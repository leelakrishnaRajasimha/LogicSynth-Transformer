import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, EarlyStopping, global_step_from_engine, DiskSaver
from ignite.contrib.handlers import ProgressBar

from dataset import LogicSynthDataset, LogicTokenizer
from model import ModelConfig, LogicSynthTransformer

# 1.)Defaults—main.py can override these via argparse, but if I run train.py
#    directly these kick in.
DEFAULTS = dict(batch=256, lr=3e-4, epochs=150, patience=200, seq_len=256, seed=42)

def run_training(batch=None, lr=None, epochs=None, **_):
    # 2.)Merge whatever main.py passes in with my defaults.
    batch    = batch    or DEFAULTS["batch"]
    lr       = lr       or DEFAULTS["lr"]
    epochs   = epochs   or DEFAULTS["epochs"]
    patience = DEFAULTS["patience"]
    seq_len  = DEFAULTS["seq_len"]
    seed     = DEFAULTS["seed"]

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = DEVICE.type == "cuda"
    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = LogicTokenizer()

    """ CURRICULUM SCHEDULE
    The model learns addition/subtraction starting from easy (1-3 digit)
    problems and progressing to harder (1-10 digit) problems. Each stage
    trains for a fixed number of epochs before the difficulty increases.
    This is the KEY trick that makes 15-digit OOD generalization work:
    the model first masters the carry algorithm on small numbers, then
    extends it to longer sequences. Without this, training on 1-8 digits
    uniformly means the model sees very few hard (8-digit) carry chains
    and never truly learns the general algorithm."""

    curriculum = [
        # (max_digits, n_examples, epochs_for_this_stage)
        (3,  300_000,  5),    # Stage 1: Master basics — 1-3 digits
        (5,  400_000,  8),    # Stage 2: Medium — 1-5 digits
        (8,  500_000,  12),   # Stage 3: Full range — 1-8 digits
        (10, 600_000,  15),   # Stage 4: Push further — 1-10 digits
        (10, 600_000,  10),   # Stage 5: Polish — more 1-10 digit practice
    ]

    # 4.)Collate function—Since it's decoder-only, I am merging query and answer.
    # CRITICAL: Returns (seq, labels) where labels mask query tokens with PAD so
    # the loss is computed ONLY on the answer portion.
    def merge_collate(samples):
        pad, sos = tokenizer.pad_id, tokenizer.sos_id
        seqs = []
        answer_starts = []
        for s in samples:
            src_clean = s["src"][s["src"] != pad]
            tgt_clean = s["tgt"][s["tgt"] != pad]
            if len(tgt_clean) > 0 and tgt_clean[0] == sos:
                tgt_clean = tgt_clean[1:]
            seqs.append(torch.cat([src_clean, tgt_clean]))
            answer_starts.append(len(src_clean))

        max_len = max(len(s) for s in seqs)
        out = torch.full((len(samples), max_len), pad, dtype=torch.long)
        labels = torch.full((len(samples), max_len), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
            ans_start = answer_starts[i]
            labels[i, :ans_start] = tokenizer.pad_id  # Masking everything before the answer starts.
            labels[i, ans_start:len(s)] = s[ans_start:]
        return out, labels

    # 5.)Init—Using AdamW and Mixed Precision (AMP).
    cfg   = ModelConfig(vocab_size=len(tokenizer), max_seq_len=seq_len)
    model = LogicSynthTransformer(cfg).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 6.)Training step—Loss is only computed on answer tokens.
    def train_step(engine, batch_data):
        model.train()
        seq, labels = batch_data
        seq, labels = seq.to(DEVICE), labels.to(DEVICE)
        inp = seq[:, :-1]
        tgt = labels[:, 1:]

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        return {"loss": loss.item(), "grad_norm": grad_norm.item()}

    # 7.)Val step—Accuracy measured ONLY on answer tokens.
    def val_step(engine, batch_data):
        model.eval()
        seq, labels = batch_data
        seq, labels = seq.to(DEVICE), labels.to(DEVICE)
        inp = seq[:, :-1]
        tgt = labels[:, 1:]

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(inp)

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_tgt    = tgt.reshape(-1)
        mask = flat_tgt != tokenizer.pad_id
        return flat_logits[mask], flat_tgt[mask]

    # 8.)Greedy decode for OOD testing at the end of each stage.
    @torch.no_grad()
    def greedy_decode(mdl, query_str, max_new=50):
        mdl.eval()
        toks = [tokenizer.sos_id] + tokenizer.encode(query_str)
        inp  = torch.tensor([toks], dtype=torch.long, device=DEVICE)
        for _ in range(max_new):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = mdl(inp)
            nxt = logits[:, -1, :].argmax(-1).item()
            if nxt == tokenizer.eos_id:
                break
            inp = torch.cat([inp, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        gen_ids = inp[0, len(toks):].tolist()
        return tokenizer.decode(gen_ids)

    def quick_ood_test(mdl, n_digits=15, n_samples=20):
        """Quick exact-match test on n_digits-digit problems."""
        correct = 0
        for _ in range(n_samples):
            op = random.choice(['+', '-'])
            a = random.randint(10**(n_digits-1), 10**n_digits - 1)
            b = random.randint(10**(n_digits-1), 10**n_digits - 1)
            if op == '-' and a < b: a, b = b, a
            target = a + b if op == '+' else a - b
            pred_rev = greedy_decode(mdl, f"{a}{op}{b}=")
            pred_str = pred_rev[::-1]
            try: val = int(pred_str)
            except: val = -1
            if val == target: correct += 1
        return correct, n_samples

    # CURRICULUM TRAINING LOOP
    print(f"\n   Device : {DEVICE}")
    print(f"   Batch  : {batch} | LR: {lr}")
    print(f"   Curriculum: {len(curriculum)} stages\n")

    global_epoch = 0
    best_val_acc = 0.0

    for stage_idx, (max_d, n_examples, stage_epochs) in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"   STAGE {stage_idx+1}/{len(curriculum)} — max_digits={max_d}, "
              f"samples={n_examples:,}, epochs={stage_epochs}")
        print(f"{'='*60}\n")

        # Build fresh datasets for this curriculum stage.
        train_ds = LogicSynthDataset(
            n_examples=n_examples, min_d=1, max_d=max_d,
            math_ops="+-", seq_len=seq_len, tokenizer=tokenizer
        )
        val_ds = LogicSynthDataset(
            n_examples=10_000, min_d=1, max_d=max_d,
            math_ops="+-", seq_len=seq_len, tokenizer=tokenizer
        )
        train_loader = DataLoader(train_ds, batch, shuffle=True,
                                  collate_fn=merge_collate, pin_memory=True)
        val_loader = DataLoader(val_ds, batch, shuffle=False,
                                collate_fn=merge_collate, pin_memory=True)

        # New scheduler per stage — OneCycleLR with warmup.
        total_steps = stage_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=lr, total_steps=total_steps,
            pct_start=0.05, anneal_strategy='cos'
        )

        # Engines (fresh per stage so Ignite state resets).
        trainer   = Engine(train_step)
        evaluator = Engine(val_step)

        Accuracy().attach(evaluator, "accuracy")
        Loss(criterion).attach(evaluator, "loss")

        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]})

        
        @trainer.on(Events.ITERATION_COMPLETED(every=100))
        def monitor_health(engine):
            grad_norm = engine.state.output["grad_norm"]
            if grad_norm > 0.9:
                pbar.log_message(f"   [Warning] High Gradient Norm: {grad_norm:.2f}")


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation(engine):
            nonlocal global_epoch, best_val_acc
            global_epoch += 1
            evaluator.run(val_loader)
            m = evaluator.state.metrics
            acc = m['accuracy']
            pbar.log_message(
                f"[Global {global_epoch}] Stage {stage_idx+1} Epoch {engine.state.epoch}/{stage_epochs} — "
                f"val_loss: {m['loss']:.4f}  val_acc: {acc:.4f}"
            )
            # Save best model.
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "global_epoch": global_epoch,
                    "val_acc": acc,
                    "stage": stage_idx + 1,
                }, f"./checkpoints/best_model.pt")
                pbar.log_message(f"   ★ New best model saved! val_acc={acc:.4f}")

        # Checkpoint top 2 per stage.
        ckpt_handler = Checkpoint(
            {"model": model, "optimizer": optim},
            DiskSaver("./checkpoints", create_dir=True, require_empty=False),
            n_saved=2,
            score_function=lambda e: e.state.metrics["accuracy"],
            score_name="val_acc",
            global_step_transform=lambda e, _: global_epoch,
        )
        evaluator.add_event_handler(Events.COMPLETED, ckpt_handler)

        trainer.run(train_loader, max_epochs=stage_epochs)

        # Quick OOD check after each stage.
        ood_correct, ood_total = quick_ood_test(model, n_digits=15, n_samples=20)
        print(f"\n   Stage {stage_idx+1} OOD check (15-digit): "
              f"{ood_correct}/{ood_total} ({100*ood_correct/ood_total:.0f}%)\n")

    # ─── FINAL OOD EVALUATION ──────────────────────────────────────────
    print("\n" + "="*60)
    print("   FINAL OOD TEST — 15-Digit (Unseen Lengths)")
    print("="*60)

    correct = 0
    test_ops = ['+', '-']
    n = 30
    for _ in range(n):
        op = random.choice(test_ops)
        a = random.randint(10**14, 10**15 - 1)
        b = random.randint(10**14, 10**15 - 1)
        if op == '-' and a < b: a, b = b, a

        target = a + b if op == '+' else a - b

        pred_rev = greedy_decode(model, f"{a}{op}{b}=")
        pred_str = pred_rev[::-1]

        try: val = int(pred_str)
        except: val = -1

        status = "PASS" if val == target else "FAIL"
        print(f"   [{status}] {a} {op} {b} = {target} (Model: {pred_str})")
        if val == target: correct += 1

    print(f"\n   Final OOD Accuracy: {correct}/{n} ({100*correct/n:.1f}%)")
    print(f"   Best val_acc across all stages: {best_val_acc:.4f}\n")

if __name__ == "__main__":
    run_training()