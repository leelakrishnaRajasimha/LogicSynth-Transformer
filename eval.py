import os
import glob
import random
import torch
from dataset import LogicTokenizer
from model import ModelConfig, LogicSynthTransformer

# 1.)Module-level setup so playground.py and other scripts can import these.
SEQ_LEN   = 128
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp   = DEVICE.type == "cuda"
tokenizer = LogicTokenizer()

# 2.)Loading the best checkpoint—I'm sorting by name to get the latest saved state.
def load_best_checkpoint(model):
    ckpt_dir = "./checkpoints"

    # Prefer the dedicated best_model.pt from curriculum training.
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_path):
        print(f"   Loading checkpoint: best_model.pt")
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            print(f"   (val_acc={ckpt.get('val_acc', '?')}, stage={ckpt.get('stage', '?')})")
        else:
            model.load_state_dict(ckpt)
        model.eval()
        return model

    # Fallback: grab latest numbered checkpoint.
    pattern  = os.path.join(ckpt_dir, "checkpoint_*.pt")
    files    = sorted(glob.glob(pattern))

    if not files:
        print("   [WARN] No checkpoints found! Training must happen first.")
        return model

    best = files[-1]
    print(f"   Loading checkpoint: {os.path.basename(best)}")
    ckpt = torch.load(best, map_location=DEVICE, weights_only=False)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

# 3.)Greedy decode—This is how the model generates answers one digit at a time.
@torch.no_grad()
def greedy_decode(model, query_str, max_new=64, temperature=0.7):
    toks = [tokenizer.sos_id] + tokenizer.encode(query_str)
    inp  = torch.tensor([toks], dtype=torch.long, device=DEVICE)

    for _ in range(max_new):
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(inp)
        
        next_token_logits = logits[:, -1, :] / temperature
        nxt = next_token_logits.argmax(-1).item()
        
        if nxt == tokenizer.eos_id:
            break
        inp = torch.cat([inp, torch.tensor([[nxt]], device=DEVICE)], dim=1)

    gen_ids = inp[0, len(toks):].tolist()
    return tokenizer.decode(gen_ids)

def run_eval():
    # 4.)Seeding here so eval runs are reproducible but imports don't trigger side effects.
    SEED = 123
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 5.)Stress test—Checking if the model can generalize to unseen lengths.
    def stress_test(model, n_digits, n_samples=50, ops="+-"):
        correct  = 0
        failures = []

        for _ in range(n_samples):
            op = random.choice(list(ops))
            a  = random.randint(10**(n_digits-1), 10**n_digits - 1)
            b  = random.randint(10**(n_digits-1), 10**n_digits - 1)

            if op == "-" and a < b:
                a, b = b, a

            if op == "+": target = a + b
            elif op == "-": target = a - b
            else: target = a * b

            query    = f"{a}{op}{b}="
            pred_full = greedy_decode(model, query)
            if "Ans:" in pred_full:
                pred_str = pred_full.split("Ans:")[-1][::-1]
            else:
                pred_str = pred_full[::-1]

            try:
                pred_val = int(pred_str)
            except ValueError:
                pred_val = -1

            if pred_val == target:
                correct += 1
            else:
                failures.append({
                    "query": f"{a} {op} {b}",
                    "expected": str(target),
                    "got": pred_str,
                })

        acc = 100 * correct / n_samples
        return acc, failures

    # 6.)Table printer—A clean summary to show my results at a glance.
    def print_results_table(results):
        print("\n" + "=" * 65)
        print(f"   {'Digits':>6}   {'Samples':>8}   {'EM Acc (%)':>12}   {'Status'}")
        print("-" * 65)
        for r in results:
            tag = "ELITE" if r["acc"] >= 95 else ("PASS" if r["acc"] >= 75 else "FAIL")
            print(f"   {r['digits']:>6}   {r['samples']:>8}   {r['acc']:>12.1f}   {tag}")
        print("=" * 65)

    # 7.)Error analysis—Helping me understand if it's a minor carry-over error.
    def print_error_analysis(digits, failures, max_show=5):
        if not failures:
            print(f"   [{digits}-digit] No errors—Perfect Generalization.")
            return

        print(f"\n   [{digits}-digit] Showing {min(len(failures), max_show)} sample errors:")
        print(f"   {'Query':<40} {'Expected':<25} {'Output':<25}")
        print("   " + "-" * 100)

        for f in failures[:max_show]:
            exp, got = f["expected"], f["got"]
            print(f"   {f['query']:<40} {exp:<25} {got:<25}")

    # 8.)Build model + load checkpoint.
    cfg   = ModelConfig(vocab_size=len(tokenizer), max_seq_len=SEQ_LEN)
    model = LogicSynthTransformer(cfg).to(DEVICE)
    model = load_best_checkpoint(model)

    # 9.)The main eval routine—Testing range from training (8) to extreme (20).
    print("\n" + "=" * 65)
    print("   LogicSynth Transformer — Full Evaluation Suite")
    print("   Testing OOD Generalization (Trained up to 8 digits)")
    print("=" * 65)

    test_configs = [
        {"digits": 8,  "samples": 50, "ops": "+-"},
        {"digits": 10, "samples": 50, "ops": "+-"},
        {"digits": 15, "samples": 50, "ops": "+-"},
        {"digits": 20, "samples": 30, "ops": "+-"},
    ]

    results  = []
    all_fails = {}

    for cfg in test_configs:
        d, n, ops = cfg["digits"], cfg["samples"], cfg["ops"]
        acc, fails = stress_test(model, d, n, ops)
        results.append({"digits": d, "samples": n, "acc": acc})
        all_fails[d] = fails
        print(f"   [{d:>2}-digit] Accuracy: {acc:.1f}%")

    print_results_table(results)

    print("\n" + "=" * 65)
    print("   Detailed Failure Analysis")
    print("=" * 65)
    for d, fails in all_fails.items():
        print_error_analysis(d, fails)

# 10.)So I can still run `python eval.py` directly if I want.
if __name__ == "__main__":
    run_eval()