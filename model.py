import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# 1.)Config—Updated vocab for negative sign and longer max_seq_len for 15-digit math.
@dataclass
class ModelConfig:
    vocab_size: int = 19         # 3 special + 11 digits (incl '-') + 5 ops.
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    max_seq_len: int = 256       # Increased so the model can handle long 15-digit sequences.
    dropout: float = 0.1
    pad_id: int = 0              # Matches <PAD> index in LogicTokenizer.

# 2.)Rotary Positional Embeddings—This allows the model to "extrapolate" to 15 digits.
class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        # Precomputing frequencies—this stays the same as your high-speed version.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        if self.cos_cached is None or seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def _rotate_half(x):
    # Helper—Standard RoPE math to rotate the vector components.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    # Applying the rotation—Matches your original logic but with better broadcasting.
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k

# 3.)Causal Self-Attention—Now using Flash Attention (SDPA) for better OOD stability.
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)

        self.rope = RoPE(self.head_dim, cfg.max_seq_len)

    def forward(self, x):
        B, T, C = x.shape
        # Single matmul then split—keeping your optimized QKV approach.
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 4.)Inject the rotary embeddings—This is key for the 10-to-15 digit jump.
        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)

        # 5.)Flash Attention logic—Replaces manual triu masking for much better speed and math accuracy.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.attn_drop.p if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))

# 6.)Single decoder block—The standard sequence of Norm, Attn, and FFN.
class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        # Expanding 4x then squishing back down—The "brain" part of the block.
        self.ffn = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# 7.)The full model—Stacks the blocks and ties weights for better efficiency.
class LogicSynthTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd, padding_idx=cfg.pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying—Using the same matrix for input and output to save params.
        self.head.weight = self.tok_emb.weight

        # Quick count for the console.
        n_params = sum(p.numel() for p in self.parameters())
        print(f"LogicSynth model loaded—{n_params/1e6:.2f}M parameters")

    def forward(self, idx):
        # idx shape: (batch, seq_len)
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x) 

# 8.)Quick shape test to ensure Flash Attention and RoPE are working together.
if __name__ == "__main__":
    cfg = ModelConfig()
    model = LogicSynthTransformer(cfg)

    dummy = torch.randint(0, cfg.vocab_size, (2, 32))  # batch=2, seq=32.
    logits = model(dummy)
    print(f"Input: {dummy.shape}  ->  Logits: {logits.shape}")