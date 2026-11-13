"""GPT-style decoder (EXPERIMENT_DESIGN.md §9): pre-norm RMSNorm, RoPE, QK-norm, no biases,
GELU MLP, tied input embedding / LM head.

q, k, v, o are separate Linear layers so that Muon orthogonalizes each one on its own, and so the
diagnostics can be grouped by matrix type.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Names of the hidden 2-D matrices (Muon-managed in Muon arms; same tensors used for diagnostics).
HIDDEN_TYPES = ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.fc", "mlp.proj")


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)  # (T, hd/2)
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x):  # x: (B, T, nh, hd)
        T = x.size(1)
        cos, sin = self.cos[:, :T].to(x.dtype), self.sin[:, :T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, d: int, n_head: int, max_len: int):
        super().__init__()
        self.n_head, self.hd = n_head, d // n_head
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rotary = Rotary(self.hd, max_len)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd)
        k = self.k(x).view(B, T, self.n_head, self.hd)
        v = self.v(x).view(B, T, self.n_head, self.hd)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))  # QK-norm
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, max_len: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(d)
        self.attn = Attention(d, n_head, max_len)
        self.ln2 = nn.RMSNorm(d)
        self.mlp = MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_embd: int, n_head: int, seq_len: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, seq_len) for _ in range(n_layer)])
        self.ln_f = nn.RMSNorm(n_embd)
        self.n_layer = n_layer
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02
                if name.endswith("attn.o.weight") or name.endswith("mlp.proj.weight"):
                    std = 0.02 / math.sqrt(2 * self.n_layer)
                nn.init.normal_(p, mean=0.0, std=std)
            else:
                nn.init.ones_(p)  # RMSNorm gains

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = F.linear(self.ln_f(x), self.wte.weight)  # tied head
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    @staticmethod
    def is_hidden(name: str) -> bool:
        return name.startswith("blocks.") and name.endswith(".weight") and any(t in name for t in HIDDEN_TYPES)

    def param_counts(self) -> dict:
        hidden = sum(p.numel() for n, p in self.named_parameters() if self.is_hidden(n))
        total = sum(p.numel() for p in self.parameters())
        emb = self.wte.weight.numel()
        return {"total": total, "embedding": emb, "non_embedding": total - emb, "hidden_matrices": hidden}


def hidden_type(name: str) -> str:
    for t in HIDDEN_TYPES:
        if f".{t}." in name:
            return t.replace(".", "_")
    raise KeyError(name)
