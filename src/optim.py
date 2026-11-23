"""Inner optimizers (per worker) and the LR schedule.

Parameter split: hidden 2-D block matrices -> Muon (Muon arms) or AdamW (AdamW arms);
embedding/LM head and RMSNorm gains -> AdamW in every arm.
"""
from __future__ import annotations

import math

import torch

from .config import Config


def split_params(model) -> tuple[list, list]:
    hidden, other = [], []
    for name, p in model.named_parameters():
        (hidden if model.is_hidden(name) else other).append((name, p))
    return hidden, other


class InnerOptimizer:
    """A worker's inner optimizer(s). Every group stores `base_lr`; set_lr_mult scales it."""

    def __init__(self, model, cfg: Config):
        hidden, other = split_params(model)
        fused = next(model.parameters()).is_cuda
        adam_kw = dict(betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        self.opts = []
        if cfg.inner == "adamw":
            self.opts.append(torch.optim.AdamW([p for _, p in hidden + other], lr=cfg.lr, fused=fused, **adam_kw))
        elif cfg.inner == "muon":
            self.opts.append(torch.optim.Muon([p for _, p in hidden], lr=cfg.lr_muon, weight_decay=cfg.weight_decay,
                                              momentum=cfg.muon_momentum, nesterov=True,
                                              ns_steps=cfg.muon_ns_steps, adjust_lr_fn=cfg.muon_adjust_lr))
            self.opts.append(torch.optim.AdamW([p for _, p in other], lr=cfg.lr, fused=fused, **adam_kw))
        elif cfg.inner == "sgd":  # test-only
            self.opts.append(torch.optim.SGD([p for _, p in hidden + other], lr=cfg.lr))
        else:
            raise ValueError(cfg.inner)
        for opt in self.opts:
            for g in opt.param_groups:
                g["base_lr"] = g["lr"]

    def set_lr_mult(self, mult: float):
        for opt in self.opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult

    def step(self):
        for opt in self.opts:
            opt.step()

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad(set_to_none=True)

    def state_dict(self):
        return [opt.state_dict() for opt in self.opts]

    def load_state_dict(self, sds):
        for opt, sd in zip(self.opts, sds):
            opt.load_state_dict(sd)


def averaged_state(workers: list[InnerOptimizer]) -> list[dict]:
    """Element-wise average of the workers' optimizer states (used for the DP reference learner)."""
    sds = [w.state_dict() for w in workers]
    out = []
    for i in range(len(sds[0])):
        base = sds[0][i]
        state = {}
        for pid, st in base["state"].items():
            state[pid] = {}
            for key, val in st.items():
                if torch.is_tensor(val) and val.is_floating_point() and val.dim() > 0:
                    state[pid][key] = sum(sd[i]["state"][pid][key] for sd in sds) / len(sds)
                else:
                    state[pid][key] = val
        out.append({"state": state, "param_groups": base["param_groups"]})
    return out


def lr_mult(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to min_lr_frac, indexed by 1-based global inner step."""
    if step <= cfg.warmup_steps:
        return step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
