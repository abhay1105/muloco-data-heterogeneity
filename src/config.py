"""Run configuration: one dataclass, filled from defaults <- optional YAML (--config) <- CLI flags.

Defaults are the fixed (untuned) hyperparameters shared by every run. Tuned values (inner LRs, outer
lr/momentum) live in configs/frozen.yaml and are passed with --config.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field, fields

import yaml

# method -> (mode, inner optimizer for hidden matrices, outer optimizer)
METHODS = {
    "dp_adamw": ("dp", "adamw", None),
    "dp_muon": ("dp", "muon", None),
    "diloco": ("local", "adamw", "nesterov"),
    "muloco": ("local", "muon", "nesterov"),
    "fedavg_adamw": ("local", "adamw", "fedavg"),
    "fedavg_muon": ("local", "muon", "fedavg"),
    # test-only: plain SGD inner optimizer, used by the averaging-exactness test
    "fedavg_sgd": ("local", "sgd", "fedavg"),
    "dp_sgd": ("dp", "sgd", None),
}

DOMAINS = ("arxiv", "github", "stackexchange", "wikipedia")


@dataclass
class Config:
    # --- identity ---
    method: str = "muloco"
    p: float = 0.25  # domain skew: worker k gets fraction p of domain k (0.25 = IID for 4 domains)
    H: int = 30  # inner steps per outer round
    K: int = 4  # workers (must equal number of domains for local mode, or 1)
    seed: int = 0
    run_id: str = ""  # auto-generated if empty
    tag: str = ""  # optional suffix for run_id (e.g. tuning point)

    # --- io ---
    data_dir: str = "data_bin"
    out_dir: str = "runs"
    save_checkpoint: bool = True

    # --- model ---
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    seq_len: int = 1024
    vocab_size: int = 50304

    # --- batch / schedule ---
    global_batch_seqs: int = 64  # sequences per step summed over workers (DP batch)
    micro_batch_seqs: int = 16  # forward/backward chunk (grad accumulation)
    steps: int = 7500  # inner steps per worker == DP steps
    warmup_steps: int = 250
    min_lr_frac: float = 0.1

    # --- inner optimizers ---
    lr: float = 3e-3  # AdamW lr (all AdamW-managed params)
    lr_muon: float = 3e-3  # Muon lr for hidden matrices (AdamW-matched scale)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr: str = "match_rms_adamw"
    grad_clip: float = 1.0

    # --- outer optimizer (ignored for fedavg, which is forced to lr=1, momentum=0) ---
    outer_lr: float = 0.7
    outer_momentum: float = 0.9

    # --- eval / diagnostics ---
    eval_every: int = 300  # steps; must be a multiple of H in local mode
    eval_seqs: int = 250  # per domain, periodic eval
    final_eval_seqs: int = 1000  # per domain, final eval
    log_every: int = 50  # DP train-loss logging interval (local mode logs every round)
    spectral_every: int = 300  # steps; spectral diagnostics cadence (multiple of H)
    probe_fracs: tuple = (0.25, 0.5, 0.75)  # rounds for reference-alignment + specialization
    probe: bool = True

    # --- runtime ---
    device: str = "auto"
    dtype: str = "bf16"  # bf16 autocast or fp32
    compile: bool = True

    # derived (not CLI)
    mode: str = field(default="", init=False)
    inner: str = field(default="", init=False)
    outer: str = field(default="", init=False)

    def finalize(self) -> "Config":
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method}; choose from {list(METHODS)}")
        self.mode, self.inner, self.outer = METHODS[self.method]
        self.probe_fracs = tuple(float(x) for x in self.probe_fracs)
        if self.mode == "dp":
            self.K = 1
        if self.outer == "fedavg":
            self.outer_lr, self.outer_momentum = 1.0, 0.0
        if self.mode == "local":
            if self.K not in (1, len(DOMAINS)):
                raise ValueError("local mode supports K=1 or K=number of domains")
            for name in ("steps", "eval_every", "spectral_every"):
                if getattr(self, name) % self.H:
                    raise ValueError(f"{name}={getattr(self, name)} must be a multiple of H={self.H}")
        if self.global_batch_seqs % self.K:
            raise ValueError("global_batch_seqs must be divisible by K")
        if (self.global_batch_seqs // self.K) % self.micro_batch_seqs:
            raise ValueError("per-worker batch must be divisible by micro_batch_seqs")
        if not self.run_id:
            self.run_id = default_run_id(self)
        return self

    @property
    def worker_batch_seqs(self) -> int:
        return self.global_batch_seqs // self.K

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def default_run_id(c: Config) -> str:
    if c.mode == "dp":
        rid = f"{c.method}_s{c.seed}"
    else:
        rid = f"{c.method}_p{c.p:.2f}_H{c.H}_s{c.seed}"
    return rid + (f"_{c.tag}" if c.tag else "")


def _parse_bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y"):
        return True
    if s.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {s}")


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="DiLoCo / MuLoCo / DP training")
    ap.add_argument("--config", default=None, help="YAML file of overrides (e.g. configs/frozen.yaml)")
    init_fields = [f for f in fields(Config) if f.init]
    for f in init_fields:
        default = f.default
        if isinstance(default, bool):
            ap.add_argument(f"--{f.name}", type=_parse_bool, default=None)
        elif isinstance(default, tuple):
            ap.add_argument(f"--{f.name}", type=lambda s: tuple(float(x) for x in s.split(",") if x), default=None)
        else:
            ap.add_argument(f"--{f.name}", type=type(default), default=None)
    args = ap.parse_args(argv)
    kwargs = {}
    if args.config:
        # YAML layout: flat keys apply to every method; an optional `methods:` mapping holds
        # per-method overrides, e.g. methods: {muloco: {lr_muon: 6e-3, outer_lr: 0.8}}.
        with open(args.config) as fh:
            y = yaml.safe_load(fh) or {}
        per_method = y.pop("methods", {}) or {}
        method = args.method or y.get("method") or Config.method
        y.update(per_method.get(method, {}) or {})
        unknown = set(y) - {f.name for f in init_fields}
        if unknown:
            raise ValueError(f"unknown keys in {args.config}: {unknown}")
        kwargs.update(y)
    for f in init_fields:
        v = getattr(args, f.name)
        if v is not None:
            kwargs[f.name] = v
    return Config(**kwargs).finalize()


if __name__ == "__main__":
    print(json.dumps(parse_args().to_dict(), indent=2))
