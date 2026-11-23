"""Single entry point for every run: DP baselines and DiLoCo-family (local) training.

    python -m src.train --method muloco --p 1.0 --seed 0 [--config configs/frozen.yaml] [...]

Local mode simulates K workers sequentially on one device: one model,
one inner optimizer per worker (persistent state), fp32 outer parameters + outer SGD/Nesterov.

Outputs in <out_dir>/<run_id>/: config.json, metrics.jsonl, summary.json (written last = completion
marker), final.pt (bf16 outer model, optional).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time

import torch

from .config import DOMAINS, Config, parse_args
from .data import TokenData, build_train_pools, eval_rows, n_train_needed, probe_pool, to_device
from .diagnostics import reference_metrics, round_metrics, spectral_metrics
from .model import GPT
from .optim import InnerOptimizer, averaged_state, lr_mult


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Logger:
    def __init__(self, run_dir: str):
        self.fh = open(os.path.join(run_dir, "metrics.jsonl"), "w")

    def log(self, kind: str, **kw):
        kw = {"kind": kind, "time": round(time.time(), 2), **kw}
        self.fh.write(json.dumps(kw) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.data = TokenData(cfg.data_dir, cfg.seq_len, n_train_needed(cfg, len(DOMAINS)))
        torch.manual_seed(cfg.seed)  # same init for every method at a given seed
        self.model = GPT(cfg.vocab_size, cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.seq_len).to(self.device)
        self.fwd = torch.compile(self.model) if cfg.compile else self.model
        self.params = list(self.model.parameters())
        self.names = [n for n, _ in self.model.named_parameters()]
        self.hidden_idx = [i for i, n in enumerate(self.names) if self.model.is_hidden(n)]
        self.pools = build_train_pools(cfg, self.data, self.device)
        self.autocast = (lambda: torch.autocast(self.device.type, dtype=torch.bfloat16)) if cfg.dtype == "bf16" \
            else contextlib.nullcontext
        self.run_dir = os.path.join(cfg.out_dir, cfg.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
        self.log = Logger(self.run_dir)
        self.train_time = 0.0
        self.diverged = False

    # ------------------------------------------------------------------ basic steps
    def train_step(self, opt: InnerOptimizer, pool, pool_step: int, global_step: int, wrap: bool = False):
        cfg = self.cfg
        opt.set_lr_mult(lr_mult(global_step, cfg))
        x, y = pool.batch(pool_step, wrap=wrap)
        n_micro = x.size(0) // cfg.micro_batch_seqs
        total = torch.zeros((), device=self.device)
        for i in range(n_micro):
            sl = slice(i * cfg.micro_batch_seqs, (i + 1) * cfg.micro_batch_seqs)
            with self.autocast():
                loss = self.fwd(x[sl], y[sl])
            (loss / n_micro).backward()
            total += loss.detach()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        return total / n_micro

    @torch.no_grad()
    def evaluate(self, n_seqs: int) -> dict:
        n_seqs = min(n_seqs, self.data.n_seq["val"])
        mb = self.cfg.micro_batch_seqs
        out = {}
        for d, name in enumerate(self.data.domains):
            rows = eval_rows(d, n_seqs)
            tot, cnt = 0.0, 0
            for i in range(0, n_seqs, mb):
                x, y = to_device(self.data.gather("val", rows[i:i + mb]), self.device)
                with self.autocast():
                    loss = self.fwd(x, y)
                tot += float(loss) * len(x)
                cnt += len(x)
            out[name] = tot / cnt
        out["balanced"] = sum(out[d] for d in self.data.domains) / len(self.data.domains)
        out["worst"] = max(out[d] for d in self.data.domains)
        return out

    @torch.no_grad()
    def load_params(self, src: list[torch.Tensor]):
        for p, s in zip(self.params, src):
            p.copy_(s)

    def _check_finite(self, value: float, step: int) -> bool:
        if not math.isfinite(value):
            self.diverged = True
            self.log.log("diverged", step=step, loss=value)
            print(f"[{self.cfg.run_id}] non-finite loss at step {step}; stopping", flush=True)
            return False
        return True

    # ------------------------------------------------------------------ DP
    def run_dp(self) -> int:
        cfg = self.cfg
        opt = InnerOptimizer(self.model, cfg)
        pool = self.pools[0]
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            loss = self.train_step(opt, pool, step - 1, step)
            if step % cfg.log_every == 0 or step == cfg.steps:
                lv = float(loss)
                self.train_time += time.time() - t0
                if not self._check_finite(lv, step):
                    return step
                self.log.log("train", step=step, loss=lv, lr_mult=lr_mult(step, cfg),
                             tok_per_s=self._tok_per_s(step))
                t0 = time.time()
            if step % cfg.eval_every == 0 and step != cfg.steps:
                self.train_time += time.time() - t0
                self.log.log("eval", step=step, **self.evaluate(cfg.eval_seqs))
                self._progress(step)
                t0 = time.time()
        return cfg.steps

    # ------------------------------------------------------------------ local (DiLoCo family)
    def run_local(self) -> int:
        cfg, K, H = self.cfg, self.cfg.K, self.cfg.H
        outer = [p.detach().clone().float() for p in self.params]
        outer_opt = torch.optim.SGD(outer, lr=cfg.outer_lr, momentum=cfg.outer_momentum,
                                    nesterov=cfg.outer_momentum > 0)
        workers = [InnerOptimizer(self.model, cfg) for _ in range(K)]
        hidden_names = [self.names[i] for i in self.hidden_idx]
        hidden_slot = dict(zip(self.hidden_idx, hidden_names))
        n_rounds = cfg.steps // H
        probe_rounds = {}
        if cfg.probe and K > 1:
            for j, f in enumerate(cfg.probe_fracs):
                probe_rounds[min(n_rounds - 1, int(round(f * n_rounds)))] = j
        ppool = probe_pool(cfg, self.data, self.device) if probe_rounds else None

        for r in range(n_rounds):
            step_end = (r + 1) * H
            ref, spec = None, None
            if r in probe_rounds:  # DP reference pseudo-gradient from the same start (diagnostic only)
                t_probe = time.time()
                self.load_params(outer)
                ref_opt = InnerOptimizer(self.model, cfg)
                ref_opt.load_state_dict(averaged_state(workers))
                for t in range(H):
                    self.train_step(ref_opt, ppool, probe_rounds[r] * H + t, r * H + t + 1, wrap=True)
                with torch.no_grad():
                    ref = {self.names[i]: (outer[i] - self.params[i]).clone() for i in self.hidden_idx}
                del ref_opt
                spec = {}
                t_probe = time.time() - t_probe

            t0 = time.time()
            delta_sum = [torch.zeros_like(o) for o in outer]
            hdeltas = {n: torch.empty((K, *outer[i].shape), device=self.device)
                       for n, i in zip(hidden_names, self.hidden_idx)}
            losses = []
            for k in range(K):
                self.load_params(outer)
                acc = torch.zeros((), device=self.device)
                for t in range(H):
                    acc += self.train_step(workers[k], self.pools[k], r * H + t, r * H + t + 1)
                losses.append(acc / H)
                if spec is not None:
                    self.train_time += time.time() - t0
                    spec[f"worker{k}"] = self.evaluate(cfg.eval_seqs)
                    t0 = time.time()
                with torch.no_grad():
                    for i, (o, p) in enumerate(zip(outer, self.params)):
                        d = o - p
                        delta_sum[i] += d
                        if i in hidden_slot:
                            hdeltas[hidden_slot[i]][k] = d
            worker_losses = [float(x) for x in losses]  # one sync per round
            self.train_time += time.time() - t0
            if not self._check_finite(sum(worker_losses) / K, step_end):
                return step_end

            mom = None
            if outer_opt.state:
                mom = {n: outer_opt.state[outer[i]]["momentum_buffer"] for n, i in zip(hidden_names, self.hidden_idx)
                       if "momentum_buffer" in outer_opt.state[outer[i]]}
            diag = round_metrics(hdeltas, mom)
            if step_end % cfg.spectral_every == 0:
                diag.update(spectral_metrics(hdeltas))
            if ref is not None:
                diag.update(reference_metrics(hdeltas, ref))
                diag["specialization"] = spec
            with torch.no_grad():
                for o, d in zip(outer, delta_sum):
                    o.grad = d / K
            outer_opt.step()
            outer_opt.zero_grad(set_to_none=True)
            del hdeltas, delta_sum

            self.log.log("train", step=step_end, round=r, loss=sum(worker_losses) / K, worker_loss=worker_losses,
                         lr_mult=lr_mult(step_end, cfg), tok_per_s=self._tok_per_s(step_end))
            self.log.log("diag", step=step_end, round=r, **diag)
            if step_end % cfg.eval_every == 0 and step_end != cfg.steps:
                self.load_params(outer)
                self.log.log("eval", step=step_end, **self.evaluate(cfg.eval_seqs))
                self._progress(step_end)
        self.load_params(outer)
        return cfg.steps

    # ------------------------------------------------------------------ helpers
    def _tok_per_s(self, step: int) -> float:
        return round(step * self.cfg.global_batch_seqs * self.cfg.seq_len / max(self.train_time, 1e-9), 1)

    def _progress(self, step: int):
        print(f"[{self.cfg.run_id}] step {step}/{self.cfg.steps}  train_time {self.train_time / 60:.1f} min  "
              f"tok/s {self._tok_per_s(step):,.0f}", flush=True)

    def run(self) -> dict:
        cfg = self.cfg
        t_start = time.time()
        counts = self.model.param_counts()
        self.log.log("start", run_id=cfg.run_id, device=str(self.device), params=counts,
                     data_hash=self.data.data_hash, torch=torch.__version__,
                     gpu=torch.cuda.get_device_name() if self.device.type == "cuda" else None)
        print(f"[{cfg.run_id}] device={self.device} params={counts} data_hash={self.data.data_hash}", flush=True)
        steps_done = self.run_dp() if cfg.mode == "dp" else self.run_local()
        final = None
        if not self.diverged:
            final = self.evaluate(cfg.final_eval_seqs)
            self.log.log("final_eval", step=steps_done, **final)
            if cfg.save_checkpoint:
                sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in self.model.state_dict().items()}
                torch.save(sd, os.path.join(self.run_dir, "final.pt"))
        summary = {
            "run_id": cfg.run_id, "method": cfg.method, "p": cfg.p, "H": cfg.H, "K": cfg.K, "seed": cfg.seed,
            "tag": cfg.tag, "steps_completed": steps_done, "diverged": self.diverged, "final_eval": final,
            "train_time_s": round(self.train_time, 1), "wall_time_s": round(time.time() - t_start, 1),
            "train_tok_per_s": self._tok_per_s(steps_done), "data_hash": self.data.data_hash, "params": counts,
            "device": str(self.device), "config": cfg.to_dict(),
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        self.log.close()
        print(f"[{cfg.run_id}] done: balanced={final and round(final['balanced'], 4)} diverged={self.diverged} "
              f"train_time={self.train_time / 60:.1f} min", flush=True)
        return summary


def train(cfg: Config) -> dict:
    return Runner(cfg).run()


def main(argv=None) -> dict:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
