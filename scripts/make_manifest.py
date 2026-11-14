"""Write run manifests (one run's CLI args per line) for each stage of EXPERIMENT_DESIGN.md §5-6.

    python scripts/make_manifest.py pilot
    python scripts/make_manifest.py stageA [--adamw_lrs ...] [--muon_lrs ...] [--muon_adam_lr X]
    python scripts/make_manifest.py stageB --lr_adamw X --lr_muon Y            # inner LRs from stage A
    python scripts/make_manifest.py stageC --lr_adamw X --lr_muon Y --diloco_outer LR,MU --muloco_outer LR,MU
    python scripts/make_manifest.py phase2 [--parts 4]                          # needs configs/frozen.yaml

Then: modal run --detach modal_app.py::launch --manifest manifests/<name>.txt
"""
from __future__ import annotations

import argparse
import os

HALF = ["--steps", "3750"]  # half-length tuning runs (schedule compressed to fit)


def fmt(x: float) -> str:
    return f"{x:g}"


def write(name: str, lines: list[str], header: str = ""):
    os.makedirs("manifests", exist_ok=True)
    path = os.path.join("manifests", f"{name}.txt")
    with open(path, "w") as fh:
        if header:
            fh.write("".join(f"# {h}\n" for h in header.splitlines()))
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines)} runs)")


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "stageA", "stageB", "stageC", "phase2"])
    ap.add_argument("--adamw_lrs", default="7.5e-4,1.5e-3,3e-3,6e-3")
    ap.add_argument("--muon_lrs", default="1.5e-3,3e-3,6e-3,1.2e-2")
    ap.add_argument("--muon_adam_lr", type=float, default=None, help="AdamW lr for non-hidden params in DP-Muon")
    ap.add_argument("--lr_adamw", type=float, default=None)
    ap.add_argument("--lr_muon", type=float, default=None)
    ap.add_argument("--outer_lrs", default="0.5,0.8,1.0")
    ap.add_argument("--outer_moms", default="0.6,0.9")
    ap.add_argument("--diloco_outer", default=None, help="LR,MU best for diloco from stage B")
    ap.add_argument("--muloco_outer", default=None, help="LR,MU best for muloco from stage B")
    ap.add_argument("--parts", type=int, default=1, help="split phase2 into N teammate manifests")
    a = ap.parse_args()

    if a.stage == "pilot":
        # Short runs: throughput per method + compile check. ~300 steps each.
        short = "--steps 300 --eval_every 300 --spectral_every 300 --warmup_steps 30 --save_checkpoint 0 --tag pilot"
        write("pilot", [
            f"--method dp_adamw {short}",
            f"--method dp_muon {short}",
            f"--method diloco --p 1.0 {short}",
            f"--method muloco --p 1.0 {short} --probe_fracs 0.5",
        ], "E0 pilot: throughput + end-to-end check on the GPU (not used for results)")

    elif a.stage == "stageA":
        lines = [f"--method dp_adamw --lr {fmt(lr)} --tag lr{fmt(lr)}" for lr in floats(a.adamw_lrs)]
        if a.muon_adam_lr is None:
            print("note: --muon_adam_lr not given; DP-Muon lines are omitted (run them after the AdamW sweep)")
        else:
            lines += [f"--method dp_muon --lr {fmt(a.muon_adam_lr)} --lr_muon {fmt(lr)} --tag lrm{fmt(lr)}"
                      for lr in floats(a.muon_lrs)]
        write("stageA", lines, "Stage A: DP inner-LR sweeps (seed 0, full length). Best run of each = DP seed-0 baseline.")

    elif a.stage == "stageB":
        assert a.lr_adamw and a.lr_muon, "--lr_adamw and --lr_muon (from stage A) are required"
        lines = []
        for method in ("diloco", "muloco"):
            for olr in floats(a.outer_lrs):
                for mom in floats(a.outer_moms):
                    lines.append(f"--method {method} --p 0.25 --lr {fmt(a.lr_adamw)} --lr_muon {fmt(a.lr_muon)} "
                                 f"--outer_lr {fmt(olr)} --outer_momentum {fmt(mom)} {' '.join(HALF)} "
                                 f"--save_checkpoint 0 --tag eo{fmt(olr)}_mu{fmt(mom)}")
        write("stageB", lines, "Stage B: outer (lr, momentum) sweep, IID, H=30, seed 0, half length.")

    elif a.stage == "stageC":
        assert a.lr_adamw and a.lr_muon and a.diloco_outer and a.muloco_outer
        lines = []
        for method, outer in (("diloco", a.diloco_outer), ("muloco", a.muloco_outer)):
            olr, mom = floats(outer)
            for mult in (0.5, 2.0):
                lr = a.lr_adamw * (mult if method == "diloco" else 1.0)
                lrm = a.lr_muon * (mult if method == "muloco" else 1.0)
                lines.append(f"--method {method} --p 0.25 --lr {fmt(lr)} --lr_muon {fmt(lrm)} --outer_lr {fmt(olr)} "
                             f"--outer_momentum {fmt(mom)} {' '.join(HALF)} --save_checkpoint 0 --tag innerx{fmt(mult)}")
        write("stageC", lines, "Stage C: inner-LR x{0.5,2} check at the best outer config (half length).\n"
                               "Compare against the matching stage-B run (inner x1).")

    elif a.stage == "phase2":
        assert os.path.exists("configs/frozen.yaml"), "write configs/frozen.yaml first (end of Phase 1)"
        cfg = "--config configs/frozen.yaml"
        order = []  # seed 0 of everything first, then remaining seeds (EXPERIMENT_DESIGN.md §6.1)
        for seed in (0, 1, 2):
            for method in ("diloco", "muloco"):
                for p in ("0.25", "0.5", "0.75", "1.0"):
                    order.append((seed, f"{cfg} --method {method} --p {p} --seed {seed}"))
            for method in ("fedavg_adamw", "fedavg_muon"):
                for p in ("0.25", "1.0"):
                    order.append((seed, f"{cfg} --method {method} --p {p} --seed {seed}"))
            if seed > 0:
                for method in ("dp_adamw", "dp_muon"):
                    order.append((seed, f"{cfg} --method {method} --seed {seed}"))
        lines = [l for _, l in order]
        assert len(lines) == 40, len(lines)
        if a.parts == 1:
            write("phase2", lines, "Phase 2 production runs (40). Seed 0 first.")
        else:  # round-robin so every part gets a share of seed 0
            for i in range(a.parts):
                write(f"phase2_part{i + 1}", lines[i::a.parts], f"Phase 2, part {i + 1}/{a.parts}")


if __name__ == "__main__":
    main()
