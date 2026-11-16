"""Modal wrapper (EXPERIMENT_DESIGN.md §6.2-6.3). The training code itself is platform-independent.

One-time setup (once per Modal workspace):
    pip install modal && modal setup
    modal run modal_prepare.py                          # builds the tokenized data volume (CPU only)

Launch runs (each manifest line = CLI args for one run of `python -m src.train`):
    modal run --detach modal_app.py::launch --manifest manifests/stageA.txt
    modal run modal_app.py::launch --args "--method muloco --p 1.0 --seed 0"     # a single run
    modal run modal_app.py::status --manifest manifests/stageA.txt               # what is done

Download results:
    modal volume get muloco-runs / runs/
"""
from __future__ import annotations

import json
import shlex

import modal

APP_NAME = "muloco"
DATA_DIR, RUNS_DIR = "/vol/data", "/vol/runs"
DEFAULT_GPU = "A100-80GB"  # H100 is not available on the Modal Starter plan (2026-11-14)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.15.0", "numpy", "tiktoken", "pyarrow", "huggingface_hub", "pyyaml")
    .add_local_python_source("src")
    .add_local_dir("configs", "/root/configs")
)
data_vol = modal.Volume.from_name("muloco-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("muloco-runs", create_if_missing=True)


@app.function(image=image, gpu=DEFAULT_GPU, timeout=3 * 3600, max_containers=10, retries=1,
              volumes={DATA_DIR: data_vol, RUNS_DIR: runs_vol})
def train(args: list[str]) -> dict:
    import os

    from src.config import parse_args
    from src.train import train as run_training

    cfg = parse_args(list(args) + ["--data_dir", f"{DATA_DIR}/bin", "--out_dir", RUNS_DIR])
    done = os.path.join(RUNS_DIR, cfg.run_id, "summary.json")
    if os.path.exists(done):
        return {"run_id": cfg.run_id, "skipped": True}
    try:
        s = run_training(cfg)
    finally:
        runs_vol.commit()
    fe = s["final_eval"] or {}
    return {"run_id": s["run_id"], "balanced": fe.get("balanced"), "diverged": s["diverged"],
            "train_min": round(s["train_time_s"] / 60, 1), "tok_per_s": s["train_tok_per_s"]}


def _read_manifest(manifest: str, args: str) -> list[list[str]]:
    if args:
        return [shlex.split(args)]
    lines = [ln.strip() for ln in open(manifest)]
    return [shlex.split(ln) for ln in lines if ln and not ln.startswith("#")]


def _run_id(argv: list[str]) -> str:
    from src.config import parse_args
    return parse_args(list(argv)).run_id


def _completed() -> set[str]:
    done = set()
    try:
        for entry in runs_vol.listdir("/", recursive=True):
            if entry.path.endswith("summary.json"):
                done.add(entry.path.strip("/").split("/")[0])
    except Exception:
        pass
    return done


@app.local_entrypoint()
def launch(manifest: str = "", args: str = "", gpu: str = DEFAULT_GPU):
    runs = _read_manifest(manifest, args)
    done = _completed()
    todo = [a for a in runs if _run_id(a) not in done]
    print(f"{len(runs)} runs in manifest, {len(runs) - len(todo)} already complete, launching {len(todo)} on {gpu}", flush=True)
    fn = train if gpu == DEFAULT_GPU else train.with_options(gpu=gpu)
    for res in fn.map(todo, order_outputs=False, return_exceptions=True):
        print(res if isinstance(res, dict) else f"FAILED: {res!r}", flush=True)


@app.local_entrypoint()
def status(manifest: str):
    done = _completed()
    for a in _read_manifest(manifest, ""):
        rid = _run_id(a)
        print(("done  " if rid in done else "todo  ") + rid)
