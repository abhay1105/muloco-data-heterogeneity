"""CPU-only Modal app that builds the tokenized data volume (kept separate from the GPU app so it can
run without GPU access / before a payment method is on file).

    modal run modal_prepare.py            # ~20-40 min of CPU; prints data_hash (must match across teammates)
    modal run modal_prepare.py --force    # rebuild
"""
from __future__ import annotations

import json

import modal

DATA_DIR = "/vol/data"

app = modal.App("muloco-prepare")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "tiktoken", "pyarrow", "huggingface_hub", "pyyaml")
    .add_local_python_source("src")
)
data_vol = modal.Volume.from_name("muloco-data", create_if_missing=True)


@app.function(image=image, cpu=8, memory=32768, timeout=4 * 3600, volumes={DATA_DIR: data_vol})
def prepare(force: bool = False) -> str:
    import os

    from src import prepare_data

    out = f"{DATA_DIR}/bin"
    if os.path.exists(f"{out}/meta.json") and not force:
        meta = json.load(open(f"{out}/meta.json"))
        print(f"data already prepared (data_hash={meta['data_hash']}); pass --force to rebuild")
        return meta["data_hash"]
    prepare_data.main(["--out", out, "--workdir", "/tmp/hf_parquet"])
    data_vol.commit()
    meta = json.load(open(f"{out}/meta.json"))
    for name, f in sorted(meta["files"].items()):
        print(f"  {name:28s} {f['tokens']:>12,d} tokens")
    return meta["data_hash"]


@app.local_entrypoint()
def main(force: bool = False):
    print("data_hash =", prepare.remote(force))
