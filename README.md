# MuLoCo Under Non-IID Data (EECS 182 Final Project)

Does MuLoCo (Muon as DiLoCo's inner optimizer) keep its advantage over DiLoCo when each worker trains on a
different text domain, and does DiLoCo's outer Nesterov momentum protect Muon from the client drift that federated-learning theory predicts?

- **Design:** [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md), covering setup, hypotheses, tuning protocol, budget and the code map (§10).
- **Results log:** [`RESULTS.md`](RESULTS.md), with every run, table and dated observation.
- **Run logs:** `runs/<run_id>/` (`config.json`, `metrics.jsonl`, `summary.json`), versioned in git.
- **Checkpoints:** `final.pt` files (~102 MB each, too large for GitHub) are on Hugging Face at
  [`paidvenk/muloco-data-heterogeneity`](https://huggingface.co/paidvenk/muloco-data-heterogeneity) (private until the
  report is submitted), together with a second copy of every run log. `CHECKPOINTS.sha256` lists their SHA-256 hashes.

---

## Running the experiments
All runs execute on [Modal](https://modal.com) GPUs (A100-80GB). Each full run takes ~25–35 minutes and costs ~$1.05–1.35;
`launch` runs up to 10 in parallel and the laptop can sleep once runs are launched with `--detach`.

### One-time setup
1. Modal account with a payment method and a spending limit (Settings → Usage & Billing).
2. Code and environment (Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/)):
   ```bash
   git clone https://github.com/abhay1105/muloco-data-heterogeneity.git
   cd muloco-data-heterogeneity
   uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
   .venv/bin/python -m pytest -q          # should print "22 passed"
   .venv/bin/modal setup                  # browser login
   ```
3. Build the data (CPU only, ~5–10 min, a few cents). It must print **`data_hash = 33c0c658fbf49a8a`**:
   ```bash
   .venv/bin/modal run modal_prepare.py
   ```

### Phase 2 (production, 40 runs, frozen hyperparameters in `configs/frozen.yaml`)
```bash
.venv/bin/modal run --detach modal_app.py::launch --manifest manifests/phase2_batch1.txt   # seed 0, 12 runs
.venv/bin/modal run modal_app.py::status --manifest manifests/phase2_batch1.txt            # progress
.venv/bin/python scripts/sync_runs.py      # download finished runs, verify, checksum, back up to Hugging Face
git add runs CHECKPOINTS.sha256 && git commit -m "..." && git push
```
Then the same for `manifests/phase2_batch2.txt` (seeds 1–2, 28 runs). `manifests/phase2.txt` lists all 40.
- Re-running a launch is safe: finished runs (with a `summary.json`) are skipped. This is also how failed runs are retried.
- A run that prints `non-finite loss` / `diverged` is a valid result, not a bug.
- Don't edit `configs/frozen.yaml` or the manifests.

---

## Developer quick reference
```bash
.venv/bin/python -m pytest -q                                   # correctness tests (CPU, ~10 s)
.venv/bin/python -m src.prepare_data --out data_tiny --synthetic # tiny local data
.venv/bin/python -m src.train --method muloco --p 1.0 --data_dir data_tiny --n_layer 2 --n_embd 128 \
    --n_head 4 --steps 60 --H 10 --global_batch_seqs 16 --micro_batch_seqs 4 --eval_every 30 \
    --spectral_every 30 --warmup_steps 10 --compile 0             # local smoke run
python scripts/make_manifest.py {pilot,stageA,stageB,stageC,phase2} ...   # run lists
python analysis/summarize.py "<run-id glob>" --diag              # markdown results table
.venv/bin/python analysis/plots.py                               # report figures + H1/H2 tables -> figures/
```
