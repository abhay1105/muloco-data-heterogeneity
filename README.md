# MuLoCo Under Non-IID Data (EECS 182 Final Project)

Does MuLoCo (Muon as DiLoCo's inner optimizer) keep its advantage over DiLoCo when each worker trains on a
different text domain, and does DiLoCo's outer Nesterov momentum protect Muon from the client drift that federated-learning theory predicts?

- **Design:** [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md), covering setup, hypotheses, tuning protocol, budget and the code map (§10).
- **Results log:** [`RESULTS.md`](RESULTS.md), with every run, table and dated observation.
- **Run logs:** `runs/<run_id>/` (`config.json`, `metrics.jsonl`, `summary.json`). Model checkpoints (`final.pt`) are not in git.

---

## Teammate guide: running Phase 2

You'll run a list of training jobs on [Modal](https://modal.com) GPUs from your laptop. Each run takes about 25–35 minutes on an A100. Runs execute in the cloud in parallel (up to 10 at once), and your laptop can sleep once they're launched.

### One-time setup (~15 minutes)
1. **Modal account:** sign up at modal.com, then go to Settings → Usage & Billing. Add a payment method (required for GPU functions) and **set a spending limit** (e.g. $30). The free $30/month credit is applied first.
2. **Code and environment** (needs Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/), or use `python -m venv`):
   ```bash
   git clone https://github.com/abhay1105/muloco-data-heterogeneity.git
   cd muloco-data-heterogeneity
   uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
   .venv/bin/python -m pytest -q          # should print "22 passed"
   .venv/bin/modal setup                  # browser login
   ```
3. **Build the data** in your Modal workspace (CPU only, ~5–10 min, a few cents):
   ```bash
   .venv/bin/modal run modal_prepare.py
   ```
   It must print **`data_hash = 33c0c658fbf49a8a`**. If it prints anything else, stop and tell the group.

### Run your assigned manifest
Each line of a manifest is one training run. Use your part, e.g. `manifests/phase2_part2.txt`:
```bash
.venv/bin/modal run --detach modal_app.py::launch --manifest manifests/phase2_partN.txt
```
- `--detach` keeps the runs going if your laptop disconnects.
- Re-running the same command is safe: runs that already finished (they have a `summary.json`) are skipped. This is also how you retry failed runs.
- Check progress at any time:
  ```bash
  .venv/bin/modal run modal_app.py::status --manifest manifests/phase2_partN.txt
  ```
  You can also watch the app on the Modal dashboard.

### Hand back the results
When `status` shows every run as `done`:
```bash
.venv/bin/modal volume get muloco-runs / runs/
```
Then send the new `runs/<run_id>/` folders back to the group. *The exact mechanism (git commit vs. shared drive, and where checkpoints go) is still to be decided.*

### Rules
- **Don't edit `configs/frozen.yaml` or the manifests.** Every run must use the frozen hyperparameters.
- Don't change the GPU type unless asked. Results don't depend on it, but timing and cost do.
- If a run prints `non-finite loss` / `diverged`, that is a valid result, not a bug. Leave it and report it.

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
```
