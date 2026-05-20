# Remote Execution Scripts

Run long lmms-eval jobs on a lab VM without keeping your laptop awake. Every
script in this folder is a thin SSH/rsync wrapper around a shared config block
and a per-job declarative `.conf`. The VM runs the actual work inside a tmux
session so you can disconnect, reconnect, and tail logs at will.

## Layout

```
remote_execution_scripts/
├── .env.example              # template — copy to .env and set SSH_KEY
├── config.sh                 # VM identity (user, host, port, workdir)
├── jobs/
│   ├── example.conf                                 # boilerplate template
│   └── dynamic_graph_benchmark_qwen25vl_3b.conf     # ready-made job
├── lib/
│   └── common.sh             # shared bash helpers (ssh, rsync, tmux names)
├── 00_wsl_fix_key.sh         # copy your SSH key from /mnt/c to ~/.ssh w/ chmod 600
├── 01_deploy.sh <job>        # rsync repo up + run REMOTE_SETUP_CMD
├── 02_run.sh    <job>        # start the job in tmux, tee to run.log
├── 03_logs.sh   <job> [...]  # tail -F run.log (or --attach, --status, --tail N)
├── 04_fetch.sh  <job>        # rsync RESULT_PATHS back into ./remote_results/<job>/
├── 05_stop.sh   <job>        # kill the tmux session
├── 06_compare_direct_disguise.sh <job> [--timestamp TS]
│                             # (postprocess) write an Excel comparing direct vs disguise per pair
├── 07_batch_report.sh   -f <manifest.txt> [...]
│                             # (postprocess) one xlsx aggregating every job in a batch
├── run_batch.sh         -f <manifest.txt> [...]
│                             # chain 01→04 across many jobs
├── batches/<group>.txt       # per-axis manifests (run with -f, see "Batches" below)
└── jobs/generate_graph_benchmark_jobs.py   # regenerates jobs/graph_benchmark/* + batches/*
```

## Prerequisites

- Bash (Linux, macOS, or WSL on Windows)
- `rsync` and `ssh` on your local machine; `tmux` + `uv` already installed on the VM
- SSH access to the VM with a key you own

### VM layout assumptions

`config.sh` defaults assume the VM03 layout (SSD-backed; home partition is small):

| What | Path |
|---|---|
| `dynamic-lmms-eval` repo | `/media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval` |
| sibling `dynamic-dataset` repo | `/media/vm03/ssd1T/andrew/dynamic/dynamic-dataset` |
| HuggingFace cache (`HF_HOME`) | `/media/vm03/ssd1T/andrew/hf_cache` |
| uv cache (`UV_CACHE_DIR`) | `/media/vm03/ssd1T/andrew/uv_cache` |

All four are created on first deploy; override via env vars (`REMOTE_WORKDIR`, `REMOTE_DATASET_DIR`, `REMOTE_HF_HOME`, `REMOTE_UV_CACHE_DIR`) for other VMs.

> **One-time bootstrap:** `uv` is not installed globally on VM03 — it lives inside `$REMOTE_WORKDIR/.venv`. Before the first `./01_deploy.sh`, create the venv and install `uv` into it once on the VM:
> ```bash
> ssh vm03 "cd /media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval && python3 -m venv .venv && .venv/bin/pip install uv"
> ```
> Subsequent deploys activate `.venv` automatically and call `uv sync`.

### SSH key

Copy `.env.example` to `.env` and point `SSH_KEY` at your private key:

```bash
cp remote_execution_scripts/.env.example remote_execution_scripts/.env
# then edit .env so SSH_KEY=... matches your key path
```

`.env` is gitignored and loaded automatically by every script. You can also
override per-invocation (`SSH_KEY=... ./02_run.sh <job>`) — shell env wins
over `.env`.

On WSL with a Windows-side key (`/mnt/c/Users/…`), use the helper to copy
it into `~/.ssh` with `chmod 600`:

```bash
source ./00_wsl_fix_key.sh /mnt/c/Users/Andrew/.ssh/vm_key
```

It exports `SSH_KEY` for the current shell; for persistence put the resulting
path in `.env` instead.

## Typical flow

```bash
# one-time: point config.sh at your VM if needed (VM_USER, VM_HOST, VM_PORT,
# REMOTE_WORKDIR) — or override inline via env vars / .env.
# SSH_KEY is read from remote_execution_scripts/.env (see "SSH key" above).

JOB=dynamic_graph_benchmark_qwen25vl_3b

./01_deploy.sh $JOB       # rsync repo to ~/dynamic-lmms-eval on the VM + uv sync
./02_run.sh   $JOB        # tmux session `lmms_<job>` starts running the eval
./03_logs.sh  $JOB        # live-tail run.log; Ctrl-C detaches, job keeps going
#   ...optionally disconnect your laptop, grab coffee...
./03_logs.sh  $JOB --status   # quick non-interactive check
./04_fetch.sh $JOB        # once it's done, rsync logs/ and data back
```

## Postprocessing

Result-processing scripts are numbered `06+` and run **locally** against
`./remote_results/<job>/`. Each one is a thin bash dispatcher that reads
processor-specific keys from the job `.conf` and invokes a Python script
under `tools/postprocess/`.

### `06_compare_direct_disguise.sh` — direct vs disguise Excel

For benchmarks with paired `_direct` / `_disguise` task variants (e.g.
Dynamic Graph Benchmark), this writes one workbook to
`./remote_results/<job>/processed/compare_direct_disguise.xlsx`:

- a `main` sheet with one row per pair (sample count, direct accuracy,
  disguise accuracy, paired accuracy = both correct on the same `doc_id`);
- one sheet per pair with one row per `doc_id` (target, both responses,
  both correctness flags, `both_correct`) plus a final TOTAL row.

The job `.conf` declares the mapping:

```bash
COMPARE_PAIRS=(
    "coloring:dynamic_graph_benchmark_coloring_direct:dynamic_graph_benchmark_coloring_disguise"
    "directed_connectivity:dynamic_graph_benchmark_directed_connectivity_direct:dynamic_graph_benchmark_directed_connectivity_disguise"
)
```

Run after `04_fetch.sh`:

```bash
./remote_execution_scripts/06_compare_direct_disguise.sh \
    dynamic_graph_benchmark_coloring_directed_qwen25vl_3b
# pin a specific run if multiple timestamps exist:
./remote_execution_scripts/06_compare_direct_disguise.sh \
    dynamic_graph_benchmark_coloring_directed_qwen25vl_3b --timestamp 20260505_055447
```

By default the latest timestamp covering every referenced task is used.

### `07_batch_report.sh` — one xlsx per batch

After `run_batch.sh` (or any sequence of `04_fetch.sh` calls) has populated
`remote_results/<job>/` for every job in a batch, generate one cross-job
workbook:

```bash
./remote_execution_scripts/07_batch_report.sh -f remote_execution_scripts/batches/ablation_labels.txt
# pin a sample timestamp for every job in the batch:
./remote_execution_scripts/07_batch_report.sh \
    -f remote_execution_scripts/batches/standard_4b.txt --timestamp 20260505_055447
# abort instead of marking missing jobs NO_DATA:
./remote_execution_scripts/07_batch_report.sh -f remote_execution_scripts/batches/sweep_nodes.txt --strict
```

Output lands in `remote_results/_batch_reports/<batch>_<gen_ts>.xlsx`. The
`<gen_ts>` filename suffix is the **report generation time**, so re-running
07 after a partial re-fetch never overwrites a prior report. A
`<batch>_latest.xlsx` copy next to it is refreshed to point at the most
recent run.

Workbook layout:

- **`summary`** — one row per job: `job`, `axis`, `axis_value`, `model`,
  `model_pretrained`, `samples_ts`, `n_samples`, then per base-task accuracy
  columns (`<task>_direct_acc` / `<task>_disguise_acc` / `<task>_paired_acc`),
  ending with overall accuracy across all pairs. Jobs without fetched data
  show up as `samples_ts=NO_DATA` rows so a partial report is still useful.
- **One tab per job** — paired jobs (every standard / ablation_* batch) get
  the same `consolidated` table that `06_compare_direct_disguise.sh` writes
  (one row per task pair). Sweep jobs (`sweep_nodes` / `sweep_edges`) get
  long-form `(base_task, variant, x, n, accuracy)` rows so the curve
  information is preserved.

## Batches

Run a whole group of jobs end-to-end with `run_batch.sh -f`. Each manifest
under `batches/` lists the jobs in one ablation axis:

```bash
cd remote_execution_scripts

# regenerate confs + manifests after editing the generator
python jobs/generate_graph_benchmark_jobs.py

# run a batch (default: stop on first failure)
./run_batch.sh -f batches/standard_4b.txt

# typical overnight form: tolerate per-job failures, poll less often
./run_batch.sh -f batches/ablation_labels.txt --keep-going --poll 60

# skip the trailing aggregated xlsx (07_batch_report.sh) — useful if you'll
# re-fetch later and only want one final report
./run_batch.sh -f batches/sweep_nodes.txt --no-report
```

`run_batch.sh` calls `07_batch_report.sh` once the queue is done (whether
every job succeeded or some failed — missing/failed jobs show up as
`NO_DATA` rows in the report).

| Manifest | Jobs | Axis | n / job |
|---|---|---|---|
| `standard_4b.txt` | 4 | baseline (numeric labels, default color, no adj matrix) | 5000 |
| `ablation_labels.txt` | 8 | label_style ∈ {letters, none} × 4 models | 500 |
| `ablation_color.txt` | 4 | node_color = `#F1948A` | 500 |
| `ablation_adjmatrix.txt` | 4 | INCLUDE_ADJ_MATRIX=1 | 500 |
| `ablation_thinking.txt` | 2 | Qwen3-VL-{4B,8B}-Thinking SKUs | 500 |
| `ablation_model_size.txt` | 6 | larger panel: 8B–11B mixed family | 500 |
| `sweep_nodes.txt` | 4 | vertex sweep (3..14, 250/value) | n/a |
| `sweep_edges.txt` | 4 | edge sweep (3,5,8,12,18,25,35, 100/value) | n/a |

After every job in a batch has been fetched, run
`./07_batch_report.sh -f batches/<name>.txt` to consolidate the run into one
xlsx (see above).

## Resumable chunked runs (dynamic_graph_benchmark)

Long monolithic runs are fragile — a single mid-run CUDA OOM, vllm engine
hiccup, or VM reboot loses everything. The `dynamic_graph_benchmark` runner
(`examples/models/dynamic_graph_benchmark/run_eval.sh`) supports a
**chunked**, **resumable** mode driven by one env var on the job conf:

```bash
export CHUNK_SIZE=2000   # rows per lmms-eval invocation; 0/unset = monolithic
```

When `CHUNK_SIZE > 0`:

1. `tools/prepare_dynamic_graph_benchmark.py` writes the dataset **and**
   pre-shards it under `${DATASET_DIR}/chunks/chunk_NNNN/` plus a
   `chunks.toc.json` index. Chunk boundaries are pair-aligned so a
   `direct`/`disguise` pair is never split. A `prepare_meta.json`
   fingerprint at `${DATASET_DIR}/prepare_meta.json` makes the prepare
   step **idempotent**: a re-run with identical args reuses the on-disk
   dataset without regenerating.
2. `run_eval.sh` iterates the TOC, points `./dynamic_graph_benchmark_data`
   at each chunk, and invokes `lmms-eval` into a chunk-scoped output
   dir `logs/${JOB_NAME}/chunks/chunk_NNNN/`. Each chunk writes a
   completion sentinel `${RUN_DIR}/chunks/chunk_NNNN.status` (`done` /
   `failed:<rc>` / `in_progress`).
3. After every chunk is `done`, `tools/postprocess/merge_chunked_run.py`
   concatenates the per-chunk `*_samples_<task>.jsonl` files into
   `logs/${JOB_NAME}/<model_dir>/<merge_ts>_samples_<task>.jsonl`, so
   `06_compare_direct_disguise.sh` and `07_batch_report.sh` see the
   single-timestamp shape they expect.

**Resume on failure.** The mechanism is just `./02_run.sh <job>` again.
The fresh tmux session re-enters `run_eval.sh`; prepare sees its
fingerprint matches and exits as a no-op; the chunk loop skips chunks
already marked `done` and retries the first failed/pending one.
Per-chunk lmms-eval invocations pay one vllm warmup each, so
`CHUNK_SIZE` should be picked to amortise that (the generator uses
2000 for standard runs, 500 for ablations).

**Forcing a full restart.** Change any prepare fingerprint input (e.g.
bump `SEED`, change `NUM_SAMPLES`, edit a knob) — prepare detects the
mismatch, regenerates the dataset, and clears stale `*.status` files
under `${RUN_DIR}/chunks/`. Wiping `${DATASET_DIR}/prepare_meta.json`
manually forces the same effect.

The generator (`jobs/generate_graph_benchmark_jobs.py`) sets a sensible
`CHUNK_SIZE` for every batch class — see the `CHUNK_*` constants at the
top of the file.

## How it works

- **Session naming.** The tmux session is `lmms_<job_name>` so only one run per
  job can exist at a time. `02_run.sh` refuses to start if a session is alive;
  use `./05_stop.sh <job>` first.
- **Logs.** The remote command runs inside tmux, tee'd to
  `$REMOTE_RUNS_DIR/<job>/run.log`, so both `tail -F` and `tmux attach` work. An
  `exit_code` sentinel is written on completion so `04_fetch.sh` can report
  success / failure.
- **Results.** `04_fetch.sh` refuses to pull while the session is alive (pass
  `--force` for a mid-run snapshot). Results land in
  `./remote_results/<job>/` alongside a `.run/` subdir holding the log and
  exit-code sentinel.
- **Deploy.** `01_deploy.sh` rsyncs two trees:
  - `UPLOAD_PATHS` (relative to the local repo root) → `REMOTE_WORKDIR`.
  - `DATASET_UPLOAD_PATHS` (relative to `LOCAL_DATASET_ROOT`, the sibling
    `dynamic-dataset` repo) → `REMOTE_DATASET_DIR`. Only jobs that call
    `tools/prepare_dynamic_graph_*.py` need this — omit the array otherwise.

  Then it runs `REMOTE_SETUP_CMD` (e.g. `uv sync`) with `UV_CACHE_DIR` and
  `HF_HOME` pointed at the SSD, so the download cache doesn't fill `/home`.
  Setup output streams live (no tmux).
- **Run.** `02_run.sh`'s launcher exports `HF_HOME` / `UV_CACHE_DIR`, cds into
  `REMOTE_WORKDIR`, and sources `.venv/bin/activate` before executing
  `REMOTE_RUN_CMD`. The `[env]` banner in `run.log` records the resolved paths
  and `which python` so you can verify the right venv is active.

## Adding a new job

1. Copy `jobs/example.conf` to `jobs/<your_job>.conf`.
2. Fill in:
   ```bash
   UPLOAD_PATHS=( "lmms_eval" "tools" "pyproject.toml" "uv.lock" )
   # Optional — only for jobs that need the sibling dynamic-dataset repo:
   DATASET_UPLOAD_PATHS=( "src" "pyproject.toml" "uv.lock" )
   REMOTE_SETUP_CMD="cd '$REMOTE_WORKDIR' && uv sync"
   REMOTE_RUN_CMD="bash examples/models/<foo>/<bar>.sh"
   RESULT_PATHS=( "logs/<foo>" )
   ```
3. `./01_deploy.sh <your_job> && ./02_run.sh <your_job>`.

`REMOTE_RUN_CMD` doesn't need its own `cd` or venv activation — the launcher
handles both. That's it — no top-level script edits.

## Config precedence

Everything in `config.sh` is overridable via env var, e.g.:

```bash
VM_HOST=10.0.0.5 REMOTE_WORKDIR=/scratch/lmms ./01_deploy.sh <job>
```

Useful when you have multiple VMs and don't want to keep editing `config.sh`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied (publickey)` | `SSH_KEY` unset or wrong permissions. On WSL use `source ./00_wsl_fix_key.sh`. |
| `tmux: not found` on the VM | `ssh <vm> sudo apt-get install -y tmux` once. |
| `rsync: command not found` | Install rsync locally. `sudo apt install rsync` on WSL. |
| 02_run refuses to start | An old session exists: `./05_stop.sh <job>` then retry. |
| Log appears frozen but job is running | Some programs buffer heavily; use `./03_logs.sh <job> --attach` to confirm, or rely on tqdm progress bars inside tmux. |
| Results not fetched — "not present on VM" | The run likely failed before producing that path. Inspect `remote_results/<job>/.run/run.log`. |
