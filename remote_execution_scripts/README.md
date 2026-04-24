# Remote Execution Scripts

Run long lmms-eval jobs on a lab VM without keeping your laptop awake. Every
script in this folder is a thin SSH/rsync wrapper around a shared config block
and a per-job declarative `.conf`. The VM runs the actual work inside a tmux
session so you can disconnect, reconnect, and tail logs at will.

## Layout

```
remote_execution_scripts/
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
└── 05_stop.sh   <job>        # kill the tmux session
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

Export your key before running anything:

```bash
export SSH_KEY="$HOME/.ssh/vm_key"
```

On WSL with a Windows-side key (`/mnt/c/Users/…`), use the helper:

```bash
source ./00_wsl_fix_key.sh /mnt/c/Users/Andrew/.ssh/vm_key
```

## Typical flow

```bash
# one-time: point config.sh at your VM if needed (VM_USER, VM_HOST, VM_PORT,
# REMOTE_WORKDIR) — or override inline via env vars.
export SSH_KEY="$HOME/.ssh/vm03_pk"

JOB=dynamic_graph_benchmark_qwen25vl_3b

./01_deploy.sh $JOB       # rsync repo to ~/dynamic-lmms-eval on the VM + uv sync
./02_run.sh   $JOB        # tmux session `lmms_<job>` starts running the eval
./03_logs.sh  $JOB        # live-tail run.log; Ctrl-C detaches, job keeps going
#   ...optionally disconnect your laptop, grab coffee...
./03_logs.sh  $JOB --status   # quick non-interactive check
./04_fetch.sh $JOB        # once it's done, rsync logs/ and data back
```

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
