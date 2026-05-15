---
name: vm-batch-runner
description: Run lmms-eval jobs and batches on the remote GPU VM, monitor them, diagnose failures, smoke-test fixes on a small sample, then redeploy. Use when the user asks to "run the batch", "kick off X on the VM", "check that run", "retry/fix and rerun", or any combination of remote run → monitor → fix → re-run. Covers the deploy/run/fetch/stop pipeline under `remote_execution_scripts/` without hard-coding its current command surface — the skill points the agent at the scripts to inspect.
---

# Running Things on VM03 (or any equivalent GPU VM)

This skill makes you able to **run, monitor, diagnose, fix, smoke-test, and re-run** lmms-eval workloads on the remote VM without supervision. The goal is autonomous iteration: when something fails, identify the root cause, apply the smallest fix, validate with a few samples, then redeploy the real workload.

## When to Use

Trigger on requests like:
- "run the batch / kick off `X.txt` / dispatch `<job-id>`"
- "watch the run / wait for it to finish"
- "it failed — fix and rerun"
- "smoke test this fix"
- one-off ad-hoc commands on the VM ("ssh in and check disk", "purge the cache")

If the user just wants to inspect *local* results that have already been fetched, this skill isn't needed.

## Mental Model — The 5 Tools Under `remote_execution_scripts/`

You should **never assume the current flag surface**. Each task, read the script's header comment first to see the exact options it accepts. The scripts always live in `remote_execution_scripts/`:

| Script | Purpose | What to verify in the header before invoking |
|---|---|---|
| `01_deploy.sh <job>` | rsync repo → VM, run `REMOTE_SETUP_CMD` | What `UPLOAD_PATHS` / `DATASET_UPLOAD_PATHS` are; whether `uv sync` is implied |
| `02_run.sh <job>` | Generate launcher, start detached `tmux` session | What env vars get forwarded; how the launcher activates `.venv` |
| `03_logs.sh <job> [--tail N\|--status\|--until-done [POLL_S]\|--attach]` | Stream / status / wait-for-done | Whether `--until-done` blocks (yes, that's how we wait); what its exit code reflects |
| `04_fetch.sh <job> [--force]` | rsync results from VM → `remote_results/<job>/` | Which `RESULT_PATHS` get pulled; whether it refuses while tmux is still running |
| `05_stop.sh <job>` | Kill the tmux session | (no surprises) |
| `run_batch.sh [-f manifest.txt] [<job>...] [--keep-going] [--poll N] [--no-report]` | deploy → run → wait → fetch sequentially | What `--keep-going` vs default does; whether `07_batch_report.sh` is auto-invoked at the end |
| `07_batch_report.sh <batch-name> [...]` | Aggregate batch results into an xlsx | Whether it shells out to `uv run` (yes — `lib/common.sh` prepends `~/.local/bin` to `PATH` so uv resolves in non-interactive WSL) |

Job ids are paths under `remote_execution_scripts/jobs/`, e.g. `graph_benchmark/graph_bench_ablation_adjmatrix_qwen35_4b`. Batch manifests are under `remote_execution_scripts/batches/`.

**Before running anything, read the script's leading comment block.** The exact flag names and behaviors drift; don't burn time on a wrong-flag error.

## Environment / Where Things Live

- **You are on Windows.** The remote_execution_scripts/ shell scripts depend on `rsync` and POSIX tools that don't exist in Git Bash. **Always wrap invocations with `wsl.exe -- bash -c "..."`** — never try them through git-bash directly. WSL has `rsync`, `uv` (after this session's install), and the SSH key at `/home/andrew/.ssh/vm_key` (mounted to `/mnt/c/...` for the repo).
- **VM connection:** see `remote_execution_scripts/config.sh` for `VM_USER`, `VM_HOST`, `VM_PORT`. Don't hard-code these — they may move. SSH key path is in `.env`. For ad-hoc `ssh ...` calls from Windows directly, use `~/.ssh/vm_key` and the port from `config.sh`.
- **On VM03 specifically (verify against current `[[vm03_layout]]` memory):**
  - Repo: `/media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval`
  - SSD-backed caches set by the launcher in `02_run.sh`: `HF_HOME`, `UV_CACHE_DIR`, `XDG_CACHE_HOME`, `TRITON_HOME`, `TRITON_CACHE_DIR`, `TMPDIR`, `FLASHINFER_WORKSPACE_BASE` — all under `/media/vm03/ssd1T/andrew/`. The `/home` partition is small; never let model/vllm/triton caches land there.
  - `uv` lives at `~/.local/bin/uv` (NOT in `.venv`). Non-interactive `ssh vm03 'cmd'` calls need `export PATH="$HOME/.local/bin:$PATH"` before invoking `uv`.

## Core Loop

For any new run, work through these steps. Don't skip; don't reorder.

### 1. Confirm the request scope

- Is it a single job, a batch manifest, or an ad-hoc shell command?
- If batch: which manifest exactly? (`cat remote_execution_scripts/batches/<name>.txt` to confirm contents.)
- If the user said "run X", check whether X is a conf path, a model short-name, or a batch name. Job confs are under `remote_execution_scripts/jobs/`; batches under `remote_execution_scripts/batches/`.

### 2. Inspect — never assume

Before invoking any of the suite's scripts, **read its header**:

```bash
head -40 remote_execution_scripts/run_batch.sh   # or whichever script
```

This catches breaking changes to flag names, new flags you should pass, removed flags you'd otherwise pass.

### 3. Launch in the background — get notified, don't poll

Use the Bash tool's `run_in_background: true` with `--until-done` baked in. That way you get exactly one notification when the run ends, and the harness streams nothing in between:

```bash
wsl.exe -- bash -c "cd /mnt/c/Users/Andrew/Msc/dynamic-lmms-eval/remote_execution_scripts \
  && ./run_batch.sh -f batches/<name>.txt --keep-going --poll 60 2>&1"
```

For single jobs: `./02_run.sh <job>` to launch, then a separate background `./03_logs.sh <job> --until-done 60`. (`02_run.sh` returns immediately once tmux is up; the waiter is a separate call.)

**Do not chain `sleep` calls to poll.** The harness blocks long leading sleeps anyway. Trust the notification.

### 4. When the watcher notifies, read the tail of the watcher's output

The `--until-done` mode prints the final 80 lines of `run.log` plus the exit sentinel. Parse for:

- "**No such file**" / "command not found" → setup issue (`uv` not on path, missing dataset, etc.)
- "**CUDA out of memory**" → GPU pressure (see below)
- "**Engine core initialization failed**" → vllm couldn't bring up the engine; look earlier in the log for the *real* cause (KV cache exhausted, sampler warmup OOM, model arch reject, etc.)
- "**ValueError: prompt length X > max_model_len Y**" → `max_model_len` too tight for this dataset
- "**TypeError: ... unexpected keyword argument**" → API drift in vllm or in the wrapper
- "**accuracy: 0.000 ...** while `filtered_resps` is reasoning prose" → the model is in thinking mode; output got truncated before the answer (see "Thinking-mode pitfalls" below)
- exit_code: `0` despite a Python traceback in the tail → lmms-eval swallows errors and exits 0. Treat any traceback in the tail as failure regardless of exit code.

**Important:** `exit_code: 0` from the run sentinel does NOT mean success. Always look at the table at the end of the log (`|Tasks | ... | accuracy |`) — if it's absent, the run died before scoring.

### 5. Diagnose against the catalog below

Most issues fall into a handful of categories. Match symptom → category, apply the fix, then go to step 6.

### 6. Smoke-test the fix before re-running the real workload

Don't burn 1+ hours on the full ablation just to discover the fix doesn't hold. Smoke test pattern:

1. Pick a single conf from the failed batch.
2. Lower its sample count: `sed -i 's/NUM_SAMPLES="[0-9]*"/NUM_SAMPLES="20"/' <conf>` (or whatever the conf's knob is named).
3. Launch just that one job (`./02_run.sh <job>`, then `--until-done` in background).
4. Verify it reaches the final results table.
5. **Restore the original sample count** (or just `python jobs/generate_graph_benchmark_jobs.py` to regenerate — note this wipes any manual overrides).
6. Launch the full batch.

### 7. Fetch and report

After a real run finishes, `./04_fetch.sh <job>` pulls jsonls into `remote_results/<job>/`. For batches, `run_batch.sh` does this automatically; `07_batch_report.sh` then emits an xlsx aggregate.

## Diagnostic Catalog

### CUDA OOM during model load

```
torch.OutOfMemoryError: ... this process has 11.55 GiB memory in use
ERROR ... gpu_model_runner: Failed to load model - not enough GPU memory
```

The model weights don't fit. `gpu_memory_utilization` only controls KV-cache budgeting after weights load — lowering it doesn't fix this. Options in order of effort:

1. Verify the actual weight size on disk: `ssh vm03 'du -sh /media/vm03/ssd1T/andrew/hf_cache/hub/models--<org>--<model>/'`. The "Effective NB" naming on MatFormer models hides the true total parameter count.
2. Pick a smaller variant. Update `MODEL_PRETRAINED` in the conf (or in `jobs/generate_graph_benchmark_jobs.py` if generated) and rename JOB_NAME/files to match.
3. `quantization=fp8` is **not** a load-time fix — vllm loads bf16 weights *first* then quantizes. It helps inference memory, not loading.
4. Tensor parallelism would help, but VM03 has one GPU.

### CUDA OOM during KV cache / sampler warmup

```
ValueError: No available memory for the cache blocks
RuntimeError: CUDA out of memory ... warming up sampler with 256 dummy requests
```

Weights fit, but vllm pre-allocates KV cache + a warmup pool. Knobs (per-model overrides in `run_eval.sh`):
- `gpu_memory_utilization=0.97` — squeeze more headroom out of the budget. 0.98 is the practical ceiling.
- `max_model_len=4096` (or lower) — caps per-sequence KV. Verify your prompts fit (check the longest one).
- `max_num_seqs=4` — shrinks vllm's warmup pool from the 256 default. Safe when `batch_size=1`.

### Disk full on `/home`

```
OSError: [Errno 28] No space left on device: '/home/.../triton/cache/...'
```

Something escaped the SSD redirect. The launcher in `02_run.sh` exports `XDG_CACHE_HOME`, `TRITON_HOME`, `TRITON_CACHE_DIR`, `TMPDIR`, `FLASHINFER_WORKSPACE_BASE` — verify they all still appear there. To free `/home`:

```bash
ssh vm03 'df -h /home && du -sh ~/.cache ~/.triton 2>/dev/null'
# ask the user before deleting; the auto-classifier will refuse rm on shared VMs
ssh vm03 'rm -rf ~/.cache ~/.triton'   # ONLY with explicit user authorization
```

### Wrong model_args kwarg

```
TypeError: EngineArgs.__init__() got an unexpected keyword argument 'pretrained'
```

The vllm wrapper expects `model=`, not `pretrained=`. The mapping happens in `run_eval.sh`'s `MODEL_ARGS` default — check that the vllm/vllm_chat branch uses `model=$MODEL_PRETRAINED` and not the generic `pretrained=` template.

### Thinking-mode pitfalls

Symptom: `filtered_resps` is multi-paragraph reasoning prose, `accuracy ≈ 0.00` across all tasks, throughput drops to 50–80 s/prompt.

The model is in thinking mode and burning the entire `max_new_tokens` budget before reaching an answer. Two principled fixes:

- **Disable thinking** (default for everything except `*-Thinking` SKUs):
  - vllm wrapper: `chat_template_kwargs={"enable_thinking":false}` — pre-injects an empty `<think></think>` block at the prompt level. **This is the reliable Qwen3 mechanism.** The text directive `/no_think` is unreliable on some SKUs.
  - HF wrappers (`qwen3_vl`, etc.): `reasoning_prompt=\n/no_think` — they don't expose `chat_template_kwargs` directly, but accept the directive.
- **Allow thinking, strip from output** (only for `*-Thinking` ablations):
  - Bump `max_new_tokens` via `--gen_kwargs max_new_tokens=4096` (or higher) so the chain-of-thought can finish.
  - `lmms_eval/api/reasoning.py::strip_reasoning_tags` handles the three output shapes: balanced, close-only (Qwen3 chat-template injection), open-only (truncated). Configure tags in the task yaml's `reasoning_tags`.

Verify which mode you're in by inspecting one `filtered_resps` from the latest jsonl on the VM:
```bash
ssh vm03 "head -c 800 /media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval/logs/<job>/<model_dir>/<latest>_samples_*.jsonl | head -3"
```

### Generated conf got overwritten by the user's edits

The generator under `remote_execution_scripts/jobs/generate_graph_benchmark_jobs.py` **overwrites** confs in `jobs/graph_benchmark/` on every run. If you manually tweaked a generated conf (e.g. `NUM_SAMPLES=100` for a smoke test), you must either:

- Restore the original via `python jobs/generate_graph_benchmark_jobs.py` before launching the batch, or
- Make the edit in the generator itself so it survives.

Orphaned confs (e.g. after a model rename) are NOT auto-deleted — remove them with `rm jobs/graph_benchmark/*_<old_short>*.conf` to avoid stale batch references.

### lmms-eval exit code lies

`cli_evaluate` catches exceptions and returns 0 even on hard failures. So:
- `run_batch.sh` may mark a job "OK" when it actually crashed mid-init.
- Always confirm by either (a) checking the final results table is in `run.log`, or (b) checking `logs/<job>/<model_dir>/` exists with `_samples_*.jsonl` files.

## Smoke-Test Recipe

When fixing a bug that's only manifested in a long run, use this short loop:

```bash
# 1. Pick one failing conf, set tiny sample count
sed -i 's/NUM_SAMPLES="[0-9]\+"/NUM_SAMPLES="20"/' \
  remote_execution_scripts/jobs/<path>/<conf>.conf

# 2. Deploy + run + watch in one shot
wsl.exe -- bash -c "cd /mnt/c/Users/Andrew/Msc/dynamic-lmms-eval/remote_execution_scripts \
  && ./01_deploy.sh <job> 2>&1 | tail -3 \
  && ./02_run.sh <job> 2>&1 | tail -3"
# then in a SEPARATE background Bash call:
wsl.exe -- bash -c "cd /mnt/c/Users/Andrew/Msc/dynamic-lmms-eval/remote_execution_scripts \
  && ./03_logs.sh <job> --until-done 30 2>&1"  # run_in_background: true

# 3. On notification: read the watcher output. Look for the final results table.
#    If still failing, diagnose against the catalog and iterate.

# 4. Restore via generator (preferred — keeps state reproducible)
wsl.exe -- bash -c "cd /mnt/c/Users/Andrew/Msc/dynamic-lmms-eval/remote_execution_scripts \
  && python3 jobs/generate_graph_benchmark_jobs.py"

# 5. Launch the real batch
wsl.exe -- bash -c "cd /mnt/c/Users/Andrew/Msc/dynamic-lmms-eval/remote_execution_scripts \
  && ./run_batch.sh -f batches/<name>.txt --keep-going --poll 60 2>&1"  # run_in_background: true
```

## Anti-Patterns To Avoid

- **Do not poll `--status` in a sleep loop.** Use `--until-done` once, in the background, and trust the notification.
- **Do not launch the full batch first to "see if it works."** Smoke-test on one conf with `NUM_SAMPLES=20` first; the feedback loop is 5–15 min instead of multiple hours.
- **Do not assume `exit_code: 0` means success.** Always confirm a results table was emitted.
- **Do not `git-bash` the shell scripts.** Use WSL — git-bash lacks rsync and breaks `01_deploy.sh` silently or noisily.
- **Do not delete VM-side caches without explicit user authorization.** The auto-classifier blocks destructive remote operations; ask first via `AskUserQuestion`, propose the exact command, then run it once approved.
- **Do not `git commit` mid-debugging.** Wait until the user signs off on the final fix; intermediate states are usually wrong.

## When You Need an Ad-Hoc VM Command

For one-off ops (peek at the log, check disk, inspect a model cache, etc.), prefer a direct SSH using the key path from `config.sh`. Wrap in `wsl.exe -- bash -c` if you also need rsync; raw `ssh` from Windows works for pure-ssh commands. Examples:

```bash
# Live tail (cancel with Ctrl-C; run does not stop on cancel)
wsl.exe -- bash -c "cd /mnt/c/Users/Andrew/Msc/dynamic-lmms-eval/remote_execution_scripts \
  && ./03_logs.sh <job> --tail 100"

# Direct read of a partial result file
ssh -i ~/.ssh/vm_key -p 5022 vm03@<host> \
  "head -c 1000 /media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval/logs/<job>/<model_dir>/<latest>_samples_*.jsonl | head -3"
```

## What This Skill Does NOT Cover

- **Model-side or task-side debugging** (wrong prompt, broken `doc_to_visual`, etc.). Use `lmms-eval-guide` instead — it has the right context for the lmms-eval pipeline internals.
- **Local-only evaluation runs** (no VM involved). Run `lmms_eval` directly via `uv run python -m lmms_eval ...`.
- **Generating new ablation matrices**. That's a one-time scripting task in `jobs/generate_graph_benchmark_jobs.py`; do it inline with the user, not via this skill.
