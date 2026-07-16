# Rerun plan — renderer fix + fp8 vision fix + extraction fix (2026-07-16)

Handoff for the executing agent. Everything referenced is committed
(dynamic-dataset: renderer fixes + rng BFS-root coloring fix;
dynamic-lmms-eval: fp8 bf16-vision plugin, InternVL think force-close,
answer extraction, prepare fingerprint code-hash, merged job layout).
Use the `vm-batch-runner` skill for all execution mechanics.

**Merged job layout (Stage 1, 2026-07-16):** χ-controlled coloring
(uniform {2,3,4}, linear 2→3→4 per sample index) is now the prepare
tool's UNCONDITIONAL default, so the old separate `*_coloring_*` jobs no
longer exist — every base job runs all three tasks. Job counts below
are the new (halved) ones. The coloring χ=2 duplicate collapse was also
fixed (rng BFS root in dynamic-dataset), so all coloring graphs differ
from any pre-2026-07-16 fetch.

**Fresh campaign tree (10+):** the rerun files into NEW numbered
campaign dirs `10_standard, 11_sweep_size, 12_abl_adjlist,
13_abl_labels, 14_abl_color, 15_abl_think, 16_abl_thinkadj,
17_abl_scram` (stamped in every conf; the organizer SPECS map job names
there automatically). `remote_results/01_*..09_*` are the untouched
PRE-FIX legacy archive — never file into, read from, or report over
them for the paper. `job_data_dir` resolves a job name present in both
trees to the newest copy (mtime), so reports pick the rerun.

## Why everything below is invalid

1. **Direct-render bug (all models, all campaigns):** straight edges passed
   through collinear nodes — 33–75% (sp), 36–77% (conn), 3–22% (coloring) of
   direct images per difficulty had ≥1 invisible/misattributed edge, and
   ~60% of sp images had a weight label sitting on a foreign edge. Image
   cells were unsolvable in principle. Fixed in dynamic-dataset (arc
   routing + layer stagger + label placement).
2. **fp8 vision blindness (Qwen + InternVL, campaigns 07/08/09 only):** vllm
   online fp8 quantized the ViT → the models never saw the image. Fixed via
   the `fp8_keep_bf16` vllm plugin (run_eval.sh sets FP8_KEEP_BF16_PATTERNS
   automatically).
3. Riding along (no extra cost): InternVL native think force-close
   (INTERNVL_FORCE_CLOSE=1 default), robust answer extraction in utils.py.

Every job regenerates its dataset automatically on first run: the prepare
fingerprint now includes a hash of the `src.benchmark` sources, so the
deployed renderer change forces regeneration. **Verify this on the first
job**: run.log must show prepare REGENERATING, not "fingerprint matches —
skipping". If it skips, the deploy did not reach the VM — stop and fix.

## Execution order

| # | Campaign (10+ tree) | Batch manifest(s) | Jobs | Models | Why |
|---|---------------------|-------------------|------|--------|-----|
| 0 | smoke gate (_scratch) | `fp8smoke_vm03.txt` (+`_rest`) | 24 tiny (n=2/task) | Qwen, InternVL | validate NEW images end-to-end before burning GPU-days; includes sp-solvability hand-check (see below) |
| 1 | 15_abl_think | `think_ablation_vm02.txt` + `think_ablation_vm03.txt` | 18 (n=100, all 3 tasks each) | all 3 | render fix + fp8 blindness + force-close + extraction |
| 2 | 10_standard | `standard.txt` | 9 (n=500, all 3 tasks each; includes the old 03 coloring content) | all 3 | render fix; sp accuracy was suspiciously low — now measurable on readable images |
| 3 | 16_abl_thinkadj | `thinkadj_ablation_vm02.txt` + `_vm03.txt` | 18 (n=100) | all 3 | fp8 blindness invalidated the whole design (blind image arm degenerates to scramble); render fix |
| 4 | 12_abl_adjlist | `ablation_adjlist.txt` | 9 | all 3 | image arm of img-vs-img+adj comparison was corrupted |
| 5 | 11_sweep_size | `sweep_nodes.txt`, `sweep_edges.txt` | 6 | all 3 | direct images |
| 6 | 13_abl_labels | `ablation_labels.txt` (or per-style `_letters`/`_none`) | 18 | all 3 | direct images |
| 7 | 14_abl_color | `ablation_color.txt` | 9 | all 3 | direct images |
| 8 | 17_abl_scram | `scram_ablation_vm02.txt` + `_vm03.txt` — Qwen + InternVL jobs only | 12 of 18 | Qwen, InternVL | fp8 fix + control consistency. Gemma scram jobs OPTIONAL (scrambled pixels carry no structure either way) |
| — | 16 (_inc leaves) | `thinkadj500inc_*.txt` | 18 | all 3 | DEFERRED (tier-4 n=500 increments) — only if time remains |

NOT rerun: nothing else. Old sp-disguise-map results were faithful (the map
change is readability polish), but they get replaced anyway because every
job bundles direct+disguise. There is no separate coloring campaign anymore
(χ-control is the default; 03_coloring_chi is legacy/archived). `cat` each
manifest before dispatch to confirm the job list — do not trust this table
over the manifest.

## Step 0 in detail (the smoke gate)

1. Deploy one smoke job per VM (`01_deploy.sh`; run_batch also deploys) —
   this rsyncs BOTH repos (UPLOAD_PATHS + DATASET_UPLOAD_PATHS).
2. Run the `fp8smoke` batch. Confirm in run.log: (a) prepare regenerates
   the dataset (code-hash mismatch), (b) the fp8 runs log the
   `fp8_keep_bf16` plugin load, (c) final results table present.
3. Organize with `tools/postprocess/organize_fp8smoke_pairs.py`, then
   hand-verify: every sp direct image's edges readable against
   adjacency.txt (this is the user's "are the tasks solvable" validation);
   think traces reference real image content; no blindness signatures
   ("storage bags", generic scene descriptions).
4. Only then start batch #1.

## Operational cautions (from prior burns)

- Split big batches across vm02 + vm03 (the `_vm02`/`_vm03` manifests
  exist for 07/08; for single manifests, split or alternate). Never run
  the SAME job id on both VMs — fetches overwrite the same local dir.
- `run_batch.sh --keep-going --poll 60` in background via WSL; wait for the
  notification, never sleep-poll. lmms-eval exit code 0 LIES — check the
  results table / `_samples_*.jsonl` exist per job.
- Resume after a mid-run crash = just `./02_run.sh <job>` again (chunk
  sentinels skip done chunks). Read the failed CHUNK's log, not run.log.
- vm02's SSD is shared and near-full: `df -h` before n=500 datasets;
  never delete colleagues' data.
- After each batch: `04_fetch.sh` (run_batch does it), organizer files
  results into `remote_results/<NN_campaign>/...`, then
  `07_batch_report.sh` (reports must include the run_info tab).
- Spot-check one think-arm jsonl per model per batch with
  `verify_thinking.py` (reasoning present, budgets sane, answers extracted).
