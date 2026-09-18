#!/bin/bash
# =============================================================================
# Dynamic Graph Benchmark runner.
#
# This script is ORCHESTRATION ONLY: prepare the dataset, ask prompting/plan.py
# what to run, run it, merge the chunks. Every *decision* -- backend selection,
# VRAM tuning, the per-model thinking mechanism, token budgets, prompt text --
# lives in prompting/models.py and prompting/plan.py, where it is unit-tested
# against argv captured from the previous version of this script
# (test/eval/test_run_eval_plan.py).
#
# The conf contract is unchanged, so every jobs/*.conf keeps working.
#
# Required env vars:
#   MODEL_PRETRAINED   HF identifier
#   NUM_SAMPLES        Generations per task (standard mode)
#   DIFFICULTY         easy|medium|hard
#   DATASET_DIR        Local path the prepared HF dataset is written to
#   JOB_NAME           Logical job id; drives ./logs/$JOB_NAME
#
# Optional (ablation axes):
#   TASKS                  Space-separated task names (default: all three)
#   LABEL_STYLE            numeric|letters|none          NODE_COLOR  hex
#   EDGE_STYLE             straight|curved               SEED        int
#   INCLUDE_ADJ_MATRIX     1 appends a text adjacency list to the prompt
#   DIFFICULTY_OVERRIDES   space-separated "task=level" pairs
#   START_INDEX            first sample index (incremental runs)
#   THINKING               1 enables the model's native reasoning mode
#   PROMPT_ID              prompt template (default direct_v1); see prompting/
#   MODEL_NAME_OVERRIDE    force a backend (e.g. InternVL through vllm)
#   INTERNVL_R1_VARIANT    R1 system-prompt preset for the InternVL think arm
#   INTERNVL_FORCE_CLOSE   0 opts out of the native thinking-token budget
#   CHUNK_SIZE             >0 shards the dataset for resumable runs
#   BATCH_SIZE             override the profile's batch size
#
# Sweep mode: CONSTRAINT, CONSTRAINT_VALUES, SAMPLES_PER_VALUE
# =============================================================================

set -euo pipefail

: "${MODEL_PRETRAINED:?MODEL_PRETRAINED is required}"
: "${NUM_SAMPLES:?NUM_SAMPLES is required (ignored in sweep mode)}"
: "${DATASET_DIR:?DATASET_DIR is required}"
: "${JOB_NAME:?JOB_NAME is required}"
: "${DIFFICULTY:=medium}"
: "${TASKS:=coloring directed_connectivity shortest_path}"
: "${LABEL_STYLE:=numeric}"
: "${NODE_COLOR:=#AED6F1}"
: "${EDGE_STYLE:=straight}"
: "${INCLUDE_ADJ_MATRIX:=0}"
: "${START_INDEX:=0}"
: "${DIFFICULTY_OVERRIDES:=}"
: "${THINKING:=0}"
: "${PROMPT_ID:=direct_v1}"
: "${CONSTRAINT:=}"
: "${CONSTRAINT_VALUES:=}"
: "${SAMPLES_PER_VALUE:=}"
: "${SEED:=42}"
: "${CHUNK_SIZE:=0}"
: "${INTERNVL_FORCE_CLOSE:=1}"

# PROMPT_ID reaches the task hooks through the environment: process_results is
# arity-2 and never receives lmms_eval_specific_kwargs, so this is the only
# transport that reaches the text, visual and scoring hooks alike.
export PROMPT_ID

# --- Step 1: ask for the plan --------------------------------------------------
# One source of truth, shared with run_local_api.py. Fails loudly if a template
# asks for more generation than the model's window can hold.
PLAN_ARGS=(
    --pretrained "$MODEL_PRETRAINED"
    --thinking "$THINKING"
    --tasks "$TASKS"
    --job-name "$JOB_NAME"
    --prompt-id "$PROMPT_ID"
    --force-close "$INTERNVL_FORCE_CLOSE"
)
[ -n "${MODEL_NAME_OVERRIDE:-}" ] && PLAN_ARGS+=(--backend "$MODEL_NAME_OVERRIDE")
[ -n "${INTERNVL_R1_VARIANT:-}" ] && PLAN_ARGS+=(--system-prompt "$INTERNVL_R1_VARIANT")

mapfile -t LMMS_ARGS < <(python -m prompting.plan "${PLAN_ARGS[@]}" --format argv)
while IFS='=' read -r _k _v; do
    [ -n "$_k" ] && export "$_k=$_v"
done < <(python -m prompting.plan "${PLAN_ARGS[@]}" --format env)

echo "[run_eval] job=$JOB_NAME pretrained=$MODEL_PRETRAINED thinking=$THINKING prompt=$PROMPT_ID"
echo "[run_eval]   tasks=$TASKS dataset_dir=$DATASET_DIR chunk_size=$CHUNK_SIZE"
echo "[run_eval]   label_style=$LABEL_STYLE node_color=$NODE_COLOR edge_style=$EDGE_STYLE adj=$INCLUDE_ADJ_MATRIX"
echo "[run_eval]   plan: ${LMMS_ARGS[*]}"

# --- Helpers -------------------------------------------------------------------

_build_prepare_args() {
    local args=(
        --seed "$SEED"
        --tasks $TASKS
        --output-dir "$DATASET_DIR"
        --label-style "$LABEL_STYLE"
        --node-color "$NODE_COLOR"
        --edge-style "$EDGE_STYLE"
    )
    [ "$INCLUDE_ADJ_MATRIX" = "1" ] && args+=(--include-adjacency-matrix)
    local ov
    for ov in $DIFFICULTY_OVERRIDES; do
        args+=(--difficulty-override "$ov")
    done
    if [ -n "$CONSTRAINT" ]; then
        args+=(--constraint "$CONSTRAINT" --constraint-values "$CONSTRAINT_VALUES")
        [ -n "$SAMPLES_PER_VALUE" ] && args+=(--samples-per-value "$SAMPLES_PER_VALUE")
    else
        args+=(--num-samples "$NUM_SAMPLES" --difficulty "$DIFFICULTY" --start-index "$START_INDEX")
    fi
    printf '%s\n' "${args[@]}"
}

_point_canonical_dataset_at() {
    # The task yamls always load ./dynamic_graph_benchmark_data.
    local target_abs="$1" canonical="./dynamic_graph_benchmark_data"
    if [ -L "$canonical" ] || [ -e "$canonical" ]; then
        rm -rf "$canonical"
    fi
    ln -s "$target_abs" "$canonical"
}

_run_lmms_eval() {
    # Trailing flags win in argparse, so the per-chunk output path and a conf's
    # BATCH_SIZE override are simply appended to the plan.
    local overrides=(--output_path "$1")
    [ -n "${BATCH_SIZE:-}" ] && overrides+=(--batch_size "$BATCH_SIZE")
    accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
        "${LMMS_ARGS[@]}" "${overrides[@]}"
}

# --- Step 2: generate the dataset ----------------------------------------------

mapfile -t PREPARE_ARGS < <(_build_prepare_args)

if [ "$CHUNK_SIZE" -gt 0 ]; then
    RUNS_CHUNK_DIR="${RUN_DIR:-./.runs/$JOB_NAME}/chunks"
    mkdir -p "$RUNS_CHUNK_DIR"
    PREPARE_ARGS+=(--chunk-size "$CHUNK_SIZE" --reset-status-dir "$RUNS_CHUNK_DIR")
fi

python tools/prepare_dynamic_graph_benchmark.py "${PREPARE_ARGS[@]}"

# --- Step 3: run ---------------------------------------------------------------

if [ "$CHUNK_SIZE" -le 0 ]; then
    _point_canonical_dataset_at "$(realpath "$DATASET_DIR")"
    _run_lmms_eval "./logs/${JOB_NAME}"
    exit 0
fi

TOC_PATH="$DATASET_DIR/chunks/chunks.toc.json"
if [ ! -f "$TOC_PATH" ]; then
    echo "[run_eval] FATAL: chunks TOC missing at $TOC_PATH despite CHUNK_SIZE=$CHUNK_SIZE" >&2
    exit 2
fi

mapfile -t CHUNK_NAMES < <(python -c "
import json
print('\n'.join(c['name'] for c in json.load(open('$TOC_PATH'))['chunks']))
")
echo "[run_eval] chunked mode: ${#CHUNK_NAMES[@]} chunks (CHUNK_SIZE=$CHUNK_SIZE) - state under $RUNS_CHUNK_DIR"

for chunk in "${CHUNK_NAMES[@]}"; do
    status_file="$RUNS_CHUNK_DIR/${chunk}.status"
    if [ -f "$status_file" ] && [ "$(cat "$status_file")" = "done" ]; then
        echo "[run_eval] $chunk: already done - skipping"
        continue
    fi

    chunk_out="./logs/${JOB_NAME}/chunks/${chunk}"
    # Wipe partial output from a previous failed attempt so the merger does not
    # see two timestamps under one chunk dir.
    rm -rf "$chunk_out"

    echo "in_progress" > "$status_file"
    _point_canonical_dataset_at "$(realpath "$DATASET_DIR/chunks/$chunk")"
    echo "[run_eval] $chunk: running lmms-eval -> $chunk_out"

    if _run_lmms_eval "$chunk_out"; then
        # cli_evaluate catches exceptions and still returns 0, so a chunk can
        # "succeed" having written nothing. Verify before trusting it.
        if [ "$(find "$chunk_out" -name '*_samples_*.jsonl' 2>/dev/null | wc -l)" -eq 0 ]; then
            echo "failed:no_output" > "$status_file"
            echo "[run_eval] $chunk: exited 0 but wrote no samples - re-run ./02_run.sh to retry" >&2
            exit 1
        fi
        echo "done" > "$status_file"
    else
        rc=$?
        echo "failed:$rc" > "$status_file"
        echo "[run_eval] $chunk: exited $rc - aborting; re-run ./02_run.sh to resume" >&2
        exit "$rc"
    fi
done

# --- Step 4: merge the per-chunk logs ------------------------------------------

MERGE_STATUS="$RUNS_CHUNK_DIR/merge.status"
if [ ! -f "$MERGE_STATUS" ] || [ "$(cat "$MERGE_STATUS")" != "done" ]; then
    echo "in_progress" > "$MERGE_STATUS"
    echo "[run_eval] merging ${#CHUNK_NAMES[@]} chunks into ./logs/${JOB_NAME}"
    if python tools/postprocess/merge_chunked_run.py --job-dir "./logs/${JOB_NAME}" --toc "$TOC_PATH"; then
        echo "done" > "$MERGE_STATUS"
    else
        rc=$?
        echo "failed:$rc" > "$MERGE_STATUS"
        echo "[run_eval] merge step exited $rc" >&2
        exit "$rc"
    fi
else
    echo "[run_eval] merge already done - skipping"
fi

echo "[run_eval] all chunks merged; results under ./logs/${JOB_NAME}"
