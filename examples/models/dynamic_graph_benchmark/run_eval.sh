#!/bin/bash
# =============================================================================
# Generic Dynamic Graph Benchmark runner for the ablation campaign.
#
# All configuration is read from environment variables so a single shell
# script can drive standard, ablation, and sweep jobs uniformly. Conf
# files under remote_execution_scripts/jobs/ export these variables
# before sourcing the dispatch.
#
# Required env vars:
#   MODEL_PRETRAINED   HF identifier passed to lmms-eval as pretrained=
#   NUM_SAMPLES        Generations per task (standard mode) or ignored (sweep)
#   DIFFICULTY         easy|medium|hard (ignored in sweep mode)
#   DATASET_DIR        Local path the prepared HF dataset is written to
#   JOB_NAME           Logical job id; drives ./logs/$JOB_NAME output path
#
# Optional env vars (ablation axes):
#   TASKS                  Space-separated task names (default: all 4)
#   LABEL_STYLE            numeric|letters|none (default numeric)
#   NODE_COLOR             hex (default #AED6F1)
#   EDGE_STYLE             straight|curved (default straight)
#   INCLUDE_ADJ_MATRIX     1 to enable, 0/empty to disable
#   DIFFICULTY_OVERRIDES   space-separated "task=level" pairs forwarded raw
#
# Optional env vars (sweep mode):
#   CONSTRAINT             nodes|edges
#   CONSTRAINT_VALUES      e.g. "5..14" or "3,6,9,12"
#   SAMPLES_PER_VALUE      defaults to 250 (nodes) or 100 (edges)
#
# Required only when an lmms-eval task group YAML doesn't already point
# at DATASET_DIR; we always symlink ./dynamic_graph_benchmark_data ->
# $DATASET_DIR so the existing task YAMLs Just Work.
# =============================================================================

set -euo pipefail

: "${MODEL_PRETRAINED:?MODEL_PRETRAINED is required}"
: "${NUM_SAMPLES:?NUM_SAMPLES is required (ignored in sweep mode)}"
: "${DIFFICULTY:=medium}"
: "${DATASET_DIR:?DATASET_DIR is required}"
: "${JOB_NAME:?JOB_NAME is required}"
: "${TASKS:=coloring directed_connectivity shortest_path}"
: "${LABEL_STYLE:=numeric}"
: "${NODE_COLOR:=#AED6F1}"
: "${EDGE_STYLE:=straight}"
: "${INCLUDE_ADJ_MATRIX:=0}"
: "${DIFFICULTY_OVERRIDES:=}"
: "${CONSTRAINT:=}"
: "${CONSTRAINT_VALUES:=}"
: "${SAMPLES_PER_VALUE:=}"
: "${SEED:=42}"

# Map HF pretrained id to lmms-eval --model registry name.
case "$MODEL_PRETRAINED" in
    *Qwen3-VL*)                   MODEL_NAME="qwen3_vl" ;;
    *Qwen2.5-VL*|*Qwen2_5-VL*)    MODEL_NAME="qwen2_5_vl" ;;
    *InternVL3*|*internvl3*)      MODEL_NAME="internvl3_5" ;;
    *LLaVA-OneVision-1.5*)        MODEL_NAME="llava_onevision1_5" ;;
    *MiniCPM-V*)                  MODEL_NAME="minicpm_v" ;;
    *Llama-3.2-*-Vision*|*llama-3.2-*-vision*) MODEL_NAME="llama_vision" ;;
    *Qwen3.5*|*gemma-4*)          MODEL_NAME="vllm" ;;
    *)
        echo "Unknown model family for MODEL_PRETRAINED=$MODEL_PRETRAINED" >&2
        echo "Set MODEL_NAME explicitly to override." >&2
        exit 1
        ;;
esac
MODEL_NAME="${MODEL_NAME_OVERRIDE:-$MODEL_NAME}"

echo "[run_eval] job=$JOB_NAME model=$MODEL_NAME pretrained=$MODEL_PRETRAINED"
echo "[run_eval]   tasks=$TASKS dataset_dir=$DATASET_DIR"
echo "[run_eval]   label_style=$LABEL_STYLE node_color=$NODE_COLOR edge_style=$EDGE_STYLE adj_matrix=$INCLUDE_ADJ_MATRIX"
if [ -n "$CONSTRAINT" ]; then
    echo "[run_eval]   sweep: constraint=$CONSTRAINT values=$CONSTRAINT_VALUES spv=$SAMPLES_PER_VALUE"
else
    echo "[run_eval]   standard: num_samples=$NUM_SAMPLES difficulty=$DIFFICULTY"
fi

# --- Step 1: generate dataset --------------------------------------------------

PREPARE_ARGS=(
    --seed "$SEED"
    --tasks $TASKS
    --output-dir "$DATASET_DIR"
    --label-style "$LABEL_STYLE"
    --node-color "$NODE_COLOR"
    --edge-style "$EDGE_STYLE"
)
if [ "$INCLUDE_ADJ_MATRIX" = "1" ]; then
    PREPARE_ARGS+=(--include-adjacency-matrix)
fi
for ov in $DIFFICULTY_OVERRIDES; do
    PREPARE_ARGS+=(--difficulty-override "$ov")
done

if [ -n "$CONSTRAINT" ]; then
    PREPARE_ARGS+=(--constraint "$CONSTRAINT" --constraint-values "$CONSTRAINT_VALUES")
    if [ -n "$SAMPLES_PER_VALUE" ]; then
        PREPARE_ARGS+=(--samples-per-value "$SAMPLES_PER_VALUE")
    fi
else
    PREPARE_ARGS+=(--num-samples "$NUM_SAMPLES" --difficulty "$DIFFICULTY")
fi

python tools/prepare_dynamic_graph_benchmark.py "${PREPARE_ARGS[@]}"

# Symlink the per-job dataset dir to the canonical path the lmms-eval
# task YAMLs reference. Cheaper than parameterising every YAML.
CANONICAL_DATASET="./dynamic_graph_benchmark_data"
if [ -L "$CANONICAL_DATASET" ] || [ -e "$CANONICAL_DATASET" ]; then
    rm -rf "$CANONICAL_DATASET"
fi
ln -s "$(realpath "$DATASET_DIR")" "$CANONICAL_DATASET"

# --- Step 2: run lmms-eval -----------------------------------------------------

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model "$MODEL_NAME" \
    --model_args "pretrained=$MODEL_PRETRAINED" \
    --tasks dynamic_graph_benchmark \
    --batch_size 1 \
    --log_samples \
    --output_path "./logs/${JOB_NAME}"
