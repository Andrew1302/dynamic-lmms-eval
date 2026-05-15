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

# Default model_args follow the registry wrapper's convention. The vllm wrapper
# uses `model=` instead of `pretrained=` and needs explicit memory tuning to
# fit on VM03's 12 GiB RTX 4070 (max_model_len cap, eager mode, bf16).
# A conf can set MODEL_ARGS to override the default verbatim.
#
# Thinking control:
#   - vllm wrapper: chat_template_kwargs={"enable_thinking":false} — official
#     Qwen3/vllm mechanism, pre-injects empty <think></think> at template time.
#   - HF wrappers (qwen3_vl etc.): reasoning_prompt="/no_think" — appends the
#     directive to user content (chat template applied inside the wrapper).
# Thinking-mode ablation uses *-Thinking model SKUs; we detect that suffix and
# omit thinking-disable args so the model reasons freely. The task yaml's
# reasoning_tags + strip_reasoning_tags filter the <think>...</think> block
# (including the close-only shape Qwen3 chat templates produce) before scoring.
case "$MODEL_PRETRAINED" in
    *Thinking*)
        VLLM_NO_THINK=""
        HF_NO_THINK=""
        ;;
    *)
        VLLM_NO_THINK=',chat_template_kwargs={"enable_thinking":false}'
        HF_NO_THINK=",reasoning_prompt=\\n/no_think"
        ;;
esac

# Per-model memory tuning. Most fit comfortably; Gemma-4-E2B is mis-marketed —
# "Effective 2 B" but its total weight on disk is ~9.6 GB, leaving < 1 GiB for
# KV cache on the 12 GiB RTX 4070. Tighten max_model_len there so the KV cache
# has any room at all, and shrink max_num_seqs (vllm warms up with 256 dummy
# concurrent requests by default — wasted memory since we run batch_size=1).
case "$MODEL_PRETRAINED" in
    *gemma-4-E2B*|*gemma-4-e2b*)
        VLLM_GPU_UTIL=0.97
        VLLM_MAX_MODEL_LEN=4096
        VLLM_MAX_NUM_SEQS=4
        ;;
    *)
        VLLM_GPU_UTIL=0.93
        VLLM_MAX_MODEL_LEN=12288
        VLLM_MAX_NUM_SEQS=256
        ;;
esac

VLLM_DEFAULT_ARGS="model=$MODEL_PRETRAINED,gpu_memory_utilization=$VLLM_GPU_UTIL,max_model_len=$VLLM_MAX_MODEL_LEN,max_num_seqs=$VLLM_MAX_NUM_SEQS,enforce_eager=True,dtype=bfloat16,trust_remote_code=True,limit_mm_per_prompt={\"image\":1}${VLLM_NO_THINK}"
case "$MODEL_NAME" in
    vllm|vllm_chat)
        : "${MODEL_ARGS:=$VLLM_DEFAULT_ARGS}" ;;
    qwen3_vl|qwen2_5_vl|llava_onevision1_5|gemma3)
        # These HF wrappers expose reasoning_prompt as a named __init__ arg.
        : "${MODEL_ARGS:=pretrained=$MODEL_PRETRAINED${HF_NO_THINK}}" ;;
    internvl3_5)
        # InternVL3 dynamic-preprocess tiles each image up to max_num times
        # (default 12 → ~3k vision tokens). On VM03's 12 GiB GPU the 4B/8B
        # variants OOM during attention softmax mid-run; cap at 6 tiles.
        : "${MODEL_ARGS:=pretrained=$MODEL_PRETRAINED,max_num=6}" ;;
    *)
        # Non-reasoning wrappers (minicpm_v, llama_vision):
        # reasoning_prompt isn't accepted and /no_think wouldn't be recognized
        # by the model's chat template anyway — pass nothing.
        : "${MODEL_ARGS:=pretrained=$MODEL_PRETRAINED}" ;;
esac

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

# Thinking-mode ablations need a much larger max_new_tokens than the task yaml's
# default (sized for short non-thinking answers). Override via --gen_kwargs.
EXTRA_LMMS_ARGS=()
case "$MODEL_PRETRAINED" in
    *Thinking*) EXTRA_LMMS_ARGS+=(--gen_kwargs "max_new_tokens=4096") ;;
esac

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model "$MODEL_NAME" \
    --model_args "$MODEL_ARGS" \
    --tasks dynamic_graph_benchmark \
    --batch_size 1 \
    --log_samples \
    --output_path "./logs/${JOB_NAME}" \
    "${EXTRA_LMMS_ARGS[@]}"
