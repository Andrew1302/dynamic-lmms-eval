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
#   SPECIAL_COLORING       1 to plant the coloring task's chromatic number
#                          uniformly across {2,3,4} (linear 2→3→4 per sample);
#                          0/empty keeps the default full-triangulation graphs
#   THINKING               1 enables the model's reasoning mode (per-request,
#                          single checkpoint); 0/empty disables it. Also bumps
#                          max_new_tokens to 4096 so the <think> block can close
#                          before the answer. See the thinking-control block.
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
: "${SPECIAL_COLORING:=0}"
: "${DIFFICULTY_OVERRIDES:=}"
: "${THINKING:=0}"
: "${CONSTRAINT:=}"
: "${CONSTRAINT_VALUES:=}"
: "${SAMPLES_PER_VALUE:=}"
: "${SEED:=42}"

# CHUNK_SIZE controls per-chunk row count for resumable runs. 0/unset keeps
# the original monolithic path: one prepare + one lmms-eval invocation across
# every row. When > 0, the prepare step pre-shards the dataset under
# $DATASET_DIR/chunks/, this script iterates the shards, and a re-invocation
# (after a mid-run failure) skips every chunk already marked done.
: "${CHUNK_SIZE:=0}"

# Build the explicit lmms-eval subtask list from $TASKS. Running the full
# `dynamic_graph_benchmark` group always loads all 6 subtasks (coloring,
# directed_connectivity, shortest_path × direct/disguise); a single-task
# dataset (e.g. the coloring-only re-run) has 0 rows for the others, so
# lmms-eval crashes at load time (`test_doc = self.task_docs[0]` →
# IndexError on an empty filtered split). Requesting only the direct+disguise
# subtasks for the tasks actually present avoids that and is equivalent to the
# group when all three tasks are selected.
_lmms_tasks=()
for _t in $TASKS; do
    _lmms_tasks+=("dynamic_graph_benchmark_${_t}_direct" "dynamic_graph_benchmark_${_t}_disguise")
done
LMMS_TASKS="$(IFS=,; echo "${_lmms_tasks[*]}")"

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
# Thinking control. THINKING=1 turns on the model's reasoning mode; 0 disables
# it. All three panel models toggle thinking on a *single* checkpoint, but the
# mechanism differs per wrapper:
#   - vllm (Qwen3.5, Gemma-4): chat_template_kwargs={"enable_thinking":true|false}
#     — the official flag the chat template reads. Qwen3.5 defaults thinking ON,
#     Gemma-4 defaults OFF, so we always pass it explicitly.
#   - internvl3_5 (HF): a `think=1` model_arg makes the wrapper set the R1
#     system prompt (InternVL3.5's documented thinking trigger).
#   - other HF wrappers (qwen3_vl etc.): reasoning_prompt="/no_think" disables
#     thinking; empty leaves the model reasoning.
# A *-Thinking model SKU (e.g. Qwen3-VL-4B-Thinking) always reasons regardless
# of THINKING. The task yaml's reasoning_tags + strip_reasoning_tags filter the
# <think>...</think> block (including the close-only shape the chat templates
# produce) before scoring.
case "$MODEL_PRETRAINED" in
    *Thinking*) THINKING=1 ;;
esac

if [ "$THINKING" = "1" ]; then
    VLLM_THINK=',chat_template_kwargs={"enable_thinking":true}'
    HF_NO_THINK=""
    INTERNVL_THINK=",think=1"
else
    VLLM_THINK=',chat_template_kwargs={"enable_thinking":false}'
    HF_NO_THINK=",reasoning_prompt=\\n/no_think"
    INTERNVL_THINK=""
fi

# Per-model memory tuning. Most fit comfortably; Gemma-4-E2B is mis-marketed —
# "Effective 2 B" but its total weight on disk is ~9.6 GB, leaving < 1 GiB for
# KV cache on the 12 GiB RTX 4070. Tighten max_model_len there so the KV cache
# has any room at all, and shrink max_num_seqs (vllm warms up with 256 dummy
# concurrent requests by default — wasted memory since we run batch_size=1).
VLLM_QUANT=""   # per-model quantization arg (set for InternVL below); "" = bf16
case "$MODEL_PRETRAINED" in
    *gemma-4-E2B*|*gemma-4-e2b*)
        VLLM_GPU_UTIL=0.97
        VLLM_MAX_MODEL_LEN=4096
        VLLM_MAX_NUM_SEQS=4
        # Thinking needs room for the reasoning chain plus the answer; the 4096
        # window (sized for terse no-think answers) truncates it. Widen it in
        # the thinking arm — this is a pure sequence-length knob (inputs/tiling
        # unchanged), so it preserves comparability. max_num_seqs stays tiny so
        # the KV cache has room on the 12 GiB card; watch for OOM on the smoke.
        if [ "$THINKING" = "1" ]; then
            # Gemma uses sliding-window attention, so a long single request only
            # needs window-sized KV — measured on the 12 GiB card, even a 40k
            # window loads (KV=14.5k tokens, 1.25x concurrency). Use a 16384
            # window (same as InternVL) so the generation budget (12288, below,
            # matching InternVL) has ~4k of prompt headroom.
            VLLM_MAX_MODEL_LEN=16384
        fi
        ;;
    *InternVL3*|*internvl3*)
        # InternVL3.5-4B via vllm, fp8-quantized. In bf16 the 8.88 GiB weights
        # left only a ~9k-token window on the 12 GiB card, which truncated the
        # (faithful, verbose) R1 reasoning ~40% of the time on hard graphs — a
        # physical VRAM limit, not a tuning issue (cpu_offload restored the window
        # but at 1.4 tok/s, ~36x slower — unusable). fp8 halves the weights to
        # ~4.4 GiB, so a 16k window fits (measured KV ~27k tokens) AND runs faster
        # (81 vs 50 tok/s). The vision tower stays bf16; only the LM is fp8.
        VLLM_GPU_UTIL=0.92
        VLLM_MAX_MODEL_LEN=16384
        VLLM_MAX_NUM_SEQS=2
        VLLM_QUANT=",quantization=fp8"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ;;
    *Qwen3.5*)
        # Qwen3.5-4B (dense) is severely KV-starved on the 12 GiB card: measured
        # at a 12288 window, only ~0.44 GiB / ~3.2k tokens of KV remain — so a
        # bf16 16384 window + 12288 think budget does NOT fit (no sliding-window
        # trick like Gemma). fp8 halves the weights (exactly as for InternVL) to
        # make the InternVL-matched 16384-window / 12288-budget config fit. Both
        # arms run fp8 so the think-vs-nothink comparison differs only in thinking.
        VLLM_GPU_UTIL=0.92
        VLLM_MAX_MODEL_LEN=16384
        # fp8 frees enough KV for ~6.77x concurrency at 16384 (measured 31k KV
        # tokens), so unlike InternVL (bf16, KV-tight → 2) Qwen can batch more.
        # 6 keeps well within that headroom while ~3x the throughput of 2.
        VLLM_MAX_NUM_SEQS=6
        VLLM_QUANT=",quantization=fp8"
        # Think arm uses a native thinking-token budget (12288) + 1024 answer
        # allowance = 13312 total; widen the window to 20480 so that plus the
        # image prompt fits (still ≤ the ~31k KV pool, so it loads).
        if [ "$THINKING" = "1" ]; then
            VLLM_MAX_MODEL_LEN=20480
        fi
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ;;
    *)
        VLLM_GPU_UTIL=0.93
        VLLM_MAX_MODEL_LEN=12288
        VLLM_MAX_NUM_SEQS=256
        ;;
esac

# Gemma-4's thinking output wraps reasoning in the special tokens
# <|channel>thought ... <channel|>. Preserve them (skip_special_tokens=False) so
# the task yaml's reasoning_tags can strip the channel block and isolate the
# answer; with the default (True) the delimiters vanish and the answer regex
# reads the reasoning. Only needed for Gemma's thinking arm.
VLLM_SKIP_SPECIAL=""
case "$MODEL_PRETRAINED" in
    *gemma-4-E2B*|*gemma-4-e2b*)
        [ "$THINKING" = "1" ] && VLLM_SKIP_SPECIAL=",skip_special_tokens=False" ;;
esac

# Qwen3.5's no-think arm answers with a verbose worked solution (plain prose, NOT
# a <think> block — enable_thinking=false genuinely suppresses reasoning, verified
# 0/24 <think> in the no-think smoke) that overflows the terse no-think budget.
# Coerce a concise answer with a directive appended to the user turn — only Qwen,
# only no-think (the think arm must stay free to reason). reasoning_prompt lands
# the directive at the end of the user message; the wrapper turns \n into newlines.
VLLM_TERSE=""
case "$MODEL_PRETRAINED" in
    *Qwen3.5*)
        [ "$THINKING" = "0" ] && VLLM_TERSE=',reasoning_prompt=\n\nReply with only the final answer and nothing else. No reasoning or explanation.' ;;
esac

# Qwen think arm: enable vllm's native thinking-token budget. Setting a
# reasoning_parser makes the wrapper build a ReasoningConfig (simple/vllm.py) so
# vllm knows the </think> delimiter to force-close at the budget (set per-request
# via gen_kwargs thinking_token_budget, below). Without this the budget is
# rejected ("thinking_token_budget is set but reasoning_config is not configured").
VLLM_REASONING=""
case "$MODEL_PRETRAINED" in
    *Qwen3.5*)
        [ "$THINKING" = "1" ] && VLLM_REASONING=",reasoning_parser=qwen3" ;;
esac

# Per-family thinking arg for the vllm path. InternVL3.5 has no enable_thinking
# flag — its thinking is triggered by an R1 *system* prompt (passed as the preset
# system_prompt=internvl_r1). Qwen3.5/Gemma-4 use the enable_thinking chat-template
# flag (VLLM_THINK), and Gemma additionally needs skip_special_tokens=False.
case "$MODEL_PRETRAINED" in
    *InternVL3*|*internvl3*)
        # INTERNVL_R1_VARIANT selects the R1 system-prompt preset (internvl_r1 |
        # internvl_r1_v2 | _v3 | _v4) for prompt-tuning experiments; default is
        # the committed internvl_r1.
        VLLM_MODEL_THINK=""
        [ "$THINKING" = "1" ] && VLLM_MODEL_THINK=",system_prompt=${INTERNVL_R1_VARIANT:-internvl_r1}"
        ;;
    *)
        VLLM_MODEL_THINK="${VLLM_THINK}${VLLM_SKIP_SPECIAL}${VLLM_TERSE}${VLLM_REASONING}"
        ;;
esac

VLLM_DEFAULT_ARGS="model=$MODEL_PRETRAINED,gpu_memory_utilization=$VLLM_GPU_UTIL,max_model_len=$VLLM_MAX_MODEL_LEN,max_num_seqs=$VLLM_MAX_NUM_SEQS,enforce_eager=True,dtype=bfloat16,trust_remote_code=True,limit_mm_per_prompt={\"image\":1}${VLLM_MODEL_THINK}${VLLM_QUANT}"
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
        #
        # Even at 6 tiles, medium/hard graphs run ~45 MiB over the edge while
        # ~1 GiB sits "reserved but unallocated" (allocator fragmentation).
        # expandable_segments reclaims that fragmented reserve — it's a pure
        # CUDA-allocator strategy, so model inputs/tiling/outputs are unchanged
        # and accuracy stays comparable to easy and to the other models.
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        : "${MODEL_ARGS:=pretrained=$MODEL_PRETRAINED,max_num=6${INTERNVL_THINK}}" ;;
    *)
        # Non-reasoning wrappers (minicpm_v, llama_vision):
        # reasoning_prompt isn't accepted and /no_think wouldn't be recognized
        # by the model's chat template anyway — pass nothing.
        : "${MODEL_ARGS:=pretrained=$MODEL_PRETRAINED}" ;;
esac

# Batch size. batch_size=1 serializes vllm (one sequence at a time) and is the
# throughput bottleneck for verbose models — Qwen3.5 emits ~200 tok/answer at
# ~42 tok/s, ~4 h per standard job. vllm batches many prompts concurrently, so
# raising it for the vllm path gives a ~5-10x speedup. Greedy decoding
# (temperature=0) is per-sequence identical regardless of batch, so results are
# unchanged. Gemma is capped at its tight max_num_seqs (KV-starved on the 12 GiB
# card); HF wrappers (internvl3_5 etc.) stay at 1 (already fast, untested batched).
# Override with BATCH_SIZE in the conf/env.
case "$MODEL_NAME" in
    vllm|vllm_chat)
        case "$MODEL_PRETRAINED" in
            *gemma-4-E2B*|*gemma-4-e2b*) : "${BATCH_SIZE:=4}" ;;
            # Hand vllm a whole chunk at once. The wrapper blocks on the entire
            # batch, so a small batch is gated by its single longest (1024-tok)
            # generation; a large batch lets vllm's continuous batching keep the
            # KV-limited running set full and amortizes that tail across the
            # chunk. max_num_seqs (256) caps actual concurrency; the rest queue.
            *)                           : "${BATCH_SIZE:=256}" ;;
        esac
        ;;
    *) : "${BATCH_SIZE:=1}" ;;
esac

echo "[run_eval] job=$JOB_NAME model=$MODEL_NAME pretrained=$MODEL_PRETRAINED batch_size=$BATCH_SIZE thinking=$THINKING"
echo "[run_eval]   tasks=$TASKS dataset_dir=$DATASET_DIR"
echo "[run_eval]   lmms_tasks=$LMMS_TASKS"
echo "[run_eval]   label_style=$LABEL_STYLE node_color=$NODE_COLOR edge_style=$EDGE_STYLE adj_matrix=$INCLUDE_ADJ_MATRIX special_coloring=$SPECIAL_COLORING"
if [ -n "$CONSTRAINT" ]; then
    echo "[run_eval]   sweep: constraint=$CONSTRAINT values=$CONSTRAINT_VALUES spv=$SAMPLES_PER_VALUE"
else
    echo "[run_eval]   standard: num_samples=$NUM_SAMPLES difficulty=$DIFFICULTY"
fi

# --- Helpers -------------------------------------------------------------------

_build_prepare_args() {
    # Echoes the prepare CLI args, one per line, to be read with mapfile.
    # Centralises the standard-vs-sweep + ablation knob translation so both
    # the monolithic and chunked code paths produce identical fingerprints.
    local args=(
        --seed "$SEED"
        --tasks $TASKS
        --output-dir "$DATASET_DIR"
        --label-style "$LABEL_STYLE"
        --node-color "$NODE_COLOR"
        --edge-style "$EDGE_STYLE"
    )
    if [ "$INCLUDE_ADJ_MATRIX" = "1" ]; then
        args+=(--include-adjacency-matrix)
    fi
    if [ "$SPECIAL_COLORING" = "1" ]; then
        args+=(--special-coloring)
    fi
    local ov
    for ov in $DIFFICULTY_OVERRIDES; do
        args+=(--difficulty-override "$ov")
    done
    if [ -n "$CONSTRAINT" ]; then
        args+=(--constraint "$CONSTRAINT" --constraint-values "$CONSTRAINT_VALUES")
        if [ -n "$SAMPLES_PER_VALUE" ]; then
            args+=(--samples-per-value "$SAMPLES_PER_VALUE")
        fi
    else
        args+=(--num-samples "$NUM_SAMPLES" --difficulty "$DIFFICULTY")
    fi
    printf '%s\n' "${args[@]}"
}

_point_canonical_dataset_at() {
    # Atomically repoint ./dynamic_graph_benchmark_data → the requested
    # absolute path. The task yamls always load from the canonical name.
    local target_abs="$1"
    local canonical="./dynamic_graph_benchmark_data"
    if [ -L "$canonical" ] || [ -e "$canonical" ]; then
        rm -rf "$canonical"
    fi
    ln -s "$target_abs" "$canonical"
}

_run_lmms_eval() {
    # Invoke lmms-eval into the given --output_path. Extra args (e.g.
    # --gen_kwargs for thinking SKUs) come from EXTRA_LMMS_ARGS in the
    # caller's scope.
    local output_path="$1"
    accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
        --model "$MODEL_NAME" \
        --model_args "$MODEL_ARGS" \
        --tasks "$LMMS_TASKS" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --output_path "$output_path" \
        "${EXTRA_LMMS_ARGS[@]}"
}

# --- Step 1: generate dataset --------------------------------------------------

mapfile -t PREPARE_ARGS < <(_build_prepare_args)

# Chunked mode pre-shards the dataset for resumable runs. RUN_DIR comes from
# 02_run.sh's launcher (it points at $REMOTE_RUNS_DIR/$JOB_NAME); we fall
# back to a sibling-of-cwd path when the script is invoked directly (e.g.
# during local smoke tests).
if [ "$CHUNK_SIZE" -gt 0 ]; then
    RUNS_CHUNK_DIR="${RUN_DIR:-./.runs/$JOB_NAME}/chunks"
    mkdir -p "$RUNS_CHUNK_DIR"
    PREPARE_ARGS+=(--chunk-size "$CHUNK_SIZE" --reset-status-dir "$RUNS_CHUNK_DIR")
fi

python tools/prepare_dynamic_graph_benchmark.py "${PREPARE_ARGS[@]}"

# --- Step 2: run lmms-eval -----------------------------------------------------

# The thinking arm reasons inside <think>...</think> and needs a much larger
# max_new_tokens than the task yaml's default (64) so the reasoning block can
# close before the answer. Override via --gen_kwargs. The no-think arm keeps
# the 64-tok default (terse single-integer / Yes-No answers).
# Answer allowance: tokens reserved for the post-reasoning answer. Standard runs
# use the task-yaml default (64); we use 1024 (>64) so a (possibly verbose)
# answer always fits — reused as the no-think budget AND the think-arm's extra on
# top of the thinking budget.
ANSWER_BUDGET=1024
EXTRA_LMMS_ARGS=()
if [ "$THINKING" = "1" ]; then
    # temperature=0.6 is the official guidance for the panel's thinking modes;
    # do_sample=true is required for the HF wrappers to sample (no-op for vllm).
    THINK_MAXTOK=4096
    THINK_TOKEN_BUDGET=""   # native vllm thinking-token budget (Qwen only); "" = off
    case "$MODEL_PRETRAINED" in
        # InternVL runs fp8 with a 16k window (see memory tuning), so give its
        # verbose R1 reasoning a generous budget that rarely truncates.
        *InternVL3*|*internvl3*) THINK_MAXTOK=12288 ;;
        # Gemma mostly commits within 4096, but coloring-hard over-deliberates:
        # ~14% hit the 4096 cap. Its sliding-window attention makes a wider
        # window cheap, so give it the same 12288 budget as InternVL (fits the
        # 16384 window with ~4k prompt headroom) to capture the reasoning tail.
        *gemma-4-E2B*|*gemma-4-e2b*) THINK_MAXTOK=12288 ;;
        # Qwen3.5 over-deliberates hardest (~40% truncated even at 12288). Use
        # vllm's NATIVE thinking-token budget to force-close </think> at 12288
        # thinking tokens, then ANSWER_BUDGET (1024) more for the answer, so the
        # answer is never lost. Total max_new_tokens = 12288 + 1024 = 13312, which
        # fits the widened 20480 window. Needs reasoning_parser (set above).
        *Qwen3.5*) THINK_TOKEN_BUDGET=12288; THINK_MAXTOK=$((12288 + ANSWER_BUDGET)) ;;
    esac
    _GEN="max_new_tokens=$THINK_MAXTOK,temperature=0.6,do_sample=true"
    [ -n "$THINK_TOKEN_BUDGET" ] && _GEN="$_GEN,thinking_token_budget=$THINK_TOKEN_BUDGET"
    EXTRA_LMMS_ARGS+=(--gen_kwargs "$_GEN")
else
    # No-think arm. Standard/task default is 64; Qwen3.5's answers (coerced terse)
    # fit trivially but we give the ANSWER_BUDGET (1024) so a verbose answer never
    # truncates. Other models keep the task-yaml default (their no-think numbers
    # already landed at 64).
    case "$MODEL_PRETRAINED" in
        *Qwen3.5*) EXTRA_LMMS_ARGS+=(--gen_kwargs "max_new_tokens=$ANSWER_BUDGET") ;;
    esac
fi

if [ "$CHUNK_SIZE" -le 0 ]; then
    # Monolithic path: one symlink, one lmms-eval invocation. Identical to
    # pre-chunking behaviour.
    _point_canonical_dataset_at "$(realpath "$DATASET_DIR")"
    _run_lmms_eval "./logs/${JOB_NAME}"
    exit 0
fi

# Chunked path. Walk the TOC produced by prepare; for each chunk, point the
# canonical dataset symlink at its slice and run lmms-eval into its own
# output dir. Status sentinels under $RUNS_CHUNK_DIR drive resume.

TOC_PATH="$DATASET_DIR/chunks/chunks.toc.json"
if [ ! -f "$TOC_PATH" ]; then
    echo "[run_eval] FATAL: chunks TOC missing at $TOC_PATH despite CHUNK_SIZE=$CHUNK_SIZE" >&2
    exit 2
fi

# Read chunk names + n_chunks via a tiny python helper. Avoids a jq dep.
mapfile -t CHUNK_NAMES < <(python -c "
import json, sys
with open('$TOC_PATH') as f:
    toc = json.load(f)
for c in toc['chunks']:
    print(c['name'])
")
N_CHUNKS="${#CHUNK_NAMES[@]}"
echo "[run_eval] chunked mode: $N_CHUNKS chunks (CHUNK_SIZE=$CHUNK_SIZE) — state under $RUNS_CHUNK_DIR"

for chunk in "${CHUNK_NAMES[@]}"; do
    status_file="$RUNS_CHUNK_DIR/${chunk}.status"
    if [ -f "$status_file" ] && [ "$(cat "$status_file")" = "done" ]; then
        echo "[run_eval] $chunk: already done — skipping"
        continue
    fi

    chunk_out="./logs/${JOB_NAME}/chunks/${chunk}"
    # Wipe any partial output left behind by a previous failed/in-progress run
    # of this chunk so we don't end up with two timestamps under the same
    # chunk dir (would confuse the merger's TOC-order pass).
    rm -rf "$chunk_out"

    echo "in_progress" > "$status_file"
    _point_canonical_dataset_at "$(realpath "$DATASET_DIR/chunks/$chunk")"

    echo "[run_eval] $chunk: running lmms-eval → $chunk_out"
    if _run_lmms_eval "$chunk_out"; then
        # lmms-eval's cli_evaluate catches exceptions and returns 0 even when
        # the chunk crashed mid-run (CUDA OOM, dataset load failure, etc.).
        # Verify at least one *_samples_*.jsonl was written before trusting
        # the success; otherwise mark failed and abort so resume re-runs it
        # instead of silently dropping a chunk.
        jsonl_count=$(find "$chunk_out" -name "*_samples_*.jsonl" 2>/dev/null | wc -l)
        if [ "$jsonl_count" -eq 0 ]; then
            echo "failed:no_output" > "$status_file"
            echo "[run_eval] $chunk: lmms-eval exited 0 but wrote no *_samples_*.jsonl files — treating as failure; re-run ./02_run.sh to retry" >&2
            exit 1
        fi
        echo "done" > "$status_file"
    else
        rc=$?
        echo "failed:$rc" > "$status_file"
        echo "[run_eval] $chunk: lmms-eval exited $rc — aborting; re-run ./02_run.sh to resume" >&2
        exit "$rc"
    fi
done

# --- Step 3: merge per-chunk logs ---------------------------------------------

# Produce one timestamped log dir per (model_dir, task) so the existing
# postprocess scripts (06_compare_direct_disguise.sh, 07_batch_report.sh)
# see a shape identical to a non-chunked run.
MERGE_STATUS="$RUNS_CHUNK_DIR/merge.status"
if [ ! -f "$MERGE_STATUS" ] || [ "$(cat "$MERGE_STATUS")" != "done" ]; then
    echo "in_progress" > "$MERGE_STATUS"
    echo "[run_eval] merging $N_CHUNKS chunks into ./logs/${JOB_NAME}"
    if python tools/postprocess/merge_chunked_run.py \
        --job-dir "./logs/${JOB_NAME}" \
        --toc "$TOC_PATH"; then
        echo "done" > "$MERGE_STATUS"
    else
        rc=$?
        echo "failed:$rc" > "$MERGE_STATUS"
        echo "[run_eval] merge step exited $rc" >&2
        exit "$rc"
    fi
else
    echo "[run_eval] merge already done — skipping"
fi

echo "[run_eval] all chunks merged; results under ./logs/${JOB_NAME}"
