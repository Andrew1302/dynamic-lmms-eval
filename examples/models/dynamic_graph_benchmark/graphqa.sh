#!/bin/bash
# Evaluate a Qwen-VL model on the Dynamic Graph Benchmark restricted to the
# coloring + directed_connectivity + shortest_path tasks (direct + disguise of
# each).
#
# Run from the dynamic-lmms-eval repo root:
#   bash examples/models/dynamic_graph_benchmark/graphqa.sh \
#        [NUM_SAMPLES] [DIFFICULTY] [MODEL_PRETRAINED]
#
# Requirements:
#   - dynamic-dataset repo at ../dynamic-dataset (sibling of dynamic-lmms-eval)
#   - uv environment with lmms-eval, datasets, networkx, matplotlib, Pillow installed
#
# JOB_NAME is exported by remote_execution_scripts/02_run.sh and drives the
# log output path so the conf's RESULT_PATHS rsync entry always matches what
# this script produces. The fallback lets the script also run standalone.

NUM_SAMPLES=${1:-1000}
DIFFICULTY=${2:-medium}
MODEL_PRETRAINED=${3:-Qwen/Qwen2.5-VL-3B-Instruct}
JOB_NAME=${JOB_NAME:-dynamic_graphqa_benchmark}

# Map the HF pretrained id to the lmms-eval --model registry name. The Qwen 2.5
# and Qwen 3 VL families have separate model classes in lmms_eval.
case "$MODEL_PRETRAINED" in
    *Qwen3-VL*)            MODEL_NAME="qwen3_vl" ;;
    *Qwen2.5-VL*|*Qwen2_5-VL*) MODEL_NAME="qwen2_5_vl" ;;
    *)
        echo "Unknown Qwen VL family for MODEL_PRETRAINED=$MODEL_PRETRAINED" >&2
        exit 1
        ;;
esac

# Step 1: generate the dataset (direct + disguise rows per task), restricted
# to coloring, directed_connectivity, and shortest_path to avoid producing 10k
# unused undirected-connectivity samples.
python tools/prepare_dynamic_graph_benchmark.py \
    --num-samples "$NUM_SAMPLES" \
    --difficulty "$DIFFICULTY" \
    --difficulty-override shortest_path=easy \
    --seed 42 \
    --tasks coloring directed_connectivity shortest_path \
    --output-dir ./dynamic_graph_benchmark_data

# Step 2: full evaluation across the six leaf tasks via the group alias.
# The group `dynamic_graph_benchmark` resolves to coloring_{direct,disguise},
# directed_connectivity_{direct,disguise}, and shortest_path_{direct,disguise}
# per the group YAML.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model "$MODEL_NAME" \
    --model_args "pretrained=$MODEL_PRETRAINED" \
    --tasks dynamic_graph_benchmark \
    --batch_size 1 \
    --log_samples \
    --output_path "./logs/${JOB_NAME}"
