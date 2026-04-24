#!/bin/bash
# Evaluate Qwen2.5-VL-3B-Instruct on the Dynamic Graph Benchmark (direct + disguise).
#
# Run from the dynamic-lmms-eval repo root:
#   bash examples/models/dynamic_graph_benchmark/qwen25vl_3b.sh
#
# Requirements:
#   - dynamic-dataset repo at ../dynamic-dataset (sibling of dynamic-lmms-eval)
#   - uv environment with lmms-eval, datasets, networkx, matplotlib, Pillow installed

NUM_SAMPLES=${1:-100}
DIFFICULTY=${2:-medium}

# Step 1: generate the dataset (direct + disguise rows per task).
python tools/prepare_dynamic_graph_benchmark.py \
    --num-samples "$NUM_SAMPLES" \
    --difficulty "$DIFFICULTY" \
    --seed 42 \
    --output-dir ./dynamic_graph_benchmark_data

# Step 2 (optional): smoke test a single leaf task.
# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
#     --tasks dynamic_graph_benchmark_connectivity_direct \
#     --batch_size 1 \
#     --limit 4

# Step 3: full evaluation across all 4 leaf tasks via the group alias.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks dynamic_graph_benchmark \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/dynamic_graph_benchmark_qwen25vl_3b
