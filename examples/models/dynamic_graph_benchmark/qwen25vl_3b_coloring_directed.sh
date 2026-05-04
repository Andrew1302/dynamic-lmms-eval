#!/bin/bash
# Evaluate Qwen2.5-VL-3B-Instruct on the Dynamic Graph Benchmark restricted to
# the coloring + directed_connectivity tasks (direct + disguise of each).
#
# Run from the dynamic-lmms-eval repo root:
#   bash examples/models/dynamic_graph_benchmark/qwen25vl_3b_coloring_directed.sh
#
# Requirements:
#   - dynamic-dataset repo at ../dynamic-dataset (sibling of dynamic-lmms-eval)
#   - uv environment with lmms-eval, datasets, networkx, matplotlib, Pillow installed

NUM_SAMPLES=${1:-10000}
DIFFICULTY=${2:-medium}

# Step 1: generate the dataset (direct + disguise rows per task), restricted
# to coloring and directed_connectivity to avoid producing 10k unused
# undirected-connectivity samples.
python tools/prepare_dynamic_graph_benchmark.py \
    --num-samples "$NUM_SAMPLES" \
    --difficulty "$DIFFICULTY" \
    --seed 42 \
    --tasks coloring directed_connectivity \
    --output-dir ./dynamic_graph_benchmark_data

# Step 2: full evaluation across the four leaf tasks via the group alias.
# The group `dynamic_graph_benchmark` resolves to coloring_{direct,disguise}
# and directed_connectivity_{direct,disguise} per the group YAML.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks dynamic_graph_benchmark \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/dynamic_graph_benchmark_coloring_directed_qwen25vl_3b
