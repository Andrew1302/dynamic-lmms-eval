#!/bin/bash
# Evaluate Qwen2.5-VL-7B-Instruct on the Dynamic Graph QA benchmark.
#
# Run from the dynamic-lmms-eval repo root:
#   bash examples/models/dynamic_graph_qa_qwen25vl.sh
#
# Requirements:
#   - dynamic-dataset repo at ../dynamic-dataset (sibling of dynamic-lmms-eval)
#   - uv environment with lmms-eval, datasets, networkx, matplotlib, Pillow installed

export HF_HOME="~/.cache/huggingface"

NUM_SAMPLES=${1:-140}

# Step 1: generate the dataset (140 samples = 10 per task, small graphs, fixed seed)
python tools/prepare_dynamic_graph_qa.py \
    --num-samples "$NUM_SAMPLES" \
    --size all \
    --seed 42 \
    --output-dir ./dynamic_graph_qa_data

# Step 2 (optional): smoke test — first 8 samples only, no GPU required
# python -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056 \
#     --tasks dynamic_graph_qa \
#     --batch_size 1 \
#     --limit 8

# Step 3: full evaluation
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks dynamic_graph_qa \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/dynamic_graph_qa_qwen25vl_3b
