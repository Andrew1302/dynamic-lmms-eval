#!/bin/bash
# Evaluate Qwen2.5-VL-7B-Instruct on the Google GraphQA benchmark (text-only).
#
# Run from the dynamic-lmms-eval repo root:
#   bash examples/models/google_graphqa_qwen25vl.sh
#
# Evaluates 11 graph task configs (zero_shot_test split) as a group.
# To run a single config, replace --tasks google_graphqa with e.g.:
#   --tasks google_graphqa_cycle_check
#
# Sampling: pass a fraction (0.0-1.0) as the first argument to evaluate that
# proportion of each config, or omit for the full dataset.
#   bash examples/models/google_graphqa/qwen25vl_3b.sh 0.1   # 10% of each config
#   bash examples/models/google_graphqa/qwen25vl_3b.sh 0.5   # 50% of each config
#   bash examples/models/google_graphqa/qwen25vl_3b.sh       # full dataset

LIMIT=${0.1:-""}

# Smoke test (optional): first 8 samples of cycle_check only
# python -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056 \
#     --tasks google_graphqa_cycle_check \
#     --batch_size 1 \
#     --limit 8

# Full evaluation across all 12 configs
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks google_graphqa \
    --batch_size 1 \
    --log_samples \
    ${LIMIT:+--limit $LIMIT} \
    --output_path ./logs/google_graphqa_qwen25vl_3b
