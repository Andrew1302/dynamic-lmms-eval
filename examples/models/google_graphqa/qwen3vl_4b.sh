#!/bin/bash
# Evaluate Qwen2.5-VL-7B-Instruct on the Google GraphQA benchmark (text-only).
#
# Run from the dynamic-lmms-eval repo root:
#   bash examples/models/google_graphqa_qwen25vl.sh
#
# Evaluates all 12 graph task configs (zero_shot_test split) as a group.
# To run a single config, replace --tasks google_graphqa with e.g.:
#   --tasks google_graphqa_cycle_check
#
# Sampling: set LIMIT to a fraction (0.0-1.0) to evaluate that proportion of
# each config, or leave empty for the full dataset.
#   LIMIT=0.1 bash examples/models/google_graphqa_qwen25vl.sh   # 10% of each config
#   LIMIT=0.5 bash examples/models/google_graphqa_qwen25vl.sh   # 50% of each config
#   bash examples/models/google_graphqa_qwen25vl.sh             # full dataset

# Set to e.g. 0.1 for 10%, or leave empty for all samples
LIMIT="0.025"

export HF_HOME="~/.cache/huggingface"

# Smoke test (optional): first 8 samples of cycle_check only
# python -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056 \
#     --tasks google_graphqa_cycle_check \
#     --batch_size 1 \
#     --limit 8

# Full evaluation across all 12 configs
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
    --tasks google_graphqa \
    --batch_size 1 \
    --log_samples \
    ${LIMIT:+--limit $LIMIT} \
    --output_path ./logs/google_graphqa_qwen3vl_4b
