#!/bin/bash
OUT="/mnt/c/Users/Andrew/AppData/Local/Temp/claude/C--Users-Andrew-Msc-dynamic-lmms-eval/7c571ee9-90e5-4111-9e7e-53e5a310bd45/tasks/bnol0b981.output"
start=$(date +%s)
for i in $(seq 1 120); do
  # how many jobs have we started, and which is latest?
  latest=$(grep -oP '(?<=starting: graph_benchmark/)\S+' "$OUT" 2>/dev/null | tail -1)
  nstarted=$(grep -c 'starting: graph_benchmark/' "$OUT" 2>/dev/null)
  el=$(( ($(date +%s)-start)/60 ))
  if echo "$latest" | grep -q 'medium_qwen35'; then
    echo "RESUME OK after ${el}min: reached job '$latest' (jobs 1-4 fast-skipped). nstarted=$nstarted"; exit 0
  fi
  if echo "$latest" | grep -qE 'medium_gemma|hard_'; then
    echo "PAST medium_qwen35 after ${el}min: latest='$latest' nstarted=$nstarted"; exit 0
  fi
  sleep 30
done
echo "TIMEOUT after 60min: latest='$latest' nstarted=$nstarted (jobs 1-4 may be re-running — investigate)"
