#!/bin/bash
KEY=/home/andrew/.ssh/vm_key; HOST=vm03@143.107.165.250
R=/media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval
SSH="ssh -i $KEY -p 5022 -o StrictHostKeyChecking=no $HOST"
start=$(date +%s)
for i in $(seq 1 30); do
  read mem util <<< $($SSH "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)
  thr=$($SSH "grep -oE 'generation throughput: [0-9.]+ tokens/s' $R/.runs/graph_bench_standard_medium_qwen35_4b/run.log 2>/dev/null | tail -1")
  el=$(( ($(date +%s)-start)/60 ))
  if [ "${mem:-0}" -gt 5000 ] 2>/dev/null; then
    echo "ACTIVE after ${el}min: GPU mem=${mem}MiB util=${util}% | $thr"; exit 0
  fi
  ec=$($SSH "cat $R/.runs/graph_bench_standard_medium_qwen35_4b/exit_code 2>/dev/null")
  [ -n "$ec" ] && { echo "JOB ALREADY EXITED ec=$ec after ${el}min"; exit 0; }
  sleep 20
done
echo "STILL IDLE after 10min — GPU mem=${mem}MiB. Possible hang; last log:"
$SSH "tail -6 $R/.runs/graph_bench_standard_medium_qwen35_4b/run.log"
