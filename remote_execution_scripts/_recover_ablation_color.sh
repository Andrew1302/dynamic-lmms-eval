#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

JOB="graph_benchmark/graph_bench_ablation_color_qwen3vl_4b"
SESSION="lmms_graph_bench_ablation_color_qwen3vl_4b"
RUN_DIR="/media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval/.runs/graph_bench_ablation_color_qwen3vl_4b"
SSH_OPTS=(-i /home/andrew/.ssh/vm_key -p 5022 -o ConnectTimeout=10 -o ServerAliveInterval=20 -o StrictHostKeyChecking=no)
HOST="vm03@143.107.165.250"

echo "=== [wait] polling for $SESSION (SSH-retry on 255) ==="
while :; do
    out=$(ssh "${SSH_OPTS[@]}" "$HOST" "tmux has-session -t $SESSION 2>/dev/null && echo ALIVE || echo GONE" 2>&1)
    rc=$?
    if [ $rc -eq 255 ]; then
        echo "[wait] ssh rc=255 transport failure, retry in 30s"
        sleep 30
        continue
    fi
    case "$out" in
        *ALIVE*) sleep 60 ;;
        *GONE*)  echo "[wait] tmux session ended"; break ;;
        *)       echo "[wait] unexpected response: $out"; sleep 30 ;;
    esac
done

echo "=== [tail] final run.log tail ==="
for i in 1 2 3 4 5; do
    if ssh "${SSH_OPTS[@]}" "$HOST" "tail -n 40 $RUN_DIR/run.log; echo ---; if [ -f $RUN_DIR/exit_code ]; then echo exit_code: \$(cat $RUN_DIR/exit_code); else echo exit_code: missing; fi"; then
        break
    fi
    echo "[tail] ssh failed attempt $i, retry 15s"
    sleep 15
done

echo "=== [fetch] qwen3vl_4b ==="
for i in 1 2 3 4 5; do
    if ./04_fetch.sh "$JOB"; then
        break
    fi
    echo "[fetch] failed attempt $i, retry 20s"
    sleep 20
done

echo "=== [batch] internvl35_4b + gemma4_e2b ==="
./run_batch.sh \
    graph_benchmark/graph_bench_ablation_color_internvl35_4b \
    graph_benchmark/graph_bench_ablation_color_gemma4_e2b \
    --keep-going --poll 60 --no-report 2>&1

echo "=== [report] aggregating ablation_color ==="
./07_batch_report.sh -f batches/ablation_color.txt 2>&1

echo "=== [done] recovery chain complete ==="
