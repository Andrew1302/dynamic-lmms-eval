#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

JOB="graph_benchmark/graph_bench_ablation_color_internvl35_4b"
SESSION="lmms_graph_bench_ablation_color_internvl35_4b"
RUN_DIR="/media/vm03/ssd1T/andrew/dynamic/dynamic-lmms-eval/.runs/graph_bench_ablation_color_internvl35_4b"
SSH_OPTS=(-i /home/andrew/.ssh/vm_key -p 5022 -o ConnectTimeout=10 -o ServerAliveInterval=20 -o StrictHostKeyChecking=no)
HOST="vm03@143.107.165.250"

echo "=== [deploy] $JOB ==="
for i in 1 2 3 4 5; do
    if ./01_deploy.sh "$JOB"; then break; fi
    echo "[deploy] failed attempt $i, retry 20s"; sleep 20
done

echo "=== [run] $JOB ==="
./02_run.sh "$JOB" || exit 1

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
        *ALIVE*) sleep 30 ;;
        *GONE*)  echo "[wait] tmux session ended"; break ;;
        *)       echo "[wait] unexpected response: $out"; sleep 30 ;;
    esac
done

echo "=== [tail] final run.log tail ==="
for i in 1 2 3 4 5; do
    if ssh "${SSH_OPTS[@]}" "$HOST" "tail -n 60 $RUN_DIR/run.log; echo ---; if [ -f $RUN_DIR/exit_code ]; then echo exit_code: \$(cat $RUN_DIR/exit_code); else echo exit_code: missing; fi"; then
        break
    fi
    echo "[tail] ssh failed attempt $i, retry 15s"
    sleep 15
done
echo "=== [done] smoke test complete ==="
