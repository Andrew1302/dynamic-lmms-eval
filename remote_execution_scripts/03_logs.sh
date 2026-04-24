#!/bin/bash
# =============================================================================
# 03_logs.sh <job> [--attach|--tail N|--status]
# -----------------------------------------------------------------------------
# Default: stream run.log live (like `tail -f`). Ctrl-C detaches; the remote
# process keeps running inside tmux.
#
# --attach   : attach to the tmux session itself (interactive, Ctrl-b d detaches).
# --tail N   : print the last N lines and exit (N defaults to 100).
# --status   : print session state + last 20 lines + exit-code sentinel. Non-interactive.
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
bootstrap

load_job "${1:-}"
shift || true

MODE="follow"
TAIL_N=100
while [ $# -gt 0 ]; do
    case "$1" in
        --attach)  MODE="attach"; shift ;;
        --tail)    MODE="tail"; TAIL_N="${2:-100}"; shift 2 ;;
        --status)  MODE="status"; shift ;;
        *) fail "Unknown flag: $1" ;;
    esac
done

SESSION="$(tmux_session "$JOB_NAME")"
RUN_DIR="$(remote_run_dir "$JOB_NAME")"
LOG_PATH="$(remote_log_path "$JOB_NAME")"

case "$MODE" in
    attach)
        log "attaching to tmux session '$SESSION' (Ctrl-b d to detach, job keeps running)"
        ssh_tty "tmux attach -t '$SESSION'"
        ;;
    tail)
        ssh_cmd "tail -n $TAIL_N '$LOG_PATH' 2>/dev/null || echo '(no log yet at $LOG_PATH)'"
        ;;
    status)
        ssh_cmd "bash -s" <<REMOTE
set -u
echo '=== session ==='
if tmux has-session -t '$SESSION' 2>/dev/null; then
    echo 'status: RUNNING'
    tmux list-sessions | grep '^$SESSION:' || true
else
    echo 'status: NOT RUNNING'
fi
echo
echo '=== exit sentinel ==='
if [ -f '$RUN_DIR/exit_code' ]; then
    echo "exit_code: \$(cat '$RUN_DIR/exit_code')"
else
    echo 'exit_code: <not yet written>'
fi
echo
echo '=== last 20 lines of run.log ==='
if [ -f '$LOG_PATH' ]; then
    tail -n 20 '$LOG_PATH'
else
    echo '(no log yet at $LOG_PATH)'
fi
REMOTE
        ;;
    follow)
        log "tailing $LOG_PATH (Ctrl-C stops tail; remote job keeps running)"
        # -F retries if the file doesn't exist yet; --pid exits when tmux dies.
        ssh_tty "tail -F '$LOG_PATH'"
        ;;
esac
