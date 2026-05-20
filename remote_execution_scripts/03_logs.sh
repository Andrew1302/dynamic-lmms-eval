#!/bin/bash
# =============================================================================
# 03_logs.sh <job> [--attach|--tail N|--status|--until-done [N]]
# -----------------------------------------------------------------------------
# Default: stream run.log live (like `tail -f`). Ctrl-C detaches; the remote
# process keeps running inside tmux.
#
# --attach     : attach to the tmux session itself (Ctrl-b d to detach).
# --tail N     : print the last N lines and exit (N defaults to 100).
# --status     : print session state + last 20 lines + exit-code sentinel.
# --until-done [N]: poll every N seconds (default 30) until the tmux session
#                   ends; then dump the final tail (last 80 lines) and the
#                   exit sentinel. Used by run_batch.sh to chain jobs.
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
bootstrap

load_job "${1:-}"
shift || true

MODE="follow"
TAIL_N=100
POLL_S=30
while [ $# -gt 0 ]; do
    case "$1" in
        --attach)     MODE="attach"; shift ;;
        --tail)       MODE="tail"; TAIL_N="${2:-100}"; shift 2 ;;
        --status)     MODE="status"; shift ;;
        --until-done) MODE="until-done"
                      if [ "${2:-}" ] && [ "${2#-}" = "$2" ]; then
                          POLL_S="$2"; shift 2
                      else
                          shift
                      fi
                      ;;
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
    until-done)
        log "polling tmux session '$SESSION' every ${POLL_S}s until it ends"
        # Distinguish "tmux session ended" (rc=1) from "ssh transport failed"
        # (rc=255 typically). A bare `while ssh_cmd ...` exits on both, which
        # made the watcher return prematurely on any network blip and let
        # downstream callers (run_batch.sh) launch the next job on top of a
        # still-running tmux session — causing GPU contention.
        while true; do
            probe_rc=0
            ssh_cmd "tmux has-session -t '$SESSION' 2>/dev/null" || probe_rc=$?
            if [ "$probe_rc" -eq 0 ]; then
                sleep "$POLL_S"
            elif [ "$probe_rc" -eq 1 ]; then
                # tmux confirmed session is gone — exit the loop cleanly.
                break
            else
                # Transport error (ssh exit 255, network reset, etc.). Retry
                # rather than treating it as end-of-job. Print so the operator
                # can see if blips become persistent.
                log "ssh probe returned $probe_rc (transport error?), retrying after ${POLL_S}s"
                sleep "$POLL_S"
            fi
        done
        log "session ended; dumping final tail + exit sentinel"
        ssh_cmd "bash -s" <<REMOTE
set -u
echo '=== final tail (80 lines) ==='
tail -n 80 '$LOG_PATH' 2>/dev/null || echo '(no log at $LOG_PATH)'
echo
echo '=== exit sentinel ==='
if [ -f '$RUN_DIR/exit_code' ]; then
    ec=\$(cat '$RUN_DIR/exit_code')
    echo "exit_code: \$ec"
    exit "\$ec"
else
    echo 'exit_code: <missing>'
    exit 0
fi
REMOTE
        ;;
esac
