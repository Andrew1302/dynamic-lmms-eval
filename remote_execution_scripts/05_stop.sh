#!/bin/bash
# =============================================================================
# 05_stop.sh <job> — Kill the tmux session for a job.
# -----------------------------------------------------------------------------
# Sends `tmux kill-session`. The REMOTE_RUN_CMD children are killed with it.
# Safe to call even if the session no longer exists.
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
bootstrap

load_job "${1:-}"

SESSION="$(tmux_session "$JOB_NAME")"

if ssh_cmd "tmux has-session -t '$SESSION' 2>/dev/null"; then
    ssh_cmd "tmux kill-session -t '$SESSION'"
    log "${C_G}killed tmux session '$SESSION'${C_RESET}"
else
    warn "no tmux session '$SESSION' is running — nothing to stop"
fi
