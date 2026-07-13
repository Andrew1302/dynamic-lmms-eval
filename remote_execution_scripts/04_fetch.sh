#!/bin/bash
# =============================================================================
# 04_fetch.sh <job> [--force]
# -----------------------------------------------------------------------------
# Rsyncs each RESULT_PATHS entry from the VM into $LOCAL_RESULTS_DIR/<job>/,
# plus the run.log. Refuses to fetch if the tmux session is still running
# unless --force is passed (useful for partial mid-run snapshots).
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
bootstrap "$@"
set -- ${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}

load_job "${1:-}"
shift || true

FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; fi

SESSION="$(tmux_session "$JOB_NAME")"
RUN_DIR="$(remote_run_dir "$JOB_NAME")"
LOG_PATH="$(remote_log_path "$JOB_NAME")"
DEST="$(local_results_dir "$JOB_NAME")"

if ssh_cmd "tmux has-session -t '$SESSION' 2>/dev/null"; then
    if [ "$FORCE" -eq 0 ]; then
        fail "tmux session '$SESSION' is still running. Pass --force to fetch a snapshot anyway, or wait + re-run."
    fi
    warn "session still running — fetching a mid-run snapshot (--force)"
fi

mkdir -p "$DEST"
log "fetching results for '$JOB_NAME' → $DEST"

# Always pull the run log + exit sentinel.
log "  ↓ .runs/$JOB_NAME/"
rsync_down ".runs/$JOB_NAME/" "$DEST/.run/"

for path in "${RESULT_PATHS[@]}"; do
    log "  ↓ $path"
    # Create the parent dir locally so rsync preserves the relative layout.
    rsync_down "$path" "$DEST/$path" || warn "  (not present on VM: $path)"
done

EXIT_CODE_FILE="$DEST/.run/exit_code"
ec=""
if [ -f "$EXIT_CODE_FILE" ]; then
    ec="$(cat "$EXIT_CODE_FILE")"
    if [ "$ec" = "0" ]; then
        log "${C_G}fetch complete — remote job exited 0${C_RESET}"
    else
        warn "remote job exited with code $ec — inspect $DEST/.run/run.log"
    fi
else
    warn "no exit sentinel found — job may still be running (used --force?)"
fi

# Auto-file the just-fetched run into the numbered campaign tree
# (NN_campaign/<task>/<original|disguise>/<leaf>.jsonl). Only on a clean,
# complete fetch — a --force mid-run snapshot is left flat so a partial run
# isn't prematurely organized. Idempotent; safe to re-run manually:
#   python tools/postprocess/organize_results.py --root "$LOCAL_RESULTS_DIR" --apply
if [ "$FORCE" -eq 0 ] && [ "$ec" = "0" ]; then
    ORG="$REPO_ROOT/tools/postprocess/organize_results.py"
    PY="$(command -v python || command -v python3 || true)"
    if [ -n "$PY" ] && [ -f "$ORG" ]; then
        log "organizing → numbered campaign tree"
        "$PY" "$ORG" --root "$LOCAL_RESULTS_DIR" --apply --quiet \
            || warn "organize step failed — results fetched OK; run organize_results.py manually"
    else
        warn "python/organizer not found — results left flat; run organize_results.py to file them"
    fi
fi
