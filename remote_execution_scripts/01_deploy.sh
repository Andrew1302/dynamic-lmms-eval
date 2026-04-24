#!/bin/bash
# =============================================================================
# 01_deploy.sh <job> — Rsync the repo up to the VM and run setup.
# =============================================================================
# Reads jobs/<job>.conf, rsyncs each entry in UPLOAD_PATHS to $REMOTE_WORKDIR
# on the VM, then runs REMOTE_SETUP_CMD (if defined) with live output.
#
# Safe to re-run: rsync only transfers what changed, and REMOTE_SETUP_CMD
# should be idempotent (e.g. `uv sync`).
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
bootstrap

load_job "${1:-}"

log "deploying job '$JOB_NAME' to ${VM_USER}@${VM_HOST}:${REMOTE_WORKDIR}"

ssh_cmd "mkdir -p '$REMOTE_WORKDIR' '$REMOTE_RUNS_DIR' '$REMOTE_DATASET_DIR' '$REMOTE_HF_HOME' '$REMOTE_UV_CACHE_DIR'"

log "rsync → $REMOTE_WORKDIR"
for path in "${UPLOAD_PATHS[@]}"; do
    src="$LOCAL_REPO_ROOT/$path"
    if [ ! -e "$src" ]; then
        warn "skipping missing upload path: $path"
        continue
    fi
    # Trailing slash on dirs ⇒ rsync the contents; files go as-is.
    if [ -d "$src" ]; then
        log "  ↑ $path/"
        rsync_up "$src/" "$path/"
    else
        log "  ↑ $path"
        rsync_up "$src" "$path"
    fi
done

if [ ${#DATASET_UPLOAD_PATHS[@]} -gt 0 ]; then
    if [ ! -d "$LOCAL_DATASET_ROOT" ]; then
        fail "DATASET_UPLOAD_PATHS is set but LOCAL_DATASET_ROOT does not exist: $LOCAL_DATASET_ROOT"
    fi
    log "rsync → $REMOTE_DATASET_DIR (from $LOCAL_DATASET_ROOT)"
    for path in "${DATASET_UPLOAD_PATHS[@]}"; do
        src="$LOCAL_DATASET_ROOT/$path"
        if [ ! -e "$src" ]; then
            warn "skipping missing dataset upload path: $path"
            continue
        fi
        if [ -d "$src" ]; then
            log "  ↑ $path/"
            rsync_up "$src/" "$path/" "$REMOTE_DATASET_DIR"
        else
            log "  ↑ $path"
            rsync_up "$src" "$path" "$REMOTE_DATASET_DIR"
        fi
    done
fi

if [ -n "${REMOTE_SETUP_CMD:-}" ]; then
    log "running setup command on VM (UV_CACHE_DIR=$REMOTE_UV_CACHE_DIR):"
    echo "${C_DIM}$REMOTE_SETUP_CMD${C_RESET}"
    # uv lives inside .venv on this VM — activate it before REMOTE_SETUP_CMD so
    # `uv sync` (or any `uv …` call) resolves. If .venv is missing we emit a
    # hint; it must be bootstrapped once on the VM before the first deploy.
    ssh_cmd "bash -lc '
set -e
export UV_CACHE_DIR=\"$REMOTE_UV_CACHE_DIR\" HF_HOME=\"$REMOTE_HF_HOME\"
cd \"$REMOTE_WORKDIR\"
if [ ! -f .venv/bin/activate ]; then
    echo \"[fatal] .venv/bin/activate missing in $REMOTE_WORKDIR — bootstrap uv once: python3 -m venv .venv && .venv/bin/pip install uv\" >&2
    exit 1
fi
source .venv/bin/activate
$REMOTE_SETUP_CMD
'"
fi

log "${C_G}deploy complete${C_RESET} — next: ./02_run.sh $JOB_NAME"
