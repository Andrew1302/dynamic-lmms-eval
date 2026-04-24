#!/bin/bash
# =============================================================================
# lib/common.sh — Shared helpers for remote_execution_scripts/*.
# =============================================================================
# Sourced (not executed) by every top-level script. Provides:
#   - require_env        : fail loudly when $SSH_KEY is missing
#   - load_job           : source a jobs/<name>.conf and validate it
#   - ssh_cmd / ssh_tty  : run a command on the VM (batch / interactive)
#   - rsync_up / rsync_down : transfer files to/from the VM
#   - tmux_session       : canonical tmux session name for a job
#   - remote_log_path    : canonical remote log path for a run
# =============================================================================

set -euo pipefail

# --- color helpers (no-op when stdout isn't a TTY) ---------------------------
if [ -t 1 ]; then
    C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'
    C_B=$'\033[34m'; C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
else
    C_R=""; C_G=""; C_Y=""; C_B=""; C_DIM=""; C_RESET=""
fi

log()  { echo "${C_B}[remote]${C_RESET} $*"; }
warn() { echo "${C_Y}[remote]${C_RESET} $*" >&2; }
fail() { echo "${C_R}[remote] ERROR:${C_RESET} $*" >&2; exit 1; }

# --- env validation ----------------------------------------------------------
require_env() {
    if [ -z "${SSH_KEY:-}" ]; then
        fail "SSH_KEY is not set. Export it first: export SSH_KEY=\"\$HOME/.ssh/your_key\""
    fi
    if [ ! -f "$SSH_KEY" ]; then
        fail "SSH_KEY points to a file that does not exist: $SSH_KEY"
    fi
}

# --- job loader --------------------------------------------------------------
# Sources remote_execution_scripts/jobs/<name>.conf and validates its fields.
# Sets JOB_NAME if the .conf didn't.
load_job() {
    local job_arg="$1"
    if [ -z "${job_arg:-}" ]; then
        fail "No job specified. Available jobs:$(ls "$REPO_ROOT/remote_execution_scripts/jobs/" | sed 's|^| - |; s|\.conf$||' | awk 'NR==1{print ""; print $0; next} {print}')"
    fi

    local conf="$REPO_ROOT/remote_execution_scripts/jobs/${job_arg}.conf"
    if [ ! -f "$conf" ]; then
        fail "Job config not found: $conf"
    fi

    # shellcheck disable=SC1090
    source "$conf"

    : "${JOB_NAME:=$job_arg}"
    : "${UPLOAD_PATHS:?jobs/${job_arg}.conf must define UPLOAD_PATHS (bash array)}"
    : "${REMOTE_RUN_CMD:?jobs/${job_arg}.conf must define REMOTE_RUN_CMD}"
    : "${RESULT_PATHS:?jobs/${job_arg}.conf must define RESULT_PATHS (bash array)}"
    # REMOTE_SETUP_CMD is optional.
    # DATASET_UPLOAD_PATHS is optional — only jobs that depend on the sibling
    # dynamic-dataset repo need it. Default to an empty array so 01_deploy.sh
    # can safely iterate.
    if ! declare -p DATASET_UPLOAD_PATHS >/dev/null 2>&1; then
        DATASET_UPLOAD_PATHS=()
    fi
}

# --- ssh helpers -------------------------------------------------------------
_ssh_opts=(-i "${SSH_KEY:-}" -p "${VM_PORT:-22}" -o StrictHostKeyChecking=no -o ServerAliveInterval=30)

ssh_cmd() {
    # Non-interactive remote exec. Stdin from caller is forwarded.
    ssh "${_ssh_opts[@]}" -o BatchMode=yes "${VM_USER}@${VM_HOST}" "$@"
}

ssh_tty() {
    # Interactive session (for `tmux attach`).
    ssh -t "${_ssh_opts[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

rsync_up() {
    # rsync local -> remote. Args: <local_path> <remote_relative_path> [remote_base=$REMOTE_WORKDIR]
    local src="$1" dst="$2" base="${3:-$REMOTE_WORKDIR}"
    rsync -az --delete \
        -e "ssh ${_ssh_opts[*]}" \
        "$src" "${VM_USER}@${VM_HOST}:${base}/${dst}"
}

rsync_down() {
    # rsync remote -> local. Args: <remote_relative_path> <local_path>
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    rsync -az \
        -e "ssh ${_ssh_opts[*]}" \
        "${VM_USER}@${VM_HOST}:${REMOTE_WORKDIR}/${src}" "$dst"
}

# --- naming conventions ------------------------------------------------------
tmux_session() { echo "lmms_${1}"; }
remote_log_path() { echo "${REMOTE_RUNS_DIR}/${1}/run.log"; }
remote_run_dir()  { echo "${REMOTE_RUNS_DIR}/${1}"; }
local_results_dir() { echo "${LOCAL_RESULTS_DIR}/${1}"; }

# --- bootstrapping -----------------------------------------------------------
# Every top-level script sources this, then sources config.sh.
# $REPO_ROOT is set before sourcing so load_job can find jobs/.
bootstrap() {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/remote_execution_scripts/config.sh"
    require_env
    # re-populate ssh opts now that VM_PORT / SSH_KEY are known
    _ssh_opts=(-i "$SSH_KEY" -p "$VM_PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30)
}
