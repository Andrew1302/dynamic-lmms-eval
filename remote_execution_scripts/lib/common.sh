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

# Ensure user-local bin is on PATH so non-interactive shells (e.g. run_batch.sh
# invoked from WSL via `wsl.exe -- bash -c ...`) can find `uv` even though
# .profile is not sourced. uv installs itself to ~/.local/bin by default.
if [ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

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
        fail "SSH_KEY is not set. Put it in remote_execution_scripts/.env (see .env.example) or export SSH_KEY=\"\$HOME/.ssh/your_key\"."
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

    # Snapshot of currently-exported variable names so we can compute which
    # vars the .conf adds. 02_run.sh forwards those (and only those) into the
    # remote launcher — local `export` in the .conf does not reach SSH, so
    # without this run_eval.sh sees an empty $MODEL_PRETRAINED on the VM.
    local _before_exports
    _before_exports="$(compgen -e | sort)"

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

    # Compute the set of newly-exported variable names (job env vars).
    local _after_exports
    _after_exports="$(compgen -e | sort)"
    JOB_EXPORTS=()
    local _var
    while IFS= read -r _var; do
        [ -n "$_var" ] && JOB_EXPORTS+=("$_var")
    done < <(comm -13 <(printf '%s\n' "$_before_exports") <(printf '%s\n' "$_after_exports"))
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

scp_up() {
    # scp a single local file to an absolute remote path. Args: <local_file> <remote_abs_path>
    # scp uses -P for port (ssh uses -p), so we can't pass _ssh_opts verbatim.
    scp -i "$SSH_KEY" -P "$VM_PORT" -o StrictHostKeyChecking=no -q "$1" "${VM_USER}@${VM_HOST}:$2"
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
# Set BOOTSTRAP_LOCAL=1 before calling for local-only scripts (no SSH).
bootstrap() {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
    # Load remote_execution_scripts/.env if present. Anything already in the
    # shell environment wins (so a one-off `SSH_KEY=... ./02_run.sh ...` still
    # overrides). .env is gitignored — see .env.example for the template.
    local env_file="$REPO_ROOT/remote_execution_scripts/.env"
    if [ -f "$env_file" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            # skip blanks and comments
            [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
            # split KEY=VALUE on the first =
            local key="${line%%=*}"
            local val="${line#*=}"
            key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
            # strip a single layer of surrounding single or double quotes
            if [[ "$val" =~ ^\".*\"$ ]] || [[ "$val" =~ ^\'.*\'$ ]]; then
                val="${val:1:${#val}-2}"
            fi
            # only set if not already in the shell env
            if [ -z "${!key:-}" ]; then
                # eval so values like $HOME or ~ expand the same way they
                # would on a normal `export` line
                eval "export $key=\"$val\""
            fi
        done < "$env_file"
    fi
    # shellcheck disable=SC1091
    source "$REPO_ROOT/remote_execution_scripts/config.sh"
    if [ "${BOOTSTRAP_LOCAL:-0}" != "1" ]; then
        require_env
        # re-populate ssh opts now that VM_PORT / SSH_KEY are known
        _ssh_opts=(-i "$SSH_KEY" -p "$VM_PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30)
    fi
}
