#!/bin/bash
# =============================================================================
# config.sh — remote path layout shared by every remote_execution_scripts/*.
# =============================================================================
# VM *identity* (user/host/port) lives in profiles/<name>.sh, selected with
# `--vm <name>` (default vm03) and sourced by lib/common.sh::bootstrap before
# this file. This file only derives the on-VM path layout from that identity.
# A job .conf file never repeats any of these values — it references relative
# paths and $REMOTE_WORKDIR etc.
#
# Precedence for every var below: explicit shell env > profile > .env > the
# defaults here (all use ${VAR:-default}, so the first one set wins).
# =============================================================================

# Fallback so config.sh is still valid if sourced without a profile (e.g. a
# one-off `VM_USER=... source config.sh`). Normally the profile sets these.
VM_USER="${VM_USER:-vm03}"
VM_HOST="${VM_HOST:-143.107.165.250}"
VM_PORT="${VM_PORT:-5022}"

# Single SSD base every remote path hangs off. The VM home partition is small,
# so everything lives on the SSD. Mirrors per-VM by username (/media/vm03/...,
# /media/vm02/...). A profile can override REMOTE_SSD_BASE if a VM's SSD mounts
# elsewhere — every path below then follows automatically.
REMOTE_SSD_BASE="${REMOTE_SSD_BASE:-/media/${VM_USER}/ssd1T/andrew}"

# Where the project lives on the VM. 01_deploy.sh creates it if missing.
REMOTE_WORKDIR="${REMOTE_WORKDIR:-${REMOTE_SSD_BASE}/dynamic/dynamic-lmms-eval}"

# Sibling dynamic-dataset repo. Required by tools/prepare_dynamic_graph_*.py.
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-${REMOTE_SSD_BASE}/dynamic/dynamic-dataset}"

# SSD-backed caches. Home partition can't hold HF models, uv's download cache,
# triton compile cache, or generic XDG caches — vllm fills any of them quickly.
REMOTE_HF_HOME="${REMOTE_HF_HOME:-${REMOTE_SSD_BASE}/hf_cache}"
REMOTE_UV_CACHE_DIR="${REMOTE_UV_CACHE_DIR:-${REMOTE_SSD_BASE}/uv_cache}"
REMOTE_XDG_CACHE_HOME="${REMOTE_XDG_CACHE_HOME:-${REMOTE_SSD_BASE}/.cache}"
REMOTE_TRITON_HOME="${REMOTE_TRITON_HOME:-${REMOTE_SSD_BASE}}"
REMOTE_TRITON_CACHE_DIR="${REMOTE_TRITON_CACHE_DIR:-${REMOTE_SSD_BASE}/.triton/cache}"
REMOTE_TMPDIR="${REMOTE_TMPDIR:-${REMOTE_SSD_BASE}/tmp}"
REMOTE_FLASHINFER_WORKSPACE_BASE="${REMOTE_FLASHINFER_WORKSPACE_BASE:-${REMOTE_SSD_BASE}}"

# Where per-run artifacts (logs, metadata) live on the VM.
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR:-${REMOTE_WORKDIR}/.runs}"

# Local repo root (this file is in remote_execution_scripts/).
LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Local dynamic-dataset repo (sibling of dynamic-lmms-eval).
LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-${LOCAL_REPO_ROOT}/../dynamic-dataset}"

# Local directory where fetched results land (one subdir per job).
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-${LOCAL_REPO_ROOT}/remote_results}"

# SSH_KEY is required. Preferred: copy .env.example to .env and set it there
# (loaded automatically by lib/common.sh). One-off override still works:
#   SSH_KEY="$HOME/.ssh/vm_key" ./02_run.sh <job>
