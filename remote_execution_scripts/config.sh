#!/bin/bash
# =============================================================================
# config.sh — VM identity used by every remote_execution_scripts/* script.
# =============================================================================
# Edit the block below for your lab's host. A job .conf file never repeats
# these values — it only references $REMOTE_WORKDIR etc.
# =============================================================================

VM_USER="${VM_USER:-vm03}"
VM_HOST="${VM_HOST:-143.107.165.250}"
VM_PORT="${VM_PORT:-5022}"

# Where the project lives on the VM. 01_deploy.sh creates it if missing.
# VM home partition is small; the repo lives on the SSD under /media/vm03/ssd1T.
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/media/${VM_USER}/ssd1T/andrew/dynamic/dynamic-lmms-eval}"

# Sibling dynamic-dataset repo. Required by tools/prepare_dynamic_graph_*.py.
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-/media/${VM_USER}/ssd1T/andrew/dynamic/dynamic-dataset}"

# SSD-backed caches. Home partition can't hold HF models or uv's download cache.
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/media/${VM_USER}/ssd1T/andrew/hf_cache}"
REMOTE_UV_CACHE_DIR="${REMOTE_UV_CACHE_DIR:-/media/${VM_USER}/ssd1T/andrew/uv_cache}"

# Where per-run artifacts (logs, metadata) live on the VM.
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR:-${REMOTE_WORKDIR}/.runs}"

# Local repo root (this file is in remote_execution_scripts/).
LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Local dynamic-dataset repo (sibling of dynamic-lmms-eval).
LOCAL_DATASET_ROOT="${LOCAL_DATASET_ROOT:-${LOCAL_REPO_ROOT}/../dynamic-dataset}"

# Local directory where fetched results land (one subdir per job).
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-${LOCAL_REPO_ROOT}/remote_results}"

# SSH_KEY is required — export it before running any script:
#   export SSH_KEY="$HOME/.ssh/vm_key"
