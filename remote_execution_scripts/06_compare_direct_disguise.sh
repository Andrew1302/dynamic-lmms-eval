#!/bin/bash
# =============================================================================
# 06_compare_direct_disguise.sh <job> [--timestamp TS]
# -----------------------------------------------------------------------------
# Local postprocessing — no SSH. Reads $LOCAL_RESULTS_DIR/<job>/logs/, picks
# the latest timestamped lmms-eval run, joins each direct/disguise task pair
# declared by COMPARE_PAIRS in the .conf, and writes one Excel workbook with
# a `main` summary tab plus one tab per pair.
#
# .conf must declare:
#   COMPARE_PAIRS=(
#       "label:direct_task_name:disguise_task_name"
#       ...
#   )
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
# Local-only: this script never SSHes, so skip the SSH_KEY check. --vm is
# accepted (and stripped) for command-line symmetry even though it's a no-op
# here — local result paths are VM-independent.
BOOTSTRAP_LOCAL=1 bootstrap "$@"
set -- ${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}

load_job "${1:-}"
shift || true

TIMESTAMP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --timestamp) TIMESTAMP="${2:-}"; shift 2 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

if ! declare -p COMPARE_PAIRS >/dev/null 2>&1; then
    fail "jobs/${JOB_NAME}.conf must define COMPARE_PAIRS=(\"label:direct:disguise\" ...) for 06_compare_direct_disguise.sh"
fi
if [ "${#COMPARE_PAIRS[@]}" -eq 0 ]; then
    fail "COMPARE_PAIRS in jobs/${JOB_NAME}.conf is empty — add at least one entry"
fi

JOB_RESULTS_DIR="$(local_results_dir "$JOB_NAME")"
if [ ! -d "$JOB_RESULTS_DIR/logs" ]; then
    fail "no logs/ subtree under $JOB_RESULTS_DIR — run ./04_fetch.sh $JOB_NAME first"
fi

OUTPUT="$JOB_RESULTS_DIR/processed/compare_direct_disguise.xlsx"
SCRIPT="$REPO_ROOT/tools/postprocess/compare_direct_disguise.py"

PAIR_ARGS=()
for pair in "${COMPARE_PAIRS[@]}"; do
    PAIR_ARGS+=("--pair" "$pair")
done

TS_ARGS=()
if [ -n "$TIMESTAMP" ]; then
    TS_ARGS+=("--timestamp" "$TIMESTAMP")
fi

log "comparing direct vs disguise for '$JOB_NAME' (${#COMPARE_PAIRS[@]} pair(s))"
# --no-project + --with avoids forcing a full local `uv sync` (torch etc.) just
# to run this lightweight script — its only third-party dep is openpyxl.
( cd "$REPO_ROOT" && uv run --no-project --with openpyxl python "$SCRIPT" \
    --results-dir "$JOB_RESULTS_DIR" \
    --output "$OUTPUT" \
    "${PAIR_ARGS[@]}" \
    "${TS_ARGS[@]}" )

log "${C_G}wrote${C_RESET} $OUTPUT"
