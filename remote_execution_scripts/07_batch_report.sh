#!/bin/bash
# =============================================================================
# 07_batch_report.sh -f <manifest.txt> [job ...] [options]
# -----------------------------------------------------------------------------
# Local postprocessing — no SSH. Aggregates every job in a batch into a single
# Excel workbook:
#
#   * `summary` sheet — one row per job (axis, model, per-task accuracy, overall)
#   * one tab per job — paired jobs get the `consolidated` table that 06 writes;
#     sweep jobs get long-form (base_task, variant, x, n, accuracy) rows.
#
# Inputs:
#   -f FILE              Batch manifest (one job id per line, '#' OK). Same
#                        format as run_batch.sh's -f. The basename (sans .txt)
#                        is used as the batch name and the output filename
#                        prefix.
#   <job-id>             Direct job id, e.g. graph_benchmark/graph_bench_*.
#                        Multiple direct ids may be passed and combined with -f.
#
# Options:
#   -o FILE              Output xlsx path. Default:
#                        remote_results/_batch_reports/<batch>_<gen_ts>.xlsx
#   --timestamp TS       Pin a specific {YYYYMMDD_HHMMSS} sample timestamp for
#                        every job. Jobs without that timestamp are NO_DATA.
#   --strict             Abort if any job is missing data. Default: emit a
#                        NO_DATA row and continue.
#
# Re-running this script after a fresh fetch never overwrites a prior report
# (the generation timestamp goes in the filename). A `<batch>_latest.xlsx`
# convenience copy is refreshed to point at the most recent run.
# =============================================================================

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/lib/common.sh"
BOOTSTRAP_LOCAL=1 bootstrap

JOBS=()
BATCH_NAME=""
TIMESTAMP=""
STRICT=0
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--file)
            [ -f "$2" ] || fail "manifest not found: $2"
            if [ -z "$BATCH_NAME" ]; then
                BATCH_NAME="$(basename "$2")"
                BATCH_NAME="${BATCH_NAME%.txt}"
            fi
            while IFS= read -r line; do
                line="${line%%#*}"
                line="${line//[$'\t\r\n ']/}"
                [ -n "$line" ] && JOBS+=("$line")
            done <"$2"
            shift 2
            ;;
        --timestamp) TIMESTAMP="${2:-}"; shift 2 ;;
        --strict)    STRICT=1; shift ;;
        -o|--output) OUTPUT="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# ====/p' "$0" | head -n 36 | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) JOBS+=("$1"); shift ;;
    esac
done

if [ "${#JOBS[@]}" -eq 0 ]; then
    fail "no jobs specified. Pass job ids directly or use -f manifest.txt"
fi

BATCH_NAME="${BATCH_NAME:-batch}"
GEN_TS="$(date +%Y%m%d_%H%M%S)"

if [ -z "$OUTPUT" ]; then
    OUTPUT="$LOCAL_RESULTS_DIR/_batch_reports/${BATCH_NAME}_${GEN_TS}.xlsx"
fi
LATEST_LINK="$LOCAL_RESULTS_DIR/_batch_reports/${BATCH_NAME}_latest.xlsx"

# Build a TSV of per-job metadata that the Python side reads.
# Columns (tab-separated, one job per line):
#   job_id   job_name   results_dir   model_pretrained   compare_pairs_joined   constraint
# compare_pairs_joined is the COMPARE_PAIRS array joined by '|'.
# constraint is "" for non-sweep jobs, or "nodes"/"edges" for sweep jobs —
# used by the Python side to decide the per-job tab layout (consolidated
# vs long-form).
# Sourcing each .conf inside a *subshell* keeps job vars from bleeding across
# iterations.

TMPTSV="$(mktemp -t batch_report.XXXXXX.tsv)"
trap 'rm -f "$TMPTSV"' EXIT

for j in "${JOBS[@]}"; do
    # Write each job's TSV row from a subshell so per-job env never leaks
    # across iterations. `set -e` outside makes a failing load_job abort the
    # whole script (rather than silently writing a blank row).
    (
        set -e
        load_job "$j"
        MODEL_PRETRAINED="${MODEL_PRETRAINED:-}"
        if declare -p COMPARE_PAIRS >/dev/null 2>&1 && [ "${#COMPARE_PAIRS[@]}" -gt 0 ]; then
            pairs_joined="$(IFS='|'; echo "${COMPARE_PAIRS[*]}")"
        else
            pairs_joined=""
        fi
        results_dir="$(local_results_dir "$JOB_NAME")"
        constraint="${CONSTRAINT:-}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$j" "$JOB_NAME" "$results_dir" "$MODEL_PRETRAINED" "$pairs_joined" "$constraint"
    ) >>"$TMPTSV"
done

log "writing batch report for '$BATCH_NAME' (${#JOBS[@]} jobs) → $OUTPUT"

STRICT_FLAG=()
if [ "$STRICT" -eq 1 ]; then STRICT_FLAG=(--strict); fi
TS_FLAG=()
if [ -n "$TIMESTAMP" ]; then TS_FLAG=(--timestamp "$TIMESTAMP"); fi

# --no-project + --with keeps this lightweight (openpyxl only, no torch).
( cd "$REPO_ROOT" && uv run --no-project --with openpyxl python \
    "$REPO_ROOT/tools/postprocess/batch_report.py" \
    --jobs-tsv "$TMPTSV" \
    --output "$OUTPUT" \
    --latest-link "$LATEST_LINK" \
    --batch-name "$BATCH_NAME" \
    "${TS_FLAG[@]}" "${STRICT_FLAG[@]}" )

log "${C_G}done${C_RESET} $OUTPUT (latest: $LATEST_LINK)"
