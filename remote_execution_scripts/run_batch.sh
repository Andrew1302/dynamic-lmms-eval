#!/bin/bash
# =============================================================================
# run_batch.sh [-f manifest.txt] [job1 job2 ...] [--keep-going]
# -----------------------------------------------------------------------------
# Sequentially run several jobs end-to-end (deploy → run → wait → fetch).
#
# Inputs:
#   -f FILE       Read job ids (one per line, '#' comments OK) from FILE.
#   <job-id>      Direct job id, e.g. graph_benchmark/graph_bench_standard_qwen3vl_4b.
#                 Multiple direct ids may be passed; combine freely with -f.
#
# Options:
#   --vm NAME     Target VM profile (default vm03). Forwarded to every sub-step
#                 via the VM_PROFILE env var. See profiles/<name>.sh.
#   --keep-going  Continue to the next job after a failure (default: stop).
#   --poll N      03_logs.sh --until-done poll interval, seconds (default 30).
#   --no-report   Skip the trailing 07_batch_report.sh invocation.
#
# A short pass/fail summary is printed at the end. Per-job logs land in the
# usual remote_results/<job> path via 04_fetch.sh; this script does not parse
# accuracy numbers — see tools/postprocess/ for that. Once the queue is done
# (whether every job succeeded or some failed), this script also calls
# 07_batch_report.sh to write one aggregated xlsx covering the batch
# (missing/failed jobs surface as NO_DATA rows). Pass --no-report to skip.
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
KEEP_GOING=0
POLL_S=30
JOBS=()
MANIFEST=""
RUN_REPORT=1

while [ $# -gt 0 ]; do
    case "$1" in
        --vm)
            # Select the target VM and export it so every child script
            # (01/02/03/04, 07) inherits it via bootstrap's VM_PROFILE lookup.
            VM_PROFILE="${2:-}"; export VM_PROFILE
            [ -n "$VM_PROFILE" ] || { echo "--vm needs a value (e.g. --vm vm02)" >&2; exit 2; }
            shift 2
            ;;
        --vm=*) VM_PROFILE="${1#*=}"; export VM_PROFILE; shift ;;
        -f|--file)
            [ -f "$2" ] || { echo "manifest not found: $2" >&2; exit 2; }
            # Remember the manifest path so 07 can use it for batch-name +
            # job list rather than re-parsing what we read here.
            if [ -z "$MANIFEST" ]; then MANIFEST="$2"; fi
            while IFS= read -r line; do
                line="${line%%#*}"
                line="${line//[$'\t\r\n ']/}"
                [ -n "$line" ] && JOBS+=("$line")
            done <"$2"
            shift 2
            ;;
        --keep-going) KEEP_GOING=1; shift ;;
        --poll) POLL_S="$2"; shift 2 ;;
        --no-report)  RUN_REPORT=0; shift ;;
        -h|--help)
            sed -n '2,/^# ====/p' "$0" | head -n 28 | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) JOBS+=("$1"); shift ;;
    esac
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "run_batch.sh: no jobs specified. Pass job ids directly or use -f manifest.txt" >&2
    exit 2
fi

echo "[run_batch] target VM: ${VM_PROFILE:-vm03}"
echo "[run_batch] queue (${#JOBS[@]} jobs):"
for j in "${JOBS[@]}"; do echo "  - $j"; done

declare -A RESULT
for j in "${JOBS[@]}"; do
    echo
    echo "============================================================"
    echo "[run_batch] $(date -Iseconds)  starting: $j"
    echo "============================================================"
    if "$HERE/01_deploy.sh" "$j" \
        && "$HERE/02_run.sh" "$j" \
        && "$HERE/03_logs.sh" "$j" --until-done "$POLL_S" \
        && "$HERE/04_fetch.sh" "$j"; then
        RESULT["$j"]="ok"
        echo "[run_batch] $j: OK"
    else
        rc=$?
        RESULT["$j"]="FAIL(rc=$rc)"
        echo "[run_batch] $j: FAIL (rc=$rc)" >&2
        if [ "$KEEP_GOING" -eq 0 ]; then
            echo "[run_batch] aborting batch (pass --keep-going to continue past failures)" >&2
            break
        fi
    fi
done

echo
echo "============================================================"
echo "[run_batch] summary"
echo "============================================================"
overall=0
for j in "${JOBS[@]}"; do
    state="${RESULT[$j]:-skipped}"
    printf "  %-12s %s\n" "$state" "$j"
    case "$state" in
        ok)      ;;
        skipped) ;;
        *)       overall=1 ;;
    esac
done

# Aggregate every job (including failures, which become NO_DATA rows) into
# one xlsx so the operator can read the whole batch in one place. Skip on
# --no-report or if 07 itself isn't present.
if [ "$RUN_REPORT" -eq 1 ] && [ -x "$HERE/07_batch_report.sh" ]; then
    echo
    echo "============================================================"
    echo "[run_batch] batch report"
    echo "============================================================"
    REPORT_ARGS=()
    if [ -n "$MANIFEST" ]; then
        REPORT_ARGS+=(-f "$MANIFEST")
    else
        # Direct-id form: pass every job through individually. The batch
        # name defaults to "batch" inside 07.
        REPORT_ARGS+=("${JOBS[@]}")
    fi
    report_rc=0
    "$HERE/07_batch_report.sh" "${REPORT_ARGS[@]}" || report_rc=$?
    if [ "$report_rc" -ne 0 ]; then
        # Don't let a postprocess failure mask the run's own pass/fail status —
        # the per-job results are still on disk and 07 can be re-run later.
        echo "[run_batch] WARNING: 07_batch_report.sh failed (exit $report_rc) — batch results are still on disk; re-run 07 manually" >&2
    fi
fi

exit "$overall"
