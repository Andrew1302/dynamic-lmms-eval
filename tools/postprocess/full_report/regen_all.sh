#!/bin/bash
# Regenerate the full-results set end-to-end: assemble -> batch_report -> run_info.
# Usage: bash regen_all.sh <scratch_assembled_root>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD="$REPO/tools/postprocess/full_report"
OUT="$REPO/remote_results/_full_reports"
ASM="${1:?pass an assembled-root scratch dir}"
TS=$(date +%Y%m%d_%H%M%S)
cd "$REPO"

rm -rf "$ASM"; mkdir -p "$ASM"

report () {  # <family> <manifest> <tsv>
  local nd
  nd=$(uv run --no-project --with openpyxl --with datasets python tools/postprocess/batch_report.py \
    --jobs-tsv "$3" --output "$OUT/full_$1_${TS}.xlsx" \
    --latest-link "$OUT/full_$1_latest.xlsx" --batch-name "full_$1" --strict 2>&1 \
    | grep -cE NO_DATA || true)
  echo "  full_$1 NO_DATA: $nd"
}

for fam in standard adjlist color think sweep_nodes labels; do
  uv run --no-project python "$BUILD/assemble_full.py" \
    "$BUILD/manifests/full_$fam.txt" "$ASM/$fam" "$ASM/$fam.tsv" >/dev/null
done

# 6 families -> 7 reports (labels split into letters/none)
report standard    "$BUILD/manifests/full_standard.txt"    "$ASM/standard.tsv"
report adjlist     "$BUILD/manifests/full_adjlist.txt"     "$ASM/adjlist.tsv"
report color       "$BUILD/manifests/full_color.txt"       "$ASM/color.tsv"
report think       "$BUILD/manifests/full_think.txt"       "$ASM/think.tsv"
report sweep_nodes "$BUILD/manifests/full_sweep_nodes.txt" "$ASM/sweep_nodes.tsv"
grep "_letters_" "$ASM/labels.tsv" > "$ASM/labels_letters.tsv"
grep "_none_"    "$ASM/labels.tsv" > "$ASM/labels_none.tsv"
report labels_letters "" "$ASM/labels_letters.tsv"
report labels_none    "" "$ASM/labels_none.tsv"

PYTHONIOENCODING=utf-8 uv run --no-project --with openpyxl --with datasets python \
  "$BUILD/add_full_run_info.py" "$ASM" "$OUT" | sed 's/^/  /'
echo "regen done (ts=$TS)"
