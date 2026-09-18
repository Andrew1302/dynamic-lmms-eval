#!/bin/bash
# =============================================================================
# Capture the exact lmms-eval argv that run_eval.sh would execute, for every
# (model, thinking, force-close, backend) cell, into a JSON fixture.
#
# This is the ground truth for prompting/plan.py: 15 bash glob branches cannot
# be ported reviewably by reading them, but they can be replayed. The fixture
# it writes is asserted against in test/eval/test_run_eval_plan.py.
#
#   bash tools/devtools/capture_run_eval_argv.sh test/eval/fixtures/legacy_run_eval_argv.json
#
# Re-run ONLY when run_eval.sh's invocation is intentionally changed -- the
# whole point is that this file does not move when plan.py is refactored.
#
# It shadows three commands on PATH for the duration:
#   accelerate - prints the argv (and the tuning env) instead of launching
#   python     - no-ops the dataset prepare step, passes everything else through
#   ln         - mkdir instead of symlink (Git Bash cannot always symlink)
# =============================================================================
set -euo pipefail

OUT="${1:?usage: capture_run_eval_argv.sh <out.json>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK" "$REPO/dataset_argvcapture" "$REPO/dynamic_graph_benchmark_data"' EXIT

mkdir -p "$WORK/bin"

cat > "$WORK/bin/accelerate" <<'STUB'
#!/bin/bash
"$REAL_PYTHON" - "$@" <<'PYEOF'
import json, os, sys
# run_eval.sh also *exports* tuning env vars; they are part of the behaviour the
# port must reproduce, so capture them alongside the argv.
keep = ("FP8_KEEP_BF16_PATTERNS", "PYTORCH_CUDA_ALLOC_CONF")
print("ARGV_CAPTURE " + json.dumps({"argv": sys.argv[1:], "env": {k: os.environ[k] for k in keep if k in os.environ}}))
PYEOF
STUB

cat > "$WORK/bin/python" <<'STUB'
#!/bin/bash
for a in "$@"; do
  case "$a" in *prepare_dynamic_graph_benchmark.py) exit 0 ;; esac
done
exec "$REAL_PYTHON" "$@"
STUB

cat > "$WORK/bin/ln" <<'STUB'
#!/bin/bash
mkdir -p "${!#}" 2>/dev/null || true
STUB

chmod +x "$WORK/bin/"*
REAL_PYTHON="$(command -v python)"
export REAL_PYTHON

cd "$REPO"
mkdir -p ./dataset_argvcapture

# pretrained|thinking|internvl_force_close|backend_override|r1_variant
CELLS=(
  "Qwen/Qwen3.5-4B|0|||"
  "Qwen/Qwen3.5-4B|1|||"
  "google/gemma-4-E2B-it|0|||"
  "google/gemma-4-E2B-it|1|||"
  "OpenGVLab/InternVL3_5-4B|0|||"
  "OpenGVLab/InternVL3_5-4B|1|||"
  "OpenGVLab/InternVL3_5-4B|0||vllm|"
  "OpenGVLab/InternVL3_5-4B|1||vllm|"
  "OpenGVLab/InternVL3_5-4B|1|0|vllm|"
  "OpenGVLab/InternVL3_5-4B|1||vllm|internvl_r1_v3"
  "Qwen/Qwen3-VL-4B-Instruct|0|||"
  "Qwen/Qwen3-VL-4B-Instruct|1|||"
  "Qwen/Qwen3-VL-4B-Thinking|0|||"
  "Qwen/Qwen2.5-VL-3B-Instruct|0|||"
)

echo "{" > "$OUT"
first=1
for cell in "${CELLS[@]}"; do
  IFS='|' read -r pretrained thinking forceclose override variant <<< "$cell"
  key="${pretrained}|think=${thinking}|fc=${forceclose}|name=${override}|var=${variant}"

  # PATH is prepended rather than replaced: `env -i` drops APPDATA, which is
  # where this platform's Python keeps its user site-packages.
  captured="$(
    env PATH="$WORK/bin:$PATH" \
      MODEL_PRETRAINED="$pretrained" THINKING="$thinking" \
      INTERNVL_FORCE_CLOSE="${forceclose:-1}" \
      MODEL_NAME_OVERRIDE="$override" \
      INTERNVL_R1_VARIANT="$variant" \
      NUM_SAMPLES=10 DIFFICULTY=easy DATASET_DIR=./dataset_argvcapture \
      JOB_NAME=argvcapture CHUNK_SIZE=0 PROMPT_ID=direct_v1 \
      bash examples/models/dynamic_graph_benchmark/run_eval.sh 2>&1 \
      | grep '^ARGV_CAPTURE ' | head -1 | sed 's/^ARGV_CAPTURE //'
  )" || true

  if [ -z "$captured" ]; then
    echo "  !! no argv captured for $key" >&2
    continue
  fi
  [ $first -eq 1 ] || echo "," >> "$OUT"
  first=0
  printf '  %s: %s' "$("$REAL_PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$key")" "$captured" >> "$OUT"
  echo "  captured: $key"
done
echo "" >> "$OUT"
echo "}" >> "$OUT"
"$REAL_PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'{len(d)} cells written to {sys.argv[1]}')" "$OUT"
