#!/usr/bin/env python
"""Build the batch-report .xlsx from local run_local_api.py logs.

The VM report flow (``07_batch_report.sh`` -> ``tools/postprocess/batch_report.py``)
sources each job's ``.conf`` and hands the Python side a TSV of
``job_id, job_name, results_dir, model_pretrained, compare_pairs, constraint``.
Our local runs have no ``.conf`` files, so this helper synthesizes the same TSV
directly from the ``./logs/<job>`` trees that ``run_local_api.py`` produced, then
invokes the unchanged ``batch_report.py``.

Two small adapters make the existing report tooling "just work":

* ``batch_report.py`` expects ``results_dir/logs/.../*_samples_*.jsonl`` (it
  guards on ``results_dir/logs`` existing). Our logs live at
  ``./logs/<job>/<model>/...`` with no nested ``logs/``, so we stage each job's
  tree into a temp ``<stage>/<diff>/logs/<job>`` layout (copy, not move).
* ``batch_report.parse_job_id`` derives the difficulty from a
  ``graph_bench_standard_<diff>_*`` job-name prefix. We label the synthetic TSV
  rows that way so each difficulty becomes a clean ``axis=standard`` row with the
  difficulty in ``axis_value`` — identical in shape to the VM standard runs.

Usage (from repo root, repo venv python):

    .venv\\Scripts\\python examples/models/dynamic_graph_benchmark/make_local_report.py

    # match a non-default run:
    .venv\\Scripts\\python examples/models/dynamic_graph_benchmark/make_local_report.py \\
        --job-prefix local_gemini25flash_shortest_path \\
        --difficulties easy medium hard --tasks shortest_path \\
        --model-version gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = REPO_ROOT / "logs"
BATCH_REPORT = REPO_ROOT / "tools" / "postprocess" / "batch_report.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-prefix", default="local_gemini25flash_shortest_path", help="prefix of the ./logs/<prefix>_<difficulty> dirs to aggregate")
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"], choices=["easy", "medium", "hard"])
    p.add_argument("--tasks", nargs="+", default=["shortest_path"], help="benchmark tasks evaluated (drives the compare-pairs spec)")
    p.add_argument("--model-version", default="gemini-2.5-flash", help="model label shown in the report's model_pretrained column")
    p.add_argument("--output", default=None, help="output .xlsx path (default: ./logs/_reports/<prefix>_<gents>.xlsx)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_tag = args.model_version.replace("-", "").replace(".", "").replace("/", "_")
    pairs = "|".join(
        f"{t}:dynamic_graph_benchmark_{t}_direct:dynamic_graph_benchmark_{t}_disguise"
        for t in args.tasks
    )

    gen_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else (LOGS_DIR / "_reports" / f"{args.job_prefix}_{gen_ts}.xlsx")

    stage = Path(tempfile.mkdtemp(prefix="local_report_"))
    tsv_path = stage / "jobs.tsv"
    tsv_lines: list[str] = []
    staged = 0
    try:
        for diff in args.difficulties:
            src = LOGS_DIR / f"{args.job_prefix}_{diff}"
            if not src.is_dir() or not any(src.rglob("*_samples_*.jsonl")):
                print(f"[make_local_report] skip {diff}: no samples under {src}")
                continue
            # Stage into the layout batch_report.py expects: results_dir/logs/<job>/...
            job_name = f"graph_bench_standard_{diff}_{model_tag}"
            results_dir = stage / diff
            dst = results_dir / "logs" / job_name
            shutil.copytree(src, dst)
            # TSV: job_id, job_name, results_dir, model_pretrained, compare_pairs, constraint
            tsv_lines.append("\t".join([job_name, job_name, str(results_dir), args.model_version, pairs, ""]))
            staged += 1

        if staged == 0:
            print("[make_local_report] FATAL: no job logs found to report on.", file=sys.stderr)
            return 1

        tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(BATCH_REPORT),
            "--jobs-tsv", str(tsv_path),
            "--output", str(out_path),
            "--batch-name", args.job_prefix,
        ]
        print(f"[make_local_report] $ {' '.join(cmd)}")
        # Force UTF-8 so any non-ASCII console output doesn't crash on Windows cp1252.
        env = {**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
        print(f"[make_local_report] wrote {out_path}")
        return 0
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
