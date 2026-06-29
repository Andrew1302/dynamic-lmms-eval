#!/usr/bin/env python
"""Local launcher for API-backed models (e.g. Gemini) on the dynamic graph benchmark.

This is the Windows-friendly local analogue of ``run_eval.sh``: it reuses the exact
same dataset prep (``tools/prepare_dynamic_graph_benchmark.py``), the same task YAMLs,
and the same registered model wrappers + lmms-eval — but drops the VM-only machinery
(SSH deploy, ``ln -s`` symlink, ``accelerate launch``, chunking). API models need no
GPU, so running them locally is both cheaper and simpler than a VM round-trip.

It writes the prepared dataset straight into the canonical path the task YAMLs load
(``./dynamic_graph_benchmark_data``, ``load_from_disk: True``), so no symlink is needed.
Difficulties are run sequentially: prepare -> eval -> prepare (overwrite) -> eval -> ...

Usage (from the repo root, with the repo venv's python and GOOGLE_API_KEY set):

    # PowerShell:  $env:GOOGLE_API_KEY = "<key>"
    .venv\\Scripts\\python examples/models/dynamic_graph_benchmark/run_local_api.py

    # 1-sample smoke on a single difficulty (~4 API calls):
    .venv\\Scripts\\python examples/models/dynamic_graph_benchmark/run_local_api.py \\
        --num-samples 1 --difficulties easy

Defaults mirror the standard .conf (graph_bench_standard_*_*.conf): shortest_path,
10 samples/difficulty, numeric labels, node color #AED6F1, straight edges, seed 42,
model gemini_api / gemini-2.5-flash.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Repo root = three levels up from examples/models/dynamic_graph_benchmark/.
REPO_ROOT = Path(__file__).resolve().parents[3]
# The path the task YAMLs load via load_from_disk (see _default_template_yaml).
CANONICAL_DATASET_DIR = "./dynamic_graph_benchmark_data"


def _run(cmd: list[str]) -> None:
    """Run a subprocess from the repo root, streaming output; raise on failure."""
    print(f"\n[run_local_api] $ {' '.join(cmd)}", flush=True)
    # Force UTF-8 in the child so lmms-eval's results table (which contains
    # non-ASCII glyphs like the up-arrow) doesn't crash on a Windows cp1252
    # console with UnicodeEncodeError.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemini_api", help="lmms-eval --model registry name (default: gemini_api)")
    p.add_argument("--model-version", default="gemini-2.5-flash", help="API model version passed as model_version= (default: gemini-2.5-flash)")
    p.add_argument("--tasks", nargs="+", default=["shortest_path"], help="benchmark tasks (default: shortest_path). Each expands to _direct + _disguise subtasks.")
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"], choices=["easy", "medium", "hard"], help="difficulties to run sequentially (default: easy medium hard)")
    p.add_argument("--num-samples", type=int, default=10, help="generations per task per difficulty (default: 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-style", default="numeric", choices=["numeric", "letters", "none"])
    p.add_argument("--node-color", default="#AED6F1")
    p.add_argument("--edge-style", default="straight", choices=["straight", "curved"])
    p.add_argument("--include-adjacency-matrix", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=None, help="override per-generation output budget via --gen_kwargs max_new_tokens=N. Needed for reasoning models (e.g. gemini-3.x-pro) that would otherwise spend the task YAML's 64-token cap on thinking and return an empty answer.")
    p.add_argument("--max-tokens", type=int, default=None, help="optional hard token-budget ceiling across the run (lmms-eval --max_tokens); off by default")
    p.add_argument("--job-prefix", default=None, help="output dir prefix under ./logs (default derived from model-version + tasks)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Fail fast before any (cost-incurring) dataset prep or API call.
    if not os.environ.get("GOOGLE_API_KEY"):
        print("[run_local_api] FATAL: GOOGLE_API_KEY is not set. In PowerShell: $env:GOOGLE_API_KEY = \"<key>\"", file=sys.stderr)
        return 2

    py = sys.executable
    variants = ("direct", "disguise")
    lmms_tasks = ",".join(f"dynamic_graph_benchmark_{t}_{v}" for t in args.tasks for v in variants)

    # A compact, filesystem-safe tag for the model version (e.g. gemini-2.5-flash -> gemini25flash).
    model_tag = args.model_version.replace("-", "").replace(".", "").replace("/", "_")
    tasks_tag = "_".join(args.tasks)
    job_prefix = args.job_prefix or f"local_{model_tag}_{tasks_tag}"

    print(f"[run_local_api] model={args.model} version={args.model_version}")
    print(f"[run_local_api] tasks={args.tasks} -> {lmms_tasks}")
    print(f"[run_local_api] difficulties={args.difficulties} num_samples={args.num_samples}")
    print(f"[run_local_api] repo_root={REPO_ROOT}")

    completed: list[tuple[str, str]] = []
    for diff in args.difficulties:
        job = f"{job_prefix}_{diff}"
        out_path = f"./logs/{job}"

        # --- Step 1: prepare the dataset directly into the canonical dir (overwrites
        # the previous difficulty's dataset; runs are strictly sequential). ----------
        prepare_args = [
            py, "tools/prepare_dynamic_graph_benchmark.py",
            "--seed", str(args.seed),
            "--tasks", *args.tasks,
            "--output-dir", CANONICAL_DATASET_DIR,
            "--num-samples", str(args.num_samples),
            "--difficulty", diff,
            "--label-style", args.label_style,
            "--node-color", args.node_color,
            "--edge-style", args.edge_style,
        ]
        if args.include_adjacency_matrix:
            prepare_args.append("--include-adjacency-matrix")
        _run(prepare_args)

        # --- Step 2: run lmms-eval on the direct + disguise subtasks. ----------------
        # Plain `python -m lmms_eval` (gemini_api's Accelerator() runs fine
        # single-process); avoids accelerate launch's process-spawn quirks on Windows.
        eval_args = [
            py, "-m", "lmms_eval",
            "--model", args.model,
            "--model_args", f"model_version={args.model_version}",
            "--tasks", lmms_tasks,
            "--batch_size", str(args.batch_size),
            "--limit", str(args.num_samples),  # belt-and-suspenders cap on docs evaluated
            "--log_samples",
            "--output_path", out_path,
        ]
        if args.max_new_tokens is not None:
            eval_args += ["--gen_kwargs", f"max_new_tokens={args.max_new_tokens}"]
        if args.max_tokens is not None:
            eval_args += ["--max_tokens", str(args.max_tokens)]
        _run(eval_args)

        completed.append((diff, out_path))

    print("\n[run_local_api] done. Results:")
    for diff, out_path in completed:
        print(f"  {diff:6s} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
