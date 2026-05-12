"""Plot accuracy vs. number of vertices, grouped by (task, variant, model).

Usage:
    python tools/postprocess/plot_accuracy_vs_nodes.py \
        --results remote_results/graph_bench_sweep_nodes_qwen3vl_4b/logs \
        --results remote_results/graph_bench_sweep_nodes_internvl35_4b/logs \
        --output-dir analysis/

Produces, per base task:
    analysis/plots/accuracy_vs_nodes_<base_task>.png
    analysis/tables/accuracy_vs_nodes.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _logs import find_sample_jsonls, iter_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", action="append", required=True,
                   help="One or more results dirs (each gets a series). Repeat the flag.")
    p.add_argument("--output-dir", default="./analysis")
    p.add_argument("--xkey", choices=["n_vertices", "constraint_value"],
                   default="n_vertices",
                   help="Bin by observed n_vertices (default) or by the requested sweep value.")
    return p.parse_args()


def _job_label(path: str) -> str:
    """Derive a human-readable series label from the results-dir path."""
    return Path(path).parent.name or Path(path).name


def main() -> int:
    args = parse_args()
    plots_dir = Path(args.output_dir) / "plots"
    tables_dir = Path(args.output_dir) / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # accuracy[base_task][variant][job_label][x] = (correct, total)
    acc: dict[str, dict[str, dict[str, dict[int, list[int]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))))

    for results_dir in args.results:
        label = _job_label(results_dir)
        rows = list(iter_rows(find_sample_jsonls(Path(results_dir)), job_label=label))
        if not rows:
            print(f"[warn] no sample rows under {results_dir}")
            continue
        for r in rows:
            x = getattr(r, args.xkey)
            if x <= 0:
                continue
            bucket = acc[r.base_task][r.variant][r.job][x]
            bucket[0] += r.correct
            bucket[1] += 1

    # CSV with every (base_task, variant, job, x) point.
    csv_path = tables_dir / "accuracy_vs_nodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base_task", "variant", "job", "x", "correct", "total", "accuracy"])
        for base, by_var in sorted(acc.items()):
            for var, by_job in sorted(by_var.items()):
                for job, by_x in sorted(by_job.items()):
                    for x, (c, n) in sorted(by_x.items()):
                        w.writerow([base, var, job, x, c, n, c / n if n else 0])
    print(f"[csv] wrote {csv_path}")

    # One PNG per base_task: x = node count, y = accuracy, lines per (variant, job).
    for base, by_var in sorted(acc.items()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for var, by_job in sorted(by_var.items()):
            for job, by_x in sorted(by_job.items()):
                xs = sorted(by_x.keys())
                ys = [by_x[x][0] / by_x[x][1] if by_x[x][1] else 0 for x in xs]
                ls = "-" if var == "direct" else "--"
                ax.plot(xs, ys, marker="o", linestyle=ls, label=f"{job} ({var})")
        ax.set_title(f"{base}: accuracy vs {args.xkey}")
        ax.set_xlabel(args.xkey)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"accuracy_vs_nodes_{base}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
