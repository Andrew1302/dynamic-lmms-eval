"""Plot accuracy vs. number of edges (binned), grouped by (task, variant, model).

Coloring's edges are determined by Delaunay so the edge sweep is
*observed* rather than directly controlled — we bin into uniform-width
buckets so the curve stays smooth. Other tasks bin at integer edge
counts directly.

Usage mirrors ``plot_accuracy_vs_nodes.py``:

    python tools/postprocess/plot_accuracy_vs_edges.py \
        --results remote_results/graph_bench_sweep_edges_qwen3vl_4b/logs \
        --output-dir analysis/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _logs import find_sample_jsonls, iter_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", action="append", required=True)
    p.add_argument("--output-dir", default="./analysis")
    p.add_argument("--bin-width", type=int, default=2,
                   help="Edge-count bin width for observed edge axis (default 2).")
    return p.parse_args()


def _job_label(path: str) -> str:
    return Path(path).parent.name or Path(path).name


def main() -> int:
    args = parse_args()
    plots_dir = Path(args.output_dir) / "plots"
    tables_dir = Path(args.output_dir) / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # acc[base][variant][job][bin_center] = [correct, total]
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0]))))
    bin_w = max(1, args.bin_width)

    for results_dir in args.results:
        label = _job_label(results_dir)
        rows = list(iter_rows(find_sample_jsonls(Path(results_dir)), job_label=label))
        if not rows:
            print(f"[warn] no sample rows under {results_dir}")
            continue
        for r in rows:
            if r.n_edges <= 0:
                continue
            center = (r.n_edges // bin_w) * bin_w + bin_w / 2
            b = acc[r.base_task][r.variant][r.job][center]
            b[0] += r.correct
            b[1] += 1

    csv_path = tables_dir / "accuracy_vs_edges.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base_task", "variant", "job", "edge_bin_center", "correct", "total", "accuracy"])
        for base, by_var in sorted(acc.items()):
            for var, by_job in sorted(by_var.items()):
                for job, by_x in sorted(by_job.items()):
                    for x, (c, n) in sorted(by_x.items()):
                        w.writerow([base, var, job, x, c, n, c / n if n else 0])
    print(f"[csv] wrote {csv_path}")

    for base, by_var in sorted(acc.items()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for var, by_job in sorted(by_var.items()):
            for job, by_x in sorted(by_job.items()):
                xs = sorted(by_x.keys())
                ys = [by_x[x][0] / by_x[x][1] if by_x[x][1] else 0 for x in xs]
                ls = "-" if var == "direct" else "--"
                ax.plot(xs, ys, marker="o", linestyle=ls, label=f"{job} ({var})")
        ax.set_title(f"{base}: accuracy vs n_edges (bin={bin_w})")
        ax.set_xlabel("n_edges (bin center)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = plots_dir / f"accuracy_vs_edges_{base}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
