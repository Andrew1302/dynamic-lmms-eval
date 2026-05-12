"""Generate a Dynamic Graph Benchmark dataset and save it to disk for lmms-eval.

Unlike ``prepare_dynamic_graph_qa.py`` (which targets ``src.dataset-generator``),
this script targets the newer ``src.benchmark`` package, whose ``BenchmarkTask``
contract produces *two* paired samples per generation: a ``direct`` rendering
(plain graph) and a ``disguise`` rendering (maze / map / etc.), both sharing
the same ground-truth answer.

Run from the dynamic-lmms-eval repo root:
    python tools/prepare_dynamic_graph_benchmark.py
    python tools/prepare_dynamic_graph_benchmark.py --num-samples 60 --difficulty medium
    python tools/prepare_dynamic_graph_benchmark.py --tasks connectivity coloring --seed 0

Ablation flags (forwarded to RenderConfig in the benchmark package):
    --label-style {numeric,letters,none}
    --node-color HEX
    --edge-style {straight,curved}
    --include-adjacency-matrix

Sweep mode (vertex- or edge-count axis):
    --constraint {nodes,edges}
    --constraint-values  LO..HI[:STEP] | v1,v2,v3
    --samples-per-value  N

When --constraint is set, --num-samples and --difficulty are ignored;
the script materializes rows for every (task, value, sample_idx) and
attaches achieved n_vertices/n_edges to each row.

The script expects the dynamic-dataset sibling repo at ../dynamic-dataset
(i.e. C:/Users/Andrew/Msc/dynamic-dataset).

Output is a HuggingFace DatasetDict saved to --output-dir (default:
./dynamic_graph_benchmark_data). Each BenchmarkTask generation emits two rows:
one with ``variant="direct"`` and one with ``variant="disguise"``.
"""

from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
from pathlib import Path

_FORK_ROOT = Path(__file__).resolve().parent.parent
_DYNAMIC_DATASET_ROOT = _FORK_ROOT.parent / "dynamic-dataset"

if not _DYNAMIC_DATASET_ROOT.exists():
    sys.exit(
        f"[prepare_dynamic_graph_benchmark] dynamic-dataset repo not found at "
        f"{_DYNAMIC_DATASET_ROOT}. Clone it as a sibling of dynamic-lmms-eval."
    )

if str(_DYNAMIC_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYNAMIC_DATASET_ROOT))

benchmark = importlib.import_module("src.benchmark")
RenderConfig = benchmark.RenderConfig


def _parse_difficulty_override(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--difficulty-override must be TASK=LEVEL, got {spec!r}"
        )
    task, level = spec.split("=", 1)
    if level not in {"easy", "medium", "hard"}:
        raise argparse.ArgumentTypeError(
            f"--difficulty-override level must be easy/medium/hard, got {level!r}"
        )
    return task, level


def _parse_constraint_values(spec: str) -> list[int]:
    if ".." in spec:
        body, _, step_s = spec.partition(":")
        step = int(step_s) if step_s else 1
        lo_s, _, hi_s = body.partition("..")
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            raise argparse.ArgumentTypeError(f"constraint range lo > hi: {spec!r}")
        return list(range(lo, hi + 1, step))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(spec)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Dynamic Graph Benchmark dataset for lmms-eval")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of generations per task (each yields 1 direct + 1 disguise row)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                        help="Default difficulty applied to every task. Override per task with --difficulty-override.")
    parser.add_argument("--difficulty-override", action="append", default=[], type=_parse_difficulty_override,
                        metavar="TASK=LEVEL",
                        help="Per-task difficulty override, repeatable (e.g. --difficulty-override shortest_path=easy).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./dynamic_graph_benchmark_data")
    parser.add_argument("--tasks", nargs="+", default=None, help="Subset of tasks to include (default: all registered)")

    # Ablation knobs -> RenderConfig
    parser.add_argument("--label-style", choices=["numeric", "letters", "none"], default="numeric",
                        help="Node-label style on the direct-view image.")
    parser.add_argument("--node-color", default="#AED6F1",
                        help="Default node fill colour (hex). Highlights are unaffected.")
    parser.add_argument("--edge-style", choices=["straight", "curved"], default="straight",
                        help="Edge style for weighted graphs. Unweighted graphs are always straight.")
    parser.add_argument("--include-adjacency-matrix", action="store_true",
                        help="Append a text adjacency matrix to the direct-view prompt.")

    # Sweep mode
    parser.add_argument("--constraint", choices=["nodes", "edges"], default=None,
                        help="Sweep over this axis. When set, --num-samples and --difficulty are ignored.")
    parser.add_argument("--constraint-values", default=None,
                        help="Values to sweep: LO..HI[:STEP] or comma-separated list.")
    parser.add_argument("--samples-per-value", type=int, default=None,
                        help="Samples per constraint value. Defaults: 250 (nodes), 100 (edges).")
    parser.add_argument("--edge-tolerance", type=float, default=0.15,
                        help="For --constraint edges: accept samples within (1±tol) of target.")
    parser.add_argument("--edge-max-attempts", type=int, default=200,
                        help="Max rejection-sampling attempts per (task, edge target, sample).")

    args = parser.parse_args()
    if args.constraint is not None and args.constraint_values is None:
        parser.error("--constraint requires --constraint-values")
    return args


def _build_config(args: argparse.Namespace):
    return RenderConfig(
        label_style=args.label_style,
        node_color=args.node_color,
        edge_style=args.edge_style,
    )


def _sample_for_edge_target(task, target_edges, cfg, include_adj, base_seed, tol, max_attempts):
    """Rejection-sample by varying node count to hit the edge target."""
    v_lo = max(3, target_edges // 3 - 2)
    v_hi = max(v_lo + 2, target_edges + 4)
    tol_lo, tol_hi = (1 - tol) * target_edges, (1 + tol) * target_edges
    rng = random.Random(base_seed)
    best_sample = None
    best_diff = math.inf
    for attempt in range(max_attempts):
        nc = rng.randint(v_lo, v_hi)
        seed = base_seed + attempt * 101
        sample = task.generate(
            seed=seed, difficulty="medium", config=cfg,
            include_adjacency_matrix=include_adj, node_count=nc,
        )
        diff = abs(sample["n_edges"] - target_edges)
        if diff < best_diff:
            best_sample = sample
            best_diff = diff
        if tol_lo <= sample["n_edges"] <= tol_hi:
            return sample, seed
    assert best_sample is not None
    return best_sample, base_seed


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)

    all_tasks = benchmark.get_all_tasks()
    task_names = args.tasks if args.tasks else sorted(all_tasks.keys())
    unknown = set(task_names) - set(all_tasks.keys())
    if unknown:
        sys.exit(f"[prepare_dynamic_graph_benchmark] Unknown tasks: {unknown}. Available: {sorted(all_tasks.keys())}")

    if args.constraint is None:
        rows = _generate_standard(args, cfg, task_names, all_tasks)
        mode_desc = f"standard mode: {args.num_samples} generations per task"
    else:
        rows = _generate_sweep(args, cfg, task_names, all_tasks)
        mode_desc = (
            f"sweep mode: constraint={args.constraint} "
            f"values={_parse_constraint_values(args.constraint_values)} "
            f"samples_per_value={args.samples_per_value or (250 if args.constraint == 'nodes' else 100)}"
        )

    print(f"[prepare_dynamic_graph_benchmark] {mode_desc}; generated {len(rows)} rows. Building HuggingFace dataset ...")

    import datasets
    from datasets import DatasetDict, Features, Image, Value

    features = Features({
        "id": Value("string"),
        "task": Value("string"),
        "variant": Value("string"),
        "difficulty": Value("string"),
        "seed": Value("int64"),
        "prompt": Value("string"),
        "image": Image(),
        "answer": Value("string"),
        "n_vertices": Value("int64"),
        "n_edges": Value("int64"),
        "label_style": Value("string"),
        "node_color": Value("string"),
        "edge_style": Value("string"),
        "include_adjacency_matrix": Value("bool"),
        "constraint": Value("string"),
        "constraint_value": Value("int64"),
    })

    ds = datasets.Dataset.from_list(rows, features=features)
    dataset_dict = DatasetDict({"test": ds})

    import shutil

    output_dir = Path(args.output_dir)
    # Wipe stale shards from previous runs — save_to_disk overwrites the
    # primary arrow files but can leave orphaned shards behind when the new
    # row count maps to fewer shards, which silently truncates loaded data.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"[prepare_dynamic_graph_benchmark] Saved to {output_dir.resolve()}")
    print(f"[prepare_dynamic_graph_benchmark] Split 'test' has {len(ds)} samples.")

    images_dir = output_dir.parent / f"{output_dir.name}_images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    for row in rows:
        fname = images_dir / f"{row['id']}.png"
        row["image"].save(str(fname))
    print(f"[prepare_dynamic_graph_benchmark] Exported {len(rows)} images to {images_dir.resolve()}")


def _generate_standard(args, cfg, task_names, all_tasks) -> list[dict]:
    difficulty_for: dict[str, str] = {t: args.difficulty for t in task_names}
    for task, level in args.difficulty_override:
        if task not in difficulty_for:
            sys.exit(
                f"[prepare_dynamic_graph_benchmark] --difficulty-override references "
                f"unknown/unselected task {task!r}. Selected tasks: {task_names}"
            )
        difficulty_for[task] = level

    print(f"[prepare_dynamic_graph_benchmark] Generating {args.num_samples} generations per task "
          f"across {task_names} (difficulty={difficulty_for}, seed={args.seed}, "
          f"label_style={args.label_style}, edge_style={args.edge_style}, "
          f"adj_matrix={args.include_adjacency_matrix})")

    rows = []
    for task_name in task_names:
        task_cls = all_tasks[task_name]
        task = task_cls()
        difficulty = difficulty_for[task_name]
        for i in range(args.num_samples):
            seed = args.seed + i * 1000 + (hash(task_name) % 1000)
            sample = task.generate(
                seed=seed, difficulty=difficulty, config=cfg,
                include_adjacency_matrix=args.include_adjacency_matrix,
            )
            rows.extend(_row_pair(
                task_name=task_name, variant_seed=i, sample=sample, args=args,
                difficulty=difficulty, seed=seed, constraint=None, constraint_value=-1,
            ))
    return rows


def _generate_sweep(args, cfg, task_names, all_tasks) -> list[dict]:
    values = _parse_constraint_values(args.constraint_values)
    spv = args.samples_per_value or (250 if args.constraint == "nodes" else 100)
    print(f"[prepare_dynamic_graph_benchmark] Sweep: constraint={args.constraint} "
          f"values={values} spv={spv} tasks={task_names} "
          f"label_style={args.label_style} adj_matrix={args.include_adjacency_matrix}")

    rows = []
    counter = 0
    for task_name in task_names:
        task_cls = all_tasks[task_name]
        task = task_cls()
        for value in values:
            for s in range(spv):
                seed = args.seed + counter * 7919
                counter += 1
                if args.constraint == "nodes":
                    sample = task.generate(
                        seed=seed, difficulty="medium", config=cfg,
                        include_adjacency_matrix=args.include_adjacency_matrix,
                        node_count=value,
                    )
                    used_seed = seed
                else:
                    sample, used_seed = _sample_for_edge_target(
                        task=task, target_edges=value, cfg=cfg,
                        include_adj=args.include_adjacency_matrix,
                        base_seed=seed, tol=args.edge_tolerance,
                        max_attempts=args.edge_max_attempts,
                    )
                rows.extend(_row_pair(
                    task_name=task_name, variant_seed=counter, sample=sample, args=args,
                    difficulty="medium", seed=used_seed,
                    constraint=args.constraint, constraint_value=value,
                ))
            print(f"  [{task_name}] value={value}: {spv} samples done")
    return rows


def _row_pair(*, task_name, variant_seed, sample, args, difficulty, seed,
              constraint, constraint_value) -> list[dict]:
    common = {
        "task": task_name,
        "difficulty": difficulty,
        "seed": seed,
        "n_vertices": int(sample["n_vertices"]),
        "n_edges": int(sample["n_edges"]),
        "label_style": args.label_style,
        "node_color": args.node_color,
        "edge_style": args.edge_style,
        "include_adjacency_matrix": bool(args.include_adjacency_matrix),
        "constraint": constraint or "",
        "constraint_value": int(constraint_value),
        "answer": str(sample["answer"]),
    }
    return [
        {**common,
         "id": f"{task_name}_direct_{variant_seed:06d}",
         "variant": "direct",
         "prompt": sample["direct_prompt"],
         "image": sample["direct_image"].convert("RGB")},
        {**common,
         "id": f"{task_name}_disguise_{variant_seed:06d}",
         "variant": "disguise",
         "prompt": sample["disguise_prompt"],
         "image": sample["disguise_image"].convert("RGB")},
    ]


if __name__ == "__main__":
    main()
