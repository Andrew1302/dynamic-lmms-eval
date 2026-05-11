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

The script expects the dynamic-dataset sibling repo at ../dynamic-dataset
(i.e. C:/Users/Andrew/Msc/dynamic-dataset).

Output is a HuggingFace DatasetDict saved to --output-dir (default:
./dynamic_graph_benchmark_data). Each BenchmarkTask generation emits two rows:
one with ``variant="direct"`` and one with ``variant="disguise"``.
"""

from __future__ import annotations

import argparse
import importlib
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_tasks = benchmark.get_all_tasks()
    task_names = args.tasks if args.tasks else sorted(all_tasks.keys())

    unknown = set(task_names) - set(all_tasks.keys())
    if unknown:
        sys.exit(f"[prepare_dynamic_graph_benchmark] Unknown tasks: {unknown}. Available: {sorted(all_tasks.keys())}")

    difficulty_for: dict[str, str] = {t: args.difficulty for t in task_names}
    for task, level in args.difficulty_override:
        if task not in difficulty_for:
            sys.exit(
                f"[prepare_dynamic_graph_benchmark] --difficulty-override references "
                f"unknown/unselected task {task!r}. Selected tasks: {task_names}"
            )
        difficulty_for[task] = level

    print(f"[prepare_dynamic_graph_benchmark] Generating {args.num_samples} generations per task "
          f"across {task_names} (difficulty={difficulty_for}, seed={args.seed})")

    rows = []
    for task_name in task_names:
        task_cls = all_tasks[task_name]
        task = task_cls()
        difficulty = difficulty_for[task_name]
        for i in range(args.num_samples):
            seed = args.seed + i * 1000 + (hash(task_name) % 1000)
            sample = task.generate(seed=seed, difficulty=difficulty)
            answer = str(sample["answer"])

            rows.append({
                "id": f"{task_name}_direct_{i:04d}",
                "task": task_name,
                "variant": "direct",
                "difficulty": difficulty,
                "seed": seed,
                "prompt": sample["direct_prompt"],
                "image": sample["direct_image"].convert("RGB"),
                "answer": answer,
            })
            rows.append({
                "id": f"{task_name}_disguise_{i:04d}",
                "task": task_name,
                "variant": "disguise",
                "difficulty": difficulty,
                "seed": seed,
                "prompt": sample["disguise_prompt"],
                "image": sample["disguise_image"].convert("RGB"),
                "answer": answer,
            })

    print(f"[prepare_dynamic_graph_benchmark] Generated {len(rows)} rows "
          f"({len(rows)//2} direct + {len(rows)//2} disguise). Building HuggingFace dataset ...")

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

    # Export images for debugging / inspection.
    images_dir = output_dir.parent / "dynamic_graph_benchmark_images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    for row in rows:
        fname = images_dir / f"{row['id']}.png"
        row["image"].save(str(fname))

    print(f"[prepare_dynamic_graph_benchmark] Exported {len(rows)} images to {images_dir.resolve()}")


if __name__ == "__main__":
    main()
