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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Dynamic Graph Benchmark dataset for lmms-eval")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of generations per task (each yields 1 direct + 1 disguise row)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
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

    print(f"[prepare_dynamic_graph_benchmark] Generating {args.num_samples} generations per task "
          f"across {task_names} (difficulty={args.difficulty}, seed={args.seed})")

    rows = []
    for task_name in task_names:
        task_cls = all_tasks[task_name]
        task = task_cls()
        for i in range(args.num_samples):
            seed = args.seed + i * 1000 + (hash(task_name) % 1000)
            sample = task.generate(seed=seed, difficulty=args.difficulty)
            answer = str(sample["answer"])

            rows.append({
                "id": f"{task_name}_direct_{i:04d}",
                "task": task_name,
                "variant": "direct",
                "difficulty": args.difficulty,
                "seed": seed,
                "prompt": sample["direct_prompt"],
                "image": sample["direct_image"].convert("RGB"),
                "answer": answer,
            })
            rows.append({
                "id": f"{task_name}_disguise_{i:04d}",
                "task": task_name,
                "variant": "disguise",
                "difficulty": args.difficulty,
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

    output_dir = Path(args.output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"[prepare_dynamic_graph_benchmark] Saved to {output_dir.resolve()}")
    print(f"[prepare_dynamic_graph_benchmark] Split 'test' has {len(ds)} samples.")

    # Export images for debugging / inspection.
    import shutil

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
