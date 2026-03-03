"""Generate a Dynamic Graph QA dataset and save it to disk for lmms-eval.

Run from the dynamic-lmms-eval repo root:
    python tools/prepare_dynamic_graph_qa.py
    python tools/prepare_dynamic_graph_qa.py --num-samples 280 --size medium --seed 0
    python tools/prepare_dynamic_graph_qa.py --tasks node_count cycle_check shortest_path

The script expects the dynamic-dataset sibling repo at ../dynamic-dataset relative to
the repo root (i.e. C:/Users/Andrew/Msc/dynamic-dataset).

Output is a HuggingFace DatasetDict saved to --output-dir (default: ./dynamic_graph_qa_data).
Load it in lmms-eval via dataset_kwargs: {load_from_disk: True}.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve dynamic-dataset sibling repo and add to sys.path so that
# importlib.import_module("src.dataset-generator") works.
# ---------------------------------------------------------------------------
_FORK_ROOT = Path(__file__).resolve().parent.parent  # dynamic-lmms-eval/
_DYNAMIC_DATASET_ROOT = _FORK_ROOT.parent / "dynamic-dataset"

if not _DYNAMIC_DATASET_ROOT.exists():
    sys.exit(f"[prepare_dynamic_graph_qa] dynamic-dataset repo not found at {_DYNAMIC_DATASET_ROOT}. " "Clone it as a sibling of dynamic-lmms-eval.")

if str(_DYNAMIC_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYNAMIC_DATASET_ROOT))

pkg = importlib.import_module("src.dataset-generator")

# ---------------------------------------------------------------------------
# Graph generators per task family
# ---------------------------------------------------------------------------
_YESNO_TASKS = {"cycle_check", "edge_existence", "reachability", "connectivity_check"}
_LIST_TASKS = {"connected_nodes", "disconnected_nodes"}
_WEIGHTED_TASKS = {"mst", "shortest_path"}
_FLOW_TASKS = {"maximum_flow"}

# Tasks that need possibly-disconnected graphs for interesting samples
_POSSIBLY_DISCONNECTED_TASKS = {"reachability", "connectivity_check", "connected_components"}


def _make_graph(task_name: str, size: str):
    gg = pkg.graph_generator
    n = gg.random_node_count(size)
    if task_name in _FLOW_TASKS:
        return gg.random_directed_weighted_graph(n)
    if task_name in _WEIGHTED_TASKS:
        return gg.random_weighted_connected_graph(n)
    if task_name in _POSSIBLY_DISCONNECTED_TASKS:
        return gg.random_possibly_disconnected(n)
    return gg.random_graph(n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Dynamic Graph QA dataset for lmms-eval")
    parser.add_argument("--num-samples", type=int, default=140, help="Total number of samples to generate (default: 140 = 10 per task)")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small", help="Graph size preset (default: small, 5-9 nodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="./dynamic_graph_qa_data", help="Output directory for the saved HuggingFace dataset (default: ./dynamic_graph_qa_data)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Task names to include (default: all 14 tasks)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    all_tasks = pkg.get_all_tasks()
    task_names = args.tasks if args.tasks else list(all_tasks.keys())

    # Validate requested task names
    unknown = set(task_names) - set(all_tasks.keys())
    if unknown:
        sys.exit(f"[prepare_dynamic_graph_qa] Unknown tasks: {unknown}. Available: {list(all_tasks.keys())}")

    print(f"[prepare_dynamic_graph_qa] Generating {args.num_samples} samples across tasks: {task_names}")
    print(f"[prepare_dynamic_graph_qa] Graph size: {args.size}, seed: {args.seed}")

    rows = []
    for i in range(args.num_samples):
        task_name = task_names[i % len(task_names)]
        task_cls = all_tasks[task_name]
        G = _make_graph(task_name, args.size)
        sample = task_cls().generate(G)
        rows.append(
            {
                "id": i,
                "task": task_name,
                "prompt": sample["prompt"],
                "image": sample["image"].convert("RGB"),
                "answer": str(sample["answer"]),
            }
        )

    print(f"[prepare_dynamic_graph_qa] Generated {len(rows)} samples. Building HuggingFace dataset ...")

    import datasets
    from datasets import DatasetDict, Features, Image, Value

    features = Features(
        {
            "id": Value("int32"),
            "task": Value("string"),
            "prompt": Value("string"),
            "image": Image(),
            "answer": Value("string"),
        }
    )

    ds = datasets.Dataset.from_list(rows, features=features)
    dataset_dict = DatasetDict({"test": ds})

    output_dir = Path(args.output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"[prepare_dynamic_graph_qa] Saved to {output_dir.resolve()}")
    print(f"[prepare_dynamic_graph_qa] Split 'test' has {len(ds)} samples.")


if __name__ == "__main__":
    main()
