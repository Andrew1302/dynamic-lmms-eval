"""Dynamic Graph Benchmark task hooks for lmms-eval.

This module is a thin adapter. Everything about *how* a question is asked and
*how* the answer is read back lives in the top-level ``prompting`` package,
selected per run by the ``PROMPT_ID`` environment variable (default
``direct_v1``, which reproduces every campaign up to 17_abl_scram byte for
byte). ``process_results`` is arity-2 and never receives
``lmms_eval_specific_kwargs``, so an env var is the only transport that reaches
the text, visual and scoring hooks alike.

What stays here: dataset slicing (``process_docs`` filters), the image-control
ablations, and score aggregation.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger as eval_logger

# House style (tools/prepare_dynamic_graph_benchmark.py:62, run_local_responses.py:51):
# make the repo root importable explicitly rather than relying on cwd. `!function`
# loads this file by path, so the task directory is not on sys.path either.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prompting import render  # noqa: E402
from prompting.registry import active_template  # noqa: E402


def _scramble_pixels(img, seed: int = 20260707):
    """Deterministically permute every pixel of an RGB image.

    Destroys all spatial structure (the graph is unrecognizable) while keeping
    the exact dimensions and colour histogram — so the model still receives an
    image of identical vision-token cost that carries no usable graph info.
    Used by the "no-image (scrambled)" control ablation: if accuracy is
    unchanged vs. the intact-image run, the image was not contributing.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(img)
    n = arr.shape[0] * arr.shape[1]
    perm = np.random.default_rng(seed).permutation(n)
    flat = arr.reshape(n, -1)
    return Image.fromarray(flat[perm].reshape(arr.shape))


def dynamic_graph_benchmark_doc_to_visual(doc):
    # Image ablations (control conditions). Gated by env so a single conf toggles
    # them without touching the dataset render, keeping graphs/prompts identical:
    #   NO_IMAGE=1        -> drop the image entirely (true no-image ablation)
    #   SCRAMBLE_IMAGE=1  -> pixel-shuffle the image (present but unrecognizable)
    # These are image controls, not prompt configuration — deliberately NOT part
    # of the prompt template.
    if os.environ.get("NO_IMAGE") == "1":
        return []
    if doc.get("image") is not None:
        img = doc["image"].convert("RGB")
        if os.environ.get("SCRAMBLE_IMAGE") == "1":
            img = _scramble_pixels(img)
        return [img]
    return []


def dynamic_graph_benchmark_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Flat prompt for the simple (single-turn) model wrappers."""
    template = active_template()
    text, _ = render.to_flat(template.turns(doc), template.system(doc), dynamic_graph_benchmark_doc_to_visual(doc))
    return text


def dynamic_graph_benchmark_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Multi-turn messages for the chat wrappers — the only path that can carry
    few-shot exemplars as real prior turns."""
    template = active_template()
    return render.to_messages(template.turns(doc), template.system(doc), dynamic_graph_benchmark_doc_to_visual(doc))


def dynamic_graph_benchmark_process_results(doc, results):
    prediction = results[0] if results else ""
    task = doc.get("task", "unknown")
    variant = doc.get("variant", "unknown")
    answer = str(doc.get("answer", ""))

    template = active_template()
    spec = template.answer_spec(task)
    score = 1.0 if spec.parse(prediction) == spec.parse(answer) else 0.0

    return {
        "accuracy": {
            "task": task,
            "variant": variant,
            "score": score,
            # lmms-eval strips most doc fields from the saved jsonl; surface
            # the axis info here so post-hoc reports can pivot per constraint
            # value without round-tripping through the cached HF dataset.
            "n_vertices": int(doc.get("n_vertices", 0) or 0),
            "n_edges": int(doc.get("n_edges", 0) or 0),
            "constraint": str(doc.get("constraint", "") or ""),
            "constraint_value": int(doc.get("constraint_value", -1) or -1),
            # Prompt provenance: which template produced this row, and a content
            # hash of it. Without this, a results tree cannot state how it was
            # asked — the gap that made prompt experiments unreproducible.
            "prompt_id": template.id,
            "prompt_fingerprint": template.fingerprint(),
        }
    }


def dynamic_graph_benchmark_aggregate_results(results):
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        bucket[(r["task"], r["variant"])].append(r["score"])

    total_correct = 0.0
    total_samples = 0
    for (task, variant), scores in sorted(bucket.items()):
        acc = sum(scores) / len(scores)
        eval_logger.info(f"dynamic_graph_benchmark | {task}/{variant}: {acc:.3f} ({int(sum(scores))}/{len(scores)})")
        total_correct += sum(scores)
        total_samples += len(scores)

    overall = total_correct / total_samples if total_samples > 0 else 0.0
    if results:
        eval_logger.info(f"dynamic_graph_benchmark | prompt: {results[0].get('prompt_id')} ({results[0].get('prompt_fingerprint')})")
    eval_logger.info(f"dynamic_graph_benchmark | overall: {overall:.3f} ({int(total_correct)}/{total_samples})")
    return overall


def _filter(dataset, task: str, variant: str):
    return dataset.filter(lambda row: row["task"] == task and row["variant"] == variant)


def filter_coloring_direct(dataset):
    return _filter(dataset, "coloring", "direct")


def filter_coloring_disguise(dataset):
    return _filter(dataset, "coloring", "disguise")


def filter_directed_connectivity_direct(dataset):
    return _filter(dataset, "directed_connectivity", "direct")


def filter_directed_connectivity_disguise(dataset):
    return _filter(dataset, "directed_connectivity", "disguise")


def filter_shortest_path_direct(dataset):
    return _filter(dataset, "shortest_path", "direct")


def filter_shortest_path_disguise(dataset):
    return _filter(dataset, "shortest_path", "disguise")
