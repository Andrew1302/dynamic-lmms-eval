"""Dynamic Graph Benchmark task utilities for lmms-eval.

The benchmark package (``dynamic-dataset/src/benchmark``) generates, for each
graph task, a pair of samples: a ``direct`` rendering (plain graph image) and
a ``disguise`` rendering (maze / map / etc.), sharing a single ground-truth
answer. This module:

  * filters a combined on-disk dataset down to a specific (task, variant) slice
    via ``process_docs_*`` hooks bound from each leaf YAML;
  * formats prompts for both legacy ``doc_to_text`` consumers and chat-style
    ``doc_to_messages`` models;
  * normalizes predictions per-task (yes/no vs. integer) so that model output
    shape matches the ground-truth answer shape;
  * aggregates scores into per-(task, variant) accuracy plus an overall number.
"""

from __future__ import annotations

import re
from collections import defaultdict

from loguru import logger as eval_logger

_YESNO_TASKS = {"connectivity"}
_INTEGER_TASKS = {"coloring"}

_YES_PATTERNS = {"yes", "y", "true", "t"}
_NO_PATTERNS = {"no", "n", "false", "f"}


def _normalize(prediction: str, task: str) -> str:
    pred = (prediction or "").strip()

    if task in _YESNO_TASKS:
        first = pred.lower().split()
        if not first:
            return ""
        token = re.sub(r"[^a-z]", "", first[0])
        if token in _YES_PATTERNS:
            return "yes"
        if token in _NO_PATTERNS:
            return "no"
        return token

    if task in _INTEGER_TASKS:
        match = re.search(r"-?\d+", pred)
        return match.group(0) if match else pred.lower()

    return pred.lower()


def _normalize_answer(answer: str, task: str) -> str:
    return _normalize(answer, task)


def dynamic_graph_benchmark_doc_to_visual(doc):
    if doc.get("image") is not None:
        return [doc["image"].convert("RGB")]
    return []


def dynamic_graph_benchmark_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    return f"{pre}{doc['prompt']}{post}"


def dynamic_graph_benchmark_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    visuals = dynamic_graph_benchmark_doc_to_visual(doc)
    text = dynamic_graph_benchmark_doc_to_text(doc, lmms_eval_specific_kwargs)

    content = []
    for visual in visuals:
        content.append({"type": "image", "url": visual})
    content.append({"type": "text", "text": text})

    return [{"role": "user", "content": content}]


def dynamic_graph_benchmark_process_results(doc, results):
    prediction = results[0] if results else ""
    task = doc.get("task", "unknown")
    variant = doc.get("variant", "unknown")
    answer = str(doc.get("answer", ""))

    norm_pred = _normalize(prediction, task)
    norm_gt = _normalize_answer(answer, task)

    score = 1.0 if norm_pred == norm_gt else 0.0
    return {"accuracy": {"task": task, "variant": variant, "score": score}}


def dynamic_graph_benchmark_aggregate_results(results):
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        bucket[(r["task"], r["variant"])].append(r["score"])

    total_correct = 0.0
    total_samples = 0
    for (task, variant), scores in sorted(bucket.items()):
        acc = sum(scores) / len(scores)
        eval_logger.info(
            f"dynamic_graph_benchmark | {task}/{variant}: "
            f"{acc:.3f} ({int(sum(scores))}/{len(scores)})"
        )
        total_correct += sum(scores)
        total_samples += len(scores)

    overall = total_correct / total_samples if total_samples > 0 else 0.0
    eval_logger.info(
        f"dynamic_graph_benchmark | overall: "
        f"{overall:.3f} ({int(total_correct)}/{total_samples})"
    )
    return overall


def _filter(dataset, task: str, variant: str):
    return dataset.filter(lambda row: row["task"] == task and row["variant"] == variant)


def filter_connectivity_direct(dataset):
    return _filter(dataset, "connectivity", "direct")


def filter_connectivity_disguise(dataset):
    return _filter(dataset, "connectivity", "disguise")


def filter_coloring_direct(dataset):
    return _filter(dataset, "coloring", "direct")


def filter_coloring_disguise(dataset):
    return _filter(dataset, "coloring", "disguise")
