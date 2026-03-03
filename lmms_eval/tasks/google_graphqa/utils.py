"""Google GraphQA task utilities for lmms-eval.

Text-only benchmark: baharef/GraphQA on HuggingFace.
Each document has a graph described in natural language, a question, and an answer.

Supported configs (dataset_name):
    connected_nodes, cycle_check, disconnected_nodes, edge_count, edge_existence,
    maximum_flow, node_classification, node_count, node_degree, reachability,
    shortest_path, triangle_counting

Answer normalization is inferred from the ground-truth answer format:
  - Yes / No answers     → lowercase first word
  - Comma-separated list → strip items, sort numerically, rejoin
  - Plain number         → strip whitespace and trailing punctuation
"""

from __future__ import annotations

import re

from loguru import logger as eval_logger


# ---------------------------------------------------------------------------
# Answer normalisation helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip whitespace and trailing period/comma from a string."""
    return text.strip().rstrip(".,")


def _is_yesno(answer: str) -> bool:
    first = _clean(answer).split()[0].lower() if _clean(answer) else ""
    return first in {"yes", "no"}


def _is_list(answer: str) -> bool:
    return "," in _clean(answer)


def _normalize_answer(raw: str) -> str:
    cleaned = _clean(raw)
    if _is_yesno(cleaned):
        return cleaned.split()[0].lower()
    if _is_list(cleaned):
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
        try:
            items = sorted(items, key=lambda x: int(x))
        except ValueError:
            items = sorted(items)
        return ", ".join(items)
    # numeric — extract first integer/float found
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return match.group(0) if match else cleaned.lower()


def _normalize_prediction(pred: str, gt_answer: str) -> str:
    """Normalise model prediction using the same strategy as the ground truth."""
    gt_clean = _clean(gt_answer)
    if _is_yesno(gt_clean):
        # Extract first yes/no word anywhere in the prediction
        first = _clean(pred).split()[0].lower() if pred.strip() else ""
        if first in {"yes", "no"}:
            return first
        lower = pred.lower()
        if lower.startswith("yes"):
            return "yes"
        if lower.startswith("no"):
            return "no"
        return first
    if _is_list(gt_clean):
        items = [item.strip() for item in _clean(pred).split(",") if item.strip()]
        try:
            items = sorted(items, key=lambda x: int(x))
        except ValueError:
            items = sorted(items)
        return ", ".join(items)
    match = re.search(r"-?\d+(?:\.\d+)?", pred)
    return match.group(0) if match else _clean(pred).lower()


# ---------------------------------------------------------------------------
# lmms-eval interface
# ---------------------------------------------------------------------------

def google_graphqa_doc_to_visual(doc):
    """Text-only task — no visual input."""
    return []


def google_graphqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Return the question string (already ends with 'A: ')."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    return f"{pre}{doc['question']}{post}"


def google_graphqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Build a text-only ChatMessages-protocol message (no image content)."""
    text = google_graphqa_doc_to_text(doc, lmms_eval_specific_kwargs)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def google_graphqa_process_results(doc, results):
    """Score a single sample and return per-sample accuracy."""
    prediction = results[0] if results else ""
    gt_answer = str(doc.get("answer", ""))

    norm_pred = _normalize_prediction(prediction, gt_answer)
    norm_gt = _normalize_answer(gt_answer)

    score = 1.0 if norm_pred == norm_gt else 0.0
    return {"accuracy": score}


def google_graphqa_aggregate_results(results):
    """Aggregate per-sample accuracy scores into an overall mean."""
    if not results:
        return 0.0
    overall = sum(results) / len(results)
    eval_logger.info(f"google_graphqa | overall accuracy: {overall:.3f} ({int(sum(results))}/{len(results)})")
    return overall
