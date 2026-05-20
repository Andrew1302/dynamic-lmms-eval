"""Shared jsonl loaders for fetched lmms-eval sample logs.

Each ``*_samples_<task>.jsonl`` row produced by lmms-eval contains:
    - ``doc_id`` — HF dataset row index
    - ``doc`` — the dataset row (so ``doc["n_vertices"]`` etc. land here)
    - ``filtered_resps`` — model response (string or [string])
    - ``target`` — ground-truth answer
    - ``accuracy.score`` — 0 or 1, set by the task's process_results hook

Loader helpers normalise these into a small ``SampleRow`` dataclass so
the analysis scripts don't repeat boilerplate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


SAMPLES_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_samples_(?P<task>.+)\.jsonl$")


@dataclass(frozen=True)
class SampleRow:
    job: str
    task: str                       # full lmms-eval task name (e.g. ..._direct)
    base_task: str                  # short task ("shortest_path")
    variant: str                    # "direct" | "disguise"
    doc_id: int
    target: str
    response: str
    correct: int                    # 0 or 1
    n_vertices: int
    n_edges: int
    label_style: str
    edge_style: str
    node_color: str
    include_adjacency_matrix: bool
    constraint: str
    constraint_value: int


def _resp(rec: dict) -> str:
    val = rec.get("filtered_resps", "")
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val)


def _correct(rec: dict) -> int:
    acc = rec.get("accuracy")
    if isinstance(acc, dict) and "score" in acc:
        return int(round(float(acc["score"])))
    return 0


def _base_task_and_variant(task_full: str) -> tuple[str, str]:
    """Split 'dynamic_graph_benchmark_<base>_<variant>' -> (base, variant)."""
    prefix = "dynamic_graph_benchmark_"
    name = task_full[len(prefix):] if task_full.startswith(prefix) else task_full
    if name.endswith("_direct"):
        return name[: -len("_direct")], "direct"
    if name.endswith("_disguise"):
        return name[: -len("_disguise")], "disguise"
    return name, "unknown"


def _doc(rec: dict) -> dict:
    """Return the doc payload, tolerating either rec["doc"] or flat layout."""
    if "doc" in rec and isinstance(rec["doc"], dict):
        return rec["doc"]
    return rec


def iter_rows(jsonl_paths: Iterable[Path], job_label: str) -> Iterator[SampleRow]:
    for p in jsonl_paths:
        m = SAMPLES_RE.match(p.name)
        if not m:
            continue
        task_full = m.group("task")
        base_task, variant = _base_task_and_variant(task_full)
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc = _doc(rec)
                yield SampleRow(
                    job=job_label,
                    task=task_full,
                    base_task=base_task,
                    variant=variant,
                    doc_id=int(rec.get("doc_id", -1)),
                    target=str(rec.get("target", "")),
                    response=_resp(rec),
                    correct=_correct(rec),
                    n_vertices=int(doc.get("n_vertices", 0)),
                    n_edges=int(doc.get("n_edges", 0)),
                    label_style=str(doc.get("label_style", "")),
                    edge_style=str(doc.get("edge_style", "")),
                    node_color=str(doc.get("node_color", "")),
                    include_adjacency_matrix=bool(doc.get("include_adjacency_matrix", False)),
                    constraint=str(doc.get("constraint", "")),
                    constraint_value=int(doc.get("constraint_value", -1)),
                )


def find_sample_jsonls(root: Path) -> list[Path]:
    """All ``*_samples_<task>.jsonl`` under ``root`` (recursive).

    Excludes per-chunk outputs under ``chunks/`` — those are inputs to the
    merge step, not authoritative results. The merge writes the combined
    file to ``<job>/<model_dir>/`` which is what callers want here.
    """
    return sorted(
        p for p in root.rglob("*_samples_*.jsonl")
        if SAMPLES_RE.match(p.name) and "chunks" not in p.parts
    )
