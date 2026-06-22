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


def _acc(rec: dict) -> dict:
    """The accuracy dict carries task/variant/score plus (for recent runs) the
    constraint axis info that lmms-eval strips from rec[\"doc\"]."""
    acc = rec.get("accuracy")
    return acc if isinstance(acc, dict) else {}


def _build_doc_lookup(
    dataset_root: Path, task_full: str
) -> dict[int, dict]:
    """Reconstruct a global ``doc_id -> {n_vertices, n_edges, constraint_value, ...}``
    map for ``task_full`` by walking the cached chunked HF dataset.

    The merge step (``merge_chunked_run.py``) globally renumbers ``doc_id`` to
    a running offset across chunks, processed in chunk-index order, after
    filtering each chunk to the task's (task, variant) slice. We replay that
    exact ordering here so older jsonls — which were written before
    ``process_results`` surfaced these fields — still get an axis label.
    """
    base, variant = _base_task_and_variant(task_full)
    # Find the chunk directories. Layout produced by the prepare pipeline:
    #   dataset_<job>/dataset_<job>/chunks/chunk_NNNN/
    # but the wrapping nesting depends on the fetch path. Search a couple of
    # levels deep to stay robust.
    # An HF DatasetDict chunk is identifiable by a sibling ``dataset_dict.json``
    # inside the chunk_NNNN directory. Filter to those so we don't grab
    # ``.run/chunks/`` (per-chunk status sentinels) or ``logs/.../chunks/``
    # (per-chunk lmms-eval jsonls).
    chunk_dirs: list[Path] = []
    for candidate in sorted(dataset_root.rglob("chunks")):
        if not candidate.is_dir():
            continue
        kids = sorted(
            c for c in candidate.iterdir()
            if c.is_dir()
            and c.name.startswith("chunk_")
            and (c / "dataset_dict.json").exists()
        )
        if kids:
            chunk_dirs = kids
            break
    if not chunk_dirs:
        return {}

    try:
        from datasets import load_from_disk  # type: ignore[import-not-found]
    except Exception:
        return {}

    # Columns we actually consume. Skipping image-typed columns avoids the
    # ``Pillow``-required decode path when iterating rows.
    wanted_cols = (
        "task", "variant", "n_vertices", "n_edges", "constraint",
        "constraint_value", "label_style", "edge_style", "node_color",
        "include_adjacency_matrix",
    )

    out: dict[int, dict] = {}
    global_idx = 0
    for ch in chunk_dirs:
        try:
            ds = load_from_disk(str(ch))
        except Exception:
            continue
        split = ds["test"] if hasattr(ds, "keys") and "test" in ds else ds
        keep = [c for c in wanted_cols if c in split.column_names]
        split = split.select_columns(keep)
        for row in split:
            if row.get("task") != base or row.get("variant") != variant:
                continue
            out[global_idx] = {
                "n_vertices": int(row.get("n_vertices", 0) or 0),
                "n_edges": int(row.get("n_edges", 0) or 0),
                "label_style": str(row.get("label_style", "") or ""),
                "edge_style": str(row.get("edge_style", "") or ""),
                "node_color": str(row.get("node_color", "") or ""),
                "include_adjacency_matrix": bool(row.get("include_adjacency_matrix", False)),
                "constraint": str(row.get("constraint", "") or ""),
                "constraint_value": int(row.get("constraint_value", -1) or -1),
            }
            global_idx += 1
    return out


def iter_rows(
    jsonl_paths: Iterable[Path],
    job_label: str,
    dataset_root: Path | None = None,
) -> Iterator[SampleRow]:
    """Stream ``SampleRow``s from ``jsonl_paths``.

    If ``dataset_root`` is given, missing axis info (n_edges, constraint_value,
    etc.) is back-filled lazily per-task from the cached HF dataset there.
    """
    # Cache the doc lookup per task_full so we only walk the dataset once.
    lookups: dict[str, dict[int, dict]] = {}
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
                acc = _acc(rec)
                doc_id = int(rec.get("doc_id", -1))

                def _pick(key: str, default, cast):
                    if key in acc and acc[key] not in (None, ""):
                        return cast(acc[key])
                    if key in doc and doc[key] not in (None, ""):
                        return cast(doc[key])
                    return default

                n_vertices = _pick("n_vertices", 0, int)
                n_edges = _pick("n_edges", 0, int)
                constraint = _pick("constraint", "", str)
                constraint_value = _pick("constraint_value", -1, int)

                # Fall back to the cached HF dataset for legacy jsonls that
                # don't carry the axis info in `accuracy`.
                if constraint_value == -1 and dataset_root is not None:
                    if task_full not in lookups:
                        lookups[task_full] = _build_doc_lookup(dataset_root, task_full)
                    info = lookups[task_full].get(doc_id)
                    if info is not None:
                        n_vertices = info["n_vertices"] or n_vertices
                        n_edges = info["n_edges"] or n_edges
                        constraint = info["constraint"] or constraint
                        constraint_value = info["constraint_value"]

                yield SampleRow(
                    job=job_label,
                    task=task_full,
                    base_task=base_task,
                    variant=variant,
                    doc_id=doc_id,
                    target=str(rec.get("target", "")),
                    response=_resp(rec),
                    correct=_correct(rec),
                    n_vertices=n_vertices,
                    n_edges=n_edges,
                    label_style=str(doc.get("label_style", "")),
                    edge_style=str(doc.get("edge_style", "")),
                    node_color=str(doc.get("node_color", "")),
                    include_adjacency_matrix=bool(doc.get("include_adjacency_matrix", False)),
                    constraint=constraint,
                    constraint_value=constraint_value,
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
