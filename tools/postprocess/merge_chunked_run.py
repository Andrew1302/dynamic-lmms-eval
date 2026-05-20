"""Concatenate per-chunk lmms-eval sample logs into one timestamped log dir.

When ``run_eval.sh`` runs in chunked mode it produces

    logs/<job>/chunks/chunk_0000/<model_dir>/<ts>_samples_<task>.jsonl
    logs/<job>/chunks/chunk_0001/<model_dir>/<ts>_samples_<task>.jsonl
    ...

Each chunk has its own timestamp, which breaks downstream postprocess
scripts (``compare_direct_disguise.py``, ``batch_report.py``) — they
require a single timestamp that covers every referenced task.

This script reads each chunk's ``*_samples_<task>.jsonl`` in TOC order
and writes one merged file per task with a single fresh timestamp:

    logs/<job>/<model_dir>/<merge_ts>_samples_<task>.jsonl

The per-chunk files are left in place for forensics; ``04_fetch.sh``
brings them down via ``RESULT_PATHS=("logs/${JOB_NAME}")`` so both views
end up under ``remote_results/<job>/``.

Invocation (from the runner):

    python tools/postprocess/merge_chunked_run.py \\
        --job-dir ./logs/<job> \\
        --toc <DATASET_DIR>/chunks/chunks.toc.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_SAMPLES_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_samples_(?P<task>.+)\.jsonl$")
_MERGE_META = ".merge_meta.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", required=True, type=Path,
                   help="Root for this job's lmms-eval output (logs/<job>).")
    p.add_argument("--toc", required=True, type=Path,
                   help="Path to chunks.toc.json from the prepare step.")
    return p.parse_args()


def _load_toc(toc_path: Path) -> list[str]:
    with toc_path.open("r", encoding="utf-8") as fh:
        toc = json.load(fh)
    return [c["name"] for c in toc["chunks"]]


def _discover_chunk_samples(
    chunks_root: Path, chunk_names: list[str]
) -> dict[tuple[Path, str], list[Path]]:
    """Return ``{(rel_parent, task): [chunk_path_in_order]}``.

    ``rel_parent`` is the directory of the jsonl relative to its chunk_dir
    (e.g. ``Path("dynamic_graph_benchmark/Qwen__Qwen3-VL-4B-Instruct")``).
    Merged files are written into ``job_dir / rel_parent / ...`` so the
    output mirrors lmms-eval's native layout — that is what
    ``compare_direct_disguise.py`` and ``batch_report.py`` find via
    ``rglob``.

    Aborts with a SystemExit if a chunk is missing its log, has multiple
    timestamped logs for the same task (would indicate a failed-mid-chunk
    cleanup), or different chunks emit jsonls under different relative
    layouts (which would mean different models — not supported).
    """
    by_key: dict[tuple[Path, str], dict[str, Path]] = defaultdict(dict)
    rel_parents_per_chunk: dict[str, set[Path]] = {}
    for chunk in chunk_names:
        chunk_dir = chunks_root / chunk
        if not chunk_dir.is_dir():
            sys.exit(f"[merge_chunked_run] missing chunk output dir: {chunk_dir}")

        # lmms-eval writes <output_path>/<experiment>/<model_dir>/<ts>_samples_<task>.jsonl
        # — possibly with additional intermediate dirs. We don't care about
        # the exact depth; we just preserve the relative layout in the merge.
        jsonls = sorted(chunk_dir.rglob("*_samples_*.jsonl"))
        if not jsonls:
            sys.exit(f"[merge_chunked_run] no _samples_*.jsonl files under {chunk_dir}")

        per_key: dict[tuple[Path, str], list[Path]] = defaultdict(list)
        rel_parents: set[Path] = set()
        for jp in jsonls:
            m = _SAMPLES_RE.match(jp.name)
            if not m:
                continue
            rel_parent = jp.parent.relative_to(chunk_dir)
            rel_parents.add(rel_parent)
            per_key[(rel_parent, m.group("task"))].append(jp)
        rel_parents_per_chunk[chunk] = rel_parents

        for key, paths in per_key.items():
            if len(paths) != 1:
                sys.exit(
                    f"[merge_chunked_run] {chunk_dir / key[0]} has {len(paths)} jsonls "
                    f"for task {key[1]!r}; expected exactly 1. Files: {[p.name for p in paths]}"
                )
            by_key[key][chunk] = paths[0]

    # All chunks must share the same relative layout (same model).
    layouts = {frozenset(p) for p in rel_parents_per_chunk.values()}
    if len(layouts) != 1:
        sys.exit(
            "[merge_chunked_run] chunks emit jsonls under inconsistent layouts: "
            f"{ {c: sorted(map(str, p)) for c, p in rel_parents_per_chunk.items()} }"
        )

    # Every (rel_parent, task) must appear in every chunk.
    expected_chunks = set(chunk_names)
    out: dict[tuple[Path, str], list[Path]] = {}
    for key, chunk_map in by_key.items():
        missing = expected_chunks - chunk_map.keys()
        if missing:
            sys.exit(
                f"[merge_chunked_run] task {key[1]!r} (under {key[0]}) missing from "
                f"chunks: {sorted(missing)}"
            )
        out[key] = [chunk_map[c] for c in chunk_names]
    return out


def _concat_jsonls(sources: list[Path], dest: Path) -> int:
    """Stream-copy lines from each source into dest with globally-unique doc_ids.

    lmms-eval numbers ``doc_id`` from 0 within each chunk's invocation, so
    naive concatenation produces N chunks × ~K rows of colliding ids. Anything
    downstream that keys by ``doc_id`` (e.g., ``compare_direct_disguise.py``'s
    direct/disguise pair-matching dict) then collapses to ~K unique entries.

    Rewrite each row's ``doc_id`` to a running offset: chunk i's first row gets
    (sum of all earlier chunks' row counts), then increments by 1 per row. Pair
    matching across direct/disguise jsonls stays consistent because both files
    iterate chunks in the same TOC order with the same per-chunk row counts.
    Returns total rows written.
    """
    rows = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as out_fh:
        for src in sources:
            with src.open("r", encoding="utf-8") as in_fh:
                for line in in_fh:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Preserve malformed lines rather than silently dropping
                        # them — downstream will fail loudly on the bad jsonl.
                        out_fh.write(line if line.endswith("\n") else line + "\n")
                        rows += 1
                        continue
                    rec["doc_id"] = rows
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows += 1
    return rows


def main() -> None:
    args = _parse_args()
    job_dir: Path = args.job_dir.resolve()
    chunks_root = job_dir / "chunks"
    if not chunks_root.is_dir():
        sys.exit(f"[merge_chunked_run] no chunks dir under {job_dir}")

    chunk_names = _load_toc(args.toc)
    grouped = _discover_chunk_samples(chunks_root, chunk_names)

    merge_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: list[dict] = []
    for (rel_parent, task), sources in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        dest = job_dir / rel_parent / f"{merge_ts}_samples_{task}.jsonl"
        n = _concat_jsonls(sources, dest)
        print(f"[merge_chunked_run] {task}: {n} rows -> {dest.relative_to(job_dir)}")
        summary.append({
            "rel_parent": str(rel_parent),
            "task": task,
            "merged_path": str(dest.relative_to(job_dir)),
            "n_rows": n,
            "sources": [str(p.relative_to(job_dir)) for p in sources],
        })

    (job_dir / _MERGE_META).write_text(
        json.dumps({
            "merge_ts": merge_ts,
            "chunks": chunk_names,
            "outputs": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[merge_chunked_run] wrote {_MERGE_META} ({len(summary)} task files at ts {merge_ts})")


if __name__ == "__main__":
    main()
