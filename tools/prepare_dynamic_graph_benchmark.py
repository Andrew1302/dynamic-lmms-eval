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
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import zlib
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

    # Parallelism. 1 keeps the serial codepath for bit-for-bit comparison /
    # debugging. Default = all cores. Deliberately excluded from _fingerprint
    # so changing worker count does not invalidate the on-disk dataset.
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1,
                        help="Worker processes for rendering. 1 = in-process serial.")

    # Chunking. When --chunk-size > 0 the script also writes a sharded copy of
    # the dataset under <chunks-dir>/chunk_NNNN, plus a TOC, so a remote runner
    # can iterate chunks and resume from a failed one. The canonical
    # <output-dir> dataset is still written unchanged.
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="If > 0, also emit per-chunk HF datasets of this many rows. "
                             "Chunk boundaries are pair-aligned (even row index) so "
                             "direct/disguise pairs are never split.")
    parser.add_argument("--chunks-dir", type=str, default=None,
                        help="Where to write chunk_NNNN subdirs and chunks.toc.json "
                             "(default: <output-dir>/chunks).")
    parser.add_argument("--reset-status-dir", type=str, default=None,
                        help="If set, AND the script regenerates (fingerprint changed or "
                             "first run with chunks), wipe *.status files in this directory. "
                             "Used by run_eval.sh to clear stale chunk completion markers "
                             "when the dataset is rebuilt.")

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


# Per-process handle to the imported `src.benchmark` module. Populated by
# _worker_init under spawn, and lazily populated in-process for the serial path.
_WORKER_BENCHMARK = None


def _worker_init():
    """Spawn-safe init: re-add dynamic-dataset to sys.path and import src.benchmark.

    Importing the rendering submodules sets matplotlib.use('Agg') at module load,
    so the Agg backend is live before the worker handles its first payload.
    """
    global _WORKER_BENCHMARK
    if str(_DYNAMIC_DATASET_ROOT) not in sys.path:
        sys.path.insert(0, str(_DYNAMIC_DATASET_ROOT))
    _WORKER_BENCHMARK = importlib.import_module("src.benchmark")


def _ensure_benchmark_loaded():
    global _WORKER_BENCHMARK
    if _WORKER_BENCHMARK is None:
        _WORKER_BENCHMARK = benchmark
    return _WORKER_BENCHMARK


def _encode_png(img):
    """RGB-convert and PNG-encode a PIL Image to bytes.

    Done inside the worker so (a) the main process never receives the ~MB-scale
    decoded pixel buffer over IPC (the shortest_path map is 2550x2550), and
    (b) PNG encoding runs in parallel instead of serially during the HF
    writer's flush.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _encode_sample_images(sample):
    sample["direct_image"] = _encode_png(sample["direct_image"])
    sample["disguise_image"] = _encode_png(sample["disguise_image"])
    return sample


def _render_standard_one(payload):
    """Render one (task_name, i) standard-mode sample. Picklable, side-effect free."""
    task_name, i, seed, difficulty, cfg, include_adj = payload
    mod = _ensure_benchmark_loaded()
    task = mod.get_all_tasks()[task_name]()
    sample = task.generate(
        seed=seed, difficulty=difficulty, config=cfg,
        include_adjacency_matrix=include_adj,
    )
    return task_name, i, seed, _encode_sample_images(sample)


def _render_sweep_one(payload):
    """Render one sweep-mode sample. Handles both 'nodes' and 'edges' constraints."""
    (task_name, ordering_idx, base_seed, constraint, value,
     cfg, include_adj, tol, max_attempts) = payload
    mod = _ensure_benchmark_loaded()
    task = mod.get_all_tasks()[task_name]()
    if constraint == "nodes":
        sample = task.generate(
            seed=base_seed, difficulty="medium", config=cfg,
            include_adjacency_matrix=include_adj, node_count=value,
        )
        used_seed = base_seed
    else:
        sample, used_seed = _sample_for_edge_target(
            task=task, target_edges=value, cfg=cfg, include_adj=include_adj,
            base_seed=base_seed, tol=tol, max_attempts=max_attempts,
        )
    return task_name, ordering_idx, used_seed, value, _encode_sample_images(sample)


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


_META_FILENAME = "prepare_meta.json"
_TOC_FILENAME = "chunks.toc.json"


def _fingerprint(args: argparse.Namespace) -> dict:
    """Stable, hashable view of the args that determine the dataset content.

    Two prepare runs with identical fingerprints produce byte-identical rows.
    Used to skip regeneration when the on-disk dataset already matches.
    """
    # difficulty_override is a list of (task, level) tuples; normalise to
    # lists-of-lists so the JSON roundtrip on prepare_meta.json compares equal.
    overrides = sorted([list(o) for o in args.difficulty_override])
    return {
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "tasks": sorted(args.tasks) if args.tasks else None,
        "difficulty": args.difficulty,
        "difficulty_override": overrides,
        "label_style": args.label_style,
        "node_color": args.node_color,
        "edge_style": args.edge_style,
        "include_adjacency_matrix": bool(args.include_adjacency_matrix),
        "constraint": args.constraint,
        "constraint_values": args.constraint_values,
        "samples_per_value": args.samples_per_value,
        "edge_tolerance": args.edge_tolerance,
        "edge_max_attempts": args.edge_max_attempts,
        "chunk_size": int(args.chunk_size),
        # Bumped when the on-disk chunk layout changes (e.g. row ordering)
        # so previously-built chunks/ trees are detected as stale and
        # regenerated. v2: chunks interleave pairs across tasks so every
        # chunk spans all selected tasks (v1 grouped contiguous rows per
        # task, which made lmms-eval's per-task filter return 0 rows on
        # single-task chunks).
        "chunk_layout_version": 2 if int(args.chunk_size) > 0 else 0,
    }


def _read_meta(output_dir: Path) -> dict | None:
    meta_path = output_dir / _META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _clear_status_dir(status_dir: Path) -> None:
    if not status_dir.is_dir():
        return
    removed = 0
    for p in status_dir.glob("*.status"):
        p.unlink()
        removed += 1
    if removed:
        print(f"[prepare_dynamic_graph_benchmark] Cleared {removed} stale *.status files in {status_dir}")


def _resolve_chunks_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    return Path(args.chunks_dir) if args.chunks_dir else output_dir / "chunks"


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir.parent / f"{output_dir.name}_images"
    chunks_dir = _resolve_chunks_dir(args, output_dir)

    fp_new = _fingerprint(args)
    fp_old = _read_meta(output_dir)

    # Short-circuit: identical fingerprint AND every required artifact is on
    # disk → reuse. This is what makes resume cheap on the VM.
    if fp_old == fp_new and output_dir.is_dir():
        chunks_ok = (args.chunk_size <= 0) or (chunks_dir / _TOC_FILENAME).is_file()
        if chunks_ok:
            print(f"[prepare_dynamic_graph_benchmark] Dataset at {output_dir} already matches "
                  f"requested fingerprint — skipping regeneration.")
            return

    # About to regenerate — wipe stale chunk status files first so the runner
    # doesn't think a previously-failed chunk index is still done. Only acts
    # when the caller asked us to manage that dir.
    if args.reset_status_dir:
        _clear_status_dir(Path(args.reset_status_dir))

    cfg = _build_config(args)

    all_tasks = benchmark.get_all_tasks()
    task_names = args.tasks if args.tasks else sorted(all_tasks.keys())
    unknown = set(task_names) - set(all_tasks.keys())
    if unknown:
        sys.exit(f"[prepare_dynamic_graph_benchmark] Unknown tasks: {unknown}. Available: {sorted(all_tasks.keys())}")

    # Wrap task_names so HF's Dataset.from_generator does not interpret a list
    # kwarg as a sharding hint and call our generator once per task (which would
    # also spin the worker Pool up N times).
    task_names_tuple = tuple(task_names)
    if args.constraint is None:
        gen_fn = _generate_standard
        gen_kwargs = {"args": args, "cfg": cfg, "task_names": task_names_tuple}
        mode_desc = f"standard mode: {args.num_samples} generations per task"
    else:
        gen_fn = _generate_sweep
        gen_kwargs = {"args": args, "cfg": cfg, "task_names": task_names_tuple}
        mode_desc = (
            f"sweep mode: constraint={args.constraint} "
            f"values={_parse_constraint_values(args.constraint_values)} "
            f"samples_per_value={args.samples_per_value or (250 if args.constraint == 'nodes' else 100)}"
        )

    print(f"[prepare_dynamic_graph_benchmark] {mode_desc}; building HuggingFace dataset (streaming) ...")

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

    # Stream rendered rows straight into arrow shards on disk so peak RAM is
    # bounded to ~writer_batch_size rows instead of holding every PIL image
    # until the final save. cache_dir is wiped before and after to avoid
    # accumulating stale arrow files between runs.
    cache_dir = output_dir.parent / f".{output_dir.name}_hfcache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    ds = datasets.Dataset.from_generator(
        gen_fn,
        gen_kwargs=gen_kwargs,
        features=features,
        cache_dir=str(cache_dir),
        writer_batch_size=200,
    )
    dataset_dict = DatasetDict({"test": ds})

    # Wipe stale shards from previous runs — save_to_disk overwrites the
    # primary arrow files but can leave orphaned shards behind when the new
    # row count maps to fewer shards, which silently truncates loaded data.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"[prepare_dynamic_graph_benchmark] Saved to {output_dir.resolve()}")
    print(f"[prepare_dynamic_graph_benchmark] Split 'test' has {len(ds)} samples.")

    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    n_imgs = 0
    for row in ds:  # HF lazy-decodes one row at a time; no in-memory list
        row["image"].save(str(images_dir / f"{row['id']}.png"))
        n_imgs += 1
    print(f"[prepare_dynamic_graph_benchmark] Exported {n_imgs} images to {images_dir.resolve()}")

    if args.chunk_size > 0:
        _write_chunks(ds, args.chunk_size, chunks_dir)

    # Drop the generator cache now that save_to_disk has the canonical copy.
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Meta written last so an interrupted prepare doesn't falsely advertise
    # "fingerprint matches" on next invocation.
    (output_dir / _META_FILENAME).write_text(
        json.dumps(fp_new, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _interleave_pair_indices_by_task(ds) -> list[int]:
    """Round-robin pair indices so each chunk slice spans every task.

    ``_row_pair`` writes each direct/disguise pair as two consecutive rows,
    and ``_generate_*`` emits all of one task's rows before moving to the
    next. Without reordering, contiguous chunks would each contain a single
    task — lmms-eval's per-task filter then matches 0 rows on 2/3 chunks
    and ``task_docs[0]`` raises IndexError. Returns a flat list of pair
    indices (one per pair) in round-robin order.
    """
    tasks = ds["task"]  # one entry per row; pair p occupies rows 2p, 2p+1
    n_pairs = len(tasks) // 2
    pairs_per_task: dict[str, list[int]] = {}
    for p in range(n_pairs):
        pairs_per_task.setdefault(tasks[2 * p], []).append(p)
    # Stable round-robin over insertion order — matches the task order the
    # generator used, so a single-task selection still produces a contiguous
    # layout (no spurious reordering when there is nothing to interleave).
    buckets = list(pairs_per_task.values())
    max_len = max((len(b) for b in buckets), default=0)
    out: list[int] = []
    for i in range(max_len):
        for b in buckets:
            if i < len(b):
                out.append(b[i])
    return out


def _write_chunks(ds, chunk_size: int, chunks_dir: Path) -> None:
    """Slice the on-disk dataset into pair-aligned chunks and save each as a DatasetDict.

    ``_row_pair`` emits each (direct, disguise) sample as two consecutive
    entries, so we pad every chunk boundary to an even index — a chunk is
    always a whole number of pairs. Slicing via ``ds.select`` is lazy, so we
    never hold all rows in memory simultaneously.

    Pairs are first reordered round-robin across tasks (see
    ``_interleave_pair_indices_by_task``) so each chunk contains a slice of
    every selected task. Required for lmms-eval's per-task filtering to
    find at least one row per task in every chunk.
    """
    from datasets import DatasetDict

    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True)

    if chunk_size % 2 != 0:
        # Round up to the next even number so direct/disguise pairs stay
        # together. We document this in the help; do not silently differ.
        chunk_size = chunk_size + 1
        print(f"[prepare_dynamic_graph_benchmark] chunk_size was odd; bumped to {chunk_size} "
              f"so direct/disguise pairs stay together.")

    total = len(ds)
    # Interleave pairs across tasks and expand back to row indices.
    pair_order = _interleave_pair_indices_by_task(ds)
    row_order = [r for p in pair_order for r in (2 * p, 2 * p + 1)]
    if len(row_order) != total:
        # Trailing unpaired row (only possible if total is odd) — drop it
        # rather than producing a chunk with a half-pair.
        print(f"[prepare_dynamic_graph_benchmark] WARNING: dropping {total - len(row_order)} "
              f"trailing unpaired row(s) from chunking")

    toc_entries: list[dict] = []
    chunk_idx = 0
    for start in range(0, len(row_order), chunk_size):
        end = min(start + chunk_size, len(row_order))
        if (end - start) % 2 != 0:
            # Defensive: can't happen with even chunk_size and pair-aligned
            # row_order, but guard against truncating a pair.
            end -= 1
        if end <= start:
            continue
        name = f"chunk_{chunk_idx:04d}"
        sub_ds = ds.select(row_order[start:end])
        DatasetDict({"test": sub_ds}).save_to_disk(str(chunks_dir / name))
        toc_entries.append({
            "idx": chunk_idx,
            "name": name,
            "start": start,
            "end": end,
            "n_rows": end - start,
        })
        chunk_idx += 1

    toc = {
        "chunk_size": chunk_size,
        "n_chunks": len(toc_entries),
        "total_rows": total,
        "chunks": toc_entries,
    }
    (chunks_dir / _TOC_FILENAME).write_text(
        json.dumps(toc, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[prepare_dynamic_graph_benchmark] Wrote {len(toc_entries)} chunks of up to "
          f"{chunk_size} rows each to {chunks_dir.resolve()}")


def _resolve_difficulty_for(args, task_names) -> dict[str, str]:
    difficulty_for: dict[str, str] = {t: args.difficulty for t in task_names}
    for task, level in args.difficulty_override:
        if task not in difficulty_for:
            sys.exit(
                f"[prepare_dynamic_graph_benchmark] --difficulty-override references "
                f"unknown/unselected task {task!r}. Selected tasks: {task_names}"
            )
        difficulty_for[task] = level
    return difficulty_for


def _dispatch(work_fn, payloads, num_workers):
    """Yield work_fn(p) for each p in payloads, in input order.

    num_workers <= 1 runs in-process (no Pool, easier debugging). Otherwise a
    'spawn' Pool drives work_fn — spawn so behaviour is identical on Linux and
    Windows and we don't inherit fork-time matplotlib state.
    """
    if num_workers <= 1:
        for p in payloads:
            yield work_fn(p)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_worker_init) as pool:
        # imap (ordered) so on-disk row order is reproducible; the pool buffers
        # ~chunksize*num_workers results ahead of us, which is bounded.
        for result in pool.imap(work_fn, payloads, chunksize=4):
            yield result


def _generate_standard(args, cfg, task_names):
    """Generator: yield direct/disguise row pairs for the standard mode."""
    difficulty_for = _resolve_difficulty_for(args, task_names)

    print(f"[prepare_dynamic_graph_benchmark] Generating {args.num_samples} generations per task "
          f"across {task_names} (difficulty={difficulty_for}, seed={args.seed}, "
          f"label_style={args.label_style}, edge_style={args.edge_style}, "
          f"adj_matrix={args.include_adjacency_matrix}, num_workers={args.num_workers})")

    payloads = []
    for task_name in task_names:
        difficulty = difficulty_for[task_name]
        # zlib.crc32 instead of Python's hash(): hash() is randomized per
        # process, which made seeds — and therefore generated graphs — differ
        # across invocations of this script even with --seed fixed.
        task_salt = zlib.crc32(task_name.encode("utf-8")) % 1000
        for i in range(args.num_samples):
            seed = args.seed + i * 1000 + task_salt
            payloads.append((task_name, i, seed, difficulty, cfg,
                             args.include_adjacency_matrix))

    total = len(payloads)
    done = 0
    for task_name, i, seed, sample in _dispatch(_render_standard_one, payloads, args.num_workers):
        for row in _row_pair(
            task_name=task_name, variant_seed=i, sample=sample, args=args,
            difficulty=difficulty_for[task_name], seed=seed,
            constraint=None, constraint_value=-1,
        ):
            yield row
        done += 1
        if done % 100 == 0 or done == total:
            print(f"[prepare_dynamic_graph_benchmark] rendered {done}/{total} samples")


def _generate_sweep(args, cfg, task_names):
    """Generator: yield direct/disguise row pairs for sweep mode."""
    values = _parse_constraint_values(args.constraint_values)
    spv = args.samples_per_value or (250 if args.constraint == "nodes" else 100)
    print(f"[prepare_dynamic_graph_benchmark] Sweep: constraint={args.constraint} "
          f"values={values} spv={spv} tasks={task_names} "
          f"label_style={args.label_style} adj_matrix={args.include_adjacency_matrix} "
          f"num_workers={args.num_workers}")

    # Build payloads with the exact same counter sequence as the original serial
    # loop so seeds and variant_seed (== counter after increment) match bit-for-bit.
    payloads = []
    counter = 0
    for task_name in task_names:
        for value in values:
            for _ in range(spv):
                base_seed = args.seed + counter * 7919
                counter += 1
                payloads.append((task_name, counter, base_seed, args.constraint, value,
                                 cfg, args.include_adjacency_matrix,
                                 args.edge_tolerance, args.edge_max_attempts))

    per_bucket_remaining: dict[tuple[str, int], int] = {}
    for p in payloads:
        per_bucket_remaining[(p[0], p[4])] = per_bucket_remaining.get((p[0], p[4]), 0) + 1

    total = len(payloads)
    done = 0
    for task_name, ordering_idx, used_seed, value, sample in _dispatch(
        _render_sweep_one, payloads, args.num_workers
    ):
        for row in _row_pair(
            task_name=task_name, variant_seed=ordering_idx, sample=sample, args=args,
            difficulty="medium", seed=used_seed,
            constraint=args.constraint, constraint_value=value,
        ):
            yield row
        done += 1
        per_bucket_remaining[(task_name, value)] -= 1
        if per_bucket_remaining[(task_name, value)] == 0:
            print(f"  [{task_name}] value={value}: {spv} samples done")
        if done % 100 == 0 or done == total:
            print(f"[prepare_dynamic_graph_benchmark] rendered {done}/{total} samples")


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
    # sample["*_image"] is PNG-encoded bytes produced by _encode_sample_images.
    # HF Image feature accepts the {"bytes", "path"} dict form and stores it
    # verbatim — no re-encoding in the writer.
    return [
        {**common,
         "id": f"{task_name}_direct_{variant_seed:06d}",
         "variant": "direct",
         "prompt": sample["direct_prompt"],
         "image": {"bytes": sample["direct_image"], "path": None}},
        {**common,
         "id": f"{task_name}_disguise_{variant_seed:06d}",
         "variant": "disguise",
         "prompt": sample["disguise_prompt"],
         "image": {"bytes": sample["disguise_image"], "path": None}},
    ]


if __name__ == "__main__":
    main()
