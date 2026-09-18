#!/usr/bin/env python
"""Responses-API runner for the dynamic graph benchmark — captures the model's
*thinking* (reasoning summary) alongside the answer.

Why this exists: gpt-5.6-terra (Azure, aboda-openai v1 endpoint) is a reasoning
model whose thinking canNOT be turned off, and Chat Completions exposes only a
``reasoning_tokens`` COUNT — no text. The Responses API DOES return a reasoning
summary. So for any analysis of *what the model is thinking* (especially on
misses), we must drive the benchmark through /responses with
``reasoning.summary=detailed``. This runner does that while reusing the exact
canonical prompt (doc['prompt']), image (doc['image']) and scorer
(utils.dynamic_graph_benchmark_process_results) — so scores match the lmms-eval
pipeline, only the transport and the captured thinking differ.

Resumable by construction: prepare is prefix-stable — sample i is a pure function
of (seed, task, i) — so --start-index slices a larger run. Run i in [0,5) now,
i in [5,100) next: byte-identical graphs, zero overlap. Always keep --seed fixed
(default 42) and advance --start-index by the number of samples already run.

Per sample we store: global index, task, variant, difficulty, gold answer, model
prediction, canonical score, reasoning_summary (thinking), input/output/reasoning
token counts, and the encrypted reasoning blob (for later rehydration).

Creds: AZURE_OPENAI_API_KEY + AZURE_OPENAI_API_BASE from .env (auto-loaded).

Run from repo root with .venv-api's python:
    .venv-api\\Scripts\\python examples/models/dynamic_graph_benchmark/run_local_responses.py \\
        --num-samples 5 --start-index 0 --difficulties easy medium hard
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DATASET_DIR = REPO_ROOT / "dynamic_graph_benchmark_data"
load_dotenv(REPO_ROOT / ".env")

# Make the canonical task utils importable for the shared scorer + prompt.
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-version", default="gpt-5.6-terra", help="Azure deployment / model id (default: gpt-5.6-terra)")
    p.add_argument("--tasks", nargs="+", default=["coloring", "directed_connectivity", "shortest_path"])
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"], choices=["easy", "medium", "hard"])
    p.add_argument("--num-samples", type=int, default=5, help="samples (pairs) per task per difficulty")
    p.add_argument("--start-index", type=int, default=0, help="global sample index to start at (resume: advance by samples already run)")
    p.add_argument("--seed", type=int, default=42, help="dataset seed — KEEP FIXED across resumes")
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max-output-tokens", type=int, default=8000, help="hard cap on output (reasoning+answer) per call")
    p.add_argument("--num-concurrent", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=0, help="split the slice into sub-chunks of this many samples, preparing+inferring each (kill-safe resume). 0 = one chunk.")
    p.add_argument("--num-workers", type=int, default=0, help="prepare render workers (0 = prepare default = cpu_count)")
    p.add_argument("--label-style", default="numeric", choices=["numeric", "letters", "none"])
    p.add_argument("--node-color", default="#AED6F1")
    p.add_argument("--edge-style", default="straight", choices=["straight", "curved"])
    p.add_argument("--out-dir", default=None, help="output root (default ./logs_responses/<model>_s<seed>)")
    p.add_argument("--api-version", default="preview")
    return p.parse_args()


def build_prepare_cmd(args, diff, cstart, cnum):
    cmd = [
        sys.executable, "tools/prepare_dynamic_graph_benchmark.py",
        "--seed", str(args.seed),
        "--tasks", *args.tasks,
        "--output-dir", str(CANONICAL_DATASET_DIR),
        "--num-samples", str(cnum),
        "--start-index", str(cstart),
        "--difficulty", diff,
        "--label-style", args.label_style,
        "--node-color", args.node_color,
        "--edge-style", args.edge_style,
    ]
    if args.num_workers:
        cmd += ["--num-workers", str(args.num_workers)]
    return cmd


def encode_png(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_response(d: dict):
    """Extract (answer, reasoning_summary, encrypted, in_tok, out_tok, reason_tok)."""
    answer_parts, summary_parts, encrypted = [], [], None
    for o in d.get("output", []):
        t = o.get("type")
        if t == "message":
            for c in o.get("content", []):
                if c.get("text"):
                    answer_parts.append(c["text"])
        elif t == "reasoning":
            for s in o.get("summary", []):
                if isinstance(s, dict) and s.get("text"):
                    summary_parts.append(s["text"])
            encrypted = o.get("encrypted_content") or encrypted
    u = d.get("usage", {}) or {}
    reason_tok = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return (
        "".join(answer_parts).strip(),
        "\n\n".join(summary_parts).strip(),
        encrypted,
        u.get("input_tokens", 0) or 0,
        u.get("output_tokens", 0) or 0,
        reason_tok,
    )


def main() -> int:
    args = parse_args()
    # Windows consoles default to cp1252; model output / progress can carry
    # non-ASCII. Force UTF-8 so prints never crash the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    base = os.environ.get("AZURE_OPENAI_API_BASE")
    if not key or not base:
        print(f"[responses] FATAL: need AZURE_OPENAI_API_KEY + AZURE_OPENAI_API_BASE in {REPO_ROOT/'.env'}", file=sys.stderr)
        return 2
    url = base.rstrip("/") + f"/responses?api-version={args.api_version}"
    headers = {"api-key": key, "Content-Type": "application/json"}

    import subprocess
    import datasets
    from lmms_eval.tasks.dynamic_graph_benchmark.utils import dynamic_graph_benchmark_process_results as score_fn

    out_root = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "logs_responses" / f"{args.model_version}_s{args.seed}")
    out_root.mkdir(parents=True, exist_ok=True)

    # RESUME: skip any (difficulty, task, variant, global_index) already on disk
    # (this env kills long runs, so a slice is driven in bounded per-difficulty
    # invocations; re-running never re-calls a finished sample). Failed rows are
    # NOT marked done, so they get retried.
    done = set()
    for f in glob.glob(str(out_root / "*" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not str(r.get("prediction", "")).startswith("[REQUEST_FAILED"):
                done.add((r["difficulty"], r["task"], r["variant"], r["global_index"]))
    print(f"[responses] resume set: {len(done)} samples already on disk under {out_root}")

    def call_one(prompt, b64):
        payload = {
            "model": args.model_version,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
            ]}],
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
            "max_output_tokens": args.max_output_tokens,
        }
        for attempt in range(5):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=300)
                if r.status_code == 200:
                    return r.json(), None
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as exc:
                err = str(exc)[:200]
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
        return None, err

    chunk = args.chunk_size or args.num_samples
    grand = []  # (task, variant, diff, score, in, out, reason)
    for diff in args.difficulties:
        for cstart in range(args.start_index, args.start_index + args.num_samples, chunk):
            cnum = min(chunk, args.start_index + args.num_samples - cstart)
            # Skip a chunk whose whole slice is already on disk WITHOUT rendering
            # (render is the slow part). Makes kill-resume cheap. Failed rows are
            # not in `done`, so they re-run.
            expected = {(diff, t, v, i)
                        for t in args.tasks for v in ("direct", "disguise")
                        for i in range(cstart, cstart + cnum)}
            if expected <= done:
                print(f"[responses] {diff} i=[{cstart},{cstart+cnum}) complete — skip.", flush=True)
                continue
            print(f"\n[responses] === {diff} chunk i=[{cstart},{cstart+cnum}): prepare ===", flush=True)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            subprocess.run(build_prepare_cmd(args, diff, cstart, cnum), cwd=REPO_ROOT, check=True, env=env)
            ds = datasets.load_from_disk(str(CANONICAL_DATASET_DIR))
            ds = ds["test"] if isinstance(ds, dict) or hasattr(ds, "keys") else ds

            diff_dir = out_root / diff
            diff_dir.mkdir(parents=True, exist_ok=True)
            writers = {}

            def process(idx, _ds=ds, _diff=diff):
                doc = _ds[idx]
                gi = int(doc["id"].rsplit("_", 1)[1])
                resp, err = call_one(doc["prompt"], encode_png(doc["image"]))
                if resp is None:
                    answer, summary, enc, it, ot, rt = f"[REQUEST_FAILED] {err}", "", None, 0, 0, 0
                else:
                    answer, summary, enc, it, ot, rt = parse_response(resp)
                sc = score_fn(doc, [answer])["accuracy"]
                return {
                    "global_index": gi, "id": doc["id"], "task": doc["task"], "variant": doc["variant"],
                    "difficulty": _diff, "target": doc["answer"], "prediction": answer, "score": sc["score"],
                    "reasoning_summary": summary, "input_tokens": it, "output_tokens": ot,
                    "reasoning_tokens": rt, "n_vertices": doc["n_vertices"], "n_edges": doc["n_edges"],
                    "encrypted_reasoning": enc,
                }

            todo = []
            for i, _id in enumerate(ds["id"]):
                task, variant, gi = _id.rsplit("_", 2)
                if (diff, task, variant, int(gi)) not in done:
                    todo.append(i)
            print(f"[responses] {diff} i=[{cstart},{cstart+cnum}): {len(todo)} to run", flush=True)

            with ThreadPoolExecutor(max_workers=args.num_concurrent) as ex:
                futs = {ex.submit(process, i): i for i in todo}
                for k, fut in enumerate(as_completed(futs), 1):
                    row = fut.result()
                    key2 = (row["task"], row["variant"])
                    if key2 not in writers:
                        writers[key2] = open(diff_dir / f"{row['task']}_{row['variant']}.jsonl", "a", encoding="utf-8")
                    writers[key2].write(json.dumps(row, ensure_ascii=False) + "\n")
                    writers[key2].flush()  # persist immediately — kill-safe
                    grand.append((row["task"], row["variant"], diff, row["score"], row["input_tokens"], row["output_tokens"], row["reasoning_tokens"]))
                    if not str(row["prediction"]).startswith("[REQUEST_FAILED"):
                        done.add((row["difficulty"], row["task"], row["variant"], row["global_index"]))
                    print(f"  [{k}/{len(todo)}] {row['task']}/{row['variant']} i={row['global_index']} "
                          f"gold={row['target']} pred={str(row['prediction'])[:20]!r} score={row['score']} "
                          f"reason_tok={row['reasoning_tokens']}", flush=True)
            for w in writers.values():
                w.close()

    # manifest + summary
    manifest = {
        "model": args.model_version, "seed": args.seed, "start_index": args.start_index,
        "num_samples": args.num_samples, "tasks": args.tasks, "difficulties": args.difficulties,
        "reasoning_effort": args.reasoning_effort,
        "next_start_index": args.start_index + args.num_samples,
    }
    (out_root / f"manifest_s{args.seed}_start{args.start_index}_n{args.num_samples}.json").write_text(json.dumps(manifest, indent=2))

    print("\n[responses] ===== SCORE SUMMARY =====")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    tin = tout = 0
    for task, variant, diff, sc, it, ot, rt in grand:
        agg[(task, variant)][0] += sc; agg[(task, variant)][1] += 1
        tin += it; tout += ot
    for (task, variant), (s, n) in sorted(agg.items()):
        print(f"  {task:22} {variant:9} {s:.0f}/{n} = {s/n:.2f}")
    direct = [g for g in grand if g[1] == "direct"]; disg = [g for g in grand if g[1] == "disguise"]
    for name, grp in (("direct", direct), ("disguise", disg), ("ALL", grand)):
        if grp:
            print(f"  {name:9} {sum(g[3] for g in grp):.0f}/{len(grp)} = {sum(g[3] for g in grp)/len(grp):.2f}")
    print(f"\n[responses] tokens: input={tin:,} output={tout:,} (calls={len(grand)})")
    print(f"[responses] NEXT run: --start-index {args.start_index + args.num_samples} (same --seed {args.seed})")
    print(f"[responses] output -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
