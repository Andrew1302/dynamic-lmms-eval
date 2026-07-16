"""Explode the fp8fix pre-flight smoke into a browsable per-question tree.

Layout (mirrors _inspect/thinkadj_n100, with the ablation tier prepended):

    <out>/<adj|img>/<arm>/<task>/<difficulty>/<orig|disg>/q_<NNN>/
        prompt.txt               full model prompt
        answer.txt               ground-truth target
        image.png                the exact stored image the model was shown
        qwen_35_answer.txt       Qwen3.5-4B     output (raw + extracted + score)
        internvl_35_answer.txt   InternVL3.5-4B output

    adj = tier-1 / 08-style (adjacency list in prompt) ; img = tier-2 / 07-style
    (image only). Both models run vllm fp8 with the FP8_KEEP_BF16_PATTERNS
    vision fix; InternVL think additionally runs the new qwen3 reasoning-parser
    force-close (12288 budget + 1024 answer).

A root ``_index.tsv`` lists every question with each model's extracted answer +
score. ``_timing_notes.txt`` records per-job wall time / throughput (from each
job's run.log Throughput Summary) plus per-sample raw-output lengths, so the
think arms' cost is documented next to the data.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
from pathlib import Path

RESULTS = Path(r"C:/Users/Andrew/Msc/dynamic-lmms-eval/remote_results")
DEFAULT_OUT = RESULTS / "_inspect" / "fp8smoke"

MODELS = {  # cell token -> answer filename stem
    "qwen35_4b": "qwen_35_answer",
    "internvl35_4b": "internvl_35_answer",
}
MODEL_PRETTY = {"qwen35_4b": "Qwen3.5-4B", "internvl35_4b": "InternVL3.5-4B"}
TIERS = ["adj", "img"]
ARMS = ["think", "nothink"]
TASKS = ["directed_connectivity", "shortest_path", "coloring"]
DIFFS = ["easy", "medium", "hard"]
VARIANTS = {"direct": "orig", "disguise": "disg"}


def job_name(tier: str, arm: str, diff: str, model: str) -> str:
    return f"graph_bench_fp8smoke_{tier}_{arm}_{diff}_{model}"


def find_job_dir(job: str) -> Path | None:
    for cand in (RESULTS / "_scratch" / job, RESULTS / job, RESULTS / "_smoke" / job):
        if cand.is_dir():
            return cand
    return None


def latest_samples(job_dir: Path, task: str, variant: str) -> Path | None:
    pat = f"*_samples_dynamic_graph_benchmark_{task}_{variant}.jsonl"
    cands = [f for f in job_dir.rglob(pat) if "/chunks/" not in f.as_posix()]
    return max(cands, key=lambda p: p.name) if cands else None


def image_path(job_dir: Path, job: str, task: str, variant: str, doc_id: int) -> Path | None:
    for base in (job_dir / f"dataset_{job}_images" / f"dataset_{job}_images",
                 job_dir / f"dataset_{job}_images"):
        p = base / f"{task}_{variant}_{doc_id:06d}.png"
        if p.exists():
            return p
    return None


def flat(x):
    while isinstance(x, list) and x:
        x = x[0]
    return x


def extracted(rec: dict) -> str:
    v = flat(rec.get("filtered_resps"))
    return "" if v is None else str(v)


def raw_output(rec: dict) -> str:
    r = flat(rec.get("resps"))
    if r is None or (isinstance(r, str) and r.strip() in ("", "None")):
        r = flat(rec.get("filtered_resps"))
    return "" if r is None else str(r)


def load_cell(job_dir: Path, task: str, variant: str) -> dict[int, dict]:
    f = latest_samples(job_dir, task, variant)
    if f is None:
        return {}
    out = {}
    for line in f.open(encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["doc_id"]] = rec
    return out


def link_or_copy(src: Path | None, dst: Path) -> str:
    if src is None:
        return "MISSING"
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "ok"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def write_answer_file(path: Path, model_pretty: str, task: str, variant: str,
                      diff: str, doc_id: int, target: str, rec: dict | None):
    if rec is None:
        path.write_text(f"model: {model_pretty}\nSTATUS: MISSING (no sample logged)\n", encoding="utf-8")
        return
    ans = extracted(rec)
    score = (rec.get("accuracy") or {}).get("score")
    verdict = "CORRECT" if (score is not None and score > 0.5) else "WRONG"
    raw = raw_output(rec)
    body = (
        f"model: {model_pretty}\n"
        f"task: {task} | variant: {variant} | difficulty: {diff} | q: {doc_id}\n"
        f"ground_truth: {target}\n"
        f"extracted_answer: {ans}\n"
        f"score: {score}  ({verdict})\n"
        f"raw_output_chars: {len(raw)}\n"
        f"{'-' * 40} raw model output {'-' * 40}\n"
        f"{raw}\n"
    )
    path.write_text(body, encoding="utf-8")


# --- timing ------------------------------------------------------------------

_THROUGHPUT_KEYS = ("total_gen_tokens", "total_elapsed_time", "avg_speed", "avg_ttft")


def job_timing(job_dir: Path) -> dict:
    """Parse the fetched run.log for wall time + the Throughput Summary table."""
    info: dict = {}
    log = job_dir / ".run" / "run.log"
    if not log.exists():
        logs = list(job_dir.rglob("run.log"))
        log = logs[0] if logs else None
    if log is None or not log.exists():
        return info
    text = log.read_text(encoding="utf-8", errors="replace")
    # wall clock from the launcher's [start]/[end] stamps
    m0 = re.search(r"^\[start\] (\S+)", text, re.M)
    m1 = re.search(r"^\[end\] (\S+)", text, re.M)
    if m0 and m1:
        info["start"], info["end"] = m0.group(1), m1.group(1)
    for key in _THROUGHPUT_KEYS:
        m = re.search(rf"\|{key}\s*\|\s*([0-9.]+)\|", text)
        if m:
            info[key] = float(m.group(1))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clean", action="store_true", help="wipe --out first")
    args = ap.parse_args(argv)

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    index_rows, timing_rows = [], []
    n_q = n_img_missing = 0
    missing_jobs = []

    for tier in TIERS:
        for arm in ARMS:
            for diff in DIFFS:
                # jobs are per (tier, arm, diff, model) and hold all 3 tasks
                job_dirs: dict[str, tuple[str, Path]] = {}
                for model in MODELS:
                    job = job_name(tier, arm, diff, model)
                    jd = find_job_dir(job)
                    if jd is None:
                        missing_jobs.append(job)
                        continue
                    job_dirs[model] = (job, jd)
                    t = job_timing(jd)
                    timing_rows.append((job, tier, arm, diff, model, t))
                for task in TASKS:
                    for variant, vfolder in VARIANTS.items():
                        recs: dict[str, dict[int, dict]] = {}
                        img_src = None
                        for model, (job, jd) in job_dirs.items():
                            recs[model] = load_cell(jd, task, variant)
                        doc_ids = sorted({d for r in recs.values() for d in r})
                        for doc_id in doc_ids:
                            rep = next((recs[m][doc_id] for m in MODELS if doc_id in recs.get(m, {})), None)
                            if rep is None:
                                continue
                            qdir = args.out / tier / arm / task / diff / vfolder / f"q_{doc_id:03d}"
                            qdir.mkdir(parents=True, exist_ok=True)
                            target = str(rep.get("target", ""))
                            (qdir / "prompt.txt").write_text(str(rep.get("input", "")), encoding="utf-8")
                            (qdir / "answer.txt").write_text(target + "\n", encoding="utf-8")
                            img_src = None
                            for model, (job, jd) in job_dirs.items():
                                img_src = img_src or image_path(jd, job, task, variant, doc_id)
                            if link_or_copy(img_src, qdir / "image.png") == "MISSING":
                                n_img_missing += 1
                            row = [tier, arm, task, diff, variant, doc_id, target]
                            for model, stem in MODELS.items():
                                rec = recs.get(model, {}).get(doc_id)
                                write_answer_file(qdir / f"{stem}.txt", MODEL_PRETTY[model], task, variant,
                                                  diff, doc_id, target, rec)
                                if rec is None:
                                    row += ["MISSING", "", ""]
                                else:
                                    sc = (rec.get("accuracy") or {}).get("score")
                                    row += [extracted(rec).replace("\t", " ").replace("\n", " ")[:60],
                                            sc, len(raw_output(rec))]
                            index_rows.append(row)
                            n_q += 1

    hdr = ["tier", "arm", "task", "difficulty", "variant", "q", "target"]
    for stem in MODELS.values():
        base = stem.replace("_answer", "")
        hdr += [f"{base}_ans", f"{base}_score", f"{base}_rawchars"]
    with (args.out / "_index.tsv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(hdr) + "\n")
        for row in index_rows:
            fh.write("\t".join("" if c is None else str(c) for c in row) + "\n")

    # timing notes
    lines = [
        "fp8fix smoke — per-job wall time and throughput (source: fetched run.log)",
        "job = one (tier, arm, difficulty, model); 3 tasks x 2 renders x n=2 = 12 samples/job",
        "wall time includes dataset prepare + vllm engine load/quantize; generation-only",
        "cost is total_elapsed_time (s) from lmms-eval's Throughput Summary.",
        "",
        f"{'job':64s} {'gen_s':>8s} {'tok':>8s} {'tok/s':>7s} {'ttft_s':>8s}  start..end",
    ]
    for job, tier, arm, diff, model, t in timing_rows:
        lines.append(
            f"{job:64s} {t.get('total_elapsed_time', float('nan')):8.1f} "
            f"{t.get('total_gen_tokens', float('nan')):8.0f} "
            f"{t.get('avg_speed', float('nan')):7.1f} "
            f"{t.get('avg_ttft', float('nan')):8.1f}  "
            f"{t.get('start','?')} .. {t.get('end','?')}")
    (args.out / "_timing_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (args.out / "_README.txt").write_text(
        "fp8fix pre-flight smoke (fp8 LM + bf16 ViT fix; InternVL think force-close),\n"
        "exploded per question.\n\n"
        "Path: <adj|img>/<arm>/<task>/<difficulty>/<orig|disg>/q_<NNN>/\n"
        "  adj = tier-1 / 08-style (adjacency list in prompt); img = tier-2 / 07-style (image only).\n"
        "  orig = direct node-link image ; disg = disguise region/map render (same graph).\n"
        "  prompt.txt = full prompt; answer.txt = ground truth; image.png = exact stored image;\n"
        "  <model>_answer.txt = raw output + extracted + score + raw length.\n\n"
        "_index.tsv     every question x model (extracted answer, score, raw chars).\n"
        "_timing_notes.txt  per-job wall/generation time + throughput.\n"
        "_analysis_notes.txt  senior-researcher pass: vision grounding, oddities, hypotheses.\n\n"
        "Seed 42 = same graphs as the campaign datasets. NOTE: coloring direct q_000 has a\n"
        "known stored-image render bug (wrong graph in the image; see project memory) —\n"
        "in the img tier it doubles as a vision canary.\n",
        encoding="utf-8")

    print(f"questions written: {n_q}")
    print(f"images missing:    {n_img_missing}")
    if missing_jobs:
        print(f"jobs missing ({len(missing_jobs)}):")
        for j in missing_jobs:
            print("  ", j)
    print(f"output root:       {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
