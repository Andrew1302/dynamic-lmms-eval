"""Explode the think+adjacency ablation (n=100 base) into a browsable per-question tree.

Layout (difficulty inserted so easy/medium/hard doc_id 0-99 don't collide):

    <out>/<arm>/<task>/<difficulty>/<orig|disg>/q_<NNN>/
        prompt.txt              full model prompt (incl. text adjacency list)
        answer.txt              ground-truth target
        image.png               the exact stored image the model was shown
        qwen_35_answer.txt       Qwen3.5-4B    output (raw + extracted + score)
        gemma_4_2b_answer.txt    Gemma-4-E2B   output
        internvl_35_answer.txt   InternVL3.5-4B output

Prompt / answer / image are model-independent (graphs are seed-identical), written
once per question from whichever model is available. A root ``_index.tsv`` lists every
question with each model's extracted answer + score for fast filtering during analysis.

Reads ONLY merged (non-/chunks/) sample logs. Coloring is sourced from the dedicated
``_coloring_`` jobs (special-χ). Images are hardlinked (fallback copy) to save disk.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
from pathlib import Path

JOBS = Path(r"C:/Users/Andrew/Msc/dynamic-lmms-eval/remote_results/08_abl_thinkadj/_jobs")
DEFAULT_OUT = Path(r"C:/Users/Andrew/Msc/dynamic-lmms-eval/remote_results/_inspect/thinkadj_n100")

MODELS = {  # cell token -> answer filename stem
    "qwen35_4b": "qwen_35_answer",
    "gemma4_e2b": "gemma_4_2b_answer",
    "internvl35_4b": "internvl_35_answer",
}
TASKS = ["directed_connectivity", "shortest_path", "coloring"]
DIFFS = ["easy", "medium", "hard"]
ARMS = ["think", "nothink"]
VARIANTS = {"direct": "orig", "disguise": "disg"}


def job_name(arm: str, task: str, diff: str, model: str) -> str:
    col = "coloring_" if task == "coloring" else ""
    return f"graph_bench_thinkadj_{arm}_{col}{diff}_{model}"


def latest_merged(job_dir: Path, task: str, variant: str) -> Path | None:
    """Newest merged (non-chunk) samples file for a (task, variant)."""
    pat = f"*_samples_dynamic_graph_benchmark_{task}_{variant}.jsonl"
    cands = [f for f in job_dir.rglob(pat) if "/chunks/" not in f.as_posix()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.name)  # timestamp prefix sorts lexically


def image_path(job_dir: Path, job: str, task: str, variant: str, doc_id: int) -> Path:
    base = job_dir / f"dataset_{job}_images" / f"dataset_{job}_images"
    return base / f"{task}_{variant}_{doc_id:06d}.png"


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
        r = flat(rec.get("filtered_resps"))          # nothink: nothing was stripped
    return "" if r is None else str(r)


def load_cell(job_dir: Path, task: str, variant: str) -> dict[int, dict]:
    f = latest_merged(job_dir, task, variant)
    if f is None:
        return {}
    out = {}
    for line in f.open(encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["doc_id"]] = rec
    return out


def link_or_copy(src: Path, dst: Path) -> str:
    if not src.exists():
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
    body = (
        f"model: {model_pretty}\n"
        f"task: {task} | variant: {variant} | difficulty: {diff} | q: {doc_id}\n"
        f"ground_truth: {target}\n"
        f"extracted_answer: {ans}\n"
        f"score: {score}  ({verdict})\n"
        f"{'-' * 40} raw model output {'-' * 40}\n"
        f"{raw_output(rec)}\n"
    )
    path.write_text(body, encoding="utf-8")


MODEL_PRETTY = {"qwen35_4b": "Qwen3.5-4B", "gemma4_e2b": "Gemma-4-E2B", "internvl35_4b": "InternVL3.5-4B"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--diffs", nargs="+", default=DIFFS, choices=DIFFS)
    ap.add_argument("--limit", type=int, default=None, help="max questions per cell (for a quick look)")
    ap.add_argument("--clean", action="store_true", help="wipe --out first")
    args = ap.parse_args(argv)

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    index_rows = []
    n_q = n_img_missing = 0

    for arm in args.arms:
        for task in args.tasks:
            for diff in args.diffs:
                for variant, vfolder in VARIANTS.items():
                    # load every model's records for this cell
                    recs: dict[str, dict[int, dict]] = {}
                    img_job = img_dir = None
                    for model in MODELS:
                        job = job_name(arm, task, diff, model)
                        jd = JOBS / job
                        if not jd.is_dir():
                            recs[model] = {}
                            continue
                        recs[model] = load_cell(jd, task, variant)
                        if img_job is None and recs[model]:
                            img_job, img_dir = job, jd
                    doc_ids = sorted({d for r in recs.values() for d in r})
                    if args.limit is not None:
                        doc_ids = doc_ids[: args.limit]
                    for doc_id in doc_ids:
                        # a representative record for prompt/answer/image
                        rep = next((recs[m][doc_id] for m in MODELS if doc_id in recs.get(m, {})), None)
                        if rep is None:
                            continue
                        qdir = args.out / arm / task / diff / vfolder / f"q_{doc_id:03d}"
                        qdir.mkdir(parents=True, exist_ok=True)
                        target = str(rep.get("target", ""))
                        (qdir / "prompt.txt").write_text(str(rep.get("input", "")), encoding="utf-8")
                        (qdir / "answer.txt").write_text(target + "\n", encoding="utf-8")
                        if img_dir is not None:
                            st = link_or_copy(image_path(img_dir, img_job, task, variant, doc_id), qdir / "image.png")
                            if st == "MISSING":
                                n_img_missing += 1
                        row = [arm, task, diff, variant, doc_id, target]
                        for model, stem in MODELS.items():
                            rec = recs.get(model, {}).get(doc_id)
                            write_answer_file(qdir / f"{stem}.txt", MODEL_PRETTY[model], task, variant,
                                              diff, doc_id, target, rec)
                            if rec is None:
                                row += ["MISSING", ""]
                            else:
                                sc = (rec.get("accuracy") or {}).get("score")
                                row += [extracted(rec).replace("\t", " ").replace("\n", " ")[:60], sc]
                        index_rows.append(row)
                        n_q += 1

    # root index for fast filtering
    hdr = ["arm", "task", "difficulty", "variant", "q", "target"]
    for stem in MODELS.values():
        hdr += [f"{stem.replace('_answer','')}_ans", f"{stem.replace('_answer','')}_score"]
    with (args.out / "_index.tsv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(hdr) + "\n")
        for row in index_rows:
            fh.write("\t".join("" if c is None else str(c) for c in row) + "\n")

    (args.out / "_README.txt").write_text(
        "Think + adjacency-list ablation (n=100 base), exploded per question.\n\n"
        "Path: <arm>/<task>/<difficulty>/<orig|disg>/q_<NNN>/\n"
        "  orig = direct node-link image ; disg = disguise region/map render (same graph).\n"
        "  prompt.txt = full prompt incl. text adjacency; answer.txt = ground truth;\n"
        "  image.png = exact stored image shown; <model>_answer.txt = raw output + extracted + score.\n\n"
        "_index.tsv lists every question with each model's extracted answer + score.\n"
        "NOTE: coloring direct q_000 has a known stored-image render bug (see project memory).\n",
        encoding="utf-8")

    print(f"questions written: {n_q}")
    print(f"images missing:    {n_img_missing}")
    print(f"output root:       {args.out}")
    print(f"index:             {args.out / '_index.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
