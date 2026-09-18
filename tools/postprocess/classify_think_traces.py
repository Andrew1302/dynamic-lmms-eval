"""Classify failing THINK-arm traces into three failure modes and surface excerpts.

Modes (per the analysis spec):
  A misread-then-consistent : the chain asserts structure (edge/corridor/border)
      absent from the true adjacency, then reasons correctly over that wrong
      graph to a wrong answer.
  B read-then-flip          : the chain first extracts structure correctly and
      reaches the right answer, then doubts it and commits to a different wrong one.
  C truncated               : the chain never emits a final answer (hit the token
      budget mid-reasoning).

Scope: connectivity + coloring, both surfaces, think arm, 3 models, 3 difficulties,
for BOTH the thinking ablation (07_abl_think, image-only) and the adjacency+thinking
ablation (08_abl_thinkadj, adjacency in prompt).

CAVEAT (printed in the report): 07/08 are the PRE-FIX runs — the vision tower was
fp8-quantized (blind) for Qwen/InternVL, so on the IMAGE (07 / original-surface)
cells category A is dominated by blindness, not genuine misreading. The 08 adj arm
(adjacency supplied as text) is unaffected by vision blindness and is the meaningful
signal. Every number here is preliminary; regenerate on the post-fix rerun.

Heuristics are deliberately conservative and dump their evidence so a human (or a
verifier subagent) can audit. Output: <out>/summary.txt (+ per-cell tsv + excerpts/).

  python tools/postprocess/classify_think_traces.py --root remote_results --out remote_results/_inspect/think_analysis
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import defaultdict

BATCHES = {"07_abl_think": "think(img)", "08_abl_thinkadj": "adj+think"}
TASKS = ["connectivity", "coloring"]
SURFACES = ["original", "disguise"]
MODELS = ["gemma", "internvl", "qwen"]
DIFFS = ["easy", "medium", "hard"]

FLIP = re.compile(r"\b(wait|actually|but wait|on second thought|let me (re|double)|"
                  r"reconsider|rethink|hold on|hmm,? let|recheck|re-check|"
                  r"i made (a|an) (mistake|error)|that'?s wrong|correction)\b", re.I)
# thinking-channel / think-tag markers used to split reasoning from the final answer
THINK_SPLIT = re.compile(r"</think>|<\|channel\|?>final|<\|start\|>assistant|\bFinal answer\b|\bAnswer:\s", re.I)


def flat(x):
    while isinstance(x, list) and x:
        x = x[0]
    return x


def final_answer(rec) -> str:
    v = flat(rec.get("filtered_resps"))
    return "" if v is None else str(v).strip()


def raw_cot(rec) -> str:
    r = flat(rec.get("resps"))
    return "" if r is None else str(r)


def gold(rec) -> str:
    return str(rec.get("target", "")).strip()


def out_tokens(rec) -> int:
    tc = flat(rec.get("token_counts")) or {}
    if isinstance(tc, dict):
        return int(tc.get("output_tokens", 0) or 0)
    return 0


def true_adj(rec) -> dict[int, set] | None:
    """Parse the adjacency list from the prompt (present only in the adj arm)."""
    inp = flat(rec.get("input")) or ""
    adj = {}
    for m in re.finditer(r"Adj\[(\d+)\]\s*=\s*\[([^\]]*)\]", str(inp)):
        u = int(m.group(1))
        nbrs = set()
        for tok in re.findall(r"\d+", m.group(2)):
            nbrs.add(int(tok))
        adj[u] = nbrs
    return adj or None


def asserted_edges(cot: str) -> set[tuple[int, int]]:
    """Edges the chain claims, from 'a -> b', 'a - b', 'a,b', 'edge (a,b)' forms."""
    e = set()
    for a, b in re.findall(r"(\d+)\s*(?:->|→|--|-|to|,|and)\s*(\d+)", cot):
        e.add((int(a), int(b)))
    return e


def classify(rec, budget: int):
    g = gold(rec)
    fa = final_answer(rec)
    cot = raw_cot(rec)
    toks = out_tokens(rec)

    # normalize yes/no + integer answers
    def norm(s):
        s = s.lower()
        if "yes" in s and "no" not in s.replace("node", ""):
            return "yes"
        if re.search(r"\bno\b", s):
            return "no"
        m = re.search(r"-?\d+", s)
        return m.group(0) if m else s.strip()

    gn, fan = norm(g), norm(fa)

    # C truncated: no usable final answer AND the budget was (near) exhausted
    near_budget = budget and toks >= budget - 64
    no_answer = fan == "" or fan == fa.strip().lower() and not re.search(r"yes|no|\d", fan)
    # split reasoning vs tail; truncation = no answer segment after the reasoning
    has_answer_seg = bool(THINK_SPLIT.search(cot)) or bool(re.search(r"(?:answer|chromatic number|is)\s*[:=]?\s*(yes|no|\d+)\s*$", cot.strip(), re.I))
    if (no_answer and (near_budget or not has_answer_seg)):
        return "C", f"toks={toks} near_budget={bool(near_budget)} extracted={fa[:20]!r}"

    # B read-then-flip: gold appears as an intermediate conclusion, a flip marker
    # follows, and the committed answer differs from gold.
    body = cot
    # find gold mentioned as a value/conclusion inside the chain (not only prompt echo)
    if gn in ("yes", "no"):
        gold_hits = [m.start() for m in re.finditer(rf"\b{gn}\b", body, re.I)]
    else:
        gold_hits = [m.start() for m in re.finditer(rf"(?<!\d){re.escape(gn)}(?!\d)", body)]
    flip_hits = [m.start() for m in FLIP.finditer(body)]
    if fan != gn and gold_hits and flip_hits:
        # a flip marker occurs at/after a gold-as-conclusion mention
        if any(fh > gh - 40 for gh in gold_hits for fh in flip_hits):
            ev = f"gold={g!r} committed={fa[:20]!r} flip_markers={len(flip_hits)}"
            return "B", ev

    # A misread-then-consistent (residual, verified against text adjacency when present)
    adj = true_adj(rec)
    ev = f"gold={g!r} committed={fa[:24]!r}"
    if adj:
        claimed = asserted_edges(cot)
        phantom = [(a, b) for (a, b) in claimed
                   if a in adj and b not in adj[a] and b in adj and a not in adj.get(b, set())]
        ev += f" phantom_edges={phantom[:4]}"
    return "A", ev


def budget_for(batch, model):
    # think budgets used by run_eval.sh (approx caps for truncation detection)
    return 12288 if model in ("internvl", "qwen") else 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("remote_results"))
    ap.add_argument("--out", type=Path, default=Path("remote_results/_inspect/think_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "excerpts").mkdir(exist_ok=True)

    per_model = defaultdict(lambda: defaultdict(int))          # model -> {A,B,C,total_fail,total}
    per_cell = []                                              # rows for tsv
    examples = defaultdict(list)                               # (model,mode,batch,surface) -> recs

    for batch in BATCHES:
        for task in TASKS:
            for surface in SURFACES:
                for model in MODELS:
                    for diff in DIFFS:
                        f = args.root / batch / task / surface / f"{model}_{diff}_think.jsonl"
                        if not f.exists():
                            continue
                        recs = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
                        budget = budget_for(batch, model)
                        nA = nB = nC = 0
                        for i, r in enumerate(recs):
                            if (r.get("accuracy") or {}).get("score", 0) >= 0.5:
                                continue
                            mode, ev = classify(r, budget)
                            if mode == "A": nA += 1
                            elif mode == "B": nB += 1
                            else: nC += 1
                            per_model[model][mode] += 1
                            per_model[model]["fail"] += 1
                            examples[(model, mode, batch, surface, task)].append(
                                (f, i, r.get("doc_id"), diff, ev))
                        per_model[model]["total"] += len(recs)
                        per_cell.append((batch, task, surface, model, diff,
                                         len(recs), nA + nB + nC, nA, nB, nC))

    # ---- write summary ----
    lines = []
    lines.append("PRE-FIX think-trace failure-mode classification (PRELIMINARY).")
    lines.append("07=thinking ablation (image-only, fp8-BLIND for qwen/internvl); "
                 "08=adjacency+thinking (adjacency as text, blindness-immune).")
    lines.append("A=misread-then-consistent  B=read-then-flip  C=truncated\n")
    lines.append(f"{'model':10s} {'fails':>6s} {'A':>5s} {'B':>5s} {'C':>5s}  {'A%':>5s} {'B%':>5s} {'C%':>5s}  A:B")
    for model in MODELS:
        d = per_model[model]
        fa = d["fail"] or 1
        ab = f"{d['A']}:{d['B']}"
        lines.append(f"{model:10s} {d['fail']:6d} {d['A']:5d} {d['B']:5d} {d['C']:5d}  "
                     f"{100*d['A']/fa:4.0f}% {100*d['B']/fa:4.0f}% {100*d['C']/fa:4.0f}%  {ab}")
    lines.append("")
    # split by batch (07 image vs 08 adj) for the caveat-aware read
    for batch in BATCHES:
        lines.append(f"--- {batch} ({BATCHES[batch]}) per-model A/B/C ---")
        bm = defaultdict(lambda: defaultdict(int))
        for (b, task, surface, model, diff, n, nf, nA, nB, nC) in per_cell:
            if b != batch: continue
            bm[model]["A"] += nA; bm[model]["B"] += nB; bm[model]["C"] += nC; bm[model]["fail"] += nf
        for model in MODELS:
            d = bm[model]; fa = d["fail"] or 1
            lines.append(f"  {model:10s} fails={d['fail']:4d}  A={d['A']:4d} ({100*d['A']/fa:3.0f}%)  "
                         f"B={d['B']:4d} ({100*d['B']/fa:3.0f}%)  C={d['C']:4d} ({100*d['C']/fa:3.0f}%)")
        lines.append("")
    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- per-cell tsv ----
    with (args.out / "per_cell.tsv").open("w", encoding="utf-8") as fh:
        fh.write("batch\ttask\tsurface\tmodel\tdiff\tn\tfail\tA\tB\tC\n")
        for row in per_cell:
            fh.write("\t".join(str(x) for x in row) + "\n")

    # ---- candidate-excerpt index (paths so a verifier can pull the raw CoT) ----
    with (args.out / "excerpt_candidates.tsv").open("w", encoding="utf-8") as fh:
        fh.write("model\tmode\tbatch\tsurface\ttask\tfile\trec_index\tdoc_id\tdiff\tevidence\n")
        for (model, mode, batch, surface, task), lst in examples.items():
            for (f, i, doc, diff, ev) in lst[:20]:
                fh.write(f"{model}\t{mode}\t{batch}\t{surface}\t{task}\t{f}\t{i}\t{doc}\t{diff}\t{ev}\n")

    print((args.out / "summary.txt").read_text(encoding="utf-8"))
    print(f"\n[written] {args.out}/summary.txt, per_cell.tsv, excerpt_candidates.tsv")


if __name__ == "__main__":
    main()
