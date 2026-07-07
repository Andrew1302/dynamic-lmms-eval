"""Validate a thinking-vs-no-thinking ablation run from the lmms-eval sample logs.

The thinking arm is only trustworthy if, per model:
  1. the model actually emitted reasoning (a closed <think>...</think> block), and
  2. the scored answer was extracted from the *post-reasoning* text via the task
     regex — not from a number that happened to appear inside the reasoning.

This script proves both from the fetched ``*_samples_<task>.jsonl`` files. It
reads the RAW ``resps`` (pre-strip) alongside ``filtered_resps`` (post-strip) —
the evaluator keeps ``resps`` only when it differs from ``filtered_resps``, i.e.
exactly when a reasoning block was stripped (evaluator.py:1122,
evaluation_tracker.py:280), and falls back to ``filtered_resps`` otherwise. It
reuses the PRODUCTION ``strip_reasoning_tags`` + task ``_normalize`` so the
verdict matches what scoring actually did. Reasoning presence is detected from
the raw text (the vllm wrapper never populates ``token_counts.reasoning_tokens``,
so token counts are unavailable for Qwen3.5 / Gemma-4).

Usage:
    python tools/postprocess/verify_thinking.py [ROOT ...] [--dump N] [--yaml PATH]

ROOT defaults to ``remote_results``. Pass one or more fetched job/batch dirs.
Per (model, arm) it prints reasoning-present / truncation / answer-isolation /
extraction-consistency rates and accuracy, then dumps the first N raw→filtered→
extracted→target→score samples per model so extraction is eyeballable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # tools/postprocess -> repo root

# Reuse the sample-jsonl discovery + filename parsing from the shared loader.
sys.path.insert(0, str(_HERE))
from _logs import SAMPLES_RE, _base_task_and_variant, find_sample_jsonls  # noqa: E402


def _ensure_loguru_stub() -> None:
    """The task utils import ``loguru`` at module scope (used only by the
    aggregate helpers, not ``_normalize``). loguru lives in the remote eval
    ``.venv``; when running this validator locally it may be absent. Stub it so
    we can import the REAL ``_normalize`` (no copy = no drift) without pulling
    the dependency."""
    try:
        import loguru  # noqa: F401
    except ModuleNotFoundError:
        import types

        stub = types.ModuleType("loguru")
        stub.logger = types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None,
            error=lambda *a, **k: None, debug=lambda *a, **k: None,
        )
        sys.modules["loguru"] = stub


def _load_by_path(mod_name: str, rel_path: str):
    """Import a single module by file path, bypassing the heavy ``lmms_eval``
    package __init__ (which would pull in every model wrapper)."""
    spec = importlib.util.spec_from_file_location(mod_name, _REPO_ROOT / rel_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ensure_loguru_stub()
_reasoning = _load_by_path("_dgb_reasoning", "lmms_eval/api/reasoning.py")
_utils = _load_by_path("_dgb_utils", "lmms_eval/tasks/dynamic_graph_benchmark/utils.py")
strip_reasoning_tags = _reasoning.strip_reasoning_tags
_normalize = _utils._normalize

_DEFAULT_TAGS = [["<think>", "</think>"]]
_MODEL_SHORTS = ("internvl35_4b", "qwen35_4b", "gemma4_e2b")


def _load_reasoning_tags(yaml_path: Path | None) -> list[list[str]]:
    """The tag pairs the eval actually stripped with — read from the task yaml
    so this stays in sync (incl. any Gemma tag added later). Falls back to the
    <think> default if the file/key is unavailable."""
    path = yaml_path or (
        _REPO_ROOT / "lmms_eval/tasks/dynamic_graph_benchmark/_default_template_yaml"
    )
    try:
        import yaml  # lmms_eval depends on PyYAML

        # The task yaml uses `!function ...` tags (doc_to_text etc.) that
        # SafeLoader rejects. Ignore any custom tag — we only want reasoning_tags.
        class _TolerantLoader(yaml.SafeLoader):
            pass

        _TolerantLoader.add_multi_constructor(
            "!", lambda loader, suffix, node: None
        )
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_TolerantLoader)
        tags = data.get("reasoning_tags")
        if tags:
            return [list(pair) for pair in tags]
    except Exception as exc:  # noqa: BLE001
        print(f"[verify] WARN: could not read reasoning_tags from {path}: {exc}")
    return _DEFAULT_TAGS


def _job_label(path: Path) -> str:
    """Nearest ancestor dir encoding the job (model+arm+difficulty). Matches any
    graph_bench_* job dir (campaign `graph_bench_think_*` and `thinksmoke_*`)."""
    for part in path.parts:
        if part.startswith("graph_bench_"):
            return part
    return path.parent.name


def _model_of(label: str) -> str:
    for short in _MODEL_SHORTS:
        if short in label:
            return short
    return "unknown_model"


def _arm_of(label: str) -> str:
    # "nothink" contains "think", so test it first.
    if "nothink" in label:
        return "nothink"
    if "think" in label:
        return "think"
    return "unknown"


def _raw_resp(rec: dict) -> str:
    """The pre-strip response. Mirrors evaluator.py:1122: prefer ``resps`` (kept
    only when a reasoning block was stripped), else the unmodified response."""
    val = rec.get("resps", rec.get("filtered_resps", ""))
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val)


def _filtered_resp(rec: dict) -> str:
    val = rec.get("filtered_resps", "")
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val)


def _score(rec: dict) -> int:
    acc = rec.get("accuracy")
    if isinstance(acc, dict) and "score" in acc:
        return int(round(float(acc["score"])))
    return 0


def _out_tokens(rec: dict) -> int | None:
    """Exact generated (output) token count, if the model wrapper recorded it
    (vllm path). rec['token_counts'] is a per-request list of dicts."""
    tc = rec.get("token_counts")
    if isinstance(tc, list) and tc and isinstance(tc[0], dict):
        v = tc[0].get("output_tokens")
        return int(v) if v is not None else None
    return None


@dataclass
class Diag:
    label: str
    model: str
    arm: str
    base_task: str
    variant: str
    doc_id: int
    raw: str
    filtered: str
    target: str
    extracted: str
    score: int
    reasoning_present: bool   # a closed reasoning block exists in raw
    no_close_tag: bool        # reasoning arm but no closing tag (truncation suspect)
    answer_isolated: bool     # filtered carries no reasoning tag
    consistent: bool          # re-normalized filtered reproduces the logged score
    out_tokens: int | None    # exact generated token count (vllm), else None


def _diagnose(rec: dict, task_full: str, label: str, tags: list[list[str]]) -> Diag:
    base_task, variant = _base_task_and_variant(task_full)
    raw = _raw_resp(rec)
    filtered = _filtered_resp(rec)
    target = str(rec.get("target", ""))
    score = _score(rec)
    arm = _arm_of(label)

    opens = [t[0] for t in tags]
    closes = [t[1] for t in tags]
    has_close = any(c in raw for c in closes)
    has_open = any(o in raw for o in opens)
    # A closed reasoning block present in the raw text = reasoning genuinely emitted.
    reasoning_present = has_close
    # Truncation suspect: reasoning started (open tag, or a long think-arm body)
    # but never closed -> strip either drops everything or nothing, losing the
    # answer. For the close-only Qwen shape there's no literal open tag, so also
    # flag long think-arm responses that never closed.
    no_close_tag = (
        arm == "think"
        and not has_close
        and (has_open or len(raw) > 200)
    )
    # The answer regex must only see post-reasoning text.
    answer_isolated = not any(o in filtered or c in filtered for o, c in zip(opens, closes))

    extracted = _normalize(filtered, base_task)
    recomputed = 1 if extracted == _normalize(target, base_task) else 0
    consistent = recomputed == score

    return Diag(
        label=label,
        model=_model_of(label),
        arm=arm,
        base_task=base_task,
        variant=variant,
        doc_id=int(rec.get("doc_id", -1)),
        raw=raw,
        filtered=filtered,
        target=target,
        extracted=extracted,
        score=score,
        reasoning_present=reasoning_present,
        no_close_tag=no_close_tag,
        answer_isolated=answer_isolated,
        consistent=consistent,
        out_tokens=_out_tokens(rec),
    )


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):5.1f}%" if den else "  n/a"


def _truncate(s: str, n: int = 240) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", default=["remote_results"],
                    help="Fetched job/batch dirs to scan (default: remote_results)")
    ap.add_argument("--dump", type=int, default=10,
                    help="Raw→filtered samples to print per model (default 10)")
    ap.add_argument("--yaml", type=Path, default=None,
                    help="Task yaml to read reasoning_tags from")
    args = ap.parse_args(argv)

    # Non-ASCII marks (→ ✓ …) crash on Windows cp1252 stdout; force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    tags = _load_reasoning_tags(args.yaml)
    print(f"[verify] reasoning_tags = {tags}\n")

    jsonls: list[Path] = []
    for root in args.roots:
        jsonls.extend(find_sample_jsonls(Path(root)))
    if not jsonls:
        print(f"[verify] no *_samples_*.jsonl found under {args.roots}")
        return 1

    diags: list[Diag] = []
    for p in jsonls:
        m = SAMPLES_RE.match(p.name)
        if not m:
            continue
        label = _job_label(p)
        task_full = m.group("task")
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                diags.append(_diagnose(json.loads(line), task_full, label, tags))

    if not diags:
        print("[verify] no sample rows parsed")
        return 1

    # --- Per (model, arm) summary --------------------------------------------
    groups: dict[tuple[str, str], list[Diag]] = defaultdict(list)
    for d in diags:
        groups[(d.model, d.arm)].append(d)

    print("=" * 100)
    print(f"{'model':16} {'arm':8} {'n':>5} {'reason+':>8} {'no-close':>9} "
          f"{'ans-iso':>8} {'extract=score':>13} {'acc':>7}")
    print("-" * 100)
    ok = True
    for (model, arm), rows in sorted(groups.items()):
        n = len(rows)
        reason = sum(r.reasoning_present for r in rows)
        noclose = sum(r.no_close_tag for r in rows)
        iso = sum(r.answer_isolated for r in rows)
        cons = sum(r.consistent for r in rows)
        acc = sum(r.score for r in rows)
        print(f"{model:16} {arm:8} {n:5d} {_pct(reason, n):>8} {_pct(noclose, n):>9} "
              f"{_pct(iso, n):>8} {_pct(cons, n):>13} {_pct(acc, n):>7}")
        # Gate: extraction must always reproduce the score; answers must be
        # isolated; the think arm must actually reason.
        if cons != n or iso != n:
            ok = False
        if arm == "think" and reason < n:
            ok = False
    print("=" * 100)
    print("reason+   = raw has a closed reasoning block (thinking genuinely emitted)")
    print("no-close  = think arm, reasoning never closed (truncation suspect → answer lost)")
    print("ans-iso   = filtered_resps carries NO reasoning tag (regex saw answer only)")
    print("extract=score = re-normalizing filtered_resps reproduces the logged score")

    # --- Output-token usage per job (think arm; vllm records exact counts) ----
    import statistics as _st

    def _pctl(vals: list[int], q: float) -> int:
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else 0

    tok = defaultdict(list)   # label -> [out_tokens]
    ncl = defaultdict(int)    # label -> no_close count
    for d in diags:
        if d.arm == "think" and d.out_tokens is not None:
            tok[d.label].append(d.out_tokens)
            if d.no_close_tag:
                ncl[d.label] += 1
    if tok:
        print("\noutput tokens per response (think arm):")
        print(f"  {'job':50} {'n':>4} {'med':>6} {'p90':>6} {'max':>6} {'trunc':>7}")
        for lbl in sorted(tok):
            v = tok[lbl]
            print(f"  {lbl[:50]:50} {len(v):4d} {int(_st.median(v)):6d} "
                  f"{_pctl(v, 0.9):6d} {max(v):6d} {_pct(ncl[lbl], len(v)):>7}")
    print()

    # --- Flagged rows (the actionable failures) ------------------------------
    flagged = [d for d in diags if not d.consistent or not d.answer_isolated
               or d.no_close_tag]
    if flagged:
        print(f"[verify] {len(flagged)} flagged row(s) (inconsistent / not-isolated / truncated):")
        for d in flagged[:40]:
            why = []
            if not d.consistent:
                why.append("extract≠score")
            if not d.answer_isolated:
                why.append("tag-in-answer")
            if d.no_close_tag:
                why.append("no-close-tag")
            print(f"  [{','.join(why)}] {d.label} {d.base_task}/{d.variant} "
                  f"doc={d.doc_id} score={d.score} extracted={d.extracted!r} target={d.target!r}")
            print(f"      raw:      {_truncate(d.raw)}")
            print(f"      filtered: {_truncate(d.filtered)}")
        print()

    # --- Per-model eyeball dump (>=10 samples) -------------------------------
    by_model: dict[str, list[Diag]] = defaultdict(list)
    for d in diags:
        by_model[d.model].append(d)
    for model, rows in sorted(by_model.items()):
        print("#" * 100)
        print(f"# {model}: first {args.dump} samples (think arm prioritized)")
        print("#" * 100)
        rows_sorted = sorted(rows, key=lambda r: (r.arm != "think", r.base_task, r.doc_id))
        for d in rows_sorted[: args.dump]:
            mark = "✓" if d.score else "✗"
            print(f"[{d.arm}] {d.base_task}/{d.variant} doc={d.doc_id} {mark} "
                  f"reason+={int(d.reasoning_present)} iso={int(d.answer_isolated)}")
            print(f"    raw:      {_truncate(d.raw)}")
            print(f"    filtered: {_truncate(d.filtered)}")
            print(f"    extract={d.extracted!r}  target={d.target!r}")
        print()

    print(f"[verify] GATE: {'PASS' if ok else 'FAIL'} "
          f"(think arm reasons, answers isolated, extraction reproduces every score)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
