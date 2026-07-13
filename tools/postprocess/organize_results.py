#!/usr/bin/env python3
"""Reorganize fetched lmms-eval graph-benchmark runs into a numbered,
task-first folder hierarchy.

Target layout (under ``remote_results/``)::

    NN_campaign/
      <task>/<original|disguise>/<leaf>.jsonl   # authoritative merged samples
      _jobs/<original-job-dir>/...              # full run moved intact (dataset+images+logs)
      README.md
    _scratch/<smoke-or-superseded-job>/...      # non-campaign runs, kept not deleted

Where:
  * task folders are ``connectivity`` / ``shortest_path`` / ``coloring``
  * ``original`` == the ``_direct`` task variant, ``disguise`` == ``_disguise``
  * leaf name encodes model / difficulty / arm (the hierarchy carries the rest)

The script is idempotent and reversible: every move is recorded in a manifest
under ``remote_results/_migration/``. Run with ``--apply`` to execute; the
default is a dry run that only prints the plan.

Design notes:
  * The authoritative result for a (task, variant) is the newest ``*_samples_*``
    jsonl NOT under a ``chunks/`` path (mirrors tools/postprocess/_logs.py).
  * When two jobs map to the same destination leaf (e.g. a re-run in
    ``archive/new`` and another at top level), the newest merged jsonl wins and
    the loser's whole job dir is parked in ``_scratch/_superseded/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Campaign spec: ordered (regex on bare job name) -> (campaign dir, leaf tmpl).
# Bare name = job dir name with the leading "graph_bench_" stripped. First
# match wins, so list the more specific prefixes first.
# --------------------------------------------------------------------------
MODELS = {
    "internvl35_4b": "internvl",
    "qwen35_4b": "qwen",
    "gemma4_e2b": "gemma",
}

_DIFF = r"(?P<diff>easy|medium|hard)"
_MODEL = r"(?P<model>[a-z0-9]+_[a-z0-9]+)"
_COL = r"(?:(?P<col>coloring)_)?"  # optional coloring-only job marker

SPECS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(rf"^scram_(?P<arm>think|nothink)_{_COL}{_DIFF}_{_MODEL}$"),
     "09_abl_scram", "{model}_{diff}_{arm}"),
    (re.compile(rf"^thinkadj500inc_(?P<arm>think|nothink)_{_COL}{_DIFF}_{_MODEL}$"),
     "08_abl_thinkadj", "{model}_{diff}_{arm}_inc"),
    (re.compile(rf"^thinkadj_(?P<arm>think|nothink)_{_COL}{_DIFF}_{_MODEL}$"),
     "08_abl_thinkadj", "{model}_{diff}_{arm}"),
    (re.compile(rf"^think_(?P<arm>think|nothink)_{_COL}{_DIFF}_{_MODEL}$"),
     "07_abl_think", "{model}_{diff}_{arm}"),
    (re.compile(rf"^ablation_adjlist_{_COL}{_DIFF}_{_MODEL}$"),
     "04_abl_adjlist", "{model}_{diff}"),
    (re.compile(rf"^ablation_labels_(?P<style>letters|none)_{_COL}{_DIFF}_{_MODEL}$"),
     "05_abl_labels", "{model}_{diff}_{style}"),
    (re.compile(rf"^ablation_color_{_COL}{_DIFF}_{_MODEL}$"),
     "06_abl_color", "{model}_{diff}"),
    (re.compile(rf"^coloring_{_DIFF}_{_MODEL}$"),
     "03_coloring_chi", "{model}_{diff}"),
    (re.compile(rf"^standard_{_DIFF}_{_MODEL}$"),
     "01_standard", "{model}_{diff}"),
    (re.compile(rf"^sweep_(?P<constraint>edges|nodes)_(?P<col>coloring)_{_MODEL}$"),
     "02_sweep_size", "{model}_{constraint}"),
    (re.compile(rf"^sweep_(?P<constraint>edges|nodes)_{_MODEL}$"),
     "02_sweep_size", "{model}_{constraint}"),
]

SCRATCH_RE = re.compile(r"smoke|promptexp|fp8think|v1ab|gemmatok")

TASK_FOLDER = {
    "directed_connectivity": "connectivity",
    "shortest_path": "shortest_path",
    "coloring": "coloring",
}
VARIANT_FOLDER = {"direct": "original", "disguise": "disguise"}

CAMPAIGN_DESC = {
    "01_standard": "Baseline benchmark, n=500/task, 3 difficulties x 3 models. All three tasks, original + disguise.",
    "02_sweep_size": "Graph-size scaling sweeps (edges & nodes; *_chi = special-coloring sweep). Difficulty replaced by the swept size; leaf = <model>_<constraint>.",
    "03_coloring_chi": "Coloring re-run with planted chromatic number chi in {2,3,4}, n=500/task, 3 difficulties x 3 models.",
    "04_abl_adjlist": "Adjacency list injected into the prompt. n=100/task, 3 difficulties x 3 models.",
    "05_abl_labels": "Node-label style ablation (letters / none vs the numeric baseline). n=100/task.",
    "06_abl_color": "Node fill-colour ablation. n=100/task, 3 difficulties x 3 models.",
    "07_abl_think": "Thinking on/off ablation (image-only prompt). Both arms, n=100/task.",
    "08_abl_thinkadj": "Thinking x adjacency-list ablation. Both arms. n=500 = base (n=100) pooled with _inc (n=400); InternVL base-only. leaf arm in {think,nothink}, _inc marks the 400-sample increment.",
    "09_abl_scram": "Scrambled ('no-info') image control. Qwen only, both arms, medium+hard.",
}

SAMPLES_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_samples_(?P<task>.+)\.jsonl$")


@dataclass
class Leaf:
    task_folder: str
    variant_folder: str
    leaf_name: str
    src_jsonl: Path
    mtime: float
    is_coloring: bool = False
    special_coloring: bool = False


@dataclass
class JobPlan:
    job_dir: Path
    bare: str
    campaign: str
    leaf_tmpl: str
    fields: dict
    leaves: list[Leaf] = field(default_factory=list)
    scratch_reason: str = ""  # non-empty => goes to _scratch


def match_job(job_name: str):
    """Classify a job by name -> (campaign, leaf_tmpl, fields) or None.

    Single source of truth shared with the generator and the image linker.
    ``job_name`` may include or omit the ``graph_bench_`` prefix.
    """
    bare = job_name[len("graph_bench_"):] if job_name.startswith("graph_bench_") else job_name
    if SCRATCH_RE.search(bare):
        return None
    for rx, campaign, tmpl in SPECS:
        m = rx.match(bare)
        if m and m.groupdict().get("model", "") in MODELS:
            fields = dict(m.groupdict())
            fields["model"] = MODELS[fields["model"]]
            return campaign, tmpl, fields
    return None


def campaign_for_name(job_name: str) -> str:
    """Return the ``NN_campaign`` a job belongs to, or "" for scratch/unknown."""
    m = match_job(job_name)
    return m[0] if m else ""


def _base_variant(task_full: str) -> tuple[str, str]:
    prefix = "dynamic_graph_benchmark_"
    n = task_full[len(prefix):] if task_full.startswith(prefix) else task_full
    if n.endswith("_direct"):
        return n[:-7], "direct"
    if n.endswith("_disguise"):
        return n[:-9], "disguise"
    return n, "unknown"


def authoritative_jsonls(job_dir: Path) -> dict[tuple[str, str], Path]:
    """Newest non-chunk merged samples file per (base_task, variant)."""
    best: dict[tuple[str, str], tuple[float, Path]] = {}
    for p in job_dir.rglob("*_samples_*.jsonl"):
        if "chunks" in p.parts:
            continue
        m = SAMPLES_RE.match(p.name)
        if not m:
            continue
        base, variant = _base_variant(m.group("task"))
        key = (base, variant)
        mt = p.stat().st_mtime
        if key not in best or mt > best[key][0]:
            best[key] = (mt, p)
    return {k: v[1] for k, v in best.items()}


def classify(job_dir: Path) -> JobPlan | None:
    name = job_dir.name
    if not name.startswith("graph_bench_"):
        return None
    bare = name[len("graph_bench_"):]
    if SCRATCH_RE.search(bare):
        return JobPlan(job_dir, bare, "", "", {}, scratch_reason="smoke/experimental")
    for rx, campaign, tmpl in SPECS:
        m = rx.match(bare)
        if not m:
            continue
        gd = m.groupdict()
        model_raw = gd.get("model", "")
        if model_raw not in MODELS:
            # unknown model token (e.g. legacy qwen3vl) -> leave for review
            return JobPlan(job_dir, bare, "", "", {}, scratch_reason=f"unknown-model:{model_raw}")
        fields = dict(gd)
        fields["model"] = MODELS[model_raw]
        return JobPlan(job_dir, bare, campaign, tmpl, fields)
    # graph_bench_* that matched nothing
    return JobPlan(job_dir, bare, "", "", {}, scratch_reason="UNMATCHED")


def build_leaves(jp: JobPlan) -> None:
    # Special-chi coloring is identified reliably from the job NAME: a dedicated
    # coloring job carries the '_coloring_' marker (fields['col']), and every
    # 03_coloring_chi job is special by construction. Base all-task jobs emit
    # default-chi coloring. (run.log's 'special_coloring=' marker is missing on
    # older runs, so name is authoritative.)
    special = jp.fields.get("col") == "coloring" or jp.campaign == "03_coloring_chi"
    for (base, variant), src in authoritative_jsonls(jp.job_dir).items():
        tf = TASK_FOLDER.get(base)
        vf = VARIANT_FOLDER.get(variant)
        if tf is None or vf is None:
            continue
        leaf = jp.leaf_tmpl.format(**jp.fields)
        is_col = base == "coloring"
        jp.leaves.append(Leaf(tf, vf, leaf, src, src.stat().st_mtime,
                              is_coloring=is_col,
                              special_coloring=is_col and special))


def route_default_coloring(plans: list[JobPlan]) -> None:
    """In a campaign that has a special-chi coloring result, any *default*-chi
    coloring leaf (a leftover from an all-tasks base job) is routed to a
    ``coloring_default/`` folder so it never collides with the headline
    special-chi coloring under ``coloring/``."""
    has_special: dict[str, bool] = {}
    for jp in plans:
        if jp.scratch_reason:
            continue
        for lf in jp.leaves:
            if lf.special_coloring:
                has_special[jp.campaign] = True
    for jp in plans:
        if jp.scratch_reason:
            continue
        if not has_special.get(jp.campaign):
            continue
        for lf in jp.leaves:
            if lf.is_coloring and not lf.special_coloring:
                lf.task_folder = "coloring_default"


def scan(rr: Path) -> list[JobPlan]:
    roots: list[Path] = []
    # top-level dirs
    for d in rr.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(("_", ".")) or re.match(r"^\d\d_", d.name):
            continue
        if d.name == "archive":
            continue
        roots.append(d)
    # archive/new (curated standard/coloring/sweep)
    an = rr / "archive" / "new"
    if an.is_dir():
        roots += [d for d in an.iterdir() if d.is_dir() and d.name.startswith("graph_bench_")]

    plans: list[JobPlan] = []
    for d in roots:
        jp = classify(d)
        if jp is None:
            continue
        if not jp.scratch_reason:
            build_leaves(jp)
        plans.append(jp)
    return plans


def resolve_collisions(plans: list[JobPlan]) -> list[dict]:
    """Newest merged-jsonl wins a leaf; losers' whole job dir -> _scratch.

    Collision key = (campaign, task_folder, variant_folder, leaf_name).
    """
    owners: dict[tuple, tuple[float, JobPlan]] = {}
    for jp in plans:
        if jp.scratch_reason:
            continue
        for lf in jp.leaves:
            key = (jp.campaign, lf.task_folder, lf.variant_folder, lf.leaf_name)
            cur = owners.get(key)
            if cur is None or lf.mtime > cur[0]:
                owners[key] = (lf.mtime, jp)

    winners = {id(jp) for _, jp in owners.values()}
    collisions = []
    for jp in plans:
        if jp.scratch_reason:
            continue
        if id(jp) not in winners:
            jp.scratch_reason = "superseded"
            collisions.append({"job": jp.job_dir.name, "campaign": jp.campaign})
    return collisions


def plan_summary(plans: list[JobPlan]) -> None:
    by_campaign: dict[str, list[JobPlan]] = {}
    scratch: list[JobPlan] = []
    unmatched: list[JobPlan] = []
    for jp in plans:
        if jp.scratch_reason == "UNMATCHED" or jp.scratch_reason.startswith("unknown-model"):
            unmatched.append(jp)
        elif jp.scratch_reason:
            scratch.append(jp)
        else:
            by_campaign.setdefault(jp.campaign, []).append(jp)

    print("=" * 70)
    print("CAMPAIGNS")
    print("=" * 70)
    for camp in sorted(by_campaign):
        jps = by_campaign[camp]
        nleaf = sum(len(j.leaves) for j in jps)
        print(f"\n{camp}   ({len(jps)} jobs, {nleaf} result files)")
        # unique leaves per task/variant
        tv: dict[tuple[str, str], list[str]] = {}
        for j in jps:
            for lf in j.leaves:
                tv.setdefault((lf.task_folder, lf.variant_folder), []).append(lf.leaf_name)
        for (tf, vf) in sorted(tv):
            names = sorted(set(tv[(tf, vf)]))
            shown = ", ".join(names[:6]) + (" ..." if len(names) > 6 else "")
            print(f"    {tf}/{vf}: {len(names)}  [{shown}]")

    print("\n" + "=" * 70)
    print(f"SCRATCH (-> _scratch/): {len(scratch)} jobs "
          "(smoke/experimental + superseded re-runs)")
    print("=" * 70)
    for jp in sorted(scratch, key=lambda j: j.job_dir.name):
        print(f"    {jp.job_dir.name}   [{jp.scratch_reason}]")

    if unmatched:
        print("\n" + "!" * 70)
        print(f"UNMATCHED / NEEDS REVIEW: {len(unmatched)}")
        print("!" * 70)
        for jp in unmatched:
            print(f"    {jp.job_dir.name}   [{jp.scratch_reason}]")


def apply(plans: list[JobPlan], rr: Path) -> list[Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"stamp": stamp, "moves": [], "copies": []}
    scratch_root = rr / "_scratch"
    superseded_root = scratch_root / "_superseded"
    filed: list[Path] = []  # _jobs/<job> dests of the runs we just filed

    for jp in plans:
        if jp.scratch_reason:
            dest_root = superseded_root if jp.scratch_reason == "superseded" else scratch_root
            dest_root.mkdir(parents=True, exist_ok=True)
            dest = dest_root / jp.job_dir.name
            if dest.exists():
                shutil.rmtree(str(dest))
            shutil.move(str(jp.job_dir), str(dest))
            manifest["moves"].append([str(jp.job_dir), str(dest)])
            continue

        camp_dir = rr / jp.campaign
        # copy leaves first (job dir still in place); overwrite on re-fetch.
        for lf in jp.leaves:
            out = camp_dir / lf.task_folder / lf.variant_folder
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"{lf.leaf_name}.jsonl"
            shutil.copy2(str(lf.src_jsonl), str(dst))
            manifest["copies"].append(str(dst))
        # move whole job dir into _jobs/ (replace on re-fetch so a fresher,
        # more-complete run supersedes the previously filed one).
        jobs_dir = camp_dir / "_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        dest = jobs_dir / jp.job_dir.name
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.move(str(jp.job_dir), str(dest))
        manifest["moves"].append([str(jp.job_dir), str(dest)])
        filed.append(dest)

    # per-campaign README
    for camp, desc in CAMPAIGN_DESC.items():
        cdir = rr / camp
        if not cdir.is_dir():
            continue
        readme = cdir / "README.md"
        readme.write_text(
            f"# {camp}\n\n{desc}\n\n"
            "Layout: `<task>/<original|disguise>/<model>_<difficulty>[_<arm/variant>].jsonl`\n"
            "`original` = the `_direct` task variant, `disguise` = `_disguise`.\n"
            "Full runs (dataset + images + logs) are preserved under `_jobs/`.\n",
            encoding="utf-8",
        )

    mdir = rr / "_migration"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"manifest_{stamp}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nAPPLIED. {len(manifest['moves'])} dirs moved, "
          f"{len(manifest['copies'])} result files placed.")
    print(f"Manifest: {mdir / f'manifest_{stamp}.json'}")
    return filed


_IMG_RE = re.compile(r"^(?P<base>.+?)_(?P<variant>direct|disguise)_(?P<idx>\d+)\.png$")


def _link_one_job(camp_dir: Path, job_dir: Path) -> int:
    """Hardlink one filed job's PNGs into its leaf-parallel .images/ folders."""
    m = match_job(job_dir.name)
    if not m:
        return 0
    campaign, tmpl, fields = m
    leaf = tmpl.format(**fields)
    special = fields.get("col") == "coloring" or campaign == "03_coloring_chi"
    has_default = (camp_dir / "coloring_default").exists()
    made = 0
    for png in job_dir.rglob("*.png"):
        im = _IMG_RE.match(png.name)
        if not im:
            continue
        base = im.group("base")
        if base == "coloring":
            tf = "coloring" if special else ("coloring_default" if has_default else "coloring")
        else:
            tf = TASK_FOLDER.get(base)
        vf = VARIANT_FOLDER.get(im.group("variant"))
        if tf is None or vf is None:
            continue
        out = camp_dir / tf / vf / f"{leaf}.images"
        out.mkdir(parents=True, exist_ok=True)
        dst = out / f"{im.group('idx')}.png"
        if dst.exists():
            continue
        try:
            os.link(str(png), str(dst))
        except OSError:
            shutil.copy2(str(png), str(dst))  # cross-volume fallback
        made += 1
    return made


def link_images(rr: Path, quiet: bool = False, only: list[Path] | None = None) -> int:
    """Expose each filed job's exported PNGs next to the matching result leaf:
    ``NN/<task>/<original|disguise>/<leaf>.images/<idx>.png`` (hardlinked, so no
    extra disk on the same volume). Idempotent. Returns the number of links made.

    ``only`` = specific ``_jobs/<job>`` dirs to (re)link (used by the post-fetch
    hook so a single fetch doesn't rescan the whole tree). Default: all filed jobs.
    """
    made = 0
    if only is not None:
        for job_dir in only:
            camp_dir = job_dir.parent.parent  # NN_campaign/_jobs/<job>
            made += _link_one_job(camp_dir, job_dir)
    else:
        for camp_dir in sorted(p for p in rr.iterdir()
                               if p.is_dir() and re.match(r"^\d\d_", p.name)):
            jobs_dir = camp_dir / "_jobs"
            if not jobs_dir.is_dir():
                continue
            for job_dir in sorted(jobs_dir.iterdir()):
                if job_dir.is_dir():
                    made += _link_one_job(camp_dir, job_dir)
    if not quiet:
        print(f"linked {made} images into <leaf>.images/ folders")
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="remote_results", help="results root")
    ap.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the plan dump (for the post-fetch hook)")
    ap.add_argument("--link-images", action="store_true",
                    help="(re)build <leaf>.images/ hardlink folders for filed jobs and exit")
    args = ap.parse_args()

    rr = Path(args.root).resolve()

    if args.link_images:
        link_images(rr, quiet=args.quiet)
        return

    plans = scan(rr)
    route_default_coloring(plans)
    resolve_collisions(plans)
    # scan() only returns still-flat (unorganized) job dirs, so an empty result
    # means everything is already filed — stay silent for the post-fetch hook.
    if not plans:
        if not args.quiet:
            print("nothing to organize (all runs already filed)")
        return
    if not args.quiet:
        plan_summary(plans)
    if args.apply:
        filed = apply(plans, rr)
        # Link only the just-filed jobs so a single fetch stays fast.
        link_images(rr, quiet=args.quiet, only=filed)
    elif not args.quiet:
        print("\n(dry run — pass --apply to execute)")


if __name__ == "__main__":
    main()
