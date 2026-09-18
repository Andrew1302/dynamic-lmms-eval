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

# Sibling module: this file's directory is on sys.path both when run as a
# script and when imported by the callers that insert tools/postprocess.
import _families

# --------------------------------------------------------------------------
# Campaign spec: ordered (regex on bare job name) -> (campaign dir, leaf tmpl).
# Bare name = job dir name with the leading "graph_bench_" stripped. First
# match wins, so list the more specific prefixes first.
# --------------------------------------------------------------------------
MODELS = _families.MODELS

_DIFF = r"(?P<diff>easy|medium|hard)"
_MODEL = r"(?P<model>[a-z0-9]+_[a-z0-9]+)"
_COL = r"(?:(?P<col>coloring)_)?"  # optional coloring-only job marker

# Campaigns 01-09 are the PRE-FIX tree (render bug / fp8 blindness / default-chi
# coloring); they stay on disk untouched as the archive. Everything fetched from
# 2026-07-16 on files into the fresh 10+ tree so old and new results never mix.
# SPECS and SCRATCH_RE moved to _families.py so the filing rules and the
# report's axis grouping cannot drift apart. Re-exported for importers.
SPECS = [(f.pattern, f.campaign, f.leaf_tmpl) for f in _families.FAMILIES]

SCRATCH_RE = _families.SCRATCH_RE

TASK_FOLDER = {
    "directed_connectivity": "connectivity",
    "shortest_path": "shortest_path",
    "coloring": "coloring",
}
VARIANT_FOLDER = {"direct": "original", "disguise": "disguise"}

_LEGACY_NOTE = ("PRE-FIX LEGACY ARCHIVE (superseded 2026-07-16 by the 10+ tree: "
                "direct-render occlusion bug, fp8 vision blindness, default-chi "
                "coloring). Do not mix with 10+ results. ")
_LEGACY_CAMPAIGN_DESC = {
    # ---- legacy pre-fix tree (01-09): archive only, nothing new files here
    # (except 03, which still catches refetches of the retired coloring jobs).
    "01_standard": _LEGACY_NOTE + "Baseline benchmark, n=500/task.",
    "02_sweep_size": _LEGACY_NOTE + "Graph-size scaling sweeps.",
    "03_coloring_chi": _LEGACY_NOTE + "Dedicated special-chi coloring jobs (family retired: chi-control is the prepare default and coloring rides in every base job).",
    "04_abl_adjlist": _LEGACY_NOTE + "Adjacency-list ablation.",
    "05_abl_labels": _LEGACY_NOTE + "Label-style ablation.",
    "06_abl_color": _LEGACY_NOTE + "Node-colour ablation.",
    "07_abl_think": _LEGACY_NOTE + "Thinking on/off ablation.",
    "08_abl_thinkadj": _LEGACY_NOTE + "Thinking x adjacency-list ablation.",
    "09_abl_scram": _LEGACY_NOTE + "Scrambled-image control.",
    # ---- post-fix rerun tree (10+): occlusion-aware renders, bf16 vision under
    # fp8, chi-controlled coloring in every base job (all three tasks per job).
    "10_standard": "Baseline benchmark v2, n=500/task, 3 difficulties x 3 models. All three tasks (coloring chi-controlled), original + disguise.",
    "11_sweep_size": "Graph-size scaling sweeps v2 (edges & nodes), all three tasks per job; leaf = <model>_<constraint>.",
    "12_abl_adjlist": "Adjacency list injected into the prompt. n=100/task, 3 difficulties x 3 models, all three tasks.",
    "13_abl_labels": "Node-label style ablation (letters / none vs the numeric baseline). n=100/task, all three tasks.",
    "14_abl_color": "Node fill-colour ablation. n=100/task, 3 difficulties x 3 models, all three tasks.",
    "15_abl_think": "Thinking on/off ablation (image-only prompt). Both arms, n=100/task, all three tasks.",
    "16_abl_thinkadj": "Thinking x adjacency-list ablation. Both arms, n=100/task, all three tasks; _inc marks the deferred 400-sample increment (pool for n=500).",
    "17_abl_scram": "Scrambled ('no-info') image control + adjacency list. Both arms, 3 difficulties; Qwen + InternVL required, Gemma optional.",
}

# Post-fix campaigns describe themselves in the family registry, so a new
# family needs no second entry here. The legacy 01-09 archive has no families.
CAMPAIGN_DESC = {**_LEGACY_CAMPAIGN_DESC, **{f.campaign: f.description for f in _families.FAMILIES if f.description}}

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

    Thin adapter over the ``_families`` registry, which is the single source of
    truth shared with batch_report.py and the job generator. The return shape is
    kept for callers (the generator's drift guard unpacks it).
    ``job_name`` may include or omit the ``graph_bench_`` prefix.
    """
    matched = _families.match_job(job_name)
    if matched is None:
        return None
    family, fields = matched
    return family.campaign, family.leaf_tmpl, fields


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


def _meta_special_coloring(job_dir: Path) -> bool:
    """prepare_meta.json authority for merged-era jobs (2026-07-16+): chi-
    controlled coloring became the prepare DEFAULT, so base all-task jobs now
    emit special-chi coloring under the same names that used to emit
    default-chi. The fetched dataset's meta (special_coloring: true/false)
    is the only era-proof discriminator."""
    for meta in job_dir.glob("dataset_*/prepare_meta.json"):
        try:
            return bool(json.loads(
                meta.read_text(encoding="utf-8")).get("special_coloring"))
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    return False


def build_leaves(jp: JobPlan) -> None:
    # Special-chi coloring: a dedicated coloring job carries the '_coloring_'
    # name marker (fields['col']) and every 03_coloring_chi job is special by
    # construction — but since 2026-07-16 base all-task jobs are special too
    # (chi-control is the prepare default), so fall back to the fetched
    # dataset's prepare_meta.json when the name says nothing.
    special = (jp.fields.get("col") == "coloring"
               or jp.campaign == "03_coloring_chi"
               or _meta_special_coloring(jp.job_dir))
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
    special = (fields.get("col") == "coloring" or campaign == "03_coloring_chi"
               or _meta_special_coloring(job_dir))
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
