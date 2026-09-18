"""Assemble synthetic per-job dirs for the labels/color/think 'full results'
merge, sourcing each cell's 6 task jsonls from the organizer's AUTHORITATIVE
per-task folders (rerun campaign leaf if present, else base campaign leaf).

No base tree mutation: files are COPIED verbatim into a scratch tree; metrics
are computed later by the standard batch_report.py. Emits a jobs TSV.
"""
import os
import sys
import shutil
import glob
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]

TOOLS = REPO / "tools" / "postprocess"
sys.path.insert(0, str(TOOLS))
from organize_results import match_job  # noqa: E402

RES = REPO / "remote_results"
JOBS_CONF = REPO / "remote_execution_scripts" / "jobs" / "graph_benchmark"

# rerun campaign (from match_job SPECS) -> base campaign
RERUN2BASE = {"13_abl_labels": "05_abl_labels", "14_abl_color": "06_abl_color",
              "15_abl_think": "07_abl_think", "10_standard": "01_standard",
              "12_abl_adjlist": "04_abl_adjlist", "11_sweep_size": "02_sweep_size",
              "16_abl_thinkadj": "08_abl_thinkadj"}

# WINNER_MODE: "merged" (default) = rerun leaf if present else base; "postfix"
# = rerun only (base never used — a cell with no rerun is an error). Set via the
# WINNER_MODE env var so the same assembler produces both report families.
WINNER_MODE = os.environ.get("WINNER_MODE", "merged")

# Base campaigns whose own coloring/ folder is NOT the chi-controlled (special)
# coloring -> pull coloring for their base cells from here instead. Only
# 01_standard is affected (its coloring/ is the legacy default-chi); every other
# base campaign's coloring/ is already special-chi.
BASE_COLORING = {"01_standard": "03_coloring_chi"}

# task (jsonl label) -> authoritative folder name
TASK_FOLDER = {"coloring": "coloring", "directed_connectivity": "connectivity",
               "shortest_path": "shortest_path"}
VARIANTS = {"direct": "original", "disguise": "disguise"}
TASKS = ["coloring", "directed_connectivity", "shortest_path"]

COMPARE_PAIRS = "|".join(
    "%s:dynamic_graph_benchmark_%s_direct:dynamic_graph_benchmark_%s_disguise" % (t, t, t)
    for t in TASKS)

FIXED_TS = "20260718_000000"


def model_pretrained(job_name):
    conf = os.path.join(JOBS_CONF, job_name + ".conf")
    with open(conf, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("MODEL_PRETRAINED="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v
    return ""


def main():
    manifest = sys.argv[1]
    out_root = sys.argv[2]   # scratch tree root
    tsv_path = sys.argv[3]
    jobs = [l.strip() for l in open(manifest) if l.strip() and not l.startswith("#")]

    rows = []
    provenance = []  # (job, winner_campaign, leaf) for audit
    for job_id in jobs:
        job_name = job_id.split("/")[-1]
        m = match_job(job_name)
        if not m:
            raise SystemExit("match_job failed for %s" % job_name)
        rerun_camp, tmpl, fields = m
        leaf = tmpl.format(**fields)
        base_camp = RERUN2BASE[rerun_camp]

        # winner = rerun if that leaf exists in the rerun campaign, else base.
        rerun_probe = os.path.join(RES, rerun_camp, "connectivity", "original", leaf + ".jsonl")
        rerun_present = os.path.isfile(rerun_probe)
        if WINNER_MODE == "postfix" and not rerun_present:
            raise SystemExit("postfix mode: no rerun leaf for %s (leaf=%s)" % (job_name, leaf))
        winner = rerun_camp if rerun_present else base_camp

        job_dir = os.path.join(out_root, job_name)
        logs_dir = os.path.join(job_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        col_src = None
        for task in TASKS:
            folder = TASK_FOLDER[task]
            # Coloring must be the chi-controlled (special) coloring. Base
            # 01_standard/coloring/ is the OLD default-chi (degenerate chi
            # distribution) — the special-chi standard coloring lives in
            # 03_coloring_chi. Redirect coloring for base-standard cells there.
            src_camp = winner
            if task == "coloring" and winner == base_camp:
                src_camp = BASE_COLORING.get(base_camp, base_camp)
            if task == "coloring":
                col_src = src_camp
            for variant, vfolder in VARIANTS.items():
                src = os.path.join(RES, src_camp, folder, vfolder, leaf + ".jsonl")
                if not os.path.isfile(src):
                    raise SystemExit("MISSING source: %s (job %s)" % (src, job_name))
                dst = os.path.join(
                    logs_dir,
                    "%s_samples_dynamic_graph_benchmark_%s_%s.jsonl" % (FIXED_TS, task, variant))
                shutil.copy2(src, dst)

        constraint = fields.get("constraint", "") or ""
        rows.append((job_id, job_name, job_dir, model_pretrained(job_name), COMPARE_PAIRS, constraint))
        provenance.append((job_name, winner, leaf, col_src))

    with open(tsv_path, "w", encoding="utf-8", newline="") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")

    print("assembled %d jobs -> %s" % (len(rows), out_root))
    for jn, win, leaf, col in provenance:
        extra = ("  coloring<-%s" % col) if col and col != win else ""
        print("  %-56s <- %-16s leaf=%s%s" % (jn, win, leaf, extra))


if __name__ == "__main__":
    main()
