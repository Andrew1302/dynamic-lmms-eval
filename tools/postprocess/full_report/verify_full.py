"""Independently recompute every cell of each full-results report from the
organizer's AUTHORITATIVE per-task jsonls (winner = rerun leaf if present else
base), using the canonical build_pair_rows metric fn, and diff against the
report. Any mismatch > 1e-9 is a FAIL.
"""
import os
import sys
import glob
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]

TOOLS = REPO / "tools" / "postprocess"
sys.path.insert(0, str(TOOLS))
import openpyxl
from organize_results import match_job
from compare_direct_disguise import build_pair_rows, load_jsonl

RES = REPO / "remote_results"
FULL = RES / "_full_reports"
RERUN2BASE = {"13_abl_labels": "05_abl_labels", "14_abl_color": "06_abl_color",
              "15_abl_think": "07_abl_think", "10_standard": "01_standard",
              "12_abl_adjlist": "04_abl_adjlist", "11_sweep_size": "02_sweep_size"}
TASK_FOLDER = {"coloring": "coloring", "directed_connectivity": "connectivity",
               "shortest_path": "shortest_path"}


BASE_COLORING = {"01_standard": "03_coloring_chi"}


def recompute(winner, leaf):
    out = {}
    for task, folder in TASK_FOLDER.items():
        src_camp = winner
        if task == "coloring" and winner in BASE_COLORING:
            src_camp = BASE_COLORING[winner]
        d = RES / src_camp / folder / "original" / (leaf + ".jsonl")
        g = RES / src_camp / folder / "disguise" / (leaf + ".jsonl")
        if not d.is_file() or not g.is_file():
            continue
        direct = load_jsonl(d)
        disguise = load_jsonl(g)
        _, t = build_pair_rows(direct, disguise)
        out[task] = (
            round(t["direct_correct"] / (t["n_direct"] or 1), 4),
            round(t["disguise_correct"] / (t["n_disguise"] or 1), 4),
            round(t["paired_correct"] / (t["n_paired"] or 1), 4),
        )
    return out


def report_cells(xlsx):
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["summary"]
    rows = list(ws.iter_rows(values_only=True))
    h = rows[0]
    idx = {c: i for i, c in enumerate(h)}
    cells = {}
    for r in rows[1:]:
        job = r[idx["job"]].split("/")[-1]
        d = {}
        for task in TASK_FOLDER:
            trip = tuple(r[idx["%s_%s_acc" % (task, v)]] for v in ("direct", "disguise", "paired"))
            d[task] = trip
        cells[job] = d
    return cells


FAMS = {"full_standard": None, "full_adjlist": None,
        "full_color": None, "full_think": None, "full_sweep_nodes": None, "full_labels_letters": None, "full_labels_none": None}

total_cells = 0
total_fail = 0
for fam in FAMS:
    xlsx = FULL / (fam + "_latest.xlsx")
    if not xlsx.is_file():
        print("MISSING report:", xlsx)
        continue
    cells = report_cells(xlsx)
    print("\n===== %s : %d jobs =====" % (fam, len(cells)))
    fails = 0
    for job, rep in cells.items():
        m = match_job(job)
        rerun_camp, tmpl, fields = m
        leaf = tmpl.format(**fields)
        base_camp = RERUN2BASE[rerun_camp]
        probe = RES / rerun_camp / "connectivity" / "original" / (leaf + ".jsonl")
        winner = rerun_camp if probe.is_file() else base_camp
        recomp = recompute(winner, leaf)
        for task, rep_trip in rep.items():
            got = recomp.get(task)
            total_cells += 1
            if got is None:
                # report has a value but source missing -> only fail if report non-empty
                if any(x not in (None, "") for x in rep_trip):
                    print("  FAIL %s %s: report has %s but no source" % (job, task, rep_trip))
                    fails += 1
                    total_fail += 1
                continue
            for k, (a, b) in enumerate(zip(rep_trip, got)):
                av = 0.0 if a in (None, "") else float(a)
                if abs(av - b) > 1e-9:
                    print("  FAIL %s %s[%d]: report=%s recomputed=%s (winner=%s)"
                          % (job, task, k, a, b, winner))
                    fails += 1
                    total_fail += 1
    print("  cell-triples checked, fails in family: %d" % fails)

print("\n########################################")
print("TOTAL task-cells checked: %d   FAILS: %d" % (total_cells, total_fail))
print("RESULT:", "ALL MATCH" if total_fail == 0 else "MISMATCHES FOUND")
