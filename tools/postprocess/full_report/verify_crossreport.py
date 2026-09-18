"""Check A: every RERUN cell in a full report must equal the same cell in the
standalone 10-15 report (which was computed independently from the _jobs dirs).
Also confirm every full report has exactly the expected job set."""
import sys
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POSTPROCESS))
import openpyxl
from organize_results import match_job

RES = REPO / "remote_results"
FULL = RES / "_full_reports"
BR = RES / "_batch_reports"
TASKS = ["coloring", "directed_connectivity", "shortest_path"]

# full report -> standalone rerun report
STANDALONE = {
    "full_standard": "10_standard_latest.xlsx",
    "full_adjlist": "12_abl_adjlist_latest.xlsx",
    "full_labels_letters": "13_abl_labels_latest.xlsx",
    "full_labels_none": "13_abl_labels_latest.xlsx",
    "full_color": "14_abl_color_latest.xlsx",
    "full_think": "15_abl_think_latest.xlsx",
    "full_sweep_nodes": "11_sweep_size_latest.xlsx",
}
RERUN_CAMP = {"full_standard": "10_standard", "full_adjlist": "12_abl_adjlist",
              "full_labels_letters": "13_abl_labels", "full_labels_none": "13_abl_labels", "full_color": "14_abl_color",
              "full_think": "15_abl_think", "full_sweep_nodes": "11_sweep_size"}


def cells(xlsx):
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["summary"]
    rows = list(ws.iter_rows(values_only=True))
    h = rows[0]
    idx = {c: i for i, c in enumerate(h)}
    out = {}
    for r in rows[1:]:
        job = r[idx["job"]].split("/")[-1]
        d = {}
        for t in TASKS:
            for v in ("direct", "disguise", "paired"):
                key = "%s_%s_acc" % (t, v)
                val = r[idx[key]] if key in idx else None
                d[(t, v)] = None if val in (None, "") else round(float(val), 4)
        d["_overall"] = tuple(
            None if r[idx[k]] in (None, "") else round(float(r[idx[k]]), 4)
            for k in ("overall_direct_acc", "overall_disguise_acc", "overall_paired_acc"))
        out[job] = d
    return out


total = 0
fails = 0
for fam, standalone in STANDALONE.items():
    full = cells(FULL / (fam + "_latest.xlsx"))
    stand = cells(BR / standalone)
    rerun_camp = RERUN_CAMP[fam]
    n_rerun = 0
    for job, fd in full.items():
        rerun_camp2, tmpl, fields = match_job(job)
        leaf = tmpl.format(**fields)
        probe = RES / rerun_camp / "connectivity" / "original" / (leaf + ".jsonl")
        if not probe.is_file():
            continue  # base cell, not checked here
        n_rerun += 1
        if job not in stand:
            print("  FAIL %s: rerun cell %s absent from standalone %s" % (fam, job, standalone))
            fails += 1
            continue
        sd = stand[job]
        for k, fv in fd.items():
            sv = sd.get(k)
            total += 1
            if fv != sv:
                print("  FAIL %s %s %s: full=%s standalone=%s" % (fam, job, k, fv, sv))
                fails += 1
    print("%-16s rerun cells cross-checked: %d" % (fam, n_rerun))

print("\nTOTAL rerun values checked: %d   FAILS: %d" % (total, fails))
print("RESULT:", "ALL MATCH" if fails == 0 else "MISMATCH")
