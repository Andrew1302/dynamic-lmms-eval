"""Verify run_info tabs: present as first sheet, size-range table populated,
and node/edge ranges match an independent recompute from the assembled jsonls."""
import sys, json, re
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POSTPROCESS))
import openpyxl
from _logs import iter_rows

ASM = Path(sys.argv[1])
FULL = REPO / "remote_results" / "_full_reports"

# report -> (family assembled dir, RANGE job template, difficulties or None for sweep)
ABLATION = {
    "full_standard_latest.xlsx": ("standard", "graph_bench_standard_%s_internvl35_4b"),
    "full_adjlist_latest.xlsx": ("adjlist", "graph_bench_ablation_adjlist_%s_internvl35_4b"),
    "full_color_latest.xlsx": ("color", "graph_bench_ablation_color_%s_internvl35_4b"),
    "full_labels_letters_latest.xlsx": ("labels", "graph_bench_ablation_labels_letters_%s_internvl35_4b"),
    "full_labels_none_latest.xlsx": ("labels", "graph_bench_ablation_labels_none_%s_internvl35_4b"),
    "full_think_latest.xlsx": ("think", "graph_bench_think_think_%s_internvl35_4b"),
}
DIFFS = ["easy", "medium", "hard"]
TASKS = ["coloring", "directed_connectivity", "shortest_path"]


def recompute(family, jobtmpl):
    """(task,diff) -> [nvmin,nvmax,nemin,nemax,n] from assembled direct rows."""
    out = {}
    for diff in DIFFS:
        job = jobtmpl % diff
        root = ASM / family / job
        paths = list(root.rglob("*_samples_*.jsonl"))
        for r in iter_rows(paths, job_label=job, dataset_root=root):
            if r.variant != "direct":
                continue
            k = (r.base_task, diff)
            a = out.setdefault(k, [10**9, -1, 10**9, -1, 0])
            a[0] = min(a[0], r.n_vertices); a[1] = max(a[1], r.n_vertices)
            a[2] = min(a[2], r.n_edges); a[3] = max(a[3], r.n_edges); a[4] += 1
    return out


def report_ranges(xlsx):
    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames[0] == "run_info", "run_info not first sheet in %s" % xlsx.name
    ws = wb["run_info"]
    rows = list(ws.iter_rows(values_only=True))
    out = {}
    grab = False
    for v in rows:
        vals = [c for c in v]
        if vals and vals[0] == "task":
            grab = True; continue
        if grab:
            if not vals or vals[0] in (None, "") or (isinstance(vals[0], str) and vals[0].startswith("Total")):
                grab = False; continue
            task, diff, nvmn, nvmx, nemn, nemx, ng = vals[:7]
            out[(task, diff)] = [nvmn, nvmx, nemn, nemx, ng]
    return out


fails = 0
checked = 0
allranges = {}
for report, (family, jobtmpl) in ABLATION.items():
    xlsx = FULL / report
    rep = report_ranges(xlsx)
    rec = recompute(family, jobtmpl)
    nrows = len(rep)
    print("%-38s size-rows=%d" % (report, nrows))
    if nrows == 0:
        print("  FAIL empty size table"); fails += 1; continue
    for k, rv in rep.items():
        checked += 1
        gv = rec.get(k)
        if gv is None:
            print("  FAIL %s: in report but not recomputed" % (k,)); fails += 1; continue
        if [int(x) for x in rv] != gv:
            print("  FAIL %s: report=%s recompute=%s" % (k, rv, gv)); fails += 1
        allranges.setdefault(k, rv)
        # consistency: same (task,diff) range across families
        if allranges[k] != rv:
            print("  WARN %s differs across families: %s vs %s" % (k, allranges[k], rv))

print("\nsize-cells checked: %d  fails: %d" % (checked, fails))
print("RESULT:", "ALL RUN_INFO RANGES MATCH" if fails == 0 else "MISMATCH")
