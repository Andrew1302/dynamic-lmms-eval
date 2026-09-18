"""Build BOTH report families for every campaign, from the AUTHORITATIVE per-task
leaf jsonls (ground truth), reusing the shipped batch_report + run_info + metrics.

Two modes (WINNER handled by assemble_full via WINNER_MODE env):
  postfix : rerun (post-fix) leaf only; error if a cell has no post-fix run.
  merged  : rerun leaf if present else base campaign leaf (post-fix wins).

Outputs:
  postfix -> remote_results/<campaign>/report_<campaign>.xlsx
  merged  -> remote_results/_full_reports/merged_<name>_latest.xlsx

Run:  uv run --no-project --with openpyxl --with datasets python build_all_reports.py <mode> <scratch_root>
"""
import os, sys, subprocess, importlib
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]

MODE = sys.argv[1]              # "postfix" | "merged"
ASM = Path(sys.argv[2])         # scratch assembled root
assert MODE in ("postfix", "merged")

REPO = REPO
BUILD = REPO / "remote_results" / "_full_reports" / "_build"
MANI = BUILD / "manifests"
FULL = REPO / "remote_results" / "_full_reports"
TOOLS = REPO / "tools" / "postprocess"
RES = REPO / "remote_results"
sys.path.insert(0, str(TOOLS))
import openpyxl

import _runinfo
from _runinfo_specs import RUN_INFO_SPECS

# ---- families ---------------------------------------------------------------
# name, manifest, campaign, runinfo kind + params, metrics kind, split
FAMS = [
    dict(name="standard",    mani="full_standard.txt",    camp="10_standard",   ri=("std",),           spd=500),
    dict(name="adjlist",     mani="full_adjlist.txt",     camp="12_abl_adjlist",ri=("adj",),           spd=100),
    dict(name="color",       mani="full_color.txt",       camp="14_abl_color",  ri=("abl","color"),    spd=100),
    dict(name="labels",      mani="full_labels.txt",      camp="13_abl_labels", ri=("abl_split",),     spd=100, split=True),
    dict(name="think",       mani="full_think.txt",       camp="15_abl_think",  ri=("thinking",),      metrics="thinking"),
    dict(name="thinkadj",    mani="full_thinkadj.txt",    camp="16_abl_thinkadj",ri=("thinkadj",),     metrics="thinkadj"),
    dict(name="sweep_nodes", mani="full_sweep_nodes.txt", camp="11_sweep_size", ri=("sweep","nodes")),
    dict(name="sweep_edges", mani="full_sweep_edges.txt", camp="11_sweep_size", ri=("sweep","edges")),
]

# injected config text for standard/adjlist (no shipped generator) -------------


def uv(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout + r.stderr

def assemble(mani, out_family, tsv):
    env = dict(os.environ, WINNER_MODE=MODE)
    r = subprocess.run(["uv","run","--no-project","python", str(BUILD/"assemble_full.py"),
                        str(MANI/mani), str(out_family), str(tsv)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"assemble FAILED ({mani}):\n{r.stdout}\n{r.stderr}")
    return r.stdout

def batch_report(tsv, out_xlsx, name):
    r = subprocess.run(["uv","run","--no-project","--with","openpyxl","--with","datasets","python",
                        str(TOOLS/"batch_report.py"), "--jobs-tsv", str(tsv),
                        "--output", str(out_xlsx), "--batch-name", name],
                       capture_output=True, text=True)
    nd = (r.stdout+r.stderr).count("NO_DATA")
    if not Path(out_xlsx).exists():
        raise SystemExit(f"batch_report FAILED ({name}):\n{r.stdout}\n{r.stderr}")
    return nd


def add_runinfo(fam, out_xlsx, asm_family):
    """Write the run_info sheet from the family registry.

    This used to mutate module globals on the add_run_info_* scripts (RES, SPD,
    ARMS, _config_rows) and inject configuration prose that existed in no file,
    which had already drifted from add_full_run_info.py's copy of the same text.
    All four are now parameters; the prose lives in _runinfo_specs.py.
    """
    kind = fam["ri"][0]
    spec_key = {"std": "standard", "adj": "adjlist", "abl": "ablation_color",
                "thinking": "think", "thinkadj": "thinkadj"}.get(kind)
    if kind == "abl_split":
        spec_key = f"ablation_{fam['_arm']}"

    wb = openpyxl.load_workbook(out_xlsx)
    if spec_key:
        _runinfo.build_run_info(wb, RUN_INFO_SPECS[spec_key], results_root=Path(asm_family))
    elif kind == "sweep":
        # sweep sources coloring and conn/shortest_path rows from different
        # jobs, so it keeps its own builder (see tools/postprocess/add_run_info.py).
        sw = importlib.import_module("add_run_info_sweep")
        sw.RES = asm_family
        sw.build_run_info(fam["ri"][1], wb)
    wb.save(out_xlsx)

def add_metrics(fam, out_xlsx, asm_family):
    mk = fam.get("metrics")
    if not mk: return
    mod = importlib.import_module(f"add_metrics_{mk}")
    mod.RES = asm_family
    wb = openpyxl.load_workbook(out_xlsx)
    mod.build_metrics(wb)
    wb.save(out_xlsx)

def out_path(fam, label=None):
    nm = fam["name"] if not label else f"{fam['name']}_{label}"
    if MODE == "merged":
        return FULL / f"merged_{nm}_latest.xlsx"
    else:
        camp = fam["camp"]
        # disambiguate families that share a campaign (sweep nodes/edges -> 11)
        extra = f"_{fam['ri'][1]}" if fam["ri"][0] == "sweep" else ""
        if label:
            extra += f"_{label}"
        return RES / camp / f"report_{camp}{extra}.xlsx"

built = []
for fam in FAMS:
    asm_family = ASM / fam["name"]
    tsv = ASM / f"{fam['name']}.tsv"
    prov = assemble(fam["mani"], asm_family, tsv)
    base_used = [ln for ln in prov.splitlines() if "<- 0" in ln or ("<-" in ln and "abl" in ln.split("<-")[1] and MODE=="merged" and False)]
    if fam.get("split"):
        # split tsv into letters/none
        letters_tsv = ASM / "labels_letters.tsv"; none_tsv = ASM / "labels_none.tsv"
        lines = [l for l in tsv.read_text().splitlines() if l.strip()]
        letters_tsv.write_text("\n".join(l for l in lines if "_letters_" in l)+"\n")
        none_tsv.write_text("\n".join(l for l in lines if "_none_" in l)+"\n")
        for label, t in (("letters", letters_tsv), ("none", none_tsv)):
            op = out_path(fam, label)
            nd = batch_report(t, op, f"{MODE}_labels_{label}")
            fam["_arm"] = label
            add_runinfo(fam, op, asm_family)
            built.append((op.name, nd))
    else:
        op = out_path(fam)
        nd = batch_report(tsv, op, f"{MODE}_{fam['name']}")
        add_runinfo(fam, op, asm_family)
        add_metrics(fam, op, asm_family)
        built.append((op.name, nd))

print(f"\n===== {MODE.upper()} BUILD DONE =====")
for name, nd in built:
    print(f"  {name:44} NO_DATA={nd}")
