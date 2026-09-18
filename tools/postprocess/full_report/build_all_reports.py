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
import add_run_info_ablation as abl
_COMMON_TAIL = [
    ("Coloring construction", "special-coloring: chromatic number planted uniformly over {2,3,4}"),
    ("conn / shortest_path construction", "standard difficulty presets"),
    ("Edge-count convention", "directed_connectivity = directed out-edges; coloring/shortest_path = undirected"),
    ("Graphs across models", "identical per (task, difficulty) — same seeds; ranges below from one model"),
]
def _std_config(_arm):
    return [("Models", ", ".join(abl.MODELS)), ("Tasks", ", ".join(abl.TASKS)),
            ("Difficulties", "easy, medium, hard (pure per-difficulty)"),
            ("Generations per task per difficulty", f"{abl.SPD}  (→ {abl.SPD} direct + {abl.SPD} disguise)"),
            ("Setting", "baseline / standard run (no ablation)"),
            ("Prompt augmentation", "none — image only (no adjacency list)"),
            ("Render settings", "label_style=numeric, node_color=#AED6F1, edge_style=straight"), *_COMMON_TAIL]
def _adj_config(_arm):
    return [("Models", ", ".join(abl.MODELS)), ("Tasks", ", ".join(abl.TASKS)),
            ("Difficulties", "easy, medium, hard (pure per-difficulty)"),
            ("Generations per task per difficulty", f"{abl.SPD}  (→ {abl.SPD} direct + {abl.SPD} disguise)"),
            ("Ablated setting", "adjacency list injected into the prompt (textual edge list alongside the image)"),
            ("Baseline for comparison", "standard run — image only, no adjacency list"),
            ("Prompt augmentation", "adjacency list included (image + text edge list)"),
            ("Render settings", "label_style=numeric, node_color=#AED6F1, edge_style=straight"), *_COMMON_TAIL]

STD_ARM = {"title": "Standard baseline benchmark — run configuration",
           "job_base": "graph_bench_standard",
           "ablated": ("Setting", "baseline"), "render": ("Render", "standard")}
ADJ_ARM = {"title": "Adjacency-list ablation — run configuration",
           "job_base": "graph_bench_ablation_adjlist",
           "ablated": ("Ablated setting", "adjacency list in prompt"),
           "render": ("Render", "standard")}

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
    kind = fam["ri"][0]
    wb = openpyxl.load_workbook(out_xlsx)
    if kind in ("std","adj","abl","abl_split"):
        abl.RES = asm_family
        orig_spd = abl.SPD; abl.SPD = fam.get("spd",100)
        orig_cfg = abl._config_rows
        try:
            if kind == "std":
                abl.ARMS["standard"] = STD_ARM
                abl._config_rows = _std_config; abl.build_run_info("standard", wb)
            elif kind == "adj":
                abl.ARMS["adjlist"] = ADJ_ARM
                abl._config_rows = _adj_config; abl.build_run_info("adjlist", wb)
            elif kind == "abl":
                abl.build_run_info(fam["ri"][1], wb)      # color
            elif kind == "abl_split":
                abl.build_run_info(fam["_arm"], wb)        # letters|none
        finally:
            abl.SPD = orig_spd; abl._config_rows = orig_cfg
    elif kind == "thinking":
        thk = importlib.import_module("add_run_info_thinking"); thk.RES = asm_family; thk.build_run_info(wb)
    elif kind == "thinkadj":
        tj = importlib.import_module("add_run_info_thinkadj"); tj.RES = asm_family; tj.build_run_info(wb)
    elif kind == "sweep":
        sw = importlib.import_module("add_run_info_sweep"); sw.RES = asm_family; sw.build_run_info(fam["ri"][1], wb)
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
