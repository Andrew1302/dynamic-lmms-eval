"""Add a run_info sheet to each full-results report, reusing the shipped
add_run_info_* generators. The only new content is factual config text for
standard/adjlist (no shipped generator exists); the size-range table and
prompt-count table are computed by the shipped code from the assembled jsonls.

Data-location fix: the shipped scripts read `RES / <job>`; the full reports are
assembled elsewhere, so we redirect each module's RES to the assembled family
dir before calling its build_run_info. No shipped file is edited.

Usage: python add_full_run_info.py <assembled_root> <full_reports_dir>
  <assembled_root>   = scratch dir holding <family>/<job>/logs/*.jsonl
  <full_reports_dir> = remote_results/_full_reports
"""
import sys
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]

TOOLS = REPO / "tools" / "postprocess"
sys.path.insert(0, str(TOOLS))
import openpyxl
import add_run_info_ablation as abl
import add_run_info_thinking as thk
import add_run_info_sweep as swp

ASM = Path(sys.argv[1])
FULL = Path(sys.argv[2])

# --- factual config text for the two families with no shipped generator ------
# Mirrors add_run_info_ablation._config_rows but with accurate per-family lines
# (the ablation template hard-codes "no adjacency list", wrong for adjlist).
_COMMON_TAIL = [
    ("Coloring construction",
     "special-coloring: chromatic number planted uniformly over {2,3,4}"),
    ("conn / shortest_path construction", "standard difficulty presets"),
    ("Edge-count convention",
     "directed_connectivity = directed out-edges; coloring/shortest_path = undirected"),
    ("Graphs across models",
     "identical per (task, difficulty) — same seeds; ranges below from one model"),
]


def _std_config(_arm):
    return [
        ("Models", ", ".join(abl.MODELS)),
        ("Tasks", ", ".join(abl.TASKS)),
        ("Difficulties", "easy, medium, hard (pure per-difficulty; no per-task override)"),
        ("Generations per task per difficulty",
         f"{abl.SPD}  (→ {abl.SPD} direct + {abl.SPD} disguise prompts)"),
        ("Setting", "baseline / standard run (no ablation)"),
        ("Prompt augmentation", "none — image only (no adjacency list in prompt)"),
        ("Render settings",
         "label_style=numeric, node_color=#AED6F1, edge_style=straight"),
        *_COMMON_TAIL,
    ]


def _adj_config(_arm):
    return [
        ("Models", ", ".join(abl.MODELS)),
        ("Tasks", ", ".join(abl.TASKS)),
        ("Difficulties", "easy, medium, hard (pure per-difficulty; no per-task override)"),
        ("Generations per task per difficulty",
         f"{abl.SPD}  (→ {abl.SPD} direct + {abl.SPD} disguise prompts)"),
        ("Ablated setting",
         "adjacency list injected into the prompt (textual edge list alongside the image)"),
        ("Baseline for comparison", "standard run — image only, no adjacency list"),
        ("Prompt augmentation", "adjacency list included (image + text edge list)"),
        ("Render settings",
         "label_style=numeric, node_color=#AED6F1, edge_style=straight"),
        *_COMMON_TAIL,
    ]


def run_ablation(report, arm, family, config_override=None, arms_entry=None, spd=100):
    abl.RES = ASM / family
    if arms_entry:
        abl.ARMS[arm] = arms_entry
    orig = abl._config_rows
    orig_spd = abl.SPD
    abl.SPD = spd  # standard is n=500/task; ablations n=100
    if config_override:
        abl._config_rows = config_override
    try:
        p = FULL / report
        wb = openpyxl.load_workbook(p)
        abl.build_run_info(arm, wb)
        wb.save(p)
    finally:
        abl._config_rows = orig
        abl.SPD = orig_spd
    print("run_info +", report, "(ablation:%s, spd=%d)" % (arm, spd))


def run_thinking(report, family):
    thk.RES = ASM / family
    p = FULL / report
    wb = openpyxl.load_workbook(p)
    thk.build_run_info(wb)
    wb.save(p)
    print("run_info +", report, "(thinking)")


def run_sweep(report, axis, family):
    swp.RES = ASM / family
    p = FULL / report
    wb = openpyxl.load_workbook(p)
    swp.build_run_info(axis, wb)
    wb.save(p)
    print("run_info +", report, "(sweep:%s)" % axis)


STD_ARM = {"title": "Standard baseline benchmark — run configuration",
           "job_base": "graph_bench_standard",
           "ablated": ("Setting", "baseline"), "render": ("Render", "standard")}
ADJ_ARM = {"title": "Adjacency-list ablation — run configuration",
           "job_base": "graph_bench_ablation_adjlist",
           "ablated": ("Ablated setting", "adjacency list in prompt"),
           "render": ("Render", "standard")}

run_ablation("full_standard_latest.xlsx", "standard", "standard", _std_config, STD_ARM, spd=500)
run_ablation("full_adjlist_latest.xlsx", "adjlist", "adjlist", _adj_config, ADJ_ARM)
run_ablation("full_color_latest.xlsx", "color", "color")
run_ablation("full_labels_letters_latest.xlsx", "letters", "labels")
run_ablation("full_labels_none_latest.xlsx", "none", "labels")
run_thinking("full_think_latest.xlsx", "think")
run_sweep("full_sweep_nodes_latest.xlsx", "nodes", "sweep_nodes")
print("done")
