#!/usr/bin/env python
"""Add the run_info tab to every full-results workbook.

    python tools/postprocess/full_report/add_full_run_info.py <asm_root> <reports_dir>

This used to import the four ``add_run_info_*`` scripts and mutate their module
globals before calling them -- ``RES`` to point at the assembled tree, ``SPD``
to 500 for the standard family, ``ARMS[...]`` to graft on two synthetic arms,
and ``_config_rows`` to inject configuration prose that was never shipped in any
file. That prose then drifted from the copy in build_all_reports.py, so two
workbooks could describe the same family differently.

All four are now real parameters: the text lives in _runinfo_specs.py and the
assembled root is an argument. Nothing is monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POSTPROCESS))

import openpyxl

import _runinfo
from _runinfo_specs import RUN_INFO_SPECS

# report stem -> (run_info spec, assembled-tree subdir)
REPORTS = {
    "full_standard": ("standard", "standard"),
    "full_adjlist": ("adjlist", "adjlist"),
    "full_color": ("ablation_color", "color"),
    "full_labels_letters": ("ablation_letters", "labels"),
    "full_labels_none": ("ablation_none", "labels"),
    "full_think": ("think", "think"),
    "full_thinkadj": ("thinkadj", "thinkadj"),
}

# The sweep family still has its own script: it sources coloring and
# conn/shortest_path rows from different jobs, so it is not a parameterisation
# of the shared layout. See tools/postprocess/add_run_info.py.
SWEEP_REPORTS = {"full_sweep_nodes": "nodes", "full_sweep_edges": "edges"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <asm_root> <reports_dir>", file=sys.stderr)
        return 2
    asm_root, reports_dir = Path(argv[0]), Path(argv[1])

    touched = 0
    for stem, (spec_key, subdir) in REPORTS.items():
        path = reports_dir / f"{stem}_latest.xlsx"
        if not path.is_file():
            print(f"  skip {stem}: no workbook")
            continue
        results_root = asm_root / subdir
        if not results_root.is_dir():
            print(f"  skip {stem}: no assembled tree at {results_root}")
            continue
        wb = openpyxl.load_workbook(path)
        _runinfo.build_run_info(wb, RUN_INFO_SPECS[spec_key], results_root=results_root)
        wb.save(path)
        print(f"  run_info -> {path.name}  ({spec_key})")
        touched += 1

    for stem, axis in SWEEP_REPORTS.items():
        path = reports_dir / f"{stem}_latest.xlsx"
        if path.is_file():
            print(f"  note {stem}: run `add_run_info_sweep.py {axis}` (separate sourcing rules)")

    print(f"[full run_info] {touched} workbooks updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
