"""Tests for the consolidated run_info builder.

The strong evidence that this reproduces the three scripts it replaced is the
cell-level diff against the seven shipped full-report workbooks (0 differences,
recorded in the commit). These tests keep the contract from drifting afterwards
without needing those workbooks, which live under the gitignored remote_results/.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "postprocess"))

openpyxl = pytest.importorskip("openpyxl")

import _runinfo  # noqa: E402
from _runinfo_specs import RUN_INFO_SPECS  # noqa: E402

# Families the shared builder owns. sweep_* keep their own script: they source
# coloring and conn/shortest_path rows from different jobs, which is a different
# sourcing shape rather than a different parameterisation.
SHARED = ["think", "thinkadj", "ablation_letters", "ablation_none", "ablation_color", "standard", "adjlist"]


@pytest.fixture
def workbook():
    wb = openpyxl.Workbook()
    wb.active.title = "summary"
    wb.active["A1"] = "placeholder"
    return wb


@pytest.mark.parametrize("key", sorted(RUN_INFO_SPECS))
def test_every_spec_is_complete(key):
    spec = RUN_INFO_SPECS[key]
    required = {"title", "models", "tasks", "arms", "cells", "cell_header", "cell_noun", "cells_plural", "samples_per_cell", "range_job_tmpls", "cell_source", "config_rows", "col_widths"}
    assert required <= set(spec), f"{key} missing {sorted(required - set(spec))}"
    assert spec["title"].strip()
    assert spec["models"] and spec["tasks"] and spec["cells"]
    assert spec["samples_per_cell"] > 0
    assert len(spec["col_widths"]) == 7, f"{key} needs a width per column A-G"
    assert spec["cell_source"] in ("job", "constraint_value")


@pytest.mark.parametrize("key", sorted(RUN_INFO_SPECS))
def test_config_rows_are_pairs_of_nonempty_text(key):
    for row in RUN_INFO_SPECS[key]["config_rows"]:
        assert len(row) == 2, f"{key}: {row!r}"
        assert row[0].strip() and str(row[1]).strip(), f"{key}: {row!r}"


@pytest.mark.parametrize("key", SHARED)
def test_builds_the_sheet_at_index_zero(workbook, key, tmp_path):
    _runinfo.build_run_info(workbook, RUN_INFO_SPECS[key], results_root=tmp_path)
    assert workbook.sheetnames[0] == "run_info", "verify_runinfo asserts run_info is the first sheet"
    assert "summary" in workbook.sheetnames, "the existing sheets must survive"


@pytest.mark.parametrize("key", SHARED)
def test_rebuilding_is_idempotent(workbook, key, tmp_path):
    spec = RUN_INFO_SPECS[key]
    _runinfo.build_run_info(workbook, spec, results_root=tmp_path)
    first = [[c.value for c in row] for row in workbook["run_info"].iter_rows()]
    _runinfo.build_run_info(workbook, spec, results_root=tmp_path)
    assert workbook.sheetnames.count("run_info") == 1
    assert [[c.value for c in row] for row in workbook["run_info"].iter_rows()] == first


def test_arm_families_get_a_six_column_prompt_table(workbook, tmp_path):
    """The only structural difference between the merged scripts."""
    _runinfo.build_run_info(workbook, RUN_INFO_SPECS["think"], results_root=tmp_path)
    headers = {tuple(c.value for c in row if c.value) for row in workbook["run_info"].iter_rows()}
    assert ("model", "arm", "task", "direct", "disguise", "total") in headers


def test_armless_families_get_a_five_column_prompt_table(workbook, tmp_path):
    _runinfo.build_run_info(workbook, RUN_INFO_SPECS["ablation_color"], results_root=tmp_path)
    headers = {tuple(c.value for c in row if c.value) for row in workbook["run_info"].iter_rows()}
    assert ("model", "task", "direct", "disguise", "total") in headers


def test_the_title_is_the_first_cell(workbook, tmp_path):
    spec = RUN_INFO_SPECS["standard"]
    _runinfo.build_run_info(workbook, spec, results_root=tmp_path)
    assert workbook["run_info"]["A1"].value == spec["title"]


def test_standard_reports_its_own_sample_count(workbook, tmp_path):
    """n=500 for standard vs 100 for the ablations. It was previously injected
    by a monkeypatched global, and the extraction initially got it wrong."""
    rows = dict(RUN_INFO_SPECS["standard"]["config_rows"])
    generations = next(v for k, v in rows.items() if "Generations" in k)
    assert generations.startswith("500")
    assert RUN_INFO_SPECS["standard"]["samples_per_cell"] == 500
    assert RUN_INFO_SPECS["ablation_color"]["samples_per_cell"] == 100


def test_extra_rows_are_appended_after_the_config_block(workbook, tmp_path):
    """The hook the CoT campaign uses to record prompt provenance per run."""
    spec = RUN_INFO_SPECS["think"]
    _runinfo.build_run_info(workbook, spec, results_root=tmp_path, extra_rows=[("Prompt template", "cot_zeroshot_v1 (abc123)")])
    values = [c.value for row in workbook["run_info"].iter_rows() for c in row]
    assert "Prompt template" in values
    assert values.index("Prompt template") > values.index(spec["config_rows"][0][0])


def test_families_keep_their_own_column_widths():
    """They genuinely differ (B was 74, 66 and 60) -- a detail the 'these files
    are near-identical' framing hid."""
    widths = {k: RUN_INFO_SPECS[k]["col_widths"][1] for k in ("think", "ablation_color", "sweep_nodes")}
    assert len(set(widths.values())) == 3, widths


def test_the_replaced_scripts_are_gone():
    post = REPO / "tools" / "postprocess"
    for gone in ("add_run_info_thinking.py", "add_run_info_thinkadj.py", "add_run_info_ablation.py"):
        assert not (post / gone).exists(), f"{gone} should have been replaced by add_run_info.py"
    assert (post / "add_run_info.py").exists()
    # sweep deliberately survives
    assert (post / "add_run_info_sweep.py").exists()


def test_cli_rejects_an_unknown_family():
    import add_run_info

    with pytest.raises(SystemExit):
        add_run_info.main(["--family", "nope", "--xlsx", "x.xlsx"])


# ==========================================================================
# metrics consolidation
# ==========================================================================
def test_metrics_families_are_registered():
    import add_metrics

    assert sorted(add_metrics.FAMILIES) == ["think", "thinkadj"]
    for prefix, title in add_metrics.FAMILIES.values():
        assert prefix.startswith("graph_bench_")
        assert "quality metrics" in title


def test_the_replaced_metrics_scripts_are_gone():
    post = REPO / "tools" / "postprocess"
    for gone in ("add_metrics_thinking.py", "add_metrics_thinkadj.py"):
        assert not (post / gone).exists(), f"{gone} should have been replaced by add_metrics.py"
    assert (post / "add_metrics.py").exists()


def test_metrics_cli_rejects_an_unknown_family():
    import add_metrics

    with pytest.raises(SystemExit):
        add_metrics.main(["--family", "nope", "--xlsx", "x.xlsx"])


def test_verify_thinking_uses_the_production_parser():
    """It used to read _normalize off the task utils module; that moved into
    prompting.answers, which briefly left this (and add_metrics) broken."""
    import verify_thinking

    assert verify_thinking._normalize("The chromatic number is 4.", "coloring") == "4"
    assert verify_thinking._normalize("A: Yes", "directed_connectivity") == "yes"
