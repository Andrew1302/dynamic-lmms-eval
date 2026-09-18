"""The family registry must agree with the two tables it replaces.

`_families.py` unifies `organize_results.SPECS` (regexes -> campaign + leaf) and
`batch_report._AXIS_PREFIXES` (prefix strings -> axis + axis_value). Before
either is rewired to it, it has to produce identical answers for every job name
that actually exists -- every generated conf and every batch manifest line.

Two divergences are expected and asserted explicitly rather than tolerated:

  * `thinkadj500inc_*` currently reports axis="unknown" in every batch report,
    because it has a SPEC but no matching prefix. The registry fixes it, so
    those names legitimately differ. That is the drift the registry exists to
    remove; it is pinned here so the fix is visible rather than silent.
  * `promptexp_*` stays scratch until the CoT family lands.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POSTPROCESS = REPO / "tools" / "postprocess"
sys.path.insert(0, str(POSTPROCESS))

import _families  # noqa: E402
import organize_results as legacy_org  # noqa: E402

JOBS_DIR = REPO / "remote_execution_scripts" / "jobs" / "graph_benchmark"
BATCH_DIR = REPO / "remote_execution_scripts" / "batches"


def _all_job_names() -> list[str]:
    names = {p.stem for p in JOBS_DIR.glob("*.conf")}
    for manifest in BATCH_DIR.glob("*.txt"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                names.add(line.rsplit("/", 1)[-1])
    return sorted(names)


JOB_NAMES = _all_job_names()

# thinkadj500inc has a SPEC but no axis prefix -- the live drift the registry fixes.
KNOWN_AXIS_FIX = re.compile(r"^graph_bench_thinkadj500inc_")


def test_the_corpus_is_not_empty():
    """Guard against the globs silently matching nothing."""
    assert len(JOB_NAMES) > 150, f"only found {len(JOB_NAMES)} job names"


@pytest.mark.parametrize("job", JOB_NAMES)
def test_campaign_and_leaf_match_organize_results(job):
    """Filing must not move: a changed campaign or leaf would relocate results."""
    legacy = legacy_org.match_job(job)
    new = _families.match_job(job)

    if legacy is None:
        assert new is None, f"{job}: registry classified a job the organizer calls scratch"
        return
    assert new is not None, f"{job}: registry failed to classify a job the organizer files"

    legacy_campaign, legacy_leaf_tmpl, legacy_fields = legacy
    family, fields = new
    assert family.campaign == legacy_campaign, job
    assert family.leaf(fields) == legacy_leaf_tmpl.format(**legacy_fields), job


LEGACY_AXIS = json.loads((REPO / "test" / "eval" / "fixtures" / "legacy_axis_mapping.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("job", JOB_NAMES)
def test_axis_matches_the_frozen_legacy_mapping(job):
    """Report grouping must not move, except for the one known fix.

    Compared against a frozen snapshot of the pre-rewire ``_AXIS_PREFIXES``
    table rather than against the live ``batch_report``, which now derives from
    this registry -- that comparison would be a tautology.
    """
    import batch_report

    legacy = LEGACY_AXIS.get(job)
    if legacy is None:
        pytest.skip(f"{job} not in the frozen corpus")
    produced = batch_report.parse_job_id(job)

    if KNOWN_AXIS_FIX.match(job):
        # The drift the registry removes: a filing rule existed, a prefix did not.
        assert legacy["axis"] == "unknown", f"{job}: frozen snapshot no longer shows the drift"
        assert produced["axis"] == "thinking_adj", job
        assert produced["model_short"] in _families.MODELS, job
        return

    assert produced == legacy, job


def test_the_registry_fixed_exactly_the_jobs_we_expect():
    """Guard the blast radius: only thinkadj500inc may change classification."""
    import batch_report

    changed = {j for j, was in LEGACY_AXIS.items() if batch_report.parse_job_id(j) != was}
    assert changed, "expected the thinkadj500inc fix to show up"
    assert all(KNOWN_AXIS_FIX.match(j) for j in changed), sorted(changed - {j for j in changed if KNOWN_AXIS_FIX.match(j)})
    assert len(changed) == 18, f"expected 18 thinkadj500inc jobs, got {len(changed)}"


def test_campaign_for_name_keeps_its_signature():
    """The job generator imports this and runs a drift check against it."""
    assert _families.campaign_for_name("graph_bench_standard_easy_qwen35_4b") == "10_standard"
    assert _families.campaign_for_name("graph_bench_thinksmoke_think_hard_qwen35_4b") == ""
    assert _families.campaign_for_name("nonsense") == ""


def test_every_campaign_dir_is_well_formed():
    for family in _families.FAMILIES:
        assert re.match(r"^\d\d_[a-z_]+$", family.campaign), family.key


def test_family_keys_are_unique():
    assert len(_families.FAMILY_KEYS) == len(set(_families.FAMILY_KEYS))


def test_more_specific_families_are_matched_first():
    """thinkadj500inc must win over thinkadj, and thinkadj over think."""
    for job, expected in [
        ("graph_bench_thinkadj500inc_think_easy_qwen35_4b", "thinkadj_inc"),
        ("graph_bench_thinkadj_think_easy_qwen35_4b", "thinkadj"),
        ("graph_bench_think_think_easy_qwen35_4b", "think"),
    ]:
        family, _ = _families.match_job(job)
        assert family.key == expected, job


def test_job_dir_prefers_a_flat_fetch_then_the_family_campaign(tmp_path):
    family = _families.family_by_key("standard")
    job = "graph_bench_standard_easy_qwen35_4b"
    assert family.job_dir(tmp_path, job) == tmp_path / family.campaign / "_jobs" / job
    (tmp_path / job).mkdir()
    assert family.job_dir(tmp_path, job) == tmp_path / job


def test_the_cot_family_needs_no_change_outside_the_registry():
    """The acceptance criterion for this refactor: a new experiment family is
    one registry entry. Filing, report grouping and the campaign description all
    follow from it, with no second table to update."""
    import batch_report
    import organize_results

    job = "graph_bench_cot_cot_zeroshot_v1_hard_qwen35_4b"

    family, fields = _families.match_job(job)
    assert family.key == "cot_prompt"
    assert family.campaign == "18_abl_cot"
    assert family.leaf(fields) == "qwen_hard_cot_zeroshot_v1"

    # the organizer files it without knowing anything about CoT
    assert organize_results.campaign_for_name(job) == "18_abl_cot"
    assert organize_results.match_job(job)[0] == "18_abl_cot"
    assert "18_abl_cot" in organize_results.CAMPAIGN_DESC

    # the report groups it without knowing anything about CoT
    assert batch_report.parse_job_id(job) == {
        "axis": "cot_prompt",
        "axis_value": "cot_zeroshot_v1/hard",
        "model_short": "qwen35_4b",
    }


@pytest.mark.parametrize("prompt_id", ["direct_v1", "cot_zeroshot_v1", "cot_fewshot_img_v1"])
def test_every_shipped_prompt_template_yields_a_classifiable_job(prompt_id):
    """The registry's job-name pattern must accept every real PROMPT_ID."""
    job = f"graph_bench_cot_{prompt_id}_medium_internvl35_4b"
    matched = _families.match_job(job)
    assert matched is not None, f"{prompt_id} produces an unclassifiable job name"
    family, fields = matched
    assert fields["prompt"] == prompt_id
    assert family.leaf(fields) == f"internvl_medium_{prompt_id}"


def test_legacy_campaign_descriptions_survive_the_derivation():
    import organize_results

    assert organize_results.CAMPAIGN_DESC["01_standard"].startswith("PRE-FIX LEGACY ARCHIVE")
    assert len(organize_results.CAMPAIGN_DESC) >= 18


def test_coloring_only_jobs_keep_their_marker_in_the_axis_value():
    family, fields = _families.match_job("graph_bench_think_think_coloring_easy_internvl35_4b")
    assert family.axis_value(fields) == "think/coloring/easy"
    family, fields = _families.match_job("graph_bench_think_think_easy_internvl35_4b")
    assert family.axis_value(fields) == "think/easy"
