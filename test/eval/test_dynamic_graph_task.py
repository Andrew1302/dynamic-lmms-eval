"""Tests for the dynamic graph benchmark task hooks.

Covers everything lmms-eval calls on this task except the model itself:
doc_to_visual (including the NO_IMAGE / SCRAMBLE_IMAGE control ablations),
doc_to_text, doc_to_messages, process_results and the aggregation.

utils.py is loaded **by file path**, exactly as lmms-eval's `!function`
constructor loads it (lmms_eval/utils.py:890-897), so these tests exercise the
real loading mode and do not drag in the whole lmms_eval package.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO / "lmms_eval" / "tasks" / "dynamic_graph_benchmark" / "utils.py"


def _load_utils():
    spec = importlib.util.spec_from_file_location("dgb_utils", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


@pytest.fixture
def image():
    from PIL import Image

    # deterministic, non-uniform: a uniform image would survive scrambling
    img = Image.new("RGB", (8, 6))
    img.putdata([(i * 3 % 256, i * 7 % 256, i * 11 % 256) for i in range(8 * 6)])
    return img


def _doc(task="coloring", variant="direct", answer="3", image=None, **extra):
    doc = {
        "task": task,
        "variant": variant,
        "answer": answer,
        "prompt": "Q: what is it?\nA:",
        "image": image,
        "n_vertices": 6,
        "n_edges": 9,
        "constraint": "",
        "constraint_value": -1,
    }
    doc.update(extra)
    return doc


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from the shipped defaults."""
    for var in ("PROMPT_ID", "NO_IMAGE", "SCRAMBLE_IMAGE"):
        monkeypatch.delenv(var, raising=False)


# ==========================================================================
# doc_to_visual + the image control ablations
# ==========================================================================
def test_doc_to_visual_returns_the_image_as_rgb(image):
    out = utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image))
    assert len(out) == 1
    assert out[0].mode == "RGB"
    assert out[0].size == image.size


def test_doc_to_visual_handles_a_doc_with_no_image():
    assert utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=None)) == []


def test_no_image_ablation_drops_the_image(monkeypatch, image):
    monkeypatch.setenv("NO_IMAGE", "1")
    assert utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image)) == []


def test_scramble_ablation_destroys_structure_but_keeps_cost(monkeypatch, image):
    """The control must cost the same vision tokens while carrying no graph."""
    monkeypatch.setenv("SCRAMBLE_IMAGE", "1")
    scrambled = utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image))[0]
    original = image.convert("RGB")
    assert scrambled.size == original.size
    assert sorted(scrambled.getdata()) == sorted(original.getdata()), "colour histogram must be preserved"
    assert list(scrambled.getdata()) != list(original.getdata()), "pixels must actually move"


def test_scramble_is_deterministic(monkeypatch, image):
    """Two runs of the same ablation must produce identical inputs."""
    monkeypatch.setenv("SCRAMBLE_IMAGE", "1")
    a = utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image))[0]
    b = utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image))[0]
    assert list(a.getdata()) == list(b.getdata())


def test_no_image_wins_over_scramble(monkeypatch, image):
    monkeypatch.setenv("NO_IMAGE", "1")
    monkeypatch.setenv("SCRAMBLE_IMAGE", "1")
    assert utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image)) == []


def test_image_ablations_are_off_by_default(image):
    out = utils.dynamic_graph_benchmark_doc_to_visual(_doc(image=image))
    assert list(out[0].getdata()) == list(image.convert("RGB").getdata())


# ==========================================================================
# doc_to_text / doc_to_messages
# ==========================================================================
def test_doc_to_text_defaults_to_the_historical_composition():
    text = utils.dynamic_graph_benchmark_doc_to_text(_doc())
    assert text == "Answer with a single integer and nothing else.\nQ: what is it?\nA:"


def test_doc_to_text_uses_the_right_instruction_per_task():
    assert utils.dynamic_graph_benchmark_doc_to_text(_doc(task="directed_connectivity", answer="Yes")).startswith("Answer with only 'Yes' or 'No'.")


def test_prompt_id_changes_the_rendered_text(monkeypatch):
    baseline = utils.dynamic_graph_benchmark_doc_to_text(_doc())
    monkeypatch.setenv("PROMPT_ID", "cot_zeroshot_v1")
    cot = utils.dynamic_graph_benchmark_doc_to_text(_doc())
    assert cot != baseline
    assert "step by step" in cot


def test_doc_to_text_accepts_the_legacy_kwargs_argument():
    """lmms-eval passes lmms_eval_specific_kwargs when a task defines it."""
    assert utils.dynamic_graph_benchmark_doc_to_text(_doc(), None) == utils.dynamic_graph_benchmark_doc_to_text(_doc())


def test_doc_to_messages_shape(image):
    msgs = utils.dynamic_graph_benchmark_doc_to_messages(_doc(image=image))
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert [c["type"] for c in msgs[0]["content"]] == ["image", "text"]


def test_doc_to_messages_respects_the_no_image_ablation(monkeypatch, image):
    monkeypatch.setenv("NO_IMAGE", "1")
    msgs = utils.dynamic_graph_benchmark_doc_to_messages(_doc(image=image))
    assert [c["type"] for c in msgs[0]["content"]] == ["text"]


def test_few_shot_produces_prior_turns_on_the_chat_path(monkeypatch, image):
    monkeypatch.setenv("PROMPT_ID", "cot_fewshot_img_v1")
    msgs = utils.dynamic_graph_benchmark_doc_to_messages(_doc(image=image))
    assert len(msgs) == 5, "two exemplar pairs plus the real question"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]


def test_text_and_messages_agree_on_the_final_question(image):
    """The two backends must ask the same thing."""
    doc = _doc(image=image)
    text = utils.dynamic_graph_benchmark_doc_to_text(doc)
    msgs = utils.dynamic_graph_benchmark_doc_to_messages(doc)
    assert [c for c in msgs[0]["content"] if c["type"] == "text"][0]["text"] == text


# ==========================================================================
# process_results
# ==========================================================================
def test_scores_a_correct_and_an_incorrect_answer():
    assert utils.dynamic_graph_benchmark_process_results(_doc(answer="3"), ["3"])["accuracy"]["score"] == 1.0
    assert utils.dynamic_graph_benchmark_process_results(_doc(answer="3"), ["4"])["accuracy"]["score"] == 0.0


def test_scoring_normalises_both_sides():
    """The prediction and the gold answer go through the same parser."""
    assert utils.dynamic_graph_benchmark_process_results(_doc(task="directed_connectivity", answer="Yes"), ["yes"])["accuracy"]["score"] == 1.0
    assert utils.dynamic_graph_benchmark_process_results(_doc(task="directed_connectivity", answer="Yes"), ["A: Yes"])["accuracy"]["score"] == 1.0


def test_empty_response_scores_zero_without_raising():
    assert utils.dynamic_graph_benchmark_process_results(_doc(), [])["accuracy"]["score"] == 0.0
    assert utils.dynamic_graph_benchmark_process_results(_doc(), [""])["accuracy"]["score"] == 0.0


def test_results_carry_prompt_provenance():
    """Without this a results tree cannot state how it was asked."""
    acc = utils.dynamic_graph_benchmark_process_results(_doc(), ["3"])["accuracy"]
    assert acc["prompt_id"] == "direct_v1"
    assert len(acc["prompt_fingerprint"]) == 16


def test_provenance_tracks_the_selected_template(monkeypatch):
    monkeypatch.setenv("PROMPT_ID", "cot_zeroshot_v1")
    acc = utils.dynamic_graph_benchmark_process_results(_doc(), ["3"])["accuracy"]
    assert acc["prompt_id"] == "cot_zeroshot_v1"


def test_results_carry_the_axis_fields_reports_pivot_on():
    acc = utils.dynamic_graph_benchmark_process_results(_doc(n_vertices=11, n_edges=20, constraint="nodes", constraint_value=11), ["3"])["accuracy"]
    assert (acc["n_vertices"], acc["n_edges"]) == (11, 20)
    assert (acc["constraint"], acc["constraint_value"]) == ("nodes", 11)
    assert (acc["task"], acc["variant"]) == ("coloring", "direct")


def test_cot_scoring_reads_a_chain_from_the_end(monkeypatch):
    """The same response scores differently under the two conditions -- which is
    exactly why the parser has to travel with the template."""
    chain = "No path at first glance.\nTracing 5 -> 1 -> 4.\nYes"
    doc = _doc(task="directed_connectivity", answer="Yes")
    assert utils.dynamic_graph_benchmark_process_results(doc, [chain])["accuracy"]["score"] == 0.0
    monkeypatch.setenv("PROMPT_ID", "cot_zeroshot_v1")
    assert utils.dynamic_graph_benchmark_process_results(doc, [chain])["accuracy"]["score"] == 1.0


# ==========================================================================
# aggregation
# ==========================================================================
def _row(task, variant, score):
    return {"task": task, "variant": variant, "score": score, "prompt_id": "direct_v1", "prompt_fingerprint": "x" * 16}


def test_aggregate_is_the_overall_mean_not_a_mean_of_means():
    rows = [_row("coloring", "direct", 1.0)] * 3 + [_row("coloring", "disguise", 0.0)]
    assert utils.dynamic_graph_benchmark_aggregate_results(rows) == pytest.approx(0.75)


def test_aggregate_handles_an_empty_run():
    assert utils.dynamic_graph_benchmark_aggregate_results([]) == 0.0


def test_aggregate_spans_tasks_and_variants():
    rows = [
        _row("coloring", "direct", 1.0),
        _row("coloring", "disguise", 0.0),
        _row("shortest_path", "direct", 1.0),
        _row("shortest_path", "disguise", 1.0),
    ]
    assert utils.dynamic_graph_benchmark_aggregate_results(rows) == pytest.approx(0.75)


# ==========================================================================
# process_docs filters
# ==========================================================================
class _FakeDataset:
    """Minimal stand-in for a HF Dataset: only .filter is used."""

    def __init__(self, rows):
        self.rows = rows

    def filter(self, fn):
        return _FakeDataset([r for r in self.rows if fn(r)])


ALL_ROWS = [{"task": t, "variant": v} for t in ("coloring", "directed_connectivity", "shortest_path") for v in ("direct", "disguise")]


@pytest.mark.parametrize(
    "fn_name,task,variant",
    [
        ("filter_coloring_direct", "coloring", "direct"),
        ("filter_coloring_disguise", "coloring", "disguise"),
        ("filter_directed_connectivity_direct", "directed_connectivity", "direct"),
        ("filter_directed_connectivity_disguise", "directed_connectivity", "disguise"),
        ("filter_shortest_path_direct", "shortest_path", "direct"),
        ("filter_shortest_path_disguise", "shortest_path", "disguise"),
    ],
)
def test_each_filter_selects_exactly_its_own_slice(fn_name, task, variant):
    out = getattr(utils, fn_name)(_FakeDataset(ALL_ROWS))
    assert out.rows == [{"task": task, "variant": variant}]


def test_no_filter_for_the_deleted_undirected_connectivity_task():
    assert not hasattr(utils, "filter_connectivity_direct")
    assert not hasattr(utils, "filter_connectivity_disguise")
