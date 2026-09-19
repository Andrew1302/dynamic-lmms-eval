"""Prompt stability + answer-parsing regressions for the dynamic graph benchmark.

Sibling of test_prompt_stability.py, reusing its snapshot directory and the
shared ``--update-snapshots`` flag. It is a separate module because that file's
CASES schema assumes a ``doc_to_text(fixture, kwargs)`` signature, whereas these
prompts are template-first and selected by the PROMPT_ID env var.

Two of these tests are the safety net for the prompting refactor:

  * test_direct_v1_reproduces_recorded_prompts -- direct_v1 must render exactly
    the prompt string recorded in real run logs. If it drifts, every published
    campaign number silently becomes incomparable.
  * test_answer_parsing_matches_recorded -- the moved parser must agree with the
    pre-refactor utils._normalize on real model responses.

Both read fixtures/dynamic_graph_prompts.json, extracted once from
remote_results/ (which is gitignored, so the data cannot be read at test time).

    # after an INTENTIONAL template change:
    pytest test/eval/prompt_stability/ --update-snapshots -v
"""

import json
from pathlib import Path

import pytest

from prompting import render
from prompting.answers import cot_answer_spec, default_answer_spec
from prompting.registry import PROMPT_IDS, active_template, load_template
from prompting.file_template import load_exemplar

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "dynamic_graph_prompts.json").read_text(encoding="utf-8"))

# Eval graphs are seeded from base 42 (tools/prepare_dynamic_graph_benchmark.py);
# exemplars must come from far outside that space or a worked example could be a
# question the model is later scored on.
EVAL_SEED_CEILING = 100_000_000

TASK_VARIANTS = sorted(FIXTURES["prompts"])


def _doc(task, variant, prompt):
    return {"task": task, "variant": variant, "prompt": prompt}


def _stored_prompt(task, recorded):
    """Strip the eval-time instruction prefix to recover doc["prompt"]."""
    instruction = default_answer_spec(task).instruction() + "\n"
    assert recorded.startswith(instruction), "fixture lost its instruction prefix"
    return recorded[len(instruction) :]


# --------------------------------------------------------------------------
# The refactor safety net
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key", TASK_VARIANTS)
def test_direct_v1_reproduces_recorded_prompts(key):
    """direct_v1 renders byte-for-byte what the shipped campaigns actually sent."""
    task, variant = key.split("|")
    template = load_template("direct_v1")
    for recorded in FIXTURES["prompts"][key]:
        doc = _doc(task, variant, _stored_prompt(task, recorded))
        text, _ = render.to_flat(template.turns(doc), template.system(doc), [])
        assert text == recorded


@pytest.mark.parametrize("task", sorted(FIXTURES["parses"]))
def test_answer_parsing_matches_recorded(task):
    """The moved parser agrees with pre-refactor utils._normalize on real output."""
    spec = default_answer_spec(task)
    for raw, expected in FIXTURES["parses"][task]:
        assert spec.parse(raw) == expected


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------
@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
@pytest.mark.parametrize("key", TASK_VARIANTS)
def test_rendered_prompt_snapshot(prompt_id, key, update_snapshots):
    """Freeze every template's rendered output; a wording change must be explicit."""
    task, variant = key.split("|")
    template = load_template(prompt_id)
    doc = _doc(task, variant, _stored_prompt(task, FIXTURES["prompts"][key][0]))
    payload = {
        "prompt_id": prompt_id,
        "task": task,
        "variant": variant,
        "system": template.system(doc),
        "turns": [
            {
                "role": t.role,
                "text": t.text,
                "images": [i if i == "<doc-image>" else Path(i).name for i in t.images],
            }
            for t in template.turns(doc)
        ],
        "answer_instruction": template.answer_spec(task).instruction(),
        "gen_kwargs": template.budget().as_gen_kwargs(),
    }
    path = SNAPSHOT_DIR / f"dgb__{prompt_id}__{task}__{variant}.json"
    if update_snapshots:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        pytest.skip(f"snapshot updated: {path.name}")
    assert path.exists(), f"no snapshot for {path.name}; generate with --update-snapshots"
    assert payload == json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_fingerprint_is_stable_and_distinct(prompt_id):
    """Fingerprints are deterministic, and no two templates collide."""
    assert load_template(prompt_id).fingerprint() == load_template(prompt_id).fingerprint()
    assert len({load_template(p).fingerprint() for p in PROMPT_IDS}) == len(PROMPT_IDS)


def test_cot_templates_read_the_answer_from_the_end():
    """A reasoning template must not be scored on the model's opening word."""
    chain = "No path is obvious at first.\nFollowing the arrows: 5 -> 1 -> 4.\nYes"
    assert default_answer_spec("directed_connectivity").parse(chain) == "no"
    assert cot_answer_spec("directed_connectivity").parse(chain) == "yes"
    for prompt_id in ("cot_zeroshot_v1", "cot_fewshot_img_v1"):
        assert load_template(prompt_id).answer_spec("directed_connectivity").parse(chain) == "yes"


# --------------------------------------------------------------------------
# Exemplar integrity
# --------------------------------------------------------------------------


def test_fewshot_exemplars_are_variant_matched():
    """cot_fewshot_img_v1 follows the document's variant. (sp_fewshot_direct_ex1_v1
    deliberately does not -- that cross-domain pinning is the PoC's probe.)"""
    template = load_template("cot_fewshot_img_v1")
    doc = _doc("coloring", "disguise", "Q: x\nA:")
    names = [Path(i).name for t in template.turns(doc) for i in t.images if i != "<doc-image>"]
    assert names and all("disguise" in n for n in names)


def test_prompt_id_env_selects_the_template(monkeypatch):
    monkeypatch.delenv("PROMPT_ID", raising=False)
    assert active_template().id == "direct_v1"
    monkeypatch.setenv("PROMPT_ID", "cot_zeroshot_v1")
    assert active_template().id == "cot_zeroshot_v1"


def test_the_poc_fewshot_template_is_pinned_to_the_direct_exemplar():
    """This one must NOT be variant-matched: the experiment asks whether a
    plain-graph worked example transfers to the map framing."""
    template = load_template("sp_fewshot_direct_ex1_v1")
    for variant in ("direct", "disguise"):
        doc = _doc("shortest_path", variant, "Q: x\nA:")
        names = [Path(i).name for t in template.turns(doc) for i in t.images if i != "<doc-image>"]
        assert names == ["shortest_path_direct_ex1.png"], f"{variant}: {names}"
