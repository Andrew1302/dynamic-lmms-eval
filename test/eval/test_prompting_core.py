"""Unit tests for the prompting package: answers, base types, render, registry.

Nothing here runs a model. Everything is pure string/structure work, which is
the point -- the prompt layer is deliberately inference-free so it can be
regression-tested in under a second.
"""

import re
from pathlib import Path

import pytest
import yaml

from prompting import render
from prompting.answers import (
    INTEGER_INSTRUCTION,
    YESNO_INSTRUCTION,
    IntegerAnswer,
    YesNoAnswer,
    cot_answer_spec,
    default_answer_spec,
)
from prompting.base import CANONICAL_DOCS, DOC_IMAGE, PromptTemplate, TokenBudget, Turn
from prompting.registry import (
    DEFAULT_PROMPT_ID,
    PROMPT_IDS,
    active_template,
    load_template,
)
from prompting.file_template import LIBRARY_DIR, load_exemplar

REPO = Path(__file__).resolve().parents[2]
TASK_DIR = REPO / "lmms_eval" / "tasks" / "dynamic_graph_benchmark"
ASSETS = REPO / "prompting" / "assets"
VARIANTS = ("direct", "disguise")
TASKS = ("coloring", "directed_connectivity", "shortest_path")
# Eval graphs are seeded from base 42; exemplars must sit far outside that
# space so a worked example can never be a question the model is scored on.
EVAL_SEED_CEILING = 100_000_000
STEMS = sorted(p.stem for p in ASSETS.glob("*.yaml"))


# ==========================================================================
# answers.py
# ==========================================================================
def test_default_instructions_are_the_historical_strings():
    """These exact strings were the pre_prompt in every shipped campaign."""
    assert default_answer_spec("coloring").instruction() == INTEGER_INSTRUCTION
    assert default_answer_spec("shortest_path").instruction() == INTEGER_INSTRUCTION
    assert default_answer_spec("directed_connectivity").instruction() == YESNO_INSTRUCTION
    assert INTEGER_INSTRUCTION == "Answer with a single integer and nothing else."
    assert YESNO_INSTRUCTION == "Answer with only 'Yes' or 'No'."


def test_default_answer_spec_rejects_unknown_task():
    with pytest.raises(KeyError, match="no answer format registered"):
        default_answer_spec("connectivity")  # the deleted undirected task


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3", "3"),
        ("  3  ", "3"),
        ("A: 4", "4"),
        ("Answer: 4", "4"),
        ("final answer: 12", "12"),
        ("-2", "-2"),
        ("", ""),
        ("no digits here", "no digits here"),
        # multi-integer: an explicit final statement wins over position
        ("Nodes 0,1,2 form a triangle. The chromatic number is 4.", "4"),
        ("colors 1 2 3 used; answer: 3", "3"),
        # multi-integer, no statement: a last line holding one int wins
        ("step 5 then 8\n7", "7"),
    ],
)
def test_integer_parse(raw, expected):
    assert IntegerAnswer().parse(raw) == expected


def test_integer_prefer_last_only_changes_the_tiebreak():
    """shortest_path echoes vertex ids, so a first-int rule locks onto one."""
    ambiguous = "from vertex 0 to vertex 4 costs 9 then 13"
    assert IntegerAnswer(prefer_last=False).parse(ambiguous) == "0"
    assert IntegerAnswer(prefer_last=True).parse(ambiguous) == "13"
    # a single integer is unambiguous either way
    assert IntegerAnswer(prefer_last=False).parse("7") == "7"
    assert IntegerAnswer(prefer_last=True).parse("7") == "7"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Yes", "yes"),
        ("no", "no"),
        ("A: Yes", "yes"),
        ("True", "yes"),
        ("f", "no"),
        ("*? No.", "no"),
        ("", ""),
    ],
)
def test_yesno_parse(raw, expected):
    assert YesNoAnswer().parse(raw) == expected


def test_yesno_length_cap_stops_long_prose_flipping_on_a_stray_word():
    """The <=120 char unique-word fallback must not apply to reasoning prose."""
    assert YesNoAnswer().parse("hmm... no.") == "no"
    long_prose = "x" * 130 + " yes"
    assert YesNoAnswer().parse(long_prose) not in ("yes", "no")


def test_yesno_prefer_last_reads_the_conclusion_not_the_opening():
    chain = "No obvious path at first.\nTracing 5 -> 1 -> 4.\nYes"
    assert YesNoAnswer(prefer_last=False).parse(chain) == "no"
    assert YesNoAnswer(prefer_last=True).parse(chain) == "yes"


def test_yesno_prefer_last_falls_back_to_last_mention():
    """Last line holds no verdict -> use the last one stated anywhere."""
    assert YesNoAnswer(prefer_last=True).parse("first no, later yes.\nDone.") == "yes"
    assert YesNoAnswer(prefer_last=True).parse("nothing conclusive") == "nothing"


def test_with_instruction_keeps_the_parser():
    spec = default_answer_spec("coloring").with_instruction("Say a number.")
    assert spec.instruction() == "Say a number."
    assert spec.parse("The chromatic number is 4.") == "4"


def test_cot_answer_spec_changes_instruction_and_tiebreak_together():
    """Instruction and parser must move as one -- that is the whole design."""
    for task in TASKS:
        terse, cot = default_answer_spec(task), cot_answer_spec(task)
        assert cot.instruction() != terse.instruction()
        assert "last line" in cot.instruction()
        assert cot.prefer_last is True
        assert type(cot) is type(terse)  # same parser family


def test_answer_specs_are_frozen():
    with pytest.raises(Exception):
        default_answer_spec("coloring").instruction_text = "mutated"


# ==========================================================================
# base.py
# ==========================================================================
def test_turn_rejects_a_bad_role():
    Turn("user", "x")
    Turn("assistant", "x")
    with pytest.raises(ValueError, match="must be 'user' or 'assistant'"):
        Turn("system", "x")


def test_token_budget_gen_kwargs():
    assert TokenBudget(max_new_tokens=64).as_gen_kwargs() == {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "do_sample": False,
    }
    g = TokenBudget(max_new_tokens=100, thinking_token_budget=50, temperature=0.6, do_sample=True).as_gen_kwargs()
    assert g["thinking_token_budget"] == 50
    assert g["temperature"] == 0.6
    assert g["do_sample"] is True


def test_canonical_docs_cover_every_task():
    assert {d["task"] for d in CANONICAL_DOCS} == set(TASKS)


def test_fingerprint_reacts_to_text_version_and_budget():
    class Base(PromptTemplate):
        id = "t"

        def turns(self, doc):
            return [Turn("user", "hello", images=(DOC_IMAGE,))]

        def budget(self):
            return TokenBudget(max_new_tokens=8)

    class ChangedText(Base):
        def turns(self, doc):
            return [Turn("user", "hello!", images=(DOC_IMAGE,))]

    class ChangedBudget(Base):
        def budget(self):
            return TokenBudget(max_new_tokens=9)

    class ChangedVersion(Base):
        version = 2

    base = Base().fingerprint()
    assert base == Base().fingerprint()
    for other in (ChangedText(), ChangedBudget(), ChangedVersion()):
        assert other.fingerprint() != base


# ==========================================================================
# render.py
# ==========================================================================
def _single(text="q"):
    return [Turn("user", text, images=(DOC_IMAGE,))]


def test_to_flat_single_turn_passes_through_unchanged():
    """This identity is what keeps direct_v1 byte-compatible."""
    text, images = render.to_flat(_single("Answer.\nQ: x\nA:"), None, ["IMG"])
    assert text == "Answer.\nQ: x\nA:"
    assert images == ["IMG"]


def test_to_flat_multi_turn_annotates_and_keeps_image_order():
    exemplar_png = next(ASSETS.glob("*.png"))
    turns = [
        Turn("user", "ex-q", images=(exemplar_png,)),
        Turn("assistant", "ex-a"),
        Turn("user", "real-q", images=(DOC_IMAGE,)),
    ]
    text, images = render.to_flat(turns, "SYS", ["DOC"])
    assert text.startswith("SYS")
    assert "Question: ex-q" in text
    assert "Answer: ex-a" in text
    assert text.rstrip().endswith("real-q")
    assert len(images) == 2
    assert images[-1] == "DOC"  # doc image last, after the exemplar


def test_to_flat_folds_a_system_message_in():
    text, _ = render.to_flat(_single("q"), "SYS", [])
    assert text.startswith("SYS")
    assert text.rstrip().endswith("q")


def test_to_messages_puts_images_before_text():
    msgs = render.to_messages(_single("q"), None, ["IMG"])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert [c["type"] for c in msgs[0]["content"]] == ["image", "text"]
    assert msgs[0]["content"][0]["url"] == "IMG"


def test_to_messages_emits_a_system_turn():
    msgs = render.to_messages(_single(), "SYS", [])
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"][0]["text"] == "SYS"


def test_doc_image_sentinel_expands_to_every_doc_image():
    msgs = render.to_messages(_single(), None, ["A", "B"])
    assert [c["url"] for c in msgs[0]["content"] if c["type"] == "image"] == ["A", "B"]


def test_no_image_ablation_leaves_the_text_turn_intact():
    msgs = render.to_messages(_single(), None, [])
    assert [c["type"] for c in msgs[0]["content"]] == ["text"]


def test_assistant_turns_carry_no_images():
    msgs = render.to_messages([Turn("assistant", "a")], None, ["IMG"])
    assert [c["type"] for c in msgs[0]["content"]] == ["text"]


# ==========================================================================
# registry.py
# ==========================================================================
def test_registry_contents_and_default():
    """The library grows as experiments are added, so assert the core templates
    are present rather than pinning an exact set -- but the default must not move,
    because it is what every historical campaign used."""
    assert {"direct_v1", "cot_zeroshot_v1", "cot_fewshot_img_v1"} <= set(PROMPT_IDS)
    assert DEFAULT_PROMPT_ID == "direct_v1"


def test_unknown_prompt_id_fails_loudly_and_lists_the_known_ones():
    with pytest.raises(KeyError) as e:
        load_template("cot_v99")
    assert "cot_v99" in str(e.value)
    assert "direct_v1" in str(e.value)


def test_load_template_is_cached():
    """utils.py re-executes once per leaf YAML; fingerprinting hashes assets."""
    assert load_template("direct_v1") is load_template("direct_v1")


def test_active_template_reads_the_env(monkeypatch):
    monkeypatch.delenv("PROMPT_ID", raising=False)
    assert active_template().id == DEFAULT_PROMPT_ID
    for pid in PROMPT_IDS:
        monkeypatch.setenv("PROMPT_ID", pid)
        assert active_template().id == pid


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_every_template_declares_a_usable_budget(prompt_id):
    b = load_template(prompt_id).budget()
    assert b.max_new_tokens > 0
    if b.thinking_token_budget is not None:
        assert b.max_new_tokens > b.thinking_token_budget, "no room left for the answer"


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
@pytest.mark.parametrize("task", TASKS)
def test_every_template_renders_every_task_and_variant(prompt_id, task):
    template = load_template(prompt_id)
    for variant in VARIANTS:
        doc = {"task": task, "variant": variant, "prompt": "Q: x\nA:"}
        turns = template.turns(doc)
        assert turns, "a template must produce at least one turn"
        assert turns[-1].role == "user", "the last turn must be the real question"
        assert DOC_IMAGE in turns[-1].images
        assert template.answer_spec(task).instruction() in turns[-1].text


def test_cot_templates_strip_the_trailing_answer_cue():
    """'A:' reads as 'answer now' and is what models echo as 'A: Yes'."""
    doc = {"task": "coloring", "variant": "direct", "prompt": "Q: x\nA:"}
    assert load_template("direct_v1").turns(doc)[-1].text.endswith("A:")
    for pid in ("cot_zeroshot_v1", "cot_fewshot_img_v1"):
        assert not load_template(pid).turns(doc)[-1].text.rstrip().endswith("A:")


# ==========================================================================
# exemplar assets
# ==========================================================================








def test_exemplar_prompt_text_is_ascii():
    """Prompt text crosses shells, env vars and tokenizers; keep it predictable."""
    for f in sorted(ASSETS.glob("*.yaml")):
        meta = yaml.safe_load(f.read_text(encoding="utf-8"))
        offenders = {c for c in meta["worked_solution"] if ord(c) > 127}
        assert not offenders, f"{f.name} has non-ascii {offenders}"


def test_exemplar_sidecars_have_the_expected_schema():
    required = {"task", "variant", "seed", "question", "answer", "worked_solution"}
    for f in sorted(ASSETS.glob("*.yaml")):
        assert required <= set(yaml.safe_load(f.read_text(encoding="utf-8"))), f.name


def test_no_source_file_carries_a_stray_control_character():
    """A shell-escaping slip once turned a regex word-boundary escape into a
    literal backspace, silently disabling the match. Cheap to keep checking."""
    for root in (REPO / "prompting", TASK_DIR, Path(__file__).parent):
        for f in root.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".yaml", ".json", ".md"):
                ctrl = {c for c in f.read_bytes() if c < 9 or (13 < c < 32)}
                assert not ctrl, f"{f} contains control bytes {ctrl}"


# ==========================================================================
# wiring: the things that break a deploy rather than a computation
# ==========================================================================
def test_prompting_is_declared_as_a_shipped_package():
    """include = ['lmms_eval*'] alone silently omits a new top-level package."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    line = re.search(r"^include = \[.*\]$", text, re.M).group(0)
    assert "prompting*" in line


def test_every_generated_conf_uploads_the_prompting_package():
    """Missing here => a clean deploy, then ModuleNotFoundError on the VM."""
    confs = sorted((REPO / "remote_execution_scripts" / "jobs" / "graph_benchmark").glob("*.conf"))
    assert confs, "no generated confs found"
    missing = [c.name for c in confs if '"prompting"' not in c.read_text(encoding="utf-8")]
    assert not missing, f"{len(missing)} confs would not ship prompting/: {missing[:3]}"


def test_leaf_task_yamls_no_longer_carry_prompt_text():
    """The instruction comes from AnswerSpec now; a stale pre_prompt would double it."""
    for f in TASK_DIR.glob("dynamic_graph_benchmark_*_*.yaml"):
        body = f.read_text(encoding="utf-8")
        assert "pre_prompt" not in body, f.name
        assert "lmms_eval_specific_kwargs" not in body, f.name


def test_task_yamls_only_reference_functions_that_exist():
    utils_src = (TASK_DIR / "utils.py").read_text(encoding="utf-8")
    defined = set(re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)", utils_src, re.M))
    referenced = set()
    for f in list(TASK_DIR.glob("*.yaml")) + [TASK_DIR / "_default_template_yaml"]:
        if f.exists():
            referenced |= set(re.findall(r"!function utils\.([a-zA-Z_][a-zA-Z0-9_]*)", f.read_text(encoding="utf-8")))
    assert referenced, "no !function references found -- the test is not looking where it thinks"
    assert referenced <= defined, f"YAMLs reference missing utils functions: {sorted(referenced - defined)}"


def test_the_deleted_undirected_connectivity_task_is_fully_gone():
    """It matched 0 rows: dynamic-dataset registers no undirected connectivity task."""
    for p in TASK_DIR.glob("*connectivity*"):
        assert "directed_connectivity" in p.name, f"stale task file {p.name}"
    group = (TASK_DIR / "dynamic_graph_benchmark.yaml").read_text(encoding="utf-8")
    for line in group.splitlines():
        if "connectivity" in line:
            assert "directed_connectivity" in line


@pytest.mark.parametrize("stem", STEMS)
def test_every_exemplar_asset_is_usable(stem):
    """Every shipped worked example must load, be uncontaminated, and end in the
    exact answer format its own instruction asks for."""
    ex = load_exemplar(stem)
    spec = cot_answer_spec(ex.task)
    assert ex.image.exists()
    assert "TODO" not in ex.worked_solution, f"{stem} still has a placeholder solution"
    assert ex.seed > EVAL_SEED_CEILING, f"{stem} seed {ex.seed} may collide with eval graphs"
    assert spec.parse(ex.worked_solution) == spec.parse(ex.answer), f"{stem} does not reach its own gold answer"
    assert not any(ord(c) > 127 for c in ex.worked_solution), f"{stem} has non-ascii prompt text"


def test_the_library_covers_every_template_id():
    """PROMPT_IDS is derived from the files on disk, so it cannot drift."""
    from prompting.registry import PROMPT_IDS

    assert set(PROMPT_IDS) == {p.stem for p in LIBRARY_DIR.glob("*.yaml")}
