"""prompting/plan.py must reproduce what run_eval.sh actually did.

The fixture in fixtures/legacy_run_eval_argv.json was captured mechanically
from the real script by tools/devtools/capture_run_eval_argv.sh (stub
`accelerate` on PATH, print the argv instead of launching). Porting 15 bash
glob branches by reading them is not reviewable; replaying them is.

The contract is *semantic* equivalence, not byte-identical argv: run_eval.sh
omits --gen_kwargs entirely when the task yaml default already applies, so the
comparison merges those defaults in on both sides before comparing.
"""

import json
from pathlib import Path

import pytest

from prompting.models import profile_for
from prompting.plan import TASK_YAML_GEN_KWARGS, BudgetError, build_plan, resolve_budget
from prompting.registry import load_template

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_run_eval_argv.json").read_text(encoding="utf-8"))
CELLS = sorted(FIXTURE)


def _parse_legacy(argv):
    """Turn a captured argv into the same shape build_plan produces."""
    flags = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and "=" not in tok:
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                flags[tok] = argv[i + 1]
                i += 2
                continue
            flags[tok] = True
        i += 1
    gen = dict(TASK_YAML_GEN_KWARGS)
    for part in flags.get("--gen_kwargs", "").split(","):
        if not part:
            continue
        key, _, value = part.partition("=")
        if value in ("true", "false"):
            value = value == "true"
        elif value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        gen[key] = value
    return {
        "model": flags["--model"],
        "model_args": flags["--model_args"],
        "tasks": flags["--tasks"],
        "batch_size": int(flags["--batch_size"]),
        "output_path": flags["--output_path"],
        "gen_kwargs": gen,
    }


def _plan_for(cell):
    pretrained, thinking, force_close, backend, variant = (p.split("=", 1)[-1] if "=" in p else p for p in cell.split("|"))
    return build_plan(
        pretrained=pretrained,
        thinking=thinking == "1",
        tasks="coloring directed_connectivity shortest_path",
        job_name="argvcapture",
        backend_override=backend or None,
        system_prompt=variant or "internvl_r1",
        force_close=force_close != "0",
    )


@pytest.mark.parametrize("cell", CELLS)
def test_plan_matches_the_legacy_script(cell):
    legacy = _parse_legacy(FIXTURE[cell]["argv"])
    plan = _plan_for(cell)
    assert plan.model == legacy["model"], cell
    assert plan.tasks == legacy["tasks"], cell
    assert plan.batch_size == legacy["batch_size"], cell
    assert plan.output_path == legacy["output_path"], cell
    assert plan.gen_kwargs == legacy["gen_kwargs"], cell
    # model_args order is the script's; compare as a set of settings
    assert sorted(plan.model_args.split(",")) == sorted(legacy["model_args"].split(",")), cell


@pytest.mark.parametrize("cell", CELLS)
def test_exported_env_matches_the_legacy_script(cell):
    """FP8_KEEP_BF16_PATTERNS is the fp8 vision-blindness fix; losing it
    silently blinds the model, which is how a whole campaign was wasted."""
    assert _plan_for(cell).env == FIXTURE[cell]["env"], cell


def test_the_fixture_actually_covers_the_panel():
    models = {cell.split("|")[0] for cell in CELLS}
    assert {"Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it", "OpenGVLab/InternVL3_5-4B"} <= models
    assert any("think=1" in c for c in CELLS) and any("think=0" in c for c in CELLS)


# ==========================================================================
# the new behaviour the port exists to enable
# ==========================================================================
def test_a_template_that_cannot_fit_the_window_raises():
    """Silent truncation lost ~40% of one campaign's answers. It must crash."""

    class Greedy:
        id = "greedy"

        def budget(self):
            from prompting.base import TokenBudget

            return TokenBudget(max_new_tokens=999_999)

    with pytest.raises(BudgetError, match="needs a"):
        resolve_budget(Greedy(), profile_for("google/gemma-4-E2B-it"), thinking=False)


def test_the_answer_floor_lifts_a_terse_budget_but_never_lowers_one():
    qwen = profile_for("Qwen/Qwen3.5-4B")
    gemma = profile_for("google/gemma-4-E2B-it")
    direct = load_template("direct_v1")
    assert resolve_budget(direct, qwen, thinking=False)["max_new_tokens"] == 1024  # floor applies
    assert resolve_budget(direct, gemma, thinking=False)["max_new_tokens"] == 64  # no floor
    cot = load_template("cot_zeroshot_v1")
    assert resolve_budget(cot, qwen, thinking=False)["max_new_tokens"] == 2048  # template wins


def test_cot_templates_get_their_requested_budget_onto_the_command_line():
    plan = build_plan(pretrained="Qwen/Qwen3.5-4B", thinking=False, tasks="coloring", job_name="j", prompt_id="cot_zeroshot_v1")
    assert "--gen_kwargs" in plan.argv()
    assert "max_new_tokens=2048" in plan.shell()


def test_direct_v1_on_a_plain_model_omits_gen_kwargs_entirely():
    """It equals the task yaml default, so the historical argv is preserved."""
    plan = build_plan(pretrained="google/gemma-4-E2B-it", thinking=False, tasks="coloring", job_name="j")
    assert "--gen_kwargs" not in plan.argv()


def test_prompt_id_does_not_perturb_engine_tuning():
    """A template shapes the prompt and the budget, never the engine config."""
    kwargs = dict(pretrained="Qwen/Qwen3.5-0.8B", thinking=False, tasks="coloring", job_name="j")
    a = build_plan(prompt_id="direct_v1", **kwargs)
    b = build_plan(prompt_id="cot_fewshot_img_v1", **kwargs)

    # Two engine settings are legitimately prompt-derived: the terse directive
    # and the image limit. Everything else is the card's business, not the
    # prompt's.
    def engine(plan):
        # three settings are legitimately prompt-derived: the terse directive,
        # the image limit, and the window (grown to hold a few-shot prompt)
        return sorted(p for p in plan.model_args.split(",") if not p.startswith(("reasoning_prompt=", "limit_mm_per_prompt=", "max_model_len=")))

    assert engine(a) == engine(b)
    assert a.env == b.env
    assert a.batch_size == b.batch_size
    assert a.gen_kwargs != b.gen_kwargs  # the budget is the template's business


def test_the_image_limit_follows_the_template():
    """vllm rejects a request carrying more images than limit_mm_per_prompt
    allows, so pinning it at 1 made few-shot exemplars unrunnable."""
    kwargs = dict(pretrained="Qwen/Qwen3.5-0.8B", thinking=False, tasks="coloring", job_name="j")
    assert 'limit_mm_per_prompt={"image":1}' in build_plan(prompt_id="direct_v1", **kwargs).model_args
    assert 'limit_mm_per_prompt={"image":1}' in build_plan(prompt_id="cot_zeroshot_v1", **kwargs).model_args
    # two exemplar images plus the document's own
    assert 'limit_mm_per_prompt={"image":3}' in build_plan(prompt_id="cot_fewshot_img_v1", **kwargs).model_args


def test_the_terse_directive_is_never_applied_to_a_reasoning_template():
    """Qwen3.5's no-think arm carries 'reply with only the final answer'. Handing
    that to a CoT prompt would give the model two contradictory instructions --
    the collision this refactor exists to make impossible."""
    kwargs = dict(pretrained="Qwen/Qwen3.5-0.8B", thinking=False, tasks="coloring", job_name="j")
    assert "Reply with only the final answer" not in build_plan(prompt_id="cot_zeroshot_v1", **kwargs).model_args
    kwargs = dict(pretrained="Qwen/Qwen3.5-4B", thinking=False, tasks="coloring", job_name="j")
    assert "Reply with only the final answer" in build_plan(prompt_id="direct_v1", **kwargs).model_args
    for cot in ("cot_zeroshot_v1",):
        assert "Reply with only the final answer" not in build_plan(prompt_id=cot, **kwargs).model_args


def test_task_expansion_covers_both_variants():
    plan = build_plan(pretrained="Qwen/Qwen3.5-4B", thinking=False, tasks="coloring shortest_path", job_name="j")
    assert plan.tasks == ("dynamic_graph_benchmark_coloring_direct,dynamic_graph_benchmark_coloring_disguise," "dynamic_graph_benchmark_shortest_path_direct,dynamic_graph_benchmark_shortest_path_disguise")


def test_a_thinking_sku_reasons_even_when_thinking_is_off():
    plan = build_plan(pretrained="Qwen/Qwen3-VL-4B-Thinking", thinking=False, tasks="coloring", job_name="j")
    assert "no_think" not in plan.model_args


def test_unknown_checkpoint_fails_with_a_pointer():
    with pytest.raises(KeyError, match="prompting/models.py"):
        build_plan(pretrained="acme/mystery-7b", thinking=False, tasks="coloring", job_name="j")


def test_a_few_shot_prompt_grows_the_window_only_where_the_card_allows():
    """The few-shot prompt measured 18,602 tokens on the VM and overflowed a
    16,384 window mid-run. The template now declares what it needs; a small
    checkpoint has the KV headroom to grant it, the 4B does not and must say so
    rather than dying inside vllm after the model has loaded."""
    small = build_plan(pretrained="Qwen/Qwen3.5-0.8B", thinking=False, tasks="coloring", job_name="j", prompt_id="cot_fewshot_img_v1")
    assert "max_model_len=24576" in small.model_args

    with pytest.raises(BudgetError, match="needs a 24576-token window"):
        build_plan(pretrained="Qwen/Qwen3.5-4B", thinking=False, tasks="coloring", job_name="j", prompt_id="cot_fewshot_img_v1")


def test_single_image_templates_do_not_grow_the_window():
    """Growth must be driven by need, or every run's engine config would drift."""
    for pid in ("direct_v1", "cot_zeroshot_v1"):
        plan = build_plan(pretrained="Qwen/Qwen3.5-4B", thinking=False, tasks="coloring", job_name="j", prompt_id=pid)
        assert "max_model_len=16384" in plan.model_args, pid
