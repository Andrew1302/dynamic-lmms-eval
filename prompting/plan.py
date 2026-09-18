"""Resolve (template, model, arm) into one lmms-eval invocation.

This is the decision layer that used to live in 567 lines of bash. It has two
callers -- examples/models/dynamic_graph_benchmark/run_eval.sh on the VMs and
examples/models/dynamic_graph_benchmark/run_local_api.py locally -- which is
why it earns its keep: those two previously duplicated the same knowledge.

The budget contract, which is the reason this is Python and tested:

    the prompt template REQUESTS   (a chain of thought needs room)
    the model profile CONSTRAINS   (the card has a context window)
    a request that cannot fit RAISES, rather than being silently truncated

Silent truncation is what lost ~40% of one campaign's answers; it must be a
crash, not a shrug.

    python -m prompting.plan --pretrained Qwen/Qwen3.5-4B --thinking 0 \
        --tasks "coloring shortest_path" --job-name myjob
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys

from prompting.models import ModelProfile, Thinking, is_thinking_sku, profile_for
from prompting.registry import DEFAULT_PROMPT_ID, load_template

# Window budgeting. The prompt has to fit alongside the generation, and for an
# image benchmark the prompt is mostly vision tokens.
#
# TEXT_PROMPT_HEADROOM is deliberately modest: Gemma's no-think window is only
# 4096 *in total* and ran a full campaign, so a large text allowance would
# reject a configuration already known to work.
#
# Vision tokens are deliberately NOT estimated here. They vary with the render
# and the processor, and an invented per-image constant rejects configurations
# that have demonstrably run. A template that needs an unusual window declares
# it via PromptTemplate.min_window, from a measurement.
TEXT_PROMPT_HEADROOM = 1024

# The task yaml's generation_kwargs. gen_kwargs are only passed on the command
# line when they differ from these, so a plain run keeps the historical argv.
TASK_YAML_GEN_KWARGS = {"max_new_tokens": 64, "temperature": 0.0, "do_sample": False}

VARIANTS = ("direct", "disguise")


class BudgetError(RuntimeError):
    """A template asked for more room than the model can give."""


@dataclasses.dataclass(frozen=True)
class Plan:
    model: str
    model_args: str
    tasks: str
    batch_size: int
    output_path: str
    gen_kwargs: dict
    env: dict

    def argv(self) -> list[str]:
        """The lmms-eval argument vector, minus the `accelerate launch` prefix."""
        out = ["--model", self.model, "--model_args", self.model_args, "--tasks", self.tasks, "--batch_size", str(self.batch_size), "--log_samples", "--output_path", self.output_path]
        if self.gen_kwargs != TASK_YAML_GEN_KWARGS:
            out += ["--gen_kwargs", _format_gen_kwargs(self.gen_kwargs)]
        return out

    def shell(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv())


def _format_gen_kwargs(gen: dict) -> str:
    parts = []
    for key, value in gen.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _task_list(tasks: str) -> str:
    """Expand task names into the explicit direct+disguise subtask list.

    Running the whole group always loads all six subtasks; a single-task dataset
    then has 0 rows for the others and lmms-eval raises IndexError at load.
    """
    return ",".join(f"dynamic_graph_benchmark_{t}_{v}" for t in tasks.split() for v in VARIANTS)


def resolve_budget(template, profile: ModelProfile, thinking: bool) -> dict:
    """Combine what the template asks for with what the model allows."""
    want = template.budget()
    gen = dict(want.as_gen_kwargs())

    if thinking:
        # Thinking is a decode-time mode owned by the model, not the prompt:
        # its budget comes from the profile.
        gen["max_new_tokens"] = profile.think_max_new_tokens
        gen["temperature"] = 0.6
        gen["do_sample"] = True
        if profile.thinking_token_budget is not None and profile.reasoning_parser:
            gen["thinking_token_budget"] = profile.thinking_token_budget
    elif profile.answer_floor:
        # A model that answers verbosely needs room even under a terse prompt.
        gen["max_new_tokens"] = max(gen["max_new_tokens"], profile.answer_floor)

    if profile.backend in ("vllm", "vllm_chat"):
        needed = max(gen["max_new_tokens"] + TEXT_PROMPT_HEADROOM, getattr(template, "min_window", None) or 0)
        window = profile.window(thinking, required=needed)
        if needed > window:
            raise BudgetError(
                f"{template.id} needs a {needed}-token window but {profile.backend} allows {window} "
                f"(generation {gen['max_new_tokens']}, declared minimum {getattr(template, 'min_window', None) or 0}). "
                f"Raise max_model_len/max_model_len_ceiling for this model, or use fewer exemplars."
            )
    return gen


def _vllm_model_args(pretrained: str, profile: ModelProfile, thinking: bool, system_prompt: str | None, terse_directive: str | None, max_images: int = 1, window: int | None = None) -> str:
    args = [
        f"model={pretrained}",
        f"gpu_memory_utilization={profile.gpu_util}",
        f"max_model_len={window or profile.window(thinking)}",
        f"max_num_seqs={profile.max_num_seqs}",
        "enforce_eager=True",
        "dtype=bfloat16",
        "trust_remote_code=True",
        'limit_mm_per_prompt={"image":%d}' % max_images,
    ]
    if profile.thinking is Thinking.R1_SYSTEM_PROMPT:
        if thinking:
            args.append(f"system_prompt={system_prompt}")
    else:
        args.append('chat_template_kwargs={"enable_thinking":%s}' % ("true" if thinking else "false"))
        if thinking and not profile.skip_special_tokens_when_thinking:
            args.append("skip_special_tokens=False")
        if terse_directive:
            args.append(f"reasoning_prompt={terse_directive}")
    if thinking and profile.reasoning_parser:
        args.append(f"reasoning_parser={profile.reasoning_parser}")
    if profile.quantization:
        args.append(f"quantization={profile.quantization}")
    return ",".join(args)


def _window_for(template, profile: ModelProfile, thinking: bool) -> int:
    """The window the engine is actually configured with, after any growth
    needed to hold this template's prompt."""
    needed = max(template.budget().max_new_tokens + TEXT_PROMPT_HEADROOM, getattr(template, "min_window", None) or 0)
    return profile.window(thinking, required=needed)


def build_plan(
    *,
    pretrained: str,
    thinking: bool,
    tasks: str,
    job_name: str,
    prompt_id: str = DEFAULT_PROMPT_ID,
    backend_override: str | None = None,
    system_prompt: str = "internvl_r1",
    force_close: bool = True,
) -> Plan:
    profile = profile_for(pretrained)
    thinking = thinking or is_thinking_sku(pretrained)
    backend = backend_override or profile.backend
    template = load_template(prompt_id)

    if not force_close:
        # Opt out of vllm's native thinking-token budget: without the parser
        # there is nothing to force-close against, so the widened window and the
        # answer allowance come off too (INTERNVL_FORCE_CLOSE=0 in run_eval.sh).
        profile = dataclasses.replace(
            profile,
            reasoning_parser=None,
            thinking_token_budget=None,
            max_model_len_thinking=None,
            # Without a forced close there is no separate answer allowance: the
            # whole generation is the reasoning budget.
            think_max_new_tokens=profile.thinking_token_budget or profile.think_max_new_tokens,
        )

    # A "answer tersely" directive belongs only to a template that is not asking
    # for reasoning; applying both would hand the model contradictory orders.
    terse_directive = None
    if not thinking and not template.expects_reasoning:
        terse_directive = profile.terse_directive

    if backend in ("vllm", "vllm_chat"):
        model_args = _vllm_model_args(pretrained, profile, thinking, system_prompt, terse_directive, template.max_images(), window=_window_for(template, profile, thinking))
        batch_size = profile.batch_size
    elif backend == "internvl3_5":
        # InternVL3 tiles each image up to max_num times (default 12, ~3k vision
        # tokens) and OOMs mid-run on a 12 GiB card; 6 tiles fits.
        model_args = f"pretrained={pretrained},max_num=6" + (",think=1" if thinking else "")
        batch_size = 1
    elif profile.thinking is Thinking.REASONING_PROMPT:
        model_args = f"pretrained={pretrained}" + ("" if thinking else ",reasoning_prompt=\\n/no_think")
        batch_size = 1
    else:
        model_args = f"pretrained={pretrained}"
        batch_size = 1

    # Tuning env is a property of the checkpoint, not of the backend: run_eval.sh
    # exported it before dispatching on the wrapper, so the HF path gets it too.
    env = profile.env()

    return Plan(
        model=backend,
        model_args=model_args,
        tasks=_task_list(tasks),
        batch_size=batch_size,
        output_path=f"./logs/{job_name}",
        gen_kwargs=resolve_budget(template, profile, thinking),
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    # Shell callers read this output with `mapfile` / `read`, which do not strip
    # the CR that Windows text mode would otherwise append to every line -- it
    # would end up inside argv values and env var contents.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--thinking", default="0")
    ap.add_argument("--tasks", default="coloring directed_connectivity shortest_path")
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--prompt-id", default=DEFAULT_PROMPT_ID)
    ap.add_argument("--backend", default=None, help="override the profile's backend (e.g. run InternVL through vllm)")
    ap.add_argument("--system-prompt", default="internvl_r1")
    ap.add_argument("--force-close", default="1", help="0 opts out of the native thinking-token budget")
    ap.add_argument("--format", choices=("shell", "json", "argv", "env"), default="shell")
    args = ap.parse_args(argv)

    plan = build_plan(
        pretrained=args.pretrained,
        thinking=args.thinking == "1",
        tasks=args.tasks,
        job_name=args.job_name,
        prompt_id=args.prompt_id,
        backend_override=args.backend,
        system_prompt=args.system_prompt,
        force_close=args.force_close != "0",
    )
    if args.format == "json":
        print(json.dumps(dataclasses.asdict(plan), indent=2))
    elif args.format == "argv":
        # one argument per line, for `mapfile -t ARGS < <(... --format argv)`;
        # no argument ever contains a newline
        for arg in plan.argv():
            print(arg)
    elif args.format == "env":
        for key, value in plan.env.items():
            print(f"{key}={value}")
    else:
        for key, value in plan.env.items():
            print(f"export {shlex.quote(key)}={shlex.quote(value)}")
        print(plan.shell())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
