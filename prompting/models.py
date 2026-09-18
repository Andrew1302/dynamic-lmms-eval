"""Per-model backend and VRAM tuning, as data instead of bash globs.

This replaces the `case "$MODEL_PRETRAINED"` ladders in
examples/models/dynamic_graph_benchmark/run_eval.sh -- 15 glob branches that
mixed backend selection, VRAM tuning, the thinking mechanism and prompt text in
one 567-line script. Every value here is ported verbatim from that script; the
comments explaining *why* each number is what it is come with them, because
they are the expensive part.

A profile only ever *constrains*: it says what the card and the checkpoint can
take. What to ask for comes from the prompt template. prompting/plan.py puts
the two together and fails loudly when they cannot both be satisfied.
"""

from __future__ import annotations

import dataclasses
from enum import Enum


class Thinking(Enum):
    """How a checkpoint is switched into reasoning mode.

    All three panel models toggle thinking on a *single* checkpoint, but the
    mechanism differs per wrapper -- which is exactly why this was a glob
    ladder and is now a field.
    """

    CHAT_TEMPLATE_FLAG = "chat_template_flag"  # enable_thinking, read by the chat template
    R1_SYSTEM_PROMPT = "r1_system_prompt"  # InternVL3.5's documented trigger
    REASONING_PROMPT = "reasoning_prompt"  # HF wrappers: "/no_think" disables
    ALWAYS_ON = "always_on"  # a *-Thinking SKU always reasons


@dataclasses.dataclass(frozen=True)
class ModelProfile:
    """Everything run_eval.sh knew about one checkpoint."""

    match: tuple[str, ...]  # case-insensitive substrings of the HF id
    backend: str  # lmms-eval --model registry name

    # --- vllm engine tuning (ignored by the HF wrappers) -------------------
    gpu_util: float = 0.93
    max_model_len: int = 12288
    max_model_len_thinking: int | None = None  # None => same as max_model_len
    # How far max_model_len may be grown to fit a large prompt (few-shot images
    # are ~6k tokens each). None => no growth: the configured window is a hard
    # limit, because on the 12 GiB cards a wider window costs KV cache the model
    # does not have. Set it only where the headroom has actually been checked.
    max_model_len_ceiling: int | None = None
    max_num_seqs: int = 256
    quantization: str | None = None
    keep_bf16_patterns: str | None = None  # fp8 vision-blindness fix
    expandable_segments: bool = False  # PYTORCH_CUDA_ALLOC_CONF
    batch_size: int = 256
    skip_special_tokens_when_thinking: bool = True
    reasoning_parser: str | None = None  # enables the native thinking budget

    # --- generation ---------------------------------------------------------
    thinking: Thinking = Thinking.CHAT_TEMPLATE_FLAG
    think_max_new_tokens: int = 4096
    thinking_token_budget: int | None = None  # None => no forced close
    answer_floor: int | None = None  # min tokens for the no-think answer
    # Qwen3.5's no-think arm answers with a verbose worked solution (plain prose,
    # not a <think> block) that overflows the terse budget. This directive coerces
    # a concise answer; plan.py applies it only to templates that are NOT asking
    # for reasoning, so it can never contradict a CoT instruction.
    terse_directive: str | None = None

    def window(self, thinking: bool, required: int = 0) -> int:
        base = self.max_model_len_thinking if (thinking and self.max_model_len_thinking is not None) else self.max_model_len
        if required > base and self.max_model_len_ceiling:
            return min(max(base, required), self.max_model_len_ceiling)
        return base

    def env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.keep_bf16_patterns:
            out["FP8_KEEP_BF16_PATTERNS"] = self.keep_bf16_patterns
        if self.expandable_segments:
            out["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return out


# Gemma-4-E2B is mis-marketed: "Effective 2B" but ~9.6 GB on disk, leaving
# < 1 GiB of KV cache on a 12 GiB RTX 4070. Hence the tight window and tiny
# max_num_seqs (vllm otherwise warms up with 256 dummy concurrent requests).
# Its sliding-window attention makes a *long* single request cheap, so the
# thinking arm can afford a 16384 window on the same card.
GEMMA_4_E2B = ModelProfile(
    match=("gemma-4-e2b",),
    backend="vllm",
    gpu_util=0.97,
    max_model_len=4096,
    max_model_len_thinking=16384,
    max_num_seqs=4,
    batch_size=4,
    # Gemma wraps reasoning in the special tokens <|channel>thought ... <channel|>;
    # with skip_special_tokens=True those delimiters vanish and the answer cannot
    # be told apart from the reasoning.
    skip_special_tokens_when_thinking=False,
    think_max_new_tokens=12288,
)

# InternVL3.5-4B via vllm, fp8. In bf16 the 8.88 GiB weights left only a ~9k
# window, truncating the (faithful, verbose) R1 reasoning ~40% of the time on
# hard graphs. fp8 halves the weights so a 16k window fits AND runs faster
# (81 vs 50 tok/s). Only the LM is fp8; the vision tower stays bf16 -- vllm's
# online fp8 otherwise quantizes the ViT and destroys the percept.
INTERNVL3_5 = ModelProfile(
    match=("internvl3",),
    backend="internvl3_5",
    gpu_util=0.92,
    max_model_len=16384,
    max_model_len_thinking=20480,
    max_num_seqs=2,
    quantization="fp8",
    keep_bf16_patterns="vision_model.,mlp1",
    expandable_segments=True,
    thinking=Thinking.R1_SYSTEM_PROMPT,
    # Its LM is Qwen3, so vllm's qwen3 reasoning parser applies: the native
    # budget force-closes </think> instead of losing the answer to truncation.
    reasoning_parser="qwen3",
    thinking_token_budget=12288,
    think_max_new_tokens=13312,
)

# Qwen3.5-4B (dense) is severely KV-starved on a 12 GiB card: at a 12288 window
# only ~0.44 GiB / ~3.2k tokens of KV remain, and it has no sliding-window trick.
# fp8 halves the weights to make the InternVL-matched config fit; both arms run
# fp8 so the comparison differs only in thinking.
QWEN3_5 = ModelProfile(
    match=("qwen3.5",),
    backend="vllm",
    gpu_util=0.92,
    max_model_len=16384,
    max_model_len_thinking=20480,
    max_num_seqs=6,  # fp8 frees ~31k KV tokens: room for far more than InternVL's 2
    quantization="fp8",
    keep_bf16_patterns="visual.",
    expandable_segments=True,
    reasoning_parser="qwen3",
    thinking_token_budget=12288,
    think_max_new_tokens=13312,
    # Qwen's no-think arm answers with a verbose worked solution that overflows
    # the task yaml's 64-token default.
    answer_floor=1024,
    # Literal backslash-n: the vllm wrapper expands these itself when it appends
    # the directive to the user turn.
    terse_directive=r"\n\nReply with only the final answer and nothing else. No reasoning or explanation.",
)

# Qwen3.5-0.8B: small enough that none of the 4B entry's contortions apply.
# bf16 (fp8 buys nothing at this size and adds a variable), a roomy window, and
# no answer floor. Used for fast end-to-end smokes of the prompt layer.
QWEN3_5_SMALL = ModelProfile(
    match=("qwen3.5-0.8b", "qwen3.5-0_8b"),
    backend="vllm",
    gpu_util=0.85,
    max_model_len=16384,
    max_num_seqs=8,
    # A 0.8B leaves plenty of KV headroom on the 12 GiB card, so its window can
    # grow to hold a few-shot prompt (measured ~19.5k for two exemplar images).
    max_model_len_ceiling=32768,
    reasoning_parser="qwen3",
    thinking_token_budget=4096,
    think_max_new_tokens=5120,
)

QWEN3_VL = ModelProfile(match=("qwen3-vl",), backend="qwen3_vl", thinking=Thinking.REASONING_PROMPT, batch_size=1)
QWEN2_5_VL = ModelProfile(match=("qwen2.5-vl", "qwen2_5-vl"), backend="qwen2_5_vl", thinking=Thinking.REASONING_PROMPT, batch_size=1)
LLAVA_OV_1_5 = ModelProfile(match=("llava-onevision-1.5",), backend="llava_onevision1_5", thinking=Thinking.REASONING_PROMPT, batch_size=1)
MINICPM_V = ModelProfile(match=("minicpm-v",), backend="minicpm_v", thinking=Thinking.REASONING_PROMPT, batch_size=1)
LLAMA_VISION = ModelProfile(match=("llama-3.2",), backend="llama_vision", thinking=Thinking.REASONING_PROMPT, batch_size=1)

# Order matters: first substring match wins, exactly as the bash case did.
PROFILES: tuple[ModelProfile, ...] = (
    QWEN3_VL,
    QWEN2_5_VL,
    INTERNVL3_5,
    LLAVA_OV_1_5,
    MINICPM_V,
    LLAMA_VISION,
    QWEN3_5_SMALL,
    QWEN3_5,
    GEMMA_4_E2B,
)


def profile_for(pretrained: str) -> ModelProfile:
    lowered = pretrained.lower()
    for profile in PROFILES:
        if any(token in lowered for token in profile.match):
            return profile
    raise KeyError(f"no ModelProfile matches {pretrained!r}; add one to prompting/models.py")


def is_thinking_sku(pretrained: str) -> bool:
    """A *-Thinking checkpoint always reasons, whatever THINKING says."""
    return "thinking" in pretrained.lower()
