"""Shared wording for the reasoning templates.

Both CoT templates compose their final user turn identically; keeping that here
means the directive and the answer instruction cannot drift between them.
"""

from __future__ import annotations

from prompting.answers import AnswerSpec

COT_DIRECTIVE = "Think step by step about what the image shows, then answer."


def cot_user_text(doc: dict, spec: AnswerSpec) -> str:
    """Question, reasoning directive, answer-format instruction — in that order.

    The trailing "A:" answer cue is stripped: it reads as an instruction to
    answer immediately, and it is the cue models echo as "A: Yes", which the
    scorer then has to undo (see answers._ANSWER_PREFIX_RE).
    """
    question = doc["prompt"].removesuffix("A:").rstrip()
    return f"{question}\n\n{COT_DIRECTIVE}\n{spec.instruction()}"
