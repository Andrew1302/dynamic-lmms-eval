"""cot_zeroshot_v1 — "think step by step", no exemplars."""

from __future__ import annotations

from prompting.answers import AnswerSpec, cot_answer_spec
from prompting.base import DOC_IMAGE, PromptTemplate, TokenBudget, Turn
from prompting.templates._cot import cot_user_text


class ZeroShotCoT(PromptTemplate):
    id = "cot_zeroshot_v1"

    def answer_spec(self, task: str) -> AnswerSpec:
        return cot_answer_spec(task)

    def turns(self, doc: dict) -> list[Turn]:
        return [Turn("user", cot_user_text(doc, self.answer_spec(doc["task"])), images=(DOC_IMAGE,))]

    def budget(self) -> TokenBudget:
        # Prose reasoning, not a <think> block, so there is no separate thinking
        # budget to cap — the whole chain comes out of max_new_tokens.
        # temperature follows the panel's documented reasoning-mode guidance.
        return TokenBudget(max_new_tokens=2048, temperature=0.6, do_sample=True)
