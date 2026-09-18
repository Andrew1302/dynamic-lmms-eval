"""direct_v1 — the terse baseline every campaign through 17_abl_scram used.

Byte-identical to the historical `pre_prompt + doc["prompt"]` composition. It is
pinned by a golden test against prompts recorded in real result logs; if this
template's output moves, every existing campaign number becomes incomparable.
"""

from __future__ import annotations

from prompting.base import DOC_IMAGE, PromptTemplate, TokenBudget, Turn


class DirectAnswer(PromptTemplate):
    id = "direct_v1"

    def turns(self, doc: dict) -> list[Turn]:
        return [Turn("user", self.compose(doc, doc["prompt"]), images=(DOC_IMAGE,))]

    def budget(self) -> TokenBudget:
        # Matches the task YAML's historical generation_kwargs: a single integer
        # or a Yes/No needs nothing more.
        return TokenBudget(max_new_tokens=64)
