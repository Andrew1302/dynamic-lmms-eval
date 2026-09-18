"""Prompt template lookup.

PROMPT_ID is the transport: `process_results` is arity-2 and never receives
`lmms_eval_specific_kwargs`, so an env var is the only channel that reaches all
three task hooks (text, visual, scoring) alike.
"""

from __future__ import annotations

import functools
import os

from prompting.base import PromptTemplate
from prompting.templates.direct import DirectAnswer
from prompting.templates.few_shot import FewShotImageCoT
from prompting.templates.zero_shot import ZeroShotCoT

DEFAULT_PROMPT_ID = "direct_v1"

_TEMPLATES: dict[str, type[PromptTemplate]] = {t.id: t for t in (DirectAnswer, ZeroShotCoT, FewShotImageCoT)}

PROMPT_IDS = sorted(_TEMPLATES)


@functools.lru_cache(maxsize=None)
def load_template(prompt_id: str) -> PromptTemplate:
    """Instantiate by id. Cached: `!function` re-executes a task utils.py once
    per leaf YAML (6x per run), and fingerprinting hashes asset bytes."""
    try:
        return _TEMPLATES[prompt_id]()
    except KeyError:
        raise KeyError(f"unknown PROMPT_ID {prompt_id!r}; known: {PROMPT_IDS}") from None


def active_template() -> PromptTemplate:
    return load_template(os.environ.get("PROMPT_ID", DEFAULT_PROMPT_ID))
