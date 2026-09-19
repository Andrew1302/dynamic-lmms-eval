"""Prompt template lookup.

Templates are declared in prompting/library/*.yaml -- one file per template,
containing the literal prompt. Adding a condition means adding a file; there is
no Python to write and no second place to look.

PROMPT_ID is the transport: process_results is arity-2 and never receives
lmms_eval_specific_kwargs, so an env var is the only channel that reaches all
three task hooks (text, visual, scoring) alike.
"""

from __future__ import annotations

import functools
import os

from prompting.base import PromptTemplate
from prompting.file_template import load_library

DEFAULT_PROMPT_ID = "direct_v1"


def _library() -> dict[str, PromptTemplate]:
    return load_library()


@functools.lru_cache(maxsize=None)
def load_template(prompt_id: str) -> PromptTemplate:
    """Look up a template by id. Cached: `!function` re-executes a task's
    utils.py once per leaf YAML, and fingerprinting hashes asset bytes."""
    lib = _library()
    try:
        return lib[prompt_id]
    except KeyError:
        raise KeyError(f"unknown PROMPT_ID {prompt_id!r}; known: {sorted(lib)}") from None


def active_template() -> PromptTemplate:
    return load_template(os.environ.get("PROMPT_ID", DEFAULT_PROMPT_ID))


def __getattr__(name):
    # PROMPT_IDS is derived from the library on disk, so it cannot drift from it.
    if name == "PROMPT_IDS":
        return sorted(_library())
    raise AttributeError(name)
