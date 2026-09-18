"""Core types for eval-time prompt construction.

A ``PromptTemplate`` owns everything about how a benchmark question is put to a
model: the system message, the conversation turns (text *and* images), the
answer format instruction, the parser that reads the answer back, and the token
budget the format needs. Pairing those in one object is the point — the previous
arrangement scattered them across task YAMLs, a model backend and a bash script,
so changing how a question was asked could silently break how it was scored.

Templates *wrap* the stored question (``doc["prompt"]``, already rendered at
dataset-prep time by the sibling dynamic-dataset package). They cannot reword it:
the endpoint node ids the question interpolates are not stored as dataset
columns. See prompting/README.md.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from prompting.answers import AnswerSpec

# Sentinel for "the graph image belonging to the document under evaluation", as
# opposed to a static exemplar image shipped in prompting/assets/.
DOC_IMAGE = "<doc-image>"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class Turn:
    """One conversation turn. ``images`` holds DOC_IMAGE and/or asset Paths."""

    role: str  # "user" | "assistant"
    text: str
    images: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"Turn.role must be 'user' or 'assistant', got {self.role!r}")


@dataclass(frozen=True)
class TokenBudget:
    """Generation budget a prompt format needs.

    ``max_new_tokens`` is the total output cap. ``thinking_token_budget`` is
    vllm's native reasoning cap (forces the reasoning-end token so the answer is
    never lost to truncation); None disables it. A ModelProfile may only
    *constrain* these, never silently shrink them — see prompting/plan.py.
    """

    max_new_tokens: int
    thinking_token_budget: int | None = None
    temperature: float = 0.0
    do_sample: bool = False

    def as_gen_kwargs(self) -> dict[str, Any]:
        gen: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
        if self.thinking_token_budget is not None:
            gen["thinking_token_budget"] = self.thinking_token_budget
        return gen


# Canonical documents used only to compute a template's fingerprint. Hashing the
# template's *rendered output* for a fixed input captures every static string and
# every referenced asset without making subclasses declare them twice.
CANONICAL_DOCS: tuple[dict[str, Any], ...] = (
    {"task": "coloring", "variant": "direct", "prompt": "Q: CANON\nA:"},
    {"task": "directed_connectivity", "variant": "direct", "prompt": "Q: CANON\nA:"},
    {"task": "shortest_path", "variant": "disguise", "prompt": "Q: CANON\nA:"},
)


class PromptTemplate(ABC):
    """Base class for every prompt condition, including the non-CoT baseline."""

    id: str
    version: int = 1

    # True when the template asks the model to reason before answering. Some
    # models carry a profile-level "answer tersely" directive for their
    # non-reasoning arm; it must not be applied on top of a template that is
    # asking for a chain of thought, or the two instructions contradict.
    expects_reasoning: bool = False

    # --- the four things a subclass defines -------------------------------
    def system(self, doc: dict) -> str | None:
        """System message for this doc, or None for no system turn."""
        return None

    @abstractmethod
    def turns(self, doc: dict) -> list[Turn]:
        """The conversation. The last turn must be the user's real question."""

    def answer_spec(self, task: str) -> AnswerSpec:
        """Instruction + parser for this task. Defaults to the task's natural format."""
        from prompting.answers import default_answer_spec

        return default_answer_spec(task)

    @abstractmethod
    def budget(self) -> TokenBudget:
        """Generation budget this format needs."""

    # --- provided once, never overridden ----------------------------------
    def compose(self, doc: dict, body: str) -> str:
        """Prepend the answer-format instruction to a turn body.

        Kept here so no subclass can forget it and drift from its own
        AnswerSpec. Reproduces the historical `pre_prompt + prompt` byte layout.
        """
        return f"{self.answer_spec(doc['task']).instruction()}\n{body}"

    def fingerprint(self) -> str:
        """Content hash of this template: id, version, rendered text, asset bytes."""
        h = hashlib.sha256()
        h.update(f"{self.id}\x00{self.version}\x00".encode())
        for doc in CANONICAL_DOCS:
            h.update(json.dumps(self.system(doc) or "").encode())
            for turn in self.turns(doc):
                h.update(f"\x00{turn.role}\x00{turn.text}\x00".encode())
                for img in turn.images:
                    h.update(_image_token(img).encode())
        h.update(json.dumps(self.budget().as_gen_kwargs(), sort_keys=True).encode())
        return h.hexdigest()[:16]


def _image_token(img: Any) -> str:
    """Stable identity for an image reference: the sentinel, or a content hash."""
    if img is DOC_IMAGE or img == DOC_IMAGE:
        return DOC_IMAGE
    path = Path(img)
    return f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
