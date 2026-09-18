"""Answer formats: the instruction that asks for one, and the parser that reads it.

These two are deliberately one object. They were previously separated — the
instruction sat in each task YAML's ``pre_prompt`` while the parser lived in the
task's ``utils.py`` — so a reworded prompt could silently degrade scoring with
nothing failing. A template cannot now change how it asks without seeing the
parser that reads the reply.

The parsing logic is moved **verbatim** from
``lmms_eval/tasks/dynamic_graph_benchmark/utils.py``; the regexes below are the
ones benchmarked against the thinkadj sample logs (see their comments). The only
behavioural change: an unrecognised task now raises instead of silently
lower-casing the prediction, which previously masked malformed rows.
"""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod

# Models (InternVL3.5 especially) echo the prompt's answer cue, producing
# "A: Yes" / "Answer: 4". The first-token rule read that as "a" and scored a
# correct answer 0 — 66 connectivity think-arm answers in the thinkadj n=100
# run alone. Strip the cue before token rules.
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:a|answer|final answer)\s*[:\-]\s*", re.IGNORECASE)

_YES_PATTERNS = {"yes", "y", "true", "t"}
_NO_PATTERNS = {"no", "n", "false", "f"}

# Verbose worked solutions bury the answer mid-prose where neither first- nor
# last-integer is reliable (node ids, color-class listings, edge weights all
# emit integers). An explicit final-answer statement ("the chromatic number
# is 4", "answer: 14", "\boxed{3}") is; take the LAST such statement — the
# model's standing claim. The copula (is|=|:|{) is required so question echoes
# ("path total from vertex 0") don't match. Benchmarked on all thinkadj
# jsonls: coloring +160/-20 vs first-int, shortest_path ±2, connectivity
# +66/-0 (with the prefix strip + unique-word fallback below).
_FINAL_STMT_RE = re.compile(
    r"(?:final answer|answer|chromatic number(?: of the graph)?"
    r"|minimum number of colors(?: needed| required)?"
    r"|\bchi\b(?:\(g\))?|boxed"
    r"|minimum total driving time|minimum[- ]weight path total)"
    r"[^.\d\n]{0,20}?(?:is|=|:|\{)\s*\**\s*(-?\d+)",
    re.IGNORECASE,
)

INTEGER_INSTRUCTION = "Answer with a single integer and nothing else."
YESNO_INSTRUCTION = "Answer with only 'Yes' or 'No'."


class AnswerSpec(ABC):
    """An answer format: how to ask for it, and how to read it back."""

    instruction_text: str

    def instruction(self) -> str:
        return self.instruction_text

    @abstractmethod
    def parse(self, raw: str) -> str:
        """Normalize a model response (or a gold answer) to comparable form."""

    def with_instruction(self, text: str) -> "AnswerSpec":
        """Same parser, different wording — what a CoT template needs."""
        return dataclasses.replace(self, instruction_text=text)


@dataclasses.dataclass(frozen=True)
class IntegerAnswer(AnswerSpec):
    instruction_text: str = INTEGER_INSTRUCTION
    # Prompts that make the model echo vertex indices ("from vertex X to
    # vertex Y") would lock onto a vertex id under a first-integer rule; the
    # last integer is far more often the real answer there.
    prefer_last: bool = False

    def parse(self, raw: str) -> str:
        pred = _strip_cue(raw)
        ints = re.findall(r"-?\d+", pred)
        if not ints:
            return pred.lower()
        if len(ints) == 1:
            return ints[0]
        stmts = list(_FINAL_STMT_RE.finditer(pred))
        if stmts:
            return stmts[-1].group(1)
        last_line_ints = re.findall(r"-?\d+", pred.strip().splitlines()[-1])
        if len(last_line_ints) == 1:
            return last_line_ints[0]
        return ints[-1] if self.prefer_last else ints[0]


@dataclasses.dataclass(frozen=True)
class YesNoAnswer(AnswerSpec):
    instruction_text: str = YESNO_INSTRUCTION
    # Terse answers put the verdict first; a chain of thought puts it LAST, and
    # the first-token rule would score the model's opening word. A template that
    # asks for reasoning must therefore pair itself with prefer_last=True.
    prefer_last: bool = False

    def parse(self, raw: str) -> str:
        pred = _strip_cue(raw)
        if self.prefer_last:
            tail = self._from_tail(pred)
            if tail is not None:
                return tail
        first = pred.lower().split()
        token = re.sub(r"[^a-z]", "", first[0]) if first else ""
        if token in _YES_PATTERNS:
            return "yes"
        if token in _NO_PATTERNS:
            return "no"
        # Short answers whose first token is noise ("*? No.") but that contain
        # exactly one yes/no word are unambiguous. Length-capped so long
        # reasoning prose can't flip on an incidental "yes".
        if len(pred) <= 120:
            words = set(re.findall(r"\b(yes|no)\b", pred.lower()))
            if len(words) == 1:
                return words.pop()
        return token

    @staticmethod
    def _from_tail(pred: str) -> str | None:
        """Read the verdict from the end: last line first, else last mention."""
        lines = [ln for ln in pred.strip().splitlines() if ln.strip()]
        if lines:
            in_last = re.findall(r"\b(yes|no)\b", lines[-1].lower())
            if len(set(in_last)) == 1:
                return in_last[-1]
        anywhere = re.findall(r"\b(yes|no)\b", pred.lower())
        return anywhere[-1] if anywhere else None


def _strip_cue(raw: str) -> str:
    return _ANSWER_PREFIX_RE.sub("", (raw or "").strip())


# The benchmark's task set is closed: dynamic-dataset registers exactly these
# three (src/benchmark/tasks/__init__.py).
_DEFAULTS: dict[str, AnswerSpec] = {
    "coloring": IntegerAnswer(),
    "shortest_path": IntegerAnswer(prefer_last=True),
    "directed_connectivity": YesNoAnswer(),
}


def default_answer_spec(task: str) -> AnswerSpec:
    """The terse format every campaign up to and including 17_abl_scram used."""
    try:
        return _DEFAULTS[task]
    except KeyError:
        raise KeyError(f"no answer format registered for task {task!r}; known: {sorted(_DEFAULTS)}") from None


COT_TAIL = "When you have finished reasoning, state your final answer on the last line"

_COT_TAIL_BY_KIND = {
    IntegerAnswer: f"{COT_TAIL} as a single integer.",
    YesNoAnswer: f"{COT_TAIL} as only 'Yes' or 'No'.",
}


def cot_answer_spec(task: str) -> AnswerSpec:
    """Same parsers, retuned to read the answer from the END of a chain of thought.

    Shared by every reasoning template so the instruction wording and the
    prefer_last switch can never drift apart.
    """
    spec = default_answer_spec(task)
    return dataclasses.replace(
        spec,
        instruction_text=_COT_TAIL_BY_KIND[type(spec)],
        prefer_last=True,
    )
