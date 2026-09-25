"""One template class. The prompts themselves live in prompting/library/*.yaml.

The point is that you can answer "what exactly gets sent?" by reading one file,
without reading any Python. A library file contains the literal prompt with two
placeholders:

    {question}     the stored dataset question, with the trailing "A:" answer
                   cue kept or stripped per strip_answer_cue
    {instruction}  the answer-format sentence, from answer_style

and an optional list of worked-example stems, which may themselves contain
{task} / {variant} to follow the document.

    python -m prompting.show sp_explain_v1 --variant disguise

prints the fully rendered payload for a sample document.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import yaml

from prompting.answers import AnswerSpec, cot_answer_spec, default_answer_spec
from prompting.base import ASSETS_DIR, DOC_IMAGE, PromptTemplate, TokenBudget, Turn

LIBRARY_DIR = Path(__file__).resolve().parent / "library"

ANSWER_STYLES = {"terse": default_answer_spec, "cot": cot_answer_spec}


@dataclasses.dataclass(frozen=True)
class Exemplar:
    task: str
    variant: str
    question: str
    answer: str
    worked_solution: str
    seed: int
    image: Path


@functools.lru_cache(maxsize=None)
def load_exemplar(stem: str) -> Exemplar:
    """Load one worked example by asset stem, e.g. shortest_path_direct_ex1."""
    sidecar = ASSETS_DIR / f"{stem}.yaml"
    if not sidecar.is_file():
        available = sorted(p.stem for p in ASSETS_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no exemplar {stem!r} in {ASSETS_DIR}; have: {available}")
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    image = sidecar.with_suffix(".png")
    if not image.is_file():
        raise FileNotFoundError(f"exemplar image missing for {stem}: {image}")
    return Exemplar(
        task=meta["task"],
        variant=meta["variant"],
        question=meta["question"],
        answer=str(meta["answer"]),
        worked_solution=meta["worked_solution"],
        seed=int(meta["seed"]),
        image=image,
    )


class FileTemplate(PromptTemplate):
    """A prompt template defined by a library YAML file."""

    def __init__(self, spec: dict):
        self._spec = spec
        self.id = spec["id"]
        self.version = int(spec.get("version", 1))
        self.summary = spec.get("summary", "").strip()
        self.expects_reasoning = spec["answer_style"] == "cot"
        self.min_window = spec.get("min_window")

    # --- the pieces the base class asks for -------------------------------
    def system(self, doc: dict) -> str | None:
        return self._spec.get("system")

    def answer_spec(self, task: str) -> AnswerSpec:
        return ANSWER_STYLES[self._spec["answer_style"]](task)

    def budget(self) -> TokenBudget:
        b = self._spec.get("budget", {})
        return TokenBudget(
            max_new_tokens=int(b.get("max_new_tokens", 64)),
            thinking_token_budget=b.get("thinking_token_budget"),
            temperature=float(b.get("temperature", 0.0)),
            do_sample=bool(b.get("do_sample", False)),
        )

    def exemplar_stems(self, doc: dict) -> list[str]:
        return [s.format(task=doc["task"], variant=doc["variant"]) for s in self._spec.get("exemplars", [])]

    def turns(self, doc: dict) -> list[Turn]:
        spec = self.answer_spec(doc["task"])
        turns: list[Turn] = []
        for stem in self.exemplar_stems(doc):
            ex = load_exemplar(stem)
            # An exemplar from another task keeps its own answer format: its
            # worked solution ends in that task's answer, not the document's.
            turns.append(Turn("user", self._render(ex.question, self.answer_spec(ex.task)), images=(ex.image,)))
            turns.append(Turn("assistant", ex.worked_solution))
        turns.append(Turn("user", self._render(doc["prompt"], spec), images=(DOC_IMAGE,)))
        return turns

    # --- rendering ---------------------------------------------------------
    def _render(self, question: str, spec: AnswerSpec) -> str:
        if self._spec.get("strip_answer_cue"):
            # "A:" reads as "answer now" and is what models echo back as
            # "A: Yes", which the scorer then has to undo.
            question = question.removesuffix("A:").rstrip()
        return self._spec["user_turn"].format(question=question, instruction=spec.instruction())


@functools.lru_cache(maxsize=None)
def load_library() -> dict[str, FileTemplate]:
    out: dict[str, FileTemplate] = {}
    for path in sorted(LIBRARY_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if spec["id"] != path.stem:
            raise ValueError(f"{path.name}: id {spec['id']!r} does not match the filename")
        if spec["answer_style"] not in ANSWER_STYLES:
            raise ValueError(f"{path.name}: answer_style must be one of {sorted(ANSWER_STYLES)}")
        out[spec["id"]] = FileTemplate(spec)
    return out
