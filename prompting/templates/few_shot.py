"""cot_fewshot_img_v1 — worked examples carrying their own graph images.

Each exemplar is a PNG plus a YAML sidecar in prompting/assets/:

    coloring_direct_ex1.png
    coloring_direct_ex1.yaml     task / variant / question / worked_solution / seed

Exemplars are **variant-matched**: a disguise question gets disguise exemplars.
A plain-graph demo in front of a maze or map question would shift the domain
between the example and the question, confounding the disguise arm with a
few-shot-transfer effect.

Exemplars MUST be rendered from a seed disjoint from every eval seed — see
prompting/assets/README.md. `test_no_exemplar_contamination` enforces it.

Only the chat path can carry real multi-turn exemplars; the simple wrappers get
an annotated flattening (prompting/render.py), which is recorded in run_info.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import yaml

from prompting.answers import AnswerSpec, cot_answer_spec
from prompting.base import ASSETS_DIR, DOC_IMAGE, PromptTemplate, TokenBudget, Turn
from prompting.templates._cot import cot_user_text

N_EXEMPLARS = 2
EXEMPLAR_TASKS = ("coloring", "directed_connectivity", "shortest_path")


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
def load_exemplars(task: str, variant: str) -> tuple[Exemplar, ...]:
    """Exemplars for a (task, variant), ordered by filename. Fails loudly if absent."""
    out = []
    for sidecar in sorted(ASSETS_DIR.glob(f"{task}_{variant}_ex*.yaml")):
        meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        image = sidecar.with_suffix(".png")
        if not image.exists():
            raise FileNotFoundError(f"exemplar image missing for {sidecar.name}: {image}")
        out.append(Exemplar(task=meta["task"], variant=meta["variant"], question=meta["question"], answer=str(meta["answer"]), worked_solution=meta["worked_solution"], seed=int(meta["seed"]), image=image))
    if len(out) < N_EXEMPLARS:
        raise FileNotFoundError(f"need {N_EXEMPLARS} exemplars for {task!r}/{variant!r}, found {len(out)} in {ASSETS_DIR}")
    return tuple(out[:N_EXEMPLARS])


class FewShotImageCoT(PromptTemplate):
    id = "cot_fewshot_img_v1"
    expects_reasoning = True
    # Measured on vm03: a 3-image request (two exemplars + the document) came to
    # 18,602 prompt tokens, which overflowed a 16,384 window mid-run. 24,576
    # holds that plus the 2,048-token generation with room to spare.
    min_window = 24576

    def answer_spec(self, task: str) -> AnswerSpec:
        return cot_answer_spec(task)

    def turns(self, doc: dict) -> list[Turn]:
        spec = self.answer_spec(doc["task"])
        turns: list[Turn] = []
        for ex in load_exemplars(doc["task"], doc["variant"]):
            turns.append(Turn("user", cot_user_text({"prompt": ex.question}, spec), images=(ex.image,)))
            turns.append(Turn("assistant", ex.worked_solution))
        turns.append(Turn("user", cot_user_text(doc, spec), images=(DOC_IMAGE,)))
        return turns

    def budget(self) -> TokenBudget:
        return TokenBudget(max_new_tokens=2048, temperature=0.6, do_sample=True)
