"""Turn[] -> what a model backend actually accepts.

Two backends, one source of truth:

  * chat models (vllm) take real multi-turn messages, so exemplars survive.
  * the simple wrappers (internvl3_5 and the other HF paths) have a single
    prompt string, so a multi-turn template is flattened into one annotated
    block. The flattening is reported, never silent -- see `flattened()`.
"""

from __future__ import annotations

from typing import Any, Sequence

from prompting.base import DOC_IMAGE, Turn

_EXEMPLAR_HEADER = "Here are worked examples."
_QUESTION_HEADER = "Now answer this one."


def _resolve(images: Sequence[Any], doc_images: Sequence[Any]) -> list[Any]:
    """Substitute the DOC_IMAGE sentinel; open exemplar paths lazily.

    PIL is imported only when an exemplar is actually present, so text-only use
    (tests, prompt inspection) needs no imaging stack.
    """
    out: list[Any] = []
    for img in images:
        if img == DOC_IMAGE:
            out.extend(doc_images)
        else:
            from PIL import Image

            out.append(Image.open(img).convert("RGB"))
    return out


def to_messages(turns: Sequence[Turn], system: str | None, doc_images: Sequence[Any]) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    for turn in turns:
        content: list[dict] = [{"type": "image", "url": img} for img in _resolve(turn.images, doc_images)]
        content.append({"type": "text", "text": turn.text})
        messages.append({"role": turn.role, "content": content})
    return messages


def to_flat(turns: Sequence[Turn], system: str | None, doc_images: Sequence[Any]) -> tuple[str, list[Any]]:
    """(prompt_text, images) for backends without a multi-turn interface.

    A single user turn with no system message passes through untouched -- this
    is what keeps direct_v1 byte-identical to the historical prompt.
    """
    if system is None and len(turns) == 1 and turns[0].role == "user":
        return turns[0].text, _resolve(turns[0].images, doc_images)

    blocks: list[str] = [system] if system else []
    images: list[Any] = []
    exemplars = turns[:-1]
    if exemplars:
        blocks.append(_EXEMPLAR_HEADER)
        for turn in exemplars:
            images.extend(_resolve(turn.images, doc_images))
            label = "Question" if turn.role == "user" else "Answer"
            blocks.append(f"{label}: {turn.text}")
        blocks.append(_QUESTION_HEADER)
    final = turns[-1]
    images.extend(_resolve(final.images, doc_images))
    blocks.append(final.text)
    return "\n\n".join(blocks), images


def flattened(turns: Sequence[Turn], system: str | None) -> bool:
    """True when to_flat would lose multi-turn structure -- recorded in run_info."""
    return not (system is None and len(turns) == 1)
