import re
from typing import List, Optional, Union


def strip_reasoning_tags(text: str, tag_pairs: List[List[str]]) -> str:
    """Remove reasoning tag blocks from model output.

    Args:
        text: Raw model output string
        tag_pairs: List of [start_tag, end_tag] pairs,
                   e.g. [["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]

    Returns:
        Cleaned text with reasoning blocks removed.

    Notes:
        Handles three shapes per tag pair:
        - balanced (`<think>...</think>...answer`): strip the wrapped block.
        - close-only (`...reasoning...</think>...answer`): chat templates that
          inject the opening `<think>\\n` into the prompt (e.g. Qwen3 with
          enable_thinking=True) emit only the close tag in the generated text.
          Treat everything up to and including the first close tag as reasoning.
        - open-only (`<think>...reasoning` truncated by max_new_tokens):
          everything from the open tag onward is unfinished reasoning with no
          answer to recover; drop it.
    """
    result = text
    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag) :]
            else:
                break
        if start_tag not in result and end_tag in result:
            end = result.find(end_tag)
            result = result[end + len(end_tag) :]
        if start_tag in result and end_tag not in result:
            result = result[: result.find(start_tag)]
    return result.strip()


def parse_reasoning_tags_config(cli_value: Optional[str] = None, task_value: Optional[object] = None) -> Optional[List[List[str]]]:
    """Resolve reasoning_tags from CLI + task config.

    Priority: task_value > cli_value.
    "none" / None = disabled.
    """
    import json

    effective = task_value if task_value is not None else cli_value
    if effective is None or effective == "none" or effective is False:
        return None
    if isinstance(effective, str):
        return json.loads(effective)
    return effective
