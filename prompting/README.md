# `prompting/` — eval-time prompt templates

Everything about **how a benchmark question is put to a model** and **how the answer is read
back** lives here, selected per run by the `PROMPT_ID` environment variable.

```bash
export PROMPT_ID=cot_zeroshot_v1     # default: direct_v1
```

| id | what it does |
|---|---|
| `direct_v1` | The terse baseline. Byte-identical to every campaign through `17_abl_scram`. |
| `cot_zeroshot_v1` | "Think step by step", no exemplars. |
| `cot_fewshot_img_v1` | Two worked examples, each with its own graph image. |

## Why a template owns its parser

`AnswerSpec` pairs `instruction()` with `parse()` in one object. Previously the instruction sat
in each task YAML's `pre_prompt` while the parser lived in the task's `utils.py`, so a reworded
prompt could silently degrade scoring with nothing failing.

This is not hypothetical: `YesNoAnswer` reads the **first** token, which is right for a terse
answer and wrong for a chain of thought, where it would score the model's opening word. The CoT
templates therefore pair themselves with `prefer_last=True` via `cot_answer_spec()`. You cannot
change how a question is asked here without seeing the parser that reads the reply.

## Why `PROMPT_ID` is an environment variable

`process_results(doc, results)` is arity-2 and never receives `lmms_eval_specific_kwargs`
(`lmms_eval/api/task.py`). An env var is the only transport that reaches the text, visual and
scoring hooks alike.

## Scope boundary: templates wrap, they do not reword

`doc["prompt"]` is the question as rendered at dataset-prep time by the sibling `dynamic-dataset`
package. A template wraps it — instruction, reasoning directive, exemplars — but **cannot reword
or regenerate it**, because the endpoint node ids the question interpolates
(`G.graph["entrance"/"exit"/"source"/"sink"]`) are not stored as dataset columns. Rewording the
question itself would first require adding those columns in
`tools/prepare_dynamic_graph_benchmark.py::_row_pair`.

The upside: prompt choice needs **no dataset regeneration**, so a CoT arm and its baseline see
exactly the same graphs.

## Exemplar assets

`assets/<task>_<variant>_ex<k>.{png,yaml}`, regenerated with:

```bash
uv run --no-project --with matplotlib --with networkx --with pillow --with pyyaml \
       --with scipy --with adjustText --with cartopy \
       python prompting/assets/make_exemplars.py
```

Three rules, all enforced by `test/eval/prompt_stability/test_graph_prompts.py`:

1. **No contamination.** Exemplar seeds sit three orders of magnitude above the eval seed space.
2. **Variant-matched.** A disguise question gets disguise exemplars; a plain-graph demo in front
   of a maze question would confound the disguise arm with a few-shot-transfer effect.
3. **Self-consistent.** Each worked solution must parse — with its own template's `AnswerSpec` —
   to its own gold answer, and end in the exact format the instruction asks for. An exemplar that
   reasons correctly but never states the answer teaches the model not to answer.

The generator writes `worked_solution: TODO`; narratives are **written by hand against the
rendered picture**, because a wrong exemplar silently poisons every few-shot run. Seeds are
pinned per task rather than taken blindly from the base — one candidate was rejected for placing
two region labels on top of each other.

## Provenance

Every scored row carries `prompt_id` and `prompt_fingerprint` (a content hash over the template's
rendered text, its budget and its asset bytes). A results tree can therefore state how it was
asked — the gap that made prompt experiments unreproducible.
