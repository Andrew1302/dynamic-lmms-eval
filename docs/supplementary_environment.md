# Supplementary: experimental environment and reproducibility (VM03)

Machine profile captured live from VM03 on **2026-07-31**; software/config values
read from the repository state described in [Code versions](#code-versions).
VM03 is the primary evaluation host for the dynamic graph benchmark campaigns.

---

## 1. Hardware

| Item | Value |
|---|---|
| Host | `vm03` (lab GPU workstation), reached over SSH |
| CPU | 13th Gen Intel(R) Core(TM) i5-13400F — 10 cores (6 P-cores + 4 E-cores), 16 threads, 1 socket |
| CPU clocks | 0.80 GHz min / 4.60 GHz max |
| L3 cache | 20 MiB |
| System RAM | 62 GiB usable (+1 GiB swap) |
| GPU | 1 × NVIDIA GeForce RTX 4070 |
| GPU memory | 12282 MiB (12 GiB) |
| GPU compute capability | 8.9 (Ada Lovelace) |
| GPU power cap | 200 W |
| GPU count used per job | 1 (no tensor/data parallelism) |
| Storage | 1.9 TB SSD (`/dev/sda2`) mounted at `/media/vm03/ssd1T` |

The 12 GiB VRAM budget is the binding constraint behind every per-model
precision/context choice in [§5](#5-inference-configuration) and [§6](#6-thinking-vs-no-thinking-configuration-matrix).

## 2. Operating system and toolchain

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | Linux 6.8.0-124-generic, x86_64 |
| glibc | 2.35 |
| gcc | 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3) |
| NVIDIA driver | 560.35.03 |
| CUDA driver API | 12.6 |
| CUDA runtime | supplied by the PyTorch wheel (cu128 / CUDA 12.8); no system-wide `nvcc` |
| cuDNN | 9.10.02 (bundled with the PyTorch wheel) |

## 3. Python environment

Dependency management: **uv 0.9.16**, project virtualenv at `<repo>/.venv`.
`pyproject.toml` declares `requires-python = ">=3.10"`; the resolved interpreter
on VM03 is **CPython 3.12.11**. (only cite this, don't need to talk about uv)

Reproduce with:

```bash
uv sync                              # resolves against the committed uv.lock
uv pip install adjustText cartopy    # job REMOTE_SETUP_CMD extras (map disguise renderer)
```

Lockfile identity at capture time: `uv.lock` MD5 `df9c947d7e26aa0c7ce2f64e77efbefd`,
`pyproject.toml` MD5 `683138872322f1ebc386f977a54d1353`.

262 distributions are installed. Versions relevant to the results:

| Package | Version | | Package | Version |
|---|---|---|---|---|
| torch | 2.10.0+cu128 | | transformers | 5.8.1 |
| torchvision | 0.25.0 | | tokenizers | 0.22.2 |
| torchaudio | 2.10.0 | | accelerate | 1.12.0 |
| vllm | 0.19.1 | | safetensors | 0.7.0 |
| triton | 3.6.0 | | datasets | 4.6.1 |
| flashinfer-python | 0.6.6 | | pyarrow | 23.0.1 |
| numpy | 2.2.6 | | pandas | 3.0.1 |
| scipy | 1.17.1 | | pillow | 12.1.1 |
| networkx | 3.6.1 | | matplotlib | 3.10.8 |
| timm | 1.0.25 | | einops | 0.8.2 |
| qwen-vl-utils | 0.0.14 | | sentencepiece | 0.2.1 |
| protobuf | 5.29.6 | | openai | 2.24.0 |
| av | 15.1.0 | | decord | 0.6.0 |

Neither `flash-attn` nor `xformers` is installed; vLLM runs with
`enforce_eager=True` and FlashInfer available.

### Environment variables

```

Set conditionally by `examples/models/dynamic_graph_benchmark/run_eval.sh`:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # InternVL3.5 and Qwen3.5 (both arms)
FP8_KEEP_BF16_PATTERNS=vision_model.,mlp1          # InternVL3.5 (both arms)
FP8_KEEP_BF16_PATTERNS=visual.                     # Qwen3.5      (both arms)
```

## 4. Models

Evaluated panel, pinned to the exact Hugging Face revisions present in the VM03
cache:

| Short name | Hugging Face id | Revision (commit) |
|---|---|---|
| `internvl35_4b` | `OpenGVLab/InternVL3_5-4B` | `481f6e32467eab4e922ccd7fd6cf420441a62331` |
| `qwen35_4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `gemma4_e2b` | `google/gemma-4-E2B-it` | `179516f0c449474fdc46f08f30ead5b11e178497` |


## 5. Inference configuration

All three models are served through **vLLM 0.19.1** (`lmms-eval --model vllm`),
launched with `accelerate launch --num_processes=1` — one GPU, no tensor or data
parallelism. Shared engine arguments:

```
enforce_eager=True
dtype=bfloat16
trust_remote_code=True
limit_mm_per_prompt={"image": 1}
seed=1                                 # vLLM engine seed, hard-coded in the wrapper
```

Per-model engine settings (no-thinking arm; thinking-arm deltas in [§6](#6-thinking-vs-no-thinking-configuration-matrix)):

| Model | Weight precision | `gpu_memory_utilization` | `max_model_len` | `max_num_seqs` | `batch_size` |
|---|---|---|---|---|---|
| InternVL3.5-4B | fp8 LM + bf16 vision tower | 0.92 | 16384 | 2 | 256 |
| Qwen3.5-4B | fp8 LM + bf16 vision tower | 0.92 | 16384 | 6 | 256 |
| Gemma-4-E2B-it | bf16 (whole model) | 0.97 | 4096 | 4 | 4 |

`batch_size` is a submission-side knob only: vLLM's continuous batching caps
real concurrency at `max_num_seqs`, and every sequence is decoded independently,
so the batch size does not affect outputs.

# NOTE: It's important to explain that the fp8 wasn't applied to internVl's ViT

### Why fp8 for two of the three models

In bf16, InternVL3.5-4B's 8.88 GiB of weights leave roughly a 9k-token context
on a 12 GiB card, which truncated its (faithful, verbose) reasoning on about 40 %
of hard graphs. Online fp8 quantization halves the weights to ~4.4 GiB, which
buys a 16k-token window *and* runs faster (81 vs. 50 tok/s measured). CPU
offloading was tried and rejected: it restored the window at ~1.4 tok/s (~36×
slower). Qwen3.5-4B is dense and has no sliding-window trick, so it is even more
KV-starved and receives the same treatment. Gemma-4-E2B stays bf16 — its
sliding-window attention makes a wide context cheap without quantization.

Both arms of the thinking ablation use identical precision for a given model, so
the think/no-think contrast is never confounded by quantization.

### fp8 vision-tower fix (`fp8_keep_bf16`)

vLLM's online fp8 path quantizes *every* `LinearBase` module it finds, including
the vision transformer. Probing on VM03 (2026-07-14) showed this destroys the
visual percept outright: the bf16 model describes a maze render correctly while
the fp8 model reports "a stack of grey fabric storage bags" regardless of input.
vLLM protects InternVL's ViT for AWQ checkpoints but has no equivalent guard for
online fp8.

`lmms_eval/vllm_plugins.py::fp8_keep_bf16` closes the gap. It is registered as a
`vllm.general_plugins` entry point (necessary because the v1 engine core runs in
a spawned subprocess, where an in-process monkeypatch would never reach model
construction) and patches `Fp8Config.get_quant_method` to return
`UnquantizedLinearMethod` for any linear whose module prefix matches a pattern in
`FP8_KEEP_BF16_PATTERNS`. Matching modules stay bf16; the language model keeps
the exact fp8 profile of the earlier campaigns. The variable is unset for Gemma,
making the plugin a no-op there.

**Caveat for the write-up:** results produced *before* this fix landed with an
fp8 InternVL3.5 or Qwen3.5 and an image input are effectively blind-model runs
and must not be reported as sighted conditions. We reran everything after we catch the error, so all results are correct

## 6. Thinking vs. no-thinking configuration matrix

The `THINKING` environment variable selects the arm. All three models toggle
reasoning on a *single* checkpoint — no separate "-Thinking" SKU is used — but
the trigger mechanism, the token budget, and the answer-extraction path differ
per model. Everything below is implemented in
`examples/models/dynamic_graph_benchmark/run_eval.sh`, keyed on
`MODEL_PRETRAINED`.

### 6.1 What is identical across both arms

| Held constant | Value |
|---|---|
| Weight precision | Per-model, as in §5 (fp8 for InternVL/Qwen, bf16 for Gemma) — **same in both arms** |
| `FP8_KEEP_BF16_PATTERNS` | Same in both arms (vision tower always bf16 where fp8 applies) |
| `gpu_memory_utilization` | Same in both arms |
| `max_num_seqs`, `batch_size` | Same in both arms |
| Dataset, seed, prompt text, image | Identical (same prepare fingerprint) |

The only per-arm changes are the reasoning trigger, the context window (widened
purely to fit longer generations), the generation budget, and the decoding
temperature.

### 6.2 Per-model differences

| | **InternVL3.5-4B** | **Qwen3.5-4B** | **Gemma-4-E2B-it** |
|---|---|---|---|
| Precision (both arms) | fp8 LM + bf16 ViT | fp8 LM + bf16 ViT | bf16 |
| Thinking trigger | R1 **system prompt** (`system_prompt=internvl_r1`) — the model has no `enable_thinking` flag | `chat_template_kwargs={"enable_thinking":true}` | `chat_template_kwargs={"enable_thinking":true}` |
| No-think mechanism | System prompt simply omitted (no system message at all) | `chat_template_kwargs={"enable_thinking":false}` **plus** a terseness directive appended to the user turn | `chat_template_kwargs={"enable_thinking":false}` |
| `max_model_len` — no-think | 16384 | 16384 | 4096 |
| `max_model_len` — think | **20480** | **20480** | **16384** |
| Reasoning parser (think only) | `reasoning_parser=qwen3` (its LM is Qwen3, so the `<think>` delimiters apply) | `reasoning_parser=qwen3` | none |
| Native thinking-token budget (think only) | `thinking_token_budget=12288` (force-closes `</think>`) | `thinking_token_budget=12288` | not available (no parser) |
| `max_new_tokens` — think | 13312 (= 12288 thinking + 1024 answer) | 13312 (= 12288 + 1024) | 12288 |
| `max_new_tokens` — no-think | 64 (task-YAML default) | **1024** | 64 (task-YAML default) |
| Special-token handling | default | default | **`skip_special_tokens=False` in the think arm only** |
| Reasoning delimiters stripped before scoring | `<think>` … `</think>` | `<think>` … `</think>` | `<\|channel>` … `<channel\|>` |
| Decoding — no-think | greedy (`temperature=0`, `do_sample=false`) | greedy | greedy |
| Decoding — think | `temperature=0.6`, `do_sample=true` | `temperature=0.6`, `do_sample=true` | `temperature=0.6`, `do_sample=true` |

NOTE: I thin in text simply said 12288. Go with that. Don't extra detail what may contradict the main paper

### 6.3 Rationale for each per-model deviation

- **InternVL3.5 — R1 system prompt.** InternVL3.5 exposes no `enable_thinking`
  chat-template flag; its documented thinking trigger is an R1-style system
  prompt instructing the model to enclose its analysis in `<think>`/`</think>`
  and then give a standalone answer. It is passed as the preset token
  `system_prompt=internvl_r1` because the literal multi-line string cannot
  survive the comma/newline-delimited `model_args` parser. Alternative prompt
  variants (`internvl_r1_v1` … `_v4`) exist for prompt-tuning experiments; the
  committed default is `internvl_r1`.
- **InternVL3.5 — force-close.** Without a reasoning parser, ~40 % of hard-graph
  generations produced a 12k-token unclosed `<think>` block whose reasoning
  *contained* the right answer but never emitted it. Enabling
  `reasoning_parser=qwen3` lets vLLM force-close `</think>` at the 12288-token
  budget, reserving 1024 tokens for the answer. `INTERNVL_FORCE_CLOSE=0` opts
  out (and then falls back to `max_new_tokens=12288` with no budget).
- **Qwen3.5 — force-close.** Qwen3.5 over-deliberates hardest (~40 % truncated
  even at 12288 tokens); the same native budget scheme brings answer loss to
  ~0 %.
- **Qwen3.5 — no-think terseness directive.** With `enable_thinking=false`,
  Qwen3.5 genuinely suppresses reasoning (0/24 `<think>` blocks in the smoke
  test) but answers with a verbose worked solution in plain prose. That prose is
  not a reasoning block, so it cannot be stripped; at the 64-token default it was
  simply truncated mid-answer. Two fixes apply together: the no-think budget is
  raised to 1024 tokens, and a directive ("Reply with only the final answer and
  nothing else. No reasoning or explanation.") is appended to the user turn via
  `reasoning_prompt`. The directive is applied **only** to Qwen3.5 and **only**
  in the no-think arm, so the thinking arm remains free to reason.
- **Gemma-4-E2B — `skip_special_tokens=False`.** Gemma wraps its reasoning in
  the special tokens `<|channel>thought … <channel|>` rather than `<think>` tags.
  With the default `skip_special_tokens=True` those delimiters are dropped from
  the decoded text and the answer regex reads the reasoning instead of the
  answer. Preserving them lets the task YAML's `reasoning_tags` isolate the
  answer. Needed only in Gemma's thinking arm.
- **Gemma-4-E2B — window widening.** Gemma's 4096 window was sized for terse
  no-think answers and truncates reasoning. Its sliding-window attention makes a
  long single request cheap (KV grows with the window, not the sequence), so the
  think arm widens to 16384 with a 12288-token generation budget while
  `max_num_seqs` stays small.
- **Context widening is comparability-safe.** All window changes are pure
  sequence-length knobs: inputs, image tiling, and precision are untouched, so
  accuracy remains comparable across arms and models.

### 6.4 Non-vLLM (HuggingFace) wrappers

Some earlier / auxiliary runs use HF wrappers rather than vLLM. For those,
no-think is expressed as `reasoning_prompt=\n/no_think` appended to the prompt,
and the InternVL HF wrapper takes `think=1` plus `max_num=6` (tile cap, needed to
avoid OOM in attention softmax on the 12 GiB card). The reported panel runs
through vLLM.

## 7. Seeds and determinism

| Component | Seed / setting |
|---|---|
| Dataset generation (global) | `SEED=42` (`run_eval.sh` default → `prepare_dynamic_graph_benchmark.py --seed`) |
| Per-sample seed, standard mode | `seed_i = 42 + i*1000 + (crc32(task_name) mod 1000)` |
| Per-sample seed, sweep mode | `base_i = 42 + counter*7919`; edge-count targeting retries at `base_i + attempt*101` |
| Graph layout | `networkx.spring_layout(G, seed=42)` |
| vLLM engine | `seed=1` (`lmms_eval/models/simple/vllm.py`) |
| Decoding, no-thinking arm | `temperature=0`, `top_p=1.0`, `do_sample=false`, `num_beams=1` — **deterministic** |
| Decoding, thinking arm | `temperature=0.6`, `do_sample=true` — **stochastic** |

Sample *i* is a pure function of `(seed, task, i)`, so generation is
prefix-stable: a run can be extended with `--start-index` and the first *N*
samples remain byte-identical. Each prepared dataset directory stores a
**fingerprint** of all generation arguments *plus* a hash of the benchmark
generator code; an identical fingerprint guarantees byte-identical rows and
short-circuits regeneration.

**Reproducibility caveat.** The no-thinking arm is greedy and reproduces exactly
given the same weights, vLLM version, and engine configuration. The thinking arm
samples at `temperature=0.6` (the official per-model guidance for these models'
reasoning modes) and therefore reproduces only up to sampling noise; repeated
runs will differ.

## 8. Stimuli

Graph images are rendered with matplotlib: `figsize=(6, 6)` at `dpi=120`,
i.e. **720 × 720 px PNG**, one image per prompt (`limit_mm_per_prompt={"image":1}`).
Disk-persisted copies are written with `dpi=(120, 120)`.

## 9. Code versions

| Repository | Commit | Date |
|---|---|---|
| `dynamic-lmms-eval` | `23a020c9bdc34bfc8dcfd56ea2713337ba4b535e` | 2026-07-16 |
| `dynamic-dataset` | `a11aa60161fc1461dda250b10f01231bb7d1471d` | 2026-07-16 |

> **Open item.** `remote_execution_scripts/01_deploy.sh` rsyncs the local
> *working tree* to the VM rather than checking out a git ref, so VM03's own
> checkout reports a stale HEAD (`40d7bf59`, 2026-03-04) with modified files. At
> capture time the local working tree also carried 55 uncommitted changes in
> `dynamic-lmms-eval` and 5 in `dynamic-dataset`. No single commit hash therefore
> describes the code that produced the current results. Commit or tag the working
> tree and cite that hash here before submission.

## 10. Reproduction checklist

1. Provision a machine matching §1–§2 (a 12 GiB Ada-class GPU is assumed by every
   memory setting; larger cards can override `VLLM_GPU_UTIL_OVERRIDE`,
   `VLLM_MAX_MODEL_LEN_OVERRIDE`, `VLLM_MAX_NUM_SEQS_OVERRIDE`).
2. Clone `dynamic-lmms-eval` and its sibling `dynamic-dataset` at the commits in §9.
3. `uv sync && uv pip install adjustText cartopy`.
4. Fetch the model revisions in §4.
5. Export the environment of §3.
6. Run a job configuration from `remote_execution_scripts/jobs/graph_benchmark/`;
   each `.conf` fixes the campaign, sample count, difficulty, task set, and
   ablation flags, and dispatches `run_eval.sh`, which regenerates the dataset
   (fingerprint-checked) before invoking `lmms-eval`.
