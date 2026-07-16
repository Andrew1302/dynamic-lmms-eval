"""vLLM general plugins registered by lmms_eval (see pyproject entry points).

fp8_keep_bf16 — keep vision-tower linears in bf16 under online fp8.

vLLM's online fp8 path (``quantization=fp8`` with a bf16 checkpoint) quantizes
every ``LinearBase`` it can find, including the vision transformer. On
Qwen3.5-4B and InternVL3.5-4B this destroys the visual percept outright:
probed 2026-07-14 on VM03, the bf16 model describes a maze render correctly
while the fp8 model sees "a stack of grey fabric storage bags" regardless of
input (project memory: fp8-vision-blindness). vLLM already protects InternVL's
ViT for AWQ checkpoints (``_patch_quant_config`` adds ``vision_model`` to
``modules_to_not_convert``) but has no equivalent for online fp8 — this plugin
fills that gap.

Gated by the ``FP8_KEEP_BF16_PATTERNS`` env var: a comma-separated list of
substrings matched against each layer's module prefix (e.g. ``visual.`` for
Qwen3.5, ``vision_model.,mlp1`` for InternVL3.5). Matching linears get
``UnquantizedLinearMethod`` (stay bf16); everything else quantizes as before,
so the language model keeps the exact fp8 profile of the earlier campaigns.
Unset or empty → no-op.

Registered as a ``vllm.general_plugins`` entry point because the v1 engine
core runs in a spawned subprocess: an in-process monkeypatch never reaches
model construction, but general plugins load in every vLLM process (and env
vars propagate to spawned children).
"""

from __future__ import annotations

import os


def fp8_keep_bf16() -> None:
    patterns = [p.strip() for p in os.environ.get("FP8_KEEP_BF16_PATTERNS", "").split(",") if p.strip()]
    if not patterns:
        return

    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import fp8 as _fp8

    # Plugins load once per process but may be re-imported; make idempotent.
    if getattr(_fp8.Fp8Config, "_lmms_keep_bf16_patterns", None) == patterns:
        return

    _orig_get_quant_method = _fp8.Fp8Config.get_quant_method

    def _patched_get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and any(p in prefix for p in patterns):
            return UnquantizedLinearMethod()
        return _orig_get_quant_method(self, layer, prefix)

    _fp8.Fp8Config.get_quant_method = _patched_get_quant_method
    _fp8.Fp8Config._lmms_keep_bf16_patterns = patterns
    print(f"[lmms_eval.vllm_plugins] fp8_keep_bf16 active, patterns={patterns}", flush=True)
