#!/usr/bin/env python
"""Local launcher for API-backed models (Gemini, OpenAI, Azure OpenAI) on the
dynamic graph benchmark.

This is the Windows-friendly local analogue of ``run_eval.sh``: it reuses the exact
same dataset prep (``tools/prepare_dynamic_graph_benchmark.py``), the same task YAMLs,
and the same registered model wrappers + lmms-eval — but drops the VM-only machinery
(SSH deploy, ``ln -s`` symlink, ``accelerate launch``, chunking). API models need no
GPU, so running them locally is both cheaper and simpler than a VM round-trip.

Post-fix guarantees (these ship in the sibling ``dynamic-dataset`` renderer + this
prepare, so a fresh local prep is automatically post-fix):
  * occlusion-aware direct render + sp-map disambiguation (no absorbed edges);
  * χ-controlled coloring is UNCONDITIONAL — every coloring dataset plants the
    chromatic number linearly across {2, 3, 4} (sample i cycles 2→3→4), so the
    answer distribution is exactly uniform. There is no flag to turn this off.
The fp8 vision fix is irrelevant here (API models are not quantized).

It writes the prepared dataset straight into the canonical path the task YAMLs load
(``./dynamic_graph_benchmark_data``, ``load_from_disk: True``), so no symlink is needed.
Difficulties are run sequentially: prepare -> eval -> prepare (overwrite) -> eval -> ...

Credentials are read from the repo ``.env`` (auto-loaded) or the environment.
  * Gemini      (``--model gemini_api``): GOOGLE_API_KEY
  * OpenAI      (``--model openai``):     OPENAI_API_KEY  [+ optional OPENAI_API_BASE]
  * Azure OpenAI(``--model openai --azure``): AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_BASE (endpoint). The aboda-openai resource is the new
        Azure OpenAI v1 API (serves at the endpoint ROOT, Bearer auth, no
        api-version), so --azure just points the plain OpenAI client at that
        endpoint + key — the classic /openai/deployments/…?api-version client
        404s there. ``--model-version`` is the deployment name (e.g.
        gpt-5.6-terra); keep "gpt-5" in it so the wrapper routes reasoning SKUs
        correctly (drops temperature, uses max_completion_tokens).

Usage (from the repo root, with the dedicated API venv's python — .venv-api,
which holds the api-only stack: torch/transformers/openai/accelerate + the
dynamic-dataset render deps, minus vllm):

    # Azure gpt-5.6-terra — 1-sample smoke on easy, all 3 standard tasks (~6 calls):
    .venv-api\\Scripts\\python examples/models/dynamic_graph_benchmark/run_local_api.py \\
        --model openai --azure --model-version gpt-5.6-terra \\
        --num-samples 1 --difficulties easy

    # Full standard ablation — 100 samples/diff × 3 diffs × 3 tasks × 2 surfaces:
    .venv-api\\Scripts\\python examples/models/dynamic_graph_benchmark/run_local_api.py \\
        --model openai --azure --model-version gpt-5.6-terra --num-samples 100

Defaults mirror the standard campaign (coloring + directed_connectivity +
shortest_path; numeric labels, node color #AED6F1, straight edges, seed 42,
no adjacency list, no thinking).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root = three levels up from examples/models/dynamic_graph_benchmark/.
REPO_ROOT = Path(__file__).resolve().parents[3]
# The path the task YAMLs load via load_from_disk (see _default_template_yaml).
CANONICAL_DATASET_DIR = "./dynamic_graph_benchmark_data"

# Load the repo .env up front so the credential preflight below can see keys the
# user dropped there (the model wrapper also calls load_dotenv(), but that fires
# inside the eval subprocess — too late for our own fail-fast check).
load_dotenv(REPO_ROOT / ".env")


# Per-provider required env vars for the fail-fast preflight.
#
# NOTE ON AZURE: the aboda-openai resource is the *new Azure OpenAI v1 API* — it
# serves inference at the endpoint root (POST {endpoint}/chat/completions with the
# deployment name in the body), accepts Bearer auth, and needs no api-version.
# The classic AzureOpenAI client (/openai/deployments/{dep}/...?api-version=) 404s
# there. So --azure just points the *plain* OpenAI client at the Azure endpoint +
# key (verified: Bearer + no api-version + root path → 200). Hence no
# AZURE_OPENAI_API_VERSION requirement.
_REQUIRED_ENV = {
    "gemini": ("GOOGLE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_BASE"),
}


def _run(cmd: list[str]) -> None:
    """Run a subprocess from the repo root, streaming output; raise on failure."""
    print(f"\n[run_local_api] $ {' '.join(cmd)}", flush=True)
    # Force UTF-8 in the child so lmms-eval's results table (which contains
    # non-ASCII glyphs like the up-arrow) doesn't crash on a Windows cp1252
    # console with UnicodeEncodeError.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemini_api", help="lmms-eval --model registry name (default: gemini_api; use 'openai' for OpenAI/Azure)")
    p.add_argument("--model-version", default="gemini-2.5-flash", help="API model version / Azure deployment name passed as model_version= (default: gemini-2.5-flash)")
    p.add_argument("--azure", action="store_true", help="route the openai model through Azure OpenAI (AzureOpenAI client + AZURE_OPENAI_* env). Implies --model openai.")
    p.add_argument("--num-concurrent", type=int, default=None, help="in-flight request concurrency for the openai wrapper (num_concurrent=; default: wrapper default 32). Lower it if Azure returns 429s.")
    p.add_argument("--tasks", nargs="+", default=["coloring", "directed_connectivity", "shortest_path"], help="benchmark tasks (default: the standard trio). Each expands to _direct + _disguise subtasks.")
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"], choices=["easy", "medium", "hard"], help="difficulties to run sequentially (default: easy medium hard)")
    p.add_argument("--num-samples", type=int, default=10, help="generations per task per difficulty (default: 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-style", default="numeric", choices=["numeric", "letters", "none"])
    p.add_argument("--node-color", default="#AED6F1")
    p.add_argument("--edge-style", default="straight", choices=["straight", "curved"])
    p.add_argument("--include-adjacency-matrix", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=None, help="override per-generation output budget via --gen_kwargs max_new_tokens=N. Needed for reasoning models (e.g. gemini-3.x-pro) that would otherwise spend the task YAML's 64-token cap on thinking and return an empty answer.")
    p.add_argument("--max-tokens", type=int, default=None, help="optional hard token-budget ceiling across the run (lmms-eval --max_tokens); off by default")
    p.add_argument("--job-prefix", default=None, help="output dir prefix under ./logs (default derived from model-version + tasks)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --azure implies the openai wrapper; promote the default model so the user
    # doesn't have to pass both flags.
    if args.azure and args.model == "gemini_api":
        args.model = "openai"

    # Resolve which credential set this run needs, then fail fast (before any
    # cost-incurring dataset prep or API call) if any required var is missing.
    if args.model == "gemini_api":
        provider = "gemini"
    elif args.model == "openai":
        provider = "azure" if args.azure else "openai"
    else:
        provider = None  # unknown/other wrapper: skip the preflight, let it self-report

    if provider is not None:
        missing = [v for v in _REQUIRED_ENV[provider] if not os.environ.get(v)]
        if missing:
            print(
                f"[run_local_api] FATAL: {provider} run needs {', '.join(_REQUIRED_ENV[provider])} "
                f"but these are unset: {', '.join(missing)}. Add them to {REPO_ROOT / '.env'} "
                f"(or the environment).",
                file=sys.stderr,
            )
            return 2

    # Azure (new v1 API): drive the *plain* OpenAI client at the Azure endpoint
    # root using Bearer auth. Feed the wrapper via the OPENAI_* env it already
    # reads (base_url = OPENAI_API_BASE, api_key = OPENAI_API_KEY); keeps the key
    # out of --model_args / logs. These propagate to the eval subprocess via
    # os.environ, and load_dotenv() there won't clobber them (.env has no OPENAI_*).
    if provider == "azure":
        os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
        os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_API_BASE"].rstrip("/")

    if args.azure and "gpt-5" not in args.model_version and not any(k in args.model_version for k in ("o1", "o3", "o4")):
        print(
            f"[run_local_api] WARNING: model_version='{args.model_version}' has no reasoning marker "
            "(gpt-5/o1/o3/o4); the openai wrapper will send temperature+max_tokens, which reasoning "
            "SKUs reject. For gpt-5.6-terra keep 'gpt-5' in the deployment name.",
            file=sys.stderr,
        )

    py = sys.executable
    variants = ("direct", "disguise")
    lmms_tasks = ",".join(f"dynamic_graph_benchmark_{t}_{v}" for t in args.tasks for v in variants)

    # lmms-eval --model_args (comma-separated key=value). base_url + api_key come
    # from OPENAI_* env (set above for Azure), NOT here — so no secret hits the
    # CLI/logs. num_concurrent only when explicitly overridden.
    model_arg_parts = [f"model_version={args.model_version}"]
    if args.num_concurrent is not None:
        model_arg_parts.append(f"num_concurrent={args.num_concurrent}")
    model_args = ",".join(model_arg_parts)

    # A compact, filesystem-safe tag for the model version (e.g. gemini-2.5-flash -> gemini25flash).
    model_tag = args.model_version.replace("-", "").replace(".", "").replace("/", "_")
    tasks_tag = "_".join(args.tasks)
    job_prefix = args.job_prefix or f"local_{model_tag}_{tasks_tag}"

    print(f"[run_local_api] model={args.model} version={args.model_version} provider={provider} model_args={model_args}")
    print(f"[run_local_api] tasks={args.tasks} -> {lmms_tasks}")
    print(f"[run_local_api] difficulties={args.difficulties} num_samples={args.num_samples}")
    print(f"[run_local_api] repo_root={REPO_ROOT}")

    completed: list[tuple[str, str]] = []
    for diff in args.difficulties:
        job = f"{job_prefix}_{diff}"
        out_path = f"./logs/{job}"

        # --- Step 1: prepare the dataset directly into the canonical dir (overwrites
        # the previous difficulty's dataset; runs are strictly sequential). ----------
        prepare_args = [
            py, "tools/prepare_dynamic_graph_benchmark.py",
            "--seed", str(args.seed),
            "--tasks", *args.tasks,
            "--output-dir", CANONICAL_DATASET_DIR,
            "--num-samples", str(args.num_samples),
            "--difficulty", diff,
            "--label-style", args.label_style,
            "--node-color", args.node_color,
            "--edge-style", args.edge_style,
        ]
        if args.include_adjacency_matrix:
            prepare_args.append("--include-adjacency-matrix")
        _run(prepare_args)

        # --- Step 2: run lmms-eval on the direct + disguise subtasks. ----------------
        # Plain `python -m lmms_eval` (gemini_api's Accelerator() runs fine
        # single-process); avoids accelerate launch's process-spawn quirks on Windows.
        eval_args = [
            py, "-m", "lmms_eval",
            "--model", args.model,
            "--model_args", model_args,
            "--tasks", lmms_tasks,
            "--batch_size", str(args.batch_size),
            "--limit", str(args.num_samples),  # belt-and-suspenders cap on docs evaluated
            "--log_samples",
            "--output_path", out_path,
        ]
        if args.max_new_tokens is not None:
            eval_args += ["--gen_kwargs", f"max_new_tokens={args.max_new_tokens}"]
        if args.max_tokens is not None:
            eval_args += ["--max_tokens", str(args.max_tokens)]
        _run(eval_args)

        completed.append((diff, out_path))

    print("\n[run_local_api] done. Results:")
    for diff, out_path in completed:
        print(f"  {diff:6s} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
