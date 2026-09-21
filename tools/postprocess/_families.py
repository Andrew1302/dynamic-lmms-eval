"""One registry describing every experiment family.

The same fact -- "which family does this job belong to, and what are its axis
coordinates" -- was previously encoded twice, in two incompatible notations:

  * ``organize_results.SPECS``      regexes with named groups -> campaign + leaf
  * ``batch_report._AXIS_PREFIXES`` ~20 literal prefixes built by 4 generator
                                    functions -> axis + axis_value

They drifted, as duplicated tables do: ``thinkadj500inc`` has a SPEC but no
prefix, so all 18 of those jobs report as ``axis="unknown"`` in every batch
report. One regex with named groups plus two small templates produces both
outputs, and the drift becomes impossible rather than merely unlikely.

The generator already declared this intent -- "Single source of truth for the
job-name -> NN_campaign mapping ... adding a new ablation family means adding
one SPEC entry" (generate_graph_benchmark_jobs.py) -- this generalises it.

Adding a family is one entry here. Nothing else.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

# Short aliases used in results-tree leaf names.
MODELS = {"internvl35_4b": "internvl", "qwen35_4b": "qwen", "qwen35_2b": "qwen2b", "gemma4_e2b": "gemma"}
MODEL_PRETTY = {"internvl35_4b": "InternVL3.5-4B", "qwen35_4b": "Qwen3.5-4B", "qwen35_2b": "Qwen3.5-2B", "gemma4_e2b": "Gemma-4-E2B"}
PRETRAINED = {
    "internvl35_4b": "OpenGVLab/InternVL3_5-4B",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "qwen35_2b": "Qwen/Qwen3.5-2B",
    "gemma4_e2b": "google/gemma-4-E2B-it",
}

TASKS = ("coloring", "directed_connectivity", "shortest_path")
DIFFICULTIES = ("easy", "medium", "hard")
VARIANTS = ("direct", "disguise")

_DIFF = r"(?P<diff>easy|medium|hard)"
_MODEL = r"(?P<model>[a-z0-9]+_[a-z0-9]+)"
_COL = r"(?:(?P<col>coloring)_)?"  # optional coloring-only job marker
_ARM = r"(?P<arm>think|nothink)"
_CTASK = r"(?P<task>sp|coloring|conn)"  # task marker for the cott_ families

# Jobs whose results are inspection/smoke material, never a campaign. Checked
# before the patterns, mirroring organize_results.match_job.
SCRATCH_RE = re.compile(r"smoke|promptexp|fp8think|v1ab|gemmatok")


@dataclasses.dataclass(frozen=True)
class Family:
    """One experiment family: how its jobs are named, filed and reported."""

    key: str  # stable id; also the --family CLI choice
    pattern: re.Pattern  # matches the BARE job name (graph_bench_ stripped)
    campaign: str  # remote_results/NN_campaign

    leaf_tmpl: str  # results-tree leaf; str.format over the match groups
    axis: str  # summary-sheet "axis" column
    axis_parts: tuple[str, ...] = ()  # groups joined with "/" (empties dropped)

    description: str = ""  # replaces the CAMPAIGN_DESC entry

    def leaf(self, fields: dict[str, str]) -> str:
        return self.leaf_tmpl.format(**fields)

    def axis_value(self, fields: dict[str, str]) -> str:
        return "/".join(fields.get(p) or "" for p in self.axis_parts if fields.get(p))

    def job_dir(self, results_root: Path, job: str) -> Path:
        """Where a job's fetched data lives: a flat fresh fetch, else this
        family's campaign. Deterministic -- the campaign is a property of the
        family, so no newest-mtime glob and no JOB_CAMPAIGN_PIN are needed."""
        flat = results_root / job
        return flat if flat.is_dir() else results_root / self.campaign / "_jobs" / job


# Order matters: first match wins, most specific first (thinkadj500inc before
# thinkadj before think), exactly as the two tables it replaces were ordered.
FAMILIES: tuple[Family, ...] = (
    Family(
        key="scram",
        pattern=re.compile(rf"^scram_{_ARM}_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="17_abl_scram",
        leaf_tmpl="{model}_{diff}_{arm}",
        axis="scram_img",
        axis_parts=("arm", "col", "diff"),
        description="Scrambled ('no-info') image control + adjacency list. Both arms, 3 difficulties.",
    ),
    Family(
        key="thinkadj_inc",
        pattern=re.compile(rf"^thinkadj500inc_{_ARM}_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="16_abl_thinkadj",
        leaf_tmpl="{model}_{diff}_{arm}_inc",
        axis="thinking_adj",
        axis_parts=("arm", "col", "diff"),
        description="Deferred 400-sample increment of the thinkadj ablation; pools with the n=100 run for n=500.",
    ),
    Family(
        key="thinkadj",
        pattern=re.compile(rf"^thinkadj_{_ARM}_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="16_abl_thinkadj",
        leaf_tmpl="{model}_{diff}_{arm}",
        axis="thinking_adj",
        axis_parts=("arm", "col", "diff"),
        description="Thinking x adjacency-list ablation. Both arms, n=100/task, all three tasks.",
    ),
    Family(
        key="think",
        pattern=re.compile(rf"^think_{_ARM}_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="15_abl_think",
        leaf_tmpl="{model}_{diff}_{arm}",
        axis="thinking",
        axis_parts=("arm", "col", "diff"),
        description="Thinking on/off ablation (image-only prompt). Both arms, n=100/task, all three tasks.",
    ),
    Family(
        key="adjlist",
        pattern=re.compile(rf"^ablation_adjlist_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="12_abl_adjlist",
        leaf_tmpl="{model}_{diff}",
        axis="adjlist",
        axis_parts=("diff",),
        description="Adjacency list injected into the prompt. n=100/task, 3 difficulties x 3 models.",
    ),
    Family(
        key="labels",
        pattern=re.compile(rf"^ablation_labels_(?P<style>letters|none)_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="13_abl_labels",
        leaf_tmpl="{model}_{diff}_{style}",
        axis="labels",
        axis_parts=("style", "col", "diff"),
        description="Node-label style ablation (letters / none vs the numeric baseline). n=100/task.",
    ),
    Family(
        key="color",
        pattern=re.compile(rf"^ablation_color_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="14_abl_color",
        leaf_tmpl="{model}_{diff}",
        axis="color",
        axis_parts=("col", "diff"),
        description="Node fill-colour ablation. n=100/task, 3 difficulties x 3 models.",
    ),
    Family(
        key="coloring_legacy",
        pattern=re.compile(rf"^coloring_{_DIFF}_{_MODEL}$"),
        campaign="03_coloring_chi",
        leaf_tmpl="{model}_{diff}",
        axis="coloring",
        axis_parts=("diff",),
        description="Retired dedicated coloring jobs; chi-control is now the prepare default.",
    ),
    Family(
        key="standard",
        pattern=re.compile(rf"^standard_{_DIFF}_{_MODEL}$"),
        campaign="10_standard",
        leaf_tmpl="{model}_{diff}",
        axis="standard",
        axis_parts=("diff",),
        description="Baseline benchmark v2, n=500/task, 3 difficulties x 3 models, original + disguise.",
    ),
    Family(
        key="cot_task_inc",
        # Deferred increment of a cot_task cell: same seed, higher --start-index.
        pattern=re.compile(rf"^cottinc_{_CTASK}_(?P<prompt>[a-z0-9_]+_v\d+)_{_DIFF}_{_MODEL}$"),
        campaign="18_abl_cot",
        leaf_tmpl="{model}_{diff}_{task}_{prompt}_inc",
        axis="cot_prompt",
        axis_parts=("task", "prompt", "diff"),
        description="Deferred increment of the per-task CoT runs; pools with the base run because sample generation is index-pure.",
    ),
    Family(
        key="cot_task",
        # Task-scoped CoT arms. The prefix is "cott_", NOT "cot_<task>_": prompt
        # ids already start with the task ("sp_explain_v1"), so a "cot_sp_..."
        # pattern would also match the existing cot_prompt jobs and re-file them.
        pattern=re.compile(rf"^cott_{_CTASK}_(?P<prompt>[a-z0-9_]+_v\d+)_{_DIFF}_{_MODEL}$"),
        campaign="18_abl_cot",
        leaf_tmpl="{model}_{diff}_{task}_{prompt}",
        axis="cot_prompt",
        axis_parts=("task", "prompt", "diff"),
        description="Chain-of-thought prompt experiment run per task. Image and graph are identical across arms; only the instruction text differs.",
    ),
    Family(
        key="cot_prompt",
        # PROMPT_ID values are <family>_v<n> (prompting/registry.py).
        pattern=re.compile(rf"^cot_(?P<prompt>[a-z0-9_]+_v\d+)_{_COL}{_DIFF}_{_MODEL}$"),
        campaign="18_abl_cot",
        leaf_tmpl="{model}_{diff}_{prompt}",
        axis="cot_prompt",
        axis_parts=("prompt", "col", "diff"),
        description=("Chain-of-thought prompt-template experiment. The image and the graph are identical across arms; only the instruction text differs. Includes direct_v1 as the in-campaign baseline."),
    ),
    Family(
        key="sweep_nodes",
        pattern=re.compile(rf"^sweep_(?P<constraint>nodes)_{_COL}{_MODEL}$"),
        campaign="11_sweep_size",
        leaf_tmpl="{model}_{constraint}",
        axis="sweep_nodes",
        description="Graph-size scaling sweep over the vertex-count axis.",
    ),
    Family(
        key="sweep_edges",
        pattern=re.compile(rf"^sweep_(?P<constraint>edges)_{_COL}{_MODEL}$"),
        campaign="11_sweep_size",
        leaf_tmpl="{model}_{constraint}",
        axis="sweep_edges",
        description="Graph-size scaling sweep over the edge-count axis.",
    ),
)

FAMILY_KEYS = [f.key for f in FAMILIES]
_BY_KEY = {f.key: f for f in FAMILIES}


def family_by_key(key: str) -> Family:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown family {key!r}; known: {FAMILY_KEYS}") from None


def match_job(job_name: str) -> tuple[Family, dict[str, str]] | None:
    """Classify a job name. Returns (family, fields) or None for scratch/unknown.

    ``fields["model"]`` is the short alias used in leaf names; the raw token is
    kept as ``fields["model_raw"]`` because the report's summary sheet shows it.
    """
    bare = job_name[len("graph_bench_") :] if job_name.startswith("graph_bench_") else job_name
    if SCRATCH_RE.search(bare):
        return None
    for family in FAMILIES:
        m = family.pattern.match(bare)
        if not m:
            continue
        fields = {k: (v or "") for k, v in m.groupdict().items()}
        raw = fields.get("model", "")
        if raw not in MODELS:
            return None
        fields["model_raw"] = raw
        fields["model"] = MODELS[raw]
        return family, fields
    return None


def campaign_for_name(job_name: str) -> str:
    """The NN_campaign dir a job files into, or "" for scratch/unknown."""
    matched = match_job(job_name)
    return matched[0].campaign if matched else ""
