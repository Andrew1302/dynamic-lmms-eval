# `full_report/` — the assembled "full results" build

Builds the cross-campaign **full results** workbooks: for each family, it assembles
the authoritative per-task jsonls into a synthetic per-job tree, runs the standard
`batch_report.py` over it, and adds the `run_info` tab.

**This code used to live in `remote_results/_full_reports/_build/`, which
`.gitignore:16` excludes — 847 lines of build logic and the project's only reporting
regression gate were not in version control at all.** A `remote_results/` wipe would
have destroyed them. Outputs still go to `remote_results/_full_reports/`; only the
code moved.

## Running it

```bash
bash tools/postprocess/full_report/regen_all.sh <scratch_assembled_root>
```

`regen_all.sh` chains: `assemble_full.py` → `batch_report.py` → `add_full_run_info.py`.
The scratch root holds the assembled trees and per-family TSVs; nothing under
`remote_results/` is read-modified, only written.

## The regression gates

Both verifiers recompute independently from the assembled jsonls and compare against
the shipped workbooks, so they can be run **without** regenerating anything:

```bash
# assemble once into a scratch dir, then:
python tools/postprocess/full_report/verify_runinfo.py    <asm_root>
python tools/postprocess/full_report/verify_crossreport.py <asm_root>
```

`verify_runinfo.py` is the gate that matters for any reporting refactor. Expected
output:

```
size-cells checked: 54  fails: 0
RESULT: ALL RUN_INFO RANGES MATCH
```

Its `WARN ... differs across families` lines are informational: the standard family
runs n=500 per cell and the ablations n=100, so the graph counts legitimately differ.

## Known pre-existing failure — `verify_crossreport.py`

`verify_crossreport.py` currently reports **48 fails / 270 values checked**, e.g.

```
FAIL full_think: rerun cell graph_bench_think_think_medium_qwen35_4b
     absent from standalone 15_abl_think_latest.xlsx
```

This predates the move — the untracked original produces byte-identical output on the
same input. The cause is staleness, not disagreement: the standalone batch reports in
`remote_results/_batch_reports/` are dated 2026-07-18, older than the cells they are
missing, so the full reports contain rerun cells the standalone ones never saw.
Regenerating those standalone reports should clear it.

**It is recorded here so a future refactor is measured against 48, not against 0, and
so the failure is never mistakenly attributed to that refactor.** `verify_runinfo.py`
is clean and is the gate to hold at zero.

## Note on `build_all_reports.py`

There are two wrappers here: `add_full_run_info.py` (2026-07-20) and
`build_all_reports.py` (2026-07-23, the newer and in practice the live driver). Both
monkeypatch module globals on the `add_run_info_*` scripts (`RES`, `ARMS`, `SPD`,
`_config_rows`), and **their injected configuration prose has already drifted apart** —
two workbooks for the same family can describe their configuration differently. That
duplication is what the planned `_families.py` registry removes; until then, prefer
`add_full_run_info.py`'s wording, which matches the shipped generators.
