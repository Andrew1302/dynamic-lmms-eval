"""Audit the chi (target) distribution of EVERY coloring source folder used by
the full-results merge. Special-chi = ~uniform over {2,3,4}; default-chi =
skewed hard to 4. Flag any non-sweep leaf whose most-common target > 60%."""
import json, os
from collections import Counter
from pathlib import Path

# Repo-relative roots. This file used to live in the gitignored directory
# remote_results/_full_reports/_build/ and hardcoded absolute paths; it is now
# tracked, so it must work from any checkout.
REPO = Path(__file__).resolve().parents[3]
POSTPROCESS = Path(__file__).resolve().parents[1]

RES = REPO / "remote_results"

# All coloring sources the assembler actually reads (base + rerun per family).
# 01_standard is intentionally EXCLUDED (base-standard coloring redirects to 03).
SOURCES = ["03_coloring_chi", "10_standard",           # standard
           "02_sweep_size", "11_sweep_size",           # sweep (clamped -> skip skew flag)
           "04_abl_adjlist", "12_abl_adjlist",          # adjlist
           "05_abl_labels", "13_abl_labels",            # labels
           "06_abl_color", "14_abl_color",              # color
           "07_abl_think", "15_abl_think"]              # think
SWEEP = {"02_sweep_size", "11_sweep_size"}  # legit chi-clamp skew on small graphs

# Show 01_standard too, for the record (should look degenerate = the bug).
WATCH = ["01_standard"]


def dist(p):
    c = Counter()
    for l in open(p, encoding="utf-8"):
        c[str(json.loads(l).get("target")).strip()] += 1
    return c


def audit(camp, flag=True):
    folder = RES / camp / "coloring" / "original"
    if not folder.is_dir():
        print("  %s: NO coloring/original" % camp); return 0
    flags = 0
    for jf in sorted(folder.glob("*.jsonl")):
        c = dist(jf)
        n = sum(c.values())
        top_frac = max(c.values()) / n if n else 0
        skew = flag and camp not in SWEEP and top_frac > 0.60
        mark = "  <<< SKEWED (default-chi?)" if skew else ""
        if skew or camp in WATCH:
            print("  %-16s %-22s n=%-4d %s top=%.2f%s" % (
                camp, jf.stem, n, dict(sorted(c.items())), top_frac, mark))
        if skew:
            flags += 1
    return flags


print("=== WATCH (01_standard: expected degenerate = the known bug, NOT used) ===")
for c in WATCH:
    audit(c, flag=False)
print("\n=== SOURCES actually used by the merge (flag if skewed) ===")
total = 0
for c in SOURCES:
    total += audit(c)
print("\nSKEWED leaves among used sources: %d" % total)
print("RESULT:", "ALL USED COLORING SOURCES ARE SPECIAL-CHI" if total == 0 else "PROBLEM FOUND")
