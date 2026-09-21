"""Print exactly what a template sends to the model.

    python -m prompting.show                      # list the library
    python -m prompting.show sp_explain_v1        # direct variant
    python -m prompting.show sp_explain_v1 --variant disguise --task shortest_path
"""

from __future__ import annotations

import argparse

from prompting.base import DOC_IMAGE
from prompting.registry import DEFAULT_PROMPT_ID, load_template
from prompting.file_template import load_library

SAMPLE_QUESTIONS = {
    ("shortest_path", "direct"): "Q: In the weighted directed acyclic graph shown, what is the minimum-weight path total from vertex 0 to vertex 4?\nA:",
    ("shortest_path", "disguise"): "Q: The map of Latin America below shows several cities and the available driving routes between them. Each arrow indicates a one-way driving connection from one city to another, and is labeled with the typical driving time in hours. What is the minimum total driving time, in hours, from the start city to the end city?\nA:",
    ("coloring", "direct"): "Q: What is the minimum number of colors needed to color this graph so that no two adjacent nodes share a color?\nA:",
    ("coloring", "disguise"): "Q: How many colors are needed to color this map so that no two regions sharing a land border have the same color? Regions separated by water, or touching only at a single point, are not neighbors.\nA:",
    ("directed_connectivity", "direct"): "Q: Following the arrow directions, is there a directed path from node 5 to node 3?\nA:",
    ("directed_connectivity", "disguise"): "Q: Each corridor passage in this maze has an arrow showing the only allowed direction of travel. Following the arrows, is it possible to travel from the green cell to the red cell?\nA:",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt_id", nargs="?", help="omit to list the library")
    ap.add_argument("--task", default="shortest_path", choices=sorted({t for t, _ in SAMPLE_QUESTIONS}))
    ap.add_argument("--variant", default="direct", choices=("direct", "disguise"))
    args = ap.parse_args(argv)

    lib = load_library()
    if not args.prompt_id:
        print(f"{len(lib)} templates in prompting/library/\n")
        for pid, tpl in sorted(lib.items()):
            mark = " (default)" if pid == DEFAULT_PROMPT_ID else ""
            print(f"  {pid}{mark}\n      {tpl.summary}\n")
        return 0

    template = load_template(args.prompt_id)
    doc = {"task": args.task, "variant": args.variant, "prompt": SAMPLE_QUESTIONS[(args.task, args.variant)]}
    budget = template.budget()

    print("=" * 78)
    print(f"{template.id}   [{args.task} / {args.variant}]")
    print("=" * 78)
    print(f"fingerprint : {template.fingerprint()}")
    print(f"answer      : {template.answer_spec(args.task).instruction()}")
    print(f"budget      : {budget.as_gen_kwargs()}")
    print(f"images/req  : {template.max_images()}" + (f"   min_window={template.min_window}" if template.min_window else ""))
    if template.system(doc):
        print(f"\n--- system ---\n{template.system(doc)}")
    for i, turn in enumerate(template.turns(doc)):
        imgs = [DOC_IMAGE if im == DOC_IMAGE else f"assets/{im.name}" for im in turn.images]
        print(f"\n--- turn {i}: {turn.role}{('  ' + ', '.join(imgs)) if imgs else ''} ---")
        print(turn.text)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
