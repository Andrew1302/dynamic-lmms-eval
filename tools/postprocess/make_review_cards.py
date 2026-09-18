"""Build one human-reviewable "review card" PNG per question from an organized
sample tree (as produced by organize_fp8smoke_pairs.py / organize_thinkadj_pairs.py).

For every ``q_<NNN>/`` leaf that has an ``image.png``, composite:

    [ model-shown image ]  |  [ text panel ]

where the text panel carries, top to bottom:
    task | variant | difficulty | doc_id
    GROUND TRUTH: <target>
    adjacency list (parsed from prompt.txt)   <- what the image must encode
    per-model: extracted answer + CORRECT/WRONG verdict

This is the "is the task solvable / did the model actually see the image"
artifact: a reviewer opens card.png and checks the picture against the
adjacency + ground truth without opening five text files. A root
``_cards_index.tsv`` lists every card with the models' verdicts so you can
filter to the disagreements or the wrong-answer cases.

Usage:
  python tools/postprocess/make_review_cards.py --tree remote_results/_inspect/fp8smoke
  python tools/postprocess/make_review_cards.py --tree <dir> --only shortest_path
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


MONO = _font(15)
MONO_B = _font(17)


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def parse_answer_file(text: str) -> dict:
    out = {"model": "", "extracted": "", "verdict": "", "score": "", "gt": ""}
    m = re.search(r"^model:\s*(.+)$", text, re.M)
    if m:
        out["model"] = m.group(1).strip()
    m = re.search(r"^ground_truth:\s*(.+)$", text, re.M)
    if m:
        out["gt"] = m.group(1).strip()
    m = re.search(r"^extracted_answer:\s*(.*)$", text, re.M)
    if m:
        out["extracted"] = m.group(1).strip()
    m = re.search(r"^score:\s*(\S+)\s*\((\w+)\)", text, re.M)
    if m:
        out["score"], out["verdict"] = m.group(1), m.group(2)
    return out


def extract_adjacency(prompt: str) -> str:
    """Pull the adjacency-list block out of a prompt if present, else empty."""
    lines = prompt.splitlines()
    keep = [ln for ln in lines if re.match(r"\s*(Adj\[|\d+\s*(->|:|—|-))", ln)]
    return "\n".join(keep[:40])


def paired_adjacency(qdir: Path) -> str:
    """Image-only prompts omit the adjacency by design. Recover the exact
    ground-truth graph from the sibling adj-tier question (same seed → same
    graph) so a reviewer can check the picture against it."""
    parts = list(qdir.parts)
    # .../<tier>/<arm>/<task>/<diff>/<variant>/q_NNN  → swap tier to 'adj'
    if len(parts) < 6:
        return ""
    parts[-6] = "adj"
    sib = Path(*parts) / "prompt.txt"
    if sib.exists():
        return extract_adjacency(read(sib))
    return ""


def wrap(text: str, width: int) -> list[str]:
    out = []
    for raw in text.splitlines() or [""]:
        if len(raw) <= width:
            out.append(raw)
        else:
            for i in range(0, len(raw), width):
                out.append(raw[i:i + width])
    return out


def build_card(qdir: Path, answer_stems: list[str]) -> Image.Image | None:
    img_p = qdir / "image.png"
    if not img_p.exists():
        return None
    img = Image.open(img_p).convert("RGB")
    H = 760
    img = img.resize((int(img.width * H / img.height), H))

    prompt = read(qdir / "prompt.txt")
    target = read(qdir / "answer.txt").strip()
    adj = extract_adjacency(prompt) or paired_adjacency(qdir)

    panel_lines: list[tuple[str, object, str]] = []
    parts = qdir.parts
    crumb = "/".join(parts[-6:-1])  # tier/arm/task/diff/variant
    panel_lines.append((f"{crumb}/{qdir.name}", MONO_B, "#111111"))
    panel_lines.append(("", MONO, "#000000"))
    panel_lines.append((f"GROUND TRUTH: {target}", MONO_B, "#0B6E4F"))
    panel_lines.append(("", MONO, "#000000"))
    panel_lines.append(("adjacency (from prompt):", MONO_B, "#111111"))
    for ln in wrap(adj, 46) or ["(none in prompt)"]:
        panel_lines.append(("  " + ln, MONO, "#333333"))
    panel_lines.append(("", MONO, "#000000"))
    for stem in answer_stems:
        a = parse_answer_file(read(qdir / f"{stem}.txt"))
        if not a["model"]:
            continue
        color = "#0B6E4F" if a["verdict"] == "CORRECT" else "#B00020"
        panel_lines.append((f"{a['model']}: {a['verdict']} (score {a['score']})",
                            MONO_B, color))
        for ln in wrap(f"  ans: {a['extracted']}", 46):
            panel_lines.append((ln, MONO, "#333333"))
        panel_lines.append(("", MONO, "#000000"))

    line_h = 22
    panel_w = 560
    panel_h = max(H, 20 + line_h * len(panel_lines))
    canvas_h = max(H, panel_h)
    out = Image.new("RGB", (img.width + panel_w + 24, canvas_h + 16), "white")
    out.paste(img, (8, 8))
    d = ImageDraw.Draw(out)
    x0 = img.width + 20
    d.line([(x0 - 6, 8), (x0 - 6, canvas_h)], fill="#CCCCCC", width=1)
    y = 12
    for txt, font, color in panel_lines:
        d.text((x0, y), txt, fill=color, font=font)
        y += line_h
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True,
                    help="organized sample tree root (contains q_NNN leaves)")
    ap.add_argument("--only", default="", help="substring filter on the leaf path")
    ap.add_argument("--stems", nargs="*",
                    default=["qwen_35_answer", "internvl_35_answer",
                             "gemma_e2b_answer"],
                    help="per-model answer file stems to render")
    args = ap.parse_args(argv)

    qdirs = sorted(p for p in args.tree.rglob("q_*") if p.is_dir())
    if args.only:
        qdirs = [p for p in qdirs if args.only in p.as_posix()]
    index = ["path\tground_truth\t" + "\t".join(args.stems)]
    n = 0
    for qdir in qdirs:
        card = build_card(qdir, args.stems)
        if card is None:
            continue
        card.save(qdir / "card.png")
        target = read(qdir / "answer.txt").strip()
        verds = []
        for stem in args.stems:
            a = parse_answer_file(read(qdir / f"{stem}.txt"))
            verds.append(a["verdict"] or "-")
        index.append(f"{qdir.relative_to(args.tree).as_posix()}\t{target}\t"
                     + "\t".join(verds))
        n += 1
    (args.tree / "_cards_index.tsv").write_text("\n".join(index) + "\n",
                                                encoding="utf-8")
    print(f"[cards] wrote {n} card.png under {args.tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
