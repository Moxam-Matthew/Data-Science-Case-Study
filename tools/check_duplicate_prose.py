"""
Flag near-duplicate paragraphs inside a generated transcript's ANSWERS block.

This exists because a real defect got committed. An answer in
11_ceiling_and_transport.py was corrected by ADDING a rewritten paragraph
without deleting the original, so the transcript argued the same point twice in
slightly different words. The numbers in these reports are interpolated from the
run and pinned by tests; the prose around them is written by hand and has no
other guard.

WHAT DOES NOT WORK, since both were tried:

  Whole-paragraph similarity. A restated point diverges in the tail, so the real
  instance scored 0.63 on its shared half and lower across the whole paragraph
  -- under any threshold that does not also flag every paragraph on a shared
  topic.

  Comparing the first N words. The two openings were "What the curve still
  cannot show" and "What the curve cannot show is": one inserted word shifts
  every position and the comparison fails.

WHAT WORKS is a shared word SHINGLE -- any run of 7 consecutive content words
appearing in both paragraphs. Insertions elsewhere do not disturb it, and
ordinary English reuse does not produce one. Calibrated across all twelve
transcripts: length 7 flags 2 paragraph pairs and catches the known defect,
length 9 catches nothing.

Comparison is WITHIN a file only. These reports deliberately restate context
across scripts, so cross-file repetition is intended rather than a defect.

    python tools/check_duplicate_prose.py [output/11_ceiling_and_transport.txt ...]

With no arguments it checks every transcript in output/. Exits non-zero if
anything is flagged, so it can gate a commit.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

SHINGLE = 7      # consecutive words that must coincide
MIN_CHARS = 120  # ignore one-line fragments


def paragraphs(text: str) -> list[str]:
    if "ANSWERS" not in text:
        return []
    block = text[text.index("ANSWERS"):]
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", block)
            if len(" ".join(p.split())) >= MIN_CHARS]


def shingles(paragraph: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z]+", paragraph.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def check(path: Path) -> list[str]:
    paras = paragraphs(path.read_text(encoding="utf-8", errors="replace"))
    flags = []
    for a, b in itertools.combinations(paras, 2):
        shared = shingles(a) & shingles(b)
        if shared:
            phrase = " ".join(sorted(shared)[0])
            flags.append(f'  shared phrase: "...{phrase}..."\n'
                         f"    A: {a[:100]}...\n"
                         f"    B: {b[:100]}...")
    return flags


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or sorted(
        (Path(__file__).resolve().parent.parent / "output").glob("*.txt"))
    total = 0
    for path in targets:
        flags = check(path)
        if flags:
            total += len(flags)
            print(f"\n{path.name}: {len(flags)} possible duplicate(s)")
            print("\n".join(flags))
    if total:
        print(f"\n{total} flagged. Each is a candidate, not a verdict -- read "
              f"both and delete whichever is redundant.")
        return 1
    print(f"No duplicated prose across {len(targets)} transcript(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
