"""
Shared reporting scaffolding, and the mechanism that stops prose drifting
away from results.

Two jobs here.

The first is mundane: the rules, headers and question banners that were
copy-pasted verbatim across three scripts now live in one place.

The second matters more. Every narrative number in this project -- the follow-up
ratio, the albumin skew, the count of impossible cells -- is interpolated into
the prose from the frame that produced it, never typed. Hardcoding them is not a
tidiness problem, it is a correctness problem: the numbers were originally
written against a full-cohort run, and when the analysis moved to a training
partition the prose silently kept the old values. A reader comparing the printed
table with the paragraph beneath it would have found them disagreeing, in a
repository whose entire argument is that the details were checked.

`Facts` closes that class of error permanently. If a value appears in a
sentence, it is looked up; if the lookup key is missing, formatting raises
rather than printing something stale.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import pandas as pd

RULE = "=" * 78
SUB = "-" * 78


def configure_pandas(width: int = 220) -> None:
    pd.set_option("display.width", width)
    pd.set_option("display.max_columns", 60)


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def question(n: int, text: str) -> None:
    print(f"\n{SUB}\nQUESTION {n}: {text}\n{SUB}")


class Facts(dict):
    """
    A dict that refuses to silently supply a missing narrative number.

    `"{alb_skew}".format(**facts)` on an absent key raises KeyError, which is
    the desired behaviour: a question header or an answer paragraph that
    references a quantity the run did not produce is a bug, not a blank.
    """

    def __missing__(self, key: str) -> Any:  # pragma: no cover - defensive
        raise KeyError(
            f"narrative fact '{key}' was referenced in prose but never computed. "
            f"Add it in the script's collect_facts(). Available: "
            f"{', '.join(sorted(self))}"
        )


def render_answers(template: str, facts: "Facts | dict[str, Any]",
                   width: int = 78) -> str:
    """
    Interpolate the narrative, then re-wrap it so interpolation cannot break
    the layout.

    The ANSWERS blocks are written pre-wrapped to 78 columns, but `Facts`
    substitutes verdict sentences that are decided at runtime and can run to
    300 characters. Those land inside an already-wrapped paragraph and produce
    one enormous line, so a reader opening the committed transcript sees a
    report that visibly came apart -- in a project whose argument is that the
    details were checked. Formatting first and wrapping second makes the
    template's own line breaks irrelevant and the result stable no matter how
    long a verdict turns out to be.

    Paragraphs that are laid out deliberately are left exactly as written.
    Those are identified by column alignment -- three or more consecutive
    spaces between non-space characters -- or by mixing more than two levels of
    indentation, which is what the small aligned tables inside several answers
    look like. Re-wrapping one of those would destroy it.
    """
    import re
    import textwrap

    # Call sites build the mapping with dict(facts, rule=RULE), which would
    # otherwise discard the Facts subclass and with it the guarantee that a
    # referenced-but-uncomputed number raises instead of printing a blank.
    if not isinstance(facts, Facts):
        facts = Facts(facts)

    # format_map, not format(**facts): unpacking looks keys up in a plain kwargs
    # dict, so Facts.__missing__ never fired and a missing narrative number
    # raised a bare KeyError instead of the message naming what to compute.
    is_rule = lambda ln: bool(re.fullmatch(r"[=\-_~]{4,}", ln.strip()))

    def wrap_run(lines: list[str]) -> str:
        """Wrap one run of ordinary prose lines."""
        indents = [len(ln) - len(ln.lstrip()) for ln in lines]
        # An answer heading ("A50. WHO THESE PATIENTS ARE") sits at a shallower
        # indent than the body that follows it with no blank line between. It
        # is its own line, not the first sentence of the paragraph, so joining
        # the two would read as "A50. WHO THESE PATIENTS ARE 2,458 adults...".
        head = ""
        if len(lines) > 1 and (
                # Outdented relative to the body, e.g. "A50. WHO THESE PATIENTS ARE".
                indents[0] < min(indents[1:])
                # Or an all-caps sub-heading sitting at the body's own indent,
                # which would otherwise be joined onto the sentence beneath it
                # and read as "AND THE CONSTRAINT THAT ENDS THE DISCUSSION Even
                # a genuine improvement could not be adopted here."
                or (lines[0].strip() == lines[0].strip().upper()
                    and any(c.isalpha() for c in lines[0])
                    and len(lines[0].strip()) < width)):
            head, lines, indents = lines[0].rstrip(), lines[1:], indents[1:]
        body = textwrap.fill(
            " ".join(ln.strip() for ln in lines), width=width,
            initial_indent=" " * min(indents),
            subsequent_indent=" " * (sorted(set(indents))[1]
                                     if len(set(indents)) > 1 else min(indents)),
            break_long_words=False, break_on_hyphens=False)
        return f"{head}\n{body}" if head else body

    out = []
    for block in re.split(r"\n\s*\n", template.format_map(facts)):
        if not block.strip():
            continue
        lines = [ln for ln in block.split("\n") if ln.strip()]
        indents = {len(ln) - len(ln.lstrip()) for ln in lines}
        # Leave alone anything laid out deliberately: column-aligned text, or
        # more than two indent levels, which is what the small aligned tables
        # inside several answers look like.
        if re.search(r"\S {3,}\S", block) or len(indents) > 2:
            out.append(block.rstrip())
            continue

        # Rules are written flush against the text above them, with no blank
        # line -- "ANSWERS" sits directly on top of one. Split the block into
        # runs of rule and non-rule lines so the rule keeps its position while
        # the prose beside it still gets wrapped, and rejoin with a single
        # newline so the original spacing is preserved.
        parts, run = [], []
        for ln in lines:
            if is_rule(ln):
                if run:
                    parts.append(wrap_run(run))
                    run = []
                parts.append(ln.rstrip())
            else:
                run.append(ln)
        if run:
            parts.append(wrap_run(run))
        out.append("\n".join(parts))
    return "\n\n".join(out)


def fmt_p(p: float | None) -> str:
    """Format a p- or q-value the way a journal would."""
    if p is None or pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_table(df: pd.DataFrame, round_map: dict[str, int] | None = None,
              p_cols: tuple[str, ...] = ()) -> str:
    """Round, format p-columns, and render without the index."""
    out = df.copy()
    for col, nd in (round_map or {}).items():
        if col in out:
            out[col] = out[col].round(nd)
    for col in p_cols:
        if col in out:
            out[col] = out[col].apply(fmt_p)
    return out.to_string(index=False)


def run_and_capture(main: Callable[[], None], out_path: Path) -> None:
    """
    Run a script's main(), printing to the terminal and writing the same text
    to `out_path`.

    The transcripts are committed. A reviewer who will not clone and run the
    project -- which is most of them -- can otherwise see no number behind any
    claim in the README, and figures alone do not carry the tables.

    Order matters here. The file is written BEFORE the terminal, because the
    terminal is the fragile one: a Windows console on a legacy codepage raises
    UnicodeEncodeError on any character it cannot map, and doing that first once
    destroyed a five-minute run at its final line. The expensive artefact is
    saved first, then the display is attempted defensively.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        # Console cannot represent some character; degrade rather than lose the
        # run. The committed transcript is already complete and correct.
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
        sys.stdout.write(
            f"\n[note: some characters were not representable in {enc} and were "
            f"replaced for display only; {out_path.name} is unaffected]\n")
    print(f"\n[transcript written to {out_path.name}]")
