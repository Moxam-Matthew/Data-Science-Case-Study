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
