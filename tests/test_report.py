"""
Tests for the narrative renderer.

Every committed transcript passes through render_answers, so a regression here
disfigures twelve reports at once. Two properties matter: prose must be wrapped
AFTER interpolation (the whole reason the function exists), and anything laid
out deliberately -- aligned mini-tables, horizontal rules -- must survive
untouched.
"""

from __future__ import annotations

import pytest

from report import RULE, Facts, render_answers

WIDTH = 78


class TestWrapping:
    def test_a_long_interpolated_verdict_is_wrapped(self):
        """The defect this was written for: a runtime verdict of 300 characters
        substituted into a hand-wrapped paragraph produced one enormous line in
        the committed transcript."""
        verdict = ("It holds. XGBoost minus elastic net is +0.0038 with interval "
                   "[-0.0061, +0.0133], crossing zero at EPV 28 just as it did "
                   "at EPV 6.4, which makes the earlier result a finding rather "
                   "than an artefact of the sample size.")
        out = render_answers("A1. HEAD\n    {v} And the prose that followed it.",
                             {"v": verdict})
        assert max(len(ln) for ln in out.splitlines()) <= WIDTH

    def test_the_heading_is_not_folded_into_the_body(self):
        out = render_answers("A50. WHO THESE PATIENTS ARE\n    {n} adults.",
                             {"n": "2,458"})
        assert out.splitlines()[0] == "A50. WHO THESE PATIENTS ARE"

    def test_paragraph_indentation_is_preserved(self):
        out = render_answers(
            "A1. HEAD\n    " + "word " * 60, {})
        body = [ln for ln in out.splitlines() if ln.strip()][1:]
        assert all(ln.startswith("    ") for ln in body)

    def test_blank_line_between_paragraphs_survives(self):
        out = render_answers("A1. HEAD\n    First para.\n\n    Second para.", {})
        assert "\n\n" in out


class TestPreservation:
    def test_aligned_tables_are_left_exactly_as_written(self):
        """Column alignment carries meaning. Re-wrapping one of these would
        turn a small table into a paragraph of numbers."""
        table = ("        bun      100.0% missing early vs 0.9% late\n"
                 "        urine    100.0% vs 9.1%\n"
                 "        glucose  100.0% vs 6.5%")
        out = render_answers(f"A1. HEAD\n    Prose.\n\n{table}", {})
        assert table in out

    def test_a_rule_flush_against_prose_keeps_its_place(self):
        """Rules are written with no blank line above them. An earlier version
        preserved the whole block on seeing a rule, which silently exempted the
        adjoining paragraph from wrapping."""
        long_line = "x" * 40 + " " + "y" * 40
        out = render_answers(f"A1. HEAD\n    {long_line}\n{RULE}", {})
        lines = out.splitlines()
        assert lines[-1] == RULE
        assert max(len(ln) for ln in lines[:-1]) <= WIDTH

    def test_answers_banner_stays_on_top_of_its_rule(self):
        out = render_answers("ANSWERS\n{rule}\n\nA1. HEAD\n    Prose.",
                             {"rule": RULE})
        assert out.startswith(f"ANSWERS\n{RULE}")


class TestFactsContract:
    def test_a_missing_narrative_number_raises_by_name(self):
        """format(**facts) unpacks into a plain kwargs dict, so Facts.__missing__
        never fired and the helpful message was dead code. format_map keeps the
        contract the project's write-up relies on."""
        with pytest.raises(KeyError, match="never computed"):
            render_answers("A1. HEAD\n    Value {absent} here.", {"present": "1"})

    def test_a_plain_dict_is_still_held_to_the_contract(self):
        """Call sites build the mapping with dict(facts, rule=RULE), which drops
        the Facts subclass. The guarantee has to survive that."""
        with pytest.raises(KeyError, match="never computed"):
            render_answers("A1. HEAD\n    {absent}", dict(a="1"))

    def test_facts_instances_pass_through_unchanged(self):
        out = render_answers("A1. HEAD\n    {v}.", Facts(v="value"))
        assert "value." in out


class TestSubHeadings:
    """An all-caps line inside an answer is a sub-heading, not the opening words
    of the paragraph below it. Without this the transcript read
    "AND THE CONSTRAINT THAT ENDS THE DISCUSSION Even a genuine improvement..."."""

    def test_an_all_caps_line_is_not_folded_into_its_paragraph(self):
        out = render_answers(
            "A1. HEAD\n    Body text.\n\n    A SUB HEADING IN CAPS\n    "
            + "word " * 40, {})
        assert "A SUB HEADING IN CAPS\n" in out
        assert "A SUB HEADING IN CAPS word" not in out

    def test_ordinary_prose_starting_with_an_acronym_still_wraps(self):
        """Guard against over-matching: a normal sentence that merely begins
        with capitals must not be treated as a heading."""
        out = render_answers(
            "A1. HEAD\n    AUC is the metric here and this sentence carries on "
            "for long enough to need wrapping across more than one line.", {})
        assert "AUC is the metric" in out
        body = [l for l in out.splitlines() if l.strip()][1:]
        assert len(body) > 1
