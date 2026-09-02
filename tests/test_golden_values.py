"""
Golden values: the numbers this project publishes.

Every constant below appears in the README or in an answer block. The point is
not to test arithmetic -- it is to make a silent change LOUD.

The failure this exists for is real and was observed. Running the analysis under
pandas 2.3.3 instead of the pinned 2.2.2 changed which logistic fits converged.
A variable dropped out of the functional-form family, the family shrank from
nine tests to eight, and the FDR correction therefore became slightly weaker, so
a published q-value moved. Nothing raised. No traceback, no warning that
survived to the terminal -- just a README that quietly no longer matched the
code that produced it.

A dependency bump should turn that into a red build, not a discrepancy someone
notices six months later in an interview.

If a test here fails, do not adjust the constant to make it pass. Work out what
changed, decide whether the new number is more correct, and then update BOTH the
constant and every place the old one is quoted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from support2 import analysis_frames

ROOT = Path(__file__).resolve().parent.parent


def _load(stem: str):
    """Import a numbered analysis script, whose name is not a valid identifier."""
    path = ROOT / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("_", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cohort():
    return analysis_frames()


class TestCohortShape:
    def test_training_cohort_size(self, cohort):
        assert len(cohort.chf_train) == 978
        assert int(cohort.chf_train["death"].sum()) == 599

    def test_implausible_cell_count(self, cohort):
        """Quoted in 01_eda.py A6 and 02_profile.py A8."""
        assert cohort.n_voided == 349

    def test_holdout_size(self, cohort):
        assert cohort.n_test == 2731


class TestHeadlineFinding:
    """The BUN follow-up artefact -- the result the README leads with."""

    @pytest.fixture(scope="class")
    def comparison(self, cohort):
        return _load("01_eda").compute_binary_vs_logrank(cohort.full_train)

    def test_bun_follow_up_ratio(self, comparison):
        bun = comparison.set_index("variable").loc["bun"]
        assert bun.fu_ratio == pytest.approx(2.55, abs=0.01)

    def test_bun_binary_gap_is_large_but_logrank_is_null(self, comparison):
        bun = comparison.set_index("variable").loc["bun"]
        assert bun.gap_pp == pytest.approx(20.9, abs=0.1)
        assert bun.p_binary < 0.001
        assert bun.q_logrank > 0.5, "the whole point: no hazard difference"
        assert bun.verdict == "follow-up artefact"

    def test_survivors_are_the_interview_variables(self, comparison):
        real = set(comparison.loc[comparison.verdict == "real signal", "variable"])
        assert real == {"income", "adlp"}

    def test_lab_variables_are_all_artefact(self, comparison):
        art = set(comparison.loc[comparison.verdict == "follow-up artefact", "variable"])
        assert {"bun", "urine", "glucose"} <= art

    def test_education_does_not_replicate(self, comparison):
        """Reported in the README as a finding that failed under correction."""
        edu = comparison.set_index("variable").loc["edu"]
        assert edu.q_logrank > 0.05
        assert edu.verdict == "no signal"


class TestFunctionalForm:
    @pytest.fixture(scope="class")
    def linearity(self, cohort):
        return _load("02_profile").compute_linearity(cohort.chf_train)

    def test_family_size_is_fixed(self, linearity):
        """If this drops, a fit stopped converging and the FDR correction just
        got weaker for every other variable. That is the pandas-2.3.3 failure."""
        assert len(linearity) == 9
        assert linearity.p_nonlinearity.notna().sum() == 9, (
            "a fit failed to converge; the family size is preserved but the "
            "correction is now based on fewer live tests"
        )

    def test_creatinine_is_the_only_nonlinear_variable(self, linearity):
        nonlinear = set(linearity.loc[linearity.verdict == "NON-LINEAR", "variable"])
        assert nonlinear == {"crea"}

    def test_creatinine_q_value(self, linearity):
        crea = linearity.set_index("variable").loc["crea"]
        assert crea.q_nonlinearity == pytest.approx(0.036, abs=0.005)
        assert crea.aic_gain > 5

    def test_heart_rate_fails_correction(self, linearity):
        """Significant at p, not at q. The README says so explicitly."""
        hrt = linearity.set_index("variable").loc["hrt"]
        assert hrt.p_nonlinearity < 0.05
        assert hrt.q_nonlinearity > 0.05
        assert hrt.q_nonlinearity == pytest.approx(0.093, abs=0.01)


class TestEnrolmentWaves:
    """The mechanism proof. A 100.0 / 0.9 split is the claim; anything softer
    means the wave cut or the cohort changed."""

    @pytest.fixture(scope="class")
    def enrolment(self, cohort):
        return _load("03_cohort").compute_enrolment(cohort.chf_train)

    def test_censoring_gap_is_empty(self, enrolment):
        assert enrolment["gap_n"] <= 2
        assert min(enrolment["gap_neighbours"]) >= 20

    def test_labs_are_wholly_absent_in_the_early_wave(self, enrolment):
        t = enrolment["table"].set_index("variable")
        for col in ("bun", "urine", "glucose"):
            assert t.loc[col, "missing_early_wave"] == pytest.approx(100.0, abs=0.1)
            assert t.loc[col, "missing_late_wave"] < 10.0

    def test_interview_variables_are_balanced_across_waves(self, enrolment):
        t = enrolment["table"].set_index("variable")
        for col in ("adlp", "income"):
            assert abs(t.loc[col, "difference_pp"]) < 15.0

    def test_converse_separation(self, enrolment):
        c = enrolment["converse"]
        assert c["missing"]["pct_early"] > 95.0
        assert c["recorded"]["pct_early"] < 5.0
