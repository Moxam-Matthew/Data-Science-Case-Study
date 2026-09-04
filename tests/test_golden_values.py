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

from modelling import default_predictors, make_outcome
from support2 import SEPSIS_LABEL, analysis_frames

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


class TestSepsisReplication:
    """
    The replication arm's findings (12_replication.py, 13_horizon.py).

    Pinned for the same reason as the CHF ones, and for one more: the whole
    claim of the replication is that these numbers were produced by IDENTICAL
    code on a second cohort. If a change to the shared pipeline moves one arm
    and not the other, the comparison silently stops being like-for-like.

    Model AUCs are deliberately absent here, as they are for CHF: refitting a
    nested cross-validation inside a test suite would trade a fast build for a
    guarantee the transcripts already provide.
    """

    @pytest.fixture(scope="class")
    def sepsis(self):
        return analysis_frames(group=SEPSIS_LABEL).chf_train

    @pytest.fixture(scope="class")
    def eda(self, sepsis):
        mod = _load("12_replication")
        return mod.replicate_eda(sepsis, make_outcome(sepsis).values)

    @pytest.fixture(scope="class")
    def horizon(self, sepsis):
        return _load("13_horizon").horizon_accounting(sepsis)

    def test_cohort_is_the_size_the_write_up_claims(self, sepsis):
        assert len(sepsis) == 2458
        assert int(make_outcome(sepsis).sum()) == 1091

    def test_events_per_variable_clears_the_conventional_floor(self, sepsis):
        """The entire premise of the replication: EPV 28 against CHF's 6.4. If
        this drops below 10 the arm no longer tests what it says it tests."""
        mod = _load("12_replication")
        k = mod.design_width(sepsis, default_predictors(sepsis))
        assert k == 39
        assert make_outcome(sepsis).sum() / k == pytest.approx(28.0, abs=0.1)

    def test_the_enrolment_wave_replicates(self, eda):
        w = eda["wave"].set_index("variable")
        for col in ("bun", "urine", "glucose"):
            assert w.loc[col, "missing_early_wave"] > 99.0
            assert w.loc[col, "missing_late_wave"] < 12.0

    def test_the_negative_controls_show_no_wave_structure(self, eda):
        """These carry the argument. Without them the wave split could just be
        separating sicker patients."""
        w = eda["wave"].set_index("variable")
        for col in ("adlp", "income"):
            assert abs(w.loc[col, "missing_early_wave"]
                       - w.loc[col, "missing_late_wave"]) < 15.0

    def test_artefacts_shrink_and_real_effects_grow(self, eda):
        """A51's one-line diagnostic: accounting for exposure time collapses an
        artefact by an order of magnitude and strengthens a real association."""
        a = eda["artefact"].set_index("variable")
        for col in ("bun", "urine", "glucose"):
            assert a.loc[col, "shrinkage"] > 5.0
            assert a.loc[col, "fu_ratio"] > 2.0
        for col in ("adlp", "income"):
            assert a.loc[col, "shrinkage"] < 1.0, "a real effect should GROW"
            assert a.loc[col, "fu_ratio"] == pytest.approx(1.0, abs=0.10)

    def test_the_dnr_contrast_does_not_replicate(self, eda):
        """Reported as a non-replication. If this ever starts matching CHF,
        A51 is wrong and must be rewritten rather than quietly passing."""
        d = eda["dnr"].set_index("dnr")
        before = d.loc["dnr before sadm"]
        assert before.n == 45
        assert before.gap_vs_no_dnr > 40.0
        assert before.p_vs_no_dnr < 0.05

    def test_dropping_the_horizon_doubles_the_wave_contamination(self, horizon):
        """13_horizon.py A54. The any-horizon label absorbs the enrolment
        calendar; the 180-day label largely does not."""
        assert horizon["n_180"] == 1091
        assert horizon["n_any"] == 1440
        assert horizon["gap_any"] > 2.0 * horizon["gap_180"] - 0.5
        assert horizon["fu_ratio"] > 2.0

    def test_a_protocol_flag_predicts_the_wrong_outcome(self, horizon):
        """'BUN was not recorded' carries no clinical information. Against the
        180-day outcome it should be near chance; against any-horizon it is
        not, and that gap is the demonstration."""
        assert horizon["flag_auc_180"] < 0.54
        assert horizon["flag_auc_any"] > horizon["flag_auc_180"] + 0.02
