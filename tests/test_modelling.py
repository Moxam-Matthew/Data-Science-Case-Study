"""
Tests for the modelling layer.

The failures guarded here are the ones that do not raise. A leaked imputation,
a mislabelled outcome and a decision curve with the wrong exchange rate all
produce plausible numbers, and the only way to notice is to check them against
something computed independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelling import (
    HORIZON_DAYS,
    MISSINGNESS_INDICATORS,
    PHYSICIAN_BENCHMARK,
    MissingnessIndicator,
    build_pipeline,
    calibration_metrics,
    default_predictors,
    make_outcome,
    net_benefit,
    treat_all_net_benefit,
)
from support2 import OUTCOME_EVENT, OUTCOME_TIME, analysis_frames


@pytest.fixture(scope="module")
def chf():
    return analysis_frames().chf_train


class TestOutcome:
    def test_horizon_has_no_prior_censoring(self, chf):
        """The reason 180 days was chosen. If a future change breaks this, the
        binary label silently starts encoding censored patients as survivors."""
        censored_early = ((chf[OUTCOME_EVENT] == 0)
                          & (chf[OUTCOME_TIME] < HORIZON_DAYS))
        assert not censored_early.any()

    def test_outcome_matches_an_independent_computation(self, chf):
        y = make_outcome(chf)
        manual = ((chf["death"] == 1) & (chf["d.time"] <= HORIZON_DAYS)).astype(int)
        pd.testing.assert_series_equal(y, manual, check_names=False)

    def test_raises_rather_than_mislabelling(self, chf):
        """At a horizon with prior censoring, refuse instead of guessing."""
        with pytest.raises(ValueError, match="censored before"):
            make_outcome(chf, horizon=1500)

    def test_event_rate_is_usable(self, chf):
        y = make_outcome(chf)
        assert 0.10 < y.mean() < 0.45, "outcome should be neither rare nor saturated"

    def test_physician_benchmark_is_never_a_predictor(self, chf):
        assert PHYSICIAN_BENCHMARK not in default_predictors(chf)


class TestPipeline:
    def test_imputer_is_refit_per_fold(self, chf):
        """The single most important guarantee in the project. Imputing once on
        all training data leaks held-out information into the model that
        predicts it, and nothing about the output looks wrong when it happens."""
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        y = make_outcome(chf).values
        preds = default_predictors(chf)
        pipe = build_pipeline(chf, preds, LogisticRegression(max_iter=500))

        states = []
        for tr, _ in StratifiedKFold(4, shuffle=True, random_state=1).split(
                chf[preds], y):
            p = clone(pipe)
            p.fit(chf[preds].iloc[tr], y[tr])
            imp = (p.named_steps["prep"].named_transformers_["num"]
                   .named_steps["impute"])
            states.append(np.round(imp.initial_imputer_.statistics_, 8).tobytes())
        assert len(set(states)) == len(states), (
            "the imputer produced identical fitted state on every fold, which "
            "means it saw the same rows each time"
        )

    def test_indicators_are_added_before_imputation(self, chf):
        """Once the imputer has run there is nothing left to flag."""
        ind = MissingnessIndicator(MISSINGNESS_INDICATORS)
        out = ind.fit_transform(chf[default_predictors(chf)])
        for c in MISSINGNESS_INDICATORS:
            assert f"{c}_missing" in out.columns
            assert out[f"{c}_missing"].sum() == chf[c].isna().sum()

    def test_no_missing_values_reach_the_estimator(self, chf):
        from sklearn.linear_model import LogisticRegression

        y = make_outcome(chf).values
        preds = default_predictors(chf)
        pipe = build_pipeline(chf, preds, LogisticRegression(max_iter=500))
        pipe.fit(chf[preds], y)
        Z = pipe.named_steps["prep"].transform(
            pipe.named_steps["indicators"].transform(chf[preds]))
        assert not np.isnan(Z).any()

    def test_scaling_is_applied_only_where_asked(self, chf):
        from sklearn.linear_model import LogisticRegression

        y = make_outcome(chf).values
        preds = default_predictors(chf)
        for scale, expect_unit_variance in ((True, True), (False, False)):
            pipe = build_pipeline(chf, preds, LogisticRegression(max_iter=2000),
                                  scale=scale)
            pipe.fit(chf[preds], y)
            Z = pipe.named_steps["prep"].transform(
                pipe.named_steps["indicators"].transform(chf[preds]))
            age_col = list(pipe.named_steps["prep"]
                           .get_feature_names_out()).index("age")
            sd = float(np.std(Z[:, age_col]))
            assert (abs(sd - 1) < 0.05) == expect_unit_variance


class TestMetrics:
    def test_calibration_slope_is_one_for_a_perfect_model(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.98, 20000)
        y = rng.binomial(1, p)
        m = calibration_metrics(y, p)
        assert m["calibration_slope"] == pytest.approx(1.0, abs=0.08)
        assert m["calibration_intercept"] == pytest.approx(0.0, abs=0.08)

    def test_overconfident_predictions_give_slope_below_one(self):
        """The signature of overfitting: predictions too spread out."""
        rng = np.random.default_rng(1)
        p_true = rng.uniform(0.1, 0.9, 20000)
        y = rng.binomial(1, p_true)
        logit = np.log(p_true / (1 - p_true))
        p_over = 1 / (1 + np.exp(-logit * 3))       # inflate the spread
        assert calibration_metrics(y, p_over)["calibration_slope"] < 0.6

    def test_auc_is_blind_to_a_monotone_distortion(self):
        """Why calibration must be reported separately: a transform that
        destroys calibration leaves discrimination untouched."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, 5000)
        y = rng.binomial(1, p)
        squashed = p ** 3
        assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, squashed))
        assert (calibration_metrics(y, p)["calibration_slope"]
                != pytest.approx(calibration_metrics(y, squashed)["calibration_slope"],
                                 abs=0.1))

    def test_net_benefit_matches_the_published_formula(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        p = np.array([.9, .8, .2, .7, .6, .1, .1, .1, .1, .1])
        pt = 0.5
        nb = net_benefit(y, p, np.array([pt]))[0]
        flag = p >= pt
        tp, fp = np.sum(flag & (y == 1)), np.sum(flag & (y == 0))
        assert nb == pytest.approx(tp / 10 - (fp / 10) * (pt / (1 - pt)))

    def test_treat_all_equals_the_model_that_flags_everyone(self):
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.3, 500)
        th = np.array([0.1, 0.2, 0.3, 0.4])
        always = net_benefit(y, np.ones_like(y, dtype=float), th)
        np.testing.assert_allclose(always, treat_all_net_benefit(y, th), atol=1e-12)

    def test_treat_none_is_zero_net_benefit(self):
        rng = np.random.default_rng(4)
        y = rng.binomial(1, 0.3, 500)
        th = np.array([0.1, 0.3, 0.5])
        never = net_benefit(y, np.zeros_like(y, dtype=float) - 1, th)
        np.testing.assert_allclose(never, np.zeros_like(th), atol=1e-12)

    def test_net_benefit_at_prevalence_threshold(self):
        """At pt equal to prevalence, treat-all has exactly zero net benefit --
        the point where acting on everyone stops being free."""
        y = np.concatenate([np.ones(300), np.zeros(700)])
        nb = treat_all_net_benefit(y, np.array([0.3]))[0]
        assert nb == pytest.approx(0.0, abs=1e-12)
