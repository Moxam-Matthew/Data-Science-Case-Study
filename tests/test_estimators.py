"""
Tests for the hand-rolled estimators.

Not coverage theatre. Each test protects a specific way this project could be
silently wrong, and every one of them is an estimator whose output appears in
the README as a claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stats_utils import (
    add_fdr,
    median_followup,
    smd_binary,
    smd_categorical,
    smd_continuous,
)


class TestSMD:
    """The only estimator here that is implemented from a paper rather than a
    library. If Yang & Dalton is coded wrong, every Table 1 imbalance flag is
    wrong, and nothing else in the project would notice."""

    def test_binary_matches_closed_form(self):
        a = pd.Series([1] * 30 + [0] * 70)
        b = pd.Series([1] * 50 + [0] * 50)
        p1, p2 = 0.30, 0.50
        expected = (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        assert smd_categorical(a, b) == pytest.approx(expected, abs=1e-9)

    def test_continuous_matches_closed_form(self):
        a = pd.Series([1.0, 2, 3, 4, 5])
        b = pd.Series([3.0, 4, 5, 6, 7])
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        assert smd_continuous(a, b) == pytest.approx((a.mean() - b.mean()) / pooled)

    def test_identical_groups_give_zero(self):
        s = pd.Series(list("aabbcc"))
        assert abs(smd_categorical(s, s.copy())) == pytest.approx(0, abs=1e-9)
        x = pd.Series([1.0, 2, 3, 4])
        assert abs(smd_continuous(x, x.copy())) == pytest.approx(0, abs=1e-9)

    def test_multilevel_is_not_the_per_level_binary(self):
        """The whole point of Yang & Dalton: for k>2 the Mahalanobis-style
        distance is a different quantity from any per-level binary SMD, and
        quoting the largest per-level value instead is the common error."""
        a = pd.Series(["x"] * 50 + ["y"] * 30 + ["z"] * 20)
        b = pd.Series(["x"] * 20 + ["y"] * 30 + ["z"] * 50)
        multi = smd_categorical(a, b)
        per_level = max(
            abs(smd_binary((a == lv).mean(), (b == lv).mean())) for lv in "xyz"
        )
        assert multi > 0
        assert multi != pytest.approx(per_level, abs=1e-6)

    def test_multilevel_is_symmetric_and_nonnegative(self):
        a = pd.Series(["x"] * 50 + ["y"] * 30 + ["z"] * 20)
        b = pd.Series(["x"] * 20 + ["y"] * 30 + ["z"] * 50)
        assert smd_categorical(a, b) == pytest.approx(smd_categorical(b, a))
        assert smd_categorical(a, b) >= 0


class TestMedianFollowup:
    """The reverse-KM estimator underpins the project's headline concept. A
    silent error here would not raise -- it would just print a plausible
    number, which is exactly how the naive `time.median()` version survived."""

    def test_hand_computed_reverse_km(self):
        """Six patients. Censoring is the 'event' for follow-up, so the reverse
        KM is over the three censored times 4, 8, 12 with deaths as censored.

        Product-limit on the censoring indicator:
          t=4  : at risk 6, censor-events 1 -> S=5/6      = 0.833
          t=8  : at risk 4, censor-events 1 -> S=0.833*3/4= 0.625
          t=12 : at risk 2, censor-events 1 -> S=0.625*1/2= 0.3125
        S first drops to <= 0.5 at t=12, so the median follow-up is 12.
        """
        time = pd.Series([2, 4, 6, 8, 10, 12])
        event = pd.Series([1, 0, 1, 0, 1, 0])
        assert median_followup(time, event) == pytest.approx(12.0)

    def test_differs_from_naive_median_when_deaths_are_early(self):
        """The failure this replaced: early deaths drag the naive median down,
        so it reports a number far smaller than actual follow-up."""
        time = pd.Series([1, 1, 1, 1, 100, 200, 300, 400])
        event = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
        assert time.median() < 100
        assert median_followup(time, event) > time.median()

    def test_all_censored_equals_plain_median(self):
        time = pd.Series([10.0, 20, 30, 40, 50])
        event = pd.Series([0, 0, 0, 0, 0])
        assert median_followup(time, event) == pytest.approx(time.median())


class TestFDR:
    """A correction applied to the wrong column, or to a family whose size
    changes with the results, silently alters which findings are reported."""

    def test_matches_statsmodels(self):
        from statsmodels.stats.multitest import multipletests

        p = [0.001, 0.008, 0.021, 0.04, 0.12, 0.3, 0.5, 0.7, 0.9]
        got = add_fdr(pd.DataFrame({"p": p}))
        _, expected, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        np.testing.assert_allclose(got["q_fdr"].values, expected, rtol=1e-12)

    def test_q_is_never_below_p(self):
        p = [0.001, 0.01, 0.02, 0.04, 0.2, 0.6]
        got = add_fdr(pd.DataFrame({"p": p}))
        assert (got.q_fdr >= got.p - 1e-12).all()

    def test_nan_rows_stay_in_the_family(self):
        """A non-converging fit must keep its row. Dropping it shrinks the
        denominator and inflates every other q-value -- manufacturing
        significance without anyone noticing."""
        with_nan = add_fdr(pd.DataFrame({"p": [0.01, 0.02, np.nan, 0.04]}))
        assert len(with_nan) == 4
        assert with_nan.q_fdr.isna().sum() == 1
        assert with_nan.survives_fdr.iloc[2] == ""

    def test_bonferroni_flag_uses_family_size(self):
        p = [0.006] * 9
        got = add_fdr(pd.DataFrame({"p": p}))
        # 0.05/9 = 0.00556, so 0.006 must NOT clear Bonferroni.
        assert (got.survives_bonf == "").all()
