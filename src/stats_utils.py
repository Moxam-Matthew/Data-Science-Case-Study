"""
Statistical helpers where the naive implementation is wrong in a way that
matters clinically.

Three things live here, each because the obvious approach fails:

  * Multiplicity. Running one test is inference; running sixty and reporting
    the small p-values is selection. Every multi-variable table in this project
    carries a Benjamini-Hochberg q-value.
  * Standardised mean difference for multi-level categoricals. The binary
    formula applied per level is not the right answer.
  * Median follow-up. The median of the time column is a median time-to-event,
    contaminated by the deaths. Follow-up needs the reverse Kaplan-Meier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Multiplicity ─────────────────────────────────────────────────────────────
def add_fdr(table: pd.DataFrame, p_col: str = "p",
            q_col: str = "q_fdr", alpha: float = 0.05) -> pd.DataFrame:
    """
    Append Benjamini-Hochberg q-values and a survives-correction flag.

    BH controls the false discovery rate -- the expected share of claimed
    findings that are false -- rather than the family-wise error rate.
    Bonferroni controls the latter and is far stricter; for exploratory
    screening across many variables, FDR is the conventional and more
    defensible choice. The Bonferroni threshold is reported alongside so a
    reader can apply the stricter bar themselves.
    """
    from statsmodels.stats.multitest import multipletests

    out = table.copy()
    mask = out[p_col].notna()
    out[q_col] = np.nan
    if mask.sum():
        _, q, _, _ = multipletests(out.loc[mask, p_col].values, alpha=alpha,
                                   method="fdr_bh")
        out.loc[mask, q_col] = q
    out["survives_fdr"] = np.where(out[q_col] < alpha, "yes", "")
    out["survives_bonf"] = np.where(out[p_col] < alpha / max(int(mask.sum()), 1),
                                    "yes", "")
    return out


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    return alpha / max(n_tests, 1)


# ── Standardised mean difference ─────────────────────────────────────────────
def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return np.nan if pooled == 0 else (a.mean() - b.mean()) / pooled


def smd_binary(p1: float, p2: float) -> float:
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return np.nan if pooled == 0 else (p1 - p2) / pooled


def smd_categorical(a: pd.Series, b: pd.Series) -> float:
    """
    Multi-level SMD (Yang & Dalton, 2012).

    For a variable with k levels the difference is a (k-1)-vector of proportion
    differences, and the standardised distance is Mahalanobis-like:

        d = sqrt( (p_a - p_b)' S^-1 (p_a - p_b) )

    with S the average of the two multinomial covariance matrices. Computing a
    separate binary SMD per level and quoting the largest is not this quantity
    and does not have the same 0.1 threshold interpretation.
    """
    levels = sorted(set(a.dropna().unique()) | set(b.dropna().unique()))
    if len(levels) < 2:
        return np.nan
    if len(levels) == 2:
        return smd_binary((a == levels[1]).mean(), (b == levels[1]).mean())

    pa = np.array([(a == lv).mean() for lv in levels[:-1]])   # drop last level
    pb = np.array([(b == lv).mean() for lv in levels[:-1]])

    def cov(p):
        S = -np.outer(p, p)
        np.fill_diagonal(S, p * (1 - p))
        return S

    S = (cov(pa) + cov(pb)) / 2
    diff = pa - pb
    try:
        return float(np.sqrt(diff @ np.linalg.pinv(S) @ diff))
    except np.linalg.LinAlgError:
        return np.nan


# ── Follow-up ────────────────────────────────────────────────────────────────
def median_followup(time: pd.Series, event: pd.Series) -> float:
    """
    Median follow-up by reverse Kaplan-Meier.

    The naive `time.median()` answers a different question: it is the median
    time until death *or* censoring, so it is pulled downward by every death
    and is not a measure of how long the cohort was observed. The reverse KM
    inverts the indicator -- censoring becomes the "event" -- and estimates how
    long patients would have been followed had they not died.

    On the CHF cohort the two differ by roughly a factor of three, which is
    material in a project whose central finding concerns follow-up duration.
    """
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(time, 1 - event)
    return float(km.median_survival_time_)


def followup_summary(df: pd.DataFrame, time_col: str, event_col: str) -> dict:
    t, e = df[time_col], df[event_col]
    return {
        "n": len(df),
        "events": int(e.sum()),
        "censored": int((e == 0).sum()),
        "median_followup_reverseKM": median_followup(t, e),
        "median_time_to_event_or_censor": float(t.median()),
        "median_time_among_censored": float(t[e == 0].median()),
        "max_observed": float(t.max()),
    }
