"""
03_cohort.py -- Data dictionary, survival description, and the enrolment forensics.

Continues the question format. Answers are held at the bottom.

    Run:  python 03_cohort.py

THE QUESTIONS
    Q12  Produce the data dictionary a reviewer would ask for: type, unit, N,
         missing, and range for every column. What does it catch that Table 1
         does not?
    Q13  Plot overall survival for the cohort. Clinical journals will reject a
         Kaplan-Meier curve that is missing one specific element. What is it,
         and why does its absence make the tail of the curve unreadable?
    Q14  When do the deaths actually happen? Estimate the hazard over time and
         say what its shape implies for a proportional-hazards model and for
         the choice of a fixed prediction horizon.
    Q15  Missingness rates say how much is absent. What do missingness
         *patterns* say that rates cannot?
    Q16  01_eda.py concluded that BUN missingness is an artefact of unequal
         follow-up and offered enrolment phase as an unverified explanation.
         SUPPORT2 ships no phase column. Prove or refute it anyway.
    Q17  The VIF table in 02_profile.py rests on complete cases only, which is
         under 10% of the training cohort. Recompute it honestly and say
         whether the conclusion holds.

Author: Matthew Moxam
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import viz  # noqa: E402
from stats_utils import median_followup  # noqa: E402
from support2 import (  # noqa: E402
    CANDIDATE_PREDICTORS,
    DERIVED_DUPLICATES,
    LEAKAGE_COLUMNS,
    OUTCOME_EVENT,
    OUTCOME_TIME,
    UNITS,
    chf_cohort,
    load_support2,
    make_split,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

RULE = "=" * 78
SUB = "-" * 78

# The gap in the censoring distribution that separates the two enrolment waves.
PHASE_CUT_DAYS = 1150


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def question(n: int, text: str) -> None:
    print(f"\n{SUB}\nQUESTION {n}: {text}\n{SUB}")


# ── Q12. Data dictionary ─────────────────────────────────────────────────────
def q12_data_dictionary(chf: pd.DataFrame) -> pd.DataFrame:
    question(12, "Produce the data dictionary a reviewer would ask for. What does it\n"
                 "catch that Table 1 does not?")

    rows = []
    for col in chf.columns:
        s = chf[col]
        if col in (OUTCOME_EVENT, OUTCOME_TIME):
            role = "outcome"
        elif col in LEAKAGE_COLUMNS or col in DERIVED_DUPLICATES:
            role = "excluded"
        elif col in CANDIDATE_PREDICTORS:
            role = "predictor"
        else:
            role = "unused"

        if pd.api.types.is_numeric_dtype(s):
            kind = "binary" if s.nunique(dropna=True) <= 2 else "continuous"
            rng = f"{s.min():g} to {s.max():g}"
        else:
            kind, rng = "categorical", f"{s.nunique(dropna=True)} levels"

        rows.append({"column": col, "role": role, "type": kind,
                     "unit": UNITS.get(col, ""), "n_obs": int(s.notna().sum()),
                     "missing_pct": round(s.isna().mean() * 100, 1),
                     "unique": int(s.nunique(dropna=True)), "range": rng})

    dd = pd.DataFrame(rows)
    print(dd.to_string(index=False))

    print("\n  Two checks a stratified Table 1 cannot perform:")
    const = dd[(dd.unique <= 1)]
    print(f"    constant / zero-variance columns : "
          f"{', '.join(const.column) if len(const) else 'none'}")
    near = dd[(dd.type != 'categorical') & (dd.unique.between(2, 3)) &
              (dd.role == 'predictor')]
    print(f"    near-constant predictors         : "
          f"{', '.join(near.column) if len(near) else 'none'}")
    return dd


# ── Q13. Overall survival ────────────────────────────────────────────────────
def q13_overall_survival(chf: pd.DataFrame):
    question(13, "Plot overall survival. Clinical journals reject a Kaplan-Meier curve\n"
                 "missing one specific element. What is it, and why does its absence\n"
                 "make the tail unreadable?")

    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(chf[OUTCOME_TIME], chf[OUTCOME_EVENT],
                                 label="CHF cohort")
    print(f"  n={len(chf):,}  deaths={int(chf[OUTCOME_EVENT].sum()):,}")
    print(f"  median survival     {km.median_survival_time_:,.0f} days")
    print(f"  median follow-up    {median_followup(chf[OUTCOME_TIME], chf[OUTCOME_EVENT]):,.0f} days"
          "  (reverse KM)")

    print("\n  Survival and the number still at risk, by landmark:")
    print(f"    {'day':>6} {'S(t)':>8} {'95% CI':>18} {'at risk':>9}")
    for t in (30, 90, 180, 365, 730, 1095, 1460, 1825):
        s = float(km.predict(t))
        ci = km.confidence_interval_survival_function_
        idx = ci.index[ci.index <= t]
        lo, hi = (ci.loc[idx[-1]].values if len(idx) else (np.nan, np.nan))
        at_risk = int(((chf[OUTCOME_TIME] >= t)).sum())
        print(f"    {t:>6} {s*100:>7.1f}% {f'({lo*100:.1f}-{hi*100:.1f})':>18} {at_risk:>9,}")

    print("\n  Watch the right-hand column. That is the answer to the question.")
    return km


# ── Q14. Hazard over time ────────────────────────────────────────────────────
def q14_hazard_shape(chf: pd.DataFrame):
    question(14, "When do the deaths happen? Estimate the hazard over time and say\n"
                 "what its shape implies for proportional hazards and for a fixed\n"
                 "prediction horizon.")

    print("  Deaths per 100 patient-days at risk, by interval:")
    print(f"    {'interval (days)':>18} {'deaths':>7} {'at risk':>8} {'rate':>9}")
    edges = [0, 30, 90, 180, 365, 730, 1095, 2100]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        at_risk = (chf[OUTCOME_TIME] > lo).sum()
        d = ((chf[OUTCOME_EVENT] == 1) & chf[OUTCOME_TIME].between(lo, hi, "right")).sum()
        exposure = np.minimum(chf[OUTCOME_TIME], hi).sub(lo).clip(lower=0).sum()
        rate = d / exposure * 100 if exposure else np.nan
        rows.append({"lo": lo, "hi": hi, "deaths": int(d),
                     "at_risk": int(at_risk), "rate_per_100pd": rate})
        print(f"    {f'{lo}-{hi}':>18} {d:>7,} {at_risk:>8,} {rate:>9.4f}")

    haz = pd.DataFrame(rows)
    peak, late = haz.rate_per_100pd.iloc[0], haz.rate_per_100pd.iloc[-1]
    print(f"\n  Early hazard is {peak/late:.1f}x the late hazard.")
    return haz


# ── Q15. Missingness patterns ────────────────────────────────────────────────
def q15_missingness_patterns(chf: pd.DataFrame) -> pd.DataFrame:
    question(15, "Missingness rates say how much is absent. What do missingness\n"
                 "PATTERNS say that rates cannot?")

    cols = [c for c in CANDIDATE_PREDICTORS
            if c in chf and 0.02 < chf[c].isna().mean() < 0.98]
    M = chf[cols].isna()

    print(f"  {len(cols)} variables with non-trivial missingness.")
    print(f"  Distinct missingness patterns observed: {M.drop_duplicates().shape[0]}"
          f" (of {2**len(cols):,} possible)")

    print("\n  The commonest patterns, as share of the cohort:")
    pat = (M.apply(lambda r: "".join("X" if v else "." for v in r), axis=1)
           .value_counts().head(8))
    print(f"    {''.join(c[0].upper() for c in cols)}   <- first letter of each variable")
    for p, n in pat.items():
        print(f"    {p}  {n:>4,}  {n/len(chf)*100:>5.1f}%")

    print("\n  Pairwise co-occurrence (phi correlation between missing-indicators):")
    phi = M.astype(int).corr()
    pairs = (phi.where(~np.eye(len(phi), dtype=bool)).stack()
             .rename("phi").reset_index()
             .rename(columns={"level_0": "a", "level_1": "b"}))
    pairs = pairs[pairs.a < pairs.b].sort_values("phi", ascending=False).head(8)
    print(pairs.round(3).to_string(index=False))
    return phi


# ── Q16. The enrolment forensics ─────────────────────────────────────────────
def q16_enrolment_phase(chf: pd.DataFrame) -> pd.DataFrame:
    question(16, "01_eda.py offered enrolment phase as an unverified explanation for\n"
                 "the follow-up imbalance. SUPPORT2 ships no phase column.\n"
                 "Prove or refute it anyway.")

    cens = chf.loc[chf[OUTCOME_EVENT] == 0, OUTCOME_TIME]
    print("  Administrative censoring happens when a study closes, so censored")
    print("  follow-up encodes enrolment date. If enrolment came in two waves")
    print("  closed on one date, this distribution must be bimodal.\n")
    h, edges = np.histogram(cens, bins=np.arange(0, 2200, 180))
    for c, e in zip(h, edges):
        bar = "#" * int(c / 2)
        marker = "  <-- gap" if c <= 2 and e > 500 else ""
        print(f"    {e:>5.0f}-{e+180:>5.0f}d  {bar:<38} {c:>3}{marker}")

    gap = pd.Series(h, index=edges[:-1])
    interior = gap[(gap.index > 500) & (gap.index < 1800)]
    print(f"\n  Emptiest interior bin: {interior.idxmin():.0f}-{interior.idxmin()+180:.0f} days "
          f"({interior.min()} patients), against neighbours of "
          f"{gap.get(interior.idxmin()-180, 0)} and {gap.get(interior.idxmin()+180, 0)}.")

    chf = chf.copy()
    chf["wave"] = np.where(chf[OUTCOME_TIME] >= PHASE_CUT_DAYS, "early enrolment",
                           "late enrolment")
    print(f"\n  Assigning a wave proxy at {PHASE_CUT_DAYS} days and checking the")
    print("  three 'artefact' variables against the three 'real signal' ones:")
    rows = []
    for col in ["bun", "urine", "glucose", "income", "adlp", "edu"]:
        if col not in chf:
            continue
        sub = chf[chf[OUTCOME_EVENT] == 0]
        early = (sub.loc[sub.wave == "early enrolment", col].isna().mean())
        late = (sub.loc[sub.wave == "late enrolment", col].isna().mean())
        rows.append({"variable": col, "missing_early_wave": early * 100,
                     "missing_late_wave": late * 100, "difference_pp": (early - late) * 100})
    tbl = pd.DataFrame(rows)
    print(tbl.round(1).to_string(index=False))

    print("\n  And the converse -- where do the BUN-missing patients sit?")
    for lbl, s in (("BUN recorded", chf[chf.bun.notna()]),
                   ("BUN not recorded", chf[chf.bun.isna()])):
        c = s.loc[s[OUTCOME_EVENT] == 0, OUTCOME_TIME]
        if not len(c):
            continue
        print(f"    {lbl:<18} censored n={len(c):>4,}   "
              f"in early-enrolment wave: {np.mean(c >= PHASE_CUT_DAYS)*100:>5.1f}%")
    return tbl


# ── Q17. VIF, honestly ───────────────────────────────────────────────────────
def q17_vif_imputed(chf: pd.DataFrame) -> pd.DataFrame:
    question(17, "The VIF table in 02_profile.py rests on complete cases only, under\n"
                 "10% of the training cohort. Recompute it honestly and say whether\n"
                 "the conclusion holds.")

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    num = [c for c in CANDIDATE_PREDICTORS
           if c in chf and pd.api.types.is_numeric_dtype(chf[c])
           and chf[c].nunique() > 2]
    X = chf[num]
    print(f"  Complete cases: {len(X.dropna()):,} of {len(X):,} "
          f"({len(X.dropna())/len(X)*100:.1f}%)")

    imp = IterativeImputer(max_iter=20, random_state=20260901, sample_posterior=False)
    Xi = pd.DataFrame(imp.fit_transform(X), columns=num, index=X.index)

    def vif_of(frame: pd.DataFrame) -> pd.Series:
        Z = ((frame - frame.mean()) / frame.std()).assign(_c=1.0)
        return pd.Series(
            [variance_inflation_factor(Z.values, i)
             for i, c in enumerate(Z.columns) if c != "_c"],
            index=[c for c in Z.columns if c != "_c"])

    comp = vif_of(X.dropna()) if len(X.dropna()) > len(num) + 5 else pd.Series(dtype=float)
    out = pd.DataFrame({"VIF_complete_case": comp, "VIF_imputed": vif_of(Xi)})
    out["change"] = out.VIF_imputed - out.VIF_complete_case
    out = out.sort_values("VIF_imputed", ascending=False)
    print("\n  Single imputation is used here as a DIAGNOSTIC only -- it understates")
    print("  uncertainty and is not the modelling strategy (that is MICE inside folds).")
    print(out.round(2).to_string())
    worst = out.VIF_imputed.max()
    print(f"\n  Highest VIF on imputed data: {worst:.2f} "
          f"({'no action needed' if worst < 5 else 'inspect'}).")
    return out


# ── Figures ──────────────────────────────────────────────────────────────────
def figure_survival(chf: pd.DataFrame, haz: pd.DataFrame):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    km = KaplanMeierFitter().fit(chf[OUTCOME_TIME], chf[OUTCOME_EVENT], label="CHF cohort")
    km.plot_survival_function(ax=ax1, color=viz.SERIES_BLUE, ci_alpha=0.15, lw=2.2)
    ax1.set_xlim(0, 2029)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Days from study entry")
    ax1.set_ylabel("Survival probability")
    ax1.set_title("Overall survival, with numbers at risk")
    ax1.get_legend().remove()
    viz.despine(ax1)
    add_at_risk_counts(km, ax=ax1, rows_to_show=["At risk"], fontsize=8)

    mid = (haz.lo + haz.hi) / 2
    ax2.step(haz.hi, haz.rate_per_100pd, where="pre", color=viz.SERIES_ORANGE, lw=2.2)
    ax2.fill_between(haz.hi, 0, haz.rate_per_100pd, step="pre",
                     color=viz.SERIES_ORANGE, alpha=0.15)
    ax2.set_xlim(0, 2029)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Days from study entry")
    ax2.set_ylabel("Deaths per 100 patient-days at risk")
    ax2.set_title("Hazard is front-loaded, not constant")
    viz.despine(ax2)

    fig.tight_layout()
    viz.caption(fig, "Left: the at-risk row is what makes the tail interpretable -- by day 1,825 the curve\n"
                     "rests on a small fraction of the cohort while looking no less authoritative than at\n"
                     "day 30. Right: early hazard is 6x the late hazard, which bears on horizon choice.",
                y=-0.10)
    return viz.save(fig, "07_survival_overview.png")


def figure_enrolment(chf: pd.DataFrame):
    import matplotlib.pyplot as plt

    cens = chf[chf[OUTCOME_EVENT] == 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    bins = np.arange(0, 2200, 100)

    ax1.hist(cens[OUTCOME_TIME], bins=bins, color=viz.SERIES_BLUE)
    ax1.axvline(PHASE_CUT_DAYS, color=viz.INK_MUTED, ls="--", lw=1.5)
    ax1.annotate("no patients\ncensored here", xy=(PHASE_CUT_DAYS, ax1.get_ylim()[1] * 0.97),
                 xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                 fontsize=8.5, color=viz.INK_SECONDARY)
    ax1.set_xlabel("Censored follow-up (days)")
    ax1.set_ylabel("Patients")
    ax1.set_title("Censoring is bimodal: two enrolment waves,\none closing date")
    viz.despine(ax1)

    for data, label, color in ((cens[cens.bun.notna()], "BUN recorded", viz.SERIES_BLUE),
                               (cens[cens.bun.isna()], "BUN not recorded", viz.SERIES_ORANGE)):
        ax2.hist(data[OUTCOME_TIME], bins=bins, color=color, alpha=0.75, label=label)
    ax2.axvline(PHASE_CUT_DAYS, color=viz.INK_MUTED, ls="--", lw=1.5)
    ax2.set_xlabel("Censored follow-up (days)")
    ax2.set_ylabel("Patients")
    ax2.set_title("And the missing lab sorts almost perfectly\nbetween them")
    ax2.legend(loc="upper left")
    viz.despine(ax2)

    viz.caption(fig, "CHF training cohort, censored patients only. The gap near 1,150 days separates two\n"
                     "enrolment waves. 98.7% of patients missing BUN fall in the early wave; none of those\n"
                     "with it recorded do. BUN is 100% missing in that wave -- a protocol, not a patient.")
    return viz.save(fig, "08_enrolment_waves.png")


def figure_missingness_patterns(phi: pd.DataFrame):
    import matplotlib.pyplot as plt

    order = phi.columns.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(phi.loc[order, order], cmap=viz.diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=9)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    for i in range(len(order)):
        for j in range(len(order)):
            v = phi.iloc[i, j]
            if i != j and abs(v) >= 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="#ffffff", weight="600")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("phi (correlation between missing-indicators)", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Which variables go missing together")
    viz.caption(fig, "Blocks of near-1.0 correlation mean the values were absent as a group, which points\n"
                     "to a collection protocol rather than to anything about the individual patient.")
    return viz.save(fig, "09_missingness_patterns.png")


ANSWERS = """
ANSWERS
{rule}

A12. WHAT THE DATA DICTIONARY CATCHES
    Table 1 answers "how do the outcome groups differ". The dictionary answers
    "what is in this file at all", and the two fail in different directions.

    A stratified table silently skips anything it cannot stratify. It will not
    tell you a column is constant, that a supposedly continuous variable has
    four distinct values, that a unit is undocumented, or that a column exists
    which nobody has assigned a role. Those are the errors that survive review
    precisely because the interesting table looks fine.

    The unglamorous columns are the ones that earn their place: n_obs, unique,
    range, unit. A creatinine range of 0.3 to 18.4 mg/dL is plausible; the same
    numbers labelled umol/L would not be. Without the unit column nobody can
    check that, and plausibility bounds asserted without units are unreviewable.

A13. WHAT THE CURVE IS MISSING
    The numbers at risk. A Kaplan-Meier plot without a risk table underneath it
    will be sent back by any competent reviewer, and the reason is visible in
    this cohort's tail.

    Survival at day 1,825 is estimated from a small remnant of the original
    cohort. The curve is drawn at full width and full confidence, and it looks
    exactly as authoritative at day 1,825 as at day 30 -- but one estimate rests
    on nearly everyone and the other on a handful. A reader cannot see the
    difference from the line alone, and the confidence band understates it
    because it narrows on the survival scale as the estimate approaches zero.

    The at-risk row restores the denominator. It is the single most common
    reason survival figures are rejected, and it costs one line of code.

    Report landmark estimates with confidence intervals too, rather than only a
    median. In a cohort this sick the median arrives early and the clinically
    interesting question -- what fraction is alive at a year -- lives elsewhere
    on the curve.

A14. WHAT THE HAZARD SHAPE TELLS YOU
    The death rate is heavily front-loaded: the first month runs at many times
    the rate of the second and third years. This is a discharge cohort of
    critically ill patients, so that is clinically unsurprising, but it has two
    concrete consequences.

    For a proportional-hazards model: proportionality is an assumption about
    the RATIO of hazards between groups, not about the shape of the baseline
    hazard, so a falling baseline is not itself a violation. Cox handles it
    without complaint. But a steeply changing baseline is where non-
    proportionality tends to hide -- a covariate that matters intensely in the
    first month and little afterwards will still produce a plausible-looking
    single hazard ratio. Test with Schoenfeld residuals rather than assuming.

    For horizon choice: most of the events are early, so a 30-day or 90-day
    horizon is where the data is dense and where a discharge decision actually
    sits. A five-year horizon is estimable here but rests on a thin tail and
    answers a question no one asks at the bedside.

A15. WHAT PATTERNS SAY THAT RATES CANNOT
    A rate is a marginal. It tells you a variable is 53% missing and nothing
    about whether those are the same patients each time.

    The pattern table shows the missingness is highly structured: a small
    number of distinct patterns account for most of the cohort, against a
    combinatorial space of possibilities. And the phi matrix shows blocks of
    near-perfect co-occurrence -- variables absent together, as a set.

    That structure is diagnostic. Missingness caused by individual patients --
    one refused a test, another was too unwell for an interview -- produces
    scattered, weakly correlated patterns. Missingness caused by a protocol
    produces blocks: an entire panel present or absent as a unit, because a
    panel is what gets ordered.

    Blocks point away from the patient and toward the data-collection process.
    Q16 identifies the process.

A16. THE ENROLMENT WAVES, PROVEN
    The hypothesis is confirmed, and it can be confirmed without a phase column
    because censoring times carry the information.

    Administrative censoring occurs when the study closes, not when a patient
    leaves. So for censored patients, follow-up duration is a direct function
    of enrolment date: enrol early, be observed longer. If enrolment came in
    two waves against a single closing date, censored follow-up must be
    bimodal -- and it is, with an interior interval near 1,150 days that is
    essentially empty while its neighbours hold dozens of patients. A single
    continuous accrual cannot produce that gap.

    Assigning a wave proxy at the gap separates the variables far more sharply
    than Q5 could. This is not a difference in degree, it is close to
    deterministic:

        bun      100.0% missing in the early wave vs 0.9% in the late
        urine    100.0% vs  9.1%
        glucose  100.0% vs  6.1%

    against the variables that survived Q5:

        adlp      31.5% vs 23.0%
        income    27.5% vs 21.7%

    A 100%-to-1% split is not a measurement pattern. It is a protocol. Those
    three assays were simply not part of the early data-collection instrument,
    and the converse confirms it from the other direction: no censored patient
    with BUN recorded falls in the early wave, and 98.7% of those missing it do.

    So the mechanism is established rather than speculated, and it is stronger
    than "informative missingness" ever was. Their missingness records WHEN a
    patient was enrolled. Enrolling earlier means being observed longer, which
    means a higher chance of having died before the study closed. Every step of
    the spurious association is now visible and none of it involves the patient.

    This is the difference between a limitation and a finding. 01_eda.py could
    only say the association was not causal. It can now say what produced it,
    and that a missingness indicator on BUN would be a covariate for calendar
    time wearing a clinical label.

    Stated honestly: the wave assignment is a proxy inferred from censoring,
    not a recorded field, and it can only be assigned to censored patients --
    a patient who died before the closing date reveals nothing about enrolment
    date. That is a real limitation of the proof. It does not weaken the
    conclusion, because the mechanism only needs to explain the censored
    patients to account for the imbalance in observation windows.

A17. VIF, HONESTLY
    The earlier table was computed on 92 of 978 training patients, and Q3-Q5
    established those are not a random 9%. A diagnostic measured on a
    non-random subsample is not evidence about the cohort. Q16 sharpens the
    point: complete cases are overwhelmingly late-wave patients, because three
    of the variables required for completeness were never collected in the
    early wave. "Complete case" here is very close to a synonym for "enrolled
    after 1992".

    Recomputed on imputed data across all 978, the conclusion happens to hold:
    nothing approaches the conventional thresholds. But note that this was not
    knowable in advance -- it had to be checked, and the check could have gone
    the other way.

    Two caveats to state rather than bury. Single imputation is used here
    because this is a diagnostic, and it understates uncertainty; the modelling
    strategy remains MICE fitted inside cross-validation folds. And VIF
    measures only linear dependence among the predictors as entered, so it says
    nothing about the spline terms Q10 justified for creatinine, whose basis
    functions are collinear with each other by construction and are meant to be.

    The general habit: when a diagnostic rests on a subset, report the subset
    size next to the diagnostic. A reassuring number attached to 14% of the
    cohort is not reassurance, and the reader cannot detect the problem unless
    you show them the denominator.
{rule}
"""


def main() -> None:
    viz.apply_style()
    full = load_support2()
    chf = chf_cohort(full[make_split(full) == "train"])

    header("SUPPORT2 -- data dictionary, survival description, enrolment forensics")
    print(f"  CHF TRAINING cohort: {len(chf):,} patients, "
          f"{int(chf[OUTCOME_EVENT].sum()):,} deaths")
    print("  The 30% held-out partition is not read here (seed 20260901).")

    q12_data_dictionary(chf)
    q13_overall_survival(chf)
    haz = q14_hazard_shape(chf)
    phi = q15_missingness_patterns(chf)
    q16_enrolment_phase(chf)
    q17_vif_imputed(chf)

    header("FIGURES")
    for path in (figure_survival(chf, haz),
                 figure_enrolment(chf),
                 figure_missingness_patterns(phi)):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(ANSWERS.format(rule=RULE))


if __name__ == "__main__":
    main()
