"""
02_profile.py -- Cohort profile, data quality, and functional form.

Continues the question format of 01_eda.py. Answers are held at the bottom.

    Run:  python 02_profile.py

THE QUESTIONS
    Q7   Every clinical paper opens with Table 1: baseline characteristics
         stratified by outcome. Build one. Then justify why it reports
         standardised mean differences instead of the p-values you have
         probably seen in published tables.
    Q8   Albumin in this cohort has a skew of 12.1 against a physiologic range
         of roughly 1-6 g/dL, and mean arterial pressure has a minimum of zero.
         What are you looking at, and what should happen to those rows?
    Q9   Six continuous predictors are strongly right-skewed. Does that matter
         for logistic regression, and is transforming them the right response?
    Q10  Mortality by creatinine octile runs 52, 55, 67, 66, 67, 65 percent.
         What does a linear term in the logit assert about that pattern, and
         what does it cost you when the assertion is wrong?
    Q11  Which predictors are collinear, and why does that matter far more for
         this project than it would for a pure prediction task?

Author: Matthew Moxam
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import viz  # noqa: E402
from support2 import (  # noqa: E402
    CANDIDATE_PREDICTORS,
    OUTCOME_EVENT,
    PHYSIOLOGY,
    chf_cohort,
    load_support2,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

RULE = "=" * 78
SUB = "-" * 78

# Physiologic bounds. Values outside these are recording artefacts, not
# extreme patients -- a mean arterial pressure of zero is not a measurement.
PLAUSIBLE = {
    "meanbp": (20, 200), "hrt": (20, 250), "resp": (4, 60), "temp": (30, 43),
    "sod": (110, 175), "crea": (0.1, 20), "wblc": (0.1, 100),
    "ph": (6.8, 7.8), "glucose": (20, 1000), "alb": (0.5, 7.0),
    "bili": (0.05, 60), "bun": (1, 250), "pafi": (20, 700),
}


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def question(n: int, text: str) -> None:
    print(f"\n{SUB}\nQUESTION {n}: {text}\n{SUB}")


# ── Q7. Table 1 ──────────────────────────────────────────────────────────────
def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return np.nan if pooled == 0 else (a.mean() - b.mean()) / pooled


def smd_binary(p1: float, p2: float) -> float:
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return np.nan if pooled == 0 else (p1 - p2) / pooled


def q7_table_one(chf: pd.DataFrame) -> pd.DataFrame:
    question(7, "Build Table 1, stratified by outcome. Then justify why it reports\n"
                "standardised mean differences rather than p-values.")

    died = chf[chf[OUTCOME_EVENT] == 1]
    lived = chf[chf[OUTCOME_EVENT] == 0]
    rows = []

    for col in CANDIDATE_PREDICTORS:
        if col not in chf:
            continue
        s = chf[col]
        if s.dtype == object or s.nunique() <= 2:
            # Report the modal / positive level as a percentage.
            if s.dtype == object:
                level = s.mode(dropna=True)
                if level.empty:
                    continue
                level = level.iloc[0]
                p1 = (died[col] == level).mean()
                p2 = (lived[col] == level).mean()
                label = f"{col} = {level}"
            else:
                p1, p2 = died[col].mean(), lived[col].mean()
                label = f"{col} = 1"
            rows.append({"characteristic": label,
                         "died": f"{p1*100:.1f}%", "survived": f"{p2*100:.1f}%",
                         "smd": abs(smd_binary(p1, p2)), "missing_pct": s.isna().mean()*100})
        else:
            m1, m2 = died[col].median(), lived[col].median()
            q1 = died[col].quantile([.25, .75]).values
            q2 = lived[col].quantile([.25, .75]).values
            rows.append({"characteristic": col,
                         "died": f"{m1:.1f} [{q1[0]:.1f}-{q1[1]:.1f}]",
                         "survived": f"{m2:.1f} [{q2[0]:.1f}-{q2[1]:.1f}]",
                         "smd": abs(smd_continuous(died[col], lived[col])),
                         "missing_pct": s.isna().mean()*100})

    t1 = pd.DataFrame(rows).sort_values("smd", ascending=False).reset_index(drop=True)
    t1["imbalanced"] = np.where(t1.smd > 0.1, "yes", "")

    print(f"  Died n={len(died):,}   Survived n={len(lived):,}")
    print("  Continuous: median [IQR].  Binary/categorical: percentage.\n")
    show = t1.copy()
    show["smd"] = show.smd.round(3)
    show["missing_pct"] = show.missing_pct.round(1)
    print(show.to_string(index=False))
    print(f"\n  {int((t1.smd > 0.1).sum())} of {len(t1)} characteristics exceed the "
          f"conventional |SMD| > 0.1 imbalance threshold.")
    return t1


# ── Q8. Data quality ─────────────────────────────────────────────────────────
def q8_plausibility(chf: pd.DataFrame) -> pd.DataFrame:
    question(8, "Albumin skews 12.1 against a physiologic range of ~1-6 g/dL, and\n"
                "mean arterial pressure has a minimum of zero. What are you looking\n"
                "at, and what should happen to those rows?")

    rows = []
    for col, (lo, hi) in PLAUSIBLE.items():
        if col not in chf:
            continue
        s = chf[col].dropna()
        below, above = (s < lo).sum(), (s > hi).sum()
        if below or above:
            offenders = sorted(set(s[(s < lo) | (s > hi)].round(2)))[:6]
            rows.append({"variable": col, "bound": f"[{lo}, {hi}]",
                         "below": below, "above": above,
                         "observed_min": s.min(), "observed_max": s.max(),
                         "offending_values": ", ".join(str(v) for v in offenders)})
    out = pd.DataFrame(rows)
    if out.empty:
        print("  No implausible values detected.")
        return out
    print(out.to_string(index=False))
    total = int(out.below.sum() + out.above.sum())
    print(f"\n  {total} implausible cell values across {len(out)} variables "
          f"({total/(len(chf)*len(PLAUSIBLE))*100:.3f}% of all cells).")
    print("  Note this is a cell-level problem, not a row-level one -- which")
    print("  determines the correct remedy.")
    return out


# ── Q9. Distributions ────────────────────────────────────────────────────────
def q9_distributions(chf: pd.DataFrame) -> pd.DataFrame:
    question(9, "Six continuous predictors are strongly right-skewed. Does that\n"
                "matter for logistic regression, and is transforming them right?")

    rows = []
    for col in PHYSIOLOGY:
        s = chf[col].dropna()
        if len(s) < 50:
            continue
        pos = s[s > 0]
        rows.append({"variable": col, "n": len(s), "skew": s.skew(),
                     "skew_log": np.log(pos).skew() if len(pos) == len(s) else np.nan,
                     "median": s.median(), "p99_over_median": s.quantile(.99)/s.median()})
    out = pd.DataFrame(rows).sort_values("skew", key=abs, ascending=False)
    show = out.copy()
    for c in ("skew", "skew_log", "p99_over_median"):
        show[c] = show[c].round(2)
    show["median"] = show["median"].round(1)
    print(show.to_string(index=False))
    print("\n  `skew_log` is the skew after a natural-log transform. Where it is")
    print("  much closer to zero, a log transform would work. Where it is not,")
    print("  the shape is not a simple multiplicative one.")
    return out


# ── Q10. Functional form ─────────────────────────────────────────────────────
def q10_linearity(chf: pd.DataFrame) -> pd.DataFrame:
    question(10, "Mortality by creatinine octile runs 52, 55, 67, 66, 67, 65 percent.\n"
                 "What does a linear term in the logit assert about that pattern,\n"
                 "and what does it cost when the assertion is wrong?")

    import statsmodels.formula.api as smf

    print("  Likelihood-ratio test: linear term vs natural cubic spline (4 df).")
    print("  A significant result means linearity is rejected by the data.\n")
    rows = []
    for col in ["age", "crea", "meanbp", "hrt", "resp", "sod", "bun", "wblc", "temp"]:
        lo, hi = PLAUSIBLE.get(col, (-np.inf, np.inf))
        d = chf[[col, OUTCOME_EVENT]].dropna()
        d = d[(d[col] >= lo) & (d[col] <= hi)]   # Q8's bounds, applied
        if len(d) < 100 or d[col].nunique() < 10:
            continue
        d = d.rename(columns={col: "x", OUTCOME_EVENT: "y"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lin = smf.logit("y ~ x", data=d).fit(disp=0)
                spl = smf.logit("y ~ cr(x, df=4)", data=d).fit(disp=0)
            except Exception as exc:
                print(f"  {col:8s} SKIPPED -- fit failed ({type(exc).__name__})")
                continue
        # A spline that failed to converge cannot be compared on likelihood.
        converged = (lin.mle_retvals.get("converged", True)
                     and spl.mle_retvals.get("converged", True))
        if not converged:
            print(f"  {col:8s} SKIPPED -- maximum likelihood did not converge "
                  f"(likely sparse or separated at the spline knots)")
            continue
        lr = 2 * (spl.llf - lin.llf)
        ddf = int(spl.df_model - lin.df_model)
        p = stats.chi2.sf(lr, ddf) if ddf > 0 else np.nan
        rows.append({"variable": col, "n": len(d), "lr_chi2": lr, "df": ddf,
                     "p_nonlinearity": p, "aic_linear": lin.aic, "aic_spline": spl.aic,
                     "aic_gain": lin.aic - spl.aic})
    out = pd.DataFrame(rows).sort_values("lr_chi2", ascending=False)
    show = out.copy()
    show["verdict"] = np.where(show.p_nonlinearity < 0.05, "NON-LINEAR", "linear ok")
    for c in ("lr_chi2", "aic_linear", "aic_spline", "aic_gain"):
        show[c] = show[c].round(1)
    show["p_nonlinearity"] = show.p_nonlinearity.apply(
        lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    print(show.to_string(index=False))
    return out


# ── Q11. Collinearity ────────────────────────────────────────────────────────
def q11_collinearity(chf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    question(11, "Which predictors are collinear, and why does that matter far more\n"
                 "for this project than it would for a pure prediction task?")

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    num = [c for c in CANDIDATE_PREDICTORS
           if c in chf and pd.api.types.is_numeric_dtype(chf[c]) and chf[c].nunique() > 2]
    corr = chf[num].corr(method="spearman")

    pairs = (corr.where(~np.eye(len(corr), dtype=bool))
             .stack().rename("rho").reset_index()
             .rename(columns={"level_0": "a", "level_1": "b"}))
    pairs = pairs[pairs.a < pairs.b].reindex(
        pairs.rho.abs().sort_values(ascending=False).index).dropna().head(10)
    print("  Strongest Spearman correlations (rank-based; these are not normal):")
    print(pairs.round(3).to_string(index=False))

    d = chf[num].dropna()
    vif = pd.DataFrame()
    if len(d) > 50:
        Z = (d - d.mean()) / d.std()
        # A rank-deficient design has an exactly-determined column: some variable
        # is a deterministic function of the others. VIF is undefined there, so
        # find and report the culprits rather than emitting a warning.
        rank, ncol = np.linalg.matrix_rank(Z.values), Z.shape[1]
        aliased = []
        if rank < ncol:
            keep = []
            for c in Z.columns:
                trial = keep + [c]
                if np.linalg.matrix_rank(Z[trial].values) == len(trial):
                    keep.append(c)
                else:
                    aliased.append(c)
            print(f"\n  Design matrix is rank-deficient: rank {rank} of {ncol} columns.")
            print(f"  Exactly determined by the others: {', '.join(aliased)}")
            print("  This is not near-collinearity. It is an identity, and it means one")
            print("  of these columns is a derived summary of the rest.")
            Z = Z[keep]

        X = Z.assign(_const=1.0)
        vif = pd.DataFrame({
            "variable": [c for c in X.columns if c != "_const"],
            "VIF": [variance_inflation_factor(X.values, i)
                    for i, c in enumerate(X.columns) if c != "_const"],
        }).sort_values("VIF", ascending=False)
        vif["flag"] = np.where(vif.VIF > 10, "problem",
                               np.where(vif.VIF > 5, "inspect", ""))
        print(f"\n  Variance inflation factors (complete cases, n={len(d):,}):")
        print(vif.round(2).to_string(index=False))

    events = int(chf[OUTCOME_EVENT].sum())
    print(f"\n  Sample size check: {events:,} events, {len(CANDIDATE_PREDICTORS)} "
          f"candidate predictors.")
    print(f"  Events per variable = {events/len(CANDIDATE_PREDICTORS):.1f} "
          f"(conventional floor: 10).")
    return corr, vif


# ── Figures ──────────────────────────────────────────────────────────────────
def figure_distributions(chf: pd.DataFrame, dist: pd.DataFrame):
    import matplotlib.pyplot as plt

    cols = dist.variable.tolist()[:12]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, col in zip(axes.ravel(), cols):
        s = chf[col].dropna()
        lo, hi = PLAUSIBLE.get(col, (-np.inf, np.inf))
        ok, bad = s[(s >= lo) & (s <= hi)], s[(s < lo) | (s > hi)]
        ax.hist(ok, bins=40, color=viz.SERIES_BLUE, alpha=0.85)
        if len(bad):
            ax.axvline(ok.max(), color=viz.SERIES_ORANGE, lw=1.5, ls="--")
            ax.text(0.97, 0.86, f"{len(bad)} implausible", transform=ax.transAxes,
                    ha="right", fontsize=7.5, color=viz.SERIES_ORANGE, weight="600")
        sk = dist.loc[dist.variable == col, "skew"].iloc[0]
        ax.set_title(f"{col}   skew {sk:.1f}", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", visible=False)
        viz.despine(ax)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distributions of physiologic predictors (CHF cohort)",
                 fontsize=12, fontweight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    viz.caption(fig, "Plotted over physiologically plausible ranges. Dashed rule marks where\n"
                     "implausible values were excluded from the axis, not from the data.")
    return viz.save(fig, "04_distributions.png")


def figure_linearity(chf: pd.DataFrame, lin: pd.DataFrame):
    import matplotlib.pyplot as plt

    top = lin.head(4).variable.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(3.4 * len(top), 4.2))
    for ax, col in zip(np.atleast_1d(axes), top):
        lo, hi = PLAUSIBLE.get(col, (-np.inf, np.inf))
        d = chf[[col, OUTCOME_EVENT]].dropna()
        d = d[(d[col] >= lo) & (d[col] <= hi)]
        d["bin"] = pd.qcut(d[col], 8, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(x=(col, "median"),
                                                p=(OUTCOME_EVENT, "mean"),
                                                n=(OUTCOME_EVENT, "size"))
        se = np.sqrt(g.p * (1 - g.p) / g.n)
        ax.errorbar(g.x, g.p * 100, yerr=se * 100, fmt="o", color=viz.SERIES_BLUE,
                    ms=7, capsize=3, lw=1.5, mec=viz.SURFACE, mew=1.2)
        m, b = np.polyfit(d[col], d[OUTCOME_EVENT] * 100, 1)
        xs = np.linspace(d[col].min(), d[col].max(), 50)
        ax.plot(xs, m * xs + b, color=viz.SERIES_ORANGE, lw=2, ls="--")
        row = lin[lin.variable == col].iloc[0]
        p = "<0.001" if row.p_nonlinearity < 0.001 else f"{row.p_nonlinearity:.3f}"
        ax.set_title(f"{col}\nnon-linearity p {p}", fontsize=10)
        ax.set_xlabel(col)
        viz.despine(ax)
    np.atleast_1d(axes)[0].set_ylabel("Observed mortality (%)")
    fig.suptitle("What a linear term assumes, against what the data does",
                 fontsize=12, fontweight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    viz.caption(fig, "Blue: observed mortality by octile with binomial standard errors. Orange dashed:\n"
                     "the straight line a linear term is constrained to fit. Where they diverge, the\n"
                     "linear model is asserting something the data contradicts.")
    return viz.save(fig, "05_functional_form.png")


def figure_correlation(corr: pd.DataFrame):
    import matplotlib.pyplot as plt

    order = corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.loc[order, order], cmap=viz.diverging_cmap(),
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8.5)
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    for i in range(len(order)):
        for j in range(len(order)):
            v = corr.iloc[i, j]
            if i != j and abs(v) >= 0.4:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="#ffffff", weight="600")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Spearman rho", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Rank correlation among continuous predictors")
    viz.caption(fig, "Spearman rather than Pearson: these variables are skewed and their\n"
                     "relationships monotonic but not linear. Cells |rho| >= 0.4 are labelled.")
    return viz.save(fig, "06_correlation.png")


ANSWERS = """
ANSWERS
{rule}

A7. TABLE 1, AND WHY SMD RATHER THAN P
    Table 1 is the first table in essentially every clinical paper, and its job
    is descriptive: who is in this cohort, and how do the outcome groups differ
    at baseline. Reviewers look for it. Its absence is conspicuous in a way it
    simply is not in a machine learning writeup.

    The p-value convention is a habit the methods literature has been trying to
    kill for two decades. Three reasons it is wrong here:

      * It answers a question nobody asked. A p-value tests whether the groups
        differ in some hypothetical source population. But these ARE the two
        groups -- everyone who died and everyone who did not. There is no
        sampling to make inference about.
      * It is a function of sample size, not of difference. With 1,387 patients
        a clinically trivial gap in sodium reaches p<0.05; with 80 patients a
        large one does not. The p-value is telling you about n.
      * It invites the reader to scan for stars and treat them as importance.

    The standardised mean difference is the difference in units of pooled
    standard deviation, and it does not move with n. |SMD| > 0.1 is the
    conventional threshold for meaningful imbalance -- the same convention
    used to judge covariate balance after propensity score matching.

    Report SMD. If a reviewer insists on p-values, they are cheap to add, but
    lead with the effect size.

A8. THE IMPOSSIBLE VALUES
    Albumin of 29.0 g/dL is not a sick patient. Serum albumin runs 3.5-5.0 in
    health, falls in critical illness, and is incompatible with life much above
    7. A single value of 29 against a median of 3.3 is the entire reason skew
    reads 12.1. It is almost certainly a decimal or unit error -- 2.9 misplaced.

    Zeros in meanbp, hrt and resp are the same class of problem. A mean
    arterial pressure of zero is not a low blood pressure; it is the absence of
    circulation, and these patients were alive at enrolment with follow-up
    recorded afterwards. It is a coding convention for unmeasured, or a
    transcription failure.

    What matters is that this is a CELL-level problem, not a ROW-level one. The
    instinct is to drop the patient. That is wrong twice over: it discards
    every other valid measurement on that patient, and it deletes rows on the
    basis of a data-entry error, which is a form of selection.

    The correct move is to set the offending cells to missing and let the
    imputation machinery handle them alongside the values that were never
    recorded. That keeps the patient, keeps their other data, and -- crucially
    -- makes the decision explicit and reversible, recorded in PLAUSIBLE at the
    top of this file rather than buried in a cleaning script.

    State the bounds before you look at the outcome. Bounds chosen after seeing
    which exclusions help your model are not quality control.

A9. SKEW, AND WHETHER TO TRANSFORM
    First, the answer most people get wrong: logistic regression makes NO
    normality assumption about its predictors. None. The normality assumption
    in linear models is about residuals, and logistic regression does not have
    those in the same sense. Skewed predictors are not, in themselves, a
    violation of anything.

    So skew matters for two narrower reasons:

      * Leverage. A handful of extreme values can dominate the fit of a linear
        term, so the coefficient describes the tail rather than the bulk.
      * Interpretability. A one-unit change in creatinine means something very
        different at 0.8 than at 12.

    A log transform fixes the arithmetic but costs you the clinical reading:
    an odds ratio per log-unit of bilirubin is not a sentence you can say to a
    cardiologist. Given that Q10 shows these relationships are not linear on
    any scale, the better answer is usually to model the shape directly with a
    spline and present the result as a plotted risk curve, rather than to
    transform the variable and still assume linearity in the transformed space.

    Transform when the mechanism is genuinely multiplicative. Otherwise let the
    spline do the work and show the curve.

A10. WHAT A LINEAR TERM ASSERTS
    A linear term in the logit asserts that every one-unit increase in the
    predictor multiplies the odds of death by the same constant, everywhere
    along its range. For creatinine that is a strong claim, and the octile
    pattern -- 52, 55, 67, 66, 67, 65 -- contradicts it directly. Risk steps up
    around 1.2 mg/dL and then flattens. It does not keep climbing.

    A linear fit through that shape does two bad things at once. It understates
    the jump at moderate elevation, which is where most patients actually sit
    and where the clinical decision lives. And it extrapolates a rising slope
    into the far tail, so it overstates risk for patients with creatinine of 8
    or 12 -- the sickest few, where being wrong is least excusable.

    The likelihood-ratio test above formalises it: compare a linear term with a
    natural cubic spline on the same variable and test the difference. Where
    p is small, the data has rejected linearity and you have a measurable AIC
    improvement to show for modelling the shape.

    But read the table honestly, because it does not say what you might expect.
    Only creatinine (p<0.001, AIC gain 10.4) and heart rate (p=0.024) reject
    linearity. Age, mean arterial pressure, temperature and BUN all test as
    adequately linear, and their spline fits are WORSE by AIC -- the extra
    degrees of freedom bought nothing. Eyeballing the age octiles earlier
    suggested curvature; the formal test disagreed. Testing beat assuming in
    both directions, which is the actual lesson.

    So the Harrell-school default -- assume non-linearity for continuous
    predictors and spend a few degrees of freedom on a restricted cubic spline
    -- is the right posture, but it is a hypothesis to test, not a conclusion
    to assert. Fit the spline, test it, and keep it only where the data pays
    for it. Where it does, report the result as a plotted risk curve rather
    than a single odds ratio: a curve is more honest than a number when the
    effect is not constant, and clinicians read curves fluently.

    One warning the table also delivers. `resp` tested non-linear at p=0.040
    before Q8's plausibility bounds were applied and linear at p=0.085 after --
    a verdict flipped by the removal of a single impossible value (a
    respiratory rate of 76). Functional-form conclusions can hang on one bad
    cell. Clean first, then test, and say which order you did it in.

    Do not select the shape by looking at outcomes first and then choosing.
    Pre-specify a spline with a fixed number of knots and report it.

A11. COLLINEARITY, AND WHY IT MATTERS MORE HERE
    Two distinct findings here, and they need different responses.

    The first is not collinearity at all. `adls` and `adlsc` correlate at
    rho = 1.000, and the design matrix comes back rank-deficient with `adlsc`
    exactly determined by the others. That is an identity, not an association:
    `adlsc` is a derived ADL summary reconstructed from the components. Putting
    both in a model is not a statistical judgement call, it is asking the fit
    for a coefficient that does not exist. Drop one, and say which and why.

    The second is real: `bun` and `crea` correlate at rho = 0.79. That is
    clinically unsurprising -- both measure renal function -- and it is exactly
    the case that hurts an interpretable model.

    For pure prediction, collinearity is close to harmless. If two predictors
    carry the same information, a model can lean on either and the predictions
    barely move. This is why the machine learning habit is to ignore it.

    This project is not a pure prediction task. The deliverable is an odds
    ratio per predictor that a cardiologist can act on, and collinearity
    attacks exactly that. It inflates standard errors, so confidence intervals
    widen and a real effect can look null. It makes coefficients unstable, so
    small changes to the cohort flip signs. And it splits shared signal
    arbitrarily between correlated variables, so which one "wins" is close to
    a coin toss -- and you will then be asked to explain, clinically, why the
    winner mattered and the loser did not.

    VIF above about 5 warrants a look; above 10 is a problem. Nothing here
    exceeds 2.7 -- but read that number with suspicion, because the VIF table
    is computed on complete cases and only 136 of 1,387 CHF patients have every
    continuous predictor recorded. That is under 10% of the cohort, and Q3-Q5
    established those patients are not a random 10%. Recompute VIF on the
    imputed data before trusting it. A reassuring diagnostic measured on a
    biased subsample is not reassurance.

    The remedy for genuine collinearity is rarely mechanical deletion. Prefer
    collapsing correlated measures into a single clinically meaningful
    construct, or pre-specifying which member of a correlated set to keep on
    clinical grounds -- decided before fitting, and stated. For bun and crea,
    creatinine is the more standard renal marker in heart failure reporting,
    so it is the defensible one to keep if only one survives.

    Note also the sample size line above. Events per variable is comfortable
    here, which means the constraint on this model is not statistical power. It
    is the interpretability budget: how many coefficients a clinical reader can
    actually hold and act on. That is a smaller number than the data permits,
    and it is the real reason to be selective.
{rule}
"""


def main() -> None:
    viz.apply_style()
    chf = chf_cohort(load_support2())

    header("SUPPORT2 -- cohort profile, data quality, functional form")
    print(f"  CHF cohort: {len(chf):,} patients, {int(chf[OUTCOME_EVENT].sum()):,} deaths")

    q7_table_one(chf)
    q8_plausibility(chf)
    dist = q9_distributions(chf)
    lin = q10_linearity(chf)
    corr, _ = q11_collinearity(chf)

    header("FIGURES")
    for path in (figure_distributions(chf, dist),
                 figure_linearity(chf, lin),
                 figure_correlation(corr)):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(ANSWERS.format(rule=RULE))


if __name__ == "__main__":
    main()
