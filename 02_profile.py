"""
02_profile.py -- Cohort profile, data quality, and functional form.

Continues the question format of 01_eda.py. Answers are held at the bottom, and
every number in them is interpolated from the run.

    Run:  python 02_profile.py

THE QUESTIONS
    Q7   Every clinical paper opens with Table 1: baseline characteristics
         stratified by outcome. Build one. Then justify why it reports
         standardised mean differences instead of the p-values you have
         probably seen in published tables.
    Q8   Albumin arrives with a value physiologically incompatible with life,
         and mean arterial pressure with a minimum of zero. What are you
         looking at, and what should happen to those rows?
    Q9   Several continuous predictors are strongly right-skewed. Does that
         matter for logistic regression, and is transforming them the right
         response?
    Q10  Mortality by creatinine octile rises and then flattens. What does a
         linear term in the logit assert about that pattern, and what does it
         cost you when the assertion is wrong?
    Q11  Which predictors are collinear, and why does that matter far more for
         this project than it would for a pure prediction task?

Author: Matthew Moxam
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import viz
from report import (
    RULE,
    Facts,
    configure_pandas,
    fmt_p,
    header,
    question,
    render_answers,
    run_and_capture,
)
from stats_utils import add_fdr, smd_categorical, smd_continuous
from support2 import (
    CANDIDATE_PREDICTORS,
    DERIVED_DUPLICATES,
    OUTCOME_EVENT,
    PHYSIOLOGY,
    PLAUSIBLE_BOUNDS,
    UNITS,
    analysis_frames,
    find_implausible,
)

OUT_DIR = Path(__file__).resolve().parent / "output"

# Variables carried into the functional-form family. Fixed here, before the
# tests run, so the family size cannot change with the results -- a variable
# that fails to converge keeps its row with p = NaN rather than vanishing and
# silently shrinking the multiplicity correction.
LINEARITY_FAMILY = ["age", "crea", "meanbp", "hrt", "resp", "sod", "bun",
                    "wblc", "temp"]


# ═══ Q7. Table 1 ═════════════════════════════════════════════════════════════
def compute_table_one(chf: pd.DataFrame) -> pd.DataFrame:
    died, lived = chf[chf[OUTCOME_EVENT] == 1], chf[chf[OUTCOME_EVENT] == 0]
    rows = []
    for col in CANDIDATE_PREDICTORS:
        if col not in chf:
            continue
        s = chf[col]
        is_cat = (s.dtype == object) or (s.nunique(dropna=True) <= 2)
        if is_cat:
            var_smd = abs(smd_categorical(died[col], lived[col]))
            levels = sorted(s.dropna().unique(), key=str)
            for i, lv in enumerate(levels):
                rows.append({
                    "characteristic": f"{col} = {lv}" if len(levels) > 1 else col,
                    "unit": "%",
                    "died": f"{(died[col] == lv).mean()*100:.1f}",
                    "survived": f"{(lived[col] == lv).mean()*100:.1f}",
                    "smd": var_smd if i == 0 else np.nan,
                    "missing_pct": s.isna().mean() * 100})
        else:
            q1 = died[col].quantile([.25, .75]).values
            q2 = lived[col].quantile([.25, .75]).values
            rows.append({
                "characteristic": col, "unit": UNITS.get(col, ""),
                "died": f"{died[col].median():.1f} [{q1[0]:.1f}-{q1[1]:.1f}]",
                "survived": f"{lived[col].median():.1f} [{q2[0]:.1f}-{q2[1]:.1f}]",
                "smd": abs(smd_continuous(died[col], lived[col])),
                "missing_pct": s.isna().mean() * 100})

    t1 = pd.DataFrame(rows)
    t1["_var"] = t1.characteristic.str.split(" = ").str[0]
    key = t1.groupby("_var").smd.transform("max")
    t1 = (t1.assign(_key=key).sort_values(["_key", "_var"], ascending=[False, True])
          .drop(columns=["_key", "_var"]).reset_index(drop=True))
    t1["imbalanced"] = np.where(t1.smd > 0.1, "yes", "")
    return t1


def report_table_one(t1: pd.DataFrame, chf: pd.DataFrame) -> None:
    question(7, "Build Table 1, stratified by outcome. Then justify why it reports\n"
                "standardised mean differences rather than p-values.")
    n_died = int((chf[OUTCOME_EVENT] == 1).sum())
    print(f"  Died n={n_died:,}   Survived n={len(chf)-n_died:,}")
    print("  Continuous: median [IQR] in the stated unit. Categorical: percent of group.")
    print("  One SMD per variable (multi-level uses Yang & Dalton), not per level.\n")
    show = t1.copy()
    show["smd"] = show.smd.round(3).fillna("")
    show["missing_pct"] = show.missing_pct.round(1)
    print(show.to_string(index=False))
    n_vars = int(t1.smd.notna().sum())
    print(f"\n  {int((t1.smd > 0.1).sum())} of {n_vars} variables exceed the "
          f"conventional |SMD| > 0.1 imbalance threshold.")


# ═══ Q8. Data quality ════════════════════════════════════════════════════════
def report_plausibility(caught: pd.DataFrame, n_voided: int, n_cells: int) -> None:
    question(8, "Albumin arrives with a value physiologically incompatible with life,\n"
                "and mean arterial pressure with a minimum of zero. What are you\n"
                "looking at, and what should happen to those rows?")
    print("  Bounds are declared in support2.PLAUSIBLE_BOUNDS and applied by")
    print("  analysis_frames() before any script sees the data. This reports what")
    print("  was caught on the training rows, from the uncleaned source.\n")
    if caught.empty:
        print("  No implausible values detected.")
        return
    print(caught.to_string(index=False))
    total = int(caught.below.sum() + caught.above.sum())
    print(f"\n  {total} implausible cells on the CHF training rows "
          f"({total/n_cells*100:.3f}% of bounded cells).")
    print(f"  {n_voided} across the whole file, all already set to missing.")
    print("  Note this is a CELL-level problem, not a ROW-level one -- which")
    print("  determines the correct remedy.")


# ═══ Q9. Distributions ═══════════════════════════════════════════════════════
def compute_distributions(chf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in PHYSIOLOGY:
        s = chf[col].dropna()
        if len(s) < 50:
            continue
        pos = s[s > 0]
        rows.append({"variable": col, "unit": UNITS.get(col, ""), "n": len(s),
                     "skew": s.skew(),
                     "skew_log": np.log(pos).skew() if len(pos) == len(s) else np.nan,
                     "median": s.median(),
                     "p99_over_median": s.quantile(.99) / s.median()})
    return pd.DataFrame(rows).sort_values("skew", key=abs, ascending=False)


def report_distributions(dist: pd.DataFrame) -> None:
    question(9, "Several continuous predictors are strongly right-skewed. Does that\n"
                "matter for logistic regression, and is transforming them right?")
    show = dist.copy()
    for c in ("skew", "skew_log", "p99_over_median"):
        show[c] = show[c].round(2)
    show["median"] = show["median"].round(1)
    print(show.to_string(index=False))
    print("\n  `skew_log` is the skew after a natural-log transform. Where it is much")
    print("  closer to zero a log transform would work; where it is not, the shape")
    print("  is not a simple multiplicative one.")


# ═══ Q10. Functional form ════════════════════════════════════════════════════
def compute_linearity(chf: pd.DataFrame) -> pd.DataFrame:
    """
    Likelihood-ratio test of a linear term against a natural cubic spline.

    Every variable in LINEARITY_FAMILY keeps a row. A fit that fails to converge
    yields p = NaN rather than being dropped, because dropping it would shrink
    the multiplicity family and silently inflate everyone else's q-value.
    """
    import statsmodels.formula.api as smf

    rows = []
    for col in LINEARITY_FAMILY:
        rec = {"variable": col, "n": 0, "lr_chi2": np.nan, "df": 0,
               "p_nonlinearity": np.nan, "aic_linear": np.nan,
               "aic_spline": np.nan, "aic_gain": np.nan, "note": ""}
        if col not in chf:
            rec["note"] = "absent"
            rows.append(rec)
            continue

        d = chf[[col, OUTCOME_EVENT]].dropna().rename(
            columns={col: "x", OUTCOME_EVENT: "y"})
        rec["n"] = len(d)
        if len(d) < 100 or d["x"].nunique() < 10:
            rec["note"] = "too sparse"
            rows.append(rec)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lin = smf.logit("y ~ x", data=d).fit(disp=0, method="bfgs", maxiter=200)
                spl = smf.logit("y ~ cr(x, df=4)", data=d).fit(
                    disp=0, method="bfgs", maxiter=200)
            except Exception as exc:
                rec["note"] = f"fit failed: {type(exc).__name__}"
                rows.append(rec)
                continue

        if not (lin.mle_retvals.get("converged", True)
                and spl.mle_retvals.get("converged", True)):
            rec["note"] = "did not converge"
            rows.append(rec)
            continue

        lr = 2 * (spl.llf - lin.llf)
        ddf = int(spl.df_model - lin.df_model)
        rec.update({"lr_chi2": lr, "df": ddf,
                    "p_nonlinearity": stats.chi2.sf(lr, ddf) if ddf > 0 else np.nan,
                    "aic_linear": lin.aic, "aic_spline": spl.aic,
                    "aic_gain": lin.aic - spl.aic})
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("lr_chi2", ascending=False, na_position="last")
    out = add_fdr(out, p_col="p_nonlinearity", q_col="q_nonlinearity")
    out["verdict"] = np.where(out.q_nonlinearity < 0.05, "NON-LINEAR",
                              np.where(out.p_nonlinearity.notna(), "linear ok",
                                       "not tested"))
    return out


def report_linearity(lin: pd.DataFrame) -> None:
    question(10, "Mortality by creatinine octile rises and then flattens. What does a\n"
                 "linear term in the logit assert about that pattern, and what does\n"
                 "it cost when the assertion is wrong?")
    print("  Likelihood-ratio test: linear term vs natural cubic spline.")
    print("  Family is fixed in advance, so a non-converging fit keeps its row.\n")
    show = lin.copy()
    for c in ("lr_chi2", "aic_linear", "aic_spline", "aic_gain"):
        show[c] = show[c].round(1)
    for c in ("p_nonlinearity", "q_nonlinearity"):
        show[c] = show[c].apply(fmt_p)
    print(show[["variable", "n", "lr_chi2", "df", "p_nonlinearity", "q_nonlinearity",
                "aic_linear", "aic_spline", "aic_gain", "verdict", "note"]]
          .to_string(index=False))
    n = len(lin)
    print(f"\n  FDR family size is {n} (fixed), of which "
          f"{int(lin.p_nonlinearity.notna().sum())} produced a p-value.")
    print(f"  The Bonferroni bar would be {0.05/n:.4f}. The verdict column uses q.")


# ═══ Q11. Collinearity ═══════════════════════════════════════════════════════
def compute_collinearity(chf: pd.DataFrame) -> dict:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # adlsc is excluded from CANDIDATE_PREDICTORS on this evidence, so it is
    # re-added here purely to demonstrate the exclusion. A claim in the README
    # that nothing in the repo computes is a claim a reader cannot check.
    demo_cols = [c for c in CANDIDATE_PREDICTORS
                 if c in chf and pd.api.types.is_numeric_dtype(chf[c])
                 and chf[c].nunique() > 2]
    identity = None
    if "adlsc" in chf and "adls" in chf:
        both = chf[["adls", "adlsc"]].dropna()
        identity = {"n": len(both),
                    "spearman": both.adls.corr(both.adlsc, method="spearman"),
                    "pearson": both.adls.corr(both.adlsc),
                    "max_abs_diff": float((both.adls - both.adlsc).abs().max())}

    corr = chf[demo_cols].corr(method="spearman")
    pairs = (corr.where(~np.eye(len(corr), dtype=bool)).stack().rename("rho")
             .reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
    pairs = pairs[pairs.a < pairs.b].reindex(
        pairs.rho.abs().sort_values(ascending=False).index).dropna().head(10)

    d = chf[demo_cols].dropna()
    vif = pd.DataFrame()
    if len(d) > len(demo_cols) + 5:
        Z = (d - d.mean()) / d.std()
        X = Z.assign(_const=1.0)
        vif = pd.DataFrame({
            "variable": [c for c in X.columns if c != "_const"],
            "VIF": [variance_inflation_factor(X.values, i)
                    for i, c in enumerate(X.columns) if c != "_const"],
        }).sort_values("VIF", ascending=False)
        vif["flag"] = np.where(vif.VIF > 10, "problem",
                               np.where(vif.VIF > 5, "inspect", ""))
    return {"corr": corr, "pairs": pairs, "vif": vif,
            "identity": identity, "n_complete": len(d)}


def report_collinearity(r: dict, chf: pd.DataFrame) -> None:
    question(11, "Which predictors are collinear, and why does that matter far more\n"
                 "for this project than it would for a pure prediction task?")

    ident = r["identity"]
    if ident:
        print("  First, a demonstrated exclusion. `adlsc` is not in the candidate set;")
        print("  this is the evidence for that, recomputed so a reader can check it:")
        print(f"    adls vs adlsc   n={ident['n']:,}   spearman={ident['spearman']:.6f}   "
              f"pearson={ident['pearson']:.6f}")
        print(f"    max |difference| = {ident['max_abs_diff']:.6f}")
        print(f"    -> {DERIVED_DUPLICATES['adlsc']}")
        print("    This is an identity, not an association.\n")

    print("  Strongest Spearman correlations among candidate predictors:")
    print(r["pairs"].round(3).to_string(index=False))
    if not r["vif"].empty:
        print(f"\n  Variance inflation factors (complete cases, n={r['n_complete']:,} "
              f"of {len(chf):,} -- see 03_cohort.py Q17):")
        print(r["vif"].round(2).to_string(index=False))

    events = int(chf[OUTCOME_EVENT].sum())
    print(f"\n  Sample size: {events:,} events, {len(CANDIDATE_PREDICTORS)} candidate "
          f"predictors -> {events/len(CANDIDATE_PREDICTORS):.1f} events per variable "
          f"(floor: 10).")


# ═══ Facts ═══════════════════════════════════════════════════════════════════
def collect_facts(t1: pd.DataFrame, caught: pd.DataFrame, dist: pd.DataFrame,
                  lin: pd.DataFrame, coll: dict, n_voided: int) -> Facts:
    d, l = dist.set_index("variable"), lin.set_index("variable")
    nonlinear = l[l.verdict == "NON-LINEAR"].index.tolist()
    worst_skew = dist.iloc[0]
    alb_max = caught.set_index("variable").loc["alb", "observed_max"] if "alb" in caught.variable.values else np.nan
    ident = coll["identity"] or {}
    return Facts(
        n_caught=str(int(caught.below.sum() + caught.above.sum())) if not caught.empty else "0",
        n_voided=str(n_voided),
        alb_max=f"{alb_max:.1f}" if not pd.isna(alb_max) else "n/a",
        # Bracket access, not attribute: `row.skew` resolves to Series.skew, the
        # method, and formats as a bound method rather than raising.
        worst_skew_var=str(worst_skew["variable"]),
        worst_skew=f"{worst_skew['skew']:.1f}",
        worst_skew_log=("n/a" if pd.isna(worst_skew["skew_log"])
                        else f"{worst_skew['skew_log']:.2f}"),
        crea_p=fmt_p(l.loc["crea", "p_nonlinearity"]),
        crea_q=fmt_p(l.loc["crea", "q_nonlinearity"]),
        crea_gain=f"{l.loc['crea', 'aic_gain']:.1f}",
        hrt_p=fmt_p(l.loc["hrt", "p_nonlinearity"]),
        hrt_q=fmt_p(l.loc["hrt", "q_nonlinearity"]),
        nonlinear_vars=", ".join(nonlinear) if nonlinear else "none",
        n_family=str(len(lin)),
        bonf=f"{0.05/len(lin):.4f}",
        n_smd_imbalanced=str(int((t1.smd > 0.1).sum())),
        n_smd_vars=str(int(t1.smd.notna().sum())),
        adlsc_rho=f"{ident.get('spearman', float('nan')):.6f}",
        adlsc_maxdiff=f"{ident.get('max_abs_diff', float('nan')):.6f}",
        adlsc_n=f"{ident.get('n', 0):,}",
        vif_max=f"{coll['vif'].VIF.max():.2f}" if not coll["vif"].empty else "n/a",
        vif_n=f"{coll['n_complete']:,}",
    )


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_distributions(chf: pd.DataFrame, dist: pd.DataFrame):
    import matplotlib.pyplot as plt

    cols = dist.variable.tolist()[:12]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, col in zip(axes.ravel(), cols):
        s = chf[col].dropna()
        ax.hist(s, bins=40, color=viz.SERIES_BLUE, alpha=0.9)
        sk = dist.loc[dist.variable == col, "skew"].iloc[0]
        unit = UNITS.get(col, "")
        ax.set_title(f"{col} ({unit})   skew {sk:.1f}", fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", visible=False)
        viz.despine(ax)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distributions of physiologic predictors, after plausibility bounds",
                 fontsize=12, fontweight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    viz.caption(fig, f"CHF training cohort, n={len(chf):,}. Physiologically impossible values were set to\n"
                     f"missing upstream by analysis_frames(), so no bar here is an artefact.")
    return viz.save(fig, "04_distributions.png")


def figure_linearity(chf: pd.DataFrame, lin: pd.DataFrame):
    import matplotlib.pyplot as plt

    top = lin.dropna(subset=["lr_chi2"]).head(4).variable.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(3.4 * len(top), 4.2))
    for ax, col in zip(np.atleast_1d(axes), top):
        d = chf[[col, OUTCOME_EVENT]].dropna()
        d = d.assign(bin=pd.qcut(d[col], 8, duplicates="drop"))
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
        ax.set_title(f"{col} ({UNITS.get(col,'')})\nq {fmt_p(row.q_nonlinearity)}",
                     fontsize=10)
        ax.set_xlabel(col)
        viz.despine(ax)
    np.atleast_1d(axes)[0].set_ylabel("Observed mortality (%)")
    fig.suptitle("What a linear term assumes, against what the data does",
                 fontsize=12, fontweight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    viz.caption(fig, f"CHF training cohort, n={len(chf):,}. Blue: observed mortality by octile with binomial\n"
                     f"standard errors. Orange dashed: the straight line a linear term is constrained to fit.\n"
                     f"q-values are FDR-corrected across a fixed family of {len(lin)} variables.")
    return viz.save(fig, "05_functional_form.png")


def figure_correlation(corr: pd.DataFrame, n: int):
    import matplotlib.pyplot as plt

    order = corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.loc[order, order], cmap=viz.diverging_cmap(), vmin=-1, vmax=1)
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
    ax.set_title("Rank correlation among candidate predictors")
    viz.caption(fig, f"CHF training cohort, n={n:,}. Spearman rather than Pearson: these variables are\n"
                     f"skewed and their relationships monotonic but not linear. Cells |rho| >= 0.4 labelled.")
    return viz.save(fig, "06_correlation.png")


ANSWERS = """
ANSWERS
{rule}

A7. TABLE 1, AND WHY SMD RATHER THAN P
    Table 1 is the first table in essentially every clinical paper, and its job
    is descriptive: who is in this cohort, and how do the outcome groups differ
    at baseline. Reviewers look for it. Its absence is conspicuous in a way it
    simply is not in a machine learning writeup. Here {n_smd_imbalanced} of
    {n_smd_vars} variables exceed the imbalance threshold.

    The p-value convention is a habit the methods literature has spent two
    decades trying to kill. Three reasons it is wrong here:

      * It answers a question nobody asked. A p-value tests whether the groups
        differ in some hypothetical source population. But these ARE the two
        groups -- everyone who died and everyone who did not. There is no
        sampling to make inference about.
      * It is a function of sample size, not of difference. At this n a
        clinically trivial gap in sodium reaches p<0.05; at n=80 a large one
        does not. The p-value is telling you about n.
      * It invites the reader to scan for stars and treat them as importance.

    The standardised mean difference is the difference in units of pooled
    standard deviation and does not move with n. |SMD| > 0.1 is the conventional
    threshold, the same one used to judge covariate balance after propensity
    score matching. Multi-level categoricals get one Yang-Dalton SMD for the
    variable rather than a binary SMD per level, which is a different quantity
    and does not share the 0.1 interpretation.

A8. THE IMPOSSIBLE VALUES
    Albumin of {alb_max} g/dL is not a sick patient. Serum albumin runs 3.5-5.0
    in health, falls in critical illness, and is incompatible with life much
    above 7. Against a median near 3.3 it is almost certainly a decimal or unit
    error. Zeros in meanbp, hrt and resp are the same class of problem: a mean
    arterial pressure of zero is not low blood pressure, it is the absence of
    circulation, and these patients were alive at enrolment with follow-up
    recorded afterwards.

    What matters is that this is a CELL-level problem, not a ROW-level one. The
    instinct is to drop the patient. That is wrong twice over: it discards every
    other valid measurement that patient contributed, and it deletes rows on the
    basis of a data-entry error, which is a form of selection.

    The correct move is to set the offending cells to missing and let the
    imputation machinery handle them alongside values that were never recorded.
    An impossible measurement and an absent one are the same kind of ignorance.

    Where that happens matters as much as what it does. These bounds live in
    support2.PLAUSIBLE_BOUNDS and are applied by analysis_frames() before any
    script receives data -- {n_voided} cells across the file, {n_caught} of them
    on the CHF training rows. An earlier version of this project kept the bounds
    in this script and applied them in one function, with the result that the
    data dictionary in 03_cohort.py published an albumin maximum of {alb_max}
    two sections after the README called that value incompatible with life. The
    rule "clean first, then test" is not implemented by a constant in one file.

    State the bounds before looking at the outcome. Bounds chosen after seeing
    which exclusions help your model are not quality control.

A9. SKEW, AND WHETHER TO TRANSFORM
    First, the answer most people get wrong: logistic regression makes NO
    normality assumption about its predictors. None. The normality assumption in
    linear models concerns residuals, and logistic regression does not have
    those in the same sense. Skewed predictors are not, in themselves, a
    violation of anything.

    So skew matters for two narrower reasons. Leverage: a handful of extreme
    values can dominate the fit of a linear term, so the coefficient describes
    the tail rather than the bulk. And interpretability: a one-unit change in
    creatinine means something very different at 0.8 than at 12.

    {worst_skew_var} is the most skewed at {worst_skew}, falling to
    {worst_skew_log} after a log transform. But a log transform costs the
    clinical reading -- an odds ratio per log-unit of bilirubin is not a
    sentence you can say to a cardiologist. Given that Q10 shows the
    relationships that matter are not linear on any scale, the better answer is
    usually to model the shape directly with a spline and present a plotted risk
    curve, rather than transform and still assume linearity in the new space.

    Transform when the mechanism is genuinely multiplicative. Otherwise let the
    spline do the work and show the curve.

A10. WHAT A LINEAR TERM ASSERTS
    A linear term in the logit asserts that every one-unit increase multiplies
    the odds of death by the same constant, everywhere along the range. For
    creatinine that is a strong claim and the octile pattern contradicts it:
    risk steps up around 1.2 mg/dL and then flattens rather than continuing to
    climb.

    A linear fit through that shape does two bad things at once. It understates
    the jump at moderate elevation, which is where most patients sit and where
    the clinical decision lives. And it extrapolates a rising slope into the far
    tail, so it overstates risk for patients with creatinine of 8 or 12 -- the
    sickest few, where being wrong is least excusable.

    The likelihood-ratio test formalises it. Creatinine rejects linearity at
    p={crea_p}, q={crea_q}, with an AIC gain of {crea_gain}. Surviving the
    correction: {nonlinear_vars}.

    Watch what multiplicity does to heart rate. Unadjusted it reads p={hrt_p};
    across the family of {n_family} it is q={hrt_q}. The Bonferroni bar would be
    {bonf}. A family that size at alpha 0.05 hands you roughly one false
    positive for free, and heart rate is the likeliest candidate. Reporting it
    as non-linear would be reporting the multiplicity, not the biology.

    Two structural points about that family. It is fixed in LINEARITY_FAMILY
    before any test runs, so a variable that fails to converge keeps its row
    with p = NaN instead of vanishing -- dropping it would shrink the
    denominator and silently inflate everyone else's q-value, which is a way to
    manufacture significance without noticing. And the fits use BFGS with a
    raised iteration cap, because the default optimiser failed on two variables
    in an earlier version and those failures were being swallowed by a
    module-level warnings filter.

    So the Harrell-school default -- assume non-linearity and spend a few
    degrees of freedom on a restricted cubic spline -- is the right posture, but
    it is a hypothesis to test, not a conclusion to assert. Fit the spline, test
    it, correct for the family, and keep it only where the data pays.

A11. COLLINEARITY, AND WHY IT MATTERS MORE HERE
    Two distinct findings, needing different responses.

    The first is not collinearity at all. `adls` and `adlsc` correlate at
    rho = {adlsc_rho} with a maximum absolute difference of {adlsc_maxdiff}
    across {adlsc_n} rows. They are the same number. `adlsc` is a derived ADL
    summary reconstructed from its components, and asking a model for a
    coefficient on both is asking for one that does not exist. It is excluded
    from CANDIDATE_PREDICTORS for that reason -- and recomputed here anyway, so
    the exclusion is evidence a reader can check rather than an assertion.

    The second is real: bun and crea correlate strongly. Clinically
    unsurprising, since both measure renal function, and exactly the case that
    hurts an interpretable model.

    For pure prediction, collinearity is close to harmless. If two predictors
    carry the same information a model leans on either and predictions barely
    move, which is why the machine learning habit is to ignore it.

    This project is not a pure prediction task. The deliverable is an odds ratio
    a cardiologist can act on, and collinearity attacks exactly that. It inflates
    standard errors, so intervals widen and a real effect can look null. It makes
    coefficients unstable, so small cohort changes flip signs. And it splits
    shared signal arbitrarily, so which correlate "wins" is near a coin toss --
    and you will then be asked to explain, clinically, why the winner mattered.

    VIF above about 5 warrants a look, above 10 is a problem. The maximum here
    is {vif_max}, but read that with suspicion: it rests on {vif_n} complete
    cases, and 03_cohort.py Q17 recomputes it on imputed data and explains why
    "complete case" in this cohort is nearly a synonym for one enrolment wave.

    The remedy for genuine collinearity is rarely mechanical deletion. Prefer
    collapsing correlated measures into one clinically meaningful construct, or
    pre-specifying which member to keep on clinical grounds -- decided before
    fitting, and stated. For bun and crea, creatinine is the more standard renal
    marker in heart failure reporting.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    cohort = analysis_frames()
    chf = cohort.chf_train

    header("SUPPORT2 -- cohort profile, data quality, functional form")
    print(f"  CHF TRAINING cohort: {len(chf):,} patients, "
          f"{int(chf[OUTCOME_EVENT].sum()):,} deaths")
    print("  The 30% held-out partition is never returned by analysis_frames().")

    t1 = compute_table_one(chf)
    report_table_one(t1, chf)

    # Q8 reports what the upstream cleaning caught, read from the raw source
    # restricted to the same training rows.
    caught = find_implausible(cohort.raw.loc[chf.index])
    report_plausibility(caught, cohort.n_voided, len(chf) * len(PLAUSIBLE_BOUNDS))

    dist = compute_distributions(chf)
    report_distributions(dist)

    lin = compute_linearity(chf)
    report_linearity(lin)

    coll = compute_collinearity(chf)
    report_collinearity(coll, chf)

    header("FIGURES")
    for path in (figure_distributions(chf, dist),
                 figure_linearity(chf, lin),
                 figure_correlation(coll["corr"], len(chf))):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    facts = collect_facts(t1, caught, dist, lin, coll, cohort.n_voided)
    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    run_and_capture(main, OUT_DIR / "02_profile.txt")
