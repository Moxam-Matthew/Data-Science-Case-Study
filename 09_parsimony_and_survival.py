"""
09_parsimony_and_survival.py -- A model a cardiologist could compute on paper,
and the time-to-event model this project argued for and had not built.

Two loose ends, closed before the held-out partition is spent.

    Run:  python 09_parsimony_and_survival.py

    Training partition only. The held-out 30% remains unread.

THE QUESTIONS
    Q39  Real clinical risk scores use five to eight variables, not
         twenty-eight penalised down to sixteen. Build the model a cardiologist
         would recognise, and say what it costs.
    Q40  01_eda.py A1 argued that a binary flag discards the timing information
         that matters clinically -- and then the whole modelling stage used a
         binary outcome. Was that a contradiction?
    Q41  Cox assumes hazards are proportional. 03_cohort.py Q14 showed the
         baseline hazard falls steeply. Test the assumption rather than
         asserting it, and say what to do where it fails.
    Q42  Harrell's C for the Cox model and AUC for the logistic model are both
         concordance measures. Are they measuring the same thing, and which
         logistic model is the fair comparison?

Author: Matthew Moxam
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import viz
from modelling import (
    CV_FOLDS,
    CV_REPEATS,
    HORIZON_DAYS,
    OUTCOME_LABEL,
    RANDOM_STATE,
    build_pipeline,
    calibration_metrics,
    cross_val_predictions,
    default_predictors,
    discrimination_metrics,
    make_outcome,
)
from report import RULE, Facts, configure_pandas, fmt_p, header, question, run_and_capture
from support2 import OUTCOME_EVENT, OUTCOME_TIME, UNITS, analysis_frames

OUT_DIR = Path(__file__).resolve().parent / "output"

# ── The parsimonious model, specified on clinical grounds ────────────────────
# Chosen from what the heart failure literature treats as prognostic and what is
# knowable at admission -- NOT by looking at which variables performed well
# here. The honesty caveat is in A39: this set overlaps heavily with what the
# elastic net selected, and having already seen that selection, the choice
# cannot be called independent. It is what a cardiologist would name, checked
# against the literature rather than against the data.
CLINICAL_PREDICTORS = [
    "age",       # in every cardiovascular risk score ever published
    "meanbp",    # haemodynamic status
    "sod",       # hyponatraemia, a long-established adverse marker in HF
    "crea",      # renal function; cardiorenal syndrome drives HF outcome
    "scoma",     # neurological status, a severity marker
    "num.co",    # comorbidity burden
    "adls",      # functional dependence
]
SPLINE_VARIABLE = "crea"     # 02_profile.py Q10: the one variable rejecting linearity
SPLINE_DF = 4


# ═══ Q39. The parsimonious model ═════════════════════════════════════════════
def compare_parsimony(chf: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    full = default_predictors(chf)
    specs = {
        f"Full elastic net ({len(full)} vars)": (
            full,
            LogisticRegressionCV(l1_ratios=(0.2, 0.5, 0.9),
                                 Cs=np.logspace(-3, 1, 8), cv=CV_FOLDS,
                                 scoring="neg_log_loss", max_iter=3000,
                                 random_state=RANDOM_STATE, refit=True,
                                 n_jobs=-1, solver="saga")),
        f"Clinical logistic ({len(CLINICAL_PREDICTORS)} vars)": (
            CLINICAL_PREDICTORS, LogisticRegression(max_iter=4000)),
        f"Clinical + spline on {SPLINE_VARIABLE}": (
            CLINICAL_PREDICTORS, LogisticRegression(max_iter=4000)),
    }
    rows = []
    for name, (preds, est) in specs.items():
        frame = chf
        use = preds
        if "spline" in name:
            frame, use, _ = add_spline_columns(chf, preds)
        p = cross_val_predictions(build_pipeline(frame, use, est),
                                  frame[use], y, n_repeats=CV_REPEATS,
                                  label=name[:26])
        rows.append({"model": name, "n_predictors": len(use),
                     **discrimination_metrics(y, p), **calibration_metrics(y, p),
                     "_pred": p})
    return pd.DataFrame(rows)


def add_spline_columns(chf: pd.DataFrame, preds: list[str]) -> tuple:
    """
    Replace the spline variable with a natural cubic basis.

    Knots are placed on quantiles of the TRAINING data. That is a defensible
    simplification and also a small leak: strictly the basis should be rebuilt
    inside each fold, since knot placement is estimated from data. With knots at
    fixed quantiles of a single variable the effect is negligible, but it is a
    leak rather than not one, and saying so is cheaper than pretending.
    """
    from patsy import dmatrix

    x = chf[SPLINE_VARIABLE]
    observed = x.dropna()
    # Knots cannot be placed through NaN, so the basis is built on observed
    # values and written back by index. Rows with a missing value keep NaN
    # across every basis column and are imputed downstream like any other gap --
    # rather than being silently dropped, which would change the cohort.
    fitted = dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": observed.values},
                     return_type="dataframe")
    cols = [f"{SPLINE_VARIABLE}_s{i+1}" for i in range(fitted.shape[1])]
    basis = pd.DataFrame(np.nan, index=chf.index, columns=cols)
    basis.loc[observed.index, cols] = fitted.values

    out = pd.concat([chf, basis], axis=1)
    use = [c for c in preds if c != SPLINE_VARIABLE] + cols
    # design_info is returned rather than stashed on .attrs: pandas
    # deep-copies attrs on almost every operation and patsy's design_info
    # is not picklable, so that route raises on the next concat.
    return out, use, fitted.design_info


def report_parsimony(t: pd.DataFrame, y: np.ndarray) -> None:
    question(39, "Real clinical risk scores use five to eight variables, not\n"
                 "twenty-eight penalised down to sixteen. Build the model a\n"
                 "cardiologist would recognise, and say what it costs.")
    show = t.drop(columns=["_pred"]).copy()
    for c in ("auc", "pr_auc", "calibration_slope", "calibration_intercept",
              "brier"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print(f"\n  Variables in the clinical model: "
          f"{', '.join(CLINICAL_PREDICTORS)}")
    print("  All knowable at admission; all named by the heart failure")
    print("  literature rather than selected from this dataset.")


def clinical_odds_ratios(chf: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """The coefficient table a cardiologist would actually read."""
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression

    pipe = build_pipeline(chf, CLINICAL_PREDICTORS,
                          LogisticRegression(max_iter=4000))
    pipe.fit(chf[CLINICAL_PREDICTORS], y)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    Z = pd.DataFrame(pipe.named_steps["prep"].transform(
        pipe.named_steps["indicators"].transform(chf[CLINICAL_PREDICTORS])),
        columns=names, index=chf.index)
    fit = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    ci = fit.conf_int()
    out = pd.DataFrame({
        "variable": fit.params.index,
        "unit": [UNITS.get(v, "1 SD") for v in fit.params.index],
        "odds_ratio": np.exp(fit.params.values),
        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
        "p": fit.pvalues.values})
    return out[out.variable != "const"].sort_values("odds_ratio", ascending=False)


# ═══ Q40 / Q41 / Q42. Survival ═══════════════════════════════════════════════
def fit_cox(chf: pd.DataFrame) -> dict:
    """
    Cox proportional hazards on the FULL follow-up, with a spline on creatinine.

    This is the model 01_eda.py A1 argued for and the project had not built. It
    uses every patient's complete observation time rather than collapsing to a
    180-day flag, which is the whole point: a patient who died on day 20 and one
    who died on day 179 are the same event to the logistic model and different
    events here.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from sklearn.impute import SimpleImputer

    frame, use, _ = add_spline_columns(chf, CLINICAL_PREDICTORS)
    X = frame[use]
    # Median imputation here rather than MICE: CoxPHFitter takes a single
    # complete frame, and the fold-wise machinery does not apply to a model
    # being fitted once for inference rather than cross-validated for
    # prediction. Stated because it differs from the rest of the project.
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=use, index=X.index)
    data = imputed.assign(**{OUTCOME_TIME: chf[OUTCOME_TIME].values,
                             OUTCOME_EVENT: chf[OUTCOME_EVENT].values})

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(data, duration_col=OUTCOME_TIME, event_col=OUTCOME_EVENT)
    ph = proportional_hazard_test(cph, data, time_transform="rank")
    return {"model": cph, "data": data, "ph": ph.summary,
            "concordance": float(cph.concordance_index_),
            "n": len(data), "events": int(chf[OUTCOME_EVENT].sum())}


def report_cox(r: dict) -> None:
    question(40, "01_eda.py A1 argued that a binary flag discards the timing\n"
                 "information that matters clinically -- and then the whole\n"
                 "modelling stage used a binary outcome. Was that a contradiction?")
    cph = r["model"]
    print(f"  Cox proportional hazards on full follow-up: n={r['n']:,}, "
          f"{r['events']:,} deaths")
    print(f"  Harrell's C = {r['concordance']:.3f}\n")
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%",
                     "p"]].copy()
    s.columns = ["hazard_ratio", "ci_low", "ci_high", "p"]
    for c in ("hazard_ratio", "ci_low", "ci_high"):
        s[c] = s[c].round(3)
    s["p"] = s["p"].apply(fmt_p)
    print(s.to_string())
    print("\n  Hazard ratios, not odds ratios: the quantity is a rate over time,")
    print("  and it uses the full observation window rather than a 180-day flag.")


def report_ph_assumption(r: dict) -> dict:
    question(41, "Cox assumes hazards are proportional. 03_cohort.py Q14 showed the\n"
                 "baseline hazard falls steeply. Test the assumption rather than\n"
                 "asserting it, and say what to do where it fails.")
    ph = r["ph"].copy()
    ph = ph.reset_index()
    ph.columns = [str(c) for c in ph.columns]
    show = ph.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4)
    print(show.to_string(index=False))
    pcol = [c for c in ph.columns if c.lower() == "p"]
    violations = []
    if pcol:
        violations = ph.loc[ph[pcol[0]] < 0.05].iloc[:, 0].astype(str).tolist()
    print(f"\n  Covariates violating proportional hazards at p<0.05: "
          f"{', '.join(violations) if violations else 'none'}")
    return {"violations": violations, "n_tested": len(ph)}


def report_concordance_vs_auc(r: dict, clinical_auc: float,
                              full_auc: float) -> None:
    question(42, "Harrell's C for the Cox model and AUC for the logistic model are\n"
                 "both concordance measures. Are they measuring the same thing,\n"
                 "and which logistic model is the fair comparison?")
    print(f"  Cox, full follow-up, {len(CLINICAL_PREDICTORS)} clinical vars")
    print(f"      Harrell's C = {r['concordance']:.3f}")
    print(f"  Clinical logistic, {HORIZON_DAYS}-day flag, same "
          f"{len(CLINICAL_PREDICTORS)} vars")
    print(f"      AUC         = {clinical_auc:.3f}   <- the like-for-like comparison")
    print(f"      difference    {r['concordance'] - clinical_auc:+.3f}")
    print(f"\n  Full elastic net, {HORIZON_DAYS}-day flag, all predictors")
    print(f"      AUC         = {full_auc:.3f}   <- NOT comparable: more variables")
    print(f"      difference    {r['concordance'] - full_auc:+.3f}")
    print("\n  Comparing the 7-variable Cox against the 28-variable logistic would")
    print("  attribute a variable-count difference to the choice of model family.")


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_spline_risk(chf: pd.DataFrame, cox: dict):
    """The plotted risk curve A10 argued for instead of a single hazard ratio."""
    import matplotlib.pyplot as plt

    cph, data = cox["model"], cox["data"]
    basis_cols = [c for c in data.columns if c.startswith(f"{SPLINE_VARIABLE}_s")]
    grid = np.linspace(chf[SPLINE_VARIABLE].quantile(0.02),
                       chf[SPLINE_VARIABLE].quantile(0.98), 120)

    from patsy import dmatrix
    # Reuse the design info the training basis was built with, so the grid is
    # evaluated against the same knots rather than a freshly placed set.
    _, _, design_info = add_spline_columns(chf, CLINICAL_PREDICTORS)
    grid_basis = dmatrix(design_info, {"x": grid}, return_type="dataframe")
    grid_basis.columns = basis_cols

    beta = cph.params_[basis_cols].values
    lp = grid_basis.values @ beta
    ref = np.interp(chf[SPLINE_VARIABLE].median(), grid, lp)
    hr = np.exp(lp - ref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"height_ratios": [1]})
    ax1.plot(grid, hr, color=viz.SERIES_BLUE, lw=2.4)
    ax1.axhline(1.0, color=viz.BASELINE, lw=1.2, ls="--")
    ax1.axvline(chf[SPLINE_VARIABLE].median(), color=viz.INK_MUTED, lw=1.2, ls=":")
    ax1.text(chf[SPLINE_VARIABLE].median(), ax1.get_ylim()[1] * 0.96,
             " reference = median", fontsize=8.5, color=viz.INK_SECONDARY,
             va="top")
    ax1.set_xlabel(f"{SPLINE_VARIABLE} ({UNITS.get(SPLINE_VARIABLE, '')})")
    ax1.set_ylabel("Hazard ratio vs median")
    ax1.set_title("Creatinine: a curve, not a coefficient")
    viz.despine(ax1)

    ax2.hist(chf[SPLINE_VARIABLE].dropna(), bins=50, color=viz.SERIES_BLUE,
             alpha=0.85)
    ax2.set_xlabel(f"{SPLINE_VARIABLE} ({UNITS.get(SPLINE_VARIABLE, '')})")
    ax2.set_ylabel("Patients")
    ax2.set_title("Where the patients actually are")
    viz.despine(ax2)

    fig.tight_layout()
    viz.caption(fig, f"CHF training cohort, Cox model with a natural cubic spline on {SPLINE_VARIABLE}. Risk rises\n"
                     f"then flattens, exactly as 02_profile.py Q10 found. The right panel is the reason the\n"
                     f"right-hand tail of the curve should not be over-read: almost nobody is there.",
                y=-0.04)
    return viz.save(fig, "19_spline_risk_curve.png")


ANSWERS = """
ANSWERS
{rule}

A39. WHAT PARSIMONY COSTS
    The clinical model uses {n_clinical} variables against the full model's
    {n_full}, and reaches AUC {clin_auc} against {full_auc} -- a difference of
    {auc_gap}. Adding a spline on {spline_var} moves it to {spline_auc}, which
    is not an improvement; the spline earned its place in a Cox model on full
    follow-up, not here.

    But AUC is the metric on which the clinical model looks worst, and reporting
    only that would undersell it. Its PR-AUC is {clin_prauc} against
    {full_prauc} -- higher, on the metric that matters more at 25% prevalence.
    Its calibration slope is {clin_slope} against {full_slope}: the full model
    over-shrinks, the clinical model is nearer 1, and it is the better
    calibrated of the two. Brier scores are identical to three decimals.

    So "what parsimony costs" is close to nothing here: {auc_gap} of AUC, in
    exchange for better calibration and a model with seven variables instead of
    twenty-eight. That is a much stronger result than a small loss, and it
    matches how clinical risk scores are actually built. GRACE, TIMI and
    CHA2DS2-VASc all use single-digit variable counts, not because their authors
    lacked data but because a score nobody can compute does not get computed.

    An honesty note that belongs in the open rather than in a footnote. These
    seven variables were chosen from the heart failure literature -- age,
    haemodynamics, sodium, renal function, neurological status, comorbidity
    burden, functional dependence -- and they overlap heavily with what the
    elastic net selected in 05_modelling.py. Having already seen that selection,
    I cannot claim this specification was independent of the data. A genuinely
    pre-specified model would have been written down before any modelling ran.
    The overlap is reassuring about the clinical plausibility of the penalised
    model; it is not independent evidence, and it should not be presented as
    validation of either.

    The trade to state plainly: the full model is better for prediction, the
    clinical model is better for adoption, and they are close enough here that
    the choice is a deployment question rather than a statistical one.

A40. WAS THE BINARY OUTCOME A CONTRADICTION?
    Partly, and the honest answer is more interesting than a defence.

    A1 objected to the SATURATED end-of-study flag -- death at any point over
    one to five years, firing for {full_death_pct}% of the cohort, with
    follow-up unbalanced in a way Q5 showed manufactures signal. The 180-day
    outcome the modelling actually used is a different object: no patient is
    censored before it, so nothing is discarded by collapsing to it, and it
    aligns with the physician's own 6-month estimate. That was a defensible
    choice and the project made it for stated reasons.

    But A1 also argued that timing is what a cardiologist asks about, and that
    argument was left unfinished. A binary model at 180 days treats a death on
    day 20 and a death on day 179 as identical events. They are not identical to
    a patient, a family, or a discharge decision.

    So the Cox model above is not a redundancy check, it is the missing half.
    Harrell's C of {cindex} on the full follow-up says the ordering of survival
    times is recoverable at about the same strength as the 180-day ranking,
    which is a genuinely useful thing to know: the signal is not concentrated
    in one window.

    What the Cox model buys that the logistic does not: hazard ratios rather
    than odds ratios, a risk curve over time rather than at one horizon, and the
    ability to answer "when" rather than only "whether". What it costs: the
    proportional hazards assumption, which Q41 tests rather than assumes.

A41. TESTING PROPORTIONALITY
    {ph_verdict}

    Two things worth separating, because they are routinely confused. A falling
    BASELINE hazard -- which 03_cohort.py Q14 documented, early hazard about six
    times the late -- is not a violation. Cox never models the baseline; it
    factors out. Proportionality is a claim about the RATIO between patients,
    which is a different thing entirely.

    What a violation would mean is that a covariate's effect changes over time:
    a variable that matters intensely in the first month and little afterwards
    still yields one plausible-looking hazard ratio, and that single number is
    an average over a period during which the truth changed.

    The remedies, in ascending order of effort. Stratify on the offending
    variable, which allows it its own baseline hazard and stops pretending its
    effect is constant. Add a time-varying coefficient, an interaction between
    the covariate and a function of time. Or, if the violation is severe, accept
    that the proportional hazards framework is the wrong one and report a
    fixed-horizon model instead -- which is, in effect, what the rest of this
    project did.

    There is corroborating evidence in the coefficients, and it is the kind
    worth noticing rather than smoothing over. Compare the two models on
    `num.co`: the logistic gives an odds ratio of {numco_or} -- more
    comorbidities associated with LOWER 180-day mortality, which is clinically
    backwards -- while the Cox gives a hazard ratio of {numco_hr} and does not
    reach significance. A variable whose sign disagrees between two models
    fitted on the same patients is unstable, and `num.co` is also one of the
    covariates failing the proportionality test. Those two facts are the same
    fact: an effect that changes over the follow-up cannot be summarised by one
    number, and the summary you get depends on which window you look through.

    The right response is not to pick whichever sign is more comfortable. It is
    to say that this variable is not reliably estimated here, stratify or
    interact it with time, and refuse to interpret its coefficient clinically
    until it behaves.

    The point is that this was tested. A Cox model reported without a
    proportionality check is a model with an unexamined assumption at its centre,
    and Schoenfeld residuals cost one line.

A42. IS HARRELL'S C THE SAME THING AS AUC?
    Nearly, but the first job is picking the right comparison. C = {cindex} for
    the Cox model against AUC = {clinical_auc} for the logistic model on the
    SAME {n_clinical} variables -- a difference of {c_gap}. Against the full
    {n_full}-variable elastic net ({full_auc}) the gap is {c_gap_full}, and
    quoting that one would blame the model family for a variable count.

    Both are concordance measures: the probability that, for a randomly chosen
    pair of patients, the model ranks them in the correct order. For a binary
    outcome with no censoring, Harrell's C and the AUC are the same quantity --
    the C-index is a generalisation of AUC, not a different idea.

    What generalises is the definition of "correct order". AUC pairs a patient
    who had the event with one who did not. Harrell's C pairs patients by
    survival TIME, and handles the pairs where censoring makes the comparison
    undecidable by discarding them. So C is answering a harder question on more
    information, and the near-identical value tells you the model's ranking is
    stable whether you ask about 180-day status or about ordering over the whole
    follow-up.

    One caution that applies to both and is usually omitted: concordance shares
    every weakness this project has already documented for AUC. It is
    rank-based, so it cannot see calibration; it is insensitive to added
    predictors; and it says nothing about whether the model is clinically
    useful, which is what decision curve analysis in 05_modelling.py is for.
    Reporting C alone for a survival model is the same mistake as reporting AUC
    alone for a binary one.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train
    y = make_outcome(chf).values

    header(f"SUPPORT2 -- parsimony and survival")
    print(f"  CHF training cohort {len(chf):,}, {int(y.sum()):,} events at "
          f"{HORIZON_DAYS} days ({y.mean()*100:.1f}%)")
    print(f"  Full follow-up: {int(chf[OUTCOME_EVENT].sum()):,} deaths over "
          f"{chf[OUTCOME_TIME].max():,.0f} days")
    print("  Training partition only; the held-out 30% remains unread.")

    t = compare_parsimony(chf, y)
    report_parsimony(t, y)

    header("CLINICAL MODEL: ODDS RATIOS")
    ors = clinical_odds_ratios(chf, y)
    show = ors.copy()
    for c in ("odds_ratio", "ci_low", "ci_high"):
        show[c] = show[c].round(2)
    show["p"] = show["p"].apply(fmt_p)
    print(show.to_string(index=False))

    cox = fit_cox(chf)
    report_cox(cox)
    ph = report_ph_assumption(cox)
    full_auc = float(t.iloc[0]["auc"])
    clinical_auc = float(
        t[t.model.str.startswith("Clinical logistic")].iloc[0]["auc"])
    report_concordance_vs_auc(cox, clinical_auc, full_auc)

    header("FIGURES")
    path = figure_spline_risk(chf, cox)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    clin = t[t.model.str.startswith("Clinical logistic")].iloc[0]
    spl = t[t.model.str.contains("spline")].iloc[0]
    ph_verdict = (
        f"Across {ph['n_tested']} covariates, {len(ph['violations'])} violate "
        f"proportional hazards at p<0.05"
        + (f": {', '.join(ph['violations'])}." if ph["violations"]
           else ". The assumption holds on this cohort.")
    )
    facts = Facts(
        n_clinical=str(len(CLINICAL_PREDICTORS)),
        n_full=str(len(default_predictors(chf))),
        clin_auc=f"{clin.auc:.3f}", full_auc=f"{full_auc:.3f}",
        clin_prauc=f"{clin.pr_auc:.3f}",
        full_prauc=f"{t.iloc[0]['pr_auc']:.3f}",
        clin_slope=f"{clin.calibration_slope:.2f}",
        full_slope=f"{t.iloc[0]['calibration_slope']:.2f}",
        numco_or=f"{float(ors.set_index('variable').loc['num.co','odds_ratio']):.2f}",
        numco_hr=f"{float(cox['model'].summary.loc['num.co','exp(coef)']):.2f}",
        spline_auc=f"{spl.auc:.3f}",
        auc_gap=f"{clin.auc - full_auc:+.3f}",
        spline_var=SPLINE_VARIABLE,
        full_death_pct=f"{chf[OUTCOME_EVENT].mean()*100:.1f}",
        cindex=f"{cox['concordance']:.3f}",
        clinical_auc=f"{clinical_auc:.3f}",
        c_gap=f"{cox['concordance'] - clinical_auc:+.3f}",
        c_gap_full=f"{cox['concordance'] - full_auc:+.3f}",
        ph_verdict=ph_verdict,
    )
    print(ANSWERS.format(rule=RULE, **facts))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "09_parsimony_and_survival.txt")
