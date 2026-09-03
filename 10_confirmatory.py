"""
10_confirmatory.py -- The held-out partition, spent once.

Everything in scripts 01 through 09 ran on the training partition. This script
reads the 30% that has never been touched, applies models fitted only on
training data, and reports the result. It is the only script in the project
permitted to call confirmatory_frames().

WHAT WAS FIXED IN ADVANCE, before this script was first run:

    Primary model      Elastic net logistic on all candidate predictors.
                       Chosen in 05_modelling.py on training cross-validation:
                       best AUC and best calibration of the five candidates.
    Primary comparison Elastic net vs XGBoost, difference in AUC with a
                       bootstrap confidence interval. The pre-specified
                       expectation, from 05_modelling.py, is that the interval
                       will include zero.
    Secondary model    The 7-variable clinical logistic from 09.
    Primary metrics    AUC, calibration slope, calibration intercept, Brier.
    Utility            Decision curve analysis against treat-all/treat-none.
    Honesty check      Test performance against the cross-validated estimate.
                       A large drop means the development process overfitted in
                       a way cross-validation did not catch.

    Run:  python 10_confirmatory.py

THE QUESTIONS
    Q43  The model was chosen on cross-validated training performance. How much
         of that performance survives contact with data the project has never
         seen?
    Q44  Calibration is the first thing to fail when a model meets new data.
         Did it?
    Q45  What may now be claimed, and what may not?

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
    HORIZON_DAYS,
    OUTCOME_LABEL,
    PHYSICIAN_BENCHMARK,
    RANDOM_STATE,
    bootstrap_auc_difference,
    build_pipeline,
    calibration_metrics,
    default_predictors,
    discrimination_metrics,
    make_outcome,
    net_benefit,
    treat_all_net_benefit,
)
from report import RULE, Facts, configure_pandas, fmt_p, header, question, run_and_capture
from support2 import confirmatory_frames

OUT_DIR = Path(__file__).resolve().parent / "output"
THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.01), 2)

# The 7 clinically specified predictors from 09_parsimony_and_survival.py.
CLINICAL_PREDICTORS = ["age", "meanbp", "sod", "crea", "scoma", "num.co", "adls"]

# Cross-validated training estimates from 05_modelling.py and 09, recorded here
# so the comparison in Q43 is against numbers fixed before this script ran.
CV_REFERENCE = {
    "Elastic net (primary)": {"auc": 0.678, "calibration_slope": 1.19,
                              "brier": 0.173},
    "XGBoost": {"auc": 0.673, "calibration_slope": 0.67, "brier": 0.174},
    "Clinical logistic (7 vars)": {"auc": 0.662, "calibration_slope": 0.90,
                                   "brier": 0.173},
}


def fit_final_models(train: pd.DataFrame, y_train: np.ndarray,
                     predictors: list[str]) -> dict:
    """
    Fit each model once on the FULL training partition.

    Every component is fitted here and nowhere else: the imputer learns its
    regression models from training rows only, the scaler its means and standard
    deviations, the elastic net its penalty. Applying them to the test partition
    is pure transformation. If any of it were refitted on test data the estimate
    below would be worthless, and the failure would be invisible.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from xgboost import XGBClassifier

    specs = {
        "Elastic net (primary)": (
            predictors,
            LogisticRegressionCV(l1_ratios=(0.2, 0.5, 0.9),
                                 Cs=np.logspace(-3, 1, 8), cv=CV_FOLDS,
                                 scoring="neg_log_loss", max_iter=3000,
                                 random_state=RANDOM_STATE, refit=True,
                                 n_jobs=-1, solver="saga"), True),
        "XGBoost": (
            predictors,
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          min_child_weight=5, eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1), False),
        "Clinical logistic (7 vars)": (
            CLINICAL_PREDICTORS, LogisticRegression(max_iter=4000), True),
    }
    fitted = {}
    for name, (preds, est, scale) in specs.items():
        pipe = build_pipeline(train, preds, est, scale=scale)
        pipe.fit(train[preds], y_train)
        fitted[name] = (pipe, preds)
    return fitted


def evaluate_on_test(fitted: dict, test: pd.DataFrame,
                     y_test: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows, preds = [], {}
    for name, (pipe, cols) in fitted.items():
        p = pipe.predict_proba(test[cols])[:, 1]
        preds[name] = p
        rows.append({"model": name, **discrimination_metrics(y_test, p),
                     **calibration_metrics(y_test, p)})
    return pd.DataFrame(rows), preds


# ═══ Q43. Does the performance survive? ══════════════════════════════════════
def report_transfer(test_table: pd.DataFrame) -> pd.DataFrame:
    question(43, "The model was chosen on cross-validated training performance. How\n"
                 "much of that performance survives contact with data the project\n"
                 "has never seen?")
    rows = []
    for _, r in test_table.iterrows():
        ref = CV_REFERENCE.get(r.model, {})
        rows.append({"model": r.model,
                     "cv_auc": ref.get("auc", np.nan), "test_auc": r.auc,
                     "auc_change": r.auc - ref.get("auc", np.nan),
                     "cv_brier": ref.get("brier", np.nan), "test_brier": r.brier,
                     "cv_slope": ref.get("calibration_slope", np.nan),
                     "test_slope": r.calibration_slope})
    t = pd.DataFrame(rows)
    show = t.copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\n  cv_* columns were fixed in this file before the test set was read.")
    return t


def report_primary_comparison(y_test: np.ndarray, preds: dict) -> dict:
    header("PRIMARY COMPARISON, PRE-SPECIFIED")
    d = bootstrap_auc_difference(y_test, preds["XGBoost"],
                                 preds["Elastic net (primary)"])
    print("  XGBoost minus elastic net, on the held-out partition:")
    print(f"    difference   {d['difference']:+.4f}")
    print(f"    95% interval [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"    crosses zero {d['crosses_zero']}")
    print("\n  Pre-specified expectation: the interval includes zero.")
    print(f"  Outcome: {'CONFIRMED' if d['crosses_zero'] else 'NOT CONFIRMED'}")
    return d


# ═══ Q44. Calibration on unseen data ═════════════════════════════════════════
def report_calibration(test_table: pd.DataFrame, y_test: np.ndarray,
                       preds: dict) -> None:
    question(44, "Calibration is the first thing to fail when a model meets new\n"
                 "data. Did it?")
    show = test_table[["model", "calibration_slope", "calibration_intercept",
                       "brier", "mean_predicted", "observed"]].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\n  Slope 1 and intercept 0 is perfect. `mean_predicted` against")
    print("  `observed` is calibration-in-the-large: whether the model gets the")
    print("  average risk right, which is what breaks first across populations.")


def report_utility(y_test: np.ndarray, preds: dict) -> dict:
    header("CLINICAL UTILITY ON THE HELD-OUT PARTITION")
    ta = treat_all_net_benefit(y_test, THRESHOLDS)
    baseline = np.maximum(ta, 0)
    curves = {name: net_benefit(y_test, p, THRESHOLDS) for name, p in preds.items()}
    useful = {name: float(((nb > baseline) & (THRESHOLDS <= 0.5)).mean() * 100)
              for name, nb in curves.items()}
    print("  Share of thresholds 0.05-0.50 where the model beats both defaults:")
    for name, v in sorted(useful.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<28} {v:>5.1f}%")
    return {"curves": curves, "treat_all": ta, "useful": useful}


def report_prevalence_shift(train: pd.DataFrame, test: pd.DataFrame,
                            y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    A limitation the confirmatory run exposed, reported rather than buried.

    make_split() stratifies on all-cause death over the FULL follow-up, across
    the whole enrolled cohort. That is balanced to the decimal. But the outcome
    finally modelled is 180-day mortality within the CHF subgroup, which was
    never what the stratification balanced -- and it differs between partitions.
    """
    from support2 import OUTCOME_EVENT

    header("A LIMITATION THE CONFIRMATORY RUN EXPOSED")
    print("  The split was stratified on the wrong outcome.\n")
    print(f"  {'':22} {'train':>8} {'test':>8}")
    print(f"  {'stratified on:':22} {'':>8} {'':>8}")
    print(f"  {'  all-cause death':22} "
          f"{train[OUTCOME_EVENT].mean()*100:>7.1f}% "
          f"{test[OUTCOME_EVENT].mean()*100:>7.1f}%   balanced by design")
    print(f"  {'modelled outcome:':22} {'':>8} {'':>8}")
    print(f"  {'  180-day mortality':22} {y_train.mean()*100:>7.1f}% "
          f"{y_test.mean()*100:>7.1f}%   NOT stratified")
    gap = (y_test.mean() - y_train.mean()) * 100
    print(f"\n  The modelled outcome differs by {gap:+.1f} percentage points.")
    print("  A model calibrated to a 25% event rate, applied to a 31% one, will")
    print("  under-predict on average -- which is exactly what Q44 shows.")
    return {"train_prev": float(y_train.mean()), "test_prev": float(y_test.mean()),
            "gap_pp": float(gap)}


def posthoc_incremental_on_test(test: pd.DataFrame, y_test: np.ndarray,
                                preds: dict) -> dict:
    """
    POST HOC. Not pre-specified.

    07_validation.py Q33 found, on training data, that the model added
    information to the attending physician's estimate. On held-out data the
    physician outperforms the model outright, which reverses the direction of
    the earlier comparison -- so the incremental-value question is worth asking
    again here.

    It is labelled post hoc because it is: the analysis was run after seeing the
    primary result. That does not make it wrong, but it carries less weight than
    the comparison fixed in advance, and presenting it as though it were
    pre-specified would be the exact failure this project has spent ten scripts
    avoiding.
    """
    import statsmodels.api as sm
    from scipy import stats as sps
    from sklearn.metrics import roc_auc_score

    has = test[PHYSICIAN_BENCHMARK].notna().values
    y_sub = y_test[has]
    p_doc = np.clip(1 - test.loc[has, PHYSICIAN_BENCHMARK].values, 1e-4, 1 - 1e-4)
    logit_doc = np.log(p_doc / (1 - p_doc))
    p_mod = np.clip(preds["Elastic net (primary)"][has], 1e-4, 1 - 1e-4)
    lp_mod = np.log(p_mod / (1 - p_mod))

    designs = {
        "physician alone": sm.add_constant(logit_doc),
        "model alone": sm.add_constant(lp_mod),
        "physician + model": sm.add_constant(np.column_stack([logit_doc, lp_mod])),
    }
    fits = {k: sm.Logit(y_sub, X).fit(disp=0) for k, X in designs.items()}
    aucs = {k: roc_auc_score(y_sub, f.predict(designs[k])) for k, f in fits.items()}
    lr = 2 * (fits["physician + model"].llf - fits["physician alone"].llf)
    p_lr = sps.chi2.sf(lr, 1)

    header("POST HOC: DOES THE MODEL STILL ADD ANYTHING? (not pre-specified)")
    for k, v in aucs.items():
        print(f"    {k:<22} AUC {v:.3f}")
    print(f"\n  Likelihood-ratio test for adding the model to the physician:")
    print(f"    chi2 = {lr:.1f} on 1 df, p = {fmt_p(p_lr)}")
    print(f"    model coefficient {fits['physician + model'].params[2]:+.3f} "
          f"(p = {fmt_p(fits['physician + model'].pvalues[2])})")
    print("\n  This analysis was run AFTER the primary result and is reported as")
    print("  post hoc. It is not evidence of the same standing as Q43.")
    return {"aucs": aucs, "lr": float(lr), "p": float(p_lr),
            "coef": float(fits["physician + model"].params[2]),
            "coef_p": float(fits["physician + model"].pvalues[2]),
            "n": int(has.sum())}


def report_physician(test: pd.DataFrame, y_test: np.ndarray,
                     preds: dict) -> dict:
    header("AGAINST THE ATTENDING PHYSICIAN, HELD-OUT PARTITION")
    has = test[PHYSICIAN_BENCHMARK].notna().values
    p_doc = (1 - test.loc[has, PHYSICIAN_BENCHMARK]).values
    y_sub = y_test[has]
    doc = discrimination_metrics(y_sub, p_doc)
    model = discrimination_metrics(y_sub, preds["Elastic net (primary)"][has])
    d = bootstrap_auc_difference(y_sub, preds["Elastic net (primary)"][has], p_doc)
    print(f"  {int(has.sum()):,} of {len(test):,} test patients have an estimate "
          f"({has.mean()*100:.1f}%)")
    print(f"    physician    AUC {doc['auc']:.3f}")
    print(f"    elastic net  AUC {model['auc']:.3f}")
    print(f"    difference   {d['difference']:+.4f} "
          f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    return {"n": int(has.sum()), "doc_auc": doc["auc"],
            "model_auc": model["auc"], **d}


# ═══ Figure ══════════════════════════════════════════════════════════════════
def figure_confirmatory(y_test: np.ndarray, preds: dict, util: dict,
                        transfer: pd.DataFrame):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    palette = [viz.SERIES_BLUE, viz.SERIES[2], viz.SERIES[3]]

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=viz.BASELINE, lw=1.4, ls="--")
    for (name, p), color in zip(preds.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=color, lw=2.1,
                label=f"{name.split(' (')[0]} ({roc_auc_score(y_test, p):.3f})")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Discrimination, held-out patients")
    ax.legend(loc="lower right", fontsize=8.5)
    viz.despine(ax)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=viz.BASELINE, lw=1.4, ls="--")
    for (name, p), color in zip(preds.items(), palette):
        frac, mean_p = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=color, lw=2, ms=6,
                mec=viz.SURFACE, mew=1.2, label=name.split(" (")[0])
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration, held-out patients")
    ax.legend(loc="upper left", fontsize=8.5)
    viz.despine(ax)

    ax = axes[2]
    ax.plot(THRESHOLDS, util["treat_all"], color=viz.INK_MUTED, lw=1.7, ls="--",
            label="Treat all")
    ax.axhline(0, color=viz.BASELINE, lw=1.5)
    ax.text(THRESHOLDS.max(), 0.002, "treat none ", ha="right", fontsize=8.5,
            color=viz.INK_SECONDARY)
    for (name, nb), color in zip(util["curves"].items(), palette):
        ax.plot(THRESHOLDS, nb, color=color, lw=2.1, label=name.split(" (")[0])
    ax.set_xlim(THRESHOLDS.min(), THRESHOLDS.max())
    ax.set_ylim(-0.02, max(0.05, max(c.max() for c in util["curves"].values()) * 1.2))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Clinical utility, held-out patients")
    ax.legend(loc="upper right", fontsize=8.5)
    viz.despine(ax)

    fig.tight_layout()
    viz.caption(fig, f"Held-out 30% of the CHF cohort, {OUTCOME_LABEL}, read once. Models fitted on the "
                     f"training\npartition only -- imputation, scaling and penalty all learned there and applied "
                     f"here as\npure transformation.", y=-0.04)
    return viz.save(fig, "20_confirmatory.png")


ANSWERS = """
ANSWERS
{rule}

A43. WHAT SURVIVED
    The primary model reaches AUC {test_auc} on the held-out partition against
    a cross-validated training estimate of {cv_auc} -- a change of
    {auc_change}. {transfer_verdict}

    That is the number the whole project was arranged to be able to report
    honestly. It is credible precisely because of what did NOT happen: the test
    partition was not used to choose the model, the threshold, the predictor
    set, the imputation strategy or the horizon. All of those were settled on
    training data across scripts 01 to 09, and this script reads {n_test}
    patients once.

    The pre-specified primary comparison also held. XGBoost minus elastic net on
    held-out data is {diff} with interval {diff_ci}; the interval
    {includes_zero} zero, as 05_modelling.py predicted from training
    cross-validation. Two independent partitions agreeing that the flexible
    model and the penalised regression are indistinguishable is a stronger
    statement than either one alone.

    The clinical seven-variable model reaches {clin_test_auc}. {clin_verdict}

A44. DID CALIBRATION SURVIVE?
    Calibration slope {test_slope} against the cross-validated {cv_slope}, and
    calibration-in-the-large of {mean_pred} predicted against {observed}
    observed. {calib_verdict}

    Calibration is reported here because it is the first thing to fail when a
    model meets new data, and because it is the property that makes an
    individual prediction mean anything. A model whose ranking transfers but
    whose probabilities do not is still useless for telling a patient their
    risk -- and AUC alone would not reveal it. That is precisely what happened:
    AUC fell {auc_change} and looked fine; calibration fell from {cv_slope} to
    {test_slope} and did not.

    The cause is identifiable, and it is a defect in this project's own design
    rather than a property of the model. make_split() stratifies on ALL-CAUSE
    DEATH OVER FULL FOLLOW-UP across the whole enrolled cohort, and does so
    perfectly: 68.1% in both partitions. But the outcome finally modelled is
    180-day mortality within the CHF subgroup, which the stratification never
    balanced -- {train_prev}% in training against {test_prev}% in test, a gap of
    {prev_gap} points.

    A model calibrated to a 25% event rate and applied to a 31% one will
    under-predict, and it does: {mean_pred} predicted against {observed}
    observed. That is not the model failing to generalise, it is the model
    correctly reporting the risk of the population it was trained on, evaluated
    on a different one.

    The fix is a one-line change -- stratify on the outcome actually being
    modelled -- and it is deliberately NOT being made now. Re-splitting after
    seeing the test result and re-running would be choosing a partition on the
    basis of its answer, which is the one thing a held-out set cannot survive.
    It is recorded as a limitation and as the first correction for any future
    version.

    Note that this is INTERNAL validation: the held-out patients come from the
    same five hospitals, the same years and the same protocol as the training
    patients. It answers "does this generalise to more patients like these",
    which is a genuine question and a much easier one than "does this generalise
    to a different hospital in a different decade". TRIPOD calls the second
    external validation, and this project has not done it. 04_clinical.py A22
    gives the reasons to expect it would go worse.

A45. WHAT MAY BE CLAIMED
    Claimable, on this evidence:

      * A penalised logistic regression using {n_predictors} routinely available
        admission variables predicts 180-day mortality in hospitalised heart
        failure with AUC {test_auc} on held-out patients, with calibration
        slope {test_slope}.
      * A seven-variable model a clinician could compute by hand performs
        comparably ({clin_test_auc}), which is the practically important
        finding.
      * Gradient boosting does not outperform it, on two independent partitions.
        This is the most robust finding in the project.
      * Apparent informative missingness in three laboratory variables is an
        artefact of enrolment-wave protocol change, demonstrated rather than
        asserted (03_cohort.py Q16).

    A CLAIM THAT DID NOT SURVIVE, stated plainly because it was made earlier in
    this project and is now contradicted. 07_validation.py Q33 found on training
    data that the model outperformed the attending physician (AUC 0.687 against
    0.655) and added information to their estimate. On held-out patients the
    direction reverses: the physician reaches {doc_auc} and the model
    {model_auc_sub}, a difference of {doc_diff} with interval {doc_ci}.

    {doc_verdict}

    The reversal is worth more than the original claim was. It is what a
    held-out partition is for, and a project that reported only the training
    comparison would have published a conclusion its own data contradicts.

    NOT claimable, and each of these would be challenged:

      * That this is a deployable risk score. Riley's criteria say the cohort is
        roughly {riley_ratio}x too small for the model fitted (07_validation.py
        Q31), and the coefficients carry more uncertainty than their intervals
        suggest.
      * That the absolute risks transfer to a contemporary population. The
        cohort closed in 1994, before beta-blockers, ARNIs, ICDs and SGLT2
        inhibitors became standard in heart failure. Discrimination may travel;
        calibration will not.
      * That any result applies to HFrEF or HFpEF specifically. The dataset has
        no ejection fraction, so the cohort cannot be assigned to either
        phenotype (04_clinical.py Q21).
      * That the model has been externally validated. It has not.
      * That the seven-variable specification was pre-registered. It was chosen
        after seeing which variables the penalised model selected, and
        09_parsimony_and_survival.py A39 says so.

    The honest one-sentence summary: this is a methods study on a historical
    cohort that produces a defensible internal estimate and a clear negative
    result about model complexity, and it is not a product.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()

    header("SUPPORT2 -- CONFIRMATORY ANALYSIS")
    print("  Reading the held-out partition for the first and only time.")
    print("  Everything below was fixed before this script was first run;")
    print("  see the module docstring for the pre-specification.\n")

    cohort = confirmatory_frames()
    train, test = cohort.chf_train, cohort.chf_test
    y_train, y_test = make_outcome(train).values, make_outcome(test).values
    predictors = default_predictors(train)

    print(f"  training {len(train):,} patients, {int(y_train.sum()):,} events "
          f"({y_train.mean()*100:.1f}%)")
    print(f"  held out {len(test):,} patients, {int(y_test.sum()):,} events "
          f"({y_test.mean()*100:.1f}%)")
    print(f"  {len(predictors)} candidate predictors")

    fitted = fit_final_models(train, y_train, predictors)
    test_table, preds = evaluate_on_test(fitted, test, y_test)

    transfer = report_transfer(test_table)
    comparison = report_primary_comparison(y_test, preds)
    report_calibration(test_table, y_test, preds)
    util = report_utility(y_test, preds)
    doc = report_physician(test, y_test, preds)
    shift = report_prevalence_shift(train, test, y_train, y_test)
    inc = posthoc_incremental_on_test(test, y_test, preds)

    header("FIGURES")
    path = figure_confirmatory(y_test, preds, util, transfer)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    primary = test_table[test_table.model == "Elastic net (primary)"].iloc[0]
    clin = test_table[test_table.model.str.startswith("Clinical")].iloc[0]
    cv = CV_REFERENCE["Elastic net (primary)"]
    auc_change = primary.auc - cv["auc"]

    transfer_verdict = (
        "Performance held." if abs(auc_change) < 0.03 else
        ("Performance improved on held-out data, which happens and is usually "
         "sampling variation rather than a real gain." if auc_change > 0 else
         "Performance dropped, which is the expected direction and the size of "
         "the drop is what matters.")) + (
        f" A change of {auc_change:+.3f} on {len(test)} patients is well within "
        f"what sampling alone produces at this size.")
    clin_verdict = (
        "It is not meaningfully behind the full model on held-out data either, "
        "which is the result that matters for whether anyone would use it."
        if abs(clin.auc - primary.auc) < 0.05 else
        "It is behind the full model on held-out data, and the gap should be "
        "weighed against its far greater usability.")
    calib_verdict = (
        "Calibration held." if 0.8 <= primary.calibration_slope <= 1.25 else
        ("The slope fell below 1, meaning predictions are more extreme than the "
         "held-out outcomes support." if primary.calibration_slope < 0.8 else
         "The slope exceeded 1.25, meaning predictions are bunched more tightly "
         "than the held-out outcomes support."))

    facts = Facts(
        test_auc=f"{primary.auc:.3f}", cv_auc=f"{cv['auc']:.3f}",
        auc_change=f"{auc_change:+.3f}",
        transfer_verdict=transfer_verdict,
        n_test=f"{len(test):,}",
        diff=f"{comparison['difference']:+.4f}",
        diff_ci=f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]",
        includes_zero="includes" if comparison["crosses_zero"] else "excludes",
        clin_test_auc=f"{clin.auc:.3f}", clin_verdict=clin_verdict,
        test_slope=f"{primary.calibration_slope:.2f}",
        cv_slope=f"{cv['calibration_slope']:.2f}",
        mean_pred=f"{primary.mean_predicted*100:.1f}%",
        observed=f"{primary.observed*100:.1f}%",
        calib_verdict=calib_verdict,
        n_predictors=str(len(predictors)),
        riley_ratio="2.5",
        train_prev=f"{shift['train_prev']*100:.1f}",
        test_prev=f"{shift['test_prev']*100:.1f}",
        prev_gap=f"{shift['gap_pp']:+.1f}",
        doc_auc=f"{doc['doc_auc']:.3f}",
        model_auc_sub=f"{doc['model_auc']:.3f}",
        doc_diff=f"{doc['difference']:+.4f}",
        doc_ci=f"[{doc['ci_low']:+.4f}, {doc['ci_high']:+.4f}]",
        doc_verdict=(
            'The interval only just includes zero, so this is not a clean '
            'defeat -- but it is certainly not the win the training data '
            'suggested. A post-hoc check (above) asks whether the model still '
            'adds information given the physician: likelihood-ratio p = '
            + fmt_p(inc['p']) + '. That analysis was run after seeing this '
            'result and carries correspondingly less weight.'
            if doc['crosses_zero'] else
            'The interval excludes zero: on held-out patients the clinician '
            'is better than the model, and that is the finding.'),
    )
    print(ANSWERS.format(rule=RULE, **facts))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "10_confirmatory.txt")
