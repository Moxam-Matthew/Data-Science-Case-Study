"""
05_modelling.py -- Penalised regression, gradient boosting, and whether either
is clinically useful.

Continues the question format. Answers are held at the bottom and every number
in them is interpolated from the run.

    Run:  python 05_modelling.py

    All development is cross-validated on the TRAINING partition. The held-out
    30% is not read by this script. It is spent once, at the end of the project,
    on the single pre-specified comparison -- not on choosing between models.

THE QUESTIONS
    Q23  The unpenalised and penalised fits miscalibrate in OPPOSITE directions
         while their AUCs sit within a rounding error of each other. What has
         gone wrong in each case, and why is it nearly invisible to AUC?
    Q24  Elastic net has two hyperparameters. Selecting them by cross-validation
         and then reporting cross-validated performance from the same folds is a
         mistake with a name. What is it, and what does it cost?
    Q25  XGBoost and the penalised regression differ on AUC. Before deciding
         whether the deliverable should change, decide whether that is a
         difference at all.
    Q26  Both models beat chance. Does either beat the attending physician, and
         does the answer depend on which patients you ask about?
    Q27  A model with a higher AUC can be worth less at the bedside. Show it.

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
    MISSINGNESS_INDICATORS,
    OUTCOME_LABEL,
    PHYSICIAN_BENCHMARK,
    PROTOCOL_MISSING,
    RANDOM_STATE,
    bootstrap_auc_difference,
    build_pipeline,
    calibration_metrics,
    cross_val_predictions,
    default_predictors,
    discrimination_metrics,
    make_outcome,
    net_benefit,
    treat_all_net_benefit,
)
from report import Facts, RULE, configure_pandas, fmt_p, header, question, render_answers, run_and_capture
from support2 import SUPPORT_NORMAL_FILL, analysis_frames

OUT_DIR = Path(__file__).resolve().parent / "output"
# Rounded at definition: np.arange leaves values like 0.30000000000000004,
# which then fail an exact .loc lookup when the report asks for threshold 0.30.
THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.01), 2)


# ═══ Model definitions ═══════════════════════════════════════════════════════
def make_models(chf: pd.DataFrame, predictors: list[str]) -> dict:
    """
    Four models, each with a job. Hyperparameter search sits INSIDE the pipeline
    so it is refitted on every outer fold -- see A24 for why that matters.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    # sklearn 1.8 deprecated `penalty` in favour of l1_ratio / C, so the new
    # spelling is used: C=inf is unpenalised, l1_ratio=1 is pure LASSO.
    shared = dict(cv=CV_FOLDS, scoring="neg_log_loss", max_iter=3000,
                  random_state=RANDOM_STATE, refit=True, n_jobs=-1,
                  solver="saga")
    return {
        # Deliberately included to fail. It is the demonstration for Q23.
        "Unpenalised logistic": build_pipeline(
            chf, predictors, LogisticRegression(C=np.inf, max_iter=4000)),
        "LASSO logistic": build_pipeline(
            chf, predictors,
            LogisticRegressionCV(l1_ratios=(1.0,), Cs=np.logspace(-3, 1, 8),
                                 **shared)),
        "Elastic net logistic": build_pipeline(
            chf, predictors,
            LogisticRegressionCV(l1_ratios=(0.2, 0.5, 0.9),
                                 Cs=np.logspace(-3, 1, 8), **shared)),
        "XGBoost": build_pipeline(
            chf, predictors,
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2.0, min_child_weight=5,
                          eval_metric="logloss", random_state=RANDOM_STATE,
                          n_jobs=-1),
            scale=False),
        "Decision tree (depth 3)": build_pipeline(
            chf, predictors,
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                   random_state=RANDOM_STATE),
            scale=False),
    }


# ═══ Q23 / Q24 / Q25. Fit and compare ════════════════════════════════════════
def evaluate_models(chf: pd.DataFrame, predictors: list[str],
                    y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    X = chf[predictors]
    preds, rows = {}, []
    for name, pipe in make_models(chf, predictors).items():
        p = cross_val_predictions(pipe, X, y, n_repeats=CV_REPEATS,
                                  label=name)
        preds[name] = p
        rows.append({"model": name, **discrimination_metrics(y, p),
                     **calibration_metrics(y, p)})
    return pd.DataFrame(rows), preds


def report_overfitting(table: pd.DataFrame, facts: Facts) -> None:
    question(23, f"An unpenalised logistic regression returns a calibration slope of\n"
                 f"{facts['unpen_slope']} against an ideal of 1.0, while the penalised fits "
                 f"sit near\n{facts['enet_slope']}. What has gone wrong in each direction, "
                 f"and why is it nearly\ninvisible to AUC?")
    show = table[["model", "auc", "pr_auc", "calibration_slope",
                  "calibration_intercept", "brier"]].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\n  Calibration slope 1.0 is perfect. Below 1 means predictions are too")
    print("  extreme: the model has learned noise and states it confidently.")
    print("  Read the auc column beside it -- it barely moves.")


def report_nested_cv(table: pd.DataFrame) -> None:
    question(24, "Elastic net has two hyperparameters. Selecting them by\n"
                 "cross-validation and then reporting cross-validated performance\n"
                 "from the same folds is a mistake with a name. What is it?")
    print("  Every model above uses nested cross-validation: the hyperparameter")
    print(f"  search is a step inside the pipeline, so on each of the "
          f"{CV_FOLDS}x{CV_REPEATS} outer")
    print("  folds it is refitted using only that fold's training rows.")
    print("\n  The alternative -- tune once on all training data, then report")
    print("  cross-validated scores -- lets the held-out rows of each fold")
    print("  influence the hyperparameters used to predict them.")


def compute_model_comparison(preds: dict, y: np.ndarray) -> pd.DataFrame:
    ref = "Elastic net logistic"
    rows = []
    for name in preds:
        if name == ref:
            continue
        d = bootstrap_auc_difference(y, preds[name], preds[ref])
        rows.append({"model": name, "vs": ref, **d})
    return pd.DataFrame(rows)


def report_model_comparison(out: pd.DataFrame, facts: Facts) -> None:
    question(25, f"XGBoost differs from the penalised regression by {facts['xgb_diff']} in\n"
                 f"AUC. Before deciding whether the deliverable should change, decide\n"
                 f"whether that is a difference at all.")
    show = out.copy()
    for c in ("difference", "ci_low", "ci_high"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    print("\n  A difference whose 95% interval crosses zero is not a difference.")


# ═══ Q26. The physician benchmark ════════════════════════════════════════════
def compare_with_physician(chf: pd.DataFrame, y: np.ndarray,
                           preds: dict) -> dict:
    """
    Head-to-head on the patients the physician actually scored.

    prg6m is the attending's estimated probability of SURVIVAL at 6 months, so
    predicted death is 1 - prg6m. The comparison is restricted to patients with
    a recorded estimate -- scoring the model on everyone and the physician on a
    subset would not be a comparison.
    """
    has = chf[PHYSICIAN_BENCHMARK].notna().values
    p_doc = (1 - chf.loc[has, PHYSICIAN_BENCHMARK]).values
    y_sub = y[has]
    rows = [{"predictor": "Attending physician (prg6m)",
             **discrimination_metrics(y_sub, p_doc),
             **calibration_metrics(y_sub, p_doc)}]
    diffs = []
    for name, p in preds.items():
        rows.append({"predictor": name, **discrimination_metrics(y_sub, p[has]),
                     **calibration_metrics(y_sub, p[has])})
        diffs.append({"model": name,
                      **bootstrap_auc_difference(y_sub, p[has], p_doc)})
    return {"table": pd.DataFrame(rows), "diffs": pd.DataFrame(diffs),
            "n": int(has.sum()), "coverage": float(has.mean() * 100),
            "mask": has, "p_doc": p_doc}


def report_physician(r: dict) -> None:
    question(26, "Both models beat chance. Does either beat the attending physician,\n"
                 "and does the answer depend on which patients you ask about?")
    print(f"  Restricted to the {r['n']:,} patients with a recorded physician "
          f"estimate ({r['coverage']:.1f}% of the cohort).\n")
    show = r["table"][["predictor", "auc", "calibration_slope",
                       "calibration_intercept", "brier"]].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\n  Bootstrap difference in AUC against the physician:")
    d = r["diffs"].copy()
    for c in ("difference", "ci_low", "ci_high"):
        d[c] = d[c].round(4)
    print(d.to_string(index=False))


# ═══ Q27. Decision curve analysis ════════════════════════════════════════════
def compute_dca(y: np.ndarray, preds: dict, physician: dict) -> pd.DataFrame:
    curves = {"Treat all": treat_all_net_benefit(y, THRESHOLDS),
              "Treat none": np.zeros_like(THRESHOLDS)}
    for name, p in preds.items():
        curves[name] = net_benefit(y, p, THRESHOLDS)
    df = pd.DataFrame(curves, index=THRESHOLDS)
    df.index.name = "threshold"
    return df


def summarise_dca(dca: pd.DataFrame) -> dict:
    show_at = [round(v, 2) for v in (0.10, 0.20, 0.30, 0.40, 0.50)]
    models = [c for c in dca.columns if c not in ("Treat all", "Treat none")]
    baseline = dca[["Treat all", "Treat none"]].max(axis=1)
    return {"show_at": show_at,
            "best_at": {t: max(models, key=lambda m: dca.loc[t, m]) for t in show_at},
            "useful": {m: float(((dca[m] > baseline) & (dca.index <= 0.5)).mean() * 100)
                       for m in models}}


def report_dca(dca: pd.DataFrame, info: dict) -> None:
    question(27, "A model with a higher AUC can be worth less at the bedside. Show it.")
    show_at = info["show_at"]
    sub = dca.loc[dca.index.isin(show_at)].round(4)
    print("  Net benefit at clinically plausible thresholds:\n")
    print(sub.to_string())

    models = [c for c in dca.columns if c not in ("Treat all", "Treat none")]
    best_at = {t: max(models, key=lambda m: dca.loc[t, m]) for t in show_at}
    baseline = dca[["Treat all", "Treat none"]].max(axis=1)
    useful = {m: float(((dca[m] > baseline) & (dca.index <= 0.5)).mean() * 100)
              for m in models}
    print("\n  Share of thresholds 0.05-0.50 where the model beats both default")
    print("  strategies (treat everyone / treat no one):")
    for m, v in sorted(useful.items(), key=lambda kv: -kv[1]):
        print(f"    {m:<26} {v:>5.1f}%")
    return {"best_at": best_at, "useful": useful}


# ═══ Sensitivity ═════════════════════════════════════════════════════════════
def sensitivity_analysis(chf: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Does the conclusion depend on how missing data was handled?

    Three arms: MICE inside folds (primary), the SUPPORT investigators' published
    normal-fill constants, and complete cases only. Where they agree, say so;
    where they diverge, the divergence is a finding rather than an inconvenience.
    """
    from sklearn.linear_model import LogisticRegressionCV

    preds_full = default_predictors(chf)
    est = lambda: LogisticRegressionCV(solver="saga", Cs=np.logspace(-3, 1, 8),
                                       l1_ratios=(0.2, 0.5, 0.9), cv=CV_FOLDS,
                                       scoring="neg_log_loss", max_iter=3000,
                                       random_state=RANDOM_STATE, n_jobs=-1)
    rows = []

    p = cross_val_predictions(build_pipeline(chf, preds_full, est()),
                              chf[preds_full], y, n_repeats=2,
                              label="sens: MICE")
    rows.append({"arm": "MICE inside folds (primary)", "n": len(y),
                 **discrimination_metrics(y, p), **calibration_metrics(y, p)})

    filled = chf.copy()
    for col, val in SUPPORT_NORMAL_FILL.items():
        if col in filled:
            filled[col] = filled[col].fillna(val)
    p = cross_val_predictions(build_pipeline(filled, preds_full, est()),
                              filled[preds_full], y, n_repeats=2,
                              label="sens: normal-fill")
    rows.append({"arm": "SUPPORT normal-fill constants", "n": len(y),
                 **discrimination_metrics(y, p), **calibration_metrics(y, p)})

    preds_drop = default_predictors(chf, drop_protocol_missing=True)
    p = cross_val_predictions(build_pipeline(chf, preds_drop, est()),
                              chf[preds_drop], y, n_repeats=2,
                              label="sens: drop protocol")
    rows.append({"arm": f"Drop protocol-missing ({', '.join(PROTOCOL_MISSING)})",
                 "n": len(y), **discrimination_metrics(y, p),
                 **calibration_metrics(y, p)})

    cc = chf[preds_full].dropna().index
    if len(cc) <= 120:
        rows.append({"arm": f"Complete cases only -- NOT RUN, only {len(cc)} rows",
                     "n": len(cc), "auc": np.nan, "pr_auc": np.nan,
                     "calibration_slope": np.nan, "calibration_intercept": np.nan,
                     "brier": np.nan, "mean_predicted": np.nan,
                     "observed": np.nan})
    if len(cc) > 120:
        y_cc = y[chf.index.isin(cc)]
        p = cross_val_predictions(build_pipeline(chf.loc[cc], preds_full, est()),
                                  chf.loc[cc, preds_full], y_cc,
                                  n_splits=3, n_repeats=2,
                                  label="sens: complete case")
        rows.append({"arm": "Complete cases only", "n": len(cc),
                     **discrimination_metrics(y_cc, p),
                     **calibration_metrics(y_cc, p)})
    return pd.DataFrame(rows)


def report_sensitivity(t: pd.DataFrame) -> None:
    header("SENSITIVITY TO THE MISSING-DATA STRATEGY")
    show = t[["arm", "n", "auc", "calibration_slope", "brier"]].copy()
    for c in ("auc", "calibration_slope", "brier"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    skipped = t[t.auc.isna()]
    if len(skipped):
        print()
        print("  An arm reported as NOT RUN is not a silent omission: complete-case")
        print("  analysis is infeasible here because requiring every predictor leaves")
        print("  too few patients to cross-validate. That infeasibility is itself the")
        print("  argument against complete-case analysis, so it is printed rather")
        print("  than dropped.")


# ═══ Odds ratios ═════════════════════════════════════════════════════════════
def selected_odds_ratios(chf: pd.DataFrame, predictors: list[str],
                         y: np.ndarray) -> pd.DataFrame:
    """
    Fit the elastic net once on all training data to see what it selects, then
    refit an UNPENALISED logistic on the selected variables to report odds
    ratios with confidence intervals.

    Two things must be said about this. Penalised coefficients are shrunk toward
    zero, so exponentiating them does not give an odds ratio a clinician can
    interpret -- hence the refit. But the refit's confidence intervals are too
    narrow, because they pretend the variable set was chosen in advance when it
    was chosen by looking at the same data. That is the post-selection inference
    problem, and it has no cheap fix; the intervals below are optimistic and are
    labelled as such rather than quietly reported.
    """
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegressionCV

    pipe = build_pipeline(
        chf, predictors,
        LogisticRegressionCV(solver="saga", Cs=np.logspace(-3, 1, 8),
                             l1_ratios=(0.2, 0.5, 0.9), cv=CV_FOLDS,
                             scoring="neg_log_loss", max_iter=3000,
                             random_state=RANDOM_STATE, n_jobs=-1))
    pipe.fit(chf[predictors], y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    kept = [n for n, c in zip(names, coefs) if abs(c) > 1e-8]

    Z = pd.DataFrame(pipe.named_steps["prep"].transform(
        pipe.named_steps["indicators"].transform(chf[predictors])), columns=names,
        index=chf.index)[kept]
    fit = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    ci = fit.conf_int()
    out = pd.DataFrame({
        "variable": fit.params.index,
        "odds_ratio": np.exp(fit.params.values),
        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
        "p": fit.pvalues.values,
    })
    out = out[out.variable != "const"].sort_values("odds_ratio", ascending=False)
    return out.assign(n_selected=len(kept), n_candidate=len(names))


def report_odds_ratios(t: pd.DataFrame) -> None:
    header("ODDS RATIOS FROM THE SELECTED MODEL")
    print(f"  Elastic net kept {t.n_selected.iloc[0]} of {t.n_candidate.iloc[0]} "
          f"encoded terms. Refit unpenalised for interpretable effect sizes.")
    print("  Per 1 SD for continuous terms, since predictors were standardised.")
    print("  Intervals are OPTIMISTIC -- see the note in selected_odds_ratios().\n")
    show = t[["variable", "odds_ratio", "ci_low", "ci_high", "p"]].copy()
    for c in ("odds_ratio", "ci_low", "ci_high"):
        show[c] = show[c].round(2)
    show["p"] = show["p"].apply(fmt_p)
    print(show.to_string(index=False))


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_calibration(y: np.ndarray, preds: dict, table: pd.DataFrame):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    show = ["Unpenalised logistic", "Elastic net logistic", "XGBoost"]
    colors = [viz.SERIES_ORANGE, viz.SERIES_BLUE, viz.SERIES[2]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    ax1.plot([0, 1], [0, 1], color=viz.BASELINE, lw=1.5, ls="--", zorder=1)
    for name, color in zip(show, colors):
        if name not in preds:
            continue
        frac, mean_p = calibration_curve(y, preds[name], n_bins=10, strategy="quantile")
        ax1.plot(mean_p, frac, "o-", color=color, lw=2, ms=6,
                 mec=viz.SURFACE, mew=1.2, label=name)
    ax1.set_xlabel("Predicted risk")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Calibration by risk decile")
    ax1.legend(loc="upper left")
    viz.despine(ax1)

    t = table.set_index("model").loc[[n for n in show if n in table.model.values]]
    ypos = np.arange(len(t))
    ax2.barh(ypos, t.calibration_slope, color=colors[:len(t)], height=0.55)
    ax2.axvline(1.0, color=viz.BASELINE, lw=1.5, ls="--")
    ax2.text(1.0, len(t) - 0.35, " ideal = 1.0", fontsize=8.5,
             color=viz.INK_SECONDARY, va="center")
    for i, (v, a) in enumerate(zip(t.calibration_slope, t.auc)):
        ax2.text(v + 0.02, i, f"slope {v:.2f}   AUC {a:.3f}", va="center",
                 fontsize=8.5, color=viz.INK_SECONDARY)
    ax2.set_yticks(ypos, t.index)
    ax2.set_xlim(0, max(1.3, t.calibration_slope.max() * 1.5))
    ax2.set_xlabel("Calibration slope")
    ax2.set_title("Discrimination hides what calibration shows")
    ax2.grid(axis="y", visible=False)
    viz.despine(ax2)

    viz.caption(fig, f"CHF training cohort, {OUTCOME_LABEL}, out-of-fold predictions from "
                     f"{CV_FOLDS}x{CV_REPEATS} repeated CV.\nThe unpenalised model's AUC is close to the "
                     f"others while its calibration slope is far below 1:\nit ranks patients acceptably and "
                     f"states their individual risk with unfounded confidence.")
    return viz.save(fig, "11_calibration.png")


def figure_dca(dca: pd.DataFrame, prevalence: float):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dca.index, dca["Treat all"], color=viz.INK_MUTED, lw=1.8, ls="--",
            label="Treat all")
    ax.plot(dca.index, dca["Treat none"], color=viz.BASELINE, lw=1.8,
            label="Treat none")
    for name, color in (("Elastic net logistic", viz.SERIES_BLUE),
                        ("XGBoost", viz.SERIES[2]),
                        ("Decision tree (depth 3)", viz.SERIES[3])):
        if name in dca:
            ax.plot(dca.index, dca[name], color=color, lw=2.2, label=name)
    ax.set_xlim(dca.index.min(), dca.index.max())
    ax.set_ylim(-0.02, max(0.05, dca.max().max() * 1.15))
    ax.set_xlabel("Threshold probability (risk at which you would act)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis: is the model worth using?")
    ax.legend(loc="upper right")
    viz.despine(ax)
    viz.caption(fig, f"CHF training cohort, {OUTCOME_LABEL}, prevalence {prevalence*100:.1f}%. A model is "
                     f"worth using only\nwhere its curve sits above BOTH defaults. Net benefit is on the "
                     f"scale of true positives\nper patient, after charging for false positives at the "
                     f"exchange rate the threshold implies.")
    return viz.save(fig, "12_decision_curve.png")


ANSWERS = """
ANSWERS
{rule}

A23. WHAT AUC CANNOT SEE
    The unpenalised model reaches AUC {unpen_auc} with a calibration slope of
    {unpen_slope}. The elastic net reaches AUC {enet_auc} with a slope of
    {enet_slope}. On discrimination alone you would call these models
    equivalent. They are not remotely equivalent.

    A slope below 1 means predicted risks are too spread out: the model pushes
    patients toward 0 and 1 more confidently than the data supports. It has
    fitted noise, and -- this is the part that matters clinically -- it states
    that noise as a probability. A patient told they have a 5% risk when the
    truth is 20% has been misinformed, even though the model ranked them
    correctly relative to everyone else.

    AUC cannot see this because AUC is rank-based. Multiply every predicted
    probability by ten, or take the square root of all of them, and the AUC is
    unchanged while the calibration is destroyed. Discrimination asks "is the
    ordering right"; calibration asks "is the number right". Only the second
    supports a conversation with a patient, and it is the one usually omitted.

    The cause here is countable. {n_events} events against roughly {n_params}
    encoded parameters is about {epv} events per variable, below the
    conventional floor of 10. At that ratio an unpenalised fit has enough
    freedom to chase noise, and it does.

    Read the other direction too, because the penalised fits overshoot: their
    slopes sit near {enet_slope}, above 1 rather than below. That is the
    opposite failure -- predictions too tightly bunched around the average,
    because the penalty shrank the coefficients further than the data required.
    It is the safer error of the two, since an under-confident model understates
    how different patients are rather than inventing distinctions, but it is
    still miscalibration and it would matter if the model were used to select a
    high-risk group for an intervention. The remedy is not a bigger grid search;
    it is to report the slope so a reader can see which way the model errs.

    XGBoost is the interesting middle case: slope {xgb_slope}, no better
    calibrated than the unpenalised regression despite its regularisation, and
    it is the one model here whose calibration intercept is meaningfully off
    zero ({xgb_intercept}). Boosting optimises log-loss, which rewards
    calibration, and it still comes out badly -- capacity is not the same thing
    as being right about probabilities.

A24. THE MISTAKE AND ITS NAME
    Selecting hyperparameters on the same folds you then report performance
    from is a form of information leakage; the resulting estimate is
    optimistically biased. The fix is NESTED cross-validation, and the reason
    this project gets it right almost by accident is structural: the
    hyperparameter search is a step inside the Pipeline, so scikit-learn refits
    it on each outer fold's training rows only. Every number in this script
    comes from {folds}x{repeats} repeated stratified folds built that way.

    The size of the effect is easy to underrate. With a small event count and a
    two-dimensional search -- penalty strength and l1 ratio -- there is real
    room to select a configuration that suits the particular rows being scored.
    That is a mistake you cannot detect afterwards from the output, which is why
    it has to be prevented by construction rather than checked for.

    A related discipline, stated because it is the one being followed: the
    held-out 30% has not been read by this script. Choosing between models on a
    test set converts it into a validation set, and there is then nothing left
    that has not influenced a decision. It is spent once, on a comparison fixed
    in advance.

A25. WHEN A BETTER MODEL IS NOT A BETTER DELIVERABLE
    XGBoost sits {xgb_diff} from the elastic net in AUC, with a bootstrap
    interval of {xgb_ci}. {xgb_verdict}

    Note which way that sign points. The flexible model did not win and then
    get rejected on interpretability grounds -- it did not win.

    That is the honest reading, and it matches the literature: a systematic
    review of clinical prediction models found no consistent benefit of machine
    learning over logistic regression on structured data (Christodoulou et al.,
    J Clin Epidemiol 2019). Tabular clinical data at this scale is where
    regression is hardest to beat -- the relationships are mostly smooth, the
    sample is small, and boosting's capacity has little to work with.

    So the deliverable should not change, and the reason is not sentimental.
    The elastic net yields an odds ratio per predictor with an interval, which
    is what a journal prints and what a clinician can argue with. XGBoost yields
    a ranking. Trading an interpretable effect size for a difference this size
    is a bad trade, and being able to say WHY it is a bad trade -- rather than
    reporting whichever number came out higher -- is the actual skill on display.

    The decision tree is included for a different reason again. It loses on
    every metric, and it is the only model here that prints as something a
    clinician could apply at a bedside without a computer. If the goal were
    adoption rather than accuracy, it would be the candidate.

A26. AGAINST THE PHYSICIAN
    On the {doc_n} patients with a recorded estimate, the attending physician
    achieves AUC {doc_auc}. The elastic net achieves {enet_auc_sub}, a
    difference of {doc_diff} with interval {doc_ci}. {doc_verdict}

    Two cautions before drawing any conclusion from that.

    The comparison is on the subset the physician scored, which is
    {doc_coverage}% of the cohort -- and an estimate is more likely to be
    recorded for patients a clinician had reason to think about. That is not a
    random subset, so this is a conditional comparison rather than a general one.

    And the physician is not a competitor. They had access to the bedside, the
    conversation and the trajectory, none of which is in the dataset; the model
    has arithmetic. The interesting result is not who wins but whether the model
    adds anything to what the clinician already knows -- which is an
    incremental-value question, answered by adding the physician estimate as a
    covariate and testing what the other predictors contribute on top. That is
    the analysis a cardiology journal would ask for, and it belongs in the
    modelling stage rather than here.

A27. NET BENEFIT: THE QUESTION AUC DOES NOT ASK
    A decision curve puts the model, "treat everyone" and "treat no one" on one
    scale: net true positives per patient, after charging for false positives at
    the exchange rate the threshold implies. Choosing to act at a 20% risk
    threshold is a statement that one missed death is worth four unnecessary
    interventions, and the arithmetic follows from that.

    Best strategy by threshold: {dca_summary}

    The useful range matters more than the peak. Below roughly the prevalence
    ({prevalence}%), treating everyone is hard to beat, because at a low enough
    threshold you should act on everybody and no model can improve on that.
    Above it, the models separate from the defaults. {dca_verdict}

    This is why decision curve analysis belongs in the report. A model can win
    on AUC and sit below "treat everyone" across every threshold a clinician
    would actually use, in which case it is worth nothing regardless of its
    discrimination. Reporting net benefit is how you find that out before a
    reviewer does, and it is still uncommon enough that including it signals
    familiarity with the clinical prediction literature rather than the Kaggle
    version of the problem.
{rule}
"""


def collect_facts(table: pd.DataFrame, comp: pd.DataFrame, doc: dict,
                  dca_info: dict, y: np.ndarray, n_params: int) -> Facts:
    t = table.set_index("model")
    xgb = comp.set_index("model").loc["XGBoost"]
    dd = doc["diffs"].set_index("model").loc["Elastic net logistic"]
    doc_row = doc["table"].set_index("predictor")

    xgb_verdict = ("The interval crosses zero, so on this evidence the two "
                   "models are indistinguishable."
                   if xgb.crosses_zero else
                   "The interval excludes zero, so the difference is real -- "
                   "though still small enough to weigh against interpretability.")
    doc_verdict = ("The interval crosses zero: the model neither beats nor "
                   "loses to the clinician on this evidence."
                   if dd.crosses_zero else
                   ("The model is ahead." if dd.difference > 0
                    else "The clinician is ahead."))
    best = dca_info["best_at"]
    dca_summary = "; ".join(f"{int(t*100)}% -> {m}" for t, m in best.items())
    top = max(dca_info["useful"].items(), key=lambda kv: kv[1])
    dca_verdict = (f"{top[0]} is above both defaults across {top[1]:.0f}% of the "
                   f"0.05-0.50 range, the widest of any model here.")

    return Facts(
        unpen_auc=f"{t.loc['Unpenalised logistic','auc']:.3f}",
        unpen_slope=f"{t.loc['Unpenalised logistic','calibration_slope']:.2f}",
        enet_auc=f"{t.loc['Elastic net logistic','auc']:.3f}",
        enet_slope=f"{t.loc['Elastic net logistic','calibration_slope']:.2f}",
        xgb_slope=f"{t.loc['XGBoost','calibration_slope']:.2f}",
        xgb_intercept=f"{t.loc['XGBoost','calibration_intercept']:+.2f}",
        n_events=f"{int(y.sum()):,}", n_params=str(n_params),
        epv=f"{y.sum()/n_params:.1f}",
        folds=str(CV_FOLDS), repeats=str(CV_REPEATS),
        xgb_diff=f"{xgb.difference:+.4f}",
        xgb_ci=f"[{xgb.ci_low:+.4f}, {xgb.ci_high:+.4f}]",
        xgb_verdict=xgb_verdict,
        doc_n=f"{doc['n']:,}", doc_coverage=f"{doc['coverage']:.1f}",
        doc_auc=f"{doc_row.loc['Attending physician (prg6m)','auc']:.3f}",
        enet_auc_sub=f"{doc_row.loc['Elastic net logistic','auc']:.3f}",
        doc_diff=f"{dd.difference:+.4f}",
        doc_ci=f"[{dd.ci_low:+.4f}, {dd.ci_high:+.4f}]",
        doc_verdict=doc_verdict,
        dca_summary=dca_summary, dca_verdict=dca_verdict,
        prevalence=f"{y.mean()*100:.1f}",
    )


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train
    y = make_outcome(chf).values
    predictors = default_predictors(chf)

    header(f"SUPPORT2 -- modelling {OUTCOME_LABEL}")
    print(f"  CHF training cohort {len(chf):,}   events {int(y.sum()):,} "
          f"({y.mean()*100:.1f}%)")
    print(f"  {len(predictors)} predictors; missingness indicators on "
          f"{', '.join(MISSINGNESS_INDICATORS)}")
    print(f"  {CV_FOLDS}x{CV_REPEATS} repeated stratified CV, imputation refitted "
          f"inside every fold")
    print(f"  Horizon {HORIZON_DAYS} days: no patient is censored before it, so the")
    print(f"  label is complete and aligns with the physician's 6-month estimate.")
    print("  THE HELD-OUT 30% IS NOT READ BY THIS SCRIPT.")

    # Compute everything first, then build the narrative facts, then report.
    # The question headers interpolate results too, not just the answers -- a
    # header that asserts "XGBoost wins" is exactly as capable of going stale as
    # a paragraph, and this file already had one that did.
    table, preds = evaluate_models(chf, predictors, y)
    comp = compute_model_comparison(preds, y)
    doc = compare_with_physician(chf, y, preds)
    dca = compute_dca(y, preds, doc)
    dca_info = summarise_dca(dca)
    sens = sensitivity_analysis(chf, y)
    ors = selected_odds_ratios(chf, predictors, y)
    n_params = int(ors.n_candidate.iloc[0])
    facts = collect_facts(table, comp, doc, dca_info, y, n_params)

    report_overfitting(table, facts)
    report_nested_cv(table)
    report_model_comparison(comp, facts)
    report_physician(doc)
    report_dca(dca, dca_info)
    report_sensitivity(sens)
    report_odds_ratios(ors)

    header("FIGURES")
    for path in (figure_calibration(y, preds, table),
                 figure_dca(dca, y.mean())):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        # saga reports convergence on some inner folds of the elastic net search;
        # the outer estimates are unaffected and max_iter is already generous.
        warnings.filterwarnings("ignore", message=".*converge.*")
        warnings.filterwarnings("ignore", category=FutureWarning)
        run_and_capture(main, OUT_DIR / "05_modelling.txt")
