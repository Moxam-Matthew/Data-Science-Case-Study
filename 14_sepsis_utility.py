"""
14_sepsis_utility.py -- Is the sepsis model USEFUL, not merely accurate?

12_replication.py established that the sepsis model discriminates (held-out AUC
0.739) and is calibrated (slope 0.957). It then stopped, and stopping there is
the thing a principal reviewer notices first. Discrimination and calibration say
the probabilities are ordered and honest. Neither says anyone should act on
them.

The CHF arm could not settle this. Its held-out model was miscalibrated (slope
0.706), and net benefit computed from probabilities that are systematically
wrong is not interpretable. The sepsis cohort can: calibrated, at 45%
prevalence, which is the regime where a model has room to be worth something.

Five questions, chosen because each gives a materially better answer here
than it did in CHF rather than merely repeating one.

    Run:  python 14_sepsis_utility.py

    Q57 and Q58 read the sepsis HOLDOUT. That partition was already spent in
    12_replication.py, so this is a second look at a spent holdout -- permitted
    for describing a model already fixed, NOT for choosing between models. No
    model, threshold or predictor set is selected anywhere in this file; the
    elastic net and its coefficients are exactly the ones 12_replication.py
    fitted before any of these numbers existed.

THE QUESTIONS
    Q57  Decision curve analysis. Across the thresholds a clinician might
         actually use, does the model beat treating everyone and treating
         nobody?
    Q58  The same model at 45% prevalence instead of 25%. Which of the four
         confusion-matrix numbers move, and which do not?
    Q59  The physician is ahead on AUC. Does the model add anything ON TOP of
         the physician, which is the only version of the question that matters?
    Q60  CHF's learning curve was still climbing at n=978. Is this one flat at
         2,458 -- and does that settle whether more data was the answer?
    Q61  A bigger grid bought nothing (11 Q47). Does COMBINING the models --
         which a grid cannot reach -- do better than either alone?

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
    PHYSICIAN_BENCHMARK,
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
from report import RULE, Facts, configure_pandas, header, question, render_answers, run_and_capture
from support2 import SEPSIS_LABEL, analysis_frames, confirmatory_frames
from thresholds import metrics_at, ppv_npv_at_prevalence

OUT_DIR = Path(__file__).resolve().parent / "output"

# Thresholds a clinician might plausibly act on in an ICU sepsis population.
DCA_THRESHOLDS = np.round(np.arange(0.10, 0.81, 0.02), 2)
# Prevalences to re-express the operating point under. 0.450 is this holdout;
# 0.254 is the CHF cohort, carried so the two arms sit on one scale.
DEPLOY_PREVALENCES = [0.10, 0.254, 0.450, 0.60, 0.80]
LEARNING_FRACTIONS = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
LEARNING_REPEATS = 20
# Q58's operating point, fixed in advance rather than read off the decision
# curve. Picking the DCA optimum would be selecting a threshold on a holdout
# that 12_replication.py already spent -- harmless to the Bayes arithmetic,
# which holds at any threshold, but it would contradict the discipline this
# file claims. 0.50 is the project default throughout 08_operating_points.py.
PRESPECIFIED_THRESHOLD = 0.50
# Q61 only. The stacked model refits a nested hyperparameter search inside every
# outer fold, so the full 5x5 protocol used everywhere else costs hours for a
# comparison whose answer 11_ceiling_and_transport.py Q47 already makes likely.
# One 5-fold pass is used instead. All four arms are measured by the identical
# procedure, so the COMPARISON is fair; what it costs is precision on the
# absolute values, and the bootstrap interval on the difference reflects that
# honestly rather than hiding it.
ENSEMBLE_REPEATS = 1
CHF_REFERENCE = {"prevalence": 0.254, "test_auc": 0.655, "test_slope": 0.706,
                 "learning_gain_doubling": 0.028}
# The head-to-head 12_replication.py reported on this same held-out partition.
SEPSIS_HOLDOUT = {"physician_auc": 0.767, "model_auc": 0.735}


def primary_estimator():
    """The model 12_replication.py fitted. Re-declared, never re-chosen."""
    from sklearn.linear_model import LogisticRegressionCV

    return LogisticRegressionCV(
        l1_ratios=(0.2, 0.5, 0.9), Cs=np.logspace(-3, 1, 8), cv=CV_FOLDS,
        scoring="neg_log_loss", max_iter=3000, random_state=RANDOM_STATE,
        refit=True, n_jobs=-1, solver="saga")


def fit_primary() -> dict:
    coh = confirmatory_frames(group=SEPSIS_LABEL)
    tr, te = coh.chf_train, coh.chf_test
    ytr, yte = make_outcome(tr).values, make_outcome(te).values
    predictors = default_predictors(tr)
    pipe = build_pipeline(tr, predictors, primary_estimator(), scale=True)
    pipe.fit(tr[predictors], ytr)
    p = pipe.predict_proba(te[predictors])[:, 1]
    return {"train": tr, "test": te, "y_train": ytr, "y_test": yte,
            "p_test": p, "predictors": predictors, "pipe": pipe,
            **discrimination_metrics(yte, p), **calibration_metrics(yte, p)}


# ═══ Q57. Decision curve analysis ════════════════════════════════════════════
def decision_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    nb_model = net_benefit(y, p, DCA_THRESHOLDS)
    nb_all = treat_all_net_benefit(y, DCA_THRESHOLDS)
    d = pd.DataFrame({"threshold": DCA_THRESHOLDS, "model": nb_model,
                      "treat_all": nb_all, "treat_none": 0.0})
    d["best_alternative"] = d[["treat_all", "treat_none"]].max(axis=1)
    d["advantage"] = d.model - d.best_alternative
    # Net benefit is in units of true positives per patient. Dividing by the
    # exchange rate converts it to something a clinician can hear: how many
    # unnecessary interventions you avoid per 100 patients at the same number
    # of cases found.
    odds = d.threshold / (1 - d.threshold)
    d["avoided_per_100"] = d.advantage / odds * 100
    return d


def report_dca(d: pd.DataFrame, prevalence: float) -> dict:
    question(57, "Decision curve analysis. Across the thresholds a clinician\n"
                 "might actually use, does the model beat treating everyone and\n"
                 "treating nobody?")
    show = d[d.threshold.isin([0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80])]
    print(f"  Prevalence in this partition is {prevalence*100:.1f}%, so "
          f"'treat all' is a\n  serious competitor up to a threshold near that "
          f"value.\n")
    print(show.round(4).to_string(index=False))

    useful = d[d.advantage > 0]
    r = {"prevalence": prevalence,
         "n_useful": len(useful), "n_total": len(d),
         "lo": float(useful.threshold.min()) if len(useful) else np.nan,
         "hi": float(useful.threshold.max()) if len(useful) else np.nan,
         "best_t": float(d.loc[d.advantage.idxmax(), "threshold"]),
         "best_adv": float(d.advantage.max()),
         "best_avoided": float(d.loc[d.advantage.idxmax(), "avoided_per_100"])}
    if len(useful):
        print(f"\n  The model has positive net benefit over the best "
              f"alternative at\n  {r['n_useful']} of {r['n_total']} thresholds, "
              f"spanning {r['lo']:.2f} to {r['hi']:.2f}.")
        print(f"  Largest advantage at threshold {r['best_t']:.2f}: "
              f"{r['best_adv']:+.4f} net benefit,")
        print(f"  equivalent to avoiding {r['best_avoided']:.1f} unnecessary "
              f"interventions per 100\n  patients at the same number of deaths "
              f"correctly identified.")
    else:
        print("\n  The model does not beat the best alternative at ANY "
              "threshold tested.")
    return r


# ═══ Q58. The same model, a different prevalence ═════════════════════════════
def prevalence_table(y: np.ndarray, p: np.ndarray, threshold: float) -> pd.DataFrame:
    m = metrics_at(y, p, threshold)
    rows = []
    for prev in DEPLOY_PREVALENCES:
        ppv, npv = ppv_npv_at_prevalence(m["sensitivity"], m["specificity"], prev)
        rows.append({"prevalence": prev, "sensitivity": m["sensitivity"],
                     "specificity": m["specificity"], "ppv": ppv, "npv": npv})
    return pd.DataFrame(rows)


def report_prevalence(y: np.ndarray, p: np.ndarray,
                      threshold: float = PRESPECIFIED_THRESHOLD) -> dict:
    question(58, "The same model at 45% prevalence instead of 25%. Which of the\n"
                 "four confusion-matrix numbers move, and which do not?")
    t = prevalence_table(y, p, threshold)
    show = t.copy()
    for c in show.columns:
        show[c] = show[c].round(3)
    print(f"  Operating point pre-specified at {threshold:.2f}, NOT read off the")
    print(f"  decision curve -- selecting it there would be choosing a threshold")
    print(f"  on a holdout 12_replication.py already spent. The MODEL does not")
    print(f"  change between these rows; only the population does.\n")
    print(show.to_string(index=False))

    here = t[np.isclose(t.prevalence, 0.450)].iloc[0]
    chf = t[np.isclose(t.prevalence, CHF_REFERENCE["prevalence"])].iloc[0]
    print(f"\n  Moving from the CHF prevalence ({CHF_REFERENCE['prevalence']*100:.1f}%) "
          f"to this one ({here.prevalence*100:.1f}%):")
    print(f"    sensitivity  {chf.sensitivity:.3f} -> {here.sensitivity:.3f}   (fixed)")
    print(f"    specificity  {chf.specificity:.3f} -> {here.specificity:.3f}   (fixed)")
    print(f"    PPV          {chf.ppv:.3f} -> {here.ppv:.3f}   "
          f"({here.ppv - chf.ppv:+.3f})")
    print(f"    NPV          {chf.npv:.3f} -> {here.npv:.3f}   "
          f"({here.npv - chf.npv:+.3f})")
    return {"ppv_here": float(here.ppv), "npv_here": float(here.npv),
            "ppv_chf": float(chf.ppv), "npv_chf": float(chf.npv),
            "sens": float(here.sensitivity), "spec": float(here.specificity),
            "threshold": threshold}


# ═══ Q59. Incremental value over the physician ═══════════════════════════════
def incremental_value(f: dict) -> dict:
    """
    Does the model add anything to what the clinician already knows?

    The right comparison is not model vs physician -- that contest is
    uninteresting because a clinician who has examined the patient holds
    information no dataset here contains. It is physician alone against
    physician PLUS model. Both are fitted by cross-validation on the TRAINING
    partition so the comparison is out-of-fold rather than in-sample; an
    earlier version of this analysis in the CHF arm used in-sample predictions
    and overstated the gain by roughly half.
    """
    from sklearn.linear_model import LogisticRegression

    tr = f["train"]
    has = tr[PHYSICIAN_BENCHMARK].notna()
    sub = tr[has].copy()
    y = make_outcome(sub).values
    sub["_doc"] = 1 - sub[PHYSICIAN_BENCHMARK]        # survival -> death risk

    def oof(cols, label):
        return cross_val_predictions(
            build_pipeline(sub, cols, LogisticRegression(C=1.0, max_iter=3000),
                           scale=True),
            sub[cols], y, n_repeats=CV_REPEATS, label=label)

    p_doc = oof(["_doc"], "physician alone")
    p_both = oof(["_doc"] + f["predictors"], "physician + model")
    p_model = oof(f["predictors"], "model alone")

    return {"n": int(has.sum()),
            "auc_doc": discrimination_metrics(y, p_doc)["auc"],
            "auc_model": discrimination_metrics(y, p_model)["auc"],
            "auc_both": discrimination_metrics(y, p_both)["auc"],
            "gain": bootstrap_auc_difference(y, p_both, p_doc)}


def report_incremental(r: dict) -> dict:
    question(59, "The physician is ahead on AUC. Does the model add anything ON\n"
                 "TOP of the physician, which is the only version of the\n"
                 "question that matters?")
    print(f"  {r['n']:,} training patients carry a physician estimate. All three\n"
          f"  models are out-of-fold, {CV_FOLDS}-fold x {CV_REPEATS} repeats.\n")
    print(f"    physician alone           AUC {r['auc_doc']:.3f}")
    print(f"    model alone               AUC {r['auc_model']:.3f}")
    print(f"    physician + model         AUC {r['auc_both']:.3f}")
    g = r["gain"]
    print(f"\n  Incremental value of the model over the physician:")
    print(f"    {g['difference']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]  "
          f"crosses zero {g['crosses_zero']}")
    return r


# ═══ Q60. The learning curve, at 2.5x the CHF cohort ═════════════════════════
def learning_curve(tr: pd.DataFrame, y: np.ndarray,
                   predictors: list[str]) -> pd.DataFrame:
    import sys
    import time

    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    rows, started, done = [], time.time(), 0
    total = len(LEARNING_FRACTIONS) * LEARNING_REPEATS
    for frac in LEARNING_FRACTIONS:
        aucs = []
        for rep in range(LEARNING_REPEATS):
            idx = (np.arange(len(y)) if frac >= 1.0 else
                   train_test_split(np.arange(len(y)), train_size=frac,
                                    stratify=y, random_state=1000 + rep)[0])
            Xs, ys = tr.iloc[idx][predictors], y[idx]
            pipe = build_pipeline(tr, predictors,
                                  LogisticRegression(C=1.0, max_iter=2000))
            cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=rep)
            oof = np.zeros(len(ys))
            for a, b in cv.split(Xs, ys):
                m = clone(pipe)
                m.fit(Xs.iloc[a], ys[a])
                oof[b] = m.predict_proba(Xs.iloc[b])[:, 1]
            aucs.append(roc_auc_score(ys, oof))
            done += 1
            if done % 10 == 0:
                el = time.time() - started
                print(f"    learning curve   {done:>3}/{total}  {el:5.0f}s, "
                      f"~{el/done*(total-done):4.0f}s left",
                      file=sys.stderr, flush=True)
        a = np.array(aucs)
        rows.append({"fraction": frac, "n": len(idx), "mean_auc": a.mean(),
                     "sd": a.std(ddof=1), "lo": np.percentile(a, 2.5),
                     "hi": np.percentile(a, 97.5)})
    return pd.DataFrame(rows)


def report_learning(t: pd.DataFrame) -> dict:
    question(60, "CHF's learning curve was still climbing at n=978. Is this one\n"
                 "flat at 2,458 -- and does that settle whether more data was\n"
                 "the answer?")
    show = t.copy()
    for c in ("mean_auc", "sd", "lo", "hi"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    half = t.iloc[len(t) // 2:]
    slope = float(np.polyfit(half.n, half.mean_auc, 1)[0] * 100)
    first, last = t.iloc[0], t.iloc[-1]
    r = {"slope_per_100": slope,
         "gain_total": float(last.mean_auc - first.mean_auc),
         "extrapolated_double": float(slope * last.n / 100),
         "n_first": int(first.n), "n_last": int(last.n),
         "auc_first": float(first.mean_auc), "auc_last": float(last.mean_auc),
         "sd_first": float(first.sd), "sd_last": float(last.sd)}
    print(f"\n  AUC from n={r['n_first']:,} to n={r['n_last']:,}: "
          f"{r['auc_first']:.4f} -> {r['auc_last']:.4f} ({r['gain_total']:+.4f})")
    print(f"  Slope over the upper half: {slope:+.4f} AUC per 100 patients")
    print(f"  Extrapolated gain from doubling to {r['n_last']*2:,}: "
          f"{r['extrapolated_double']:+.3f}")
    print(f"  CHF, same extrapolation at n=978: "
          f"{CHF_REFERENCE['learning_gain_doubling']:+.3f}")
    print(f"  Spread across resamples narrowed from sd {r['sd_first']:.4f} "
          f"to {r['sd_last']:.4f}.")
    return r


# ═══ Q61. The last lever: does combining the models beat either alone? ═══════
def stacked_ensemble(f: dict) -> dict:
    """
    Stack the elastic net and XGBoost, and see whether the combination beats
    either on its own.

    This is the last "but did you try..." on the list. 11_ceiling_and_transport
    Q47 answered the bigger-grid version with a rigorous negative; this answers
    the combine-the-models version, which a bigger grid cannot reach.

    Two rules keep it honest. The base models are refitted INSIDE every outer
    fold, so the meta-learner never sees a base prediction made on a row the
    base model was trained on -- the classic way a stacking result flatters
    itself. And this runs on the TRAINING partition only: the sepsis holdout
    was spent in 12_replication.py, so whatever comes out cannot be confirmed
    and therefore cannot become the project's model. That constraint is the
    finding as much as the number is.
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    tr, y, cols = f["train"], f["y_train"], f["predictors"]

    def xgb():
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             min_child_weight=5, eval_metric="logloss",
                             random_state=RANDOM_STATE, n_jobs=-1)

    specs = {
        "Elastic net alone": (primary_estimator(), True),
        "XGBoost alone": (xgb(), False),
        "Stacked (net + XGBoost)": (StackingClassifier(
            estimators=[("net", primary_estimator()), ("xgb", xgb())],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            cv=3, stack_method="predict_proba", n_jobs=1), True),
        "Simple average of the two": (None, None),
    }
    rows, preds = [], {}
    for name, (est, scale) in specs.items():
        if est is None:
            continue
        p = cross_val_predictions(build_pipeline(tr, cols, est, scale=scale),
                                  tr[cols], y, n_repeats=ENSEMBLE_REPEATS,
                                  label=name)
        preds[name] = p
        rows.append({"model": name, **discrimination_metrics(y, p),
                     **calibration_metrics(y, p)})

    # The poor man's ensemble, included because it usually matches stacking on
    # tabular data and costs nothing -- if it does, that is worth knowing.
    avg = (preds["Elastic net alone"] + preds["XGBoost alone"]) / 2
    preds["Simple average of the two"] = avg
    rows.append({"model": "Simple average of the two",
                 **discrimination_metrics(y, avg), **calibration_metrics(y, avg)})

    t = pd.DataFrame(rows)
    best = t.loc[t.auc.idxmax()]
    base = float(t[t.model == "Elastic net alone"].auc.iloc[0])
    return {"table": t, "preds": preds, "base_auc": base,
            "best_name": str(best.model), "best_auc": float(best.auc),
            "stack_gain": float(t[t.model == "Stacked (net + XGBoost)"].auc.iloc[0]) - base,
            "avg_gain": float(t[t.model == "Simple average of the two"].auc.iloc[0]) - base,
            "vs_stack": bootstrap_auc_difference(
                y, preds["Stacked (net + XGBoost)"], preds["Elastic net alone"]),
            "vs_avg": bootstrap_auc_difference(
                y, avg, preds["Elastic net alone"])}


def report_ensemble(r: dict) -> dict:
    question(61, "The last lever. A bigger grid bought nothing (11 Q47). Does\n"
                 "COMBINING the models -- which a grid cannot reach -- do better\n"
                 "than either alone?")
    show = r["table"][["model", "auc", "pr_auc", "calibration_slope", "brier"]].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    for label, key in (("Stacked", "vs_stack"), ("Simple average", "vs_avg")):
        d = r[key]
        print(f"\n  {label} minus elastic net alone:")
        print(f"    {d['difference']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
              f"crosses zero {d['crosses_zero']}")
    return r


# ═══ Figure ══════════════════════════════════════════════════════════════════
def figure_utility(dca: pd.DataFrame, prev: pd.DataFrame, inc: dict,
                   lc: pd.DataFrame, prevalence: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(dca.threshold, dca.model, color=viz.SERIES_BLUE, lw=2.4, label="model")
    ax1.plot(dca.threshold, dca.treat_all, color=viz.SERIES_ORANGE, lw=1.8,
             ls="--", label="treat all")
    ax1.axhline(0, color=viz.INK_MUTED, lw=1.4, ls=":", label="treat none")
    good = dca[dca.advantage > 0]
    if len(good):
        ax1.axvspan(good.threshold.min(), good.threshold.max(),
                    color=viz.SERIES_BLUE, alpha=0.08)
    ax1.set_xlabel("threshold probability")
    ax1.set_ylabel("net benefit")
    ax1.set_title("Decision curve")
    ax1.legend(fontsize=8.5)
    viz.despine(ax1)

    ax2.plot(prev.prevalence * 100, prev.ppv, "o-", color=viz.SERIES_BLUE,
             lw=2.2, label="PPV")
    ax2.plot(prev.prevalence * 100, prev.npv, "o-", color=viz.SERIES_ORANGE,
             lw=2.2, label="NPV")
    ax2.axhline(prev.sensitivity.iloc[0], color=viz.INK_MUTED, lw=1.4, ls=":")
    ax2.text(prev.prevalence.max() * 100, prev.sensitivity.iloc[0] + 0.015,
             "sensitivity (fixed)", fontsize=8, color=viz.INK_MUTED, ha="right")
    for x, lbl in ((CHF_REFERENCE["prevalence"] * 100, "CHF"),
                   (prevalence * 100, "sepsis")):
        ax2.axvline(x, color=viz.BASELINE, lw=1.2)
        ax2.text(x, 0.02, lbl, fontsize=8, color=viz.INK_SECONDARY,
                 ha="center", rotation=90)
    ax2.set_xlabel("prevalence (%)")
    ax2.set_ylabel("value")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Same model, different population")
    ax2.legend(fontsize=8.5, loc="center right")
    viz.despine(ax2)

    names = ["physician\nalone", "model\nalone", "physician\n+ model"]
    vals = [inc["auc_doc"], inc["auc_model"], inc["auc_both"]]
    cols = [viz.SERIES_ORANGE, viz.BASELINE, viz.SERIES_BLUE]
    ax3.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax3.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax3.set_xticks(range(3), names, fontsize=8.5)
    ax3.set_ylim(min(vals) - 0.03, max(vals) + 0.03)
    ax3.set_ylabel("out-of-fold AUC")
    ax3.set_title("Does the model add to the clinician?")
    ax3.grid(axis="x", visible=False)
    viz.despine(ax3)

    ax4.fill_between(lc.n, lc.lo, lc.hi, color=viz.SERIES_BLUE, alpha=0.15)
    ax4.plot(lc.n, lc.mean_auc, "o-", color=viz.SERIES_BLUE, lw=2.2)
    ax4.axvline(978, color=viz.BASELINE, lw=1.4, ls="--")
    ax4.text(978, lc.lo.min(), " CHF cohort size", fontsize=8,
             color=viz.INK_SECONDARY, rotation=90, va="bottom")
    ax4.set_xlabel("training patients")
    ax4.set_ylabel("cross-validated AUC")
    ax4.set_title("Would more data still help?")
    viz.despine(ax4)

    fig.tight_layout()
    viz.caption(fig, f"{OUTCOME_LABEL}, ARF/MOSF w/Sepsis. Decision curve and prevalence panels use the held-out partition; "
                     f"incremental value and the\nlearning curve use the training partition out-of-fold. The sepsis holdout was "
                     f"spent in 12_replication.py -- nothing here selects a model.", y=-0.06)
    return viz.save(fig, "24_sepsis_utility.png")


ANSWERS = """
ANSWERS
{rule}

A57. IS IT USEFUL, NOT JUST ACCURATE?
    {dca_verdict}

    This is the question 12_replication.py stopped one step short of, and the
    one a reviewer reaches first. AUC says the probabilities are ordered.
    Calibration says they are honest. Neither says a clinician should do
    anything differently, because neither knows what an error costs.

    Net benefit does. Choosing to intervene at a threshold of {best_t} is a
    statement that one missed death is worth roughly {exchange} unnecessary
    interventions -- that is what the odds of the threshold mean. Decision curve
    analysis puts the model, "treat everyone" and "treat nobody" on that single
    scale, so the comparison is against what a department would otherwise do
    rather than against chance.

    The comparison is unusually demanding here, and worth flagging before the
    result is quoted. At {prevalence}% prevalence "treat everyone" is a strong
    default: nearly half these patients die, so a blanket policy is right nearly
    half the time and stays competitive until the threshold approaches the
    prevalence. A model that beats treat-all in this cohort has cleared a higher
    bar than the same model would clear at the CHF arm's
    {chf_prevalence}%.

    Why this could not be settled in the CHF arm: its held-out calibration slope
    was {chf_slope}, so its predicted probabilities were systematically wrong.
    Net benefit is computed BY thresholding those probabilities, so a decision
    curve drawn on miscalibrated predictions measures the miscalibration as much
    as the model. At {slope} here, the curve means what it says.

A58. THE SAME MODEL, A DIFFERENT POPULATION
    Sensitivity and specificity are properties of the MODEL. Predictive values
    are properties of the model AND the population, and only the second pair
    moves between the two cohorts.

    Holding the operating point at {threshold}, pre-specified rather than read
    off the decision curve above: sensitivity {sens} and specificity {spec} are
    identical in both arms by construction. PPV goes from {ppv_chf} at the CHF
    cohort's {chf_prevalence}% prevalence to {ppv_here} here, and NPV goes the
    other way, {npv_chf} to {npv_here}.

    The threshold was fixed in advance deliberately. Taking the DCA optimum
    instead would have been choosing an operating point on a holdout that
    12_replication.py already spent. It would not have changed this
    demonstration -- the arithmetic below holds at every threshold -- but a file
    that argues for holdout discipline should not quietly relax it for
    convenience, and the cost of being strict here is nil.

    Nothing about the model changed. This is Bayes' rule, and it is the single
    most common way a published model is misread: a PPV quoted without the
    prevalence it was measured at is not a portable claim. The same model
    deployed on a general ward, where 180-day mortality is far below either
    figure here, would return a PPV that would get it withdrawn -- not because
    it stopped working, but because most flagged patients would survive.

    The practical consequence for anyone deploying this: sensitivity and
    specificity travel, predictive values do not. Recalibration to local
    prevalence is not optional tuning, it is a precondition, and it is cheap --
    the intercept moves, the coefficients do not.

A59. DOES IT ADD ANYTHING TO THE CLINICIAN?
    {incremental_verdict}

    This is the question worth asking, and it is not "does the model beat the
    doctor". That contest is rigged and uninteresting: the physician examined
    the patient and holds information -- appearance, trajectory, the family
    conversation -- that no column in this file contains. 12_replication.py
    found the physician ahead on the holdout ({holdout_note}), and treated that as
    the uncomfortable result. It is the expected one.

    The deployable question is whether a model built from routine data adds
    anything to a judgement the clinician has already formed. If it does, it is
    worth having even while losing head-to-head, because it costs nothing to
    compute and the clinician's estimate is not free. If it does not, then no
    amount of AUC justifies putting it in front of anyone.

    Two honest limits on this comparison. The physician estimate is recorded for
    {inc_n} of the training patients, not all of them, and those are unlikely to
    be a random subset -- an attending is more likely to commit a number for a
    patient they have formed a view about. And `prg6m` is a 6-month survival
    estimate against a {horizon}-day outcome: close, but not the same question,
    which if anything handicaps the physician slightly.

A60. WOULD MORE DATA STILL HELP?
    {learning_verdict}

    The CHF learning curve was still climbing at n=978, and A46 recorded that as
    a wrong expectation -- the reasoning had been that five model families
    landing within 0.05 AUC of each other implied an information limit rather
    than a sample limit, and the curve said otherwise. Doubling that cohort
    extrapolated to {chf_gain} AUC.

    This cohort is 2.5 times larger and answers the question the CHF arm could
    only extrapolate toward. It also separates the two constraints that A46
    conflated: precision and performance. Whatever the curve does, the spread
    across resamples narrowed from sd {sd_first} to {sd_last}, so the larger
    cohort bought steadier estimates regardless of whether it bought a higher
    ceiling.

    What no learning curve here can show is the effect of more VARIABLES.
    Lactate, culture results, antibiotic timing, vasopressor and fluid data --
    the entire content of the Surviving Sepsis bundles -- are absent from this
    dataset (12_replication.py A50). The ranking of what to spend money on is
    unchanged and is worth stating plainly to anyone funding this work: better
    variables first, more patients second, better algorithms last.

A61. THE LAST LEVER
    {ensemble_verdict}

    This closes the last "but did you try..." on the list. 11_ceiling_and_transport
    Q47 widened the elastic net from 24 configurations to 250 ({grid_net}) and
    replaced XGBoost's single hand-set configuration with a 120-candidate random
    search over nine hyperparameters ({grid_xgb}). Both were nested, both were
    honest, both were negative. A grid search cannot reach the one thing left,
    which is combining model families rather than choosing among them.

    One measurement caveat, stated rather than buried. Q61 uses a single 5-fold
    pass where every other analysis here uses 5 folds x 5 repeats, because the
    stack refits a nested search inside each fold and the full protocol costs
    hours. All four arms above go through the identical procedure, so the
    comparison between them is fair; what is lost is precision on the absolute
    values, and the bootstrap intervals reflect that.

    A simple average of the two probability vectors is reported beside the
    stack deliberately. On tabular data it usually matches a trained
    meta-learner, and when it does, the stack's extra machinery is buying
    complexity rather than performance -- which is worth knowing before anyone
    puts a StackingClassifier into a production pipeline.

    AND THE CONSTRAINT THAT ENDS THE DISCUSSION
    Even a genuine improvement could not be adopted here. The sepsis holdout
    was spent in 12_replication.py and the CHF holdout in 10_confirmatory.py.
    A new model requires a fresh holdout, and this dataset no longer has one to
    give; reporting a stacked model against a partition used to evaluate its
    components would be exactly the contamination the whole project is built to
    avoid. So the honest status of anything found here is "promising, and
    unvalidated" -- which is not a model you deploy, it is a hypothesis for the
    next cohort.

    That is the real answer to "can we squeeze out a better score". The place
    to spend effort is not the estimator. Five model families land within 0.05
    AUC of one another (05_modelling.py Q23), the optimism bootstrap puts the
    shrinkage factor at 1.00 (07_validation.py Q32), a ten-fold bigger grid
    moves nothing, and Q60 above shows doubling the cohort would buy
    {learning_double}. Every one of those points at the same conclusion: the
    limit is the information in these twenty-eight variables, and the only
    lever with real headroom is measuring something this dataset never
    recorded.

    A NOTE ON WHAT THIS FILE MAY AND MAY NOT DO
    Q57 and Q58 read a holdout that 12_replication.py already spent. That is
    legitimate for DESCRIBING a model whose every parameter was fixed
    beforehand, and illegitimate for choosing between models -- the second look
    is free only because nothing was selected on it. No threshold reported here
    was chosen to make the model look better; the DCA optimum is read off the
    curve as a description, and the model, its predictors and its coefficients
    are exactly those fitted before any number in this file existed. If a
    genuinely new model came out of this analysis, it would need a fresh
    holdout, and this cohort no longer has one to give.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()

    header("SUPPORT2 -- is the sepsis model useful, not merely accurate?")
    f = fit_primary()
    prevalence = float(f["y_test"].mean())
    print(f"  {SEPSIS_LABEL}: {len(f['train']):,} training, "
          f"{len(f['test']):,} held out.")
    print(f"  Primary model re-fitted exactly as in 12_replication.py: "
          f"AUC {f['auc']:.3f}, calibration slope {f['calibration_slope']:.3f}.")
    print(f"  Held-out prevalence {prevalence*100:.1f}%.")

    dca = decision_curve(f["y_test"], f["p_test"])
    d = report_dca(dca, prevalence)

    pv = report_prevalence(f["y_test"], f["p_test"])

    inc = report_incremental(incremental_value(f))

    lc = learning_curve(f["train"], f["y_train"], f["predictors"])
    lr = report_learning(lc)

    ens = report_ensemble(stacked_ensemble(f))

    header("FIGURES")
    path = figure_utility(dca, prevalence_table(f["y_test"], f["p_test"],
                                                PRESPECIFIED_THRESHOLD),
                          inc, lc, prevalence)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    # ── verdicts, computed ──────────────────────────────────────────────────
    dca_verdict = (
        f"Yes, over a usable range. The model has higher net benefit than both "
        f"treating everyone and treating nobody at {d['n_useful']} of "
        f"{d['n_total']} thresholds tested, spanning {d['lo']:.2f} to "
        f"{d['hi']:.2f}. At its best point, threshold {d['best_t']:.2f}, the "
        f"advantage is equivalent to avoiding {d['best_avoided']:.1f} "
        f"unnecessary interventions per 100 patients while identifying the same "
        f"number of deaths."
        if d["n_useful"] else
        f"No. Across every threshold tested the model fails to beat the better "
        f"of treating everyone and treating nobody. Discrimination of "
        f"{f['auc']:.3f} and near-perfect calibration are not enough to make it "
        f"worth acting on in a population where {prevalence*100:.0f}% die, and "
        f"reporting that plainly is the point of running the analysis.")

    g = inc["gain"]
    incremental_verdict = (
        f"No, not measurably. Physician alone reaches AUC {inc['auc_doc']:.3f}; "
        f"physician plus model reaches {inc['auc_both']:.3f}, a gain of "
        f"{g['difference']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}] whose "
        f"interval includes zero. On this evidence the model does not earn its "
        f"place beside a clinician who has already formed a view."
        if g["crosses_zero"] else
        f"Yes. Physician alone reaches AUC {inc['auc_doc']:.3f}; physician plus "
        f"model reaches {inc['auc_both']:.3f}, a gain of {g['difference']:+.4f} "
        f"[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}], excluding zero. The model "
        f"loses to the physician head-to-head and still adds information the "
        f"physician does not have -- which is the combination that makes a "
        f"decision aid worth deploying.")

    climbing = lr["extrapolated_double"] > 0.010
    learning_verdict = (
        f"It has largely flattened. From n={lr['n_first']:,} to "
        f"n={lr['n_last']:,} the cross-validated AUC moves "
        f"{lr['gain_total']:+.4f}, and the slope over the upper half of the "
        f"curve extrapolates to {lr['extrapolated_double']:+.3f} from doubling "
        f"the cohort again -- against {CHF_REFERENCE['learning_gain_doubling']:+.3f} "
        f"for the same extrapolation in CHF. More patients of this kind would "
        f"buy very little. That is the answer the CHF arm could not give, and "
        f"it settles the sample-size question: the limit here is the "
        f"information in these variables, not the number of rows."
        if not climbing else
        f"It is still climbing. From n={lr['n_first']:,} to n={lr['n_last']:,} "
        f"the cross-validated AUC moves {lr['gain_total']:+.4f}, and the upper "
        f"half of the curve extrapolates to {lr['extrapolated_double']:+.3f} "
        f"from doubling again -- against "
        f"{CHF_REFERENCE['learning_gain_doubling']:+.3f} in CHF. Even at 2,458 "
        f"patients the sample is a live constraint, which is a more useful "
        f"finding than a flat curve: it says the ceiling has not been reached "
        f"and more of the same data is still worth collecting.")

    st, av = ens["vs_stack"], ens["vs_avg"]
    beat = (not st["crosses_zero"]) and st["difference"] > 0
    ensemble_verdict = (
        f"No. Stacking the elastic net and XGBoost reaches AUC "
        f"{ens['base_auc'] + ens['stack_gain']:.4f} against {ens['base_auc']:.4f} "
        f"for the elastic net alone, a difference of {st['difference']:+.4f} "
        f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] whose interval includes "
        f"zero. A plain average of the two probability vectors moves it "
        f"{av['difference']:+.4f} [{av['ci_low']:+.4f}, {av['ci_high']:+.4f}]. "
        f"Combining the families buys nothing measurable, which is what a "
        f"cohort where every family already agrees should produce."
        if not beat else
        f"Yes, marginally. Stacking reaches AUC "
        f"{ens['base_auc'] + ens['stack_gain']:.4f} against {ens['base_auc']:.4f}, "
        f"a difference of {st['difference']:+.4f} [{st['ci_low']:+.4f}, "
        f"{st['ci_high']:+.4f}], excluding zero. Read the size before the sign: "
        f"this is a gain of well under one AUC point, bought with a model far "
        f"harder to explain to a clinician and impossible to confirm on a "
        f"holdout that no longer exists.")

    odds = d["best_t"] / (1 - d["best_t"])
    facts = Facts(
        dca_verdict=dca_verdict, best_t=f"{d['best_t']:.2f}",
        exchange=f"{1/odds:.1f}",
        prevalence=f"{prevalence*100:.1f}",
        chf_prevalence=f"{CHF_REFERENCE['prevalence']*100:.1f}",
        chf_slope=f"{CHF_REFERENCE['test_slope']:.3f}",
        slope=f"{f['calibration_slope']:.3f}",
        threshold=f"{pv['threshold']:.2f}", sens=f"{pv['sens']:.3f}",
        spec=f"{pv['spec']:.3f}", ppv_chf=f"{pv['ppv_chf']:.3f}",
        ppv_here=f"{pv['ppv_here']:.3f}", npv_chf=f"{pv['npv_chf']:.3f}",
        npv_here=f"{pv['npv_here']:.3f}",
        incremental_verdict=incremental_verdict,
        holdout_note=(f"physician {SEPSIS_HOLDOUT['physician_auc']:.3f}, "
                      f"model {SEPSIS_HOLDOUT['model_auc']:.3f}"),
        inc_n=f"{inc['n']:,}", horizon=str(HORIZON_DAYS),
        learning_verdict=learning_verdict,
        chf_gain=f"{CHF_REFERENCE['learning_gain_doubling']:+.3f}",
        sd_first=f"{lr['sd_first']:.4f}", sd_last=f"{lr['sd_last']:.4f}",
        ensemble_verdict=ensemble_verdict,
        grid_net="-0.0043 AUC", grid_xgb="+0.0035 AUC",
        learning_double=f"{lr['extrapolated_double']:+.3f} AUC",
    )
    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "14_sepsis_utility.txt")
