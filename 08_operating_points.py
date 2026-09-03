"""
08_operating_points.py -- Recall, precision, F1, accuracy, ROC: the threshold
metrics, and how to report them without misleading anybody.

Everything in 05_modelling.py is threshold-free. AUC asks whether the ranking is
right; calibration asks whether the probabilities are right. Neither requires a
decision. The moment you quote recall or accuracy you have made a decision --
you have chosen a cut-off -- and if you do not say which, the number is
uninterpretable.

    Run:  python 08_operating_points.py

THE QUESTIONS
    Q35  Report accuracy at the default 0.5 cut-off. Then compare it with a
         model that predicts nobody dies. What does the comparison tell you,
         and what does it tell you about accuracy as a metric here?
    Q36  Sensitivity, specificity, PPV and NPV all move as the threshold moves,
         in different directions. Which one should the threshold be chosen for?
    Q37  F1 is maximised at one threshold and net benefit at another. Both are
         principled. Which is the clinical answer, and why?
    Q38  Sensitivity and specificity are properties of the model. PPV and NPV
         are not. What breaks when this model is deployed somewhere with a
         different case mix?

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
    cross_val_predictions,
    default_predictors,
    make_outcome,
    net_benefit,
    treat_all_net_benefit,
)
from report import RULE, Facts, configure_pandas, header, question, run_and_capture
from support2 import analysis_frames

OUT_DIR = Path(__file__).resolve().parent / "output"

DEFAULT_THRESHOLD = 0.50
SWEEP = np.round(np.arange(0.05, 0.91, 0.01), 2)
# Prevalences to show PPV/NPV under, for Q38. The middle one is this cohort.
DEPLOY_PREVALENCES = [0.05, 0.10, 0.254, 0.40, 0.60]
RULE_OUT_SENSITIVITY = 0.90


# ═══ Metric machinery ════════════════════════════════════════════════════════
def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flag = p >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """
    Every threshold metric anyone asks for, plus the ones that are actually
    informative at 25% prevalence.

    `mcc` (Matthews correlation) is included because it is the honest answer to
    "give me one number": it uses all four cells of the confusion matrix and,
    unlike F1, does not ignore true negatives or assume the two error types cost
    the same.
    """
    c = confusion_at(y, p, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn

    sens = tp / (tp + fn) if (tp + fn) else np.nan          # recall
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan           # precision
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / n if n else np.nan
    f1 = (2 * ppv * sens / (ppv + sens)
          if (ppv + sens) and not np.isnan(ppv) else np.nan)

    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else np.nan

    return {"threshold": threshold, **c, "flagged": tp + fp,
            "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "accuracy": acc, "f1": f1,
            "balanced_accuracy": (sens + spec) / 2,
            "youden_j": sens + spec - 1, "mcc": mcc}


def sweep_metrics(y: np.ndarray, p: np.ndarray,
                  thresholds: np.ndarray = SWEEP) -> pd.DataFrame:
    return pd.DataFrame([metrics_at(y, p, t) for t in thresholds])


def ppv_npv_at_prevalence(sens: float, spec: float, prev: float) -> tuple:
    """
    PPV and NPV implied by a fixed sensitivity/specificity at a new prevalence
    (Bayes). This is the arithmetic behind Q38: the model does not change, the
    population does, and two of the four numbers move anyway.
    """
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    npv = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)
    return float(ppv), float(npv)


def classification_report_all(y: np.ndarray, preds: dict,
                              threshold: float) -> pd.DataFrame:
    """
    The full classification report every model, at one stated threshold.

    Threshold-free columns (AUC, PR-AUC) come first because they do not depend
    on the cut-off; everything after `accuracy` does, and changes if the
    threshold changes. Keeping the two groups visibly separate is the point of
    the table -- a reader should never have to wonder which of these numbers
    would move if someone picked 0.3 instead.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for name, p in preds.items():
        m = metrics_at(y, p, threshold)
        rows.append({
            "model": name,
            "auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["ppv"],
            "npv": m["npv"],
            "f1": m["f1"],
            "balanced_acc": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return pd.DataFrame(rows)


def report_classification_reports(y: np.ndarray, preds: dict,
                                  thresholds: list[float]) -> None:
    header("CLASSIFICATION REPORT, EVERY MODEL, AT EACH STATED THRESHOLD")
    nir = max(y.mean(), 1 - y.mean())
    print(f"  Out-of-fold predictions. Prevalence {y.mean()*100:.1f}%, "
          f"no-information rate {nir*100:.1f}%.")
    print("  auc and pr_auc are threshold-free; every column after them moves")
    print("  when the threshold does.\n")
    for t in thresholds:
        note = ("the conventional default"
                if abs(t - 0.5) < 1e-9 else
                "the prevalence, a defensible reference point"
                if abs(t - round(float(y.mean()), 2)) < 1e-9 else
                "chosen from the decision curve")
        print(f"  -- threshold {t:.2f} ({note}) " + "-" * 28)
        t_df = classification_report_all(y, preds, t)
        show = t_df.copy()
        for c in ("auc", "pr_auc", "accuracy", "sensitivity", "specificity",
                  "precision", "npv", "f1", "balanced_acc", "mcc"):
            show[c] = show[c].round(3)
        print(show.to_string(index=False))
        best_acc = t_df.loc[t_df.accuracy.idxmax()]
        print(f"    best accuracy: {best_acc.model} at {best_acc.accuracy:.3f} "
              f"against a no-information rate of {nir:.3f}\n")


# ═══ Q35. Accuracy and the no-information rate ═══════════════════════════════
def report_accuracy_trap(y: np.ndarray, p: np.ndarray) -> None:
    question(35, f"Report accuracy at the default {DEFAULT_THRESHOLD} cut-off. Then compare it\n"
                 f"with a model that predicts nobody dies. What does the comparison\n"
                 f"tell you about accuracy as a metric here?")
    m = metrics_at(y, p, DEFAULT_THRESHOLD)
    nir = max(y.mean(), 1 - y.mean())
    print(f"  At threshold {DEFAULT_THRESHOLD}:")
    print(f"    accuracy          {m['accuracy']*100:>5.1f}%")
    print(f"    sensitivity       {m['sensitivity']*100:>5.1f}%   "
          f"({m['tp']} of {m['tp']+m['fn']} deaths identified)")
    print(f"    specificity       {m['specificity']*100:>5.1f}%")
    print(f"    precision (PPV)   {m['ppv']*100:>5.1f}%")
    print(f"    F1                {m['f1']:>5.3f}")
    print(f"    patients flagged  {m['flagged']:>5,} of {len(y):,}")
    print()
    print(f"  A model that predicts NOBODY dies:")
    print(f"    accuracy          {nir*100:>5.1f}%   <- the no-information rate")
    print(f"    sensitivity        0.0%   ({0} of {int(y.sum())} deaths identified)")
    print()
    delta = (m["accuracy"] - nir) * 100
    print(f"  The model's accuracy advantage over predicting nothing: "
          f"{delta:+.1f} points.")


# ═══ Q36. The threshold trade-off ════════════════════════════════════════════
def report_threshold_sweep(sweep: pd.DataFrame, y: np.ndarray) -> dict:
    question(36, "Sensitivity, specificity, PPV and NPV all move as the threshold\n"
                 "moves, in different directions. Which one should the threshold\n"
                 "be chosen for?")
    show_at = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    sub = sweep[sweep.threshold.isin(show_at)].copy()
    cols = ["threshold", "flagged", "tp", "fp", "fn", "tn", "sensitivity",
            "specificity", "ppv", "npv", "f1", "accuracy", "mcc"]
    disp = sub[cols].copy()
    for c in ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy", "mcc"):
        disp[c] = disp[c].round(3)
    print(disp.to_string(index=False))

    # Clinically anchored operating points, rather than 0.5 by default.
    best_f1 = sweep.loc[sweep.f1.idxmax()]
    best_j = sweep.loc[sweep.youden_j.idxmax()]
    best_mcc = sweep.loc[sweep.mcc.idxmax()]
    rule_out = sweep[sweep.sensitivity >= RULE_OUT_SENSITIVITY]
    rule_out = rule_out.iloc[-1] if len(rule_out) else None

    print("\n  Operating points chosen by different rules:")
    print(f"    max F1              threshold {best_f1.threshold:.2f}  "
          f"sens {best_f1.sensitivity:.2f}  spec {best_f1.specificity:.2f}  "
          f"PPV {best_f1.ppv:.2f}")
    print(f"    max Youden J        threshold {best_j.threshold:.2f}  "
          f"sens {best_j.sensitivity:.2f}  spec {best_j.specificity:.2f}  "
          f"PPV {best_j.ppv:.2f}")
    print(f"    max MCC             threshold {best_mcc.threshold:.2f}  "
          f"sens {best_mcc.sensitivity:.2f}  spec {best_mcc.specificity:.2f}  "
          f"PPV {best_mcc.ppv:.2f}")
    if rule_out is not None:
        print(f"    >={RULE_OUT_SENSITIVITY:.0%} sensitivity   threshold "
              f"{rule_out.threshold:.2f}  sens {rule_out.sensitivity:.2f}  "
              f"spec {rule_out.specificity:.2f}  NPV {rule_out.npv:.2f}")
    return {"best_f1": best_f1, "best_j": best_j, "best_mcc": best_mcc,
            "rule_out": rule_out}


# ═══ Q37. F1 versus net benefit ══════════════════════════════════════════════
def report_f1_vs_net_benefit(sweep: pd.DataFrame, y: np.ndarray,
                             p: np.ndarray) -> dict:
    question(37, "F1 is maximised at one threshold and net benefit at another. Both\n"
                 "are principled. Which is the clinical answer, and why?")
    nb = net_benefit(y, p, sweep.threshold.values)
    ta = treat_all_net_benefit(y, sweep.threshold.values)
    baseline = np.maximum(ta, 0)
    gain = nb - baseline

    best_f1 = sweep.loc[sweep.f1.idxmax()]
    best_nb_idx = int(np.argmax(gain))
    best_nb_t = float(sweep.threshold.iloc[best_nb_idx])

    print(f"  F1 peaks at threshold {best_f1.threshold:.2f} "
          f"(F1 = {best_f1.f1:.3f})")
    print(f"  Net benefit over the best default peaks at threshold {best_nb_t:.2f} "
          f"(gain = {gain[best_nb_idx]:.4f})")
    print()
    print(f"  {'threshold':>9} {'F1':>7} {'net benefit':>12} {'gain vs default':>16}")
    for t in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        i = int(np.where(sweep.threshold.values == t)[0][0])
        print(f"  {t:>9.2f} {sweep.f1.iloc[i]:>7.3f} {nb[i]:>12.4f} "
              f"{gain[i]:>16.4f}")
    print("\n  F1 does not know what a false negative costs. Net benefit does --")
    print("  the threshold IS the statement of cost.")
    return {"best_f1_t": float(best_f1.threshold), "best_f1": float(best_f1.f1),
            "best_nb_t": best_nb_t, "best_nb_gain": float(gain[best_nb_idx]),
            "nb": nb, "gain": gain}


# ═══ Q38. Transportability of the four numbers ═══════════════════════════════
def report_prevalence_dependence(sweep: pd.DataFrame, anchor_t: float) -> dict:
    question(38, "Sensitivity and specificity are properties of the model. PPV and NPV\n"
                 "are not. What breaks when this model is deployed somewhere with a\n"
                 "different case mix?")
    row = sweep[sweep.threshold == anchor_t].iloc[0]
    sens, spec = float(row.sensitivity), float(row.specificity)
    print(f"  Holding the model and the threshold fixed at {anchor_t:.2f}:")
    print(f"    sensitivity {sens:.3f} and specificity {spec:.3f} do not change.\n")
    print(f"  {'prevalence':>11} {'PPV':>7} {'NPV':>7}")
    rows = []
    for prev in DEPLOY_PREVALENCES:
        ppv, npv = ppv_npv_at_prevalence(sens, spec, prev)
        marker = "  <- this cohort" if abs(prev - 0.254) < 1e-9 else ""
        print(f"  {prev*100:>10.1f}% {ppv:>7.3f} {npv:>7.3f}{marker}")
        rows.append({"prevalence": prev, "ppv": ppv, "npv": npv})
    t = pd.DataFrame(rows)
    lo = t.iloc[0]
    hi = t.iloc[-1]
    print(f"\n  PPV moves from {lo.ppv:.2f} to {hi.ppv:.2f} across that range "
          f"while the model is unchanged.")
    return {"sens": sens, "spec": spec, "table": t,
            "ppv_low": float(lo.ppv), "ppv_high": float(hi.ppv),
            "npv_low": float(lo.npv), "npv_high": float(hi.npv)}


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_roc_pr(y: np.ndarray, preds: dict, sweep: pd.DataFrame, ops: dict):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    # One colour per model. A short list would make zip() silently drop the
    # models past its end, which is exactly what happened the first time.
    palette = (viz.SERIES * 3)[:len(preds)]

    ax1.plot([0, 1], [0, 1], color=viz.BASELINE, lw=1.4, ls="--", zorder=1)
    for (name, p), color in zip(preds.items(), palette):
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=color, lw=2.0,
                 label=f"{name} (AUC {roc_auc_score(y, p):.3f})")

    # Operating points on the reference model's curve. Several criteria land on
    # the same threshold here, so coincident points are merged into one marker
    # rather than drawn on top of each other with unreadable labels.
    wanted = [("best_j", "max Youden J"), ("best_f1", "max F1"),
              ("rule_out", f"{RULE_OUT_SENSITIVITY:.0%} sensitivity")]
    grouped: dict[float, list[str]] = {}
    for key, lbl in wanted:
        r = ops.get(key)
        if r is not None:
            grouped.setdefault(round(float(r.threshold), 4), []).append(lbl)
    for i, (t, labels) in enumerate(sorted(grouped.items())):
        r = sweep[sweep.threshold == t].iloc[0]
        ax1.scatter([1 - r.specificity], [r.sensitivity], s=95, marker="o",
                    color=viz.SERIES_ORANGE, edgecolor=viz.SURFACE, linewidth=1.6,
                    zorder=4)
        ax1.annotate(f"{' / '.join(labels)}  (t={t:.2f})",
                     xy=(1 - r.specificity, r.sensitivity),
                     xytext=(10, -14 if i % 2 == 0 else 10),
                     textcoords="offset points", fontsize=8.5,
                     color=viz.INK_SECONDARY)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("1 − specificity (false positive rate)")
    ax1.set_ylabel("Sensitivity (recall)")
    ax1.set_title("ROC, with operating points marked")
    ax1.legend(loc="lower right", fontsize=8.5)
    viz.despine(ax1)

    prevalence = y.mean()
    ax2.axhline(prevalence, color=viz.BASELINE, lw=1.4, ls="--")
    ax2.text(0.02, prevalence + 0.015, f"no-skill = prevalence {prevalence:.3f}",
             fontsize=8.5, color=viz.INK_SECONDARY)
    for (name, p), color in zip(preds.items(), palette):
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=color, lw=2.0,
                 label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Recall (sensitivity)")
    ax2.set_ylabel("Precision (PPV)")
    ax2.set_title("Precision–recall, which ROC flatters")
    ax2.legend(loc="upper right", fontsize=8.5)
    viz.despine(ax2)

    viz.caption(fig, f"CHF training cohort, {OUTCOME_LABEL}, out-of-fold predictions. ROC's baseline is the\n"
                     f"diagonal regardless of prevalence; the precision-recall baseline is the prevalence "
                     f"itself\n({prevalence:.3f}), which is why PR is the more honest picture when the "
                     f"outcome is uncommon.")
    return viz.save(fig, "17_roc_pr.png")


def figure_threshold_sweep(sweep: pd.DataFrame, y: np.ndarray, ops: dict,
                           f1nb: dict):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    series = [("sensitivity", "Sensitivity (recall)", viz.SERIES_BLUE),
              ("specificity", "Specificity", viz.SERIES_ORANGE),
              ("ppv", "Precision (PPV)", viz.SERIES[2]),
              ("npv", "NPV", viz.SERIES[3])]
    for col, label, color in series:
        ax1.plot(sweep.threshold, sweep[col], color=color, lw=2.2, label=label)
    ax1.axvline(y.mean(), color=viz.INK_MUTED, ls=":", lw=1.4)
    ax1.text(y.mean() + 0.01, 0.03, f"prevalence {y.mean():.3f}", rotation=90,
             fontsize=8, color=viz.INK_SECONDARY, va="bottom")
    ax1.set_xlim(sweep.threshold.min(), sweep.threshold.max())
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Value")
    ax1.set_title("The four numbers move in different directions")
    ax1.legend(loc="center right", fontsize=8.5)
    viz.despine(ax1)

    for col, label, color in (("f1", "F1", viz.SERIES_BLUE),
                              ("accuracy", "Accuracy", viz.SERIES[3]),
                              ("mcc", "Matthews corr.", viz.SERIES[2])):
        ax2.plot(sweep.threshold, sweep[col], color=color, lw=2.2, label=label)
    nir = max(y.mean(), 1 - y.mean())
    ax2.axhline(nir, color=viz.SERIES_ORANGE, ls="--", lw=1.6)
    ax2.text(sweep.threshold.max(), nir + 0.015,
             f"no-information rate {nir:.3f} ", ha="right", fontsize=8.5,
             color=viz.SERIES_ORANGE, weight="600")
    ax2.axvline(f1nb["best_f1_t"], color=viz.SERIES_BLUE, ls=":", lw=1.4)
    ax2.axvline(f1nb["best_nb_t"], color=viz.INK_MUTED, ls=":", lw=1.4)
    ax2.annotate(f"F1 peak\nt={f1nb['best_f1_t']:.2f}",
                 xy=(f1nb["best_f1_t"], 0.08), xytext=(4, 0),
                 textcoords="offset points", fontsize=8, color=viz.SERIES_BLUE)
    ax2.annotate(f"net benefit peak\nt={f1nb['best_nb_t']:.2f}",
                 xy=(f1nb["best_nb_t"], 0.30), xytext=(4, 0),
                 textcoords="offset points", fontsize=8, color=viz.INK_SECONDARY)
    ax2.set_xlim(sweep.threshold.min(), sweep.threshold.max())
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Threshold")
    ax2.set_title("Accuracy never falls below the no-information rate")
    ax2.legend(loc="lower right", fontsize=8.5)
    viz.despine(ax2)

    viz.caption(fig, f"CHF training cohort, {OUTCOME_LABEL}. Left: no single threshold optimises all four, so\n"
                     f"the choice is a clinical statement rather than a tuning step. Right: accuracy sits near\n"
                     f"the no-information rate across the whole range, which is why it is uninformative here.")
    return viz.save(fig, "18_threshold_sweep.png")


ANSWERS = """
ANSWERS
{rule}

A35. WHY ACCURACY IS THE WRONG HEADLINE HERE
    At the default 0.5 cut-off the model is {acc}% accurate. A model that
    predicts nobody dies is {nir}% accurate. The entire value of the fitting,
    the imputation, the penalisation and the cross-validation, measured this
    way, is {acc_delta} percentage points.

    That is not because the model is worthless -- 05_modelling.py shows it
    discriminates and is well calibrated -- but because accuracy at a fixed
    threshold is close to uninformative when one class holds
    {prevalence}% of the sample. The no-information rate is the floor any
    metric must beat, and accuracy starts most of the way there for free.

    Look at what the 0.5 threshold actually does: it flags {flagged} patients of
    {n}, identifying {tp} of {events} deaths. Sensitivity {sens_at_50}. A model
    deployed to find high-risk patients that misses most of them is not usable,
    and its accuracy is the number that hides this.

    Two conclusions. Accuracy should not be the headline for an imbalanced
    clinical outcome; report it, but beside the no-information rate so the
    comparison cannot be avoided. And 0.5 is not a neutral default -- it is a
    specific and usually wrong claim about the relative cost of the two errors.

A36. WHICH NUMBER TO CHOOSE THE THRESHOLD FOR
    None of them, on their own. The four move in different directions by
    construction, and the sweep shows there is no threshold that is good at all
    of them: raise it and specificity and PPV improve while sensitivity and NPV
    fall; lower it and the reverse. That is not a defect to be optimised away,
    it is the trade the threshold exists to express.

    Which number matters depends on what the model is FOR, and the four map onto
    different clinical uses:

      * SENSITIVITY (recall) for a rule-out tool. If the question is "who can
        safely be discharged", missing a death is the expensive error and you
        accept false positives to avoid it. Here {rule_out_desc}
      * SPECIFICITY and PPV for a rule-in tool, where flagging triggers a scarce
        or invasive intervention and a false positive has real cost.
      * NPV for reassurance -- the number a clinician implicitly wants when they
        say "so this patient is low risk?"

    Now look at where the named optimisation rules actually land: F1 at {f1_t},
    Youden's J at {j_t}, Matthews correlation at {mcc_t}. They very nearly
    agree, and they cluster around the prevalence ({prevalence}%).

    That agreement is not reassuring, and it is worth being precise about why.
    It is a property of a reasonably calibrated model at moderate prevalence --
    several of these criteria drift toward the prevalence in that situation --
    rather than evidence that they have found the right answer. They agree
    because they are all asking a similar arithmetic question. None of them
    asked a clinician anything.

    The threshold that does NOT cluster with them is the one derived from a
    clinical requirement: {rule_out_t} for {rule_out_sens_pct}% sensitivity, far
    below the rest. That is the tell. Formula-driven thresholds converge on an
    answer with no clinical content; a decision-driven threshold sits wherever
    the decision puts it.

    Of the single-number summaries, MCC is the one to prefer if you must pick.
    It uses all four cells of the confusion matrix, whereas F1 ignores true
    negatives entirely -- which is a strange thing to ignore when three-quarters
    of your patients are one.

A37. F1 VERSUS NET BENEFIT
    F1 peaks at threshold {f1_t}; net benefit over the best default strategy
    peaks at {nb_t}. On this cohort they very nearly coincide -- and that near
    agreement is the most misleading result in this file, because it invites the
    conclusion that F1 was fine all along.

    It was not. The agreement is a coincidence of this dataset, and the way to
    see that is to ask what would happen if the clinical preference changed.
    Suppose a missed death were judged ten times worse than an unnecessary
    intervention rather than three. Net benefit moves immediately, because the
    exchange rate is an input: you evaluate at a threshold near 0.09 instead.
    F1 does not move at all. It cannot, because it has no way to represent the
    preference -- there is no argument to F1 in which a cost ratio could be
    supplied. Its peak stays where it was, and it would now be recommending an
    operating point the clinician has explicitly rejected.

    F1 is the harmonic mean of precision and recall. It treats a false positive
    and a false negative as equally regrettable, and it never mentions true
    negatives. Neither assumption is a clinical statement -- they are properties
    of the formula, inherited from information retrieval, where the cost of
    showing someone an irrelevant document and the cost of hiding a relevant one
    are indeed roughly symmetric. In heart failure they are not remotely
    symmetric, and nobody has said what the ratio should be.

    Net benefit asks the question the other way round. You pick the threshold
    FIRST, because the threshold is the clinical statement: acting at 20% risk
    says one missed death is worth four unnecessary interventions. Net benefit
    then tells you whether the model is worth using at that stated exchange
    rate. The preference is an input, not something the metric invents.

    So the clinical answer is: do not let a metric choose the threshold. Choose
    the threshold from the decision, then report sensitivity, specificity, PPV
    and NPV AT that threshold, and use decision curve analysis to check the
    model beats treat-all and treat-none there. Reporting "F1 = {f1_best}" with
    no threshold attached is reporting the output of an optimisation nobody
    asked for.

    Report F1 and accuracy because reviewers expect them, and be able to say
    exactly this about why they are not driving the decision.

A38. WHAT TRANSPORTS AND WHAT DOES NOT
    Sensitivity and specificity are conditional on the true state -- among those
    who died, what fraction did we flag -- so they are properties of the model
    and its threshold. Move the model to a different hospital and, case mix
    aside, they travel.

    PPV and NPV are conditional on the PREDICTION -- among those we flagged,
    what fraction died -- so they depend on how many people in the population
    actually have the outcome. Holding this model and threshold fixed at {anchor}
    with sensitivity {sens_anchor} and specificity {spec}, PPV runs from {ppv_low} at
    5% prevalence to {ppv_high} at 60%, while NPV runs the other way from
    {npv_low} to {npv_high}. The model has not changed at all.

    That is why a published PPV is close to meaningless without the prevalence
    beside it, and why "our model has 60% precision" is not a portable claim. It
    also compounds the transportability problem from 04_clinical.py A22: this
    cohort is from 1989-94 with {prevalence}% 180-day mortality, and a
    contemporary heart failure population on modern therapy has substantially
    lower mortality. Deploy this threshold there and the PPV falls even if the
    model's discrimination is unchanged.

    The practical rule: quote sensitivity and specificity as model properties;
    quote PPV and NPV only with the prevalence they were computed at; and if you
    are asked what the model would do somewhere else, recompute them from Bayes
    rather than assuming they carry over.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train
    y = make_outcome(chf).values
    predictors = default_predictors(chf)

    header(f"SUPPORT2 -- operating points, {OUTCOME_LABEL}")
    print(f"  CHF training cohort {len(chf):,}, {int(y.sum()):,} events "
          f"({y.mean()*100:.1f}%)")
    print("  Out-of-fold predictions, same CV design as 05_modelling.py.")
    print("  Training partition only; the held-out 30% remains unread.")

    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    # Same five models and settings as 05_modelling.py, so the threshold metrics
    # here attach to the models whose AUC and calibration are reported there.
    shared = dict(cv=CV_FOLDS, scoring="neg_log_loss", max_iter=3000,
                  random_state=RANDOM_STATE, refit=True, n_jobs=-1,
                  solver="saga")
    specs = {
        "Unpenalised logistic": (LogisticRegression(C=np.inf, max_iter=4000), True),
        "LASSO": (LogisticRegressionCV(l1_ratios=(1.0,),
                                       Cs=np.logspace(-3, 1, 8), **shared), True),
        "Elastic net": (LogisticRegressionCV(l1_ratios=(0.2, 0.5, 0.9),
                                             Cs=np.logspace(-3, 1, 8), **shared), True),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=3,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=2.0,
                                  min_child_weight=5, eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=-1), False),
        "Decision tree (d3)": (DecisionTreeClassifier(max_depth=3,
                                                     min_samples_leaf=40,
                                                     random_state=RANDOM_STATE), False),
    }
    preds = {
        name: cross_val_predictions(
            build_pipeline(chf, predictors, est, scale=scale),
            chf[predictors], y, n_repeats=CV_REPEATS, label=name)
        for name, (est, scale) in specs.items()
    }
    p = preds["Elastic net"]          # the reference model throughout
    sweep = sweep_metrics(y, p)

    m50 = metrics_at(y, p, DEFAULT_THRESHOLD)
    nir = max(y.mean(), 1 - y.mean())

    report_accuracy_trap(y, p)
    ops = report_threshold_sweep(sweep, y)
    f1nb = report_f1_vs_net_benefit(sweep, y, p)
    anchor_t = f1nb["best_nb_t"]
    prev_dep = report_prevalence_dependence(sweep, anchor_t)

    # Every metric, every model, at three stated thresholds -- the conventional
    # default, the prevalence, and the one the decision curve points to.
    report_classification_reports(
        y, preds, sorted({DEFAULT_THRESHOLD, round(float(y.mean()), 2), anchor_t}))

    header("FIGURES")
    for path in (figure_roc_pr(y, preds, sweep, ops),
                 figure_threshold_sweep(sweep, y, ops, f1nb)):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    ro = ops["rule_out"]
    rule_out_desc = (
        f"{RULE_OUT_SENSITIVITY:.0%} sensitivity needs a threshold of "
        f"{ro.threshold:.2f}, which flags {int(ro.flagged):,} patients and "
        f"gives an NPV of {ro.npv:.2f}."
        if ro is not None else
        "no threshold in the swept range reaches 90% sensitivity.")

    facts = Facts(
        acc=f"{m50['accuracy']*100:.1f}", nir=f"{nir*100:.1f}",
        acc_delta=f"{(m50['accuracy']-nir)*100:+.1f}",
        prevalence=f"{y.mean()*100:.1f}",
        flagged=f"{m50['flagged']:,}", n=f"{len(y):,}",
        tp=str(m50["tp"]), events=f"{int(y.sum()):,}",
        sens_at_50=f"{m50['sensitivity']:.3f}",
        rule_out_desc=rule_out_desc,
        j_t=f"{ops['best_j'].threshold:.2f}",
        f1_t=f"{ops['best_f1'].threshold:.2f}",
        mcc_t=f"{ops['best_mcc'].threshold:.2f}",
        nb_t=f"{f1nb['best_nb_t']:.2f}",
        f1_best=f"{f1nb['best_f1']:.3f}",
        anchor=f"{anchor_t:.2f}",
        rule_out_t=(f"{ops['rule_out'].threshold:.2f}"
                    if ops["rule_out"] is not None else "n/a"),
        rule_out_sens_pct=f"{RULE_OUT_SENSITIVITY*100:.0f}",
        sens_anchor=f"{prev_dep['sens']:.3f}",
        spec=f"{prev_dep['spec']:.3f}",
        ppv_low=f"{prev_dep['ppv_low']:.2f}", ppv_high=f"{prev_dep['ppv_high']:.2f}",
        npv_low=f"{prev_dep['npv_low']:.2f}", npv_high=f"{prev_dep['npv_high']:.2f}",
    )
    print(ANSWERS.format(rule=RULE, **facts))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "08_operating_points.txt")
