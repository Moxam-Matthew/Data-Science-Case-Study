"""
Threshold metrics: the numbers that appear once a probability is turned into a
decision.

Extracted from 08_operating_points.py when the sepsis replication needed the
same machinery. Duplicating it would have been the worse option twice over: the
two arms would drift, and the whole argument of 14_sepsis_utility.py is that
these numbers move with PREVALENCE while the model stays fixed -- a claim that
is only checkable if both cohorts are measured by identical code.

Nothing here decides a threshold. Choosing one is a clinical judgement about
the exchange rate between a missed case and an unnecessary intervention, and it
belongs in the analysis scripts where that trade-off is argued.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 0.05 to 0.90 in hundredths. np.round is not cosmetic: np.arange on floats
# produces values like 0.30000000000000004, which then print as a threshold
# nobody chose and break equality lookups against a named operating point.
SWEEP = np.round(np.arange(0.05, 0.91, 0.01), 2)


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
    informative when the classes are unbalanced.

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
    (Bayes). The model does not change, the population does, and two of the four
    numbers move anyway -- which is why a reported PPV without its prevalence is
    not a portable claim about a model.
    """
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    npv = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)
    return float(ppv), float(npv)
