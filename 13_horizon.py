"""
13_horizon.py -- Why not just predict death, and use every event?

The 180-day horizon throws away events. In the sepsis training partition 1,440
patients died at some point and only 1,091 died within 180 days, so relaxing
the horizon recovers about a third more events for free. It is the obvious
move, it is the one most people reach for, and it deserves a measured answer
rather than an appeal to principle.

`death` in SUPPORT2 does not mean "died". It means "was observed to die before
last contact", and 03_cohort.py Q16 established that contact was not equal:
the study enrolled in two waves and followed the first for roughly twice as
long. A model trained on that label may be learning part of the enrolment
calendar alongside the biology.

So this file does the thing rather than arguing about it. It fits the same
model on both outcomes, compares them, and then decomposes any difference to
find where it came from. It closes with the analysis that does use every
patient and every timeframe -- a survival model, because handling unequal
follow-up is what censoring is for.

    Run:  python 13_horizon.py

    Training partition only. The sepsis holdout was spent in 12_replication.py
    and nothing here may revise that result.

THE QUESTIONS
    Q54  How many more events does dropping the horizon buy, and what does the
         extra label encode besides death?
    Q55  Fit both outcomes and compare. If the any-horizon model scores higher,
         where do the extra points come from?
    Q56  What does it cost to do this properly? Fit the survival model that
         uses every patient and every timeframe, and compare.

A PREDICTION, RECORDED BEFORE RUNNING
    Because A46 in 11_ceiling_and_transport.py got its expectation wrong and
    said so, the same discipline applies here:
      1. The any-horizon model will score HIGHER than the 180-day model.
      2. Removing the three protocol-differing labs will cost the any-horizon
         model MORE than it costs the 180-day model, because that is where the
         wave signature lives.
      3. Harrell's C for the Cox model will land close to the 180-day AUC,
         since both are concordance measures over largely the same information.
    Whichever of these is wrong is the interesting part.

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
    RANDOM_STATE,
    bootstrap_auc_difference,
    build_pipeline,
    calibration_metrics,
    cross_val_predictions,
    default_predictors,
    discrimination_metrics,
    make_outcome,
)
from report import RULE, Facts, configure_pandas, fmt_p, header, question, render_answers, run_and_capture
from support2 import (
    OUTCOME_EVENT,
    OUTCOME_TIME,
    SEPSIS_LABEL,
    analysis_frames,
)

OUT_DIR = Path(__file__).resolve().parent / "output"

WAVE_MARKER = "bun"      # 03_cohort.py Q16: ~100% missing in the early wave


def any_horizon_outcome(df: pd.DataFrame) -> np.ndarray:
    """
    The tempting label: died at any point before last contact.

    Deliberately NOT routed through make_outcome, which refuses to label
    patients whose status at a fixed horizon is unknown. That guard is the
    right behaviour and this outcome sidesteps it, which is precisely the
    problem being demonstrated.
    """
    return (df[OUTCOME_EVENT] == 1).astype(int).values


# ═══ Q54. What the extra events actually encode ══════════════════════════════
def horizon_accounting(sep: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    y180, yany = make_outcome(sep).values, any_horizon_outcome(sep)
    early = sep[WAVE_MARKER].isna().values

    rows = []
    for name, mask in (("early wave", early), ("late wave", ~early)):
        cens = mask & (sep[OUTCOME_EVENT].values == 0)
        rows.append({
            "wave": name, "n": int(mask.sum()),
            "median_fu_censored_d": float(np.median(sep[OUTCOME_TIME].values[cens])),
            "max_fu_observed_d": float(sep[OUTCOME_TIME].values[mask].max()),
            "died_180d_pct": 100 * float(y180[mask].mean()),
            "died_any_pct": 100 * float(yany[mask].mean()),
        })
    waves = pd.DataFrame(rows)

    flag = sep[WAVE_MARKER].isna().astype(int).values
    return {
        "waves": waves, "y180": y180, "yany": yany, "early": early,
        "n_180": int(y180.sum()), "n_any": int(yany.sum()),
        "gap_180": abs(waves.died_180d_pct.diff().iloc[-1]),
        "gap_any": abs(waves.died_any_pct.diff().iloc[-1]),
        "flag_auc_180": roc_auc_score(y180, flag),
        "flag_auc_any": roc_auc_score(yany, flag),
        "fu_ratio": waves.median_fu_censored_d.iloc[0] / waves.median_fu_censored_d.iloc[1],
    }


def report_accounting(r: dict) -> None:
    question(54, "How many more events does dropping the horizon buy, and what\n"
                 "does the extra label encode besides death?")
    extra = r["n_any"] - r["n_180"]
    print(f"  {HORIZON_DAYS}-day deaths {r['n_180']:,}   any-horizon deaths "
          f"{r['n_any']:,}   (+{extra:,}, +{100*extra/r['n_180']:.0f}%)\n")
    show = r["waves"].copy()
    for c in show.columns[2:]:
        show[c] = show[c].round(1)
    print(show.to_string(index=False))
    print(f"\n  Censored follow-up runs {r['fu_ratio']:.2f}x longer in the early wave.")
    print(f"  Between-wave mortality gap, {HORIZON_DAYS}-day : "
          f"{r['gap_180']:.1f} points")
    print(f"  Between-wave mortality gap, any horizon : "
          f"{r['gap_any']:.1f} points  "
          f"({r['gap_any']/r['gap_180']:.1f}x larger)")
    print(f"\n  A single flag -- '{WAVE_MARKER} was not recorded', a fact about a\n"
          f"  data-collection protocol carrying no clinical information:")
    print(f"    predicting {HORIZON_DAYS}-day death   AUC {r['flag_auc_180']:.3f}")
    print(f"    predicting any-horizon death  AUC {r['flag_auc_any']:.3f}")


# ═══ Q55. Fit both, and find the source of the difference ════════════════════
def compare_horizons(sep: pd.DataFrame, r: dict) -> pd.DataFrame:
    """
    Four models: two outcomes x two predictor sets.

    Dropping bun, urine and glucose removes the variables whose MISSINGNESS
    encodes the enrolment wave. If the any-horizon model is leaning on the
    calendar, it should lose more from that removal than the 180-day model
    does -- which is a testable decomposition rather than an assertion.
    """
    from sklearn.linear_model import LogisticRegressionCV

    full = default_predictors(sep)
    lean = default_predictors(sep, drop_protocol_missing=True)

    def estimator():
        return LogisticRegressionCV(
            l1_ratios=(0.2, 0.5, 0.9), Cs=np.logspace(-3, 1, 8), cv=CV_FOLDS,
            scoring="neg_log_loss", max_iter=3000, random_state=RANDOM_STATE,
            refit=True, n_jobs=-1, solver="saga")

    rows, preds = [], {}
    for oname, y in ((f"{HORIZON_DAYS}-day", r["y180"]), ("any horizon", r["yany"])):
        for pname, cols in (("all predictors", full),
                            ("minus protocol labs", lean)):
            label = f"{oname}, {pname}"
            p = cross_val_predictions(
                build_pipeline(sep, cols, estimator(), scale=True),
                sep[cols], y, n_repeats=CV_REPEATS, label=label)
            preds[label] = p
            rows.append({"outcome": oname, "predictors": pname,
                         "n_predictors": len(cols),
                         **discrimination_metrics(y, p),
                         **calibration_metrics(y, p)})
    t = pd.DataFrame(rows)
    t.attrs["preds"] = preds
    return t


def predict_the_wave(sep: pd.DataFrame) -> dict:
    """
    Can the predictor set identify which enrolment wave a patient came from?

    If it can, then any label correlated with follow-up length is partly
    learnable from the features -- which is the mechanism, stated as a number
    instead of a worry.
    """
    from sklearn.linear_model import LogisticRegression

    # The protocol labs are excluded outright: their missingness DEFINES the
    # wave, so leaving them in would ask whether a tautology is predictable.
    lean = default_predictors(sep, drop_protocol_missing=True)
    wave = sep[WAVE_MARKER].isna().astype(int).values
    p = cross_val_predictions(
        build_pipeline(sep, lean, LogisticRegression(C=1.0, max_iter=3000),
                       scale=True),
        sep[lean], wave, n_repeats=1, label="wave from clinical variables")
    return {"auc": discrimination_metrics(wave, p)["auc"], "n_predictors": len(lean)}


def report_comparison(t: pd.DataFrame, wave: dict, r: dict) -> dict:
    question(55, "Fit both outcomes and compare. If the any-horizon model scores\n"
                 "higher, where do the extra points come from?")
    show = t[["outcome", "predictors", "n_predictors", "auc", "pr_auc",
              "calibration_slope", "brier"]].copy()
    for c in ("auc", "pr_auc", "calibration_slope", "brier"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    def get(o, p):
        return float(t[(t.outcome == o) & (t.predictors == p)].auc.iloc[0])

    h = f"{HORIZON_DAYS}-day"
    d = {
        "auc_180_full": get(h, "all predictors"),
        "auc_any_full": get("any horizon", "all predictors"),
        "auc_180_lean": get(h, "minus protocol labs"),
        "auc_any_lean": get("any horizon", "minus protocol labs"),
    }
    d["headline_gain"] = d["auc_any_full"] - d["auc_180_full"]
    d["cost_180"] = d["auc_180_lean"] - d["auc_180_full"]
    d["cost_any"] = d["auc_any_lean"] - d["auc_any_full"]

    print(f"\n  Apparent gain from dropping the horizon: "
          f"{d['headline_gain']:+.4f} AUC")
    print(f"\n  Cost of removing the three protocol-differing labs:")
    print(f"    {h:<12} outcome  {d['cost_180']:+.4f}")
    print(f"    any-horizon  outcome  {d['cost_any']:+.4f}")
    print(f"    the any-horizon model is hurt "
          f"{abs(d['cost_any']) - abs(d['cost_180']):+.4f} more")

    preds = t.attrs["preds"]
    diff = bootstrap_auc_difference(
        r["yany"], preds["any horizon, all predictors"],
        preds["any horizon, minus protocol labs"])
    d["removal_diff"] = diff
    print(f"    bootstrap on the any-horizon model: {diff['difference']:+.4f} "
          f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
          f"crosses zero {diff['crosses_zero']}")

    print(f"\n  Can the CLINICAL variables alone identify the enrolment wave?")
    print(f"    {wave['n_predictors']} predictors, protocol labs excluded: "
          f"AUC {wave['auc']:.3f}")
    d["wave_auc"] = wave["auc"]
    return d


# ═══ Q56. The analysis that uses every timeframe properly ════════════════════
def fit_cox(sep: pd.DataFrame) -> dict:
    """
    Cox proportional hazards on the full follow-up.

    This is what "use every event regardless of timeframe" actually looks like
    when done correctly. A patient followed 600 days and a patient followed
    2,000 days contribute the exposure they actually had, so the unequal
    follow-up that poisons the binary label is handled rather than absorbed.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from sklearn.impute import SimpleImputer

    cols = [c for c in default_predictors(sep)
            if pd.api.types.is_numeric_dtype(sep[c])]
    X = sep[cols]
    imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                           columns=cols, index=X.index)
    data = imputed.assign(**{OUTCOME_TIME: sep[OUTCOME_TIME].values,
                             OUTCOME_EVENT: sep[OUTCOME_EVENT].values})
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(data, duration_col=OUTCOME_TIME, event_col=OUTCOME_EVENT)
    ph = proportional_hazard_test(cph, data, time_transform="rank")
    return {"model": cph, "concordance": float(cph.concordance_index_),
            "n": len(data), "events": int(sep[OUTCOME_EVENT].sum()),
            "ph": ph.summary, "n_violating": int((ph.summary.p < 0.05).sum()),
            "n_terms": len(cols),
            "median_fu": float(sep[OUTCOME_TIME].median())}


def report_cox(c: dict, auc_180: float) -> None:
    question(56, "What does it cost to do this properly? Fit the survival model\n"
                 "that uses every patient and every timeframe, and compare.")
    print(f"  {c['n']:,} patients, {c['events']:,} deaths, {c['n_terms']} terms.")
    print(f"  Every patient contributes the follow-up they actually had; "
          f"median {c['median_fu']:.0f} days.\n")
    print(f"    Harrell's C (Cox, full follow-up)   {c['concordance']:.3f}")
    print(f"    AUC ({HORIZON_DAYS}-day logistic)              {auc_180:.3f}")
    print(f"    difference                          "
          f"{c['concordance'] - auc_180:+.3f}")
    print(f"\n  Proportional hazards test: {c['n_violating']} of {c['n_terms']} "
          f"terms violate at p < 0.05.")
    top = c["ph"].sort_values("p").head(5)[["test_statistic", "p"]].copy()
    top["p"] = top.p.apply(fmt_p)
    print(top.round(2).to_string())


# ═══ Figure ══════════════════════════════════════════════════════════════════
def figure_horizon(t: pd.DataFrame, acc: dict, cox: dict, auc_180: float):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    w = acc["waves"]
    x = np.arange(2)
    ax1.bar(x - 0.19, w.died_180d_pct, width=0.36, color=viz.SERIES_BLUE,
            label=f"died within {HORIZON_DAYS} d")
    ax1.bar(x + 0.19, w.died_any_pct, width=0.36, color=viz.SERIES_ORANGE,
            label="died, any horizon")
    ax1.set_xticks(x, [f"{r.wave}\n(median follow-up {r.median_fu_censored_d:.0f} d)"
                       for r in w.itertuples()], fontsize=8.5)
    ax1.set_ylabel("mortality (%)")
    ax1.set_title("The label moves with follow-up,\nnot with illness")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(axis="x", visible=False)
    viz.despine(ax1)

    h = f"{HORIZON_DAYS}-day"
    order = [(h, "all predictors"), (h, "minus protocol labs"),
             ("any horizon", "all predictors"), ("any horizon", "minus protocol labs")]
    vals = [float(t[(t.outcome == o) & (t.predictors == p)].auc.iloc[0])
            for o, p in order]
    cols = [viz.SERIES_BLUE, viz.BASELINE, viz.SERIES_ORANGE, viz.BASELINE]
    ax2.bar(range(4), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax2.set_xticks(range(4), [f"{h}\nall", f"{h}\nminus labs",
                              "any horizon\nall", "any horizon\nminus labs"],
                   fontsize=8)
    ax2.set_ylim(min(vals) - 0.03, max(vals) + 0.03)
    ax2.set_ylabel("cross-validated AUC")
    ax2.set_title("The higher score, and what it rests on")
    ax2.grid(axis="x", visible=False)
    viz.despine(ax2)

    labels = [f"flag alone\n{h}", f"flag alone\nany horizon",
              f"model\n{h}", "Cox\nfull follow-up"]
    vals3 = [acc["flag_auc_180"], acc["flag_auc_any"], auc_180, cox["concordance"]]
    cols3 = [viz.BASELINE, viz.SERIES_ORANGE, viz.SERIES_BLUE, viz.SERIES[2]]
    ax3.bar(range(4), vals3, color=cols3, width=0.62)
    ax3.axhline(0.5, color=viz.INK_MUTED, lw=1.2, ls=":")
    ax3.text(3.45, 0.505, "chance", fontsize=8, color=viz.INK_MUTED, ha="right")
    for i, v in enumerate(vals3):
        ax3.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax3.set_xticks(range(4), labels, fontsize=8)
    ax3.set_ylim(0.45, max(vals3) + 0.05)
    ax3.set_ylabel("concordance")
    ax3.set_title("A protocol flag should not\npredict death")
    ax3.grid(axis="x", visible=False)
    viz.despine(ax3)

    fig.tight_layout()
    viz.caption(fig, "ARF/MOSF w/Sepsis training partition, 2,458 patients. 'Protocol labs' are bun, urine and\n"
                     "glucose, absent for essentially the whole early enrolment wave (03_cohort.py Q16). The Cox\n"
                     "model uses every patient's actual follow-up rather than a fixed window.", y=-0.05)
    return viz.save(fig, "23_horizon.png")


ANSWERS = """
ANSWERS
{rule}

A54. WHAT THE EXTRA EVENTS ENCODE
    Dropping the horizon takes the sepsis training cohort from {n_180} events to
    {n_any}, a gain of {extra_pct}%. For a cohort that spent 07_validation.py
    Q31 failing Riley's sample-size criteria, a third more events for free is
    exactly the trade anyone would want to make.

    The trade is not free. `death` in this dataset means "was observed to die
    before last contact", and contact was not equal. Censored follow-up runs
    {fu_ratio}x longer in the early enrolment wave -- and the longest follow-up
    recorded in the late wave is shorter than the MEDIAN follow-up in the early
    one, so a late-wave patient cannot be observed to die past a date the early
    wave routinely passed.

    Watch what that does to the label. Between the two waves the {horizon}-day
    mortality gap is {gap_180} points. The any-horizon gap is {gap_any} points,
    {gap_ratio}x larger. Nobody got sicker between waves; the study just looked
    at one of them for twice as long, and the extra events are concentrated
    where it looked longest.

    The cleanest statement of the problem is a single variable. "{marker} was
    not recorded" is a fact about a data-collection protocol and carries no
    clinical information whatsoever. Against the {horizon}-day outcome it scores
    {flag_180} -- essentially chance, which is correct. Against the any-horizon
    outcome it scores {flag_any}. The label has absorbed the calendar.

A55. WHERE THE EXTRA POINTS COME FROM
    The any-horizon model scores {gain_direction}: {auc_any_full} against
    {auc_180_full}, a difference of {headline_gain}. Anyone optimising for a
    number on a slide would stop here and ship it.

    {decomposition_verdict}

    The decomposition is the point. Removing bun, urine and glucose removes the
    variables whose MISSINGNESS carries the wave signature -- their values are
    ordinary labs, but their absence is a timestamp. If the any-horizon model
    were learning biology, removing three labs would cost it about what it costs
    the {horizon}-day model. If it were learning the enrolment calendar, it
    should lose more.

    {wave_verdict}

    The general lesson transfers well past this dataset, and it is the one worth
    carrying into any modelling review: an outcome defined by "did we observe
    the event" rather than "did the event happen by time T" imports the
    observation process into the label. Every mechanism that governed who was
    watched, and for how long, becomes learnable signal. The model gets better
    at the benchmark and worse at the job.

    So the honest answer to "why not use every death" is not that it is
    forbidden. It is that the resulting number is not measuring what its name
    says, and the project has the receipts to show it.

A56. THE VERSION THAT ACTUALLY USES EVERY TIMEFRAME
    A survival model. This is not a consolation prize -- it is the analysis the
    binary label was a shortcut for, and it uses strictly more information than
    either logistic model above: {cox_n} patients, {cox_events} deaths, and
    every patient's real exposure time rather than a fixed window.

    Censoring is exactly the machinery for unequal follow-up. A patient last
    seen alive on day 700 contributes 700 days of survived risk and no death,
    rather than being scored as a survivor on a 2,000-day question nobody asked
    them. The differential follow-up that corrupts the binary label is handled
    rather than absorbed.

    Harrell's C is {cindex} against the {horizon}-day model's {auc_180},
    a difference of {c_gap}. {cox_verdict}

    That comparison needs a caveat before anyone quotes the gap, because the
    two numbers are not answering the same question and the naive reading --
    "the survival model is worse" -- is wrong. AUC at a fixed horizon asks one
    binary thing: was this patient dead at day {horizon}. Harrell's C asks the
    model to order every comparable PAIR of patients across the whole of
    follow-up, which means it must also separate a death on day 200 from a
    death on day 1,500. That is a strictly harder discrimination, and it is
    normal for C over years of follow-up to sit below an AUC at a single early
    horizon. The project made the mirror-image error once already, in
    09_parsimony_and_survival.py, by comparing a seven-variable Cox model
    against a twenty-eight-variable elastic net and reading the difference as a
    verdict on survival modelling.

    So the gap is not evidence that the horizon approach is better. It is
    evidence that the two are measuring different things -- which is the reason
    to prefer the survival model where the timing matters clinically, and the
    reason the two numbers should never be put in the same column of a results
    table without this sentence attached.

    One caveat reported rather than buried: {ph_verdict} Proportional hazards is
    an assumption about the RATIO of hazards being constant over time, and in a
    cohort followed for years through an acute illness there is no reason to
    expect it to hold everywhere. It bears on the interpretation of individual
    coefficients more than on the concordance, but a model whose assumption is
    violated should say so.

    WHAT I PREDICTED, AND WHETHER I WAS RIGHT
{scorecard}
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    sep = analysis_frames(group=SEPSIS_LABEL).chf_train

    header(f"SUPPORT2 -- the horizon, and what a bigger label set costs")
    print(f"  {SEPSIS_LABEL} training partition, {len(sep):,} patients.")
    print(f"  The holdout was spent in 12_replication.py and is not read here.")

    acc = horizon_accounting(sep)
    report_accounting(acc)

    t = compare_horizons(sep, acc)
    wave = predict_the_wave(sep)
    d = report_comparison(t, wave, acc)

    cox = fit_cox(sep)
    report_cox(cox, d["auc_180_full"])

    header("FIGURES")
    path = figure_horizon(t, acc, cox, d["auc_180_full"])
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    # ── verdicts, computed rather than asserted ──────────────────────────────
    hurt_more = abs(d["cost_any"]) > abs(d["cost_180"])
    p1 = d["headline_gain"] > 0
    p2 = hurt_more
    p3 = abs(cox["concordance"] - d["auc_180_full"]) < 0.05

    decomposition_verdict = (
        f"It does not survive the decomposition. Removing the three "
        f"protocol-differing labs costs the {HORIZON_DAYS}-day model "
        f"{d['cost_180']:+.4f} AUC and the any-horizon model "
        f"{d['cost_any']:+.4f} -- {abs(d['cost_any']) - abs(d['cost_180']):+.4f} "
        f"more. The apparent advantage of the bigger label set is bound up in "
        f"the variables that encode when a patient was enrolled."
        if hurt_more else
        f"The decomposition does NOT support the contamination story. Removing "
        f"the protocol labs costs the {HORIZON_DAYS}-day model "
        f"{d['cost_180']:+.4f} and the any-horizon model {d['cost_any']:+.4f}, "
        f"so the any-horizon model is not leaning on them more heavily. The "
        f"wave signal is real -- A54 measured it -- but this particular test "
        f"does not show it driving the AUC difference, and the honest reading "
        f"is that the mechanism is more diffuse than a three-variable "
        f"decomposition can isolate.")

    wave_verdict = (
        f"And the enrolment wave is learnable from the clinical variables "
        f"alone. With the protocol labs excluded entirely, "
        f"{wave['n_predictors']} ordinary clinical predictors identify which "
        f"wave a patient came from at AUC {wave['auc']:.3f}. "
        + ("That is close to chance, so the leakage route really is the three "
           "labs rather than the whole feature set -- dropping them is an "
           "effective remedy."
           if wave["auc"] < 0.60 else
           "That is well above chance, which is the uncomfortable version: the "
           "calendar is diffused across the ordinary clinical variables too, so "
           "dropping three columns does not fully remove it. Only a horizon "
           "that every patient was observed through can."))

    cox_verdict = (
        "The two land in the same place, which is the expected result -- both "
        "are concordance measures over largely the same information, and the "
        "survival model's advantage is in what it does NOT corrupt rather than "
        "in a higher score."
        if p3 else
        "They differ by more than the 0.05 the prediction allowed, and the "
        "expectation behind it was too loose: it treated 'both are concordance "
        "measures' as though that made them commensurable, which it does not.")

    ph_verdict = (
        f"{cox['n_violating']} of {cox['n_terms']} terms violate proportional "
        f"hazards at p < 0.05."
        if cox["n_violating"] else
        "no term violates proportional hazards at p < 0.05.")

    marks = [("the any-horizon model would score higher", p1),
             ("removing the protocol labs would cost it more", p2),
             ("Harrell's C would land near the 180-day AUC", p3)]
    scorecard = "\n".join(
        f"      {('RIGHT' if ok else 'WRONG'):<9}{txt}" for txt, ok in marks)
    n_right = sum(ok for _, ok in marks)
    scorecard += (f"\n\n    {n_right} of 3. "
                  + ("The expectations were recorded before the run and are "
                     "reported whether or not they held, which is the only way "
                     "a stated prediction is worth anything."
                     if n_right == 3 else
                     "The misses are left in rather than quietly rewritten. A "
                     "prediction that is only reported when it succeeds is not "
                     "a prediction."))

    facts = Facts(
        n_180=f"{acc['n_180']:,}", n_any=f"{acc['n_any']:,}",
        extra_pct=f"{100*(acc['n_any']-acc['n_180'])/acc['n_180']:.0f}",
        fu_ratio=f"{acc['fu_ratio']:.2f}", horizon=str(HORIZON_DAYS),
        gap_180=f"{acc['gap_180']:.1f}", gap_any=f"{acc['gap_any']:.1f}",
        gap_ratio=f"{acc['gap_any']/acc['gap_180']:.1f}",
        marker=WAVE_MARKER,
        flag_180=f"{acc['flag_auc_180']:.3f}", flag_any=f"{acc['flag_auc_any']:.3f}",
        gain_direction="higher" if p1 else "lower",
        auc_any_full=f"{d['auc_any_full']:.4f}",
        auc_180_full=f"{d['auc_180_full']:.4f}",
        headline_gain=f"{d['headline_gain']:+.4f}",
        decomposition_verdict=decomposition_verdict,
        wave_verdict=wave_verdict,
        cox_n=f"{cox['n']:,}", cox_events=f"{cox['events']:,}",
        cindex=f"{cox['concordance']:.3f}", auc_180=f"{d['auc_180_full']:.3f}",
        c_gap=f"{cox['concordance'] - d['auc_180_full']:+.3f}",
        cox_verdict=cox_verdict, ph_verdict=ph_verdict, scorecard=scorecard,
    )
    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "13_horizon.txt")
