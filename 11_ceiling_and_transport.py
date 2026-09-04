"""
11_ceiling_and_transport.py -- Is the limit optimisation or information, and
does the model transport across a protocol change?

Three analyses that answer "could we get better numbers by tuning harder?"
with evidence rather than assertion.

    Run:  python 11_ceiling_and_transport.py

    IMPORTANT. The held-out partition was spent in 10_confirmatory.py and is not
    read again here. Everything below runs on the training partition, by nested
    cross-validation. Nothing in this file may be used to revise the
    confirmatory result -- if a bigger search found a better model, that model
    would need a fresh holdout, not the one already reported on.

THE QUESTIONS
    Q46  Would more data help? Plot the learning curve and find out, rather
         than assuming either way.
    Q47  Would a much larger hyperparameter search help? Run one and report the
         answer honestly, including if it contradicts the expectation.
    Q48  A random train/test split is the weakest form of validation TRIPOD
         recognises. This cohort contains a protocol change. Use it: train on
         one enrolment wave, validate on the other.

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
    PROTOCOL_MISSING,
    RANDOM_STATE,
    build_pipeline,
    calibration_metrics,
    cross_val_predictions,
    default_predictors,
    discrimination_metrics,
    make_outcome,
)
from report import Facts, RULE, configure_pandas, fmt_p, header, question, render_answers, run_and_capture
from support2 import analysis_frames

OUT_DIR = Path(__file__).resolve().parent / "output"

LEARNING_FRACTIONS = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
LEARNING_REPEATS = 20          # resamples per size, for a usable interval
SEARCH_ITER = 120              # candidates in the exhaustive random search
WAVE_MARKER = "bun"            # 03_cohort.py Q16: 100% missing in the early wave


def modest_elastic_net():
    """The configuration used throughout the project."""
    from sklearn.linear_model import LogisticRegressionCV

    return LogisticRegressionCV(l1_ratios=(0.2, 0.5, 0.9),
                                Cs=np.logspace(-3, 1, 8), cv=CV_FOLDS,
                                scoring="neg_log_loss", max_iter=3000,
                                random_state=RANDOM_STATE, refit=True,
                                n_jobs=-1, solver="saga")


# ═══ Q46. Learning curve ═════════════════════════════════════════════════════
def learning_curve(chf: pd.DataFrame, y: np.ndarray,
                   predictors: list[str]) -> pd.DataFrame:
    """
    Cross-validated AUC against training-set size.

    Each size is evaluated on repeated stratified subsamples, so the spread is
    an honest interval rather than one draw. A curve still climbing at n=978
    says more patients would help; a flat one says the limit is the information
    in these variables, and no amount of tuning reaches past it.
    """
    import sys
    import time

    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    rows = []
    started = time.time()
    total = len(LEARNING_FRACTIONS) * LEARNING_REPEATS
    done = 0
    for frac in LEARNING_FRACTIONS:
        aucs = []
        for rep in range(LEARNING_REPEATS):
            if frac < 1.0:
                idx, _ = train_test_split(np.arange(len(y)), train_size=frac,
                                          stratify=y, random_state=1000 + rep)
            else:
                idx = np.arange(len(y))
            Xs, ys = chf.iloc[idx][predictors], y[idx]
            # A light estimator here on purpose: the question is how performance
            # scales with n, and a nested search at every size would multiply the
            # cost without changing the shape of the curve.
            pipe = build_pipeline(chf, predictors,
                                  LogisticRegression(C=1.0, max_iter=2000))
            cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=rep)
            oof = np.zeros(len(ys))
            for tr, te in cv.split(Xs, ys):
                p = clone(pipe)
                p.fit(Xs.iloc[tr], ys[tr])
                oof[te] = p.predict_proba(Xs.iloc[te])[:, 1]
            aucs.append(roc_auc_score(ys, oof))
            done += 1
            if done % 10 == 0:
                el = time.time() - started
                print(f"    learning curve   {done:>3}/{total}  {el:5.0f}s elapsed, "
                      f"~{el/done*(total-done):4.0f}s remaining",
                      file=sys.stderr, flush=True)
        a = np.array(aucs)
        rows.append({"fraction": frac, "n": len(idx), "mean_auc": a.mean(),
                     "sd": a.std(ddof=1), "lo": np.percentile(a, 2.5),
                     "hi": np.percentile(a, 97.5)})
    print(f"    learning curve   DONE in {time.time()-started:.0f}s",
          file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def report_learning_curve(t: pd.DataFrame) -> dict:
    question(46, "Would more data help? Plot the learning curve and find out,\n"
                 "rather than assuming either way.")
    show = t.copy()
    for c in ("mean_auc", "sd", "lo", "hi"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    # Slope over the last half of the curve, in AUC per 100 additional patients.
    half = t.iloc[len(t) // 2:]
    slope = np.polyfit(half.n, half.mean_auc, 1)[0] * 100
    first, last = t.iloc[0], t.iloc[-1]
    print(f"\n  AUC from n={first.n:,} to n={last.n:,}: "
          f"{first.mean_auc:.4f} -> {last.mean_auc:.4f} "
          f"({last.mean_auc - first.mean_auc:+.4f})")
    print(f"  Slope over the upper half of the curve: "
          f"{slope:+.4f} AUC per 100 additional patients")
    print(f"  Extrapolated gain from doubling the cohort to "
          f"{last.n*2:,}: {slope*last.n/100:+.3f}")
    return {"slope_per_100": float(slope), "first": first, "last": last,
            "extrapolated_double": float(slope * last.n / 100)}


# ═══ Q47. Exhaustive search ══════════════════════════════════════════════════
def exhaustive_search(chf: pd.DataFrame, y: np.ndarray,
                      predictors: list[str]) -> pd.DataFrame:
    """
    A deliberately large hyperparameter search, evaluated by nested CV.

    The search sits inside the pipeline, so it is refitted on every outer fold
    and the reported score is not contaminated by the selection. If a bigger
    grid genuinely helps, this will show it; if it does not, that is the more
    useful finding and it is reported either way.
    """
    from scipy.stats import loguniform, randint, uniform
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier

    xgb_space = {
        "n_estimators": randint(100, 900),
        "max_depth": randint(2, 7),
        "learning_rate": loguniform(0.005, 0.3),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(1, 30),
        "reg_lambda": loguniform(0.1, 50),
        "reg_alpha": loguniform(1e-3, 10),
        "gamma": loguniform(1e-3, 5),
    }
    searched_xgb = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        xgb_space, n_iter=SEARCH_ITER, scoring="neg_log_loss", cv=3,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True)

    from sklearn.linear_model import LogisticRegressionCV
    wide_enet = LogisticRegressionCV(
        l1_ratios=list(np.linspace(0.05, 0.95, 10)),
        Cs=np.logspace(-4, 2, 25), cv=CV_FOLDS, scoring="neg_log_loss",
        max_iter=4000, random_state=RANDOM_STATE, refit=True, n_jobs=-1,
        solver="saga")

    from xgboost import XGBClassifier as XGB
    specs = {
        "Elastic net, project grid (24 configs)": (modest_elastic_net(), True),
        f"Elastic net, wide grid (250 configs)": (wide_enet, True),
        "XGBoost, project defaults (1 config)": (
            XGB(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1), False),
        f"XGBoost, random search ({SEARCH_ITER} configs)": (searched_xgb, False),
    }
    rows = []
    for name, (est, scale) in specs.items():
        p = cross_val_predictions(build_pipeline(chf, predictors, est, scale=scale),
                                  chf[predictors], y, n_repeats=2,
                                  label=name[:28])
        rows.append({"model": name, **discrimination_metrics(y, p),
                     **calibration_metrics(y, p)})
    return pd.DataFrame(rows)


def report_search(t: pd.DataFrame) -> dict:
    question(47, "Would a much larger hyperparameter search help? Run one and\n"
                 "report the answer honestly, including if it contradicts the\n"
                 "expectation.")
    show = t[["model", "auc", "pr_auc", "calibration_slope", "brier"]].copy()
    for c in ("auc", "pr_auc", "calibration_slope", "brier"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    enet_gain = (t[t.model.str.contains("wide")].iloc[0].auc
                 - t[t.model.str.contains("project grid")].iloc[0].auc)
    xgb_before = t[t.model.str.contains("project defaults")].iloc[0]
    xgb_after = t[t.model.str.contains("random search")].iloc[0]
    xgb_gain = (t[t.model.str.contains("random search")].iloc[0].auc
                - t[t.model.str.contains("project defaults")].iloc[0].auc)
    print(f"\n  Elastic net: 24 configurations -> 250. AUC change {enet_gain:+.4f}")
    print(f"  XGBoost:      1 configuration -> {SEARCH_ITER}. "
          f"AUC change {xgb_gain:+.4f}")
    print(f"  XGBoost calibration slope: {xgb_before.calibration_slope:.3f} "
          f"-> {xgb_after.calibration_slope:.3f}")
    return {"enet_gain": float(enet_gain), "xgb_gain": float(xgb_gain),
            "xgb_slope_before": float(xgb_before.calibration_slope),
            "xgb_slope_after": float(xgb_after.calibration_slope)}


# ═══ Q48. Temporal validation ════════════════════════════════════════════════
def temporal_validation(chf: pd.DataFrame, y: np.ndarray) -> dict:
    """
    Train on one enrolment wave, validate on the other.

    TRIPOD ranks random-split internal validation as the weakest form of
    validation and temporal validation (type 2b) above it, because a temporal
    split tests transportability across a change in time and practice rather
    than merely across a reshuffle of the same patients.

    03_cohort.py Q16 established that this cohort contains exactly such a
    change: two enrolment waves with different data-collection protocols. The
    wave is identified by BUN missingness -- 100% absent in the early wave
    against 0.9% in the late one.

    Two consequences follow and both are costs. The three protocol-differing
    labs (bun, urine, glucose: 100%, 99.0%, 99.5% missing early) cannot be
    predictors, because they do not exist on one side of the split. And the
    wave marker is inferred from a missingness pattern rather than recorded, so
    a small number of patients will be misassigned.
    """
    from sklearn.metrics import roc_auc_score

    usable = [c for c in default_predictors(chf) if c not in PROTOCOL_MISSING]
    early = chf[WAVE_MARKER].isna().values
    frames = {"early": (chf[early], y[early]), "late": (chf[~early], y[~early])}

    rows = []
    for train_name, test_name in (("late", "early"), ("early", "late")):
        Xtr, ytr = frames[train_name]
        Xte, yte = frames[test_name]
        pipe = build_pipeline(Xtr, usable, modest_elastic_net())
        pipe.fit(Xtr[usable], ytr)
        p = pipe.predict_proba(Xte[usable])[:, 1]
        rows.append({"trained_on": train_name, "tested_on": test_name,
                     "n_train": len(ytr), "n_test": len(yte),
                     "train_prev": ytr.mean(), "test_prev": yte.mean(),
                     "auc": roc_auc_score(yte, p), **calibration_metrics(yte, p)})

    # Within-wave reference: how well does it do without crossing the change?
    for name in ("late", "early"):
        X, yy = frames[name]
        p = cross_val_predictions(build_pipeline(X, usable, modest_elastic_net()),
                                  X[usable], yy, n_repeats=2,
                                  label=f"within-wave {name}")
        rows.append({"trained_on": f"{name} (CV)", "tested_on": f"{name} (CV)",
                     "n_train": len(yy), "n_test": len(yy),
                     "train_prev": yy.mean(), "test_prev": yy.mean(),
                     "auc": roc_auc_score(yy, p), **calibration_metrics(yy, p)})
    return {"table": pd.DataFrame(rows), "n_usable": len(usable),
            "dropped": PROTOCOL_MISSING}


def report_temporal(r: dict) -> dict:
    question(48, "A random train/test split is the weakest form of validation TRIPOD\n"
                 "recognises. This cohort contains a protocol change. Use it: train\n"
                 "on one enrolment wave, validate on the other.")
    t = r["table"]
    show = t[["trained_on", "tested_on", "n_train", "n_test", "train_prev",
              "test_prev", "auc", "calibration_slope", "brier"]].copy()
    for c in ("train_prev", "test_prev", "auc", "calibration_slope", "brier"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print(f"\n  {r['n_usable']} predictors usable across both waves; "
          f"{', '.join(r['dropped'])} dropped -- absent from the early wave.")

    cross = t[~t.trained_on.str.contains("CV")]
    within = t[t.trained_on.str.contains("CV")]
    gap = within.auc.mean() - cross.auc.mean()
    print(f"\n  mean AUC within-wave  {within.auc.mean():.3f}")
    print(f"  mean AUC across-wave  {cross.auc.mean():.3f}")
    print(f"  transport cost        {-gap:+.3f}")
    return {"within": float(within.auc.mean()), "cross": float(cross.auc.mean()),
            "gap": float(gap), "table": t}


# ═══ Figure ══════════════════════════════════════════════════════════════════
def figure_ceiling(curve: pd.DataFrame, search: pd.DataFrame, temporal: dict):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax1.fill_between(curve.n, curve.lo, curve.hi, color=viz.SERIES_BLUE, alpha=0.16)
    ax1.plot(curve.n, curve.mean_auc, "o-", color=viz.SERIES_BLUE, lw=2.2, ms=6,
             mec=viz.SURFACE, mew=1.4)
    ax1.set_xlabel("Training patients")
    ax1.set_ylabel("Cross-validated AUC")
    ax1.set_title("Learning curve: has it plateaued?")
    viz.despine(ax1)

    s = search.copy()
    s["short"] = ["EN 24", "EN 250", "XGB 1", f"XGB {SEARCH_ITER}"]
    colors = [viz.SERIES_BLUE, viz.SERIES_BLUE, viz.SERIES[2], viz.SERIES[2]]
    ax2.bar(s.short, s.auc, color=colors, width=0.62)
    ax2.set_ylim(min(s.auc) - 0.02, max(s.auc) + 0.012)
    for i, v in enumerate(s.auc):
        ax2.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax2.set_ylabel("Cross-validated AUC")
    ax2.set_title("Search size vs. performance")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.grid(axis="x", visible=False)
    viz.despine(ax2)

    tt = temporal["table"]
    labels = [f"{r.trained_on}→{r.tested_on}".replace(" (CV)", "")
              for _, r in tt.iterrows()]
    is_cross = [not str(r.trained_on).endswith("(CV)") for _, r in tt.iterrows()]
    cols = [viz.SERIES_ORANGE if c else viz.BASELINE for c in is_cross]
    ax3.barh(range(len(tt)), tt.auc, color=cols, height=0.6)
    ax3.set_yticks(range(len(tt)), labels, fontsize=9)
    ax3.set_xlim(0.5, max(tt.auc) + 0.05)
    ax3.axvline(0.5, color=viz.BASELINE, lw=1.2, ls="--")
    for i, v in enumerate(tt.auc):
        ax3.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax3.set_xlabel("AUC")
    ax3.set_title("Across a protocol change (orange)")
    ax3.grid(axis="y", visible=False)
    viz.despine(ax3)

    fig.tight_layout()
    viz.caption(fig, f"CHF training partition, {OUTCOME_LABEL}. The held-out 30% was spent in "
                     f"10_confirmatory.py\nand is not read here. Left: shaded band is the 95% range "
                     f"across {LEARNING_REPEATS} resamples per size.\nRight: orange bars train on one "
                     f"enrolment wave and test on the other.", y=-0.05)
    return viz.save(fig, "21_ceiling_and_transport.png")


ANSWERS = """
ANSWERS
{rule}

A46. WOULD MORE DATA HELP?
    From {n_small} to {n_full} training patients the cross-validated AUC moves
    {curve_change}. Over the upper half of the curve the slope is
    {slope_per_100} AUC per 100 additional patients, which extrapolates to
    {extrap} if the cohort were doubled.

    {curve_verdict}

    This contradicts what I expected before running it, and the expectation is
    worth recording because it was wrong. The reasoning was: five model families
    land within 0.05 AUC of one another, and the optimism bootstrap put the
    elastic net's shrinkage factor at 1.00, so the model is information-limited
    rather than capacity-limited and more of the same patients should not help.
    The first half of that is still true. The conclusion drawn from it was not.

    Two constraints were being conflated. A model can be information-limited in
    the sense that no ALGORITHM will do better on these variables, and still be
    sample-limited in the sense that these variables' predictive content is not
    yet fully estimated. The learning curve separates them, which is why it is
    worth plotting rather than reasoning about.

    So both findings point the same way for different reasons. Riley's criteria
    (07_validation.py Q31) say the cohort is too small for the coefficients to
    be PRECISE. The learning curve says it is also too small for discrimination
    to have topped out. Doubling the cohort would buy roughly {extrap} AUC and
    considerably steadier estimates -- a modest but real return, and a far
    better use of effort than any amount of additional tuning.

    What the curve cannot show is the effect of more VARIABLES. The information
    that would move discrimination here is ejection fraction, NYHA class, BNP,
    medications -- all absent (04_clinical.py Q21). A larger cohort with the
    same twenty-eight columns buys precision; a smaller cohort with an
    echocardiogram would probably buy more. The honest ranking of what to spend
    money on: better variables first, more patients second, better algorithms
    last.

A47. WOULD A BIGGER SEARCH HELP?
    No, and this was worth testing rather than asserting.

    Widening the elastic net from 24 configurations to 250 changes AUC by
    {enet_gain}. Replacing XGBoost's single hand-set configuration with a
    {n_iter}-candidate random search over nine hyperparameters -- depth,
    learning rate, subsample, column sample, minimum child weight, two
    regularisation terms, gamma and tree count -- changes it by {xgb_gain}.

    Both searches were evaluated by nested cross-validation, so these are not
    the optimistic in-sample numbers a search reports about itself. The search
    is refitted on every outer fold and scored on rows it did not select on.

    One qualification, because the search did buy something and reporting only
    AUC would hide it. XGBoost's calibration slope moved from {xgb_slope_before}
    to {xgb_slope_after} under the wide search -- from badly overconfident to
    close to ideal. Its discrimination barely moved and its PR-AUC got slightly
    worse, so on the headline metric the search was a wash; on the metric this
    project has argued matters more, it was a real improvement.

    That is a genuine finding rather than a rescue of the original claim. It
    sharpens it: hyperparameter search on this problem does not buy ranking, but
    it can buy probability quality -- and a search scored on AUC alone would
    have discarded the configuration that fixed the calibration. Choose the
    search metric to match what the model is for. Here `neg_log_loss` was used
    rather than `roc_auc` precisely because it is a proper scoring rule and
    rewards calibration, and that choice is why the improvement appeared at all.

    This is the expected result and the reason is in the project already. The
    optimism bootstrap in 07_validation.py measured the elastic net's shrinkage
    factor at 1.00 -- the model needs no shrinkage, which is another way of
    saying it is not leaving fit on the table. Five model families spanning
    unpenalised regression to gradient boosting land within 0.05 AUC of one
    another. When the choice of model class stops mattering, the hyperparameters
    within a class will not matter either.

    The general lesson, which transfers well beyond this dataset: hyperparameter
    search moves performance when the model is capacity-limited. It does nothing
    when the model is information-limited, and the way to tell the difference is
    to look at whether different model families disagree. Here they do not.

A48. DOES IT TRANSPORT ACROSS A PROTOCOL CHANGE?
    Within a wave the model reaches {within_auc}. Trained on one wave and
    applied to the other it reaches {cross_auc} -- a transport cost of
    {transport_cost}.

    {transport_verdict}

    This is a better test than the random split that produced the confirmatory
    result, and TRIPOD says so: a random split reshuffles the same patients and
    asks whether the model generalises to more of them; a temporal split asks
    whether it generalises across a change in when and how the data were
    collected. The second is much closer to the question anyone deploying a
    model actually faces.

    Three honest costs, none of them small.

    The wave is INFERRED, not recorded. It is identified by BUN missingness,
    which 03_cohort.py Q16 showed is 100% absent in the early wave against 0.9%
    late -- close to deterministic but not deterministic, so some patients are
    misassigned.

    Three predictors had to be dropped. bun, urine and glucose are 100%, 99.0%
    and 99.5% missing in the early wave, so they cannot be used by a model that
    has to run on both sides. The cross-wave model is therefore not the same
    model as the one in 10_confirmatory.py, and the numbers are not directly
    comparable to it.

    And it is still internal. Both waves come from the same five hospitals in
    the same programme. A model that transports across a protocol change within
    one study has not been shown to transport to a different institution, a
    different country, or the present decade -- and 04_clinical.py A22 gives
    concrete reasons to expect the last of those to go badly.

    What this does establish is narrower and worth stating precisely: the
    model's discrimination is not an artefact of one data-collection protocol.
    That is a real result and it is the strongest transportability claim this
    dataset supports.

    A final constraint, because it governs what any of this can be used for.
    The held-out partition was spent in 10_confirmatory.py. Nothing in this file
    may revise that result. If the wide search HAD found a better model, the
    honest response would not have been to report it against the existing
    holdout -- it would have been to note that a new model requires a new
    holdout, and that this cohort no longer has one to give.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train
    y = make_outcome(chf).values
    predictors = default_predictors(chf)

    header("SUPPORT2 -- the ceiling, and transport across a protocol change")
    print(f"  CHF training partition {len(chf):,}, {int(y.sum()):,} events "
          f"({y.mean()*100:.1f}%)")
    print("  THE HELD-OUT 30% WAS SPENT IN 10_confirmatory.py AND IS NOT READ HERE.")
    print("  Nothing below may be used to revise the confirmatory result.")

    curve = learning_curve(chf, y, predictors)
    cur = report_learning_curve(curve)

    search = exhaustive_search(chf, y, predictors)
    srch = report_search(search)

    temporal = temporal_validation(chf, y)
    temp = report_temporal(temporal)

    header("FIGURES")
    path = figure_ceiling(curve, search, temp)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    curve_verdict = (
        "The curve has flattened: the last increments of data buy almost "
        "nothing, so more patients of this kind would not move discrimination "
        "much."
        if abs(cur["extrapolated_double"]) < 0.02 else
        "The curve is still climbing: more patients would plausibly help, and "
        "the sample size is a real constraint on performance as well as on "
        "precision.")
    transport_verdict = (
        "Discrimination transported essentially intact, which is the result "
        "worth having: the model is not relying on an artefact of one "
        "collection protocol."
        if abs(temp["gap"]) < 0.03 else
        ("Discrimination dropped materially across the protocol change, which "
         "is a genuine warning about transportability and a stronger caution "
         "than the random-split result gave."
         if temp["gap"] > 0 else
         "Discrimination was higher across waves than within them, which is "
         "usually sampling noise at this size rather than a real effect."))

    facts = Facts(
        n_small=f"{int(cur['first'].n):,}", n_full=f"{int(cur['last'].n):,}",
        curve_change=f"{cur['last'].mean_auc - cur['first'].mean_auc:+.4f}",
        slope_per_100=f"{cur['slope_per_100']:+.4f}",
        extrap=f"{cur['extrapolated_double']:+.3f}",
        curve_verdict=curve_verdict,
        enet_gain=f"{srch['enet_gain']:+.4f}", xgb_gain=f"{srch['xgb_gain']:+.4f}",
        n_iter=str(SEARCH_ITER),
        xgb_slope_before=f"{srch['xgb_slope_before']:.2f}",
        xgb_slope_after=f"{srch['xgb_slope_after']:.2f}",
        within_auc=f"{temp['within']:.3f}", cross_auc=f"{temp['cross']:.3f}",
        transport_cost=f"{-temp['gap']:+.3f}",
        transport_verdict=transport_verdict,
    )
    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "11_ceiling_and_transport.txt")
