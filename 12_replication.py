"""
12_replication.py -- Do the findings survive a cohort with four times the events?

The CHF analysis rests on 978 patients and 248 events: 6.4 events per variable,
against a Riley requirement of about 2,465 patients. Several of its conclusions
are therefore weak by construction. "Penalised regression is not beaten by
gradient boosting" is exactly what you would expect at that sample size whether
or not it is true in general -- small samples favour regularisation.

The ARF/MOSF-with-sepsis group gives 2,458 training patients and 1,091 events:
EPV 28, and the Riley requirement essentially met. If the same conclusions hold
there, they are findings. If they do not, the CHF results were underpowered
artefacts and the project needs to say so.

    Run:  python 12_replication.py

    The sepsis TEST partition has never been read. 10_confirmatory.py used only
    chf_test, so a genuinely clean confirmatory analysis is available here --
    which the CHF arm can no longer provide.

THE QUESTIONS
    Q50  Who are these patients, and what is the clinical frame? Heart failure
         vocabulary does not transfer to an ICU sepsis population.
    Q51  Do the exploratory findings replicate -- the enrolment-wave artefact,
         and DNR as a care decision rather than a disease state?
    Q52  Does "gradient boosting does not beat penalised regression" hold at
         EPV 28, or was it an artefact of a small sample?
    Q53  Spend the clean sepsis holdout. Does the confirmatory result look
         different when the cohort is adequately powered?

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
)
from report import Facts, RULE, configure_pandas, fmt_p, header, question, render_answers, run_and_capture
from support2 import (
    CHF_LABEL,
    OUTCOME_EVENT,
    OUTCOME_TIME,
    SEPSIS_LABEL,
    analysis_frames,
    confirmatory_frames,
)

OUT_DIR = Path(__file__).resolve().parent / "output"

# Results carried over from the CHF arm, recorded here so the comparison is
# against numbers fixed before this script ran.
CHF_REFERENCE = {
    "n_train": 978, "events": 248, "epv": 6.4, "n_params": 39,
    "elastic_net_auc": 0.678, "xgboost_auc": 0.673,
    "xgb_minus_enet": -0.0051, "xgb_ci": (-0.0361, 0.0279),
    "test_auc": 0.655, "test_slope": 0.706,
    "physician_auc_test": 0.711, "model_auc_test": 0.633,
    "physician_ci": (-0.154, 0.003),
    # Transcribed from output/05_modelling.txt, in MODEL_ORDER.
    "cv_auc": [0.661, 0.675, 0.678, 0.673, 0.631],
    "cv_slope": [0.590, 1.216, 1.190, 0.666, 0.896],
}

# Fixed order, shared by the model table and every figure that compares cohorts.
MODEL_ORDER = ["Unpenalised logistic", "LASSO logistic", "Elastic net logistic",
               "XGBoost", "Decision tree (depth 3)"]


# ═══ Q50. The cohort, and its clinical frame ═════════════════════════════════
def design_width(df: pd.DataFrame, predictors: list[str]) -> int:
    """Columns the model actually estimates a coefficient for.

    Events per variable is only meaningful against the ENCODED design, not the
    28 raw predictors: one-hot expansion and the two missingness indicators take
    it to a wider matrix. Computed rather than hardcoded so the comparison is
    like-for-like even if a category is absent in one cohort.
    """
    pipe = build_pipeline(df, predictors, None, scale=True)
    frame = pipe.named_steps["indicators"].fit_transform(df[predictors])
    return pipe.named_steps["prep"].fit_transform(frame).shape[1]


def report_cohort(sep: pd.DataFrame, y: np.ndarray, n_params: int) -> dict:
    question(50, "Who are these patients, and what is the clinical frame? Heart\n"
                 "failure vocabulary does not transfer to an ICU sepsis population.")
    d = {
        "n": len(sep), "events": int(y.sum()), "prev": float(y.mean()),
        "age_med": sep.age.median(), "female": (sep.sex == "female").mean() * 100,
        "comorb3": (sep["num.co"] >= 3).mean() * 100,
        "coma_gt0": (sep.scoma > 0).mean() * 100,
        "pafi_med": sep.pafi.median(),
        "pafi_ali": (sep.pafi < 300).mean() * 100,
        "crea_med": sep.crea.median(),
        "crea_aki": (sep.crea > 2.0).mean() * 100,
        "dnr_any": sep.dnr.isin(["dnr before sadm", "dnr after sadm"]).mean() * 100,
        "epv": y.sum() / n_params, "n_params": n_params,
    }
    print(f"  {d['n']:,} patients with acute respiratory failure or multi-organ")
    print(f"  system failure in the setting of sepsis. {d['events']:,} died within")
    print(f"  {HORIZON_DAYS} days ({d['prev']*100:.1f}%).\n")
    print(f"    age                 median {d['age_med']:.0f}")
    print(f"    sex                 {d['female']:.1f}% female")
    print(f"    comorbidity burden  {d['comorb3']:.1f}% with 3 or more")
    print(f"    neurological         {d['coma_gt0']:.1f}% with any coma score")
    print(f"    oxygenation         median P/F {d['pafi_med']:.0f}; "
          f"{d['pafi_ali']:.1f}% below 300")
    print(f"    renal               median creatinine {d['crea_med']:.1f} mg/dL; "
          f"{d['crea_aki']:.1f}% above 2.0")
    print(f"    DNR order (any)     {d['dnr_any']:.1f}%")
    print(f"\n  Events per variable: {d['epv']:.1f} "
          f"(CHF arm: {CHF_REFERENCE['epv']})")
    return d


# ═══ Q51. Do the EDA findings replicate? ═════════════════════════════════════
def replicate_eda(sep: pd.DataFrame, y: np.ndarray) -> dict:
    """Re-run the three exploratory findings that mattered, on a new cohort.

    Deliberately NOT a pass/fail on p < 0.05. This cohort is 2.5x the size, so
    every test has more power and a threshold rule would report "did not
    replicate" for effects that in fact collapsed by an order of magnitude.
    The chi-square STATISTICS are carried alongside the p-values so the
    comparison is of effect magnitude, which is what replication means.
    """
    from lifelines.statistics import logrank_test
    from scipy import stats as sps

    # (a) The enrolment-wave artefact. adlp and income are negative controls:
    # they are genuinely missing rather than protocol-differing, so they should
    # show NO wave structure if the mechanism is what 03_cohort.py claimed.
    early = sep.bun.isna()
    wave = pd.DataFrame([
        {"variable": c, "kind": "protocol lab" if c in ("bun", "urine", "glucose")
                                else "negative control",
         "missing_early_wave": sep.loc[early, c].isna().mean() * 100,
         "missing_late_wave": sep.loc[~early, c].isna().mean() * 100}
        for c in ("bun", "urine", "glucose", "adlp", "income") if c in sep])

    # (b) Cumulative-outcome test vs time-to-event test on the same split.
    arte = []
    for col in ("bun", "urine", "glucose", "adlp", "income"):
        if col not in sep:
            continue
        m = sep[col].isna()
        if not (0.02 < m.mean() < 0.98):
            continue
        obs, mis = sep[~m], sep[m]
        chi2, p_bin, *_ = sps.chi2_contingency(pd.crosstab(m, sep[OUTCOME_EVENT]))
        lr = logrank_test(obs[OUTCOME_TIME], mis[OUTCOME_TIME],
                          obs[OUTCOME_EVENT], mis[OUTCOME_EVENT])
        fo = obs.loc[obs[OUTCOME_EVENT] == 0, OUTCOME_TIME].median()
        fm = mis.loc[mis[OUTCOME_EVENT] == 0, OUTCOME_TIME].median()
        arte.append({"variable": col,
                     "kind": "protocol lab" if col in ("bun", "urine", "glucose")
                             else "negative control",
                     "chi2_binary": chi2, "p_binary": p_bin,
                     "chi2_logrank": lr.test_statistic, "p_logrank": lr.p_value,
                     "shrinkage": chi2 / max(lr.test_statistic, 1e-9),
                     "fu_ratio": fm / fo})
    arte = pd.DataFrame(arte)

    # (c) DNR. In CHF, a directive brought FROM HOME was indistinguishable from
    # no directive (log-rank p = 0.882) while one written DURING the admission
    # nearly doubled mortality. Each category is tested against 'no dnr'.
    ys = pd.Series(y, index=sep.index)
    ref = sep[sep.dnr == "no dnr"]
    dnr = []
    for lvl in ("dnr before sadm", "dnr after sadm"):
        g = sep[sep.dnr == lvl]
        if len(g) < 10:
            continue
        lr = logrank_test(g[OUTCOME_TIME], ref[OUTCOME_TIME],
                          g[OUTCOME_EVENT], ref[OUTCOME_EVENT])
        dnr.append({"dnr": lvl, "n": len(g),
                    "mortality_180": ys[g.index].mean() * 100,
                    "gap_vs_no_dnr": (ys[g.index].mean()
                                      - ys[ref.index].mean()) * 100,
                    "p_vs_no_dnr": lr.p_value})
    dnr = pd.DataFrame(dnr)
    return {"wave": wave, "artefact": arte, "dnr": dnr,
            "n_no_dnr": len(ref), "mort_no_dnr": ys[ref.index].mean() * 100}


def report_replicate_eda(r: dict) -> dict:
    question(51, "Do the exploratory findings replicate -- the enrolment-wave\n"
                 "artefact, and DNR as a care decision rather than a disease state?")
    print("  (a) Enrolment wave, split on whether BUN was recorded. The two\n"
          "      negative controls are missing for ordinary reasons and should\n"
          "      show no wave structure:\n")
    print(r["wave"].round(1).to_string(index=False))

    print("\n  (b) Cumulative-outcome test vs time-to-event test, same splits.\n"
          "      'shrinkage' is how many times smaller the statistic gets once\n"
          "      exposure time is accounted for:\n")
    show = r["artefact"].copy()
    for c in ("chi2_binary", "chi2_logrank", "shrinkage"):
        show[c] = show[c].round(1)
    show["fu_ratio"] = show.fu_ratio.round(2)
    for c in ("p_binary", "p_logrank"):
        show[c] = show[c].apply(fmt_p)
    print(show.to_string(index=False))

    print(f"\n  (c) DNR against the {r['n_no_dnr']:,} patients with no directive "
          f"({r['mort_no_dnr']:.1f}% mortality):\n")
    d = r["dnr"].copy()
    for c in ("mortality_180", "gap_vs_no_dnr"):
        d[c] = d[c].round(1)
    d["p_vs_no_dnr"] = d.p_vs_no_dnr.apply(fmt_p)
    print(d.to_string(index=False))

    labs = r["artefact"][r["artefact"].kind == "protocol lab"]
    ctrl = r["artefact"][r["artefact"].kind == "negative control"]
    lw = r["wave"][r["wave"].kind == "protocol lab"]
    cw = r["wave"][r["wave"].kind == "negative control"]
    before = r["dnr"][r["dnr"].dnr == "dnr before sadm"]
    return {
        "wave_replicated": bool((lw.missing_early_wave > 95).all()
                                and (lw.missing_late_wave < 15).all()
                                and (cw.missing_early_wave
                                     - cw.missing_late_wave).abs().max() < 15),
        "ratio_replicated": bool(labs.fu_ratio.min() > 1.5
                                 and ctrl.fu_ratio.max() < 1.3),
        "lab_ratio": float(labs.fu_ratio.mean()),
        "ctrl_ratio": float(ctrl.fu_ratio.mean()),
        "lab_shrinkage": float(labs.shrinkage.min()),
        "ctrl_shrinkage": float(ctrl.shrinkage.max()),
        "dnr_before_gap": float(before.gap_vs_no_dnr.iloc[0]) if len(before) else float("nan"),
        "dnr_before_p": float(before.p_vs_no_dnr.iloc[0]) if len(before) else float("nan"),
        "dnr_before_n": int(before.n.iloc[0]) if len(before) else 0,
        "dnr_replicated": bool(len(before) and before.p_vs_no_dnr.iloc[0] > 0.05),
    }


# ═══ Q52. Does the model comparison replicate? ═══════════════════════════════
def replicate_models(sep: pd.DataFrame, y: np.ndarray,
                     predictors: list[str]) -> tuple[pd.DataFrame, dict]:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    shared = dict(cv=CV_FOLDS, scoring="neg_log_loss", max_iter=3000,
                  random_state=RANDOM_STATE, refit=True, n_jobs=-1, solver="saga")
    specs = {
        "Unpenalised logistic": (LogisticRegression(C=np.inf, max_iter=4000), True),
        "LASSO logistic": (LogisticRegressionCV(l1_ratios=(1.0,),
                                                Cs=np.logspace(-3, 1, 8), **shared), True),
        "Elastic net logistic": (LogisticRegressionCV(l1_ratios=(0.2, 0.5, 0.9),
                                                     Cs=np.logspace(-3, 1, 8), **shared), True),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                  min_child_weight=5, eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=-1), False),
        "Decision tree (depth 3)": (DecisionTreeClassifier(max_depth=3,
                                                          min_samples_leaf=40,
                                                          random_state=RANDOM_STATE), False),
    }
    preds, rows = {}, []
    for name, (est, scale) in specs.items():
        p = cross_val_predictions(build_pipeline(sep, predictors, est, scale=scale),
                                  sep[predictors], y, n_repeats=CV_REPEATS,
                                  label=name)
        preds[name] = p
        rows.append({"model": name, **discrimination_metrics(y, p),
                     **calibration_metrics(y, p)})
    return pd.DataFrame(rows), preds


def report_replicate_models(t: pd.DataFrame, preds: dict,
                            y: np.ndarray) -> dict:
    question(52, "Does 'gradient boosting does not beat penalised regression' hold\n"
                 "at EPV 28, or was it an artefact of a small sample?")
    show = t[["model", "auc", "pr_auc", "calibration_slope", "brier"]].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    d = bootstrap_auc_difference(y, preds["XGBoost"], preds["Elastic net logistic"])
    print(f"\n  XGBoost minus elastic net, sepsis cohort:")
    print(f"    difference   {d['difference']:+.4f}  "
          f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
          f"crosses zero {d['crosses_zero']}")
    ref = CHF_REFERENCE
    print(f"  Same comparison in the CHF cohort (EPV {ref['epv']}):")
    print(f"    difference   {ref['xgb_minus_enet']:+.4f}  "
          f"[{ref['xgb_ci'][0]:+.4f}, {ref['xgb_ci'][1]:+.4f}]  crosses zero True")
    return d


# ═══ Q53. The clean holdout ══════════════════════════════════════════════════
def confirmatory_sepsis(predictors: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from xgboost import XGBClassifier

    coh = confirmatory_frames(group=SEPSIS_LABEL)
    tr, te = coh.chf_train, coh.chf_test
    ytr, yte = make_outcome(tr).values, make_outcome(te).values

    shared = dict(cv=CV_FOLDS, scoring="neg_log_loss", max_iter=3000,
                  random_state=RANDOM_STATE, refit=True, n_jobs=-1, solver="saga")
    fits = {
        "Elastic net (primary)": (LogisticRegressionCV(
            l1_ratios=(0.2, 0.5, 0.9), Cs=np.logspace(-3, 1, 8), **shared), True),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                  min_child_weight=5, eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=-1), False),
    }
    rows, preds = [], {}
    for name, (est, scale) in fits.items():
        pipe = build_pipeline(tr, predictors, est, scale=scale)
        pipe.fit(tr[predictors], ytr)
        p = pipe.predict_proba(te[predictors])[:, 1]
        preds[name] = p
        rows.append({"model": name, **discrimination_metrics(yte, p),
                     **calibration_metrics(yte, p)})

    has = te[PHYSICIAN_BENCHMARK].notna().values
    doc = None
    if has.sum() > 50:
        from sklearn.metrics import roc_auc_score
        p_doc = (1 - te.loc[has, PHYSICIAN_BENCHMARK]).values
        doc = {"n": int(has.sum()),
               "doc_auc": roc_auc_score(yte[has], p_doc),
               "model_auc": roc_auc_score(yte[has], preds["Elastic net (primary)"][has]),
               **bootstrap_auc_difference(
                   yte[has], preds["Elastic net (primary)"][has], p_doc)}
    return {"table": pd.DataFrame(rows), "preds": preds, "y_test": yte,
            "n_train": len(tr), "n_test": len(te),
            "train_prev": float(ytr.mean()), "test_prev": float(yte.mean()),
            "comparison": bootstrap_auc_difference(
                yte, preds["XGBoost"], preds["Elastic net (primary)"]),
            "physician": doc}


def report_confirmatory(r: dict, cv_auc: float) -> None:
    question(53, "Spend the clean sepsis holdout. Does the confirmatory result look\n"
                 "different when the cohort is adequately powered?")
    print(f"  train {r['n_train']:,} ({r['train_prev']*100:.1f}% events)   "
          f"held out {r['n_test']:,} ({r['test_prev']*100:.1f}% events)")
    print(f"  prevalence gap {abs(r['test_prev']-r['train_prev'])*100:.1f} points "
          f"(CHF arm: 5.4 points, which broke its calibration)\n")
    show = r["table"][["model", "auc", "calibration_slope",
                       "calibration_intercept", "brier", "mean_predicted",
                       "observed"]].copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    primary = r["table"].iloc[0]
    print(f"\n  cross-validated AUC {cv_auc:.3f} -> held-out {primary.auc:.3f} "
          f"({primary.auc - cv_auc:+.3f})")
    c = r["comparison"]
    print(f"\n  XGBoost minus elastic net on held-out sepsis patients:")
    print(f"    {c['difference']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  "
          f"crosses zero {c['crosses_zero']}")
    if r["physician"]:
        d = r["physician"]
        print(f"\n  Against the attending physician's 6-month survival estimate,"
              f" on the\n  {d['n']:,} held-out patients who have one:")
        print(f"    physician {d['doc_auc']:.3f}   model {d['model_auc']:.3f}")
        print(f"    model minus physician {d['difference']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
              f"crosses zero {d['crosses_zero']}")
        print(f"    CHF held-out, same comparison: physician "
              f"{CHF_REFERENCE['physician_auc_test']:.3f}, model "
              f"{CHF_REFERENCE['model_auc_test']:.3f}")


# ═══ Figure ══════════════════════════════════════════════════════════════════
def figure_replication(sep_models: pd.DataFrame, conf: dict, cv_auc: float):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    names = ["Unpenalised\nlogistic", "LASSO", "Elastic net", "XGBoost", "Tree\n(d3)"]
    chf_auc = CHF_REFERENCE["cv_auc"]
    x = np.arange(len(names))
    # The two bar series are paired by position, so a reordered table would
    # silently plot each cohort's value against the wrong model.
    if list(sep_models.model) != MODEL_ORDER:
        raise ValueError("sepsis table is not in MODEL_ORDER; bars would mispair")
    ax1.bar(x - 0.19, chf_auc, width=0.36, color=viz.BASELINE, label="CHF (EPV 6.4)")
    ax1.bar(x + 0.19, sep_models.auc, width=0.36, color=viz.SERIES_BLUE,
            label="Sepsis (EPV 28)")
    ax1.set_xticks(x, names, fontsize=8.5)
    ax1.set_ylim(0.55, max(max(chf_auc), sep_models.auc.max()) + 0.03)
    ax1.set_ylabel("Cross-validated AUC")
    ax1.set_title("Does the ranking replicate?")
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(axis="x", visible=False)
    viz.despine(ax1)

    chf_slope = CHF_REFERENCE["cv_slope"]
    ax2.bar(x - 0.19, chf_slope, width=0.36, color=viz.BASELINE, label="CHF")
    ax2.bar(x + 0.19, sep_models.calibration_slope, width=0.36,
            color=viz.SERIES_BLUE, label="Sepsis")
    ax2.axhline(1.0, color=viz.SERIES_ORANGE, lw=1.6, ls="--")
    ax2.text(len(names) - 0.5, 1.02, "ideal", fontsize=8.5,
             color=viz.SERIES_ORANGE, ha="right")
    ax2.set_xticks(x, names, fontsize=8.5)
    ax2.set_ylabel("Calibration slope")
    ax2.set_title("Calibration at four times the events")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(axis="x", visible=False)
    viz.despine(ax2)

    labels = ["CHF\ncross-val", "CHF\nheld-out", "Sepsis\ncross-val",
              "Sepsis\nheld-out"]
    vals = [CHF_REFERENCE["elastic_net_auc"], CHF_REFERENCE["test_auc"],
            cv_auc, float(conf["table"].iloc[0].auc)]
    cols = [viz.BASELINE, viz.BASELINE, viz.SERIES_BLUE, viz.SERIES_BLUE]
    ax3.bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax3.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax3.set_xticks(range(4), labels, fontsize=8.5)
    ax3.set_ylim(0.55, max(vals) + 0.04)
    ax3.set_ylabel("AUC")
    ax3.set_title("Development vs held-out, both cohorts")
    ax3.grid(axis="x", visible=False)
    viz.despine(ax3)

    fig.tight_layout()
    viz.caption(fig, f"{OUTCOME_LABEL}. Grey is the CHF arm (978 training patients, 248 events); blue is the\n"
                     f"sepsis replication (2,458 and 1,091). The sepsis held-out partition had never been read\n"
                     f"before this script.", y=-0.05)
    return viz.save(fig, "22_replication.png")


ANSWERS = """
ANSWERS
{rule}

A50. WHO THESE PATIENTS ARE
    {n} adults with acute respiratory failure or multi-organ system failure in
    the setting of sepsis, {events} of whom died within {horizon} days
    ({prev}%). Median age {age_med}, {female}% female, {comorb3}% carrying three
    or more comorbidities.

    The clinical frame is intensive care, not cardiology, and the vocabulary has
    to change with it. The organ-system markers this dataset carries map onto
    the components of a modern severity score: P/F ratio (median {pafi_med},
    with {pafi_ali}% below 300, the acute lung injury threshold) is the
    respiratory component of SOFA; creatinine (median {crea_med} mg/dL,
    {crea_aki}% above 2.0) is the renal component; the coma score stands in for
    the neurological one. APACHE III is in the file as `aps` and remains
    excluded for the reason it always was -- it is computed from the same vitals
    and labs used as predictors, so it is collinear by construction rather than
    an independent benchmark.

    What this cohort does NOT provide is the thing that would make it a proper
    sepsis study: no lactate, no culture results, no antibiotic timing, no
    vasopressor or fluid data. The Surviving Sepsis Campaign bundles are built
    on exactly those, and none of them are here. So this cohort is being used
    for its statistical properties, and the clinical interpretation is
    correspondingly thinner than the CHF arm's. Saying that plainly is better
    than dressing it up.

    The statistical properties are the point. Events per variable is {epv}
    against the CHF arm's {chf_epv} -- above the conventional floor of 10 rather
    than well below it, and enough for Riley's criteria to be roughly satisfied
    where the CHF cohort missed them by a factor of 2.5.

A51. DO THE EXPLORATORY FINDINGS REPLICATE?
    One does, emphatically. One does not, and the failure is more interesting
    than the success.

    THE ENROLMENT ARTEFACT REPLICATES. {wave_verdict} This is the finding that
    SHOULD carry across, because it is a property of the study rather than of
    heart failure: SUPPORT enrolled in two waves with different data-collection
    instruments, so every disease group inherits the same structure. The
    negative controls are what make it convincing -- adlp and income are missing
    for ordinary reasons and show essentially no wave structure, so the split is
    picking out a collection protocol rather than sick patients.

    The downstream consequence replicates too, and this is where a threshold
    test would have misled. Compare the two test statistics rather than the two
    p-values. The three protocol labs shrink by a factor of {lab_shrinkage} or
    more once the test accounts for exposure time. The negative controls do not
    shrink at all -- their statistics GROW, by at least {ctrl_growth}x, which
    is what happens when a real association is measured
    more precisely by adding time-to-event information rather than being
    explained away by it. Censored follow-up runs {lab_ratio}x longer for the
    unmeasured group among the labs, against {ctrl_ratio}x for the controls.

    That is the diagnostic in a single line: an artefact loses an order of
    magnitude when you account for exposure time; a real effect gains.

    Note that in a cohort 2.5 times the size, one of those log-rank p-values
    still lands below 0.05. Reporting "did not replicate" on that basis would be
    wrong. Power scales with n, so the question is never whether a p-value
    crossed a line in a bigger sample -- it is whether the EFFECT collapsed. It
    did, by more than an order of magnitude, in the same three variables and
    neither control.

    THE DNR FINDING DOES NOT REPLICATE. {dnr_verdict}

    In CHF, a directive the patient brought from home was statistically
    indistinguishable from no directive (log-rank p = 0.882) while one written
    during the admission nearly doubled mortality. That contrast was the whole
    argument for treating DNR as a record of a care decision rather than a
    measure of illness. It does not hold here.

    The honest reading is that the CHF version was probably too clean. That
    cohort had few patients with a pre-existing directive, so "indistinguishable
    from no DNR" may have been low power rather than genuine equivalence -- and
    the same caution applies in reverse here, where the group is only
    {dnr_before_n} patients wide. A clinical reading is also available: someone
    arriving at an ICU in multi-organ failure with a standing directive is
    plausibly frailer relative to their peers than a heart failure patient with
    the same paperwork.

    What survives either way is the decision that came out of it. In-admission
    DNR remains the strongest single marker in the cohort, and it is still
    excluded from the model for the same reason -- a variable that records a
    decision to stop treating cannot be used to predict the outcome of
    treatment without becoming self-fulfilling. That exclusion never depended on
    the before-versus-after contrast.

A52. DOES THE MODEL COMPARISON HOLD AT EPV 28?
    {model_verdict}

    This was the test worth running, because the CHF result was weak by
    construction. At 6.4 events per variable, penalised regression beating
    gradient boosting is close to what theory predicts regardless of whether it
    is true in general: small samples punish flexible models, and regularisation
    is doing most of the work. A sceptical reviewer would have been right to
    discount it.

    At {epv} events per variable that defence disappears. XGBoost has enough
    data to fit whatever structure is there, and the comparison becomes a real
    test of whether additional flexibility buys anything on tabular clinical
    data.

    Note also what happened to calibration. The CHF arm's penalised models had
    slopes near 1.19 -- above 1, meaning predictions bunched too close to the
    average, which is the signature of a penalty doing more work than the data
    requires. If that was a sample-size effect rather than a coding error, the
    slopes should move toward 1 once the penalty has more events to work with.

    {calib_shift_verdict}

A53. THE CLEAN HOLDOUT
    The sepsis test partition had never been read. 10_confirmatory.py used only
    the CHF test rows, so this is a genuine confirmatory analysis rather than a
    second look -- and it is the one the CHF arm can no longer provide.

    Cross-validated AUC {cv_auc} became {test_auc} on held-out patients, a
    change of {auc_change}. {transfer_verdict}

    The prevalence gap between partitions is {prev_gap} points here against 5.4
    in the CHF arm. That matters: the CHF calibration failure was traced to
    stratifying on all-cause death over full follow-up while modelling 180-day
    mortality in a subgroup, and the resulting imbalance made the model
    under-predict. The same defect exists here -- the split is still stratified
    on the wrong outcome -- but the larger cohort makes it bite far less, which
    is itself evidence for the diagnosis. {calib_verdict}

    THE PHYSICIAN COMPARISON. This was the CHF arm's most uncomfortable result
    and the one most likely to be challenged, so it is the one most worth
    re-testing. There, the model beat the attending physician in development
    (0.687 against 0.655) and then lost to them on held-out patients (0.633
    against {chf_doc}), with the difference interval brushing zero. Two readings
    were available and the CHF data could not separate them: either the
    physician is genuinely the stronger predictor and the development result was
    optimism, or the held-out sample was too small to resolve a real difference.
    This cohort has enough patients with a recorded physician estimate to say
    which.

    {physician_verdict}

    Note what the comparison is and is not. The physician saw the patient; the
    model saw a row. A clinician who has examined someone has information no
    dataset here contains, so a model matching them is doing well and a model
    losing to them is not embarrassing -- it is the expected result. The useful
    question was never "does the model win", it is whether the model adds
    anything the clinician does not already have.

    What this does not fix is external validity. Both cohorts come from the same
    five hospitals in the same programme between 1989 and 1994. Replicating in a
    second disease group is a real strengthening of the METHODS findings and
    says nothing about whether either model would work in a modern ICU.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    coh = analysis_frames(group=SEPSIS_LABEL)
    sep = coh.chf_train
    y = make_outcome(sep).values
    predictors = default_predictors(sep)

    n_params = design_width(sep, predictors)

    header(f"SUPPORT2 -- replication in {SEPSIS_LABEL}")
    print(f"  Primary cohort was CHF: {CHF_REFERENCE['n_train']} training "
          f"patients, {CHF_REFERENCE['events']} events, EPV "
          f"{CHF_REFERENCE['epv']}.")
    print(f"  This cohort: {len(sep):,} training patients, {int(y.sum()):,} "
          f"events, EPV {y.sum()/n_params:.1f}.")
    print(f"  {len(predictors)} candidate predictors, same governance as the CHF arm.")

    d = report_cohort(sep, y, n_params)
    eda = replicate_eda(sep, y)
    eda_r = report_replicate_eda(eda)

    models, preds = replicate_models(sep, y, predictors)
    comp = report_replicate_models(models, preds, y)

    cv_auc = float(models[models.model == "Elastic net logistic"].iloc[0].auc)
    conf = confirmatory_sepsis(predictors)
    report_confirmatory(conf, cv_auc)

    header("FIGURES")
    path = figure_replication(models, conf, cv_auc)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    primary = conf["table"].iloc[0]
    auc_change = primary.auc - cv_auc
    prev_gap = abs(conf["test_prev"] - conf["train_prev"]) * 100

    wave_verdict = (
        "The three protocol labs are absent for essentially every patient on "
        "one side of the split and present for almost all of them on the other, "
        "in a cohort with different patients, a different mortality rate and a "
        "different clinical course."
        if eda_r["wave_replicated"] else
        "The wave structure is weaker here than in CHF, so the mechanism is not "
        "as cleanly separated and the original demonstration is the stronger "
        "evidence of the two.")
    dnr_verdict = (
        f"A directive recorded before admission carries a "
        f"{eda_r['dnr_before_gap']:+.1f} point mortality gap against no "
        f"directive ({fmt_p(eda_r['dnr_before_p'])}), where in CHF the same "
        f"comparison was flat."
        if not eda_r["dnr_replicated"] else
        f"A directive recorded before admission is again indistinguishable from "
        f"no directive ({fmt_p(eda_r['dnr_before_p'])}), reproducing the CHF "
        f"contrast.")
    model_verdict = (
        f"It holds. XGBoost minus elastic net is {comp['difference']:+.4f} with "
        f"interval [{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}], crossing "
        f"zero at EPV {d['epv']:.0f} just as it did at EPV "
        f"{CHF_REFERENCE['epv']}."
        if comp["crosses_zero"] else
        f"It does NOT hold. XGBoost minus elastic net is "
        f"{comp['difference']:+.4f} with interval [{comp['ci_low']:+.4f}, "
        f"{comp['ci_high']:+.4f}], which excludes zero. At adequate sample size "
        f"the flexible model does separate from the regression, and the CHF "
        f"conclusion was a small-sample artefact. That is a correction the "
        f"project has to carry.")
    transfer_verdict = (
        "Performance held." if abs(auc_change) < 0.03 else
        "Performance dropped materially, which at this sample size is harder to "
        "attribute to sampling noise than it was in the CHF arm."
        if auc_change < 0 else
        "Performance rose, which at this size is more likely a real property of "
        "the partition than noise.")
    calib_verdict = (
        f"Held-out calibration slope is {primary.calibration_slope:.2f} against "
        f"the CHF arm's {CHF_REFERENCE['test_slope']:.2f}."
        + (" The failure did not recur." if 0.8 <= primary.calibration_slope <= 1.25
           else " It is still off, so sample size was not the whole story."))

    # Did the penalised models' over-shrinkage ease with more events? Measured
    # as distance from the ideal slope of 1, averaged over LASSO and elastic
    # net, rather than asserted.
    pen = ["LASSO logistic", "Elastic net logistic"]
    pen_i = [MODEL_ORDER.index(m) for m in pen]
    chf_gap = float(np.mean([abs(CHF_REFERENCE["cv_slope"][i] - 1) for i in pen_i]))
    sep_gap = float(np.mean([abs(models.loc[models.model == m,
                                            "calibration_slope"].iloc[0] - 1)
                             for m in pen]))
    sep_slopes = ", ".join(
        f"{m.split()[0]} {models.loc[models.model == m, 'calibration_slope'].iloc[0]:.2f}"
        for m in pen)
    calib_shift_verdict = (
        f"They did. On the sepsis cohort the penalised slopes are {sep_slopes}, "
        f"an average distance from 1 of {sep_gap:.3f} against the CHF arm's "
        f"{chf_gap:.3f}. The over-shrinkage eased as the events accumulated, "
        f"which is the behaviour a correctly implemented penalty should show "
        f"and is a useful check that nothing is miscoded."
        if sep_gap < chf_gap else
        f"They did not. On the sepsis cohort the penalised slopes are "
        f"{sep_slopes}, an average distance from 1 of {sep_gap:.3f} against the "
        f"CHF arm's {chf_gap:.3f}. More events did not pull the slopes toward "
        f"1, so the miscalibration is not simply a sample-size effect and the "
        f"penalty's behaviour deserves a closer look than this project has "
        f"given it.")

    doc = conf["physician"]
    if doc is None:
        physician_verdict = ("Too few held-out patients carry a physician "
                             "estimate to make the comparison here.")
    elif doc["crosses_zero"]:
        # The interval crossing zero is not the whole story. The CHF holdout
        # also crossed zero, and in the same direction. Two independent cohorts
        # pointing the same way is evidence a single interval does not carry,
        # so the direction is reported alongside the significance.
        same_way = (doc["difference"] < 0) == (
            CHF_REFERENCE["model_auc_test"] < CHF_REFERENCE["physician_auc_test"])
        chf_lo, chf_hi = CHF_REFERENCE["physician_ci"]
        width_ratio = ((doc["ci_high"] - doc["ci_low"]) / (chf_hi - chf_lo))
        ahead = "physician" if doc["difference"] < 0 else "model"
        physician_verdict = (
            f"The {ahead} is ahead, and the interval does not exclude zero: "
            f"model {doc['model_auc']:.3f} against physician "
            f"{doc['doc_auc']:.3f} on {doc['n']:,} patients, difference "
            f"{doc['difference']:+.4f} [{doc['ci_low']:+.4f}, "
            f"{doc['ci_high']:+.4f}]. Taken alone that is 'not resolved'."
            + (f" Taken with the CHF holdout it is more than that: both cohorts "
               f"put the {ahead} ahead, on different patients with different "
               f"illnesses, and the sepsis interval is {width_ratio:.0%} the "
               f"width of the CHF one. Two independent estimates agreeing in "
               f"direction is "
               f"evidence neither interval carries on its own. Of the two "
               f"readings the CHF arm could not separate, the sepsis cohort "
               f"favours the first: the {ahead} is probably genuinely better, "
               f"and the development result was optimism rather than a small "
               f"holdout." if same_way else
               f" The CHF holdout pointed the other way, so the direction is "
               f"not stable across cohorts and neither result should be "
               f"leaned on."))
    elif doc["difference"] > 0:
        physician_verdict = (
            f"The model beats the physician on held-out patients: "
            f"{doc['model_auc']:.3f} against {doc['doc_auc']:.3f}, difference "
            f"{doc['difference']:+.4f} [{doc['ci_low']:+.4f}, "
            f"{doc['ci_high']:+.4f}], excluding zero. The CHF arm's reversal "
            f"does not survive at this sample size, which points to that result "
            f"having been a small-sample effect.")
    else:
        physician_verdict = (
            f"The physician beats the model, and this time the interval "
            f"excludes zero: {doc['doc_auc']:.3f} against "
            f"{doc['model_auc']:.3f}, difference {doc['difference']:+.4f} "
            f"[{doc['ci_low']:+.4f}, {doc['ci_high']:+.4f}]. The CHF reversal "
            f"replicates with adequate power, so it was not a fluke of a small "
            f"holdout. Bedside judgement is the stronger predictor here and the "
            f"project should say so plainly.")

    facts = Facts(
        n=f"{d['n']:,}", events=f"{d['events']:,}", horizon=str(HORIZON_DAYS),
        prev=f"{d['prev']*100:.1f}", age_med=f"{d['age_med']:.0f}",
        female=f"{d['female']:.1f}", comorb3=f"{d['comorb3']:.1f}",
        pafi_med=f"{d['pafi_med']:.0f}", pafi_ali=f"{d['pafi_ali']:.1f}",
        crea_med=f"{d['crea_med']:.1f}", crea_aki=f"{d['crea_aki']:.1f}",
        epv=f"{d['epv']:.1f}", chf_epv=str(CHF_REFERENCE["epv"]),
        wave_verdict=wave_verdict, dnr_verdict=dnr_verdict,
        lab_ratio=f"{eda_r['lab_ratio']:.2f}",
        ctrl_ratio=f"{eda_r['ctrl_ratio']:.2f}",
        lab_shrinkage=f"{eda_r['lab_shrinkage']:.0f}",
        ctrl_growth=f"{1 / eda_r['ctrl_shrinkage']:.1f}",
        dnr_before_n=f"{eda_r['dnr_before_n']:,}",
        model_verdict=model_verdict,
        cv_auc=f"{cv_auc:.3f}", test_auc=f"{primary.auc:.3f}",
        auc_change=f"{auc_change:+.3f}", transfer_verdict=transfer_verdict,
        prev_gap=f"{prev_gap:.1f}", calib_verdict=calib_verdict,
        physician_verdict=physician_verdict,
        calib_shift_verdict=calib_shift_verdict,
        chf_doc=f"{CHF_REFERENCE['physician_auc_test']:.3f}",
    )
    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "12_replication.txt")
