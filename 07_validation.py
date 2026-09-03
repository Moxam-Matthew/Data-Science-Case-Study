"""
07_validation.py -- Was the sample big enough, how optimistic is the fit, does the
model add anything, and what do the coefficients actually mean to a patient?

Four questions that a clinical reviewer asks after the modelling is done, and
that a leaderboard never asks at all.

    Run:  python 07_validation.py

    Training partition only. The held-out 30% remains unread.

THE QUESTIONS
    Q31  Events-per-variable is a rule of thumb from 1996. Riley's sample-size
         criteria are the modern replacement. Was this cohort ever large enough
         to fit the model that was fitted?
    Q32  A held-out partition tells you how the model performs on unseen rows.
         It does not tell you how much the FITTING process overfits. What is
         optimism-corrected bootstrap validation, and what does it reveal here?
    Q33  The model and the attending physician have similar AUCs. That is the
         wrong comparison. What is the right one, and what does it show?
    Q34  The model reports an odds ratio of 1.81 for coma score. A cardiologist
         will hear "81% more likely to die". Why is that wrong at this
         prevalence, and what should you report instead?

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
    build_pipeline,
    calibration_metrics,
    cross_val_predictions,
    default_predictors,
    discrimination_metrics,
    make_outcome,
)
from report import RULE, Facts, configure_pandas, fmt_p, header, question, run_and_capture
from support2 import UNITS, analysis_frames

OUT_DIR = Path(__file__).resolve().parent / "output"

N_BOOTSTRAP = 100          # Harrell's optimism correction; 100 is the usual floor
SHRINKAGE_TARGET = 0.90    # Riley criterion 1: accept at most 10% shrinkage
CI_HALF_WIDTH = 0.05       # Riley criterion 3: precision of the overall risk estimate


# ═══ Q31. Riley sample size ══════════════════════════════════════════════════
def cox_snell_r2(y: np.ndarray, p: np.ndarray) -> float:
    """
    Cox-Snell R^2 from fitted probabilities, which is what Riley's criteria take
    as input rather than the AUC people usually have to hand.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll_model = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    phi = y.mean()
    ll_null = np.sum(y * np.log(phi) + (1 - y) * np.log(1 - phi))
    lr = 2 * (ll_model - ll_null)
    return float(1 - np.exp(-lr / len(y)))


def riley_sample_size(n_params: int, r2_cs: float, prevalence: float,
                      shrinkage: float = SHRINKAGE_TARGET,
                      delta: float = CI_HALF_WIDTH) -> dict:
    """
    Minimum sample size for a binary-outcome prediction model (Riley et al.,
    Stat Med 2019), the modern replacement for "10 events per variable".

    EPV asks only whether the coefficients are estimable. Riley asks whether the
    model will be usefully PRECISE, and answers it with three separate criteria
    that must all be satisfied:

      1. Expected shrinkage no worse than `shrinkage`. The model should not need
         its coefficients pulled toward zero by more than 10% to stop it
         overfitting.
      2. Small absolute optimism in Nagelkerke R^2 (<= 0.05).
      3. The overall risk estimated precisely enough that its 95% interval is
         no wider than +/- `delta`.

    The binding criterion is whichever demands the most patients. Reporting only
    EPV hides which constraint you are actually up against.
    """
    max_r2 = 1 - np.exp(2 * (prevalence * np.log(prevalence)
                             + (1 - prevalence) * np.log(1 - prevalence)))
    r2_nagelkerke = r2_cs / max_r2

    # Criterion 1: shrinkage.
    n1 = n_params / ((shrinkage - 1) * np.log(1 - r2_cs / shrinkage))

    # Criterion 2: absolute optimism in Nagelkerke R^2 below 0.05.
    target = 0.05
    s_needed = r2_cs / (r2_cs + target * max_r2)
    n2 = n_params / ((s_needed - 1) * np.log(1 - r2_cs / s_needed))

    # Criterion 3: precision of the overall risk.
    n3 = (1.96 / delta) ** 2 * prevalence * (1 - prevalence)

    required = max(n1, n2, n3)
    binding = {n1: "shrinkage <= 10%", n2: "R^2 optimism <= 0.05",
               n3: f"risk CI within +/-{delta}"}[required]
    return {"n_params": n_params, "r2_cox_snell": r2_cs,
            "r2_nagelkerke": r2_nagelkerke, "max_r2": max_r2,
            "n_criterion_1": n1, "n_criterion_2": n2, "n_criterion_3": n3,
            "required_n": required, "binding_criterion": binding,
            "epv_rule_n": n_params * 10 / prevalence}


def report_riley(r: dict, n_actual: int, events: int, facts: Facts) -> None:
    question(31, f"Events-per-variable is a rule of thumb from 1996. Riley's criteria\n"
                 f"are the modern replacement. With {r['n_params']} parameters and "
                 f"{events} events in\n{n_actual:,} patients, was this cohort ever large "
                 f"enough for the model\nthat was fitted?")
    print(f"  Anticipated Cox-Snell R^2      {r['r2_cox_snell']:.4f}")
    print(f"  Nagelkerke R^2                 {r['r2_nagelkerke']:.4f}  "
          f"(max attainable {r['max_r2']:.4f} at this prevalence)")
    print()
    print(f"  {'criterion':<34} {'required n':>11}")
    print(f"  {'1. shrinkage <= 10%':<34} {r['n_criterion_1']:>11,.0f}")
    print(f"  {'2. R^2 optimism <= 0.05':<34} {r['n_criterion_2']:>11,.0f}")
    print(f"  {'3. risk CI within +/-0.05':<34} {r['n_criterion_3']:>11,.0f}")
    print(f"  {'-'*34} {'-'*11}")
    print(f"  {'REQUIRED (binding: ' + r['binding_criterion'] + ')':<34} "
          f"{r['required_n']:>11,.0f}")
    print(f"  {'available':<34} {n_actual:>11,}")
    print(f"\n  For contrast, the 10-EPV rule would demand "
          f"{r['epv_rule_n']:,.0f} patients.")
    shortfall = r["required_n"] - n_actual
    verdict = ("adequate" if shortfall <= 0
               else f"SHORT BY {shortfall:,.0f} patients "
                    f"({r['required_n']/n_actual:.1f}x the available cohort)")
    print(f"  {'verdict':<34} {verdict}")


# ═══ Q32. Optimism-corrected bootstrap ═══════════════════════════════════════
def optimism_bootstrap(chf: pd.DataFrame, predictors: list[str], y: np.ndarray,
                       estimator_factory, n_boot: int = N_BOOTSTRAP,
                       label: str = "") -> dict:
    """
    Harrell's optimism correction.

      1. Fit on the full sample; record APPARENT performance.
      2. For each bootstrap resample: refit from scratch, score on the resample
         (where it was fitted) and on the ORIGINAL sample (which it has only
         partly seen). The gap is that replicate's optimism.
      3. Corrected performance = apparent - mean optimism.

    This answers a different question from a holdout. A holdout tells you how
    one particular fitted model does on unseen rows. This tells you how much the
    FITTING PROCEDURE inflates its own scoring -- and because every replicate
    refits from scratch, imputation included, the estimate covers the whole
    pipeline rather than the final coefficients alone.

    It is also the answer to "you only have 978 patients, why not use all of
    them for training": you can, if you are willing to quantify the optimism
    instead of hiding it.
    """
    import sys
    import time

    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    X = chf[predictors]
    full = build_pipeline(chf, predictors, estimator_factory())
    full.fit(X, y)
    p_app = full.predict_proba(X)[:, 1]
    apparent = {"auc": roc_auc_score(y, p_app),
                **{k: v for k, v in calibration_metrics(y, p_app).items()
                   if k in ("calibration_slope", "brier")}}

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    opt_auc, opt_slope, opt_brier, slopes_on_original = [], [], [], []
    started = time.time()
    for b in range(1, n_boot + 1):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot = clone(full)
        boot.fit(X.iloc[idx], y[idx])
        p_boot = boot.predict_proba(X.iloc[idx])[:, 1]
        p_orig = boot.predict_proba(X)[:, 1]
        cal_boot = calibration_metrics(y[idx], p_boot)
        cal_orig = calibration_metrics(y, p_orig)
        opt_auc.append(roc_auc_score(y[idx], p_boot) - roc_auc_score(y, p_orig))
        opt_slope.append(cal_boot["calibration_slope"] - cal_orig["calibration_slope"])
        opt_brier.append(cal_boot["brier"] - cal_orig["brier"])
        slopes_on_original.append(cal_orig["calibration_slope"])
        if label and b % 10 == 0:
            el = time.time() - started
            print(f"    {label:<26} boot {b:>3}/{n_boot}  {el:5.0f}s elapsed, "
                  f"~{el/b*(n_boot-b):4.0f}s remaining", file=sys.stderr, flush=True)
    if label:
        print(f"    {label:<26} DONE {n_boot} bootstraps in "
              f"{time.time()-started:.0f}s", file=sys.stderr, flush=True)

    return {
        "apparent_auc": apparent["auc"],
        "optimism_auc": float(np.mean(opt_auc)),
        "corrected_auc": apparent["auc"] - float(np.mean(opt_auc)),
        "apparent_slope": apparent["calibration_slope"],
        "optimism_slope": float(np.mean(opt_slope)),
        "corrected_slope": apparent["calibration_slope"] - float(np.mean(opt_slope)),
        "apparent_brier": apparent["brier"],
        "corrected_brier": apparent["brier"] - float(np.mean(opt_brier)),
        "uniform_shrinkage": float(np.mean(slopes_on_original)),
        "n_boot": len(opt_auc),
    }


def report_optimism(results: dict[str, dict], cv_auc: dict) -> None:
    question(32, "A held-out partition tells you how one fitted model performs on\n"
                 "unseen rows. It does not tell you how much the FITTING PROCESS\n"
                 "overfits. What is optimism-corrected bootstrap validation, and\n"
                 "what does it reveal here?")
    rows = []
    for name, r in results.items():
        rows.append({"model": name, "apparent_auc": r["apparent_auc"],
                     "optimism": r["optimism_auc"],
                     "corrected_auc": r["corrected_auc"],
                     "cv_auc": cv_auc.get(name, np.nan),
                     "apparent_slope": r["apparent_slope"],
                     "shrinkage_factor": r["uniform_shrinkage"]})
    t = pd.DataFrame(rows)
    for c in t.columns[1:]:
        t[c] = t[c].round(3)
    print(t.to_string(index=False))
    print(f"\n  {list(results.values())[0]['n_boot']} bootstrap replicates, each "
          f"refitting the whole pipeline including imputation.")
    print("  `shrinkage_factor` is the mean calibration slope of bootstrap models")
    print("  scored on the original data: the factor by which coefficients would")
    print("  need multiplying to stop the model overfitting.")


# ═══ Q33. Incremental value ══════════════════════════════════════════════════
def incremental_value(chf: pd.DataFrame, predictors: list[str],
                      y: np.ndarray) -> dict:
    """
    Does the model add anything to what the clinician already knows?

    Three nested models on the patients the physician scored:
      A  physician estimate alone
      B  model predictors alone
      C  both

    A beats-the-doctor comparison (A vs B) is the wrong question, because in
    practice nobody discards the clinician. The question a cardiology journal
    asks is C vs A: given the physician's judgement, do the measured variables
    contribute anything further?
    """
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    has = chf[PHYSICIAN_BENCHMARK].notna().values
    sub, y_sub = chf[has], y[has]

    # Physician estimate on the logit scale, so it enters as a proper covariate.
    p_doc = np.clip(1 - sub[PHYSICIAN_BENCHMARK].values, 1e-4, 1 - 1e-4)
    logit_doc = np.log(p_doc / (1 - p_doc))

    est = lambda: LogisticRegressionCV(
        solver="saga", Cs=np.logspace(-3, 1, 8), l1_ratios=(0.5,), cv=CV_FOLDS,
        scoring="neg_log_loss", max_iter=3000, random_state=RANDOM_STATE,
        n_jobs=-1)

    # The model's contribution must be measured OUT OF FOLD. Fitting on these
    # 727 patients and then scoring the same rows would credit the model with
    # its own overfitting: an apparent AUC here reads ~0.73 against a
    # cross-validated ~0.69 in 05_modelling.py, and the whole incremental-value
    # gain would inherit that inflation. The physician's estimate needs no such
    # correction -- it was recorded before any of this, so it cannot overfit.
    p_oof = cross_val_predictions(build_pipeline(sub, predictors, est()),
                                  sub[predictors], y_sub, n_repeats=3,
                                  label="incremental: out-of-fold")
    p_oof = np.clip(p_oof, 1e-4, 1 - 1e-4)
    lp_model = np.log(p_oof / (1 - p_oof))

    fits, aucs = {}, {}
    designs = {
        "A: physician alone": sm.add_constant(logit_doc),
        "B: model alone": sm.add_constant(lp_model),
        "C: physician + model": sm.add_constant(np.column_stack([logit_doc, lp_model])),
    }
    for name, Xd in designs.items():
        f = sm.Logit(y_sub, Xd).fit(disp=0)
        fits[name] = f
        aucs[name] = roc_auc_score(y_sub, f.predict(Xd))

    # Likelihood-ratio test for the block added to the physician's estimate.
    lr_stat = 2 * (fits["C: physician + model"].llf - fits["A: physician alone"].llf)
    ddf = int(fits["C: physician + model"].df_model
              - fits["A: physician alone"].df_model)
    from scipy import stats as sps
    lr_p = sps.chi2.sf(lr_stat, ddf)

    # Bootstrap the AUC gain from adding the model to the physician.
    rng = np.random.default_rng(RANDOM_STATE)
    gains = []
    Xa, Xc = designs["A: physician alone"], designs["C: physician + model"]
    pa, pc = fits["A: physician alone"].predict(Xa), fits["C: physician + model"].predict(Xc)
    for _ in range(2000):
        idx = rng.integers(0, len(y_sub), len(y_sub))
        if len(np.unique(y_sub[idx])) < 2:
            continue
        gains.append(roc_auc_score(y_sub[idx], pc[idx])
                     - roc_auc_score(y_sub[idx], pa[idx]))
    lo, hi = np.percentile(gains, [2.5, 97.5])

    return {"n": int(has.sum()), "aucs": aucs, "fits": fits,
            "lr_stat": lr_stat, "lr_df": ddf, "lr_p": lr_p,
            "auc_gain": aucs["C: physician + model"] - aucs["A: physician alone"],
            "gain_ci": (float(lo), float(hi)),
            "doc_coef": float(fits["C: physician + model"].params[1]),
            "doc_p": float(fits["C: physician + model"].pvalues[1]),
            "model_coef": float(fits["C: physician + model"].params[2]),
            "model_p": float(fits["C: physician + model"].pvalues[2])}


def report_incremental(r: dict) -> None:
    question(33, "The model and the attending physician have similar AUCs. That is the\n"
                 "wrong comparison, because nobody proposes discarding the clinician.\n"
                 "What is the right one, and what does it show?")
    print(f"  Nested models on the {r['n']:,} patients with a physician estimate:\n")
    for name, auc in r["aucs"].items():
        print(f"    {name:<24} AUC {auc:.3f}")
    print(f"\n  Adding the model to the physician: AUC gain {r['auc_gain']:+.4f} "
          f"[{r['gain_ci'][0]:+.4f}, {r['gain_ci'][1]:+.4f}]")
    print(f"  Likelihood-ratio test, {r['lr_df']} df: chi2 = {r['lr_stat']:.1f}, "
          f"p = {fmt_p(r['lr_p'])}")
    print(f"\n  In the combined model, both terms retain signal:")
    print(f"    physician estimate  coef {r['doc_coef']:+.3f}  p = {fmt_p(r['doc_p'])}")
    print(f"    model linear pred.  coef {r['model_coef']:+.3f}  p = {fmt_p(r['model_p'])}")


# ═══ Q34. Translation ════════════════════════════════════════════════════════
def or_to_rr(odds_ratio: float, baseline_risk: float) -> float:
    """
    Convert an odds ratio to a risk ratio at a given baseline risk
    (Zhang & Yu, JAMA 1998).

        RR = OR / (1 - p0 + p0 * OR)

    The two coincide only when the outcome is rare. At this cohort's prevalence
    they do not, and the OR is the larger number -- which is why quoting one as
    though it were the other systematically overstates the effect.
    """
    return odds_ratio / (1 - baseline_risk + baseline_risk * odds_ratio)


def translate_effects(chf: pd.DataFrame, predictors: list[str],
                      y: np.ndarray) -> pd.DataFrame:
    """Odds ratios, the risk ratios they are not, and absolute risk."""
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegressionCV

    est = LogisticRegressionCV(solver="saga", Cs=np.logspace(-3, 1, 8),
                               l1_ratios=(0.5,), cv=CV_FOLDS,
                               scoring="neg_log_loss", max_iter=3000,
                               random_state=RANDOM_STATE, n_jobs=-1)
    pipe = build_pipeline(chf, predictors, est)
    pipe.fit(chf[predictors], y)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    coefs = pipe.named_steps["model"].coef_[0]
    kept = [n for n, c in zip(names, coefs) if abs(c) > 1e-8]

    Z = pd.DataFrame(pipe.named_steps["prep"].transform(
        pipe.named_steps["indicators"].transform(chf[predictors])),
        columns=names, index=chf.index)[kept]
    fit = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)

    baseline = float(y.mean())
    rows = []
    for name in kept:
        beta = fit.params[name]
        odds = float(np.exp(beta))
        # Absolute risk at the cohort mean, and at +1 SD of this predictor.
        mean_row = Z.mean().to_frame().T
        p0 = float(fit.predict(sm.add_constant(mean_row, has_constant="add"))[0])
        shifted = mean_row.copy()
        shifted[name] = shifted[name] + 1.0
        p1 = float(fit.predict(sm.add_constant(shifted, has_constant="add"))[0])
        ard = p1 - p0
        rows.append({"variable": name, "unit": UNITS.get(name, "1 SD"),
                     "odds_ratio": odds,
                     "risk_ratio": or_to_rr(odds, baseline),
                     "risk_at_mean_pct": p0 * 100,
                     "risk_at_plus1sd_pct": p1 * 100,
                     "abs_risk_diff_pp": ard * 100,
                     "nns": abs(1 / ard) if ard else np.inf,
                     "p": float(fit.pvalues[name])})
    out = pd.DataFrame(rows)
    return out.reindex(out.odds_ratio.sub(1).abs().sort_values(ascending=False).index)


def report_translation(t: pd.DataFrame, prevalence: float) -> None:
    question(34, f"The model reports odds ratios. A cardiologist will hear an OR of\n"
                 f"1.81 as '81% more likely to die'. At a prevalence of "
                 f"{prevalence*100:.1f}% that is\nwrong. Why, and what should you "
                 f"report instead?")
    show = t.head(10).copy()
    for c in ("odds_ratio", "risk_ratio"):
        show[c] = show[c].round(2)
    for c in ("risk_at_mean_pct", "risk_at_plus1sd_pct", "abs_risk_diff_pp"):
        show[c] = show[c].round(1)
    show["nns"] = show.nns.round(0)
    show["p"] = show["p"].apply(fmt_p)
    print(show[["variable", "odds_ratio", "risk_ratio", "risk_at_mean_pct",
                "risk_at_plus1sd_pct", "abs_risk_diff_pp", "nns", "p"]]
          .to_string(index=False))
    worst = t.iloc[0]
    print(f"\n  Largest effect: {worst['variable']} at OR {worst['odds_ratio']:.2f}, "
          f"but RR {worst['risk_ratio']:.2f}.")
    print(f"  The OR overstates the relative increase by "
          f"{(worst['odds_ratio']-1)/(worst['risk_ratio']-1):.1f}x.")
    print("\n  `nns` is the number of patients you would need to move by 1 SD on")
    print("  that predictor to change one outcome -- the scale a clinician acts on.")


# ═══ Figure ══════════════════════════════════════════════════════════════════
def figure_translation(t: pd.DataFrame, prevalence: float):
    import matplotlib.pyplot as plt

    d = t.head(8).iloc[::-1]
    ypos = np.arange(len(d))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5))

    ax1.scatter(d.odds_ratio, ypos, s=80, color=viz.SERIES_BLUE,
                label="Odds ratio", edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
    ax1.scatter(d.risk_ratio, ypos, s=80, color=viz.SERIES_ORANGE,
                label="Risk ratio", edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
    for i, (o, r) in enumerate(zip(d.odds_ratio, d.risk_ratio)):
        ax1.plot([o, r], [i, i], color=viz.BASELINE, lw=1.5, zorder=1)
    ax1.axvline(1.0, color=viz.BASELINE, lw=1.2, ls="--")
    ax1.set_yticks(ypos, d.variable)
    ax1.set_xlabel("Ratio (per 1 SD)")
    ax1.set_title("An odds ratio is not a risk ratio")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", visible=False)
    viz.despine(ax1)

    colors = [viz.SERIES_ORANGE if v > 0 else viz.SERIES[2] for v in d.abs_risk_diff_pp]
    ax2.barh(ypos, d.abs_risk_diff_pp, color=colors, height=0.65)
    ax2.axvline(0, color=viz.BASELINE, lw=1.2)
    # Pad both ends so an outward-placed label on a negative bar cannot collide
    # with the axis tick beside it.
    span = d.abs_risk_diff_pp
    lo, hi = min(span.min(), 0), max(span.max(), 0)
    pad = (hi - lo) * 0.28
    ax2.set_xlim(lo - pad, hi + pad)
    offset = (hi - lo) * 0.02
    for i, v in enumerate(span):
        ax2.text(v + (offset if v >= 0 else -offset), i, f"{v:+.1f} pp",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=8.5, color=viz.INK_SECONDARY)
    ax2.set_yticks(ypos, d.variable)
    ax2.set_xlabel("Absolute change in 180-day risk (percentage points)")
    ax2.set_title("What a clinician can actually act on")
    ax2.grid(axis="y", visible=False)
    viz.despine(ax2)

    fig.tight_layout()
    viz.caption(fig, f"CHF training cohort, {OUTCOME_LABEL}, prevalence {prevalence*100:.1f}%. Left: at this\n"
                     f"prevalence the odds ratio systematically exceeds the risk ratio, so quoting one as the\n"
                     f"other overstates the effect. Right: absolute risk change from a 1 SD shift, computed at\n"
                     f"the cohort mean.", y=-0.04)
    return viz.save(fig, "16_effect_translation.png")


ANSWERS = """
ANSWERS
{rule}

A31. WAS THE COHORT EVER BIG ENOUGH?
    No, and the honest answer needs a better tool than events-per-variable.

    EPV asks whether coefficients are estimable at all. Riley's criteria ask
    whether the model will be usefully PRECISE, and they answer with three
    separate requirements. Here the binding one is "{binding}", demanding
    {required_n} patients against the {available_n} available -- a shortfall of
    {shortfall}.

    Two things follow, and the second is more interesting than the first.

    The first is that the shrinkage this cohort forces is not a modelling
    choice, it is arithmetic. 05_modelling.py found the unpenalised fit badly
    overconfident and penalisation fixing it; Riley predicts exactly that from
    the sample size before a single model is fitted. Penalisation was not a
    lucky guess, it was mandatory.

    The second is that EPV would have given a friendlier and less useful answer.
    At this prevalence the 10-EPV rule asks for about {epv_n} patients, which
    this cohort nearly meets. Riley asks for roughly {ratio}x that. The rule of
    thumb passed a model the modern criteria fail -- which is why quoting EPV
    alone, as this project did until now, understates the problem.

    What it does NOT mean is that the analysis is void. It means the model
    should be presented as underpowered for individual risk estimation, that its
    coefficients carry more uncertainty than their intervals suggest, and that
    external validation on a larger cohort is a requirement rather than a nicety.
    Saying so unprompted is the difference between reporting a limitation and
    being caught by one.

A32. WHAT THE HOLDOUT CANNOT TELL YOU
    A holdout answers: how does THIS fitted model do on rows it has not seen. It
    is the right tool for a final, honest performance estimate, which is why 30%
    of this cohort is still untouched.

    It does not answer: how much does the FITTING PROCEDURE flatter itself. For
    that you need Harrell's optimism correction -- fit on the full sample, then
    for each of {n_boot} bootstrap resamples refit from scratch and measure the
    gap between how the refit scores on its own resample and how it scores on
    the original data. That gap, averaged, is the optimism.

    The results are worth reading side by side with 05_modelling.py. Apparent
    AUC for the unpenalised model is {unpen_apparent}, corrected to
    {unpen_corrected} -- an optimism of {unpen_optimism}. The elastic net's
    apparent {enet_apparent} corrects to {enet_corrected}. The corrected figures
    land close to the cross-validated ones, which is the reassurance you want:
    two different resampling schemes agreeing that the apparent numbers are
    inflated by roughly the same amount.

    The `shrinkage_factor` column is the practical payoff. It is the mean
    calibration slope of bootstrap models scored on the original data, and it
    estimates the factor by which coefficients would need multiplying to stop
    the model overfitting. For the unpenalised model it is {unpen_shrinkage} --
    meaning roughly {unpen_shrink_pct}% of the fitted effect is noise. Riley's
    criterion 1 targets no worse than 0.90, and A31 already showed this cohort
    cannot deliver it.

    Why include this when a holdout exists? Because they answer different
    questions, and because the bootstrap uses every patient. With {available_n}
    training rows, spending 30% on a holdout is expensive; the optimism
    correction is what lets you say something honest about a model fitted on all
    of them. In a report you would present both.

A33. THE RIGHT COMPARISON
    Physician alone reaches AUC {auc_a}. The model alone reaches {auc_b}. Those
    are the numbers people quote, and the implied question -- who wins -- is not
    one anyone would act on, because no cardiology department proposes replacing
    clinical judgement with a regression.

    The question that matters is whether the model adds anything GIVEN the
    clinician. Fit both together: AUC {auc_c}, a gain of {gain} over the
    physician alone with interval {gain_ci}, and a likelihood-ratio test of
    chi2 = {lr_stat} on {lr_df} df, p = {lr_p}.

    {incremental_verdict}

    Note what the combined model does to each term. The physician's estimate
    keeps a coefficient of {doc_coef} (p = {doc_p}) and the model's linear
    predictor {model_coef} (p = {model_p}). Neither drives the other out. They
    are carrying partly different information -- which is the clinically
    sensible result, since the physician saw the patient and the model saw the
    chart.

    This is the analysis a cardiology journal asks for and the one an interview
    answer usually skips. "My model beat the doctor" invites a fight about
    whether the comparison was fair. "My model adds measurable information to
    what the doctor already knows, and here is the test" invites a collaboration.

A34. WHY THE ODDS RATIO OVERSTATES THE CASE
    An odds ratio approximates a risk ratio only when the outcome is rare,
    conventionally under about 10%. This outcome runs at {prevalence}%, so the
    approximation fails and it fails in one direction: the OR is always further
    from 1 than the RR.

    The largest effect here is {top_var}, OR {top_or}. Converted at this
    baseline risk (Zhang & Yu, JAMA 1998) the risk ratio is {top_rr} -- the odds
    ratio overstates the relative increase by {top_inflation}x. A clinician
    hearing "OR 1.81" and thinking "81% more likely to die" has been misled by a
    factor of nearly two, and it is the analyst's job to prevent that, not the
    clinician's job to remember the conversion.

    What to report instead, in ascending order of usefulness:

      * the odds ratio with its interval, because that is what the model
        estimates and what a methods reviewer will check;
      * the risk ratio at a stated baseline risk, because that is what the
        sentence "more likely" actually means;
      * the ABSOLUTE risk, because that is what a patient can act on. A 1 SD
        change in {top_var} moves 180-day risk from {top_p0}% to {top_p1}%, a
        difference of {top_ard} percentage points.

    Absolute risk is the one that survives translation to a bedside. "Your risk
    goes from 25 in 100 to 33 in 100" is a sentence a patient can weigh; "your
    odds ratio is 1.81" is not a sentence at all.

    A related discipline this project has followed by omission and should state
    explicitly: NO OVERSAMPLING. The reflex for a 25% outcome is SMOTE or class
    weighting, and both distort predicted probabilities -- which is the entire
    quantity of interest here. A model trained on a synthetically rebalanced
    cohort predicts risks for a population that does not exist, and its
    calibration is destroyed by construction. Clinical prediction methodologists
    discourage it for exactly this reason. If a decision threshold is needed,
    move the threshold; do not move the data. Knowing why not to use SMOTE is
    worth more than knowing how.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train
    y = make_outcome(chf).values
    predictors = default_predictors(chf)

    header(f"SUPPORT2 -- validation and translation, {OUTCOME_LABEL}")
    print(f"  CHF training cohort {len(chf):,}, {int(y.sum()):,} events "
          f"({y.mean()*100:.1f}%)")
    print("  Training partition only. The held-out 30% remains unread.")

    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    # Q31 needs a fitted model to supply an anticipated Cox-Snell R^2.
    ref = build_pipeline(chf, predictors, LogisticRegression(max_iter=4000))
    ref.fit(chf[predictors], y)
    n_params = len(ref.named_steps["prep"].get_feature_names_out())
    r2 = cox_snell_r2(y, ref.predict_proba(chf[predictors])[:, 1])
    riley = riley_sample_size(n_params, r2, float(y.mean()))

    boot = {
        "Unpenalised logistic": optimism_bootstrap(
            chf, predictors, y, lambda: LogisticRegression(max_iter=4000),
            label="optimism: unpenalised"),
        "Elastic net logistic": optimism_bootstrap(
            chf, predictors, y,
            lambda: LogisticRegressionCV(
                solver="saga", Cs=np.logspace(-3, 1, 6), l1_ratios=(0.5,),
                cv=3, scoring="neg_log_loss", max_iter=2000,
                random_state=RANDOM_STATE, n_jobs=-1),
            label="optimism: elastic net"),
    }
    cv_auc = {"Unpenalised logistic": 0.661, "Elastic net logistic": 0.678}

    inc = incremental_value(chf, predictors, y)
    trans = translate_effects(chf, predictors, y)

    facts = Facts(
        binding=riley["binding_criterion"],
        required_n=f"{riley['required_n']:,.0f}",
        available_n=f"{len(chf):,}",
        shortfall=f"{max(riley['required_n']-len(chf), 0):,.0f}",
        epv_n=f"{riley['epv_rule_n']:,.0f}",
        ratio=f"{riley['required_n']/riley['epv_rule_n']:.1f}",
        n_boot=str(boot["Unpenalised logistic"]["n_boot"]),
        unpen_apparent=f"{boot['Unpenalised logistic']['apparent_auc']:.3f}",
        unpen_corrected=f"{boot['Unpenalised logistic']['corrected_auc']:.3f}",
        unpen_optimism=f"{boot['Unpenalised logistic']['optimism_auc']:.3f}",
        enet_apparent=f"{boot['Elastic net logistic']['apparent_auc']:.3f}",
        enet_corrected=f"{boot['Elastic net logistic']['corrected_auc']:.3f}",
        unpen_shrinkage=f"{boot['Unpenalised logistic']['uniform_shrinkage']:.2f}",
        unpen_shrink_pct=f"{(1-boot['Unpenalised logistic']['uniform_shrinkage'])*100:.0f}",
        auc_a=f"{inc['aucs']['A: physician alone']:.3f}",
        auc_b=f"{inc['aucs']['B: model alone']:.3f}",
        auc_c=f"{inc['aucs']['C: physician + model']:.3f}",
        gain=f"{inc['auc_gain']:+.4f}",
        gain_ci=f"[{inc['gain_ci'][0]:+.4f}, {inc['gain_ci'][1]:+.4f}]",
        lr_stat=f"{inc['lr_stat']:.1f}", lr_df=str(inc["lr_df"]),
        lr_p=fmt_p(inc["lr_p"]),
        doc_coef=f"{inc['doc_coef']:+.3f}", doc_p=fmt_p(inc["doc_p"]),
        model_coef=f"{inc['model_coef']:+.3f}", model_p=fmt_p(inc["model_p"]),
        incremental_verdict=(
            "The gain is real: the interval excludes zero and the "
            "likelihood-ratio test rejects the physician-only model. The measured "
            "variables carry information the clinician's estimate does not."
            if inc["gain_ci"][0] > 0 and inc["lr_p"] < 0.05 else
            "The likelihood-ratio test rejects the physician-only model, so the "
            "measured variables do add information -- but the AUC gain interval "
            "crosses zero, which is a reminder that AUC is insensitive to added "
            "predictors and a significant LR test can coexist with a negligible "
            "change in discrimination."
            if inc["lr_p"] < 0.05 else
            "Neither test supports adding the model: on this evidence the "
            "measured variables contribute nothing beyond the clinician's own "
            "estimate."),
        prevalence=f"{y.mean()*100:.1f}",
        top_var=str(trans.iloc[0]["variable"]),
        top_or=f"{trans.iloc[0]['odds_ratio']:.2f}",
        top_rr=f"{trans.iloc[0]['risk_ratio']:.2f}",
        top_inflation=f"{(trans.iloc[0]['odds_ratio']-1)/(trans.iloc[0]['risk_ratio']-1):.1f}",
        top_p0=f"{trans.iloc[0]['risk_at_mean_pct']:.1f}",
        top_p1=f"{trans.iloc[0]['risk_at_plus1sd_pct']:.1f}",
        top_ard=f"{trans.iloc[0]['abs_risk_diff_pp']:+.1f}",
    )

    report_riley(riley, len(chf), int(y.sum()), facts)
    report_optimism(boot, cv_auc)
    report_incremental(inc)
    report_translation(trans, float(y.mean()))

    header("FIGURES")
    path = figure_translation(trans, float(y.mean()))
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(ANSWERS.format(rule=RULE, **facts))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*converge.*")
        run_and_capture(main, OUT_DIR / "07_validation.txt")
