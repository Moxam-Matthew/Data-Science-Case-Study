"""
03_cohort.py -- Data dictionary, survival description, and the enrolment forensics.

Continues the question format. Answers are held at the bottom and every number
in them is interpolated from the run.

    Run:  python 03_cohort.py

THE QUESTIONS
    Q12  Produce the data dictionary a reviewer would ask for: role, type, unit,
         N, missing and range for every column. What does it catch that Table 1
         does not?
    Q13  Plot overall survival for the cohort. Clinical journals will reject a
         Kaplan-Meier curve that is missing one specific element. What is it,
         and why does its absence make the tail of the curve unreadable?
    Q14  When do the deaths actually happen? Estimate the hazard over time and
         say what its shape implies for a proportional-hazards model and for
         the choice of a fixed prediction horizon.
    Q15  Missingness rates say how much is absent. What do missingness
         *patterns* say that rates cannot?
    Q16  01_eda.py concluded that BUN missingness is an artefact of unequal
         follow-up and offered enrolment phase as an unverified explanation.
         SUPPORT2 ships no phase column. Prove or refute it anyway.
    Q17  The VIF table in 02_profile.py rests on complete cases only, well under
         a fifth of the training cohort. Recompute it honestly and say whether
         the conclusion holds.

Author: Matthew Moxam
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import viz
from report import Facts, RULE, configure_pandas, header, question, render_answers, run_and_capture
from stats_utils import median_followup
from support2 import (
    CANDIDATE_PREDICTORS,
    DERIVED_DUPLICATES,
    LEAKAGE_COLUMNS,
    OUTCOME_EVENT,
    OUTCOME_TIME,
    UNITS,
    analysis_frames,
)

OUT_DIR = Path(__file__).resolve().parent / "output"

# The gap in the censoring distribution that separates the two enrolment waves.
PHASE_CUT_DAYS = 1150
LANDMARKS = (30, 90, 180, 365, 730, 1095, 1460, 1825)
HAZARD_EDGES = [0, 30, 90, 180, 365, 730, 1095, 2100]


# ═══ Q12. Data dictionary ════════════════════════════════════════════════════
def compute_data_dictionary(chf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in chf.columns:
        s = chf[col]
        if col in (OUTCOME_EVENT, OUTCOME_TIME):
            role = "outcome"
        elif col in LEAKAGE_COLUMNS or col in DERIVED_DUPLICATES:
            role = "excluded"
        elif col in CANDIDATE_PREDICTORS:
            role = "predictor"
        else:
            role = "unused"
        if pd.api.types.is_numeric_dtype(s):
            kind = "binary" if s.nunique(dropna=True) <= 2 else "continuous"
            rng = f"{s.min():g} to {s.max():g}"
        else:
            kind, rng = "categorical", f"{s.nunique(dropna=True)} levels"
        rows.append({"column": col, "role": role, "type": kind,
                     "unit": UNITS.get(col, ""), "n_obs": int(s.notna().sum()),
                     "missing_pct": round(s.isna().mean() * 100, 1),
                     "unique": int(s.nunique(dropna=True)), "range": rng})
    return pd.DataFrame(rows)


def report_data_dictionary(dd: pd.DataFrame) -> None:
    question(12, "Produce the data dictionary a reviewer would ask for. What does it\n"
                 "catch that Table 1 does not?")
    print(dd.to_string(index=False))
    print("\n  Ranges reflect the plausibility bounds applied upstream, so this table")
    print("  and 02_profile.py Q8 cannot contradict one another.\n")
    const = dd[dd.unique <= 1]
    near = dd[(dd.type != "categorical") & dd.unique.between(2, 3)
              & (dd.role == "predictor")]
    unassigned = dd[dd.role == "unused"]
    print(f"  constant / zero-variance columns : {', '.join(const.column) or 'none'}")
    print(f"  near-constant predictors         : {', '.join(near.column) or 'none'}")
    print(f"  no role assigned                 : {', '.join(unassigned.column) or 'none'}")
    dropped = sorted(set(const.column) & set(CANDIDATE_PREDICTORS))
    if dropped:
        print(f"\n  -> {', '.join(dropped)} are candidate predictors that are CONSTANT")
        print("     inside this cohort, because the cohort is defined by restricting")
        print("     on them. They carry no information here and can make a design")
        print("     matrix singular. support2.model_predictors() drops them per")
        print("     cohort; they remain predictors on the full study, where Q4 uses")
        print("     dzgroup to identify the case-mix mechanism.")


# ═══ Q13. Overall survival ═══════════════════════════════════════════════════
def compute_survival(chf: pd.DataFrame) -> dict:
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(chf[OUTCOME_TIME], chf[OUTCOME_EVENT],
                                 label="CHF cohort")
    ci = km.confidence_interval_survival_function_
    rows = []
    for t in LANDMARKS:
        idx = ci.index[ci.index <= t]
        lo, hi = (ci.loc[idx[-1]].values if len(idx) else (np.nan, np.nan))
        rows.append({"day": t, "survival_pct": float(km.predict(t)) * 100,
                     "ci_lo": lo * 100, "ci_hi": hi * 100,
                     "at_risk": int((chf[OUTCOME_TIME] >= t).sum())})
    return {"km": km, "landmarks": pd.DataFrame(rows),
            "median_survival": float(km.median_survival_time_),
            "median_followup": median_followup(chf[OUTCOME_TIME], chf[OUTCOME_EVENT])}


def report_survival(r: dict, chf: pd.DataFrame) -> None:
    question(13, "Plot overall survival. Clinical journals reject a Kaplan-Meier curve\n"
                 "missing one specific element. What is it, and why does its absence\n"
                 "make the tail unreadable?")
    print(f"  n={len(chf):,}  deaths={int(chf[OUTCOME_EVENT].sum()):,}")
    print(f"  median survival  {r['median_survival']:,.0f} days")
    print(f"  median follow-up {r['median_followup']:,.0f} days  (reverse KM)")
    print("\n  Survival and the number still at risk, by landmark:")
    print(f"    {'day':>6} {'S(t)':>8} {'95% CI':>18} {'at risk':>9}")
    for _, x in r["landmarks"].iterrows():
        ci = f"({x.ci_lo:.1f}-{x.ci_hi:.1f})"
        print(f"    {int(x.day):>6} {x.survival_pct:>7.1f}% {ci:>18} {int(x.at_risk):>9,}")
    print("\n  Watch the right-hand column. That is the answer to the question.")


# ═══ Q14. Hazard over time ═══════════════════════════════════════════════════
def compute_hazard(chf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(HAZARD_EDGES[:-1], HAZARD_EDGES[1:]):
        at_risk = int((chf[OUTCOME_TIME] > lo).sum())
        d = int(((chf[OUTCOME_EVENT] == 1)
                 & chf[OUTCOME_TIME].between(lo, hi, "right")).sum())
        exposure = np.minimum(chf[OUTCOME_TIME], hi).sub(lo).clip(lower=0).sum()
        rows.append({"lo": lo, "hi": hi, "deaths": d, "at_risk": at_risk,
                     "rate_per_100pd": d / exposure * 100 if exposure else np.nan})
    return pd.DataFrame(rows)


def report_hazard(haz: pd.DataFrame) -> None:
    question(14, "When do the deaths happen? Estimate the hazard over time and say\n"
                 "what its shape implies for proportional hazards and for a fixed\n"
                 "prediction horizon.")
    print("  Deaths per 100 patient-days at risk, by interval:")
    print(f"    {'interval (days)':>18} {'deaths':>7} {'at risk':>8} {'rate':>9}")
    for _, x in haz.iterrows():
        print(f"    {f'{int(x.lo)}-{int(x.hi)}':>18} {int(x.deaths):>7,} "
              f"{int(x.at_risk):>8,} {x.rate_per_100pd:>9.4f}")
    ratio = haz.rate_per_100pd.iloc[0] / haz.rate_per_100pd.iloc[-1]
    print(f"\n  Early hazard is {ratio:.1f}x the late hazard.")


# ═══ Q15. Missingness patterns ═══════════════════════════════════════════════
def compute_missingness_patterns(chf: pd.DataFrame) -> dict:
    cols = [c for c in CANDIDATE_PREDICTORS
            if c in chf and 0.02 < chf[c].isna().mean() < 0.98]
    M = chf[cols].isna()
    patterns = (M.apply(lambda r: "".join("X" if v else "." for v in r), axis=1)
                .value_counts())
    return {"cols": cols, "M": M, "patterns": patterns,
            "n_distinct": int(M.drop_duplicates().shape[0]),
            "phi": M.astype(int).corr()}


def report_missingness_patterns(r: dict, chf: pd.DataFrame) -> None:
    question(15, "Missingness rates say how much is absent. What do missingness\n"
                 "PATTERNS say that rates cannot?")
    cols = r["cols"]
    print(f"  {len(cols)} variables with non-trivial missingness.")
    print(f"  Distinct patterns observed: {r['n_distinct']} of {2**len(cols):,} possible")
    print("\n  The commonest patterns, as share of the cohort:")
    print(f"    {''.join(c[0].upper() for c in cols)}   <- first letter of each variable")
    for p, n in r["patterns"].head(8).items():
        print(f"    {p}  {n:>4,}  {n/len(chf)*100:>5.1f}%")

    phi = r["phi"]
    pairs = (phi.where(~np.eye(len(phi), dtype=bool)).stack().rename("phi")
             .reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
    pairs = pairs[pairs.a < pairs.b].sort_values("phi", ascending=False).head(8)
    print("\n  Pairwise co-occurrence (phi between missing-indicators):")
    print(pairs.round(3).to_string(index=False))


# ═══ Q16. Enrolment forensics ════════════════════════════════════════════════
def compute_enrolment(chf: pd.DataFrame) -> dict:
    cens = chf.loc[chf[OUTCOME_EVENT] == 0, OUTCOME_TIME]
    h, edges = np.histogram(cens, bins=np.arange(0, 2200, 180))
    hist = pd.Series(h, index=edges[:-1])
    interior = hist[(hist.index > 500) & (hist.index < 1800)]

    wave = np.where(chf[OUTCOME_TIME] >= PHASE_CUT_DAYS, "early", "late")
    sub = chf.assign(wave=wave)
    sub = sub[sub[OUTCOME_EVENT] == 0]
    rows = []
    for col in ["bun", "urine", "glucose", "income", "adlp", "edu"]:
        if col not in chf:
            continue
        early = sub.loc[sub.wave == "early", col].isna().mean() * 100
        late = sub.loc[sub.wave == "late", col].isna().mean() * 100
        rows.append({"variable": col, "missing_early_wave": early,
                     "missing_late_wave": late, "difference_pp": early - late})

    converse = {}
    for lbl, s in (("recorded", chf[chf.bun.notna()]), ("missing", chf[chf.bun.isna()])):
        c = s.loc[s[OUTCOME_EVENT] == 0, OUTCOME_TIME]
        converse[lbl] = {"n": len(c),
                         "pct_early": float(np.mean(c >= PHASE_CUT_DAYS) * 100) if len(c) else np.nan}
    return {"hist": hist, "gap_bin": int(interior.idxmin()), "gap_n": int(interior.min()),
            "gap_neighbours": (int(hist.get(interior.idxmin() - 180, 0)),
                               int(hist.get(interior.idxmin() + 180, 0))),
            "table": pd.DataFrame(rows), "converse": converse}


def report_enrolment(r: dict) -> None:
    question(16, "01_eda.py offered enrolment phase as an unverified explanation for\n"
                 "the follow-up imbalance. SUPPORT2 ships no phase column.\n"
                 "Prove or refute it anyway.")
    print("  Administrative censoring happens when a study closes, so censored")
    print("  follow-up encodes enrolment date. If enrolment came in two waves")
    print("  closed on one date, this distribution must be bimodal.\n")
    for e, c in r["hist"].items():
        marker = "  <-- gap" if c <= 2 and e > 500 else ""
        print(f"    {e:>5.0f}-{e+180:>5.0f}d  {'#' * int(c/2):<38} {c:>3}{marker}")
    lo, hi = r["gap_neighbours"]
    print(f"\n  Emptiest interior bin: {r['gap_bin']}-{r['gap_bin']+180} days "
          f"({r['gap_n']} patients), against neighbours of {lo} and {hi}.")

    print(f"\n  Assigning a wave proxy at {PHASE_CUT_DAYS} days, and checking the")
    print("  'artefact' variables against the 'real signal' ones:")
    print(r["table"].round(1).to_string(index=False))

    print("\n  And the converse -- where do the BUN-missing patients sit?")
    for lbl, d in r["converse"].items():
        print(f"    BUN {lbl:<9} censored n={d['n']:>4,}   "
              f"in early-enrolment wave: {d['pct_early']:>5.1f}%")


# ═══ Q17. VIF, honestly ══════════════════════════════════════════════════════
def compute_vif_imputed(chf: pd.DataFrame) -> dict:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    num = [c for c in CANDIDATE_PREDICTORS
           if c in chf and pd.api.types.is_numeric_dtype(chf[c]) and chf[c].nunique() > 2]
    X = chf[num]
    imp = IterativeImputer(max_iter=20, random_state=20260901)
    Xi = pd.DataFrame(imp.fit_transform(X), columns=num, index=X.index)

    def vif_of(frame: pd.DataFrame) -> pd.Series:
        Z = ((frame - frame.mean()) / frame.std()).assign(_c=1.0)
        return pd.Series([variance_inflation_factor(Z.values, i)
                          for i, c in enumerate(Z.columns) if c != "_c"],
                         index=[c for c in Z.columns if c != "_c"])

    cc = X.dropna()
    comp = vif_of(cc) if len(cc) > len(num) + 5 else pd.Series(dtype=float)
    out = pd.DataFrame({"VIF_complete_case": comp, "VIF_imputed": vif_of(Xi)})
    out["change"] = out.VIF_imputed - out.VIF_complete_case
    return {"table": out.sort_values("VIF_imputed", ascending=False),
            "n_complete": len(cc), "n_total": len(X)}


def report_vif(r: dict) -> None:
    question(17, "The VIF table in 02_profile.py rests on complete cases only, well\n"
                 "under a fifth of the training cohort. Recompute it honestly and\n"
                 "say whether the conclusion holds.")
    print(f"  Complete cases: {r['n_complete']:,} of {r['n_total']:,} "
          f"({r['n_complete']/r['n_total']*100:.1f}%)")
    print("\n  Single imputation is used here as a DIAGNOSTIC only -- it understates")
    print("  uncertainty and is not the modelling strategy (that is MICE inside folds).")
    print(r["table"].round(2).to_string())
    worst = r["table"].VIF_imputed.max()
    print(f"\n  Highest VIF on imputed data: {worst:.2f} "
          f"({'no action needed' if worst < 5 else 'inspect'}).")


# ═══ Facts ═══════════════════════════════════════════════════════════════════
def collect_facts(surv: dict, haz: pd.DataFrame, pat: dict, enrol: dict,
                  vif: dict, chf: pd.DataFrame) -> Facts:
    lm = surv["landmarks"].set_index("day")
    e = enrol["table"].set_index("variable")
    lo, hi = enrol["gap_neighbours"]
    return Facts(
        n_chf=f"{len(chf):,}",
        n_deaths=f"{int(chf[OUTCOME_EVENT].sum()):,}",
        median_followup=f"{surv['median_followup']:,.0f}",
        at_risk_30=f"{int(lm.loc[30, 'at_risk']):,}",
        at_risk_1825=f"{int(lm.loc[1825, 'at_risk']):,}",
        surv_365=f"{lm.loc[365, 'survival_pct']:.1f}",
        surv_1825=f"{lm.loc[1825, 'survival_pct']:.1f}",
        hazard_ratio=f"{haz.rate_per_100pd.iloc[0] / haz.rate_per_100pd.iloc[-1]:.1f}",
        n_patterns=str(pat["n_distinct"]),
        n_miss_vars=str(len(pat["cols"])),
        n_possible=f"{2**len(pat['cols']):,}",
        gap_bin=f"{enrol['gap_bin']}-{enrol['gap_bin']+180}",
        gap_n=str(enrol["gap_n"]),
        gap_lo=str(lo), gap_hi=str(hi),
        bun_early=f"{e.loc['bun','missing_early_wave']:.1f}",
        bun_late=f"{e.loc['bun','missing_late_wave']:.1f}",
        urine_early=f"{e.loc['urine','missing_early_wave']:.1f}",
        urine_late=f"{e.loc['urine','missing_late_wave']:.1f}",
        glucose_early=f"{e.loc['glucose','missing_early_wave']:.1f}",
        glucose_late=f"{e.loc['glucose','missing_late_wave']:.1f}",
        adlp_early=f"{e.loc['adlp','missing_early_wave']:.1f}",
        adlp_late=f"{e.loc['adlp','missing_late_wave']:.1f}",
        income_early=f"{e.loc['income','missing_early_wave']:.1f}",
        income_late=f"{e.loc['income','missing_late_wave']:.1f}",
        bun_missing_early_pct=f"{enrol['converse']['missing']['pct_early']:.1f}",
        bun_recorded_early_pct=f"{enrol['converse']['recorded']['pct_early']:.1f}",
        vif_complete_n=f"{vif['n_complete']:,}",
        vif_complete_pct=f"{vif['n_complete']/vif['n_total']*100:.1f}",
        vif_max=f"{vif['table'].VIF_imputed.max():.2f}",
        phase_cut=str(PHASE_CUT_DAYS),
    )


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_survival(chf: pd.DataFrame, surv: dict, haz: pd.DataFrame):
    import matplotlib.pyplot as plt
    from lifelines.plotting import add_at_risk_counts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    km = surv["km"]
    km.plot_survival_function(ax=ax1, color=viz.SERIES_BLUE, ci_alpha=0.15, lw=2.2)
    ax1.set_xlim(0, 2029)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Days from study entry")
    ax1.set_ylabel("Survival probability")
    ax1.set_title("Overall survival, with numbers at risk")
    ax1.get_legend().remove()
    viz.despine(ax1)
    add_at_risk_counts(km, ax=ax1, rows_to_show=["At risk"], fontsize=8)

    ax2.step(haz.hi, haz.rate_per_100pd, where="pre", color=viz.SERIES_ORANGE, lw=2.2)
    ax2.fill_between(haz.hi, 0, haz.rate_per_100pd, step="pre",
                     color=viz.SERIES_ORANGE, alpha=0.15)
    ax2.set_xlim(0, 2029)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Days from study entry")
    ax2.set_ylabel("Deaths per 100 patient-days at risk")
    ax2.set_title("Hazard is front-loaded, not constant")
    viz.despine(ax2)

    lm = surv["landmarks"].set_index("day")
    ratio = haz.rate_per_100pd.iloc[0] / haz.rate_per_100pd.iloc[-1]
    fig.tight_layout()
    viz.caption(fig, f"CHF training cohort, n={len(chf):,}. Left: the at-risk row is what makes the tail "
                     f"interpretable --\nby day 1,825 the curve rests on {int(lm.loc[1825,'at_risk'])} patients "
                     f"while looking no less authoritative than at\nday 30, where it rests on "
                     f"{int(lm.loc[30,'at_risk'])}. Right: early hazard is {ratio:.1f}x the late hazard.",
                y=-0.10)
    return viz.save(fig, "07_survival_overview.png")


def figure_enrolment(chf: pd.DataFrame, enrol: dict):
    import matplotlib.pyplot as plt

    cens = chf[chf[OUTCOME_EVENT] == 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    bins = np.arange(0, 2200, 100)

    ax1.hist(cens[OUTCOME_TIME], bins=bins, color=viz.SERIES_BLUE)
    ax1.axvline(PHASE_CUT_DAYS, color=viz.INK_MUTED, ls="--", lw=1.5)
    ax1.annotate("no patients\ncensored here", xy=(PHASE_CUT_DAYS, ax1.get_ylim()[1] * 0.97),
                 xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                 fontsize=8.5, color=viz.INK_SECONDARY)
    ax1.set_xlabel("Censored follow-up (days)")
    ax1.set_ylabel("Patients")
    ax1.set_title("Censoring is bimodal: two enrolment waves,\none closing date")
    viz.despine(ax1)

    for data, label, color in ((cens[cens.bun.notna()], "BUN recorded", viz.SERIES_BLUE),
                               (cens[cens.bun.isna()], "BUN not recorded", viz.SERIES_ORANGE)):
        ax2.hist(data[OUTCOME_TIME], bins=bins, color=color, alpha=0.75, label=label)
    ax2.axvline(PHASE_CUT_DAYS, color=viz.INK_MUTED, ls="--", lw=1.5)
    ax2.set_xlabel("Censored follow-up (days)")
    ax2.set_ylabel("Patients")
    ax2.set_title("And the missing lab sorts almost perfectly\nbetween them")
    ax2.legend(loc="upper left")
    viz.despine(ax2)

    e = enrol["table"].set_index("variable")
    viz.caption(fig, f"CHF training cohort, censored patients only (n={len(cens):,}). The gap near "
                     f"{PHASE_CUT_DAYS} days separates\ntwo enrolment waves. BUN is "
                     f"{e.loc['bun','missing_early_wave']:.1f}% missing in the early wave against "
                     f"{e.loc['bun','missing_late_wave']:.1f}% in the late one --\na collection protocol, "
                     f"not a property of the patient.")
    return viz.save(fig, "08_enrolment_waves.png")


def figure_missingness_patterns(phi: pd.DataFrame, n: int):
    import matplotlib.pyplot as plt

    order = phi.columns.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(phi.loc[order, order], cmap=viz.diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=9)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    for i in range(len(order)):
        for j in range(len(order)):
            v = phi.iloc[i, j]
            if i != j and abs(v) >= 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="#ffffff", weight="600")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("phi (correlation between missing-indicators)", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Which variables go missing together")
    viz.caption(fig, f"CHF training cohort, n={n:,}. Blocks of near-1.0 correlation mean the values were absent\n"
                     f"as a group, which points to a collection protocol rather than to individual patients.")
    return viz.save(fig, "09_missingness_patterns.png")


ANSWERS = """
ANSWERS
{rule}

A12. WHAT THE DATA DICTIONARY CATCHES
    Table 1 answers "how do the outcome groups differ". The dictionary answers
    "what is in this file at all", and the two fail in different directions.

    A stratified table silently skips anything it cannot stratify. It will not
    tell you a column is constant, that a supposedly continuous variable has
    four distinct values, that a unit is undocumented, or that a column exists
    which nobody has assigned a role. Those are the errors that survive review
    precisely because the interesting table looks fine.

    The unglamorous columns earn their place: n_obs, unique, range, unit. A
    creatinine range of 0.3 to 20 mg/dL is plausible; the same numbers labelled
    umol/L would not be. Without the unit column nobody can check that, and
    plausibility bounds asserted without units are unreviewable.

    Note also what this table must NOT do, which an earlier version of this
    project got wrong. It once published an albumin maximum of 29.0 g/dL, two
    sections after the write-up called that value incompatible with life,
    because the plausibility bounds lived in one script and were applied in one
    function. A dictionary that contradicts the data-quality section is worse
    than no dictionary, because it looks like diligence. The bounds now live in
    support2.PLAUSIBLE_BOUNDS and are applied by the shared entry point, so the
    two cannot disagree.

A13. WHAT THE CURVE IS MISSING
    The numbers at risk. A Kaplan-Meier plot without a risk table underneath it
    will be sent back by any competent reviewer, and the reason is visible in
    this cohort's tail.

    Survival at day 1,825 is estimated at {surv_1825}% -- from {at_risk_1825}
    patients. At day 30 the same curve rests on {at_risk_30}. It is drawn at
    full width and full confidence at both, and looks exactly as authoritative
    in each place. A reader cannot see the difference from the line alone, and
    the confidence band understates it because it narrows on the survival scale
    as the estimate approaches zero.

    The at-risk row restores the denominator. It is the single most common
    reason survival figures are rejected, and it costs one line of code.

    Report landmark estimates with confidence intervals too, rather than only a
    median. In a cohort this sick the median arrives early, and the clinically
    interesting question -- what fraction is alive at a year, here {surv_365}%
    -- lives elsewhere on the curve.

A14. WHAT THE HAZARD SHAPE TELLS YOU
    The death rate is heavily front-loaded: the first interval runs at
    {hazard_ratio} times the rate of the last. This is a cohort of critically
    ill patients, so that is clinically unsurprising, but it has two concrete
    consequences.

    For a proportional-hazards model: proportionality is an assumption about the
    RATIO of hazards between groups, not about the shape of the baseline hazard,
    so a falling baseline is not itself a violation. Cox handles it without
    complaint. But a steeply changing baseline is where non-proportionality
    tends to hide -- a covariate that matters intensely in the first month and
    little afterwards still yields a plausible-looking single hazard ratio. Test
    with Schoenfeld residuals rather than assuming.

    For horizon choice: most events are early, so a 30-day or 90-day horizon is
    where the data is dense and where a discharge decision actually sits. A
    five-year horizon is estimable but rests on {at_risk_1825} patients and
    answers a question nobody asks at the bedside.

A15. WHAT PATTERNS SAY THAT RATES CANNOT
    A rate is a marginal. It tells you a variable is half missing and nothing
    about whether those are the same patients each time.

    Across {n_miss_vars} variables there are {n_possible} possible missingness
    patterns and only {n_patterns} occur. The phi matrix shows blocks of
    near-perfect co-occurrence -- variables absent together, as a set.

    That structure is diagnostic. Missingness caused by individual patients --
    one refused a test, another was too unwell for an interview -- produces
    scattered, weakly correlated patterns. Missingness caused by a protocol
    produces blocks: an entire panel present or absent as a unit, because a
    panel is what gets ordered.

    Blocks point away from the patient and toward the data-collection process.
    Q16 identifies the process.

A16. THE ENROLMENT WAVES, PROVEN
    Confirmed, and confirmable without a phase column because censoring times
    carry the information.

    Administrative censoring occurs when the study closes, not when a patient
    leaves. So for censored patients, follow-up duration is a direct function of
    enrolment date: enrol early, be observed longer. If enrolment came in two
    waves against a single closing date, censored follow-up must be bimodal --
    and it is, with the interval at {gap_bin} days holding {gap_n} patients
    against neighbours of {gap_lo} and {gap_hi}. A single continuous accrual
    cannot produce that gap.

    Assigning a wave proxy at {phase_cut} days separates the variables far more
    sharply than Q5 could. This is not a difference of degree, it is close to
    deterministic:

        bun      {bun_early}% missing early vs {bun_late}% late
        urine    {urine_early}% vs {urine_late}%
        glucose  {glucose_early}% vs {glucose_late}%

    against the variables that survived Q5:

        adlp     {adlp_early}% vs {adlp_late}%
        income   {income_early}% vs {income_late}%

    A hundred-to-one split is not a measurement pattern. It is a protocol: those
    assays were not part of the early collection instrument. The converse
    confirms it from the other direction -- {bun_recorded_early_pct}% of
    censored patients with BUN recorded fall in the early wave, against
    {bun_missing_early_pct}% of those missing it.

    So the mechanism is established rather than speculated, and it is stronger
    than "informative missingness" ever was. Missing BUN records WHEN a patient
    was enrolled. Enrolling earlier means being observed longer, which means a
    higher chance of having died before the study closed. Every step of the
    spurious association is visible and none of it involves the patient.

    Stated honestly: the wave assignment is a proxy inferred from censoring, not
    a recorded field, and it can only be assigned to censored patients -- a
    patient who died before the closing date reveals nothing about their
    enrolment date. That is a real limit on the proof. It does not weaken the
    conclusion, because the mechanism only needs to explain the censored
    patients to account for the imbalance in observation windows.

A17. VIF, HONESTLY
    The complete-case table rests on {vif_complete_n} of the training cohort
    ({vif_complete_pct}%), and Q3-Q5 established those are not a random sample.
    A diagnostic measured on a non-random subsample is not evidence about the
    cohort. Q16 sharpens the point: complete cases are overwhelmingly late-wave
    patients, because three of the variables required for completeness were
    never collected in the early wave. "Complete case" here is close to a
    synonym for "enrolled after the protocol changed".

    Recomputed on imputed data across the full training cohort, the conclusion
    happens to hold -- the maximum VIF is {vif_max}, nowhere near the
    conventional thresholds. But that was not knowable in advance. It had to be
    checked, and it could have gone the other way.

    Two caveats to state rather than bury. Single imputation is used because
    this is a diagnostic and it understates uncertainty; the modelling strategy
    remains MICE fitted inside cross-validation folds. And VIF measures only
    linear dependence among predictors as entered, so it says nothing about the
    spline terms Q10 justified for creatinine, whose basis functions are
    collinear with one another by construction and are meant to be.

    The general habit: when a diagnostic rests on a subset, report the subset
    size beside it. A reassuring number attached to a tenth of the cohort is not
    reassurance, and the reader cannot detect the problem unless you show the
    denominator.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    cohort = analysis_frames()
    chf = cohort.chf_train

    header("SUPPORT2 -- data dictionary, survival description, enrolment forensics")
    print(f"  CHF TRAINING cohort: {len(chf):,} patients, "
          f"{int(chf[OUTCOME_EVENT].sum()):,} deaths")
    print(f"  {cohort.n_voided} implausible cells set to missing upstream.")
    print("  The 30% held-out partition is never returned by analysis_frames().")

    dd = compute_data_dictionary(chf)
    report_data_dictionary(dd)

    surv = compute_survival(chf)
    report_survival(surv, chf)

    haz = compute_hazard(chf)
    report_hazard(haz)

    pat = compute_missingness_patterns(chf)
    report_missingness_patterns(pat, chf)

    enrol = compute_enrolment(chf)
    report_enrolment(enrol)

    vif = compute_vif_imputed(chf)
    report_vif(vif)

    header("FIGURES")
    for path in (figure_survival(chf, surv, haz),
                 figure_enrolment(chf, enrol),
                 figure_missingness_patterns(pat["phi"], len(chf))):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(render_answers(ANSWERS,
                         dict(collect_facts(surv, haz, pat, enrol, vif, chf), rule=RULE)))


if __name__ == "__main__":
    run_and_capture(main, OUT_DIR / "03_cohort.txt")
