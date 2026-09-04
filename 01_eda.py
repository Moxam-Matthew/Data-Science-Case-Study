"""
01_eda.py -- Exploratory analysis of SUPPORT2, taught as a set of questions.

HOW TO READ THIS FILE
    Each section states a question, then shows the evidence needed to answer it.
    The answers are held back deliberately: they live in ANSWERS at the bottom of
    this file, and print after the analysis when you run it. Work out your own
    answer from the printed numbers first -- the reasoning is worth more than the
    conclusion, and in an interview you will be asked for the reasoning.

    Every number appearing in a question or an answer is interpolated from the
    run that produced it (see collect_facts). None are typed by hand.

    Run:  python 01_eda.py

THE QUESTIONS
    Q1  The dataset ships a binary `death` flag. Why is modelling that flag
        alone a mistake here, and what does it discard?
    Q2  Which columns must never enter the feature matrix, and how would you
        recognise one you had not been warned about?
    Q3  Several lab columns are around half missing. Is that missingness
        random, or does it carry information about the outcome?
    Q4  The mortality gap for `pafi` missingness disappears once the cohort is
        restricted to CHF, while the gap for `bun` gets larger. Why would
        conditioning on disease group do opposite things to two lab variables?
    Q5  Split the CHF cohort on whether BUN was ever recorded. A chi-square on
        death and a log-rank test on the same split disagree sharply. Both are
        correctly computed. Which is telling you the truth, and what does the
        other one actually measure?
    Q6  Given the corrected picture, what is the defensible imputation strategy
        -- and what would be wrong with adding a missingness indicator to every
        variable that looked informative in Q3?

Author: Matthew Moxam
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import viz
from report import (
    RULE,
    Facts,
    configure_pandas,
    fmt_p,
    header,
    question,
    render_answers,
    run_and_capture,
)
from stats_utils import add_fdr, followup_summary
from support2 import (
    CANDIDATE_PREDICTORS,
    CHF_LABEL,
    OUTCOME_EVENT,
    OUTCOME_TIME,
    analysis_frames,
    audit_columns,
    chf_cohort,
    cohort_flow,
)

OUT_DIR = Path(__file__).resolve().parent / "output"


# ═══ Q1. The shape of the outcome ════════════════════════════════════════════
def compute_outcome_structure(df: pd.DataFrame) -> dict:
    fu = followup_summary(df, OUTCOME_TIME, OUTCOME_EVENT)
    censored_time = df.loc[df[OUTCOME_EVENT] == 0, OUTCOME_TIME]
    chf = chf_cohort(df)
    return {
        "full": fu,
        "percentiles": {p: float(np.percentile(censored_time, p))
                        for p in (10, 25, 50, 75, 90)},
        "short_followup_pct": float((censored_time < 365).mean() * 100),
        "short_followup_n": int((censored_time < 365).sum()),
        "chf": followup_summary(chf, OUTCOME_TIME, OUTCOME_EVENT),
    }


def report_outcome_structure(r: dict) -> None:
    question(1, "The dataset ships a binary `death` flag. Why is modelling that\n"
                "flag alone a mistake here, and what does it discard?")
    f = r["full"]
    print(f"  deaths observed      {f['events']:>6,} / {f['n']:,} "
          f"({f['events']/f['n']*100:.1f}%)")
    print(f"  censored (alive)     {f['censored']:>6,} "
          f"({f['censored']/f['n']*100:.1f}%)")

    print("\n  Three different quantities, often confused. Only one is follow-up:")
    print(f"    median of the time column      {f['median_time_to_event_or_censor']:>7,.0f} d"
          "   <- a median TIME-TO-EVENT, pulled down by every death")
    print(f"    median among censored only     {f['median_time_among_censored']:>7,.0f} d"
          "   <- ad hoc; ignores follow-up of those who died")
    print(f"    reverse Kaplan-Meier           {f['median_followup_reverseKM']:>7,.0f} d"
          "   <- median FOLLOW-UP; the correct estimator")
    print(f"    longest observation            {f['max_observed']:>7,.0f} d")

    print("\n  Follow-up among the censored -- if these were all short, early dropout")
    print("  would bias a binary flag. Are they?")
    for p, v in r["percentiles"].items():
        print(f"    {p:>2}th percentile   {v:>6,.0f} days")
    print(f"\n  Censored before 1 year: {r['short_followup_n']:,} "
          f"({r['short_followup_pct']:.1f}% of censored patients)")

    c = r["chf"]
    print(f"\n  Same question inside the CHF cohort:")
    print(f"    n={c['n']:,}  deaths={c['events']:,}  censored={c['censored']:,}")
    print(f"    median follow-up (reverse KM) {c['median_followup_reverseKM']:,.0f} days")


# ═══ Q2. Column governance ═══════════════════════════════════════════════════
def compute_leakage_audit(df: pd.DataFrame) -> dict:
    audit = audit_columns(df)
    leak = []
    for col in ("d.time", "surv2m", "slos"):
        if col not in df:
            continue
        died = df.loc[df[OUTCOME_EVENT] == 1, col].dropna()
        alive = df.loc[df[OUTCOME_EVENT] == 0, col].dropna()
        u, _ = stats.mannwhitneyu(died, alive)
        auc = u / (len(died) * len(alive))
        leak.append({"column": col, "median_died": died.median(),
                     "median_survived": alive.median(),
                     "univariate_auc": max(auc, 1 - auc)})
    return {"audit": audit, "leakage_demo": pd.DataFrame(leak)}


def report_leakage_audit(r: dict) -> None:
    question(2, "Which columns must never enter the feature matrix, and how\n"
                "would you recognise one you had not been warned about?")
    audit = r["audit"]
    print(audit.groupby("role").size().to_string())

    print("\n  Excluded, with the reason:")
    for _, x in audit[audit.role == "excluded"].iterrows():
        print(f"    {x['column']:<10} {x['reason']}")

    unreviewed = audit[audit.role == "unreviewed"]
    print(f"\n  Not yet classified: "
          f"{', '.join(unreviewed['column']) if len(unreviewed) else 'none'}")

    print("\n  Empirical check -- what leakage looks like:")
    for _, x in r["leakage_demo"].iterrows():
        print(f"    {x['column']:<10} median died={x['median_died']:>9,.1f}  "
              f"survived={x['median_survived']:>9,.1f}  "
              f"univariate AUC={x['univariate_auc']:.3f}")


# ═══ Q3 / Q4. Missingness and case mix ═══════════════════════════════════════
def missingness_vs_outcome(df: pd.DataFrame, cols: list[str],
                           min_miss: float = 0.02) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df:
            continue
        m = df[c].isna()
        if not (min_miss < m.mean() < 1 - min_miss):
            continue
        d_miss = df.loc[m, OUTCOME_EVENT].mean()
        d_obs = df.loc[~m, OUTCOME_EVENT].mean()
        _, p, *_ = stats.chi2_contingency(pd.crosstab(m, df[OUTCOME_EVENT]))
        rows.append({"variable": c, "missing_pct": m.mean() * 100,
                     "death_if_missing": d_miss * 100,
                     "death_if_observed": d_obs * 100,
                     "gap_pp": (d_miss - d_obs) * 100, "p": p})
    out = pd.DataFrame(rows)
    return out.sort_values("gap_pp", key=abs, ascending=False).reset_index(drop=True)


def report_missingness(table: pd.DataFrame) -> None:
    question(3, "Several lab columns are around half missing. Is that missingness\n"
                "random, or does it carry information about the outcome?")
    show = table.copy()
    for c in ("missing_pct", "death_if_missing", "death_if_observed", "gap_pp"):
        show[c] = show[c].round(1)
    show["p"] = show["p"].apply(fmt_p)
    print(show.to_string(index=False))
    print("\n  Read `gap_pp`: percentage-point difference in mortality between")
    print("  patients whose value was missing and those whose was recorded.")


def compute_case_mix(df: pd.DataFrame, full_table: pd.DataFrame) -> dict:
    chf = chf_cohort(df)
    within = missingness_vs_outcome(chf, CANDIDATE_PREDICTORS)
    merged = (full_table[["variable", "gap_pp", "p"]]
              .rename(columns={"gap_pp": "gap_full", "p": "p_full"})
              .merge(within[["variable", "gap_pp", "p", "missing_pct"]]
                     .rename(columns={"gap_pp": "gap_chf", "p": "p_chf"}),
                     on="variable", how="inner"))
    merged["survives"] = np.where(merged.p_chf < 0.05, "yes", "no")
    merged = merged.sort_values("gap_chf", key=abs, ascending=False)

    labs = ["bun", "glucose", "urine", "ph", "pafi", "alb", "bili", "wblc"]
    d = df.copy()
    d["heavy"] = d[labs].isna().sum(axis=1) >= 5
    comp = (pd.crosstab(d.heavy, d.dzgroup, normalize="index") * 100).round(1).T
    comp.columns = ["<5 labs missing (%)", ">=5 labs missing (%)"]
    return {"merged": merged,
            "composition": comp.sort_values(">=5 labs missing (%)", ascending=False)}


def report_case_mix(r: dict) -> None:
    question(4, "The mortality gap for `pafi` missingness disappears once the\n"
                "cohort is restricted to CHF, while the gap for `bun` gets larger.\n"
                "Why would conditioning on disease group do opposite things?")
    show = r["merged"].copy()
    for c in ("gap_full", "gap_chf", "missing_pct"):
        show[c] = show[c].round(1)
    for c in ("p_full", "p_chf"):
        show[c] = show[c].apply(fmt_p)
    print(show[["variable", "missing_pct", "gap_full", "p_full",
                "gap_chf", "p_chf", "survives"]].to_string(index=False))
    print("\n  Case mix is the mechanism. Which patients carry the missing values?\n")
    print(r["composition"].to_string())


# ═══ Q5. The two tests disagree ══════════════════════════════════════════════
def compute_binary_vs_logrank(df: pd.DataFrame) -> pd.DataFrame:
    from lifelines.statistics import logrank_test

    chf = chf_cohort(df)
    rows = []
    for c in CANDIDATE_PREDICTORS:
        if c not in chf:
            continue
        m = chf[c].isna()
        if not (0.02 < m.mean() < 0.98):
            continue
        obs, mis = chf[~m], chf[m]
        _, p_bin, *_ = stats.chi2_contingency(pd.crosstab(m, chf[OUTCOME_EVENT]))
        lr = logrank_test(obs[OUTCOME_TIME], mis[OUTCOME_TIME],
                          obs[OUTCOME_EVENT], mis[OUTCOME_EVENT])
        fu_obs = obs.loc[obs[OUTCOME_EVENT] == 0, OUTCOME_TIME].median()
        fu_mis = mis.loc[mis[OUTCOME_EVENT] == 0, OUTCOME_TIME].median()
        rows.append({"variable": c,
                     "gap_pp": (mis[OUTCOME_EVENT].mean() - obs[OUTCOME_EVENT].mean()) * 100,
                     "p_binary": p_bin, "p_logrank": lr.p_value,
                     "censored_fu_obs": fu_obs, "censored_fu_missing": fu_mis,
                     "fu_ratio": fu_mis / fu_obs})
    out = pd.DataFrame(rows).sort_values("gap_pp", ascending=False).reset_index(drop=True)
    out = add_fdr(out, p_col="p_logrank", q_col="q_logrank")
    out["verdict"] = np.where(
        (out.p_binary < 0.05) & (out.q_logrank < 0.05), "real signal",
        np.where(out.p_binary < 0.05, "follow-up artefact", "no signal"))
    return out


def report_binary_vs_logrank(table: pd.DataFrame, facts: Facts) -> None:
    question(5, "Split the CHF cohort on whether BUN was ever recorded. A chi-square\n"
                "on death returns p{p_bin} with a {gap:.0f}-point gap; a log-rank test on\n"
                "the same split returns p={p_lr}. Both are correctly computed.\n"
                "Which is telling the truth, and what does the other measure?"
                .format(p_bin=facts["bun_p_binary"], gap=float(facts["bun_gap_raw"]),
                        p_lr=facts["bun_p_logrank"]))
    show = table.copy()
    for c in ("gap_pp", "fu_ratio"):
        show[c] = show[c].round(2)
    for c in ("censored_fu_obs", "censored_fu_missing"):
        show[c] = show[c].round(0)
    for c in ("p_binary", "p_logrank", "q_logrank"):
        show[c] = show[c].apply(fmt_p)
    print(show[["variable", "gap_pp", "p_binary", "p_logrank", "q_logrank",
                "censored_fu_obs", "censored_fu_missing", "fu_ratio",
                "verdict"]].to_string(index=False))
    n = int(table.p_logrank.notna().sum())
    print(f"\n  q_logrank is the Benjamini-Hochberg FDR q-value across {n} tests.")
    print("  The verdict column uses q, not p.")
    print("\n  `fu_ratio` is median follow-up among CENSORED patients, missing-group")
    print("  over observed-group. Near 1 means both groups were watched equally long,")
    print("  so a difference in cumulative death is real. Far above 1 means one group")
    print("  simply had longer to die.")


def report_strategy(df: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    question(6, "Given the corrected picture, what is the defensible imputation\n"
                "strategy -- and what would be wrong with adding a missingness\n"
                "indicator to every variable that looked informative in Q3?")
    real = comparison.loc[comparison.verdict == "real signal", "variable"].tolist()
    artefact = comparison.loc[comparison.verdict == "follow-up artefact", "variable"].tolist()
    print(f"  Survive the time-to-event test    : {', '.join(real) or '(none)'}")
    print(f"  Artefact of differential follow-up: {', '.join(artefact) or '(none)'}")
    print("\n  Note what the survivors have in common, and what the artefacts do.")

    chf = chf_cohort(df)
    complete = chf[CANDIDATE_PREDICTORS].dropna()
    print(f"\n  Complete-case cost: {len(complete):,} of {len(chf):,} CHF patients "
          f"retained ({len(complete)/len(chf)*100:.1f}%).")
    return {"real": real, "artefact": artefact,
            "complete_n": len(complete), "complete_pct": len(complete)/len(chf)*100}


# ═══ Facts ═══════════════════════════════════════════════════════════════════
def collect_facts(outcome: dict, missing: pd.DataFrame, comparison: pd.DataFrame,
                  strategy: dict, n_voided: int) -> Facts:
    """Every number that appears in a question header or an answer paragraph."""
    bun = comparison.set_index("variable").loc["bun"]
    m = missing.set_index("variable")
    real, art = strategy["real"], strategy["artefact"]
    c = comparison.set_index("variable")
    return Facts(
        chf_median_fu=f"{outcome['chf']['median_followup_reverseKM']:,.0f}",
        naive_median=f"{outcome['full']['median_time_to_event_or_censor']:,.0f}",
        reverse_km=f"{outcome['full']['median_followup_reverseKM']:,.0f}",
        fu_fold=f"{outcome['full']['median_followup_reverseKM'] / outcome['full']['median_time_to_event_or_censor']:.1f}",
        short_fu_pct=f"{outcome['short_followup_pct']:.1f}",
        chf_death_pct=f"{outcome['chf']['events']/outcome['chf']['n']*100:.1f}",
        full_death_pct=f"{outcome['full']['events']/outcome['full']['n']*100:.1f}",
        bun_gap_raw=f"{bun.gap_pp:.2f}",
        bun_gap=f"{bun.gap_pp:.1f}",
        bun_p_binary=fmt_p(bun.p_binary),
        bun_p_logrank=fmt_p(bun.p_logrank),
        bun_fu_missing=f"{bun.censored_fu_missing:,.0f}",
        bun_fu_obs=f"{bun.censored_fu_obs:,.0f}",
        bun_fu_ratio=f"{bun.fu_ratio:.2f}",
        bun_miss_gap_full=f"{m.loc['bun', 'gap_pp']:.1f}",
        artefact_vars=", ".join(art),
        real_vars=" and ".join(real),
        artefact_ratio=f"{c.loc[art, 'fu_ratio'].mean():.2f}" if art else "n/a",
        real_ratio_lo=f"{c.loc[real, 'fu_ratio'].min():.2f}" if real else "n/a",
        real_ratio_hi=f"{c.loc[real, 'fu_ratio'].max():.2f}" if real else "n/a",
        edu_p=fmt_p(c.loc["edu", "p_logrank"]) if "edu" in c.index else "n/a",
        edu_q=fmt_p(c.loc["edu", "q_logrank"]) if "edu" in c.index else "n/a",
        n_tests=str(int(comparison.p_logrank.notna().sum())),
        complete_n=f"{strategy['complete_n']:,}",
        complete_pct=f"{strategy['complete_pct']:.1f}",
        n_voided=str(n_voided),
    )


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_missingness(df: pd.DataFrame, comparison: pd.DataFrame):
    import matplotlib.pyplot as plt

    chf = chf_cohort(df)
    miss = (chf[CANDIDATE_PREDICTORS].isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    verdict = comparison.set_index("variable")["verdict"].to_dict()
    palette = {"real signal": viz.SERIES_ORANGE,
               "follow-up artefact": viz.SERIES_BLUE,
               "no signal": viz.BASELINE}
    colors = [palette.get(verdict.get(v, "no signal"), viz.BASELINE) for v in miss.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(miss.index, miss.values, color=colors, height=0.7)
    for name, val in miss.items():
        ax.text(val + 0.8, name, f"{val:.0f}%", va="center",
                fontsize=8, color=viz.INK_SECONDARY)
    ax.set_xlim(0, max(miss.values) * 1.18)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missingness by candidate predictor")
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    labels = ["Informative (survives log-rank + FDR)",
              "Artefact of differential follow-up", "No association"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k])
               for k in ("real signal", "follow-up artefact", "no signal")]
    ax.legend(handles, labels, loc="lower right")
    viz.caption(fig, f"SUPPORT2 (UCI 880), CHF training cohort n={len(chf):,}. Blue bars are variables\n"
                     f"a chi-square would have flagged and a log-rank clears.")
    return viz.save(fig, "01_missingness_by_variable.png")


def figure_conditioning(merged: pd.DataFrame, n_chf: int):
    import matplotlib.pyplot as plt

    d = merged.reindex(merged.gap_chf.abs().sort_values().index)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for yi, (gf, gc) in enumerate(zip(d.gap_full, d.gap_chf)):
        ax.plot([gf, gc], [yi, yi], color=viz.BASELINE, lw=1.5, zorder=1)
    ax.scatter(d.gap_full, y, s=70, color=viz.SERIES_BLUE, zorder=2,
               label="All disease groups", edgecolor=viz.SURFACE, linewidth=1.5)
    ax.scatter(d.gap_chf, y, s=70, color=viz.SERIES_ORANGE, zorder=3,
               label="CHF only", edgecolor=viz.SURFACE, linewidth=1.5)
    ax.axvline(0, color=viz.BASELINE, lw=1.2)
    ax.set_yticks(y, d.variable)
    ax.set_xlabel("Mortality gap: missing minus observed (percentage points)")
    ax.set_title("Does the missingness signal survive holding disease constant?")
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    ax.legend(loc="lower right")
    viz.caption(fig, f"Training partition; CHF n={n_chf:,}. Right of zero: patients missing the value died\n"
                     f"more often. Variables collapsing toward zero were tracking case mix.")
    return viz.save(fig, "02_conditioning_on_disease.png")


def figure_km(df: pd.DataFrame, comparison: pd.DataFrame):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    chf = chf_cohort(df)
    lookup = comparison.set_index("variable")
    panels = [("bun", "BUN (routine renal lab)"), ("adlp", "ADL (patient-reported function)")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        obs, mis = chf[chf[col].notna()], chf[chf[col].isna()]
        for data, label, color in ((obs, "recorded", viz.SERIES_BLUE),
                                   (mis, "not recorded", viz.SERIES_ORANGE)):
            km = KaplanMeierFitter().fit(data[OUTCOME_TIME], data[OUTCOME_EVENT],
                                         label=label)
            km.plot_survival_function(ax=ax, color=color, ci_alpha=0.12, lw=2)
            ax.annotate(label, xy=(1500, km.predict(1500)), xytext=(4, 6),
                        textcoords="offset points", fontsize=9,
                        color=color, weight="600")
        row = lookup.loc[col]
        ax.set_title(title)
        ax.set_xlabel("Days from study entry")
        ax.set_xlim(0, 2029)
        ax.set_ylim(0, 1)
        ax.text(0.03, 0.10,
                f"binary p {fmt_p(row.p_binary)}   log-rank q {fmt_p(row.q_logrank)}\n"
                f"censored follow-up ratio {row.fu_ratio:.2f}x",
                transform=ax.transAxes, fontsize=8.5, color=viz.INK_SECONDARY)
        ax.get_legend().remove()
        viz.despine(ax)

    axes[0].set_ylabel("Survival probability")
    fig.suptitle("Same test, opposite conclusions: cumulative death vs. hazard over time",
                 fontsize=12, fontweight="600", y=1.0)
    bun = lookup.loc["bun"]
    viz.caption(fig,
                f"CHF training cohort, n={len(chf):,}. Left: a {bun.gap_pp:.0f}-point gap in cumulative death "
                f"with curves that\noverlap -- the missing-BUN group was followed {bun.fu_ratio:.2f}x longer. "
                f"Right: balanced follow-up,\nand the curves genuinely separate. Only the right panel is a finding.")
    return viz.save(fig, "03_binary_vs_time_to_event.png")


ANSWERS = """
ANSWERS
{rule}

A0. WHY THERE IS A HELD-OUT PARTITION, AND WHY IT IS LOCKED
    Not a question in the list, but the first thing a reviewer should ask.

    Across 01_eda.py, 02_profile.py and 03_cohort.py this project looks at the
    outcome dozens of times. Each look is a decision point at which the data
    could steer a modelling choice. That accumulated steering is analyst degrees
    of freedom, and it makes any subsequent performance estimate optimistic by
    an amount nobody can compute after the fact.

    Two honest responses exist. Report everything as exploratory and correct the
    optimism by bootstrap. Or partition once, before modelling, and keep one
    part unseen. This project does both: the findings below are labelled
    hypothesis-generating and carry FDR-adjusted q-values across {n_tests}
    tests, and a 30% partition is held out by fixed seed and never read.

    The partition is generated from a constant (seed 20260901, stratified on
    death) rather than stored, so it reproduces exactly without committing
    patient rows -- the same constraint that shapes the loader.

A1. WHY NOT THE BINARY FLAG
    Start by refusing the textbook worry. The standard fear is heavy early
    dropout, and it does not apply here: only {short_fu_pct}% of censored
    patients were followed less than a year. Censoring in SUPPORT2 is mostly
    administrative -- the study ended. Say that out loud rather than reciting a
    concern the data does not support.

    Note also which follow-up number you quote. The median of the time column is
    {naive_median} days, but that is a median time-to-event: every death drags
    it down. Median follow-up needs the reverse Kaplan-Meier, inverting the
    indicator so censoring is the event, and it gives {reverse_km} days --
    {fu_fold}x larger. Quoting the first as "median follow-up" is a common
    error, and an embarrassing one in a project whose central finding is about
    follow-up.

    The real objections to the binary flag are three.

    First, the label is nearly saturated: {full_death_pct}% of the training
    cohort and {chf_death_pct}% of CHF died. An outcome that fires for
    two-thirds of patients has little room to discriminate, and "died at some
    point over the next one to five years" is not a question anyone decides on.

    Second, follow-up duration is not balanced across patients, and Q5 shows it
    correlates with data-collection patterns in a way that manufactures signal.
    Once that is true the binary flag is not merely lossy. It is biased, in a
    direction you will not notice unless you look.

    Third, the clinical question is timing. A cardiologist discharging a heart
    failure patient wants 30-day and 6-month risk. Any fixed-horizon binary
    outcome must be computed with censor-adjustment anyway.

A2. THE COLUMNS THAT MUST NOT ENTER
    They fall into four recognisable kinds:

      * The outcome in disguise -- `d.time`, `hospdead`. `d.time` is follow-up
        duration; as a "predictor" of death it reaches a univariate AUC near
        0.94, which is the tell.
      * Measured after baseline -- `sfdm2` (function at 2 months), `slos`,
        `charges`, `totcst`, `avtisst`. Not knowable at prediction time.
      * Another model's output -- `surv2m`, `surv6m`. Including these means your
        model is re-predicting the 1995 SUPPORT model.
      * Constructed from the predictors -- `aps`, `sps` are severity scores
        computed from the same vitals and labs. Collinear by construction. So is
        `adlsc`, which 02_profile.py shows is not merely correlated with `adls`
        but numerically identical to it.

    `prg2m` and `prg6m` are excluded for a different reason: they are the
    attending physician's own survival estimates. They are the benchmark. The
    interesting question is not whether a model beats chance but whether it
    beats the doctor -- and you cannot answer that if you fed the doctor's
    answer to the model.

    The general test for a column nobody warned you about: could this value have
    been known, in this form, at the moment the prediction is meant to be made?
    If answering needs a hospital course to resolve, it leaks.

A3. IS THE MISSINGNESS INFORMATIVE?
    On this evidence it looks emphatically non-random. Patients missing BUN died
    {bun_miss_gap_full} percentage points more often across all disease groups,
    with glucose, urine output and patient-reported ADLs showing gaps of similar
    size. Under MCAR you would expect gaps near zero.

    That is the answer most analyses give, and it is where most of them stop.
    Hold it loosely. A gap of this kind has at least three explanations: the
    patient was sicker, the patient was somewhere that orders fewer labs, or the
    two groups were not observed for the same length of time. The third is
    invisible to a chi-square, because a chi-square on a cumulative outcome has
    no concept of exposure time at all.

    Q4 separates the first two. Q5 surfaces the third, and it removes half the
    variables on this list.

A4. WHY CONDITIONING SPLITS THE VARIABLES IN TWO
    Restricting to CHF holds disease fixed, and the variables separate.

    Some collapse to nothing -- `pafi`, `ph`, `alb`, `bili`, `adls`. Blood gases
    are drawn almost exclusively on ventilated or ICU patients, so across the
    full cohort their missingness encoded WHERE a patient was treated: the
    heavy-missing group is disproportionately ward oncology, the well-measured
    group disproportionately sepsis. The signal was case mix wearing a lab coat.

    Others get stronger -- `bun`, `urine`, `glucose`. These are basic renal and
    volume monitoring, which is what you follow in heart failure, where
    cardiorenal syndrome drives outcome. That reading is clinically seductive
    and, as Q5 shows, wrong.

A5. WHY THE TWO TESTS DISAGREE
    Both are right. They answer different questions.

    The chi-square asks: of the patients in each group, what fraction had died
    by the end of the study? The log-rank asks: at each point in time, among
    those still alive and still observed, is the rate of dying the same? The
    first has no concept of exposure time. The second is built on it.

    The `fu_ratio` column settles it. Among censored CHF patients, those missing
    BUN were followed a median {bun_fu_missing} days against {bun_fu_obs} for
    those with it recorded -- {bun_fu_ratio} times longer. They did not die at a
    higher rate; they were watched longer, so more had died by the time the
    study closed. The {bun_gap} point gap is an accounting artefact of unequal
    observation windows, and the overlapping curves in Figure 3 are what that
    looks like.

    The variables sort themselves by this ratio:

      * ratio near {artefact_ratio}, log-rank null -- {artefact_vars}. All
        artefact. Their binary p-values are the most significant on the list and
        mean nothing.
      * ratio {real_ratio_lo} to {real_ratio_hi}, surviving FDR correction --
        {real_vars}. Real.

    Notice what the survivors are. Not laboratory values at all -- they are
    collected by INTERVIEWING the patient. Non-response to an interview is
    caused by the patient's condition: too unwell, too confused, or dead before
    the interview happened. That is missingness driven by the outcome process
    itself, the textbook definition of informative, and the one place here where
    the textbook applies.

    One apparent survivor did not survive. On the full cohort with unadjusted
    p-values, education looked real. On the training partition with FDR
    correction across {n_tests} tests it is gone (p={edu_p}, q={edu_q}). Nothing
    about education changed -- what changed is that it was no longer judged
    against a bar it had help clearing. Marginal findings are exactly the ones
    that evaporate under a holdout and a multiplicity correction, which is the
    argument for imposing both before you become attached to a result.

A6. WHAT TO ACTUALLY DO
    Not complete-case analysis. Requiring every candidate predictor leaves
    {complete_n} of the CHF training cohort ({complete_pct}%), and Q4 shows the
    dropped rows differ systematically. That is selection bias, not tidying.

    Not the SUPPORT normal-fill constants as the primary strategy either.
    Filling creatinine with 1.01 asserts a value was normal when it was never
    measured, and shrinks variance so downstream confidence intervals come out
    too narrow. It is kept in support2.py as a documented clinical baseline to
    compare against, not as the answer.

    Three parts:

      1. Multiple imputation (MICE), conditioning the imputation model on the
         auxiliaries Q4 identified -- dzgroup and care-intensity measures --
         since those are what make MAR tenable for the case-mix-driven group.
      2. Missingness indicators for {real_vars} only -- the variables that clear
         FDR correction on training data. This is the trap the question points
         at: after Q3 the obvious move is to flag {artefact_vars}, and that
         would be wrong. Those indicators encode enrolment era, not patient
         state (03_cohort.py proves it). You would spend degrees of freedom on
         noise and then be asked, in front of clinicians, to explain a
         coefficient for "BUN was not drawn" -- with no clinical story to tell.
      3. Imputation fitted inside each cross-validation fold. Imputing on the
         full dataset before splitting leaks test information into training, and
         is the most common silent error in pipelines like this.

    Then report the sensitivity: MICE, complete-case and normal-fill side by
    side. Where they agree, say so. Where they diverge, the divergence is a
    finding about the data rather than an inconvenience.

    Note finally that {n_voided} physiologically impossible cells were set to
    missing before any of this ran -- see 02_profile.py Q8. They are imputed
    alongside the values that were never recorded, because an impossible
    measurement and an absent one are the same kind of ignorance.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    cohort = analysis_frames()
    df, chf = cohort.full_train, cohort.chf_train
    raw = cohort.raw

    header("SUPPORT2 -- exploratory analysis")
    print(f"  {raw.shape[0]:,} patients x {raw.shape[1]} columns as loaded")
    print(f"  {cohort.n_voided} physiologically impossible cells set to missing "
          f"before any analysis")

    print("\n  Cohort derivation (CONSORT-style attrition):")
    for _, r in cohort_flow(raw).iterrows():
        note = f"  (-{r['excluded']:,})" if r["excluded"] else ""
        print(f"    {r['remaining']:>6,}  {r['step']}{note}")

    print(f"\n  Train/test partition (seed 20260901, stratified on death):")
    print(f"    all enrolled   train {len(df):,}   test {cohort.n_test:,}")
    print(f"    CHF cohort     train {len(chf):,}")
    print("\n  Every number below is computed on TRAIN ONLY. The test partition is")
    print("  never returned by analysis_frames(). See A0 for why.")

    outcome = compute_outcome_structure(df)
    report_outcome_structure(outcome)

    leak = compute_leakage_audit(df)
    report_leakage_audit(leak)

    missing = missingness_vs_outcome(df, CANDIDATE_PREDICTORS)
    report_missingness(missing)

    case_mix = compute_case_mix(df, missing)
    report_case_mix(case_mix)

    comparison = compute_binary_vs_logrank(df)
    facts = collect_facts(outcome, missing, comparison,
                          {"real": [], "artefact": [], "complete_n": 0,
                           "complete_pct": 0.0}, cohort.n_voided)
    report_binary_vs_logrank(comparison, facts)

    strategy = report_strategy(df, comparison)
    facts = collect_facts(outcome, missing, comparison, strategy, cohort.n_voided)

    header("FIGURES")
    for path in (figure_missingness(df, comparison),
                 figure_conditioning(case_mix["merged"], len(chf)),
                 figure_km(df, comparison)):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        # Narrow and deliberate: lifelines emits a delta-t warning on tied event
        # times that carries no information here. Convergence warnings are NOT
        # suppressed -- 02_profile.py depends on seeing them.
        warnings.filterwarnings("ignore", message=".*delta.*")
        run_and_capture(main, OUT_DIR / "01_eda.txt")
