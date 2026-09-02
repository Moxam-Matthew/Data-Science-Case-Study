"""
01_eda.py -- Exploratory analysis of SUPPORT2, taught as a set of questions.

HOW TO READ THIS FILE
    Each section states a question, then shows the evidence needed to answer it.
    The answers are held back deliberately: they live in ANSWERS at the bottom of
    this file, and print after the analysis when you run it. Work out your own
    answer from the printed numbers first -- the reasoning is worth more than the
    conclusion, and in an interview you will be asked for the reasoning.

    Run:  python 01_eda.py

THE QUESTIONS
    Q1  The dataset ships a binary `death` flag. Why is modelling that flag
        alone a mistake here, and what does it discard?
    Q2  Which columns must never enter the feature matrix, and how would you
        recognise one you had not been warned about?
    Q3  Roughly half of several lab columns are missing. Is that missingness
        random, or does it carry information about the outcome?
    Q4  The mortality gap for `pafi` missingness disappears once the cohort is
        restricted to CHF, while the gap for `bun` gets larger. Why would
        conditioning on disease group do opposite things to two lab variables?
    Q5  Split the CHF cohort on whether BUN was ever recorded. A chi-square on
        death returns p<0.001 with a 20-point gap; a log-rank test on the same
        split returns p=0.67. Both are correctly computed. Which one is telling
        you the truth, and what does the other one actually measure?
    Q6  Given the corrected picture, what is the defensible imputation strategy
        -- and what would be wrong with adding a missingness indicator to every
        variable that looked informative in Q3?

Author: Matthew Moxam
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import viz  # noqa: E402
from stats_utils import add_fdr, followup_summary  # noqa: E402
from support2 import (  # noqa: E402
    CANDIDATE_PREDICTORS,
    CHF_LABEL,
    OUTCOME_EVENT,
    OUTCOME_TIME,
    audit_columns,
    chf_cohort,
    cohort_flow,
    load_support2,
    make_split,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

RULE = "=" * 78
SUB = "-" * 78


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def question(n: int, text: str) -> None:
    print(f"\n{SUB}\nQUESTION {n}: {text}\n{SUB}")


# ── Q1. The shape of the outcome ─────────────────────────────────────────────
def q1_outcome_structure(df: pd.DataFrame) -> None:
    question(1, "The dataset ships a binary `death` flag. Why is modelling that\n"
                "flag alone a mistake here, and what does it discard?")

    event, time = df[OUTCOME_EVENT], df[OUTCOME_TIME]
    censored = (event == 0)
    print(f"  deaths observed      {event.sum():>6,} / {len(df):,} ({event.mean()*100:.1f}%)")
    print(f"  censored (alive)     {censored.sum():>6,} ({censored.mean()*100:.1f}%)")

    print("\n  Three different quantities, often confused. Only one is follow-up:")
    fu = followup_summary(df, OUTCOME_TIME, OUTCOME_EVENT)
    print(f"    median of the time column      {fu['median_time_to_event_or_censor']:>7,.0f} days"
          "   <- a median TIME-TO-EVENT, pulled down by every death")
    print(f"    median among censored only     {fu['median_time_among_censored']:>7,.0f} days"
          "   <- ad hoc; ignores the observed follow-up of those who died")
    print(f"    reverse Kaplan-Meier           {fu['median_followup_reverseKM']:>7,.0f} days"
          "   <- median FOLLOW-UP; the correct estimator")
    print(f"    longest observation            {fu['max_observed']:>7,.0f} days")

    print("\n  Follow-up among the censored -- if these were all short, early dropout")
    print("  would bias a binary flag. Are they?")
    censored_time = time[censored]
    for pct in (10, 25, 50, 75, 90):
        print(f"    {pct:>2}th percentile   {np.percentile(censored_time, pct):>6,.0f} days")
    print(f"\n  Censored before 1 year: {(censored_time < 365).sum():,} "
          f"({(censored_time < 365).mean()*100:.1f}% of censored patients)")

    print("\n  Same question inside the CHF cohort:")
    chf = chf_cohort(df)
    cfu = followup_summary(chf, OUTCOME_TIME, OUTCOME_EVENT)
    print(f"    n={cfu['n']:,}  deaths={cfu['events']:,}  censored={cfu['censored']:,}")
    print(f"    median follow-up (reverse KM) {cfu['median_followup_reverseKM']:,.0f} days")
    chf_cens = chf.loc[chf[OUTCOME_EVENT] == 0, OUTCOME_TIME]
    print(f"    {(chf_cens < 365).mean()*100:.1f}% of censored followed <1 year")


# ── Q2. Column governance ────────────────────────────────────────────────────
def q2_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    question(2, "Which columns must never enter the feature matrix, and how\n"
                "would you recognise one you had not been warned about?")

    audit = audit_columns(df)
    print(audit.groupby("role").size().to_string())

    print("\n  Excluded, with the reason:")
    for _, r in audit[audit.role == "excluded"].iterrows():
        print(f"    {r['column']:<10} {r['reason']}")

    unreviewed = audit[audit.role == "unreviewed"]
    if len(unreviewed):
        print("\n  Not yet classified (decide before modelling):")
        print("   ", ", ".join(unreviewed["column"]))

    print("\n  Empirical check -- univariate association with death for the three")
    print("  strongest excluded columns, to show what leakage looks like:")
    for col in ("d.time", "surv2m", "slos"):
        if col not in df:
            continue
        a = df.loc[df[OUTCOME_EVENT] == 1, col].median()
        b = df.loc[df[OUTCOME_EVENT] == 0, col].median()
        u, p = stats.mannwhitneyu(df.loc[df[OUTCOME_EVENT] == 1, col].dropna(),
                                  df.loc[df[OUTCOME_EVENT] == 0, col].dropna())
        auc = u / (df[OUTCOME_EVENT].sum() * (len(df) - df[OUTCOME_EVENT].sum()))
        print(f"    {col:<10} median died={a:>9,.1f}  survived={b:>9,.1f}  "
              f"univariate AUC={max(auc, 1-auc):.3f}")
    return audit


# ── Q3. Is missingness informative? ──────────────────────────────────────────
def missingness_vs_outcome(df: pd.DataFrame, cols: list[str],
                           min_miss: float = 0.02) -> pd.DataFrame:
    """Death rate when a value is missing vs observed, with a chi-square test."""
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


def _fmt(t: pd.DataFrame) -> str:
    t = t.copy()
    t["p"] = t["p"].apply(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    for c in ("missing_pct", "death_if_missing", "death_if_observed", "gap_pp"):
        t[c] = t[c].round(1)
    return t.to_string(index=False)


def q3_informative_missingness(df: pd.DataFrame) -> pd.DataFrame:
    question(3, "Roughly half of several lab columns are missing. Is that\n"
                "missingness random, or does it carry information about the outcome?")
    full = missingness_vs_outcome(df, CANDIDATE_PREDICTORS)
    print(_fmt(full))
    print("\n  Read the `gap_pp` column: percentage-point difference in mortality")
    print("  between patients whose value was missing and those whose was recorded.")
    return full


# ── Q4. Does it survive conditioning on disease? ─────────────────────────────
def q4_condition_on_disease(df: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    question(4, "The mortality gap for `pafi` missingness disappears once the\n"
                "cohort is restricted to CHF, while the gap for `bun` gets larger.\n"
                "Why would conditioning on disease group do opposite things?")

    chf = chf_cohort(df)
    within = missingness_vs_outcome(chf, CANDIDATE_PREDICTORS)

    merged = (full[["variable", "gap_pp", "p"]]
              .rename(columns={"gap_pp": "gap_full", "p": "p_full"})
              .merge(within[["variable", "gap_pp", "p", "missing_pct"]]
                     .rename(columns={"gap_pp": "gap_chf", "p": "p_chf"}),
                     on="variable", how="inner"))
    merged["survives"] = np.where(merged.p_chf < 0.05, "yes", "no")
    merged["shrinkage"] = merged.gap_full - merged.gap_chf
    merged = merged.sort_values("gap_chf", key=abs, ascending=False)

    show = merged.copy()
    for c in ("gap_full", "gap_chf", "shrinkage", "missing_pct"):
        show[c] = show[c].round(1)
    for c in ("p_full", "p_chf"):
        show[c] = show[c].apply(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    print(show[["variable", "missing_pct", "gap_full", "p_full",
                "gap_chf", "p_chf", "survives"]].to_string(index=False))

    print("\n  Case mix is the mechanism. Which patients carry the missing values?")
    labs = ["bun", "glucose", "urine", "ph", "pafi", "alb", "bili", "wblc"]
    d = df.copy()
    d["n_missing_labs"] = d[labs].isna().sum(axis=1)
    d["heavy"] = d.n_missing_labs >= 5
    comp = (pd.crosstab(d.heavy, d.dzgroup, normalize="index") * 100).round(1).T
    comp.columns = ["<5 labs missing (%)", ">=5 labs missing (%)"]
    print("\n" + comp.sort_values(">=5 labs missing (%)", ascending=False).to_string())
    return merged


# ── Q5. The two tests disagree ───────────────────────────────────────────────
def q5_binary_vs_logrank(df: pd.DataFrame) -> pd.DataFrame:
    question(5, "Split the CHF cohort on whether BUN was ever recorded. A chi-square\n"
                "on death returns p<0.001 with a 20-point gap; a log-rank test on the\n"
                "same split returns p=0.67. Both are correctly computed. Which is\n"
                "telling you the truth, and what does the other one measure?")

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
    # Multiplicity: this is one test per variable, so the small p-values must
    # clear a corrected bar before they count as findings.
    out = add_fdr(out, p_col="p_logrank", q_col="q_logrank")
    out["verdict"] = np.where(
        (out.p_binary < 0.05) & (out.q_logrank < 0.05), "real signal",
        np.where(out.p_binary < 0.05, "follow-up artefact", "no signal"))

    show = out.copy()
    for c in ("gap_pp", "fu_ratio"):
        show[c] = show[c].round(2)
    for c in ("censored_fu_obs", "censored_fu_missing"):
        show[c] = show[c].round(0)
    for c in ("p_binary", "p_logrank", "q_logrank"):
        show[c] = show[c].apply(
            lambda v: "" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
    print(show[["variable", "gap_pp", "p_binary", "p_logrank", "q_logrank",
                "censored_fu_obs", "censored_fu_missing", "fu_ratio",
                "verdict"]].to_string(index=False))
    print(f"\n  q_logrank is the Benjamini-Hochberg FDR q-value across "
          f"{int(out.p_logrank.notna().sum())} tests. The verdict column uses q, not p.")

    print("\n  `fu_ratio` is median follow-up among CENSORED patients, missing-group")
    print("  over observed-group. A ratio near 1 means the two groups were watched")
    print("  for equally long, so a difference in cumulative death is real. A ratio")
    print("  of 2.5 means one group simply had far longer to die.")
    return out


def q6_strategy(df: pd.DataFrame, comparison: pd.DataFrame):
    question(6, "Given the corrected picture, what is the defensible imputation\n"
                "strategy -- and what would be wrong with adding a missingness\n"
                "indicator to every variable that looked informative in Q3?")

    real = comparison.loc[comparison.verdict == "real signal", "variable"].tolist()
    artefact = comparison.loc[comparison.verdict == "follow-up artefact", "variable"].tolist()
    print(f"  Survive the time-to-event test : {', '.join(real) if real else '(none)'}")
    print(f"  Artefact of differential follow-up: {', '.join(artefact) if artefact else '(none)'}")
    print("\n  Note what the survivors have in common, and what the artefacts do.")

    chf = chf_cohort(df)
    complete = chf[CANDIDATE_PREDICTORS].dropna()
    print(f"\n  Complete-case cost: {len(complete):,} of {len(chf):,} CHF patients retained "
          f"({len(complete)/len(chf)*100:.1f}%).")
    return real, artefact


# ── Figures ──────────────────────────────────────────────────────────────────
def figure_missingness(df: pd.DataFrame, comparison: pd.DataFrame):
    import matplotlib.pyplot as plt

    miss = (df[CANDIDATE_PREDICTORS].isna().mean() * 100).sort_values()
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

    labels = ["Informative (survives log-rank)",
              "Artefact of differential follow-up",
              "No association"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k])
               for k in ("real signal", "follow-up artefact", "no signal")]
    ax.legend(handles, labels, loc="lower right")
    viz.caption(fig, "SUPPORT2 (UCI 880), n=9,105; classification computed on the CHF cohort.\n"
                     "Blue bars are variables a chi-square would have flagged and a log-rank clears.")
    return viz.save(fig, "01_missingness_by_variable.png")


def figure_conditioning(merged: pd.DataFrame):
    import matplotlib.pyplot as plt

    d = merged.reindex(merged.gap_chf.abs().sort_values().index)
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for yi, (gf, gc) in enumerate(zip(d.gap_full, d.gap_chf)):
        ax.plot([gf, gc], [yi, yi], color=viz.BASELINE, lw=1.5, zorder=1)
    ax.scatter(d.gap_full, y, s=70, color=viz.SERIES_BLUE, zorder=2,
               label="All patients", edgecolor=viz.SURFACE, linewidth=1.5)
    ax.scatter(d.gap_chf, y, s=70, color=viz.SERIES_ORANGE, zorder=3,
               label="CHF only", edgecolor=viz.SURFACE, linewidth=1.5)
    ax.axvline(0, color=viz.BASELINE, lw=1.2)
    ax.set_yticks(y, d.variable)
    ax.set_xlabel("Mortality gap: missing minus observed (percentage points)")
    ax.set_title("Does the missingness signal survive holding disease constant?")
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    ax.legend(loc="lower right")
    viz.caption(fig, "Right of zero: patients missing the value died more often. Variables\n"
                     "collapsing toward zero were tracking case mix, not the patient.")
    return viz.save(fig, "02_conditioning_on_disease.png")


def figure_km(df: pd.DataFrame, comparison: pd.DataFrame):
    """Two panels: a variable whose signal is artefact, beside one that is real."""
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    chf = chf_cohort(df)
    lookup = comparison.set_index("variable")
    panels = [("bun", "BUN (routine renal lab)"), ("adlp", "ADL (patient-reported function)")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        obs, mis = chf[chf[col].notna()], chf[chf[col].isna()]
        lr = logrank_test(obs[OUTCOME_TIME], mis[OUTCOME_TIME],
                          obs[OUTCOME_EVENT], mis[OUTCOME_EVENT])
        for data, label, color in ((obs, "recorded", viz.SERIES_BLUE),
                                   (mis, "not recorded", viz.SERIES_ORANGE)):
            km = KaplanMeierFitter()
            km.fit(data[OUTCOME_TIME], data[OUTCOME_EVENT], label=label)
            km.plot_survival_function(ax=ax, color=color, ci_alpha=0.12, lw=2)
            ax.annotate(label, xy=(1500, km.predict(1500)), xytext=(4, 6),
                        textcoords="offset points", fontsize=9,
                        color=color, weight="600")

        row = lookup.loc[col]
        p_b = "<0.001" if row.p_binary < 0.001 else f"{row.p_binary:.3f}"
        p_l = "<0.001" if row.p_logrank < 0.001 else f"{row.p_logrank:.3f}"
        ax.set_title(title)
        ax.set_xlabel("Days from study entry")
        ax.set_xlim(0, 2029)
        ax.set_ylim(0, 1)
        ax.text(0.03, 0.10,
                f"binary p {p_b}   log-rank p {p_l}\n"
                f"censored follow-up ratio {row.fu_ratio:.2f}x",
                transform=ax.transAxes, fontsize=8.5, color=viz.INK_SECONDARY)
        ax.get_legend().remove()
        viz.despine(ax)

    axes[0].set_ylabel("Survival probability")
    fig.suptitle("Same test, opposite conclusions: cumulative death vs. hazard over time",
                 fontsize=12, fontweight="600", y=1.0)
    viz.caption(fig,
                "n=1,387 CHF patients. Left: a 20-point gap in cumulative death with curves that\n"
                "overlap -- the missing-BUN group was simply followed 2.6x longer. Right: balanced\n"
                "follow-up, and the curves genuinely separate. Only the right panel is a finding.")
    return viz.save(fig, "03_binary_vs_time_to_event.png")


ANSWERS = """
ANSWERS
{rule}

A0. WHY THERE IS A HELD-OUT PARTITION, AND WHY IT IS LOCKED
    Not a question in the list, but the first thing a reviewer should ask.

    Across 01_eda.py and 02_profile.py this project looks at the outcome
    roughly sixty times: twelve missingness contrasts, twelve log-rank tests,
    around thirty Table 1 comparisons, seven tests of functional form. Each one
    is a decision point at which the data could steer a modelling choice. That
    accumulated steering is analyst degrees of freedom, and it makes every
    subsequent performance estimate optimistic by an amount nobody can compute
    after the fact.

    Two honest responses exist. Report everything as exploratory and correct
    the optimism by bootstrap. Or partition once, before modelling, and keep
    one part unseen. This project does both: the exploratory findings below are
    labelled hypothesis-generating and carry FDR-adjusted q-values, and a 30%
    partition is held out by fixed seed and never read.

    The partition is generated from a constant (seed 20260901, stratified on
    death) rather than stored, so it reproduces exactly without committing
    patient rows -- the same constraint that shapes the loader.

    What this buys, said plainly: the confirmatory estimates will come from
    data that had no opportunity to influence any choice made here. Without it,
    a held-out AUC reported at the end of this project would be a number with
    no defensible interpretation.

A1. WHY NOT THE BINARY FLAG
    Start by refusing the textbook worry. The standard fear is heavy early
    dropout, and it does not apply here: only about 2% of censored patients
    were followed less than a year. Censoring in SUPPORT2 is mostly
    administrative -- the study ended. Say that out loud rather than reciting a
    concern the data does not support.

    Note also which follow-up number you quote. The median of the time column
    is a median time-to-event: every death drags it down, and it says nothing
    about how long the cohort was watched. Median follow-up needs the reverse
    Kaplan-Meier, inverting the indicator so censoring is the event. On this
    cohort the two differ by roughly threefold. Quoting the first as "median
    follow-up" is a common and consequential error, and it is especially
    embarrassing in a project whose central finding is about follow-up.

    The real objections are different, and stronger.

    First, the label is nearly saturated: 68% of the full cohort and 61% of the
    CHF cohort died. An outcome that fires for two-thirds of patients has
    little room to discriminate, and "died at some point over the next one to
    five years" is not a question anyone makes a decision on.

    Second, and this is the one that bites, follow-up duration is not balanced
    across patients -- and Q5 shows it correlates with data-collection patterns
    in a way that manufactures signal. Once that is true, the binary flag is
    not merely lossy. It is biased, and biased in a direction you will not
    notice unless you look.

    Third, the clinical question is timing. A cardiologist discharging a heart
    failure patient wants 30-day and 6-month risk. Any fixed-horizon binary
    outcome has to be computed with censor-adjustment anyway, so you need the
    time variable regardless.

    That is the honest case for Kaplan-Meier and Cox here -- not that survival
    analysis is impressive, but that the binary alternative is actively
    misleading on this dataset.

A2. THE COLUMNS THAT MUST NOT ENTER
    Fifteen columns are excluded, and they fall into four recognisable kinds:

      * The outcome in disguise -- `d.time`, `hospdead`. `d.time` is follow-up
        duration; as a "predictor" it achieves a near-perfect univariate AUC,
        which is the tell.
      * Measured after baseline -- `sfdm2` (function at 2 months), `slos`,
        `charges`, `totcst`, `avtisst`. Not knowable at prediction time.
      * Another model's output -- `surv2m`, `surv6m`. Including these means
        your model is re-predicting the 1995 SUPPORT model.
      * Constructed from the predictors -- `aps`, `sps` are severity scores
        computed from the same vitals and labs. Collinear by construction.

    `prg2m` and `prg6m` are excluded for a different reason: they are the
    attending physician's own survival estimates. They are the benchmark. The
    interesting question is not whether a model beats chance, but whether it
    beats the doctor -- and you cannot answer that if you fed the doctor's
    answer to the model.

    The general test, for a column nobody warned you about: could this value
    have been known, in this form, at the moment the prediction is meant to be
    made? If the answer needs a hospital course to resolve, it leaks.

A3. IS THE MISSINGNESS INFORMATIVE?
    On this evidence it looks emphatically non-random. Patients missing BUN
    died 12.5 percentage points more often than those with it recorded;
    glucose, urine output and patient-reported ADLs show gaps of similar size,
    all at p<0.001. Under MCAR you would expect gaps near zero.

    That is the answer most analyses give, and it is where most of them stop.
    Hold it loosely. A gap of this kind has at least three explanations: the
    patient was sicker, the patient was somewhere that orders fewer labs, or
    the two groups were not observed for the same length of time. The third is
    invisible to a chi-square, because a chi-square on a cumulative outcome has
    no concept of exposure time at all.

    Q4 separates the first two. Q5 is where the third one surfaces, and it
    removes half the variables on this list.

A4. WHY CONDITIONING SPLITS THE VARIABLES IN TWO
    Restricting to CHF holds the disease fixed, and the variables separate:

      * Collapses to nothing -- `pafi` (p=0.73), `ph` (p=0.70), `alb`, `bili`,
        `adls`. `pafi` and `ph` come from arterial blood gases, drawn almost
        exclusively on ventilated or ICU patients. Across the full cohort their
        missingness encoded *where the patient was being treated*: the
        heavy-missing group is 20% lung cancer and 13% colon cancer against 19%
        sepsis, while the well-measured group is 44% sepsis. Ward oncology
        patients do not get serial blood gases. The signal was case mix wearing
        a lab coat.

      * Gets stronger -- `bun` (+20.1pp), `urine` (+19.7pp), `glucose`
        (+18.4pp), `adlp` (+12.0pp), all p<0.001. These are basic renal and
        volume monitoring, which is precisely what you follow in heart failure,
        where cardiorenal syndrome drives outcome. A CHF patient whose renal
        function was not being tracked is a genuinely different patient. And
        `adlp` is patient-reported function, absent when someone is too unwell
        or confused to be interviewed -- the missingness is caused by the thing
        you are trying to predict.

    So the mechanism is not uniform across columns. Some missingness tracks
    case mix and care setting. Whether any of it tracks the patient is a
    question Q4 cannot answer, because every test so far has been run on a
    cumulative outcome.

A5. WHY THE TWO TESTS DISAGREE
    Both are right. They are answering different questions.

    The chi-square asks: of the patients in each group, what fraction had died
    by the end of the study? The log-rank asks: at each point in time, among
    those still alive and still being observed, is the rate of dying the same?
    The first has no concept of exposure time. The second is built on it.

    The `fu_ratio` column settles it. Among censored CHF patients, those
    missing BUN were followed for a median 1,689 days against 655 for those
    with it recorded -- 2.6 times longer. They also died later (median 275 vs
    179 days). They did not die at a higher rate. They were watched for longer,
    so more of them had died by the time the study closed. The 20-point gap is
    an accounting artefact of unequal observation windows, and the overlapping
    curves in Figure 3 (left) are what that looks like.

    The variables sort themselves cleanly by this ratio:

      * ratio ~2.55, log-rank null -- bun, urine, glucose. All artefact. Their
        binary p-values are the most significant on the list and mean nothing.
      * ratio ~1.1, surviving FDR correction -- income and adlp. Real.

    Notice what the survivors are. Not laboratory values at all -- they are
    variables collected by *interviewing the patient*. Non-response to an
    interview is caused by the patient's condition: too unwell, too confused,
    or dead before the interview happened. That is missingness driven by the
    outcome process itself, which is the textbook definition of informative,
    and it is the one place here where the textbook actually applies.

    One survivor did not survive. On the full cohort with unadjusted p-values,
    education looked real (p=0.004). On the training partition with FDR
    correction across twelve tests it is gone (p=0.124, q=0.253). Nothing about
    education changed -- what changed is that it was no longer being judged
    against a bar it had help clearing. Marginal findings are exactly the ones
    that evaporate under a holdout and a multiplicity correction, which is the
    argument for imposing both before you become attached to a result.

    Why the labs carry a 2.6x follow-up imbalance is worth stating as a
    hypothesis rather than a fact: SUPPORT enrolled in two phases several years
    apart, and phase-specific collection protocols would produce exactly this
    pattern -- earlier enrolment giving longer follow-up, alongside a different
    set of routinely captured labs. The dataset ships no phase indicator, so
    this is unverified here. It should be flagged as a limitation, not asserted.

    The transferable habit: whenever a group difference in a cumulative outcome
    is the headline, check whether the groups were observed for the same length
    of time before believing it.

A6. WHAT TO ACTUALLY DO
    Not complete-case analysis. Requiring every candidate predictor drops the
    CHF cohort to a small fraction of itself, and Q4 shows the dropped rows
    differ systematically. That is selection bias, not tidying.

    Not the SUPPORT normal-fill constants as the primary strategy either.
    Filling creatinine with 1.01 asserts a value was normal when it was never
    measured, and it shrinks variance so downstream confidence intervals come
    out too narrow. It is kept in support2.py as a documented clinical baseline
    to compare against, not as the answer.

    Three parts:

      1. Multiple imputation (MICE), conditioning the imputation model on the
         auxiliaries Q4 identified -- dzgroup and care-intensity measures --
         since those are what make the MAR assumption tenable for the
         case-mix-driven group.
      2. Missingness indicators for adlp and income only -- the two that clear
         FDR correction on training data. This is the trap the question points
         at: after Q3 the obvious move is to flag bun, urine and glucose, and
         that would be wrong. Those indicators encode enrolment era, not
         patient state. You would spend degrees of freedom on noise and then be
         asked, in front of clinicians, to explain a coefficient for "BUN was
         not drawn" -- with no clinical story to tell, because there isn't one.
      3. Imputation fitted inside each cross-validation fold. Imputing on the
         full dataset before splitting leaks test information into training,
         and it is the most common silent error in pipelines like this.

    Then report the sensitivity: MICE, complete-case, and normal-fill side by
    side. Where they agree, say so. Where they diverge, the divergence is a
    finding about the data rather than an inconvenience.

    The clinical framing worth keeping: an indicator for "this patient could
    not complete the functional interview" is not a data-quality artefact to be
    scrubbed. It records something true about the patient, and Figure 3 (right)
    shows the two groups have genuinely different survival. A model may use
    that -- provided you can say out loud what it means.
{rule}
"""


def main() -> None:
    viz.apply_style()
    full = load_support2()

    header("SUPPORT2 -- exploratory analysis")
    print(f"  {full.shape[0]:,} patients x {full.shape[1]} columns")

    print("\n  Cohort derivation (CONSORT-style attrition):")
    flow = cohort_flow(full)
    for _, r in flow.iterrows():
        note = f"  (-{r['excluded']:,})" if r["excluded"] else ""
        print(f"    {r['remaining']:>6,}  {r['step']}{note}")

    # The partition is defined on the whole enrolled cohort, not on CHF alone,
    # because Q4 compares CHF against the other disease groups and both sides
    # of that comparison must come from the same side of the split.
    split = make_split(full)
    df = full[split == "train"].copy()
    chf_all, chf_tr = chf_cohort(full), chf_cohort(df)

    print(f"\n  Train/test partition (seed 20260901, stratified on death):")
    print(f"    all enrolled   train {(split=='train').sum():,}   "
          f"test {(split=='test').sum():,}")
    print(f"    CHF cohort     train {len(chf_tr):,}   test {len(chf_all)-len(chf_tr):,}")
    print(f"    event rate     train {df[OUTCOME_EVENT].mean()*100:.1f}%   "
          f"test {full.loc[split=='test', OUTCOME_EVENT].mean()*100:.1f}%")
    print("\n  Every number below is computed on TRAIN ONLY. The test partition is")
    print("  not read by this script or by 02_profile.py. See A0 for why.")

    q1_outcome_structure(df)
    q2_leakage_audit(df)
    full = q3_informative_missingness(df)
    merged = q4_condition_on_disease(df, full)
    comparison = q5_binary_vs_logrank(df)
    q6_strategy(df, comparison)

    header("FIGURES")
    for path in (figure_missingness(df, comparison),
                 figure_conditioning(merged),
                 figure_km(df, comparison)):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(ANSWERS.format(rule=RULE))


if __name__ == "__main__":
    main()
