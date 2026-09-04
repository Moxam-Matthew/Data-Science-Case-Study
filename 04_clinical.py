"""
04_clinical.py -- The questions a clinician asks, which the statistics do not.

The first three scripts ask whether the analysis is sound. This one asks whether
the result would survive a room of interventional cardiologists, which is a
different test and fails for different reasons.

    Run:  python 04_clinical.py

THE QUESTIONS
    Q18  Write the opening paragraph of a Results section. Who are these
         patients, in the terms a clinician thinks in?
    Q19  DNR status is the strongest single predictor in this cohort, ahead of
         every physiologic variable. Should it be in the model?
    Q20  A prediction model needs an origin: the moment at which the prediction
         is made. Is there one here, and is it the same moment for everyone?
    Q21  Name the variables a heart failure specialist would expect and this
         dataset does not contain. What can you not claim as a result?
    Q22  The cohort was assembled between 1989 and 1994. Which of your findings
         transfer to a patient admitted this year, and which do not?

Author: Matthew Moxam
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import viz
from report import Facts, RULE, configure_pandas, fmt_p, header, question, render_answers, run_and_capture
from support2 import (
    DNR_IN_ADMISSION_LABEL,
    DNR_PREEXISTING_LABEL,
    OUTCOME_EVENT,
    OUTCOME_TIME,
    analysis_frames,
)

OUT_DIR = Path(__file__).resolve().parent / "output"

# Variables a heart failure specialist would expect at the bedside, and the
# token that would identify each if the dataset held it.
EXPECTED_CARDIOLOGY = {
    "Ejection fraction (LVEF)": ["ef", "lvef", "eject"],
    "NYHA functional class": ["nyha"],
    "BNP / NT-proBNP": ["bnp", "natriuretic"],
    "ECG / rhythm": ["ecg", "ekg", "rhythm", "afib"],
    "Echocardiography": ["echo", "lvedd", "valve"],
    "Heart failure medications": ["betablock", "aceinhib", "diuretic", "digoxin"],
    "Revascularisation history": ["pci", "cabg", "revasc", "stent"],
    "Cause of death": ["causedeath", "cvdeath"],
}

MODERN_THERAPIES = [
    ("1996-99", "beta-blockers established in HFrEF (CIBIS-II, MERIT-HF)"),
    ("1999", "spironolactone (RALES)"),
    ("2002-05", "ICD for primary prevention (MADIT-II, SCD-HeFT); CRT (COMPANION)"),
    ("2014", "sacubitril/valsartan (PARADIGM-HF)"),
    ("2019-20", "SGLT2 inhibitors (DAPA-HF, EMPEROR-Reduced)"),
]


# Q18. Who are these patients ================================================
def compute_cohort_description(chf: pd.DataFrame) -> dict:
    return {
        "n": len(chf),
        "deaths": int(chf[OUTCOME_EVENT].sum()),
        "age_med": chf.age.median(),
        "age_q1": chf.age.quantile(.25), "age_q3": chf.age.quantile(.75),
        "age_over_75": (chf.age >= 75).mean() * 100,
        "female": (chf.sex == "female").mean() * 100,
        "comorb_med": chf["num.co"].median(),
        "comorb_3plus": (chf["num.co"] >= 3).mean() * 100,
        "diabetes": chf.diabetes.mean() * 100,
        "dementia": chf.dementia.mean() * 100,
        "cancer": (chf.ca != "no").mean() * 100,
        "race": chf.race.value_counts(normalize=True).mul(100),
        "dnr_any": chf.dnr.isin(
            [DNR_PREEXISTING_LABEL, DNR_IN_ADMISSION_LABEL]).mean() * 100,
        "crea_med": chf.crea.median(),
        "renal_impair": (chf.crea > 1.5).mean() * 100,
        "meanbp_med": chf.meanbp.median(),
        "adl_dependent": (chf.adls >= 2).mean() * 100,
    }


def report_cohort_description(d: dict) -> None:
    question(18, "Write the opening paragraph of a Results section. Who are these\n"
                 "patients, in the terms a clinician thinks in?")
    print(f"  {d['n']:,} patients with a primary diagnosis of congestive heart failure,")
    print(f"  of whom {d['deaths']:,} died during follow-up.\n")
    print(f"    age                 median {d['age_med']:.0f} "
          f"(IQR {d['age_q1']:.0f}-{d['age_q3']:.0f}); {d['age_over_75']:.1f}% aged 75+")
    print(f"    sex                 {d['female']:.1f}% female")
    print(f"    comorbidity burden  median {d['comorb_med']:.0f}; "
          f"{d['comorb_3plus']:.1f}% with 3 or more")
    print(f"    diabetes            {d['diabetes']:.1f}%")
    print(f"    dementia            {d['dementia']:.1f}%")
    print(f"    any cancer          {d['cancer']:.1f}%")
    print(f"    creatinine          median {d['crea_med']:.1f} mg/dL; "
          f"{d['renal_impair']:.1f}% above 1.5")
    print(f"    mean arterial BP    median {d['meanbp_med']:.0f} mmHg")
    print(f"    ADL dependence      {d['adl_dependent']:.1f}% dependent in 2+ activities")
    print(f"    DNR order (any)     {d['dnr_any']:.1f}%")
    print("    race                " + ", ".join(
        f"{k} {v:.1f}%" for k, v in d["race"].items()))


# Q19. DNR ===================================================================
def compute_dnr(chf: pd.DataFrame) -> dict:
    from lifelines.statistics import logrank_test

    g = chf.groupby("dnr").agg(n=(OUTCOME_EVENT, "size"),
                               deaths=(OUTCOME_EVENT, "sum"),
                               mortality=(OUTCOME_EVENT, "mean"),
                               median_surv=(OUTCOME_TIME, "median"))
    g["mortality"] = g.mortality * 100

    pre = chf[chf.dnr == DNR_PREEXISTING_LABEL]
    ina = chf[chf.dnr == DNR_IN_ADMISSION_LABEL]
    none = chf[chf.dnr == "no dnr"]
    lr_pre = logrank_test(pre[OUTCOME_TIME], none[OUTCOME_TIME],
                          pre[OUTCOME_EVENT], none[OUTCOME_EVENT])
    lr_ina = logrank_test(ina[OUTCOME_TIME], none[OUTCOME_TIME],
                          ina[OUTCOME_EVENT], none[OUTCOME_EVENT])

    rows = []
    for c in ["crea", "meanbp", "alb", "sod", "hrt", "resp", "age", "num.co", "scoma"]:
        d = chf[[c, OUTCOME_EVENT]].dropna()
        if d[c].nunique() < 3:
            continue
        hi = d[c] > d[c].median()
        rows.append({"variable": c,
                     "gap_pp": (d.loc[hi, OUTCOME_EVENT].mean()
                                - d.loc[~hi, OUTCOME_EVENT].mean()) * 100})
    rows.append({"variable": "DNR in admission",
                 "gap_pp": (ina[OUTCOME_EVENT].mean() - none[OUTCOME_EVENT].mean()) * 100})
    rows.append({"variable": "DNR pre-existing",
                 "gap_pp": (pre[OUTCOME_EVENT].mean() - none[OUTCOME_EVENT].mean()) * 100})
    ranking = pd.DataFrame(rows).sort_values("gap_pp", key=abs, ascending=False)
    return {"levels": g, "ranking": ranking,
            "p_pre": lr_pre.p_value, "p_ina": lr_ina.p_value}


def report_dnr(r: dict) -> None:
    question(19, "DNR status is the strongest single predictor in this cohort, ahead\n"
                 "of every physiologic variable. Should it be in the model?")
    show = r["levels"].copy()
    show["mortality"] = show.mortality.round(1)
    print(show.to_string())
    print(f"\n  vs no DNR, log-rank:  pre-existing p={fmt_p(r['p_pre'])}   "
          f"in-admission p={fmt_p(r['p_ina'])}")
    print("\n  Ranked against the physiologic predictors (median split):")
    print(r["ranking"].round(1).to_string(index=False))


# Q20. Prediction origin =====================================================
def compute_prediction_origin(chf: pd.DataFrame) -> dict:
    day1 = chf.hday <= 1
    _, p, *_ = stats.chi2_contingency(pd.crosstab(day1, chf[OUTCOME_EVENT]))
    return {
        "day1_pct": day1.mean() * 100,
        "day2_7_pct": chf.hday.between(2, 7).mean() * 100,
        "late_pct": (chf.hday > 7).mean() * 100,
        "max_hday": int(chf.hday.max()),
        "mort_day1": chf.loc[day1, OUTCOME_EVENT].mean() * 100,
        "mort_later": chf.loc[~day1, OUTCOME_EVENT].mean() * 100,
        "p": p,
        "n_later": int((~day1).sum()),
        "day1_frame": chf[day1],
    }


def report_prediction_origin(r: dict) -> None:
    question(20, "A prediction model needs an origin: the moment at which the\n"
                 "prediction is made. Is there one here, and is it the same\n"
                 "moment for everyone?")
    print(f"  Enrolled on hospital day 1      {r['day1_pct']:>5.1f}%")
    print(f"  Enrolled on days 2-7            {r['day2_7_pct']:>5.1f}%")
    print(f"  Enrolled after day 7            {r['late_pct']:>5.1f}%   "
          f"(latest: day {r['max_hday']})")
    print(f"\n  Mortality, enrolled day 1  : {r['mort_day1']:.1f}%")
    print(f"  Mortality, enrolled later  : {r['mort_later']:.1f}%  "
          f"(n={r['n_later']}, p={fmt_p(r['p'])})")
    print("\n  The later-enrolled are not a random sample of the cohort, and their")
    print("  'baseline' physiology was measured after days of hospital course.")


def sensitivity_day1(chf: pd.DataFrame, day1: pd.DataFrame) -> pd.DataFrame:
    """Does the headline missingness finding hold on an aligned cohort?"""
    from lifelines.statistics import logrank_test

    rows = []
    for label, frame in (("full training cohort", chf),
                         ("day-1 enrolments only", day1)):
        obs, mis = frame[frame.bun.notna()], frame[frame.bun.isna()]
        lr = logrank_test(obs[OUTCOME_TIME], mis[OUTCOME_TIME],
                          obs[OUTCOME_EVENT], mis[OUTCOME_EVENT])
        fu_o = obs.loc[obs[OUTCOME_EVENT] == 0, OUTCOME_TIME].median()
        fu_m = mis.loc[mis[OUTCOME_EVENT] == 0, OUTCOME_TIME].median()
        rows.append({"cohort": label, "n": len(frame),
                     "binary_gap_pp": (mis[OUTCOME_EVENT].mean()
                                       - obs[OUTCOME_EVENT].mean()) * 100,
                     "logrank_p": lr.p_value, "fu_ratio": fu_m / fu_o})
    out = pd.DataFrame(rows)
    print("\n  Sensitivity -- does the BUN finding survive an aligned origin?")
    show = out.copy()
    show["binary_gap_pp"] = show.binary_gap_pp.round(1)
    show["fu_ratio"] = show.fu_ratio.round(2)
    show["logrank_p"] = show.logrank_p.apply(fmt_p)
    print(show.to_string(index=False))
    return out


# Q21. What is absent ========================================================
def compute_absent_variables(chf: pd.DataFrame) -> pd.DataFrame:
    cols = " ".join(chf.columns).lower().replace("_", "").replace(".", "")
    rows = []
    for label, keys in EXPECTED_CARDIOLOGY.items():
        found = [k for k in keys if k in cols]
        rows.append({"expected_variable": label,
                     "present": "yes" if found else "NO",
                     "matched": ", ".join(found) if found else ""})
    return pd.DataFrame(rows)


def report_absent_variables(t: pd.DataFrame) -> None:
    question(21, "Name the variables a heart failure specialist would expect and this\n"
                 "dataset does not contain. What can you not claim as a result?")
    print(t.to_string(index=False))
    print(f"\n  {int((t.present == 'NO').sum())} of {len(t)} absent.")


# Q22. Transportability ======================================================
def report_transportability(d: dict) -> None:
    question(22, "The cohort was assembled between 1989 and 1994. Which findings\n"
                 "transfer to a patient admitted this year, and which do not?")
    print("  Therapies now standard in heart failure that post-date this cohort:")
    for era, item in MODERN_THERAPIES:
        print(f"    {era:<9} {item}")
    print(f"\n  Observed mortality in this cohort: "
          f"{d['deaths']/d['n']*100:.1f}% over follow-up.")
    print("  Representativeness:")
    print(f"    female {d['female']:.1f}%   "
          + "   ".join(f"{k} {v:.1f}%" for k, v in d["race"].items()))


# Facts ======================================================================
def collect_facts(d: dict, dnr: dict, origin: dict, absent: pd.DataFrame,
                  sens: pd.DataFrame) -> Facts:
    lv = dnr["levels"]
    rank = dnr["ranking"].set_index("variable")
    s = sens.set_index("cohort")
    return Facts(
        n=f"{d['n']:,}", deaths=f"{d['deaths']:,}",
        mortality=f"{d['deaths']/d['n']*100:.1f}",
        age_med=f"{d['age_med']:.0f}", age_q1=f"{d['age_q1']:.0f}",
        age_q3=f"{d['age_q3']:.0f}", female=f"{d['female']:.1f}",
        comorb_3plus=f"{d['comorb_3plus']:.1f}", diabetes=f"{d['diabetes']:.1f}",
        white=f"{d['race'].get('white', float('nan')):.1f}",
        black=f"{d['race'].get('black', float('nan')):.1f}",
        dnr_any=f"{d['dnr_any']:.1f}",
        mort_ina=f"{lv.loc[DNR_IN_ADMISSION_LABEL, 'mortality']:.1f}",
        mort_pre=f"{lv.loc[DNR_PREEXISTING_LABEL, 'mortality']:.1f}",
        mort_none=f"{lv.loc['no dnr', 'mortality']:.1f}",
        n_ina=f"{int(lv.loc[DNR_IN_ADMISSION_LABEL, 'n']):,}",
        n_pre=f"{int(lv.loc[DNR_PREEXISTING_LABEL, 'n']):,}",
        surv_ina=f"{lv.loc[DNR_IN_ADMISSION_LABEL, 'median_surv']:.0f}",
        surv_none=f"{lv.loc['no dnr', 'median_surv']:.0f}",
        p_pre=fmt_p(dnr["p_pre"]), p_ina=fmt_p(dnr["p_ina"]),
        dnr_gap=f"{rank.loc['DNR in admission', 'gap_pp']:.1f}",
        crea_gap=f"{rank.loc['crea', 'gap_pp']:.1f}",
        day1=f"{origin['day1_pct']:.1f}", late=f"{origin['late_pct']:.1f}",
        max_hday=str(origin["max_hday"]),
        mort_day1=f"{origin['mort_day1']:.1f}",
        mort_later=f"{origin['mort_later']:.1f}",
        origin_p=fmt_p(origin["p"]),
        n_absent=str(int((absent.present == "NO").sum())),
        n_expected=str(len(absent)),
        sens_ratio=f"{s.loc['day-1 enrolments only', 'fu_ratio']:.2f}",
        sens_gap=f"{s.loc['day-1 enrolments only', 'binary_gap_pp']:.1f}",
        sens_p=fmt_p(s.loc["day-1 enrolments only", "logrank_p"]),
    )


# Figure =====================================================================
def figure_dnr(chf: pd.DataFrame):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    spec = [("no dnr", "No DNR order", viz.SERIES_BLUE),
            (DNR_PREEXISTING_LABEL, "DNR pre-existing (advance directive)",
             viz.SERIES[2]),
            (DNR_IN_ADMISSION_LABEL, "DNR written during admission",
             viz.SERIES_ORANGE)]
    for level, label, color in spec:
        d = chf[chf.dnr == level]
        if len(d) < 5:
            continue
        km = KaplanMeierFitter().fit(d[OUTCOME_TIME], d[OUTCOME_EVENT], label=label)
        km.plot_survival_function(ax=ax, color=color, ci_alpha=0.10, lw=2.2)
    ax.set_xlim(0, 2029)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Days from study entry")
    ax.set_ylabel("Survival probability")
    ax.set_title("A care decision, not a disease state")
    ax.legend(loc="upper right")
    viz.despine(ax)
    viz.caption(fig, f"CHF training cohort, n={len(chf):,}. A pre-existing directive tracks the no-DNR\n"
                     f"curve closely; an order written during the admission separates sharply. The variable\n"
                     f"conflates a patient's stated preference with a clinical decision to limit treatment.")
    return viz.save(fig, "10_dnr_survival.png")


ANSWERS = """
ANSWERS
{rule}

A18. WHO THESE PATIENTS ARE
    {n} adults admitted with congestive heart failure, median age {age_med}
    (IQR {age_q1}-{age_q3}), {female}% female, {comorb_3plus}% carrying three or
    more comorbidities and {diabetes}% diabetic. {white}% white and {black}%
    Black. {dnr_any}% had a DNR order. {deaths} died during follow-up, a crude
    mortality of {mortality}%.

    Two things a clinician reads off that paragraph immediately, neither of
    which appears anywhere in the statistics.

    First, {female}% female is low for a heart failure population. Women are
    roughly half of heart failure admissions and are over-represented in the
    preserved-ejection-fraction phenotype. A cohort this male-weighted is
    plausibly enriched for reduced ejection fraction -- which, as Q21 notes, is
    exactly the thing this dataset cannot confirm.

    Second, a mortality of {mortality}% is far above what a modern heart failure
    cohort produces, and that gap is the subject of Q22.

    Write this paragraph before the modelling, not after. It is what tells a
    reader whether the result could apply to their patients, and no amount of
    discrimination or calibration substitutes for it.

A19. DNR: THE STRONGEST PREDICTOR, AND THE ONE TO REFUSE
    Split by level, the variable falls apart:

        no DNR                {mort_none}% mortality, median survival {surv_none} d
        DNR pre-existing      {mort_pre}% (n={n_pre}), log-rank vs none p={p_pre}
        DNR during admission  {mort_ina}% (n={n_ina}), median survival {surv_ina} d,
                              log-rank vs none p={p_ina}

    A pre-existing directive carries essentially the no-DNR mortality. An order
    written during the admission nearly doubles it -- a {dnr_gap} point gap
    against no DNR, where creatinine, the strongest physiologic predictor
    available, manages {crea_gap}.

    That asymmetry is the argument. If DNR status were a marker of how sick a
    patient is, both levels would move together. They do not. What separates
    them is not the patient's condition but WHEN the decision was made and by
    whom: an advance directive is the patient's own statement of values, known
    at admission; an order written on day four is a clinician's response to
    deterioration, and usually a decision to limit treatment.

    A model that uses the second one learns that clinicians judged this patient
    to be dying, and then predicts death. It will discriminate beautifully and
    it is unusable. Deployed, it would identify as high-risk precisely those
    patients already receiving less aggressive care, and recommend less
    aggressive care -- a self-fulfilling prophecy with an AUC attached. This is
    a recognised hazard in prognostic modelling for critical illness, and it is
    the objection most likely to end a presentation to clinicians badly.

    So `dnr` is replaced by two derived variables. `dnr_preexisting` stays: it
    is knowable at the origin and it reflects the patient. `dnr_in_admission`
    joins the exclusion list beside `dnrday`, which was already excluded for the
    same reason -- an inconsistency in this project's own rule until now.

    Note what would have been lost by handling this mechanically. Dropping the
    whole variable discards a legitimate advance directive; keeping it whole
    imports the prophecy. The split is only defensible because the two levels
    behave differently, which is a fact about the data rather than a preference.

A20. THE PREDICTION ORIGIN
    {day1}% were enrolled on hospital day 1, but {late}% after day 7, the latest
    on day {max_hday}. Their mortality differs sharply: {mort_day1}% for day-1
    enrolments against {mort_later}% for the rest (p={origin_p}).

    That breaks an assumption the project has been making silently. A prediction
    model needs an origin -- the moment the prediction is made -- and every
    patient's covariates must be measured at the same point relative to it.
    Here, "baseline" physiology for a patient enrolled on day 12 was recorded
    after twelve days of hospital course, treatment and possible deterioration.
    They are not comparable to a patient measured on admission, and they are
    self-selected: to be enrolled on day 12 you must survive to day 12, and
    still be sick enough to qualify.

    The honest response is to state the origin explicitly -- admission for this
    cohort -- and show that the conclusions do not depend on the misalignment.
    The sensitivity analysis above restricts to day-1 enrolments and the BUN
    finding is unchanged: follow-up ratio {sens_ratio}, binary gap {sens_gap}
    points, log-rank p={sens_p}. Still an artefact, on an aligned cohort.

    For the modelling stage the same discipline applies: report the primary
    analysis on the full cohort and the day-1 restriction alongside it. If they
    disagree, the disagreement is the finding.

A21. WHAT IS NOT HERE
    {n_absent} of {n_expected} variables a heart failure specialist would expect
    are absent, and the first one is disqualifying for some claims.

    There is no EJECTION FRACTION. Modern heart failure is defined by it --
    reduced (HFrEF, LVEF <= 40%), mildly reduced, and preserved (HFpEF, >= 50%)
    -- and the three differ in pathophysiology, in prognosis, and in which drugs
    work at all. Every guideline, every trial and every clinic conversation is
    organised around that split. This cohort cannot be assigned to it. So no
    result here may be stated as applying to HFrEF or to HFpEF; it applies to
    "hospitalised heart failure" as a single undifferentiated group, which is
    not how the condition has been thought about for twenty years.

    There is no NYHA class, so functional severity is unmeasured except through
    generic ADLs. No BNP or NT-proBNP -- unsurprising, since they entered
    practice after this cohort closed, but they are now the standard objective
    marker of decompensation. No echocardiography, no ECG or rhythm, so atrial
    fibrillation is invisible. No medications, so it is impossible to know who
    was treated and how, and heart failure prognosis is substantially a function
    of treatment. No revascularisation history, which is precisely the exposure
    an interventional cardiologist cares most about.

    And no cause of death. Only all-cause mortality is available, so
    cardiovascular death cannot be separated from death with heart failure
    incidentally present -- in a cohort with this much cancer and dementia, that
    distinction matters. It is also the one framing in which competing risks
    would legitimately apply, and their absence is why this project does not
    use them.

    State these before someone asks. A limitation you raise yourself reads as
    command of the material; the same limitation raised by a reviewer reads as
    an oversight.

A22. WHAT TRANSFERS AND WHAT DOES NOT
    The cohort closed in 1994. Essentially the entire modern heart failure
    armamentarium post-dates it: beta-blockers established in HFrEF from
    1996-99, spironolactone in 1999, implantable defibrillators and
    resynchronisation in the early 2000s, sacubitril/valsartan in 2014, SGLT2
    inhibitors in 2019-20. A patient admitted today receives treatment that did
    not exist for anyone in this dataset.

    So the {mortality}% mortality here does not transfer, and neither does any
    absolute risk estimate built from it. A model calibrated on this cohort
    would systematically over-predict death in a contemporary population, and
    calibration -- the thing this project argues hardest for -- is exactly what
    fails first when a model moves across eras.

    What does transfer is structural, and it is worth being precise about the
    distinction:

      * that missingness can encode a data-collection protocol rather than a
        patient state, and that a cumulative-outcome test cannot see it;
      * that renal function is prognostic in heart failure and its effect
        saturates rather than climbing linearly;
      * that a treatment-limitation decision will dominate any physiologic
        predictor if you let it into the model.

    Mechanisms travel; coefficients do not. Presenting this as "a model for
    predicting heart failure mortality" would be indefensible. Presenting it as
    a methods study on a historical cohort, whose value is the reasoning rather
    than the numbers, is both defensible and the more interesting claim.

    The honest closing statement: this cohort supports conclusions about how to
    analyse clinical data. It does not support a deployable risk score, and no
    amount of additional modelling on it would change that.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train

    header("SUPPORT2 -- the clinical read")
    print(f"  CHF TRAINING cohort: {len(chf):,} patients, "
          f"{int(chf[OUTCOME_EVENT].sum()):,} deaths")

    d = compute_cohort_description(chf)
    report_cohort_description(d)

    dnr = compute_dnr(chf)
    report_dnr(dnr)

    origin = compute_prediction_origin(chf)
    report_prediction_origin(origin)
    sens = sensitivity_day1(chf, origin["day1_frame"])

    absent = compute_absent_variables(chf)
    report_absent_variables(absent)

    report_transportability(d)

    header("FIGURES")
    path = figure_dnr(chf)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    print(render_answers(ANSWERS,
                         dict(collect_facts(d, dnr, origin, absent, sens), rule=RULE)))


if __name__ == "__main__":
    run_and_capture(main, OUT_DIR / "04_clinical.txt")
