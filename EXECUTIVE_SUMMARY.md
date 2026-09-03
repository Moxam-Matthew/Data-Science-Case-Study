# Executive Summary

**Predicting 180-day mortality after heart failure admission — SUPPORT2, n=1,387**

A clinical prediction study run to the standards journals apply, with a held-out
partition spent once. The headline result is a **negative** one, and the most
useful outputs are two claims that did not survive validation.

---

## Bottom line

**A penalised logistic regression using 28 routine admission variables predicts
180-day mortality with AUC 0.655 on 409 patients it had never seen.** Gradient
boosting does not beat it. A seven-variable model a clinician could compute by hand
is close behind.

**This is not a deployable risk score, and three findings say so explicitly.** They
are stated here rather than in an appendix because they are the point.

| | Held-out result |
|---|---|
| Primary model (elastic net, 28 vars) | **AUC 0.655** |
| XGBoost | AUC 0.647 |
| Clinical model (7 vars, hand-computable) | AUC 0.628 |
| **XGBoost − elastic net** | **−0.007 [−0.055, +0.043]** — indistinguishable |
| Attending physician, same patients | **AUC 0.711** |

---

## What was confirmed

**Machine learning did not outperform regression.** Pre-specified before the holdout
was read: the AUC difference between XGBoost and the penalised regression would
include zero. It did, on both partitions independently — cross-validated
−0.005 [−0.036, +0.028], held-out −0.007 [−0.055, +0.043]. This is the most robust
finding in the study and it matches the literature (Christodoulou et al.,
*J Clin Epidemiol* 2019: no consistent ML benefit on structured clinical data).

**Discrimination transferred.** AUC fell 0.023 from the cross-validated estimate —
within sampling variation at n=409.

**Parsimony is nearly free.** Seven clinically chosen variables (`age`, mean arterial
pressure, `sodium`, `creatinine`, coma score, comorbidity count, ADL dependence) cost
0.015 AUC in development while achieving *better* calibration and *higher* PR-AUC. For
adoption, that is the model that matters.

---

## What did not survive — reported because it is the point

### 1. The physician outperformed the model on held-out patients

On training data the model led the attending physician's own prognosis (AUC 0.687 vs
0.655). **On held-out patients the direction reverses: physician 0.711, model 0.633**,
a difference of −0.079 [−0.154, +0.003].

A post-hoc check found the model still adds information *given* the physician
(likelihood-ratio χ²=5.0, p=0.026; combined AUC 0.728). That analysis was run after
seeing the primary result and is labelled post-hoc; it carries less weight than the
pre-specified comparison.

**A study reporting only the training comparison would have published a conclusion its
own data contradicts.** This is what a held-out partition is for.

### 2. Calibration failed, and the cause is our own design defect

Calibration slope fell from 1.19 to **0.706**. The model predicted a mean risk of
**26.0%** against **30.8%** observed — systematic under-prediction.

The cause is identifiable and is not the model. The train/test split was stratified on
*all-cause death over full follow-up* — perfectly balanced, 68.1% in both partitions —
but the outcome actually modelled was *180-day mortality in the CHF subgroup*, which
the stratification never balanced: **25.4% training vs 30.8% held-out, a 5.4-point
gap.** A model calibrated to a 25% event rate applied to a 31% one will under-predict.

**The fix is one line, and it was deliberately not applied.** Re-splitting after seeing
the test result would mean choosing a partition on the basis of its answer, which is
the one thing a held-out set cannot survive. It is recorded as the first correction for
any future version.

### 3. The cohort was never large enough

Riley et al. (2019) minimum sample size for this model: **2,465 patients**. Available:
**978**. The binding criterion is shrinkage ≤10%, and the shortfall is roughly 2.5×.
The older "10 events per variable" rule would have demanded only 1,538 — it passes a
model the modern criteria fail.

Bootstrap optimism correction measured what Riley predicted: the unpenalised model
needs its coefficients multiplied by **0.74** to stop overfitting. The penalised model
needs **1.00**. Penalisation was not a judgement call; it was arithmetically required.

---

## Method integrity

| Safeguard | Implementation |
|---|---|
| Held-out partition | 30%, seeded, **read exactly once**, by a separately named function so no other script can reach it |
| Imputation | MICE refitted **inside every CV fold**; a test asserts the fitted state differs per fold |
| Model selection | Nested CV — hyperparameter search sits inside the pipeline |
| Multiplicity | Benjamini–Hochberg across all multi-variable tables |
| Reporting standard | TRIPOD+AI: calibration reported alongside discrimination throughout |
| Reproducibility | Dependencies pinned exactly; golden-value tests pin every published number so a dependency change breaks the build |

Every quantitative claim in this repository is interpolated from a run rather than
typed. Full console transcripts are committed under [`output/`](output/).

**Two findings this discipline killed during development**, both of which looked real
first: education-related missingness (p=0.004 unadjusted → q=0.253 corrected) and
heart-rate non-linearity (p=0.021 → q=0.093).

---

## The most transferable finding

Six variables appeared informatively missing on a conventional chi-square test, with
mortality gaps of 10–20 points at p<0.001. **Half were artefacts.**

Patients missing BUN were followed a median 1,690 days against 664 — 2.55× longer. More
of them had died by study close because they were *watched longer*, not because they
died faster. A log-rank test on the same split is null.

The mechanism was then proven rather than hypothesised: censoring times are bimodal
with an empty interval at ~1,150 days — the signature of two enrolment waves closed on
one date. **BUN, urine and glucose are 100.0% missing in the early wave against
0.9–9.1% in the late one.** That is a data-collection protocol, not a patient
characteristic.

*A cumulative-outcome test cannot see exposure time. Whenever a group difference in a
cumulative outcome is the headline, check whether the groups were observed for equally
long before believing it.*

---

## What may not be claimed

- **Not a deployable score.** Underpowered by Riley's criteria; coefficients carry more
  uncertainty than their intervals suggest.
- **Absolute risks do not transfer to modern practice.** The cohort closed in 1994 —
  before beta-blockers in HFrEF, spironolactone, ICD/CRT, ARNIs and SGLT2 inhibitors.
  Discrimination may travel; calibration will not.
- **No HFrEF/HFpEF claim is possible.** The dataset has no ejection fraction, so the
  cohort cannot be assigned to either phenotype — nor NYHA class, BNP, ECG,
  medications, revascularisation history, or cause of death.
- **Internal validation only.** Same five hospitals, same years, same protocol. No
  external cohort.
- **The seven-variable specification was not pre-registered.** It was chosen after
  seeing which variables the penalised model selected.

---

## Honest one-line summary

**This is a methods study on a historical cohort that produces a defensible internal
estimate, a clear negative result about model complexity, and two claims that did not
survive validation. It is not a product.**

---

*Data: SUPPORT2, UCI Machine Learning Repository [dataset 880](https://archive.ics.uci.edu/dataset/880/support2).
Harrell, F. (1995). No patient-level data is committed to this repository.
Full methodology: [README.md](README.md) — 45 questions across 10 analysis scripts.*
