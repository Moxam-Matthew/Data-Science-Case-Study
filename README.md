# Clinical Prediction Modelling — SUPPORT2

A worked clinical prediction study on **SUPPORT2** (9,105 seriously ill hospitalised
adults), built to the standards clinical journals actually apply: interpretable
effect estimates with confidence intervals, calibration reported alongside
discrimination, explicit handling of censoring and competing risks, and a stated
missing-data mechanism rather than a default imputation call.

The analysis files are written as **questions with the answers held back**. Each
script poses what a reviewer would ask, shows the evidence, and puts the reasoning
at the bottom of the file. The reasoning is the point; the conclusion is cheap.

**Data:** SUPPORT2, UCI Machine Learning Repository [dataset 880](https://archive.ics.uci.edu/dataset/880/support2).
Harrell, F. (1995). https://doi.org/10.3886/ICPSR02957.v2

---

## The headline finding so far

Six variables look informatively missing when you test them the obvious way — a
chi-square on the death flag returns p<0.001 with mortality gaps of 10 to 20
percentage points. **Half of those are artefacts**, and a log-rank test on the same
splits says so.

![Binary vs time-to-event](output/figures/03_binary_vs_time_to_event.png)

Patients missing BUN were followed for a median **1,690 days against 664** — 2.55×
longer. More of them had died by the time the study closed because they were
watched for longer, not because they died faster. Their survival curves are
indistinguishable despite a 21-point gap in cumulative death.

The variables that *do* survive the time-to-event test are not laboratory values at
all. They are collected by **interviewing the patient** — functional status and
income — where follow-up is balanced and the curves genuinely separate. Interview
non-response is caused by the patient's condition. That is informative missingness
in the textbook sense, and it is the one place here the textbook applies.

| | binary p | log-rank q (FDR) | censored follow-up ratio | verdict |
|---|---|---|---|---|
| `bun`, `urine`, `glucose` | <0.001 | 0.82 – 0.95 | **2.55×** | follow-up artefact |
| `income`, `adlp` | <0.001 – 0.003 | <0.001 – 0.004 | 1.13 – 1.15 | real signal |
| `edu` | 0.124 | 0.253 | 1.12 | *did not replicate* |

That last row is the discipline working. On the full cohort with unadjusted
p-values, education looked real (p=0.004). On the training partition with FDR
correction across twelve tests it is gone. Nothing about education changed — it was
simply no longer being judged against a bar it had help clearing.

The practical consequence: a missingness indicator belongs on `income` and `adlp`
only. Adding one for `bun` would encode enrolment era, not patient state — and leave
you explaining a coefficient for "BUN was not drawn" to a room of clinicians with no
clinical story to tell.

---

## Cohort profile and functional form

[`02_profile.py`](02_profile.py) covers what a clinical reviewer checks next.

**Table 1** is reported with **standardised mean differences, not p-values**. A
p-value here measures sample size rather than importance, and there is no sampling
to make inference about since these *are* the two outcome groups. |SMD| > 0.1 is the
conventional imbalance threshold. Multi-level categoricals get a single
Yang–Dalton SMD for the variable rather than one per level, and every level is
reported rather than only the modal one.

**Physiologic plausibility** flagged 22 impossible cell values, including an albumin
of **29.0 g/dL** (normal 3.5–5.0; incompatible with life above ~7) which alone
produces the skew of 12.1, and zeros in mean arterial pressure, heart rate and
respiratory rate. These are set to missing rather than dropped — it is a cell-level
error, and deleting the patient discards their valid measurements too.

**Linearity is tested, not assumed.** A likelihood-ratio test of a linear term
against a natural cubic spline:

![Functional form](output/figures/05_functional_form.png)

Exactly one variable survives: **creatinine** (q=0.036, AIC gain 7.1) — risk steps
up around 1.2 mg/dL then **plateaus**, while a linear term extrapolates a rising
slope into a tail where almost no patients exist.

Heart rate is the instructive near-miss. Unadjusted it looks like a finding at
p=0.021; across nine tests the FDR q-value is 0.093 and it fails. Nine tests at
α=0.05 hand you roughly one false positive for free, and reporting `hrt` as
non-linear would be reporting the multiplicity rather than the biology. Everything
else tests as adequately linear with splines that are *worse* by AIC.

One caution surfaced during development: `resp` changed verdict once the
plausibility bounds were applied, because a single impossible value — a respiratory
rate of 76 — was doing the work. Clean first, then test, and say which order.

**Collinearity** turned up an identity rather than an association: `adls` and `adlsc`
correlate at **rho = 1.000** and the design matrix is rank-deficient, because `adlsc`
is a derived summary of the components. `bun` and `crea` at rho = 0.79 is the real
concern for an interpretable model, where collinearity inflates standard errors and
makes which predictor "wins" close to arbitrary.

---

## Why these methods

**Discrimination is not enough.** A model can reach AUC 0.75 and still tell a patient
their risk is 30% when it is 60%. Calibration slope, intercept and Brier score are
reported alongside AUC, per [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378).

**Odds ratios need confidence intervals.** The interpretable model is fitted with
`statsmodels`, not `sklearn` — `sklearn.LogisticRegression` applies L2 regularisation
by default, so its coefficients are penalised rather than maximum-likelihood, and it
exposes no standard errors.

**The benchmark is the clinician.** SUPPORT2 records the attending physician's own
survival estimates (`prg2m`, `prg6m`). The question worth answering is not whether a
model beats chance but whether it beats the doctor — which means those columns are
the comparator, never model inputs.

**There is no competing risk here, and saying so matters.** Competing-risks methods
are standard in cardiology and it would be easy to reach for them, but the outcome in
this project is *all-cause* mortality — `death` is complete, and `hospdead` is a
strict subset with zero contradictions. Nothing competes with dying. Fine–Gray or
Aalen–Johansen would apply if the outcome were readmission (death precludes it) or a
cause-specific death, and neither is what is modelled here. Applying them anyway
would be methodological theatre.

**The exploratory work is labelled as exploratory.** Roughly sixty outcome-aware
comparisons were made across the two EDA scripts. All multi-variable tables carry
Benjamini–Hochberg q-values, and a 30% partition (seed `20260901`, stratified on
death) is held out and never read before modelling.

---

## Structure

```
├── 01_eda.py                  # Q1–6:  outcome structure, leakage audit,
│                              #        missingness mechanism, binary vs time-to-event
├── 02_profile.py              # Q7–11: Table 1, physiologic plausibility,
│                              #        distributions, functional form, collinearity
├── src/
│   ├── support2.py            # Loading + column governance (what may be a predictor)
│   └── viz.py                 # Shared figure styling; CVD-validated palette
├── output/figures/            # Generated figures
└── requirements.txt
```

Run:

```bash
pip install -r requirements.txt
python 01_eda.py
python 02_profile.py
```

No patient data is committed. The loader reads a local copy if present and otherwise
downloads from UCI. That is deliberate: most clinical data use agreements prohibit
redistributing row-level records, so the project is built to that standard from the
first commit rather than retrofitted.

---

## Analytic discipline

**Cohort derivation is stated, not assumed.** 9,105 enrolled → 1,387 CHF → complete
outcome → follow-up > 0. Every exclusion carries its cost in patients. A sample size
that appears without derivation is not reviewable.

**A 30% partition is held out and never read.** The two EDA scripts make ~60
outcome-aware comparisons, each a point where the data could steer a modelling
choice. That is analyst degrees of freedom, and it makes any later performance
estimate optimistic by an amount nobody can recover after the fact. The split is
generated from a fixed seed rather than stored, so it reproduces exactly without
committing patient rows.

**Every multi-variable table carries FDR q-values.** Running one test is inference;
running sixty and reporting the small p-values is selection. Two claimed findings
(`edu` missingness, `hrt` non-linearity) do not survive correction and are reported
as not surviving rather than quietly dropped.

**Median follow-up uses reverse Kaplan–Meier.** The median of the time column is a
median *time-to-event*, dragged down by every death; it is not follow-up. On this
cohort the two differ roughly threefold.

---

## Column governance

The most consequential decision here is not the algorithm — it is which columns are
refused. UCI ships SUPPORT2 with only `death` and `hospdead` flagged as targets,
leaving fifteen leakage-prone columns sitting in the feature matrix. `d.time` alone
achieves a univariate AUC of **0.937** against death, because it is follow-up
duration. Handing that frame straight to a model produces excellent metrics and a
worthless result.

The exclusions fall into four kinds, enumerated with reasons in
[`src/support2.py`](src/support2.py): the outcome in disguise, measured-after-baseline,
another model's output, and constructed-from-the-predictors.

---

## Status

- [x] Exploratory analysis, missingness mechanism, leakage audit
- [x] Table 1 with SMDs, plausibility bounds, functional form, collinearity
- [ ] Multiple imputation inside CV folds; complete-case and normal-fill sensitivity
- [ ] Logistic regression with odds ratios and 95% CIs
- [ ] Gradient boosting comparator, with a confidence interval on ΔAUC
- [ ] Calibration curves, Brier score, decision curve analysis
- [ ] Cox and Aalen–Johansen with death as a competing risk
- [ ] Benchmark against physician prognosis (`prg2m` / `prg6m`)

## Limitations

SUPPORT2 was collected at five US teaching hospitals between 1989 and 1994; case
mix, practice patterns and available therapies have all moved since. The follow-up
imbalance documented above is consistent with the study's two enrolment phases, but
the public file ships no phase indicator, so that explanation is a hypothesis rather
than a verified cause. All validation here is internal — no external cohort.
