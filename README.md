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

Patients missing BUN were followed for a median **1,689 days against 655** — 2.6×
longer. More of them had died by the time the study closed because they were
watched for longer, not because they died faster. Their survival curves are
indistinguishable (log-rank p=0.67) despite a 20-point gap in cumulative death.

The variables that *do* survive the time-to-event test are not laboratory values at
all. They are the three collected by **interviewing the patient** — functional
status, income, education — where follow-up is balanced (ratio 1.03–1.13) and the
curves genuinely separate. Interview non-response is caused by the patient's
condition. That is informative missingness in the textbook sense, and it is the one
place here the textbook applies.

| | binary p | log-rank p | censored follow-up ratio | verdict |
|---|---|---|---|---|
| `bun`, `urine`, `glucose` | <0.001 | 0.50 – 0.91 | ~2.5× | follow-up artefact |
| `adlp`, `income`, `edu` | <0.001 – 0.010 | <0.001 – 0.004 | 1.03 – 1.13 | real signal |

The practical consequence: a missingness indicator belongs on `adlp`, `income` and
`edu` only. Adding one for `bun` would encode enrolment era, not patient state — and
leave you explaining a coefficient for "BUN was not drawn" to a room of clinicians
with no clinical story to tell.

---

## Cohort profile and functional form

[`02_profile.py`](02_profile.py) covers what a clinical reviewer checks next.

**Table 1** is reported with **standardised mean differences, not p-values**. With
n=1,387 a clinically trivial difference clears p<0.05, so the p-value measures
sample size rather than importance — and there is no sampling to make inference
about, since these *are* the two outcome groups. |SMD| > 0.1 is the conventional
imbalance threshold.

**Physiologic plausibility** flagged 22 impossible cell values, including an albumin
of **29.0 g/dL** (normal 3.5–5.0; incompatible with life above ~7) which alone
produces the skew of 12.1, and zeros in mean arterial pressure, heart rate and
respiratory rate. These are set to missing rather than dropped — it is a cell-level
error, and deleting the patient discards their valid measurements too.

**Linearity is tested, not assumed.** A likelihood-ratio test of a linear term
against a natural cubic spline:

![Functional form](output/figures/05_functional_form.png)

Creatinine rejects linearity decisively (p<0.001, AIC gain 10.4) — risk steps up
around 1.2 mg/dL then **plateaus**, while a linear term extrapolates a rising slope
into a tail where almost no patients exist. Heart rate also rejects (p=0.024). But
age, mean arterial pressure, temperature and BUN all test as adequately linear, and
their splines are *worse* by AIC. Eyeballing the age octiles suggested curvature;
the formal test disagreed. Testing beat assuming in both directions.

One caution the test surfaced: `resp` read non-linear at p=0.040 before the
plausibility bounds were applied and linear at p=0.085 after — a verdict flipped by
one impossible value. Clean first, then test.

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

**Death is a competing risk.** You cannot be readmitted after dying. Cumulative
incidence is estimated with Aalen–Johansen rather than treating death as censoring.

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
