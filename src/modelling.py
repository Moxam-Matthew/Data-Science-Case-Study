"""
Modelling scaffolding: the outcome, the preprocessing pipeline, and evaluation.

Three things live here because getting any of them wrong is silent.

The OUTCOME is 180-day all-cause mortality. The horizon is not arbitrary: no
patient in the CHF training cohort is censored before day 180, so the binary
label is complete and needs no censoring assumption, and `prg6m` -- the
attending physician's own 6-month survival estimate -- is measured at exactly
that horizon, which makes the benchmark a direct head-to-head rather than an
approximation. 03_cohort.py Q14 also showed the hazard is front-loaded, so this
is where the data is dense and where a discharge decision actually sits.

The PIPELINE exists so that imputation is fitted inside each cross-validation
fold. Imputing once on the full training set before splitting leaks information
from held-out rows into the model that predicts them, and it is the most common
silent error in clinical prediction pipelines. Every estimator here is wrapped
so that scikit-learn refits the imputer on each training fold.

EVALUATION reports calibration alongside discrimination. A model can reach AUC
0.75 and still tell a patient their risk is 30% when it is 60%; discrimination
ranks patients, calibration is what makes an individual number mean something.
TRIPOD+AI requires both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from support2 import OUTCOME_EVENT, OUTCOME_TIME, model_predictors

# ── The outcome ──────────────────────────────────────────────────────────────
HORIZON_DAYS = 180
OUTCOME_LABEL = f"{HORIZON_DAYS}-day all-cause mortality"

# Physician benchmark measured at the same horizon. Never a model input.
PHYSICIAN_BENCHMARK = "prg6m"

# Missingness indicators are added for these and no others. 01_eda.py Q5 showed
# these are the only variables whose missingness survives a time-to-event test
# with balanced follow-up; the lab variables' apparent signal was an artefact of
# enrolment wave, so an indicator on `bun` would encode calendar time.
MISSINGNESS_INDICATORS = ["income", "adlp"]

# Structurally absent for a whole enrolment wave (03_cohort.py Q16: 100% missing
# early, ~1-9% late). Kept in the primary model because MICE conditioned on
# strong auxiliaries is the right tool for missing-by-design, but the sensitivity
# analysis drops them -- for those patients the imputed value carries no
# information about the patient, only about the cohort.
PROTOCOL_MISSING = ["bun", "urine", "glucose"]

RANDOM_STATE = 20260901
CV_FOLDS = 5
CV_REPEATS = 5


def make_outcome(df: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.Series:
    """
    Binary mortality by `horizon` days.

    Anyone censored before the horizon has unknown status and cannot be labelled.
    At 180 days there are none, which is checked rather than assumed -- if a
    future cohort or horizon changes that, this raises instead of silently
    labelling censored patients as survivors, which would bias the outcome
    downward exactly among the least-observed patients.
    """
    censored_early = (df[OUTCOME_EVENT] == 0) & (df[OUTCOME_TIME] < horizon)
    if censored_early.any():
        raise ValueError(
            f"{int(censored_early.sum())} patients are censored before day "
            f"{horizon}; their {horizon}-day status is unknown. Either restrict "
            f"the cohort or use a time-to-event model."
        )
    return ((df[OUTCOME_EVENT] == 1) & (df[OUTCOME_TIME] <= horizon)).astype(int)


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """
    Append `<col>_missing` flags before imputation fills the values in.

    Order matters: once the imputer has run there is nothing left to flag, so
    this must sit ahead of it in the pipeline rather than beside it.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X, y=None):
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X):
        out = X.copy()
        for c in self.columns_:
            out[f"{c}_missing"] = out[c].isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) +
                          [f"{c}_missing" for c in self.columns_])


def split_feature_types(df: pd.DataFrame, predictors: list[str]
                        ) -> tuple[list[str], list[str]]:
    numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in predictors if c not in numeric]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, predictors: list[str],
                       scale: bool = True) -> ColumnTransformer:
    """
    Numeric columns: MICE-style iterative imputation, then optional scaling.
    Categorical: mode imputation, then one-hot.

    `sample_posterior=False` means this is single imputation within each fold
    rather than true multiple imputation. That understates uncertainty in the
    coefficients, and the honest fix is to fit across several imputations and
    pool by Rubin's rules. It is a documented simplification, not an oversight;
    05_modelling.py reports the complete-case and normal-fill arms alongside so
    the sensitivity to the choice is visible.

    Scaling is required for penalised regression -- an unscaled elastic net
    penalises a coefficient in mg/dL differently from one in mmHg, so the
    variables it selects would depend on their units. Tree models do not need
    it and do not get it.
    """
    numeric, categorical = split_feature_types(df, predictors)

    num_steps = [("impute", IterativeImputer(max_iter=20, random_state=RANDOM_STATE,
                                             sample_posterior=False))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore",
                                         drop="if_binary", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(df: pd.DataFrame, predictors: list[str], estimator,
                   scale: bool = True,
                   indicators: list[str] | None = None) -> Pipeline:
    """Indicators -> impute -> encode -> estimate, as one refittable unit."""
    ind = MissingnessIndicator(indicators if indicators is not None
                               else MISSINGNESS_INDICATORS)
    # Indicators are added only for columns actually in THIS model's predictor
    # set. Keying off df.columns instead was a latent bug: the transformer sees
    # df[predictors] and so never creates an indicator for a column outside it,
    # while the preprocessor was told to expect one. It stayed hidden until a
    # model used a subset of the predictors.
    augmented = list(predictors) + [f"{c}_missing" for c in ind.columns
                                    if c in predictors]
    frame = ind.fit(df[predictors]).transform(df[predictors])
    return Pipeline([
        ("indicators", ind),
        ("prep", build_preprocessor(frame, augmented, scale=scale)),
        ("model", estimator),
    ])


def default_predictors(df: pd.DataFrame, drop_protocol_missing: bool = False
                       ) -> list[str]:
    preds = model_predictors(df)
    if drop_protocol_missing:
        preds = [c for c in preds if c not in PROTOCOL_MISSING]
    return preds


# ── Evaluation ───────────────────────────────────────────────────────────────
def calibration_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    """
    Calibration-in-the-large and calibration slope, on the logit scale.

    Slope 1 and intercept 0 is perfect. Slope below 1 means predictions are too
    extreme -- the usual signature of overfitting. Intercept away from 0 means
    the average predicted risk is wrong, which is what breaks first when a model
    is moved to a population with different prevalence.
    """
    import statsmodels.api as sm

    eps = 1e-9
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    slope_fit = sm.Logit(y_true, sm.add_constant(logit)).fit(disp=0)
    intercept_fit = sm.Logit(y_true, np.ones_like(logit),
                             offset=logit).fit(disp=0)
    return {
        "calibration_slope": float(slope_fit.params[1]),
        "calibration_intercept": float(intercept_fit.params[0]),
        "brier": float(np.mean((p - y_true) ** 2)),
        "mean_predicted": float(p.mean()),
        "observed": float(y_true.mean()),
    }


def discrimination_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {"auc": float(roc_auc_score(y_true, p)),
            "pr_auc": float(average_precision_score(y_true, p))}


def bootstrap_auc_difference(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                             n_boot: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """
    Percentile bootstrap CI for AUC(a) - AUC(b), resampling patients.

    A point estimate of a difference in AUC is not a finding. Two models that
    differ by 0.02 with an interval spanning zero are, on this evidence, the
    same model -- and saying so is the answer that distinguishes judgement from
    a leaderboard.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = (roc_auc_score(y_true[idx], p_a[idx])
                    - roc_auc_score(y_true[idx], p_b[idx]))
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(roc_auc_score(y_true, p_a)
                                - roc_auc_score(y_true, p_b)),
            "ci_low": float(lo), "ci_high": float(hi),
            "crosses_zero": bool(lo <= 0 <= hi)}


def net_benefit(y_true: np.ndarray, p: np.ndarray,
                thresholds: np.ndarray) -> np.ndarray:
    """
    Decision curve analysis (Vickers & Elkin).

        NB = TP/n - (FP/n) * pt/(1-pt)

    The weight pt/(1-pt) is the exchange rate a clinician implicitly accepts by
    choosing a threshold: treating at 20% risk says one missed case is worth
    four unnecessary treatments. Net benefit puts model, treat-all and
    treat-none on that single scale, which answers "is this clinically useful"
    rather than "is this statistically accurate" -- a model can improve AUC and
    still be worth nothing at every threshold anyone would use.
    """
    n = len(y_true)
    out = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        flag = p >= pt
        tp = np.sum(flag & (y_true == 1))
        fp = np.sum(flag & (y_true == 0))
        out[i] = tp / n - (fp / n) * (pt / (1 - pt))
    return out


def treat_all_net_benefit(y_true: np.ndarray,
                          thresholds: np.ndarray) -> np.ndarray:
    prevalence = y_true.mean()
    return prevalence - (1 - prevalence) * thresholds / (1 - thresholds)


def cross_val_predictions(pipeline, X: pd.DataFrame, y: np.ndarray,
                          n_splits: int = CV_FOLDS, n_repeats: int = CV_REPEATS,
                          seed: int = RANDOM_STATE,
                          label: str = "") -> np.ndarray:
    """
    Out-of-fold predicted probabilities, averaged over repeats.

    Repeated stratified k-fold rather than a single split: one 5-fold run gives
    an estimate with enough variance that model rankings can flip between runs,
    which is a bad way to choose anything.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    import sys
    import time

    from sklearn.base import clone

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    total = n_splits * n_repeats
    acc = np.zeros(len(y))
    counts = np.zeros(len(y))
    started = time.time()
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        pipe = clone(pipeline)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        acc[test_idx] += pipe.predict_proba(X.iloc[test_idx])[:, 1]
        counts[test_idx] += 1
        if label:
            # stderr, not stdout: run_and_capture() redirects stdout to build the
            # committed transcript, so progress written there would be invisible
            # during the run and would pollute the transcript afterwards.
            elapsed = time.time() - started
            eta = elapsed / i * (total - i)
            print(f"    {label:<26} fold {i:>2}/{total}  "
                  f"{elapsed:5.0f}s elapsed, ~{eta:4.0f}s remaining",
                  file=sys.stderr, flush=True)
    if label:
        print(f"    {label:<26} DONE {total} folds in "
              f"{time.time() - started:.0f}s", file=sys.stderr, flush=True)
    return acc / np.maximum(counts, 1)
