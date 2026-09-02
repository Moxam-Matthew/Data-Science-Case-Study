"""
Loading and column governance for the SUPPORT2 dataset.

SUPPORT2 (Study to Understand Prognoses and Preferences for Outcomes and Risks
of Treatments) follows 9,105 seriously ill hospitalised adults across five US
teaching hospitals. It is the reference dataset in Harrell's *Regression
Modelling Strategies*, which is why it is used here: the methods this project
demonstrates -- calibration, competing risks, multiple imputation -- are the
methods that dataset was assembled to teach.

Source: UCI Machine Learning Repository, dataset 880.
Citation: Harrell, F. (1995). SUPPORT2. https://doi.org/10.3886/ICPSR02957.v2

No row-level data is committed to this repository. `load_support2()` reads a
local copy if one exists and otherwise downloads from UCI. That constraint is
deliberate rather than incidental -- most clinical data use agreements prohibit
redistributing patient records, so the project is built to that standard from
the start.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "support2.csv"
ZIP_PATH = DATA_DIR / "support2csv.zip"
UCI_DATASET_ID = 880


# ── Column governance ────────────────────────────────────────────────────────
# The single most consequential decision in a clinical prediction model is not
# which algorithm you pick. It is which columns you refuse to use.
#
# UCI ships SUPPORT2 with only `death` and `hospdead` flagged as targets, which
# leaves everything below sitting in the feature matrix. Handing that frame
# straight to a model produces excellent metrics and a worthless model.

OUTCOME_EVENT = "death"        # 1 = died during follow-up, 0 = censored alive
OUTCOME_TIME = "d.time"        # days from study entry to death or last contact
OUTCOME_INHOSP = "hospdead"    # 1 = died before discharge

LEAKAGE_COLUMNS: dict[str, str] = {
    "d.time": "Follow-up duration. The survival *time*, not a covariate. As a "
              "predictor of death it is close to an oracle.",
    "hospdead": "In-hospital death. A component of the outcome.",
    "sfdm2": "Functional disability measured at 2 months. Post-baseline; its "
             "largest category is literally '<2 mo. follow-up'.",
    "surv2m": "The original SUPPORT model's own 2-month survival estimate. "
              "Using it means re-predicting an existing model's output.",
    "surv6m": "As surv2m, at 6 months.",
    "prg2m": "Attending physician's 2-month survival estimate. This is the "
             "clinical benchmark to beat, not a model input.",
    "prg6m": "As prg2m, at 6 months.",
    "aps": "APACHE III physiology score, computed from the same vitals and "
           "labs used as predictors. Collinear by construction.",
    "sps": "SUPPORT physiology score. As aps.",
    "slos": "Length of stay. Known only after the episode ends.",
    "charges": "Total charges. Post-hoc resource use.",
    "totcst": "Total ratio-of-cost-to-charges. Post-hoc resource use.",
    "totmcst": "Total micro-cost. Post-hoc resource use.",
    "avtisst": "Average TISS score across the stay. Post-hoc care intensity.",
    "dnrday": "Day the DNR order was written. Encodes timing of a decision "
              "that often follows clinical deterioration.",
}

# Available at or near admission, so legitimate candidate predictors.
DEMOGRAPHIC = ["age", "sex", "race", "edu", "income"]
CLINICAL = ["dzgroup", "dzclass", "num.co", "scoma", "diabetes", "dementia",
            "ca", "hday", "dnr"]
PHYSIOLOGY = ["meanbp", "hrt", "resp", "temp", "wblc", "pafi", "alb", "bili",
              "crea", "sod", "ph", "glucose", "bun", "urine"]
FUNCTIONAL = ["adlp", "adls"]

# Excluded from the candidate set on evidence, not on principle. 02_profile.py
# found adlsc correlates with adls at rho = 1.000 and leaves the design matrix
# rank-deficient: it is a derived ADL summary, exactly determined by its
# components. Including it alongside them asks the fit for a coefficient that
# does not exist.
DERIVED_DUPLICATES = {
    "adlsc": "Derived ADL summary; rho = 1.000 with adls, design matrix "
             "rank-deficient when both are present.",
}

CANDIDATE_PREDICTORS = DEMOGRAPHIC + CLINICAL + PHYSIOLOGY + FUNCTIONAL

# Units matter: a plausibility bound without a unit is not reviewable, and a
# clinician reading an odds ratio needs to know what one unit is.
UNITS = {
    "age": "years", "edu": "years of schooling", "num.co": "count",
    "scoma": "SUPPORT coma score (0-100)", "hday": "days",
    "meanbp": "mmHg", "hrt": "beats/min", "resp": "breaths/min", "temp": "degC",
    "wblc": "10^3/uL", "pafi": "PaO2/FiO2 ratio", "alb": "g/dL",
    "bili": "mg/dL", "crea": "mg/dL", "sod": "mEq/L", "ph": "pH units",
    "glucose": "mg/dL", "bun": "mg/dL", "urine": "mL/24h",
    "adlp": "ADL count (0-7)", "adls": "ADL count (0-7)",
    "d.time": "days", "slos": "days",
}

# Recommended normal-fill values published by the SUPPORT investigators for the
# physiologic variables. Kept here as a documented clinical baseline to compare
# multiple imputation against -- not as the project's imputation strategy.
SUPPORT_NORMAL_FILL = {
    "alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01,
    "bun": 6.51, "wblc": 9.0, "urine": 2502.0,
}

CHF_LABEL = "CHF"


def load_support2(path: Path | None = None) -> pd.DataFrame:
    """
    Return the full SUPPORT2 frame (9,105 x 47).

    Resolution order: an explicit path, then the extracted CSV, then the
    downloaded zip, then a fresh download from UCI.
    """
    if path is not None:
        return pd.read_csv(path)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as fh:
                return pd.read_csv(fh)
    return _download_from_uci()


def _download_from_uci() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "No local copy of SUPPORT2 found and `ucimlrepo` is not installed.\n"
            "Either `pip install ucimlrepo`, or download dataset 880 from\n"
            "https://archive.ics.uci.edu/dataset/880/support2 into ./data/."
        ) from exc

    repo = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def chf_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to the congestive heart failure disease group.

    Holding the disease group fixed is not merely a scoping choice -- it is the
    control condition for the missingness analysis in 01_eda.py. Several
    variables look informatively missing across the full cohort purely because
    they track which service a patient was on.
    """
    return df[df["dzgroup"] == CHF_LABEL].copy()


def cohort_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    CONSORT-style attrition: how a cohort of 9,105 becomes the analysis set.

    A sample size that appears without derivation is not reviewable. Every
    exclusion below is stated with its reason and its cost in patients.
    """
    steps = []
    n = len(df)
    steps.append({"step": "SUPPORT2 enrolled patients", "excluded": 0, "remaining": n})

    chf = df[df["dzgroup"] == CHF_LABEL]
    steps.append({"step": f"Restrict to dzgroup == '{CHF_LABEL}'",
                  "excluded": n - len(chf), "remaining": len(chf)})
    n = len(chf)

    have_outcome = chf[chf[OUTCOME_EVENT].notna() & chf[OUTCOME_TIME].notna()]
    steps.append({"step": "Complete outcome (death and follow-up time)",
                  "excluded": n - len(have_outcome), "remaining": len(have_outcome)})
    n = len(have_outcome)

    positive_fu = have_outcome[have_outcome[OUTCOME_TIME] > 0]
    steps.append({"step": "Follow-up time > 0 days",
                  "excluded": n - len(positive_fu), "remaining": len(positive_fu)})

    return pd.DataFrame(steps)


def make_split(df: pd.DataFrame, test_frac: float = 0.30,
               seed: int = 20260901) -> pd.Series:
    """
    Deterministic, outcome-stratified train/test assignment.

    Returns a Series of 'train'/'test' aligned to df.index.

    Why this exists, and why it is fixed by seed rather than stored: the
    exploratory work in 01_eda.py and 02_profile.py looked at the outcome
    roughly sixty times across three scripts. Every estimate produced by that
    process is optimistic by an unquantifiable amount. The only repair is a
    partition made once, held fixed, and never inspected -- so it is generated
    from a constant seed and reproduces exactly without needing patient data
    committed to the repository.

    The test set is not touched by any exploratory analysis. Everything before
    the modelling stage runs on train only.
    """
    rng = np.random.default_rng(seed)
    assignment = pd.Series("train", index=df.index, dtype=object)
    # Stratify on the event so both partitions carry the same event fraction.
    for _, idx in df.groupby(df[OUTCOME_EVENT]).groups.items():
        idx = np.sort(np.asarray(idx))          # sort => order-independent
        n_test = int(round(len(idx) * test_frac))
        assignment.loc[rng.choice(idx, size=n_test, replace=False)] = "test"
    return assignment


def train_set(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """The partition all exploratory and model-fitting work may see."""
    return df[make_split(df, **kw) == "train"].copy()


def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every column as outcome, excluded, or candidate predictor."""
    rows = []
    for col in df.columns:
        if col in (OUTCOME_EVENT, OUTCOME_TIME, OUTCOME_INHOSP):
            role, reason = "outcome", "Defines the event or the time to it."
        elif col in LEAKAGE_COLUMNS:
            role, reason = "excluded", LEAKAGE_COLUMNS[col]
        elif col in DERIVED_DUPLICATES:
            role, reason = "excluded", DERIVED_DUPLICATES[col]
        elif col in CANDIDATE_PREDICTORS:
            role, reason = "predictor", "Available at or near admission."
        else:
            role, reason = "unreviewed", "Not yet classified."
        rows.append({"column": col, "role": role,
                     "missing_pct": round(df[col].isna().mean() * 100, 1),
                     "reason": reason})
    return pd.DataFrame(rows)
