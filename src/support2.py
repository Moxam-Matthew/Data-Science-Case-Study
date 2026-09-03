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
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "support2.csv"
ZIP_PATH = DATA_DIR / "support2csv.zip"
UCI_DATASET_ID = 880

# The shipped CSV carries an unnamed leading id column: 47 header fields against
# 48 data fields. Named here so the promotion to index is declared rather than
# inferred from that off-by-one, and so both load paths agree.
ID_COLUMN = "id"
N_COLUMNS = 47

# Column order as shipped. ucimlrepo returns features and targets separately,
# which moves `death` from position 1 to position 45; restoring this order keeps
# the download path interchangeable with the local-file path.
SHIPPED_COLUMN_ORDER = ['age', 'death', 'sex', 'hospdead', 'slos', 'd.time', 'dzgroup', 'dzclass', 'num.co', 'edu', 'income', 'scoma', 'charges', 'totcst', 'totmcst', 'avtisst', 'race', 'sps', 'aps', 'surv2m', 'surv6m', 'hday', 'diabetes', 'dementia', 'ca', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls', 'sfdm2', 'adlsc']


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
    "dnr": "Superseded by dnr_preexisting and dnr_in_admission, which separate a "
           "patient's advance directive from a care-limitation decision taken "
           "during the admission. The raw column conflates them.",
    "dnr_in_admission": "DNR order written DURING the study admission. Not a "
                        "disease state -- a clinical response to deterioration "
                        "and usually a decision to limit treatment. Mortality "
                        "86.7% against 56.8% with no DNR, while a PRE-EXISTING "
                        "directive carries 58.8%, essentially the no-DNR rate. "
                        "A model using it learns that clinicians judged the "
                        "patient to be dying and then predicts death: a "
                        "self-fulfilling prophecy that would recommend less "
                        "aggressive care for patients already receiving it.",
}

# Available at or near admission, so legitimate candidate predictors.
DEMOGRAPHIC = ["age", "sex", "race", "edu", "income"]
CLINICAL = ["dzgroup", "dzclass", "num.co", "scoma", "diabetes", "dementia",
            "ca", "hday", "dnr_preexisting"]
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

# ── Physiologic plausibility ─────────────────────────────────────────────────
# Bounds are declared here, before anything looks at an outcome, and applied by
# every entry point. They previously lived inside 02_profile.py and were used in
# exactly one function, which meant the data dictionary in 03_cohort.py happily
# published an albumin maximum of 29.0 g/dL two sections after the README called
# that value incompatible with life, and the imputation diagnostic was fitting on
# a mean arterial pressure of zero.
#
# The project's own stated rule is "clean first, then test, and say which order".
# A constant in one script does not implement that rule; a pipeline step does.
#
# Units are in UNITS below. A bound without a unit is not reviewable.
PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "meanbp": (20, 200),     # mmHg; 0 is absence of circulation, not hypotension
    "hrt": (20, 250),        # beats/min
    "resp": (4, 60),         # breaths/min
    "temp": (30, 43),        # degC
    "sod": (110, 175),       # mEq/L
    "crea": (0.1, 20),       # mg/dL
    "wblc": (0.1, 100),      # 10^3/uL
    "ph": (6.8, 7.8),        # pH units
    "glucose": (20, 1000),   # mg/dL
    "alb": (0.5, 7.0),       # g/dL; >7 is incompatible with life
    "bili": (0.05, 60),      # mg/dL
    "bun": (1, 250),         # mg/dL
    "pafi": (20, 700),       # PaO2/FiO2 ratio
}


DNR_PREEXISTING_LABEL = "dnr before sadm"
DNR_IN_ADMISSION_LABEL = "dnr after sadm"


def derive_dnr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split `dnr` into the two clinically distinct things it conflates.

    `dnr_preexisting`  -- an advance directive the patient arrived with. A
        statement of their own values, known at the prediction origin, and a
        legitimate covariate. Mortality 58.8%, against 56.8% for no DNR.

    `dnr_in_admission` -- an order written during the study admission.
        Mortality 86.7%. This is not a property of the patient; it is a decision
        taken in response to deterioration, and usually a decision to limit
        treatment. It is excluded for the same reason `dnrday` is.

    That the pre-existing level carries essentially the no-DNR mortality while
    the in-admission level nearly doubles it is the evidence for the split. Had
    both levels behaved alike, the variable would be a patient characteristic
    and could stay whole.
    """
    out = df.copy()
    if "dnr" not in out:
        return out
    known = out["dnr"].notna()
    out["dnr_preexisting"] = np.where(
        known, (out["dnr"] == DNR_PREEXISTING_LABEL).astype(float), np.nan)
    out["dnr_in_admission"] = np.where(
        known, (out["dnr"] == DNR_IN_ADMISSION_LABEL).astype(float), np.nan)
    return out


def find_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Locate cells outside physiologic bounds, without changing anything."""
    rows = []
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        if col not in df:
            continue
        s = df[col].dropna()
        below, above = int((s < lo).sum()), int((s > hi).sum())
        if below or above:
            offenders = sorted(set(s[(s < lo) | (s > hi)].round(2)))[:6]
            rows.append({"variable": col, "unit": UNITS.get(col, "?"),
                         "bound": f"[{lo}, {hi}]", "below": below, "above": above,
                         "observed_min": s.min(), "observed_max": s.max(),
                         "offending_values": ", ".join(str(v) for v in offenders)})
    return pd.DataFrame(rows)


def apply_plausibility_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Set physiologically impossible cells to missing.

    Cell-level, never row-level. Dropping the patient would discard every valid
    measurement they contributed and would delete rows on the basis of a
    transcription error, which is a form of selection. Voided cells join the
    values that were never recorded and are handled by the same imputation.

    Returns the cleaned frame and the number of cells voided, so the count is
    reportable rather than silent.
    """
    out, n = df.copy(), 0
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        if col not in out:
            continue
        bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        n += int(bad.sum())
        out.loc[bad, col] = np.nan
    return out, n


def load_support2(path: Path | None = None) -> pd.DataFrame:
    """
    Return the SUPPORT2 frame, 9,105 x 47, indexed by patient id.

    Resolution order: an explicit path, then the extracted CSV, then the
    downloaded zip, then a fresh download from UCI.

    A note on the id column, because it is not obvious from the file. The
    shipped CSV has 47 header fields and 48 data fields -- the leading column
    holds a patient id that the header never names. Pandas resolves that
    mismatch by silently promoting the first column to the index, which happens
    to be correct, but the project relied on it without saying so.

    That mattered more than it looks: make_split() partitions on df.index, so
    the train/test assignment was keyed on an identifier nobody had declared.
    The id is now named explicitly and validated, so the behaviour is a
    guarantee rather than a coincidence.
    """
    if path is not None:
        df = _read_csv_with_id(path)
    elif CSV_PATH.exists():
        df = _read_csv_with_id(CSV_PATH)
    elif ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as fh:
                df = _read_csv_with_id(fh)
    else:
        df = _download_from_uci()
    return _validate(df)


def _read_csv_with_id(source) -> pd.DataFrame:
    """
    Read the shipped CSV, promoting the unnamed leading column to a named index.

    `index_col=0` is explicit rather than relying on pandas inferring it from
    the off-by-one header. If UCI ever ships a file whose header names all 48
    columns, the inference would stop firing and every column would shift by
    one; this raises instead, and _validate() catches it either way.
    """
    df = pd.read_csv(source, index_col=0)
    df.index.name = ID_COLUMN
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fail loudly on a source file that is not the one this project was written
    against. Silent misalignment is the failure mode being guarded: a shifted
    header produces a frame that loads cleanly and analyses to nonsense.
    """
    if df.shape[1] != N_COLUMNS:
        raise ValueError(
            f"expected {N_COLUMNS} columns, got {df.shape[1]}. If the source file "
            f"now names its id column, read it with index_col='id' instead."
        )
    missing = {"age", "death", "d.time", "dzgroup"} - set(df.columns)
    if missing:
        raise ValueError(f"expected columns absent: {sorted(missing)}")
    if not df.index.is_unique:
        raise ValueError("patient ids are not unique; the index is unusable as a key")
    # Cheap alignment check. A one-column header shift puts an id into `age`.
    if not df["age"].between(0, 120).all():
        raise ValueError(
            "`age` holds values outside 0-120, which is the signature of a "
            "shifted header rather than of unusual patients"
        )
    return df


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
    # ucimlrepo splits the file into features and targets, which reorders the
    # columns and drops the id. Restore the shipped column order and rebuild a
    # 1-based id, so this path and the local-CSV path are interchangeable --
    # otherwise the same code gives two different frames depending on whether a
    # local copy happened to exist.
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    df = df.reindex(columns=[c for c in SHIPPED_COLUMN_ORDER if c in df.columns])
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name=ID_COLUMN)
    DATA_DIR.mkdir(exist_ok=True)
    # index=True, so a reload sees the same 48-field rows the shipped file has.
    df.to_csv(CSV_PATH, index=True, index_label="")
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


def model_predictors(df: pd.DataFrame) -> list[str]:
    """
    Candidate predictors with zero-variance columns removed for THIS cohort.

    `dzgroup` and `dzclass` are legitimate predictors across the whole study and
    are what Q4 conditions on -- but inside the CHF restriction they are
    constant by construction, since the restriction is defined on them. A
    constant column contributes nothing and can make a design matrix singular.

    The data dictionary in 03_cohort.py surfaces this; this is where it is acted
    on. Cohort-dependent, so it is computed rather than hardcoded.
    """
    return [c for c in CANDIDATE_PREDICTORS
            if c in df and df[c].nunique(dropna=True) > 1]


class Cohort(NamedTuple):
    """Everything a script may look at, assembled once by the same code path."""
    raw: pd.DataFrame          # as loaded, uncleaned -- for Q8's before/after report
    full_train: pd.DataFrame   # all disease groups, cleaned, training partition
    chf_train: pd.DataFrame    # CHF only, cleaned, training partition
    n_voided: int              # implausible cells set to missing
    n_test: int                # held out, never returned


def analysis_frames(test_frac: float = 0.30, seed: int = 20260901) -> Cohort:
    """
    The single entry point for every analysis script.

    Order is load -> bound -> split, and it is fixed here rather than left to
    each script, because the order is itself a finding: 02_profile.py showed a
    functional-form verdict flipping on whether the bounds had been applied yet.
    A rule that lives in one script is not a rule.

    The test partition is constructed and immediately discarded from the return
    value, so a script cannot read it by accident.
    """
    raw = load_support2()
    cleaned, n_voided = apply_plausibility_bounds(raw)
    cleaned = derive_dnr_features(cleaned)
    split = make_split(cleaned, test_frac=test_frac, seed=seed)
    full_train = cleaned[split == "train"].copy()
    return Cohort(raw=raw, full_train=full_train,
                  chf_train=chf_cohort(full_train),
                  n_voided=n_voided, n_test=int((split == "test").sum()))


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
