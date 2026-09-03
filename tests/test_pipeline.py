"""
Tests for the data pipeline: the split that underwrites the reproducibility
claim, the cleaning that the write-up asserts, and column governance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from support2 import (
    CANDIDATE_PREDICTORS,
    OUTCOME_EVENT,
    PLAUSIBLE_BOUNDS,
    analysis_frames,
    apply_plausibility_bounds,
    audit_columns,
    chf_cohort,
    load_support2,
    make_split,
    model_predictors,
)


@pytest.fixture(scope="module")
def raw():
    return load_support2()


@pytest.fixture(scope="module")
def cohort():
    return analysis_frames()


class TestSplit:
    """The README leads with the holdout. If the split is not deterministic,
    disjoint and stratified, that claim is false."""

    def test_deterministic_across_calls(self, raw):
        a, b = make_split(raw), make_split(raw)
        pd.testing.assert_series_equal(a, b)

    def test_deterministic_across_row_order(self, raw):
        """Sorting indices inside make_split means a reordered frame must give
        the same assignment -- otherwise the split silently depends on how the
        CSV happened to be read."""
        shuffled = raw.sample(frac=1.0, random_state=7)
        pd.testing.assert_series_equal(
            make_split(raw).sort_index(),
            make_split(shuffled).sort_index(),
        )

    def test_partitions_are_disjoint_and_complete(self, raw):
        s = make_split(raw)
        assert set(s.unique()) <= {"train", "test"}
        assert len(s) == len(raw)
        assert not s.isna().any()

    def test_stratified_on_the_event(self, raw):
        s = make_split(raw)
        tr = raw.loc[s == "train", OUTCOME_EVENT].mean()
        te = raw.loc[s == "test", OUTCOME_EVENT].mean()
        assert tr == pytest.approx(te, abs=0.01)

    def test_test_fraction_is_honoured(self, raw):
        s = make_split(raw, test_frac=0.30)
        assert (s == "test").mean() == pytest.approx(0.30, abs=0.01)

    def test_different_seed_gives_different_split(self, raw):
        a = make_split(raw, seed=1)
        b = make_split(raw, seed=2)
        assert not a.equals(b)

    def test_analysis_frames_never_returns_the_test_partition(self, raw, cohort):
        s = make_split(raw)
        test_idx = set(raw.index[s == "test"])
        assert not (set(cohort.full_train.index) & test_idx)
        assert not (set(cohort.chf_train.index) & test_idx)


class TestPlausibility:
    """The write-up says impossible values are voided everywhere. If cleaning
    is skipped on any path, the data dictionary publishes values the
    data-quality section calls impossible -- which is the bug this replaced."""

    def test_no_value_survives_outside_its_bounds(self, cohort):
        for frame in (cohort.full_train, cohort.chf_train):
            for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
                if col not in frame:
                    continue
                s = frame[col].dropna()
                assert s.empty or (s.between(lo, hi).all()), f"{col} out of bounds"

    def test_cleaning_is_cell_level_not_row_level(self, raw):
        cleaned, n = apply_plausibility_bounds(raw)
        assert len(cleaned) == len(raw), "rows must not be dropped"
        assert n > 0
        assert cleaned.isna().sum().sum() == raw.isna().sum().sum() + n

    def test_reported_count_matches_cells_changed(self, raw):
        cleaned, n = apply_plausibility_bounds(raw)
        changed = int((raw.notna() & cleaned.isna()).sum().sum())
        assert changed == n

    def test_is_idempotent(self, raw):
        once, n1 = apply_plausibility_bounds(raw)
        twice, n2 = apply_plausibility_bounds(once)
        assert n2 == 0
        pd.testing.assert_frame_equal(once, twice)

    def test_every_bounded_column_has_a_unit(self):
        from support2 import UNITS

        missing = [c for c in PLAUSIBLE_BOUNDS if c not in UNITS]
        assert not missing, f"bounds without units are unreviewable: {missing}"


class TestColumnGovernance:
    """A new column entering the feature matrix unnoticed is the failure this
    guards. Leakage does not raise -- it improves your metrics."""

    def test_nothing_is_left_unreviewed(self, raw):
        audit = audit_columns(raw)
        unreviewed = audit.loc[audit.role == "unreviewed", "column"].tolist()
        assert not unreviewed, f"unclassified columns: {unreviewed}"

    def test_known_leaks_are_excluded(self, raw):
        audit = audit_columns(raw).set_index("column")
        for col in ("d.time", "hospdead", "sfdm2", "surv2m", "prg2m", "aps", "slos"):
            assert audit.loc[col, "role"] in ("excluded", "outcome"), col

    def test_no_predictor_is_also_an_outcome_or_leak(self, raw):
        from support2 import DERIVED_DUPLICATES, LEAKAGE_COLUMNS

        overlap = set(CANDIDATE_PREDICTORS) & (set(LEAKAGE_COLUMNS) | set(DERIVED_DUPLICATES))
        assert not overlap, f"column claimed as both: {overlap}"

    def test_model_predictors_drops_cohort_constants(self, cohort):
        """dzgroup is constant inside CHF because the cohort is defined on it."""
        chf = cohort.chf_train
        assert "dzgroup" in CANDIDATE_PREDICTORS
        assert chf["dzgroup"].nunique() == 1
        assert "dzgroup" not in model_predictors(chf)

    def test_adlsc_is_identical_to_adls_not_merely_correlated(self, cohort):
        """The documented reason adlsc is excluded. If this ever stops being
        exactly true, the stated justification is wrong."""
        both = cohort.chf_train[["adls", "adlsc"]].dropna()
        assert len(both) > 100
        assert (both.adls - both.adlsc).abs().max() == pytest.approx(0.0, abs=1e-9)


class TestSourceFileShape:
    """The shipped CSV has 47 header fields and 48 data fields: an unnamed
    leading patient id. Pandas resolves that by promoting column 0 to the index,
    which is correct but was undeclared -- and make_split() partitions on that
    index, so the train/test assignment was keyed on an identifier nobody had
    named. These pin the behaviour."""

    def test_header_is_one_field_short_of_the_data(self):
        from support2 import CSV_PATH

        with open(CSV_PATH, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split(",")
            row = fh.readline().rstrip("\n").split(",")
        assert len(row) == len(header) + 1, (
            "the source file no longer has an unnamed id column; index_col=0 "
            "would now capture a real variable and shift every column"
        )

    def test_index_is_the_named_patient_id(self, raw):
        from support2 import ID_COLUMN

        assert raw.index.name == ID_COLUMN
        assert raw.index.is_unique
        assert raw.index.min() == 1
        assert raw.index.max() == len(raw)

    def test_columns_are_not_shifted(self, raw):
        """A one-column shift is silent: the frame loads, and `age` fills with
        patient ids. These are the cheap alignment checks that catch it."""
        assert raw["age"].between(0, 120).all()
        assert set(raw["sex"].dropna().unique()) == {"male", "female"}
        assert set(raw["death"].dropna().unique()) <= {0, 1}
        assert raw["d.time"].min() > 0

    def test_validate_rejects_a_shifted_header(self, raw):
        from support2 import _validate

        shifted = raw.reset_index()
        shifted.columns = list(raw.columns) + ["extra"]
        with pytest.raises(ValueError, match="age"):
            _validate(shifted[list(raw.columns)])

    def test_validate_rejects_wrong_column_count(self, raw):
        from support2 import _validate

        with pytest.raises(ValueError, match="expected 47 columns"):
            _validate(raw.iloc[:, :40])

    def test_shipped_column_order_matches_the_file(self, raw):
        """The download path reindexes to this order so both paths agree."""
        from support2 import SHIPPED_COLUMN_ORDER

        assert list(raw.columns) == SHIPPED_COLUMN_ORDER
