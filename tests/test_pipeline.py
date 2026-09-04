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
    CHF_LABEL,
    SEPSIS_LABEL,
    analysis_frames,
    apply_plausibility_bounds,
    audit_columns,
    chf_cohort,
    disease_cohort,
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


class TestDNRSplit:
    """DNR is the strongest predictor in the cohort but conflates two things: a
    patient's advance directive and a clinician's decision to limit treatment.
    Keeping the second would encode a self-fulfilling prophecy -- the model
    predicts death because care was withdrawn. These pin the split."""

    def test_raw_dnr_is_not_a_candidate_predictor(self):
        assert "dnr" not in CANDIDATE_PREDICTORS

    def test_in_admission_level_is_excluded_with_a_reason(self):
        from support2 import LEAKAGE_COLUMNS

        assert "dnr_in_admission" in LEAKAGE_COLUMNS
        assert "self-fulfilling" in LEAKAGE_COLUMNS["dnr_in_admission"]

    def test_preexisting_directive_is_kept(self, cohort):
        assert "dnr_preexisting" in CANDIDATE_PREDICTORS
        assert "dnr_preexisting" in cohort.chf_train.columns

    def test_derived_levels_are_mutually_exclusive(self, cohort):
        chf = cohort.chf_train
        both = (chf.dnr_preexisting == 1) & (chf.dnr_in_admission == 1)
        assert not both.any()

    def test_derived_levels_reconstruct_the_original(self, cohort):
        from support2 import DNR_IN_ADMISSION_LABEL, DNR_PREEXISTING_LABEL

        chf = cohort.chf_train
        known = chf.dnr.notna()
        assert (chf.loc[known, "dnr_preexisting"] == 1).sum() == (
            chf.dnr == DNR_PREEXISTING_LABEL).sum()
        assert (chf.loc[known, "dnr_in_admission"] == 1).sum() == (
            chf.dnr == DNR_IN_ADMISSION_LABEL).sum()

    def test_missing_dnr_propagates_rather_than_becoming_zero(self, cohort):
        """A patient with unknown DNR status is not a patient without a DNR."""
        chf = cohort.chf_train
        unknown = chf.dnr.isna()
        if unknown.any():
            assert chf.loc[unknown, "dnr_preexisting"].isna().all()
            assert chf.loc[unknown, "dnr_in_admission"].isna().all()

    def test_the_split_is_justified_by_the_data(self, cohort):
        """The whole argument: a pre-existing directive behaves like no DNR,
        an in-admission order does not. If that ever stops being true, the
        split is arbitrary and the write-up is wrong."""
        from lifelines.statistics import logrank_test
        from support2 import (DNR_IN_ADMISSION_LABEL, DNR_PREEXISTING_LABEL,
                              OUTCOME_TIME)

        chf = cohort.chf_train
        none = chf[chf.dnr == "no dnr"]
        pre = chf[chf.dnr == DNR_PREEXISTING_LABEL]
        ina = chf[chf.dnr == DNR_IN_ADMISSION_LABEL]

        p_pre = logrank_test(pre[OUTCOME_TIME], none[OUTCOME_TIME],
                             pre[OUTCOME_EVENT], none[OUTCOME_EVENT]).p_value
        p_ina = logrank_test(ina[OUTCOME_TIME], none[OUTCOME_TIME],
                             ina[OUTCOME_EVENT], none[OUTCOME_EVENT]).p_value
        assert p_pre > 0.10, "pre-existing directive should track the no-DNR curve"
        assert p_ina < 0.001, "in-admission order should separate sharply"


class TestCohortParameterisation:
    """The replication arm (12_replication.py) runs the same pipeline against a
    second disease group. That is only sound if selecting a cohort cannot move
    a patient across the train/test line -- the split has to be defined on the
    whole dataset and filtered afterwards, never recomputed per cohort."""

    def test_split_does_not_depend_on_the_cohort_requested(self, raw):
        """The nastiest failure mode this guards: if the split were computed
        after subsetting, the same patient could land in train for one cohort
        and test for another, and the sepsis holdout would be contaminated by
        the CHF arm having already been read."""
        chf = analysis_frames(group=CHF_LABEL)
        sep = analysis_frames(group=SEPSIS_LABEL)
        pd.testing.assert_index_equal(chf.full_train.index.sort_values(),
                                      sep.full_train.index.sort_values())

    def test_cohorts_are_disjoint(self, raw):
        chf = set(disease_cohort(raw, CHF_LABEL).index)
        sep = set(disease_cohort(raw, SEPSIS_LABEL).index)
        assert chf and sep
        assert not (chf & sep)

    def test_sepsis_frames_never_return_the_test_partition(self, raw):
        s = make_split(raw)
        test_idx = set(raw.index[s == "test"])
        assert not (set(analysis_frames(group=SEPSIS_LABEL).chf_train.index)
                    & test_idx)

    def test_unknown_group_is_rejected(self, raw):
        """A typo in a group name must fail loudly rather than silently
        returning an empty frame that every downstream metric would happily
        compute nonsense from."""
        with pytest.raises(ValueError):
            disease_cohort(raw, "ARF/MOSF w/sepsis")   # wrong capitalisation

    def test_chf_cohort_still_matches_the_parameterised_call(self, raw):
        pd.testing.assert_frame_equal(chf_cohort(raw),
                                      disease_cohort(raw, CHF_LABEL))

    def test_sepsis_arm_is_the_larger_one(self, raw):
        """The whole reason for the replication: the CHF conclusions rest on
        6.4 events per variable. If this ever stops holding, the framing in
        12_replication.py is wrong."""
        chf = analysis_frames(group=CHF_LABEL).chf_train
        sep = analysis_frames(group=SEPSIS_LABEL).chf_train
        assert len(sep) > 2 * len(chf)
