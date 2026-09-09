"""Tests for cpcv_validator_fp.py -- same coverage as tests/test_cpcv_
validator.py, against the FP rewrite. Same "synthetic first, empty/
degenerate input before happy path" workflow as tests_fp/test_walk_
forward_validator_fp.py. Pure combinatorics (generate_cpcv_splits,
_contiguous_runs) tested directly with no DataFrame; classify_cpcv_
overall() tested directly against synthetic verdict lists; cpcv_validate()
tested end-to-end on small synthetic panels last."""
import numpy as np
import pandas as pd
import pytest

import cpcv_validator_fp as cpcv
from walk_forward_validator_fp import stat_vs_zero


# --- format_cpcv_print_line() / format_cpcv_registry_note(): pure formatting ---

def _result(overall_verdict="ROBUST", pct=0.8, n_paths=12, n_skipped=3, n_groups=6, n_test_groups=2):
    paths = [object()] * n_paths  # only len() is used by the formatters
    return cpcv.CPCVResult(n_groups=n_groups, n_test_groups=n_test_groups, paths=paths,
                            n_splits_skipped=n_skipped, pct_paths_robust_or_moderate=pct,
                            overall_verdict=overall_verdict)


def test_format_cpcv_print_line_includes_verdict_and_pct():
    line = cpcv.format_cpcv_print_line(_result(overall_verdict="ROBUST", pct=0.8, n_paths=12, n_skipped=3))
    assert "ROBUST" in line
    assert "80%" in line
    assert "12 path(s) evaluated" in line
    assert "3 skipped" in line


def test_format_cpcv_print_line_handles_none_pct():
    line = cpcv.format_cpcv_print_line(_result(pct=None))
    assert "n/a" in line


def test_format_cpcv_registry_note_includes_group_and_path_counts():
    note = cpcv.format_cpcv_registry_note(_result(overall_verdict="WEAK", pct=0.3, n_paths=10,
                                                    n_skipped=5, n_groups=6, n_test_groups=2))
    assert note.startswith(" ")  # ready to append directly onto an existing note string
    assert "WEAK" in note
    assert "30%" in note
    assert "C(6,2)=15" in note  # 10 evaluated + 5 skipped = 15 total combinatorial splits
    assert "10 evaluated path(s)" in note


def test_format_cpcv_registry_note_handles_none_pct():
    note = cpcv.format_cpcv_registry_note(_result(pct=None))
    assert "n/a" in note


# --- generate_cpcv_splits() / _contiguous_runs(): pure combinatorics -------

def test_generate_cpcv_splits_count_matches_binomial_coefficient():
    splits = list(cpcv.generate_cpcv_splits(6, 2))
    assert len(splits) == 15  # C(6,2)


def test_generate_cpcv_splits_train_test_partition_every_group_exactly_once():
    for train, test in cpcv.generate_cpcv_splits(6, 2):
        assert train | test == frozenset(range(6))
        assert train & test == frozenset()
        assert len(test) == 2


def test_generate_cpcv_splits_n_test_groups_one_yields_n_groups_splits():
    splits = list(cpcv.generate_cpcv_splits(5, 1))
    assert len(splits) == 5


def test_contiguous_runs_splits_on_gaps():
    assert cpcv._contiguous_runs([1, 3, 4, 5]) == [[1], [3, 4, 5]]


def test_contiguous_runs_all_consecutive_is_one_run():
    assert cpcv._contiguous_runs([2, 3, 4]) == [[2, 3, 4]]


def test_contiguous_runs_empty_returns_empty():
    assert cpcv._contiguous_runs([]) == []


# --- assign_groups(): empty/degenerate input first -------------------------

def test_assign_groups_empty_df_returns_empty_series():
    result = cpcv.assign_groups(pd.DataFrame(), "d", 6)
    assert result.empty


def test_assign_groups_fewer_rows_than_n_groups_degrades_gracefully():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"d": dates})
    result = cpcv.assign_groups(df, "d", 6)
    assert result.nunique() <= 3


def test_assign_groups_produces_roughly_equal_contiguous_groups():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"d": dates})
    result = cpcv.assign_groups(df, "d", 6)
    counts = result.value_counts()
    assert result.nunique() == 6
    assert counts.max() - counts.min() <= 1  # 120/6 = 20 exactly, near-perfect balance


def test_assign_groups_keeps_tied_dates_in_same_group():
    # REGRESSION GUARD, mirroring walk_forward_validator.chronological_split()'s
    # own tied-date-boundary bug: a cohort sharing one date must not be
    # split across two groups by the naive row-count cut.
    dates = list(pd.date_range("2024-01-01", periods=10, freq="D")) + [pd.Timestamp("2024-01-10")] * 20
    df = pd.DataFrame({"d": dates})
    result = cpcv.assign_groups(df, "d", 3)
    tied_group_ids = result[df["d"] == pd.Timestamp("2024-01-10")].unique()
    assert len(tied_group_ids) == 1


# --- classify_cpcv_overall(): pure decision logic ---------------------------

def test_classify_cpcv_overall_all_robust_is_robust():
    verdict, reasoning, pct = cpcv.classify_cpcv_overall(["ROBUST"] * 10)
    assert verdict == "ROBUST"
    assert pct == 1.0


def test_classify_cpcv_overall_all_overfitted_is_overfitted():
    verdict, reasoning, pct = cpcv.classify_cpcv_overall(["OVERFITTED"] * 10)
    assert verdict == "OVERFITTED"
    assert pct == 0.0


def test_classify_cpcv_overall_mixed_moderate_share_is_moderate():
    verdict, reasoning, pct = cpcv.classify_cpcv_overall(["ROBUST"] * 5 + ["WEAK"] * 5)
    assert verdict == "MODERATE"
    assert pct == pytest.approx(0.5)


def test_classify_cpcv_overall_excludes_insufficient_labels_from_denominator():
    verdict, reasoning, pct = cpcv.classify_cpcv_overall(
        ["ROBUST"] * 3 + ["INSUFFICIENT_INSAMPLE_EDGE"] * 7)
    assert pct == 1.0  # denominator is 3, not 10


def test_classify_cpcv_overall_all_insufficient_returns_insufficient_data():
    verdict, reasoning, pct = cpcv.classify_cpcv_overall(["INSUFFICIENT_DATA"] * 5)
    assert verdict == "INSUFFICIENT_DATA"
    assert pct is None


def test_classify_cpcv_overall_empty_list_returns_insufficient_data():
    verdict, reasoning, pct = cpcv.classify_cpcv_overall([])
    assert verdict == "INSUFFICIENT_DATA"


# --- build_split_frames(): purge/embargo across possibly-multiple runs -----

def test_build_split_frames_no_purge_embargo_is_pure_group_slice():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"d": dates, "v": range(60)})
    groups = cpcv.assign_groups(df, "d", 6)
    train, test, n_purged, n_embargoed = cpcv.build_split_frames(
        df, "d", groups, frozenset({0, 1, 2, 3}), frozenset({4, 5}))
    assert n_purged == 0 and n_embargoed == 0
    assert len(train) + len(test) == 60


def test_build_split_frames_purges_train_rows_near_test_boundary():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"d": dates, "v": range(60)})
    groups = cpcv.assign_groups(df, "d", 6)
    train, test, n_purged, n_embargoed = cpcv.build_split_frames(
        df, "d", groups, frozenset({0, 1, 2, 3}), frozenset({4, 5}), purge_days=15)
    assert n_purged > 0
    assert len(train) < groups.isin({0, 1, 2, 3}).sum()


def test_build_split_frames_non_contiguous_test_groups_has_two_boundaries():
    # test_groups = {1, 4} out of 6 -- two separate contiguous runs, each
    # with its own purge boundary against the surrounding train groups.
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"d": dates, "v": range(120)})
    groups = cpcv.assign_groups(df, "d", 6)
    train, test, n_purged, n_embargoed = cpcv.build_split_frames(
        df, "d", groups, frozenset({0, 2, 3, 5}), frozenset({1, 4}), purge_days=10, embargo_days=5)
    assert len(train) + len(test) < 120  # some rows dropped by purge/embargo
    assert n_purged > 0 and n_embargoed > 0


def test_build_split_frames_empty_test_returns_unmodified():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"d": dates, "v": range(10)})
    groups = pd.Series([0] * 10, index=df.index)
    train, test, n_purged, n_embargoed = cpcv.build_split_frames(
        df, "d", groups, frozenset({0}), frozenset({1}), purge_days=5)
    assert test.empty
    assert n_purged == 0 and n_embargoed == 0


# --- cpcv_validate(): end-to-end on synthetic panels ------------------------

def test_cpcv_validate_empty_df_returns_insufficient_data():
    result = cpcv.cpcv_validate(pd.DataFrame(), "d", lambda d: stat_vs_zero(d["ret"]))
    assert result.overall_verdict == "INSUFFICIENT_DATA"
    assert result.paths == []


def test_cpcv_validate_invalid_n_test_groups_returns_insufficient_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"d": dates, "ret": range(60)})
    result = cpcv.cpcv_validate(df, "d", lambda d: stat_vs_zero(d["ret"]), n_groups=4, n_test_groups=4)
    assert result.overall_verdict == "INSUFFICIENT_DATA"


def test_cpcv_validate_too_few_rows_for_n_groups_returns_insufficient_data():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"d": dates, "ret": [1.0, 2.0, 3.0]})
    result = cpcv.cpcv_validate(df, "d", lambda d: stat_vs_zero(d["ret"]), n_groups=6, n_test_groups=2)
    assert result.overall_verdict == "INSUFFICIENT_DATA"


def test_cpcv_validate_detects_genuine_stable_signal_as_robust():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=240, freq="W")
    df = pd.DataFrame({"d": dates, "ret": rng.normal(2.0, 3.0, 240)})
    result = cpcv.cpcv_validate(df, "d", lambda d: stat_vs_zero(d["ret"]), n_groups=6, n_test_groups=2)
    assert result.overall_verdict == "ROBUST"
    assert result.pct_paths_robust_or_moderate == pytest.approx(1.0)
    assert len(result.paths) == 15


def test_cpcv_validate_detects_pure_noise_as_not_robust():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=240, freq="W")
    df = pd.DataFrame({"d": dates, "ret": rng.normal(0.0, 3.0, 240)})
    result = cpcv.cpcv_validate(df, "d", lambda d: stat_vs_zero(d["ret"]), n_groups=6, n_test_groups=2)
    assert result.overall_verdict in ("WEAK", "OVERFITTED")
    assert result.pct_paths_robust_or_moderate < 0.5


def test_cpcv_validate_purge_embargo_skips_starved_paths_not_crashes():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2020-01-01", periods=240, freq="W")
    df = pd.DataFrame({"d": dates, "ret": rng.normal(2.0, 3.0, 240)})
    result = cpcv.cpcv_validate(df, "d", lambda d: stat_vs_zero(d["ret"]), n_groups=6, n_test_groups=2,
                                 purge_days=14, embargo_days=7)
    assert result.n_splits_skipped >= 0
    assert len(result.paths) + result.n_splits_skipped == 15


def test_cpcv_validate_min_n_per_split_can_skip_all_paths():
    dates = pd.date_range("2020-01-01", periods=30, freq="W")
    df = pd.DataFrame({"d": dates, "ret": [1.0] * 30})
    result = cpcv.cpcv_validate(df, "d", lambda d: stat_vs_zero(d["ret"]), n_groups=6, n_test_groups=2,
                                 min_n_per_split=100)
    assert result.overall_verdict == "INSUFFICIENT_DATA"
    assert result.n_splits_skipped == 15
