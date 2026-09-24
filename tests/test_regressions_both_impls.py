"""Regression tests for six confirmed bugs, run against BOTH the original
and the FP implementation of each module so the two can't drift apart on
these behaviors."""
import numpy as np
import pandas as pd
import pytest

import cpcv_validator
import cpcv_validator_fp
import dedup_store
import dedup_store_fp
import walk_forward_validator
import walk_forward_validator_fp

WFV = pytest.mark.parametrize("wfv", [walk_forward_validator, walk_forward_validator_fp], ids=["orig", "fp"])
CPCV = pytest.mark.parametrize("cpcv", [cpcv_validator, cpcv_validator_fp], ids=["orig", "fp"])
DEDUP = pytest.mark.parametrize("dd", [dedup_store, dedup_store_fp], ids=["orig", "fp"])


def _daily(n, start="2020-01-01", seed=0, mean=0.5):
    return pd.DataFrame({"date": pd.date_range(start, periods=n, freq="D"),
                         "r": np.random.default_rng(seed).normal(mean, 1.0, n)})


# --- Bug 1: CPCV purge must not wipe train groups AFTER a held-out block ----

@CPCV
def test_cpcv_purge_keeps_train_groups_after_a_middle_test_block(cpcv):
    df = _daily(120)
    groups = cpcv.assign_groups(df, "date", 6)  # 20 rows per group
    train, test, n_purged, n_embargoed = cpcv.build_split_frames(
        df, "date", groups, frozenset({0, 2, 3, 4, 5}), frozenset({1}), purge_days=5)
    # 5 rows purged before the block (their forward window reaches it) and
    # 5 rows purged right after it (inside the last test row's window).
    assert n_purged == 10
    assert n_embargoed == 0
    assert len(train) == 90
    assert len(test) == 20
    test_start, test_end = test["date"].min(), test["date"].max()
    after = train.loc[train["date"] > test_end, "date"]
    assert after.min() == test_end + pd.Timedelta(days=6)
    assert (groups.loc[train.index] == 5).sum() == 20  # far-later group untouched
    assert train.loc[train["date"] < test_start, "date"].max() == test_start - pd.Timedelta(days=6)


@CPCV
def test_cpcv_non_contiguous_blocks_purge_both_sides_of_each(cpcv):
    df = _daily(120)
    groups = cpcv.assign_groups(df, "date", 6)
    train, test, n_purged, _ = cpcv.build_split_frames(
        df, "date", groups, frozenset({0, 2, 3, 5}), frozenset({1, 4}), purge_days=3)
    # block {1}: 3 before + 3 after; block {4}: 3 before + 3 after.
    assert n_purged == 12
    assert len(train) == 80 - 12
    for g in (0, 2, 3, 5):
        assert (groups.loc[train.index] == g).sum() > 0


@CPCV
def test_cpcv_validate_with_purge_evaluates_every_split(cpcv):
    df = _daily(300, seed=3)
    result = cpcv.cpcv_validate(df, "date", lambda d: walk_forward_validator.stat_vs_zero(d["r"]),
                                n_groups=6, n_test_groups=2, purge_days=5, embargo_days=2)
    assert result.n_splits_skipped == 0
    assert len(result.paths) == 15


# --- Bug 2: significance_t must drive the verdict, not just the message ----

@WFV
def test_walk_forward_significance_t_changes_the_verdict(wfv):
    df = _daily(200, seed=1, mean=0.4)
    stat_fn = lambda d: wfv.stat_vs_zero(d["r"])  # noqa: E731 -- stat_fn uses its own default t=2.0
    assert wfv.walk_forward_validate(df, "date", stat_fn).train.significant
    strict = wfv.walk_forward_validate(df, "date", stat_fn, significance_t=50)
    assert strict.verdict == "INSUFFICIENT_INSAMPLE_EDGE"


@WFV
def test_classify_overfitting_recomputes_significance_from_t(wfv):
    train = wfv.StatResult(mean=1.0, t_stat=2.5, n=50, significant=True)
    test = wfv.StatResult(mean=0.9, t_stat=2.4, n=50, significant=True)
    assert wfv.classify_overfitting(train, test, significance_t=3.0)[0] == "INSUFFICIENT_INSAMPLE_EDGE"
    # And the other direction: flags say "not significant" but |t| clears a looser bar.
    train = wfv.StatResult(mean=1.0, t_stat=1.8, n=50, significant=False)
    test = wfv.StatResult(mean=0.9, t_stat=1.7, n=50, significant=False)
    assert wfv.classify_overfitting(train, test, significance_t=1.5)[0] == "ROBUST"


@CPCV
def test_cpcv_significance_t_changes_the_verdict(cpcv):
    df = _daily(300, seed=3)
    stat_fn = lambda d: walk_forward_validator.stat_vs_zero(d["r"])  # noqa: E731
    result = cpcv.cpcv_validate(df, "date", stat_fn, significance_t=50)
    assert result.overall_verdict == "INSUFFICIENT_DATA"
    assert all(p.verdict == "INSUFFICIENT_INSAMPLE_EDGE" for p in result.paths)


# --- Bug 3: an uncomputable test stat is INSUFFICIENT_DATA, not WEAK -------

@WFV
def test_nan_test_stat_is_insufficient_data(wfv):
    train = wfv.StatResult(mean=1.0, t_stat=5.0, n=50, significant=True)
    test = wfv.StatResult(mean=float("nan"), t_stat=float("nan"), n=1, significant=False)
    verdict, _, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "INSUFFICIENT_DATA"
    assert retention is None and flipped is False


@WFV
def test_infinite_t_is_still_a_real_stat(wfv):
    """Zero variance gives t=+-inf -- degenerate but computable, not missing."""
    train = wfv.StatResult(mean=5.0, t_stat=float("inf"), n=70, significant=True)
    test = wfv.StatResult(mean=-5.0, t_stat=float("-inf"), n=30, significant=True)
    assert wfv.classify_overfitting(train, test)[0] == "OVERFITTED"


@WFV
def test_group_diff_missing_group_in_test_is_insufficient_data(wfv):
    df = _daily(100, seed=2)
    df["flag"] = np.r_[np.tile([True, False], 35), [True] * 30]  # test window has only one group
    df["r"] = df["r"] + np.where(df["flag"], 2.0, 0.0)
    result = wfv.walk_forward_validate(df, "date", lambda d: wfv.stat_group_diff(d["r"], d["flag"]))
    assert result.verdict == "INSUFFICIENT_DATA"


@CPCV
def test_cpcv_excludes_uncomputable_paths_from_the_denominator(cpcv):
    df = _daily(120, seed=4)
    df["flag"] = np.tile([True, False], 60)
    df.loc[df.index[20:40], "flag"] = True  # group 1 has no False rows
    df["r"] = df["r"] + np.where(df["flag"], 3.0, 0.0)
    result = cpcv.cpcv_validate(df, "date", lambda d: walk_forward_validator.stat_group_diff(d["r"], d["flag"]),
                                n_groups=6, n_test_groups=1)
    by_group = {next(iter(p.test_groups)): p.verdict for p in result.paths}
    assert by_group[1] == "INSUFFICIENT_DATA"
    assert result.overall_verdict == "ROBUST"
    assert result.pct_paths_robust_or_moderate == 1.0


# --- Bug 4: a malformed store is preserved, and writes are atomic ----------

@DEDUP
def test_malformed_store_is_moved_aside_not_overwritten(dd, tmp_path, capsys):
    store = tmp_path / "store.csv"
    original = "id,v\n1,2\n3,4\n5,6,7,8\n"
    store.write_text(original)
    result = dd.append_dedup(pd.DataFrame({"id": ["9"], "v": [9]}), store, ["id"])
    assert list(result["id"]) == ["9"]
    backups = list(tmp_path.glob("store.csv.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    assert backups[0].name in capsys.readouterr().out


@DEDUP
def test_empty_store_is_rebuilt_without_a_backup(dd, tmp_path):
    store = tmp_path / "store.csv"
    store.write_text("")
    dd.append_dedup(pd.DataFrame({"id": ["1"]}), store, ["id"], verbose=False)
    assert not list(tmp_path.glob("*.corrupt-*"))


@DEDUP
def test_failed_write_leaves_existing_store_intact(dd, tmp_path, monkeypatch):
    store = tmp_path / "store.csv"
    dd.append_dedup(pd.DataFrame({"id": ["1", "2"], "v": [1, 2]}), store, ["id"], verbose=False)
    before = store.read_text()

    def boom(self, *args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        dd.append_dedup(pd.DataFrame({"id": ["3"], "v": [3]}), store, ["id"], verbose=False)
    monkeypatch.undo()
    assert store.read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store.csv"]  # no temp file left behind


# --- Bug 5: keys with missing values still dedupe; missing key cols raise --

@DEDUP
def test_str_key_dedupes_when_store_key_column_has_a_missing_value(dd, tmp_path):
    store = tmp_path / "store.csv"
    dd.append_dedup(pd.DataFrame({"id": ["1", "2", None], "v": [1, 2, 3]}), store, ["id"], verbose=False)
    result = dd.append_dedup(pd.DataFrame({"id": ["1"], "v": [9]}), store, ["id"], verbose=False)
    assert len(result) == 3
    assert result.loc[result["id"] == "1", "v"].tolist() == [9]


@DEDUP
def test_float_key_from_new_rows_matches_stored_str_key(dd, tmp_path):
    store = tmp_path / "store.csv"
    dd.append_dedup(pd.DataFrame({"id": ["1", "2"], "v": [1, 2]}), store, ["id"], verbose=False)
    result = dd.append_dedup(pd.DataFrame({"id": [1.0, np.nan], "v": [9, 8]}), store, ["id"], verbose=False)
    assert sorted(result["id"].dropna()) == ["1", "2"]
    assert result.loc[result["id"] == "1", "v"].tolist() == [9]


@DEDUP
def test_leading_zero_keys_survive_a_round_trip(dd, tmp_path):
    store = tmp_path / "store.csv"
    dd.append_dedup(pd.DataFrame({"id": ["007"]}), store, ["id"], verbose=False)
    result = dd.append_dedup(pd.DataFrame({"id": ["007"]}), store, ["id"], verbose=False)
    assert result["id"].tolist() == ["007"]


@DEDUP
def test_missing_dedup_col_raises_and_leaves_store_alone(dd, tmp_path):
    store = tmp_path / "store.csv"
    dd.append_dedup(pd.DataFrame({"id": ["1"], "v": [1]}), store, ["id"], verbose=False)
    before = store.read_text()
    with pytest.raises(ValueError, match="symbol"):
        dd.append_dedup(pd.DataFrame({"id": ["2"], "v": [2]}), store, ["id", "symbol"], verbose=False)
    assert store.read_text() == before


@DEDUP
def test_caller_frame_is_not_mutated(dd, tmp_path):
    rows = pd.DataFrame({"id": [1, 2]})
    dd.append_dedup(rows, tmp_path / "store.csv", ["id"], verbose=False)
    assert rows["id"].tolist() == [1, 2]
    assert rows["id"].dtype == np.int64


# --- Bug 6: purge boundary is "on or after test start" ---------------------

@WFV
def test_purge_drops_row_whose_window_ends_exactly_on_test_start(wfv):
    train = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})  # to 01-10
    test = pd.DataFrame({"date": pd.date_range("2024-01-11", periods=5, freq="D")})
    out_train, _, n_purged, _ = wfv.apply_purge_embargo(train, test, "date", purge_days=3)
    # 01-08 + 3 days = 01-11 = test start -> purged; 01-07 resolves 01-10 -> kept.
    assert n_purged == 3
    assert out_train["date"].max() == pd.Timestamp("2024-01-07")
