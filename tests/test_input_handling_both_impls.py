"""Regression tests for date handling, CPCV row order, cluster_groups
alignment, the CPCV low-cluster caveat and post-block embargo, and input
validation -- each run against BOTH the original and the FP implementation."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import cpcv_validator
import cpcv_validator_fp
import fama_macbeth
import fama_macbeth_fp
import multiple_comparison_correction
import multiple_comparison_correction_fp
import walk_forward_validator
import walk_forward_validator_fp

PAIRS = pytest.mark.parametrize("wfv,cpcv", [(walk_forward_validator, cpcv_validator),
                                             (walk_forward_validator_fp, cpcv_validator_fp)], ids=["orig", "fp"])
WFV = pytest.mark.parametrize("wfv", [walk_forward_validator, walk_forward_validator_fp], ids=["orig", "fp"])
FM = pytest.mark.parametrize("fm", [fama_macbeth, fama_macbeth_fp], ids=["orig", "fp"])
MCC = pytest.mark.parametrize("mcc", [multiple_comparison_correction, multiple_comparison_correction_fp],
                              ids=["orig", "fp"])


# --- dates: typed sort key, no lexicographic string sort, no NaT ------------

@PAIRS
def test_day_first_strings_are_rejected_not_sorted_as_text(wfv, cpcv):
    df = pd.DataFrame({"date": ["01/02/2021", "15/01/2020", "03/03/2020", "20/12/2020"], "r": [1, 2, 3, 4]})
    with pytest.raises(TypeError, match="ISO-8601"):
        wfv.chronological_split(df, "date", 0.5)
    with pytest.raises(TypeError, match="ISO-8601"):
        cpcv.assign_groups(df, "date", 2)


@PAIRS
def test_iso_strings_and_date_objects_sort_chronologically(wfv, cpcv):
    iso = pd.DataFrame({"date": ["2021-02-01", "2020-01-15", "2020-03-03", "2020-12-20"], "r": [1, 2, 3, 4]})
    train, test = wfv.chronological_split(iso, "date", 0.5)
    assert train["date"].tolist() == ["2020-01-15", "2020-03-03"]  # original values preserved
    assert test["date"].tolist() == ["2020-12-20", "2021-02-01"]
    objs = iso.assign(date=[dt.date.fromisoformat(d) for d in iso["date"]])
    assert wfv.chronological_split(objs, "date", 0.5)[1]["r"].tolist() == [4, 1]
    assert cpcv.assign_groups(iso, "date", 2).sort_index().tolist() == [1, 0, 0, 1]


@PAIRS
def test_missing_dates_raise(wfv, cpcv):
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", None, "2020-01-03", "2020-01-04"]), "r": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="missing"):
        wfv.chronological_split(df, "date", 0.5)
    with pytest.raises(ValueError, match="missing"):
        cpcv.assign_groups(df, "date", 2)


@WFV
def test_numeric_period_column_still_works(wfv):
    df = pd.DataFrame({"year": [2019, 2015, 2017, 2016, 2018, 2014], "r": range(6)})
    train, test = wfv.chronological_split(df, "year", 0.5)
    assert train["year"].tolist() == [2014, 2015, 2016] and test["year"].tolist() == [2017, 2018, 2019]


# --- CPCV hands stat_fn date-sorted frames (HAC depends on row order) --------

def _overlapping_returns(n=240, seed=0):
    rng = np.random.default_rng(seed)
    r = np.convolve(rng.normal(0.15, 1, n + 19), np.ones(20) / 5, "valid")  # 20-row overlapping windows
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "r": r})


@PAIRS
def test_cpcv_result_does_not_depend_on_input_row_order(wfv, cpcv):
    df = _overlapping_returns()
    shuffled = df.sample(frac=1, random_state=1)

    def stat_fn(d):
        return wfv.stat_vs_zero(d["r"], maxlags=19)

    a = cpcv.cpcv_validate(df, "date", stat_fn, n_groups=4, n_test_groups=1)
    b = cpcv.cpcv_validate(shuffled, "date", stat_fn, n_groups=4, n_test_groups=1)
    assert [p.train.t_stat for p in a.paths] == pytest.approx([p.train.t_stat for p in b.paths])
    assert a.overall_verdict == b.overall_verdict


@PAIRS
def test_build_split_frames_returns_date_sorted_frames(wfv, cpcv):
    shuffled = _overlapping_returns(60).sample(frac=1, random_state=2)
    groups = cpcv.assign_groups(shuffled, "date", 3)
    for purge in (0, 3):
        train, test, _, _ = cpcv.build_split_frames(shuffled, "date", groups, frozenset({0, 2}), frozenset({1}),
                                                    purge_days=purge)
        assert train["date"].is_monotonic_increasing and test["date"].is_monotonic_increasing


# --- cluster_groups: Series by label, arrays by position --------------------

@WFV
def test_array_cluster_groups_align_by_position_on_a_non_range_index(wfv):
    values = pd.Series(np.random.default_rng(0).normal(1, 1, 8), index=range(40, 48))
    clusters = np.repeat([0, 1, 2, 3], 2)
    from_array = wfv.stat_vs_zero(values, cluster_groups=clusters)
    from_series = wfv.stat_vs_zero(values, cluster_groups=pd.Series(clusters, index=values.index))
    assert from_array.n_clusters == 4
    assert from_array.t_stat == pytest.approx(from_series.t_stat)
    assert wfv.stat_vs_zero(values, cluster_groups=list(clusters)).n_clusters == 4


@WFV
def test_group_diff_array_clusters_are_not_silently_misaligned(wfv):
    values = pd.Series(np.random.default_rng(1).normal(size=8), index=range(2, 10))
    groups = np.tile([True, False], 4)
    clusters = np.repeat([0, 1, 2, 3], 2)
    res = wfv.stat_group_diff(values, groups, cluster_groups=clusters)
    ref = wfv.stat_group_diff(values, pd.Series(groups, index=values.index),
                              cluster_groups=pd.Series(clusters, index=values.index))
    assert res.n_clusters == 4
    assert res.t_stat == pytest.approx(ref.t_stat)


@WFV
def test_cluster_groups_length_or_label_mismatch_raises(wfv):
    values = pd.Series([1.0, 2.0, 3.0], index=[5, 6, 7])
    with pytest.raises(ValueError, match="length"):
        wfv.stat_vs_zero(values, cluster_groups=[0, 1])
    with pytest.raises(ValueError, match="missing"):
        wfv.stat_vs_zero(values, cluster_groups=pd.Series([0, 1, 1], index=[0, 1, 2]))
    with pytest.raises(ValueError, match="missing label"):
        wfv.stat_vs_zero(values, cluster_groups=pd.Series([0, None, 1], index=[5, 6, 7]))


@PAIRS
def test_cpcv_with_array_cluster_groups_runs(wfv, cpcv):
    """Every CPCV train frame has a non-0..n-1 index -- this used to KeyError."""
    df = _overlapping_returns(120).assign(cl=lambda d: np.arange(len(d)) // 4)
    result = cpcv.cpcv_validate(df, "date", lambda d: wfv.stat_vs_zero(d["r"], cluster_groups=d["cl"].to_numpy()),
                                n_groups=4, n_test_groups=1)
    assert len(result.paths) == 4


# --- CPCV: low-cluster caveat, post-block train embargo ---------------------

@PAIRS
def test_cpcv_warns_when_paths_rely_on_few_clusters(wfv, cpcv):
    df = _overlapping_returns(120).assign(cl=lambda d: np.arange(len(d)) // 20)  # 6 clusters in total
    result = cpcv.cpcv_validate(df, "date", lambda d: wfv.stat_vs_zero(d["r"], cluster_groups=d["cl"]),
                                n_groups=3, n_test_groups=1)
    assert "CAVEAT" in result.reasoning
    assert f"fewer than {wfv.MIN_RELIABLE_CLUSTERS} distinct clusters" in result.reasoning
    plain = cpcv.cpcv_validate(df, "date", lambda d: wfv.stat_vs_zero(d["r"]), n_groups=3, n_test_groups=1)
    assert "CAVEAT" not in plain.reasoning


@PAIRS
def test_cpcv_embargoes_train_rows_after_each_block(wfv, cpcv):
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=120), "r": 1.0})
    groups = cpcv.assign_groups(df, "date", 6)
    train, test, n_purged, n_embargoed = cpcv.build_split_frames(
        df, "date", groups, frozenset({0, 2, 3, 4, 5}), frozenset({1}), purge_days=5, embargo_days=3)
    test_end = df.loc[groups == 1, "date"].max()
    after = train.loc[train["date"] > test_end, "date"]
    assert after.min() == test_end + pd.Timedelta(days=5 + 3 + 1)  # 5 purged, then 3 embargoed
    assert n_purged == 10  # 5 before + 5 after
    assert n_embargoed == 3 + 3  # 3 test rows at the block's start + 3 train rows after the purge
    assert len(test) == 17 and len(train) == 100 - 10 - 3


# --- input validation --------------------------------------------------------

@WFV
@pytest.mark.parametrize("frac", [0, 1, 1.5, -0.2])
def test_train_frac_out_of_range_raises(wfv, frac):
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=20), "r": 1.0})
    with pytest.raises(ValueError, match="train_frac"):
        wfv.walk_forward_validate(df, "date", lambda d: wfv.stat_vs_zero(d["r"]), train_frac=frac)


@MCC
@pytest.mark.parametrize("alpha", [0, 1, 5, -0.05])
def test_alpha_out_of_range_raises(mcc, alpha):
    for fn in (mcc.bonferroni_correction, mcc.holm_correction, mcc.benjamini_hochberg_fdr):
        with pytest.raises(ValueError, match="alpha"):
            fn([0.01, 0.2], alpha=alpha)


@MCC
def test_unknown_method_raises_a_clear_error(mcc):
    with pytest.raises(ValueError, match="method must be one of"):
        mcc.summarize_correction(["a"], [0.01], method="bh")


@FM
def test_fama_macbeth_alpha_out_of_range_raises(fm):
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"c": np.repeat(range(5), 20), "x": rng.normal(size=100), "y": rng.normal(size=100)})
    with pytest.raises(ValueError, match="significance_alpha"):
        fm.fama_macbeth_regression(panel, "c", "x", "y", significance_alpha=5)
