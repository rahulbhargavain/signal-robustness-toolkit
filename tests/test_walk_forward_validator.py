"""Tests for walk_forward_validator.py -- the shared overfitting-check
module. classify_overfitting() is tested directly against synthetic
StatResult pairs first (pure logic, no DataFrame/statsmodels needed),
then walk_forward_validate()/stat_vs_zero()/stat_group_diff() are tested
end-to-end on small synthetic panels."""
import numpy as np
import pandas as pd
import pytest

import walk_forward_validator as wfv
from walk_forward_validator import StatResult


def _sr(mean, t_stat, n=50, significant=None):
    if significant is None:
        significant = abs(t_stat) >= wfv.DEFAULT_SIGNIFICANCE_T
    return StatResult(mean=mean, t_stat=t_stat, n=n, significant=significant)


# --- classify_overfitting(): pure decision-tree tests ----------------------

def test_train_not_significant_is_insufficient_edge():
    train = _sr(mean=0.5, t_stat=0.8)  # below |t|=2
    test = _sr(mean=5.0, t_stat=3.0)
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "INSUFFICIENT_INSAMPLE_EDGE"
    assert retention is None
    assert not flipped


def test_sign_flip_is_always_overfitted_even_with_high_magnitude():
    train = _sr(mean=5.0, t_stat=4.0)
    test = _sr(mean=-5.0, t_stat=-4.0)  # equal magnitude, opposite sign
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "OVERFITTED"
    assert flipped


def test_full_retention_and_significant_is_robust():
    train = _sr(mean=4.0, t_stat=3.5)
    test = _sr(mean=3.6, t_stat=3.0)  # 90% retention, same sign, significant
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "ROBUST"
    assert retention == pytest.approx(0.9)
    assert not flipped


def test_partial_retention_significant_is_moderate():
    train = _sr(mean=4.0, t_stat=3.5)
    test = _sr(mean=2.0, t_stat=2.5)  # 50% retention, still significant
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "MODERATE"
    assert retention == pytest.approx(0.5)


def test_low_retention_significant_is_weak_not_overfitted():
    train = _sr(mean=4.0, t_stat=3.5)
    test = _sr(mean=0.8, t_stat=2.1)  # 20% retention but still (barely) significant
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "WEAK"
    assert retention == pytest.approx(0.2)


def test_not_significant_with_decent_retention_is_weak():
    train = _sr(mean=4.0, t_stat=3.5)
    test = _sr(mean=2.0, t_stat=1.0, significant=False)  # 50% retention but not significant
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "WEAK"


def test_not_significant_with_near_zero_retention_is_overfitted():
    train = _sr(mean=4.0, t_stat=3.5)
    test = _sr(mean=0.1, t_stat=0.1, significant=False)  # 2.5% retention, not significant
    verdict, reasoning, retention, flipped = wfv.classify_overfitting(train, test)
    assert verdict == "OVERFITTED"


def test_retention_boundary_values_are_deterministic():
    train = _sr(mean=10.0, t_stat=5.0)
    # exactly at the robust boundary (0.7) must be ROBUST, not MODERATE
    test_at_robust = _sr(mean=7.0, t_stat=3.0)
    verdict, *_ = wfv.classify_overfitting(train, test_at_robust)
    assert verdict == "ROBUST"
    # exactly at the moderate boundary (0.4) must be MODERATE, not WEAK
    test_at_moderate = _sr(mean=4.0, t_stat=2.5)
    verdict2, *_ = wfv.classify_overfitting(train, test_at_moderate)
    assert verdict2 == "MODERATE"


# --- chronological_split(): ordering and row-count behavior -----------------

def test_chronological_split_sorts_before_splitting():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01", "2024-04-01"]),
                        "value": [3, 1, 2, 4]})
    train, test = wfv.chronological_split(df, "date", train_frac=0.5)
    assert list(train["value"]) == [1, 2]
    assert list(test["value"]) == [3, 4]


def test_chronological_split_respects_train_frac():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D"), "value": range(10)})
    train, test = wfv.chronological_split(df, "date", train_frac=0.7)
    assert len(train) == 7
    assert len(test) == 3


def test_chronological_split_snaps_boundary_away_from_tied_date_run():
    """Real bug found live 2026-08-23: a naive row-count split can land
    strictly inside a run of identical dates, putting some of that date's 
    events in train and the
    rest in test -- the same reporting-cohort shock straddling the
    boundary, which undermines "test is genuinely held-out" regardless of
    stat_group_diff's separate cluster_groups correction (that only fixes
    the within-split i.i.d. assumption). The split must snap to whichever
    edge of the tied run is closer, so no date appears on both sides."""
    dates = (list(pd.date_range("2020-01-01", periods=10, freq="D"))
             + ["2020-02-01"] * 20  # tied run straddling the naive 70% index (28 of 40)
             + list(pd.date_range("2020-03-01", periods=10, freq="D")))
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": range(40)})
    train, test = wfv.chronological_split(df, "date", train_frac=0.7)
    assert not (set(train["date"]) & set(test["date"]))
    # snapped to the closer edge of the 20-row tied run (positions 10-29;
    # naive split_idx=28 is 2 away from the end, 18 away from the start)
    assert len(train) == 30
    assert len(test) == 10


def test_chronological_split_no_tied_dates_unaffected():
    """When no dates are tied at the boundary, snapping must be a no-op --
    regression guard so the fix doesn't perturb the common case."""
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20, freq="D"), "value": range(20)})
    train, test = wfv.chronological_split(df, "date", train_frac=0.7)
    assert len(train) == 14
    assert len(test) == 6


# --- stat_vs_zero(): naive and HAC paths ------------------------------------

def test_stat_vs_zero_naive_matches_manual_t_test():
    values = pd.Series([1.0, 2.0, 3.0, -1.0, 2.5, 1.5, 0.5, 3.0])
    result = wfv.stat_vs_zero(values)
    se = values.std(ddof=1) / (len(values) ** 0.5)
    expected_t = values.mean() / se
    assert result.mean == pytest.approx(values.mean())
    assert result.t_stat == pytest.approx(expected_t, rel=1e-3)
    assert result.n == 8


def test_stat_vs_zero_too_few_observations_is_not_significant():
    result = wfv.stat_vs_zero(pd.Series([1.0]))
    assert not result.significant
    assert result.n == 1


def test_stat_vs_zero_drops_nan_before_counting():
    values = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    result = wfv.stat_vs_zero(values)
    assert result.n == 3


def test_stat_vs_zero_drops_inf_not_just_nan():
    """REAL BUG FIXED 2026-08-29 (flagged by an external review, verified
    live before trusting it): pandas dropna() does NOT remove inf/-inf.
    An upstream ratio dividing by zero silently produced mean=inf,
    t_stat=nan (confirmed live) instead of being excluded like a
    genuinely missing observation -- fama_macbeth.py already guarded
    every regression input this way; this module was missing it."""
    values = pd.Series([1.0, 2.0, float("inf"), 3.0, -1.0, 4.0, 2.5, 1.5, 3.5, 0.5])
    result = wfv.stat_vs_zero(values)
    assert result.n == 9  # the inf row excluded
    assert np.isfinite(result.mean)
    assert np.isfinite(result.t_stat)


def test_stat_vs_zero_drops_negative_inf_too():
    values = pd.Series([1.0, 2.0, float("-inf"), 3.0, 4.0, 2.5, 1.5, 3.5])
    result = wfv.stat_vs_zero(values)
    assert result.n == 7
    assert np.isfinite(result.mean)


def test_stat_vs_zero_hac_differs_from_naive_on_overlapping_data():
    """HAC standard errors should generally be larger (t-stat smaller in
    magnitude) than naive on autocorrelated data -- doesn't assert an
    exact value (that would just re-implement statsmodels), just that the
    two code paths produce different, both-finite results."""
    rng = np.random.default_rng(42)
    base = rng.normal(1.0, 2.0, 30)
    # inject autocorrelation: each value partly repeats the previous one
    autocorrelated = pd.Series(base).rolling(3, min_periods=1).mean() + 1.0
    dates = pd.Series(pd.date_range("2020-01-01", periods=30, freq="MS"))
    naive = wfv.stat_vs_zero(autocorrelated)
    hac = wfv.stat_vs_zero(autocorrelated, dates=dates, maxlags=2)
    assert np.isfinite(naive.t_stat)
    assert np.isfinite(hac.t_stat)
    assert naive.mean == pytest.approx(hac.mean)  # point estimate unchanged, only SE differs


# --- stat_group_diff(): two-group HAC comparison ----------------------------

def test_stat_group_diff_detects_a_real_group_difference():
    values = pd.Series([10.0] * 20 + [2.0] * 20)
    group = pd.Series([True] * 20 + [False] * 20)
    result = wfv.stat_group_diff(values, group)
    assert result.mean == pytest.approx(8.0)
    assert result.significant


def test_stat_group_diff_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(5.0, 1.0, 40))
    group = pd.Series([True, False] * 20)
    result = wfv.stat_group_diff(values, group)
    assert abs(result.mean) < 2.0  # should be small, near zero


def test_stat_group_diff_drops_inf_not_just_nan():
    values = pd.Series([1.0, 2.0, float("-inf"), 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    group = pd.Series([True] * 5 + [False] * 5)
    result = wfv.stat_group_diff(values, group)
    assert result.n == 9  # the -inf row excluded
    assert np.isfinite(result.mean)
    assert np.isfinite(result.t_stat)


def test_stat_group_diff_insufficient_group_size():
    values = pd.Series([1.0, 2.0, 3.0])
    group = pd.Series([True, False, False])  # only 1 True observation
    result = wfv.stat_group_diff(values, group)
    assert not result.significant


# --- walk_forward_validate(): full orchestration ----------------------------

def test_walk_forward_validate_insufficient_data():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq="D"), "value": range(6)})
    result = wfv.walk_forward_validate(df, "date", lambda d: wfv.stat_vs_zero(d["value"]),
                                        min_n_per_split=8)
    assert result.verdict == "INSUFFICIENT_DATA"


def test_walk_forward_validate_end_to_end_robust_case():
    """A stable positive-mean signal across the whole window should come
    back ROBUST when split chronologically -- no decay by construction."""
    rng = np.random.default_rng(7)
    n = 100
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "excess_pct": rng.normal(2.0, 1.0, n),  # consistently positive mean, same distribution throughout
    })
    result = wfv.walk_forward_validate(df, "date", lambda d: wfv.stat_vs_zero(d["excess_pct"]))
    assert result.verdict in ("ROBUST", "MODERATE")  # stable signal, allow for split-sample noise
    assert result.train.significant
    assert result.retention_ratio is not None
    assert result.retention_ratio > 0


def test_walk_forward_validate_end_to_end_overfitted_case():
    """A signal that is strongly positive in the first 70% of dates and
    strongly negative in the last 30% must classify as OVERFITTED."""
    n_train, n_test = 70, 30
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n_train + n_test, freq="D"),
        "excess_pct": [5.0] * n_train + [-5.0] * n_test,
    })
    result = wfv.walk_forward_validate(df, "date", lambda d: wfv.stat_vs_zero(d["excess_pct"]))
    assert result.verdict == "OVERFITTED"
    assert result.sign_flipped


# --- cluster-robust correction (added 2026-08-23) ----------------------------
# Real bug this fixes: backtest_pit_eps_growth.py's 20d alpha result looked
# "significant" (naive t=7.03, p~=0) purely because ~300 stocks per fiscal
# year share one reporting-cohort's market-regime shock in their 60-day
# forward returns -- not because they're independent trials. Confirmed
# live: clustering by reporting-year cohort collapsed that same result to
# t=1.46 (not significant). This is a DIFFERENT correction than the
# existing HAC/maxlags path (that fixes autocorrelation within ONE
# overlapping series; this fixes many entities sharing ONE cohort shock).

def test_stat_vs_zero_cluster_robust_shrinks_t_stat_on_cohort_correlated_data():
    """Values built from a per-cohort shock + small idiosyncratic noise --
    naive (i.i.d.-assumed) t-stat must be much larger in magnitude than
    the cluster-robust one, since the true independent sample size is the
    cohort count, not the row count."""
    rng = np.random.default_rng(0)
    rows = []
    for cohort in range(10):
        shock = rng.normal(0, 5)
        for _ in range(250):
            rows.append({"cohort": cohort, "value": shock + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    naive = wfv.stat_vs_zero(df["value"])
    clustered = wfv.stat_vs_zero(df["value"], cluster_groups=df["cohort"])
    assert naive.n_clusters is None
    assert clustered.n_clusters == 10
    assert abs(clustered.t_stat) < abs(naive.t_stat) / 2  # substantially more conservative


def test_stat_vs_zero_single_cluster_falls_back_to_naive_numerically():
    """Fewer than 2 distinct clusters makes the sandwich estimator
    undefined -- must fall back to a plain OLS fit (matching naive
    exactly), not crash, while still reporting the real n_clusters so a
    caller can see why it fell back."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    naive = wfv.stat_vs_zero(values)
    one_cluster = wfv.stat_vs_zero(values, cluster_groups=pd.Series([0] * 5))
    assert one_cluster.t_stat == pytest.approx(naive.t_stat)
    assert one_cluster.n_clusters == 1


def test_stat_group_diff_cluster_robust_shrinks_t_stat_on_cohort_correlated_data():
    rng = np.random.default_rng(1)
    rows = []
    for cohort in range(10):
        shock = rng.normal(0, 5)
        for i in range(250):
            rows.append({"cohort": cohort, "group": i % 2 == 0, "value": shock + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    naive = wfv.stat_group_diff(df["value"], df["group"])
    clustered = wfv.stat_group_diff(df["value"], df["group"], cluster_groups=df["cohort"])
    assert clustered.n_clusters == 10
    # the group split is orthogonal to the cohort shock by construction (i%2),
    # so both should show near-zero effect, but clustered SE should not be
    # smaller than naive (clustering should not artificially inflate significance)
    assert abs(clustered.t_stat) <= abs(naive.t_stat) + 1.0


def test_walk_forward_validate_flags_too_few_clusters_as_a_caveat():
    """A verdict computed from a cluster-robust StatResult with fewer than
    MIN_RELIABLE_CLUSTERS distinct clusters in either split must carry an
    explicit caveat in the reasoning text -- the sandwich estimator isn't
    asymptotically reliable at that count in either direction."""
    rng = np.random.default_rng(2)
    rows = []
    for cohort in range(10):  # 7 train cohorts / 3 test cohorts after a 70/30 split -- both under 20
        shock = rng.normal(0, 5)
        for _ in range(250):
            rows.append({"cohort": cohort, "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=365 * cohort),
                         "value": shock + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    wf = wfv.walk_forward_validate(
        df, date_col="date", stat_fn=lambda d: wfv.stat_vs_zero(d["value"], cluster_groups=d["cohort"]))
    assert "CAVEAT" in wf.reasoning
    assert "fewer than" in wf.reasoning
