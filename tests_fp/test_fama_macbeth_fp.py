"""Tests for fama_macbeth.py -- the per-cohort cross-sectional regression
module built 2026-08-23 to replace pooled OLS+cluster-robust. Smoke-tested
against synthetic data with known ground truth first, per this repo's
established workflow, before being trusted against real backtest data."""
import numpy as np
import pandas as pd
import pytest

from fama_macbeth_fp import fama_macbeth_group_diff, fama_macbeth_multi_regression, fama_macbeth_regression


def _cohort_panel(rng, n_cohorts, n_per_cohort, true_slope, shock_std=5.0, noise_std=1.0, x_std=2.0):
    rows = []
    for cohort in range(n_cohorts):
        shock = rng.normal(0, shock_std)
        for _ in range(n_per_cohort):
            x = rng.normal(0, x_std)
            y = shock + true_slope * x + rng.normal(0, noise_std)
            rows.append({"cohort": cohort, "x": x, "y": y})
    return pd.DataFrame(rows)


# --- fama_macbeth_regression() ------------------------------------------------

def test_detects_a_real_slope_present_in_every_cohort():
    rng = np.random.default_rng(0)
    df = _cohort_panel(rng, n_cohorts=12, n_per_cohort=250, true_slope=0.5)
    res = fama_macbeth_regression(df, "cohort", "x", "y")
    assert res.n_periods == 12
    assert res.mean_estimate == pytest.approx(0.5, abs=0.15)
    assert res.significant


def test_rejects_a_pure_cohort_shock_confound():
    """The exact failure mode that fooled pooled OLS on the real PIT data:
    y depends only on a per-cohort shock, NOT on x at all. A naive pooled
    regression across cohorts can look significant if x happens to
    correlate with the shock in the pooled sample; Fama-MacBeth must not,
    since each cohort's OWN slope estimate is independent of its shock
    (the shock is the intercept, not the slope, in a per-cohort OLS)."""
    rng = np.random.default_rng(1)
    df = _cohort_panel(rng, n_cohorts=12, n_per_cohort=250, true_slope=0.0)
    res = fama_macbeth_regression(df, "cohort", "x", "y")
    assert not res.significant


def test_drops_cohort_with_too_few_observations():
    rng = np.random.default_rng(2)
    df = _cohort_panel(rng, n_cohorts=11, n_per_cohort=250, true_slope=0.5)
    tiny = pd.DataFrame([{"cohort": 99, "x": 1.0, "y": 2.0}, {"cohort": 99, "x": 2.0, "y": 3.0}])
    df = pd.concat([df, tiny], ignore_index=True)
    res = fama_macbeth_regression(df, "cohort", "x", "y", min_obs_per_cohort=10)
    assert 99 in res.dropped_cohorts
    assert res.n_periods == 11


def test_drops_cohort_with_constant_x_not_crash():
    rng = np.random.default_rng(3)
    df = _cohort_panel(rng, n_cohorts=11, n_per_cohort=250, true_slope=0.5)
    degenerate = pd.DataFrame({"cohort": [100] * 20, "x": [5.0] * 20, "y": rng.normal(0, 1, 20)})
    df = pd.concat([df, degenerate], ignore_index=True)
    res = fama_macbeth_regression(df, "cohort", "x", "y")
    assert 100 in res.dropped_cohorts
    assert res.n_periods == 11


def test_drops_rows_with_nan_x_or_y_before_fitting():
    rng = np.random.default_rng(4)
    df = _cohort_panel(rng, n_cohorts=11, n_per_cohort=250, true_slope=0.5)
    nan_rows = pd.DataFrame({"cohort": [0] * 5, "x": [np.nan] * 5, "y": [1.0] * 5})
    df = pd.concat([df, nan_rows], ignore_index=True)
    res = fama_macbeth_regression(df, "cohort", "x", "y")  # must not raise
    assert res.n_periods == 11


def test_fewer_than_two_usable_cohorts_is_not_significant():
    df = pd.DataFrame({"cohort": [0] * 20, "x": np.random.default_rng(5).normal(0, 1, 20),
                        "y": np.random.default_rng(6).normal(0, 1, 20)})
    res = fama_macbeth_regression(df, "cohort", "x", "y", min_obs_per_cohort=10)
    assert res.n_periods <= 1
    assert not res.significant
    assert np.isnan(res.t_stat) or not res.significant


def test_uses_exact_small_t_critical_value_not_fixed_two_point_oh():
    """Real bug found live 2026-08-23 (second review pass): a fixed
    |t|>=2.0 heuristic under-rejects for the small cohort counts (T~8-15)
    this module is built for -- e.g. at df=10, the true two-tailed 5%
    critical value is |t|>=2.228, not 2.0 (t=2.0 there is p=0.073, NOT
    significant at the conventional 0.05 bar every other stats call in
    this repo already uses). Search for an 11-cohort estimate series
    whose t-stat lands just above 2.0 but below the exact df=10 critical
    value, and confirm it comes back NOT significant."""
    import fama_macbeth as fmmod
    n = 11
    rng = np.random.default_rng(42)
    t_stat = None
    for _ in range(2000):
        vals = rng.normal(1.0, 1.0, n)
        se = vals.std(ddof=1) / (n ** 0.5)
        t = vals.mean() / se if se > 0 else 0
        if 2.0 < abs(t) < 2.228:
            t_stat = t
            break
    assert t_stat is not None, "search failed to find a t-stat in the target range -- widen the search"
    mean, std_, t_stat, p_value, significant = fmmod._t_test_period_estimates(pd.Series(vals), significance_alpha=0.05)
    assert 2.0 < abs(t_stat) < 2.228
    assert p_value > 0.05
    assert not significant  # would have been wrongly flagged significant under the old |t|>=2.0 rule


def test_p_value_matches_scipy_exact_t_distribution():
    from scipy import stats as scipy_stats
    import fama_macbeth as fmmod
    values = pd.Series([1.0, 2.5, -0.5, 3.0, 1.5, 2.0, -1.0, 4.0])
    mean, std_, t_stat, p_value, significant = fmmod._t_test_period_estimates(values, significance_alpha=0.05)
    expected_p = 2 * scipy_stats.t.sf(abs(t_stat), df=len(values) - 1)
    assert p_value == pytest.approx(expected_p)


def test_regression_drops_rows_with_infinite_x_not_crash():
    """dropna() does not remove +-inf -- a stray np.inf (e.g. from a
    caller's raw ratio dividing by a near-zero denominator) must be
    filtered explicitly, not passed into statsmodels."""
    rng = np.random.default_rng(10)
    df = _cohort_panel(rng, n_cohorts=11, n_per_cohort=250, true_slope=0.5)
    inf_rows = pd.DataFrame({"cohort": [0] * 3, "x": [np.inf, -np.inf, 1.0], "y": [1.0, 2.0, 3.0]})
    df = pd.concat([df, inf_rows], ignore_index=True)
    res = fama_macbeth_regression(df, "cohort", "x", "y")  # must not raise
    assert res.n_periods == 11


# --- fama_macbeth_multi_regression() -------------------------------------------

def _multi_cohort_panel(rng, n_cohorts, n_per_cohort, true_slopes, shock_std=5.0, noise_std=1.0, x_std=2.0):
    """Same shape as _cohort_panel() but with several x columns and a
    dict of {x_col: true_slope} -- some slopes may be 0.0 to encode a
    real null alongside a real effect in the same panel, the joint-model
    analogue of test_rejects_a_pure_cohort_shock_confound()."""
    rows = []
    x_cols = list(true_slopes.keys())
    for cohort in range(n_cohorts):
        shock = rng.normal(0, shock_std)
        for _ in range(n_per_cohort):
            xs = {col: rng.normal(0, x_std) for col in x_cols}
            y = shock + sum(true_slopes[col] * xs[col] for col in x_cols) + rng.normal(0, noise_std)
            rows.append({"cohort": cohort, "y": y, **xs})
    return pd.DataFrame(rows)


def test_multi_regression_recovers_known_slopes_jointly():
    rng = np.random.default_rng(30)
    df = _multi_cohort_panel(rng, n_cohorts=12, n_per_cohort=300,
                              true_slopes={"x1": 0.5, "x2": -0.3, "x3": 0.0})
    res = fama_macbeth_multi_regression(df, "cohort", ["x1", "x2", "x3"], "y")
    assert res.n_periods == 12
    assert res.mean_estimate["x1"] == pytest.approx(0.5, abs=0.15)
    assert res.mean_estimate["x2"] == pytest.approx(-0.3, abs=0.15)
    assert res.significant["x1"]
    assert res.significant["x2"]
    assert not res.significant["x3"]


def test_multi_regression_separates_correlated_predictors_better_than_pooling_would_suggest():
    """The joint-model analogue of test_rejects_a_pure_cohort_shock_confound():
    x2 is a pure noise variable that happens to be correlated with x1
    within each cohort -- a univariate regression of y on x2 alone could
    pick up some of x1's real effect via that correlation, but the JOINT
    model, controlling for x1, should correctly attribute the true slope
    to x1 and find nothing left over for x2."""
    rng = np.random.default_rng(31)
    rows = []
    for cohort in range(12):
        shock = rng.normal(0, 5.0)
        for _ in range(300):
            x1 = rng.normal(0, 2.0)
            x2 = 0.8 * x1 + rng.normal(0, 1.0)  # correlated with x1, no independent effect on y
            y = shock + 0.6 * x1 + rng.normal(0, 1.0)
            rows.append({"cohort": cohort, "x1": x1, "x2": x2, "y": y})
    df = pd.DataFrame(rows)
    res = fama_macbeth_multi_regression(df, "cohort", ["x1", "x2"], "y")
    assert res.mean_estimate["x1"] == pytest.approx(0.6, abs=0.2)
    assert res.significant["x1"]
    assert not res.significant["x2"]


def test_multi_regression_drops_cohort_with_too_few_observations():
    rng = np.random.default_rng(32)
    df = _multi_cohort_panel(rng, n_cohorts=11, n_per_cohort=300, true_slopes={"x1": 0.5, "x2": 0.2})
    tiny = pd.DataFrame({"cohort": [99] * 5, "x1": [1.0] * 5, "x2": [1.0] * 5, "y": [2.0] * 5})
    df = pd.concat([df, tiny], ignore_index=True)
    res = fama_macbeth_multi_regression(df, "cohort", ["x1", "x2"], "y", min_obs_per_cohort=10)
    assert 99 in res.dropped_cohorts
    assert res.n_periods == 11


def test_multi_regression_drops_rank_deficient_cohort_not_crash():
    """A cohort where two x columns are perfectly collinear has no unique
    solution -- must be dropped, not raise or silently return a garbage
    coefficient from a near-singular matrix."""
    rng = np.random.default_rng(33)
    df = _multi_cohort_panel(rng, n_cohorts=11, n_per_cohort=300, true_slopes={"x1": 0.5, "x2": 0.2})
    collinear = pd.DataFrame({"cohort": [100] * 20, "x1": list(range(20)),
                               "x2": [2 * v for v in range(20)],  # x2 = 2*x1 exactly
                               "y": rng.normal(0, 1, 20)})
    df = pd.concat([df, collinear], ignore_index=True)
    res = fama_macbeth_multi_regression(df, "cohort", ["x1", "x2"], "y")  # must not raise
    assert 100 in res.dropped_cohorts
    assert res.n_periods == 11


def test_multi_regression_drops_rows_with_nan_in_any_x_col():
    rng = np.random.default_rng(34)
    df = _multi_cohort_panel(rng, n_cohorts=11, n_per_cohort=300, true_slopes={"x1": 0.5, "x2": 0.2})
    nan_rows = pd.DataFrame({"cohort": [0] * 5, "x1": [np.nan] * 5, "x2": [1.0] * 5, "y": [1.0] * 5})
    df = pd.concat([df, nan_rows], ignore_index=True)
    res = fama_macbeth_multi_regression(df, "cohort", ["x1", "x2"], "y")  # must not raise
    assert res.n_periods == 11


# --- fama_macbeth_group_diff() ------------------------------------------------

def test_group_diff_detects_a_real_group_effect():
    rng = np.random.default_rng(7)
    rows = []
    for cohort in range(10):
        shock = rng.normal(0, 5)
        for i in range(200):
            group = i % 2 == 0
            y = shock + (3.0 if group else 0.0) + rng.normal(0, 1)
            rows.append({"cohort": cohort, "group": group, "y": y})
    df = pd.DataFrame(rows)
    res = fama_macbeth_group_diff(df, "cohort", "group", "y")
    assert res.mean_estimate == pytest.approx(3.0, abs=0.3)
    assert res.significant


def test_group_diff_rejects_pure_cohort_shock_confound():
    rng = np.random.default_rng(8)
    rows = []
    for cohort in range(10):
        shock = rng.normal(0, 5)
        for i in range(200):
            group = i % 2 == 0
            y = shock + rng.normal(0, 1)  # no real group effect
            rows.append({"cohort": cohort, "group": group, "y": y})
    df = pd.DataFrame(rows)
    res = fama_macbeth_group_diff(df, "cohort", "group", "y")
    assert not res.significant


def test_group_diff_filters_infinite_group_values_not_crash():
    """Real bug found live 2026-08-23 (third review pass): the isfinite
    filter previously covered y_col only, not group_col. A stray np.inf
    in group_col survives dropna and astype(bool) silently maps it to
    True, feeding inf into sm.OLS via the float cast. Must exclude that
    row (dropping the whole cohort here since only 10 rows exist and one
    is bad), not raise."""
    df = pd.DataFrame({"cohort": [1] * 10, "group": [True] * 5 + [False] * 4 + [np.inf], "y": [1.0] * 10})
    res = fama_macbeth_group_diff(df, "cohort", "group", "y")  # must not raise
    assert 1 in res.dropped_cohorts
    assert res.n_periods == 0


def test_regression_winsorizes_x_per_cohort_when_requested():
    """A single extreme x outlier in one cohort should stop dominating
    that cohort's slope once winsorize_x_pct clips it to the cohort's own
    percentile range."""
    rng = np.random.default_rng(20)
    df = _cohort_panel(rng, n_cohorts=11, n_per_cohort=250, true_slope=0.3)
    outlier_idx = len(df)
    df.loc[outlier_idx] = {"cohort": 0, "x": 5000.0, "y": 1.0}
    no_wins = fama_macbeth_regression(df, "cohort", "x", "y")
    wins = fama_macbeth_regression(df, "cohort", "x", "y", winsorize_x_pct=0.01)
    assert abs(wins.period_estimates[0] - 0.3) < abs(no_wins.period_estimates[0] - 0.3)


def test_group_diff_drops_cohort_with_one_sided_group():
    """A cohort where every row is the SAME group has no mean-difference
    to compute -- must be dropped, not raise or fabricate a value."""
    rng = np.random.default_rng(9)
    rows = []
    for cohort in range(9):
        shock = rng.normal(0, 5)
        for i in range(200):
            group = i % 2 == 0
            rows.append({"cohort": cohort, "group": group, "y": shock + rng.normal(0, 1)})
    one_sided = pd.DataFrame({"cohort": [99] * 20, "group": [True] * 20, "y": rng.normal(0, 1, 20)})
    df = pd.concat([pd.DataFrame(rows), one_sided], ignore_index=True)
    res = fama_macbeth_group_diff(df, "cohort", "group", "y")
    assert 99 in res.dropped_cohorts
    assert res.n_periods == 9
