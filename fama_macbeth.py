"""
Fama-MacBeth cross-sectional regression for the "many entities, few time
periods" panel shape that pooled OLS (even with walk_forward_validator.py's
HAC/cluster-robust corrections) fits poorly.

On such panels naive pooled t-stats can be inflated several-fold, and
cluster-robust correction hits the Cameron/Gelbach/Miller (2008) small-G
problem once a 70/30 walk-forward split leaves only 2-4 cohorts in the
test window -- a floor no amount of standard-error patching can fix,
because the TRUE sample size is the cohort count.

METHOD: run one cross-sectional regression PER COHORT (e.g. one fiscal
year), collect that cohort's slope (or group-mean-difference) estimate,
then test whether the resulting TIME SERIES of per-cohort estimates
(typically ~10-20) has a mean distinguishable from zero via a one-sample
t-test with T-1 degrees of freedom. This sidesteps the clustering problem entirely rather than
patching around it -- "cohort" is the unit of analysis from the start, so
there's no i.i.d.-violation left to correct for. Standard reference:
Fama & MacBeth (1973), "Risk, Return, and Equilibrium: Empirical Tests".

Every entry point also takes:
- min_periods (default 3): with fewer usable cohorts, mean/std are
  reported but t_stat/p_value are NaN and nothing is significant.
- newey_west_lags (default None): replaces the plain std/sqrt(T) SE with
  a Newey-West (Bartlett) SE, for cohort estimates that are serially
  correlated -- e.g. forward-return horizons longer than the spacing
  between cohorts, so adjacent cohorts share return periods.

Deliberately NOT using walk_forward_validator.py's HAC/cluster_groups
machinery here -- that module answers "how do I correct standard errors
on a POOLED regression", this module answers "don't pool in the first
place". The two are complementary: fama_macbeth_regression()'s per-cohort
slopes can themselves be walk-forward split (train cohorts vs. test
cohorts) via walk_forward_validator.chronological_split() + stat_vs_zero()
on the slope series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DEFAULT_MIN_OBS_PER_COHORT = 10  # below this, a per-cohort slope is too noisy to trust as one data point
DEFAULT_SIGNIFICANCE_ALPHA = 0.05
DEFAULT_MIN_PERIODS = 3  # below this many cohorts, no t-stat is reported


@dataclass
class FamaMacBethResult:
    period_estimates: pd.Series  # index=cohort label, value=that cohort's slope/group-diff estimate
    mean_estimate: float
    std_estimate: float
    t_stat: float
    p_value: float
    n_periods: int  # cohorts actually used (i.e. len(period_estimates))
    significant: bool
    dropped_cohorts: list = field(default_factory=list)  # cohorts skipped for too few/degenerate obs


def _newey_west_se(estimates: pd.Series, lags: int) -> float:
    """Newey-West (Bartlett-kernel) standard error of the mean of the
    per-cohort estimate series: sqrt((g0 + 2 * sum_{l=1..L} (1 - l/(L+1)) g_l) / T),
    g_l the lag-l autocovariance of the demeaned series (divided by T).
    Use when adjacent cohorts' estimates are serially correlated, e.g.
    forward-return horizons longer than the cohort spacing."""
    x = estimates.to_numpy(dtype=float)
    n = len(x)
    d = x - x.mean()
    lrv = float(d @ d) / n
    for lag in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * float(d[lag:] @ d[:-lag]) / n
    return float(np.sqrt(max(lrv, 0.0) / n))


def _t_test_period_estimates(estimates: pd.Series, significance_alpha: float,
                             min_periods: int = DEFAULT_MIN_PERIODS,
                             newey_west_lags: int | None = None) -> tuple[float, float, float, float, bool]:
    """One-sample t-test on the per-cohort estimate series (the
    Fama-MacBeth standard error, std / sqrt(T)), using the exact Student's-t
    distribution with df=T-1 rather than a fixed |t|>=2.0 heuristic --
    material at the T=8-15 cohort counts this module typically sees (at
    T=11 the two-tailed 5% critical value is 2.228, not 2.0).

    Fewer than min_periods cohorts -> mean/std are still reported but
    t_stat/p_value are NaN and nothing is significant: a t-test on 2 or 3
    cohorts is not evidence. newey_west_lags (default None = plain FM SE)
    swaps in a Newey-West SE for serially correlated cohort estimates."""
    n = len(estimates)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), False
    mean = float(estimates.mean())
    std = float(estimates.std(ddof=1))
    if n < max(min_periods, 2):
        return mean, std, float("nan"), float("nan"), False
    se = _newey_west_se(estimates, newey_west_lags) if newey_west_lags else std / (n ** 0.5)
    if se == 0:
        return mean, std, float("nan"), float("nan"), False
    t_stat = mean / se
    p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2)
    significant = bool(np.isfinite(p_value) and p_value < significance_alpha)
    return mean, std, t_stat, p_value, significant


def fama_macbeth_regression(
    df: pd.DataFrame, cohort_col: str, x_col: str, y_col: str,
    min_obs_per_cohort: int = DEFAULT_MIN_OBS_PER_COHORT,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    min_periods: int = DEFAULT_MIN_PERIODS, newey_west_lags: int | None = None,
    winsorize_x_pct: float | None = None,
) -> FamaMacBethResult:
    """One OLS(y ~ x) regression per distinct value of cohort_col, slope
    coefficient only. A cohort is dropped (not zero-filled) if it has
    fewer than min_obs_per_cohort usable rows, if x is constant within
    that cohort (slope undefined), or if any non-finite value (+-inf,
    e.g. from a raw ratio's zero-denominator division upstream) survived
    dropna -- "skip, don't fabricate" for undefined values.

    winsorize_x_pct (e.g. 0.01 for 1%/99%): clips x to its OWN COHORT's
    percentile range before fitting -- per-cohort, not pooled, since a
    pooled-percentile clip would itself leak cross-cohort information
    into what's supposed to be an independent per-period estimate. Off
    by default (None) since it changes the estimate's economic meaning
    (attenuates a real fat-tailed effect, not just noise) -- opt in only
    when a caller's raw variable is known to have extreme un-winsorized
    outliers, e.g. EPS growth off a near-zero prior-year base, which can
    run to thousands of percent in either direction."""
    slopes = {}
    dropped = []
    for cohort, group in df.groupby(cohort_col):
        sub = group.dropna(subset=[x_col, y_col])
        sub = sub[np.isfinite(sub[x_col].astype(float)) & np.isfinite(sub[y_col].astype(float))]
        if len(sub) < min_obs_per_cohort:
            dropped.append(cohort)
            continue
        x = sub[x_col].astype(float)
        y = sub[y_col].astype(float)
        if winsorize_x_pct:
            lo, hi = x.quantile(winsorize_x_pct), x.quantile(1 - winsorize_x_pct)
            x = x.clip(lo, hi)
        if x.std(ddof=0) == 0:
            dropped.append(cohort)
            continue
        model = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit()
        slopes[cohort] = float(model.params[1])  # [const, x] -- x is always index 1 here

    period_estimates = pd.Series(slopes)
    mean, std, t_stat, p_value, significant = _t_test_period_estimates(period_estimates, significance_alpha,
        min_periods, newey_west_lags)
    return FamaMacBethResult(period_estimates=period_estimates, mean_estimate=mean, std_estimate=std,
                              t_stat=t_stat, p_value=p_value, n_periods=len(period_estimates),
                              significant=significant, dropped_cohorts=dropped)


@dataclass
class FamaMacBethMultiResult:
    period_estimates: pd.DataFrame  # index=cohort label, columns=x_cols, values=that cohort's slope for that x
    mean_estimate: pd.Series  # index=x_cols
    std_estimate: pd.Series
    t_stat: pd.Series
    p_value: pd.Series
    n_periods: int  # cohorts actually used (rows in period_estimates)
    significant: pd.Series  # bool per x_col
    dropped_cohorts: list = field(default_factory=list)


def fama_macbeth_multi_regression(
    df: pd.DataFrame, cohort_col: str, x_cols: list[str], y_col: str,
    min_obs_per_cohort: int = DEFAULT_MIN_OBS_PER_COHORT,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    min_periods: int = DEFAULT_MIN_PERIODS, newey_west_lags: int | None = None,
) -> FamaMacBethMultiResult:
    """Multivariate per-cohort OLS(y ~ x_1 + ... + x_k) -- the genuine
    Fama-MacBeth (1973) two-pass shape (their original cross-sectional
    regression used several firm characteristics jointly; fama_macbeth_
    regression() above is the univariate special case). Returns one slope PER
    x_col PER cohort, then the same per-column one-sample t-test as the
    univariate version. A cohort is dropped (not zero-filled) if it has
    fewer than min_obs_per_cohort usable rows after dropping any row with
    a NaN/non-finite value in ANY x_col or y_col (a joint model needs
    every characteristic present on the same row, unlike running each
    x_col's own univariate regression separately, which can use a
    different, larger sample per characteristic) -- or if the per-cohort
    design matrix is rank-deficient (collinear x's, or too few rows for
    the number of parameters), matching the univariate function's
    'skip, don't fabricate' convention."""
    all_cols = list(x_cols) + [y_col]
    per_cohort_slopes = {}
    dropped = []
    for cohort, group in df.groupby(cohort_col):
        sub = group.dropna(subset=all_cols)
        sub = sub[np.all(np.isfinite(sub[all_cols].astype(float).to_numpy()), axis=1)]
        n_params = len(x_cols) + 1  # +1 for the constant
        if len(sub) < min_obs_per_cohort or len(sub) <= n_params:
            dropped.append(cohort)
            continue
        X = sm.add_constant(sub[x_cols].astype(float))
        if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
            dropped.append(cohort)
            continue
        y = sub[y_col].astype(float)
        model = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
        per_cohort_slopes[cohort] = dict(zip(x_cols, model.params[1:]))  # params[0] is const

    period_estimates = pd.DataFrame.from_dict(per_cohort_slopes, orient="index", columns=x_cols)
    means, stds, tstats, pvals, sigs = {}, {}, {}, {}, {}
    for col in x_cols:
        m, s, t, p, sig = _t_test_period_estimates(period_estimates[col].dropna(), significance_alpha,
        min_periods, newey_west_lags)
        means[col], stds[col], tstats[col], pvals[col], sigs[col] = m, s, t, p, sig
    return FamaMacBethMultiResult(
        period_estimates=period_estimates,
        mean_estimate=pd.Series(means), std_estimate=pd.Series(stds),
        t_stat=pd.Series(tstats), p_value=pd.Series(pvals),
        n_periods=len(period_estimates), significant=pd.Series(sigs),
        dropped_cohorts=dropped,
    )


def fama_macbeth_group_diff(
    df: pd.DataFrame, cohort_col: str, group_col: str, y_col: str,
    min_obs_per_cohort: int = DEFAULT_MIN_OBS_PER_COHORT,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    min_periods: int = DEFAULT_MIN_PERIODS, newey_west_lags: int | None = None,
) -> FamaMacBethResult:
    """Same idea as fama_macbeth_regression() but for a BINARY group
    comparison per cohort (e.g. "ROCE improving" vs. "not") -- per-cohort
    estimate is
    mean(y[group]) - mean(y[~group]), computed via the same dummy-OLS
    shape walk_forward_validator.stat_group_diff() uses, so a cohort with
    fewer than 2 observations in EITHER group is dropped (a mean
    difference needs both sides represented)."""
    estimates = {}
    dropped = []
    for cohort, group_df in df.groupby(cohort_col):
        sub = group_df.dropna(subset=[group_col, y_col])
        # group_col is typically already bool, but must still be checked --
        # a stray np.inf from an upstream ratio survives dropna and
        # astype(bool) silently maps it to True, feeding inf into
        # sm.OLS via the group_col.astype(float) cast below.
        sub = sub[np.isfinite(sub[y_col].astype(float)) & np.isfinite(sub[group_col].astype(float))]
        n_true = int(sub[group_col].astype(bool).sum())
        n_false = int((~sub[group_col].astype(bool)).sum())
        if len(sub) < min_obs_per_cohort or n_true < 2 or n_false < 2:
            dropped.append(cohort)
            continue
        y = sub[y_col].astype(float).to_numpy()
        x = sm.add_constant(sub[group_col].astype(float))
        model = sm.OLS(y, x).fit()
        estimates[cohort] = float(model.params[group_col])

    period_estimates = pd.Series(estimates)
    mean, std, t_stat, p_value, significant = _t_test_period_estimates(period_estimates, significance_alpha,
        min_periods, newey_west_lags)
    return FamaMacBethResult(period_estimates=period_estimates, mean_estimate=mean, std_estimate=std,
                              t_stat=t_stat, p_value=p_value, n_periods=len(period_estimates),
                              significant=significant, dropped_cohorts=dropped)
