"""
fama_macbeth_fp.py -- Functional-programming rewrite of fama_macbeth.py.

DROP-IN COMPATIBLE: same public functions (fama_macbeth_regression,
fama_macbeth_multi_regression, fama_macbeth_group_diff), same dataclass
fields, same return values -- including dropped-cohort ORDER, which
matches groupby's natural (sorted-key) iteration order exactly as the
original did.

WHAT CHANGED:

The original's three top-level functions all shared one imperative
shape: `for cohort, group in df.groupby(...): ... if <skip condition>:
dropped.append(cohort); continue ... slopes[cohort] = <estimate>`. Two
mutable accumulator dicts/lists were built up by hand across a loop
body that mixed "should we skip this cohort" logic with "how do we fit
it" logic.

Replaced with: one pure per-cohort function returning `float | None` (or
`dict[str, float] | None` for the multivariate case) -- None meaning
"drop this cohort", any other value meaning "use this estimate". The
three functions now do:

    per_cohort = {cohort: _fit_one_cohort(group, ...) for cohort, group in df.groupby(...)}
    kept    = {c: v for c, v in per_cohort.items() if v is not None}
    dropped = [c for c, v in per_cohort.items() if v is None]

i.e. compute-then-partition instead of accumulate-while-branching. Each
`_fit_one_cohort*` function is now independently unit-testable against
a single cohort's DataFrame with no groupby/dict-building machinery
around it, and there's no `.append()`/`continue` control flow left in
any of the three public functions.

`_t_test_period_estimates` was already a pure function in the original
and is carried over unchanged (the exact Student's-t p-value, not a
fixed-t heuristic, is the actual point of that function -- nothing
about a functional rewrite improves on that logic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DEFAULT_MIN_OBS_PER_COHORT = 10
DEFAULT_SIGNIFICANCE_ALPHA = 0.05


@dataclass
class FamaMacBethResult:
    period_estimates: pd.Series
    mean_estimate: float
    std_estimate: float
    t_stat: float
    p_value: float
    n_periods: int
    significant: bool
    dropped_cohorts: list = field(default_factory=list)


@dataclass
class FamaMacBethMultiResult:
    period_estimates: pd.DataFrame
    mean_estimate: pd.Series
    std_estimate: pd.Series
    t_stat: pd.Series
    p_value: pd.Series
    n_periods: int
    significant: pd.Series
    dropped_cohorts: list = field(default_factory=list)


def _t_test_period_estimates(estimates: pd.Series, significance_alpha: float) -> tuple[float, float, float, float, bool]:
    """Plain one-sample t-test on the per-cohort estimate series (the
    Fama-MacBeth standard error), using the exact Student's-t survival
    function with df=n-1 rather than a fixed |t|>=2.0 heuristic --
    material at the T=8-15 cohort counts this module typically sees."""
    n = len(estimates)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), False
    mean = float(estimates.mean())
    std = float(estimates.std(ddof=1))
    se = std / (n ** 0.5)
    if se == 0:
        return mean, std, float("nan"), float("nan"), False
    t_stat = mean / se
    p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2)
    significant = bool(np.isfinite(p_value) and p_value < significance_alpha)
    return mean, std, t_stat, p_value, significant


def _clean_finite(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pure: drop rows with NaN in any of `cols`, then drop rows with
    non-finite (+-inf) values in any of `cols` -- dropna() alone
    doesn't catch inf, same gap fixed in walk_forward_validator.py and
    the original of this module."""
    sub = sub.dropna(subset=cols)
    finite_mask = np.all(np.isfinite(sub[cols].astype(float).to_numpy()), axis=1)
    return sub[finite_mask]


def _fit_one_cohort_slope(
    group: pd.DataFrame, x_col: str, y_col: str, min_obs_per_cohort: int, winsorize_x_pct: Optional[float],
) -> Optional[float]:
    """One cohort's univariate OLS(y ~ x) slope, or None if this cohort
    should be dropped (too few usable rows, or x constant within the
    cohort so the slope is undefined). winsorize_x_pct clips x to ITS
    OWN COHORT's percentile range, never a pooled one, so no cross-
    cohort information leaks into an otherwise-independent estimate."""
    sub = _clean_finite(group, [x_col, y_col])
    if len(sub) < min_obs_per_cohort:
        return None
    x = sub[x_col].astype(float)
    y = sub[y_col].astype(float)
    if winsorize_x_pct:
        lo, hi = x.quantile(winsorize_x_pct), x.quantile(1 - winsorize_x_pct)
        x = x.clip(lo, hi)
    if x.std(ddof=0) == 0:
        return None
    model = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit()
    return float(model.params[1])


def fama_macbeth_regression(
    df: pd.DataFrame, cohort_col: str, x_col: str, y_col: str,
    min_obs_per_cohort: int = DEFAULT_MIN_OBS_PER_COHORT,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
    winsorize_x_pct: Optional[float] = None,
) -> FamaMacBethResult:
    """One OLS(y ~ x) regression per distinct value of cohort_col."""
    per_cohort = {
        cohort: _fit_one_cohort_slope(group, x_col, y_col, min_obs_per_cohort, winsorize_x_pct)
        for cohort, group in df.groupby(cohort_col)
    }
    slopes = {c: v for c, v in per_cohort.items() if v is not None}
    dropped = [c for c, v in per_cohort.items() if v is None]

    period_estimates = pd.Series(slopes)
    mean, std, t_stat, p_value, significant = _t_test_period_estimates(period_estimates, significance_alpha)
    return FamaMacBethResult(period_estimates=period_estimates, mean_estimate=mean, std_estimate=std,
                              t_stat=t_stat, p_value=p_value, n_periods=len(period_estimates),
                              significant=significant, dropped_cohorts=dropped)


def _fit_one_cohort_multi_slopes(
    group: pd.DataFrame, x_cols: list[str], y_col: str, min_obs_per_cohort: int,
) -> Optional[dict[str, float]]:
    """One cohort's multivariate OLS(y ~ x_1 + ... + x_k) slopes, or
    None if this cohort should be dropped (too few usable rows for the
    number of parameters, or a rank-deficient design matrix)."""
    all_cols = list(x_cols) + [y_col]
    sub = _clean_finite(group, all_cols)
    n_params = len(x_cols) + 1
    if len(sub) < min_obs_per_cohort or len(sub) <= n_params:
        return None
    X = sm.add_constant(sub[x_cols].astype(float))
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        return None
    y = sub[y_col].astype(float)
    model = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
    return dict(zip(x_cols, model.params[1:]))  # params[0] is const


def fama_macbeth_multi_regression(
    df: pd.DataFrame, cohort_col: str, x_cols: list[str], y_col: str,
    min_obs_per_cohort: int = DEFAULT_MIN_OBS_PER_COHORT,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
) -> FamaMacBethMultiResult:
    """The genuine Fama-MacBeth (1973) two-pass shape: one slope PER
    x_col PER cohort, jointly fit, then the same per-column one-sample
    t-test as the univariate version."""
    per_cohort = {
        cohort: _fit_one_cohort_multi_slopes(group, x_cols, y_col, min_obs_per_cohort)
        for cohort, group in df.groupby(cohort_col)
    }
    kept = {c: v for c, v in per_cohort.items() if v is not None}
    dropped = [c for c, v in per_cohort.items() if v is None]

    period_estimates = pd.DataFrame.from_dict(kept, orient="index", columns=x_cols)
    per_col_stats = {col: _t_test_period_estimates(period_estimates[col].dropna(), significance_alpha)
                      for col in x_cols}
    means = {col: s[0] for col, s in per_col_stats.items()}
    stds = {col: s[1] for col, s in per_col_stats.items()}
    tstats = {col: s[2] for col, s in per_col_stats.items()}
    pvals = {col: s[3] for col, s in per_col_stats.items()}
    sigs = {col: s[4] for col, s in per_col_stats.items()}

    return FamaMacBethMultiResult(
        period_estimates=period_estimates,
        mean_estimate=pd.Series(means), std_estimate=pd.Series(stds),
        t_stat=pd.Series(tstats), p_value=pd.Series(pvals),
        n_periods=len(period_estimates), significant=pd.Series(sigs),
        dropped_cohorts=dropped,
    )


def _fit_one_cohort_group_diff(
    group_df: pd.DataFrame, group_col: str, y_col: str, min_obs_per_cohort: int,
) -> Optional[float]:
    """One cohort's mean(y[group]) - mean(y[~group]), or None if this
    cohort should be dropped (too few total rows, or fewer than 2 rows
    in either side of the binary split). group_col is checked for
    finiteness too, not just y_col -- a stray inf surviving dropna and
    an astype(bool) cast would otherwise silently map to True."""
    sub = _clean_finite(group_df, [group_col, y_col])
    n_true = int(sub[group_col].astype(bool).sum())
    n_false = int((~sub[group_col].astype(bool)).sum())
    if len(sub) < min_obs_per_cohort or n_true < 2 or n_false < 2:
        return None
    y = sub[y_col].astype(float).to_numpy()
    x = sm.add_constant(sub[group_col].astype(float))
    model = sm.OLS(y, x).fit()
    return float(model.params[group_col])


def fama_macbeth_group_diff(
    df: pd.DataFrame, cohort_col: str, group_col: str, y_col: str,
    min_obs_per_cohort: int = DEFAULT_MIN_OBS_PER_COHORT,
    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA,
) -> FamaMacBethResult:
    """Same idea as fama_macbeth_regression() but for a binary group
    comparison per cohort, via the same dummy-OLS shape
    walk_forward_validator.stat_group_diff() uses."""
    per_cohort = {
        cohort: _fit_one_cohort_group_diff(group_df, group_col, y_col, min_obs_per_cohort)
        for cohort, group_df in df.groupby(cohort_col)
    }
    estimates = {c: v for c, v in per_cohort.items() if v is not None}
    dropped = [c for c, v in per_cohort.items() if v is None]

    period_estimates = pd.Series(estimates)
    mean, std, t_stat, p_value, significant = _t_test_period_estimates(period_estimates, significance_alpha)
    return FamaMacBethResult(period_estimates=period_estimates, mean_estimate=mean, std_estimate=std,
                              t_stat=t_stat, p_value=p_value, n_periods=len(period_estimates),
                              significant=significant, dropped_cohorts=dropped)
