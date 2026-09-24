"""
walk_forward_validator_fp.py -- Functional-programming rewrite of
walk_forward_validator.py.

DROP-IN COMPATIBLE: every public name, signature, and return type matches
the original exactly. `classify_overfitting(...)` still returns the same
`tuple[str, str, float | None, bool]`; `StatResult` and `WalkForwardResult`
have the same fields. Behavior is unchanged -- this is a restructuring,
not a bug fix or a semantic change. (A couple of judgment calls noted in
the original review -- e.g. the sign-flip rule firing on near-zero means
-- are intentionally preserved as-is here; see the note above `_RULES`
for how you'd patch that in without touching the dispatch mechanism.)

WHAT CHANGED, AND WHY IT'S "MORE FP":

1. Immutability: StatResult / WalkForwardResult / Verdict are now
   frozen, slotted dataclasses. Nothing after construction can mutate
   them -- a StatResult computed on the train split literally cannot be
   accidentally overwritten with test-split numbers later in a longer
   pipeline, which is the exact class of bug PIT/backtest code is
   prone to.

2. classify_overfitting's if/elif ladder -> a declarative rule table.
   Each branch of the original conditional is now an independent pure
   function `_Ctx -> Verdict | None`, and `_RULES` is an ordered tuple
   evaluated top-down via `next(...)` until one returns non-None. This
   is the classic "chain of responsibility" / railway pattern. Payoff:
   each rule is unit-testable in isolation against a synthetic _Ctx,
   and adding a new guard (e.g. a magnitude floor on the sign-flip
   rule) means inserting one function into the tuple, not editing a
   nested conditional.

3. _fit_ols's branching -> a lazy predicate/strategy table. The three
   fit strategies (cluster-robust, HAC, plain) are each their own pure
   function; only the one whose predicate is True actually executes
   (via a generator expression, not a list comprehension -- laziness
   matters here since fitting is the expensive part).

4. No in-place mutation anywhere. The original's `split_idx = ...`
   reassignment inside chronological_split becomes a pure helper
   `_snap_to_boundary` that takes the raw index and returns the
   adjusted one. `reasoning += ...` becomes `_append_cluster_caveat`
   returning a new string.

5. Small composition helper `pipe` (reduce-based) used for the
   inf-scrubbing step, in place of two sequential statements -- mostly
   readability, but it's the same idea as `.pipe()` chains in
   pandas/polars.

What's deliberately NOT changed: this still uses pandas/numpy/statsmodels
and still returns plain dataclasses rather than e.g. a Result/Either
monad -- a full Railway-oriented rewrite (Optional -> Maybe, exceptions
-> Result) would change the calling convention for every existing caller
in the repo, which defeats "drop-in."
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import reduce
from typing import Callable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DEFAULT_SIGNIFICANCE_ALPHA = 0.05  # two-sided; drives the default small-sample critical t
DEFAULT_SIGNIFICANCE_T = 2.0  # large-sample rule of thumb; pass as significance_t for a fixed threshold
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_MIN_N_PER_SPLIT = 8
DEFAULT_ROBUST_RETENTION = 0.7
DEFAULT_MODERATE_RETENTION = 0.4
DEFAULT_OVERFIT_RETENTION = 0.3
MIN_RELIABLE_CLUSTERS = 20


# --------------------------------------------------------------------------
# Immutable data types
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StatResult:
    mean: float
    t_stat: float
    n: int
    significant: bool
    n_clusters: Optional[int] = None
    df: Optional[float] = None  # degrees of freedom for the critical t; None -> n - 1


@dataclass(frozen=True, slots=True)
class Verdict:
    """Internal: one rule's output. Same four fields classify_overfitting
    has always returned as a bare tuple -- named here so each rule
    function is self-documenting."""
    label: str
    reasoning: str
    retention_ratio: Optional[float]
    sign_flipped: bool


def critical_t(stat: StatResult, significance_t: Optional[float] = None,
               alpha: float = DEFAULT_SIGNIFICANCE_ALPHA) -> float:
    """The |t| a StatResult must reach to count as significant: significance_t
    when given (fixed threshold), else the two-sided Student-t critical value
    at alpha with stat.df degrees of freedom (n - 1 if unset); inf when there
    are no degrees of freedom."""
    if significance_t is not None:
        return float(significance_t)
    dof = stat.df if stat.df is not None else stat.n - 1
    if not dof or dof < 1 or np.isnan(dof):
        return float("inf")
    return float(stats.t.ppf(1 - alpha / 2, dof))


def _dof(n: int, n_params: int, n_clusters: Optional[int]) -> int:
    """G - 1 for a cluster-robust fit, n - n_params otherwise."""
    return n_clusters - 1 if n_clusters is not None and n_clusters >= 2 else n - n_params


def _is_significant(stat: StatResult, significance_t: Optional[float]) -> bool:
    return bool(not np.isnan(stat.t_stat) and abs(stat.t_stat) >= critical_t(stat, significance_t))


def _with_significance(stat: StatResult, significance_t: Optional[float]) -> StatResult:
    """Pure: a copy of stat with .significant filled in."""
    return replace(stat, significant=_is_significant(stat, significance_t))


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    train: StatResult
    test: StatResult
    retention_ratio: Optional[float]
    sign_flipped: bool
    verdict: str
    reasoning: str
    n_purged_train: int = 0
    n_embargoed_test: int = 0


# --------------------------------------------------------------------------
# Tiny composition helper
# --------------------------------------------------------------------------

def pipe(x, *fns: Callable):
    """Left-to-right function composition: pipe(x, f, g, h) == h(g(f(x)))."""
    return reduce(lambda acc, f: f(acc), fns, x)


# --------------------------------------------------------------------------
# chronological_split -- pure, no reassignment of the split index in place
# --------------------------------------------------------------------------

def _snap_to_boundary(dates: pd.Series, split_idx: int) -> int:
    """Given the naive row-count split index, return the adjusted index
    that avoids slicing through a run of tied dates -- same rule as the
    original, just extracted so chronological_split has no local mutation."""
    if not (0 < split_idx < len(dates)):
        return split_idx
    boundary_date = dates.iloc[split_idx]
    run_positions = np.flatnonzero((dates == boundary_date).to_numpy())
    if len(run_positions) == 0:
        return split_idx
    run_start, run_end = int(run_positions[0]), int(run_positions[-1]) + 1
    if not (run_start < split_idx < run_end):
        return split_idx
    return run_start if (split_idx - run_start) <= (run_end - split_idx) else run_end


def date_sort_key(dates: pd.Series) -> pd.Series:
    """The values rows are ORDERED by. datetime64 and numeric columns (e.g. a
    fiscal-year int) are used as-is; anything else (str/object) must parse
    as ISO-8601 -- sorting raw strings is lexicographic, so "15/01/2020"
    would land after "01/02/2021", and a day-first vs. month-first guess
    can silently scramble the timeline. Raises on missing dates rather than
    letting NaT/NaN sort to the end and fall into the test window."""
    dates = pd.Series(dates)
    if pd.api.types.is_datetime64_any_dtype(dates) or pd.api.types.is_numeric_dtype(dates):
        key = dates
    else:
        try:
            key = pd.to_datetime(dates, format="ISO8601")
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"date column {dates.name!r} must be datetime-like, numeric, or ISO-8601 strings "
                f"(e.g. '2020-01-31'); convert it first with pd.to_datetime(..., format=...). ({e})") from e
    n_missing = int(key.isna().sum())
    if n_missing:
        raise ValueError(f"date column {dates.name!r} has {n_missing} missing value(s); drop or fill them first "
                         "-- a row with no date can't be placed on either side of a split.")
    return key


def chronological_split(
    df: pd.DataFrame, date_col: str, train_frac: float = DEFAULT_TRAIN_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    key = date_sort_key(df[date_col])
    pos = np.argsort(key.to_numpy(), kind="stable")
    ordered = df.iloc[pos].reset_index(drop=True)
    split_idx = _snap_to_boundary(key.iloc[pos].reset_index(drop=True), int(len(ordered) * train_frac))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


# --------------------------------------------------------------------------
# apply_purge_embargo -- pure filter, no mutation of the frames it's given
# --------------------------------------------------------------------------

def _purge_train(train_df: pd.DataFrame, date_col: str, test_start, purge_days: int) -> tuple[pd.DataFrame, int]:
    """Pure: returns (kept_train, n_purged) without mutating train_df."""
    if purge_days <= 0 or train_df.empty:
        return train_df, 0
    train_dates = pd.to_datetime(train_df[date_col])
    cutoff = test_start - pd.Timedelta(days=purge_days)
    # Strict: a row dated exactly purge_days before test_start resolves ON
    # test_start ("on or after") -- purged.
    keep_mask = (train_dates < cutoff).to_numpy()
    return train_df.loc[keep_mask], int((~keep_mask).sum())


def _embargo_test(test_df: pd.DataFrame, test_dates: pd.Series, test_start, embargo_days: int) -> tuple[pd.DataFrame, int]:
    """Pure: returns (kept_test, n_embargoed) without mutating test_df."""
    if embargo_days <= 0:
        return test_df, 0
    cutoff = test_start + pd.Timedelta(days=embargo_days)
    keep_mask = (test_dates >= cutoff).to_numpy()
    return test_df.loc[keep_mask], int((~keep_mask).sum())


def apply_purge_embargo(
    train_df: pd.DataFrame, test_df: pd.DataFrame, date_col: str,
    purge_days: int = 0, embargo_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Drops train rows whose forward-return window bleeds into the test
    period (purge_days), and test rows within embargo_days of the test
    window's own start. Both no-ops when the corresponding *_days is 0.
    Composed from two pure single-purpose helpers rather than one function
    doing both jobs -- each is independently testable."""
    if test_df.empty or (purge_days <= 0 and embargo_days <= 0):
        return train_df, test_df, 0, 0
    test_dates = pd.to_datetime(test_df[date_col])
    test_start = test_dates.min()
    purged_train, n_purged = _purge_train(train_df, date_col, test_start, purge_days)
    embargoed_test, n_embargoed = _embargo_test(test_df, test_dates, test_start, embargo_days)
    return purged_train, embargoed_test, n_purged, n_embargoed


# --------------------------------------------------------------------------
# OLS fit dispatch -- lazy predicate/strategy table instead of if/elif
# --------------------------------------------------------------------------

def _fit_cluster(y: np.ndarray, x: pd.DataFrame, groups) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})


def _fit_hac(y: np.ndarray, x: pd.DataFrame, maxlags: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def _fit_plain(y: np.ndarray, x: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, x).fit()


def _align_to(index: pd.Index, groups, name: str = "cluster_groups") -> pd.Series:
    """Lines per-row labels (cluster ids, dates) up with the values' rows. A
    Series aligns by INDEX LABEL (so d["cl"] from the same frame just works);
    anything else (ndarray, list) aligns by POSITION and must be the same
    length. Previously a plain array was wrapped in a fresh 0..n-1 index and
    then label-matched, which raised KeyError -- or silently mislabelled
    rows -- whenever the frame's index wasn't 0..n-1 (every CPCV train frame)."""
    if isinstance(groups, pd.Series):
        missing = index.difference(groups.index)
        if len(missing):
            raise ValueError(f"{name} is a Series missing {len(missing)} of the values' index labels "
                             f"(e.g. {list(missing[:3])}); pass it from the same frame, or as an array.")
        return groups.loc[index]
    arr = np.asarray(groups)
    if len(arr) != len(index):
        raise ValueError(f"{name} has length {len(arr)} but there are {len(index)} values.")
    return pd.Series(arr, index=index)


def _require_complete(groups: pd.Series, name: str = "cluster_groups") -> pd.Series:
    n_missing = int(groups.isna().sum())
    if n_missing:
        raise ValueError(f"{name} has {n_missing} missing label(s) on rows being tested.")
    return groups


def _fit_ols(y: np.ndarray, x: pd.DataFrame, maxlags: Optional[int], cluster_groups: Optional[pd.Series]):
    """Cluster-robust (if enough distinct groups) takes priority over HAC,
    which takes priority over a plain fit. Expressed as an ordered
    (predicate, thunk) table evaluated lazily via a generator -- only the
    winning strategy's `sm.OLS(...).fit()` actually runs."""
    n_clusters = pd.Series(cluster_groups).nunique() if cluster_groups is not None else None
    strategies: tuple[tuple[bool, Callable[[], object]], ...] = (
        (n_clusters is not None and n_clusters >= 2, lambda: _fit_cluster(y, x, cluster_groups)),
        (maxlags is not None, lambda: _fit_hac(y, x, maxlags)),
        (True, lambda: _fit_plain(y, x)),
    )
    model = next(thunk() for cond, thunk in strategies if cond)
    return model, n_clusters


def _t_from_ols_const(
    y: np.ndarray, dates: Optional[pd.Series], maxlags: Optional[int],
    cluster_groups: Optional[pd.Series] = None,
) -> tuple[float, float, Optional[int]]:
    x = pd.DataFrame({"const": np.ones(len(y))})
    model, n_clusters = _fit_ols(y, x, maxlags, cluster_groups)
    return float(model.params["const"]), float(model.tvalues["const"]), n_clusters


# --------------------------------------------------------------------------
# Stat functions -- pure, Series/DataFrame in, StatResult out
# --------------------------------------------------------------------------

def _replace_inf_with_nan(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan)


def _clean_series(values) -> pd.Series:
    """dropna() alone doesn't catch inf/-inf -- scrub both, functionally."""
    return pipe(pd.Series(values), _replace_inf_with_nan, lambda s: s.dropna())


def stat_vs_zero(
    values: pd.Series, dates: Optional[pd.Series] = None, maxlags: Optional[int] = None,
    cluster_groups: Optional[pd.Series] = None, significance_t: Optional[float] = None,
) -> StatResult:
    clean = _clean_series(values)
    n = len(clean)
    if n < 2:
        return StatResult(mean=float(clean.mean()) if n else float("nan"), t_stat=float("nan"), n=n, significant=False)
    cg = (_require_complete(_align_to(pd.Series(values).index, cluster_groups).loc[clean.index])
          if cluster_groups is not None else None)
    mean, t_stat, n_clusters = _t_from_ols_const(clean.to_numpy(dtype=float), dates, maxlags, cg)
    return _with_significance(StatResult(mean=mean, t_stat=t_stat, n=n, significant=False, n_clusters=n_clusters,
                                         df=_dof(n, 1, n_clusters)), significance_t)


def stat_group_diff(
    values: pd.Series, group_bool: pd.Series, dates: Optional[pd.Series] = None,
    maxlags: Optional[int] = None, cluster_groups: Optional[pd.Series] = None,
    significance_t: Optional[float] = None,
) -> StatResult:
    df = pd.DataFrame({"value": values, "group": group_bool})
    if dates is not None:
        df["date"] = _align_to(df.index, dates, "dates")
    if cluster_groups is not None:
        df["_cluster"] = _align_to(df.index, cluster_groups)
    df = pipe(
        df.assign(value=lambda d: _replace_inf_with_nan(d["value"])),
        lambda d: d.dropna(subset=["value", "group"]),
    )
    n = len(df)
    n_true, n_false = int(df["group"].sum()), int((~df["group"].astype(bool)).sum())
    if n_true < 2 or n_false < 2:
        return StatResult(mean=float("nan"), t_stat=float("nan"), n=n, significant=False)
    y = df["value"].astype(float).to_numpy()
    x = sm.add_constant(df["group"].astype(float))
    cg = _require_complete(df["_cluster"]) if cluster_groups is not None else None
    model, n_clusters = _fit_ols(y, x, maxlags, cg)
    coef, t_stat = float(model.params["group"]), float(model.tvalues["group"])
    return _with_significance(StatResult(mean=coef, t_stat=t_stat, n=n, significant=False, n_clusters=n_clusters,
                                         df=_dof(n, 2, n_clusters)), significance_t)


# --------------------------------------------------------------------------
# classify_overfitting -- declarative rule table replacing the if/elif ladder
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Ctx:
    train: StatResult
    test: StatResult
    significance_t: Optional[float]
    robust_retention: float
    moderate_retention: float
    overfit_retention: float
    retention: Optional[float]
    sign_flipped: bool
    train_significant: bool
    test_significant: bool
    train_crit: float
    test_crit: float


def _rule_no_insample_edge(ctx: _Ctx) -> Optional[Verdict]:
    if ctx.train_significant:
        return None
    return Verdict(
        "INSUFFICIENT_INSAMPLE_EDGE",
        f"train |t|={abs(ctx.train.t_stat):.2f} < {ctx.train_crit:.2f} -- no real in-sample edge to test "
        "for overfitting in the first place.",
        None, False,
    )


# NOTE (preserved from review of the original, not fixed here to keep this
# a pure restructuring): this fires on ANY sign disagreement, including a
# train mean of +0.0001 vs. a test mean of -0.0001. If you want a magnitude
# floor, it's a one-line change: add `abs(ctx.train.mean) > eps` to the
# guard below -- the rule-table structure means that edit is isolated to
# this one function and doesn't touch anything else in the pipeline.
def _rule_test_uncomputable(ctx: _Ctx) -> Optional[Verdict]:
    if np.isfinite(ctx.test.mean) and not np.isnan(ctx.test.t_stat):
        return None
    return Verdict(
        "INSUFFICIENT_DATA",
        f"test statistic could not be computed (n={ctx.test.n}, mean={ctx.test.mean}, t={ctx.test.t_stat}) -- "
        "nothing to compare the in-sample edge against.",
        None, False,
    )


def _rule_sign_flip(ctx: _Ctx) -> Optional[Verdict]:
    if not ctx.sign_flipped:
        return None
    return Verdict(
        "OVERFITTED",
        f"sign REVERSED out-of-sample (train mean={ctx.train.mean:+.4f}, test mean={ctx.test.mean:+.4f}) -- "
        "the strongest available overfitting signal; the apparent in-sample edge was itself noise.",
        ctx.retention, True,
    )


def _rule_test_not_significant(ctx: _Ctx) -> Optional[Verdict]:
    if ctx.test_significant:
        return None
    if ctx.retention is not None and ctx.retention < ctx.overfit_retention:
        return Verdict(
            "OVERFITTED",
            f"test |t|={abs(ctx.test.t_stat):.2f} < {ctx.test_crit:.2f} (not distinguishable from zero) AND "
            f"retention={ctx.retention:.0%} < {ctx.overfit_retention:.0%} -- edge collapsed in magnitude AND "
            "significance.",
            ctx.retention, False,
        )
    retention_str = f"{ctx.retention:.0%}" if ctx.retention is not None else "n/a"
    return Verdict(
        "WEAK",
        f"test |t|={abs(ctx.test.t_stat):.2f} < {ctx.test_crit:.2f} (not distinguishable from zero) even "
        f"though some magnitude survived (retention={retention_str}) "
        "-- edge did not survive out-of-sample.",
        ctx.retention, False,
    )


def _rule_retention_undefined(ctx: _Ctx) -> Optional[Verdict]:
    if ctx.retention is not None:
        return None
    return Verdict(
        "WEAK",
        "train mean was zero -- retention ratio undefined; test is significant but this is an "
        "edge case worth manual review.",
        None, False,
    )


def _rule_robust(ctx: _Ctx) -> Optional[Verdict]:
    if ctx.retention < ctx.robust_retention:
        return None
    return Verdict(
        "ROBUST",
        f"test retains {ctx.retention:.0%} of train's magnitude and stays significant "
        f"(|t|={abs(ctx.test.t_stat):.2f}) -- edge holds up out-of-sample.",
        ctx.retention, False,
    )


def _rule_moderate(ctx: _Ctx) -> Optional[Verdict]:
    if ctx.retention < ctx.moderate_retention:
        return None
    return Verdict(
        "MODERATE",
        f"test retains {ctx.retention:.0%} of train's magnitude, still significant "
        f"(|t|={abs(ctx.test.t_stat):.2f}) -- partial decay, worth continued monitoring.",
        ctx.retention, False,
    )


def _rule_weak_default(ctx: _Ctx) -> Optional[Verdict]:
    """Terminal rule: always matches. Table must end with an unconditional
    rule the same way an if/elif ladder must end with a bare `else`."""
    return Verdict(
        "WEAK",
        f"test retains only {ctx.retention:.0%} of train's magnitude, though narrowly still significant "
        f"(|t|={abs(ctx.test.t_stat):.2f}) -- most of the economic effect did not survive out-of-sample.",
        ctx.retention, False,
    )


_RULES: tuple[Callable[[_Ctx], Optional[Verdict]], ...] = (
    _rule_no_insample_edge,
    _rule_test_uncomputable,
    _rule_sign_flip,
    _rule_test_not_significant,
    _rule_retention_undefined,
    _rule_robust,
    _rule_moderate,
    _rule_weak_default,
)


def classify_overfitting(
    train: StatResult,
    test: StatResult,
    significance_t: Optional[float] = None,
    robust_retention: float = DEFAULT_ROBUST_RETENTION,
    moderate_retention: float = DEFAULT_MODERATE_RETENTION,
    overfit_retention: float = DEFAULT_OVERFIT_RETENTION,
) -> tuple[str, str, Optional[float], bool]:
    """Pure decision logic, same signature/return shape as the original.
    Internally: build an immutable _Ctx, then walk _RULES in order and
    take the first non-None Verdict -- same semantics as the original
    if/elif chain, expressed as data (a tuple of rule functions) instead
    of control flow. significance_t is authoritative: significance is
    recomputed from each StatResult's t_stat against critical_t() (the
    small-sample Student-t value when significance_t is None), not read
    from .significant."""
    sign_flipped = (train.mean > 0 > test.mean) or (train.mean < 0 < test.mean)
    retention = (test.mean / train.mean) if train.mean != 0 else None
    ctx = _Ctx(train, test, significance_t, robust_retention, moderate_retention, overfit_retention,
               retention, sign_flipped,
               _is_significant(train, significance_t), _is_significant(test, significance_t),
               critical_t(train, significance_t), critical_t(test, significance_t))
    verdict = next(v for rule in _RULES if (v := rule(ctx)) is not None)
    return verdict.label, verdict.reasoning, verdict.retention_ratio, verdict.sign_flipped


# --------------------------------------------------------------------------
# Top-level orchestrator -- the impure shell around the pure core above
# --------------------------------------------------------------------------

def _append_cluster_caveat(reasoning: str, train_stat: StatResult, test_stat: StatResult) -> str:
    """Pure: takes a reasoning string, returns a new one. No += mutation."""
    unreliable = [
        (name, s.n_clusters) for name, s in (("train", train_stat), ("test", test_stat))
        if s.n_clusters is not None and s.n_clusters < MIN_RELIABLE_CLUSTERS
    ]
    if not unreliable:
        return reasoning
    detail = ", ".join(f"{name}={nc} clusters" for name, nc in unreliable)
    return reasoning + (
        f" CAVEAT: cluster-robust SE used with fewer than {MIN_RELIABLE_CLUSTERS} distinct "
        f"clusters ({detail}) -- the sandwich estimator is not asymptotically reliable at this "
        "count, so this verdict should be treated as inconclusive, not trusted at face value."
    )


def _append_purge_embargo_note(reasoning: str, n_purged_train: int, n_embargoed_test: int) -> str:
    """Pure: takes a reasoning string, returns a new one. No += mutation."""
    if not (n_purged_train or n_embargoed_test):
        return reasoning
    return reasoning + (f" ({n_purged_train} train row(s) purged, {n_embargoed_test} test row(s) embargoed "
                         "-- see purge_days/embargo_days.)")


def walk_forward_validate(
    df: pd.DataFrame,
    date_col: str,
    stat_fn: Callable[[pd.DataFrame], StatResult],
    train_frac: float = DEFAULT_TRAIN_FRAC,
    min_n_per_split: int = DEFAULT_MIN_N_PER_SPLIT,
    significance_t: Optional[float] = None,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> WalkForwardResult:
    train_df, test_df = chronological_split(df, date_col, train_frac)
    n_purged_train, n_embargoed_test = 0, 0
    if purge_days > 0 or embargo_days > 0:
        train_df, test_df, n_purged_train, n_embargoed_test = apply_purge_embargo(
            train_df, test_df, date_col, purge_days=purge_days, embargo_days=embargo_days)

    if len(train_df) < min_n_per_split or len(test_df) < min_n_per_split:
        return WalkForwardResult(
            train=StatResult(mean=float("nan"), t_stat=float("nan"), n=len(train_df), significant=False),
            test=StatResult(mean=float("nan"), t_stat=float("nan"), n=len(test_df), significant=False),
            retention_ratio=None, sign_flipped=False, verdict="INSUFFICIENT_DATA",
            reasoning=_append_purge_embargo_note(
                f"train n={len(train_df)}, test n={len(test_df)} -- need >= {min_n_per_split} per split "
                "for either stat to be meaningful.", n_purged_train, n_embargoed_test),
            n_purged_train=n_purged_train, n_embargoed_test=n_embargoed_test,
        )

    train_stat, test_stat = stat_fn(train_df), stat_fn(test_df)
    verdict, reasoning, retention, sign_flipped = classify_overfitting(
        train_stat, test_stat, significance_t=significance_t)
    reasoning = _append_purge_embargo_note(reasoning, n_purged_train, n_embargoed_test)
    reasoning = _append_cluster_caveat(reasoning, train_stat, test_stat)

    return WalkForwardResult(train=train_stat, test=test_stat, retention_ratio=retention,
                              n_purged_train=n_purged_train, n_embargoed_test=n_embargoed_test,
                              sign_flipped=sign_flipped, verdict=verdict, reasoning=reasoning)
