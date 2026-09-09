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

from dataclasses import dataclass
from functools import reduce
from typing import Callable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_SIGNIFICANCE_T = 2.0
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


@dataclass(frozen=True, slots=True)
class Verdict:
    """Internal: one rule's output. Same four fields classify_overfitting
    has always returned as a bare tuple -- named here so each rule
    function is self-documenting."""
    label: str
    reasoning: str
    retention_ratio: Optional[float]
    sign_flipped: bool


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    train: StatResult
    test: StatResult
    retention_ratio: Optional[float]
    sign_flipped: bool
    verdict: str
    reasoning: str


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


def chronological_split(
    df: pd.DataFrame, date_col: str, train_frac: float = DEFAULT_TRAIN_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_idx = _snap_to_boundary(ordered[date_col], int(len(ordered) * train_frac))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


# --------------------------------------------------------------------------
# OLS fit dispatch -- lazy predicate/strategy table instead of if/elif
# --------------------------------------------------------------------------

def _fit_cluster(y: np.ndarray, x: pd.DataFrame, groups) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})


def _fit_hac(y: np.ndarray, x: pd.DataFrame, maxlags: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def _fit_plain(y: np.ndarray, x: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, x).fit()


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
    cluster_groups: Optional[pd.Series] = None, significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> StatResult:
    clean = _clean_series(values)
    n = len(clean)
    if n < 2:
        return StatResult(mean=float(clean.mean()) if n else float("nan"), t_stat=float("nan"), n=n, significant=False)
    cg = pd.Series(cluster_groups).loc[clean.index] if cluster_groups is not None else None
    mean, t_stat, n_clusters = _t_from_ols_const(clean.to_numpy(dtype=float), dates, maxlags, cg)
    return StatResult(mean=mean, t_stat=t_stat, n=n, significant=abs(t_stat) >= significance_t, n_clusters=n_clusters)


def stat_group_diff(
    values: pd.Series, group_bool: pd.Series, dates: Optional[pd.Series] = None,
    maxlags: Optional[int] = None, cluster_groups: Optional[pd.Series] = None,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> StatResult:
    df = pd.DataFrame({"value": values, "group": group_bool})
    if dates is not None:
        df["date"] = dates
    if cluster_groups is not None:
        df["_cluster"] = pd.Series(cluster_groups)
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
    cg = df["_cluster"] if cluster_groups is not None else None
    model, n_clusters = _fit_ols(y, x, maxlags, cg)
    coef, t_stat = float(model.params["group"]), float(model.tvalues["group"])
    return StatResult(mean=coef, t_stat=t_stat, n=n, significant=abs(t_stat) >= significance_t, n_clusters=n_clusters)


# --------------------------------------------------------------------------
# classify_overfitting -- declarative rule table replacing the if/elif ladder
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Ctx:
    train: StatResult
    test: StatResult
    significance_t: float
    robust_retention: float
    moderate_retention: float
    overfit_retention: float
    retention: Optional[float]
    sign_flipped: bool


def _rule_no_insample_edge(ctx: _Ctx) -> Optional[Verdict]:
    if ctx.train.significant:
        return None
    return Verdict(
        "INSUFFICIENT_INSAMPLE_EDGE",
        f"train |t|={abs(ctx.train.t_stat):.2f} < {ctx.significance_t:.1f} -- no real in-sample edge to test "
        "for overfitting in the first place.",
        None, False,
    )


# NOTE (preserved from review of the original, not fixed here to keep this
# a pure restructuring): this fires on ANY sign disagreement, including a
# train mean of +0.0001 vs. a test mean of -0.0001. If you want a magnitude
# floor, it's a one-line change: add `abs(ctx.train.mean) > eps` to the
# guard below -- the rule-table structure means that edit is isolated to
# this one function and doesn't touch anything else in the pipeline.
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
    if ctx.test.significant:
        return None
    if ctx.retention is not None and ctx.retention < ctx.overfit_retention:
        return Verdict(
            "OVERFITTED",
            f"test |t|={abs(ctx.test.t_stat):.2f} < {ctx.significance_t:.1f} (not distinguishable from zero) AND "
            f"retention={ctx.retention:.0%} < {ctx.overfit_retention:.0%} -- edge collapsed in magnitude AND "
            "significance.",
            ctx.retention, False,
        )
    retention_str = f"{ctx.retention:.0%}" if ctx.retention is not None else "n/a"
    return Verdict(
        "WEAK",
        f"test |t|={abs(ctx.test.t_stat):.2f} < {ctx.significance_t:.1f} (not distinguishable from zero) even "
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
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
    robust_retention: float = DEFAULT_ROBUST_RETENTION,
    moderate_retention: float = DEFAULT_MODERATE_RETENTION,
    overfit_retention: float = DEFAULT_OVERFIT_RETENTION,
) -> tuple[str, str, Optional[float], bool]:
    """Pure decision logic, same signature/return shape as the original.
    Internally: build an immutable _Ctx, then walk _RULES in order and
    take the first non-None Verdict -- same semantics as the original
    if/elif chain, expressed as data (a tuple of rule functions) instead
    of control flow."""
    sign_flipped = (train.mean > 0 > test.mean) or (train.mean < 0 < test.mean)
    retention = (test.mean / train.mean) if train.mean != 0 else None
    ctx = _Ctx(train, test, significance_t, robust_retention, moderate_retention, overfit_retention,
               retention, sign_flipped)
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


def walk_forward_validate(
    df: pd.DataFrame,
    date_col: str,
    stat_fn: Callable[[pd.DataFrame], StatResult],
    train_frac: float = DEFAULT_TRAIN_FRAC,
    min_n_per_split: int = DEFAULT_MIN_N_PER_SPLIT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> WalkForwardResult:
    train_df, test_df = chronological_split(df, date_col, train_frac)

    if len(train_df) < min_n_per_split or len(test_df) < min_n_per_split:
        return WalkForwardResult(
            train=StatResult(mean=float("nan"), t_stat=float("nan"), n=len(train_df), significant=False),
            test=StatResult(mean=float("nan"), t_stat=float("nan"), n=len(test_df), significant=False),
            retention_ratio=None, sign_flipped=False, verdict="INSUFFICIENT_DATA",
            reasoning=f"train n={len(train_df)}, test n={len(test_df)} -- need >= {min_n_per_split} per split "
                      "for either stat to be meaningful.",
        )

    train_stat, test_stat = stat_fn(train_df), stat_fn(test_df)
    verdict, reasoning, retention, sign_flipped = classify_overfitting(
        train_stat, test_stat, significance_t=significance_t)
    reasoning = _append_cluster_caveat(reasoning, train_stat, test_stat)

    return WalkForwardResult(train=train_stat, test=test_stat, retention_ratio=retention,
                              sign_flipped=sign_flipped, verdict=verdict, reasoning=reasoning)
