"""
Shared walk-forward robustness/overfitting check.
Native to this repo: no external MCP/service dependency, pure
pandas/numpy/statsmodels on whatever DataFrame a backtest script already
builds.

CONCEPT: split a backtest's dated observations chronologically into a
train window (default first 70%) and a held-out test window (last 30%),
recompute the SAME summary statistic on each half, and classify how much
the apparent edge degrades out-of-sample. A signal that only "worked" in
the training slice and evaporates or reverses in the test slice was
fit/discovered on noise specific to that period, not a real, stable
effect -- exactly the failure mode a single whole-history backtest (which
is how every backtest in this repo up to now has been reported) cannot
by itself distinguish from a genuine edge.

WHY A SINGLE CHRONOLOGICAL SPLIT, NOT ROLLING/EXPANDING WINDOWS: most of
this repo's backtests have modest sample sizes (tens to low hundreds of
observations) -- a rolling-window scheme would slice that down further
per fold and make each fold's own t-stat too noisy to interpret. A single
fit/holdout split is also the exact methodology standard scoring
engine validation settled on for the same reason -- borrowing
a precedent that was itself chosen after testing IC-weighted vs. rolling
alternatives and finding the simple split more reliable on a similarly-
sized panel.

TWO STAT SHAPES SUPPORTED, matching what backtests in this repo actually
report:
- stat_vs_zero(): one-sample test -- "is the mean of this return series
  distinguishable from zero", the shape backtest_mtf_buildup.py and
  pairs_trading_screen.py's per-trade returns both use.
- stat_group_diff(): two-sample test -- "is the mean of group A different
  from group B", the shape backtest_crude_oil_not_spiking.py's
  not_spiking-vs-spiking comparison uses.
Both support an optional HAC (Newey-West) correction for overlapping
windows, via the same statsmodels cov_type="HAC" pattern already
established in backtest_crude_oil_not_spiking.py and
backtest_momentum_screener_rs.py (maxlags=horizon-1 convention) -- reused
here, not reinvented, and callers pass dates/maxlags explicitly rather
than this module guessing an overlap structure it can't know.

ALSO (added 2026-08-23): an optional CLUSTER-ROBUST correction via
cluster_groups, for a DIFFERENT overlap shape than HAC handles. HAC/
Newey-West corrects serial correlation WITHIN one overlapping time series
(the crude-oil/momentum-screener case). It does NOT correct for many
DIFFERENT entities' events landing on the same or nearby calendar dates
and therefore sharing one market-regime shock in their forward returns --
a cross-sectional panel-clustering problem (Petersen 2009), not a
time-series autocorrelation one.

Pass cluster_groups (e.g. the exact available_from
date, or a coarser reporting-year label) instead of dates/maxlags when
the overlap is cross-sectional like this; the two corrections are NOT
combined (statsmodels supports two-way cluster+HAC but that's overkill
for what this repo's backtests need, and untested here) -- pass one or
the other. cluster-robust SEs are only asymptotically valid with enough
DISTINCT clusters (~20-30+ is the common rule of thumb); StatResult.
n_clusters is populated whenever cluster_groups is used so a caller/
walk-forward split with too few clusters (e.g. a 30%-test window landing
on only 3 fiscal years) can be flagged as unreliable rather than trusted
at face value in either direction.

VERDICT THRESHOLDS -- a reasoned STARTING POINT, not back-tested against
a labeled corpus of known-overfit vs. known-robust strategies. Reasoning:
- significance_t=2.0 is NOT a new choice.
  Reused for consistency, not picked fresh.
- If the TRAIN window itself never reached significance, there was no
  real in-sample edge to test for overfitting -- verdict is
  INSUFFICIENT_INSAMPLE_EDGE, not a robustness grade.
- A SIGN FLIP (train edge positive, test edge negative or vice versa) is
  the single strongest overfitting signal available and short-circuits
  straight to OVERFITTED regardless of the retention ratio -- a reversed
  edge is not "a weaker version of the same effect", it's evidence the
  original in-sample sign was itself noise.
- Otherwise, retention_ratio = test_mean / train_mean (same sign by this
  point) drives the grade: >=0.7 ROBUST, 0.4-0.7 MODERATE, <0.4 WEAK --
  and if the test window additionally fails to reach significance at all,
  that's evidence of decay strong enough to downgrade a >=0.3 retention
  to WEAK and a <0.3 retention all the way to OVERFITTED, since losing
  statistical distinguishability from zero is a harder failure than
  merely losing magnitude while remaining significant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_SIGNIFICANCE_T = 2.0
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_MIN_N_PER_SPLIT = 8
DEFAULT_ROBUST_RETENTION = 0.7
DEFAULT_MODERATE_RETENTION = 0.4
DEFAULT_OVERFIT_RETENTION = 0.3  # below this AND non-significant -> OVERFITTED, not just WEAK
MIN_RELIABLE_CLUSTERS = 20  # common rule-of-thumb floor for cluster-robust SE asymptotics; below this, treat n_clusters as informational, not a guarantee


@dataclass
class StatResult:
    mean: float
    t_stat: float
    n: int
    significant: bool
    n_clusters: int | None = None  # populated only when cluster_groups was used


@dataclass
class WalkForwardResult:
    train: StatResult
    test: StatResult
    retention_ratio: float | None  # None when train.mean == 0 (division undefined) or insufficient data
    sign_flipped: bool
    verdict: str
    reasoning: str


def chronological_split(df: pd.DataFrame, date_col: str, train_frac: float = DEFAULT_TRAIN_FRAC
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sorts by date_col and splits by ROW COUNT (not by calendar span) --
    row-count splitting keeps both halves statistically comparable in
    sample size even when observations aren't evenly spaced in time
    (e.g. crude-oil's monthly panel vs. a backtest with clustered event
    dates); a calendar-span split could leave one half with almost no
    observations if events cluster in time, which is common here.

    SNAPS the split point to the nearest date-value boundary if the naive
    row-count index would land strictly inside a run of TIED dates --
    the exact same reporting-date shock split
    across the train/test boundary -- undermining the "test is a genuinely
    held-out later period" assumption walk-forward validation depends on,
    on top of (not fixed by) stat_vs_zero/stat_group_diff's cluster_groups
    correction, which only fixes the WITHIN-split i.i.d. assumption, not
    this boundary leak. Snaps to whichever edge of the tied-date run is
    closer to the naive split_idx, to stay as close to train_frac as
    possible while keeping a whole cohort on one side."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(ordered) * train_frac)
    if 0 < split_idx < len(ordered):
        boundary_date = ordered[date_col].iloc[split_idx]
        same_date_mask = (ordered[date_col] == boundary_date).to_numpy()
        run_positions = np.flatnonzero(same_date_mask)
        if len(run_positions) and run_positions[0] < split_idx < run_positions[-1] + 1:
            run_start, run_end = int(run_positions[0]), int(run_positions[-1]) + 1
            split_idx = run_start if (split_idx - run_start) <= (run_end - split_idx) else run_end
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _fit_ols(y: np.ndarray, x: pd.DataFrame, maxlags: int | None, cluster_groups: pd.Series | None):
    """Shared fit dispatch: cluster-robust (if cluster_groups given) takes
    priority over HAC (if maxlags given) -- the two aren't combined here
    (see module docstring). Falls back to a plain OLS fit when neither is
    given, or when cluster_groups has fewer than 2 distinct groups (a
    single cluster makes cluster-robust SEs undefined; naive is the
    honest fallback, not a crash)."""
    n_clusters = None
    if cluster_groups is not None:
        n_clusters = pd.Series(cluster_groups).nunique()
        if n_clusters >= 2:
            model = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(cluster_groups)})
            return model, n_clusters
    if maxlags is not None:
        return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}), n_clusters
    return sm.OLS(y, x).fit(), n_clusters


def _t_from_ols_const(y: np.ndarray, dates: pd.Series | None, maxlags: int | None,
                       cluster_groups: pd.Series | None = None) -> tuple[float, float, int | None]:
    """OLS on a constant-only regressor, optionally HAC- or cluster-robust
    -corrected -- same pattern as backtest_crude_oil_not_spiking.py's
    test_horizon_hac() and the fix applied to backtest_momentum_screener_
    rs.py (must reference the column by its auto-assigned name "const",
    not a positional index, or statsmodels raises/mis-indexes). `dates`
    is accepted for signature symmetry with callers that pass it
    alongside maxlags; the fit itself only needs y's own ordering."""
    x = pd.DataFrame({"const": np.ones(len(y))})
    model, n_clusters = _fit_ols(y, x, maxlags, cluster_groups)
    return float(model.params["const"]), float(model.tvalues["const"]), n_clusters


def stat_vs_zero(values: pd.Series, dates: pd.Series | None = None, maxlags: int | None = None,
                  cluster_groups: pd.Series | None = None,
                  significance_t: float = DEFAULT_SIGNIFICANCE_T) -> StatResult:
    """One-sample test: is mean(values) distinguishable from zero.
    Pass dates+maxlags for a HAC (Newey-West) correction when values come
    from ONE overlapping time series (e.g. monthly forward-return rows
    sharing months of the same underlying path). Pass cluster_groups
    instead when the overlap is cross-sectional -- many different
    entities' events sharing a calendar-date/regime shock (e.g. one
    reporting-season cohort across hundreds of stocks); see module
    docstring for why these are different corrections. Omit all three for
    a plain t-test when observations are already independent (e.g.
    non-overlapping per-trade returns)."""
    # REAL BUG FIXED 2026-08-29 (flagged by an external review, verified
    # live before trusting it): pandas dropna() does NOT remove inf/-inf.
    # An upstream ratio computation dividing by zero produces inf, which
    # then propagated silently into mean=inf/t_stat=nan (confirmed live --
    # not even a clean crash, just a wrong-but-plausible-looking result)
    # instead of being excluded like a genuinely missing observation.
    # fama_macbeth.py already guards every regression input this way;
    # this module was missing the same armor.
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n < 2:
        return StatResult(mean=float(clean.mean()) if n else float("nan"), t_stat=float("nan"), n=n, significant=False)
    cg = pd.Series(cluster_groups).loc[clean.index] if cluster_groups is not None else None
    mean, t_stat, n_clusters = _t_from_ols_const(clean.to_numpy(dtype=float), dates, maxlags, cg)
    return StatResult(mean=mean, t_stat=t_stat, n=n, significant=abs(t_stat) >= significance_t, n_clusters=n_clusters)


def stat_group_diff(values: pd.Series, group_bool: pd.Series, dates: pd.Series | None = None,
                     maxlags: int | None = None, cluster_groups: pd.Series | None = None,
                     significance_t: float = DEFAULT_SIGNIFICANCE_T) -> StatResult:
    """Two-sample test: is mean(values[group_bool]) - mean(values[~group_bool])
    distinguishable from zero. Same HAC-dummy-regression shape as
    backtest_crude_oil_not_spiking.py's test_horizon_hac(); see
    stat_vs_zero()'s docstring for when to use cluster_groups instead of
    dates/maxlags."""
    df = pd.DataFrame({"value": values, "group": group_bool})
    if dates is not None:
        df["date"] = dates
    if cluster_groups is not None:
        df["_cluster"] = pd.Series(cluster_groups)
    # See stat_vs_zero()'s comment: dropna() alone doesn't catch inf/-inf.
    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["value", "group"])
    n = len(df)
    n_true, n_false = int(df["group"].sum()), int((~df["group"].astype(bool)).sum())
    if n_true < 2 or n_false < 2:
        return StatResult(mean=float("nan"), t_stat=float("nan"), n=n, significant=False)
    y = df["value"].astype(float).to_numpy()
    x = sm.add_constant(df["group"].astype(float))
    cg = df["_cluster"] if cluster_groups is not None else None
    model, n_clusters = _fit_ols(y, x, maxlags, cg)
    coef = float(model.params["group"])
    t_stat = float(model.tvalues["group"])
    return StatResult(mean=coef, t_stat=t_stat, n=n, significant=abs(t_stat) >= significance_t, n_clusters=n_clusters)


def classify_overfitting(
    train: StatResult,
    test: StatResult,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
    robust_retention: float = DEFAULT_ROBUST_RETENTION,
    moderate_retention: float = DEFAULT_MODERATE_RETENTION,
    overfit_retention: float = DEFAULT_OVERFIT_RETENTION,
) -> tuple[str, str, float | None, bool]:
    """Pure decision logic (see module docstring for the reasoning behind
    each threshold) -- returns (verdict, reasoning, retention_ratio,
    sign_flipped). Deliberately separated from walk_forward_validate() so
    it can be unit-tested directly against synthetic StatResult pairs
    without building a DataFrame or calling statsmodels."""
    if not train.significant:
        return ("INSUFFICIENT_INSAMPLE_EDGE",
                f"train |t|={abs(train.t_stat):.2f} < {significance_t:.1f} -- no real in-sample edge to test "
                "for overfitting in the first place.",
                None, False)

    sign_flipped = (train.mean > 0 > test.mean) or (train.mean < 0 < test.mean)
    if sign_flipped:
        return ("OVERFITTED",
                f"sign REVERSED out-of-sample (train mean={train.mean:+.4f}, test mean={test.mean:+.4f}) -- "
                "the strongest available overfitting signal; the apparent in-sample edge was itself noise.",
                (test.mean / train.mean) if train.mean != 0 else None, True)

    retention = (test.mean / train.mean) if train.mean != 0 else None

    if not test.significant:
        if retention is not None and retention < overfit_retention:
            return ("OVERFITTED",
                    f"test |t|={abs(test.t_stat):.2f} < {significance_t:.1f} (not distinguishable from zero) AND "
                    f"retention={retention:.0%} < {overfit_retention:.0%} -- edge collapsed in magnitude AND "
                    "significance.", retention, False)
        retention_str = f"{retention:.0%}" if retention is not None else "n/a"
        return ("WEAK",
                f"test |t|={abs(test.t_stat):.2f} < {significance_t:.1f} (not distinguishable from zero) even "
                f"though some magnitude survived (retention={retention_str}) "
                "-- edge did not survive out-of-sample.", retention, False)

    if retention is None:
        return ("WEAK", "train mean was zero -- retention ratio undefined; test is significant but this is an "
                         "edge case worth manual review.", None, False)
    if retention >= robust_retention:
        return ("ROBUST", f"test retains {retention:.0%} of train's magnitude and stays significant "
                           f"(|t|={abs(test.t_stat):.2f}) -- edge holds up out-of-sample.", retention, False)
    if retention >= moderate_retention:
        return ("MODERATE", f"test retains {retention:.0%} of train's magnitude, still significant "
                             f"(|t|={abs(test.t_stat):.2f}) -- partial decay, worth continued monitoring.",
                 retention, False)
    return ("WEAK", f"test retains only {retention:.0%} of train's magnitude, though narrowly still significant "
                     f"(|t|={abs(test.t_stat):.2f}) -- most of the economic effect did not survive out-of-sample.",
             retention, False)


def walk_forward_validate(
    df: pd.DataFrame,
    date_col: str,
    stat_fn,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    min_n_per_split: int = DEFAULT_MIN_N_PER_SPLIT,
    significance_t: float = DEFAULT_SIGNIFICANCE_T,
) -> WalkForwardResult:
    """Top-level entry point. stat_fn: Callable[[pd.DataFrame], StatResult]
    -- callers build this as e.g. `lambda d: stat_vs_zero(d["excess_pct"])`
    or `lambda d: stat_group_diff(d["fwd_ret"], d["not_spiking"])`, so this
    module never needs to know a caller's column names."""
    train_df, test_df = chronological_split(df, date_col, train_frac)
    if len(train_df) < min_n_per_split or len(test_df) < min_n_per_split:
        return WalkForwardResult(
            train=StatResult(mean=float("nan"), t_stat=float("nan"), n=len(train_df), significant=False),
            test=StatResult(mean=float("nan"), t_stat=float("nan"), n=len(test_df), significant=False),
            retention_ratio=None, sign_flipped=False, verdict="INSUFFICIENT_DATA",
            reasoning=f"train n={len(train_df)}, test n={len(test_df)} -- need >= {min_n_per_split} per split "
                      "for either stat to be meaningful.",
        )

    train_stat = stat_fn(train_df)
    test_stat = stat_fn(test_df)
    verdict, reasoning, retention, sign_flipped = classify_overfitting(
        train_stat, test_stat, significance_t=significance_t)

    # A cluster-robust StatResult with too few distinct clusters in either
    # split is NOT a reliable verdict in either direction (see module
    # docstring) -- append a caveat rather than silently trusting a low-
    # cluster-count clustered t-stat the same as a high-cluster-count one.
    unreliable = [(name, s.n_clusters) for name, s in (("train", train_stat), ("test", test_stat))
                  if s.n_clusters is not None and s.n_clusters < MIN_RELIABLE_CLUSTERS]
    if unreliable:
        detail = ", ".join(f"{name}={nc} clusters" for name, nc in unreliable)
        reasoning += (f" CAVEAT: cluster-robust SE used with fewer than {MIN_RELIABLE_CLUSTERS} distinct "
                      f"clusters ({detail}) -- the sandwich estimator is not asymptotically reliable at this "
                      "count, so this verdict should be treated as inconclusive, not trusted at face value.")

    return WalkForwardResult(train=train_stat, test=test_stat, retention_ratio=retention,
                              sign_flipped=sign_flipped, verdict=verdict, reasoning=reasoning)
