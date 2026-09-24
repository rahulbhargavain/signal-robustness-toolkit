"""
cpcv_validator_fp.py -- Functional-programming rewrite of cpcv_validator.py,
matching walk_forward_validator_fp.py's own conventions (frozen/slotted
dataclasses, declarative rule tables via `next(...)`, pure helpers with no
in-place mutation).

DROP-IN COMPATIBLE: every public name, signature, and return type matches
cpcv_validator.py exactly. Behavior is unchanged -- this is a
restructuring, not a semantic change.

WHAT CHANGED, AND WHY IT'S "MORE FP" (same three moves walk_forward_
validator_fp.py already made, applied here):

1. Immutability: CPCVPathResult / CPCVResult are now frozen, slotted
   dataclasses -- a path's train/test StatResult, once computed, can't be
   silently overwritten later in a longer pipeline.

2. classify_cpcv_overall's if/elif ladder -> a declarative rule table,
   same "chain of responsibility" pattern classify_overfitting_fp uses:
   each branch is an independent pure function `_CpcvCtx -> _Overall |
   None`, evaluated top-down via `next(...)` until one matches.

3. No in-place mutation in the per-split evaluation loop: paths are built
   via a list comprehension over generate_cpcv_splits() rather than an
   accumulator list appended to in a for-loop body (the skip-counting
   still needs a running total, so that one loop stays -- turning it into
   a pure fold would obscure the skip/keep decision for no real
   readability gain, the same "don't force it" judgment call walk_
   forward_validator_fp.py's own docstring makes about not going full
   Railway-oriented).

REUSE, NOT REBUILD: every statistical primitive is imported unchanged
from walk_forward_validator_fp.py -- StatResult, classify_overfitting,
apply_purge_embargo. This module only adds the combinatorial splitting
machinery and the cross-path aggregation, exactly like the non-FP
original.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from walk_forward_validator_fp import (
    DEFAULT_MIN_N_PER_SPLIT,
    MIN_RELIABLE_CLUSTERS,
    StatResult,
    apply_purge_embargo,
    classify_overfitting,
    date_sort_key,
)

DEFAULT_N_GROUPS = 6
DEFAULT_N_TEST_GROUPS = 2
DEFAULT_PCT_ROBUST_FOR_OVERALL_ROBUST = 0.7
DEFAULT_PCT_ROBUST_FOR_OVERALL_MODERATE = 0.4


# --------------------------------------------------------------------------
# Immutable data types
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CPCVPathResult:
    test_groups: frozenset
    train: StatResult
    test: StatResult
    retention_ratio: Optional[float]
    sign_flipped: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class CPCVResult:
    n_groups: int
    n_test_groups: int
    paths: list = field(default_factory=list)  # list[CPCVPathResult]
    n_splits_skipped: int = 0
    pct_paths_robust_or_moderate: Optional[float] = None
    median_retention: Optional[float] = None
    overall_verdict: str = "INSUFFICIENT_DATA"
    reasoning: str = ""


# --------------------------------------------------------------------------
# Pure formatting functions
# --------------------------------------------------------------------------

def format_cpcv_print_line(result: CPCVResult) -> str:
    pct_str = (f"{result.pct_paths_robust_or_moderate * 100:.0f}%"
               if result.pct_paths_robust_or_moderate is not None else "n/a")
    return (f"    [CPCV check] {len(result.paths)} path(s) evaluated ({result.n_splits_skipped} skipped) "
            f"-> {result.overall_verdict} ({pct_str} of paths ROBUST/MODERATE)")


def format_cpcv_registry_note(result: CPCVResult) -> str:
    pct_str = (f"{result.pct_paths_robust_or_moderate * 100:.0f}%"
               if result.pct_paths_robust_or_moderate is not None else "n/a")
    n_total = len(result.paths) + result.n_splits_skipped
    return (f" CPCV ({result.n_groups} groups, C({result.n_groups},{result.n_test_groups})={n_total} paths): "
            f"{result.overall_verdict} ({pct_str} of {len(result.paths)} evaluated path(s) ROBUST/MODERATE).")


# --------------------------------------------------------------------------
# assign_groups -- pure, no reassignment of the cut list in place
# --------------------------------------------------------------------------

def _snap_cut(dates: np.ndarray, cut: int, n: int) -> int:
    """Pure: given one naive row-count cut, return the boundary-snapped
    index. Same rule as walk_forward_validator_fp._snap_to_boundary,
    applied to a raw ndarray of dates rather than a Series."""
    cut = max(1, min(n - 1, cut))
    boundary_date = dates[cut]
    run_start = cut
    while run_start > 0 and dates[run_start - 1] == boundary_date:
        run_start -= 1
    run_end = cut
    while run_end < n and dates[run_end] == boundary_date:
        run_end += 1
    return run_start if (cut - run_start) <= (run_end - cut) else run_end


def assign_groups(df: pd.DataFrame, date_col: str, n_groups: int) -> pd.Series:
    """Sorts by date_col and cuts into n_groups CONTIGUOUS, roughly
    equal-row-count chronological groups, snapping each cut to the
    nearest date-run boundary so no group splits one calendar date's
    cohort in two. Returns a group-id Series (0..n_groups-1) aligned to
    df's original index."""
    if df.empty or n_groups < 2:
        return pd.Series([], dtype=int)
    key = date_sort_key(df[date_col])
    pos = np.argsort(key.to_numpy(), kind="stable")
    dates = key.to_numpy()[pos]
    n = len(df)
    n_groups = min(n_groups, n)
    raw_cuts = [int(round(n * i / n_groups)) for i in range(1, n_groups)]
    snapped_cuts = sorted(set(c for c in (_snap_cut(dates, cut, n) for cut in raw_cuts) if 0 < c < n))

    group_id = np.zeros(n, dtype=int)
    for gid, (start, end) in enumerate(zip([0] + snapped_cuts, snapped_cuts + [n], strict=True)):
        group_id[start:end] = gid
    return pd.Series(group_id, index=df.index[pos])


def generate_cpcv_splits(n_groups: int, n_test_groups: int):
    """Pure generator: yields (train_groups, test_groups) as frozensets
    for every C(n_groups, n_test_groups) combination."""
    all_groups = frozenset(range(n_groups))
    for combo in combinations(range(n_groups), n_test_groups):
        test_groups = frozenset(combo)
        yield all_groups - test_groups, test_groups


def _contiguous_runs(sorted_group_ids: list[int]) -> list[list[int]]:
    """Pure: splits a sorted list of group-ids into maximal runs of
    consecutive integers, e.g. [0, 1, 3, 4, 5] -> [[0, 1], [3, 4, 5]]."""
    if not sorted_group_ids:
        return []
    runs = [[sorted_group_ids[0]]]
    for gid in sorted_group_ids[1:]:
        if gid == runs[-1][-1] + 1:
            runs[-1].append(gid)
        else:
            runs.append([gid])
    return runs


def _purge_after_run(train_df: pd.DataFrame, date_col: str, run_end, purge_days: int,
                      embargo_days: int = 0) -> tuple[pd.DataFrame, int, int]:
    """Train rows dated AFTER a held-out run, the mirror image of the
    pre-boundary purge. The last test row's forward-return window runs to
    run_end + purge_days, so train rows dated in (run_end, run_end +
    purge_days] overlap a test outcome and are PURGED; rows in the next
    embargo_days after that are EMBARGOED (Lopez de Prado's embargo: a
    buffer after the test set against serial correlation leaking test
    information into later train rows). Returns (kept, n_purged, n_embargoed)."""
    if (purge_days <= 0 and embargo_days <= 0) or train_df.empty:
        return train_df, 0, 0
    train_dates = pd.to_datetime(train_df[date_col])
    purge_end = run_end + pd.Timedelta(days=max(purge_days, 0))
    embargo_end = purge_end + pd.Timedelta(days=max(embargo_days, 0))
    purge_mask = ((train_dates > run_end) & (train_dates <= purge_end)).to_numpy()
    embargo_mask = ((train_dates > purge_end) & (train_dates <= embargo_end)).to_numpy()
    return train_df.loc[~(purge_mask | embargo_mask)], int(purge_mask.sum()), int(embargo_mask.sum())


def _run_drops(train_df: pd.DataFrame, train_dates: pd.Series, run_test_df: pd.DataFrame, date_col: str,
               purge_days: int, embargo_days: int) -> tuple[pd.Index, pd.Index, int, int]:
    """Pure: for one contiguous held-out run, returns (train index to drop,
    test index to drop, n_purged, n_embargoed). Only train rows dated
    BEFORE the run go through apply_purge_embargo() -- it assumes all train
    precedes test, and would otherwise purge every later train group."""
    run_dates = pd.to_datetime(run_test_df[date_col])
    run_start, run_end = run_dates.min(), run_dates.max()
    before = train_df.loc[train_dates.loc[train_df.index] < run_start]
    after = train_df.loc[train_dates.loc[train_df.index] > run_end]
    kept_before, kept_test, n_before, n_embargoed = apply_purge_embargo(
        before, run_test_df, date_col, purge_days=purge_days, embargo_days=embargo_days)
    kept_after, n_after, n_embargoed_after = _purge_after_run(after, date_col, run_end, purge_days, embargo_days)
    drop_train = before.index.difference(kept_before.index).union(after.index.difference(kept_after.index))
    return (drop_train, run_test_df.index.difference(kept_test.index), n_before + n_after,
            n_embargoed + n_embargoed_after)


def _sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Stable chronological row order, index labels kept."""
    return df.iloc[np.argsort(date_sort_key(df[date_col]).to_numpy(), kind="stable")]


def build_split_frames(df: pd.DataFrame, date_col: str, group_labels: pd.Series,
                        train_groups: frozenset, test_groups: frozenset,
                        purge_days: int = 0, embargo_days: int = 0
                        ) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Slices df into (train, test) for one CPCV combinatorial split, then
    applies purge/embargo at EVERY contiguous test-run boundary, leading
    (purge + test embargo via walk_forward_validator_fp.apply_purge_embargo())
    and trailing (_purge_after_run(): purge, then an embargo on the train
    rows after it). n_embargoed counts embargoed test AND train rows."""
    # Row order matters: a HAC (maxlags) stat_fn treats adjacent rows as
    # adjacent in time, so frames are handed over date-sorted whatever the
    # caller's input order.
    df = _sort_by_date(df, date_col)
    train_df = df.loc[group_labels.isin(train_groups)]
    test_df = df.loc[group_labels.isin(test_groups)]
    if test_df.empty or (purge_days <= 0 and embargo_days <= 0):
        return train_df, test_df, 0, 0

    train_dates = pd.to_datetime(train_df[date_col])
    test_group_of = group_labels.loc[test_df.index]
    run_frames = (test_df.loc[test_group_of.isin(run)] for run in _contiguous_runs(sorted(test_groups)))

    def step(acc, run_test_df):
        kept_train, drop_test, n_purged, n_embargoed = acc
        if run_test_df.empty:
            return acc
        d_train, d_test, p, e = _run_drops(kept_train, train_dates, run_test_df, date_col, purge_days, embargo_days)
        return kept_train.drop(index=d_train), drop_test.union(d_test), n_purged + p, n_embargoed + e

    kept_train, drop_test, n_purged, n_embargoed = reduce(step, run_frames, (train_df, pd.Index([]), 0, 0))
    return kept_train, test_df.drop(index=drop_test), n_purged, n_embargoed


def evaluate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, stat_fn) -> tuple[StatResult, StatResult]:
    return stat_fn(train_df), stat_fn(test_df)


# --------------------------------------------------------------------------
# classify_cpcv_overall -- declarative rule table replacing the if/elif ladder
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Overall:
    """Internal: one rule's output. Named so each rule function is
    self-documenting, same idea as walk_forward_validator_fp.Verdict."""
    label: str
    reasoning: str
    pct: Optional[float]


@dataclass(frozen=True, slots=True)
class _CpcvCtx:
    informative: tuple  # tuple[str, ...] of path verdict labels, INSUFFICIENT_* already excluded
    pct_robust_for_robust: float
    pct_robust_for_moderate: float


def _rule_no_informative_paths(ctx: _CpcvCtx) -> Optional[_Overall]:
    if ctx.informative:
        return None
    return _Overall("INSUFFICIENT_DATA",
                     "No path had sufficient in-sample edge or data to classify -- CPCV cannot say "
                     "anything about robustness here.", None)


def _pct_robust_or_moderate(informative: tuple) -> float:
    n_robust_or_moderate = sum(1 for v in informative if v in ("ROBUST", "MODERATE"))
    return n_robust_or_moderate / len(informative)


def _rule_overall_robust(ctx: _CpcvCtx) -> Optional[_Overall]:
    pct = _pct_robust_or_moderate(ctx.informative)
    if pct < ctx.pct_robust_for_robust:
        return None
    return _Overall("ROBUST", f"{pct:.0%} of {len(ctx.informative)} informative CPCV path(s) classified "
                               "ROBUST/MODERATE.", pct)


def _rule_overall_moderate(ctx: _CpcvCtx) -> Optional[_Overall]:
    pct = _pct_robust_or_moderate(ctx.informative)
    if pct < ctx.pct_robust_for_moderate:
        return None
    return _Overall("MODERATE", f"{pct:.0%} of {len(ctx.informative)} informative CPCV path(s) classified "
                                 "ROBUST/MODERATE -- partial agreement across paths.", pct)


def _rule_overall_overfitted(ctx: _CpcvCtx) -> Optional[_Overall]:
    pct = _pct_robust_or_moderate(ctx.informative)
    n_overfitted = sum(1 for v in ctx.informative if v == "OVERFITTED")
    if n_overfitted / len(ctx.informative) < 1 - ctx.pct_robust_for_moderate:
        return None
    return _Overall("OVERFITTED", f"only {pct:.0%} of {len(ctx.informative)} informative path(s) classified "
                                   "ROBUST/MODERATE, and most of the rest were OVERFITTED -- the apparent edge "
                                   "does not survive most held-out combinations.", pct)


def _rule_overall_weak_default(ctx: _CpcvCtx) -> Optional[_Overall]:
    """Terminal rule: always matches."""
    pct = _pct_robust_or_moderate(ctx.informative)
    return _Overall("WEAK", f"only {pct:.0%} of {len(ctx.informative)} informative CPCV path(s) classified "
                             "ROBUST/MODERATE -- the edge is inconsistent across held-out combinations.", pct)


_OVERALL_RULES: tuple = (
    _rule_no_informative_paths,
    _rule_overall_robust,
    _rule_overall_moderate,
    _rule_overall_overfitted,
    _rule_overall_weak_default,
)


def classify_cpcv_overall(path_verdicts: list[str],
                           pct_robust_for_robust: float = DEFAULT_PCT_ROBUST_FOR_OVERALL_ROBUST,
                           pct_robust_for_moderate: float = DEFAULT_PCT_ROBUST_FOR_OVERALL_MODERATE
                           ) -> tuple[str, str, Optional[float]]:
    """Pure decision logic, same signature/return shape as the original.
    A path whose OWN verdict is INSUFFICIENT_INSAMPLE_EDGE or
    INSUFFICIENT_DATA is excluded from the denominator entirely."""
    informative = tuple(v for v in path_verdicts if v not in ("INSUFFICIENT_INSAMPLE_EDGE", "INSUFFICIENT_DATA"))
    ctx = _CpcvCtx(informative, pct_robust_for_robust, pct_robust_for_moderate)
    result = next(v for rule in _OVERALL_RULES if (v := rule(ctx)) is not None)
    return result.label, result.reasoning, result.pct


# --------------------------------------------------------------------------
# Top-level orchestrator -- the impure shell around the pure core above
# --------------------------------------------------------------------------

def _evaluate_one_path(df: pd.DataFrame, date_col: str, group_labels: pd.Series, stat_fn,
                        train_groups: frozenset, test_groups: frozenset,
                        purge_days: int, embargo_days: int, min_n_per_split: int, significance_t: Optional[float]
                        ) -> Optional[CPCVPathResult]:
    """Pure: returns one path's result, or None if it was starved below
    min_n_per_split (the caller counts Nones as skipped)."""
    train_df, test_df, _, _ = build_split_frames(
        df, date_col, group_labels, train_groups, test_groups, purge_days=purge_days, embargo_days=embargo_days)
    if len(train_df) < min_n_per_split or len(test_df) < min_n_per_split:
        return None
    train_stat, test_stat = evaluate_split(train_df, test_df, stat_fn)
    verdict, _reasoning, retention, sign_flipped = classify_overfitting(
        train_stat, test_stat, significance_t=significance_t)
    return CPCVPathResult(test_groups=test_groups, train=train_stat, test=test_stat,
                           retention_ratio=retention, sign_flipped=sign_flipped, verdict=verdict)


def _low_cluster_caveat(paths: list) -> str:
    """Same warning walk_forward_validate() appends: cluster-robust SEs with
    fewer than MIN_RELIABLE_CLUSTERS distinct clusters aren't reliable, so
    say how many paths relied on them instead of letting the aggregate
    verdict look fully trustworthy."""
    counts = [s.n_clusters for p in paths for s in (p.train, p.test) if s.n_clusters is not None]
    low = [p for p in paths
           if any(s.n_clusters is not None and s.n_clusters < MIN_RELIABLE_CLUSTERS for s in (p.train, p.test))]
    if not low:
        return ""
    return (f" CAVEAT: {len(low)} of {len(paths)} evaluated path(s) used cluster-robust SEs with fewer than "
            f"{MIN_RELIABLE_CLUSTERS} distinct clusters (as few as {min(counts)}) -- the sandwich estimator is not "
            "asymptotically reliable at this count, so this verdict should be treated as inconclusive, not "
            "trusted at face value.")


def cpcv_validate(df: pd.DataFrame, date_col: str, stat_fn,
                   n_groups: int = DEFAULT_N_GROUPS, n_test_groups: int = DEFAULT_N_TEST_GROUPS,
                   purge_days: int = 0, embargo_days: int = 0,
                   min_n_per_split: int = DEFAULT_MIN_N_PER_SPLIT,
                   significance_t: Optional[float] = None) -> CPCVResult:
    """Top-level entry point. Same stat_fn contract as walk_forward_
    validate(): Callable[[pd.DataFrame], StatResult]. Evaluates every
    C(n_groups, n_test_groups) combinatorial split, skipping (not
    erroring on) any split that starves below min_n_per_split after
    purge/embargo."""
    if df.empty or n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        return CPCVResult(n_groups=n_groups, n_test_groups=n_test_groups,
                           reasoning="Empty input or invalid n_groups/n_test_groups -- nothing to evaluate.")

    group_labels = assign_groups(df, date_col, n_groups)
    actual_n_groups = int(group_labels.nunique()) if len(group_labels) else 0
    if actual_n_groups < 2 or n_test_groups >= actual_n_groups:
        return CPCVResult(n_groups=n_groups, n_test_groups=n_test_groups,
                           reasoning=f"Only {actual_n_groups} distinct group(s) could be formed from "
                                     f"{len(df)} row(s) -- too few for n_test_groups={n_test_groups}.")

    evaluated = [
        _evaluate_one_path(df, date_col, group_labels, stat_fn, train_groups, test_groups,
                            purge_days, embargo_days, min_n_per_split, significance_t)
        for train_groups, test_groups in generate_cpcv_splits(actual_n_groups, n_test_groups)
    ]
    paths = [p for p in evaluated if p is not None]
    n_skipped = len(evaluated) - len(paths)

    if not paths:
        return CPCVResult(n_groups=actual_n_groups, n_test_groups=n_test_groups, n_splits_skipped=n_skipped,
                           reasoning=f"All {n_skipped} combinatorial split(s) fell below min_n_per_split="
                                     f"{min_n_per_split} after purge/embargo -- no path could be evaluated.")

    overall_verdict, reasoning, pct = classify_cpcv_overall([p.verdict for p in paths])
    retentions = [p.retention_ratio for p in paths if p.retention_ratio is not None]
    median_retention = float(np.median(retentions)) if retentions else None

    return CPCVResult(n_groups=actual_n_groups, n_test_groups=n_test_groups, paths=paths,
                       n_splits_skipped=n_skipped, pct_paths_robust_or_moderate=pct,
                       median_retention=median_retention, overall_verdict=overall_verdict,
                       reasoning=reasoning + (f" ({n_skipped} split(s) skipped for insufficient data.)"
                                              if n_skipped else "") + _low_cluster_caveat(paths))
