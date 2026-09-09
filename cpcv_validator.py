"""
Full CPCV (Combinatorial Purged Cross-Validation), a stronger sibling to
walk_forward_validator.py's single chronological 70/30 split. Not a
rewrite of that module -- walk_forward_validate() keeps its exact
signature so existing callers depending on it are unaffected; this is a
separate, opt-in tool for a caller that wants the stronger check.

WHY: a single chronological split gives exactly one train/test draw --
its verdict can itself be an artifact of where the boundary happened to
land (walk_forward_validator.chronological_split()'s own docstring
already documents one such tied-date-boundary artifact). CPCV (Lopez de
Prado, Advances in Financial Machine Learning) partitions the timeline
into N contiguous groups and tests EVERY combination of k held-out
groups as a separate train/test split (purge/embargo applied at each
split's own boundary/boundaries), reporting a DISTRIBUTION of
out-of-sample verdicts across all C(N,k) paths instead of one.

REUSE, NOT REBUILD: every statistical primitive here is imported
unchanged from walk_forward_validator.py -- StatResult, stat_vs_zero,
stat_group_diff, classify_overfitting, apply_purge_embargo. This module
only adds the combinatorial splitting machinery (assign_groups,
generate_cpcv_splits, build_split_frames) and the cross-path aggregation
(cpcv_validate, classify_cpcv_overall). apply_purge_embargo() itself
handles a test-group set covering more than one contiguous run of
groups -- e.g. groups {1, 3} held out from a 6-group partition creates
TWO boundaries (before and after each contiguous run), not one, which
the original single-split caller never needed to consider.

FP-STYLE BY DESIGN, NOT BY FASHION: generate_cpcv_splits() is a pure
generator over itertools.combinations with no DataFrame dependency, and
every per-split evaluation (build_split_frames -> stat_fn -> classify_
overfitting) is independent of every other split -- no shared mutable
state across paths. This is the same pure-function-per-step pattern
walk_forward_validator.py already uses for its single split, extended
naturally to many splits via map/generator composition rather than a
stateful loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from walk_forward_validator import (
    DEFAULT_MIN_N_PER_SPLIT,
    DEFAULT_SIGNIFICANCE_T,
    StatResult,
    apply_purge_embargo,
    classify_overfitting,
)

DEFAULT_N_GROUPS = 6
DEFAULT_N_TEST_GROUPS = 2
# Fraction of individually-classified paths that must be ROBUST or
# MODERATE for the overall CPCV verdict to read ROBUST -- deliberately
# reuses walk_forward_validator's own ROBUST_RETENTION threshold value
# (0.7) for consistency rather than picking a fresh number; same
# "reasoned starting point, not back-tested against a labeled corpus"
# caveat that module's own thresholds carry.
DEFAULT_PCT_ROBUST_FOR_OVERALL_ROBUST = 0.7
DEFAULT_PCT_ROBUST_FOR_OVERALL_MODERATE = 0.4


@dataclass
class CPCVPathResult:
    test_groups: frozenset
    train: StatResult
    test: StatResult
    retention_ratio: float | None
    sign_flipped: bool
    verdict: str  # reuses classify_overfitting()'s labels, computed per-path


@dataclass
class CPCVResult:
    n_groups: int
    n_test_groups: int
    paths: list = field(default_factory=list)  # list[CPCVPathResult]
    n_splits_skipped: int = 0
    pct_paths_robust_or_moderate: float | None = None
    median_retention: float | None = None
    overall_verdict: str = "INSUFFICIENT_DATA"
    reasoning: str = ""


def format_cpcv_print_line(result: CPCVResult) -> str:
    """Pure formatting function: one CPCVResult -> a one-line "[CPCV
    check] ..." summary. Factor this out once you have more than one call
    site so the wording doesn't drift independently at each -- the same
    "one shared function, many callers" discipline dedup_store.
    append_dedup() and walk_forward_validate() itself already follow."""
    pct_str = (f"{result.pct_paths_robust_or_moderate * 100:.0f}%"
               if result.pct_paths_robust_or_moderate is not None else "n/a")
    return (f"    [CPCV check] {len(result.paths)} path(s) evaluated ({result.n_splits_skipped} skipped) "
            f"-> {result.overall_verdict} ({pct_str} of paths ROBUST/MODERATE)")


def format_cpcv_registry_note(result: CPCVResult) -> str:
    """Pure formatting function: one CPCVResult -> a note-field suffix
    suitable for appending to a persisted summary string. Companion to
    format_cpcv_print_line() -- same extraction rationale. Returns a
    string starting with a leading space, ready to append directly onto
    an existing note string (`note = base_note + format_cpcv_registry_note(result)`)."""
    pct_str = (f"{result.pct_paths_robust_or_moderate * 100:.0f}%"
               if result.pct_paths_robust_or_moderate is not None else "n/a")
    n_total = len(result.paths) + result.n_splits_skipped
    return (f" CPCV ({result.n_groups} groups, C({result.n_groups},{result.n_test_groups})={n_total} paths): "
            f"{result.overall_verdict} ({pct_str} of {len(result.paths)} evaluated path(s) ROBUST/MODERATE).")


def assign_groups(df: pd.DataFrame, date_col: str, n_groups: int) -> pd.Series:
    """Sorts by date_col and cuts into n_groups CONTIGUOUS, roughly
    equal-row-count chronological groups (row-count, not calendar-span --
    same reasoning as walk_forward_validator.chronological_split()'s own
    docstring: keeps groups statistically comparable in sample size even
    when observations aren't evenly spaced in time). Returns a group-id
    Series (0..n_groups-1) aligned to df's original index -- callers
    should not assume any particular row order back out of this.

    Uses the SAME tied-date-boundary-snap discipline as
    chronological_split(): a naive row-count cut can otherwise split one
    calendar date's cohort across two groups, which would let one
    reporting-date shock straddle a purge/embargo boundary here too.
    np.array_split's boundaries are snapped to the nearest date-run edge
    below."""
    if df.empty or n_groups < 2:
        return pd.Series([], dtype=int)
    ordered = df.sort_values(date_col)
    dates = pd.to_datetime(ordered[date_col]).to_numpy()
    n = len(ordered)
    n_groups = min(n_groups, n)  # can't make more groups than rows
    raw_cuts = [int(round(n * i / n_groups)) for i in range(1, n_groups)]

    snapped_cuts = []
    for cut in raw_cuts:
        cut = max(1, min(n - 1, cut))
        boundary_date = dates[cut]
        run_start = cut
        while run_start > 0 and dates[run_start - 1] == boundary_date:
            run_start -= 1
        run_end = cut
        while run_end < n and dates[run_end] == boundary_date:
            run_end += 1
        cut = run_start if (cut - run_start) <= (run_end - cut) else run_end
        snapped_cuts.append(cut)

    # De-dup and sort in case snapping collapsed two adjacent cuts onto
    # the same boundary (a very large tied-date run spanning multiple
    # intended cut points) -- degrades gracefully to fewer, larger groups
    # rather than producing an invalid/overlapping partition.
    snapped_cuts = sorted(set(c for c in snapped_cuts if 0 < c < n))

    group_id = np.zeros(n, dtype=int)
    for gid, (start, end) in enumerate(zip([0] + snapped_cuts, snapped_cuts + [n])):
        group_id[start:end] = gid
    return pd.Series(group_id, index=ordered.index)


def generate_cpcv_splits(n_groups: int, n_test_groups: int):
    """Pure generator, no DataFrame involved: yields (train_groups,
    test_groups) as frozensets of group-ids for every C(n_groups,
    n_test_groups) combination. train_groups is always the complement of
    test_groups within range(n_groups), so the two partition every group
    exactly once per split."""
    all_groups = frozenset(range(n_groups))
    for combo in combinations(range(n_groups), n_test_groups):
        test_groups = frozenset(combo)
        yield all_groups - test_groups, test_groups


def _contiguous_runs(sorted_group_ids: list[int]) -> list[list[int]]:
    """Splits a sorted list of group-ids into maximal runs of consecutive
    integers, e.g. [0, 1, 3, 4, 5] -> [[0, 1], [3, 4, 5]]. Each run is one
    contiguous block of held-out groups and therefore has its own
    train/test boundary (potentially two boundaries per run: before and
    after) for purge/embargo purposes."""
    if not sorted_group_ids:
        return []
    runs = [[sorted_group_ids[0]]]
    for gid in sorted_group_ids[1:]:
        if gid == runs[-1][-1] + 1:
            runs[-1].append(gid)
        else:
            runs.append([gid])
    return runs


def build_split_frames(df: pd.DataFrame, date_col: str, group_labels: pd.Series,
                        train_groups: frozenset, test_groups: frozenset,
                        purge_days: int = 0, embargo_days: int = 0
                        ) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Slices df into (train, test) for one CPCV combinatorial split, then
    applies purge/embargo at EVERY contiguous test-run boundary -- not
    just one, unlike the single chronological-split case. A non-
    contiguous test_groups set (e.g. {1, 3} out of 6) creates two
    separate contiguous runs ({1} and {3}), each with its own leading/
    trailing boundary against the surrounding train groups; a train row
    can be purged by ANY of those boundaries, and a test row can be
    embargoed at the START of its own run. Reuses walk_forward_validator.
    apply_purge_embargo() per run rather than reimplementing the purge/
    embargo arithmetic -- one source of truth for that rule."""
    train_df = df.loc[group_labels.isin(train_groups)]
    test_df = df.loc[group_labels.isin(test_groups)]
    if test_df.empty or (purge_days <= 0 and embargo_days <= 0):
        return train_df, test_df, 0, 0

    runs = _contiguous_runs(sorted(test_groups))
    keep_train_mask = pd.Series(True, index=train_df.index)
    keep_test_mask = pd.Series(True, index=test_df.index)
    n_purged_total = 0
    n_embargoed_total = 0
    for run in runs:
        run_test_df = test_df.loc[group_labels.loc[test_df.index].isin(run)]
        if run_test_df.empty:
            continue
        run_train_df = train_df.loc[keep_train_mask]
        purged_train, embargoed_test, n_purged, n_embargoed = apply_purge_embargo(
            run_train_df, run_test_df, date_col, purge_days=purge_days, embargo_days=embargo_days)
        dropped_train_idx = run_train_df.index.difference(purged_train.index)
        dropped_test_idx = run_test_df.index.difference(embargoed_test.index)
        keep_train_mask.loc[dropped_train_idx] = False
        keep_test_mask.loc[dropped_test_idx] = False
        n_purged_total += n_purged
        n_embargoed_total += n_embargoed

    return train_df.loc[keep_train_mask], test_df.loc[keep_test_mask], n_purged_total, n_embargoed_total


def evaluate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, stat_fn) -> tuple[StatResult, StatResult]:
    """Trivial wrapper: stat_fn is the same Callable[[pd.DataFrame],
    StatResult] contract walk_forward_validate() uses (e.g.
    `lambda d: stat_vs_zero(d["excess_pct"])`) -- no new statistics code."""
    return stat_fn(train_df), stat_fn(test_df)


def classify_cpcv_overall(path_verdicts: list[str], pct_robust_for_robust: float = DEFAULT_PCT_ROBUST_FOR_OVERALL_ROBUST,
                           pct_robust_for_moderate: float = DEFAULT_PCT_ROBUST_FOR_OVERALL_MODERATE
                           ) -> tuple[str, str, float | None]:
    """Pure decision logic over a list of per-path verdict labels (each
    one of classify_overfitting()'s own labels) -- separately unit-
    testable against a synthetic label list without building any
    DataFrame. Returns (overall_verdict, reasoning, pct_robust_or_moderate).

    A path whose OWN verdict is INSUFFICIENT_INSAMPLE_EDGE or
    INSUFFICIENT_DATA is excluded from the denominator entirely (it says
    nothing about robustness either way, same as walk_forward_validate()
    treats a single such split as uninformative rather than a failure)."""
    informative = [v for v in path_verdicts if v not in ("INSUFFICIENT_INSAMPLE_EDGE", "INSUFFICIENT_DATA")]
    if not informative:
        return ("INSUFFICIENT_DATA", "No path had sufficient in-sample edge or data to classify -- CPCV "
                                       "cannot say anything about robustness here.", None)
    n_robust_or_moderate = sum(1 for v in informative if v in ("ROBUST", "MODERATE"))
    pct = n_robust_or_moderate / len(informative)
    if pct >= pct_robust_for_robust:
        return ("ROBUST", f"{pct:.0%} of {len(informative)} informative CPCV path(s) classified ROBUST/MODERATE.",
                pct)
    if pct >= pct_robust_for_moderate:
        return ("MODERATE", f"{pct:.0%} of {len(informative)} informative CPCV path(s) classified "
                             "ROBUST/MODERATE -- partial agreement across paths.", pct)
    n_overfitted = sum(1 for v in informative if v == "OVERFITTED")
    if n_overfitted / len(informative) >= 1 - pct_robust_for_moderate:
        return ("OVERFITTED", f"only {pct:.0%} of {len(informative)} informative path(s) classified "
                               "ROBUST/MODERATE, and most of the rest were OVERFITTED -- the apparent edge does "
                               "not survive most held-out combinations.", pct)
    return ("WEAK", f"only {pct:.0%} of {len(informative)} informative CPCV path(s) classified ROBUST/MODERATE "
                     "-- the edge is inconsistent across held-out combinations.", pct)


def cpcv_validate(df: pd.DataFrame, date_col: str, stat_fn,
                   n_groups: int = DEFAULT_N_GROUPS, n_test_groups: int = DEFAULT_N_TEST_GROUPS,
                   purge_days: int = 0, embargo_days: int = 0,
                   min_n_per_split: int = DEFAULT_MIN_N_PER_SPLIT,
                   significance_t: float = DEFAULT_SIGNIFICANCE_T) -> CPCVResult:
    """Top-level entry point. Same stat_fn contract as walk_forward_
    validate(): Callable[[pd.DataFrame], StatResult].

    Evaluates every C(n_groups, n_test_groups) combinatorial split,
    skipping (not erroring on) any split whose train or test frame falls
    below min_n_per_split after purge/embargo -- CPCV path counts grow
    fast (C(6,2)=15, C(10,2)=45), so on a modest sample size some
    individual paths starving is expected, not a bug; it is counted in
    n_splits_skipped rather than hidden."""
    if df.empty or n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        return CPCVResult(n_groups=n_groups, n_test_groups=n_test_groups,
                           reasoning="Empty input or invalid n_groups/n_test_groups -- nothing to evaluate.")

    group_labels = assign_groups(df, date_col, n_groups)
    actual_n_groups = int(group_labels.nunique()) if len(group_labels) else 0
    if actual_n_groups < 2 or n_test_groups >= actual_n_groups:
        return CPCVResult(n_groups=n_groups, n_test_groups=n_test_groups,
                           reasoning=f"Only {actual_n_groups} distinct group(s) could be formed from "
                                     f"{len(df)} row(s) -- too few for n_test_groups={n_test_groups}.")

    paths: list[CPCVPathResult] = []
    n_skipped = 0
    for train_groups, test_groups in generate_cpcv_splits(actual_n_groups, n_test_groups):
        train_df, test_df, _, _ = build_split_frames(
            df, date_col, group_labels, train_groups, test_groups, purge_days=purge_days, embargo_days=embargo_days)
        if len(train_df) < min_n_per_split or len(test_df) < min_n_per_split:
            n_skipped += 1
            continue
        train_stat, test_stat = evaluate_split(train_df, test_df, stat_fn)
        verdict, _reasoning, retention, sign_flipped = classify_overfitting(
            train_stat, test_stat, significance_t=significance_t)
        paths.append(CPCVPathResult(test_groups=test_groups, train=train_stat, test=test_stat,
                                     retention_ratio=retention, sign_flipped=sign_flipped, verdict=verdict))

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
                                              if n_skipped else ""))
