"""
multiple_comparison_correction_fp.py -- Functional-programming rewrite of
multiple_comparison_correction.py.

DROP-IN COMPATIBLE: same function names, signatures, and return types
(list[bool] for the three correction functions, list[dict] for
summarize_correction). Behavior is unchanged.

WHAT CHANGED:

1. holm_correction's mutate-an-array-in-a-for-loop-with-break becomes:
   compute all m per-rank thresholds at once (vectorized), find the
   first rank that FAILS via next()-with-default (the functional
   equivalent of "loop until you find one and stop"), then derive the
   whole boolean result from that single index with one comparison --
   no `significant[idx] = True` mutation anywhere.

2. benjamini_hochberg_fdr already avoided a mutate-in-a-loop shape in
   the original; here it's expressed as one pipeline (sort -> compare
   -> find max passing rank -> reindex) with no intermediate mutable
   array either.

3. bonferroni_correction and summarize_correction were already pure,
   single-expression functions -- left essentially as-is, since there
   was no imperative structure to remove.
"""
from __future__ import annotations

import numpy as np


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Reject at alpha/len(p_values) instead of alpha. Controls the
    family-wise error rate."""
    if not p_values:
        return []
    threshold = alpha / len(p_values)
    return [p < threshold for p in p_values]


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Step-down Holm-Bonferroni. Sort ascending, test the smallest
    against alpha/m, the next against alpha/(m-1), etc.; the first
    p-value that fails its threshold, and everything after it (in
    sorted order), is not rejected."""
    n = len(p_values)
    if n == 0:
        return []

    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[order]
    thresholds = alpha / (n - np.arange(n))  # per-rank threshold, rank 0 first

    # Index of the first rank that fails its threshold -- everything at
    # or after this rank is not rejected. `n` (one past the last valid
    # index) is the default when every rank passes, i.e. nothing fails.
    first_failure = next((rank for rank in range(n) if sorted_p[rank] >= thresholds[rank]), n)

    sorted_significant = np.arange(n) < first_failure
    significant = np.empty(n, dtype=bool)
    significant[order] = sorted_significant
    return significant.tolist()


def benjamini_hochberg_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Controls the expected false discovery rate. Sort ascending, find
    the largest rank k where p_(k) <= (k/m)*alpha, reject that one and
    everything below it."""
    n = len(p_values)
    if n == 0:
        return []

    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passing = sorted_p <= thresholds

    if not passing.any():
        return [False] * n

    max_rank = int(np.max(np.flatnonzero(passing)))
    sorted_significant = np.arange(n) <= max_rank
    significant = np.empty(n, dtype=bool)
    significant[order] = sorted_significant
    return significant.tolist()


_CORRECTIONS = {
    "bonferroni": bonferroni_correction,
    "holm": holm_correction,
    "fdr_bh": benjamini_hochberg_fdr,
}


def summarize_correction(labels: list[str], p_values: list[float], alpha: float = 0.05,
                          method: str = "bonferroni") -> list[dict]:
    """One row per (label, p_value) with the uncorrected verdict
    alongside the corrected one."""
    if len(labels) != len(p_values):
        raise ValueError(f"labels ({len(labels)}) and p_values ({len(p_values)}) must be the same length")
    corrected = _CORRECTIONS[method](p_values, alpha)
    return [
        {"label": label, "p_value": p, "significant_uncorrected": p < alpha,
         "significant_corrected": sig, "method": method, "n_tests": len(p_values)}
        for label, p, sig in zip(labels, p_values, corrected)
    ]
