"""
Multiple-comparison p-value correction (added 2026-08-29) -- generalizes
the manual Bonferroni correction fama_macbeth.py already applies ad hoc
(m=6 across the PIT fundamentals signal audit: DeltaROCE/DeltaCCC/
reinvestment/accruals/leverage/EPS-growth)
into a reusable, tested, first-class utility any
future multi-test batch can call directly instead of hand-computing
0.05/m inline.

WHY NOT purgedcv's Deflated Sharpe Ratio (also evaluated 2026-08-29):
DSR is fundamentally a Sharpe-ratio-based correction -- it needs a return
series plus the variance of Sharpe ratios ACROSS the trials tried. Every 
walk_forward_validator.py evaluates edge via
a t-stat on a mean return or a group difference, not an annualized
Sharpe). Wiring in DSR would mean introducing a whole new metric family
this codebase has never used, purely to feed a correction it doesn't
otherwise need in that form. This module instead expresses the SAME
underlying concern (how many things did you try before finding this
result) in the t-stat/p-value idiom already used throughout
walk_forward_validator.py and every backtest script's significance
checks -- Bonferroni, Holm (a strictly more powerful step-down version of
Bonferroni that should be preferred over it in general), and
Benjamini-Hochberg FDR (controls the expected FALSE DISCOVERY rate
instead of the family-wise error rate -- less conservative, appropriate
when a batch has many related tests and some real signal is plausible,
matching the ROLLING FACTOR-FAMILY SWEEPS this repo actually runs.
"""
import numpy as np


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """The simplest, most conservative correction: reject at
    alpha/len(p_values) instead of alpha. Controls the family-wise error
    rate (probability of ANY false positive across the whole batch) --
    appropriate when even one false discovery would be costly (e.g.
    deciding whether to wire a new signal into a live trim/entry gate)."""
    if not p_values:
        return []
    threshold = alpha / len(p_values)
    return [p < threshold for p in p_values]


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Step-down Holm-Bonferroni: strictly more powerful than plain
    Bonferroni (rejects at least as many, sometimes more) while still
    controlling the same family-wise error rate -- sort p-values
    ascending, test the smallest against alpha/m, the next against
    alpha/(m-1), etc., stopping at the first failure (every p-value after
    a failure is also rejected-as-not-significant, since Holm's guarantee
    only holds for the contiguous run of successes from the smallest
    p-value up)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    significant = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] < threshold:
            significant[idx] = True
        else:
            break  # Holm stops at the first non-rejection; nothing after it can be rejected either
    return significant.tolist()


def benjamini_hochberg_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Controls the expected FALSE DISCOVERY RATE (the expected fraction
    of rejected hypotheses that are false positives) rather than the
    family-wise error rate -- less conservative than Bonferroni/Holm, the
    right choice when a batch has many related tests and some real
    signal is plausible rather than testing a single make-or-break
    decision. Sort ascending, find the largest rank k where
    p_(k) <= (k/m)*alpha, reject that one and everything below it."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passing = sorted_p <= thresholds
    significant = np.zeros(n, dtype=bool)
    if passing.any():
        max_rank = np.max(np.flatnonzero(passing))  # largest k satisfying the condition
        significant[order[:max_rank + 1]] = True
    return significant.tolist()


def summarize_correction(labels: list[str], p_values: list[float], alpha: float = 0.05,
                          method: str = "bonferroni") -> list[dict]:
    """Convenience wrapper: one row per (label, p_value) with the
    uncorrected verdict alongside the corrected one, so a caller can print
    a clear before/after table rather than two parallel lists that have
    to be zipped by hand."""
    if len(labels) != len(p_values):
        raise ValueError(f"labels ({len(labels)}) and p_values ({len(p_values)}) must be the same length")
    correction_fn = {"bonferroni": bonferroni_correction, "holm": holm_correction,
                      "fdr_bh": benjamini_hochberg_fdr}[method]
    corrected = correction_fn(p_values, alpha)
    return [
        {"label": label, "p_value": p, "significant_uncorrected": p < alpha,
         "significant_corrected": sig, "method": method, "n_tests": len(p_values)}
        for label, p, sig in zip(labels, p_values, corrected)
    ]
