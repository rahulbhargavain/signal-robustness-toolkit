"""
Multiple-comparison p-value correction: Bonferroni, Holm, and
Benjamini-Hochberg FDR, for when a batch of signals was tested and you
need to know which survivors are real rather than hand-computing 0.05/m.

- Bonferroni: reject p <= alpha/m. Controls the family-wise error rate
  (probability of ANY false positive in the batch). Most conservative.
- Holm: step-down Bonferroni. Controls the same family-wise error rate and
  never rejects fewer hypotheses than Bonferroni, so prefer it in general.
- Benjamini-Hochberg: controls the expected FALSE DISCOVERY rate instead
  -- less conservative, appropriate when a batch has many related tests
  (e.g. a factor-family sweep) and some real signal is plausible.

All three use an inclusive `p <= threshold` rule, matching
statsmodels.stats.multitest.multipletests (tests check agreement).

WHY NOT the Deflated Sharpe Ratio: DSR is a Sharpe-ratio-based correction
-- it needs a return series plus the variance of Sharpe ratios ACROSS the
trials tried. This toolkit evaluates edge via a t-stat on a mean return or
a group difference, not an annualized Sharpe, so these corrections express
the same concern (how many things did you try before finding this result)
in the t-stat/p-value idiom used throughout.
"""
import numpy as np


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """The simplest, most conservative correction: reject at
    alpha/len(p_values) instead of alpha. Controls the family-wise error
    rate (probability of ANY false positive across the whole batch) --
    appropriate when even one false discovery would be costly (e.g.
    deciding whether to trade a new signal live)."""
    _check_alpha(alpha)
    if not p_values:
        return []
    threshold = alpha / len(p_values)
    return [p <= threshold for p in p_values]


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Step-down Holm-Bonferroni: controls the same family-wise error rate
    as Bonferroni while never rejecting fewer hypotheses. Sort p-values
    ascending, test the smallest against alpha/m, the next against
    alpha/(m-1), etc., and stop at the first one that fails -- it and
    every larger p-value are not rejected."""
    _check_alpha(alpha)
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    significant = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
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
    _check_alpha(alpha)
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
    corrections = {"bonferroni": bonferroni_correction, "holm": holm_correction,
                   "fdr_bh": benjamini_hochberg_fdr}
    if method not in corrections:
        raise ValueError(f"method must be one of {sorted(corrections)}, got {method!r}")
    correction_fn = corrections[method]
    corrected = correction_fn(p_values, alpha)
    return [
        {"label": label, "p_value": p, "significant_uncorrected": p <= alpha,
         "significant_corrected": sig, "method": method, "n_tests": len(p_values)}
        for label, p, sig in zip(labels, p_values, corrected, strict=True)
    ]
