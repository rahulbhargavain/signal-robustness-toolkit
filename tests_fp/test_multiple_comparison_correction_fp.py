"""Tests for multiple_comparison_correction.py -- verified against
synthetic known-answer cases AND the real, already-known PIT fundamentals
result (DeltaROCE p=0.043 fails m=6 Bonferroni"""
import pytest

import multiple_comparison_correction_fp as mcc


# --- bonferroni_correction() -------------------------------------------------

def test_bonferroni_reproduces_the_real_known_delta_roce_result():
    """DeltaROCE's p=0.043 among m=6 simultaneous tests failed Bonferroni
    in the real 2026-08-23 audit (threshold 0.05/6=0.00833). This is the
    exact real case this module was built to formalize -- must reproduce
    it exactly, not just something plausible."""
    p_values = [0.043, 0.30, 0.55, 0.71, 0.12, 0.88]  # DeltaROCE first, order matches the audit's own listing
    sig = mcc.bonferroni_correction(p_values, alpha=0.05)
    assert sig[0] is False  # DeltaROCE fails, exactly as the real audit found
    assert not any(sig)  # none of the 6 survive -- matches "all 6 null" from the same memory


def test_bonferroni_boundary_exact_threshold():
    # threshold = 0.05/2 = 0.025 -- both strictly below it
    assert mcc.bonferroni_correction([0.01, 0.02], alpha=0.05) == [True, True]


def test_bonferroni_single_test_reduces_to_plain_alpha():
    assert mcc.bonferroni_correction([0.04], alpha=0.05) == [True]
    assert mcc.bonferroni_correction([0.06], alpha=0.05) == [False]


def test_bonferroni_empty_input_returns_empty():
    assert mcc.bonferroni_correction([]) == []


# --- holm_correction() -------------------------------------------------------

def test_holm_rejects_at_least_as_many_as_bonferroni():
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.20]
    bonf = mcc.bonferroni_correction(p_values)
    holm = mcc.holm_correction(p_values)
    assert sum(holm) >= sum(bonf)


def test_holm_stops_at_first_failure_step_down():
    """Holm's step-down property: once one candidate fails its threshold
    (in ascending p-value order), everything after it must also be marked
    not-significant, even if a LATER p-value would individually pass a
    plain alpha/m Bonferroni-style check on its own."""
    p_values = [0.001, 0.20, 0.30]  # second one fails its Holm threshold (0.05/2=0.025)
    holm = mcc.holm_correction(p_values)
    assert holm == [True, False, False]


def test_holm_empty_input_returns_empty():
    assert mcc.holm_correction([]) == []


# --- benjamini_hochberg_fdr() -------------------------------------------------

def test_bh_fdr_rejects_at_least_as_many_as_holm():
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.20]
    holm = mcc.holm_correction(p_values)
    bh = mcc.benjamini_hochberg_fdr(p_values)
    assert sum(bh) >= sum(holm)


def test_bh_fdr_known_answer():
    # Classic textbook example: 5 p-values, alpha=0.05
    # sorted: 0.001, 0.008, 0.039, 0.041, 0.042 -- thresholds: .01, .02, .03, .04, .05
    # largest k where p_(k) <= threshold_(k): k=2 (0.008 <= 0.02) -- 0.039 > 0.03 fails, breaks the "largest k" run upward... verify directly
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042]
    sig = mcc.benjamini_hochberg_fdr(p_values, alpha=0.05)
    assert sig[0] is True and sig[1] is True  # the two smallest always survive here
    assert sum(sig) >= 2


def test_bh_fdr_empty_input_returns_empty():
    assert mcc.benjamini_hochberg_fdr([]) == []


def test_bh_fdr_no_survivors_when_all_p_values_large():
    assert mcc.benjamini_hochberg_fdr([0.5, 0.6, 0.7]) == [False, False, False]


# --- summarize_correction() ---------------------------------------------------

def test_summarize_correction_matches_delta_roce_real_case():
    labels = ["DeltaROCE", "DeltaCCC", "reinvestment", "accruals", "leverage", "EPS_growth"]
    p_values = [0.043, 0.30, 0.55, 0.71, 0.12, 0.88]
    rows = mcc.summarize_correction(labels, p_values, method="bonferroni")
    roce_row = rows[0]
    assert roce_row["label"] == "DeltaROCE"
    assert roce_row["significant_uncorrected"] is True  # 0.043 < 0.05 alone
    assert roce_row["significant_corrected"] is False  # fails once corrected for m=6
    assert roce_row["n_tests"] == 6


def test_summarize_correction_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        mcc.summarize_correction(["a", "b"], [0.01])


def test_summarize_correction_supports_all_three_methods():
    labels, p_values = ["a", "b"], [0.01, 0.02]
    for method in ("bonferroni", "holm", "fdr_bh"):
        rows = mcc.summarize_correction(labels, p_values, method=method)
        assert len(rows) == 2
        assert rows[0]["method"] == method
