"""
Tests for the FP-specific internals introduced by the *_fp rewrites --
the pieces that don't exist in the original modules at all, so the
ported original test suite (test_*_fp.py, generated from tests/) can't
exercise them. These test the pure building blocks in isolation, which
is the whole point of pulling them out in the first place.
"""
import numpy as np
import pandas as pd
import pytest

import walk_forward_validator_fp as wfv
import fama_macbeth_fp as fmb
import multiple_comparison_correction_fp as mcc
import dedup_store_fp as dd


# --- walk_forward_validator_fp: individual rules in the dispatch table ----

def _sr(mean, t_stat, n=50, significant=None, n_clusters=None):
    if significant is None:
        significant = abs(t_stat) >= wfv.DEFAULT_SIGNIFICANCE_T
    return wfv.StatResult(mean=mean, t_stat=t_stat, n=n, significant=significant, n_clusters=n_clusters)


def _ctx(train, test, **overrides):
    sign_flipped = (train.mean > 0 > test.mean) or (train.mean < 0 < test.mean)
    retention = (test.mean / train.mean) if train.mean != 0 else None
    defaults = dict(
        train=train, test=test, significance_t=wfv.DEFAULT_SIGNIFICANCE_T,
        robust_retention=wfv.DEFAULT_ROBUST_RETENTION, moderate_retention=wfv.DEFAULT_MODERATE_RETENTION,
        overfit_retention=wfv.DEFAULT_OVERFIT_RETENTION, retention=retention, sign_flipped=sign_flipped,
    )
    defaults.update(overrides)
    return wfv._Ctx(**defaults)


def test_rule_no_insample_edge_fires_when_train_not_significant():
    ctx = _ctx(_sr(0.01, 0.5), _sr(0.01, 0.5))
    v = wfv._rule_no_insample_edge(ctx)
    assert v is not None and v.label == "INSUFFICIENT_INSAMPLE_EDGE"


def test_rule_no_insample_edge_is_none_when_train_significant():
    ctx = _ctx(_sr(0.5, 3.0), _sr(0.1, 1.0))
    assert wfv._rule_no_insample_edge(ctx) is None


def test_rule_sign_flip_fires_on_opposite_signs():
    ctx = _ctx(_sr(0.5, 3.0), _sr(-0.5, -3.0))
    v = wfv._rule_sign_flip(ctx)
    assert v is not None and v.label == "OVERFITTED" and v.sign_flipped is True


def test_rule_sign_flip_is_none_when_same_sign():
    ctx = _ctx(_sr(0.5, 3.0), _sr(0.3, 2.5))
    assert wfv._rule_sign_flip(ctx) is None


def test_rules_are_evaluated_in_order_first_match_wins():
    """A train-insignificant case must hit _rule_no_insample_edge even
    if it would ALSO match a later rule -- verifies dispatch order, not
    just that each rule works standalone."""
    train, test = _sr(0.01, 0.1), _sr(-0.01, -0.1)  # insignificant AND sign-flipped
    label, reasoning, _, _ = wfv.classify_overfitting(train, test)
    assert label == "INSUFFICIENT_INSAMPLE_EDGE"


def test_fit_ols_dispatch_prefers_cluster_over_hac_when_both_given():
    y = np.random.RandomState(0).normal(size=40)
    x = pd.DataFrame({"const": np.ones(40)})
    groups = pd.Series([0, 1, 2, 3] * 10)
    model, n_clusters = wfv._fit_ols(y, x, maxlags=3, cluster_groups=groups)
    assert n_clusters == 4  # cluster path taken, not HAC


def test_fit_ols_dispatch_falls_back_to_plain_when_neither_given():
    y = np.random.RandomState(0).normal(size=20)
    x = pd.DataFrame({"const": np.ones(20)})
    model, n_clusters = wfv._fit_ols(y, x, maxlags=None, cluster_groups=None)
    assert n_clusters is None


# --- fama_macbeth_fp: per-cohort fit functions in isolation ----------------

def test_fit_one_cohort_slope_returns_none_for_constant_x():
    df = pd.DataFrame({"x": [1.0] * 20, "y": np.random.randn(20)})
    assert fmb._fit_one_cohort_slope(df, "x", "y", min_obs_per_cohort=5, winsorize_x_pct=None) is None


def test_fit_one_cohort_slope_returns_none_below_min_obs():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    assert fmb._fit_one_cohort_slope(df, "x", "y", min_obs_per_cohort=10, winsorize_x_pct=None) is None


def test_fit_one_cohort_slope_recovers_known_slope():
    rng = np.random.RandomState(0)
    x = rng.normal(size=200)
    y = 3.0 * x + rng.normal(scale=0.01, size=200)
    df = pd.DataFrame({"x": x, "y": y})
    slope = fmb._fit_one_cohort_slope(df, "x", "y", min_obs_per_cohort=5, winsorize_x_pct=None)
    assert slope == pytest.approx(3.0, abs=0.05)


def test_fit_one_cohort_group_diff_returns_none_when_one_side_too_small():
    df = pd.DataFrame({"group": [True] * 10 + [False], "y": np.random.randn(11)})
    assert fmb._fit_one_cohort_group_diff(df, "group", "y", min_obs_per_cohort=5) is None


def test_clean_finite_drops_inf_and_nan():
    df = pd.DataFrame({"a": [1.0, np.inf, np.nan, 2.0], "b": [1.0, 2.0, 3.0, 4.0]})
    cleaned = fmb._clean_finite(df, ["a", "b"])
    assert list(cleaned["a"]) == [1.0, 2.0]


# --- multiple_comparison_correction_fp: holm/BH internals ------------------

def test_holm_matches_bonferroni_when_all_pvalues_tiny():
    p_values = [0.0001, 0.0002, 0.0003]
    assert mcc.holm_correction(p_values) == [True, True, True]


def test_holm_stops_at_first_failure_rank():
    # rank0 threshold alpha/3, rank1 alpha/2, rank2 alpha/1
    p_values = [0.001, 0.04, 0.9]  # rank1 (0.04) fails alpha/2=0.025 -> ranks 1,2 rejected-as-not-significant
    sig = mcc.holm_correction(p_values, alpha=0.05)
    assert sig == [True, False, False]


def test_benjamini_hochberg_rejects_more_than_bonferroni_on_correlated_batch():
    p_values = [0.001, 0.004, 0.006, 0.008, 0.30, 0.60]
    bh = mcc.benjamini_hochberg_fdr(p_values, alpha=0.05)
    bonf = mcc.bonferroni_correction(p_values, alpha=0.05)
    assert sum(bh) >= sum(bonf)


def test_empty_pvalues_returns_empty_for_all_three_methods():
    assert mcc.bonferroni_correction([]) == []
    assert mcc.holm_correction([]) == []
    assert mcc.benjamini_hochberg_fdr([]) == []


# --- dedup_store_fp: pure helpers in isolation, no disk at all -------------

def test_combine_with_no_existing_store_returns_new_rows_unchanged():
    new_rows = pd.DataFrame([{"id": "1"}])
    assert dd._combine(None, new_rows).equals(new_rows)


def test_combine_concatenates_existing_and_new():
    existing = pd.DataFrame([{"id": "1"}])
    new_rows = pd.DataFrame([{"id": "2"}])
    combined = dd._combine(existing, new_rows)
    assert len(combined) == 2


def test_cast_dedup_cols_to_str_does_not_mutate_input():
    df = pd.DataFrame({"id": [1, 2, 3]})
    _ = dd._cast_dedup_cols_to_str(df, ["id"])
    assert df["id"].dtype != object  # original untouched


def test_dedupe_and_sort_keeps_last_and_reports_drop_count():
    df = pd.DataFrame([{"id": "1", "v": "old"}, {"id": "1", "v": "new"}])
    result, n_dropped = dd._dedupe_and_sort(df, ["id"])
    assert n_dropped == 1
    assert result.iloc[0]["v"] == "new"


def test_corrupted_store_message_format():
    from pathlib import Path
    msg = dd._corrupted_store_message(Path("/tmp/store.csv"), ValueError("boom"), "treating as empty")
    assert "store.csv" in msg and "boom" in msg and "treating as empty" in msg
