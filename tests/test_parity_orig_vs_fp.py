"""Randomized parity: the original and FP implementations must return the
same results -- verdicts, statistics, counts, and reasoning text -- on the
same inputs. The README promises the FP modules are drop-in replacements;
this is what enforces it as either side changes."""
import math

import numpy as np
import pandas as pd
import pytest

import cpcv_validator
import cpcv_validator_fp
import fama_macbeth
import fama_macbeth_fp
import multiple_comparison_correction
import multiple_comparison_correction_fp
import walk_forward_validator
import walk_forward_validator_fp

SEEDS = range(40)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-12, abs=1e-15)
    return a == b


def _stat_tuple(s):
    return (s.mean, s.t_stat, s.n, s.significant, s.n_clusters, s.df)


def _assert_stats_equal(a, b):
    for x, y in zip(_stat_tuple(a), _stat_tuple(b), strict=True):
        assert _same(x, y), (_stat_tuple(a), _stat_tuple(b))


def _frame(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 150))
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.integers(0, n * 2, n)), "D")
    df = pd.DataFrame({"date": dates, "r": rng.normal(rng.uniform(-0.5, 1), 1, n), "g": rng.random(n) < 0.5,
                       "cl": rng.integers(0, 8, n)})
    return rng, df.sample(frac=1, random_state=seed)  # unsorted input on purpose


@pytest.mark.parametrize("seed", SEEDS)
def test_walk_forward_parity(seed):
    rng, df = _frame(seed)
    kw = dict(purge_days=int(rng.integers(0, 6)), embargo_days=int(rng.integers(0, 4)),
              train_frac=float(rng.uniform(0.5, 0.85)))
    fixed_t = None if rng.random() < 0.7 else 2.0
    for make in (lambda m: (lambda d: m.stat_group_diff(d.r, d.g, cluster_groups=d.cl.to_numpy())),
                 lambda m: (lambda d: m.stat_vs_zero(d.r, maxlags=3))):
        a = walk_forward_validator.walk_forward_validate(df, "date", make(walk_forward_validator),
                                                         significance_t=fixed_t, **kw)
        b = walk_forward_validator_fp.walk_forward_validate(df, "date", make(walk_forward_validator_fp),
                                                            significance_t=fixed_t, **kw)
        assert (a.verdict, a.reasoning, a.sign_flipped, a.n_purged_train, a.n_embargoed_test) == \
               (b.verdict, b.reasoning, b.sign_flipped, b.n_purged_train, b.n_embargoed_test)
        assert _same(a.retention_ratio if a.retention_ratio is not None else float("nan"),
                     b.retention_ratio if b.retention_ratio is not None else float("nan"))
        _assert_stats_equal(a.train, b.train)
        _assert_stats_equal(a.test, b.test)


@pytest.mark.parametrize("seed", SEEDS)
def test_cpcv_parity(seed):
    rng, df = _frame(seed)
    kw = dict(n_groups=int(rng.integers(3, 8)), n_test_groups=int(rng.integers(1, 3)),
              purge_days=int(rng.integers(0, 6)), embargo_days=int(rng.integers(0, 4)))
    a = cpcv_validator.cpcv_validate(
        df, "date", lambda d: walk_forward_validator.stat_vs_zero(d.r, cluster_groups=d.cl), **kw)
    b = cpcv_validator_fp.cpcv_validate(
        df, "date", lambda d: walk_forward_validator_fp.stat_vs_zero(d.r, cluster_groups=d.cl), **kw)
    assert (a.overall_verdict, a.reasoning, a.n_groups, a.n_splits_skipped) == \
           (b.overall_verdict, b.reasoning, b.n_groups, b.n_splits_skipped)
    assert [(p.test_groups, p.verdict, p.sign_flipped) for p in a.paths] == \
           [(p.test_groups, p.verdict, p.sign_flipped) for p in b.paths]
    for pa, pb in zip(a.paths, b.paths, strict=True):
        _assert_stats_equal(pa.train, pb.train)
        _assert_stats_equal(pa.test, pb.test)
    groups = cpcv_validator.assign_groups(df, "date", kw["n_groups"])
    assert groups.equals(cpcv_validator_fp.assign_groups(df, "date", kw["n_groups"]))
    for train_g, test_g in cpcv_validator.generate_cpcv_splits(int(groups.nunique()), 1):
        fa = cpcv_validator.build_split_frames(df, "date", groups, train_g, test_g, kw["purge_days"],
                                               kw["embargo_days"])
        fb = cpcv_validator_fp.build_split_frames(df, "date", groups, train_g, test_g, kw["purge_days"],
                                                  kw["embargo_days"])
        assert list(fa[0].index) == list(fb[0].index) and list(fa[1].index) == list(fb[1].index)
        assert fa[2:] == fb[2:]


@pytest.mark.parametrize("seed", SEEDS)
def test_fama_macbeth_parity(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 400))
    panel = pd.DataFrame({"c": rng.integers(0, 8, n), "x": rng.normal(size=n), "x2": rng.normal(size=n),
                          "gg": rng.random(n) < 0.5})
    panel["y"] = 0.3 * panel["x"] + rng.normal(size=n)
    kw = dict(min_obs_per_cohort=5, min_periods=int(rng.integers(2, 5)),
              newey_west_lags=None if rng.random() < 0.5 else int(rng.integers(1, 3)))
    for fn, args in (("fama_macbeth_regression", ("c", "x", "y")), ("fama_macbeth_group_diff", ("c", "gg", "y"))):
        a = getattr(fama_macbeth, fn)(panel, *args, **kw)
        b = getattr(fama_macbeth_fp, fn)(panel, *args, **kw)
        for field in ("mean_estimate", "std_estimate", "t_stat", "p_value"):
            assert _same(getattr(a, field), getattr(b, field)), field
        assert (a.n_periods, a.significant, a.dropped_cohorts) == (b.n_periods, b.significant, b.dropped_cohorts)
        pd.testing.assert_series_equal(a.period_estimates, b.period_estimates)
    a = fama_macbeth.fama_macbeth_multi_regression(panel, "c", ["x", "x2"], "y", **kw)
    b = fama_macbeth_fp.fama_macbeth_multi_regression(panel, "c", ["x", "x2"], "y", **kw)
    pd.testing.assert_frame_equal(a.period_estimates, b.period_estimates)
    pd.testing.assert_series_equal(a.t_stat, b.t_stat)
    pd.testing.assert_series_equal(a.significant, b.significant)
    assert a.dropped_cohorts == b.dropped_cohorts


@pytest.mark.parametrize("seed", SEEDS)
def test_multiple_comparison_parity(seed):
    rng = np.random.default_rng(seed)
    p = np.round(rng.uniform(0, 0.2, int(rng.integers(1, 30))), 3).tolist()
    labels = [f"s{i}" for i in range(len(p))]
    for method in ("bonferroni", "holm", "fdr_bh"):
        assert multiple_comparison_correction.summarize_correction(labels, p, method=method) == \
               multiple_comparison_correction_fp.summarize_correction(labels, p, method=method)
