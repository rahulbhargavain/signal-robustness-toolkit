"""Small-sample critical t, Fama-MacBeth min_periods / Newey-West, and
multiple-comparison agreement with statsmodels -- each run against BOTH
the original and the FP implementation."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import fama_macbeth
import fama_macbeth_fp
import multiple_comparison_correction
import multiple_comparison_correction_fp
import walk_forward_validator
import walk_forward_validator_fp

WFV = pytest.mark.parametrize("wfv", [walk_forward_validator, walk_forward_validator_fp], ids=["orig", "fp"])
FM = pytest.mark.parametrize("fm", [fama_macbeth, fama_macbeth_fp], ids=["orig", "fp"])
MCC = pytest.mark.parametrize("mcc", [multiple_comparison_correction, multiple_comparison_correction_fp],
                              ids=["orig", "fp"])


def _series_with_t(t, n, seed=0):
    """A length-n series whose one-sample t-stat is exactly t."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return pd.Series(z + t / np.sqrt(n))


# --- small-sample critical t -------------------------------------------------

@WFV
def test_critical_t_uses_student_t_with_the_stats_own_df(wfv):
    assert wfv.critical_t(wfv.StatResult(mean=1, t_stat=3, n=8, significant=True)) == pytest.approx(
        stats.t.ppf(0.975, 7))
    assert wfv.critical_t(wfv.StatResult(mean=1, t_stat=3, n=8, significant=True, df=4)) == pytest.approx(
        stats.t.ppf(0.975, 4))
    assert wfv.critical_t(wfv.StatResult(mean=1, t_stat=3, n=8, significant=True), significance_t=2.0) == 2.0
    assert wfv.critical_t(wfv.StatResult(mean=1, t_stat=3, n=1, significant=True)) == float("inf")


@WFV
def test_t_of_2_2_on_8_rows_is_not_significant_by_default(wfv):
    res = wfv.stat_vs_zero(_series_with_t(2.2, 8))
    assert res.t_stat == pytest.approx(2.2)
    assert res.df == 7
    assert not res.significant  # t(7) 5% critical value is 2.365
    assert wfv.stat_vs_zero(_series_with_t(2.2, 8), significance_t=2.0).significant
    assert wfv.stat_vs_zero(_series_with_t(2.2, 500)).significant  # large n -> ~1.96


@WFV
def test_group_diff_and_cluster_degrees_of_freedom(wfv):
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(size=40))
    groups = pd.Series(np.tile([True, False], 20))
    assert wfv.stat_group_diff(values, groups).df == 38
    clusters = pd.Series(np.repeat(np.arange(5), 8))
    res = wfv.stat_vs_zero(values, cluster_groups=clusters)
    assert res.n_clusters == 5 and res.df == 4


@WFV
def test_classify_overfitting_applies_small_sample_threshold(wfv):
    test = wfv.StatResult(mean=0.9, t_stat=3.0, n=8, significant=True)
    small = wfv.StatResult(mean=1.0, t_stat=2.2, n=8, significant=True)
    verdict, reasoning, _, _ = wfv.classify_overfitting(small, test)
    assert verdict == "INSUFFICIENT_INSAMPLE_EDGE"
    assert "2.36" in reasoning  # the critical value actually applied
    large = wfv.StatResult(mean=1.0, t_stat=2.2, n=500, significant=True)
    assert wfv.classify_overfitting(large, test)[0] == "ROBUST"
    assert wfv.classify_overfitting(small, test, significance_t=2.0)[0] == "ROBUST"


# --- Fama-MacBeth: min_periods and Newey-West -------------------------------

def _panel(n_cohorts, slope=1.0, seed=0, rows=30):
    rng = np.random.default_rng(seed)
    frames = []
    for c in range(n_cohorts):
        x = rng.normal(size=rows)
        frames.append(pd.DataFrame({"cohort": c, "x": x, "y": slope * x + rng.normal(0, 0.5, rows),
                                    "g": np.tile([True, False], rows // 2)}))
    return pd.concat(frames, ignore_index=True)


@FM
def test_too_few_cohorts_reports_mean_but_no_t_stat(fm):
    res = fm.fama_macbeth_regression(_panel(2), "cohort", "x", "y")
    assert res.n_periods == 2
    assert np.isfinite(res.mean_estimate)
    assert np.isnan(res.t_stat) and np.isnan(res.p_value) and not res.significant
    res = fm.fama_macbeth_regression(_panel(2), "cohort", "x", "y", min_periods=2)
    assert np.isfinite(res.t_stat)


@FM
def test_min_periods_applies_to_every_entry_point(fm):
    panel = _panel(4)
    assert np.isnan(fm.fama_macbeth_group_diff(panel, "cohort", "g", "y", min_periods=5).t_stat)
    multi = fm.fama_macbeth_multi_regression(panel.assign(x2=panel["x"] ** 2), "cohort", ["x", "x2"], "y",
                                             min_periods=5)
    assert multi.t_stat.isna().all()


@FM
def test_newey_west_se_matches_statsmodels_hac(fm):
    est = pd.Series(np.random.default_rng(0).normal(0.5, 1, 15)).cumsum() * 0.1
    for lags in (1, 3):
        hac = sm.OLS(est.to_numpy(), np.ones(len(est))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
        assert fm._newey_west_se(est, lags) == pytest.approx(hac.bse[0])


@FM
def test_newey_west_shrinks_t_on_autocorrelated_cohort_estimates(fm):
    """Slopes that drift slowly across cohorts (serially correlated) look
    more precise than they are under the plain FM SE."""
    rng = np.random.default_rng(3)
    drift = 0.3 + np.cumsum(rng.normal(0, 0.15, 20))
    frames = []
    for c, b in enumerate(drift):
        x = rng.normal(size=40)
        frames.append(pd.DataFrame({"cohort": c, "x": x, "y": b * x + rng.normal(0, 0.1, 40)}))
    panel = pd.concat(frames, ignore_index=True)
    plain = fm.fama_macbeth_regression(panel, "cohort", "x", "y")
    nw = fm.fama_macbeth_regression(panel, "cohort", "x", "y", newey_west_lags=3)
    assert nw.mean_estimate == plain.mean_estimate
    assert abs(nw.t_stat) < abs(plain.t_stat)
    assert nw.p_value > plain.p_value


# --- multiple-comparison corrections vs. statsmodels ------------------------

@MCC
@pytest.mark.parametrize("method,ours", [("bonferroni", "bonferroni_correction"), ("holm", "holm_correction"),
                                         ("fdr_bh", "benjamini_hochberg_fdr")])
@pytest.mark.parametrize("seed", range(20))
def test_matches_statsmodels_multipletests(mcc, method, ours, seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 40))
    # Mix of real-looking signals and nulls, rounded so ties occur.
    p = np.round(np.concatenate([rng.uniform(0, 0.01, m // 3), rng.uniform(0, 1, m - m // 3)]), 3).tolist()
    for alpha in (0.05, 0.10):
        expected = multipletests(p, alpha=alpha, method=method)[0].tolist()
        assert getattr(mcc, ours)(p, alpha=alpha) == expected


@MCC
def test_p_exactly_at_threshold_is_rejected_by_every_method(mcc):
    assert mcc.bonferroni_correction([0.025, 0.025], alpha=0.05) == [True, True]
    assert mcc.holm_correction([0.025, 0.05], alpha=0.05) == [True, True]
    assert mcc.benjamini_hochberg_fdr([0.025, 0.05], alpha=0.05) == [True, True]
    assert mcc.summarize_correction(["a"], [0.05], alpha=0.05)[0]["significant_uncorrected"] is True
