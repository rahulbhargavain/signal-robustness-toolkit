# signal-robustness-toolkit

Five small, independent, fully-tested Python modules for one recurring problem in
quantitative signal research: **a backtest result that looks statistically
significant on the whole sample is not the same as a result that's real.**


## What's here

| Module | Answers |
|---|---|
| `walk_forward_validator.py` | Split a backtest chronologically (train/test), recompute the same stat on both halves, and classify the result — `ROBUST` / `MODERATE` / `WEAK` / `OVERFITTED` / `INSUFFICIENT_INSAMPLE_EDGE`. A signal that only "worked" in-sample and evaporates or reverses out-of-sample gets caught here, not published as a finding. Also provides `apply_purge_embargo()` — drop train rows whose own forward-return window bleeds into the test period, and an extra buffer at the start of test — for when a signal's horizon is wide enough that a naive split boundary leaks. |
| `cpcv_validator.py` | A stronger sibling to `walk_forward_validator.py`'s single 70/30 split: Combinatorial Purged Cross-Validation (Lopez de Prado). Partitions the timeline into N groups and evaluates **every** C(N,k) held-out combination (purge/embargo applied at each split's own boundaries), reporting a distribution of verdicts instead of one draw. A single chronological split's verdict can be an artifact of exactly where that one boundary happened to land — CPCV has repeatedly caught cases where it disagreed with the single-split verdict in both directions (overturning a false OVERFITTED, and confirming a real one), so prefer it over the single split when you can afford C(N,k) evaluations of your stat function. Builds directly on `walk_forward_validator.py`'s `StatResult`/`classify_overfitting()`/`apply_purge_embargo()` — not a separate implementation. |
| `fama_macbeth.py` | For "many entities, few time periods" panels (e.g. ~300 stocks reporting on the same annual cycle, giving ~10-17 true independent cohorts, not thousands of independent stock-events) — pooled OLS understates the real clustering and inflates t-stats 4-5x. Runs one regression per cohort and tests the resulting time series of per-cohort estimates instead. |
| `multiple_comparison_correction.py` | Bonferroni, Holm (step-down, strictly more powerful than plain Bonferroni), and Benjamini-Hochberg FDR — for when you've tested more than one hypothesis and need to know which survivors are real. |
| `dedup_store.py` | Append-and-dedupe for an incrementally-growing CSV store, keyed on a natural key rather than row position. Closes a real, confirmed dtype-mismatch bug: an integer-looking key (like an exchange's own sequence ID) silently round-trips as `int64` after a CSV reload but stays `str` on a fresh fetch, so `drop_duplicates()` fails to recognize the duplicate across runs. |

### Architecture & Logic Flowcharts

**Walk-Forward Validation Pipeline:**
<p align="center">
  <img src="assets/walk_forward_validate_pipeline_v2.svg" alt="Walk Forward Validation Pipeline" width="50%">
</p>

**Classification Branching Logic:**
<p align="center">
  <img src="assets/classify_overfitting_branching_logic.svg" alt="Classification Branching Logic" width="75%">
</p>


## Why these five, together

They compose. A typical flow in the source pipeline: test a candidate signal → if
one whole-sample test, run `walk_forward_validator.stat_vs_zero()`, then either
`walk_forward_validator.walk_forward_validate()` for a quick single-split check or
`cpcv_validator.cpcv_validate()` for the stronger multi-path one; if a
cross-sectional panel, use `fama_macbeth.fama_macbeth_regression()` and *then*
validate the resulting cohort-estimate series the same way; if several signals were
screened at once, correct with `multiple_comparison_correction.bonferroni_correction()`
(or `holm_correction`/`benjamini_hochberg_fdr` when the batch is large and some
real signal is plausible) before trusting any single one. `dedup_store.py` is the
odd one out — infrastructure rather than statistics — included because every
signal above depends on a clean, non-duplicated input history.

## Not included

The source pipeline also has a PIT (point-in-time) fundamentals integrity checker
that follows the same discipline (survivorship bias, restatement-vintage risk),
but it's tightly coupled to that pipeline's own data-fetching/caching modules and
wouldn't run standalone — worth building your own version of the *pattern*
(validate a cached data source's integrity before any backtest is allowed to
trust it), not worth shipping the coupled code here.

## Usage

Each module is self-contained — copy the one file you need, or all five. No
`setup.py`/`pyproject.toml` provided; drop them into your own project.
`cpcv_validator.py` is the one exception to full independence: it imports from
`walk_forward_validator.py`, so copy both together if you want CPCV.

```python
import walk_forward_validator as wfv

result = wfv.walk_forward_validate(
    df, date_col="date",
    stat_fn=lambda d: wfv.stat_vs_zero(d["excess_pct"]),
    purge_days=20,  # optional: your signal's own forward-return horizon in calendar days
)
print(result.verdict)  # ROBUST / MODERATE / WEAK / OVERFITTED / INSUFFICIENT_INSAMPLE_EDGE / INSUFFICIENT_DATA
print(result.train.t_stat, result.test.t_stat, result.retention_ratio)
```

```python
import cpcv_validator as cpcv
from walk_forward_validator import stat_vs_zero

result = cpcv.cpcv_validate(
    df, date_col="date",
    stat_fn=lambda d: stat_vs_zero(d["excess_pct"]),
    n_groups=6, n_test_groups=2,  # C(6,2) = 15 evaluated paths
    purge_days=20, embargo_days=20,
)
print(result.overall_verdict)  # ROBUST / MODERATE / WEAK / OVERFITTED / INSUFFICIENT_DATA
print(result.pct_paths_robust_or_moderate, len(result.paths), result.n_splits_skipped)
```

```python
import fama_macbeth as fmb

result = fmb.fama_macbeth_regression(panel_df, cohort_col="fiscal_year", x_col="characteristic", y_col="forward_return")
print(result.n_periods, result.mean_estimate, result.t_stat, result.p_value)
```

```python
import multiple_comparison_correction as mcc

survives = mcc.bonferroni_correction(p_values=[0.01, 0.04, 0.002])
```

```python
from pathlib import Path
from dedup_store import append_dedup

append_dedup(new_rows_df, store_path=Path("my_accumulator.csv"), dedup_cols=["id"])
```

## FP rewrites

Each module has a functional-programming sibling (`walk_forward_validator_fp.py`,
`cpcv_validator_fp.py`, `fama_macbeth_fp.py`, `multiple_comparison_correction_fp.py`,
`dedup_store_fp.py`) in `tests_fp/`'s companion set at the repo root. Drop-in
compatible with the originals -- same public names, signatures, and return types,
same behavior -- restructured to be immutable throughout (frozen, slotted
dataclasses) and to express branching decision logic (`classify_overfitting`,
`classify_cpcv_overall`) as a declarative, ordered rule table evaluated via
`next(...)` instead of an if/elif ladder. Useful if you want to add a new decision
rule without touching existing branches, or want the stronger guarantee that a
`StatResult` computed on one split can never be silently mutated later in a longer
pipeline. Pick whichever style fits your own codebase; both are maintained.

## Testing

```bash
python -m pytest tests/ tests_fp/ -q
```

287 tests, no external services, no API keys, no network access required.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
