# Apex2 Plan: BCa Bootstrap CI (HEP-first, churn-enabled)

## Context
- Objective: introduce BCa confidence intervals (Efron, 1987) in NextStat.
- Product priority: **HEP first**, then churn where statistically applicable.
- Methodology: **Apex2** lifecycle:
  1. Planning
  2. Exploration
  3. Execution
  4. Verification

## 1) Planning
### Goal
Create a reusable BCa engine in `ns-inference`, then integrate it into selected workflows.

### Scope split
- Primary (HEP):
  - toy-derived estimator summaries where bootstrap asymmetry is meaningful.
  - first candidate: `unbinned-fit-toys` summary fields for POI estimator distribution.
- Secondary (churn):
  - `churn bootstrap-hr` currently percentile-only, add BCa method option.

### Non-goals (v1)
- Replacing CLs asymptotic or toy test-statistic machinery wholesale.
- Multivariate joint BCa regions.

### Output contract draft (v1)
All BCa-enabled outputs should carry explicit method metadata and diagnostics.

```json
{
  "ci": {
    "method": "percentile|bca",
    "conf_level": 0.95,
    "lower": 0.0,
    "upper": 0.0,
    "diagnostics": {
      "z0": 0.0,
      "acceleration": 0.0,
      "alpha_low": 0.025,
      "alpha_high": 0.975,
      "alpha_low_adj": 0.02,
      "alpha_high_adj": 0.98,
      "n_bootstrap": 1000,
      "n_jackknife": 1000,
      "fallback_reason": null
    }
  }
}
```

Fallback policy:
- If BCa is requested but prerequisites fail (insufficient finite samples, degenerate jackknife, etc.),
  return percentile interval and set `fallback_reason`.
- Default method remains unchanged unless explicitly switched by product decision.

## 2) Exploration
### Code-path findings
- HEP toy summary output exists in:
  - `crates/ns-cli/src/main.rs` (`cmd_unbinned_fit_toys`)
  - current summary contains `q16/q50/q84` and pull summaries.
- Churn bootstrap path is percentile-only:
  - `crates/ns-inference/src/churn.rs` (`bootstrap_hazard_ratios`)
- Existing quantile helpers are duplicated in multiple modules; good target for reuse.

### BCa applicability decision table (v1)
- `unbinned-fit-toys` POI summary intervals: **APPLY BCa** (after adding stable jackknife path for target estimator).
- `churn bootstrap-hr`: **APPLY BCa** (log-HR scale, fallback policy required).
- CLs limit scan / qmu / qtilde pipeline: **DO NOT APPLY by default** (different inferential target).

## 3) Execution (incremental)
### Iteration 1 (this change set)
- Add reusable module:
  - `crates/ns-inference/src/bootstrap_ci.rs`
  - percentile + BCa primitives:
    - quantiles,
    - `z0`,
    - jackknife acceleration `a`,
    - adjusted alpha mapping,
    - BCa interval with diagnostics.
- Export through `crates/ns-inference/src/lib.rs`.

### Iteration 2 (HEP summary integration)
- `unbinned-fit-toys` gets opt-in summary CI controls:
  - `--summary-ci-method percentile|bca`
  - `--summary-ci-level` (default `0.68`)
  - `--summary-ci-bootstrap` (default `1000`)
- Added `summary.mean_ci` block (when opt-in flag is set):
  - target: `mean` over converged finite `poi_hat`
  - requested/effective method metadata
  - BCa diagnostics (`z0`, `acceleration`, adjusted alphas, counts)
  - fallback-to-percentile with `diagnostics.fallback_reason` when BCa prerequisites fail.
- Added CLI test coverage for BCa opt-in path.

### Iteration 3 (churn integration: core + CLI + Python)
- Core:
  - added `bootstrap_hazard_ratios_with_method(..., ci_method, n_jackknife)` in `ns-inference`.
  - kept `bootstrap_hazard_ratios(...)` as backward-compatible percentile wrapper.
  - BCa computed on log-HR; fallback to percentile with explicit reason.
  - added per-coefficient diagnostics and effective-method metadata.
- CLI:
  - `nextstat churn bootstrap-hr` supports `--ci-method percentile|bca` and `--n-jackknife`.
  - output includes root-level requested method and per-coefficient diagnostics.
- Python:
  - `nextstat.churn_bootstrap_hr(..., ci_method=\"percentile\", n_jackknife=200)` parity.
  - returned payload includes method and diagnostics arrays.

### Iteration 4 (validation + benchmark artifacts)
- Added unit tests for churn percentile/BCa bootstrap paths.
- Added CLI integration tests for churn bootstrap method selection.
- Added benchmark scripts + runbook:
  - `scripts/benchmarks/bench_unbinned_summary_ci.py`
  - `scripts/benchmarks/bench_churn_bootstrap_ci_methods.py`
  - `docs/benchmarks/bca-hep-churn-ci-methods.md`

### Next iterations
- Expand full-size benchmark matrices on production-like hardware tiers.
- Define release-gate thresholds for BCa overhead/fallback rates.
- Add stricter statistical calibration experiments (coverage under controlled skew scenarios).

## 4) Verification
### Iteration 1 checks
- Unit tests for:
  - quantile behavior,
  - BCa alpha mapping identity at `z0=0, a=0`,
  - BCa interval smoke and finite diagnostics.

### Iteration 2 checks
- `cargo test -p ns-cli unbinned_fit_toys_summary_mean_ci_bca_opt_in -- --nocapture`
- `cargo check -p ns-cli`

### Iteration 3 checks
- `cargo test -p ns-inference bootstrap_hazard_ratios_ -- --nocapture`
- `cargo test -p ns-cli churn_bootstrap_hr_ -- --nocapture`
- `cargo check -p ns-py`

### Iteration 4 checks (smoke)
- `python3 scripts/benchmarks/bench_unbinned_summary_ci.py --runs 1 --n-toys 8 --summary-ci-bootstrap 16 --out-dir /tmp/bca_hep_smoke`
- `python3 scripts/benchmarks/bench_churn_bootstrap_ci_methods.py --runs 1 --n-customers 120 --n-bootstrap 12 --n-jackknife 8 --out-dir /tmp/bca_churn_smoke`

### Upcoming verification gates
- Run full-size benchmark matrix and publish summary tables (not only smoke runs).
- Add release-gate thresholds for BCa overhead/fallback rates.
