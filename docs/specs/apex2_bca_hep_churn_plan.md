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

### Iteration 5 (release-gate automation)
- Added threshold gate script:
  - `scripts/benchmarks/check_bca_ci_gates.py`
- Added CI smoke coverage for gate script pass/fail scenarios:
  - `tests/python/test_bca_ci_gates_script.py`
- Integrated BCa benchmark + gate execution into nightly Apex2 workflow:
  - `.github/workflows/apex2-nightly-slow.yml`

### Iteration 6 (controlled skew calibration harness)
- Added scenario-matrix calibration script:
  - `scripts/benchmarks/bench_bca_skew_calibration.py`
- Coverage checks now include controlled asymmetry scenarios:
  - HEP boundary pressure (`gauss_boundary_lowpoi`)
  - churn heavy-censoring small-sample regime (`heavy_censoring_small_n`)
- Added runbook commands/artifacts section in:
  - `docs/benchmarks/bca-hep-churn-ci-methods.md`
- Recorded full matrix baseline artifact:
  - `bench_results/bca_skew_calibration_2026-02-16/summary.json`
- Wired skew calibration matrix into Apex2 nightly workflow as informational artifact production.

### Iteration 7 (nextstat-bench adult rerun + interpretation hardening)
- Re-ran adult BCa benchmark matrix on `nextstat-bench` (64-thread CPU stand) with summary-only artifacts:
  - `bench_results/unbinned_summary_ci_nextstat_bench_2026-02-16/summary.json`
  - `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16/summary.json`
  - `bench_results/bca_skew_calibration_nextstat_bench_2026-02-16/summary.json`
  - `bench_results/bca_ci_gate_report_nextstat_bench_2026-02-16/summary.json`
- Added explicit binary selection support to BCa benchmark scripts:
  - `--nextstat-bin` override for `bench_unbinned_summary_ci.py`
  - `--nextstat-bin` override for `bench_churn_bootstrap_ci_methods.py`
  - `--nextstat-bin` override for `bench_bca_skew_calibration.py`
- Added HEP CI-center bias diagnostics (`median/mean center minus poi_true`) to summary outputs to disambiguate low inclusion rates.

### Iteration 8 (BCA-14 dataset-level HEP calibration)
- Added dataset-level HEP calibration harness:
  - `scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py`
- Methodology:
  - regenerate observed ROOT dataset per run,
  - fit observed POI (`unbinned-fit`),
  - run bootstrap replicates (`unbinned-fit-toys --gen mle`),
  - compute percentile + BCa intervals and coverage vs known generation truth.
- Added smoke regression test:
  - `tests/python/test_bca_hep_dataset_bootstrap_ci_smoke.py`
- Published baseline summary artifact:
  - `bench_results/hep_dataset_bootstrap_ci_2026-02-16/summary.json`
- Updated benchmark runbook with BCA-14 section and baseline table.
- Initial generator path used `root` CLI in runtime environment (later generalized in BCA-15).

### Iteration 9 (BCA-14 full-matrix on nextstat-bench + dependency bootstrap)
- Installed missing benchmark runtime on `nextstat-bench`:
  - `root` (CERN ROOT) via micromamba env (`/opt/micromamba/envs/rootenv`)
  - Python deps for ROOT/Parquet toolchain: `pyarrow`, `uproot`, `awkward`, `numpy`, `scipy`, `pandas`, `pyyaml`
- Produced native Linux `nextstat` binary on `nextstat-bench` from workspace snapshot and reran BCA-14 with adult parameters:
  - `runs=32`, `n_bootstrap=1500`, `threads=56`
- Published summary-only artifact:
  - `bench_results/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_full/summary.json`
- Result snapshot:
  - both HEP scenarios: coverage `0.96875` for percentile and BCa
  - BCa effective in all runs (`fallback_count=0`)
  - BCa median width slightly smaller than percentile in both scenarios.
- Rootless portability follow-up tracked as:
  - `BCA-15` (`d00c01cf-ca8c-424c-b7c6-aaa414216f25`) (closed in Iteration 14)

### Iteration 10 (BCA-14 long-run statistical stabilization)
- Ran long-run dataset-level calibration on `nextstat-bench` with:
  - `runs=128`, `n_bootstrap=1500`, `threads=56`
- Published summary-only artifact:
  - `bench_results/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_longrun/summary.json`
- Statistical interpretation (binomial Wilson intervals):
  - `gauss_mu_mid`: coverage `0.9766` for both percentile and BCa, Wilson 95% `[0.9334, 0.9920]`
  - `gauss_mu_boundary_low`: percentile `0.9688` (`[0.9224, 0.9878]`), BCa `0.9766` (`[0.9334, 0.9920]`)
- Operational behavior:
  - `fallback_count=0` in all methods/scenarios
  - BCa median width remained lower than percentile in both scenarios.

### Iteration 11 (Churn long-run calibration on nextstat-bench)
- Extended churn benchmark script to support explicit coverage accounting against default generator truth:
  - `scripts/benchmarks/bench_churn_bootstrap_ci_methods.py`
  - added `--use-default-truth` and summary fields:
    - `coverage_vs_true_hr`, `coverage_hits`, `coverage_total`
- Ran churn long-run matrix on `nextstat-bench`:
  - `runs=128`, `n_customers=4000`, `n_bootstrap=800`, `n_jackknife=240`
- Published summary-only artifact:
  - `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_longrun/summary.json`
- Result snapshot:
  - coverage parity: percentile `0.7500` vs BCa `0.7500` (both `384/512`)
  - Wilson 95% coverage interval: `[0.7107, 0.7856]`
  - BCa remained fallback-free and slightly narrower (`0.079411` vs `0.079607` mean width)
  - BCa runtime overhead remained bounded (`0.3019s` vs `0.2349s` median wall, ~`1.29x`).
- Investigation outcome:
  - this run used a **single fixed observed dataset** across all seeds (conditional bootstrap variability only), so the reported "coverage" was not dataset-level calibration.

### Iteration 12 (BCA-19 resolution: dataset-level churn calibration)
- Upgraded churn benchmark harness with:
  - dataset-level mode `--regenerate-data-per-run`
  - generator controls surfaced in script (`--n-cohorts`, `--max-time`, `--treatment-fraction`)
  - per-coefficient summary diagnostics (`per_coefficient` coverage/mean point/mean width).
- Ran two dataset-level matrices on `nextstat-bench` (`runs=128`, `n_customers=4000`, `n_bootstrap=800`, `n_jackknife=240`):
  - default generator regime (`treatment_fraction=0.3`):
    - percentile coverage `0.9453` (`484/512`, Wilson `[0.9221, 0.9619]`)
    - BCa coverage `0.9512` (`487/512`, Wilson `[0.9289, 0.9667]`)
  - no-treatment identifiability control (`treatment_fraction=0.0`):
    - percentile coverage `0.9297` (`476/512`, Wilson `[0.9042, 0.9488]`)
    - BCa coverage `0.9297` (`476/512`, Wilson `[0.9042, 0.9488]`)
- Published summary-only artifacts:
  - `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat30/summary.json`
  - `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat00/summary.json`
- Conclusion:
  - prior `0.75` result was a fixed-dataset artifact, not BCa failure.
  - in dataset-level calibration, coverage is near nominal for both methods and `plan_premium` no longer shows deterministic miss.

### Iteration 13 (BCA-20: Apex2 nightly wiring to dataset-level churn calibration)
- Updated Apex2 nightly workflow churn benchmark call in:
  - `.github/workflows/apex2-nightly-slow.yml`
- Nightly churn benchmark now runs with dataset-level calibration flags:
  - `--regenerate-data-per-run`
  - `--use-default-truth`
  - generator controls pinned for stability:
    - `--n-cohorts 48`
    - `--max-time 12`
    - `--treatment-fraction 0.3`
- Gate wiring remains unchanged (`artifacts/bca_ci/churn_bootstrap_ci_methods/summary.json`), but now points to dataset-level churn outputs by construction.
- Fixed-dataset churn coverage is kept as optional diagnostic mode only and is no longer the nightly calibration default.

### Iteration 14 (BCA-15: rootless HEP observed-data generator)
- Upgraded `scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py` with root-writer selection:
  - new flag: `--root-writer auto|uproot|root-cli` (default `auto`)
  - `auto` mode prefers rootless `uproot` writer and falls back to `root` CLI when needed.
- Kept compatibility with legacy `root` CLI macro writer path for environments where `uproot` is not installed.
- Updated smoke test:
  - `tests/python/test_bca_hep_dataset_bootstrap_ci_smoke.py`
  - now accepts either writer path (skip only when both `uproot` and `root` CLI are unavailable).
- Result:
  - dataset-level HEP benchmark no longer requires system ROOT by default when `uproot` is available.

### Next iterations
- Expand dataset-level calibration matrices to larger `runs` for tighter coverage uncertainty bounds.
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

### Iteration 5 checks
- `python3 scripts/benchmarks/check_bca_ci_gates.py --hep-summary bench_results/unbinned_summary_ci_2026-02-15/summary.json --churn-summary bench_results/churn_bootstrap_ci_methods_2026-02-15/summary.json --out-json /tmp/bca_ci_gate_report.json --out-md /tmp/bca_ci_gate_report.md`
- `pytest -q tests/python/test_bca_ci_gates_script.py`
- `pytest -q tests/python/test_bca_bench_results_policy.py`

### Iteration 6 checks
- `python3 scripts/benchmarks/bench_bca_skew_calibration.py --runs 1 --hep-n-toys 8 --hep-summary-ci-bootstrap 16 --churn-n-bootstrap 12 --churn-n-jackknife 8 --out-dir /tmp/bca_skew_calib_smoke`

### Iteration 7 checks
- `python3 scripts/benchmarks/bench_unbinned_summary_ci.py --runs 1 --n-toys 20 --summary-ci-bootstrap 32 --threads 2 --nextstat-bin target/debug/nextstat --out-dir /tmp/bca_hep_smoke_bias`
- `python3 scripts/benchmarks/bench_bca_skew_calibration.py --runs 1 --hep-n-toys 20 --hep-summary-ci-bootstrap 32 --hep-threads 2 --churn-n-bootstrap 40 --churn-n-jackknife 20 --nextstat-bin target/debug/nextstat --out-dir /tmp/bca_skew_smoke_bias`
- `python3 scripts/benchmarks/check_bca_ci_gates.py --hep-summary bench_results/unbinned_summary_ci_nextstat_bench_2026-02-16/summary.json --churn-summary bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16/summary.json --out-json /tmp/bca_gate_nextstat_bench.json --out-md /tmp/bca_gate_nextstat_bench.md`

### Iteration 8 checks
- `python3 scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py --runs 2 --n-events 120 --n-bootstrap 40 --threads 2 --nextstat-bin target/debug/nextstat --out-dir /tmp/hep_dataset_bca_smoke`
- `pytest -q tests/python/test_bca_hep_dataset_bootstrap_ci_smoke.py`

### Iteration 9 checks
- `pytest -q tests/python/test_bca_bench_results_policy.py`
- Remote full matrix on `nextstat-bench`:
  - `python /opt/nextstat_src_min_2026-02-16/scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat --runs 32 --n-bootstrap 1500 --threads 56 --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_full`

### Iteration 10 checks
- `pytest -q tests/python/test_bca_bench_results_policy.py`
- Remote long-run matrix on `nextstat-bench`:
  - `python /opt/nextstat_src_min_2026-02-16/scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat --runs 128 --n-bootstrap 1500 --threads 56 --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_longrun`

### Iteration 11 checks
- `python3 -m py_compile scripts/benchmarks/bench_churn_bootstrap_ci_methods.py`
- `python3 scripts/benchmarks/bench_churn_bootstrap_ci_methods.py --n-customers 300 --n-bootstrap 40 --n-jackknife 30 --runs 1 --seed0 321 --use-default-truth --nextstat-bin target/release/nextstat --out-dir /tmp/churn_ci_cov_smoke`
- `pytest -q tests/python/test_bca_bench_results_policy.py`
- Remote long-run matrix on `nextstat-bench`:
  - `python /opt/nextstat_src_min_2026-02-16/scripts/benchmarks/bench_churn_bootstrap_ci_methods.py --n-customers 4000 --n-bootstrap 800 --conf-level 0.95 --n-jackknife 240 --runs 128 --seed0 9100 --use-default-truth --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_longrun`

### Iteration 12 checks
- `python3 -m py_compile scripts/benchmarks/bench_churn_bootstrap_ci_methods.py`
- `python3 scripts/benchmarks/bench_churn_bootstrap_ci_methods.py --n-customers 300 --n-bootstrap 40 --n-jackknife 30 --runs 2 --seed0 321 --use-default-truth --regenerate-data-per-run --nextstat-bin target/release/nextstat --out-dir /tmp/churn_ci_cov_smoke3`
- `pytest -q tests/python/test_bca_bench_results_policy.py`
- Remote dataset-level matrices on `nextstat-bench`:
  - `python /opt/nextstat_src_min_2026-02-16/scripts/benchmarks/bench_churn_bootstrap_ci_methods.py --n-customers 4000 --n-cohorts 6 --max-time 24 --treatment-fraction 0.3 --n-bootstrap 800 --conf-level 0.95 --n-jackknife 240 --runs 128 --seed0 9400 --use-default-truth --regenerate-data-per-run --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat30`
  - `python /opt/nextstat_src_min_2026-02-16/scripts/benchmarks/bench_churn_bootstrap_ci_methods.py --n-customers 4000 --n-cohorts 6 --max-time 24 --treatment-fraction 0.0 --n-bootstrap 800 --conf-level 0.95 --n-jackknife 240 --runs 128 --seed0 9600 --use-default-truth --regenerate-data-per-run --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat00`

### Iteration 13 checks
- `rg -n "bench_churn_bootstrap_ci_methods.py|regenerate-data-per-run|use-default-truth|n-cohorts 48|max-time 12|treatment-fraction 0.3" .github/workflows/apex2-nightly-slow.yml`

### Iteration 14 checks
- `python3 -m py_compile scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py`
- `pytest -q tests/python/test_bca_hep_dataset_bootstrap_ci_smoke.py`

### Upcoming verification gates
- Run full-size benchmark matrix and publish summary tables (not only smoke runs).
