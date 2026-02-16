# BCa vs Percentile CI Benchmarks (HEP + Churn)

This runbook defines reproducible benchmarks for BCa vs percentile confidence intervals in two target workflows:

- HEP: `unbinned-fit-toys` summary CI (`summary.mean_ci`)
- Churn: `churn bootstrap-hr` hazard-ratio CIs

## Prerequisites

```bash
cargo build -p ns-cli
```

Optional (for faster runs):

```bash
cargo build -p ns-cli --release
```

When running on remote stands, pin the exact binary explicitly:

```bash
--nextstat-bin /absolute/path/to/nextstat
```

## 1) HEP benchmark (unbinned summary CI)

Script:

- `scripts/benchmarks/bench_unbinned_summary_ci.py`

Command:

```bash
python scripts/benchmarks/bench_unbinned_summary_ci.py \
  --runs 12 \
  --n-toys 200 \
  --summary-ci-bootstrap 400 \
  --summary-ci-level 0.68 \
  --threads 1 \
  --out-dir bench_results/unbinned_summary_ci_2026-02-15
```

Artifacts:

- `bench_results/unbinned_summary_ci_2026-02-15/raw_runs.json`
- `bench_results/unbinned_summary_ci_2026-02-15/summary.json`
- `bench_results/unbinned_summary_ci_2026-02-15/summary.md`

## 2) Churn benchmark (bootstrap HR CI method)

Script:

- `scripts/benchmarks/bench_churn_bootstrap_ci_methods.py`

Command:

```bash
python scripts/benchmarks/bench_churn_bootstrap_ci_methods.py \
  --n-customers 2000 \
  --n-bootstrap 500 \
  --n-jackknife 160 \
  --runs 8 \
  --conf-level 0.95 \
  --out-dir bench_results/churn_bootstrap_ci_methods_2026-02-15
```

Artifacts:

- `bench_results/churn_bootstrap_ci_methods_2026-02-15/raw_runs.json`
- `bench_results/churn_bootstrap_ci_methods_2026-02-15/summary.json`
- `bench_results/churn_bootstrap_ci_methods_2026-02-15/summary.md`

## Results (2026-02-15)

| Workflow | Metric | Percentile | BCa | Notes |
|---|---|---:|---:|---|
| HEP unbinned-fit-toys | Median wall (s) | 1.802 | 1.679 | BCa median overhead: `0.931x` vs percentile (noise-level difference) |
| HEP unbinned-fit-toys | Mean width (`summary.mean_ci`) | 0.004444 | 0.004446 | Practically identical on this setup |
| HEP unbinned-fit-toys | Coverage vs `poi_true` | 0.083 | 0.083 | Both methods: `1/12` runs include `poi_true` |
| HEP unbinned-fit-toys | BCa fallback count | N/A | 0 | Effective BCa in `12/12` runs |
| Churn bootstrap-hr | Median wall (s) | 4.466 | 6.075 | BCa median overhead: `1.360x` |
| Churn bootstrap-hr | Mean interval width | 0.112353 | 0.113721 | Mean over all coefficients |
| Churn bootstrap-hr | BCa fallback count | N/A | 0 | Effective BCa coeff count: `32` (8 runs × 4 coefs) |

## Notes

- BCa is implemented with percentile fallback when BCa prerequisites are insufficient for a coefficient (diagnostic field: `fallback_reason`).
- HEP defaults are unchanged unless `--summary-ci-method` is explicitly set.
- Churn defaults remain percentile unless `--ci-method bca` is explicitly set.

## 3) CI gate check

Script:

- `scripts/benchmarks/check_bca_ci_gates.py`

Command:

```bash
python scripts/benchmarks/check_bca_ci_gates.py \
  --hep-summary bench_results/unbinned_summary_ci_2026-02-15/summary.json \
  --churn-summary bench_results/churn_bootstrap_ci_methods_2026-02-15/summary.json \
  --out-json bench_results/bca_ci_gate_report_2026-02-15.json \
  --out-md bench_results/bca_ci_gate_report_2026-02-15.md \
  --hep-max-overhead 1.25 \
  --churn-max-overhead 1.75 \
  --hep-max-fallback-rate 0.05 \
  --churn-max-fallback-rate 0.05 \
  --hep-min-effective-bca-rate 0.95 \
  --churn-min-effective-bca-rate 0.95
```

Default gate thresholds:

- HEP overhead (`bca/percentile`) <= `1.25x`
- Churn overhead (`bca/percentile`) <= `1.75x`
- HEP fallback rate <= `0.05`
- Churn fallback rate <= `0.05`
- HEP effective BCa rate >= `0.95`
- Churn effective BCa rate >= `0.95`

Automation:

- `.github/workflows/apex2-nightly-slow.yml` runs both benchmark scripts and this gate checker.
- In nightly mode, churn coverage is computed in dataset-level calibration mode (`--regenerate-data-per-run --use-default-truth`) with pinned generator controls (`--n-cohorts 48 --max-time 12 --treatment-fraction 0.3`).
- `.github/workflows/apex2-nightly-slow.yml` also runs `bench_bca_skew_calibration.py` and stores results under `artifacts/bca_skew_calibration/*` (informational matrix, no extra gate threshold yet).

## 4) Skew calibration matrix (controlled scenarios)

Script:

- `scripts/benchmarks/bench_bca_skew_calibration.py`

What it runs:

- HEP scenarios:
  - `gauss_midpoi` (POI away from boundary)
  - `gauss_boundary_lowpoi` (POI near lower bound, stronger asymmetry pressure)
- Churn scenarios:
  - `baseline` (`n_customers=2000`, `max_time=24`)
  - `heavy_censoring_small_n` (`n_customers=400`, `max_time=6`)

Command (full matrix):

```bash
python scripts/benchmarks/bench_bca_skew_calibration.py \
  --runs 8 \
  --seed0 100 \
  --hep-n-toys 120 \
  --hep-summary-ci-bootstrap 300 \
  --hep-summary-ci-level 0.68 \
  --churn-n-bootstrap 300 \
  --churn-n-jackknife 120 \
  --churn-conf-level 0.95 \
  --out-dir bench_results/bca_skew_calibration_2026-02-16
```

Artifacts:

- `bench_results/bca_skew_calibration_2026-02-16/raw_runs.json`
- `bench_results/bca_skew_calibration_2026-02-16/summary.json`
- `bench_results/bca_skew_calibration_2026-02-16/summary.md`

Smoke command:

```bash
python scripts/benchmarks/bench_bca_skew_calibration.py \
  --runs 1 \
  --hep-n-toys 8 \
  --hep-summary-ci-bootstrap 16 \
  --churn-n-bootstrap 12 \
  --churn-n-jackknife 8 \
  --out-dir /tmp/bca_skew_calib_smoke
```

Results (2026-02-16, full matrix):

| Workflow | Scenario | Metric | Percentile | BCa | Notes |
|---|---|---|---:|---:|---|
| HEP unbinned-fit-toys | `gauss_midpoi` | Coverage vs `poi_true` | 1.000 | 1.000 | both methods identical in this regime |
| HEP unbinned-fit-toys | `gauss_midpoi` | Median wall (s) | 0.969 | 0.954 | BCa ~`0.984x` vs percentile |
| HEP unbinned-fit-toys | `gauss_boundary_lowpoi` | Coverage vs `poi_true` | 0.750 | 0.625 | boundary pressure increases asymmetry |
| HEP unbinned-fit-toys | `gauss_boundary_lowpoi` | Median wall (s) | 0.127 | 0.129 | BCa ~`1.010x` vs percentile |
| Churn bootstrap-hr | `baseline` | Coverage vs true HR | 0.969 | 0.969 | both methods `31/32` coefficient hits |
| Churn bootstrap-hr | `baseline` | Median wall (s) | 1.214 | 1.819 | BCa ~`1.499x` |
| Churn bootstrap-hr | `heavy_censoring_small_n` | Coverage vs true HR | 0.938 | 0.938 | both methods `30/32` coefficient hits |
| Churn bootstrap-hr | `heavy_censoring_small_n` | Median wall (s) | 0.191 | 0.267 | BCa ~`1.402x` |

## Artifact policy (repo)

For benchmark snapshots under `bench_results/*`, we keep only:

- `summary.json`

Generated/intermediate files are intentionally not tracked:

- `raw_runs.json`
- `summary.md`
- `churn_data.json`
- `unbinned_spec_summary_ci.json`

Rationale: keep the repository slim while preserving the machine-readable summary required for CI gates and documentation tables.

## NextStat Bench Adult Run (2026-02-16)

Host profile:

- `nextstat-bench` (`AMD EPYC 7502P`, `64` CPU threads, no CUDA required for these BCa paths)

Artifacts kept in repo (summary-only policy):

- `bench_results/unbinned_summary_ci_nextstat_bench_2026-02-16/summary.json`
- `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16/summary.json`
- `bench_results/bca_skew_calibration_nextstat_bench_2026-02-16/summary.json`
- `bench_results/bca_ci_gate_report_nextstat_bench_2026-02-16/summary.json`

### Main matrix (adult parameters)

| Workflow | Metric | Percentile | BCa | Notes |
|---|---|---:|---:|---|
| HEP unbinned-fit-toys (`runs=32`, `n_toys=300`, `ci_bootstrap=600`, `threads=32`) | Coverage vs `poi_true` | 0.219 | 0.250 | BCa +1 hit out of 32 |
| HEP unbinned-fit-toys | Median wall (s) | 0.0333 | 0.0332 | BCa overhead ~`1.00x` |
| HEP unbinned-fit-toys | Median CI center minus `poi_true` | +0.002235 | +0.002209 | center bias dominates half-width (~0.00183) |
| Churn bootstrap-hr (`runs=16`, `n_customers=4000`, `n_bootstrap=800`, `n_jackknife=240`) | Mean interval width | 0.082589 | 0.082940 | near-parity, BCa slightly wider |
| Churn bootstrap-hr | Median wall (s) | 0.2314 | 0.2945 | BCa overhead ~`1.27x` |

Gate check on these adult summaries:

- `scripts/benchmarks/check_bca_ci_gates.py` => `PASS`

### Churn long-run (fixed dataset, diagnostic only)

Command:

```bash
python scripts/benchmarks/bench_churn_bootstrap_ci_methods.py \
  --n-customers 4000 \
  --n-cohorts 6 \
  --max-time 24 \
  --treatment-fraction 0.3 \
  --n-bootstrap 800 \
  --conf-level 0.95 \
  --n-jackknife 240 \
  --runs 128 \
  --seed0 9100 \
  --use-default-truth \
  --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat \
  --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_longrun
```

Artifact (summary-only policy):

- `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_longrun/summary.json`

Results (`nextstat-bench`, 2026-02-16 long-run):

| Method | Coverage vs true HR | Wilson 95% CI for coverage | Median wall (s) | Mean interval width | Fallback total |
|---|---:|---:|---:|---:|---:|
| percentile | 0.7500 | [0.7107, 0.7856] | 0.2349 | 0.079607 | 0 |
| bca | 0.7500 | [0.7107, 0.7856] | 0.3019 | 0.079411 | 0 |

Notes:

- Coverage in this mode is **conditional on one fixed observed dataset** (bootstrap seeds vary, dataset does not). Treat this as diagnostic stress-test, not as frequentist calibration.
- BCa stayed fallback-free (`fallback_total=0`) and slightly narrower on average.

### Churn dataset-level calibration (`nextstat-bench`)

Command (default generator with treatment):

```bash
python scripts/benchmarks/bench_churn_bootstrap_ci_methods.py \
  --n-customers 4000 \
  --n-cohorts 6 \
  --max-time 24 \
  --treatment-fraction 0.3 \
  --n-bootstrap 800 \
  --conf-level 0.95 \
  --n-jackknife 240 \
  --runs 128 \
  --seed0 9400 \
  --use-default-truth \
  --regenerate-data-per-run \
  --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat \
  --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat30
```

Command (no-treatment identifiability control):

```bash
python scripts/benchmarks/bench_churn_bootstrap_ci_methods.py \
  --n-customers 4000 \
  --n-cohorts 6 \
  --max-time 24 \
  --treatment-fraction 0.0 \
  --n-bootstrap 800 \
  --conf-level 0.95 \
  --n-jackknife 240 \
  --runs 128 \
  --seed0 9600 \
  --use-default-truth \
  --regenerate-data-per-run \
  --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat \
  --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat00
```

Artifacts (summary-only policy):

- `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat30/summary.json`
- `bench_results/churn_bootstrap_ci_methods_nextstat_bench_2026-02-16_datasetlevel_treat00/summary.json`

Results (`nextstat-bench`, 2026-02-16 dataset-level):

| Scenario | Method | Coverage vs true HR | Wilson 95% CI | Median wall (s) | Mean width | Fallback total |
|---|---|---:|---:|---:|---:|---:|
| `treatment_fraction=0.3` | percentile | 0.9453 (484/512) | [0.9221, 0.9619] | 0.2322 | 0.076676 | 0 |
| `treatment_fraction=0.3` | bca | 0.9512 (487/512) | [0.9289, 0.9667] | 0.2976 | 0.076831 | 0 |
| `treatment_fraction=0.0` | percentile | 0.9297 (476/512) | [0.9042, 0.9488] | 0.2482 | 0.074484 | 0 |
| `treatment_fraction=0.0` | bca | 0.9297 (476/512) | [0.9042, 0.9488] | 0.3123 | 0.074631 | 0 |

Per-coefficient note:

- In dataset-level mode, `plan_premium` no longer shows deterministic miss; mean point estimate stays near truth (`~0.401` vs `0.400`) and per-coefficient coverage is in expected Monte Carlo range.

### Controlled skew matrix (adult parameters)

| Workflow | Scenario | Coverage (Percentile) | Coverage (BCa) | Interpretation |
|---|---|---:|---:|---|
| HEP | `gauss_midpoi` | 0.000 | 0.0417 | CI center bias ~`+0.0031` with median width ~`0.0044` (half-width ~`0.0022`) |
| HEP | `gauss_boundary_lowpoi` | 0.375 | 0.375 | CI center bias ~`+0.0008` with median width ~`0.00137` (half-width ~`0.00069`) |
| Churn | `baseline` | 0.9375 | 0.9375 | parity, close to target `0.95` |
| Churn | `heavy_censoring_small_n` | 0.9688 | 0.9792 | BCa slight uplift (+1 hit out of 96) |

Interpretation note:

- For HEP here we are checking inclusion of `poi_true` by CI on **mean fitted toy POI** (`summary.mean_ci`). This metric is diagnostic for estimator bias and is not used as a hard release gate; gate thresholds focus on overhead/fallback/effective-BCa rates.

## 5) HEP dataset-level bootstrap calibration (BCA-14)

Script:

- `scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py`

What it does:

- Regenerates a fresh observed ROOT dataset per run (Gaussian toy sample) via rootless `uproot` writer (default) or `root` CLI fallback.
- Fits observed POI with `nextstat unbinned-fit`.
- Runs bootstrap replicates with `nextstat unbinned-fit-toys --gen mle`.
- Computes percentile and BCa intervals from bootstrap `poi_hat`.
- Evaluates coverage against known generation truth (`mu_true`).

Command (baseline):

```bash
python scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py \
  --runs 12 \
  --seed0 100 \
  --n-events 300 \
  --n-bootstrap 200 \
  --conf-level 0.95 \
  --threads 8 \
  --root-writer auto \
  --nextstat-bin target/release/nextstat \
  --out-dir bench_results/hep_dataset_bootstrap_ci_2026-02-16
```

Artifacts:

- `bench_results/hep_dataset_bootstrap_ci_2026-02-16/summary.json`

Results (2026-02-16 baseline):

| Scenario | Method | Coverage vs true POI | Median width | Median wall total (s) | Median center minus true |
|---|---|---:|---:|---:|---:|
| `gauss_mu_mid` | percentile | 0.917 | 6.4693 | 0.0169 | +0.3579 |
| `gauss_mu_mid` | bca | 0.917 | 6.2231 | 0.0169 | +0.7093 |
| `gauss_mu_boundary_low` | percentile | 0.917 | 6.4825 | 0.0165 | +0.3230 |
| `gauss_mu_boundary_low` | bca | 0.917 | 6.2299 | 0.0165 | +0.6848 |

Command (`nextstat-bench` adult rerun):

```bash
python scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py \
  --runs 32 \
  --seed0 5000 \
  --n-events 400 \
  --sigma-true 30 \
  --conf-level 0.95 \
  --n-bootstrap 1500 \
  --threads 56 \
  --root-writer auto \
  --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat \
  --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_full
```

Artifact (summary-only policy):

- `bench_results/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_full/summary.json`

Results (`nextstat-bench`, 2026-02-16):

| Scenario | Method | Coverage vs true POI | Median width | Median wall total (s) | Median center minus true |
|---|---|---:|---:|---:|---:|
| `gauss_mu_mid` | percentile | 0.969 | 5.9378 | 0.0583 | -0.1642 |
| `gauss_mu_mid` | bca | 0.969 | 5.8949 | 0.0583 | -0.0144 |
| `gauss_mu_boundary_low` | percentile | 0.969 | 5.9231 | 0.0559 | -0.1950 |
| `gauss_mu_boundary_low` | bca | 0.969 | 5.9007 | 0.0559 | -0.0362 |

Command (`nextstat-bench` long-run calibration):

```bash
python scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py \
  --runs 128 \
  --seed0 7000 \
  --n-events 400 \
  --sigma-true 30 \
  --conf-level 0.95 \
  --n-bootstrap 1500 \
  --threads 56 \
  --root-writer auto \
  --nextstat-bin /opt/nextstat_src_min_2026-02-16/target/release/nextstat \
  --out-dir /opt/nextstat_src_min_2026-02-16/bench_out/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_longrun
```

Artifact (summary-only policy):

- `bench_results/hep_dataset_bootstrap_ci_nextstat_bench_2026-02-16_longrun/summary.json`

Results (`nextstat-bench`, 2026-02-16 long-run):

| Scenario | Method | Coverage vs true POI | Wilson 95% CI for coverage | Median width | Median wall total (s) |
|---|---|---:|---:|---:|---:|
| `gauss_mu_mid` | percentile | 0.9766 | [0.9334, 0.9920] | 5.9014 | 0.0578 |
| `gauss_mu_mid` | bca | 0.9766 | [0.9334, 0.9920] | 5.7931 | 0.0578 |
| `gauss_mu_boundary_low` | percentile | 0.9688 | [0.9224, 0.9878] | 5.8672 | 0.0563 |
| `gauss_mu_boundary_low` | bca | 0.9766 | [0.9334, 0.9920] | 5.7971 | 0.0563 |

Notes:

- This benchmark is dataset-level (new observed data each run), unlike `summary.mean_ci` diagnostics from fixed-spec toy summaries.
- In both baseline and `nextstat-bench` runs BCa was effective in all runs (`fallback_count=0`).
- Rootless writer is available (`--root-writer auto|uproot`) with `root` CLI fallback (`--root-writer root-cli`).
