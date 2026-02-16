# Bootstrap Confidence Intervals (BCa) Tutorial

Percentile and bias-corrected accelerated (BCa) confidence intervals for HEP toy studies and churn hazard-ratio bootstrap. When to use which, how to read diagnostics, and how to reproduce the benchmarks.

## Contents

1. [Background: Percentile vs BCa](#1-background-percentile-vs-bca)
2. [HEP: Toy-Summary CI](#2-hep-toy-summary-ci)
3. [Churn: Bootstrap Hazard-Ratio CI](#3-churn-bootstrap-hazard-ratio-ci)
4. [Python API](#4-python-api)
5. [Reading Diagnostics](#5-reading-diagnostics)
6. [When to Use BCa](#6-when-to-use-bca)
7. [Benchmark Results](#7-benchmark-results)

## 1. Background: Percentile vs BCa

Both methods construct confidence intervals from a bootstrap distribution of a statistic (fitted POI, hazard ratio, etc.). The difference is how they pick the quantile boundaries:

| Method | How It Works | When It Helps |
|---|---|---|
| **Percentile** | Takes the α/2 and 1−α/2 quantiles of the bootstrap distribution directly. | Symmetric distributions, large samples, quick default. |
| **BCa** | Adjusts quantile boundaries for median bias (*z₀*) and skewness (acceleration *a*, estimated via jackknife). Falls back to percentile if prerequisites fail. | Skewed distributions, small samples, boundary-adjacent parameters. |

NextStat implements both in `ns-inference` with automatic percentile fallback when BCa prerequisites are insufficient. Every CI output includes the effective method and diagnostic fields.

## 2. HEP: Toy-Summary CI

When running toy MC studies with `nextstat unbinned-fit-toys`, the summary statistics (mean POI, mean NLL) can optionally include confidence intervals:

```bash
# Percentile CI (default — fast, no jackknife)
nextstat unbinned-fit-toys spec.json \
  --n-toys 500 \
  --summary-ci-method percentile \
  --summary-ci-level 0.95 \
  --summary-ci-bootstrap 1000

# BCa CI (corrects for skew near boundaries)
nextstat unbinned-fit-toys spec.json \
  --n-toys 500 \
  --summary-ci-method bca \
  --summary-ci-level 0.95 \
  --summary-ci-bootstrap 1000
```

Output includes `summary.mean_ci` with the requested/effective method, confidence level, and diagnostic fields (`z0`, `acceleration`, `corrected_alpha_lo/hi`).

## 3. Churn: Bootstrap Hazard-Ratio CI

For subscription churn analysis, bootstrap CIs on Cox PH hazard ratios:

```bash
# Percentile (default)
nextstat churn bootstrap-hr data.json \
  --n-bootstrap 500 \
  --ci-method percentile \
  --conf-level 0.95

# BCa (per-coefficient bias correction)
nextstat churn bootstrap-hr data.json \
  --n-bootstrap 500 \
  --ci-method bca \
  --n-jackknife 160 \
  --conf-level 0.95
```

BCa output includes per-coefficient effective method and diagnostics with fallback reasons. The `--n-jackknife` parameter controls the number of jackknife replicates used to estimate acceleration.

## 4. Python API

```python
import nextstat as ns

# HEP: fit_toys with summary CI
results = ns.fit_toys(model, params, n_toys=500, seed=42)
# summary CI is controlled by CLI flags; Python returns raw toy results

# Churn: bootstrap HR with BCa
result = ns.churn_bootstrap_hr(
    data,
    n_bootstrap=500,
    ci_method="bca",       # "percentile" | "bca"
    n_jackknife=160,
    conf_level=0.95,
)
# result["coefficients"] → per-coefficient CI with effective_method + diagnostics
for coef in result["coefficients"]:
    print(f"{coef['name']}: HR={coef['hr']:.3f} "
          f"CI=[{coef['ci_lo']:.3f}, {coef['ci_hi']:.3f}] "
          f"method={coef['effective_method']}")
```

## 5. Reading Diagnostics

Every BCa CI output includes diagnostic fields that help assess interval quality:

| Field | Meaning | Typical Values |
|---|---|---|
| `z0` | Median bias correction (z-score of how many bootstrap replicates fall below the point estimate) | \|z₀\| < 0.25 → near-symmetric |
| `acceleration` | Skewness correction estimated from jackknife leave-one-out influence values | \|a\| < 0.05 → mild skew |
| `corrected_alpha_lo/hi` | Adjusted quantile positions after bias + acceleration correction | Shift vs nominal α/2 |
| `effective_method` | Actual method used (may fall back to percentile) | `"bca"` or `"percentile"` |
| `fallback_reason` | Why BCa was not used (if applicable) | `null` or `"insufficient_jackknife"` |

## 6. When to Use BCa

**Use BCa when:**
- POI near physical boundary (μ ≈ 0)
- Small bootstrap/toy sample (n < 200)
- Skewed estimator distribution
- Heavy censoring (churn/survival)
- Need second-order accuracy

**Percentile is fine when:**
- Large symmetric bootstrap distribution
- POI away from boundaries
- Speed matters more than precision
- Quick diagnostic / development runs

## 7. Benchmark Results

Benchmarks run on **nextstat-bench** (AMD EPYC 7502P, 64 threads). Full reproducible commands and artifacts in `docs/benchmarks/bca-hep-churn-ci-methods.md`.

### HEP Toy-Summary CI

| Metric | Percentile | BCa |
|---|---:|---:|
| Median wall time (s) | 0.033 | 0.033 |
| BCa overhead | — | ~1.00× |
| Fallback rate | — | 0% |

### Churn Bootstrap HR (Dataset-Level Calibration, 128 runs)

| Metric | Percentile | BCa |
|---|---:|---:|
| Coverage vs true HR | 0.945 | 0.951 |
| Wilson 95% CI | [0.922, 0.962] | [0.929, 0.967] |
| Median wall time (s) | 0.232 | 0.298 |
| BCa overhead | — | ~1.28× |
| Fallback rate | — | 0% |

### HEP Dataset-Level Bootstrap (128 runs, 1500 replicates)

| Scenario | Method | Coverage | Wilson 95% CI |
|---|---|---:|---:|
| Mid-POI | percentile | 0.977 | [0.933, 0.992] |
| Mid-POI | bca | 0.977 | [0.933, 0.992] |
| Boundary-low | percentile | 0.969 | [0.922, 0.988] |
| Boundary-low | bca | 0.977 | [0.933, 0.992] |

BCa shows a small coverage uplift at the boundary-low scenario (0.977 vs 0.969), consistent with its second-order correction for asymmetry near physical boundaries.

### Reproducibility

All benchmark commands, seeds, and CI gate thresholds are documented in `docs/benchmarks/bca-hep-churn-ci-methods.md`. Nightly CI runs both benchmarks and checks gates automatically via `.github/workflows/apex2-nightly-slow.yml`.

## Related Documents

- BCa benchmark runbook: `docs/benchmarks/bca-hep-churn-ci-methods.md`
- BCa spec: `docs/specs/apex2_bca_hep_churn_plan.md`
- CLI reference: `docs/references/cli.md`
- Python API reference: `docs/references/python-api.md`
