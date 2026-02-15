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
