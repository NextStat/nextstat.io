# Time Series Suite

This directory contains the **time series benchmark suite** for the standalone public benchmarks repo.

Scope:
- Kalman filter / smoother / EM (local level model)
- GARCH(1,1) parameter estimation

This suite produces **publishable** JSON artifacts under pinned schemas:
- `nextstat.timeseries_benchmark_result.v1` (per case)
- `nextstat.timeseries_benchmark_suite_result.v1` (suite index)

## Cases

| Case | Kind | N | Description |
|------|------|---|-------------|
| `kalman_local_level_500` | kalman_local_level | 500 | Local level model, filter + smooth + EM |
| `kalman_local_level_5000` | kalman_local_level | 5000 | Local level model, larger series |
| `garch11_1000` | garch11 | 1000 | GARCH(1,1) on simulated returns |
| `garch11_5000` | garch11 | 5000 | GARCH(1,1) on larger return series |

## Parity

- **Kalman**: log-likelihood diff, estimated Q diff, estimated R diff vs `pykalman` (primary) or `statsmodels` (fallback)
- **GARCH**: omega/alpha/beta param diffs, log-likelihood diff vs `arch`

## Run

```bash
python3 suites/timeseries/suite.py --deterministic --out-dir out/timeseries
```

Single case:

```bash
python3 suites/timeseries/run.py --case kalman_local_level_500 --kind kalman_local_level --n 500 --out out/kalman_500.json --repeat 20
python3 suites/timeseries/run.py --case garch11_1000 --kind garch11 --n 1000 --out out/garch11_1000.json --repeat 20
```

Optional baselines (skipped with `status="warn"` if missing):
- `pykalman` for Kalman EM parity
- `statsmodels` for Kalman MLE parity (fallback)
- `arch` for GARCH(1,1) parity

## Timing

Each case is timed with N=20 repeats (configurable via `--repeat`). The artifact reports min/median/p95 wall-clock times.
