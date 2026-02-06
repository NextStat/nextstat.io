# Benchmarks

NextStat has two benchmark layers:

1. **Rust micro-benchmarks** (Criterion.rs) — low-level NLL, gradient, fit kernels
2. **Python end-to-end benchmarks** (Apex2) — full GLM fit/predict, pyhf parity, regression baselines

## Rust Micro-Benchmarks (Criterion)

[Criterion.rs](https://crates.io/crates/criterion) benchmarks live in `crates/ns-inference/benches/`.

Available benchmarks:

| Bench file | What it measures |
|------------|-----------------|
| `mle_benchmark.rs` | HistFactory MLE fit (single + batch) |
| `glm_fit_predict_benchmark.rs` | GLM fit + predict for all families |
| `regression_benchmark.rs` | GLM regression NLL/gradient |
| `nuts_benchmark.rs` | NUTS sampler (warmup + sampling) |
| `kalman_benchmark.rs` | Kalman filter/smoother/EM |
| `hier_benchmark.rs` | Hierarchical model NLL/gradient |

### Local Runs

Run all benches (slow):

```bash
cargo bench --workspace
```

Run a specific bench:

```bash
cargo bench -p ns-inference --bench mle_benchmark
```

Criterion writes HTML reports to `target/criterion/**/report/index.html`.

### Quick Mode

For fast iteration (less stable numbers):

```bash
cargo bench -p ns-inference --bench mle_benchmark -- --quick
```

Use `--quick` for CI smoke runs. Do not use quick mode for published numbers.

### Baselines (Criterion)

Save a baseline:

```bash
cargo bench -p ns-inference --bench mle_benchmark -- --save-baseline main
```

Compare against a baseline:

```bash
cargo bench -p ns-inference --bench mle_benchmark -- --baseline main
```

Baselines are stored under `target/criterion`.

## Python End-to-End Benchmarks (Apex2)

The Apex2 validation system runs full Python-level benchmarks and produces machine-readable JSON reports.

### Apex2 Runners

| Script | What it measures | Output |
|--------|-----------------|--------|
| `tests/apex2_pyhf_validation_report.py` | NLL/expected_data parity vs pyhf + speedup | `tmp/apex2_pyhf_report.json` |
| `tests/benchmark_glm_fit_predict.py` | GLM fit/predict timing (linear/logistic/poisson/negbin) | `tmp/p6_glm_fit_predict.json` |
| `tests/apex2_p6_glm_benchmark_report.py` | P6 GLM regression vs baseline (slowdown detection) | `tmp/apex2_p6_glm_bench_report.json` |
| `tests/apex2_sbc_report.py` | SBC posterior calibration (NUTS) | `tmp/apex2_sbc_report.json` |
| `tests/apex2_master_report.py` | Aggregates all runners into one report | `tmp/apex2_master_report.json` |
| `tests/compare_with_latest_baseline.py` | Compare current runs vs `tmp/baselines/latest_manifest.json` | `tmp/baseline_compare_report.json` |

### Recording Baselines

Use `tests/record_baseline.py` to record reference baselines with a full environment fingerprint:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py
```

Or via `make`:

```bash
make apex2-baseline-record
```

This records both pyhf and P6 GLM baselines to `tmp/baselines/` with:
- machine hostname + timestamp in filename
- full environment metadata (Python, pyhf, nextstat, numpy versions, git commit, CPU, platform)
- a `latest_manifest.json` linking the most recently recorded *full* baseline set (pyhf + P6 GLM, and optionally ROOT suite artifacts)
- per-type pointers (`latest_pyhf_manifest.json`, `latest_p6_glm_manifest.json`, `latest_root_manifest.json`) for workflows where baselines are recorded on different machines (e.g. ROOT suite on a cluster)

Note: when you record only a subset via `--only ...` (for example `--only root` on a cluster),
the recorder does not overwrite an existing `latest_manifest.json` (to avoid clobbering a full baseline set).
Use the per-type latest manifests for that workflow.

Options:

```bash
# Record only pyhf baseline
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --only pyhf

# Record only P6 GLM baseline
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --only p6

# Custom GLM benchmark parameters
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --sizes 200,2000,20000 --p 20
```

### Comparing Against Baselines

```bash
# Compare current P6 GLM run against recorded baseline
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_p6_glm_benchmark_report.py \
  --baseline tmp/baselines/p6_glm_baseline_<host>_<date>.json \
  --out tmp/apex2_p6_glm_bench_report.json

# Or via the master report
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --p6-glm-bench \
  --p6-glm-bench-baseline tmp/baselines/p6_glm_baseline_<host>_<date>.json
```

The comparison uses a configurable slowdown threshold (default 1.3x) and skips sub-millisecond timings as too noisy.

### Compare Against Latest Baseline Manifest

After recording baselines once, compare current HEAD against the latest manifest:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/compare_with_latest_baseline.py
```

Or via `make`:

```bash
make apex2-baseline-compare
```

This writes a consolidated report to `tmp/baseline_compare_report.json` and exits with:
- `0` if parity is OK and slowdown thresholds are satisfied
- `2` if parity fails or performance regresses beyond thresholds
- `4` if a runner errors (missing deps, crash, etc.)

If the chosen manifest is missing some baseline keys (for example because it was recorded with `--only root`),
the compare runner will attempt to recover missing entries by scanning newer `baseline_manifest_*.json` in the same directory.

Note on performance noise:
- `tests/compare_with_latest_baseline.py` skips pyhf perf comparisons where the baseline per-call NLL time is below `1e-6` seconds by default (`--pyhf-min-baseline-s`), because sub-microsecond timings are dominated by timer noise.
- The GLM benchmark (`tests/benchmark_glm_fit_predict.py`) reports median timings (not min) to make regressions less sensitive to transient CPU load.
- The P6 GLM compare wrapper (`tests/apex2_p6_glm_benchmark_report.py`) skips predict comparisons when baseline `predict_s` is below `1e-3` seconds by default (`--min-baseline-predict-s`), mirroring the existing `--min-baseline-fit-s`.

For strict performance gating, require the same host as the baseline:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/compare_with_latest_baseline.py --require-same-host
```

### Baseline Environment Fingerprint

Every baseline JSON includes a `baseline_env` block:

```json
{
  "baseline_env": {
    "timestamp": 1770389196,
    "datetime_utc": "2026-02-06T14:46:36Z",
    "hostname": "MacBook-Pro.local",
    "python": "3.13.11",
    "platform": "macOS-26.2-arm64-arm-64bit-Mach-O",
    "machine": "arm64",
    "cpu": "Apple M5",
    "nextstat_version": "0.1.0",
    "pyhf_version": "0.7.6",
    "numpy_version": "2.4.2",
    "git": {
      "commit": "82418b01...",
      "branch": "main",
      "dirty": false
    }
  }
}
```

For detailed Apex2 methodology (cluster jobs, ROOT parity, etc.) see [docs/tutorials/root-trexfitter-parity.md](tutorials/root-trexfitter-parity.md).

## CI

Bench compilation and scheduled quick runs live in `.github/workflows/bench.yml`.

An opt-in, non-blocking perf smoke workflow is available as:
- `.github/workflows/perf-smoke.yml` (manual `workflow_dispatch`)

The perf smoke job is intended to catch obvious breakage (bench runtime errors) without
gating merges on absolute timing thresholds.
