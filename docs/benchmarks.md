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

## Profile Likelihood Scan (CPU)

`scan_histfactory()` uses warm-start + bounds-clamping + tape reuse for HistFactory models,
replacing the generic `scan()` cold-start + model-clone path.

### Results (release, Apple M5)

| Workspace | Params | Points | `scan()` (cold) | `scan_histfactory()` (warm) | Speedup | Iter reduction |
|-----------|--------|--------|-----------------|----------------------------|---------|----------------|
| simple | 3 | 21 | 0.000s | 0.000s | 2.2x | 1.2x |
| tHu | 184 | 21 | 11.2s | 4.2s | 2.7x | 1.2x |
| tttt-prod | 249 | 51 | 14.4s | 4.4s | 3.3x | 1.9x |

Warm-start also improves numerical accuracy at tail points: cold-start from `parameter_init()`
can get stuck in local minima at extreme mu values, while warm-start from the neighboring
scan point reaches the global minimum consistently.

### Reproducing

```bash
# All three benchmarks (release mode required for meaningful timings)
cargo test -p ns-inference --release -- test_bench_scan --ignored --nocapture
```

## Python End-to-End Benchmarks (Apex2)

The Apex2 validation system runs full Python-level benchmarks and produces machine-readable JSON reports.

## The "God Run" (Toy-based CLs)

This is the headline benchmark used in the README for toy-based CLs (q~_mu) performance.

Run:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python scripts/god_run_benchmark.py --n-toys 10000
```

Outputs:
- `tmp/god_run_report.json` (machine-readable report)
- `tmp/god_run_snippet.md` (README-ready Markdown snippet)

### Apex2 Runners

| Script | What it measures | Output |
|--------|-----------------|--------|
| `tests/apex2_pyhf_validation_report.py` | NLL/expected_data parity vs pyhf + speedup | `tmp/apex2_pyhf_report.json` |
| `tests/benchmark_glm_fit_predict.py` | GLM fit/predict timing (linear/logistic/poisson/negbin) | `tmp/p6_glm_fit_predict.json` |
| `tests/apex2_p6_glm_benchmark_report.py` | P6 GLM regression vs baseline (slowdown detection) | `tmp/apex2_p6_glm_bench_report.json` |
| `tests/apex2_gpu_bench_report.py` | CPU vs CUDA perf (fit, profile scan, batch toys) + basic parity | `tmp/apex2_gpu_bench_report.json` |
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

## ROOT TTree I/O Benchmarks

NextStat includes a native ROOT TTree reader (ns-root) with mmap I/O and rayon-parallel
basket decompression. No ROOT C++ dependency required.

### Comparison: NextStat vs uproot + numpy

Measured on the same file (`simple_tree.root`, 1000 entries, 7 branches), same machine,
release build. uproot timings are warmed (imports pre-loaded).

| Operation | NextStat (Rust) | uproot + numpy | Speedup |
|---|---:|---:|---:|
| File open (mmap) | 75 µs | 215 µs | ~3x |
| TTree metadata parse | 50 µs | 1,400 µs | ~28x |
| Read 1 branch (f64) | 65 µs | 675 µs | ~10x |
| Read all 7 branches | 200 µs | 1,300 µs | ~6.5x |
| Selection eval (`njet >= 4 && pt > 25`) | 15 µs | 26 µs | ~1.7x |
| Histogram fill (selection + weight) | 28 µs | 96 µs | ~3.4x |
| **Total pipeline** | **~430 µs** | **~3,700 µs** | **~8.5x** |

### Scaling expectations

The 1000-entry fixture measures per-event cost plus fixed overhead. At realistic
ntuple sizes (10M+ entries):

- **rayon parallel basket decompression** scales with core count (uproot is single-threaded by default).
- **mmap** enables OS-level prefetching and avoids full-file copies into RAM.
- **Expression eval** scales linearly without Python GIL overhead.
- **Expected total speedup at scale**: 10-20x vs uproot, 50-100x vs ROOT C++ `TTree::Draw`.

### Reproducing

```bash
# Generate fixture (requires uproot + numpy in .venv)
.venv/bin/python tests/fixtures/generate_root_fixtures.py

# Run Rust TTree tests
cargo test -p ns-root --test read_tree
```

## GPU Benchmarks (CUDA)

Measured on NVIDIA RTX 4000 SFF Ada (Ada Lovelace, 20GB GDDR6), CUDA 12.0, GEX44 server.
Release build (`--release`). CPU comparison on same machine (AMD EPYC, 8 cores).

### MLE Fit — CPU vs GPU

| Workspace | Params | CPU | GPU (CUDA) | Ratio |
|-----------|--------|-----|------------|-------|
| complex | 8 | 2.3 ms | 136.3 ms | CPU 59x faster |
| tHu | 184 | 520.8 ms | 1,272.0 ms | CPU 2.4x faster |

**Verdict**: Single-model GPU fit is slower than CPU at all model sizes due to kernel
launch overhead + H↔D transfer dominating the per-iteration cost. GPU single-model
fit is not recommended; use CPU.

### Profile Likelihood Scan — CPU vs GPU

| Workspace | Params | Scan Points | CPU | GPU (CUDA) | Ratio |
|-----------|--------|-------------|-----|------------|-------|
| complex | 8 | 21 | 6.3 ms | 132.4 ms | CPU 21x faster |
| tHu | 184 | 21 | 8.4 s | 7.9 s | **GPU 1.07x faster** |

**Crossover**: GPU becomes competitive for profile scans at ~150+ parameters.
Warm-start between scan points amortizes the per-point GPU overhead.

### Differentiable Layer (GPU-only)

| Workspace | Params | NLL + Signal Gradient | Profiled q₀ |
|-----------|--------|-----------------------|-------------|
| complex | 8 | 0.12 ms | 3.0 ms |
| tHu | 184 | 3.66 ms | — |

Signal gradient accuracy vs finite differences: **2.07e-9** max error.

### Neural Network Training (GPU-only)

| Metric | Value |
|--------|-------|
| 20-step training loop | 2.4 ms/step |
| Signal gradient (8 bins) | CUDA zero-copy |
| NLL convergence | Monotonically decreasing |

### Batch Toy Fitting — CPU vs GPU

Lockstep L-BFGS-B: all toys share a single kernel launch per optimizer iteration.

#### tHu (184 params) — GPU wins

**CUDA** (RTX 4000 SFF Ada, AMD EPYC 8 cores):

| n_toys | GPU (CUDA) | CPU (Rayon, 8 cores) | GPU Speedup |
|--------|-----------|---------------------|-------------|
| 100 | 20.2 s | 37.9 s | **1.8x** |
| 500 | 63.4 s | 383.7 s | **6.0x** |
| 1000 | 119.9 s | 771.4 s | **6.4x** |

**Metal** (Apple M5, 10 cores, f32):

| n_toys | GPU (Metal) | CPU (Rayon, 10 cores) | GPU Speedup |
|--------|-----------|----------------------|-------------|
| 100 | 10.7 s | 29.8 s | **2.8x** |
| 500 | 29.1 s | 175.5 s | **6.0x** |
| 1000 | 56.8 s | 359.1 s | **6.3x** |

GPU/CPU speedup ratio converges to ~6.3x on both platforms at 1000 toys.

#### complex (8 params) — CPU wins

**CUDA** (RTX 4000 SFF Ada):

| n_toys | GPU (CUDA) | CPU (Rayon, 8 cores) | CPU Speedup |
|--------|-----------|---------------------|-------------|
| 100 | 726 ms | 18 ms | CPU 40x |
| 500 | 1,169 ms | 23 ms | CPU 51x |
| 1000 | 1,838 ms | 40 ms | CPU 46x |
| 5000 | 7,412 ms | 146 ms | CPU 51x |

**Metal** (Apple M5):

| n_toys | GPU (Metal) | CPU (Rayon, 10 cores) | CPU Speedup |
|--------|-----------|----------------------|-------------|
| 100 | 1,710 ms | 31 ms | CPU 55x |
| 1000 | 2,378 ms | 132 ms | CPU 18x |
| 5000 | 8,380 ms | 226 ms | CPU 37x |

**Key insight**: GPU batch scaling is sub-linear (lockstep amortizes overhead),
while CPU scaling is super-linear for large models (memory/cache pressure at 184 params × 1000 toys).
Crossover: GPU wins for models with ~100+ parameters. Both CUDA (f64) and Metal (f32) show the same ~6.3x speedup at scale.

### Bottleneck Analysis

| Bottleneck | Impact | Recommendation |
|------------|--------|---------------|
| **Kernel launch overhead** | ~130 ms per launch on RTX 4000 | Use CPU for single-model fits and small models |
| **H↔D transfer** | Negligible for params (~2 KB), significant for repeat calls | Session reuse, warm-start |
| **Batch lockstep** | GPU 6.4x faster at 184p × 1000 toys | Use GPU for large-model toy-based CLs |
| **Profiled q₀ fits** | 2 L-BFGS-B fits per forward pass (~3 ms total) | Acceptable for NN training |
| **Single-model scan crossover** | ~150 params | Auto-dispatch: CPU below, GPU above |

### GPU Strengths

1. **Batch toy fitting**: ~6.3x faster than CPU on large models (184 params, 1000 toys). Consistent across CUDA and Metal. Scales sub-linearly with toy count.
2. **Differentiable training**: CUDA zero-copy avoids all H↔D transfers for signal data.
3. **Large-model scans**: GPU amortizes overhead when per-point fit time is large.

### GPU Weaknesses

1. **Small models**: Kernel launch overhead dominates. CPU 40-50x faster for 8-param models.
2. **Single-model evaluation**: Even for large models, CPU is 2.4x faster for one-off fits.
3. **Sequential scans**: Serial H↔D transfers per scan point.

### Reproducing

```bash
# On a CUDA machine:
cargo build --release -p ns-cli --features cuda

# Single-model fit
time nextstat fit --input tests/fixtures/complex_workspace.json --gpu cuda
time nextstat fit --input tests/fixtures/workspace_tHu.json --gpu cuda

# Profile scan
time nextstat scan --input tests/fixtures/workspace_tHu.json --start 0 --stop 5 --n-points 21 --gpu cuda

# Batch toys (the headline GPU benchmark)
time nextstat hypotest-toys --input tests/fixtures/workspace_tHu.json --mu 1.0 --n-toys 1000 --gpu cuda
time nextstat hypotest-toys --input tests/fixtures/workspace_tHu.json --mu 1.0 --n-toys 1000
```

## CI

Bench compilation and scheduled quick runs live in `.github/workflows/bench.yml`.

An opt-in, non-blocking perf smoke workflow is available as:
- `.github/workflows/perf-smoke.yml` (manual `workflow_dispatch`)

The perf smoke job is intended to catch obvious breakage (bench runtime errors) without
gating merges on absolute timing thresholds.
