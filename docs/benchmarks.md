# Benchmarks

For the canonical GitHub-release maintainer path, see: [Release Runbook](/docs/releases/release-runbook).

For the canonical overview of how tests, parity contracts, benchmark evidence,
stable-surface governance, and release gating fit together, see:
[Validation and Release Discipline](/docs/references/validation-and-release-discipline).

For the policy that defines what benchmark evidence belongs in git vs CI artifacts vs release assets, see: [Benchmark Artifact Policy](/docs/releases/benchmark-artifact-policy).

The canonical machine-readable prerelease outputs are:
- `tmp/release_surface_matrix_report.json`
- `tmp/sota_claim_matrix_report.json`
- `tmp/public_sota_bundle.json`
- `tmp/v1_sota_policy_report.json`
- `tmp/release_manifest.json`
- `tmp/release_candidate_bundle/`

For the **public benchmarks** program (trust/reproducibility spec, artifacts, and suite structure), see: [Public Benchmarks Specification](/docs/public-benchmarks).

For the live registry of published artifacts on nextstat.io, see: [Benchmark Results](/docs/benchmark-results) and [Snapshot Registry](/docs/snapshot-registry).

For a step-by-step “first snapshot” runbook, see: [First Public Benchmark Snapshot (Playbook)](/docs/benchmarks/first-public-snapshot).

For publication-grade unbinned GPU/CPU matrix execution, see: [Unbinned GPU Publication Runbook (PF3.1)](/docs/benchmarks/unbinned-publication-runbook).

For RNTuple native decode comparison on `epyc-node` (`ns-root` vs ROOT, 2026-02-16), see: [RNTuple Decode Benchmark (epyc-node, 2026-02-16)](/docs/benchmarks/rntuple-epyc-node-2026-02-16). For local reproducible reruns use `make rntuple-root-vs-nsroot`.

For the GVM measurement-combination published snapshot (Apple M5 + AMD EPYC, 2026-03-07, including Rayon thread-scaling notes), see: [GVM Benchmark Snapshot](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07).

For the `NumericalPaper` multi-start robustness snapshot (mixed literature + synthetic tiers through `128x96`, 2026-03-07), see: [GVM NumericalPaper Robustness Snapshot](/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07).

For the release-hardening memo that defines the current stable-first subset, evidence-backed operating envelope, and remaining blockers before further stable promotion, see: [GVM Stable-Surface Readiness Memo](/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07).

For the normative release-hardening policy and the explicit first-wave stable promotion decision, see:
- [GVM Stable-Surface Support Policy](/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07)
- [GVM Stable-First Promotion Decision](/docs/benchmarks/gvm-stable-first-decision-2026-03-07)

For the simplified-likelihood stable-surface acceptance criteria, release gate,
promotion runbook, and the current `nextstat-bench` benchmark evidence, see:
- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08)
- [Simplified Likelihood Release PR Checklist](/docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08)
- [Simplified Likelihood Promotion Runbook](/docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08)
- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09)
- [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09)
- [Simplified Likelihood Exporter Stable Evidence Freshness](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09)
- [Simplified Likelihood Exporter Public Validation Surface](/docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09)
- [Simplified Likelihood Exporter Stable-Review Checklist](/docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09)
- [Simplified Likelihood Exporter Stable Source-Semantics Boundary](/docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09)
- [Simplified Likelihood Exporter Stable-Candidate Blocker Matrix](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09)
- [Simplified Likelihood Exporter Stable-Candidate Review Packet](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09)
- Exporter public case catalog example:
  `docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json`
- Exporter public validation report example:
  `docs/specs/benchmarks/simplified_likelihood_export_public_validation_report_v0.example.json`
- Exporter stable evidence policy example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- Validator-facing promotion evidence bundle example:
  `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`
- Promotion evidence verification report example:
  `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_check_v0.example.json`
- Promotion bundle persistence report example:
  `docs/specs/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.example.json`
- Exporter benchmark snapshot persistence report example:
  `docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json`
- Exporter promotion evidence bundle example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json`
- Exporter promotion evidence verification report example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.example.json`
- Exporter promotion bundle persistence report example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json`
- Exporter stable-review assessment example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json`
- Exporter stable source-semantics boundary example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json`
- Exporter stable-candidate blocker matrix example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json`
- Exporter stable-candidate review packet example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json`

For the ads + weekly time-series stable-surface acceptance criteria, support
matrix, release note, promotion runbook, and current `nextstat-bench`
benchmark evidence, see:
- [Ads + Time Series Stable-Surface Acceptance](/docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08)
- [Ads + Time Series Stable-Surface Support Matrix](/docs/benchmarks/ads-timeseries-support-matrix-2026-03-08)
- [Ads + Time Series Stable-Surface Release Notes](/docs/benchmarks/ads-timeseries-release-notes-2026-03-08)
- [Ads + Time Series Stable-Surface Release PR Checklist](/docs/benchmarks/ads-timeseries-release-pr-checklist-2026-03-08)
- [Ads + Time Series Runtime Gate](/docs/benchmarks/ads-timeseries-runtime-gate.md)
- [Ads + Time Series Promotion Runbook](/docs/benchmarks/ads-timeseries-promotion-runbook-2026-03-08)
- [Ads + Time Series Benchmark Snapshot: 2026-03-08](/docs/benchmarks/ads-timeseries-benchmark-snapshot-2026-03-08)

For the realistic ads variance-reduction matrix that compares `naive`,
`CUPED(primary covariate)`, and `CURE(all covariates)` across `n`, `p`,
sparsity, and collinearity, see:
- [Ads Variance-Reduction Matrix Runbook](/docs/benchmarks/ads-variance-reduction-runbook-2026-03-08)
- [Ads Variance-Reduction Benchmark: 2026-03-08](/docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08)
- [Ads Variance-Reduction Stable-Surface Acceptance](/docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09)
- [Ads Variance-Reduction Runtime Gate](/docs/benchmarks/ads-variance-reduction-runtime-gate.md)

For the HEPData import stable-surface acceptance criteria, runtime gate,
promotion workflow, and frozen `nextstat-bench` evidence, see:
- [HEPData Import Acceptance Criteria (Stable Surface v1)](/docs/specs/hep/hepdata_import_acceptance_v1)
- [HEPData Import Runtime Gate](/docs/benchmarks/hepdata-import-runtime-gate)
- [HEPData Import Benchmark Snapshot: 2026-03-08](/docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08)
- [HEPData Import Stable-Surface Support Matrix](/docs/benchmarks/hepdata-import-support-matrix-2026-03-08)
- [HEPData Import Stable-Surface Release Notes](/docs/benchmarks/hepdata-import-release-notes-2026-03-08)
- [HEPData Import Release PR Checklist](/docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08)
- [HEPData Import Promotion Runbook](/docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08)

For the short operational matrix of what is `stable` vs `research-grade`, and the release note for the promoted subset, see:
- [GVM Stable-First Support Matrix](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07)
- [GVM Stable-First Release Notes](/docs/benchmarks/gvm-stable-first-release-notes-2026-03-07)
- [GVM Stable-First Release Candidate: v0.10.0](/docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08)
- [GVM Stable-First Release PR Checklist](/docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07)
- [GVM Stable-First Launch Checklist](/docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07)

For the simplified-likelihood stable consume path and the promoted narrow
exporter subset, see:
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08)
- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09)
- [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09)
- [Simplified Likelihood Exporter Stable Evidence Freshness](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09)
- [Simplified Likelihood Exporter Stable Source-Semantics Boundary](/docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09)
- [Simplified Likelihood Exporter Stable-Candidate Review Packet](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09)
- [Simplified Likelihood Exporter Stable Promotion Decision](/docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09)
- [Simplified Likelihood Exporter Release PR Checklist](/docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09)
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json`

Suite runbooks: [HEP](/docs/benchmarks/suites/hep) · [Pharma](/docs/benchmarks/suites/pharma) · [Bayesian](/docs/benchmarks/suites/bayesian) · [ML](/docs/benchmarks/suites/ml) · [Time Series](/docs/benchmarks/suites/timeseries) · [Econometrics](/docs/benchmarks/suites/econometrics)

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
| `measurement_combine_benchmark.rs` | Stable-first GVM core plus research-grade advanced measurement-combination workflows: direct fits, toy calibration, campaigns, and solver-parity calibration on paper + synthetic large-scale combinations |
| `flow_nll_f32_vs_f64.rs` | Flow NLL: f32 device-ptr vs f64 host upload (CUDA) |

### Local Runs

Run all benches (slow):

```bash
cargo bench --workspace
```

Run a specific bench:

```bash
cargo bench -p ns-inference --bench mle_benchmark
```

Run the dedicated GVM measurement-combination bench:

```bash
cargo bench -p ns-inference --bench measurement_combine_benchmark
```

This bench now covers:
- direct `combine_measurements_with_solver(...)`
- `calibrate_measurements_toys_with_solver(...)`
- `run_measurement_combination_calibration_campaign_with_solver(...)`
- `compare_measurement_combination_calibration_campaign_solvers(...)`

The calibration layer includes both paper-scale fixtures and synthetic large-scale cases:
- `synthetic_gvm_32x24`
- `synthetic_gvm_64x48`

For fast local iteration on the synthetic large-scale cases:

```bash
cargo bench -p ns-inference --bench measurement_combine_benchmark -- --quick
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
| `tests/apex2_simplified_likelihood_report.py` | full-vs-simplified fidelity, reduction, and reinterpretation speedup | `tmp/apex2_simplified_likelihood_report.json` |
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

Measured on NVIDIA RTX 4000 SFF Ada (Ada Lovelace, 20GB GDDR6), CUDA 12.0, dedicated GPU server.
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

### Flow NLL Reduction — f32 Zero-Copy vs f64 Host Upload

When ONNX Runtime CUDA EP produces log-prob outputs as `float*` device pointers,
the f32 zero-copy path (`nll_device_ptr_f32`) eliminates the host-to-device memcpy
entirely. The f64 host path requires uploading `n_procs × n_events` doubles (~16 bytes/event)
to GPU before each kernel launch.

Measured on RTX 4000 SFF Ada, release build, Criterion 100 samples.

#### Single process, no constraints

| Events | f64 host (H2D + kernel) | f32 device ptr (kernel only) | Speedup |
|-------:|------------------------:|-----------------------------:|--------:|
| 100 | 1.741 ms | 0.911 ms | 1.9x |
| 1,000 | 1.761 ms | 30.8 µs | **57x** |
| 10,000 | 1.947 ms | 251.8 µs | 7.7x |
| 100,000 | 3.832 ms | 3.121 ms | 1.2x |

#### Two processes + Gaussian constraint

| Events | f64 host (H2D + kernel) | f32 device ptr (kernel only) | Speedup |
|-------:|------------------------:|-----------------------------:|--------:|
| 1,000 | 374.1 µs | 48.2 µs | **7.8x** |
| 10,000 | 359.3 µs | 351.3 µs | 1.02x |
| 100,000 | 3.497 ms | 3.390 ms | 1.03x |

**Key insight**: The speedup comes entirely from eliminating H2D memcpy, not from f32
arithmetic being faster (accumulation is still f64 on GPU). The sweet spot is 100–10K events
— the typical range for unbinned flow fits in HEP — where H2D transfer dominates.
At 100K+ events, GPU compute time dominates and both paths converge.

**Numerical accuracy**: f32 path matches f64 to `rel_err < 1e-4` on standard data,
`< 1e-3` on extreme logp values ([-10..−28]).

#### Reproducing

```bash
# On a CUDA machine:
cargo bench -p ns-compute --features cuda --bench flow_nll_f32_vs_f64
```

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
