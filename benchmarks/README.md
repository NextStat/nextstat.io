# Benchmarks

This repo has three benchmark layers, each with its own output format and intended audience.

## 1) Public Benchmarks Harness (Schema'd Snapshots)

Path: `benchmarks/nextstat-public-benchmarks/`

Purpose: outsider-rerunnable benchmark snapshots with strict JSON schemas.

- Suites live in `benchmarks/nextstat-public-benchmarks/suites/`.
- Local suite outputs default to `benchmarks/nextstat-public-benchmarks/out/<suite>/...`.
- Snapshot bundles (suite outputs + `baseline_manifest.json` + `snapshot_index.json`) live under:
  - `benchmarks/nextstat-public-benchmarks/manifests/snapshots/<snapshot_id>/`
- JSON schemas live in:
  - `benchmarks/nextstat-public-benchmarks/manifests/schema/`
- Validator:
  - `python3 benchmarks/nextstat-public-benchmarks/scripts/validate_artifacts.py --strict <artifact-or-dir>`

## 2) Unbinned Cross-Framework Suite (HEP Event-Level)

Path: `benchmarks/unbinned/`

Purpose: compare unbinned fits across frameworks (NextStat CLI + RooFit + zfit + MoreFit) and
publication-grade GPU matrix runs.

- Cross-framework runner:
  - `python3 benchmarks/unbinned/run_suite.py --out tmp/unbinned_bench.json`
  - Output schema: `nextstat.unbinned_run_suite_result.v1`
- CPU symmetry benchmark (NextStat vs MoreFit, in-process timing hygiene):
  - `python3 benchmarks/unbinned/bench_cpu_symmetry.py ...`
  - Output schema: `nextstat.unbinned_cpu_symmetry_bench.v1`
- Publication matrix inputs:
  - `benchmarks/unbinned/matrices/pf31_publication_v1.json` (schema: `nextstat.pf31_publication_matrix.v1`)
  - `benchmarks/unbinned/specs/*.json` (schema: `nextstat_unbinned_spec_v0`)
- Checked-in benchmark outputs:
  - `benchmarks/unbinned/artifacts/<YYYY-MM-DD>/...`
- Validator:
  - `python3 benchmarks/unbinned/validate_artifacts.py [--strict] <artifact-or-dir>`

## 3) Rust Microbenches (Criterion)

Paths:

- `crates/*/benches/*.rs`

Purpose: tight microbenchmarks for performance regressions (SIMD, NLL kernels, NUTS transitions, parquet IO, etc).

- Run:
  - `cargo bench -p ns-inference`
  - `cargo bench -p ns-compute --bench simd_benchmark`
- Outputs:
  - `target/criterion/` (Criterion HTML + summaries)

## 4) GPU Triple Harness (LAPS vs MAMS vs BlackJAX)

Path: `benchmarks/gpu_triple_bench.py`

Purpose: apples-to-apples GPU benchmark for:

- `NS_LAPS_GPU`
- `NS_CPU_MAMS`
- `BlackJAX_GPU`

Key metric semantics (current, canonical):

- `wall_s`: total run wall time for the engine row.
- `wall_sampling`: sampling-phase wall time used for throughput metrics.
- `ESS/s(samp)`: `min_ess / wall_sampling`.
- `ESS/grad`: `min_ess / n_grad_evals`.
- `grad/s`: `n_grad_evals / wall_sampling`.

Output artifact:

- `gpu_triple_bench.json` only.
- Legacy alias `a100_triple_bench.json` is removed to avoid GPU-name confusion on non-A100 hosts (for example V100/GEX44).

Operational note (shared GPU host):

- When running LAPS and BlackJAX in the same process on one GPU, disable JAX preallocation to avoid CUDA OOM/resource contention:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
