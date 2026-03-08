# Benchmarks

This repo has three benchmark layers, each with its own output format and intended audience.

## 1) Public Benchmarks Harness (Schema'd Snapshots)

Path: `benchmarks/nextstat-public-benchmarks/`

Purpose: outsider-rerunnable benchmark snapshots with strict JSON schemas.

- Suites live in `benchmarks/nextstat-public-benchmarks/suites/`.
- Local suite outputs default to `benchmarks/nextstat-public-benchmarks/out/<suite>/...`.
- Snapshot bundles (suite outputs + `baseline_manifest.json` + `snapshot_index.json`) live under:
  - `benchmarks/nextstat-public-benchmarks/manifests/snapshots/<snapshot_id>/`
  - `snapshot_index.json` is now registry-facing as well: besides the hash inventory it can surface assessment-backed `suite_health` verdicts for suites that emit assessment artifacts
  - `snapshot_registry.json` can now be generated from one or more committed snapshot indices via `python3 scripts/benchmarks/write_snapshot_registry.py --snapshots-root benchmarks/nextstat-public-benchmarks/manifests/snapshots --out benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json`
  - Freshness gate: `python3 benchmarks/nextstat-public-benchmarks/scripts/write_snapshot_registry.py --snapshots-root benchmarks/nextstat-public-benchmarks/manifests/snapshots --out benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json --check`
- JSON schemas live in:
  - `benchmarks/nextstat-public-benchmarks/manifests/schema/`
- Validator:
  - `python3 benchmarks/nextstat-public-benchmarks/scripts/validate_artifacts.py --strict <artifact-or-dir>`
- Canonical Bayesian suite:
  - Local: `python3 benchmarks/nextstat-public-benchmarks/suites/bayesian/suite.py --out-dir benchmarks/nextstat-public-benchmarks/out/bayesian --deterministic`
  - Assessment: `python3 benchmarks/nextstat-public-benchmarks/suites/bayesian/assess.py benchmarks/nextstat-public-benchmarks/out/bayesian`
  - Remote (`nextstat-bench`): `scripts/benchmarks/bench_bayesian_suite_remote.sh`
  - Remote publisher snapshot (`nextstat-bench`): `scripts/benchmarks/publish_bayesian_snapshot_remote.sh`
    This host-backed publish lane now syncs the full snapshot root back locally, including the publisher-refreshed `snapshot_registry.json`.
  - Remote multi-seed repeatability (`nextstat-bench`): `scripts/benchmarks/bench_bayesian_multiseed_remote.sh` with canonical `nextstat,cmdstanpy` and optional `nextstat,pymc` / `nextstat,cmdstanpy,pymc` host-backed paths; the multiseed summary is now health-complete and can be regenerated from existing `seed_*` artifacts via `--reuse-existing`
  - Published snapshot verdicts: `bayesian_assessment.json` now carries a machine-readable `review_summary`, and `README_snippet_bayesian.md` now surfaces the same `core_quality` / `promotion_gate` verdict plus worst-case health metrics instead of hiding them in the raw suite table
  - Optional remote cross-framework path: set `BENCH_BACKENDS=nextstat,cmdstanpy,pymc` and the runner will self-provision `cmdstanpy` + local `vendor/cmdstan` and `pymc` + `arviz` on `nextstat-bench`; when vendor CmdStan exists it takes precedence over ambient host `~/.cmdstan`, and the PyMC lane defaults to `PYTENSOR_FLAGS=blas__ldflags=-lblas` unless overridden; the canonical `histfactory_simple_8p` case is also mapped exactly for `cmdstanpy` and `pymc` via the tracked simple HistFactory Poisson/Gamma shapesys fixture; `BENCH_EXTRA_PIP_PACKAGES` stays additive for stacks like `numpyro`
  - Suite guidance: `benchmarks/nextstat-public-benchmarks/suites/bayesian/README.md`
- Canonical MAMS suite:
  - Local: `python3 benchmarks/nextstat-public-benchmarks/suites/mams/suite.py --out-dir benchmarks/nextstat-public-benchmarks/out/mams --deterministic`
  - Assessment: `python3 benchmarks/nextstat-public-benchmarks/suites/mams/assess.py benchmarks/nextstat-public-benchmarks/out/mams`
  - Remote (`nextstat-bench`): `scripts/benchmarks/bench_mams_suite_remote.sh`
  - Remote multi-seed repeatability (`nextstat-bench`): `scripts/benchmarks/bench_mams_multiseed_remote.sh` with canonical `nextstat_mams,nextstat_nuts`, fixed `dataset_seed=12345`, stabilized MAMS defaults (`warmup=3500`, `target_accept=0.985`, `max_leapfrog=1024`), a schema-backed `mams_multiseed_summary.{json,md}`, and a separate `mams_multiseed_assessment.{json,md}` repeatability verdict that can be regenerated from existing `seed_*` artifacts via `--reuse-existing` plus `assess_multiseed.py`
  - Remote expanded stress repeatability (`nextstat-bench`): `scripts/benchmarks/bench_mams_stress_multiseed_remote.sh` with supported cases `neal_funnel_ncp_10d` and `hier_random_intercept_non_centered`, plus `neal_funnel_10d_centered` as a pathological control. This lane emits `mams_stress_multiseed_summary.{json,md}` and `mams_stress_assessment.{json,md}` under a separate schema-backed contract so canonical promotion evidence and expanded stress evidence do not get mixed.
  - Remote publisher snapshot (`nextstat-bench`): `scripts/benchmarks/publish_mams_snapshot_remote.sh`
    This host-backed publish lane syncs the full publish root back locally, including the publisher-refreshed `snapshot_registry.json`, preserves the suite-local `mams/mams_benchmark_report.md`, and emits a health-complete `README_snippet_mams.md` with `core_quality`, `promotion_gate`, failing cases, and worst reviewed-case metrics.
  - Suite guidance: `benchmarks/nextstat-public-benchmarks/suites/mams/README.md`

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
- Legacy alias `a100_triple_bench.json` is removed to avoid GPU-name confusion on non-A100 hosts (for example V100/RTX4000-node).

Operational note (shared GPU host):

- When running LAPS and BlackJAX in the same process on one GPU, disable JAX preallocation to avoid CUDA OOM/resource contention:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
