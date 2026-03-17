---
title: "Public Benchmarks"
description: "Canonical specification for NextStat's public benchmark program: protocols, correctness gates, environment pinning, artifacts, and suite structure for reproducible performance evidence across HEP, pharma, Bayesian, ML, time series, and econometrics."
status: shipped
last_updated: 2026-02-13
keywords:
  - public benchmarks
  - reproducible benchmarks
  - scientific software validation
  - HistFactory benchmark
  - pyhf comparison
  - NLME benchmark
  - NUTS sampler benchmark
  - benchmark protocol
  - NextStat
---

# Public Benchmarks (Trust Offensive)

This page is the **canonical specification** for how NextStat benchmarks are designed, executed, and published. The goal is not "a fast number" — it is **trustworthy, reproducible evidence** that other people can rerun and audit.

If you want the narrative version (why we're doing this and what it changes), see the blog posts:

- [Trust Offensive: Public Benchmarks](/blog/trust-offensive)
- [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks)
- [Benchmark Snapshots as Products](/blog/benchmark-snapshots-ci-artifacts)
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report)
- [Building a Trustworthy HEP Benchmark Harness](/blog/hep-benchmark-harness)
- [Numerical Accuracy](/blog/numerical-accuracy)
- [Differentiable HistFactory in PyTorch](/blog/differentiable-layer)
- [Bayesian Benchmarks: ESS/sec](/blog/bayesian-benchmarks-ess-per-sec)
- [Pharma Benchmarks: PK/NLME](/blog/pharma-benchmarks-pk-nlme)
- [JAX Compile vs Execution](/blog/jax-compile-vs-execution)

## What goes in docs vs blog

**Docs (this site) are canonical**: protocols, contracts, runbooks, and “how to rerun” instructions.

**Blog posts are narrative**: motivation, design rationale, interpretation of results, and “what this changes” framing.

Rule of thumb:

- If a reader needs to *execute* the benchmark, it belongs in **docs**.
- If a reader needs to *understand why* the benchmark program exists, it belongs in the **blog**.

## Scope

We benchmark **end-to-end user workflows** rather than isolated micro-kernels:

- HEP / HistFactory: NLL evaluation, gradients, MLE fits, profile scans, toy ensembles.
- Pharma: PK/NLME likelihood evaluation + fitting loops.
- Bayesian: gradient-based samplers (NUTS) with ESS/sec and wall-time.
- ML infra: compilation vs execution time (e.g., JAX compile latency vs steady-state throughput) where relevant to scientific pipelines.
- Time Series: Kalman filter/smoother throughput, EM convergence cost, forecasting latency.
- Econometrics: panel FE fit wall-time, DiD TWFE + event study wall-time, IV/2SLS first-stage + second-stage cost, AIPW doubly-robust estimator vs naive OLS.

We also publish “fast path vs reference path” comparisons (e.g., parity mode vs fast mode) when correctness contracts are part of the product.

## Non-goals

- “Hero” runs on one cherry-picked machine.
- Benchmarks that depend on undocumented caches, hidden warmups, or hand-tuned flags.
- Performance claims without a reproducibility story (exact versions, inputs, and scripts).

## Trust model (what you should be able to verify)

For every published snapshot you should be able to answer, from artifacts alone:

1. **What was measured?** (definition of tasks + metrics)
2. **On what data?** (dataset ID + hash + license)
3. **Under what environment?** (OS, CPU/GPU, compiler, Python, dependency versions)
4. **From what code?** (NextStat commit hash, dependency lockfiles, build flags)
5. **Does it still match reference?** (sanity/parity checks before timing)
6. **How stable is the number?** (repeat strategy, distributions, and reporting)

## Reproducibility contract

### Environment pinning

Published runs must include:

- `rust-toolchain.toml` and `Cargo.lock` (Rust toolchain + dependencies)
- Python version + dependency lock (e.g., uv/pip-tools/poetry lock)
- GPU runtime details when used (CUDA version / Metal / driver)

### Benchmark harness is open source

The harness scripts are part of the repo, not copy-pasted from blog posts.

Today (in this repo) the main entry points are:

- Python end-to-end comparisons vs pyhf: `tests/benchmark_pyhf_vs_nextstat.py`
- Rust-only microbenchmarks: `cargo bench --workspace` (Criterion)

### Correctness gating before timing

Before recording performance numbers, the harness must validate that the output is sane and (when applicable) matches a reference implementation within a stated tolerance.

Example: the pyhf-vs-NextStat harness verifies NLL agreement before it prints timings (`tests/benchmark_pyhf_vs_nextstat.py`).

## What we publish (artifacts)

Each benchmark snapshot should publish:

- Raw per-test measurements (not just a final table)
- Summary tables (median/best-of-N policy must be explicit)
- A **baseline manifest**: code SHA, env versions, dataset hashes, and run configuration
- A **snapshot index**: hash inventory plus any assessment-backed `suite_health` verdicts needed by registry/discovery surfaces
- A **snapshot registry** when publishing a collection of snapshots: commit-backed `snapshot_registry.json`, generated from one or more `snapshot_index.json` files, so live pages and automation can read normalized suite health without rewalking each snapshot bundle
- Any correctness/parity reports used as gating
- A **validation report** (`validation_report.json` + optional `validation_report.pdf`) produced by [`nextstat validation-report`](/docs/validation-report), containing dataset SHA-256 fingerprint, model spec, environment, and per-suite pass/fail summary

## Suites

### HEP suite (pyhf + ROOT/RooFit harness)

**Docs:** this site (how to run, what’s measured, what is gated).  
**Blog:** results + interpretation once snapshots are public.

Suite doc: [HEP Benchmark Suite](/docs/benchmarks/suites/hep).
Unbinned cross-framework tables: [Unbinned Likelihood Benchmark Suite](/docs/benchmarks/unbinned-benchmark-suite).
Reproducibility runbook + JSON contract: [Unbinned Benchmark Reproducibility](/docs/benchmarks/unbinned-reproducibility).
Publication matrix runbook (PF3.1 GPU): [Unbinned GPU Publication Runbook (PF3.1)](/docs/benchmarks/unbinned-publication-runbook).

Measurements:

- NLL time / call (CPU parity mode and fast mode)
- Gradient time / call (where exposed)
- MLE fit wall-time and convergence behavior
- Profile scan wall-time with warm-start policy
- Toy ensemble throughput (toys/sec) for CPU and GPU batch modes

Correctness gates:

- NLL parity vs pyhf at representative parameter points
- Fit-level checks (POI estimates, likelihood differences within tolerance)

GPU measurements (shipped):

- CPU vs CUDA vs Metal batch toy throughput (toys/sec at 100–5000 toys)
- Profile scan crossover analysis (~150+ parameters for GPU advantage)
- Differentiable layer latency (NLL + signal gradient, profiled q₀)

### Pharma suite (PK/NLME + analytic reference baselines)

Suite doc: [Pharma Benchmark Suite](/docs/benchmarks/suites/pharma).

Measurements:

- Likelihood + gradient time for standard models
- Fit wall-time (fixed iteration protocols to avoid “stopping rule” ambiguity)
- Scaling with subject count / observation count

### Bayesian suite (ESS/sec vs Stan + PyMC)

Suite doc: [Bayesian Benchmark Suite](/docs/benchmarks/suites/bayesian).

Primary metrics:

- ESS/sec (bulk ESS and tail ESS) per parameter group
- Wall-time per effective draw

Notes:

- ESS is only meaningful with matched model, priors, and diagnostics settings.
- We must publish the exact inference settings (step size adaptation, target accept, mass matrix policy).
- Publish raw suite execution and policy assessment separately: `bayesian_suite.json` is measurement evidence; `bayesian_assessment.json` carries `core_quality` vs `promotion_gate`.
- Canonical host-backed evidence for this suite is produced on `nextstat-bench` via `scripts/benchmarks/bench_bayesian_suite_remote.sh`.
- Canonical host-backed publisher validation for this suite is now also produced on `nextstat-bench` via `scripts/benchmarks/publish_bayesian_snapshot_remote.sh`.
- Canonical repeatability evidence for this suite is now also host-backed on `nextstat-bench` via `scripts/benchmarks/bench_bayesian_multiseed_remote.sh`, which emits a schema-backed multi-seed summary plus supplementary derived metrics from the per-seed artifacts. Optional `pymc` repeatability runs now use that same runner and carry PyMC rows in the derived metrics contract when explicitly requested. The multiseed summary itself is health-complete and can be regenerated from existing `seed_*` artifacts without rerunning the suite.
- Published Bayesian snapshot snippets are now health-complete as well: `bayesian_assessment.json` carries a machine-readable `review_summary`, and `README_snippet_bayesian.md` surfaces `core_quality`, `promotion_gate`, failing cases, and worst-case health metrics from that assessment layer.
- The snapshot-level `snapshot_index.json` now also lifts those verdicts into `suite_health`, so registry/discovery consumers can read Bayesian health and promotion readiness without opening the nested assessment artifact first.
- The commit-backed `snapshot_registry.json` now consumes those same `snapshot_index.json` verdicts through `write_snapshot_registry.py`, so publisher, registry, and live automation all read the same normalized health surface instead of maintaining a parallel registry-specific policy layer.
- The canonical promoted lane stays on `backends=[nextstat]`, while optional `cmdstanpy` and `pymc` runs are now self-provisioning on the same host when explicitly requested. When the synced seed repo snapshot contains `vendor/cmdstan/cmdstan-*`, that vendor toolchain takes precedence over ambient host `~/.cmdstan` state; the PyMC lane also defaults to an explicit `blas__ldflags=-lblas` PyTensor policy unless the operator overrides `PYTENSOR_FLAGS`.
- The canonical `histfactory_simple_8p` case is now supported on those optional `cmdstanpy` and `pymc` lanes via an exact restricted mapping of the tracked `simple_workspace.json` fixture: bounded `mu`, positive per-bin `shapesys` nuisances, and the corresponding Poisson/Gamma auxiliary constraints. This is intentionally fixture-scoped rather than a generic HistFactory-on-Stan/PyMC claim.

### MAMS suite (stable-surface candidate lane)

Suite doc: `benchmarks/nextstat-public-benchmarks/suites/mams/README.md`.

Primary metrics:

- ESS/gradient and ESS/sec for `nextstat_mams` vs `nextstat_nuts`
- sampler health metrics such as `max_r_hat` and `min_ess_bulk`
- explicit stable-surface policy separation through `mams_assessment.json`

Notes:

- Canonical host-backed evidence for this suite is produced on `nextstat-bench` via `scripts/benchmarks/bench_mams_suite_remote.sh`.
- The stabilized CPU MAMS public regime now uses `n_warmup=3500`, `target_accept=0.985`, `max_leapfrog=1024`, and `eps_jitter=0.0`; `init_l=0.0` means use the stable default `L = sqrt(d)` in preconditioned space, not an auto-tuning surface. MAMS divergence telemetry now also counts early-terminated non-finite energy-error transitions correctly instead of silently classifying them as clean rejects.
- Host-backed benchmark runners that build `bindings/ns-py` on `nextstat-bench` now install a locally built `nextstat-*.whl` with `pip install --no-deps` instead of relying on `maturin develop` dependency resolution. This keeps remote benchmark verification pinned to the synchronized source snapshot even when the published `nextstat-cli` wheel has not yet caught up to the monorepo version.
- Canonical repeatability evidence for this suite is now also host-backed on `nextstat-bench` via `scripts/benchmarks/bench_mams_multiseed_remote.sh`, which emits a schema-backed `mams_multiseed_summary.json` / `.md` plus a separate `mams_multiseed_assessment.json` / `.md` repeatability verdict, keeps `dataset_seed=12345` fixed across canonical seeds so `glm_logistic` reflects sampler variation instead of regenerated-data variation, and carries the tracked MAMS-vs-NUTS parity rows into the aggregate summary. The checked-in host-backed repeatability bundle now lives at `benchmarks/nextstat-public-benchmarks/suites/mams/results_v1/`; it is currently refreshed from `benchmarks/artifacts/mams_multiseed_validation_20260312T001153Z/nextstat-bench/`, so the tracked public evidence matches the packaging-hardened `nextstat-bench` lane. In the refreshed tracked evidence the repeatability gate passes with worst reviewed `max_r_hat=1.0087` on `neal_funnel_2d`, while parity stays clean. The repeatability artifacts can be regenerated from existing `seed_*` artifacts via `--reuse-existing` followed by `assess_multiseed.py`.
- Expanded stress evidence for this suite is now also host-backed on `nextstat-bench` via `scripts/benchmarks/bench_mams_stress_multiseed_remote.sh`. That lane deliberately stays out of the canonical promotion surface: it emits `mams_stress_multiseed_summary.json` / `.md` plus `mams_stress_assessment.json` / `.md`, reviews supported stress cases `neal_funnel_ncp_10d` and `hier_random_intercept_non_centered`, and keeps `neal_funnel_10d_centered` as a pathological control. The checked-in host-backed stress bundle now lives at `benchmarks/nextstat-public-benchmarks/suites/mams/stress_results_v1/`; in the current tracked evidence the stress lane is still not fully green because `hier_random_intercept_non_centered` reaches `max_r_hat=1.0158`, while `neal_funnel_ncp_10d` passes and the centered funnel control remains non-gating.
- Canonical host-backed publisher validation for this suite is now also produced on `nextstat-bench` via `scripts/benchmarks/publish_mams_snapshot_remote.sh`.
- Published MAMS snapshots now preserve the suite-local `mams/mams_benchmark_report.md`, while the top-level `README_snippet_mams.md` is health-complete: it surfaces `core_quality`, `promotion_gate`, failing cases, and worst reviewed-case metrics from `mams_assessment.json`.
- As with Bayesian evidence, `snapshot_index.json` lifts the MAMS assessment verdict into normalized `suite_health`, and `snapshot_registry.json` consumes that same row instead of recomputing a second policy surface.

### ML suite (compile vs execution + differentiable pipelines)

Suite doc: [ML Benchmark Suite](/docs/benchmarks/suites/ml).

Primary metrics:

- Cold-start latency (compile time)
- Warm throughput (steady-state execution)
- Memory footprint (where measurable and stable)

### Time Series suite (Kalman + state space)

Suite doc: [Time Series Benchmark Suite](/docs/benchmarks/suites/timeseries).

Measurements:

- Kalman filter/smoother throughput (states/sec) at varying state dimension
- EM convergence cost (iterations × NLL evaluations)
- Forecasting latency per horizon step

### Econometrics suite (panel + causal inference)

Suite doc: [Econometrics Benchmark Suite](/docs/benchmarks/suites/econometrics).

Measurements:

- Panel FE fit wall-time scaling with entity count and cluster count
- DiD TWFE + event study wall-time
- IV/2SLS first-stage + second-stage cost
- AIPW doubly-robust estimator vs naive OLS

## Publishing (CI artifacts + baselines)

We publish benchmark snapshots via CI:

- Every run has a unique, immutable identifier.
- Artifacts include raw results + baseline manifest + `validation_report.json`.
- Baseline comparisons are opt-in and versioned (no silent "moving targets").

Publishing + replication doc: [Publishing Benchmarks](/docs/benchmarks/publishing).
Unbinned reproducibility (commands + schema): [Unbinned Benchmark Reproducibility](/docs/benchmarks/unbinned-reproducibility).

Benchmarks repo skeleton (pinned envs, manifests): [Benchmarks Repo Skeleton](/docs/benchmarks/repo-skeleton).

First run playbook (step-by-step): [First Public Benchmark Snapshot (Playbook)](/docs/benchmarks/first-public-snapshot).

Seed harness directory (in this repo, to bootstrap the standalone benchmarks repo): `benchmarks/nextstat-public-benchmarks/`.

Seed publishing helper (in this repo): `benchmarks/nextstat-public-benchmarks/scripts/publish_snapshot.py` can generate a local snapshot directory under `benchmarks/nextstat-public-benchmarks/manifests/snapshots/<snapshot_id>/` and schema-validate the produced artifacts.

## Live pages (published artifacts)

For the live, user-facing registry of published artifacts on nextstat.io:

- Benchmark Results: [/docs/benchmark-results](/docs/benchmark-results)
- Snapshot Registry: [/docs/snapshot-registry](/docs/snapshot-registry)

Machine-readable registry:

- The standalone public benchmarks repo also maintains a commit-backed `snapshot_registry.json` (used to drive/verify the live pages and automation).
- `publish_snapshot.py` now refreshes that registry automatically after a successful publish: in the canonical repo layout it lands at `manifests/snapshot_registry.json`, while custom `--out-root` values default to a self-contained `<out-root>/snapshot_registry.json`.
- Generate it from published snapshot indices with `python3 benchmarks/nextstat-public-benchmarks/scripts/write_snapshot_registry.py --snapshots-root benchmarks/nextstat-public-benchmarks/manifests/snapshots --out benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json`.
- Verify freshness with `python3 benchmarks/nextstat-public-benchmarks/scripts/write_snapshot_registry.py --snapshots-root benchmarks/nextstat-public-benchmarks/manifests/snapshots --out benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json --check`.
- The registry validates under `nextstat.snapshot_registry.v1` and reuses the assessment-backed `suite_health` rows already exposed by each `snapshot_index.json`.
- The committed public snapshot set is now content-current for both Bayesian and MAMS evidence: `manifests/snapshots/bayesian-publisher-20260309T111048Z/` and `manifests/snapshots/mams-publisher-20260312T003824Z/` are the latest host-backed `nextstat-bench` publish bundles, and the committed registry surfaces both the promotion-passing Bayesian row and the promotion-passing MAMS stable-surface row directly from those bundles.
- Repeatability evidence is tracked separately from published snapshots: Bayesian multiseed evidence lives under `benchmarks/nextstat-public-benchmarks/suites/bayesian/results_v10/`, and MAMS multiseed evidence now lives under `benchmarks/nextstat-public-benchmarks/suites/mams/results_v1/`. The MAMS repeatability lane now carries a separate assessment artifact so its stable-surface repeatability verdict is machine-readable instead of being implicit in the raw summary arrays.

## DOI + citation

Benchmark snapshots that are stable enough to cite should be published with a DOI (e.g., Zenodo) and a machine-readable citation file (e.g., `CITATION.cff`).

Production DOI note: first production record is published at DOI `10.5281/zenodo.18542624` (https://zenodo.org/records/18542624).

Pipeline validation note: we also have a Zenodo **sandbox** record published (DOI `10.5072/zenodo.437330`). Sandbox DOIs are not intended for real citation.

## Third-party replication

The strongest trust signal is an independent rerun. The replication process should produce:

- A rerun log with the same harness
- The baseline manifest of the rerun environment
- A signed report comparing the rerun vs the published snapshot

Replication bundle (production DOI): `10.5281/zenodo.18543606` (https://zenodo.org/records/18543606). This replication record links back to the published snapshot DOI `10.5281/zenodo.18542624`.

Runbook: [Third-Party Replication Runbook](/docs/benchmarks/replication).

## Blog posts (narrative)

- [Trust Offensive: Public Benchmarks](/blog/trust-offensive) — why this exists and how to interpret it.
- [The End of the Scripting Era](/blog/end-of-scripting-era-benchmarks) — how reproducible benchmarking changes how we build scientific software.
- [Third-Party Replication: Signed Reports](/blog/third-party-replication-signed-report) — external reruns + signed validation reports as the ultimate trust signal.
