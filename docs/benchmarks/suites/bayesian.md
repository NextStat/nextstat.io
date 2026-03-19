---
title: "Benchmark Suite: Bayesian (ESS/sec vs CmdStan + optional PyMC)"
description: "Bayesian inference benchmark suite for NextStat: ESS/sec (bulk + tail), wall-time per effective draw, warmup/adaptation behavior, and SBC calibration — with a committed CmdStan-backed public proof subset and an optional PyMC harness path."
status: stable
last_updated: 2026-03-19
keywords:
  - NUTS sampler benchmark
  - ESS per second
  - Bayesian inference performance
  - Stan comparison
  - PyMC comparison
  - MCMC benchmark
  - simulation-based calibration
  - Hamiltonian Monte Carlo
  - NextStat NUTS
---

# Bayesian Benchmark Suite (ESS/sec vs CmdStan + optional PyMC)

This suite benchmarks Bayesian inference workflows with metrics that matter:

- **ESS/sec** (bulk + tail) and wall-time per effective draw
- diagnostic stability under clearly specified settings

The primary goal is to avoid “apples vs oranges” comparisons by publishing:

- the exact model + priors
- the exact inference settings
- the exact diagnostics and ESS computation policy

## Current promoted public proof subset

The current public SOTA claim for Bayesian is anchored by the committed
`results_v10` CmdStan-backed multiseed bundle:

- `benchmarks/nextstat-public-benchmarks/suites/bayesian/results_v10/bayesian_multiseed_summary.json`
- `benchmarks/nextstat-public-benchmarks/suites/bayesian/results_v10/derived_metrics.json`

That promoted subset covers:

- `histfactory_simple_8p`
- `glm_logistic_regression`
- `hier_random_intercept_non_centered`
- `eight_schools_non_centered`

with `nextstat` vs `cmdstanpy` across seeds `{42, 0, 123}` and zero committed
`warn` / `failed` rows in the promoted multiseed summary.

PyMC remains a supported optional harness path, but it is not part of the
current promoted public proof subset.

Internal sampler-parity coverage for stable product surface work also uses a
NextStat-only head-to-head harness:

- `scripts/benchmarks/bench_sampler_matrix.py` for multi-method CPU sampler discovery on a shared model matrix
- `scripts/benchmarks/bench_sampler_matrix_remote.sh` for generic EPYC remote runs with tmp-backed repo/venv/target reuse
- `scripts/benchmarks/bench_walnuts_vs_nuts.py` for `method="walnuts"` vs `method="nuts"`
- `scripts/benchmarks/bench_walnuts_vs_nuts_remote.sh` for canonical EPYC runs on `nextstat-bench`
- `scripts/benchmarks/bench_walnuts_gpu_lane_preflight.sh` for internal HTCondor GPU-lane preflight from `nextstat-bench`
- `scripts/benchmarks/bench_walnuts_gpu_transfer_smoke.sh` for internal transfer-built CUDA seam smoke via HTCondor execute nodes
- `scripts/benchmarks/bench_walnuts_gpu_cert_runner.sh` for the internal narrow GPU certification runner on that same HTCondor path
- `scripts/benchmarks/bench_walnuts_gpu_v100_cert_runner.sh` for the internal V100 GPU certification lane built on `nextstat-bench` and executed on Tesla V100 + CUDA 12.6

The canonical `nextstat-bench` host is currently a CPU-only EPYC stand. These
artifacts certify CPU sampler scope only; they do not exercise or certify CUDA
or Metal sampler paths.

Internal GPU seam work may still submit HTCondor jobs from `nextstat-bench` to
GPU execute nodes. Those artifacts remain internal execution evidence until a
separate GPU benchmark certification contract exists.

NextStat also keeps an internal V100 GPU lane for true-f64 CUDA
certification. That lane currently builds on `nextstat-bench`, stages the
compiled Rust test harness on `v100`, and executes it there via `memfd`, while
still requiring CUDA 12.6 userland on the V100 host because the accepted Volta
path depends on `compute_70`.

The current CUDA certification surfaces are split explicitly:

- the HTCondor `nextstat-bench -> gex44` lane validates the narrow batched
  step-sequence seam plus probe-only log-joint replay for WALNUTS
  reversibility checks
- the direct V100 lane validates that same narrow cert slice plus
  evaluator-backed prototype slices for linear, logistic, Poisson with
  offsets, Negative Binomial with offsets, and interval-censored Weibull AFT

Those artifacts now serve two different roles:

- the narrow StdNormal seam remains internal certification evidence only
- the evaluator-backed direct-V100 lane is the certification source for the
  shipped narrow public CUDA WALNUTS subset

The latest accepted direct-V100 refresh on March 12, 2026 kept that boundary
honest: the narrow StdNormal seam remains cert-only, while the evaluator-backed
linear, logistic, Poisson-with-offset, Negative Binomial-with-offset, and
interval-censored Weibull AFT slices now provide the positive GPU throughput
evidence for the first shipped public CUDA WALNUTS stable surface.

That same March 12, 2026 refresh also closed the first honest public-surface
acceptance loop for CUDA WALNUTS itself: `ns-py` built on V100 with CUDA 12.6,
`nextstat.has_cuda()` returned `true`, `pytest -k "walnuts and cuda"` passed,
and a representative public `PoissonRegressionModel(..., offset=...)` run via
`nextstat.sample(..., method="walnuts", device="cuda")` delivered about
`1.88x` GPU/CPU wall-time speedup at `n=12000`, `p=8`, `n_warmup=80`,
`n_samples=32`. That acceptance probe is a shipped-surface check, not a
replacement for the broader internal promotion matrix.

That direct V100 lane still carries an explicit promotion boundary. Artifact
metadata and reviewer docs distinguish the shipped narrow CUDA stable surface
from broader future GPU claims.

That harness records both:

- wall-time efficiency: min bulk/tail ESS per second
- algorithmic efficiency: min bulk/tail ESS per post-warmup leapfrog or micro-step count
- throughput: total warmup + post-warmup leapfrogs per end-to-end second
- health metrics: divergence rate, max R-hat, min E-BFMI, mean tree depth

For the internal WALNUTS-vs-NUTS parity harness:

- `ESS_bulk/LF` and `ESS_tail/LF` use post-warmup leapfrog totals only
- `LF/s` uses total warmup + post-warmup leapfrogs over end-to-end wall time
- WALNUTS benchmark runs should use shipped product defaults unless the run is
  explicitly labeled as a parameter sweep

For the generic sampler-matrix harness:

- method-specific config must be recorded explicitly in the artifact metadata
- `NUTS` / `WALNUTS` tree controls and `MAMS` leapfrog controls must not be conflated
- remote discovery batches should reuse the same tmp-backed build after the first cold compile
- sampler-matrix artifacts are internal discovery evidence, not standalone product-regression verdicts
- any discovery gate in that harness must be documented as policy, not implied to equal core diagnostics

## Named competitor baselines

| Case | NextStat method | Competitor | Library version |
| --- | --- | --- | --- |
| HistFactory simple (8p) | `sample(model, method="nuts")` | CmdStan NUTS | CmdStan ≥ 2.35 |
| GLM logistic regression (6p) | `sample(model, method="nuts")` | CmdStan NUTS | CmdStan ≥ 2.35 |
| Hierarchical random intercept NCP (22p) | `sample(model, method="nuts")` | CmdStan NUTS | CmdStan ≥ 2.35 |
| Eight Schools NCP (10p) | `sample(model, method="nuts")` | CmdStan NUTS | CmdStan ≥ 2.35 |
| GLM logistic regression (6p) | `sample(model, method="nuts")` | PyMC NUTS | PyMC ≥ 5.0 |

## What is compared

- NextStat NUTS implementation (Rust core) vs CmdStan NUTS vs PyMC NUTS (where feasible)

## What is measured

### ESS/sec

Report:

- bulk ESS/sec and tail ESS/sec (per parameter group)
- wall-time per effective draw

## Initial public baseline set (recommended)

For the first public Bayesian snapshots, we recommend a small set of models that cover distinct NUTS regimes:

1. **Simple HistFactory** (~8 parameters): fast, validates basic ESS/sec and R-hat convergence in an inference-like likelihood.
2. **Logistic regression (GLM)**: classic benchmark, straightforward to reproduce in Stan/PyMC.
3. **Hierarchical random intercepts** (non-centered): exercises funnel-like geometry and parameterization sensitivity.

Large HEP workspaces (e.g. `tHu`, 184 parameters) are valuable but typically too slow for nightly runs; keep them behind `#[ignore]` or run them as release-only smoke tests.

### Warmup + adaptation behavior

Publish:

- warmup length
- target acceptance
- step-size adaptation policy
- mass matrix policy (diag vs dense) and update schedule

## Protocol requirements (to keep the comparison honest)

- Same model and prior parameterization across frameworks.
- Same effective warmup/sampling budgets (or publish both and justify).
- Same RNG seeding policy (where supported) and deterministic preprocessing.
- Diagnostics must be computed with the same method/version (or explicitly noted).

## Harness entry points (current)

NextStat provides Criterion benches for NUTS in:

- `crates/ns-inference/benches/nuts_benchmark.rs`

Run locally:

```bash
cargo bench -p ns-inference --bench nuts_benchmark
```

For public benchmarks, we will wrap these benches (and external-framework runs) into a single harness that produces:

- raw draws (or summary traces)
- ESS metrics
- environment manifests

Sampler-parity runner inside the product repo:

```bash
python3 scripts/benchmarks/bench_walnuts_vs_nuts.py \
  --out-dir tmp/walnuts_vs_nuts \
  --seeds 42,123,777 \
  --models std_normal_10d,eight_schools,glm_logistic,funnel_ncp_10d,glm_negbin
```

The `bench_walnuts_vs_nuts.py` wrapper defaults to the canonical accepted set,
but it also accepts expansion-review candidates such as `glm_negbin`,
`glm_poisson`, and `funnel_10d` when running admission reviews under the
documented policy.

As of the 2026-03-08 admission batch on `nextstat-bench`, `glm_negbin` is
promoted into the default canonical WALNUTS-vs-NUTS review set. `glm_poisson`
and centered `funnel_10d` remain admission-review targets only.

Generic sampler-matrix runner:

```bash
python3 scripts/benchmarks/bench_sampler_matrix.py \
  --out-dir tmp/sampler_matrix \
  --seeds 42,123,777 \
  --methods nuts,walnuts,mams \
  --models std_normal_10d,eight_schools,glm_logistic,funnel_ncp_10d
```

For non-canonical product-scope reviews such as dense WALNUTS parity, the same
generic harness can be run with `--metric dense` or `--metric auto`. Those
artifacts stay in discovery scope and must not be confused with the canonical
diagonal admission contract below.

All sampler-matrix artifacts now record accelerator runtime metadata
(`cuda_runtime_available`, `metal_runtime_available`, `nvidia_smi_present`) so
host scope is explicit in the JSON/Markdown evidence bundle.

Canonical EPYC remote run:

```bash
BENCH_SSH_KEY=~/.ssh/<bench-key> \
  bash scripts/benchmarks/bench_walnuts_vs_nuts_remote.sh
```

Generic EPYC remote sampler-matrix run:

```bash
BENCH_SSH_KEY=~/.ssh/<bench-key> \
  BENCH_METHODS=nuts,walnuts,mams \
  bash scripts/benchmarks/bench_sampler_matrix_remote.sh
```

The generated `bench_walnuts_vs_nuts.md/json` artifacts report:

- `ESS_bulk/LF` and `ESS_tail/LF` against post-warmup `sample_stats["n_leapfrog"]`
- `LF/s` against total `sample_stats["n_leapfrog_warmup_total"] + sample_stats["n_leapfrog"]`
- `sample_LF`, `warmup_LF`, and `total_LF` explicitly, so warmup work and kept-draw work are not conflated

The generated `bench_sampler_matrix.md/json` artifacts add:

- a method-matrix summary for all requested sampler methods
- `method_specific_config` in metadata, so `max_treedepth` and `max_leapfrog` are not mixed implicitly
- `discovery_policy` in metadata, so stricter admission-review thresholds are explicit instead of being mistaken for public health gates
- `canonical_admission_policy` in metadata for diagonal canonical-review runs; non-diagonal discovery runs mark that policy as not applicable instead of pretending they satisfy the canonical contract

Internal canonical-admission policy for WALNUTS benchmark-set expansion:

- applies only when deciding whether a new posterior family should enter the canonical `WALNUTS` vs `NUTS` review set
- required run contract is the dedicated `scripts/benchmarks/bench_walnuts_vs_nuts.py` harness on `nextstat-bench` with `methods=[nuts, walnuts]`, `seeds=[42,123,777]`, `n_chains=4`, `n_warmup=1000`, `n_samples=1000`, `metric=diagonal`, `target_accept=0.8`, and `max_treedepth=10`
- candidate fixtures must be reproducible, have explicit priors, and emit method-specific config into the artifact
- admission requires both compared methods to clear the documented discovery thresholds in that artifact under shipped product defaults
- admission also requires the candidate to add a materially new posterior geometry class rather than duplicate an already admitted family
- the review runtime budget is capped at `600s` wall-time for the two-method / three-seed batch so the canonical set remains operational on `nextstat-bench`

Standalone public benchmarks harness (seed repo) status:

- runnable implementation: **seed (NextStat + optional Stan/PyMC)**
- suite directory: `benchmarks/nextstat-public-benchmarks/suites/bayesian/`
- CLI (single run): `python3 suites/bayesian/suite.py --out-dir ... --backends nextstat,cmdstanpy,pymc --dataset-seed 12345 --seed 42`
- CLI (assessment): `python3 suites/bayesian/assess.py <suite-dir> --promotion-backend nextstat`
- CLI (multi-seed stability): `python3 suites/bayesian/multiseed.py --out-dir ... --backends nextstat,cmdstanpy --dataset-seed 12345 --seeds 42,0,123`
- CLI (regenerate summary only): `python3 suites/bayesian/multiseed.py --out-dir ... --seeds 42,0,123 --reuse-existing`
- CLI (derived repeatability metrics): `python3 suites/bayesian/derive_metrics.py <multiseed-dir>`
- Canonical remote run: `scripts/benchmarks/bench_bayesian_suite_remote.sh`
- Canonical remote publisher run: `scripts/benchmarks/publish_bayesian_snapshot_remote.sh`
- Canonical remote multi-seed run: `scripts/benchmarks/bench_bayesian_multiseed_remote.sh`
- Publishable artifacts under pinned schemas:
  - `nextstat.bayesian_benchmark_result.v1` per case
- `nextstat.bayesian_benchmark_suite_result.v1` index
- `nextstat.bayesian_assessment.v1` assessment layer
- `nextstat.bayesian_multiseed_summary.v1` repeatability summary
- `nextstat.bayesian_derived_metrics.v2` supplementary repeatability diagnostics

The public Bayesian suite now follows the same separation contract as the canonical MAMS suite:

- `bayesian_suite.json` stays a raw measurement artifact
- `bayesian_assessment.json` / `.md` carry the explicit policy layer for `core_quality` vs `promotion_gate`
- `bayesian_assessment.json` now also includes a machine-readable `review_summary` for worst reviewed-case divergence, treedepth saturation, R-hat, E-BFMI, ESS, and ESS/sec
- the published `README_snippet_bayesian.md` now surfaces that same health verdict directly, so snapshot readers do not need to inspect raw suite rows to discover promotion failure reasons
- `snapshot_index.json` now also surfaces a normalized `suite_health` entry for the Bayesian assessment, so registry/discovery layers can read the same verdict without opening `bayesian/bayesian_assessment.json`
- the commit-backed `snapshot_registry.json` now simply consumes that same `snapshot_index.json` health row, so publisher snapshots, registry pages, and automation all share one Bayesian health vocabulary

This keeps product-quality or promotion claims out of the raw suite schema and avoids conflating public benchmark evidence with internal discovery-policy thresholds.

The canonical repeatability lane on `nextstat-bench` now has its own remote runner as well. This closes the gap where checked-in Bayesian multi-seed evidence could drift behind the current public-suite implementation: the tracked `results_v10` directory can now be refreshed from the same host-backed path, with schema-backed `bayesian_multiseed_summary.json` and `derived_metrics.json`. When you request `BENCH_BACKENDS=nextstat,pymc` or `nextstat,cmdstanpy,pymc`, that same lane now self-provisions PyMC and carries PyMC rows into the derived repeatability metrics as well.

The multiseed summary contract is now health-complete rather than throughput-only:

- per-case summary rows include arrays for `divergence_rate`, `max_treedepth_rate`, `min_ebfmi`, and `min_ess_tail`
- the Markdown summary includes a worst-across-seeds health table
- `--reuse-existing` lets you regenerate those summary artifacts from existing `seed_*` case JSON without paying for another full benchmark rerun

Canonical host-backed validation for the public Bayesian suite now has its own dedicated runner on `nextstat-bench`. The default contract is intentionally dependency-light and uses `backends=[nextstat]`.

The host-backed publisher runner now syncs the entire publish root back from `nextstat-bench`, not just the snapshot directory, so validation bundles also include the publisher-refreshed `snapshot_registry.json`.

The committed registry is now freshness-gated as well: `write_snapshot_registry.py --check` verifies that `manifests/snapshot_registry.json` still matches the committed snapshot bundles, so the registry cannot silently drift behind the published Bayesian evidence.

The committed snapshot set now also includes a host-backed health-complete Bayesian publish bundle at `benchmarks/nextstat-public-benchmarks/manifests/snapshots/bayesian-publisher-20260309T111048Z/`. That makes the committed registry not just drift-free, but also current enough to surface a promotion-passing Bayesian health row straight from tracked `nextstat-bench` evidence.

Optional external-framework backends are now self-provisioning on the remote lane when requested:

- `cmdstanpy` installs the Python package in the remote venv and provisions or reuses a local `vendor/cmdstan/cmdstan-*` toolchain inside the synchronized seed repo snapshot.
- `pymc` installs `pymc` plus `arviz` in the remote venv before executing the suite.
- `BENCH_EXTRA_PIP_PACKAGES` remains an additive escape hatch for non-default stacks such as `numpyro`; it is no longer the standard path for `cmdstanpy` or `pymc`.

When vendor CmdStan exists in the synced seed repo snapshot, the runtime now prefers it over ambient host `~/.cmdstan` state and records that exact vendor path in `backend_meta.cmdstan_path`. This keeps the canonical promoted lane stable while still making cross-framework host-backed evidence reproducible on demand without pre-baking optional dependencies into the base `nextstat-bench` image.

For the PyMC lane, the remote runner now exports `PYTENSOR_FLAGS=blas__ldflags=-lblas` by default unless the operator has already set `PYTENSOR_FLAGS`. The runtime records the effective PyTensor mode and BLAS flags in `backend_meta`, so the compute path used for host-backed evidence is explicit and reproducible.

For the canonical `histfactory_simple_8p` case, `cmdstanpy` and `pymc` now implement the exact restricted mapping of the tracked `simple_workspace.json` fixture:

- `mu` remains the bounded POI on `[0, 10]`
- per-bin `shapesys` nuisances remain positive latent scales
- the auxiliary term remains the corresponding Poisson/Gamma Barlow-Beeston constraint built from the workspace `tau = (nominal / sigma)^2`

This is a deliberate stable-surface contract for the tracked simple HistFactory fixture, not a claim of general HistFactory translation support across arbitrary Stan/PyMC workspaces. `numpyro` still reports `backend_not_supported_for_model` for this case.

Seeding policy:

- `dataset_seed` controls generated dataset content (fixed for publishable comparisons).
- `seed` controls chain RNG (varied across repeats for stability).

Dependency note (seed repo):

- Core harness deps are pinned and minimal.
- Optional backends require extra deps (and for Stan, **CmdStan** binaries). Missing optional deps are reported as `status="warn"` with a machine-readable `reason` (for example `missing_backend_dep:cmdstanpy:ImportError:...`), rather than failing the whole snapshot.

## SBC calibration suite (shipped)

The SBC (Simulation-Based Calibration) suite validates posterior correctness independently of performance:

- Generates synthetic datasets from the prior
- Runs NUTS sampling on each
- Checks rank uniformity of posterior draws vs true parameters

This is included in the Apex2 master report under the `sbc` key.

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_sbc_report.py \
  --out tmp/apex2_sbc_report.json
```

The NUTS quality smoke suite additionally checks divergence rate, R-hat, ESS, and E-BFMI floors:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_nuts_quality_report.py \
  --out tmp/apex2_nuts_quality_report.json
```

## Threats to validity (things we will document explicitly)

- Different gradient implementations (analytical vs AD) change the cost model.
- Parameterization differences (centered vs non-centered) dominate sampler efficiency.
- BLAS backend differences can dominate linear algebra-heavy models.

## Related reading

- [Public Benchmarks Specification](/docs/public-benchmarks) — canonical spec.
- [Validation Report Artifacts](/docs/validation-report) — validation pack for published snapshots.
