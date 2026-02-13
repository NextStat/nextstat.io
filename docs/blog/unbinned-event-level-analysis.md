---
title: "Unbinned Event-Level Analysis: Why Fit Every Event"
slug: unbinned-event-level-analysis
description: "A contract-first guide to extended unbinned likelihood in NextStat: event-level PDFs, Parquet ingestion, correctness gates (closure/coverage + RooFit reference), and a conservative GPU acceleration subset."
date: 2026-02-10
author: NextStat Team
status: draft
keywords:
  - unbinned likelihood
  - extended likelihood
  - event-level analysis
  - Crystal Ball
  - KDE
  - Parquet
  - GPU acceleration
  - reproducible validation
  - NextStat
category: hep
---

# Unbinned Event-Level Analysis: Why Fit Every Event

**Trust Offensive series:** [Index](/blog/trust-offensive) · **Prev:** [Where ROOT Gets It Wrong](/blog/numerical-accuracy) · **Next:** [Differentiable HistFactory in PyTorch](/blog/differentiable-layer)

---

## Abstract

Binned likelihoods (HistFactory-style) are the default for HEP because they scale well and integrate cleanly with systematic uncertainty models. But binning is a lossy projection: it discards intra-bin information and becomes brittle in low-statistics, narrow-resonance, or multi-dimensional settings.

This post is a **contract-first** guide to unbinned (event-level) analysis in NextStat 0.9.0:

- the exact extended likelihood form we minimize,
- the event storage and I/O contract (ROOT TTree and Parquet),
- the declarative model spec (`nextstat_unbinned_spec_v0`),
- correctness gates (closure/coverage and RooFit reference),
- and a conservative GPU subset for NLL+gradient evaluation.

The goal is not “an unbinned demo.” The goal is a workflow that can be rerun, audited, and regression-tested.

**Canonical references in this repo:**

- Unbinned spec: `docs/references/unbinned-spec.md`
- Unbinned Parquet schema: `docs/references/unbinned-parquet-schema.md`
- GPU contract: `docs/gpu-contract.md`

---

## 0. Versions and environment (for reproducibility)

| Component | Version |
|---|---|
| NextStat | 0.9.0 (git `2326065`, 2026-02-10) |
| Rust toolchain | 1.93.0 (`rust-toolchain.toml`) |
| Python | 3.11+ (`bindings/ns-py/pyproject.toml`) |
| NumPy | `numpy>=2.0` (optional deps) |
| pyhf | `pyhf>=0.7.6` (optional deps: `nextstat[validation]`) |
| SciPy | required only for RooFit reference data generation (`tests/validate_roofit_unbinned.py` imports `scipy.stats.crystalball`) |
| ROOT | required only for RooFit comparison (`root` on PATH); version used for reference runs: 6.38.00 |
| CUDA | optional (`cargo build --features cuda`) |
| Metal | optional (`cargo build --features metal`) |

---

## 1. Problem statement: when binning destroys information

Binning turns a continuous observation `x` into an integer “bin index.” When bins are coarse relative to the signal scale, or when event counts are low, that projection loses power.

Common regimes where unbinned likelihood is not optional:

- **Narrow resonances:** if your signal width is comparable to (or smaller than) the bin size, binning smears shape information.
- **Low statistics:** with O(10–100) events in the region of interest, “bin optimization” becomes an analysis artifact.
- **High dimension:** N-dimensional binning costs O(k^N) bins, most of which are empty.

The flip side is compute: unbinned NLL evaluation scales with the number of events.

---

## 2. Formalization

### 2.1 Binned (HistFactory-style) likelihood

For a binned model, the likelihood is a product of Poisson factors over bins with nuisance-parameter constraints:

```text
L_binned = ∏_bins Poisson(n_b | ν_b(θ)) × ∏_NP Constraint(θ_j)
```

where `ν_b(θ)` is the expected count in bin `b` after applying modifiers.

### 2.2 Extended unbinned likelihood

For an unbinned mixture model with processes `p` (signal + backgrounds), NextStat uses the **extended** likelihood:

```text
L_unbinned = exp(-ν_total) × ∏_events [ Σ_processes  ν_p · f_p(x_i | θ) ] × ∏_NP Constraint(θ_j)

ν_total = Σ_processes ν_p
```

The corresponding negative log-likelihood (NLL) minimized in MLE fits:

```text
NLL(θ) = ν_total − Σ_events log( Σ_processes ν_p · f_p(x_i | θ) ) + Σ_NP penalty(θ_j)
```

**Implementation anchor:** `crates/ns-unbinned/src/model.rs` implements the extended mixture model and its constraint terms.

---

## 3. Compute scaling: why unbinned is expensive

A single NLL evaluation requires, for each event, computing a process mixture and then a log:

```text
cost(NLL) ≈ O(N_events × N_processes)
```

Gradient evaluation adds the cost of per-event derivatives of `log f(x|θ)` for each process’ shape parameters.

This is why unbinned analysis is architecture-sensitive:

- memory layout (SoA vs AoS),
- batched PDF evaluation,
- reduction strategy,
- and (when applicable) GPU acceleration.

---

## 4. EventStore: the event-level memory contract

NextStat stores event-level data in a columnar **Structure-of-Arrays** layout:

- one `Vec<f64>` per observable,
- optional `weights: Vec<f64>`,
- plus per-observable bounds used for normalization.

**Implementation anchor:** `crates/ns-unbinned/src/event_store.rs`.

### 4.1 Bounds are part of the statistical contract

Bounds are not cosmetic metadata: they define the normalization domain Ω used by PDFs.

Event ingestion validates:

- observable values are finite,
- values are within bounds (when bounds are finite),
- column lengths match.

### 4.2 Weight policy (threat model)

Observed event weights are treated as **non-negative frequency weights**:

- each `w_i` must be finite,
- each `w_i >= 0`,
- `sum(w_i) > 0` (all-zero weights are rejected),
- negative weights are rejected.

Weights are provided either via a ROOT `TTree` weight expression (`channels[].data.weight` in `nextstat_unbinned_spec_v0`) or via an optional Parquet column `_weight: Float64`.

**Implementation anchor:** `EventStore::from_columns` validation in `crates/ns-unbinned/src/event_store.rs`.

This rule is a deliberate contract: negative weights break the probabilistic interpretation of extended likelihood and create optimizer pathologies.

---

## 5. Event I/O: ROOT TTree and Parquet

### 5.1 ROOT TTree

NextStat can read event-level columns from TTrees via the native ROOT reader (no ROOT C++ dependency), including:

- expression-based observables,
- selection cuts,
- and a nominal weight expression.

### 5.2 Parquet: unbinned event table schema v1

For reproducibility and ecosystem interoperability, NextStat defines a strict Parquet contract:

- one row per event,
- one `Float64` column per observable,
- optional `_weight: Float64`,
- optional `_channel: Utf8` for multi-channel files,
- NextStat-written Parquet files embed key-value metadata:
  - `nextstat.schema_version = "nextstat_unbinned_events_v1"`
  - `nextstat.observables = JSON array of {name,bounds}`

When this metadata is absent (e.g., files authored outside NextStat), the Parquet reader can still ingest the event table, but observable bounds must be supplied by the unbinned spec; otherwise the reader falls back to `(-inf, inf)` bounds and the normalization contract becomes the caller’s responsibility.

**Implementation anchor:** `crates/ns-unbinned/src/event_parquet.rs`.

**Canonical reference:** `docs/references/unbinned-parquet-schema.md`.

---

## 6. The declarative model contract: `nextstat_unbinned_spec_v0`

NextStat’s unbinned workflow is driven by a YAML/JSON spec with a strict schema:

- `schema_version: nextstat_unbinned_spec_v0`
- `model.parameters`: name/init/bounds (+ optional Gaussian constraint)
- `channels[]`: data source + observables + processes

**Canonical reference:** `docs/references/unbinned-spec.md`.

### 6.1 CLI entry points

```bash
nextstat config schema --name unbinned_spec_v0

nextstat unbinned-fit --config model.yaml
nextstat unbinned-scan --config model.yaml --start 0 --stop 5 --points 51
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --seed 42
nextstat unbinned-ranking --config model.yaml
nextstat unbinned-hypotest --config model.yaml --mu 1.0
nextstat unbinned-hypotest-toys --config model.yaml --mu 1.0 --n-toys 1000 --seed 42 --expected-set
```

**Implementation anchor (CLI flags):** `crates/ns-cli/src/main.rs`.

### 6.2 Fit output is also a contract

The JSON output of `unbinned-fit` is validated in CLI tests and minimally includes:

- `input_schema_version = nextstat_unbinned_spec_v0`
- `parameter_names[]`, `bestfit[]`, `uncertainties[]`
- `nll`, `converged`, `n_iter`

**Implementation anchor:** `crates/ns-cli/tests/cli_unbinned_fit.rs`.

---

## 7. Correctness is a gate: closure, coverage, RooFit reference

Unbinned analysis is compute-heavy and easy to get subtly wrong. NextStat treats correctness as a **precondition** for any performance number.

### 7.1 Closure + coverage (regression tests)

The Python regression suite enforces two statistical sanity properties:

- **Closure:** on a large synthetic dataset, fitted parameters recover truth within tolerances.
- **Coverage (slow, opt-in):** across toys, truth should fall within the nominal 1σ interval at an acceptable rate.

**Source of truth (tolerances):** `tests/python/_tolerances.py`

- `UNBINNED_CLOSURE_PARAM_ATOL = 0.15`
- `UNBINNED_CLOSURE_PARAM_RTOL = 0.10`
- `UNBINNED_COVERAGE_1SIGMA_LO = 0.55`
- `UNBINNED_COVERAGE_1SIGMA_HI = 0.98`

**Implementation anchor (tests):** `tests/python/test_unbinned_closure_coverage.py`.

### 7.2 RooFit reference (independent implementation check)

For canonical 1D models (Gaussian+Exponential, CrystalBall+Exponential), NextStat provides an independent RooFit comparison:

- generates synthetic data → Parquet,
- runs `nextstat unbinned-fit`,
- runs RooFit via a generated C++ macro,
- compares NLL and best-fit parameters,
- writes a deterministic JSON artifact.

**Implementation anchor:** `tests/validate_roofit_unbinned.py`.

RooFit comparison tolerances (intentionally generous due to normalization convention differences):

- `NLL_ATOL = 0.5`
- `PARAM_ATOL = 0.3`
- `PARAM_RTOL = 0.10`

Run:

```bash
python tests/validate_roofit_unbinned.py --cases gauss_exp,cb_exp --seed 42 --output tmp/validate_roofit_unbinned.json
```

---

## 8. GPU acceleration: conservative subset, explicit contract

NextStat’s GPU support is intentionally **subset-first**: only configurations with well-defined parity and bounded behavior are accelerated.

### 8.1 What the GPU path accelerates

GPU kernels accelerate the fused evaluation of:

- extended unbinned NLL,
- and its gradient,

while the optimizer remains the same (L-BFGS-B with box constraints).

### 8.2 Supported unbinned PDFs on GPU

The GPU subset for unbinned likelihood supports:

- `gaussian`
- `exponential`
- `crystal_ball`
- `double_crystal_ball`
- `chebyshev` (order ≤ 16)
- `histogram`

**Implementation anchor:** GPU parity tests in `crates/ns-compute/tests/unbinned_gpu_parity.rs`.

### 8.3 Supported yields and modifiers

GPU unbinned supports:

- yield forms: `fixed`, `parameter`, `scaled`
- rate modifiers: `normsys`, `weightsys`
- Gaussian constraints on nuisance parameters

(Within the subset that can be lowered to the GPU model representation.)

### 8.4 CLI flags

```bash
nextstat unbinned-fit --config model.yaml --gpu cuda
nextstat unbinned-fit --config model.yaml --gpu metal

nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu cuda
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu metal

# Experimental: sample toys on GPU (requires --gpu cuda|metal)
nextstat unbinned-fit-toys --config model.yaml --n-toys 1000 --gpu cuda --gpu-sample-toys
```

**Implementation anchor (CLI flags):** `crates/ns-cli/src/main.rs`.

### 8.5 CPU fused kernel snapshot (Gauss+Exp)

For the Gauss+Exp topology, NextStat uses a fused single-pass CPU path with SIMD
(`wide::f64x4`) and an explicit generic fallback path. The benchmark suite now
exposes both paths directly for apples-to-apples comparison:

- fused: `unbinned_nll/sig_bkg_gaussian_exp/*`
- generic: `unbinned_nll/sig_bkg_gaussian_exp_generic/*`
- fused grad: `unbinned_grad_nll/sig_bkg_gaussian_exp/*`
- generic grad: `unbinned_grad_nll/sig_bkg_gaussian_exp_generic/*`

Snapshot (Criterion medians, local run, 2026-02-11):

| Metric | N | Fused | Generic | Speedup |
|---|---:|---:|---:|---:|
| NLL | 1k | 9.188 µs | 58.479 µs | 6.36× |
| NLL | 10k | 41.751 µs | 112.12 µs | 2.69× |
| NLL | 100k | 185.54 µs | 565.08 µs | 3.05× |
| NLL+grad | 1k | 18.162 µs | 78.004 µs | 4.30× |
| NLL+grad | 10k | 58.568 µs | 186.65 µs | 3.19× |
| NLL+grad | 100k | 325.09 µs | 887.25 µs | 2.73× |

### 8.6 PF3.1 multi-GPU toy matrix snapshot (2x H100, CUDA)

Runtime matrix snapshot from `benchmarks/unbinned/artifacts/2026-02-11/pf31_matrix_20260211T182402Z/summary.json`
(stand: TensorDock 2x H100 80GB, CUDA, 4 vCPU):

| Mode | 10k toys | 50k toys | 100k toys |
|---|---:|---:|---:|
| CPU | 117 s | 588 s | 1147 s |
| 1x GPU host toys (`--gpu-devices 0`) | 134 s | 676 s | 1345 s |
| 2x GPU host toys (`--gpu-devices 0,1`) | 71 s | 342 s | 701 s |
| 1x GPU device-resident (`--gpu-sample-toys --gpu-shards 4`) | 135 s | 654 s | 1318 s |
| 2x GPU device-resident (`--gpu-sample-toys --gpu-shards 8 --gpu-devices 0,1`) | 144 s | 672 s | 1316 s |

Observed on this stand:

- Host-toy multi-GPU path scales close to 2x vs 1-GPU host path (about 1.9-2.0x).
- Device-resident sharded path is near-flat vs 1-GPU for this workload (no meaningful speedup at 100k toys).
- At 100k toys, 2x GPU host path is faster than CPU on this node (701 s vs 1147 s).

### 8.7 Single-GPU sanity snapshot (RTX 4000 Ada, CUDA)

Runtime snapshot from:

- `benchmarks/unbinned/artifacts/2026-02-12/pf31_opt2_valid_20260212T024003Z`
- `benchmarks/unbinned/artifacts/2026-02-12/pf31_opt2_focus_20260212T031632Z`

| Mode | 10k toys | 20k toys |
|---|---:|---:|
| CPU | 10.24 s | 20.35 s |
| 1x GPU host sharded (`--gpu-shards 4`) | 291.76 s | 578.42 s |
| 1x GPU device-resident sharded (`--gpu-sample-toys --gpu-shards 4`) | 291.04 s | 577.02 s |

Observed on this stand:

- For this model/workload, CPU fused path is much faster than single-GPU toy pipelines.
- `--gpu-sample-toys` lowers sampling time, but wall-time remains dominated by fit compute.
- This is expected behavior on some small/simple workloads and is a key reason to keep runtime policy explicit (do not assume GPU is always faster).

### 8.8 Toy VRAM scaling note (why sharding is required)

Toy pipelines materialize a **flat event buffer** whose size is proportional to the expected event
count at the toy generation point:

```text
bytes_per_toy ≈ E[N_events] × N_obs × sizeof(f64) × safety
```

This matters for realistic R&D workloads (e.g. O(2M) events with O(10k) toys): you cannot fit the
full ensemble into GPU memory as a single batch. The correct approach is **sharding/batching**
(e.g. `--gpu-shards` or auto-shard policy) so each shard fits in device VRAM.

### 8.9 CPU parallelism note (toys vs events)

On CPU, there are two independent sources of parallelism:

- **Across toys** (each toy is an independent fit).
- **Across events** inside a single NLL/gradient evaluation.

Naively enabling both at the same time leads to nested Rayon parallelism (lots of small tasks, poor cache locality).
NextStat therefore applies a simple heuristic for CPU toy batches:

- If `n_toys` is small relative to the thread count, run toys sequentially and parallelize the per-event loop.
- If `n_toys` is large, parallelize across toys and run the per-event loop sequentially on each worker.

`--threads 0` uses Rayon defaults (auto). Use `--threads 1` for deterministic runs.

### 8.10 CPU farm orchestration for HEP cluster runs

For CPU-only cluster studies (e.g. large toy ensembles when GPU nodes are scarce),
the repo now includes a minimal SSH-based farm workflow:

- `scripts/farm/preflight_cluster.py` — per-node logical/physical core probe.
- `scripts/farm/run_unbinned_fit_toys_cluster.py` — deterministic toy sharding
  by host weight (default physical cores), with per-shard `toy_start` and seed.
- `scripts/farm/merge_unbinned_toys_results.py` — shard output merge back into a
  single `unbinned-fit-toys`-compatible result artifact.

This path is explicitly designed for HEP-scale runs where total load is large
(`O(10k)` toys and beyond) and temporary compute stands must not lose artifacts.

### 8.11 GEX44 recovery snapshot (2026-02-13): CUDA is faster than CPU on toy batches

Latest rerun on GEX44 after CUDA optimizer stabilization and branch-level routing updates:

- stand: GEX44, RTX 4000 SFF Ada 20GB, CUDA 12.0
- artifacts:
  - `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_20260213T091253Z/summary_200.json`
  - `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_scale_20260213T091410Z/summary_scale.json`
  - `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_scale_20260213T091410Z/summary_2m_50.json`
  - `benchmarks/unbinned/artifacts/2026-02-13/gex44_cuda_opt1_scale_20260213T091410Z/summary_cb_dcb_scale.json`

10k-event spec (Gauss+Exp):

| Case | Toys | Pipeline | Wall (s) | Converged | Throughput (toys/s) |
|---|---:|---|---:|---:|---:|
| CPU | 1k | `cpu_batch` | 7.36 | 1000/1000 | 135.90 |
| CUDA (default flags) | 1k | `cuda_gpu_native` | 1.26 | 1000/1000 | 791.27 |
| CUDA (`--gpu-native`) | 1k | `cuda_gpu_native` | 1.16 | 1000/1000 | 860.30 |
| CPU | 10k | `cpu_batch` | 73.98 | 10000/10000 | 135.17 |
| CUDA (default flags) | 10k | `cuda_gpu_native` | 14.68 | 10000/10000 | 681.20 |
| CUDA (`--gpu-native`) | 10k | `cuda_gpu_native` | 14.61 | 10000/10000 | 684.63 |
| CPU | 50k | `cpu_batch` | 382.29 | 50000/50000 | 130.79 |
| CUDA | 50k | `cuda_gpu_native` | 82.26 | 50000/50000 | 607.85 |

Large-event stress spec (~2M events/toy, Gauss+Exp):

- spec: `benchmarks/unbinned/specs/pf31_gauss_exp_2m.json`

| Case | Toys | Pipeline | Wall (s) | Converged | Throughput (toys/s) |
|---|---:|---|---:|---:|---:|
| CPU | 50 | `cpu_batch` | 22.66 | 50/50 | 2.21 |
| CUDA | 50 | `cuda_gpu_native` | 7.80 | 50/50 | 6.41 |

10k-event CrystalBall / DoubleCrystalBall scale slice:

| Case | Toys | Pipeline | Wall (s) | Converged | Throughput (toys/s) |
|---|---:|---|---:|---:|---:|
| CB CPU | 1k | `cpu_batch` | 38.38 | 1000/1000 | 26.05 |
| CB CUDA | 1k | `cuda_gpu_native` | 2.26 | 1000/1000 | 442.90 |
| CB CPU | 10k | `cpu_batch` | 384.51 | 10000/10000 | 26.01 |
| CB CUDA | 10k | `cuda_gpu_native` | 24.99 | 10000/10000 | 400.11 |
| DCB CPU | 1k | `cpu_batch` | 89.82 | 1000/1000 | 11.13 |
| DCB CUDA (default) | 1k | `host` | 2.74 | 1000/1000 | 364.45 |
| DCB CUDA (`--gpu-native`) | 1k | `cuda_gpu_native` | 2.44 | 1000/1000 | 409.34 |
| DCB CPU | 10k | `cpu_batch` | 886.32 | 10000/10000 | 11.28 |
| DCB CUDA (default) | 10k | `host` | 31.25 | 10000/10000 | 320.04 |
| DCB CUDA (`--gpu-native`) | 10k | `cuda_gpu_native` | 20.08 | 10000/10000 | 498.05 |

Observed on this stand:

- CUDA now outperforms CPU in this toy pipeline by ~4.6-6.3x on the 10k-event spec (1k/10k/50k toys).
- On a much heavier ~2M-events/toy stress case, CUDA still wins (~2.9x).
- On CB/DCB 10k-event workloads, CUDA is significantly faster than CPU in all listed runs:
  - CB: ~15-17x vs CPU.
  - DCB (default host CUDA path): ~28-33x vs CPU.
  - DCB `--gpu-native`: additional gain over DCB host CUDA path (~1.12x at 1k toys, ~1.56x at 10k toys).
- Convergence is 100% in these runs (no fit errors).
- Route behavior in this branch snapshot is topology-dependent: Gauss/CB runs in this matrix report `cuda_gpu_native` with current flags, while DCB default `--gpu cuda` reports `pipeline=host` and switches to `cuda_gpu_native` only with explicit `--gpu-native`.

### 8.12 CUDA optimizer stabilization (pass 1, 2026-02-13)

To address iteration-tail dominated CUDA toy-fit wall-time, NextStat added an
optimizer stabilization pass in both CUDA-native and lockstep codepaths:

- GPU-native kernel (`unbinned_batch_lbfgs_fit`) now stops not only on projected
  gradient threshold but also on **relative objective decrease** (L-BFGS-style),
  and exits early with `failed` status when NLL/gradients become non-finite.
- Lockstep state machine (`LbfgsState`) now applies the same relative objective
  stop criterion, so host lockstep and CUDA-native paths share a consistent
  early-stop behavior.

This pass targets pathological "run to max_iter" tails. Quantitative impact is
measured in the next GEX44 rerun matrix (10k/100k toy scale).

---

## 9. Limitations and future work

Unbinned analysis is feature-complete under a strict contract, but not everything is accelerated or supported everywhere.

- GPU support is a conservative subset (explicitly enumerated above).
- Multi-dimensional unbinned PDFs exist (`KdeNdPdf`) but not all are GPU-lowered.
- Some nonparametric PDFs (KDE, spline) and neural PDFs have their own acceleration story and constraints.

For neural PDFs (ONNX-backed flows), see:

- `docs/neural-density-estimation.md`

---

## 10. Conclusion

- Unbinned likelihood removes binning artifacts at the cost of compute.
- In NextStat, unbinned analysis is defined by explicit contracts:
  - `nextstat_unbinned_spec_v0` (model spec),
  - `nextstat_unbinned_events_v1` (event Parquet table + metadata),
  - correctness gates (closure/coverage + RooFit reference).
- GPU acceleration exists as a conservative subset with explicit parity evidence.

---

## References

1. Cowan et al., “Asymptotic formulae for likelihood-based tests of new physics,” arXiv:1007.1727.
2. HistFactory model reference (ATLAS): ATL-PHYS-PUB-2019-029 (pyhf).
3. NextStat unbinned spec reference: `docs/references/unbinned-spec.md`.
4. NextStat unbinned event-level Parquet schema: `docs/references/unbinned-parquet-schema.md`.
5. GPU contract: `docs/gpu-contract.md`.
6. Implementation: `crates/ns-unbinned/src/model.rs`, `crates/ns-unbinned/src/event_store.rs`, `crates/ns-unbinned/src/event_parquet.rs`.
7. Correctness gates: `tests/python/test_unbinned_closure_coverage.py`, `tests/validate_roofit_unbinned.py`.
