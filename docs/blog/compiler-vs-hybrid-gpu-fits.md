---
title: "Compiler-Symbolic vs Hybrid-Neural: GPU-Accelerated Unbinned Fits in HEP"
slug: compiler-vs-hybrid-gpu-fits
description: "Mathematical comparison of MoreFit (symbolic differentiation + JIT OpenCL) and NextStat (analytical CUDA kernels + ONNX flows + reverse-mode AD). Three-tier gradient system, fused NLL kernel, and systematic-aware production constraints."
date: 2026-02-12
author: NextStat Team
status: draft
keywords:
  - GPU-accelerated unbinned fits
  - extended unbinned likelihood
  - analytical gradients
  - symbolic differentiation
  - reverse-mode AD
  - ONNX normalizing flows
  - CUDA
  - Metal
  - systematics
category: hep
---

# Compiler-Symbolic vs Hybrid-Neural: GPU-Accelerated Unbinned Fits in HEP

**Trust Offensive series:** [Index](/blog/trust-offensive) · **Prev:** [Unbinned Event-Level Analysis](/blog/unbinned-event-level-analysis) · **Next:** [Trust Offensive series index](/blog/trust-offensive)

---

## Abstract

Unbinned (event-level) maximum-likelihood fits are GPU-friendly because the expensive part of the objective is an embarrassingly parallel reduction over events. A recent line of work (MoreFit) advocates **compiler-symbolic optimization**: represent PDFs as a symbolic computation graph, differentiate symbolically, and JIT-compile a fused GPU kernel.

NextStat targets the same extended-likelihood objective, but optimizes for a broader task class: **rate systematics**, **hybrid parametric + neural PDFs**, **binned HistFactory integration**, and **reproducible contracts** (schema-validated configuration + correctness gates). The consequence is a hybrid architecture: analytical CUDA/Metal kernels for a conservative parametric subset, reverse-mode AD for general CPU models, and a separate GPU reduction path for ONNX flows.

This post explains the mathematical common ground, the gradient strategies that dominate performance, and the “production constraints” that decide which design wins in real HEP pipelines.

---

## 1. Extended unbinned likelihood (shared objective)

Given observed events \(x_i\) and a mixture model with processes \(p\), NextStat and MoreFit both target the **extended** unbinned likelihood. In a common form:

```text
NLL(θ) = ν_tot(θ) − Σ_i log [ Σ_p ν_p(θ) · p_p(x_i | θ) ] + C_constr(θ)
```

where:

- \(ν_p(θ)\) are per-process expected yields,
- \(p_p(x|θ)\) are normalized PDFs on a bounded observable domain,
- \(C_constr(θ)\) is a sum of constraint penalties (e.g. Gaussian nuisance priors).

The dominant cost is the per-event mixture logsumexp.

**Scaling model:** one objective evaluation is approximately

```text
cost ≈ O(N_events × N_processes)
```

so speed hinges on (1) memory layout, (2) kernel fusion, and (3) gradient evaluation strategy.

---

## 2. MoreFit: compiler optimizations on a computation graph

MoreFit’s core idea is a compiler pipeline:

- express each PDF as a symbolic DAG,
- apply graph optimizations (common subexpression elimination, hoisting of parameter-only terms, etc.),
- generate analytical derivatives symbolically,
- JIT-compile a fused kernel (e.g. OpenCL).

### 2.1 Why symbolic gradients change the game

Finite differences scale poorly: with \(P\) parameters, a central scheme needs \(2P+1\) objective evaluations per optimizer step. Symbolic (analytical) gradients reduce that to a single fused NLL+∇NLL evaluation.

### 2.2 Where compiler-symbolic becomes brittle

The compiler approach is excellent for a closed library of analytical PDFs. It becomes structurally harder as the model expands to include:

- rate modifiers that must be differentiated and validated as part of the contract,
- correctness gates (GPU↔CPU parity, toy closure/coverage),
- non-analytical components (e.g. learned PDFs that are black boxes to the graph).

---

## 3. NextStat: hybrid parametric-neural pipeline

NextStat aims to cover unbinned fits that look like production analyses rather than benchmark micro-cases.

### 3.1 Three-tier gradient system

NextStat uses three gradient regimes, selected by model component:

```text
Tier 1: Reverse-mode AD (ns-ad Tape)
  - general-purpose CPU differentiation for arbitrary model graphs
  - used for: HistFactory binned models and general CPU paths
  - anchor: crates/ns-ad/src/tape.rs

Tier 2: Analytical CUDA/Metal kernels (conservative subset)
  - fused NLL + gradient kernel for a fixed set of parametric PDFs
  - includes rate modifiers in-kernel (NormSys, WeightSys)
  - anchor: crates/ns-compute/kernels/unbinned_nll_grad.cu

Tier 3: Finite differences for ONNX flows
  - flows are treated as black boxes for ∂logp/∂θ
  - reduction (logsumexp + yields + constraints) can still run on GPU
  - anchor: crates/ns-inference/src/gpu_flow_session.rs
```

The architectural decision is not “pick one method.” It is “compose the cheapest correct gradient available for each component.”

### 3.2 What the fused kernel actually buys you

For the parametric subset, Tier 2 turns one optimizer iteration into essentially one kernel launch:

- evaluate mixture logsumexp per event,
- accumulate NLL,
- accumulate ∂NLL/∂θ for supported parameters,
- add yield terms and Gaussian constraints.

This is the same fundamental win MoreFit demonstrates: fuse NLL and gradients into one memory pass.

### 3.3 Systematics as a first-class constraint

For unbinned fits to be usable in HEP workflows, the model must support at least:

- **NormSys** (log-normal style yield modifier),
- **WeightSys** (HistFactory-style interpolation, code0 and code4p),
- Gaussian nuisance constraints.

In NextStat, these are part of the unbinned spec contract (`nextstat_unbinned_spec_v0`) and are supported in the conservative GPU subset.

---

## 4. GPU acceleration without “benchmark theater”: contracts and gates

A GPU kernel is only useful if it is trustworthy. NextStat therefore treats correctness and reproducibility as part of the performance story:

- configuration is schema-validated (`nextstat_unbinned_spec_v0`),
- GPU support is subset-first and explicit (see `docs/gpu-contract.md`),
- GPU↔CPU parity is tested under tolerances.

### 4.1 Observed-data selection and weights

Real analyses frequently require deterministic dataset definition:

- selection cuts,
- expression-based observables,
- non-negative per-event weights (frequency weights).

In NextStat, this is part of the contract:

- For ROOT TTrees: use `channels[].data.selection` and `channels[].data.weight` in `nextstat_unbinned_spec_v0`.
- For Parquet event tables: provide an optional `_weight: Float64` column.

Weights must be finite, non-negative, and non-degenerate (`sum(w_i) > 0`).

---

## 5. Gradient strategies: a compact complexity comparison

Let \(N\) be the number of events, \(P\) the number of parameters, and \(K\) the number of optimizer iterations.

- **Symbolic/analytical gradients (compiler or hand-derived):**

```text
T ≈ K × N × cost_per_event
```

- **Central finite differences (flows or black-box components):**

```text
T ≈ K × (2P+1) × N × cost_per_event
```

NextStat’s hybrid goal is to keep the expensive \((2P+1)\) factor confined to the subset of parameters that truly require it (flow parameters), while retaining analytical gradients for parametric backgrounds, yields, and rate modifiers.

---

## 6. Why ONNX over a symbolic JIT pipeline

A symbolic JIT compiler is compelling when you can represent the full model as an algebraic graph. But unbinned HEP models routinely include high-dimensional PDFs where no simple closed form exists.

ONNX normalizing flows provide a practical “escape hatch”:

- correlations in arbitrary dimension \(D\),
- training-based shape modeling,
- reusability across analysis codebases,
- portability across inference backends.

NextStat’s approach is to:

- use ONNX for components that need it,
- keep the reduction and the “physics scaffolding” (yields, constraints, mixture evaluation, parity gates) in the core inference engine.

---

## 7. R&D: pushing GPU utilization beyond a single fit

### 7.1 Device-resident toy pipeline

Toy studies dominate many validation workflows. NextStat’s CUDA batch-toy pipeline is designed to minimize host↔device traffic:

- sample toy events on device,
- keep the large `obs_flat` buffer device-resident,
- fit many toys in lockstep under L-BFGS-B.

**Implementation anchor:** `crates/ns-inference/src/unbinned_gpu_batch.rs`.

### 7.2 Metal backend (Apple Silicon)

NextStat mirrors the same conservative subset on Metal (f32). Correctness is ensured via parity tests with relaxed tolerances appropriate for f32 + atomic accumulation.

---

## 8. Conclusion

- Compiler-symbolic kernels are an elegant route to peak performance for closed-form parametric PDFs.
- Production HEP analyses need more than peak speed:
  - rate modifiers (NormSys/WeightSys),
  - deterministic dataset definition (selection + weights),
  - correctness gates and reproducibility.
- NextStat’s hybrid design keeps analytical GPU kernels where possible, and uses ONNX flows as an extensible mechanism for complex PDFs.

The outcome is a pipeline that remains applicable when the model stops being “benchmark-clean.”

---

## References

1. C. Langenbruch, “MoreFit: A More Optimised, Rapid and Efficient Fit,” arXiv:2505.12414 (2025).
2. NextStat fused parametric unbinned kernel: `crates/ns-compute/kernels/unbinned_nll_grad.cu`.
3. NextStat GPU flow reduction path: `crates/ns-inference/src/gpu_flow_session.rs`.
4. NextStat reverse-mode AD: `crates/ns-ad/src/tape.rs`.
5. NextStat batch toy fits: `crates/ns-inference/src/unbinned_gpu_batch.rs`.
6. NextStat contracts: `docs/references/unbinned-spec.md`, `docs/gpu-contract.md`.
