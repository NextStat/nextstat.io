---
title: "WALNUTS Sampler"
status: stable
---

# WALNUTS Sampler

NextStat ships WALNUTS (Window-Adaptive NUTS) as a public CPU sampler for the
accepted Euclidean HMC subset.

The recommended Python entry point is:

```python
import nextstat as ns

result = ns.sample(model, method="walnuts")
```

`nextstat.sample_walnuts(...)` is also available as an explicit alias, but the
unified `sample(..., method="walnuts")` surface remains the preferred public
entry point.

## Current v1 scope

- CPU only
- Euclidean HMC only
- `metric="diagonal"`, `metric="dense"`, and `metric="auto"`
- same top-level result contract as NUTS:
  `posterior`, `sample_stats`, `diagnostics`, `param_names`, `n_chains`,
  `n_warmup`, `n_samples`

`metric="auto"` selects the dense metric for dimensions `<= 32` and diagonal
otherwise, matching the existing NUTS warmup policy.

CUDA and Metal are not shipped WALNUTS backends today. NextStat's current GPU
sampler line remains `LAPS`. Internal GPU certification now includes both the
HTCondor narrow CUDA seam lane and a V100 / CUDA 12.6 lane built on
`nextstat-bench` and executed on `v100` via `memfd`. That V100 lane currently
covers evaluator-backed linear, logistic, Poisson-with-offset,
NegBin-with-offset, and interval-censored Weibull AFT prototype work, but that
evidence does not change the shipped public WALNUTS surface.

## Defaults

WALNUTS-specific shipped defaults:

- `max_treedepth=10`
- `max_step_halvings=4`
- `min_micro_steps=1`
- `max_energy_error=2.0`
- `target_accept=0.8`
- `target_tree_depth=4.0`
- `init_strategy="random"`
- `metric="diagonal"`
- `stepsize_jitter=0.0`

## Warmup and telemetry

Adaptive WALNUTS uses the same windowed warmup backbone as NUTS for step-size
and Euclidean mass-matrix adaptation, then applies WALNUTS trajectory rules
post-warmup.

Sampling telemetry is intentionally split:

- `sample_stats["n_leapfrog"]`: per-draw post-warmup leapfrog or micro-step counts
- `sample_stats["n_leapfrog_warmup_total"]`: per-chain warmup leapfrog totals

This keeps algorithmic-efficiency metrics honest:

- `ESS/LF` should be computed from post-warmup leapfrogs only
- end-to-end `LF/s` should use warmup + post-warmup leapfrogs over total wall time

## Initialization

Accepted init strategies:

- `"random"`
- `"mle"`
- `"pathfinder"`

When `metric="dense"` (or `metric="auto"` with `dim <= 32`), Pathfinder can
seed WALNUTS with a dense Hessian-derived metric before warmup.

`data=` is supported when sampling models that accept per-call observed data.
When sampling a `Posterior`, `data=` is not supported.

## Quality and benchmarking

Canonical WALNUTS-vs-NUTS parity is tracked on `nextstat-bench` through the
internal harness documented in:

- `docs/benchmarks/suites/bayesian.md`

`nextstat-bench` is a CPU-only EPYC host, so those artifacts certify the CPU
WALNUTS surface only. GPU scope requires a separate GPU-capable benchmark lane.

An internal V100 GPU lane now exists for prototype certification, and the
accepted artifact there now covers evaluator-backed linear, logistic,
Poisson-with-offset, NegBin-with-offset, and interval-censored Weibull AFT
slices. It still remains internal evidence until a broader shipped GPU WALNUTS
surface exists.

That lane is governed by an internal promotion contract rather than a public
backend claim.

That harness is the source of truth for:

- `ESS_bulk/s`
- `ESS_bulk/LF`
- `LF/s`
- divergence rate
- max R-hat
- min E-BFMI

## Source files

- `crates/ns-inference/src/walnuts.rs`
- `crates/ns-inference/src/chain.rs`
- `bindings/ns-py/src/lib.rs`
- `bindings/ns-py/python/nextstat/__init__.py`
- `bindings/ns-py/python/nextstat/_core.pyi`
