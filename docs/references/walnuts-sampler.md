---
title: "WALNUTS Sampler"
status: stable
---

# WALNUTS Sampler

NextStat ships WALNUTS (Window-Adaptive NUTS) as a stable public sampler for
the accepted Euclidean HMC subset.

The recommended Python entry point is:

```python
import nextstat as ns

result = ns.sample(model, method="walnuts")
```

`nextstat.sample_walnuts(...)` is also available as an explicit alias, but the
unified `sample(..., method="walnuts")` surface remains the preferred public
entry point.

## Current stable scope

- CPU:
  `metric="diagonal"`, `metric="dense"`, and `metric="auto"`
- CUDA:
  `device="cuda"` for
  `LinearRegressionModel`,
  `LogisticRegressionModel`,
  `PoissonRegressionModel`,
  `NegativeBinomialRegressionModel`,
  and `IntervalCensoredWeibullAftModel`
- Euclidean HMC only
- same top-level result contract as NUTS:
  `posterior`, `sample_stats`, `diagnostics`, `param_names`, `n_chains`,
  `n_warmup`, `n_samples`

`metric="auto"` selects the dense metric for dimensions `<= 32` and diagonal
otherwise, matching the existing NUTS warmup policy. CUDA currently supports
only `metric="diagonal"` on the stable public surface and does not accept
`Posterior` inputs; pass a supported model directly.

Metal is not a shipped WALNUTS backend today. NextStat's broader GPU sampler
line still includes `LAPS`, but WALNUTS now also ships a narrow CUDA stable
surface for the supported model families above.

The accepted March 12, 2026 GPU landing keeps the remaining non-claims honest:
the narrow StdNormal seam is still cert/parity-only, Metal is still not a
WALNUTS backend, and broader CUDA family coverage beyond the shipped subset
remains future work.

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

## CUDA stable surface boundary

The first shipped CUDA WALNUTS surface is intentionally narrow:

- supported entry points:
  `nextstat.sample(model, method="walnuts", device="cuda", ...)`
  and `nextstat.sample_walnuts(model, device="cuda", ...)`
- supported model families:
  `LinearRegressionModel`,
  `LogisticRegressionModel`,
  `PoissonRegressionModel`,
  `NegativeBinomialRegressionModel`,
  `IntervalCensoredWeibullAftModel`
- supported metric:
  `metric="diagonal"` only
- unsupported on the shipped CUDA surface:
  `Posterior`, dense metric, auto metric, and Metal

The public CUDA surface uses one visible CUDA device per call and keeps the
same result schema as CPU WALNUTS. Multi-chain calls are part of the stable
surface, but they are not yet a claim of cross-device scaling or multi-GPU
execution.

The March 12, 2026 acceptance loop for that shipped CUDA subset was closed on
direct V100 hardware through the public Python surface itself: `ns-py` built
with CUDA 12.6, `pytest -k "walnuts and cuda"` passed, and a representative
public `PoissonRegressionModel(..., offset=...)` run through
`nextstat.sample(..., method="walnuts", device="cuda")` beat the CPU path by
about `1.88x` on wall time at `n=12000`, `p=8`, `n_warmup=80`,
`n_samples=32`. That acceptance probe does not replace the broader internal GPU
promotion matrix; it proves the shipped narrow CUDA surface is real.

## Quality and benchmarking

Canonical CPU WALNUTS-vs-NUTS parity is tracked on `nextstat-bench` through the
internal harness documented in:

- `docs/benchmarks/suites/bayesian.md`

`nextstat-bench` is a CPU-only EPYC host, so those artifacts certify the CPU
WALNUTS surface only. GPU scope uses a separate GPU-capable benchmark lane.

A direct V100 GPU lane now certifies the shipped CUDA stable subset. The
accepted March 12, 2026 landing includes parity and throughput evidence for the
evaluator-backed linear, logistic, Poisson-with-offset,
NegBin-with-offset, and interval-censored Weibull AFT slices on real-f64
hardware. That same lane still keeps the narrow StdNormal seam as internal
cert-only evidence.

That lane is still governed by an internal promotion contract for any broader
future GPU expansion, but it is now also the source of truth for the shipped
CUDA WALNUTS subset above.

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
