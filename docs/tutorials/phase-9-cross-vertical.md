---
title: "Phase 9: Cross-Vertical Statistical Features — Tutorial"
status: draft
---

# Phase 9: Cross-Vertical Statistical Features

This tutorial covers the cross-vertical statistical models added to NextStat's `ns-inference` crate:

- **Gamma GLM** and **Tweedie GLM** — generalized linear models for non-negative responses
- **GEV** and **GPD** — Extreme Value Theory distributions for rare events
- **Meta-analysis** — fixed-effects and random-effects pooling of study results

All `LogDensityModel` implementations (Gamma, Tweedie, GEV, GPD) work with the standard
`nextstat.fit()` / `nextstat.MaximumLikelihoodEstimator` and NUTS sampler. Meta-analysis
uses direct closed-form computation (no optimization needed).

## Prerequisites

```bash
# From repo root — install the Python package in dev mode
cd bindings/ns-py
pip install -e .
```

## 1. Gamma GLM

Gamma regression models strictly positive continuous data with a log link function.
Useful for insurance claim amounts, hospital costs, duration data.

```python
import nextstat
import numpy as np

np.random.seed(42)
n = 200
x = np.random.randn(n, 2).tolist()
# Simulate: log(mu) = 1.0 + 0.5*x1 - 0.3*x2, alpha = 2.0
beta_true = [1.0, 0.5, -0.3]
mu = [np.exp(beta_true[0] + beta_true[1]*xi[0] + beta_true[2]*xi[1]) for xi in x]
y = [np.random.gamma(shape=2.0, scale=mi/2.0) for mi in mu]

model = nextstat.GammaRegressionModel(x, y, include_intercept=True)
print(f"Parameters: {model.parameter_names()}")
print(f"Dimension: {model.dim()}")

result = nextstat.fit(model)
print(f"Estimates: {dict(zip(model.parameter_names(), result.parameters))}")
```

**Parameters:** `[intercept, x1, x2, log_alpha]` — regression coefficients β plus the
log-transformed shape parameter α. The log link ensures μ = exp(Xβ) > 0.

## 2. Tweedie GLM

Tweedie regression handles data with exact zeros and a continuous positive component
(compound Poisson-Gamma). Power parameter `p ∈ (1, 2)` interpolates between Poisson (p=1)
and Gamma (p=2). Useful for insurance aggregate claims, rainfall amounts.

```python
model = nextstat.TweedieRegressionModel(x, y, p=1.5, include_intercept=True)
print(f"Power: {model.power()}")
print(f"Parameters: {model.parameter_names()}")

result = nextstat.fit(model)
print(f"Estimates: {dict(zip(model.parameter_names(), result.parameters))}")
```

**Parameters:** `[intercept, x1, x2, log_phi]` — regression coefficients β plus the
log-transformed dispersion parameter φ. The saddle-point NLL approximation (Dunn & Smyth 2005)
handles both the zero mass P(Y=0) = exp(−λ) and the continuous density for Y > 0.

## 3. GEV Distribution (Block Maxima)

The Generalized Extreme Value distribution models block maxima (e.g. annual maximum
river levels, maximum daily temperatures). Three sub-families:

- **Fréchet** (ξ > 0) — heavy-tailed (finance, insurance)
- **Gumbel** (ξ ≈ 0) — light-tailed (climate, hydrology)
- **Weibull** (ξ < 0) — bounded upper tail

```python
# Annual maximum river levels (meters)
annual_maxima = [8.2, 9.1, 7.8, 10.3, 8.7, 9.5, 11.2, 8.0, 9.8, 10.1,
                 7.5, 8.9, 9.3, 10.8, 8.4, 9.0, 10.5, 7.9, 9.6, 11.0]

model = nextstat.GevModel(annual_maxima)
print(f"Parameters: {model.parameter_names()}")  # [mu, log_sigma, xi]

result = nextstat.fit(model)
params = result.parameters
print(f"Location μ = {params[0]:.2f}")
print(f"Scale σ = {np.exp(params[1]):.2f}")
print(f"Shape ξ = {params[2]:.3f}")

# 100-year return level (flood planning)
z100 = nextstat.GevModel.return_level(list(params), 100.0)
print(f"100-year return level: {z100:.2f} meters")
```

**Parameters:** `[mu, log_sigma, xi]` — location μ, log-scale log(σ), and shape ξ.
The log-transformation of σ ensures positivity during optimization.

## 4. GPD (Peaks Over Threshold)

The Generalized Pareto Distribution models exceedances over a high threshold.
Complementary to GEV — while GEV uses block maxima, GPD uses all observations
above a threshold. Useful for VaR/ES in finance, reinsurance pricing.

```python
# Daily losses exceeding $1M threshold (excess amounts)
exceedances = [0.2, 0.5, 1.1, 0.3, 0.8, 2.1, 0.4, 1.5, 0.6, 3.2,
               0.7, 1.0, 0.9, 1.8, 0.3, 2.5, 0.5, 1.3, 0.4, 4.1]

model = nextstat.GpdModel(exceedances)
result = nextstat.fit(model)
params = result.parameters
print(f"Scale σ = {np.exp(params[0]):.2f}")
print(f"Shape ξ = {params[1]:.3f}")

# 99th percentile of excess (VaR contribution)
q99 = nextstat.GpdModel.quantile(list(params), 0.99)
print(f"99% excess quantile: ${q99:.2f}M")
```

**Parameters:** `[log_sigma, xi]` — log-scale and shape. For ξ > 0 the distribution
is heavy-tailed (Pareto-like); for ξ = 0 it reduces to exponential; for ξ < 0 the
distribution has a finite upper endpoint.

## 5. Meta-analysis

Meta-analysis pools results from multiple independent studies. No model fitting needed —
these are direct closed-form computations.

### Fixed-effects (inverse-variance)

Assumes a single true effect size across all studies:

```python
# Five clinical trials reporting treatment effect (log OR) and SE
estimates = [0.25, 0.30, 0.15, 0.35, 0.20]
ses = [0.10, 0.12, 0.08, 0.15, 0.11]
labels = ["Trial A", "Trial B", "Trial C", "Trial D", "Trial E"]

result = nextstat.meta_fixed(estimates, ses, labels=labels)
print(f"Pooled estimate: {result['estimate']:.3f} ± {result['se']:.3f}")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"p-value: {result['p_value']:.4f}")

het = result['heterogeneity']
print(f"I² = {het['i_squared']:.1f}%")
print(f"Q = {het['q']:.2f}, df = {het['df']}, p = {het['p_value']:.3f}")
```

### Random-effects (DerSimonian–Laird)

Assumes effect sizes vary across studies (between-study heterogeneity τ²):

```python
result = nextstat.meta_random(estimates, ses, labels=labels)
print(f"Pooled estimate: {result['estimate']:.3f} ± {result['se']:.3f}")
print(f"τ² = {result['heterogeneity']['tau_squared']:.4f}")

# Forest plot data
for row in result['forest']:
    print(f"  {row['label']}: {row['estimate']:.3f} "
          f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
          f"weight={row['weight']:.1f}%")
```

## Vertical applications

| Model | Pharma | Insurance | Finance | Epidemiology |
|-------|--------|-----------|---------|--------------|
| Gamma GLM | Drug costs | Claim amounts | — | Healthcare costs |
| Tweedie GLM | — | Aggregate claims | — | Zero-inflated costs |
| GEV | — | Catastrophe modeling | — | Epidemic peaks |
| GPD | — | Excess-of-loss | VaR / ES | — |
| Meta-analysis | Trial pooling | — | — | Study synthesis |

## Implementation notes

- All GLM/EVT models implement `LogDensityModel`, providing `nll()`, `grad_nll()`,
  `parameter_names()`, `parameter_bounds()`, `parameter_init()`. This means they
  work with MLE, NUTS, profile likelihood, and all other NextStat inference machinery.
- Meta-analysis is **not** a `LogDensityModel` — it computes closed-form estimates
  directly. Use `nextstat.meta_fixed()` / `nextstat.meta_random()` as standalone functions.
- Tweedie NLL uses the saddle-point approximation (Dunn & Smyth 2005), which is the
  standard approach since the exact density involves an infinite series.
- GEV/GPD use log-transformed scale parameters internally to ensure σ > 0 during
  unconstrained optimization.

## Test coverage

| Module | File | Tests |
|--------|------|-------|
| Gamma + Tweedie | `crates/ns-inference/src/tweedie.rs` | 13 |
| GEV + GPD | `crates/ns-inference/src/evt.rs` | 16 |
| Meta-analysis | `crates/ns-inference/src/meta_analysis.rs` | 17 |

Total: 46 unit tests covering NLL correctness, gradient parity (finite-difference),
edge cases (single observation, zero response, Gumbel limit), reference values,
and input validation.
