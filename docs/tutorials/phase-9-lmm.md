# Phase 9: Linear Mixed Models (LMM) baseline

This tutorial documents the current baseline support for Gaussian linear mixed
models (repeated measures / longitudinal data).

## What is implemented

`nextstat.LmmMarginalModel` implements an ML-style marginal likelihood for LMMs
by integrating out Normal random effects. For Gaussian outcomes this is exact
(often referred to as a "Laplace baseline" in mixed-model tooling).

Current scope:

- Gaussian outcomes only
- Random intercept, or random intercept + one random slope
- Independent (diagonal) random-effects covariance (no correlation yet)
- Maximum-likelihood fit (no priors, no REML)

## Quick start (Python)

```python
import numpy as np
import nextstat as ns

rng = np.random.default_rng(0)
n_groups = 8
m_per = 6
n = n_groups * m_per

group_idx = np.repeat(np.arange(n_groups), m_per)
x1 = rng.normal(size=n)

# Synthetic data with a random intercept per group
beta0, beta1 = 1.0, 0.5
tau_alpha, sigma_y = 0.8, 0.3
alpha = rng.normal(scale=tau_alpha, size=n_groups)
eps = rng.normal(scale=sigma_y, size=n)
y = beta0 + beta1 * x1 + alpha[group_idx] + eps

model = ns.LmmMarginalModel(
    x1.reshape(-1, 1).tolist(),
    y.tolist(),
    include_intercept=True,
    group_idx=group_idx.tolist(),
    n_groups=n_groups,
)

mle = ns.MaximumLikelihoodEstimator()
fit = mle.fit(model)
print(fit.converged, fit.nll)
print(model.parameter_names())
print(fit.parameters)
```

## Notes and limitations

- Parameter names use log-scale for standard deviations:
  `log_sigma_y`, `log_tau_alpha`, and (if enabled) `log_tau_u_betaK`.
- For fully Bayesian hierarchical modeling, prefer `nextstat.hier.*` builders
  and `nextstat.bayes.sample(...)` (NUTS) for posterior inference.

