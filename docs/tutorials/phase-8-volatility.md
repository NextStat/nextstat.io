---
title: "Phase 8: Volatility (GARCH / SV) — Tutorial"
status: draft
---

# Volatility (baseline)

This tutorial documents NextStat’s **dependency-light volatility baselines**:

- **GARCH(1,1)** with Gaussian innovations (MLE)
- **Approximate stochastic volatility (SV)** using `log(y_t^2)` with a Gaussian approximation for `log(chi^2_1)`,
  fit via Kalman MLE on an AR(1) latent log-variance

These tools are intended for **baseline workflows and reproducible reporting**, not as a full replacement for
specialized econometrics packages.

## Inputs

Both commands use a minimal JSON input:

```json
{
  "returns": [0.01, -0.02, 0.005, 0.03]
}
```

## CLI

```bash
# GARCH(1,1)
nextstat timeseries garch11-fit --input returns.json

# Approximate SV (log-chi2 + Kalman)
nextstat timeseries sv-logchi2-fit --input returns.json
```

Outputs are pretty-printed JSON containing fitted parameters, log-likelihood, and per-timestep volatility series.

## Python

```python
import nextstat

returns = [0.01, -0.02, 0.005, 0.03]

g = nextstat.timeseries.garch11_fit(returns)
sv = nextstat.timeseries.sv_logchi2_fit(returns)
```

## Key limitations

- GARCH is currently **Gaussian only** (no Student-t innovations yet).
- SV uses the classic **log-chi2 Gaussian approximation**; it is not a full Bayesian SV sampler.
- No multivariate volatility yet.

