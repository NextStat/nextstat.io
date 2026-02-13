---
title: "Phase 12: Volatility (GARCH(1,1) + Stochastic Volatility) — Tutorial"
status: draft
---

# Volatility baseline (Phase 12)

NextStat includes a minimal, dependency-light volatility baseline:

- **GARCH(1,1)** (Gaussian) MLE fit
- **Stochastic volatility (SV)** via a pragmatic **log-chi²** observation approximation, fit by MLE (Kalman likelihood)

These helpers are meant for reproducible workflows and quick checks, not as a full econometrics suite.

## Python

```python
import nextstat

rs = [0.1, -0.2, 0.05, 0.0, 0.3, -0.1]

g = nextstat.timeseries.garch11_fit(rs)
print(g["params"]["mu"], g["params"]["omega"], g["params"]["alpha"], g["params"]["beta"], g["log_likelihood"])

sv = nextstat.timeseries.sv_logchi2_fit(rs)
print(sv["params"]["mu"], sv["params"]["phi"], sv["params"]["sigma"], sv["log_likelihood"])
```

## CLI

Input JSON:

```json
{ "returns": [0.1, -0.2, 0.05, 0.0, 0.3, -0.1] }
```

Commands:

```bash
nextstat timeseries garch11-fit --input returns.json
nextstat timeseries sv-logchi2-fit --input returns.json
```

## Limitations

- GARCH is Gaussian and includes only the (1,1) order.
- SV uses a log-chi² normal approximation (baseline only).
- For serious modeling, validate against a domain library and use diagnostics (residuals, QQ plots, stability checks).
