---
title: "Phase 12: Econometrics + Causal Inference Pack — Overview"
status: draft
---

# Phase 12 overview

Phase 12 adds a dependency-light econometrics baseline:

- robust standard errors (HC / 1-way cluster)
- panel fixed effects (within estimator)
- DiD + event study (TWFE baseline)
- IV / 2SLS baseline + weak-IV diagnostics
- doubly-robust AIPW (ATE/ATT) + sensitivity hooks

## Tutorials

- [Robust SE (HC / cluster)](phase-12-robust-se.md)
- [Panel FE](phase-12-panel-fe.md)
- [DiD + event study](phase-12-did-event-study.md)
- [IV / 2SLS](phase-12-iv-2sls.md)
- [AIPW (ATE/ATT)](phase-12-aipw.md)
- [Volatility (GARCH(1,1) + SV baseline)](phase-12-volatility.md)

## Key limitations (baseline)

- 1-way clustering only (no 2-way clustering yet).
- TWFE DiD/event-study is a baseline; staggered adoption / heterogeneous effects require care.
- IV weak-instrument metrics are classic first-stage diagnostics and are intended for baseline review.
