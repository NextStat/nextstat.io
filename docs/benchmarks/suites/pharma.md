---
title: "Benchmark Suite: Pharma (PK / NLME)"
status: draft
last_updated: 2026-02-08
---

# Pharma Benchmark Suite (PK / NLME)

This suite benchmarks NextStat’s pharmacometrics baselines:

- 1-compartment oral PK (individual model)
- NLME baseline (population + subject-level random effects, MAP fit)

The goal is to publish **reproducible fit cost** for realistic problem sizes, with explicit protocols and datasets.

Related tutorials:

- PK baseline: `docs/tutorials/phase-13-pk.md`
- NLME baseline: `docs/tutorials/phase-13-nlme.md`

## What is measured

### 1) Likelihood + gradient time

Measure:

- NLL time / call
- gradient time / call (where applicable)

Report separately for:

- individual PK model
- NLME model (includes many random effects parameters)

### 2) Fit wall-time under a declared protocol

For optimization benchmarks, “time to convergence” depends on stopping criteria.

We therefore publish at least one of:

- fixed-iteration protocols (e.g., N steps of a declared optimizer)
- convergence protocols with explicit tolerances and max-iter caps

and always report:

- wall-time
- number of objective/gradient evaluations (when measurable)

### 3) Recovery / sanity tasks

Benchmarks should include correctness sanity checks such as:

- fit recovery on synthetic data (known parameters)
- predictive checks (model.predict at fitted params) for non-pathological outputs

## Datasets

Planned dataset tiers:

1. **Synthetic** datasets generated from the model (for recovery + scaling)
2. **Open** datasets (where licensing permits redistribution and deterministic preprocessing)

Every published run must include:

- dataset ID + hash
- generator parameters (for synthetic)
- preprocessing protocol (for real datasets)

## Scaling axes (what we vary)

- number of subjects (NLME)
- observations per subject
- random-effects dimension (diagonal vs future correlated Omega)
- LLOQ policies (when enabled)

## How to run locally (current)

Rust microbenchmarks live under `crates/ns-inference/benches/` and can be run via Criterion. If/when a dedicated PK/NLME bench is added, it will be listed here.

For a minimal “smoke” benchmark today, you can script repeated fits using the tutorial models from `docs/tutorials/phase-13-pk.md` and `docs/tutorials/phase-13-nlme.md` and publish your manifest (Python/Rust versions, CPU model, etc.).

