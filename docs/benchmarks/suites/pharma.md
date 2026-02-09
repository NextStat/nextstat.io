---
title: "Benchmark Suite: Pharma (PK / NLME)"
description: "Pharmacometrics benchmark suite for NextStat: PK/NLME likelihood evaluation, fit wall-time, subject-count scaling, and analytic reference baselines with correctness-gated protocols for regulated-industry validation."
status: draft
last_updated: 2026-02-08
keywords:
  - NLME benchmark
  - pharmacometrics performance
  - PK model benchmark
  - population PK fitting
  - NONMEM alternative
  - Monolix comparison
  - pharmaceutical software validation
  - NextStat pharma
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
2. **Packaged / easy-to-obtain** datasets (e.g., bundled with R packages) with deterministic preprocessing
3. **Open** datasets (where licensing permits redistribution and deterministic preprocessing)

Every published run must include:

- dataset ID + hash
- generator parameters (for synthetic)
- preprocessing protocol (for real datasets)

## Baselines (planned)

We intend to publish comparisons against at least:

- **nlmixr2** (R) for NLME fitting workflows (where we can pin the full R environment reproducibly)
- **Torsten (Stan)** for selected PK/NLME workflows (runner or recorded reference outputs)

Baselines are only meaningful under a declared protocol (optimizer choice, stopping criteria, initialization policy). Where possible, we use fixed-iteration protocols to reduce ambiguity.

## Scaling axes (what we vary)

- number of subjects (NLME)
- observations per subject
- random-effects dimension (diagonal vs future correlated Omega)
- LLOQ policies (when enabled)

## How to run locally (current)

Rust microbenchmarks live under `crates/ns-inference/benches/` and can be run via Criterion. If/when a dedicated PK/NLME bench is added, it will be listed here.

For a minimal “smoke” benchmark today, you can script repeated fits using the tutorial models from [/docs/tutorials/phase-13-pk](/docs/tutorials/phase-13-pk) and [/docs/tutorials/phase-13-nlme](/docs/tutorials/phase-13-nlme) and publish your manifest (Python/Rust versions, CPU model, etc.).

Seed harness (public benchmarks repo bootstrap):

- `benchmarks/nextstat-public-benchmarks/suites/pharma/` (single-case `run.py` and multi-case `suite.py`)
- Baseline templates (R): `benchmarks/nextstat-public-benchmarks/suites/pharma/baselines/`

## Apex2 pharma reference suite (shipped)

The `apex2_pharma_reference_report.py` runner produces a JSON report covering:

- Analytic 1-compartment PK reference (closed-form vs ODE solver parity)
- NLME fit smoke tests on synthetic population data
- Per-case pass/fail with numeric deltas

This report is included in the Apex2 master report under the `pharma_reference` key and consumed by [/docs/references/validation-report](/docs/references/validation-report) for the unified validation pack.

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_pharma_reference_report.py \
  --out tmp/apex2_pharma_reference_report.json
```

## Related reading

- [Public Benchmarks Specification](/docs/benchmarks/public-benchmarks) — canonical spec.
- [PK Baseline Tutorial](/docs/tutorials/phase-13-pk) — 1-compartment PK model walkthrough.
- [NLME Baseline Tutorial](/docs/tutorials/phase-13-nlme) — population PK with random effects.
- [Validation Report Artifacts](/docs/references/validation-report) — validation pack for published snapshots.
