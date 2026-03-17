# HEP Stable Surface

**Status**: 141/141 stable surfaces, 0 research
**Last validated**: 2026-03-17

## What is stable

Every public HEP surface in NextStat carries a `stable` maturity class.
This means:

- deterministic, reproducible results on the same input across releases
- published parity contracts with reference implementations
- CI-enforced release gates that block regression
- explicit support matrices documenting what is covered

## Stable paths at a glance

| Path | Surfaces | Reference | Parity Contract |
| --- | --- | --- | --- |
| **HistFactory** | 48 | pyhf | `twice_nll` atol=1e-8, `param_value` atol=2e-4 |
| **GVM** | 47 | literature combinations | solver-parity across 4 solvers |
| **Simplified Likelihood** | 2 | pyhf (reduced-model) | same as HistFactory core |
| **HEPData** | 2 | HEPData API | deterministic ingest parity |
| **Infrastructure** | 12 | — | internal contract |
| **Unbinned** | 11 | — | internal contract |
| **Import/Export** | 6 | — | internal contract |
| **Viz** | 11 | — | internal contract |
| **Preprocess** | 2 | — | internal contract |

## HistFactory

The largest stable path (48 surfaces across CLI, Python, tool, and server layers).

**What it covers**: fit, hypotest (asymptotic + toys), upper-limit, scan,
ranking, workspace audit, ROOT histogram I/O, XML import, Arrow round-trip,
bin-edge extraction, patchset application, Asimov/Poisson data generation,
CLs curves, reporting, validation.

**Parity reference**: pyhf (gold standard). ROOT is informational only.

**Tolerances**:

| Gate | Tolerance |
| --- | --- |
| `twice_nll` | atol=1e-8, rtol=1e-6 |
| `expected_data` | atol=1e-8 |
| `param_value` | atol=2e-4 |
| `param_uncertainty` | atol=5e-4 |
| `gradient` | atol=1e-6, rtol=1e-4 |

**Quickstart**: `nextstat run workspace.json`

**Support matrix**: [histfactory-support-matrix-2026-03-17.md](/docs/benchmarks/histfactory-support-matrix-2026-03-17.md)

## GVM (Global Vector Model)

Scalar HEP measurement combinations (47 surfaces across CLI and Python layers).

**What it covers**: foundational fit, toy calibration, calibration study,
scenario study, calibration campaign, solver-parity evidence, summarize,
brief, family report, family matrix, portfolio, portfolio stability.

**Solvers**: `auto`, `numerical-paper`, `analytic-perturbative`, `numerical`

**Evidence envelope**: literature-backed combinations up to 15x25, synthetic
stress tiers up to 128x96, multi-start robustness.

**Quickstart**: `nextstat combine-measurements spec.json`

**Support matrix**: [gvm-stable-first-support-matrix-2026-03-07.md](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)

## Simplified Likelihood

Reduced-model simplified likelihood (2 surfaces: tool layer).

**What it covers**: fit, hypotest, upper-limit, scan, discovery significance,
workspace simplification/export.

**Parity reference**: pyhf reduced-model with same fidelity tolerances as
HistFactory core.

**Quickstart**: `nextstat fit simplified.json`

**Support matrix**: [simplified-likelihood-support-matrix-2026-03-08.md](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)

## HEPData

HEPData repository import (2 surfaces: CLI layer).

**What it covers**: `nextstat import hepdata` with lock-file determinism,
schema validation, workspace generation from upstream YAML/JSON.

**Quickstart**: `nextstat import hepdata --out-dir workspaces/`

**Support matrix**: [hepdata-import-support-matrix-2026-03-08.md](/docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md)

## Release gates

Every promoted HEP slice is protected by a required-for-release CI gate:

| Gate | Make Target | CI Job |
| --- | --- | --- |
| GVM | `gvm-stable-first-gate` | `gvm-stable-first` |
| Simplified Likelihood | `simplified-likelihood-stable-surface-gate` | `simplified-likelihood-stable-surface` |
| SL Exporter | `simplified-likelihood-exporter-surface-gate` | `simplified-likelihood-exporter-surface` |
| M15 Reporting | `m15-reporting-stable-surface-gate` | `m15-reporting-stable-surface` |
| HEPData Import | `hepdata-import-stable-surface-gate` | `hepdata-import-stable-surface` |
| HistFactory | `histfactory-stable-surface-gate` | `histfactory-stable-surface` |

## Validation bundle

A single canonical artifact captures the full HEP quality state:

```bash
python -m scripts.hep_validation_bundle --check
# OK: 141/141 stable

python -m scripts.hep_validation_bundle
# Writes tmp/hep_validation_bundle.json + tmp/hep_validation_bundle.md
```

## Machine-readable inventory

The full surface-level inventory is in `hep_surface_matrix_v1.json` (141 entries).
Each entry has: `name`, `layer`, `maturity_class`, `owner_slice`, `support_matrix_ref`.

Regenerate and validate:

```bash
python scripts/hep_surface_matrix.py          # regenerate
python scripts/hep_surface_matrix.py --check  # validate
```
