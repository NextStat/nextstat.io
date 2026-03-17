# HistFactory Core Stable-Surface Support Matrix

**Date**: 2026-03-17
**Status**: Executed support matrix
**Scope**: HistFactory / pyhf core stable subset — deterministic CPU parity

## Purpose

This document is the short operational matrix for the promoted HistFactory core
subset.

It answers one narrow question:

- what is `stable` now
- what remains `research-grade`
- what parity contract and evidence applies to each class

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | public compatibility promise for the named HistFactory core subset; deterministic CPU parity with pyhf on the same JSON input |
| `research-grade` | versioned and tested, but still evolving without stable-surface promise |

## Narrow stable subset

The stable boundary is intentionally narrow: the deterministic CPU parity path
for pyhf JSON workspaces. Every surface in this subset must produce
bit-reproducible results on the same input across releases.

## Stable CLI matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat run` | `stable` | full fit + summary from pyhf JSON workspace |
| `nextstat validate` | `stable` | workspace validation / schema check |
| `nextstat fit` | `stable` | ML fit with parameter extraction |
| `nextstat audit` | `stable` | workspace audit (channel/sample/modifier inventory) |
| `nextstat hypotest` | `stable` | asymptotic CLs hypothesis test |
| `nextstat upper-limit` | `stable` | upper-limit via profile likelihood ratio scan |
| `nextstat scan` | `stable` | profile-likelihood scan over POI range |
| `nextstat hypotest-toys` | `stable` | toy-based CLs hypothesis test |
| `nextstat mass-scan` | `stable` | mass-point scan (multi-workspace) |
| `nextstat combine` | `stable` | workspace combination |
| `nextstat significance` | `stable` | discovery significance (asymptotic) |
| `nextstat goodness-of-fit` | `stable` | saturated-model goodness-of-fit |
| `nextstat report` | `stable` | structured analysis report |
| `nextstat validation-report` | `stable` | validation report with pyhf cross-checks |
| `nextstat build-hists` | `stable` | histogram building from ROOT/Arrow inputs |

## Stable Python matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `HistFactoryModel` | `stable` | model class from pyhf JSON workspace |
| `from_pyhf` | `stable` | constructor from pyhf JSON dict |
| `from_histfactory_xml` | `stable` | constructor from HistFactory XML |
| `histfactory_bin_edges_by_channel` | `stable` | bin-edge extraction |
| `from_arrow` / `to_arrow` | `stable` | Arrow serialization round-trip |
| `apply_patchset` | `stable` | patchset application |
| `fit` / `fit_batch` / `fit_toys` | `stable` | ML fit variants |
| `hypotest` / `hypotest_toys` | `stable` | asymptotic and toy CLs |
| `profile_scan` | `stable` | profile-likelihood scan |
| `upper_limit` / `upper_limits` | `stable` | upper-limit computation |
| `cls_curve` | `stable` | CLs curve extraction |
| `asimov_data` / `poisson_toys` | `stable` | data generation |
| `ranking` | `stable` | nuisance-parameter ranking |
| `workspace_audit` | `stable` | workspace audit |
| `read_root_histogram` | `stable` | ROOT TH1 histogram I/O; deterministic golden-fixture parity |

## Stable tool matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat_fit` | `stable` | tool-layer fit |
| `nextstat_hypotest` | `stable` | tool-layer asymptotic CLs |
| `nextstat_hypotest_toys` | `stable` | tool-layer toy CLs |
| `nextstat_upper_limit` | `stable` | tool-layer upper-limit |
| `nextstat_scan` | `stable` | tool-layer scan |
| `nextstat_ranking` | `stable` | tool-layer ranking |
| `nextstat_workspace_audit` | `stable` | tool-layer audit |
| `nextstat_read_root_histogram` | `stable` | ROOT TH1 histogram tool; golden-fixture parity |

## Stable server matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `POST /v1/fit` | `stable` | server-safe fit path |
| `POST /v1/ranking` | `stable` | server-safe ranking |
| `POST /v1/batch/fit` | `stable` | server-safe batch fit |
| `POST /v1/batch/toys` | `stable` | server-safe batch toys |

## Parity contract

The HistFactory core stable subset is backed by deterministic CPU parity with
pyhf (the reference Python implementation) on the same JSON workspace input.

Fidelity tolerances (from `tests/python/_tolerances.py`):

| Gate | Tolerance | Notes |
| --- | --- | --- |
| `twice_nll` | atol=1e-8, rtol=1e-6 | NLL at same parameters |
| `expected_data` | atol=1e-8 | main + auxdata ordering |
| `param_value` | atol=2e-4 | best-fit parameter values |
| `param_uncertainty` | atol=5e-4 | Hessian-based uncertainties |
| `gradient` | atol=1e-6, rtol=1e-4 | AD vs finite-diff parity |
| `expected_data_per_bin` | atol=1e-12 | near-exact arithmetic parity |

## Boundaries that remain research-grade

- GPU-accelerated batch paths — parity is tested but GPU hardware variability
  precludes deterministic promise
- Metal f32 paths — Apple Silicon has no hardware f64

## Verification lane

The stable subset is enforced by a dedicated repo gate:

- script:
  [scripts/benchmarks/histfactory_stable_surface_gate.sh](/scripts/benchmarks/histfactory_stable_surface_gate.sh)
- make target:
  `make histfactory-stable-surface-gate`

The gate verifies:

- Rust core translation/integration suite for HistFactory workspaces
- CLI HistFactory smoke tests
- Python pyhf-parity regression tests
- Python bindings API smoke for HistFactory surfaces
- formatting
- presence of the required support matrix and acceptance documents

## Companion documents

- [HistFactory Stable-Surface Acceptance](/docs/benchmarks/histfactory-stable-surface-acceptance-2026-03-17.md)

## Bottom line

The stable product promise is intentionally narrow:

- the deterministic CPU parity path for pyhf JSON workspaces is `stable`
- ROOT TH1 I/O (`read_root_histogram`) is `stable` — backed by golden-fixture parity
- every stable surface must match pyhf within the published tolerances
- GPU paths and discovery-style outputs remain `research-grade`
- future stable expansion should update this matrix explicitly, not implicitly
