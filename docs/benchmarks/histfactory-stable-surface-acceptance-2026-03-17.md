# HistFactory Stable-Surface Acceptance

**Date**: 2026-03-17
**Status**: Release-hardening acceptance policy
**Scope**: HistFactory / pyhf core stable subset — deterministic CPU parity

## Purpose

This document defines the acceptance criteria for treating the current
HistFactory core surface as a stable product surface rather than an
unclassified prototype.

It answers one narrow question:

- what must be true before we say the current HistFactory core surface is
  acceptable for stable deterministic CPU parity workflows?

## Support classes

This policy uses two support classes:

| Class | Meaning |
| --- | --- |
| `stable` | versioned contract with explicit acceptance gates and pyhf parity evidence |
| `research-grade` | versioned and tested, but not yet covered by the stable-surface acceptance promise |

## Stable subset in scope

The current stable subset is intentionally narrow:

- pyhf JSON workspace input (the only accepted input format for the stable
  parity claim)
- workspace_audit, fit, hypotest (asymptotic), upper-limit, and scan as the
  core consume path
- CLI, Python, tool, and server dispatch for HistFactory inputs
- deterministic CPU evaluation mode

Out of scope for the current stable acceptance promise:

- GPU-accelerated batch paths
- Metal f32 compute
- ROOT histogram I/O
- discovery-style output surfaces
- non-pyhf input formats beyond the existing HistFactory XML importer

## Acceptance criteria

All items below must be true for acceptance.

### 1. Contract

- pyhf JSON workspace format is the accepted input contract
- workspace_audit output contract is versioned
- CLI/Python/tool/server behavior is explicit for accepted surfaces
- the stable subset is documented in a dedicated support matrix

### 2. Verification matrix

- Rust core translation/integration coverage for HistFactory is green
- CLI smoke coverage for HistFactory workspaces is green
- Python pyhf-parity regression tests are green
- Python bindings API smoke for HistFactory surfaces is green
- no unresolved contract drift between docs and emitted artifacts

### 3. Fidelity gates

These are hard acceptance gates for pyhf parity:

- `|twice_nll_ns - twice_nll_pyhf|` <= 1e-8 (atol) or <= 1e-6 (rtol)
- `|param_value_ns - param_value_pyhf|` <= 2e-4
- `|param_uncertainty_ns - param_uncertainty_pyhf|` <= 5e-4
- `max_abs(expected_data_ns - expected_data_pyhf)` <= 1e-8
- `max_abs(gradient_ns - gradient_pyhf)` <= 1e-6 (atol) or <= 1e-4 (rtol)
- per-bin expected data parity: atol <= 1e-12

### 4. Evidence source

- pyhf is the gold-standard reference for all fidelity gates
- ROOT is informational only (not CI-gating) per ROOT/HistFactory 3-way
  comparison policy

## Operational gate surface

The stable subset is enforced through these operational entry points:

- local/CI gate:
  `make histfactory-stable-surface-gate`
- gate script:
  `scripts/benchmarks/histfactory_stable_surface_gate.sh`

## Acceptance decision rule

The surface is acceptable as `stable` only if:

1. all contract artifacts exist and are documented
2. all verification layers are green
3. all fidelity gates pass against pyhf

If any one of these fails, the affected surface remains `research-grade`.

## Current decision for March 17, 2026

Based on the current repository evidence, the acceptance criteria above are
met for the current HistFactory core surface:

- pyhf parity tests pass within published tolerances
- Rust, CLI, Python, and tool layers are green
- the stable subset is documented in a dedicated support matrix

Supporting evidence:

- [HistFactory Support Matrix](/docs/benchmarks/histfactory-support-matrix-2026-03-17.md)
- [pyhf Parity Contract](/docs/pyhf-parity-contract.md)
- [Shared Tolerances](/tests/python/_tolerances.py)

## What acceptance does not imply

Acceptance does not mean:

- arbitrary performance claims without benchmark evidence
- automatic acceptance of GPU or Metal paths
- stable support for surfaces not listed in the scoped stable subset
- parity guarantee with ROOT (informational only)
