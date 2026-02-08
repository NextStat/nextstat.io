# TRExFitter Replacement Plan (Apex2 + TDD)

Goal: a full replacement for TRExFitter workflows, with publication-ready exports (PDF/SVG) and
numeric parity (same inputs -> same numbers) while providing better diagnostics and configuration.

This document is the engineering plan and progress tracker for Apex2:
Planning -> Exploration -> Execution -> Verification.

## Scope

1. Import
   - TREx-style `.config/.trf` (ReadFrom=NTUP + ReadFrom=HIST masking semantics).
   - HistFactory exports (`combination.xml` + referenced channel XML + ROOT hists).
   - pyhf JSON workspaces and PatchSets.
2. Model + inference
   - Exact/controlled interpolation semantics (pyhf defaults vs TREx/ROOT defaults vs smooth variants).
   - Deterministic evaluation (threads=1 + stable reductions) for parity work.
3. Reporting/export
   - TREx-like artifacts plus publication-ready PDF/SVG rendering.

## Parity Targets

1. pyhf validation fixtures (committed)
   - `tests/fixtures/pyhf_xmlimport`
   - `tests/fixtures/pyhf_multichannel`
   - `tests/fixtures/pyhf_coupled_histosys`
2. ROOT parity suite (requires ROOT runtime)
   - `tests/apex2_root_suite_report.py` + `tests/record_baseline.py`
3. Realistic TREx export dirs (to be provided later)
   - 1-3 realistic export directories containing `combination.xml` + ROOT hists.

## TDD Strategy

1. Add unit/integration tests for importer semantics on committed fixtures.
2. Add contract-level tests for stable JSON outputs (deterministic ordering, stable defaults).
3. Add parity harness tests (ROOT/pyhf/NextStat) gated behind environments that provide dependencies.

## Current Status (high level)

Completed (importer semantics + tests):
- HistFactory importer: channel `StatErrorConfig` parsing and Poisson vs Gaussian mapping.
  - Poisson => `shapesys` (Barlow-Beeston), Gaussian => `staterror`.
- ShapeSys histograms interpreted as relative uncertainties and converted to absolute (sigma_abs = rel * nominal).
- Lumi semantics: `NormalizeByTheory=True` samples receive `lumi` modifier; `LumiRelErr` and `ParamSetting Const`
  are surfaced via `measurements[].config.parameters` (auxdata/sigmas + fixed).
- NormFactor `Val/Low/High` surfaced via `measurements[].config.parameters` as init + bounds.
- CLI: `--parity` enforces deterministic execution (threads=1, stable reductions, Accelerate disabled). Interpolation
  defaults are selected by the ingest path/settings and are documented in `docs/pyhf-parity-contract.md`.
- HistFactory `ConstraintTerm` support in `combination.xml` (Gamma/LogNormal/Gaussian) with ROOT semantics.
- TREx ReadFrom=HIST masking semantics (config-as-filter without variable/binning requirements).

Remaining (next focus):
1. Collect 1-3 realistic TREx export dirs (each contains `combination.xml` + ROOT hists) and record baselines.
2. Full TREx systematic coverage roadmap (smoothing/symmetrize/prune policies, reporting parity).
3. Expand expression-compat corpus coverage based on real configs (only add constructs we observe).

## BMCP Tracking

Use BMCP epics/tasks as the source of truth for execution order and progress:
- Epic: HistFactory XML import parity (pyhf + ROOT/TREx)
- Epic: TREx `.config/.trf` import compatibility
- Epic: Expression compatibility (ROOT/TMath spellings + vector branches)
