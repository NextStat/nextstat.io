# TRExFitter vs NextStat - Technical Analysis (Draft)

Status: draft (high-level notes; validate with real users).  
Goal: identify the practical requirements and UX expectations which must be met to realistically replace TRExFitter/pyhf workflows.

## 1) What TRExFitter is (in HEP workflows)

TRExFitter is commonly used as a workhorse for binned likelihood fits:

- managing many channels/samples/systematics,
- building and aggregating statistical models,
- profiles, ranking, pulls, correlation matrices,
- reporting and plotting in ATLAS/CMS-style formats.

Key observation: TRExFitter's value is not just "a fit", but the full workflow around it.

## 2) Where NextStat should win (positioning)

### 2.1 Engine correctness and portability

- The math contract is fixed and validated against pyhf (`twice_nll` parity in deterministic mode).
- Rust core + Python API improves portability and reduces reliance on ROOT by default.

### 2.2 Performance where it matters

TRExFitter workflows often bottleneck on:

- large numbers of nuisance parameters,
- ranking (many repeated fits),
- scans/limits.

NextStat should win via:

- autodiff (gradients/HVP),
- CPU parallelism + batching,
- reproducible job-array execution on clusters.

## 3) Must-have parity (practical replaceability)

To replace TRExFitter in practice, a minimum set includes:

1. Workspace ingestion
   - pyhf JSON (P0)
   - HistFactory XML import (P1) or conversion via pyhf

2. Fit and diagnostics
   - MLE fit + uncertainties
   - pulls + constraints summary
   - correlation matrix

3. Workflow primitives
   - Asimov dataset
   - profile likelihood scan
   - ranking/impact (nuisance impact on POI)

4. Systematics preprocessing (often underestimated)
   - smoothing (shape systematics)
   - pruning (low-impact nuisance parameters)
   - symmetrisation

## 4) Lessons learned

### 4.1 Copy (match user expectations)

- "One button" presets for typical analyses
- standardized reports: pulls/ranking/correlation matrices in familiar formats
- clear logs showing what was fixed/profiled and why

### 4.2 Avoid (reduce complexity and fragmentation)

- ROOT-first architecture as a hard dependency of the core
- monolithic coupling of preprocessing, inference, and plotting
- non-reproducible results due to non-deterministic parallel reductions

## 5) Compatibility strategy (practical migration)

Realistic migration path:

1. pyhf parity on a constrained model set (fixtures)
2. TREx-like outputs: pulls/ranking/corr matrices in familiar formats
3. Import pipeline: HistFactory XML -> NextStat model (directly or via pyhf)
4. Preprocessing: provide equivalent knobs, but as a modular pipeline

## 6) What to validate with users (interview checklist)

1. Which reports/plots do they actually use daily?
2. Which preprocessing steps are required (and in what order)?
3. What model sizes (bins/channels/NPs) are "normal"?
4. Where is time spent: ranking, scans, limits, toys?
5. What are the reproducibility and cluster execution requirements?

