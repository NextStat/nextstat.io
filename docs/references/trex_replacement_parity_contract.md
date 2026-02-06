---
title: "TREx Replacement Parity Contract (v0)"
status: draft
created: 2026-02-06
---

# TREx Replacement Parity Contract (v0)

This document defines what “**identical numbers**” means for the TRExFitter replacement.

**Source of truth for numeric tolerances:** `docs/plans/standards.md` and `tests/python/_tolerances.py`.

## 1) Reference surfaces

We treat these as independent parity surfaces. A replacement is only acceptable if every enabled surface matches.

### 1.1 Model ingestion

Inputs:
- pyhf JSON workspace
- HistFactory XML (`combination.xml`) + ROOT histograms
- HIST (ROOT histograms)
- NTUP (ROOT TTrees) — later milestone

Contract:
- Stable channel ordering: lexicographic by channel name (pyhf convention).
- Stable modifier/parameter ordering: matches pyhf HistFactory set construction rules.
- Deterministic output in parity mode (`threads=1`).

### 1.2 Fit outputs

Contract (deterministic mode):
- `twice_nll` parity vs pyhf within `TWICE_NLL_RTOL/TWICE_NLL_ATOL`.
- `bestfit` and `uncertainties` parity within Phase-1 contract (see `docs/plans/standards.md`).
- Parameters must be compared **by name**, never by index position.

### 1.3 Expected data (main + aux)

Contract:
- `expected_data_pyhf_main` and `expected_data_pyhf` match pyhf ordering.
- Tolerance: `EXPECTED_DATA_ATOL`.

## 2) Reporting artifacts (numbers-first)

Renderers must treat artifacts as canonical numeric truth. Artifacts are stable, versioned JSON.

### 2.1 Distributions (prefit/postfit)

Per region (channel):
- `bin_edges[]` (variable width supported)
- `data_y[]`
- data y-errors: Garwood intervals (Poisson) per bin
  - Upper: `0.5 * χ²_{1-α/2, 2(n+1)} - n`
  - Lower: `n - 0.5 * χ²_{α/2, 2n}`
  - Use α = 0.31731 (i.e. 68.2689% central interval) to match the standard HEP ±1σ convention
  - Behavior for `n=0`: lower=0, upper computed from χ² quantile
- `prefit` expected yields per sample (stack order explicit)
- `postfit` expected yields per sample (stack order explicit)
- `total_prefit` / `total_postfit`
- `ratio` series:
  - policy must be explicit: `DATA/MC`, `DATA/BKG`, etc.
  - ratio y-errors derived from data (and optionally band), but **must be specified and tested**

Parity focus:
- Numbers in the artifact must match reference (TREx/ROOT-based or committed baseline) in deterministic mode.
- Images are not compared; only the artifact numbers.

### 2.2 Pulls / constraints

Contract:
- Pull for constrained-by-normal params: `(theta_hat - center)/width`.
- Constraint: explicit definition (e.g. postfit sigma / prefit sigma) must be encoded in artifact and documented.

### 2.3 Correlation matrix

Contract:
- Derived from covariance: `corr(i,j) = cov(i,j)/(σ_i σ_j)`, compare only when covariance exists.
- Symmetric, diag=1 within numeric tolerance.

### 2.4 Ranking/impacts

Contract:
- Ranking/impact definitions must be explicit (conditional fits vs correlation-based approximation).
- Artifact must store the method used and any failure drops with reasons.

## 3) Determinism policy

Parity mode must fix:
- thread count (default: `threads=1`)
- ordering of channels/samples/modifiers/parameters
- reduction order (no non-deterministic parallel floating reductions)
- RNG seeds (only where randomness exists; report must record seeds)

## 4) Baselines and golden tests

- CI compares NextStat artifacts to committed baselines (no TRExFitter required).
- Baseline refresh happens only in an external environment where ROOT/TREx can run and is manually triggered.

