# TRExFitter ↔ NextStat Feature Parity Matrix

Status as of 2026-02-10. Based on TRExFitter (ATLAS-internal, RooStats/HistFactory) actions and NextStat CLI.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Full parity or better |
| ⚡ | Parity + GPU acceleration (NextStat advantage) |
| 🔶 | Partial — core exists but gaps remain |
| ❌ | Missing — needs implementation |
| 🟢 | NextStat-only feature (no TRExFitter equivalent) |

---

## 1. Workflow / Actions (TRExFitter single-letter actions)

TRExFitter uses single-letter action codes: `h` (build histograms), `n` (create workspace), `w` (write workspace), `f` (fit), `d` (draw prefit/postfit), `p` (draw pulls/NP), `l` (compute limit), `r` (rank NPs), `s` (significance), `b` (breakdown of mu), `m` (multi-fit/combination), `i` (importance).

| TRExFitter Action | NextStat Equivalent | Status | Notes |
|-------------------|---------------------|--------|-------|
| **h** — Build histograms from ntuples | `nextstat build-hists --config` | ✅ | Native ROOT TTree reader, TREx config parser |
| **n** — Create workspace (RooWorkspace) | `nextstat import histfactory --xml` | ✅ | Creates pyhf JSON workspace |
| **w** — Write workspace to file | Built-in (JSON output) | ✅ | pyhf JSON is the native format |
| **f** — Perform MLE fit | `nextstat fit -i workspace.json` | ⚡ | GPU (CUDA/Metal), `--fit-regions`, `--validation-regions` |
| **d** — Draw pre/postfit distributions | `nextstat viz distributions` + `nextstat report --render` | ✅ | JSON artifacts + Python PDF/SVG renderer |
| **p** — Draw NP pulls | `nextstat viz pulls --fit fit.json` | ✅ | JSON artifact + renderer |
| **l** — Compute CLs upper limit | `nextstat upper-limit --expected` | ✅ | Asymptotic q̃μ, scan + bisection modes |
| **r** — NP ranking (impact on POI) | `nextstat viz ranking` | ✅ | ±1σ pre/postfit impact |
| **s** — Significance (discovery p-value) | `nextstat significance` | ✅ | Dedicated command: p₀, Z_obs, Z_exp, q₀ |
| **b** — Breakdown of μ uncertainty | **via ranking + grouped uncertainties** | 🔶 | `--uncertainty-grouping` in report; no standalone `breakdown` command |
| **m** — Multi-fit / combination | `nextstat combine` + `fit` | ✅ | JSON-level workspace merge, then standard fit pipeline |
| **i** — Importance (grouped NP impact) | `nextstat report --uncertainty-grouping` | 🔶 | Grouping exists but not as standalone plot action |

---

## 2. Input Formats

| Feature | TRExFitter | NextStat | Status |
|---------|------------|----------|--------|
| ROOT ntuples (TTree) | ✅ ReadFrom=NTUP | ✅ `build-hists`, native `ns-root` | ✅ |
| ROOT histograms | ✅ ReadFrom=HIST | ✅ `import histfactory --xml` | ✅ |
| HistFactory XML | ✅ (via RooStats) | ✅ `import histfactory` | ✅ |
| pyhf JSON | ❌ | ✅ native format | 🟢 |
| HS3 JSON | ❌ | ✅ auto-detected | 🟢 |
| TRExFitter .config | ✅ native | ✅ `import trex-config`, `trex import-config` | ✅ |
| Parquet (event-level) | ❌ | ✅ `nextstat convert`, mmap reader | 🟢 |

---

## 3. Systematics Handling

| Feature | TRExFitter | NextStat | Status |
|---------|------------|----------|--------|
| HistoSys (shape) | ✅ | ✅ code0 + code4p | ✅ |
| NormSys (normalization) | ✅ | ✅ code1 + code4 | ✅ |
| OverallSys | ✅ | ✅ | ✅ |
| ShapeSys (stat per-bin) | ✅ | ✅ | ✅ |
| StatConfig (Barlow-Beeston) | ✅ staterror | ✅ staterror | ✅ |
| NormFactor (free float) | ✅ | ✅ normfactor | ✅ |
| ShapeFactor | ✅ | ✅ shapefactor | ✅ |
| Lumi uncertainty | ✅ lumi modifier | ✅ lumi modifier | ✅ |
| **Smoothing** (syst templates) | ✅ Smoothing=40/TRExDefault | 🔶 Python preprocessing | 🔶 |
| **Pruning** (remove small systs) | ✅ Pruning options | 🔶 Python preprocessing | 🔶 |
| **Symmetrization** | ✅ Symmetrisation options | 🔶 Python preprocessing | 🔶 |
| Interpolation code selection | ✅ per-systematic | ✅ `--interp-defaults` + per-modifier | ✅ |

---

## 4. Fit Features

| Feature | TRExFitter | NextStat | Status |
|---------|------------|----------|--------|
| MLE fit (MINUIT) | ✅ MINUIT2 | ✅ L-BFGS-B | ✅ |
| Hessian uncertainties | ✅ | ✅ | ✅ |
| Covariance matrix | ✅ | ✅ | ✅ |
| Fit to Asimov data | ✅ FitBlind | 🔶 Need explicit Asimov dataset generation | 🔶 |
| Conditional fit (fix params) | ✅ | ✅ `with_fixed_param()` | ✅ |
| Fit regions / VR exclusion | ✅ FitRegion | ✅ `--fit-regions`, `--validation-regions` | ✅ |
| GPU acceleration | ❌ | ✅ CUDA + Metal | 🟢 |
| Parity mode (vs pyhf) | ❌ | ✅ `--parity` Kahan summation | 🟢 |
| Unbinned MLE | ❌ | ✅ `nextstat unbinned-fit` | 🟢 |
| Hybrid binned+unbinned | ❌ | ✅ `nextstat hybrid-fit` | 🟢 |

---

## 5. Statistical Tests

| Feature | TRExFitter | NextStat | Status |
|---------|------------|----------|--------|
| Asymptotic CLs (q̃μ) | ✅ | ✅ `hypotest` | ✅ |
| Observed upper limit | ✅ Limit action | ✅ `upper-limit` | ✅ |
| Expected limits (Brazil band) | ✅ | ✅ `--expected` | ✅ |
| CLs scan mode | ✅ | ✅ `--scan-start/stop/points` | ✅ |
| **Mass scan (Type B Brazil)** | ✅ multi-signal Limit | ✅ `mass-scan` | ✅ |
| Toy-based CLs | ✅ (via RooStats) | ✅ `hypotest-toys` + GPU | ⚡ |
| Profile likelihood scan | ✅ | ✅ `scan` + GPU | ⚡ |
| Discovery significance (Z) | ✅ `GetSignificance` | ✅ `significance` | ✅ |
| Toy-based significance | ✅ | ✅ `hypotest-toys --mu 0` | ✅ |
| Goodness-of-fit (saturated) | ✅ | ✅ `goodness-of-fit` | ✅ |
| **Multi-POI** | 🔶 (limited) | ❌ | ❌ |

---

## 6. Output Plots / Artifacts

| Plot / Artifact | TRExFitter | NextStat | Status |
|-----------------|------------|----------|--------|
| Prefit distributions | ✅ | ✅ `viz distributions` | ✅ |
| Postfit distributions | ✅ | ✅ `viz distributions --fit` | ✅ |
| Data/MC ratio panel | ✅ | ✅ included in distributions | ✅ |
| NP pull plot | ✅ | ✅ `viz pulls` | ✅ |
| Correlation matrix | ✅ | ✅ `viz corr` | ✅ |
| NP ranking plot | ✅ | ✅ `viz ranking` | ✅ |
| Normalization factors | ✅ | ✅ in fit output JSON | ✅ |
| Yield tables (pre/postfit) | ✅ | ✅ in report artifacts | ✅ |
| Brazil band plot (CLs vs μ) | ✅ | ✅ `viz cls` | ✅ |
| Profile likelihood plot | ✅ | ✅ `viz profile` | ✅ |
| **Gammas plot** (stat NPs) | ✅ dedicated | 🔶 included in pulls, no dedicated gamma plot | 🔶 |
| **Summary plot** (μ for multiple fits) | ✅ multi-fit | ❌ | ❌ |
| **Pie chart** (composition) | ✅ | ❌ | ❌ |
| Separation plot (S vs B) | ✅ | ❌ | ❌ |
| PDF/SVG rendering | ✅ (ROOT TCanvas) | ✅ Python matplotlib + SVG | ✅ |
| Unified report bundle | ❌ (separate files) | ✅ `nextstat report` → single dir | 🟢 |

---

## 7. Advanced / Workflow Features

| Feature | TRExFitter | NextStat | Status |
|---------|------------|----------|--------|
| Config-driven workflow | ✅ .config file | ✅ `nextstat run --config spec.yaml` | ✅ |
| Config validation | ✅ (runtime) | ✅ `nextstat validate --config` | ✅ |
| JSON schema for configs | ❌ | ✅ `nextstat config schema` | 🟢 |
| **Blinding** (SR data masking) | ✅ FitBlind, BlindSR | ✅ `--blind-regions` in report | ✅ |
| Parallelism | ✅ (ROOT threads) | ✅ Rayon + `--threads` | ✅ |
| Batch/grid submission | ✅ (condor integration) | ❌ | ❌ |
| **Multi-fit / combination** | ✅ MultiFit block | ❌ | ❌ |
| Workspace patching (patchsets) | ❌ | ✅ pyhf patchset support | 🟢 |
| HistFactory XML export | ❌ (internal RooWorkspace) | ✅ `export histfactory` | 🟢 |
| Ntuple→workspace pipeline | ✅ n+w actions | ✅ `build-hists` | ✅ |
| Expression engine (TTree) | ✅ (ROOT TFormula) | ✅ native expression parser | ✅ |
| Validation report | ❌ | ✅ `nextstat validation-report` | 🟢 |
| WASM playground | ❌ | ✅ browser-based | 🟢 |
| CI metrics (`--json-metrics`) | ❌ | ✅ `nextstat_metrics_v0` schema | 🟢 |

---

## 8. Gap Analysis — Priority

### P0 (Critical for parity — blocks adoption)

| Gap | Description | Effort |
|-----|-------------|--------|
| ~~Multi-fit / Combination~~ | ✅ Done — `nextstat combine` (JSON-level merge) + existing fit pipeline | — |
| ~~Goodness-of-fit test~~ | ✅ Done — `nextstat goodness-of-fit` | — |
| ~~Discovery significance (Z)~~ | ✅ Done — `nextstat significance` | — |

### P1 (Important — expected by users)

| Gap | Description | Effort |
|-----|-------------|--------|
| ~~Asimov fit (FitBlind)~~ | ✅ Done — `nextstat fit --asimov` | — |
| ~~Gammas plot~~ | ✅ Done — `nextstat viz gammas` | — |
| ~~Summary plot~~ | ✅ Done — `nextstat viz summary` | — |
| ~~Separation plot~~ | ✅ Done — `nextstat viz separation` | — |

### P2 (Nice-to-have)

| Gap | Description | Effort |
|-----|-------------|--------|
| ~~Pie chart~~ | ✅ Done — `nextstat viz pie` | — |
| **Batch submission** | HTCondor / SLURM job submission for mass scans / toys. | Script wrapper, not core |
| ~~Smoothing CLI~~ | ✅ Done — `nextstat preprocess smooth` (native Rust 353QH,twice) | — |
| ~~Pruning CLI~~ | ✅ Done — `nextstat preprocess prune` (native Rust) | — |

---

## 9. NextStat Advantages (No TRExFitter Equivalent)

| Feature | Impact |
|---------|--------|
| **GPU acceleration** (CUDA + Metal) | 10-100× faster fits, toys, scans |
| **Unbinned likelihood** | Event-level PDFs (flow, DCR surrogate) |
| **Hybrid binned+unbinned** | Combined likelihood with shared parameters |
| **WASM playground** | Browser-based analysis, zero install |
| **Native Rust** | No ROOT dependency, single binary, deterministic |
| **pyhf / HS3 native** | Direct JSON workspace support |
| **Parquet I/O** | Modern columnar format, mmap, predicate pushdown |
| **Neural PDFs** | ONNX flow / DCR surrogate with TensorRT |
| **CI metrics schema** | `--json-metrics` for experiment tracking |
| **Validation pack** | Automated Apex2 validation reports |
