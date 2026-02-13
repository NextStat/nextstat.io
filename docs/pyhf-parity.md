# pyhf Feature Parity

Last updated: 2025-07-18

This document tracks NextStat's feature parity with [pyhf](https://github.com/scikit-hep/pyhf),
the pure-Python HistFactory implementation.

## Summary

NextStat achieves **full functional parity** with pyhf for the core HistFactory
workflow: workspace ingestion, model building, expected data, NLL evaluation,
MLE fitting, profile likelihood scans, asymptotic CLs, and toy-based hypothesis
testing.

## Feature Matrix

| pyhf Feature | NextStat Equivalent | Status |
|---|---|---|
| **Workspace JSON** | `Workspace` (serde) | ✅ Full |
| **Workspace.prune** | `Workspace::prune()` | ✅ |
| **Workspace.rename** | `Workspace::rename()` | ✅ |
| **Workspace.sorted** | `Workspace::sorted()` | ✅ |
| **Workspace.digest** | `Workspace::digest()` (SHA-256) | ✅ |
| **Workspace.combine** | `Workspace::combine()` (None/Outer/LeftOuter/RightOuter) | ✅ |
| **PatchSet / Patch** | `PatchSet`, `Patch` | ✅ |
| **Model building** | `HistFactoryModel::from_workspace()` | ✅ |
| **All modifier types** | normfactor, normsys, histosys, shapesys, shapefactor, staterror, lumi | ✅ |
| **Interpolation code0** | Piecewise linear (HistoSys default) | ✅ |
| **Interpolation code1** | Exponential (NormSys default) | ✅ |
| **Interpolation code2** | Quadratic + linear extrapolation | ✅ |
| **Interpolation code4/4p** | 6th-order polynomial + extrap (HistoSys & NormSys) | ✅ |
| **expected_data** | `HistFactoryModel::expected_data()` | ✅ |
| **NLL (logpdf)** | `HistFactoryModel::nll()` with AD gradient | ✅ |
| **Constraint terms** | Normal, Poisson (Gamma), LogNormal | ✅ |
| **MLE fit** | `MaximumLikelihoodEstimator::fit()` | ✅ |
| **Profile likelihood** | `scan_histfactory()` with warm-start + multi-start | ✅ |
| **Test stat q̃_μ** | `TestStatistic::QMuTilde` (Eq. 14, arXiv:1007.1727) | ✅ |
| **Test stat q_μ** | `TestStatistic::QMu` (Eq. 12) | ✅ |
| **Test stat t_μ** | `TestStatistic::TMu` (Eq. 8) | ✅ |
| **Test stat t̃_μ** | `TestStatistic::TMuTilde` (Eq. 11) | ✅ |
| **Asymptotic CLs** | `hypotest_qtilde()` | ✅ |
| **Toy-based CLs** | `hypotest_qtilde_toys()` | ✅ |
| **Upper limits** | `upper_limit()` (bisection + interpolation) | ✅ |
| **simplemodels.uncorrelated_background** | `pyhf::simplemodels::uncorrelated_background()` | ✅ |
| **simplemodels.correlated_background** | `pyhf::simplemodels::correlated_background()` | ✅ |
| **json2xml** | `pyhf::xml_export::workspace_to_xml()` (XML structure) | ✅ Partial¹ |
| **Optimizer: scipy** | `OptimizerStrategy::Default` (L-BFGS-B) | ✅ |
| **Optimizer: minuit** | `OptimizerStrategy::MinuitLike` (smooth bounds) | ✅ Equivalent² |
| **HistFactory XML ingestion** | `histfactory::from_xml()` | ✅ |

¹ XML structural export is complete; ROOT histogram file writing requires `root-io` feature (future).
² Uses L-BFGS-B with Minuit-style internal variable transforms (logistic/exp). Numerically equivalent
  for all tested benchmarks. A pure-Rust MIGRAD (DFP variable metric) is a future enhancement.

## Beyond pyhf

NextStat provides additional capabilities not available in pyhf:

- **GPU acceleration** — CUDA (f64) and Metal (f32) fused NLL+gradient kernels
- **Automatic differentiation** — tape-based reverse-mode AD for exact gradients
- **Warm-start profile scans** — reuse parameters across scan points
- **Multi-start robustness** — deterministic jittered restarts for non-convex NLL
- **HistFactory XML+ROOT ingestion** — read CERN HistFactory XML directly
- **TRExFitter config ingestion** — read TREx/TRExFitter configuration files
- **Parquet/Arrow pipeline** — event-level unbinned fits from Parquet
- **Batch GPU toys** — thousands of toy fits in parallel on GPU
- **Python bindings** — PyO3 with NumPy interop
- **R bindings** — extendr (experimental)
- **WASM** — browser-based inference via WebAssembly

## Crate Locations

| Module | Crate | Path |
|---|---|---|
| Workspace schema | `ns-translate` | `src/pyhf/schema.rs` |
| Model builder | `ns-translate` | `src/pyhf/model.rs` |
| Simple models | `ns-translate` | `src/pyhf/simplemodels.rs` |
| XML export | `ns-translate` | `src/pyhf/xml_export.rs` |
| MLE / fitting | `ns-inference` | `src/mle.rs` |
| Profile likelihood | `ns-inference` | `src/profile_likelihood.rs` |
| Test statistics | `ns-inference` | `src/profile_likelihood.rs` (`TestStatistic` enum) |
| Optimizer | `ns-inference` | `src/optimizer.rs` |
| SIMD interpolation | `ns-compute` | `src/simd.rs` |
