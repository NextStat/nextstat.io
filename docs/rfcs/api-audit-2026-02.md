---
title: "API Audit — February 2026"
status: draft
---

# NextStat API Audit — February 2026

Full audit of all 4 API surfaces: Python, Rust, CLI, WASM, Server.

## Executive Summary

**Rust core architecture: excellent.** Universal `LogDensityModel` trait unifies 50+ model types. All inference (MLE, NUTS, MAMS, profile, hypotest) is generic over this trait. GPU is cleanly feature-gated.

**Python API: needs cleanup.** Rapid vertical growth created function multiplication (4 variants of ranking, 5 variants of toys, 4 variants of hypotest). Device handling is split between function-name suffixes (`ranking_gpu`) and parameters (`device="cpu"`). Return types are often opaque `Dict[str, Any]`.

**Cross-layer: significant feature gaps.** Time series and survival only in CLI. Sampling/scans/hypotest absent from Server. WASM severely limited.

---

## 1. Rust Architecture (GOOD)

The Rust layer is well-designed. Key patterns:

### Universal Model Trait

```rust
pub trait LogDensityModel: Send + Sync {
    fn dim(&self) -> usize;
    fn parameter_names(&self) -> Vec<String>;
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;
    fn parameter_init(&self) -> Vec<f64>;
    fn nll(&self, params: &[f64]) -> Result<f64>;
    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>>;
    fn prepared(&self) -> Self::Prepared<'_>;
}
```

ALL models implement this: HistFactory, Unbinned, GLM, PK/PD, Survival, EVT, Ordinal, LMM. One trait → all inference methods work.

### Clean GPU Separation

GPU is compile-time optional (`#[cfg(feature = "cuda")]`). CPU inference has zero GPU dependencies.

### One Intentional Exception

`LapsModel` is an enum (not trait) because GPU-only models live entirely on device. This is correct.

### Consistency Scorecard (Rust)

| Aspect | Rating |
|--------|--------|
| Model trait unification | Excellent |
| Inference API consistency | Excellent |
| Result types | Excellent |
| Configuration patterns | Good |
| GPU abstraction | Good (intentional enum for LAPS) |

**Verdict: No major refactoring needed in Rust.**

---

## 2. Python API (NEEDS WORK)

### 2.1 Function Multiplication

The same operation has too many variants:

**Ranking (4 functions → should be 1):**

| Current | Should be |
|---------|-----------|
| `ranking(model)` | `ranking(model, device="cpu")` |
| `ranking_gpu(model)` | |
| `ranking_metal(model)` | |
| `unbinned_ranking(model)` | |

**Hypothesis Testing (4 functions → should be 2):**

| Current | Should be |
|---------|-----------|
| `hypotest(poi_test, model)` | `hypotest(poi_test, model)` |
| `hypotest_toys(poi_test, model, n_toys=...)` | `hypotest(poi_test, model, n_toys=1000)` |
| `unbinned_hypotest(mu_test, model)` | — (model type dispatch) |
| `unbinned_hypotest_toys(poi_test, model)` | — (model type dispatch) |

**Toy Fitting (5 functions → should be 1):**

| Current | Should be |
|---------|-----------|
| `fit_toys(model, params, n_toys=...)` | `fit_toys(model, params, n_toys=..., device="cpu")` |
| `fit_toys_batch(model, params)` | |
| `fit_toys_batch_gpu(model, params, device=...)` | |
| `unbinned_fit_toys(model, params)` | — (model type dispatch) |
| `poisson_toys(model, params)` | `generate_toys(model, params)` |

**Profile Scan (3 functions → should be 1):**

| Current | Should be |
|---------|-----------|
| `profile_scan(model, mu_values)` | `profile_scan(model, mu_values)` |
| `profile_curve(model, mu_values)` | `profile_scan(model, mu_values, return_curve=True)` |
| `unbinned_profile_scan(model, mu_values)` | — (model type dispatch) |

**Upper Limit (4 functions → should be 2):**

| Current | Should be |
|---------|-----------|
| `upper_limit(model)` | `upper_limit(model)` |
| `upper_limits(model, scan)` | `upper_limits(model, scan)` |
| `upper_limits_root(model)` | `upper_limit(model, method="root")` |
| `cls_curve(model, scan)` | separate (visualization) |

### 2.2 Device Handling Inconsistency

Three patterns coexist:

| Pattern | Example | Problem |
|---------|---------|---------|
| Function name suffix | `ranking_gpu()`, `ranking_metal()` | Can't add new backends |
| Parameter | `fit(..., device="cpu")` | Correct approach |
| Separate GPU function | `fit_toys_batch_gpu()` | Hybrid, confusing |

**Recommendation:** Single `device="cpu"|"cuda"|"metal"` parameter everywhere.

### 2.3 Parameter Order Inconsistency

| Function | Order | Convention |
|----------|-------|------------|
| `hypotest(poi_test, model)` | poi first | HEP convention |
| `fit(model)` | model first | Standard |
| `profile_scan(model, mu_values)` | model first | Standard |
| `panel_fe(entity_ids, x, y, p)` | metadata first, y third | **WRONG** |
| `did(y, treat, post)` | y first | Correct |

`panel_fe()` breaks its own module's convention (all other econometrics functions put `y` first).

### 2.4 Data Placement Inconsistency

| Pattern | Example |
|---------|---------|
| Data in constructor | `LinearRegressionModel(x, y)` |
| Data in function | `fit(model, data=[...])` |
| Data in both | `OneCompartmentOralPkModel(times, y, dose)` |

No clear rule when data goes in constructor vs. fit call.

### 2.5 Return Type Opacity

| Return Type | Quality | Example |
|------------|---------|---------|
| TypedDict (discoverable) | Good | `SamplerResult`, `FitResult` |
| `Dict[str, Any]` | Bad | `ranking()`, `profile_scan()`, `workspace_audit()` |
| `List[Dict[str, Any]]` | Bad | `ranking_gpu()`, `unbinned_ranking()` |
| `Any` | Bad | `hypotest_toys()`, `profile_ci()` |

~90% of dict-returning functions lack TypedDict definitions. IDE autocomplete fails.

### 2.6 Sampling API (Already Good)

Unified dispatcher exists and works:
```python
sample(model, method="nuts"|"mams"|"laps", ...)
```

But individual `sample_nuts()`, `sample_mams()`, `sample_laps()` are still exported at top level. Should be soft-deprecated in favor of unified `sample()`.

---

## 3. Cross-Layer Feature Matrix

| Operation | Python | CLI | WASM | Server |
|-----------|--------|-----|------|--------|
| MLE Fit | yes | yes | yes | yes |
| Profile Scan | yes | yes | yes | **no** |
| Hypothesis Test | yes | yes | yes | **no** |
| Toy-based CLs | yes | yes | no | yes |
| Upper Limit | yes | yes | yes | **no** |
| Ranking | yes | yes (viz) | no | yes |
| Significance | **no** | yes | no | no |
| GoF Test | **no** | yes | no | no |
| NUTS Sampling | yes | **no** | partial | **no** |
| MAMS Sampling | yes | **no** | partial | **no** |
| LAPS GPU | yes | **no** | no | **no** |
| GLM Regression | yes | **no** | yes | **no** |
| Time Series | **no** | yes | no | **no** |
| Survival | **no** | yes | no | **no** |
| Workspace Ops | yes | yes | no | **no** |
| TRExFitter Import | **no** | yes | no | no |

### Critical Gaps

1. **Python missing**: significance, GoF, time series, survival (all exist in CLI)
2. **CLI missing**: sampling (NUTS/MAMS/LAPS), GLM
3. **Server missing**: scans, hypotest, upper limits, sampling, workspace ops
4. **WASM**: intentionally limited (browser constraints)

---

## 4. Naming Conventions Across Layers

| Layer | Convention | Example |
|-------|-----------|---------|
| Python | snake_case | `profile_scan()`, `fit_toys()` |
| CLI | kebab-case | `unbinned-fit`, `hypotest-toys` |
| WASM | camelCase (JS) | `run_fit`, `run_profile_scan` |
| Server | REST paths | `POST /v1/fit`, `POST /v1/ranking` |
| Rust | snake_case | `sample_nuts()`, `from_workspace()` |

This is correct — each layer follows its ecosystem conventions.

---

## 5. Proposed Actions (Priority Order)

### P0 — Breaking but Critical (v0.10) — **ALL DONE**

| # | Action | Impact | Effort | Status |
|---|--------|--------|--------|--------|
| 1 | **Unify device parameter**: Remove `_gpu`, `_metal` suffixes. Single `device="cpu"\|"cuda"\|"metal"` param on `ranking()`, `fit_toys()`. Old functions removed (no users yet). | High — cleaner API | Medium | **DONE** |
| 2 | **Fix `panel_fe()` arg order**: `(y, x, entity_ids, p)` instead of `(entity_ids, x, y, p)` | Medium — consistency | Low | **DONE** |
| 3 | **Model-type dispatch for unbinned**: `hypotest()` and `profile_scan()` detect model type and dispatch internally. `unbinned_hypotest()`, `unbinned_profile_scan()`, etc. removed. | High — halves API surface | Medium | **DONE** |

### P1 — High Impact (v0.10-0.11) — **4 & 6 DONE**

| # | Action | Impact | Effort | Status |
|---|--------|--------|--------|--------|
| 4 | **TypedDict for all returns**: ~25 TypedDicts added to `_core.pyi` (`RankingEntry`, `ProfileScanResult`, `HypotestResult`, `WorkspaceAuditResult`, etc.) | High — IDE experience | Medium | **DONE** |
| 5 | **Expose time series + survival in Python**: Port CLI-only features to Python API | Medium — feature parity | Medium | TODO |
| 6 | **Soft-deprecate individual sampler functions**: `sample_nuts()` → `sample(method="nuts")`. Removed from `__all__`, kept as accessible aliases. | Low — already done mostly | Low | **DONE** |

### P2 — Medium (v0.11+)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 7 | **Server: add profile_scan, hypotest, upper_limit endpoints** | Medium — completeness | Medium |
| 8 | **Document data placement rules**: Constructor for static data, function for variable data | Low — clarity | Low |
| 9 | **CLI: add `sample` subcommand** (NUTS/MAMS) | Medium — completeness | Low |

### P3 — Polish (v0.12+)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 10 | **HistFactoryAnalysis wrapper class** (like UnbinnedAnalysis) for method chaining | Low — ergonomics | Medium |
| 11 | **Standardize abbreviations**: keep `nll` (established), document policy | Low | Low |
| 12 | **WASM: add ranking, GoF** (within size limits) | Low — niche | Medium |

---

## 6. What NOT To Change

1. **Rust `LogDensityModel` trait** — already excellent, don't touch
2. **`LapsModel` enum** — intentional GPU-only design, correct
3. **GPU feature-gating** — clean compile-time separation, correct
4. **Layer-specific naming** (Python snake_case, CLI kebab-case, WASM camelCase) — each follows its ecosystem
5. **Prepared caching flexibility** — models choose their own strategy, correct
6. **Sampling unified dispatcher** — `sample(method=...)` already exists, just needs promotion

---

## 7. Migration Strategy

Since there are almost no external users yet, all P0 changes were applied directly without a deprecation cycle. Old functions (`ranking_gpu`, `fit_toys_batch_gpu`, `unbinned_hypotest`, `unbinned_profile_scan`, `upper_limits_root`, etc.) were removed from `__init__.py` and `__all__`. The Rust `_impl` functions are still available internally but not exported.

Internal API conventions documented in `docs/internal/api-conventions.md` to prevent future inconsistencies.

---

## Appendix: Full Model Catalog (50+ types)

| Vertical | Count | Models |
|----------|-------|--------|
| HEP/HistFactory | 3 | HistFactoryModel, HybridModel, UnbinnedModel |
| Bayesian Built-in | 4 | GaussianMean, Funnel, StdNormal, EightSchools |
| Regression (GLM) | 8 | Linear, Logistic, Poisson, OrderedLogit, OrderedProbit, NegBinomial, Gamma, Tweedie |
| Multilevel | 2 | ComposedGlm, LmmMarginal |
| Survival | 8 | Exponential, Weibull, LogNormal AFT, Cox PH, + 4 Interval-censored |
| Pharmacokinetics | 4 | 1cpt Oral, 1cpt NLME, 2cpt IV, 2cpt Oral |
| Pharmacodynamics | 3 | Emax, SigmoidEmax, IndirectResponse |
| Extreme Value | 2 | GEV, GPD |
| Time Series | 3 | Kalman, GARCH(1,1), SV-LogChi2 |
| Econometrics | 4 | PanelFE, DiD, EventStudy, IV-2SLS |
| Churn/Survival | 2 | ChurnCoxPH, ChurnBootstrapHR |
| Meta-Analysis | 2 | FixedEffects, RandomEffects |
| Fault Tree | 1 | FaultTreeMC (CE-IS) |
| LAPS GPU-only | 6 | StdNormal, EightSchools, NealFunnel, NealFunnelNcp, GlmLogistic, Custom |

All implement `LogDensityModel` except LAPS GPU-only models (intentional).
