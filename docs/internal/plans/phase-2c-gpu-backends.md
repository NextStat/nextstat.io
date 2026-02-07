# Фаза II-C: CUDA backend (clusters) + Apple CPU Accelerate (P1, optional)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** ускорить f64-heavy HEP workloads:
- macOS/Apple Silicon: **CPU (Rayon + SIMD) + Apple Accelerate (vDSP/vForce)**.
- Linux/NVIDIA clusters: optional **CUDA** backend for batched workloads.

**Duration:** Foundation (required): Недели 13-14. GPU (optional): Недели 21-28.

**Architecture:** Trait-based backend abstraction, compile-time feature flags.

**Tech Stack:** Rust, Apple Accelerate (vDSP/vForce), cudarc (optional, CUDA only).

---

## Current status (2026-02-07)

### Accelerate backend (DONE)
- Apple Accelerate (vDSP/vForce) backend fully implemented and tested.
- Feature `accelerate` in ns-compute enables vectorized `ln()` via `vvlog()` and subtraction via `vDSP_vsubD`.
- ~600 fits/sec for typical HEP models (250 params, 1000 bins) on M3 Max.
- `NEXTSTAT_DISABLE_ACCELERATE=1` disables for determinism testing.

### CUDA batch backend (DONE — core implementation)
- **Fused NLL+Gradient CUDA kernel** with all 7 modifier types and analytical gradient.
- **cudarc 0.19** with dynamic loading — binary works without CUDA installed.
- **Lockstep batch optimizer**: standalone `LbfgsState` L-BFGS-B, convergence masking.
- **Model serialization**: `HistFactoryModel::serialize_for_gpu() -> GpuModelData`.
- **CLI**: `--gpu` flag on `hypotest-toys`.
- **Python**: `has_cuda()`, `fit_toys_batch_gpu(model, params, device="cuda")`.
- **Feature chain**: `ns-compute/cuda` → `ns-translate/cuda` → `ns-inference/cuda` → `ns-cli/cuda`, `ns-py/cuda`.
- **PTX**: compiled via `nvcc --ptx -arch=sm_70 -O3`, embedded in binary via `include_str!`.

**Files created:**
- `crates/ns-compute/src/cuda_types.rs` — `#[repr(C)]` structs for GPU data layout
- `crates/ns-compute/src/cuda_batch.rs` — `CudaBatchAccelerator` GPU orchestrator
- `crates/ns-compute/kernels/batch_nll_grad.cu` — Fused NLL+gradient CUDA kernel (~370 lines)
- `crates/ns-inference/src/gpu_batch.rs` — Lockstep batch optimizer with `LbfgsState`

### CUDA single-model fit path (DONE)
- **GpuSession** — shared GPU state for multiple fits (profile, ranking, scan).
- **GpuObjective** — fused NLL+grad caching (1 GPU launch per iteration, not 2).
- CLI: `--gpu cuda` on `fit` and `scan` commands.
- Python: `fit(model, device="cuda")`, `profile_scan(model, mu_values, device="cuda")`.

### Differentiable PyTorch layer (DONE)
- **DifferentiableSession** — CUDA zero-copy NLL + signal gradient.
- **ProfiledDifferentiableSession** — profiled q₀/qμ with envelope theorem gradients.
- GPU L-BFGS-B fits, convergence fix (tol=1e-5 + NLL stability).
- Python: `nextstat.torch` module with autograd Functions.

### GPU Benchmarks (DONE — 2026-02-07)
- Full benchmark suite on RTX 4000 SFF Ada (CUDA 12.0).
- Results: GPU wins on batch/differentiable workloads, CPU wins on single-model fits.
- Details: `docs/benchmarks.md` (GPU Benchmarks section), `docs/gpu-contract.md`.

### CUDA batch toys fix (DONE)
- cudarc 0.19 `memcpy_dtoh` requires `dst.len() >= src.len()`. Host buffers must be
  allocated at `max_batch` size, then truncated to `n_active`.

### Remaining work
- GPU CI gate (requires CUDA runner in GitHub Actions)
- Metal backend: implemented and functional (f32 precision)

### Legacy stubs
- `crates/ns-compute/src/metal.rs` and `crates/ns-compute/src/cuda.rs` — old stubs returning `NotImplemented` (superseded by new implementation)

## Precision & Parity Policy (GPU)

- **Source of truth:** deterministic CPU path (см. `docs/plans/standards.md`).
- GPU backends могут использовать **mixed precision** (например `f32` compute + `f64` accumulation).
- Для GPU CI-проверок сравнивать в первую очередь `bestfit`/`uncertainties` и `Δtwice_nll` (разности), а не абсолютное `twice_nll`.

### Metal precision reality check

Практический факт: для Metal нельзя опираться на “нормальный” быстрый FP64 в шейдерах как на базовую стратегию (на части устройств FP64 отсутствует или настолько медленный/ограниченный, что превращает backend в анти‑перформанс).

**Следствие для дизайна Metal backend:**
- Metal backend считаем **FP32-first**.
- Для “числа должны совпадать” используем CPU deterministic path; Metal — **ускоритель**, но не эталон.

**Варианты стратегии точности на Metal:**
1) `f32` kernel per-bin + **CPU `f64` reduction** (переносим накопление на CPU, GPU считает вектор `nll_i` / partial sums).
2) `f32` kernel + **compensated summation** (`Kahan/Neumaier`) в `f32` на GPU (лучше, но не заменяет FP64).
3) **Double-double emulation** (`hi/lo` пары `f32`) для суммы/критичных частей (дороже, но контролируемо; можно включать по флагу).

Policy: по умолчанию (1) как самый простой и предсказуемый для parity; (2)/(3) — опции для perf/точности tradeoff.

---

## Содержание

- [Обзор стратегии](#обзор-стратегии)
- [Sprint 2C.1: Backend Abstraction (foundation)](#sprint-2c1-backend-abstraction-foundation-недели-13-14)
- [Sprint 2C.2: Metal Backend](#sprint-2c2-metal-backend-недели-23-25)
- [Sprint 2C.3: CUDA Backend](#sprint-2c3-cuda-backend-недели-26-27)
- [Sprint 2C.4: Batched Operations](#sprint-2c4-batched-operations-недели-28)
- [Performance Targets](#performance-targets)

---

## Обзор стратегии

### Целевая аудитория

| Платформа | Пользователи | Backend |
|-----------|--------------|---------|
| macOS (Apple Silicon) | Ученые на личных машинах, разработчики | CPU + Accelerate |
| Linux + NVIDIA | HEP кластеры (CERN, Fermilab), Cloud | CUDA |
| Linux без GPU | CI, small jobs | CPU (SIMD) |
| Windows | Редко, но возможно | CUDA или CPU |

### Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Code / Python                         │
│                                                                 │
│   model = nextstat.load_workspace("ws.json")                    │
│   result = nextstat.fit(model, backend="auto")  # или "metal"   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                    ns-compute (Rust)                            │
│                                                                 │
│   pub trait ComputeBackend {                                    │
│       fn nll_batch(&self, ...) -> Result<f64>;                  │
│       fn gradient_batch(&self, ...) -> Result<Vec<f64>>;        │
│       fn batched_fits(&self, ...) -> Result<Vec<FitResult>>;    │
│   }                                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  CpuBackend   │   │ MetalBackend  │   │  CudaBackend  │
│               │   │               │   │               │
│  - SIMD       │   │  - Metal      │   │  - CUDA       │
│  - Rayon      │   │  - MPS        │   │  - cuBLAS     │
│  - ndarray    │   │  - Accelerate │   │  - cuRAND     │
└───────────────┘   └───────────────┘   └───────────────┘
     Always              macOS              Linux/Win
                    (feature flag)      (feature flag)
```

### Feature Flags

```toml
# Cargo.toml
[features]
default = ["cpu"]
cpu = []                    # Always available
metal = ["dep:metal"]       # macOS only
cuda = ["dep:cudarc"]       # NVIDIA only
wgpu = ["dep:wgpu"]         # Cross-platform fallback (WebGPU)
all-backends = ["cpu", "metal", "cuda"]
```

---

## Sprint 2C.1: Backend Abstraction (foundation) (Недели 13-14)

### Epic 2C.1.1: ComputeBackend Trait

**Цель:** Определить единый интерфейс для всех GPU backends.

---

#### Task 2C.1.1.1: Define ComputeBackend trait

**Priority:** P0
**Effort:** 4 часа
**Dependencies:** Phase 1 complete

**Files:**
- Create: `crates/ns-compute/src/backend/mod.rs`
- Create: `crates/ns-compute/src/backend/traits.rs`
- Create: `crates/ns-compute/src/backend/cpu.rs`
- Modify: `crates/ns-compute/src/lib.rs`

**Acceptance Criteria:**
- [ ] Trait определен с async-safe методами
- [ ] CPU backend реализует trait
- [ ] Существующие тесты проходят с CPU backend
- [ ] Backend auto-detection работает

**Step 1: Write failing test**

```rust
// crates/ns-compute/src/backend/tests.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_nll() {
        let backend = CpuBackend::new();

        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![10.0, 20.0, 30.0];

        let nll = backend.poisson_nll(&observed, &expected).unwrap();

        // At MLE (obs == exp), backend must match canonical Poisson NLL (see standards)
        let expected_nll: f64 = crate::poisson_nll_vec(&observed, &expected);

        assert!((nll - expected_nll).abs() < 1e-10);
    }

    #[test]
    fn test_backend_auto_detection() {
        let backend = create_backend(BackendConfig::Auto);

        // Should return some backend
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        let name = backend.name();

        // On any platform, should get at least CPU
        assert!(!name.is_empty());
    }

    #[test]
    fn test_batched_nll() {
        let backend = CpuBackend::new();

        // 3 different parameter sets
        let params_batch = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.5, 1.0, 1.0],
            vec![2.0, 1.0, 1.0],
        ];

        let model_data = ModelData {
            observed: vec![55.0, 70.0],
            // Minimal single-sample model (used only for backend API tests)
            nominals: vec![vec![55.0, 70.0]],
            modifiers: vec![],
            n_params: 3,
            poi_idx: Some(0),
            param_names: vec!["mu".into(), "gamma0".into(), "gamma1".into()],
            prefit_means: vec![1.0, 1.0, 1.0],
            prefit_sigmas: vec![1.0, 1.0, 1.0],
        };

        let results = backend.nll_batch(&model_data, &params_batch).unwrap();

        assert_eq!(results.len(), 3);
        // NLL should increase as mu moves away from MLE
        assert!(results[0] <= results[1]);
        assert!(results[1] <= results[2]);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-compute test_cpu_backend_nll
# Expected: FAIL - module not found
```

**Step 3: Implement trait definition**

```rust
// crates/ns-compute/src/backend/traits.rs
//! ComputeBackend trait definition
//!
//! This trait abstracts GPU/CPU computation backends.

use ns_core::{Result, types::Float};

/// Data needed for model evaluation
#[derive(Debug, Clone)]
pub struct ModelData {
    /// Observed data (concatenated across channels)
    pub observed: Vec<Float>,
    /// Nominal expectations per sample
    pub nominals: Vec<Vec<Float>>,
    /// Modifier specifications
    pub modifiers: Vec<ModifierSpec>,
    /// Number of parameters
    pub n_params: usize,
    /// POI index
    pub poi_idx: Option<usize>,
    /// Parameter names (length = n_params)
    pub param_names: Vec<String>,
    /// Pre-fit means (length = n_params)
    pub prefit_means: Vec<Float>,
    /// Pre-fit sigmas/uncertainties (length = n_params)
    pub prefit_sigmas: Vec<Float>,
}

/// Specification of a modifier for GPU evaluation
#[derive(Debug, Clone)]
pub enum ModifierSpec {
    NormFactor {
        param_idx: usize,
        sample_idx: usize,
    },
    StatError {
        param_indices: Vec<usize>,
        rel_uncerts: Vec<Float>,
        sample_idx: usize,
    },
    NormSys {
        param_idx: usize,
        sample_idx: usize,
        hi_factor: Float,
        lo_factor: Float,
    },
    HistoSys {
        param_idx: usize,
        sample_idx: usize,
        hi_data: Vec<Float>,
        lo_data: Vec<Float>,
    },
}

/// Result of a single fit
#[derive(Debug, Clone)]
pub struct FitResultData {
    pub bestfit: Vec<Float>,
    pub uncertainties: Vec<Float>,
    pub nll: Float,
    pub converged: bool,
}

/// Backend capabilities
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    /// Supports batched NLL computation
    pub batched_nll: bool,
    /// Supports batched gradient computation
    pub batched_gradient: bool,
    /// Supports batched full fits
    pub batched_fits: bool,
    /// Maximum batch size (0 = unlimited)
    pub max_batch_size: usize,
    /// Supports automatic differentiation
    pub autodiff: bool,
}

/// Compute backend trait
///
/// All GPU and CPU backends must implement this trait.
pub trait ComputeBackend: Send + Sync {
    /// Backend name (e.g., "cpu", "metal", "cuda")
    fn name(&self) -> &'static str;

    /// Backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Compute Poisson NLL for observed vs expected
    fn poisson_nll(&self, observed: &[Float], expected: &[Float]) -> Result<Float>;

    /// Compute expected data at given parameters
    fn expected_data(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>>;

    /// Compute full NLL (Poisson + constraints) at given parameters
    fn nll(&self, model: &ModelData, params: &[Float]) -> Result<Float> {
        let expected = self.expected_data(model, params)?;
        let poisson = self.poisson_nll(&model.observed, &expected)?;

        // Add Gaussian constraints for nuisance parameters
        let constraints = self.gaussian_constraints(model, params)?;

        Ok(poisson + constraints)
    }

    /// Compute Gaussian constraint terms
    fn gaussian_constraints(&self, model: &ModelData, params: &[Float]) -> Result<Float>;

    /// Compute gradient of NLL (optional, for gradient-based optimizers)
    fn gradient(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
        // Default: numerical gradient
        self.numerical_gradient(model, params)
    }

    /// Numerical gradient (finite differences)
    fn numerical_gradient(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
        let h = 1e-7;
        let n = params.len();
        let mut grad = vec![0.0; n];

        let f0 = self.nll(model, params)?;

        for i in 0..n {
            let mut params_plus = params.to_vec();
            params_plus[i] += h;
            let f_plus = self.nll(model, &params_plus)?;
            grad[i] = (f_plus - f0) / h;
        }

        Ok(grad)
    }

    // === Batched operations (for GPU efficiency) ===

    /// Compute NLL for multiple parameter sets in parallel
    fn nll_batch(&self, model: &ModelData, params_batch: &[Vec<Float>]) -> Result<Vec<Float>> {
        // Default: sequential (CPU), override for GPU
        params_batch.iter()
            .map(|params| self.nll(model, params))
            .collect()
    }

    /// Compute gradients for multiple parameter sets in parallel
    fn gradient_batch(&self, model: &ModelData, params_batch: &[Vec<Float>]) -> Result<Vec<Vec<Float>>> {
        params_batch.iter()
            .map(|params| self.gradient(model, params))
            .collect()
    }

    /// Run multiple independent fits in parallel (for toys, ranking)
    fn fit_batch(
        &self,
        model: &ModelData,
        init_params_batch: &[Vec<Float>],
        bounds: &[(Float, Float)],
    ) -> Result<Vec<FitResultData>> {
        // Default: sequential
        init_params_batch.iter()
            .map(|init| self.fit_single(model, init, bounds))
            .collect()
    }

    /// Single fit (used by fit_batch default impl)
    fn fit_single(
        &self,
        model: &ModelData,
        init_params: &[Float],
        bounds: &[(Float, Float)],
    ) -> Result<FitResultData>;
}

/// Boxed backend for dynamic dispatch
pub type BoxedBackend = Box<dyn ComputeBackend>;
```

**Step 4: Implement CPU backend**

```rust
// crates/ns-compute/src/backend/cpu.rs
//! CPU backend implementation using SIMD and Rayon

use rayon::prelude::*;
use ns_core::{Result, types::Float};
use super::traits::*;

/// CPU backend with SIMD and multi-threading
pub struct CpuBackend {
    /// Number of threads (0 = auto)
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { num_threads: 0 }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            batched_nll: true,
            batched_gradient: true,
            batched_fits: true,
            max_batch_size: 0, // unlimited
            autodiff: false,   // numerical only for now
        }
    }

    fn poisson_nll(&self, observed: &[Float], expected: &[Float]) -> Result<Float> {
        Ok(crate::poisson_nll_vec(observed, expected))
    }

    fn expected_data(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
        let n_bins = model.observed.len();
        let mut expected = vec![0.0; n_bins];

        // Sum contributions from all samples
        for (sample_idx, nominal) in model.nominals.iter().enumerate() {
            let mut sample_exp = nominal.clone();

            // Apply modifiers
            for modifier in &model.modifiers {
                apply_modifier_cpu(modifier, sample_idx, params, &mut sample_exp);
            }

            // Add to total
            for (i, &v) in sample_exp.iter().enumerate() {
                if i < expected.len() {
                    expected[i] += v;
                }
            }
        }

        Ok(expected)
    }

    fn gaussian_constraints(&self, model: &ModelData, params: &[Float]) -> Result<Float> {
        let mut constraint_nll = 0.0;

        for modifier in &model.modifiers {
            match modifier {
                ModifierSpec::StatError { param_indices, rel_uncerts, .. } => {
                    for (&idx, &rel_unc) in param_indices.iter().zip(rel_uncerts.iter()) {
                        let gamma = params[idx];
                        // Gaussian constraint: gamma ~ N(1, rel_unc)
                        let z = (gamma - 1.0) / rel_unc;
                        constraint_nll += 0.5 * z * z;
                    }
                }
                ModifierSpec::NormSys { param_idx, .. } |
                ModifierSpec::HistoSys { param_idx, .. } => {
                    // Standard Gaussian constraint: alpha ~ N(0, 1)
                    let alpha = params[*param_idx];
                    constraint_nll += 0.5 * alpha * alpha;
                }
                _ => {}
            }
        }

        Ok(constraint_nll)
    }

    fn nll_batch(&self, model: &ModelData, params_batch: &[Vec<Float>]) -> Result<Vec<Float>> {
        // Parallel execution with Rayon
        let results: Vec<Result<Float>> = params_batch
            .par_iter()
            .map(|params| self.nll(model, params))
            .collect();

        results.into_iter().collect()
    }

    fn gradient_batch(&self, model: &ModelData, params_batch: &[Vec<Float>]) -> Result<Vec<Vec<Float>>> {
        let results: Vec<Result<Vec<Float>>> = params_batch
            .par_iter()
            .map(|params| self.gradient(model, params))
            .collect();

        results.into_iter().collect()
    }

    fn fit_batch(
        &self,
        model: &ModelData,
        init_params_batch: &[Vec<Float>],
        bounds: &[(Float, Float)],
    ) -> Result<Vec<FitResultData>> {
        let results: Vec<Result<FitResultData>> = init_params_batch
            .par_iter()
            .map(|init| self.fit_single(model, init, bounds))
            .collect();

        results.into_iter().collect()
    }

    fn fit_single(
        &self,
        model: &ModelData,
        init_params: &[Float],
        bounds: &[(Float, Float)],
    ) -> Result<FitResultData> {
        // Use Nelder-Mead from ns-inference
        // (This is a simplified version - actual impl uses the minimizer)

        use crate::nll::nelder_mead;

        let objective = |params: &[Float]| -> Float {
            self.nll(model, params).unwrap_or(Float::INFINITY)
        };

        let result = nelder_mead(objective, init_params, bounds, 1000, 1e-8)?;

        // Compute uncertainties via Hessian
        let uncertainties = self.compute_uncertainties(model, &result.x)?;

        Ok(FitResultData {
            bestfit: result.x,
            uncertainties,
            nll: result.fun,
            converged: result.converged,
        })
    }
}

impl CpuBackend {
    fn compute_uncertainties(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
        let n = params.len();
        let h = 1e-5;
        let f0 = self.nll(model, params)?;

        let mut uncertainties = vec![0.0; n];

        for i in 0..n {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[i] += h;
            p_minus[i] -= h;

            let f_plus = self.nll(model, &p_plus)?;
            let f_minus = self.nll(model, &p_minus)?;

            let d2f = (f_plus - 2.0 * f0 + f_minus) / (h * h);

            uncertainties[i] = if d2f > 0.0 {
                (1.0 / d2f).sqrt()
            } else {
                Float::NAN
            };
        }

        Ok(uncertainties)
    }
}

fn apply_modifier_cpu(
    modifier: &ModifierSpec,
    target_sample_idx: usize,
    params: &[Float],
    data: &mut [Float],
) {
    match modifier {
        ModifierSpec::NormFactor { param_idx, sample_idx } => {
            if *sample_idx == target_sample_idx {
                let factor = params[*param_idx];
                for v in data.iter_mut() {
                    *v *= factor;
                }
            }
        }
        ModifierSpec::StatError { param_indices, sample_idx, .. } => {
            if *sample_idx == target_sample_idx {
                for (i, &idx) in param_indices.iter().enumerate() {
                    if i < data.len() {
                        data[i] *= params[idx];
                    }
                }
            }
        }
        ModifierSpec::NormSys { param_idx, sample_idx, hi_factor, lo_factor } => {
            if *sample_idx == target_sample_idx {
                let alpha = params[*param_idx];
                let factor = if alpha >= 0.0 {
                    hi_factor.powf(alpha)
                } else {
                    lo_factor.powf(-alpha)
                };
                for v in data.iter_mut() {
                    *v *= factor;
                }
            }
        }
        ModifierSpec::HistoSys { param_idx, sample_idx, hi_data, lo_data } => {
            if *sample_idx == target_sample_idx {
                let alpha = params[*param_idx];
                for (i, v) in data.iter_mut().enumerate() {
                    let nominal = *v;
                    let hi = hi_data.get(i).copied().unwrap_or(nominal);
                    let lo = lo_data.get(i).copied().unwrap_or(nominal);
                    *v = if alpha >= 0.0 {
                        nominal + alpha * (hi - nominal)
                    } else {
                        nominal + (-alpha) * (lo - nominal)
                    };
                }
            }
        }
    }
}
```

**Step 5: Create backend selection**

```rust
// crates/ns-compute/src/backend/mod.rs
//! Compute backend abstraction

pub mod traits;
pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod tests;

pub use traits::*;
pub use cpu::CpuBackend;

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

/// Backend configuration
#[derive(Debug, Clone, Default)]
pub enum BackendConfig {
    /// Auto-detect best available backend
    #[default]
    Auto,
    /// Force CPU backend
    Cpu,
    /// Force Metal backend (macOS only)
    #[cfg(feature = "metal")]
    Metal,
    /// Force CUDA backend (NVIDIA only)
    #[cfg(feature = "cuda")]
    Cuda,
}

/// Create a backend based on configuration
pub fn create_backend(config: BackendConfig) -> ns_core::Result<BoxedBackend> {
    match config {
        BackendConfig::Auto => auto_select_backend(),
        BackendConfig::Cpu => Ok(Box::new(CpuBackend::new())),
        #[cfg(feature = "metal")]
        BackendConfig::Metal => Ok(Box::new(MetalBackend::new()?)),
        #[cfg(feature = "cuda")]
        BackendConfig::Cuda => Ok(Box::new(CudaBackend::new()?)),
    }
}

/// Auto-select the best available backend
fn auto_select_backend() -> ns_core::Result<BoxedBackend> {
    // Priority: CUDA > Metal > CPU

    #[cfg(feature = "cuda")]
    {
        if let Ok(backend) = CudaBackend::new() {
            tracing::info!("Using CUDA backend");
            return Ok(Box::new(backend));
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(backend) = MetalBackend::new() {
            tracing::info!("Using Metal backend");
            return Ok(Box::new(backend));
        }
    }

    tracing::info!("Using CPU backend");
    Ok(Box::new(CpuBackend::new()))
}

/// Get the default backend (convenience function)
pub fn default_backend() -> BoxedBackend {
    create_backend(BackendConfig::Auto).expect("Failed to create backend")
}
```

**Step 6: Run tests**

```bash
cargo test -p ns-compute backend
# Expected: All PASS
```

**Step 7: Commit**

```bash
git add crates/ns-compute/src/backend/
git commit -m "feat(compute): add ComputeBackend trait and CPU implementation"
```

---

## Sprint 2C.2: Metal Backend (Недели 23-25)

### Epic 2C.2.1: Metal GPU Compute

**Цель:** Реализовать Metal backend для Apple Silicon.

---

#### Task 2C.2.1.1: Metal context and buffer management

**Priority:** P0 (для Mac users)
**Effort:** 8 часов
**Dependencies:** Task 2C.1.1.1

**Files:**
- Create: `crates/ns-compute/src/backend/metal/mod.rs`
- Create: `crates/ns-compute/src/backend/metal/context.rs`
- Create: `crates/ns-compute/src/backend/metal/buffers.rs`
- Create: `crates/ns-compute/src/backend/metal/shaders/nll.metal`

**Acceptance Criteria:**
- [ ] Metal device detection работает
- [ ] Buffer allocation/deallocation
- [ ] Shader compilation
- [ ] Basic NLL compute shader

**Step 1: Add metal dependency**

```toml
# Cargo.toml [workspace.dependencies]
metal = "0.33"
objc = "0.2"
block = "0.1"
```

```toml
# crates/ns-compute/Cargo.toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = { workspace = true, optional = true }
objc = { workspace = true, optional = true }

[features]
metal = ["dep:metal", "dep:objc"]
```

**Step 2: Implement Metal context**

```rust
// crates/ns-compute/src/backend/metal/context.rs
//! Metal GPU context management

use metal::*;
use ns_core::{Error, Result};
use std::sync::Arc;

/// Metal compute context
pub struct MetalContext {
    /// Metal device
    device: Device,
    /// Command queue
    queue: CommandQueue,
    /// Compiled compute pipelines
    pipelines: MetalPipelines,
}

struct MetalPipelines {
    poisson_nll: ComputePipelineState,
    expected_data: ComputePipelineState,
    gradient: ComputePipelineState,
}

impl MetalContext {
    /// Create new Metal context
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| Error::Numerical("No Metal device found".into()))?;

        tracing::info!(
            "Metal device: {} ({})",
            device.name(),
            if device.is_low_power() { "low power" } else { "high performance" }
        );

        let queue = device.new_command_queue();

        let pipelines = Self::compile_shaders(&device)?;

        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    fn compile_shaders(device: &Device) -> Result<MetalPipelines> {
        // Embed shader source at compile time
        let shader_source = include_str!("shaders/nll.metal");

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_source, &options)
            .map_err(|e| Error::Numerical(format!("Shader compilation failed: {}", e)))?;

        let poisson_nll = Self::create_pipeline(device, &library, "poisson_nll_kernel")?;
        let expected_data = Self::create_pipeline(device, &library, "expected_data_kernel")?;
        let gradient = Self::create_pipeline(device, &library, "gradient_kernel")?;

        Ok(MetalPipelines {
            poisson_nll,
            expected_data,
            gradient,
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        function_name: &str,
    ) -> Result<ComputePipelineState> {
        let function = library
            .get_function(function_name, None)
            .map_err(|e| Error::Numerical(format!("Function {} not found: {}", function_name, e)))?;

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| Error::Numerical(format!("Pipeline creation failed: {}", e)))
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get command queue reference
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Get pipelines reference
    pub fn pipelines(&self) -> &MetalPipelines {
        &self.pipelines
    }

    /// Create a new buffer
    pub fn create_buffer<T>(&self, data: &[T]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        );
        buffer
    }

    /// Create an empty buffer
    pub fn create_buffer_empty(&self, size: u64) -> Buffer {
        self.device.new_buffer(size, MTLResourceOptions::StorageModeShared)
    }
}
```

**Step 3: Implement Metal shaders**

```metal
// crates/ns-compute/src/backend/metal/shaders/nll.metal
#include <metal_stdlib>
using namespace metal;

/// Poisson NLL kernel
/// Computes: sum(λ - n * log(λ)) for each bin
kernel void poisson_nll_kernel(
    device const float* observed [[buffer(0)]],
    device const float* expected [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    float n = observed[tid];
    float lam = expected[tid];

    float nll = 0.0;
    if (lam > 0.0) {
        nll = lam - n * log(lam);
    } else {
        nll = INFINITY;
    }

    partial_sums[tid] = nll;
}

/// Parallel reduction for summing partial results
kernel void reduce_sum_kernel(
    device float* data [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];

    shared_data[tid] = data[bid * block_size + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in shared memory
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        result[bid] = shared_data[0];
    }
}

/// Expected data computation kernel
/// Applies modifiers to nominal histograms
kernel void expected_data_kernel(
    device const float* nominals [[buffer(0)]],
    device const float* params [[buffer(1)]],
    device const int* modifier_types [[buffer(2)]],
    device const int* modifier_params [[buffer(3)]],
    device const float* modifier_data [[buffer(4)]],
    device float* expected [[buffer(5)]],
    constant uint& n_bins [[buffer(6)]],
    constant uint& n_modifiers [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_bins) return;

    float value = nominals[tid];

    // Apply each modifier
    // (Simplified - actual implementation needs modifier dispatch)
    for (uint m = 0; m < n_modifiers; m++) {
        int mod_type = modifier_types[m];

        if (mod_type == 0) {
            // NormFactor
            int param_idx = modifier_params[m * 4];
            value *= params[param_idx];
        }
        // Other modifier types are handled via dispatch (NormSys/HistoSys/StatError, etc.)
    }

    expected[tid] = value;
}

/// Batched NLL kernel
/// Computes NLL for multiple parameter sets in parallel
kernel void batched_nll_kernel(
    device const float* observed [[buffer(0)]],
    device const float* all_expected [[buffer(1)]],  // batch_size * n_bins
    device float* results [[buffer(2)]],              // batch_size
    constant uint& n_bins [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    if (batch_idx >= batch_size) return;

    float nll = 0.0;
    uint offset = batch_idx * n_bins;

    for (uint i = 0; i < n_bins; i++) {
        float n = observed[i];
        float lam = all_expected[offset + i];

        if (lam > 0.0) {
            nll += lam - n * log(lam);
        } else {
            nll = INFINITY;
            break;
        }
    }

    results[batch_idx] = nll;
}
```

**Step 4: Implement MetalBackend**

```rust
// crates/ns-compute/src/backend/metal/mod.rs
//! Metal backend for Apple Silicon

mod context;
mod buffers;

use context::MetalContext;
use metal::*;
use ns_core::{Result, types::Float};
use super::traits::*;

/// Metal GPU backend
pub struct MetalBackend {
    context: MetalContext,
}

impl MetalBackend {
    /// Create new Metal backend
    pub fn new() -> Result<Self> {
        let context = MetalContext::new()?;
        Ok(Self { context })
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            batched_nll: true,
            batched_gradient: true,
            batched_fits: true,
            max_batch_size: 10000,  // Reasonable for GPU memory
            // GPU AD is future work; Phase 2B provides CPU AD (`Model::gradient`).
            autodiff: false,
        }
    }

    fn poisson_nll(&self, observed: &[Float], expected: &[Float]) -> Result<Float> {
        let n = observed.len();

        // Convert to f32 for Metal (f64 support is limited)
        let obs_f32: Vec<f32> = observed.iter().map(|&x| x as f32).collect();
        let exp_f32: Vec<f32> = expected.iter().map(|&x| x as f32).collect();

        // Create buffers
        let obs_buffer = self.context.create_buffer(&obs_f32);
        let exp_buffer = self.context.create_buffer(&exp_f32);
        let result_buffer = self.context.create_buffer_empty((n * 4) as u64);

        // Create command buffer
        let command_buffer = self.context.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.context.pipelines().poisson_nll);
        encoder.set_buffer(0, Some(&obs_buffer), 0);
        encoder.set_buffer(1, Some(&exp_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        // Dispatch
        let threads_per_group = 256;
        let num_groups = (n + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );

        encoder.end_encoding();

        // Execute and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results and sum
        let result_ptr = result_buffer.contents() as *const f32;
        let results = unsafe { std::slice::from_raw_parts(result_ptr, n) };
        let total: f32 = results.iter().sum();

        // Add constant term ln Γ(n+1) to match canonical Poisson NLL (see standards).
        let const_term: Float = observed
            .iter()
            .map(|&n| statrs::function::gamma::ln_gamma(n + 1.0))
            .sum();

        Ok(const_term + total as Float)
    }

    fn expected_data(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
        // For complex models, use GPU kernel
        // For simple models, CPU may be faster due to memory transfer overhead

        if model.observed.len() < 100 {
            // Fall back to CPU for small models
            return super::cpu::CpuBackend::new().expected_data(model, params);
        }

        // GPU implementation
        // Stub: until `expected_data` kernel is implemented, fall back to CPU.
        super::cpu::CpuBackend::new().expected_data(model, params)
    }

    fn gaussian_constraints(&self, model: &ModelData, params: &[Float]) -> Result<Float> {
        // Constraints are typically few - CPU is fine
        super::cpu::CpuBackend::new().gaussian_constraints(model, params)
    }

    fn nll_batch(&self, model: &ModelData, params_batch: &[Vec<Float>]) -> Result<Vec<Float>> {
        let batch_size = params_batch.len();
        let n_bins = model.observed.len();

        if batch_size < 10 {
            // Not worth GPU overhead for small batches
            return super::cpu::CpuBackend::new().nll_batch(model, params_batch);
        }

        // Compute all expected data on GPU
        let mut all_expected: Vec<f32> = Vec::with_capacity(batch_size * n_bins);

        for params in params_batch {
            let expected = self.expected_data(model, params)?;
            all_expected.extend(expected.iter().map(|&x| x as f32));
        }

        let obs_f32: Vec<f32> = model.observed.iter().map(|&x| x as f32).collect();

        // Create buffers
        let obs_buffer = self.context.create_buffer(&obs_f32);
        let exp_buffer = self.context.create_buffer(&all_expected);
        let result_buffer = self.context.create_buffer_empty((batch_size * 4) as u64);

        // Dispatch batched NLL kernel
        let command_buffer = self.context.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.context.pipelines().poisson_nll);
        encoder.set_buffer(0, Some(&obs_buffer), 0);
        encoder.set_buffer(1, Some(&exp_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        let n_bins_u32 = n_bins as u32;
        let batch_size_u32 = batch_size as u32;
        encoder.set_bytes(3, 4, &n_bins_u32 as *const _ as *const _);
        encoder.set_bytes(4, 4, &batch_size_u32 as *const _ as *const _);

        encoder.dispatch_thread_groups(
            MTLSize::new(batch_size as u64, 1, 1),
            MTLSize::new(1, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let result_ptr = result_buffer.contents() as *const f32;
        let results = unsafe { std::slice::from_raw_parts(result_ptr, batch_size) };

        // Add Gaussian constraints (small, keep on CPU)
        let mut final_results = Vec::with_capacity(batch_size);
        for (i, params) in params_batch.iter().enumerate() {
            let poisson_nll = results[i] as Float;
            let constraints = self.gaussian_constraints(model, params)?;
            final_results.push(poisson_nll + constraints);
        }

        Ok(final_results)
    }

    fn fit_single(
        &self,
        model: &ModelData,
        init_params: &[Float],
        bounds: &[(Float, Float)],
    ) -> Result<FitResultData> {
        // Use L-BFGS with GPU gradient
        // For now, delegate to CPU optimizer with GPU NLL evaluation
        super::cpu::CpuBackend::new().fit_single(model, init_params, bounds)
    }

    fn fit_batch(
        &self,
        model: &ModelData,
        init_params_batch: &[Vec<Float>],
        bounds: &[(Float, Float)],
    ) -> Result<Vec<FitResultData>> {
        // Parallel fits using GPU for NLL evaluation
        // Each fit runs independently, but NLL calls use batched GPU kernel

        use rayon::prelude::*;

        init_params_batch
            .par_iter()
            .map(|init| self.fit_single(model, init, bounds))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_poisson_nll() {
        let backend = MetalBackend::new().unwrap();

        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![10.0, 20.0, 30.0];

        let nll = backend.poisson_nll(&observed, &expected).unwrap();

        // Compare with CPU
        let cpu_nll = CpuBackend::new().poisson_nll(&observed, &expected).unwrap();

        assert!((nll - cpu_nll).abs() < 1e-4); // f32 precision
    }
}
```

**Step 5: Update Cargo.toml and feature flags**

```toml
# crates/ns-compute/Cargo.toml
[features]
default = []
metal = ["dep:metal", "dep:objc"]

[target.'cfg(target_os = "macos")'.dependencies]
metal = { version = "0.33", optional = true }
objc = { version = "0.2", optional = true }
```

**Step 6: Run tests on Mac**

```bash
cargo test -p ns-compute --features metal
# Expected: All PASS on macOS
```

**Step 7: Commit**

```bash
git add crates/ns-compute/src/backend/metal/
git commit -m "feat(compute): add Metal backend for Apple Silicon"
```

---

## Sprint 2C.3: CUDA Backend (Недели 26-27)

### Epic 2C.3.1: CUDA GPU Compute

**Цель:** Реализовать CUDA backend для NVIDIA GPUs.

*(Структура аналогична Metal, но использует cudarc crate)*

---

#### Task 2C.3.1.1: CUDA context setup

**Files:**
- Create: `crates/ns-compute/src/backend/cuda/mod.rs`
- Create: `crates/ns-compute/src/backend/cuda/kernels.cu`

**Step 1: Add CUDA dependency**

```toml
# Cargo.toml [workspace.dependencies]
cudarc = { version = "0.19", features = ["cuda-version-from-build-system"] }
```

> Note: вместо `cuda-version-from-build-system` можно выбрать **фиксированную** фичу под вашу среду (`cuda-12080`, `cuda-13010`, …). Это упрощает воспроизводимость CI, если CUDA toolkit pinned.

**Step 2: Implement CUDA kernels (PTX)**

```cuda
// crates/ns-compute/src/backend/cuda/kernels.cu
extern "C" __global__ void poisson_nll_kernel(
    const double* observed,
    const double* expected,
    double* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double n_obs = observed[idx];
    double lam = expected[idx];

    double nll = 0.0;
    if (lam > 0.0) {
        nll = lam - n_obs * log(lam);
    } else {
        nll = INFINITY;
    }

    result[idx] = nll;
}

extern "C" __global__ void batched_nll_kernel(
    const double* observed,
    const double* all_expected,
    double* results,
    int n_bins,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    double nll = 0.0;
    int offset = batch_idx * n_bins;

    for (int i = 0; i < n_bins; i++) {
        double n_obs = observed[i];
        double lam = all_expected[offset + i];

        if (lam > 0.0) {
            nll += lam - n_obs * log(lam);
        } else {
            nll = INFINITY;
            break;
        }
    }

    results[batch_idx] = nll;
}
```

*(Полная реализация CudaBackend аналогична MetalBackend)*

---

## Sprint 2C.4: Batched Operations (Неделя 28)

### Epic 2C.4.1: High-Performance Batched Fits

**Цель:** Оптимизировать batched operations для ranking plots и toy studies.

---

#### Task 2C.4.1.1: Ranking plot optimization

**Acceptance Criteria:**
- [ ] Ranking plot для 100 NP < 30 секунд
- [ ] GPU memory efficient (stream processing)
- [ ] Progress reporting

**Implementation:**

```rust
// crates/ns-inference/src/ranking.rs
//! Ranking plot computation (NP impact on POI)

use ns_compute::backend::{ComputeBackend, BoxedBackend, ModelData, FitResultData};
use ns_core::{Result, types::Float};
use rayon::prelude::*;

/// Result for a single nuisance parameter
#[derive(Debug, Clone)]
pub struct RankingEntry {
    pub param_name: String,
    pub param_idx: usize,
    /// Impact on POI when fixed at +1σ
    pub impact_up: Float,
    /// Impact on POI when fixed at -1σ
    pub impact_down: Float,
    /// Pull value (post-fit - pre-fit) / uncertainty
    pub pull: Float,
    /// Constraint (post-fit uncertainty / pre-fit uncertainty)
    pub constraint: Float,
}

/// Compute ranking plot
pub fn compute_ranking(
    backend: &dyn ComputeBackend,
    model: &ModelData,
    nominal_fit: &FitResultData,
    progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
) -> Result<Vec<RankingEntry>> {
    let poi_idx = model.poi_idx.ok_or_else(|| {
        ns_core::Error::ModelSpec("No POI defined".into())
    })?;

    let n_params = model.n_params;
    let mut entries = Vec::with_capacity(n_params - 1);

    // Pre-compute bounds for fixed fits
    let nominal_bounds: Vec<_> = (0..n_params)
        .map(|i| {
            if i == poi_idx {
                (-10.0, 10.0)  // POI always free
            } else {
                (-5.0, 5.0)    // NP bounds
            }
        })
        .collect();

    // Prepare batch of fits for GPU efficiency
    // For each NP: 2 fits (fixed at ±1σ)
    let mut fit_configs: Vec<(usize, Float, Vec<Float>)> = Vec::new();

    for np_idx in 0..n_params {
        if np_idx == poi_idx {
            continue;
        }

        let np_value = nominal_fit.bestfit[np_idx];
        let np_uncert = nominal_fit.uncertainties[np_idx];

        // Fixed at +1σ
        let mut init_up = nominal_fit.bestfit.clone();
        init_up[np_idx] = np_value + np_uncert;
        fit_configs.push((np_idx, 1.0, init_up));

        // Fixed at -1σ
        let mut init_down = nominal_fit.bestfit.clone();
        init_down[np_idx] = np_value - np_uncert;
        fit_configs.push((np_idx, -1.0, init_down));
    }

    // Run all fits
    let total_fits = fit_configs.len();

    if backend.capabilities().batched_fits {
        // GPU batched fits
        let init_params: Vec<_> = fit_configs.iter().map(|(_, _, p)| p.clone()).collect();

        // Create fixed-parameter bounds for each fit
        let all_bounds: Vec<Vec<(Float, Float)>> = fit_configs.iter()
            .map(|(np_idx, sigma, _)| {
                let mut bounds = nominal_bounds.clone();
                let fixed_val = nominal_fit.bestfit[*np_idx] + sigma * nominal_fit.uncertainties[*np_idx];
                bounds[*np_idx] = (fixed_val, fixed_val);  // Fix this parameter
                bounds
            })
            .collect();

        // This is simplified - actual impl needs per-fit bounds support
        let results = backend.fit_batch(model, &init_params, &nominal_bounds)?;

        // Process results
        for (i, ((np_idx, sigma, _), fit_result)) in fit_configs.iter().zip(results.iter()).enumerate() {
            if let Some(ref cb) = progress_callback {
                cb(i + 1, total_fits);
            }

            let poi_shift = fit_result.bestfit[poi_idx] - nominal_fit.bestfit[poi_idx];

            // Find or create entry
            let entry = entries.iter_mut().find(|e: &&mut RankingEntry| e.param_idx == *np_idx);

            if let Some(entry) = entry {
                if *sigma > 0.0 {
                    entry.impact_up = poi_shift;
                } else {
                    entry.impact_down = poi_shift;
                }
            } else {
                let mut new_entry = RankingEntry {
                    param_name: model.param_names[*np_idx].clone(),
                    param_idx: *np_idx,
                    impact_up: 0.0,
                    impact_down: 0.0,
                    pull: (nominal_fit.bestfit[*np_idx] - model.prefit_means[*np_idx])
                        / model.prefit_sigmas[*np_idx].max(1e-12),
                    constraint: nominal_fit.uncertainties[*np_idx]
                        / model.prefit_sigmas[*np_idx].max(1e-12),
                };

                if *sigma > 0.0 {
                    new_entry.impact_up = poi_shift;
                } else {
                    new_entry.impact_down = poi_shift;
                }

                entries.push(new_entry);
            }
        }
    } else {
        // Sequential CPU fits
        for (i, (np_idx, sigma, init)) in fit_configs.iter().enumerate() {
            if let Some(ref cb) = progress_callback {
                cb(i + 1, total_fits);
            }

            let mut bounds = nominal_bounds.clone();
            let fixed_val = nominal_fit.bestfit[*np_idx] + sigma * nominal_fit.uncertainties[*np_idx];
            bounds[*np_idx] = (fixed_val, fixed_val);

            let result = backend.fit_single(model, init, &bounds)?;
            let poi_shift = result.bestfit[poi_idx] - nominal_fit.bestfit[poi_idx];

            let entry = entries.iter_mut().find(|e: &&mut RankingEntry| e.param_idx == *np_idx);

            if let Some(entry) = entry {
                if *sigma > 0.0 {
                    entry.impact_up = poi_shift;
                } else {
                    entry.impact_down = poi_shift;
                }
            } else {
                let mut new_entry = RankingEntry {
                    param_name: model.param_names[*np_idx].clone(),
                    param_idx: *np_idx,
                    impact_up: 0.0,
                    impact_down: 0.0,
                    pull: (nominal_fit.bestfit[*np_idx] - model.prefit_means[*np_idx])
                        / model.prefit_sigmas[*np_idx].max(1e-12),
                    constraint: nominal_fit.uncertainties[*np_idx]
                        / model.prefit_sigmas[*np_idx].max(1e-12),
                };

                if *sigma > 0.0 {
                    new_entry.impact_up = poi_shift;
                } else {
                    new_entry.impact_down = poi_shift;
                }

                entries.push(new_entry);
            }
        }
    }

    // Sort by total impact
    entries.sort_by(|a, b| {
        let impact_a = a.impact_up.abs().max(a.impact_down.abs());
        let impact_b = b.impact_up.abs().max(b.impact_down.abs());
        impact_b.partial_cmp(&impact_a).unwrap()
    });

    Ok(entries)
}
```

---

## Performance Targets

| Operation | CPU (M3 Max) | Metal (M3 Max) | CUDA (A100) | Target |
|-----------|--------------|----------------|-------------|--------|
| Simple NLL (100 bins) | 0.1ms | 0.05ms | 0.02ms | <1ms |
| Complex fit (100 NP) | 500ms | 100ms | 50ms | <200ms |
| Ranking (100 NP) | 3 min | 40s | 15s | <30s |
| 1000 toy fits | 5 min | 30s | 10s | <1 min |
| Batched NLL (1000x) | 100ms | 5ms | 2ms | <10ms |

---

## Критерии завершения

### Exit Criteria

Phase 2C GPU backends завершена когда:

1. [x] Metal backend проходит все тесты на macOS — **DONE** (f32 precision, NLL parity 1.27e-6)
2. [x] CUDA backend проходит все тесты на Linux + NVIDIA — **DONE** (355+ tests, GEX44 RTX 4000)
3. [x] Backend auto-detection работает — **DONE** (feature-gated dispatch + runtime check)
4. [x] Ranking plot < 30s для 100 NP на GPU — **DONE** (CPU-based ranking with warm-start, GPU for batch)
5. [x] 1000 toy fits < 1 min на GPU — **DONE** (batch toys with lockstep L-BFGS-B)
6. [x] Python API поддерживает backend selection — **DONE** (`device="cuda"`, `device="metal"`)
7. [x] Documentation обновлена — **DONE** (benchmarks.md, gpu-contract.md, phase-2c plan)
8. [x] GPU benchmarks завершены — **DONE** (RTX 4000, full suite: fit/scan/diff/profiled/NN)

### Benchmarks Required

```bash
# Run all backend benchmarks
cargo bench --features "metal cuda" -- --save-baseline gpu_backends

# Compare with CPU baseline
cargo bench --features "metal cuda" -- --baseline cpu_only
```

---

*Предыдущая секция: [Phase 2B: Autodiff & Optimizers](./phase-2b-autodiff.md)*
