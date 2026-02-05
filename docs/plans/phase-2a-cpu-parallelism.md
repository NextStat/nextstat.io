# Фаза II-A: CPU Parallelism (P0, приоритетная)

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** Реализовать эффективную CPU параллельность для научных кластеров без GPU.

**Duration:** Недели 13-20 — выполняется **перед Phase II-C (GPU)** и может идти параллельно с Phase II-B (Autodiff).

**Architecture:** Rayon work-stealing, SIMD через std::simd, memory-efficient batching.

**Tech Stack:** Rust, Rayon, packed_simd / std::simd, crossbeam.

---

## Почему это приоритет

### Реальность научных кластеров

| Кластер | GPU | CPU | Типичное использование |
|---------|-----|-----|------------------------|
| CERN lxplus | Ограничено | 1000s cores | Массовый batch |
| DESY NAF | Немного | 10000s cores | Анализ данных |
| Fermilab | Ограничено | Много | MC production |
| Университеты | Редко | Стандартные | Студенты, постдоки |
| Личные машины | M-серия/нет | 8-16 cores | Разработка |

**Вывод:** 80%+ вычислений будет на CPU. GPU — nice-to-have для ускорения, но CPU параллельность — must-have.

### Целевые показатели

| Операция | Baseline (1 core) | Target (16 cores) | Speedup |
|----------|-------------------|-------------------|---------|
| Complex fit (100 NP) | 2s | 200ms | 10x |
| Ranking plot (100 NP) | 6 min | 40s | 9x |
| 1000 toy fits | 30 min | 2 min | 15x |
| Profile likelihood scan | 10 min | 1 min | 10x |

---

## Содержание

- [Sprint 2A.1: Rayon Integration](#sprint-2a1-rayon-integration-недели-13-14)
- [Sprint 2A.2: SIMD Optimization](#sprint-2a2-simd-optimization-недели-15-16)
- [Sprint 2A.3: Memory-Efficient Batching](#sprint-2a3-memory-efficient-batching-недели-17-18)
- [Sprint 2A.4: Cluster-Ready Parallelism](#sprint-2a4-cluster-ready-parallelism-недели-19-20)

---

## Sprint 2A.1: Rayon Integration (Недели 13-14)

### Epic 2A.1.1: Work-Stealing Parallelism

**Цель:** Интегрировать Rayon для автоматического распределения работы по ядрам.

---

#### Task 2A.1.1.1: Parallel NLL evaluation

**Priority:** P0
**Effort:** 4 часа
**Dependencies:** Phase 1 complete

**Files:**
- Modify: `crates/ns-compute/src/backend/cpu.rs`
- Create: `crates/ns-compute/src/parallel/mod.rs`
- Test: inline

**Acceptance Criteria:**
- [ ] NLL computation scales linearly до 8 cores
- [ ] Overhead < 1ms для small models
- [ ] Thread pool configurable

**Step 1: Write failing benchmark**

```rust
// benches/parallel_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ns_compute::backend::{CpuBackend, ComputeBackend};

fn parallel_nll_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_nll");

    // Different sizes
    for n_bins in [100, 1000, 10000, 100000].iter() {
        let observed: Vec<f64> = (0..*n_bins).map(|i| (i % 100) as f64 + 10.0).collect();
        let expected: Vec<f64> = (0..*n_bins).map(|i| (i % 100) as f64 + 10.0).collect();

        // Single-threaded baseline
        group.bench_with_input(
            BenchmarkId::new("single_thread", n_bins),
            n_bins,
            |b, _| {
                let backend = CpuBackend::with_threads(1);
                b.iter(|| backend.poisson_nll(&observed, &expected).unwrap())
            },
        );

        // Multi-threaded
        group.bench_with_input(
            BenchmarkId::new("multi_thread", n_bins),
            n_bins,
            |b, _| {
                let backend = CpuBackend::new(); // auto threads
                b.iter(|| backend.poisson_nll(&observed, &expected).unwrap())
            },
        );
    }

    group.finish();
}

criterion_group!(benches, parallel_nll_bench);
criterion_main!(benches);
```

**Step 2: Implement parallel NLL**

```rust
// crates/ns-compute/src/parallel/mod.rs
//! Parallel computation utilities

use rayon::prelude::*;
use ns_core::types::Float;

/// Threshold for switching to parallel execution
/// Below this, overhead outweighs benefits
pub const PARALLEL_THRESHOLD: usize = 1000;

/// Configure Rayon thread pool
pub fn configure_threads(num_threads: usize) {
    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok(); // Ignore if already configured
    }
}

/// Parallel Poisson NLL computation
///
/// Uses chunked parallel reduction for cache efficiency.
pub fn poisson_nll_parallel(observed: &[Float], expected: &[Float]) -> Float {
    assert_eq!(observed.len(), expected.len());
    let n = observed.len();

    if n < PARALLEL_THRESHOLD {
        // Sequential for small arrays
        return poisson_nll_sequential(observed, expected);
    }

    // Parallel reduction with chunks
    // Chunk size chosen for L1 cache efficiency (~32KB)
    let chunk_size = 4096;

    observed
        .par_chunks(chunk_size)
        .zip(expected.par_chunks(chunk_size))
        .map(|(obs_chunk, exp_chunk)| {
            poisson_nll_sequential(obs_chunk, exp_chunk)
        })
        .sum()
}

/// Deterministic Poisson NLL wrapper (required for strict parity tests).
///
/// If `deterministic=true`, forces a stable sequential sum to avoid
/// non-deterministic reduction order under Rayon work-stealing.
pub fn poisson_nll(observed: &[Float], expected: &[Float], deterministic: bool) -> Float {
    if deterministic {
        return poisson_nll_sequential(observed, expected);
    }
    poisson_nll_parallel(observed, expected)
}

/// Sequential NLL (for small arrays or within parallel chunks)
#[inline]
fn poisson_nll_sequential(observed: &[Float], expected: &[Float]) -> Float {
    observed
        .iter()
        .zip(expected.iter())
        .map(|(&n, &lam)| crate::poisson_nll(n, lam))
        .sum()
}

/// Parallel expected data computation
pub type ModifierFn = dyn Fn(&mut [Float], &[Float]) + Sync;

pub fn expected_data_parallel(
    nominals: &[Vec<Float>],
    modifiers: &[&ModifierFn],
    params: &[Float],
    n_bins: usize,
) -> Vec<Float> {
    if n_bins < PARALLEL_THRESHOLD {
        return expected_data_sequential(nominals, modifiers, params, n_bins);
    }

    // Parallel over bins
    (0..n_bins)
        .into_par_iter()
        .map(|bin_idx| {
            let mut total = 0.0;

            for (sample_idx, nominal) in nominals.iter().enumerate() {
                let mut value = nominal.get(bin_idx).copied().unwrap_or(0.0);

                // Apply modifiers for this sample
                for modifier in modifiers {
                    let mut bin_data = [value];
                    modifier(&mut bin_data, params);
                    value = bin_data[0];
                }

                total += value;
            }

            total
        })
        .collect()
}

fn expected_data_sequential(
    nominals: &[Vec<Float>],
    modifiers: &[&ModifierFn],
    params: &[Float],
    n_bins: usize,
) -> Vec<Float> {
    let mut result = vec![0.0; n_bins];

    for (sample_idx, nominal) in nominals.iter().enumerate() {
        let mut sample_data = nominal.clone();

        for modifier in modifiers {
            modifier(&mut sample_data, params);
        }

        for (i, &v) in sample_data.iter().enumerate() {
            if i < result.len() {
                result[i] += v;
            }
        }
    }

    result
}
```

**Step 3: Update CpuBackend to use parallel functions**

```diff
diff --git a/crates/ns-compute/src/backend/cpu.rs b/crates/ns-compute/src/backend/cpu.rs
@@
 use crate::parallel::{poisson_nll, configure_threads, PARALLEL_THRESHOLD};
 
 pub struct CpuBackend {
     num_threads: usize,
+    /// If true, force deterministic reductions for parity tests.
+    deterministic: bool,
 }
 
 impl CpuBackend {
     pub fn new() -> Self {
-        Self { num_threads: 0 }
+        Self { num_threads: 0, deterministic: false }
     }
 
     pub fn with_threads(num_threads: usize) -> Self {
         configure_threads(num_threads);
-        Self { num_threads }
+        Self { num_threads, deterministic: false }
     }
+
+    pub fn deterministic(mut self, deterministic: bool) -> Self {
+        self.deterministic = deterministic;
+        self
+    }
 }
 
 impl ComputeBackend for CpuBackend {
     fn poisson_nll(&self, observed: &[Float], expected: &[Float]) -> Result<Float> {
-        Ok(poisson_nll(observed, expected))
+        Ok(poisson_nll(observed, expected, self.deterministic))
     }
 }
```

**Step 4: Run benchmark**

```bash
cargo bench -p ns-compute --bench parallel_bench
# Expected: Multi-thread shows speedup for n_bins >= 1000
```

**Step 5: Commit**

```bash
git add crates/ns-compute/src/parallel/
git commit -m "feat(compute): add Rayon-based parallel NLL computation"
```

---

#### Task 2A.1.1.2: Parallel fit batch

**Priority:** P0
**Effort:** 4 часа
**Dependencies:** Task 2A.1.1.1

**Files:**
- Modify: `crates/ns-compute/src/backend/cpu.rs`
- Test: inline

**Acceptance Criteria:**
- [ ] 1000 fits use all available cores
- [ ] Memory usage stays bounded
- [ ] Progress reporting works

**Step 1: Write test**

```rust
#[test]
fn test_parallel_fit_batch() {
    let backend = CpuBackend::new();

    let model = create_test_model(10, 50); // 10 bins, 50 NP

    // 100 independent fits
    let init_params: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let mut params = vec![1.0; 50];
            params[0] = 1.0 + (i as f64) * 0.01; // Vary starting point
            params
        })
        .collect();

    let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); 50];

    let start = std::time::Instant::now();
    let results = backend.fit_batch(&model, &init_params, &bounds).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|r| r.converged));

    // Should be faster than 100 * single_fit_time
    // With 8 cores, expect ~8x speedup
    println!("100 fits in {:?}", elapsed);
}
```

**Step 2: Implement parallel fit_batch**

```rust
impl ComputeBackend for CpuBackend {
    fn fit_batch(
        &self,
        model: &ModelData,
        init_params_batch: &[Vec<Float>],
        bounds: &[(Float, Float)],
    ) -> Result<Vec<FitResultData>> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let total = init_params_batch.len();
        let completed = AtomicUsize::new(0);

        // Clone model for each thread (it's lightweight - just references)
        let results: Vec<Result<FitResultData>> = init_params_batch
            .par_iter()
            .map(|init| {
                let result = self.fit_single(model, init, bounds);

                // Progress tracking
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 10 == 0 || done == total {
                    tracing::debug!("Fit progress: {}/{}", done, total);
                }

                result
            })
            .collect();

        results.into_iter().collect()
    }
}
```

**Step 3: Commit**

```bash
git add crates/ns-compute/
git commit -m "feat(compute): parallel fit_batch with Rayon"
```

---

### Epic 2A.1.2: Parallel Gradient Computation

**Цель:** Параллельный расчёт градиентов для L-BFGS.

---

#### Task 2A.1.2.1: Parallel numerical gradient

**Priority:** P1
**Effort:** 3 часа
**Dependencies:** Task 2A.1.1.1

**Files:**
- Modify: `crates/ns-compute/src/backend/traits.rs`

**Step 1: Implement parallel gradient**

```rust
// In ComputeBackend trait default implementation

fn numerical_gradient(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
    use rayon::prelude::*;

    let h = 1e-7;
    let n = params.len();
    let f0 = self.nll(model, params)?;

    // Parallel over parameters
    let grad: Vec<Result<Float>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut params_plus = params.to_vec();
            params_plus[i] += h;
            let f_plus = self.nll(model, &params_plus)?;
            Ok((f_plus - f0) / h)
        })
        .collect();

    grad.into_iter().collect()
}

fn numerical_hessian_diagonal(&self, model: &ModelData, params: &[Float]) -> Result<Vec<Float>> {
    use rayon::prelude::*;

    let h = 1e-5;
    let n = params.len();
    let f0 = self.nll(model, params)?;

    // Parallel over parameters
    let hess_diag: Vec<Result<Float>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[i] += h;
            p_minus[i] -= h;

            let f_plus = self.nll(model, &p_plus)?;
            let f_minus = self.nll(model, &p_minus)?;

            Ok((f_plus - 2.0 * f0 + f_minus) / (h * h))
        })
        .collect();

    hess_diag.into_iter().collect()
}
```

**Step 2: Commit**

```bash
git add crates/ns-compute/
git commit -m "feat(compute): parallel numerical gradient and Hessian diagonal"
```

---

## Sprint 2A.2: SIMD Optimization (Недели 15-16)

### Epic 2A.2.1: SIMD Vectorization

**Цель:** Использовать SIMD инструкции для ускорения на одном ядре.

---

#### Task 2A.2.1.1: SIMD Poisson NLL

**Priority:** P1
**Effort:** 6 часов
**Dependencies:** Sprint 2A.1

**Files:**
- Create: `crates/ns-compute/src/simd/mod.rs`
- Modify: `crates/ns-compute/Cargo.toml`

**Acceptance Criteria:**
- [ ] 4x speedup на AVX2
- [ ] Fallback для non-SIMD CPUs
- [ ] Apple Silicon NEON support

**Step 1: Add SIMD dependency**

```toml
# Cargo.toml [workspace.dependencies]
# For SIMD on stable Rust (Rust 1.93+ baseline)
# NOTE: `std::simd` (`portable_simd`) is still unstable on stable Rust as of early 2026.
# Prefer stable `pulp` (runtime dispatch) or `std::arch` intrinsics behind `cfg`.

[features]
simd = []  # Enable SIMD optimizations
```

**Step 2: Implement SIMD NLL**

```rust
// crates/ns-compute/src/simd/mod.rs
//! SIMD-optimized computations
//!
//! Uses platform-specific SIMD when available:
//! - x86_64: AVX2, AVX-512
//! - aarch64: NEON

use ns_core::types::Float;

/// SIMD width for f64 operations
#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 4; // AVX2: 256-bit = 4 x f64

#[cfg(target_arch = "aarch64")]
const SIMD_WIDTH: usize = 2; // NEON: 128-bit = 2 x f64

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const SIMD_WIDTH: usize = 1; // Scalar fallback

/// Observed-only constant terms for Poisson NLL: `Σ ln Γ(n_i + 1)`.
///
/// These terms **do not depend on λ** and can be computed once per dataset.
#[inline]
fn poisson_const_terms(observed: &[Float]) -> Float {
    use statrs::function::gamma::ln_gamma;
    observed.iter().map(|&n| ln_gamma(n + 1.0)).sum()
}

/// SIMD-optimized Poisson NLL
///
/// Processes SIMD_WIDTH elements at a time.
///
/// Note: to keep the hot loop SIMD-friendly, we compute the variable part
/// `λ - n*ln(λ)` in SIMD and add the observed-only constants `ln Γ(n+1)`
/// outside the vector loop.
#[cfg(target_arch = "x86_64")]
pub fn poisson_nll_simd(observed: &[Float], expected: &[Float]) -> Float {
    use std::arch::x86_64::*;

    assert_eq!(observed.len(), expected.len());
    let n = observed.len();
    let const_terms = poisson_const_terms(observed);

    // Process chunks of 4 (AVX2)
    let chunks = n / 4;
    let remainder = n % 4;

    let mut total = 0.0f64;

    unsafe {
        let mut sum_vec = _mm256_setzero_pd();

        for i in 0..chunks {
            let offset = i * 4;

            // Load 4 observed and expected values
            let obs = _mm256_loadu_pd(observed.as_ptr().add(offset));
            let exp = _mm256_loadu_pd(expected.as_ptr().add(offset));

            // Compute ln(expected)
            // Note: _mm256_log_pd is not in std, need to use approximation or libm
            // For now, extract and compute scalar (can optimize with sleef or similar)
            let mut ln_exp = [0.0f64; 4];
            let exp_arr: [f64; 4] = std::mem::transmute(exp);
            for j in 0..4 {
                let lam = exp_arr[j];
                if lam <= 0.0 {
                    return Float::INFINITY;
                }
                ln_exp[j] = lam.ln();
            }
            let ln_exp_vec = _mm256_loadu_pd(ln_exp.as_ptr());

            // Variable part: λ - n*ln(λ)
            let obs_ln_exp = _mm256_mul_pd(obs, ln_exp_vec);
            let nll = _mm256_sub_pd(exp, obs_ln_exp);

            sum_vec = _mm256_add_pd(sum_vec, nll);
        }

        // Horizontal sum of sum_vec
        let sum_arr: [f64; 4] = std::mem::transmute(sum_vec);
        total = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
    }

    // Handle remainder
    for i in (n - remainder)..n {
        let lam = expected[i];
        if lam > 0.0 {
            total += lam - observed[i] * lam.ln();
        } else {
            return Float::INFINITY;
        }
    }

    total + const_terms
}

#[cfg(target_arch = "aarch64")]
pub fn poisson_nll_simd(observed: &[Float], expected: &[Float]) -> Float {
    use std::arch::aarch64::*;

    assert_eq!(observed.len(), expected.len());
    let n = observed.len();
    let const_terms = poisson_const_terms(observed);

    let chunks = n / 2;
    let remainder = n % 2;

    let mut total = 0.0f64;

    unsafe {
        let mut sum_vec = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let offset = i * 2;

            let obs = vld1q_f64(observed.as_ptr().add(offset));
            let exp = vld1q_f64(expected.as_ptr().add(offset));

            // Compute ln - need to extract for now
            let exp_arr: [f64; 2] = std::mem::transmute(exp);
            if exp_arr[0] <= 0.0 || exp_arr[1] <= 0.0 {
                return Float::INFINITY;
            }
            let ln_exp = vld1q_f64([exp_arr[0].ln(), exp_arr[1].ln()].as_ptr());

            // Variable part: λ - n*ln(λ)
            let obs_ln_exp = vmulq_f64(obs, ln_exp);
            let nll = vsubq_f64(exp, obs_ln_exp);

            sum_vec = vaddq_f64(sum_vec, nll);
        }

        // Horizontal sum
        total = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    }

    // Remainder
    for i in (n - remainder)..n {
        let lam = expected[i];
        if lam > 0.0 {
            total += lam - observed[i] * lam.ln();
        } else {
            return Float::INFINITY;
        }
    }

    total + const_terms
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn poisson_nll_simd(observed: &[Float], expected: &[Float]) -> Float {
    // Scalar fallback
    let const_terms = poisson_const_terms(observed);
    let var_terms: Float = observed
        .iter()
        .zip(expected.iter())
        .map(|(&n, &lam)| if lam > 0.0 { lam - n * lam.ln() } else { Float::INFINITY })
        .sum();

    const_terms + var_terms
}

/// Auto-vectorized alternative using iterators
/// Compiler may vectorize this with -C target-cpu=native
#[inline]
pub fn poisson_nll_autovec(observed: &[Float], expected: &[Float]) -> Float {
    let const_terms = poisson_const_terms(observed);
    let var_terms = observed.iter()
        .zip(expected.iter())
        .fold(0.0, |acc, (&n, &lam)| {
            if lam > 0.0 {
                acc + lam - n * lam.ln()
            } else {
                Float::INFINITY
            }
        })
        ;

    const_terms + var_terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_matches_scalar() {
        let observed: Vec<f64> = (0..1000).map(|i| (i % 100 + 10) as f64).collect();
        let expected: Vec<f64> = (0..1000).map(|i| (i % 100 + 10) as f64).collect();

        let simd_result = poisson_nll_simd(&observed, &expected);
        let scalar_result = poisson_nll_autovec(&observed, &expected);

        assert_relative_eq!(simd_result, scalar_result, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_performance() {
        let observed: Vec<f64> = (0..100000).map(|i| (i % 100 + 10) as f64).collect();
        let expected: Vec<f64> = (0..100000).map(|i| (i % 100 + 10) as f64).collect();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = poisson_nll_simd(&observed, &expected);
        }
        let simd_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = poisson_nll_autovec(&observed, &expected);
        }
        let scalar_time = start.elapsed();

        println!("SIMD: {:?}, Scalar: {:?}", simd_time, scalar_time);
        // SIMD should be faster (or equal if compiler auto-vectorizes well)
    }
}
```

**Step 3: Integrate with parallel module**

```rust
// crates/ns-compute/src/parallel/mod.rs
use crate::simd::poisson_nll_simd;

pub fn poisson_nll_parallel(observed: &[Float], expected: &[Float]) -> Float {
    assert_eq!(observed.len(), expected.len());
    let n = observed.len();

    if n < PARALLEL_THRESHOLD {
        // Use SIMD for small arrays (still benefit from vectorization)
        return poisson_nll_simd(observed, expected);
    }

    // Parallel chunks with SIMD within each chunk
    let chunk_size = 4096;

    observed
        .par_chunks(chunk_size)
        .zip(expected.par_chunks(chunk_size))
        .map(|(obs_chunk, exp_chunk)| {
            poisson_nll_simd(obs_chunk, exp_chunk)
        })
        .sum()
}
```

**Step 4: Commit**

```bash
git add crates/ns-compute/src/simd/
git commit -m "feat(compute): add SIMD-optimized Poisson NLL (AVX2/NEON)"
```

---

## Sprint 2A.3: Memory-Efficient Batching (Недели 17-18)

### Epic 2A.3.1: Streaming Batch Processing

**Цель:** Эффективная обработка больших батчей без OOM.

---

#### Task 2A.3.1.1: Chunked batch processing

**Priority:** P0
**Effort:** 4 часа
**Dependencies:** Sprint 2A.2

**Files:**
- Create: `crates/ns-compute/src/batch/mod.rs`

**Acceptance Criteria:**
- [ ] 10000 toy fits не вызывает OOM
- [ ] Memory usage bounded by chunk_size
- [ ] Results stream back progressively

**Step 1: Implement streaming batch**

```rust
// crates/ns-compute/src/batch/mod.rs
//! Memory-efficient batch processing

use ns_core::{Result, types::Float};
use super::backend::{ComputeBackend, ModelData, FitResultData};

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum fits to run in parallel
    /// Limited by memory (each fit needs model clone)
    pub chunk_size: usize,

    /// Enable streaming results (write to disk as completed)
    pub streaming: bool,

    /// Progress callback interval
    pub progress_interval: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            chunk_size: 100,  // Conservative default
            streaming: false,
            progress_interval: 10,
        }
    }
}

impl BatchConfig {
    /// Auto-configure based on available memory
    pub fn auto() -> Self {
        // Estimate memory per fit: ~model_size + ~1MB overhead
        let available_mb = get_available_memory_mb();
        let model_overhead_mb = 10; // Conservative estimate

        let chunk_size = (available_mb / model_overhead_mb).max(10).min(1000);

        Self {
            chunk_size,
            streaming: chunk_size < 100,
            progress_interval: (chunk_size / 10).max(1),
        }
    }
}

/// Run batch fits with memory management
pub fn run_batch_fits<F>(
    backend: &dyn ComputeBackend,
    model: &ModelData,
    init_params_batch: &[Vec<Float>],
    bounds: &[(Float, Float)],
    config: BatchConfig,
    mut progress_callback: F,
) -> Result<Vec<FitResultData>>
where
    F: FnMut(usize, usize),
{
    let total = init_params_batch.len();
    let mut all_results = Vec::with_capacity(total);

    // Process in chunks
    for (chunk_idx, chunk) in init_params_batch.chunks(config.chunk_size).enumerate() {
        let chunk_start = chunk_idx * config.chunk_size;

        // Run this chunk
        let chunk_results = backend.fit_batch(model, chunk, bounds)?;

        // Report progress
        let completed = chunk_start + chunk_results.len();
        progress_callback(completed, total);

        all_results.extend(chunk_results);

        // Optional: yield to allow GC / reduce memory pressure
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::yield_now();
    }

    Ok(all_results)
}

/// Streaming batch with disk-backed results
pub fn run_batch_fits_streaming<W, F>(
    backend: &dyn ComputeBackend,
    model: &ModelData,
    init_params_batch: &[Vec<Float>],
    bounds: &[(Float, Float)],
    config: BatchConfig,
    mut writer: W,
    mut progress_callback: F,
) -> Result<usize>
where
    W: std::io::Write,
    F: FnMut(usize, usize),
{
    let total = init_params_batch.len();
    let mut completed = 0;

    for chunk in init_params_batch.chunks(config.chunk_size) {
        let chunk_results = backend.fit_batch(model, chunk, bounds)?;

        // Write results immediately
        for result in &chunk_results {
            serde_json::to_writer(&mut writer, result)
                .map_err(|e| ns_core::Error::Io(std::io::Error::other(e)))?;
            writeln!(writer)?;
        }

        completed += chunk_results.len();
        progress_callback(completed, total);

        // Flush periodically
        if completed % 1000 == 0 {
            writer.flush()?;
        }
    }

    writer.flush()?;
    Ok(completed)
}

#[cfg(unix)]
fn get_available_memory_mb() -> usize {
    // Try to get from /proc/meminfo on Linux
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemAvailable:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        return kb / 1024; // Convert to MB
                    }
                }
            }
        }
    }
    8192 // Default 8GB assumption
}

#[cfg(not(unix))]
fn get_available_memory_mb() -> usize {
    8192 // Default assumption
}
```

**Step 2: Commit**

```bash
git add crates/ns-compute/src/batch/
git commit -m "feat(compute): add memory-efficient batch processing"
```

---

## Sprint 2A.4: Cluster-Ready Parallelism (Недели 19-20)

### Epic 2A.4.1: HTCondor/SLURM Integration

**Цель:** Подготовить NextStat для использования на научных кластерах.

---

#### Task 2A.4.1.1: Job array support

**Priority:** P1
**Effort:** 4 часа
**Dependencies:** Sprint 2A.3

**Files:**
- Create: `crates/ns-cli/src/commands/batch.rs`

**Acceptance Criteria:**
- [ ] `--job-id` и `--total-jobs` flags
- [ ] Автоматическое разбиение работы
- [ ] Результаты легко объединить

**Step 1: Implement job array CLI**

```rust
// crates/ns-cli/src/commands/batch.rs
//! Batch fitting commands for cluster execution

use clap::Args;
use std::path::PathBuf;
use ns_core::Result;

#[derive(Args, Debug)]
pub struct BatchFitArgs {
    /// Input workspace
    #[arg(short, long)]
    pub input: PathBuf,

    /// Number of toy fits to run
    #[arg(short = 'n', long, default_value = "1000")]
    pub n_toys: usize,

    /// Output directory for results
    #[arg(short, long)]
    pub output_dir: PathBuf,

    /// Random seed (for reproducibility)
    #[arg(long, default_value = "42")]
    pub seed: u64,

    // === Cluster job array support ===

    /// Current job ID (1-indexed, for HTCondor/SLURM)
    #[arg(long, env = "JOB_ID")]
    pub job_id: Option<usize>,

    /// Total number of jobs in array
    #[arg(long, env = "TOTAL_JOBS")]
    pub total_jobs: Option<usize>,

    /// Number of threads per job
    #[arg(long, default_value = "0")]
    pub threads: usize,
}

impl BatchFitArgs {
    /// Calculate which toys this job should run
    pub fn get_toy_range(&self) -> Result<(usize, usize)> {
        match (self.job_id, self.total_jobs) {
            (Some(job_id), Some(total_jobs)) => {
                if total_jobs == 0 {
                    return Err(ns_core::Error::InvalidParameter(
                        "TOTAL_JOBS must be > 0".to_string(),
                    ));
                }
                if job_id == 0 || job_id > total_jobs {
                    return Err(ns_core::Error::InvalidParameter(format!(
                        "JOB_ID must be in 1..={total_jobs}, got {job_id}"
                    )));
                }

                // Distribute toys across jobs
                let toys_per_job = self.n_toys / total_jobs;
                let remainder = self.n_toys % total_jobs;

                let start = if job_id <= remainder {
                    (job_id - 1) * (toys_per_job + 1)
                } else {
                    remainder * (toys_per_job + 1) + (job_id - 1 - remainder) * toys_per_job
                };

                let count = if job_id <= remainder {
                    toys_per_job + 1
                } else {
                    toys_per_job
                };

                Ok((start, count))
            }
            _ => Ok((0, self.n_toys)), // Single job, all toys
        }
    }

    /// Get output file path for this job
    pub fn get_output_path(&self) -> PathBuf {
        let filename = match self.job_id {
            Some(id) => format!("toys_{:04}.jsonl", id),
            None => "toys.jsonl".to_string(),
        };
        self.output_dir.join(filename)
    }

    /// Get seed for this job (ensuring reproducibility)
    pub fn get_seed(&self) -> u64 {
        match self.job_id {
            Some(id) => self.seed.wrapping_add(id as u64 * 1000),
            None => self.seed,
        }
    }
}

pub fn run_batch_fit(args: BatchFitArgs) -> Result<()> {
    use ns_compute::backend::{create_backend, BackendConfig};
    use ns_compute::batch::{run_batch_fits_streaming, BatchConfig};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Configure threads
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let backend = create_backend(BackendConfig::Cpu)?;

    // Load model
    let json = std::fs::read_to_string(&args.input)?;
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?;
    let model_data = model.to_model_data();

    // Get range for this job
    let (start, count) = args.get_toy_range()?;
    tracing::info!("Job running toys {} to {} (count: {})", start, start + count, count);

    // Generate toy initial parameters
    let mut rng = ChaCha8Rng::seed_from_u64(args.get_seed());
    let init_params: Vec<Vec<f64>> = (0..count)
        .map(|_| generate_toy_init(&model_data, &mut rng))
        .collect();

    let bounds = model.get_bounds();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Run with streaming output
    let output_path = args.get_output_path();
    let file = std::fs::File::create(&output_path)?;
    let writer = std::io::BufWriter::new(file);

    let batch_config = BatchConfig::auto();

    let completed = run_batch_fits_streaming(
        backend.as_ref(),
        &model_data,
        &init_params,
        &bounds,
        batch_config,
        writer,
        |done, total| {
            if done % 100 == 0 || done == total {
                tracing::info!("Progress: {}/{}", done, total);
            }
        },
    )?;

    tracing::info!("Completed {} fits, written to {:?}", completed, output_path);

    Ok(())
}

fn generate_toy_init(model: &ns_compute::backend::ModelData, rng: &mut impl rand::Rng) -> Vec<f64> {
    use rand::distributions::{Distribution, Uniform};

    let n = model.n_params;
    let mut params = vec![1.0; n];

    // Slightly randomize starting point for robustness
    for i in 0..n {
        let noise = Uniform::new(-0.1, 0.1);
        params[i] += noise.sample(rng);
    }

    params
}
```

**Step 2: Add merge command**

```rust
// crates/ns-cli/src/commands/merge.rs
//! Merge results from job array

use clap::Args;
use std::path::PathBuf;
use ns_core::Result;

#[derive(Args, Debug)]
pub struct MergeArgs {
    /// Directory containing result files
    #[arg(short, long)]
    pub input_dir: PathBuf,

    /// Output merged file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Pattern for input files
    #[arg(long, default_value = "toys_*.jsonl")]
    pub pattern: String,
}

pub fn run_merge(args: MergeArgs) -> Result<()> {
    use glob::glob;

    let pattern = args.input_dir.join(&args.pattern);
    let pattern_str = pattern.to_str().ok_or_else(|| {
        ns_core::Error::Io(std::io::Error::other("Invalid pattern"))
    })?;

    let mut output = std::fs::File::create(&args.output)?;
    let mut total_count = 0;

    for entry in glob(pattern_str).map_err(|e| ns_core::Error::Io(std::io::Error::other(e)))? {
        let path = entry.map_err(|e| ns_core::Error::Io(std::io::Error::other(e)))?;
        tracing::info!("Merging {:?}", path);

        let content = std::fs::read_to_string(&path)?;
        let count = content.lines().count();
        total_count += count;

        std::io::Write::write_all(&mut output, content.as_bytes())?;
    }

    tracing::info!("Merged {} total results to {:?}", total_count, args.output);

    Ok(())
}
```

**Step 3: Example HTCondor submit file**

```bash
# scripts/condor_toys.sub
# HTCondor submit file for toy fits

executable = nextstat
arguments = batch-fit --input $(workspace) --n-toys $(n_toys) --output-dir $(output_dir) --job-id $(Process) --total-jobs $(n_jobs)

log = logs/toys_$(Cluster)_$(Process).log
output = logs/toys_$(Cluster)_$(Process).out
error = logs/toys_$(Cluster)_$(Process).err

request_cpus = 4
request_memory = 4GB

queue $(n_jobs)
```

```bash
# Usage:
# condor_submit condor_toys.sub workspace=model.json n_toys=10000 output_dir=results n_jobs=100
```

**Step 4: Example SLURM script**

```bash
#!/bin/bash
# scripts/slurm_toys.sh
#SBATCH --job-name=nextstat-toys
#SBATCH --array=1-100
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1:00:00

export JOB_ID=$SLURM_ARRAY_TASK_ID
export TOTAL_JOBS=$SLURM_ARRAY_TASK_COUNT

nextstat batch-fit \
    --input $WORKSPACE \
    --n-toys $N_TOYS \
    --output-dir $OUTPUT_DIR \
    --threads $SLURM_CPUS_PER_TASK
```

**Step 5: Commit**

```bash
git add crates/ns-cli/src/commands/ scripts/
git commit -m "feat(cli): add cluster job array support (HTCondor/SLURM)"
```

---

## Performance Summary

### Expected Scaling (16-core CPU)

| Operation | 1 core | 4 cores | 16 cores | Efficiency |
|-----------|--------|---------|----------|------------|
| NLL (10K bins) | 1.0ms | 0.3ms | 0.1ms | 62% |
| Fit (100 NP) | 2s | 0.6s | 0.2s | 62% |
| 1000 toys | 30min | 8min | 2min | 94% |
| Ranking (100 NP) | 6min | 1.5min | 30s | 75% |

### Memory Usage

| Operation | Per-fit | Batch 1000 | Peak |
|-----------|---------|------------|------|
| Simple model | 1MB | 100MB | 200MB |
| Complex model | 10MB | 1GB (chunked) | 500MB |

---

## Критерии завершения

### Exit Criteria

Phase 2A CPU parallelism завершена когда:

1. [ ] Rayon integration проходит все тесты
2. [ ] SIMD дает ≥2x speedup на single core
3. [ ] 1000 fits scales to ≥10x speedup на 16 cores
4. [ ] HTCondor/SLURM job arrays работают
5. [ ] Memory usage bounded для 10K fits
6. [ ] Benchmarks documented

---

*Следующая секция: [Phase 2B: Autodiff & Optimizers](./phase-2b-autodiff.md) → затем [Phase 2C: GPU Backends](./phase-2c-gpu-backends.md) (опционально)*
