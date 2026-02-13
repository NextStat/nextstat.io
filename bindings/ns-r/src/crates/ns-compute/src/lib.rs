//! # ns-compute
//!
//! Compute backends for NextStat.
//!
//! This crate provides:
//! - **CPU backend** (Rayon + SIMD) - implements `ns_core::ComputeBackend` (Priority P0)
//! - **GPU accelerator modules** (Metal/CUDA) - feature-gated, used via `ns-inference` GPU sessions
//!
//! Note: the GPU modules do **not** currently expose a generic `ComputeBackend`
//! implementation. GPU acceleration is provided by dedicated accelerator types
//! (e.g. `cuda_batch`, `metal_batch`, differentiable sessions) with fused-kernel
//! APIs tailored to their workloads.
//!
//! ## Architecture
//!
//! High-level inference code (`ns-inference`) depends on core abstractions from
//! `ns-core` and uses session/accelerator APIs for GPU execution. This allows:
//! - Testing with the simple CPU backend
//! - Optional GPU acceleration via feature-gated accelerators
//! - Easy addition of new accelerators/backends (TPU, etc.)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod cpu;
/// Device-agnostic GPU accelerator abstraction (always available).
pub mod gpu_accel;
pub mod simd;
/// GPU data types for unbinned likelihood kernels (always available).
pub mod unbinned_types;

pub use cpu::CpuBackend;

/// Re-export cudarc driver types used in public APIs (CudaContext, CudaStream, CudaSlice).
#[cfg(feature = "cuda")]
pub use cudarc::driver as cuda_driver;

/// Re-export core Metal types so downstream crates (ns-inference, ns-cli) can reference
/// `metal_rs::Buffer`, `metal_rs::Device`, etc. without adding a direct `metal` dependency.
#[cfg(feature = "metal")]
pub mod metal_rs {
    pub use metal::Buffer;
    pub use metal::CommandQueue;
    pub use metal::Device;
    pub use metal::MTLResourceOptions;
}

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

/// GPU data types shared between Rust and CUDA (`#[repr(C)]`).
///
/// Always available (no feature gate) so that ns-translate can serialize
/// `GpuModelData` without requiring CUDA at compile time.
pub mod cuda_types;

/// GPU data types for Metal kernels (f32 precision, `#[repr(C)]`).
///
/// Always available (no feature gate) so that ns-translate can serialize
/// `MetalModelData` without requiring Metal at compile time.
pub mod metal_types;

/// CUDA batch NLL+gradient accelerator (requires `cuda` feature + NVIDIA GPU at runtime).
#[cfg(feature = "cuda")]
pub mod cuda_batch;

/// CUDA unbinned (event-level) NLL+gradient accelerator (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_unbinned;

/// CUDA unbinned (event-level) batch toy NLL+gradient accelerator (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_unbinned_batch;

/// CUDA unbinned (event-level) toy sampling accelerator (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_unbinned_toy;

/// CUDA helper kernel: per-event WeightSys morphing for unbinned workflows (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_unbinned_weight_sys;

/// CUDA NLL reduction for externally-computed log-prob (flow PDFs, requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_flow_nll;

/// CUDA batch toy NLL accelerator for flow PDFs (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_flow_batch;

/// Direct Parquet → CUDA GPU upload (SoA f64 → CudaSlice, requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod cuda_parquet;

/// CUDA differentiable NLL accelerator for PyTorch zero-copy integration.
#[cfg(feature = "cuda")]
pub mod differentiable;

/// Metal batch NLL+gradient accelerator (requires `metal` feature + Apple Silicon at runtime).
#[cfg(feature = "metal")]
pub mod metal_batch;

/// Metal unbinned (event-level) NLL+gradient accelerator (requires `metal` feature).
#[cfg(feature = "metal")]
pub mod metal_unbinned;

/// Metal unbinned (event-level) batch toy NLL+gradient accelerator (requires `metal` feature).
#[cfg(feature = "metal")]
pub mod metal_unbinned_batch;

/// Metal unbinned (event-level) toy sampling accelerator (requires `metal` feature).
#[cfg(feature = "metal")]
pub mod metal_unbinned_toy;

/// Metal differentiable NLL accelerator for profiled fitting (requires `metal` feature).
#[cfg(feature = "metal")]
pub mod metal_differentiable;

/// Direct Parquet → Metal GPU upload (SoA f64 → f32 MTLBuffer, requires `metal` feature).
#[cfg(feature = "metal")]
pub mod metal_parquet;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub mod accelerate;

#[cfg(all(feature = "accelerate", not(target_os = "macos")))]
/// Stub module for non-macOS targets.
///
/// The real Apple Accelerate backend (vDSP/vForce) is only available on macOS.
/// On other platforms we keep the same API surface but fall back to the pure
/// Rust SIMD/scalar implementations.
pub mod accelerate {
    /// Compute Poisson NLL using the pure Rust SIMD backend (fallback).
    pub fn poisson_nll_accelerate(
        expected: &[f64],
        observed: &[f64],
        ln_factorials: &[f64],
        obs_mask: &[f64],
    ) -> f64 {
        crate::simd::poisson_nll_simd(expected, observed, ln_factorials, obs_mask)
    }

    /// Compute Poisson NLL for a batch of toy experiments (fallback).
    pub fn batch_poisson_nll_accelerate(
        expected_flat: &[f64],
        observed_flat: &[f64],
        ln_factorials_flat: &[f64],
        obs_mask_flat: &[f64],
        n_bins: usize,
        n_toys: usize,
    ) -> Vec<f64> {
        assert_eq!(expected_flat.len(), n_toys * n_bins);

        let per_toy_obs = observed_flat.len() == n_toys * n_bins;
        if !per_toy_obs {
            assert_eq!(observed_flat.len(), n_bins);
            assert_eq!(ln_factorials_flat.len(), n_bins);
            assert_eq!(obs_mask_flat.len(), n_bins);
        } else {
            assert_eq!(ln_factorials_flat.len(), n_toys * n_bins);
            assert_eq!(obs_mask_flat.len(), n_toys * n_bins);
        }

        let mut results = Vec::with_capacity(n_toys);
        for toy_idx in 0..n_toys {
            let offset = toy_idx * n_bins;
            let exp = &expected_flat[offset..offset + n_bins];

            if per_toy_obs {
                results.push(crate::simd::poisson_nll_simd(
                    exp,
                    &observed_flat[offset..offset + n_bins],
                    &ln_factorials_flat[offset..offset + n_bins],
                    &obs_mask_flat[offset..offset + n_bins],
                ));
            } else {
                results.push(crate::simd::poisson_nll_simd(
                    exp,
                    observed_flat,
                    ln_factorials_flat,
                    obs_mask_flat,
                ));
            }
        }

        results
    }

    /// Compute Poisson NLL with Kahan compensated summation (fallback).
    pub fn poisson_nll_accelerate_kahan(
        expected: &[f64],
        observed: &[f64],
        ln_factorials: &[f64],
        obs_mask: &[f64],
    ) -> f64 {
        crate::simd::poisson_nll_simd_kahan(expected, observed, ln_factorials, obs_mask)
    }

    /// Clamp a vector of expected values to `[floor, +inf)` in-place (fallback).
    pub fn clamp_expected_inplace(expected: &mut [f64], floor: f64) {
        for v in expected {
            if *v < floor {
                *v = floor;
            }
        }
    }

    /// Returns false on non-macOS targets.
    pub fn is_available() -> bool {
        false
    }
}

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Evaluation mode for NLL computation.
///
/// Controls the trade-off between numerical precision and speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// Maximum speed: SIMD/Accelerate, naive summation, multi-threaded.
    /// Default mode. Results may vary slightly between runs due to
    /// non-deterministic thread scheduling.
    Fast = 0,
    /// Maximum precision: Kahan summation, Accelerate disabled, deterministic.
    /// Results are bit-exact between runs. Used for pyhf parity validation.
    Parity = 1,
}

static EVAL_MODE: AtomicU8 = AtomicU8::new(0); // 0 = Fast

/// Set the process-wide evaluation mode.
///
/// - `EvalMode::Parity`: Kahan summation, Accelerate disabled, single-thread recommended.
/// - `EvalMode::Fast`: Naive summation, Accelerate/SIMD enabled (default).
///
/// When parity mode is activated, it also disables Accelerate automatically.
pub fn set_eval_mode(mode: EvalMode) {
    EVAL_MODE.store(mode as u8, Ordering::Relaxed);
    if mode == EvalMode::Parity {
        set_accelerate_enabled(false);
    }
}

/// Get the current evaluation mode.
pub fn eval_mode() -> EvalMode {
    match EVAL_MODE.load(Ordering::Relaxed) {
        1 => EvalMode::Parity,
        _ => EvalMode::Fast,
    }
}

/// Programmatic Accelerate disable flag.
///
/// When set to `true`, `accelerate_enabled()` returns `false` regardless of
/// compile-time feature or env var. Used by deterministic mode (`--threads 1`)
/// to ensure bit-exact parity with the SIMD/scalar path.
static ACCELERATE_DISABLED: AtomicBool = AtomicBool::new(false);

/// Programmatically disable the Apple Accelerate fast-path.
///
/// This is called automatically when deterministic mode is active (`--threads 1`).
/// The effect is process-wide and persists until `set_accelerate_enabled(true)` is called.
pub fn set_accelerate_enabled(enabled: bool) {
    ACCELERATE_DISABLED.store(!enabled, Ordering::Relaxed);
}

/// Returns true if the Apple Accelerate fast-path is compiled in *and* enabled at runtime.
///
/// Three-layer gate (all must pass):
/// 1. **Compile-time**: requires `--features accelerate` on macOS.
/// 2. **Programmatic**: `set_accelerate_enabled(false)` disables (used by deterministic mode).
/// 3. **Env var**: `NEXTSTAT_DISABLE_ACCELERATE=1` disables (user override).
pub fn accelerate_enabled() -> bool {
    if !cfg!(all(feature = "accelerate", target_os = "macos")) {
        return false;
    }
    if ACCELERATE_DISABLED.load(Ordering::Relaxed) {
        return false;
    }
    std::env::var_os("NEXTSTAT_DISABLE_ACCELERATE").is_none()
}

#[cfg(test)]
mod runtime_tests {
    use super::{ACCELERATE_DISABLED, accelerate_enabled, set_accelerate_enabled};
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn accelerate_disabled_when_feature_off() {
        // This test runs in the default feature set in CI and ensures the
        // runtime flag cannot accidentally enable Accelerate without the feature.
        #[cfg(not(feature = "accelerate"))]
        assert!(!accelerate_enabled());
    }

    #[test]
    fn programmatic_disable_overrides_feature() {
        let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");

        // Reset to default state
        ACCELERATE_DISABLED.store(false, Ordering::Relaxed);

        // Programmatic disable should work regardless of feature flag
        set_accelerate_enabled(false);
        assert!(!accelerate_enabled());

        // Re-enable
        set_accelerate_enabled(true);

        // On non-accelerate builds, still disabled (feature gate)
        #[cfg(not(feature = "accelerate"))]
        assert!(!accelerate_enabled());
    }

    #[test]
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    fn accelerate_respects_env_var() {
        let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");

        // Reset programmatic flag
        ACCELERATE_DISABLED.store(false, Ordering::Relaxed);

        // NOTE: `std::env::{set_var, remove_var}` are `unsafe` on modern Rust
        // because the process environment is a shared global.
        unsafe {
            std::env::remove_var("NEXTSTAT_DISABLE_ACCELERATE");
        }
        assert!(accelerate_enabled());

        unsafe {
            std::env::set_var("NEXTSTAT_DISABLE_ACCELERATE", "1");
        }
        assert!(!accelerate_enabled());
        unsafe {
            std::env::remove_var("NEXTSTAT_DISABLE_ACCELERATE");
        }
    }

    #[test]
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    fn programmatic_disable_overrides_even_with_feature() {
        let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");

        // Ensure env var is not set
        unsafe {
            std::env::remove_var("NEXTSTAT_DISABLE_ACCELERATE");
        }
        ACCELERATE_DISABLED.store(false, Ordering::Relaxed);
        assert!(accelerate_enabled());

        // Programmatic disable
        set_accelerate_enabled(false);
        assert!(!accelerate_enabled());

        // Restore
        set_accelerate_enabled(true);
        assert!(accelerate_enabled());
    }
}
