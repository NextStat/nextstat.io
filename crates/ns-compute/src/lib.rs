//! # ns-compute
//!
//! Compute backends for NextStat.
//!
//! This crate provides implementations of the `ComputeBackend` trait:
//! - **CPU backend** (Rayon + SIMD) - Priority P0, always available
//! - **Metal backend** (macOS) - Priority P1, feature-gated
//! - **CUDA backend** (NVIDIA) - Priority P1, feature-gated
//!
//! ## Architecture
//!
//! High-level inference code (ns-inference) depends on the `ComputeBackend`
//! trait from ns-core, NOT on concrete implementations. This allows:
//! - Testing with simple CPU backend
//! - Optional GPU acceleration
//! - Easy addition of new backends (TPU, etc.)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod cpu;
pub mod simd;

pub use cpu::CpuBackend;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

/// GPU data types shared between Rust and CUDA (`#[repr(C)]`).
///
/// Always available (no feature gate) so that ns-translate can serialize
/// `GpuModelData` without requiring CUDA at compile time.
pub mod cuda_types;

/// CUDA batch NLL+gradient accelerator (requires `cuda` feature + NVIDIA GPU at runtime).
#[cfg(feature = "cuda")]
pub mod cuda_batch;

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

/// Returns true if the Apple Accelerate fast-path is compiled in *and* enabled at runtime.
///
/// Notes:
/// - Compile-time gate: requires `--features accelerate` on macOS.
/// - Runtime gate: set `NEXTSTAT_DISABLE_ACCELERATE=1` to force the pure Rust SIMD/scalar path
///   (useful for strict parity/determinism baselines).
pub fn accelerate_enabled() -> bool {
    if !cfg!(all(feature = "accelerate", target_os = "macos")) {
        return false;
    }
    std::env::var_os("NEXTSTAT_DISABLE_ACCELERATE").is_none()
}

#[cfg(test)]
mod runtime_tests {
    use super::accelerate_enabled;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn accelerate_disabled_when_feature_off() {
        // This test runs in the default feature set in CI and ensures the
        // runtime flag cannot accidentally enable Accelerate without the feature.
        #[cfg(not(feature = "accelerate"))]
        assert!(!accelerate_enabled());
    }

    #[test]
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    fn accelerate_respects_env_var() {
        let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");

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
}
