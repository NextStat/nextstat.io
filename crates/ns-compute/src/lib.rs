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

#[cfg(feature = "accelerate")]
pub mod accelerate;

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
        std::env::remove_var("NEXTSTAT_DISABLE_ACCELERATE");
        assert!(accelerate_enabled());

        std::env::set_var("NEXTSTAT_DISABLE_ACCELERATE", "1");
        assert!(!accelerate_enabled());
        std::env::remove_var("NEXTSTAT_DISABLE_ACCELERATE");
    }
}
