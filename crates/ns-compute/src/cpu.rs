//! CPU compute backend
//!
//! Priority P0 - this is the primary backend that MUST work everywhere.
//!
//! Features:
//! - Rayon for work-stealing parallelism
//! - SIMD via portable_simd (future)
//! - Memory-efficient batching
//! - HTCondor/SLURM job array support

use ns_core::{ComputeBackend, Result};

/// CPU compute backend using Rayon for parallelism
pub struct CpuBackend {
    /// Number of threads (0 = automatic)
    pub n_threads: usize,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self { n_threads: 0 }
    }

    /// Create a CPU backend with specified thread count
    pub fn with_threads(n_threads: usize) -> Self {
        Self { n_threads }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn nll(&self, _params: &[f64]) -> Result<f64> {
        Err(ns_core::Error::NotImplemented(
            "CpuBackend::nll() is not implemented. Use PreparedModel::nll() from ns-translate instead.".into()
        ))
    }

    fn gradient(&self, _params: &[f64]) -> Result<Vec<f64>> {
        Err(ns_core::Error::NotImplemented(
            "CpuBackend::gradient() is not implemented. Use ns-ad tape gradient instead.".into(),
        ))
    }

    fn hessian(&self, _params: &[f64]) -> Result<Vec<Vec<f64>>> {
        Err(ns_core::Error::NotImplemented(
            "CpuBackend::hessian() is not implemented. Use compute_hessian() from ns-inference instead.".into()
        ))
    }

    fn name(&self) -> &str {
        "CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "CPU");
    }

    #[test]
    fn test_cpu_backend_with_threads() {
        let backend = CpuBackend::with_threads(4);
        assert_eq!(backend.n_threads, 4);
    }
}
