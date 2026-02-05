//! CUDA compute backend (NVIDIA).
//!
//! Phase 2C (optional). This module is feature-gated behind `cuda`.
//!
//! Current status: stub implementation that compiles but returns
//! `Error::NotImplemented`.

use ns_core::{ComputeBackend, Error, Result};

/// CUDA backend (stub).
pub struct CudaBackend;

impl CudaBackend {
    /// Create a new CUDA backend (stub).
    pub fn new() -> Self {
        Self
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CudaBackend {
    fn nll(&self, _params: &[f64]) -> Result<f64> {
        Err(Error::NotImplemented("CUDA backend".to_string()))
    }

    fn gradient(&self, _params: &[f64]) -> Result<Vec<f64>> {
        Err(Error::NotImplemented("CUDA backend gradient".to_string()))
    }

    fn hessian(&self, _params: &[f64]) -> Result<Vec<Vec<f64>>> {
        Err(Error::NotImplemented("CUDA backend hessian".to_string()))
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}
