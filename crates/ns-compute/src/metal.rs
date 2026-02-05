//! Metal compute backend (macOS).
//!
//! Phase 2C (optional). This module is feature-gated behind `metal`.
//!
//! Current status: stub implementation that compiles but returns
//! `Error::NotImplemented`.

use ns_core::{ComputeBackend, Error, Result};

/// Metal backend (stub).
pub struct MetalBackend;

impl MetalBackend {
    /// Create a new Metal backend (stub).
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for MetalBackend {
    fn nll(&self, _params: &[f64]) -> Result<f64> {
        Err(Error::NotImplemented("Metal backend".to_string()))
    }

    fn gradient(&self, _params: &[f64]) -> Result<Vec<f64>> {
        Err(Error::NotImplemented("Metal backend gradient".to_string()))
    }

    fn hessian(&self, _params: &[f64]) -> Result<Vec<Vec<f64>>> {
        Err(Error::NotImplemented("Metal backend hessian".to_string()))
    }

    fn name(&self) -> &str {
        "Metal"
    }
}
