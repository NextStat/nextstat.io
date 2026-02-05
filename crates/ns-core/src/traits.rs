//! Core traits for NextStat
//!
//! This module defines the trait-based architecture that enables
//! dependency inversion: high-level inference logic does not depend
//! on low-level compute backends (CPU/Metal/CUDA).

use crate::Result;

/// Compute backend trait - abstraction over CPU/Metal/CUDA
///
/// This trait enables clean architecture where inference logic
/// (ns-inference) does not depend on concrete backend implementations.
pub trait ComputeBackend: Send + Sync {
    /// Compute negative log-likelihood
    fn nll(&self, params: &[f64]) -> Result<f64>;

    /// Compute gradient of NLL
    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>>;

    /// Compute Hessian matrix
    fn hessian(&self, params: &[f64]) -> Result<Vec<Vec<f64>>>;

    /// Backend name (e.g., "CPU", "Metal", "CUDA")
    fn name(&self) -> &str;
}

/// Statistical model trait
pub trait Model: Send + Sync {
    /// Number of parameters
    fn n_parameters(&self) -> usize;

    /// Parameter names
    fn parameter_names(&self) -> Vec<String>;

    /// Parameter bounds (min, max)
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBackend;

    impl ComputeBackend for DummyBackend {
        fn nll(&self, _params: &[f64]) -> Result<f64> {
            Ok(0.0)
        }

        fn gradient(&self, _params: &[f64]) -> Result<Vec<f64>> {
            Ok(vec![])
        }

        fn hessian(&self, _params: &[f64]) -> Result<Vec<Vec<f64>>> {
            Ok(vec![])
        }

        fn name(&self) -> &str {
            "Dummy"
        }
    }

    #[test]
    fn test_dummy_backend() {
        let backend = DummyBackend;
        assert_eq!(backend.name(), "Dummy");
        assert!(backend.nll(&[]).is_ok());
    }
}
