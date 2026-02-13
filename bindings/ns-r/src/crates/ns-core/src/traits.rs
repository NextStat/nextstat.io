//! Core traits for NextStat
//!
//! This module defines the trait-based architecture that enables
//! dependency inversion: high-level inference logic does not depend
//! on low-level compute backends (CPU/Metal/CUDA).

use crate::Result;

/// Prepared negative log-likelihood evaluator.
///
/// Some models can precompute/caches constants (observations, constraints, etc.)
/// to speed up repeated NLL evaluations. The inference layer should prefer
/// `prepared().nll(...)` when available.
pub trait PreparedNll: Send + Sync {
    /// Compute negative log-likelihood at `params`.
    fn nll(&self, params: &[f64]) -> Result<f64>;
}

/// Default prepared wrapper that forwards to the model's `nll`.
#[derive(Debug, Clone, Copy)]
pub struct PreparedModelRef<'a, M: LogDensityModel + ?Sized> {
    model: &'a M,
}

impl<'a, M: LogDensityModel + ?Sized> PreparedModelRef<'a, M> {
    /// Create a new prepared wrapper that forwards `nll` to the model.
    pub fn new(model: &'a M) -> Self {
        Self { model }
    }
}

impl<'a, M: LogDensityModel + ?Sized> PreparedNll for PreparedModelRef<'a, M> {
    fn nll(&self, params: &[f64]) -> Result<f64> {
        self.model.nll(params)
    }
}

/// Universal model interface for inference beyond HistFactory.
///
/// This trait is the Phase 5 foundation: generic MLE/MAP and NUTS/HMC should depend
/// on this interface, not on concrete model types.
pub trait LogDensityModel: Send + Sync {
    /// Prepared evaluator type (can cache constants).
    ///
    /// If a model has nothing to cache, use:
    /// `type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;`
    type Prepared<'a>: PreparedNll + 'a
    where
        Self: 'a;

    /// Number of parameters.
    fn dim(&self) -> usize;

    /// Parameter names (stable order).
    fn parameter_names(&self) -> Vec<String>;

    /// Parameter bounds (min, max) (stable order).
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;

    /// Suggested initial values (stable order).
    fn parameter_init(&self) -> Vec<f64>;

    /// Negative log-likelihood.
    fn nll(&self, params: &[f64]) -> Result<f64>;

    /// Gradient of NLL.
    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>>;

    /// Create a prepared evaluator.
    fn prepared(&self) -> Self::Prepared<'_>;

    /// Hint: prefer evaluating NLL and gradient together in one call.
    ///
    /// Some models can compute `nll` and `grad_nll` in a single fused pass (e.g. unbinned
    /// event-level likelihoods with per-event reductions). When this returns `true`,
    /// the optimizer may call [`Self::nll_grad_prepared`] from the cost function and
    /// cache the gradient for the subsequent gradient callback.
    ///
    /// Default is `false` for conservative performance (avoid computing gradients on
    /// cost-only evaluations).
    fn prefer_fused_eval_grad(&self) -> bool {
        false
    }

    /// Compute NLL and gradient, optionally using prepared caches.
    ///
    /// The default implementation uses `prepared.nll(params)` and `self.grad_nll(params)`.
    /// Models that can compute both in a fused pass should override this.
    fn nll_grad_prepared(
        &self,
        prepared: &Self::Prepared<'_>,
        params: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        Ok((prepared.nll(params)?, self.grad_nll(params)?))
    }
}

/// Optional extension: parameter-of-interest (POI) index.
///
/// This is primarily used by HEP workflows (profile likelihood, CLs). For general
/// models, return `None`.
pub trait PoiModel: Send + Sync {
    /// Index of POI in the model's parameter order.
    fn poi_index(&self) -> Option<usize>;
}

/// Optional extension: create a copy of the model with one parameter fixed.
///
/// This is used by profile likelihood / hypotest style workflows.
pub trait FixedParamModel: Sized + Send + Sync {
    /// Return a copy with parameter `param_idx` fixed at `value` (e.g., bounds clamped).
    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self;
}

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

/// Statistical model trait.
///
/// **Deprecated since 0.9.0**: Use [`LogDensityModel`] instead, which provides
/// the same parameter metadata plus NLL/gradient evaluation.
#[deprecated(since = "0.9.0", note = "Use LogDensityModel instead")]
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
    struct DummyModel;

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

    impl LogDensityModel for DummyModel {
        type Prepared<'a>
            = PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            2
        }

        fn parameter_names(&self) -> Vec<String> {
            vec!["a".to_string(), "b".to_string()]
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(0.0, 1.0), (-1.0, 1.0)]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.5, 0.0]
        }

        fn nll(&self, params: &[f64]) -> Result<f64> {
            Ok(params.iter().map(|&x| x * x).sum())
        }

        fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
            Ok(params.iter().map(|&x| 2.0 * x).collect())
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef { model: self }
        }
    }

    #[test]
    fn test_log_density_model_prepared_default() {
        let m = DummyModel;
        let p = m.prepared();
        assert!((p.nll(&[2.0, 3.0]).unwrap() - 13.0).abs() < 1e-12);
    }
}
