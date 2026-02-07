//! Batch toy fitting with optional Accelerate-optimized NLL.
//!
//! This module provides batch fit functions that dispatch to the Accelerate
//! backend (Apple vDSP/vForce) when available, or fall back to the standard
//! SIMD backend.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ Rayon par_iter over toys                    │
//! │  ┌─────────────────────────────────────┐    │
//! │  │ Per-toy L-BFGS-B (CPU)              │    │
//! │  │  ├── NLL: PreparedModel (SIMD or    │    │
//! │  │  │        Accelerate via feature)   │    │
//! │  │  └── Grad: reverse-mode AD (exact)  │    │
//! │  └─────────────────────────────────────┘    │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! On Apple Silicon (M3 Max, 12 P-cores), with Accelerate-backed NLL:
//! ~600 fits/sec for typical HEP models (250 params, 1000 bins),
//! completing 10K toys in ~15-20s.

use crate::mle::MaximumLikelihoodEstimator;
use crate::optimizer::OptimizerConfig;
use ns_core::traits::LogDensityModel;
use ns_core::{FitResult, Result};
use ns_translate::pyhf::HistFactoryModel;
use rayon::prelude::*;

/// Fit toys with optimized batch processing.
///
/// Uses the Accelerate backend (Apple vDSP/vForce) for NLL when the `accelerate`
/// feature is enabled in ns-compute, otherwise falls back to SIMD backend.
/// Gradient always uses exact reverse-mode AD.
///
/// For batch toys we skip Hessian/covariance computation to save time.
///
/// # Arguments
/// * `model` - Base model (used for expected data and parameter structure)
/// * `params` - Parameters to generate toys at (e.g., best-fit or Asimov)
/// * `n_toys` - Number of pseudo-experiments
/// * `seed` - Random seed for reproducibility
/// * `config` - Optimizer configuration (None for defaults)
///
/// # Returns
/// Vector of fit results, one per toy.
pub fn fit_toys_batch(
    model: &HistFactoryModel,
    params: &[f64],
    n_toys: usize,
    seed: u64,
    config: Option<OptimizerConfig>,
) -> Vec<Result<FitResult>> {
    let config = config.unwrap_or_default();
    let mle = MaximumLikelihoodEstimator::with_config(config);

    // Generate expected main data at given parameters (pyhf ordering)
    let expected = match model.expected_data_pyhf_main(params) {
        Ok(e) => e,
        Err(e) => return vec![Err(e)],
    };

    let tape_capacity = model.n_params() * 20;

    // Parallel toy generation + fitting via Rayon.
    // `map_init` creates one Tape per worker thread; it is reused across all
    // toys scheduled on that thread, so total allocations = #threads (~12)
    // instead of #toys (1000+).
    (0..n_toys)
        .into_par_iter()
        .map_init(
            || ns_ad::tape::Tape::with_capacity(tape_capacity),
            |tape, toy_idx| {
                let toy_seed = seed.wrapping_add(toy_idx as u64);
                let toy_data = crate::toys::poisson_main_from_expected(&expected, toy_seed);

                let toy_model = model.with_observed_main(&toy_data)?;
                let result = mle.fit_minimum_histfactory_with_tape(&toy_model, tape)?;

                Ok(FitResult::new(
                    result.parameters,
                    vec![0.0; toy_model.dim()],
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                ))
            },
        )
        .collect()
}

/// Check if Accelerate batch backend is available.
pub fn is_accelerate_available() -> bool {
    ns_compute::accelerate_enabled()
}

/// Check if CUDA GPU batch backend is available.
#[cfg(feature = "cuda")]
pub fn is_cuda_batch_available() -> bool {
    crate::gpu_batch::is_cuda_available()
}

/// Check if CUDA GPU batch backend is available (always false without feature).
#[cfg(not(feature = "cuda"))]
pub fn is_cuda_batch_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_fit_toys_batch_smoke() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let results = fit_toys_batch(&model, &params, 2, 123, None);
        assert_eq!(results.len(), 2);

        for (i, r) in results.iter().enumerate() {
            let fit = r.as_ref().unwrap_or_else(|e| panic!("Toy {} failed: {}", i, e));
            assert!(fit.nll.is_finite(), "Toy {} NLL should be finite", i);
            assert!(fit.converged, "Toy {} should converge", i);
        }
    }

    #[test]
    fn test_fit_toys_batch_reproducible() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let results1 = fit_toys_batch(&model, &params, 3, 42, None);
        let results2 = fit_toys_batch(&model, &params, 3, 42, None);

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            if let (Ok(a), Ok(b)) = (r1, r2) {
                assert_eq!(a.parameters, b.parameters, "Results should be reproducible");
                assert_eq!(a.nll, b.nll, "NLL should be reproducible");
            }
        }
    }

    #[test]
    fn test_fit_toys_batch_matches_standard() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let mle = MaximumLikelihoodEstimator::new();

        // Compare 2 toys: batch vs standard (same seed)
        let results_batch = fit_toys_batch(&model, &params, 2, 99, None);
        let results_std = mle.fit_toys(&model, &params, 2, 99);

        for (i, (rb, rs)) in results_batch.iter().zip(results_std.iter()).enumerate() {
            if let (Ok(b), Ok(s)) = (rb, rs) {
                // NLL values should match (same optimizer, same NLL, same gradient)
                let nll_diff = (b.nll - s.nll).abs();
                assert!(
                    nll_diff < 1e-8,
                    "Toy {}: batch NLL={} standard NLL={} diff={}",
                    i, b.nll, s.nll, nll_diff
                );

                // Parameters should match closely
                for (j, (pb, ps)) in b.parameters.iter().zip(s.parameters.iter()).enumerate() {
                    let diff = (pb - ps).abs();
                    assert!(
                        diff < 1e-6,
                        "Toy {} param {}: batch={} std={} diff={}",
                        i, j, pb, ps, diff
                    );
                }
            }
        }
    }
}
