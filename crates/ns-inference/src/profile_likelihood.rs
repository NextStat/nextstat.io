//! Profile likelihood utilities (frequentist).
//!
//! Implements the profile likelihood test statistic for upper-limit workflows:
//! `q_mu` / `qtilde_mu` in pyhf terminology (Cowan et al., arXiv:1007.1727).

use crate::MaximumLikelihoodEstimator;
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel};
use ns_core::{Error, Result};
#[cfg(feature = "cuda")]
use ns_translate::pyhf::HistFactoryModel;

/// Single point in a profile likelihood scan.
#[derive(Debug, Clone)]
pub struct ProfilePoint {
    /// Tested POI value.
    pub mu: f64,
    /// Test statistic value (pyhf `qmu` / `qmu_tilde`).
    pub q_mu: f64,
    /// Conditional NLL at `mu`.
    pub nll_mu: f64,
    /// Conditional fit convergence.
    pub converged: bool,
    /// Conditional fit iterations (argmin iterations).
    pub n_iter: u64,
}

/// Profile likelihood scan result.
#[derive(Debug, Clone)]
pub struct ProfileLikelihoodScan {
    /// POI index in NextStat parameter order.
    pub poi_index: usize,
    /// Unconditional best-fit POI value.
    pub mu_hat: f64,
    /// Unconditional NLL at the global minimum.
    pub nll_hat: f64,
    /// Per-point results.
    pub points: Vec<ProfilePoint>,
}

fn poi_index(model: &(impl PoiModel + ?Sized)) -> Result<usize> {
    model.poi_index().ok_or_else(|| Error::Validation("No POI defined".to_string()))
}

/// Compute pyhf-style `q_mu`/`qtilde_mu` for a single `mu_test`.
///
/// Note: NextStat's `HistFactoryModel::nll` is `-log L`, while pyhf uses `twice_nll = -2 log L`.
/// Therefore `q_mu = 2 * (nll(mu) - nll_hat)`, clipped at 0.
pub fn qmu_like(
    mle: &MaximumLikelihoodEstimator,
    model: &(impl LogDensityModel + PoiModel + FixedParamModel),
    mu_test: f64,
) -> Result<f64> {
    let poi = poi_index(model)?;

    let free = mle.fit_minimum(model)?;
    let mu_hat = free.parameters[poi];

    let fixed_model = model.with_fixed_param(poi, mu_test);
    let fixed = mle.fit_minimum(&fixed_model)?;

    let llr = 2.0 * (fixed.fval - free.fval);
    let mut q = llr.max(0.0);
    if mu_hat > mu_test {
        q = 0.0;
    }
    Ok(q)
}

/// Run a profile likelihood scan over the provided POI values.
pub fn scan(
    mle: &MaximumLikelihoodEstimator,
    model: &(impl LogDensityModel + PoiModel + FixedParamModel),
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    let poi = poi_index(model)?;

    let free = mle.fit_minimum(model)?;
    let mu_hat = free.parameters[poi];
    let nll_hat = free.fval;

    let mut points = Vec::with_capacity(mu_values.len());
    for &mu in mu_values {
        let fixed_model = model.with_fixed_param(poi, mu);
        let fixed = mle.fit_minimum(&fixed_model)?;

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        points.push(ProfilePoint {
            mu,
            q_mu: q,
            nll_mu: fixed.fval,
            converged: fixed.converged,
            n_iter: fixed.n_iter,
        });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
}

/// GPU-accelerated profile likelihood scan.
///
/// One `GpuSession` is shared across all scan points, avoiding repeated
/// model serialization and GPU buffer allocation. Warm-start: each point
/// starts from the previous point's best-fit parameters.
#[cfg(feature = "cuda")]
pub fn scan_gpu(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    let poi = model
        .poi_index()
        .ok_or_else(|| Error::Validation("No POI defined".to_string()))?;

    let session = crate::gpu_single::GpuSession::new(model)?;
    let config = mle.config().clone();

    // Free fit (unconditional MLE)
    let free = session.fit_minimum(model, &config)?;
    let mu_hat = free.parameters[poi];
    let nll_hat = free.fval;

    let base_bounds = model.parameter_bounds();
    let mut warm_params = free.parameters.clone();

    let mut points = Vec::with_capacity(mu_values.len());
    for &mu in mu_values {
        // Fix POI at mu via bounds clamping
        let mut bounds = base_bounds.clone();
        bounds[poi] = (mu, mu);
        warm_params[poi] = mu;

        let fixed = session.fit_minimum_from_with_bounds(
            model,
            &warm_params,
            &bounds,
            &config,
        )?;

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        // Update warm-start for next point
        warm_params = fixed.parameters.clone();

        points.push(ProfilePoint {
            mu,
            q_mu: q,
            nll_mu: fixed.fval,
            converged: fixed.converged,
            n_iter: fixed.n_iter,
        });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
}
