//! Laplace approximation utilities.
//!
//! This provides a generic Laplace approximation to the marginal likelihood:
//!
//! `Z = ∫ exp(-NLL(x)) dx ≈ exp(-NLL(x_hat)) * (2π)^(d/2) * |H|^(-1/2)`
//!
//! where `H` is the Hessian of `NLL` at the mode `x_hat`.

use nalgebra::DMatrix;
use ns_core::traits::LogDensityModel;
use ns_core::{Error, Result};

/// Result of a Laplace approximation at a given mode.
#[derive(Debug, Clone)]
pub struct LaplaceResult {
    /// Dimension of the parameter vector.
    pub dim: usize,
    /// NLL evaluated at the mode.
    pub nll_at_mode: f64,
    /// `log |H|` where `H` is the Hessian of NLL at the mode.
    pub log_det_hessian: f64,
    /// `log Z` where `Z = ∫ exp(-NLL(x)) dx` under the Laplace approximation.
    pub log_marginal: f64,
}

fn compute_hessian(model: &impl LogDensityModel, params: &[f64]) -> Result<DMatrix<f64>> {
    let n = params.len();
    let grad_center = model.grad_nll(params)?;

    let mut hessian = DMatrix::zeros(n, n);
    for j in 0..n {
        let eps = 1e-4 * params[j].abs().max(1.0);

        let mut params_plus = params.to_vec();
        params_plus[j] += eps;
        let grad_plus = model.grad_nll(&params_plus)?;

        for i in 0..n {
            hessian[(i, j)] = (grad_plus[i] - grad_center[i]) / eps;
        }
    }

    // Symmetrise: H = (H + H^T)/2.
    let ht = hessian.transpose();
    Ok((&hessian + &ht) * 0.5)
}

fn log_det_pd(h: &DMatrix<f64>) -> Result<f64> {
    let chol = nalgebra::linalg::Cholesky::new(h.clone())
        .ok_or_else(|| Error::Validation("Laplace requires a positive-definite Hessian".to_string()))?;
    let l = chol.l();
    let mut sum = 0.0;
    for i in 0..l.nrows() {
        let d = l[(i, i)];
        if !d.is_finite() || d <= 0.0 {
            return Err(Error::Validation("non-finite/negative Cholesky diagonal".to_string()));
        }
        sum += d.ln();
    }
    Ok(2.0 * sum)
}

/// Compute a Laplace approximation at `params_mode`.
pub fn laplace_log_marginal(model: &impl LogDensityModel, params_mode: &[f64]) -> Result<LaplaceResult> {
    if params_mode.len() != model.dim() {
        return Err(Error::Validation("params length mismatch".to_string()));
    }

    let nll = model.nll(params_mode)?;
    if !nll.is_finite() {
        return Err(Error::Validation("nll at mode must be finite".to_string()));
    }

    let h = compute_hessian(model, params_mode)?;
    let log_det = log_det_pd(&h)?;

    let d = params_mode.len() as f64;
    let log2pi = (2.0 * std::f64::consts::PI).ln();
    let log_marginal = -nll + 0.5 * d * log2pi - 0.5 * log_det;

    Ok(LaplaceResult {
        dim: params_mode.len(),
        nll_at_mode: nll,
        log_det_hessian: log_det,
        log_marginal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::MaximumLikelihoodEstimator;
    use crate::regression::LinearRegressionModel;

    #[test]
    fn laplace_result_is_finite_for_small_linear_regression() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![1.0, 2.1, 2.9, 4.2];
        let m = LinearRegressionModel::new(x, y, false).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let opt = mle.fit_minimum(&m).unwrap();
        let r = laplace_log_marginal(&m, &opt.parameters).unwrap();
        assert_eq!(r.dim, m.dim());
        assert!(r.nll_at_mode.is_finite());
        assert!(r.log_det_hessian.is_finite());
        assert!(r.log_marginal.is_finite());
    }
}

