//! Common data types for NextStat

use serde::{Deserialize, Serialize};

/// Fit result containing parameter estimates and uncertainties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult {
    /// Best-fit parameter values
    pub parameters: Vec<f64>,

    /// Parameter uncertainties (sqrt of covariance diagonal)
    pub uncertainties: Vec<f64>,

    /// Covariance matrix (row-major, N×N). `None` if inversion failed or the
    /// resulting covariance is numerically invalid (e.g., non-positive variances).
    pub covariance: Option<Vec<f64>>,

    /// Negative log-likelihood at minimum
    pub nll: f64,

    /// Convergence status
    pub converged: bool,

    /// Number of optimizer iterations.
    ///
    /// This is the argmin iteration counter (L-BFGS iterations), not the number
    /// of objective evaluations.
    #[serde(alias = "n_evaluations")]
    pub n_iter: usize,

    /// Number of objective (cost) evaluations performed by the optimizer.
    ///
    /// Includes evaluations during line-search.
    #[serde(default)]
    pub n_fev: usize,

    /// Number of gradient evaluations performed by the optimizer.
    #[serde(default)]
    pub n_gev: usize,

    /// Why the optimizer stopped (e.g. "SolverConverged", "MaxIterReached", "1D golden-section search").
    #[serde(default)]
    pub termination_reason: String,

    /// Gradient norm at termination. `NAN` if unavailable (e.g. 1D golden-section).
    #[serde(default = "default_nan")]
    pub final_grad_norm: f64,

    /// Negative log-likelihood before optimisation. `NAN` if unavailable.
    #[serde(default = "default_nan")]
    pub initial_nll: f64,

    /// Number of parameters sitting at their bound at the solution.
    #[serde(default)]
    pub n_active_bounds: usize,
}

fn default_nan() -> f64 {
    f64::NAN
}

impl FitResult {
    /// Create a new fit result
    pub fn new(
        parameters: Vec<f64>,
        uncertainties: Vec<f64>,
        nll: f64,
        converged: bool,
        n_iter: usize,
        n_fev: usize,
        n_gev: usize,
    ) -> Self {
        Self {
            parameters,
            uncertainties,
            covariance: None,
            nll,
            converged,
            n_iter,
            n_fev,
            n_gev,
            termination_reason: String::new(),
            final_grad_norm: f64::NAN,
            initial_nll: f64::NAN,
            n_active_bounds: 0,
        }
    }

    /// Create a fit result with covariance matrix
    #[allow(clippy::too_many_arguments)]
    pub fn with_covariance(
        parameters: Vec<f64>,
        uncertainties: Vec<f64>,
        covariance: Vec<f64>,
        nll: f64,
        converged: bool,
        n_iter: usize,
        n_fev: usize,
        n_gev: usize,
    ) -> Self {
        Self {
            parameters,
            uncertainties,
            covariance: Some(covariance),
            nll,
            converged,
            n_iter,
            n_fev,
            n_gev,
            termination_reason: String::new(),
            final_grad_norm: f64::NAN,
            initial_nll: f64::NAN,
            n_active_bounds: 0,
        }
    }

    /// Attach optimizer diagnostics (builder-style).
    pub fn with_diagnostics(
        mut self,
        termination_reason: String,
        final_grad_norm: f64,
        initial_nll: f64,
        n_active_bounds: usize,
    ) -> Self {
        self.termination_reason = termination_reason;
        self.final_grad_norm = final_grad_norm;
        self.initial_nll = initial_nll;
        self.n_active_bounds = n_active_bounds;
        self
    }

    /// Back-compat alias for older code/tests. Prefer `n_iter`.
    #[deprecated(since = "0.9.0", note = "Use the `n_iter` field directly")]
    pub fn n_evaluations(&self) -> usize {
        self.n_iter
    }

    /// Get correlation matrix element (i, j). Returns `None` if covariance is unavailable.
    pub fn correlation(&self, i: usize, j: usize) -> Option<f64> {
        let cov = self.covariance.as_ref()?;
        let n = self.parameters.len();
        if i >= n || j >= n {
            return None;
        }
        let sigma_i = self.uncertainties[i];
        let sigma_j = self.uncertainties[j];
        if sigma_i <= 0.0 || sigma_j <= 0.0 {
            return None;
        }
        Some(cov[i * n + j] / (sigma_i * sigma_j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_result() {
        let result = FitResult::new(vec![1.0, 2.0], vec![0.1, 0.2], 123.45, true, 100, 0, 0);
        assert_eq!(result.parameters.len(), 2);
        assert_eq!(result.uncertainties.len(), 2);
        assert!(result.converged);
        assert_eq!(result.n_iter, 100);
    }
}
