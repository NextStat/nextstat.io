//! Common data types for NextStat

use serde::{Deserialize, Serialize};

/// Fit result containing parameter estimates and uncertainties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult {
    /// Best-fit parameter values
    pub parameters: Vec<f64>,

    /// Parameter uncertainties (sqrt of covariance diagonal)
    pub uncertainties: Vec<f64>,

    /// Covariance matrix (row-major, N×N). `None` if Hessian inversion failed.
    pub covariance: Option<Vec<f64>>,

    /// Negative log-likelihood at minimum
    pub nll: f64,

    /// Convergence status
    pub converged: bool,

    /// Number of function evaluations
    pub n_evaluations: usize,
}

impl FitResult {
    /// Create a new fit result
    pub fn new(
        parameters: Vec<f64>,
        uncertainties: Vec<f64>,
        nll: f64,
        converged: bool,
        n_evaluations: usize,
    ) -> Self {
        Self { parameters, uncertainties, covariance: None, nll, converged, n_evaluations }
    }

    /// Create a fit result with covariance matrix
    pub fn with_covariance(
        parameters: Vec<f64>,
        uncertainties: Vec<f64>,
        covariance: Vec<f64>,
        nll: f64,
        converged: bool,
        n_evaluations: usize,
    ) -> Self {
        Self {
            parameters,
            uncertainties,
            covariance: Some(covariance),
            nll,
            converged,
            n_evaluations,
        }
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
        let result = FitResult::new(vec![1.0, 2.0], vec![0.1, 0.2], 123.45, true, 100);
        assert_eq!(result.parameters.len(), 2);
        assert_eq!(result.uncertainties.len(), 2);
        assert!(result.converged);
    }
}
