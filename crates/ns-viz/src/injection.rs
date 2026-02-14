//! Signal injection / linearity test artifact.
//!
//! TRExFitter supports signal injection studies: pseudo-experiments are
//! generated at various injected signal strengths μ_inj, fitted, and the
//! recovered μ̂ is compared to the injected value.  A linearity plot
//! (μ̂ vs μ_inj) and pull distribution are the standard outputs.
//!
//! This module provides the plot-friendly JSON artifacts.

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Summary statistics for pseudo-experiments at a single injected μ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionPoint {
    /// Injected signal strength.
    pub mu_injected: f64,
    /// Mean of recovered μ̂ across pseudo-experiments.
    pub mu_hat_mean: f64,
    /// Standard deviation of recovered μ̂.
    pub mu_hat_std: f64,
    /// Median of recovered μ̂.
    pub mu_hat_median: f64,
    /// Mean pull: (μ̂ − μ_inj) / σ_μ̂.
    pub pull_mean: f64,
    /// Standard deviation of pulls (should be ~1 if uncertainties are correct).
    pub pull_std: f64,
    /// Number of pseudo-experiments.
    pub n_toys: u32,
    /// Number of converged fits.
    pub n_converged: u32,
}

/// Plot-friendly artifact for a signal injection / linearity study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionArtifact {
    /// POI label (e.g. "μ" or "σ/σ_SM").
    pub poi_label: String,
    /// Per-injection-point results.
    pub points: Vec<InjectionPoint>,
    /// Injected μ values (same as `points[*].mu_injected`).
    pub mu_injected_values: Vec<f64>,
    /// Mean recovered μ̂ values aligned with `mu_injected_values`.
    pub mu_hat_mean_values: Vec<f64>,
    /// Error bars (std of μ̂) aligned with `mu_injected_values`.
    pub mu_hat_std_values: Vec<f64>,
    /// Linear fit slope (should be ~1 for unbiased).
    pub linearity_slope: f64,
    /// Linear fit intercept (should be ~0 for unbiased).
    pub linearity_intercept: f64,
}

impl InjectionArtifact {
    /// Build from pre-computed injection points.
    pub fn from_points(poi_label: &str, points: Vec<InjectionPoint>) -> Result<Self> {
        if points.len() < 2 {
            return Err(Error::Validation(
                "injection study requires at least 2 injection points".into(),
            ));
        }

        let mu_injected_values: Vec<f64> = points.iter().map(|p| p.mu_injected).collect();
        let mu_hat_mean_values: Vec<f64> = points.iter().map(|p| p.mu_hat_mean).collect();
        let mu_hat_std_values: Vec<f64> = points.iter().map(|p| p.mu_hat_std).collect();

        // Simple least-squares linear fit: μ̂_mean = slope * μ_inj + intercept.
        let (slope, intercept) = linear_fit(&mu_injected_values, &mu_hat_mean_values);

        Ok(Self {
            poi_label: poi_label.to_string(),
            points,
            mu_injected_values,
            mu_hat_mean_values,
            mu_hat_std_values,
            linearity_slope: slope,
            linearity_intercept: intercept,
        })
    }
}

/// Simple unweighted least-squares linear fit: y = slope * x + intercept.
fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (1.0, 0.0);
    }
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return (1.0, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points() -> Vec<InjectionPoint> {
        vec![
            InjectionPoint {
                mu_injected: 0.0,
                mu_hat_mean: 0.02,
                mu_hat_std: 0.3,
                mu_hat_median: 0.01,
                pull_mean: 0.05,
                pull_std: 1.02,
                n_toys: 1000,
                n_converged: 998,
            },
            InjectionPoint {
                mu_injected: 1.0,
                mu_hat_mean: 0.98,
                mu_hat_std: 0.3,
                mu_hat_median: 0.97,
                pull_mean: -0.07,
                pull_std: 0.99,
                n_toys: 1000,
                n_converged: 999,
            },
            InjectionPoint {
                mu_injected: 2.0,
                mu_hat_mean: 2.01,
                mu_hat_std: 0.31,
                mu_hat_median: 2.00,
                pull_mean: 0.03,
                pull_std: 1.01,
                n_toys: 1000,
                n_converged: 997,
            },
        ]
    }

    #[test]
    fn test_injection_basic() {
        let art = InjectionArtifact::from_points("mu", make_points()).unwrap();
        assert_eq!(art.mu_injected_values.len(), 3);
        // Slope should be close to 1.
        assert!((art.linearity_slope - 1.0).abs() < 0.1);
        // Intercept should be close to 0.
        assert!(art.linearity_intercept.abs() < 0.1);
    }

    #[test]
    fn test_injection_too_few_points() {
        let r = InjectionArtifact::from_points(
            "mu",
            vec![InjectionPoint {
                mu_injected: 1.0,
                mu_hat_mean: 1.0,
                mu_hat_std: 0.3,
                mu_hat_median: 1.0,
                pull_mean: 0.0,
                pull_std: 1.0,
                n_toys: 100,
                n_converged: 100,
            }],
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_linear_fit_perfect() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.5, 1.5, 2.5, 3.5];
        let (slope, intercept) = linear_fit(&x, &y);
        assert!((slope - 1.0).abs() < 1e-10);
        assert!((intercept - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_injection_serialization() {
        let art = InjectionArtifact::from_points("mu", make_points()).unwrap();
        let json = serde_json::to_string(&art).unwrap();
        let back: InjectionArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.points.len(), 3);
    }
}
