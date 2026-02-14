//! Significance scan artifact — p₀ vs mass/parameter scan.
//!
//! TRExFitter step `s` (`GetSignificance`) computes the discovery significance
//! at multiple mass or signal-strength hypotheses, plotting p₀ and Z vs the
//! scanned parameter.  This module provides the plot-friendly JSON artifact.

use ns_core::{Error, Result};
use ns_inference::{AsymptoticCLsContext, MaximumLikelihoodEstimator};
use serde::{Deserialize, Serialize};

/// Single point in a significance scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceScanPoint {
    /// Scanned parameter value (e.g. mass hypothesis or signal strength).
    pub scan_value: f64,
    /// Observed p₀ = P(q₀ ≥ q₀_obs | μ=0).
    pub p0: f64,
    /// Observed significance Z = Φ⁻¹(1 − p₀).
    pub z_obs: f64,
    /// Expected significance from Asimov dataset.
    pub z_exp: f64,
    /// Best-fit signal strength μ̂.
    pub mu_hat: f64,
    /// Test statistic q₀.
    pub q0: f64,
}

/// Plot-friendly artifact for a significance scan (p₀ / Z vs parameter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceScanArtifact {
    /// Label for the scanned axis (e.g. "m_H [GeV]" or "mu").
    pub scan_label: String,
    /// Per-point results.
    pub points: Vec<SignificanceScanPoint>,
    /// Scan x-values (same as `points[*].scan_value`).
    pub scan_values: Vec<f64>,
    /// Observed p₀ values aligned with `scan_values`.
    pub p0_values: Vec<f64>,
    /// Observed Z values aligned with `scan_values`.
    pub z_obs_values: Vec<f64>,
    /// Expected Z values aligned with `scan_values`.
    pub z_exp_values: Vec<f64>,
    /// Index of the point with maximum observed significance.
    pub best_index: usize,
    /// Maximum observed significance.
    pub best_z_obs: f64,
}

impl SignificanceScanArtifact {
    /// Build a significance scan artifact from a single workspace at
    /// a single mass point.  The discovery test (μ=0) is performed and
    /// the result is stored as a single-point scan.
    ///
    /// For multi-mass scans, call `from_points` with pre-computed results.
    pub fn from_single(
        ctx: &AsymptoticCLsContext,
        mle: &MaximumLikelihoodEstimator,
        scan_value: f64,
        scan_label: &str,
    ) -> Result<Self> {
        let r = ctx.hypotest_qtilde(mle, 0.0)?;
        let p0 = r.clsb;
        let z_obs = if r.mu_hat > 0.0 && r.q_mu > 0.0 { r.q_mu.sqrt() } else { 0.0 };
        let z_exp = if r.q_mu_a > 0.0 { r.q_mu_a.sqrt() } else { 0.0 };

        let pt =
            SignificanceScanPoint { scan_value, p0, z_obs, z_exp, mu_hat: r.mu_hat, q0: r.q_mu };

        Ok(Self {
            scan_label: scan_label.to_string(),
            points: vec![pt.clone()],
            scan_values: vec![scan_value],
            p0_values: vec![p0],
            z_obs_values: vec![z_obs],
            z_exp_values: vec![z_exp],
            best_index: 0,
            best_z_obs: z_obs,
        })
    }

    /// Build from pre-computed points (e.g. from multiple workspaces at
    /// different mass hypotheses).
    pub fn from_points(scan_label: &str, points: Vec<SignificanceScanPoint>) -> Result<Self> {
        if points.is_empty() {
            return Err(Error::Validation("significance scan must have at least 1 point".into()));
        }

        let scan_values: Vec<f64> = points.iter().map(|p| p.scan_value).collect();
        let p0_values: Vec<f64> = points.iter().map(|p| p.p0).collect();
        let z_obs_values: Vec<f64> = points.iter().map(|p| p.z_obs).collect();
        let z_exp_values: Vec<f64> = points.iter().map(|p| p.z_exp).collect();

        let (best_index, best_z_obs) = z_obs_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &z)| (i, z))
            .unwrap_or((0, 0.0));

        Ok(Self {
            scan_label: scan_label.to_string(),
            points,
            scan_values,
            p0_values,
            z_obs_values,
            z_exp_values,
            best_index,
            best_z_obs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_points_basic() {
        let pts = vec![
            SignificanceScanPoint {
                scan_value: 125.0,
                p0: 0.01,
                z_obs: 2.33,
                z_exp: 2.0,
                mu_hat: 1.1,
                q0: 5.43,
            },
            SignificanceScanPoint {
                scan_value: 130.0,
                p0: 0.05,
                z_obs: 1.64,
                z_exp: 1.5,
                mu_hat: 0.9,
                q0: 2.69,
            },
        ];
        let art = SignificanceScanArtifact::from_points("m_H [GeV]", pts).unwrap();
        assert_eq!(art.scan_values.len(), 2);
        assert_eq!(art.best_index, 0);
        assert!((art.best_z_obs - 2.33).abs() < 1e-12);
    }

    #[test]
    fn test_from_points_empty() {
        let r = SignificanceScanArtifact::from_points("mu", vec![]);
        assert!(r.is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let pt = SignificanceScanPoint {
            scan_value: 125.0,
            p0: 2.87e-7,
            z_obs: 5.0,
            z_exp: 4.5,
            mu_hat: 1.0,
            q0: 25.0,
        };
        let art = SignificanceScanArtifact::from_points("m_H", vec![pt]).unwrap();
        let json = serde_json::to_string(&art).unwrap();
        let back: SignificanceScanArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.scan_values.len(), 1);
        assert!((back.best_z_obs - 5.0).abs() < 1e-12);
    }
}
