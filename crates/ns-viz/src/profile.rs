use ns_inference::profile_likelihood::ProfileLikelihoodScan;
use serde::{Deserialize, Serialize};

/// Single point in a profile likelihood curve artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCurvePoint {
    /// Tested POI value.
    pub mu: f64,
    /// Test statistic value (q_mu / qtilde_mu).
    pub q_mu: f64,
    /// Conditional NLL at `mu`.
    pub nll_mu: f64,
    /// Whether the conditional fit converged.
    pub converged: bool,
    /// Conditional fit iterations (argmin iterations).
    pub n_iter: u64,
}

/// Plot-friendly artifact for a profile likelihood scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCurveArtifact {
    /// POI index in NextStat parameter order.
    pub poi_index: usize,
    /// Unconditional best-fit POI value.
    pub mu_hat: f64,
    /// Unconditional NLL at the global minimum.
    pub nll_hat: f64,
    /// Per-point results.
    pub points: Vec<ProfileCurvePoint>,
    /// Scan x-values (same as `points[*].mu`).
    pub mu_values: Vec<f64>,
    /// Test statistic values aligned with `mu_values`.
    pub q_mu_values: Vec<f64>,
    /// `2 * (nll_mu - nll_hat)` aligned with `mu_values` (useful for likelihood plots).
    pub twice_delta_nll: Vec<f64>,
}

impl From<ProfileLikelihoodScan> for ProfileCurveArtifact {
    fn from(scan: ProfileLikelihoodScan) -> Self {
        let ProfileLikelihoodScan { poi_index, mu_hat, nll_hat, points } = scan;

        let mut out_points = Vec::with_capacity(points.len());
        let mut mu_values = Vec::with_capacity(points.len());
        let mut q_mu_values = Vec::with_capacity(points.len());
        let mut twice_delta_nll = Vec::with_capacity(points.len());

        for p in points {
            mu_values.push(p.mu);
            q_mu_values.push(p.q_mu);
            twice_delta_nll.push(2.0 * (p.nll_mu - nll_hat));
            out_points.push(ProfileCurvePoint {
                mu: p.mu,
                q_mu: p.q_mu,
                nll_mu: p.nll_mu,
                converged: p.converged,
                n_iter: p.n_iter,
            });
        }

        Self { poi_index, mu_hat, nll_hat, points: out_points, mu_values, q_mu_values, twice_delta_nll }
    }
}
