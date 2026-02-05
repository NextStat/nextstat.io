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
}

impl From<ProfileLikelihoodScan> for ProfileCurveArtifact {
    fn from(scan: ProfileLikelihoodScan) -> Self {
        Self {
            poi_index: scan.poi_index,
            mu_hat: scan.mu_hat,
            nll_hat: scan.nll_hat,
            points: scan
                .points
                .into_iter()
                .map(|p| ProfileCurvePoint {
                    mu: p.mu,
                    q_mu: p.q_mu,
                    nll_mu: p.nll_mu,
                    converged: p.converged,
                    n_iter: p.n_iter,
                })
                .collect(),
        }
    }
}

