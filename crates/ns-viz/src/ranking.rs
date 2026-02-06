use ns_inference::RankingEntry;
use serde::{Deserialize, Serialize};

/// Ranking artifact for nuisance-parameter impact assessment (plot-friendly JSON).
///
/// This is the "data product" behind a typical HEP ranking plot:
/// - pull / constraint of each NP from the unconditional fit
/// - POI shifts when fixing the NP at +/- 1 sigma
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingArtifact {
    /// Nuisance parameter names.
    pub names: Vec<String>,
    /// POI shift when NP fixed at +1 sigma.
    pub delta_mu_up: Vec<f64>,
    /// POI shift when NP fixed at -1 sigma.
    pub delta_mu_down: Vec<f64>,
    /// Pull: (theta_hat - theta0) / sigma.
    pub pull: Vec<f64>,
    /// Constraint: sigma_hat / sigma.
    pub constraint: Vec<f64>,
}

impl From<Vec<RankingEntry>> for RankingArtifact {
    fn from(entries: Vec<RankingEntry>) -> Self {
        let mut names = Vec::with_capacity(entries.len());
        let mut delta_mu_up = Vec::with_capacity(entries.len());
        let mut delta_mu_down = Vec::with_capacity(entries.len());
        let mut pull = Vec::with_capacity(entries.len());
        let mut constraint = Vec::with_capacity(entries.len());

        for e in entries {
            names.push(e.name);
            delta_mu_up.push(e.delta_mu_up);
            delta_mu_down.push(e.delta_mu_down);
            pull.push(e.pull);
            constraint.push(e.constraint);
        }

        Self { names, delta_mu_up, delta_mu_down, pull, constraint }
    }
}
