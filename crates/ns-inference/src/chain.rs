//! Chain storage and multi-chain runner.

use crate::nuts::{NutsConfig, sample_nuts};
use ns_core::Result;
use ns_core::traits::LogDensityModel;

/// Raw MCMC chain from one NUTS run.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Draws in unconstrained space.
    pub draws_unconstrained: Vec<Vec<f64>>,
    /// Draws in constrained (model) space.
    pub draws_constrained: Vec<Vec<f64>>,
    /// Divergence flag per draw.
    pub divergences: Vec<bool>,
    /// Tree depth per draw.
    pub tree_depths: Vec<usize>,
    /// Acceptance probability per draw.
    pub accept_probs: Vec<f64>,
    /// Hamiltonian energy per draw (after momentum resampling at start of transition).
    pub energies: Vec<f64>,
    /// Configured maximum tree depth for this chain (for diagnostics).
    pub max_treedepth: usize,
    /// Final adapted step size.
    pub step_size: f64,
    /// Final adapted mass matrix diagonal.
    pub mass_diag: Vec<f64>,
}

/// Result of a multi-chain NUTS sampling run.
#[derive(Debug, Clone)]
pub struct SamplerResult {
    /// Individual chains.
    pub chains: Vec<Chain>,
    /// Parameter names.
    pub param_names: Vec<String>,
    /// Number of warmup iterations per chain.
    pub n_warmup: usize,
    /// Number of post-warmup samples per chain.
    pub n_samples: usize,
    /// Diagnostics (computed lazily via [`crate::diagnostics::compute_diagnostics`]).
    pub diagnostics: Option<crate::diagnostics::DiagnosticsResult>,
}

impl SamplerResult {
    /// Total number of post-warmup draws across all chains.
    pub fn total_draws(&self) -> usize {
        self.chains.iter().map(|c| c.draws_constrained.len()).sum()
    }

    /// Get draws for a single parameter (index) across all chains.
    pub fn param_draws(&self, param_idx: usize) -> Vec<Vec<f64>> {
        self.chains
            .iter()
            .map(|c| c.draws_constrained.iter().map(|d| d[param_idx]).collect())
            .collect()
    }

    /// Mean of a parameter across all draws and chains.
    pub fn param_mean(&self, param_idx: usize) -> f64 {
        let draws = self.param_draws(param_idx);
        let n: usize = draws.iter().map(|c| c.len()).sum();
        let sum: f64 = draws.iter().flat_map(|c| c.iter()).sum();
        sum / n as f64
    }
}

/// Run NUTS sampling on multiple chains in parallel via Rayon.
///
/// Each chain gets seed `seed + chain_id`.
pub fn sample_nuts_multichain(
    model: &(impl LogDensityModel + Sync),
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
    seed: u64,
    config: NutsConfig,
) -> Result<SamplerResult> {
    use rayon::prelude::*;

    let chains: Vec<Result<Chain>> = (0..n_chains)
        .into_par_iter()
        .map(|chain_id| {
            let chain_seed = seed.wrapping_add(chain_id as u64);
            sample_nuts(model, n_warmup, n_samples, chain_seed, config.clone())
        })
        .collect();

    let chains: Vec<Chain> = chains.into_iter().collect::<Result<Vec<_>>>()?;

    let param_names: Vec<String> = model.parameter_names();

    Ok(SamplerResult { chains, param_names, n_warmup, n_samples, diagnostics: None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;
    use ns_translate::pyhf::HistFactoryModel;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_multichain_deterministic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // No jitter so identical seeds produce identical chains.
        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
        };
        let r1 = sample_nuts_multichain(&model, 2, 50, 20, 42, config.clone()).unwrap();
        let r2 = sample_nuts_multichain(&model, 2, 50, 20, 42, config).unwrap();

        for (c1, c2) in r1.chains.iter().zip(r2.chains.iter()) {
            assert_eq!(
                c1.draws_constrained, c2.draws_constrained,
                "Multi-chain should be deterministic"
            );
        }
    }

    #[test]
    fn test_multichain_basic() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let config = NutsConfig {
            max_treedepth: 8,
            target_accept: 0.8,
            init_jitter: 0.5,
            init_jitter_rel: None,
        };
        let result = sample_nuts_multichain(&model, 2, 100, 50, 42, config).unwrap();

        assert_eq!(result.chains.len(), 2);
        assert_eq!(result.n_warmup, 100);
        assert_eq!(result.n_samples, 50);
        assert_eq!(result.total_draws(), 100);

        // POI mean should be reasonable
        let poi_mean = result.param_mean(0);
        assert!(poi_mean > 0.0 && poi_mean < 3.0, "POI mean should be reasonable: {}", poi_mean);

        for c in &result.chains {
            assert_eq!(c.energies.len(), 50);
        }
    }
}
