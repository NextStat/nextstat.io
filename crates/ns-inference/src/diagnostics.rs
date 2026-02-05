//! MCMC diagnostics: split R-hat, bulk ESS, tail ESS.
//!
//! Note: this implementation is intentionally lightweight. It computes split R-hat
//! and a simple autocorrelation-based ESS estimate. It does not (yet) implement
//! the full rank-normalized + folded diagnostics described by Vehtari et al. (2021).

/// Diagnostics for a multi-chain NUTS run.
#[derive(Debug, Clone)]
pub struct DiagnosticsResult {
    /// Split R-hat per parameter.
    pub r_hat: Vec<f64>,
    /// Bulk ESS per parameter.
    pub ess_bulk: Vec<f64>,
    /// Tail ESS per parameter.
    pub ess_tail: Vec<f64>,
    /// Fraction of divergent transitions.
    pub divergence_rate: f64,
    /// Fraction of transitions hitting max treedepth.
    pub max_treedepth_rate: f64,
}

/// Compute split R-hat for one parameter across multiple chains.
///
/// Each chain is split in half, giving 2*M half-chains.
/// R-hat = sqrt((var_hat+ / W)) where var_hat+ = (N-1)/N * W + B/N.
pub fn r_hat(chains: &[&[f64]]) -> f64 {
    if chains.is_empty() {
        return f64::NAN;
    }

    // Split each chain in half
    let mut half_chains: Vec<&[f64]> = Vec::new();
    for chain in chains {
        let n = chain.len();
        if n < 4 {
            return f64::NAN;
        }
        let mid = n / 2;
        half_chains.push(&chain[..mid]);
        half_chains.push(&chain[mid..]);
    }

    let m = half_chains.len() as f64;
    let n = half_chains[0].len() as f64;

    // Chain means
    let chain_means: Vec<f64> =
        half_chains.iter().map(|c| c.iter().sum::<f64>() / c.len() as f64).collect();
    let grand_mean: f64 = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance: B = n/(m-1) * sum((chain_mean - grand_mean)^2)
    let b: f64 =
        chain_means.iter().map(|&cm| (cm - grand_mean).powi(2)).sum::<f64>() * n / (m - 1.0);

    // Within-chain variance: W = mean of chain variances
    let w: f64 = half_chains
        .iter()
        .zip(chain_means.iter())
        .map(|(c, &cm)| {
            let var: f64 =
                c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (c.len() as f64 - 1.0);
            var
        })
        .sum::<f64>()
        / m;

    if w < 1e-30 {
        return f64::NAN;
    }

    // Marginal posterior variance estimate
    let var_hat_plus = (n - 1.0) / n * w + b / n;

    (var_hat_plus / w).sqrt()
}

/// Compute effective sample size (ESS) via initial monotone sequence estimator.
///
/// Combines all chains after split for ESS computation.
pub fn ess_bulk(chains: &[&[f64]]) -> f64 {
    if chains.is_empty() {
        return 0.0;
    }

    // Simple ESS estimate: N_eff = M*N / (1 + 2*sum(rho))
    // where rho is the autocorrelation.
    // For multi-chain, we use the formula ESS = M*N * var_hat+ / B_hat
    // but a simpler approach: compute per-chain ESS via autocorrelation.

    let mut total_ess = 0.0;

    for chain in chains {
        total_ess += chain_ess(chain);
    }

    total_ess
}

/// Compute tail ESS (ESS of indicator I(x <= median) or I(x >= median)).
pub fn ess_tail(chains: &[&[f64]]) -> f64 {
    if chains.is_empty() {
        return 0.0;
    }

    // Collect all draws and find median
    let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = all[all.len() / 2];

    // Compute ESS of the lower tail indicator
    let tail_chains: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|&x| if x <= median { 1.0 } else { 0.0 }).collect())
        .collect();
    let tail_refs: Vec<&[f64]> = tail_chains.iter().map(|c| c.as_slice()).collect();

    ess_bulk(&tail_refs)
}

/// Single-chain ESS via initial positive sequence estimator of autocorrelation.
fn chain_ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 {
        return 0.0;
    }

    let mean: f64 = chain.iter().sum::<f64>() / n as f64;
    let var: f64 = chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-30 {
        return n as f64;
    }

    // Compute autocorrelation using FFT-free approach (direct for moderate N)
    let max_lag = n - 1;
    let mut rho_sum = 0.0;

    let mut lag = 1;
    while lag < max_lag {
        let rho_lag = autocorrelation(chain, &mean, &var, lag);
        let rho_lag1 =
            if lag + 1 < max_lag { autocorrelation(chain, &mean, &var, lag + 1) } else { 0.0 };

        // Initial positive sequence: stop when pair sum is negative
        if rho_lag + rho_lag1 < 0.0 {
            break;
        }

        rho_sum += rho_lag + rho_lag1;
        lag += 2;
    }

    let tau = 1.0 + 2.0 * rho_sum;
    (n as f64 / tau).max(1.0)
}

/// Autocorrelation at given lag.
fn autocorrelation(chain: &[f64], mean: &f64, var: &f64, lag: usize) -> f64 {
    let n = chain.len();
    if lag >= n || *var < 1e-30 {
        return 0.0;
    }

    let sum: f64 = (0..n - lag).map(|i| (chain[i] - mean) * (chain[i + lag] - mean)).sum();

    sum / (n as f64 * var)
}

/// Compute full diagnostics for a SamplerResult.
pub fn compute_diagnostics(result: &crate::chain::SamplerResult) -> DiagnosticsResult {
    let n_params = result.param_names.len();

    let mut r_hat_vals = Vec::with_capacity(n_params);
    let mut ess_bulk_vals = Vec::with_capacity(n_params);
    let mut ess_tail_vals = Vec::with_capacity(n_params);

    for p in 0..n_params {
        let chain_draws: Vec<Vec<f64>> = result.param_draws(p);
        let chain_refs: Vec<&[f64]> = chain_draws.iter().map(|c| c.as_slice()).collect();

        r_hat_vals.push(r_hat(&chain_refs));
        ess_bulk_vals.push(ess_bulk(&chain_refs));
        ess_tail_vals.push(ess_tail(&chain_refs));
    }

    // Divergence and max treedepth rates
    let total_samples: usize = result.chains.iter().map(|c| c.divergences.len()).sum();
    let n_divergent: usize =
        result.chains.iter().flat_map(|c| c.divergences.iter()).filter(|&&d| d).count();
    let divergence_rate =
        if total_samples > 0 { n_divergent as f64 / total_samples as f64 } else { 0.0 };

    // Rate of transitions that hit the configured maximum treedepth.
    let n_max_depth: usize = result
        .chains
        .iter()
        .flat_map(|c| c.tree_depths.iter().map(move |&d| (d, c.max_treedepth)))
        .filter(|(d, max_d)| *d >= *max_d)
        .count();
    let max_treedepth_rate =
        if total_samples > 0 { n_max_depth as f64 / total_samples as f64 } else { 0.0 };

    DiagnosticsResult {
        r_hat: r_hat_vals,
        ess_bulk: ess_bulk_vals,
        ess_tail: ess_tail_vals,
        divergence_rate,
        max_treedepth_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_hat_well_mixed() {
        // Well-mixed chains from same distribution -> R-hat should be close to 1
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        let chain1: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng1)).collect();
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(2);
        let chain2: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng2)).collect();

        let rhat = r_hat(&[&chain1, &chain2]);
        assert!(rhat < 1.05, "R-hat for well-mixed chains should be ~1: {}", rhat);
    }

    #[test]
    fn test_r_hat_diverged_chains() {
        // Two chains at different means -> R-hat >> 1
        let chain1: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let chain2: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.01).collect();
        let rhat = r_hat(&[&chain1, &chain2]);
        assert!(rhat > 1.5, "R-hat for diverged chains should be >> 1: {}", rhat);
    }

    #[test]
    fn test_ess_constant_chain() {
        // Constant chain -> ESS = N (no autocorrelation, but var=0 => ESS=N)
        let chain = vec![1.0; 100];
        let ess = chain_ess(&chain);
        assert!(ess >= 99.0, "ESS of constant chain should be ~N: {}", ess);
    }

    #[test]
    fn test_ess_iid_chain() {
        // IID samples -> ESS should be close to N
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let chain: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

        let ess = chain_ess(&chain);
        assert!(ess > 500.0, "ESS of IID chain should be close to N: {}", ess);
    }

    #[test]
    fn test_ess_correlated_chain() {
        // Highly correlated chain (random walk) -> ESS << N
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.01).unwrap();
        let mut chain = Vec::with_capacity(1000);
        let mut x = 0.0;
        for _ in 0..1000 {
            x += normal.sample(&mut rng);
            chain.push(x);
        }

        let ess = chain_ess(&chain);
        assert!(ess < 500.0, "ESS of correlated chain should be << N: {}", ess);
    }
}
