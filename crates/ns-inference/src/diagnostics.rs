//! MCMC diagnostics: split R-hat, bulk ESS, tail ESS.
//!
//! This module implements:
//! - Split R-hat (Gelman et al.)
//! - Rank-normalized + folded split R-hat (Vehtari et al. 2021) for robustness
//! - Bulk ESS and tail ESS (lightweight approximations)

use std::fmt;

use statrs::distribution::{ContinuousCDF, Normal};

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
    /// E-BFMI per chain (energy Bayesian fraction of missing information).
    pub ebfmi: Vec<f64>,
}

/// High-level sampling quality status (non-slow gates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityStatus {
    /// All non-slow gates passed.
    Ok,
    /// Some non-slow gates emitted warnings.
    Warn,
    /// One or more non-slow gates failed.
    Fail,
}

impl fmt::Display for QualityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityStatus::Ok => write!(f, "ok"),
            QualityStatus::Warn => write!(f, "warn"),
            QualityStatus::Fail => write!(f, "fail"),
        }
    }
}

/// Thresholds for non-slow sampling quality gates.
///
/// These are deliberately conservative to avoid flakiness on short runs.
#[derive(Debug, Clone)]
pub struct QualityGates {
    /// Require at least this many chains before enabling R-hat/ESS gates.
    pub min_chains: usize,
    /// Require at least this many post-warmup draws per chain before enabling R-hat/ESS/E-BFMI gates.
    pub min_draws_per_chain: usize,

    /// Warn if divergence rate exceeds this threshold.
    pub max_divergence_rate_warn: f64,
    /// Fail if divergence rate exceeds this threshold.
    pub max_divergence_rate_fail: f64,

    /// Warn if max-treedepth rate exceeds this threshold.
    pub max_treedepth_rate_warn: f64,
    /// Fail if max-treedepth rate exceeds this threshold.
    pub max_treedepth_rate_fail: f64,

    /// Warn if max rank-normalized folded R-hat exceeds this threshold.
    pub max_rhat_warn: f64,
    /// Fail if max rank-normalized folded R-hat exceeds this threshold.
    pub max_rhat_fail: f64,

    /// Minimum bulk ESS as a fraction of total draws (n_chains * n_samples).
    pub min_ess_bulk_frac_warn: f64,
    /// Fail if bulk ESS falls below this fraction of total draws.
    pub min_ess_bulk_frac_fail: f64,

    /// Minimum E-BFMI per chain.
    pub min_ebfmi_warn: f64,
    /// Fail if E-BFMI falls below this threshold.
    pub min_ebfmi_fail: f64,
}

impl Default for QualityGates {
    fn default() -> Self {
        Self {
            min_chains: 2,
            min_draws_per_chain: 50,
            // Stan guidance + reproducibility.md: production wants <1%, tests can allow larger.
            max_divergence_rate_warn: 0.05,
            max_divergence_rate_fail: 0.20,
            max_treedepth_rate_warn: 0.05,
            max_treedepth_rate_fail: 0.20,
            // For short runs we use loose thresholds; strict gates live in slow tests.
            max_rhat_warn: 1.20,
            max_rhat_fail: 1.50,
            min_ess_bulk_frac_warn: 0.05,
            min_ess_bulk_frac_fail: 0.01,
            // Reproducibility.md uses >0.3; keep warn at 0.3 and fail at 0.2.
            min_ebfmi_warn: 0.30,
            min_ebfmi_fail: 0.20,
        }
    }
}

/// Summary of sampling run quality.
#[derive(Debug, Clone)]
pub struct QualitySummary {
    /// Aggregated status for the run.
    pub status: QualityStatus,
    /// Non-fatal issues (suggests longer warmup/samples or config tuning).
    pub warnings: Vec<String>,
    /// Hard failures (likely invalid or unusable sampling run).
    pub failures: Vec<String>,

    /// Whether R-hat/ESS/E-BFMI gates were enabled for this run.
    pub enabled: bool,
    /// Total post-warmup draws used for diagnostics.
    pub total_draws: usize,
    /// Max rank-normalized folded R-hat across parameters.
    pub max_r_hat: f64,
    /// Min bulk ESS across parameters.
    pub min_ess_bulk: f64,
    /// Min tail ESS across parameters.
    pub min_ess_tail: f64,
    /// Min E-BFMI across chains.
    pub min_ebfmi: f64,
}

/// Compute a conservative, non-slow quality summary for a sampler run.
pub fn quality_summary(
    diag: &DiagnosticsResult,
    n_chains: usize,
    n_samples: usize,
    gates: &QualityGates,
) -> QualitySummary {
    let total_draws = n_chains.saturating_mul(n_samples);
    let enabled = n_chains >= gates.min_chains && n_samples >= gates.min_draws_per_chain;

    let max_r_hat = diag
        .r_hat
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let min_ess_bulk = diag
        .ess_bulk
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let min_ess_tail = diag
        .ess_tail
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let min_ebfmi = diag
        .ebfmi
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);

    let mut warnings = Vec::new();
    let mut failures = Vec::new();

    // Always check basic finiteness.
    if !diag.divergence_rate.is_finite() {
        failures.push("divergence_rate_not_finite".to_string());
    }
    if !diag.max_treedepth_rate.is_finite() {
        failures.push("max_treedepth_rate_not_finite".to_string());
    }
    if max_r_hat == f64::NEG_INFINITY {
        failures.push("r_hat_missing".to_string());
    }
    if !max_r_hat.is_finite() {
        failures.push("r_hat_not_finite".to_string());
    }
    if min_ess_bulk == f64::INFINITY {
        failures.push("ess_bulk_missing".to_string());
    }
    if !min_ess_bulk.is_finite() {
        failures.push("ess_bulk_not_finite".to_string());
    }
    if min_ess_tail == f64::INFINITY {
        failures.push("ess_tail_missing".to_string());
    }
    if !min_ess_tail.is_finite() {
        failures.push("ess_tail_not_finite".to_string());
    }
    if min_ebfmi == f64::INFINITY {
        failures.push("ebfmi_missing".to_string());
    }
    if !min_ebfmi.is_finite() {
        warnings.push("ebfmi_not_finite".to_string());
    }

    // Divergences / treedepth are meaningful even for shorter runs.
    if diag.divergence_rate > gates.max_divergence_rate_fail {
        failures.push("divergence_rate_high".to_string());
    } else if diag.divergence_rate > gates.max_divergence_rate_warn {
        warnings.push("divergence_rate_high".to_string());
    }

    if diag.max_treedepth_rate > gates.max_treedepth_rate_fail {
        failures.push("max_treedepth_rate_high".to_string());
    } else if diag.max_treedepth_rate > gates.max_treedepth_rate_warn {
        warnings.push("max_treedepth_rate_high".to_string());
    }

    if !enabled {
        warnings.push("gates_disabled_short_run".to_string());
    } else {
        if max_r_hat > gates.max_rhat_fail {
            failures.push("r_hat_high".to_string());
        } else if max_r_hat > gates.max_rhat_warn {
            warnings.push("r_hat_high".to_string());
        }

        if total_draws > 0 {
            let warn_thr = gates.min_ess_bulk_frac_warn * (total_draws as f64);
            let fail_thr = gates.min_ess_bulk_frac_fail * (total_draws as f64);
            if min_ess_bulk < fail_thr {
                failures.push("ess_bulk_low".to_string());
            } else if min_ess_bulk < warn_thr {
                warnings.push("ess_bulk_low".to_string());
            }
            if min_ess_tail < fail_thr {
                failures.push("ess_tail_low".to_string());
            } else if min_ess_tail < warn_thr {
                warnings.push("ess_tail_low".to_string());
            }
        }

        if min_ebfmi < gates.min_ebfmi_fail {
            failures.push("ebfmi_low".to_string());
        } else if min_ebfmi < gates.min_ebfmi_warn {
            warnings.push("ebfmi_low".to_string());
        }
    }

    let status = if !failures.is_empty() {
        QualityStatus::Fail
    } else if !warnings.is_empty() {
        QualityStatus::Warn
    } else {
        QualityStatus::Ok
    };

    QualitySummary {
        status,
        warnings,
        failures,
        enabled,
        total_draws,
        max_r_hat,
        min_ess_bulk,
        min_ess_tail,
        min_ebfmi,
    }
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
    let mut half_chains_raw: Vec<&[f64]> = Vec::new();
    for chain in chains {
        let n = chain.len();
        if n < 4 {
            return f64::NAN;
        }
        let mid = n / 2;
        half_chains_raw.push(&chain[..mid]);
        half_chains_raw.push(&chain[mid..]);
    }

    let min_len = half_chains_raw.iter().map(|c| c.len()).min().unwrap_or(0);
    if min_len < 2 {
        return f64::NAN;
    }
    let half_chains: Vec<&[f64]> = half_chains_raw.iter().map(|c| &c[..min_len]).collect();

    let m = half_chains.len() as f64;
    let n = min_len as f64;

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

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let n = sorted.len() as f64;
    let pos = q * (n - 1.0);
    let i0 = pos.floor() as usize;
    let i1 = pos.ceil() as usize;
    if i0 == i1 {
        return sorted[i0];
    }
    let f = pos - i0 as f64;
    sorted[i0] * (1.0 - f) + sorted[i1] * f
}

fn median_all(chains: &[Vec<f64>]) -> f64 {
    let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    all.get(all.len() / 2).copied().unwrap_or(f64::NAN)
}

fn rank_normalize(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let normal = Normal::new(0.0, 1.0).expect("Normal(0,1) should be valid");
    let total: usize = chains.iter().map(|c| c.len()).sum();
    if total == 0 {
        return chains.to_vec();
    }

    let mut out: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();

    // Flatten draws with back-references.
    let mut flat: Vec<(f64, usize, usize)> = Vec::with_capacity(total);
    for (ci, chain) in chains.iter().enumerate() {
        for (ti, &x) in chain.iter().enumerate() {
            flat.push((x, ci, ti));
        }
    }

    // Sort by value; NaNs go to the end.
    flat.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Greater));

    // Assign average ranks for ties (1-based ranks).
    let n = flat.len();
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && flat[j].0 == flat[i].0 {
            j += 1;
        }

        let rank_lo = i as f64 + 1.0;
        let rank_hi = j as f64;
        let rank = 0.5 * (rank_lo + rank_hi);

        // Convert rank to normal quantile. Use (rank - 0.5)/N to avoid 0/1.
        let mut p = (rank - 0.5) / n as f64;
        p = p.clamp(1e-12, 1.0 - 1e-12);
        let z = normal.inverse_cdf(p);

        for k in i..j {
            let (_x, ci, ti) = flat[k];
            out[ci][ti] = z;
        }

        i = j;
    }

    out
}

fn r_hat_rank_normalized_folded(chains: &[Vec<f64>]) -> f64 {
    if chains.is_empty() || chains.iter().any(|c| c.len() < 4) {
        return f64::NAN;
    }

    let z = rank_normalize(chains);
    let z_refs: Vec<&[f64]> = z.iter().map(|c| c.as_slice()).collect();
    let r_rank = r_hat(&z_refs);

    let med = median_all(chains);
    let folded: Vec<Vec<f64>> =
        chains.iter().map(|c| c.iter().map(|&x| (x - med).abs()).collect()).collect();
    let z_fold = rank_normalize(&folded);
    let zf_refs: Vec<&[f64]> = z_fold.iter().map(|c| c.as_slice()).collect();
    let r_fold = r_hat(&zf_refs);

    r_rank.max(r_fold)
}

fn ess_bulk_rank_normalized(chains: &[Vec<f64>]) -> f64 {
    let z = rank_normalize(chains);
    let z_refs: Vec<&[f64]> = z.iter().map(|c| c.as_slice()).collect();
    ess_bulk(&z_refs)
}

fn ess_tail_quantile_indicators(chains: &[Vec<f64>]) -> f64 {
    let total: usize = chains.iter().map(|c| c.len()).sum();
    if total == 0 {
        return 0.0;
    }

    let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let q05 = quantile_sorted(&all, 0.05);
    let q95 = quantile_sorted(&all, 0.95);

    let lower: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|&x| if x <= q05 { 1.0 } else { 0.0 }).collect())
        .collect();
    let upper: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|&x| if x >= q95 { 1.0 } else { 0.0 }).collect())
        .collect();

    let lower_refs: Vec<&[f64]> = lower.iter().map(|c| c.as_slice()).collect();
    let upper_refs: Vec<&[f64]> = upper.iter().map(|c| c.as_slice()).collect();

    ess_bulk(&lower_refs).min(ess_bulk(&upper_refs))
}

fn split_chains_for_ess<'a>(chains: &[&'a [f64]]) -> Option<Vec<&'a [f64]>> {
    if chains.is_empty() {
        return None;
    }

    let mut halves: Vec<&[f64]> = Vec::with_capacity(chains.len() * 2);
    for c in chains {
        if c.len() < 4 {
            return None;
        }
        let mid = c.len() / 2;
        halves.push(&c[..mid]);
        halves.push(&c[mid..]);
    }

    let min_len = halves.iter().map(|c| c.len()).min().unwrap_or(0);
    if min_len < 4 {
        return None;
    }
    Some(halves.into_iter().map(|c| &c[..min_len]).collect())
}

fn mean_and_var(chain: &[f64]) -> (f64, f64) {
    let n = chain.len() as f64;
    let mean = chain.iter().sum::<f64>() / n;
    let var = chain.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n.max(2.0) - 1.0);
    (mean, var)
}

/// Compute effective sample size (ESS) using a multi-chain initial monotone
/// sequence estimator (Geyer) on autocorrelation estimates derived from the
/// variogram (Vehtari et al. 2021 / Stan-style).
pub fn ess_bulk(chains: &[&[f64]]) -> f64 {
    let split = match split_chains_for_ess(chains) {
        Some(v) => v,
        None => return 0.0,
    };

    let m = split.len();
    let n = split[0].len();
    let total_draws = (m * n) as f64;

    // Marginal posterior variance estimate (var_hat_plus).
    let means_vars: Vec<(f64, f64)> = split.iter().map(|c| mean_and_var(c)).collect();
    let means: Vec<f64> = means_vars.iter().map(|(mu, _)| *mu).collect();
    let vars: Vec<f64> = means_vars.iter().map(|(_, v)| *v).collect();

    let m_f = m as f64;
    let n_f = n as f64;
    let mean_all = means.iter().sum::<f64>() / m_f;
    let b = means.iter().map(|&mu| (mu - mean_all).powi(2)).sum::<f64>() * n_f / (m_f - 1.0);
    let w = vars.iter().sum::<f64>() / m_f;
    let var_hat_plus = (n_f - 1.0) / n_f * w + b / n_f;

    if !var_hat_plus.is_finite() || var_hat_plus < 1e-30 {
        return total_draws;
    }

    // Autocorrelation estimates via variogram:
    // rho_t = 1 - V_t / (2 * var_hat_plus)
    // where V_t is the mean squared difference at lag t across all chains.
    let max_lag = n - 1;
    let mut rho: Vec<f64> = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        let mut count = 0usize;
        for c in &split {
            for i in 0..(n - lag) {
                let d = c[i] - c[i + lag];
                sum += d * d;
                count += 1;
            }
        }
        if count == 0 {
            break;
        }
        let v = sum / (count as f64);
        let r = (1.0 - v / (2.0 * var_hat_plus)).clamp(-1.0, 1.0);
        rho.push(r);

        // Fast early stop: once the pair sum becomes negative we can stop building rho.
        // (Still applies monotone adjustment below.)
        if rho.len() >= 2 {
            let k = rho.len();
            if k % 2 == 0 {
                let gamma = rho[k - 2] + rho[k - 1];
                if gamma < 0.0 {
                    break;
                }
            }
        }
    }

    // Geyer initial monotone sequence estimator on paired sums.
    let mut gammas: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i + 1 < rho.len() {
        let g = rho[i] + rho[i + 1];
        if g < 0.0 {
            break;
        }
        gammas.push(g);
        i += 2;
    }

    for k in 1..gammas.len() {
        if gammas[k] > gammas[k - 1] {
            gammas[k] = gammas[k - 1];
        }
    }

    let mut tau = 1.0;
    for g in gammas {
        tau += 2.0 * g;
    }
    if !tau.is_finite() || tau <= 0.0 {
        return total_draws;
    }

    let ess = (total_draws / tau).clamp(1.0, total_draws);
    ess
}

/// Compute tail ESS as a conservative tail-mixing proxy.
///
/// This follows the common Stan/ArviZ convention: compute ESS on indicator chains
/// for the lower and upper tails, then take the minimum:
/// `min(ESS(I[x <= q05]), ESS(I[x >= q95]))`.
pub fn ess_tail(chains: &[&[f64]]) -> f64 {
    if chains.is_empty() {
        return 0.0;
    }

    let mut all: Vec<f64> = chains.iter().flat_map(|c| c.iter().copied()).collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let q05 = quantile_sorted(&all, 0.05);
    let q95 = quantile_sorted(&all, 0.95);

    let lower: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|&x| if x <= q05 { 1.0 } else { 0.0 }).collect())
        .collect();
    let upper: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|&x| if x >= q95 { 1.0 } else { 0.0 }).collect())
        .collect();

    let lower_refs: Vec<&[f64]> = lower.iter().map(|c| c.as_slice()).collect();
    let upper_refs: Vec<&[f64]> = upper.iter().map(|c| c.as_slice()).collect();

    ess_bulk(&lower_refs).min(ess_bulk(&upper_refs))
}

/// Compute full diagnostics for a SamplerResult.
pub fn compute_diagnostics(result: &crate::chain::SamplerResult) -> DiagnosticsResult {
    let n_params = result.param_names.len();

    let mut r_hat_vals = Vec::with_capacity(n_params);
    let mut ess_bulk_vals = Vec::with_capacity(n_params);
    let mut ess_tail_vals = Vec::with_capacity(n_params);

    for p in 0..n_params {
        let chain_draws: Vec<Vec<f64>> = result.param_draws(p);
        r_hat_vals.push(r_hat_rank_normalized_folded(&chain_draws));
        ess_bulk_vals.push(ess_bulk_rank_normalized(&chain_draws));
        ess_tail_vals.push(ess_tail_quantile_indicators(&chain_draws));
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
        // `tree_depths` uses 0-based depth (depth=0 means a single leapfrog step).
        // Hitting the cap means `depth >= max_treedepth`.
        .filter(|(d, max_d)| *d >= *max_d)
        .count();
    let max_treedepth_rate =
        if total_samples > 0 { n_max_depth as f64 / total_samples as f64 } else { 0.0 };

    let ebfmi: Vec<f64> = result.chains.iter().map(|c| ebfmi(&c.energies)).collect();

    DiagnosticsResult {
        r_hat: r_hat_vals,
        ess_bulk: ess_bulk_vals,
        ess_tail: ess_tail_vals,
        divergence_rate,
        max_treedepth_rate,
        ebfmi,
    }
}

/// Compute E-BFMI (energy Bayesian fraction of missing information) for one chain.
///
/// Definition (Stan): mean((E_t - E_{t-1})^2) / var(E_t).
pub fn ebfmi(energies: &[f64]) -> f64 {
    let n = energies.len();
    if n < 4 {
        return f64::NAN;
    }
    let mean: f64 = energies.iter().sum::<f64>() / n as f64;
    let var: f64 = energies.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    if var < 1e-30 {
        return f64::NAN;
    }

    let mut msd = 0.0;
    for i in 1..n {
        let d = energies[i] - energies[i - 1];
        msd += d * d;
    }
    msd /= n as f64 - 1.0;
    msd / var
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_treedepth_rate_counts_hits() {
        // Minimal synthetic SamplerResult: only tree_depths/max_treedepth are used.
        let mk_chain = |depths: Vec<usize>, max_td: usize| crate::chain::Chain {
            draws_unconstrained: vec![],
            draws_constrained: vec![],
            divergences: vec![false; depths.len()],
            tree_depths: depths,
            accept_probs: vec![],
            energies: vec![],
            max_treedepth: max_td,
            step_size: 0.0,
            mass_diag: vec![],
        };

        let r = crate::chain::SamplerResult {
            chains: vec![mk_chain(vec![0, 1, 2, 3, 3], 3)],
            param_names: vec!["x".to_string()],
            n_warmup: 0,
            n_samples: 5,
            diagnostics: None,
        };

        let d = compute_diagnostics(&r);
        // depths >= 3 are hits: 2 / 5.
        assert!((d.max_treedepth_rate - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_ebfmi_iid_energy_is_large() {
        // IID energy sequence should yield E-BFMI close to 2 (difference variance ~ 2*var).
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let energies: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
        let v = ebfmi(&energies);
        assert!(v.is_finite() && v > 0.5, "E-BFMI for IID should be comfortably > 0: {}", v);
    }

    #[test]
    fn test_rank_normalized_rhat_well_mixed() {
        // Two IID chains: rank-normalized R-hat should be close to 1.
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        let chain1: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng1)).collect();
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(2);
        let chain2: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng2)).collect();

        let rhat = r_hat_rank_normalized_folded(&vec![chain1, chain2]);
        assert!(rhat < 1.05, "Rank-normalized folded R-hat for IID chains should be ~1: {}", rhat);
    }

    #[test]
    fn test_rank_normalized_rhat_diverged_chains() {
        // Two chains at different means: rank-normalized folded R-hat should be >> 1.
        let chain1: Vec<f64> = (0..200).map(|i| i as f64 * 0.01).collect();
        let chain2: Vec<f64> = (0..200).map(|i| 10.0 + i as f64 * 0.01).collect();
        let rhat = r_hat_rank_normalized_folded(&vec![chain1, chain2]);
        assert!(rhat > 1.5, "R-hat for diverged chains should be >> 1: {}", rhat);
    }

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
        let ess = ess_bulk(&[&chain]);
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

        let ess = ess_bulk(&[&chain]);
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

        let ess = ess_bulk(&[&chain]);
        assert!(ess < 500.0, "ESS of correlated chain should be << N: {}", ess);
    }

    #[test]
    fn test_ess_tail_iid_chain_is_large() {
        // Tail ESS on IID draws should be close to N (indicator chains have no autocorrelation).
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let chain: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

        let ess = ess_tail(&[&chain]);
        assert!(ess > 500.0, "Tail ESS of IID chain should be large: {}", ess);
    }

    #[test]
    fn test_ess_tail_correlated_chain_is_smaller() {
        // Tail indicators on a random walk should show reduced ESS.
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let normal = Normal::new(0.0, 0.01).unwrap();
        let mut chain = Vec::with_capacity(1000);
        let mut x = 0.0;
        for _ in 0..1000 {
            x += normal.sample(&mut rng);
            chain.push(x);
        }

        let ess = ess_tail(&[&chain]);
        assert!(ess < 500.0, "Tail ESS of correlated chain should be reduced: {}", ess);
    }
}
