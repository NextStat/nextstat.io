//! Visual Predictive Check (VPC) and Goodness-of-Fit (GOF) diagnostics
//! for population PK models.
//!
//! # VPC
//!
//! Simulates `n_sim` replicates from the fitted model (θ, Ω, error model),
//! computes prediction intervals (PI) at each time bin, and returns
//! structured data for plotting observed vs simulated quantiles.
//!
//! # GOF
//!
//! Computes standard diagnostic quantities:
//! - **PRED** — population predictions (η = 0)
//! - **IPRED** — individual predictions (at conditional mode η̂)
//! - **IWRES** — individual weighted residuals: (DV − IPRED) / √Var
//! - **CWRES** — conditional weighted residuals (FOCE-based approximation)

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::foce::OmegaMatrix;
use crate::pk::{self, ErrorModel};

// ---------------------------------------------------------------------------
// GOF diagnostics
// ---------------------------------------------------------------------------

/// Goodness-of-fit diagnostic record for one observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GofRecord {
    /// Subject index.
    pub subject: usize,
    /// Time of observation.
    pub time: f64,
    /// Observed value (DV).
    pub dv: f64,
    /// Population prediction (η = 0).
    pub pred: f64,
    /// Individual prediction (at conditional mode η̂).
    pub ipred: f64,
    /// Individual weighted residual: (DV − IPRED) / σ(IPRED).
    pub iwres: f64,
    /// Conditional weighted residual (FOCE approximation).
    pub cwres: f64,
}

/// Compute GOF diagnostics for a fitted 1-compartment oral PK model.
///
/// # Arguments
/// - `times`, `y`, `subject_idx`: observation data
/// - `dose`, `bioav`: dosing
/// - `theta`: fitted population parameters `[CL, V, Ka]`
/// - `etas`: conditional modes per subject `eta[subject][param]`
/// - `error_model`: residual error model
pub fn gof_1cpt_oral(
    times: &[f64],
    y: &[f64],
    subject_idx: &[usize],
    dose: f64,
    bioav: f64,
    theta: &[f64],
    etas: &[Vec<f64>],
    error_model: &ErrorModel,
) -> Result<Vec<GofRecord>> {
    if theta.len() != 3 {
        return Err(Error::Validation("theta must have 3 elements".into()));
    }
    if times.len() != y.len() || times.len() != subject_idx.len() {
        return Err(Error::Validation("times/y/subject_idx length mismatch".into()));
    }

    let n_obs = times.len();
    let mut records = Vec::with_capacity(n_obs);

    for i in 0..n_obs {
        let s = subject_idx[i];
        let t = times[i];
        let dv = y[i];

        // Population prediction (eta = 0).
        let pred = pk::conc_oral(dose, bioav, theta[0], theta[1], theta[2], t);

        // Individual prediction (at conditional mode).
        let eta = &etas[s];
        let cl_i = theta[0] * eta[0].exp();
        let v_i = theta[1] * eta[1].exp();
        let ka_i = theta[2] * eta[2].exp();
        let ipred = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);

        // Residual variance at IPRED.
        let var = error_model.variance(ipred.max(1e-30));
        let sd = var.sqrt().max(1e-30);

        let iwres = (dv - ipred) / sd;

        // CWRES approximation: (DV − PRED) / √(Var_pop + H'ΩH)
        // Simplified: use PRED-based variance as denominator.
        let var_pred = error_model.variance(pred.max(1e-30));
        let cwres = (dv - pred) / var_pred.sqrt().max(1e-30);

        records.push(GofRecord { subject: s, time: t, dv, pred, ipred, iwres, cwres });
    }

    Ok(records)
}

// ---------------------------------------------------------------------------
// VPC
// ---------------------------------------------------------------------------

/// A single VPC time-bin summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpcBin {
    /// Bin center time.
    pub time: f64,
    /// Number of observed data points in this bin.
    pub n_obs: usize,
    /// Observed quantiles (e.g. 5th, 50th, 95th percentiles of DV).
    pub obs_quantiles: Vec<f64>,
    /// Simulated prediction interval lower bounds for each quantile.
    pub sim_pi_lower: Vec<f64>,
    /// Simulated prediction interval medians for each quantile.
    pub sim_pi_median: Vec<f64>,
    /// Simulated prediction interval upper bounds for each quantile.
    pub sim_pi_upper: Vec<f64>,
}

/// Result of a VPC analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpcResult {
    /// Time-binned VPC summary data.
    pub bins: Vec<VpcBin>,
    /// Quantile levels used (e.g. [0.05, 0.50, 0.95]).
    pub quantiles: Vec<f64>,
    /// Number of simulations performed.
    pub n_sim: usize,
}

/// Configuration for VPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpcConfig {
    /// Number of simulation replicates.
    pub n_sim: usize,
    /// Quantile levels to compute (default: [0.05, 0.50, 0.95]).
    pub quantiles: Vec<f64>,
    /// Number of time bins (default: 10).
    pub n_bins: usize,
    /// Random seed.
    pub seed: u64,
    /// Prediction interval level for simulated quantiles (default: 0.90 → 5th–95th).
    pub pi_level: f64,
}

impl Default for VpcConfig {
    fn default() -> Self {
        Self { n_sim: 200, quantiles: vec![0.05, 0.50, 0.95], n_bins: 10, seed: 42, pi_level: 0.90 }
    }
}

/// Run a VPC for a fitted 1-compartment oral PK model.
///
/// Simulates `n_sim` datasets by:
/// 1. Sampling η_i ~ N(0, Ω) for each subject
/// 2. Computing individual concentrations
/// 3. Adding residual noise per the error model
///
/// Then bins by time and computes observed vs simulated quantiles.
pub fn vpc_1cpt_oral(
    times: &[f64],
    y: &[f64],
    subject_idx: &[usize],
    n_subjects: usize,
    dose: f64,
    bioav: f64,
    theta: &[f64],
    omega: &OmegaMatrix,
    error_model: &ErrorModel,
    config: &VpcConfig,
) -> Result<VpcResult> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};

    if theta.len() != 3 {
        return Err(Error::Validation("theta must have 3 elements".into()));
    }
    if omega.dim() != 3 {
        return Err(Error::Validation("omega must be 3×3".into()));
    }
    if times.len() != y.len() || times.len() != subject_idx.len() {
        return Err(Error::Validation("times/y/subject_idx mismatch".into()));
    }
    if config.quantiles.is_empty() {
        return Err(Error::Validation("quantiles must not be empty".into()));
    }

    let n_obs = times.len();
    let n_q = config.quantiles.len();
    let chol = omega.cholesky();
    let n_eta = 3;

    // Determine time bins via equal-width binning.
    let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (t_max - t_min) / config.n_bins as f64;
    let bin_edges: Vec<f64> = (0..=config.n_bins).map(|i| t_min + i as f64 * bin_width).collect();
    let bin_centers: Vec<f64> =
        (0..config.n_bins).map(|i| 0.5 * (bin_edges[i] + bin_edges[i + 1])).collect();

    // Assign observations to bins.
    let mut obs_by_bin: Vec<Vec<f64>> = vec![Vec::new(); config.n_bins];
    for i in 0..n_obs {
        let bin = ((times[i] - t_min) / bin_width).floor() as usize;
        let bin = bin.min(config.n_bins - 1);
        obs_by_bin[bin].push(y[i]);
    }

    // Observed quantiles per bin.
    let obs_quantiles: Vec<Vec<f64>> = obs_by_bin
        .iter()
        .map(|vals| config.quantiles.iter().map(|&q| percentile(vals, q)).collect())
        .collect();

    // Simulate n_sim replicates and collect per-bin quantiles.
    // sim_quantiles[bin][quantile_idx] = Vec of simulated quantile values across replicates.
    let mut sim_quantiles: Vec<Vec<Vec<f64>>> =
        vec![vec![Vec::with_capacity(config.n_sim); n_q]; config.n_bins];

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let std_normal = RandNormal::new(0.0_f64, 1.0).unwrap();

    for _ in 0..config.n_sim {
        // Sample etas for each subject.
        let mut sim_etas: Vec<Vec<f64>> = Vec::with_capacity(n_subjects);
        for _ in 0..n_subjects {
            let z: Vec<f64> = (0..n_eta).map(|_| std_normal.sample(&mut rng)).collect();
            let mut eta = vec![0.0; n_eta];
            for ii in 0..n_eta {
                for jj in 0..=ii {
                    eta[ii] += chol[ii][jj] * z[jj];
                }
            }
            sim_etas.push(eta);
        }

        // Simulate observations.
        let mut sim_by_bin: Vec<Vec<f64>> = vec![Vec::new(); config.n_bins];
        for i in 0..n_obs {
            let s = subject_idx[i];
            let t = times[i];
            let eta = &sim_etas[s];
            let cl_i = theta[0] * eta[0].exp();
            let v_i = theta[1] * eta[1].exp();
            let ka_i = theta[2] * eta[2].exp();
            let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);

            // Add residual noise.
            let sd = error_model.variance(c.max(1e-30)).sqrt().max(1e-30);
            let sim_y = c + sd * std_normal.sample(&mut rng);

            let bin = ((t - t_min) / bin_width).floor() as usize;
            let bin = bin.min(config.n_bins - 1);
            sim_by_bin[bin].push(sim_y.max(0.0));
        }

        // Compute quantiles for this replicate.
        for b in 0..config.n_bins {
            for (qi, &q) in config.quantiles.iter().enumerate() {
                sim_quantiles[b][qi].push(percentile(&sim_by_bin[b], q));
            }
        }
    }

    // Summarize simulated quantiles into PI.
    let pi_lo = (1.0 - config.pi_level) / 2.0;
    let pi_hi = 1.0 - pi_lo;

    let mut bins = Vec::with_capacity(config.n_bins);
    for b in 0..config.n_bins {
        let mut pi_lower = Vec::with_capacity(n_q);
        let mut pi_median = Vec::with_capacity(n_q);
        let mut pi_upper = Vec::with_capacity(n_q);

        for qi in 0..n_q {
            pi_lower.push(percentile(&sim_quantiles[b][qi], pi_lo));
            pi_median.push(percentile(&sim_quantiles[b][qi], 0.50));
            pi_upper.push(percentile(&sim_quantiles[b][qi], pi_hi));
        }

        bins.push(VpcBin {
            time: bin_centers[b],
            n_obs: obs_by_bin[b].len(),
            obs_quantiles: obs_quantiles[b].clone(),
            sim_pi_lower: pi_lower,
            sim_pi_median: pi_median,
            sim_pi_upper: pi_upper,
        });
    }

    Ok(VpcResult { bins, quantiles: config.quantiles.clone(), n_sim: config.n_sim })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the q-th percentile of a slice (linear interpolation).
/// Returns 0.0 for empty slices.
fn percentile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if hi >= n { sorted[n - 1] } else { sorted[lo] * (1.0 - frac) + sorted[hi] * frac }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};

    fn generate_pop_data(
        n_subjects: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<Vec<f64>>) {
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let omega_sd = 0.2;
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let times_per = [0.5, 1.0, 2.0, 4.0, 8.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let eta_dist = RandNormal::new(0.0, omega_sd).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        let mut etas = Vec::new();

        for sid in 0..n_subjects {
            let eta: Vec<f64> = vec![
                eta_dist.sample(&mut rng),
                eta_dist.sample(&mut rng),
                eta_dist.sample(&mut rng),
            ];
            let cl_i = cl_pop * eta[0].exp();
            let v_i = v_pop * eta[1].exp();
            let ka_i = ka_pop * eta[2].exp();

            for &t in &times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
            etas.push(eta);
        }

        (times, y, subject_idx, etas)
    }

    #[test]
    fn gof_basic() {
        let n_subjects = 8;
        let (times, y, subject_idx, etas) = generate_pop_data(n_subjects, 42);
        let theta = [1.2, 15.0, 2.0];
        let em = ErrorModel::Additive(0.05);

        let records =
            gof_1cpt_oral(&times, &y, &subject_idx, 100.0, 1.0, &theta, &etas, &em).unwrap();

        assert_eq!(records.len(), times.len());
        for r in &records {
            assert!(r.pred.is_finite(), "PRED not finite");
            assert!(r.ipred.is_finite(), "IPRED not finite");
            assert!(r.iwres.is_finite(), "IWRES not finite");
            assert!(r.cwres.is_finite(), "CWRES not finite");
        }

        // IPRED should be closer to DV than PRED (individual fits).
        let ipred_mse: f64 =
            records.iter().map(|r| (r.dv - r.ipred).powi(2)).sum::<f64>() / records.len() as f64;
        let pred_mse: f64 =
            records.iter().map(|r| (r.dv - r.pred).powi(2)).sum::<f64>() / records.len() as f64;
        assert!(
            ipred_mse <= pred_mse * 1.5,
            "IPRED MSE {ipred_mse} much worse than PRED MSE {pred_mse}"
        );
    }

    #[test]
    fn gof_iwres_centered() {
        let n_subjects = 20;
        let (times, y, subject_idx, etas) = generate_pop_data(n_subjects, 77);
        let theta = [1.2, 15.0, 2.0];
        let em = ErrorModel::Additive(0.05);

        let records =
            gof_1cpt_oral(&times, &y, &subject_idx, 100.0, 1.0, &theta, &etas, &em).unwrap();

        let mean_iwres: f64 = records.iter().map(|r| r.iwres).sum::<f64>() / records.len() as f64;
        // IWRES should be roughly centered around 0 (not exactly, since etas are true).
        assert!(mean_iwres.abs() < 2.0, "mean IWRES should be near 0, got {mean_iwres}");
    }

    #[test]
    fn vpc_basic() {
        let n_subjects = 10;
        let (times, y, subject_idx, _etas) = generate_pop_data(n_subjects, 55);
        let theta = [1.2, 15.0, 2.0];
        let omega = OmegaMatrix::from_diagonal(&[0.2, 0.2, 0.2]).unwrap();
        let em = ErrorModel::Additive(0.05);

        let cfg = VpcConfig { n_sim: 50, n_bins: 5, seed: 42, ..VpcConfig::default() };

        let result = vpc_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            100.0,
            1.0,
            &theta,
            &omega,
            &em,
            &cfg,
        )
        .unwrap();

        assert_eq!(result.bins.len(), 5);
        assert_eq!(result.n_sim, 50);
        assert_eq!(result.quantiles.len(), 3);

        for bin in &result.bins {
            assert_eq!(bin.obs_quantiles.len(), 3);
            assert_eq!(bin.sim_pi_lower.len(), 3);
            assert_eq!(bin.sim_pi_median.len(), 3);
            assert_eq!(bin.sim_pi_upper.len(), 3);
            // PI lower ≤ median ≤ upper.
            for qi in 0..3 {
                assert!(
                    bin.sim_pi_lower[qi] <= bin.sim_pi_upper[qi] + 1e-10,
                    "PI lower > upper at bin t={}: {} > {}",
                    bin.time,
                    bin.sim_pi_lower[qi],
                    bin.sim_pi_upper[qi]
                );
            }
        }
    }

    #[test]
    fn vpc_obs_median_within_pi() {
        let n_subjects = 20;
        let (times, y, subject_idx, _) = generate_pop_data(n_subjects, 33);
        let theta = [1.2, 15.0, 2.0];
        let omega = OmegaMatrix::from_diagonal(&[0.2, 0.2, 0.2]).unwrap();
        let em = ErrorModel::Additive(0.05);

        let cfg = VpcConfig { n_sim: 100, n_bins: 5, seed: 42, ..VpcConfig::default() };

        let result = vpc_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            100.0,
            1.0,
            &theta,
            &omega,
            &em,
            &cfg,
        )
        .unwrap();

        // The observed median (quantile index 1) should fall within the
        // simulated PI for most bins when the model is correct.
        let mut n_within = 0;
        let mut n_total = 0;
        for bin in &result.bins {
            if bin.n_obs < 3 {
                continue;
            }
            n_total += 1;
            let obs_med = bin.obs_quantiles[1];
            if obs_med >= bin.sim_pi_lower[1] - 0.5 && obs_med <= bin.sim_pi_upper[1] + 0.5 {
                n_within += 1;
            }
        }
        assert!(
            n_total == 0 || n_within as f64 / n_total as f64 >= 0.5,
            "Observed median within PI in only {n_within}/{n_total} bins"
        );
    }

    #[test]
    fn percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&data, 0.5) - 3.0).abs() < 1e-10);
        assert!((percentile(&data, 1.0) - 5.0).abs() < 1e-10);
        assert!((percentile(&data, 0.25) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn percentile_empty() {
        assert!((percentile(&[], 0.5)).abs() < 1e-10);
    }

    #[test]
    fn vpc_validates_inputs() {
        let omega = OmegaMatrix::from_diagonal(&[0.2, 0.2, 0.2]).unwrap();
        let em = ErrorModel::Additive(0.05);
        let cfg = VpcConfig::default();

        let err = vpc_1cpt_oral(
            &[1.0],
            &[2.0],
            &[0],
            1,
            100.0,
            1.0,
            &[1.0, 10.0], // wrong length
            &omega,
            &em,
            &cfg,
        )
        .unwrap_err();
        assert!(err.to_string().contains("theta"));
    }
}
