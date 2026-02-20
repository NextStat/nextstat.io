//! Monte Carlo clinical trial simulation engine.
//!
//! Provides tools for simulating virtual clinical trials using population
//! pharmacokinetic (PK) models:
//!
//! - **Virtual population generation**: log-normal between-subject variability
//!   with optional correlated random effects (Cholesky decomposition).
//! - **Single trial simulation**: PK profiles, residual error, and derived
//!   endpoints (AUC, Cmax, Tmax, Ctrough).
//! - **Monte Carlo simulation**: parallel (Rayon) execution of many virtual
//!   trials with Probability of Target Attainment (PTA) estimation.
//! - **Dose optimization**: binary search over dose levels to achieve a
//!   target PTA.
//!
//! ## PK models
//!
//! Concentrations are computed via analytical superposition through
//! [`DosingRegimen`] methods (1-compartment oral, 2-compartment IV,
//! 2-compartment oral).

use ns_core::{Error, Result};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal, StandardNormal};
use rayon::prelude::*;

use crate::dosing::{DoseEvent, DosingRegimen};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Population PK parameters for virtual population generation.
///
/// Subjects are generated with log-normal between-subject variability:
///   `theta_individual[i] = theta[i] * exp(eta_i)`, where `eta ~ N(0, omega^2)`.
#[derive(Debug, Clone)]
pub struct PopulationPkParams {
    /// Fixed effects (population typical values, e.g. `[CL, V, Ka]`).
    pub theta: Vec<f64>,
    /// Between-subject variability standard deviations (log-normal scale,
    /// e.g. `[omega_CL, omega_V, omega_Ka]`).
    pub omega: Vec<f64>,
    /// Residual unexplained variability standard deviation.
    pub sigma: f64,
    /// Observation error model type.
    pub error_model: ErrorModelType,
    /// Optional correlation matrix for random effects (lower triangle, row-major).
    ///
    /// Length must be `n*(n+1)/2` where `n = theta.len()`.
    /// If `None`, assumes diagonal (independent random effects).
    pub omega_correlation: Option<Vec<f64>>,
}

/// Observation error model type for trial simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorModelType {
    /// Additive: `y = f + eps`, `eps ~ N(0, sigma)`.
    Additive,
    /// Proportional: `y = f * (1 + eps)`, `eps ~ N(0, sigma)`.
    Proportional,
    /// Combined: additive + proportional weighted by `prop_fraction`.
    /// `Var(y|f) = (sigma * prop_fraction * f)^2 + (sigma * (1 - prop_fraction))^2`.
    Combined {
        /// Fraction of variance attributable to proportional component (0, 1).
        prop_fraction: f64,
    },
}

/// Pharmacokinetic model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkModelType {
    /// 1-compartment with first-order oral absorption.
    /// Parameters: `[CL, V, Ka]`.
    OneCompartmentOral,
    /// 2-compartment IV bolus.
    /// Parameters: `[CL, V1, V2, Q]`.
    TwoCompartmentIv,
    /// 2-compartment with first-order oral absorption.
    /// Parameters: `[CL, V1, V2, Q, Ka]`.
    TwoCompartmentOral,
}

/// Configuration for a single simulated trial.
#[derive(Debug, Clone)]
pub struct TrialConfig {
    /// Number of subjects per arm.
    pub n_subjects: usize,
    /// Dosing regimen events (applied to all subjects).
    pub dosing: Vec<DoseEvent>,
    /// Observation times (hours post first dose).
    pub obs_times: Vec<f64>,
    /// PK model type.
    pub pk_model: PkModelType,
    /// Population PK parameters.
    pub population: PopulationPkParams,
    /// Random seed.
    pub seed: u64,
}

/// Result of a single simulated trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Simulated concentration profiles: `subjects x timepoints`.
    pub concentrations: Vec<Vec<f64>>,
    /// Individual PK parameters: `subjects x n_params`.
    pub individual_params: Vec<Vec<f64>>,
    /// Derived PK endpoints per subject.
    pub endpoints: TrialEndpoints,
}

/// PK endpoints derived from simulated concentration-time profiles.
#[derive(Debug, Clone)]
pub struct TrialEndpoints {
    /// AUC(0-tau) for each subject (linear trapezoidal rule).
    pub auc: Vec<f64>,
    /// Cmax for each subject.
    pub cmax: Vec<f64>,
    /// Tmax for each subject (time of maximum concentration).
    pub tmax: Vec<f64>,
    /// Ctrough for each subject (concentration at last observation time).
    pub ctrough: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Monte Carlo types
// ---------------------------------------------------------------------------

/// Configuration for Monte Carlo simulation of multiple trials.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of virtual trials to simulate.
    pub n_trials: usize,
    /// Trial configuration template (seed is varied per trial).
    pub trial: TrialConfig,
    /// Target thresholds for Probability of Target Attainment.
    pub targets: Option<PtaTargets>,
    /// Number of Rayon threads (`0` = automatic).
    pub n_threads: usize,
}

/// Target thresholds for PTA (Probability of Target Attainment) evaluation.
#[derive(Debug, Clone, Copy)]
pub struct PtaTargets {
    /// Minimum AUC target (e.g. 400 mg*h/L). Computes `P(AUC > target)`.
    pub auc_min: Option<f64>,
    /// Maximum Cmax safety limit. Computes `P(Cmax < limit)`.
    pub cmax_max: Option<f64>,
    /// Minimum Ctrough efficacy target. Computes `P(Ctrough > target)`.
    pub ctrough_min: Option<f64>,
}

/// Result of Monte Carlo trial simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Number of trials simulated.
    pub n_trials: usize,
    /// Per-trial summary statistics.
    pub trial_summaries: Vec<TrialSummary>,
    /// Probability of Target Attainment results (if targets were specified).
    pub pta: Option<PtaResult>,
    /// Total wall time in seconds.
    pub wall_s: f64,
}

/// Summary statistics for a single simulated trial.
#[derive(Debug, Clone, Copy)]
pub struct TrialSummary {
    /// Mean AUC across subjects.
    pub mean_auc: f64,
    /// Median AUC across subjects.
    pub median_auc: f64,
    /// Mean Cmax across subjects.
    pub mean_cmax: f64,
    /// Median Cmax across subjects.
    pub median_cmax: f64,
    /// Number of subjects in the trial.
    pub n_subjects: usize,
}

/// Probability of Target Attainment results.
#[derive(Debug, Clone, Copy)]
pub struct PtaResult {
    /// Fraction of subjects (across all trials) meeting AUC target.
    pub pta_auc: Option<f64>,
    /// Fraction of subjects (across all trials) meeting Cmax safety limit.
    pub pta_cmax: Option<f64>,
    /// Fraction of subjects (across all trials) meeting Ctrough target.
    pub pta_ctrough: Option<f64>,
}

// ---------------------------------------------------------------------------
// Dose optimization types
// ---------------------------------------------------------------------------

/// Which PTA target to optimize against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtaTargetType {
    /// Optimize for `P(AUC > target)`.
    Auc,
    /// Optimize for `P(Cmax < target)`.
    Cmax,
    /// Optimize for `P(Ctrough > target)`.
    Ctrough,
}

/// Result of dose optimization.
#[derive(Debug, Clone)]
pub struct DoseOptResult {
    /// Optimal dose achieving the target PTA.
    pub optimal_dose: f64,
    /// PTA achieved at the optimal dose.
    pub achieved_pta: f64,
    /// Dose-PTA curve: `(dose, pta)` pairs evaluated during the search.
    pub dose_pta_curve: Vec<(f64, f64)>,
}

// ---------------------------------------------------------------------------
// Virtual population generation
// ---------------------------------------------------------------------------

/// Generate virtual subjects from population PK parameters.
///
/// Each subject receives individual parameters drawn as:
///   `theta_individual[i] = theta_pop[i] * exp(eta_i)`
/// where `eta ~ N(0, omega^2)` (log-normal distribution for PK parameters).
///
/// When `omega_correlation` is provided, correlated random effects are generated
/// via Cholesky decomposition: `eta = L * z` where `z ~ N(0, I)`.
///
/// # Arguments
/// * `population` - Population PK parameter specification.
/// * `n_subjects` - Number of virtual subjects to generate.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
/// `Vec<Vec<f64>>` of shape `[n_subjects][n_params]`.
pub fn generate_virtual_population(
    population: &PopulationPkParams,
    n_subjects: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n_params = population.theta.len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Build Cholesky factor L from correlation + omega SDs.
    let chol = build_cholesky_factor(population);

    let mut subjects = Vec::with_capacity(n_subjects);
    for _ in 0..n_subjects {
        // Draw z ~ N(0, I).
        let z: Vec<f64> = (0..n_params)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        // eta = L * z (correlated random effects scaled by omega SDs).
        let eta = match &chol {
            Some(l_matrix) => cholesky_multiply(l_matrix, &z),
            None => {
                // Diagonal case: eta_i = omega_i * z_i.
                z.iter()
                    .zip(&population.omega)
                    .map(|(&zi, oi)| oi * zi)
                    .collect()
            }
        };

        // Individual parameters: theta_i = theta_pop * exp(eta_i).
        let individual: Vec<f64> = population
            .theta
            .iter()
            .zip(&eta)
            .map(|(&theta_pop, &eta_i)| theta_pop * eta_i.exp())
            .collect();

        subjects.push(individual);
    }

    subjects
}

/// Build Cholesky factor L = diag(omega) * L_corr from correlation matrix.
///
/// `omega_correlation` is the lower triangle of the correlation matrix (row-major):
/// for `n` params, length = `n*(n+1)/2`, diagonal elements should be 1.0.
///
/// Returns `None` if no correlation matrix is provided (diagonal omega).
fn build_cholesky_factor(population: &PopulationPkParams) -> Option<Vec<Vec<f64>>> {
    let corr_lower = population.omega_correlation.as_ref()?;
    let n = population.omega.len();

    // Unpack lower triangle into full correlation matrix.
    let mut corr = vec![vec![0.0; n]; n];
    let mut idx = 0;
    for i in 0..n {
        for j in 0..=i {
            corr[i][j] = corr_lower[idx];
            corr[j][i] = corr_lower[idx];
            idx += 1;
        }
    }

    // Cholesky decomposition of correlation matrix.
    let mut l_corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = corr[i][j];
            for k in 0..j {
                s -= l_corr[i][k] * l_corr[j][k];
            }
            if i == j {
                l_corr[i][j] = s.max(0.0).sqrt();
            } else if l_corr[j][j].abs() > 1e-15 {
                l_corr[i][j] = s / l_corr[j][j];
            }
        }
    }

    // Scale by omega SDs: L = diag(omega) * L_corr.
    for i in 0..n {
        for j in 0..=i {
            l_corr[i][j] *= population.omega[i];
        }
    }

    Some(l_corr)
}

/// Multiply Cholesky factor L by vector z: result = L * z.
fn cholesky_multiply(l_matrix: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..=i {
            s += l_matrix[i][j] * z[j];
        }
        result[i] = s;
    }
    result
}

// ---------------------------------------------------------------------------
// Single trial simulation
// ---------------------------------------------------------------------------

/// Simulate a single clinical trial.
///
/// Workflow:
/// 1. Generate virtual population with between-subject variability.
/// 2. For each subject: compute PK concentration-time profile using individual
///    parameters and the dosing regimen via analytical superposition.
/// 3. Add residual error according to the specified error model.
/// 4. Compute derived PK endpoints (AUC, Cmax, Tmax, Ctrough).
///
/// # Errors
/// Returns an error if the dosing regimen is invalid or the PK model type
/// does not match the number of population parameters.
pub fn simulate_trial(config: &TrialConfig) -> Result<TrialResult> {
    // Validate parameter count vs model type.
    let expected_params = match config.pk_model {
        PkModelType::OneCompartmentOral => 3, // CL, V, Ka
        PkModelType::TwoCompartmentIv => 4,   // CL, V1, V2, Q
        PkModelType::TwoCompartmentOral => 5, // CL, V1, V2, Q, Ka
    };
    if config.population.theta.len() != expected_params {
        return Err(Error::Validation(format!(
            "PK model {:?} requires {} parameters, got {}",
            config.pk_model,
            expected_params,
            config.population.theta.len()
        )));
    }
    if config.population.omega.len() != expected_params {
        return Err(Error::Validation(format!(
            "omega length ({}) must match theta length ({})",
            config.population.omega.len(),
            expected_params
        )));
    }
    if config.obs_times.is_empty() {
        return Err(Error::Validation(
            "obs_times must be non-empty".to_string(),
        ));
    }
    if config.n_subjects == 0 {
        return Err(Error::Validation("n_subjects must be > 0".to_string()));
    }
    if config.dosing.is_empty() {
        return Err(Error::Validation(
            "dosing must contain at least one dose event".to_string(),
        ));
    }

    // Build dosing regimen.
    let regimen = DosingRegimen::from_events(config.dosing.clone())?;

    // Generate virtual population.
    let individual_params =
        generate_virtual_population(&config.population, config.n_subjects, config.seed);

    // Simulate concentration profiles per subject.
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed.wrapping_add(0x5A5A_5A5A));

    let mut concentrations = Vec::with_capacity(config.n_subjects);

    for params in &individual_params {
        // Compute clean (noise-free) concentrations.
        let clean = compute_pk_profile(&regimen, params, &config.obs_times, config.pk_model);

        // Add residual error.
        let noisy = add_residual_error(
            &clean,
            config.population.sigma,
            &config.population.error_model,
            &mut rng,
        );

        concentrations.push(noisy);
    }

    // Compute endpoints.
    let endpoints = compute_endpoints(&concentrations, &config.obs_times);

    Ok(TrialResult {
        concentrations,
        individual_params,
        endpoints,
    })
}

/// Compute PK concentration profile at observation times for a single subject.
fn compute_pk_profile(
    regimen: &DosingRegimen,
    params: &[f64],
    obs_times: &[f64],
    pk_model: PkModelType,
) -> Vec<f64> {
    match pk_model {
        PkModelType::OneCompartmentOral => {
            let (cl, v, ka) = (params[0], params[1], params[2]);
            regimen.predict_1cpt(cl, v, ka, obs_times)
        }
        PkModelType::TwoCompartmentIv => {
            let (cl, v1, v2, q) = (params[0], params[1], params[2], params[3]);
            regimen.predict_2cpt_iv(cl, v1, v2, q, obs_times)
        }
        PkModelType::TwoCompartmentOral => {
            let (cl, v1, v2, q, ka) = (params[0], params[1], params[2], params[3], params[4]);
            regimen.predict_2cpt_oral(cl, v1, v2, q, ka, obs_times)
        }
    }
}

/// Add residual error to a concentration profile.
fn add_residual_error(
    clean: &[f64],
    sigma: f64,
    error_model: &ErrorModelType,
    rng: &mut rand::rngs::StdRng,
) -> Vec<f64> {
    let normal = RandNormal::new(0.0, 1.0).unwrap();
    clean
        .iter()
        .map(|&f| {
            let eps: f64 = normal.sample(rng);
            let noisy = match error_model {
                ErrorModelType::Additive => f + sigma * eps,
                ErrorModelType::Proportional => f * (1.0 + sigma * eps),
                ErrorModelType::Combined { prop_fraction } => {
                    let sd_prop = sigma * prop_fraction * f;
                    let sd_add = sigma * (1.0 - prop_fraction);
                    f + (sd_prop * sd_prop + sd_add * sd_add).sqrt() * eps
                }
            };
            // Concentrations cannot be negative.
            noisy.max(0.0)
        })
        .collect()
}

/// Compute PK endpoints from simulated concentration-time profiles.
fn compute_endpoints(concentrations: &[Vec<f64>], obs_times: &[f64]) -> TrialEndpoints {
    let n_subjects = concentrations.len();
    let mut auc = Vec::with_capacity(n_subjects);
    let mut cmax = Vec::with_capacity(n_subjects);
    let mut tmax = Vec::with_capacity(n_subjects);
    let mut ctrough = Vec::with_capacity(n_subjects);

    for profile in concentrations {
        // AUC: linear trapezoidal rule.
        let mut auc_val = 0.0;
        for i in 1..profile.len() {
            let dt = obs_times[i] - obs_times[i - 1];
            auc_val += 0.5 * (profile[i - 1] + profile[i]) * dt;
        }
        auc.push(auc_val);

        // Cmax and Tmax: find maximum concentration.
        let mut max_c = f64::NEG_INFINITY;
        let mut max_t = 0.0;
        for (i, &c) in profile.iter().enumerate() {
            if c > max_c {
                max_c = c;
                max_t = obs_times[i];
            }
        }
        cmax.push(max_c);
        tmax.push(max_t);

        // Ctrough: concentration at last observation time.
        ctrough.push(*profile.last().unwrap_or(&0.0));
    }

    TrialEndpoints {
        auc,
        cmax,
        tmax,
        ctrough,
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo simulation
// ---------------------------------------------------------------------------

/// Run Monte Carlo simulation of multiple virtual trials with Rayon parallelism.
///
/// Each trial uses a deterministic seed derived from `trial.seed + trial_index`,
/// ensuring reproducibility. Results include per-trial summaries and optional
/// Probability of Target Attainment (PTA) computed across all subjects in all
/// trials.
///
/// # Thread pool
/// If `config.n_threads > 0`, a custom Rayon thread pool is used; otherwise
/// Rayon's global pool (automatic thread count) is used.
///
/// # Errors
/// Returns an error if any individual trial simulation fails.
pub fn simulate_trials(config: &MonteCarloConfig) -> Result<MonteCarloResult> {
    let start = std::time::Instant::now();

    let trial_indices: Vec<usize> = (0..config.n_trials).collect();

    let run_trials = |indices: &[usize]| -> Result<Vec<TrialResult>> {
        indices
            .par_iter()
            .map(|&i| {
                let mut trial_config = config.trial.clone();
                trial_config.seed = config.trial.seed.wrapping_add(i as u64);
                simulate_trial(&trial_config)
            })
            .collect()
    };

    // Run trials in parallel.
    let results = if config.n_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.n_threads)
            .build()
            .map_err(|e| Error::Computation(format!("failed to create thread pool: {e}")))?;
        pool.install(|| run_trials(&trial_indices))?
    } else {
        run_trials(&trial_indices)?
    };

    // Compute per-trial summaries.
    let trial_summaries: Vec<TrialSummary> = results.iter().map(trial_summary).collect();

    // Compute PTA if targets specified.
    let pta = config
        .targets
        .as_ref()
        .map(|targets| compute_pta(&results, targets));

    let wall_s = start.elapsed().as_secs_f64();

    Ok(MonteCarloResult {
        n_trials: config.n_trials,
        trial_summaries,
        pta,
        wall_s,
    })
}

/// Compute summary statistics for a single trial result.
fn trial_summary(result: &TrialResult) -> TrialSummary {
    let n = result.endpoints.auc.len();
    let mean_auc = mean(&result.endpoints.auc);
    let median_auc = median(&result.endpoints.auc);
    let mean_cmax = mean(&result.endpoints.cmax);
    let median_cmax = median(&result.endpoints.cmax);
    TrialSummary {
        mean_auc,
        median_auc,
        mean_cmax,
        median_cmax,
        n_subjects: n,
    }
}

/// Compute Probability of Target Attainment across all trials.
fn compute_pta(results: &[TrialResult], targets: &PtaTargets) -> PtaResult {
    let mut total_subjects = 0usize;
    let mut auc_hits = 0usize;
    let mut cmax_hits = 0usize;
    let mut ctrough_hits = 0usize;

    for result in results {
        let n = result.endpoints.auc.len();
        total_subjects += n;

        if let Some(auc_min) = targets.auc_min {
            auc_hits += result.endpoints.auc.iter().filter(|&&a| a > auc_min).count();
        }
        if let Some(cmax_max) = targets.cmax_max {
            cmax_hits += result
                .endpoints
                .cmax
                .iter()
                .filter(|&&c| c < cmax_max)
                .count();
        }
        if let Some(ctrough_min) = targets.ctrough_min {
            ctrough_hits += result
                .endpoints
                .ctrough
                .iter()
                .filter(|&&c| c > ctrough_min)
                .count();
        }
    }

    let frac = |hits: usize| -> f64 {
        if total_subjects > 0 {
            hits as f64 / total_subjects as f64
        } else {
            0.0
        }
    };

    PtaResult {
        pta_auc: targets.auc_min.map(|_| frac(auc_hits)),
        pta_cmax: targets.cmax_max.map(|_| frac(cmax_hits)),
        pta_ctrough: targets.ctrough_min.map(|_| frac(ctrough_hits)),
    }
}

// ---------------------------------------------------------------------------
// Dose optimization
// ---------------------------------------------------------------------------

/// Find the minimum dose achieving a target Probability of Target Attainment.
///
/// Evaluates `n_dose_levels` evenly spaced doses in `dose_range` and selects
/// the lowest dose whose PTA meets or exceeds `target_pta`. For each dose
/// level, a full Monte Carlo simulation is run.
///
/// # Arguments
/// * `base_config` - Monte Carlo configuration template (dose amount will be
///   overridden for each level).
/// * `dose_range` - `(min_dose, max_dose)` search range.
/// * `target_pta` - Target PTA probability in `[0, 1]`.
/// * `pta_target_type` - Which PTA metric to optimize (AUC, Cmax, or Ctrough).
/// * `n_dose_levels` - Number of evenly spaced dose levels to evaluate.
///
/// # Errors
/// Returns an error if `n_dose_levels < 2`, if `dose_range` is invalid, or if
/// any Monte Carlo simulation fails.
pub fn find_optimal_dose(
    base_config: &MonteCarloConfig,
    dose_range: (f64, f64),
    target_pta: f64,
    pta_target_type: PtaTargetType,
    n_dose_levels: usize,
) -> Result<DoseOptResult> {
    if n_dose_levels < 2 {
        return Err(Error::Validation(format!(
            "n_dose_levels must be >= 2, got {}",
            n_dose_levels
        )));
    }
    if dose_range.0 >= dose_range.1 {
        return Err(Error::Validation(format!(
            "dose_range.0 ({}) must be < dose_range.1 ({})",
            dose_range.0, dose_range.1
        )));
    }
    if !(0.0..=1.0).contains(&target_pta) {
        return Err(Error::Validation(format!(
            "target_pta must be in [0, 1], got {}",
            target_pta
        )));
    }

    // Generate evenly spaced dose levels.
    let step = (dose_range.1 - dose_range.0) / (n_dose_levels - 1) as f64;
    let dose_levels: Vec<f64> = (0..n_dose_levels)
        .map(|i| dose_range.0 + step * i as f64)
        .collect();

    let mut dose_pta_curve = Vec::with_capacity(n_dose_levels);
    let mut optimal_dose = dose_range.1;
    let mut achieved_pta = 0.0;

    for &dose in &dose_levels {
        // Clone config and override dose amounts.
        let mut mc_config = base_config.clone();
        for event in &mut mc_config.trial.dosing {
            event.amount = dose;
        }

        let result = simulate_trials(&mc_config)?;

        let pta_val = extract_pta(&result, pta_target_type);
        dose_pta_curve.push((dose, pta_val));

        // Track first dose meeting the target.
        if pta_val >= target_pta && dose < optimal_dose {
            optimal_dose = dose;
            achieved_pta = pta_val;
        }
    }

    // If no dose met the target, report the highest PTA achieved.
    if achieved_pta == 0.0 && !dose_pta_curve.is_empty() {
        let best = dose_pta_curve
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        optimal_dose = best.0;
        achieved_pta = best.1;
    }

    Ok(DoseOptResult {
        optimal_dose,
        achieved_pta,
        dose_pta_curve,
    })
}

/// Extract the relevant PTA value from a Monte Carlo result.
fn extract_pta(result: &MonteCarloResult, target_type: PtaTargetType) -> f64 {
    let pta = match &result.pta {
        Some(p) => p,
        None => return 0.0,
    };
    match target_type {
        PtaTargetType::Auc => pta.pta_auc.unwrap_or(0.0),
        PtaTargetType::Cmax => pta.pta_cmax.unwrap_or(0.0),
        PtaTargetType::Ctrough => pta.pta_ctrough.unwrap_or(0.0),
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Arithmetic mean of a slice.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Median of a slice (sorts a copy).
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dosing::{DoseEvent, DoseRoute};

    /// Helper: create a standard 1-cpt oral trial configuration.
    fn standard_1cpt_config(n_subjects: usize, seed: u64) -> TrialConfig {
        TrialConfig {
            n_subjects,
            dosing: vec![DoseEvent {
                time: 0.0,
                amount: 500.0,
                route: DoseRoute::Oral {
                    bioavailability: 0.8,
                },
            }],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0],
            pk_model: PkModelType::OneCompartmentOral,
            population: PopulationPkParams {
                theta: vec![10.0, 50.0, 1.5], // CL=10 L/h, V=50 L, Ka=1.5 1/h
                omega: vec![0.3, 0.25, 0.2],  // ~30%, 25%, 20% CV
                sigma: 0.1,
                error_model: ErrorModelType::Proportional,
                omega_correlation: None,
            },
            seed,
        }
    }

    #[test]
    fn test_generate_population() {
        let pop = PopulationPkParams {
            theta: vec![10.0, 50.0, 1.5],
            omega: vec![0.3, 0.25, 0.2],
            sigma: 0.1,
            error_model: ErrorModelType::Additive,
            omega_correlation: None,
        };
        let subjects = generate_virtual_population(&pop, 1000, 42);
        assert_eq!(subjects.len(), 1000);

        // Each subject should have 3 parameters.
        for s in &subjects {
            assert_eq!(s.len(), 3);
            // All parameters must be positive (log-normal).
            for &val in s {
                assert!(val > 0.0, "individual parameter must be > 0, got {val}");
            }
        }

        // Population means should approximate theta (law of large numbers).
        let mean_cl: f64 = subjects.iter().map(|s| s[0]).sum::<f64>() / 1000.0;
        let mean_v: f64 = subjects.iter().map(|s| s[1]).sum::<f64>() / 1000.0;

        // Log-normal mean = theta * exp(omega^2 / 2).
        // For omega=0.3, correction factor = exp(0.045) ~ 1.046.
        // Allow 15% tolerance for 1000 subjects.
        let expected_cl = 10.0 * (0.3_f64.powi(2) / 2.0).exp();
        assert!(
            (mean_cl - expected_cl).abs() / expected_cl < 0.15,
            "mean CL = {mean_cl}, expected ~ {expected_cl}"
        );
        let expected_v = 50.0 * (0.25_f64.powi(2) / 2.0).exp();
        assert!(
            (mean_v - expected_v).abs() / expected_v < 0.15,
            "mean V = {mean_v}, expected ~ {expected_v}"
        );
    }

    #[test]
    fn test_simulate_single_trial() {
        let config = standard_1cpt_config(50, 123);
        let result = simulate_trial(&config).unwrap();

        assert_eq!(result.concentrations.len(), 50);
        assert_eq!(result.individual_params.len(), 50);
        assert_eq!(result.endpoints.auc.len(), 50);
        assert_eq!(result.endpoints.cmax.len(), 50);
        assert_eq!(result.endpoints.tmax.len(), 50);
        assert_eq!(result.endpoints.ctrough.len(), 50);

        // Concentrations shape: each subject has obs_times.len() timepoints.
        for profile in &result.concentrations {
            assert_eq!(profile.len(), 8);
        }

        // AUC, Cmax must be positive.
        for &a in &result.endpoints.auc {
            assert!(a > 0.0, "AUC must be > 0, got {a}");
        }
        for &c in &result.endpoints.cmax {
            assert!(c > 0.0, "Cmax must be > 0, got {c}");
        }
        // Tmax must be within observation range.
        for &t in &result.endpoints.tmax {
            assert!(t >= 0.5 && t <= 24.0, "Tmax out of range: {t}");
        }
    }

    #[test]
    fn test_monte_carlo_deterministic() {
        let mc_config = MonteCarloConfig {
            n_trials: 10,
            trial: standard_1cpt_config(20, 999),
            targets: Some(PtaTargets {
                auc_min: Some(100.0),
                cmax_max: None,
                ctrough_min: None,
            }),
            n_threads: 0,
        };

        let r1 = simulate_trials(&mc_config).unwrap();
        let r2 = simulate_trials(&mc_config).unwrap();

        // Same seed should give identical summaries.
        assert_eq!(r1.n_trials, r2.n_trials);
        for (s1, s2) in r1.trial_summaries.iter().zip(&r2.trial_summaries) {
            assert!(
                (s1.mean_auc - s2.mean_auc).abs() < 1e-12,
                "mean_auc mismatch: {} vs {}",
                s1.mean_auc,
                s2.mean_auc
            );
            assert!(
                (s1.mean_cmax - s2.mean_cmax).abs() < 1e-12,
                "mean_cmax mismatch: {} vs {}",
                s1.mean_cmax,
                s2.mean_cmax
            );
        }

        // PTA values must also match.
        let p1 = r1.pta.unwrap();
        let p2 = r2.pta.unwrap();
        assert!((p1.pta_auc.unwrap() - p2.pta_auc.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_pta_all_above() {
        // Very low AUC target: almost all subjects should exceed it.
        let mc_config = MonteCarloConfig {
            n_trials: 5,
            trial: standard_1cpt_config(100, 42),
            targets: Some(PtaTargets {
                auc_min: Some(0.01), // trivially low target
                cmax_max: None,
                ctrough_min: None,
            }),
            n_threads: 0,
        };

        let result = simulate_trials(&mc_config).unwrap();
        let pta = result.pta.unwrap();
        assert!(
            pta.pta_auc.unwrap() > 0.99,
            "PTA for trivially low target should be ~1.0, got {}",
            pta.pta_auc.unwrap()
        );
    }

    #[test]
    fn test_pta_all_below() {
        // Very high AUC target: almost no subjects should exceed it.
        let mc_config = MonteCarloConfig {
            n_trials: 5,
            trial: standard_1cpt_config(100, 42),
            targets: Some(PtaTargets {
                auc_min: Some(1e9), // impossibly high target
                cmax_max: None,
                ctrough_min: None,
            }),
            n_threads: 0,
        };

        let result = simulate_trials(&mc_config).unwrap();
        let pta = result.pta.unwrap();
        assert!(
            pta.pta_auc.unwrap() < 0.01,
            "PTA for impossibly high target should be ~0.0, got {}",
            pta.pta_auc.unwrap()
        );
    }

    #[test]
    fn test_dose_optimization() {
        // Higher dose should give higher AUC PTA (monotonic for oral 1-cpt).
        let trial = standard_1cpt_config(50, 42);
        let mc_config = MonteCarloConfig {
            n_trials: 5,
            trial,
            targets: Some(PtaTargets {
                auc_min: Some(100.0),
                cmax_max: None,
                ctrough_min: None,
            }),
            n_threads: 0,
        };

        let result =
            find_optimal_dose(&mc_config, (100.0, 1000.0), 0.80, PtaTargetType::Auc, 5).unwrap();

        // Dose-PTA curve should be generally monotonically increasing.
        assert!(!result.dose_pta_curve.is_empty());
        assert!(result.optimal_dose >= 100.0 && result.optimal_dose <= 1000.0);

        // Verify monotonicity (allowing small noise from MC sampling).
        let ptas: Vec<f64> = result.dose_pta_curve.iter().map(|(_, p)| *p).collect();
        // First dose PTA should be <= last dose PTA.
        assert!(
            ptas.first().unwrap() <= ptas.last().unwrap(),
            "PTA should increase with dose: {:?}",
            ptas
        );
    }

    #[test]
    fn test_correlated_random_effects() {
        let pop = PopulationPkParams {
            theta: vec![10.0, 50.0, 1.5],
            omega: vec![0.3, 0.25, 0.2],
            sigma: 0.1,
            error_model: ErrorModelType::Additive,
            // Lower triangle of 3x3 correlation: [[1], [0.5, 1], [0, 0, 1]]
            // CL-V correlation = 0.5, Ka independent.
            omega_correlation: Some(vec![1.0, 0.5, 1.0, 0.0, 0.0, 1.0]),
        };

        let subjects = generate_virtual_population(&pop, 5000, 77);
        assert_eq!(subjects.len(), 5000);

        // Check that CL and V are positively correlated.
        let log_cl: Vec<f64> = subjects.iter().map(|s| s[0].ln()).collect();
        let log_v: Vec<f64> = subjects.iter().map(|s| s[1].ln()).collect();
        let corr = pearson_correlation(&log_cl, &log_v);
        assert!(
            corr > 0.3,
            "CL-V correlation should be positive (~0.5), got {corr}"
        );

        // Ka should be roughly independent of CL.
        let log_ka: Vec<f64> = subjects.iter().map(|s| s[2].ln()).collect();
        let corr_cl_ka = pearson_correlation(&log_cl, &log_ka);
        assert!(
            corr_cl_ka.abs() < 0.15,
            "CL-Ka correlation should be ~0, got {corr_cl_ka}"
        );
    }

    #[test]
    fn test_2cpt_iv_trial() {
        let config = TrialConfig {
            n_subjects: 30,
            dosing: vec![DoseEvent {
                time: 0.0,
                amount: 100.0,
                route: DoseRoute::IvBolus,
            }],
            obs_times: vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            pk_model: PkModelType::TwoCompartmentIv,
            population: PopulationPkParams {
                theta: vec![5.0, 20.0, 40.0, 3.0], // CL, V1, V2, Q
                omega: vec![0.3, 0.25, 0.3, 0.2],
                sigma: 0.05,
                error_model: ErrorModelType::Proportional,
                omega_correlation: None,
            },
            seed: 42,
        };

        let result = simulate_trial(&config).unwrap();
        assert_eq!(result.concentrations.len(), 30);
        for &a in &result.endpoints.auc {
            assert!(a > 0.0);
        }
    }

    #[test]
    fn test_2cpt_oral_trial() {
        let config = TrialConfig {
            n_subjects: 30,
            dosing: vec![DoseEvent {
                time: 0.0,
                amount: 200.0,
                route: DoseRoute::Oral {
                    bioavailability: 0.7,
                },
            }],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0],
            pk_model: PkModelType::TwoCompartmentOral,
            population: PopulationPkParams {
                theta: vec![5.0, 20.0, 40.0, 3.0, 1.2], // CL, V1, V2, Q, Ka
                omega: vec![0.3, 0.25, 0.3, 0.2, 0.15],
                sigma: 0.08,
                error_model: ErrorModelType::Combined {
                    prop_fraction: 0.6,
                },
                omega_correlation: None,
            },
            seed: 42,
        };

        let result = simulate_trial(&config).unwrap();
        assert_eq!(result.concentrations.len(), 30);
        for &a in &result.endpoints.auc {
            assert!(a > 0.0);
        }
    }

    /// Pearson correlation coefficient (test helper).
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for (xi, yi) in x.iter().zip(y) {
            let dx = xi - mx;
            let dy = yi - my;
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        cov / (vx * vy).sqrt()
    }
}
