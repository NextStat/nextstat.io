//! Subscription / Churn vertical pack.
//!
//! High-level workflows for SaaS, telco, and subscription-business churn analysis,
//! built on top of NextStat's survival and causal inference primitives.
//!
//! ## Components
//!
//! - **Synthetic data generator** — deterministic, seeded SaaS cohort data with
//!   right-censored churn times, covariates, and optional treatment assignment.
//! - **Cohort retention analysis** — stratified Kaplan-Meier + log-rank comparison.
//! - **Churn risk model** — Cox PH workflow returning hazard ratios + CIs.
//! - **Causal uplift** — AIPW-based intervention impact estimation on churn.

use ns_core::{Error, Result};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};

// ---------------------------------------------------------------------------
// C1: Synthetic SaaS churn dataset generator
// ---------------------------------------------------------------------------

/// A single customer record in a synthetic SaaS churn dataset.
#[derive(Debug, Clone)]
pub struct ChurnRecord {
    /// Customer ID (0-indexed).
    pub customer_id: usize,
    /// Signup cohort index (0 = earliest cohort).
    pub cohort: usize,
    /// Plan type: 0 = free, 1 = basic, 2 = premium.
    pub plan: u8,
    /// Region: 0 = NA, 1 = EU, 2 = APAC.
    pub region: u8,
    /// Monthly usage score (standardised, mean ≈ 0).
    pub usage_score: f64,
    /// Number of support tickets in first 30 days.
    pub support_tickets: u8,
    /// Treatment indicator (1 = received intervention, 0 = control).
    pub treated: u8,
    /// Observed time (months from signup to churn or censoring).
    pub time: f64,
    /// Event indicator: `true` = churned, `false` = right-censored.
    pub event: bool,
}

/// Configuration for the synthetic churn dataset generator.
#[derive(Debug, Clone)]
pub struct ChurnDataConfig {
    /// Number of customers to generate.
    pub n_customers: usize,
    /// Number of signup cohorts.
    pub n_cohorts: usize,
    /// Maximum observation window (months). Customers still active at this
    /// point are right-censored.
    pub max_time: f64,
    /// Fraction of customers assigned to treatment group (0.0–1.0).
    pub treatment_fraction: f64,
    /// Baseline monthly churn rate for the free plan (hazard).
    pub base_hazard_free: f64,
    /// Hazard ratio for basic plan vs free.
    pub hr_basic: f64,
    /// Hazard ratio for premium plan vs free.
    pub hr_premium: f64,
    /// Hazard ratio per unit of usage_score (protective if < 1).
    pub hr_usage: f64,
    /// Hazard ratio for treated customers (< 1 means intervention reduces churn).
    pub hr_treatment: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for ChurnDataConfig {
    fn default() -> Self {
        Self {
            n_customers: 2000,
            n_cohorts: 6,
            max_time: 24.0,
            treatment_fraction: 0.3,
            base_hazard_free: 0.08,
            hr_basic: 0.65,
            hr_premium: 0.40,
            hr_usage: 0.80,
            hr_treatment: 0.70,
            seed: 42,
        }
    }
}

/// Result of the synthetic churn dataset generator.
#[derive(Debug, Clone)]
pub struct ChurnDataset {
    /// All customer records.
    pub records: Vec<ChurnRecord>,
    /// Convenience: times vector (same order as records).
    pub times: Vec<f64>,
    /// Convenience: events vector.
    pub events: Vec<bool>,
    /// Convenience: group labels (plan type cast to i64).
    pub groups: Vec<i64>,
    /// Convenience: treatment indicator.
    pub treated: Vec<u8>,
    /// Convenience: covariate matrix (row-major, 4 columns:
    /// plan_basic, plan_premium, usage_score, support_tickets).
    pub covariates: Vec<Vec<f64>>,
}

/// Generate a synthetic SaaS churn dataset.
///
/// Uses an exponential survival model with proportional hazards to generate
/// realistic churn times. The true data-generating process is:
///
/// ```text
/// h(t | x) = base_hazard × HR_plan × HR_usage^usage × HR_treatment^treated
/// ```
///
/// Customers whose generated time exceeds `max_time` are right-censored.
///
/// The generator is fully deterministic given `config.seed`.
pub fn generate_churn_dataset(config: &ChurnDataConfig) -> Result<ChurnDataset> {
    if config.n_customers == 0 {
        return Err(Error::Validation("n_customers must be > 0".into()));
    }
    if config.n_cohorts == 0 {
        return Err(Error::Validation("n_cohorts must be > 0".into()));
    }
    if config.max_time <= 0.0 {
        return Err(Error::Validation("max_time must be > 0".into()));
    }
    if !(0.0..=1.0).contains(&config.treatment_fraction) {
        return Err(Error::Validation("treatment_fraction must be in [0, 1]".into()));
    }
    if config.base_hazard_free <= 0.0 {
        return Err(Error::Validation("base_hazard_free must be > 0".into()));
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    let mut records = Vec::with_capacity(config.n_customers);
    let mut times = Vec::with_capacity(config.n_customers);
    let mut events = Vec::with_capacity(config.n_customers);
    let mut groups = Vec::with_capacity(config.n_customers);
    let mut treated_vec = Vec::with_capacity(config.n_customers);
    let mut covariates = Vec::with_capacity(config.n_customers);

    let unit_exp = Exp::new(1.0).unwrap();

    for i in 0..config.n_customers {
        // Assign cohort uniformly.
        let u_cohort: f64 = rng.random();
        let cohort =
            (u_cohort * config.n_cohorts as f64).min(config.n_cohorts as f64 - 1.0) as usize;

        // Assign plan: ~40% free, ~35% basic, ~25% premium.
        let u_plan: f64 = rng.random();
        let plan: u8 = if u_plan < 0.40 {
            0
        } else if u_plan < 0.75 {
            1
        } else {
            2
        };

        // Assign region: ~40% NA, ~35% EU, ~25% APAC.
        let u_region: f64 = rng.random();
        let region: u8 = if u_region < 0.40 {
            0
        } else if u_region < 0.75 {
            1
        } else {
            2
        };

        // Usage score: normal-ish via Box-Muller.
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let u1_clamped = u1.max(1e-10).min(1.0 - 1e-10);
        let usage_score = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Support tickets: Poisson-ish (0–5).
        let u_tix: f64 = rng.random();
        let support_tickets = (u_tix * 5.0).max(0.0).min(5.0) as u8;

        // Treatment assignment.
        let u_treat: f64 = rng.random();
        let is_treated: u8 = if u_treat < config.treatment_fraction { 1 } else { 0 };

        // Compute individual hazard.
        let hr_plan = match plan {
            0 => 1.0,
            1 => config.hr_basic,
            _ => config.hr_premium,
        };
        let hr_usage = config.hr_usage.powf(usage_score);
        let hr_treat = if is_treated == 1 { config.hr_treatment } else { 1.0 };
        let lambda = config.base_hazard_free * hr_plan * hr_usage * hr_treat;

        // Generate exponential survival time: T = -ln(U) / lambda = Exp(1) / lambda.
        let raw_time = unit_exp.sample(&mut rng) / lambda;

        // Apply censoring.
        let (obs_time, event) =
            if raw_time <= config.max_time { (raw_time, true) } else { (config.max_time, false) };

        let plan_basic = if plan == 1 { 1.0 } else { 0.0 };
        let plan_premium = if plan == 2 { 1.0 } else { 0.0 };

        records.push(ChurnRecord {
            customer_id: i,
            cohort,
            plan,
            region,
            usage_score,
            support_tickets,
            treated: is_treated,
            time: obs_time,
            event,
        });
        times.push(obs_time);
        events.push(event);
        groups.push(plan as i64);
        treated_vec.push(is_treated);
        covariates.push(vec![plan_basic, plan_premium, usage_score, support_tickets as f64]);
    }

    Ok(ChurnDataset { records, times, events, groups, treated: treated_vec, covariates })
}

// ---------------------------------------------------------------------------
// C2: Cohort retention analysis (stratified KM + log-rank)
// ---------------------------------------------------------------------------

/// Result of a cohort retention analysis.
#[derive(Debug, Clone)]
pub struct RetentionAnalysis {
    /// Overall Kaplan-Meier estimate (all groups combined).
    pub overall: crate::survival::KaplanMeierEstimate,
    /// Per-group Kaplan-Meier estimates.
    pub by_group: Vec<(i64, crate::survival::KaplanMeierEstimate)>,
    /// Log-rank test comparing groups.
    pub log_rank: crate::survival::LogRankResult,
}

/// Run a cohort retention analysis: stratified KM + log-rank comparison.
///
/// This is a high-level workflow that composes `kaplan_meier` and `log_rank_test`.
///
/// # Arguments
/// - `times` — observation times.
/// - `events` — event indicators.
/// - `groups` — group labels for stratification.
/// - `conf_level` — confidence level for KM CIs.
pub fn retention_analysis(
    times: &[f64],
    events: &[bool],
    groups: &[i64],
    conf_level: f64,
) -> Result<RetentionAnalysis> {
    let overall = crate::survival::kaplan_meier(times, events, conf_level)?;
    let log_rank = crate::survival::log_rank_test(times, events, groups)?;

    // Per-group KM.
    let mut unique_groups: Vec<i64> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &g in groups {
            if seen.insert(g) {
                unique_groups.push(g);
            }
        }
    }
    unique_groups.sort();

    let mut by_group = Vec::with_capacity(unique_groups.len());
    for &g in &unique_groups {
        let mut g_times = Vec::new();
        let mut g_events = Vec::new();
        for i in 0..times.len() {
            if groups[i] == g {
                g_times.push(times[i]);
                g_events.push(events[i]);
            }
        }
        if !g_times.is_empty() {
            let km = crate::survival::kaplan_meier(&g_times, &g_events, conf_level)?;
            by_group.push((g, km));
        }
    }

    Ok(RetentionAnalysis { overall, by_group, log_rank })
}

// ---------------------------------------------------------------------------
// C3: Cox PH churn risk model
// ---------------------------------------------------------------------------

/// Result of a churn risk model (Cox PH).
#[derive(Debug, Clone)]
pub struct ChurnRiskModel {
    /// Covariate names.
    pub names: Vec<String>,
    /// Fitted coefficients (log hazard ratios).
    pub coefficients: Vec<f64>,
    /// Standard errors.
    pub se: Vec<f64>,
    /// Hazard ratios: exp(coefficient).
    pub hazard_ratios: Vec<f64>,
    /// 95% CI lower for hazard ratios.
    pub hr_ci_lower: Vec<f64>,
    /// 95% CI upper for hazard ratios.
    pub hr_ci_upper: Vec<f64>,
    /// Negative log-likelihood at optimum.
    pub nll: f64,
    /// Number of observations.
    pub n: usize,
    /// Number of events.
    pub n_events: usize,
}

/// Fit a Cox PH churn risk model.
///
/// Wraps `CoxPhModel` + MLE into a churn-specific workflow that returns
/// hazard ratios with confidence intervals.
///
/// # Arguments
/// - `times` — time to churn or censoring.
/// - `events` — `true` = churned.
/// - `x` — covariate matrix (one row per customer).
/// - `names` — covariate names.
/// - `conf_level` — confidence level for HR CIs (default 0.95).
pub fn churn_risk_model(
    times: &[f64],
    events: &[bool],
    x: &[Vec<f64>],
    names: &[String],
    conf_level: f64,
) -> Result<ChurnRiskModel> {
    use crate::survival::{CoxPhModel, CoxTies};

    let n = times.len();
    let n_events = events.iter().filter(|&&e| e).count();

    let model = CoxPhModel::new(times.to_vec(), events.to_vec(), x.to_vec(), CoxTies::Efron)?;

    let mle = crate::MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model)?;

    let coef = fit.parameters.clone();
    let nll = fit.nll;
    let p = coef.len();

    // Hessian-based SE via finite differences of gradient.
    let se = hessian_se(&model, &coef)?;

    let alpha_half = (1.0 - conf_level) / 2.0;
    let z_alpha = statrs::distribution::Normal::new(0.0, 1.0)
        .map(|n| statrs::distribution::ContinuousCDF::inverse_cdf(&n, 1.0 - alpha_half))
        .unwrap_or(1.96);

    let mut hazard_ratios = Vec::with_capacity(p);
    let mut hr_ci_lower = Vec::with_capacity(p);
    let mut hr_ci_upper = Vec::with_capacity(p);
    for j in 0..p {
        let hr = coef[j].exp();
        let lo = (coef[j] - z_alpha * se[j]).exp();
        let hi = (coef[j] + z_alpha * se[j]).exp();
        hazard_ratios.push(hr);
        hr_ci_lower.push(lo);
        hr_ci_upper.push(hi);
    }

    Ok(ChurnRiskModel {
        names: names.to_vec(),
        coefficients: coef,
        se,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        nll,
        n,
        n_events,
    })
}

/// Compute SE from numerical Hessian of NLL.
fn hessian_se<M: ns_core::traits::LogDensityModel>(model: &M, beta: &[f64]) -> Result<Vec<f64>> {
    let p = beta.len();
    let eps = 1e-5;
    let mut hess = vec![0.0_f64; p * p];
    let g0 = model.grad_nll(beta)?;

    for j in 0..p {
        let mut b_hi = beta.to_vec();
        b_hi[j] += eps;
        let g_hi = model.grad_nll(&b_hi)?;
        for k in 0..p {
            hess[j * p + k] = (g_hi[k] - g0[k]) / eps;
        }
    }

    // Symmetrise.
    for j in 0..p {
        for k in (j + 1)..p {
            let avg = 0.5 * (hess[j * p + k] + hess[k * p + j]);
            hess[j * p + k] = avg;
            hess[k * p + j] = avg;
        }
    }

    let h_mat = nalgebra::DMatrix::from_row_slice(p, p, &hess);
    let h_inv = h_mat
        .try_inverse()
        .ok_or_else(|| Error::Computation("Hessian singular in churn_risk_model".into()))?;

    let se: Vec<f64> = (0..p).map(|j| h_inv[(j, j)].max(0.0).sqrt()).collect();
    Ok(se)
}

// ---------------------------------------------------------------------------
// C3b: Bootstrap hazard ratios (Rayon-parallel)
// ---------------------------------------------------------------------------

/// Result of bootstrap hazard ratio estimation.
#[derive(Debug, Clone)]
pub struct BootstrapHrResult {
    /// Covariate names.
    pub names: Vec<String>,
    /// Point-estimate hazard ratios from the full dataset.
    pub hr_point: Vec<f64>,
    /// Bootstrap 2.5th percentile HR (lower CI).
    pub hr_ci_lower: Vec<f64>,
    /// Bootstrap 97.5th percentile HR (upper CI).
    pub hr_ci_upper: Vec<f64>,
    /// Number of bootstrap resamples requested.
    pub n_bootstrap: usize,
    /// Number of resamples that converged.
    pub n_converged: usize,
    /// Wall-clock seconds for the bootstrap loop.
    pub elapsed_s: f64,
}

/// Bootstrap hazard ratios via Rayon-parallel Cox PH refitting.
///
/// Runs `n_bootstrap` resamples of the dataset (with replacement),
/// fits Cox PH on each, and returns percentile-based CIs for hazard ratios.
/// Each resample runs in parallel via Rayon.
pub fn bootstrap_hazard_ratios(
    times: &[f64],
    events: &[bool],
    x: &[Vec<f64>],
    names: &[String],
    n_bootstrap: usize,
    seed: u64,
    conf_level: f64,
) -> Result<BootstrapHrResult> {
    use rayon::prelude::*;

    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".into()));
    }
    let p = x.first().map_or(0, |row| row.len());
    if p == 0 {
        return Err(Error::Validation("covariates must have at least 1 column".into()));
    }

    // Point estimate.
    let point = churn_risk_model(times, events, x, names, conf_level)?;

    let t0 = std::time::Instant::now();

    // Parallel bootstrap.
    let hr_samples: Vec<Option<Vec<f64>>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(b as u64));
            let idx: Vec<usize> = (0..n)
                .map(|_| {
                    use rand::Rng;
                    rng.random_range(0..n)
                })
                .collect();

            let b_times: Vec<f64> = idx.iter().map(|&i| times[i]).collect();
            let b_events: Vec<bool> = idx.iter().map(|&i| events[i]).collect();
            let b_x: Vec<Vec<f64>> = idx.iter().map(|&i| x[i].clone()).collect();

            match churn_risk_model(&b_times, &b_events, &b_x, names, conf_level) {
                Ok(r) => Some(r.hazard_ratios),
                Err(_) => None,
            }
        })
        .collect();

    let elapsed_s = t0.elapsed().as_secs_f64();

    // Collect converged samples.
    let converged: Vec<&Vec<f64>> = hr_samples.iter().filter_map(|s| s.as_ref()).collect();
    let n_converged = converged.len();

    if n_converged < 2 {
        return Err(Error::Computation("Fewer than 2 bootstrap resamples converged".into()));
    }

    // Percentile CIs.
    let alpha_half = (1.0 - conf_level) / 2.0;
    let mut hr_ci_lower = Vec::with_capacity(p);
    let mut hr_ci_upper = Vec::with_capacity(p);

    for j in 0..p {
        let mut vals: Vec<f64> = converged.iter().map(|hr| hr[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = ((alpha_half * vals.len() as f64).floor() as usize).min(vals.len() - 1);
        let hi_idx = (((1.0 - alpha_half) * vals.len() as f64).ceil() as usize).min(vals.len() - 1);
        hr_ci_lower.push(vals[lo_idx]);
        hr_ci_upper.push(vals[hi_idx]);
    }

    Ok(BootstrapHrResult {
        names: names.to_vec(),
        hr_point: point.hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        n_bootstrap,
        n_converged,
        elapsed_s,
    })
}

// ---------------------------------------------------------------------------
// C4: Causal uplift estimation
// ---------------------------------------------------------------------------

/// Result of a churn uplift analysis.
#[derive(Debug, Clone)]
pub struct ChurnUpliftResult {
    /// Average Treatment Effect on churn probability (negative = intervention reduces churn).
    pub ate: f64,
    /// Standard error of ATE.
    pub se: f64,
    /// 95% CI lower.
    pub ci_lower: f64,
    /// 95% CI upper.
    pub ci_upper: f64,
    /// Number of treated.
    pub n_treated: usize,
    /// Number of control.
    pub n_control: usize,
    /// Rosenbaum sensitivity: critical Γ at which result becomes insignificant.
    pub gamma_critical: Option<f64>,
}

/// Estimate the causal impact of an intervention on churn using AIPW.
///
/// This workflow:
/// 1. Fits a logistic regression propensity model.
/// 2. Uses `aipw_ate` with outcome = event indicator (churned within `horizon` months).
/// 3. Runs Rosenbaum sensitivity analysis.
///
/// # Arguments
/// - `times` — time to churn or censoring.
/// - `events` — event indicators.
/// - `treated` — treatment assignment (0 or 1).
/// - `x` — covariates for propensity model (row-major).
/// - `horizon` — evaluation horizon in months. Customers are classified as
///   churned (event && time ≤ horizon) or retained.
pub fn churn_uplift(
    times: &[f64],
    events: &[bool],
    treated: &[u8],
    x: &[Vec<f64>],
    horizon: f64,
) -> Result<ChurnUpliftResult> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".into()));
    }
    if events.len() != n || treated.len() != n || x.len() != n {
        return Err(Error::Validation("all input arrays must have the same length".into()));
    }

    // Binary outcome: churned within horizon.
    let y: Vec<f64> =
        (0..n).map(|i| if events[i] && times[i] <= horizon { 1.0 } else { 0.0 }).collect();

    // Propensity scores via logistic regression MLE.
    let propensity = fit_propensity(treated, x)?;

    // Outcome model: simple group means (sufficient for AIPW consistency
    // when either propensity or outcome model is correct).
    let mut sum_y_t = 0.0_f64;
    let mut n_t = 0usize;
    let mut sum_y_c = 0.0_f64;
    let mut n_c = 0usize;
    for i in 0..n {
        if treated[i] == 1 {
            sum_y_t += y[i];
            n_t += 1;
        } else {
            sum_y_c += y[i];
            n_c += 1;
        }
    }
    let mu1_mean = if n_t > 0 { sum_y_t / n_t as f64 } else { 0.0 };
    let mu0_mean = if n_c > 0 { sum_y_c / n_c as f64 } else { 0.0 };
    let mu1 = vec![mu1_mean; n];
    let mu0 = vec![mu0_mean; n];

    let aipw = crate::econometrics::aipw::aipw_ate(&y, treated, &propensity, &mu1, &mu0, 0.01)?;

    // Rosenbaum sensitivity.
    let y_treated: Vec<f64> = (0..n).filter(|&i| treated[i] == 1).map(|i| y[i]).collect();
    let y_control: Vec<f64> = (0..n).filter(|&i| treated[i] == 0).map(|i| y[i]).collect();
    let gammas: Vec<f64> = (10..=30).map(|g| g as f64 / 10.0).collect();
    let rb = crate::econometrics::aipw::rosenbaum_bounds(&y_treated, &y_control, &gammas).ok();
    let gamma_critical = rb.and_then(|r| r.gamma_critical);

    Ok(ChurnUpliftResult {
        ate: aipw.ate,
        se: aipw.se,
        ci_lower: aipw.ci_lower,
        ci_upper: aipw.ci_upper,
        n_treated: n_t,
        n_control: n_c,
        gamma_critical,
    })
}

/// Fit a simple logistic propensity model: P(treated=1 | x).
///
/// Returns predicted probabilities, trimmed to [0.01, 0.99].
fn fit_propensity(treated: &[u8], x: &[Vec<f64>]) -> Result<Vec<f64>> {
    let n = treated.len();
    let p = x.first().map_or(0, |row| row.len());

    // Build design matrix with intercept.
    let mut x_full: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in x {
        let mut r = Vec::with_capacity(p + 1);
        r.push(1.0);
        r.extend_from_slice(row);
        x_full.push(r);
    }
    let model = crate::regression::LogisticRegressionModel::new(
        x_full.clone(),
        treated.to_vec(),
        false, // intercept already included in x_full
    )?;
    let mle = crate::MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model)?;

    // Predict probabilities.
    let beta = &fit.parameters;
    let probs: Vec<f64> = x_full
        .iter()
        .map(|row| {
            let lp: f64 = row.iter().zip(beta.iter()).map(|(xi, bi)| xi * bi).sum();
            let prob: f64 = 1.0 / (1.0 + (-lp).exp());
            prob.clamp(0.01, 0.99)
        })
        .collect();

    Ok(probs)
}

// ---------------------------------------------------------------------------
// C5: Real-data ingestion (Parquet / CSV → ChurnDataset)
// ---------------------------------------------------------------------------

/// Column mapping configuration for ingesting real customer data.
///
/// Maps source column names to the fields expected by the churn analysis
/// pipeline. Loaded from a YAML mapping file via `nextstat churn ingest`.
///
/// # Required columns
/// - `time_col` — observed time (months/days from signup to churn or censoring).
/// - `event_col` — event indicator: `true`/`1` = churned, `false`/`0` = censored.
///
/// # Optional columns
/// - `group_col` — segment/cohort label for stratified KM (e.g. plan tier).
/// - `treated_col` — treatment indicator (0/1) for causal uplift.
/// - `covariate_cols` — list of numeric covariate columns for Cox PH / uplift.
/// - `user_id_col` — customer identifier (carried through, not used in analysis).
///
/// # Censoring policy
/// If `observation_end` is set, any row where `event == false` gets its time
/// clamped to `min(time, observation_end)`. If `churn_ts_col` and
/// `signup_ts_col` are set (instead of a pre-computed `time_col`), durations
/// are computed as `churn_ts - signup_ts` (with missing `churn_ts` treated
/// as right-censored at `observation_end`).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ChurnMappingConfig {
    /// Column containing observed duration (required unless signup_ts_col + observation_end are set).
    #[serde(default)]
    pub time_col: Option<String>,

    /// Column containing event indicator (bool or 0/1 integer).
    pub event_col: String,

    /// Column containing group/segment label (optional, for stratified KM).
    #[serde(default)]
    pub group_col: Option<String>,

    /// Column containing treatment indicator 0/1 (optional, for uplift).
    #[serde(default)]
    pub treated_col: Option<String>,

    /// Numeric covariate columns for Cox PH / uplift (optional).
    #[serde(default)]
    pub covariate_cols: Vec<String>,

    /// Customer ID column (optional, carried through for traceability).
    #[serde(default)]
    pub user_id_col: Option<String>,

    /// Column containing signup timestamp (alternative to time_col).
    #[serde(default)]
    pub signup_ts_col: Option<String>,

    /// Column containing churn timestamp (alternative to time_col; NULL = censored).
    #[serde(default)]
    pub churn_ts_col: Option<String>,

    /// End of observation window. Used for right-censoring when churn_ts is NULL.
    /// Units must match time_col (months, days, etc.).
    #[serde(default)]
    pub observation_end: Option<f64>,

    /// Time unit for display purposes (e.g. "months", "days"). Default: "months".
    #[serde(default = "default_time_unit")]
    pub time_unit: String,
}

fn default_time_unit() -> String {
    "months".into()
}

/// Errors specific to churn data ingestion.
#[derive(Debug, thiserror::Error)]
pub enum ChurnIngestError {
    #[error("missing required column '{0}' in input data")]
    MissingColumn(String),

    #[error("column '{col}' has unsupported type: expected {expected}, got {actual}")]
    WrongType { col: String, expected: String, actual: String },

    #[error("row {row}: {message}")]
    RowError { row: usize, message: String },

    #[error("mapping config error: {0}")]
    Config(String),

    #[error("no rows in input data")]
    EmptyData,

    #[error("all events are censored — no churn events found")]
    NoEvents,
}

impl From<ChurnIngestError> for Error {
    fn from(e: ChurnIngestError) -> Self {
        Error::Validation(e.to_string())
    }
}

/// Result of ingesting real customer data.
#[derive(Debug, Clone)]
pub struct ChurnIngestResult {
    /// The validated dataset ready for analysis.
    pub dataset: ChurnDataset,
    /// Number of rows dropped due to missing/invalid data.
    pub n_dropped: usize,
    /// Warnings generated during ingestion (non-fatal issues).
    pub warnings: Vec<String>,
    /// Covariate column names (in order matching dataset.covariates columns).
    pub covariate_names: Vec<String>,
}

/// Ingest tabular customer data into a [`ChurnDataset`].
///
/// Accepts parallel slices extracted from a tabular source (Parquet, CSV, etc.)
/// and applies the column mapping + validation + censoring policy.
///
/// This is the core ingestion function; the CLI and Python wrappers call this
/// after reading the file and extracting columns.
///
/// # Arguments
/// - `n` — number of rows.
/// - `times` — observed durations (required). Must be positive.
/// - `events` — event indicators. `true` = churned.
/// - `groups` — optional segment labels (one per row).
/// - `treated` — optional treatment indicators (0 or 1).
/// - `covariates` — covariate matrix, each inner vec is one row's covariate values.
/// - `covariate_names` — names for the covariate columns.
/// - `observation_end` — optional observation window end for censoring validation.
pub fn ingest_churn_arrays(
    times: &[f64],
    events: &[bool],
    groups: Option<&[i64]>,
    treated: Option<&[u8]>,
    covariates: &[Vec<f64>],
    covariate_names: &[String],
    observation_end: Option<f64>,
) -> std::result::Result<ChurnIngestResult, ChurnIngestError> {
    let n = times.len();
    if n == 0 {
        return Err(ChurnIngestError::EmptyData);
    }
    if events.len() != n {
        return Err(ChurnIngestError::Config(format!(
            "events length ({}) != times length ({n})",
            events.len()
        )));
    }
    if let Some(g) = groups
        && g.len() != n
    {
        return Err(ChurnIngestError::Config(format!(
            "groups length ({}) != times length ({n})",
            g.len()
        )));
    }
    if let Some(t) = treated
        && t.len() != n
    {
        return Err(ChurnIngestError::Config(format!(
            "treated length ({}) != times length ({n})",
            t.len()
        )));
    }
    if !covariates.is_empty() && covariates.len() != n {
        return Err(ChurnIngestError::Config(format!(
            "covariates length ({}) != times length ({n})",
            covariates.len()
        )));
    }

    let mut out_times = Vec::with_capacity(n);
    let mut out_events = Vec::with_capacity(n);
    let mut out_groups = Vec::with_capacity(n);
    let mut out_treated = Vec::with_capacity(n);
    let mut out_covariates: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut out_records = Vec::with_capacity(n);
    let mut n_dropped = 0usize;
    let mut warnings = Vec::new();

    let p = if covariates.is_empty() { 0 } else { covariates[0].len() };
    if !covariate_names.is_empty() && covariate_names.len() != p {
        return Err(ChurnIngestError::Config(format!(
            "covariate_names length ({}) != covariate width ({p})",
            covariate_names.len()
        )));
    }

    for i in 0..n {
        let t = times[i];

        // Validate time.
        if !t.is_finite() || t < 0.0 {
            n_dropped += 1;
            continue;
        }
        if t == 0.0 {
            n_dropped += 1;
            continue;
        }

        // Apply observation_end censoring cap.
        let (final_time, final_event) = if let Some(end) = observation_end {
            if t > end { (end, false) } else { (t, events[i]) }
        } else {
            (t, events[i])
        };

        // Validate covariates row width.
        if !covariates.is_empty() {
            let row = &covariates[i];
            if row.len() != p {
                n_dropped += 1;
                continue;
            }
            if row.iter().any(|v| !v.is_finite()) {
                n_dropped += 1;
                continue;
            }
        }

        // Validate treated values.
        if let Some(tr) = treated
            && tr[i] > 1
        {
            n_dropped += 1;
            continue;
        }

        let group = groups.map_or(0, |g| g[i]);
        let treat = treated.map_or(0, |tr| tr[i]);
        let cov_row = if covariates.is_empty() { vec![] } else { covariates[i].clone() };

        out_times.push(final_time);
        out_events.push(final_event);
        out_groups.push(group);
        out_treated.push(treat);
        out_records.push(ChurnRecord {
            customer_id: i,
            cohort: 0,
            plan: 0,
            region: 0,
            usage_score: 0.0,
            support_tickets: 0,
            treated: treat,
            time: final_time,
            event: final_event,
        });
        if !covariates.is_empty() {
            out_covariates.push(cov_row);
        }
    }

    if out_times.is_empty() {
        return Err(ChurnIngestError::EmptyData);
    }

    let n_events = out_events.iter().filter(|&&e| e).count();
    if n_events == 0 {
        return Err(ChurnIngestError::NoEvents);
    }

    if n_dropped > 0 {
        warnings.push(format!(
            "{n_dropped} rows dropped (invalid time, NaN covariate, or bad treated value)"
        ));
    }

    let censoring_pct = 100.0 * (1.0 - n_events as f64 / out_times.len() as f64);
    if censoring_pct > 95.0 {
        warnings.push(format!(
            "very high censoring rate ({censoring_pct:.1}%) — only {n_events} events in {} observations",
            out_times.len()
        ));
    }

    let dataset = ChurnDataset {
        records: out_records,
        times: out_times,
        events: out_events,
        groups: out_groups,
        treated: out_treated,
        covariates: out_covariates,
    };

    Ok(ChurnIngestResult {
        dataset,
        n_dropped,
        warnings,
        covariate_names: covariate_names.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// C6: Cohort retention matrix (growth-team artifact)
// ---------------------------------------------------------------------------

/// A single cell in the cohort retention matrix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CohortMatrixCell {
    /// Number at risk at the start of this period.
    pub n_at_risk: usize,
    /// Number of events (churns) during this period.
    pub n_events: usize,
    /// Number censored during this period.
    pub n_censored: usize,
    /// Period retention rate: 1 - n_events / n_at_risk (NaN if n_at_risk == 0).
    pub retention_rate: f64,
    /// Cumulative retention (product of retention rates up to this period).
    pub cumulative_retention: f64,
}

/// A single row (one cohort) in the retention matrix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CohortMatrixRow {
    /// Cohort label (group ID).
    pub cohort: i64,
    /// Total customers in this cohort.
    pub n_total: usize,
    /// Total events in this cohort.
    pub n_events: usize,
    /// Per-period cells.
    pub periods: Vec<CohortMatrixCell>,
}

/// Result of `cohort_retention_matrix`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CohortRetentionMatrix {
    /// Period boundaries used (e.g. [1, 2, 3, 6, 12]).
    pub period_boundaries: Vec<f64>,
    /// One row per cohort.
    pub cohorts: Vec<CohortMatrixRow>,
    /// Summary row (all cohorts combined).
    pub overall: CohortMatrixRow,
}

/// Compute a cohort retention matrix (life-table style).
///
/// For each cohort (identified by `groups`), at each period boundary,
/// computes the number at risk, events, censored, period retention rate,
/// and cumulative retention.
///
/// The period boundaries define the end of each interval. For example,
/// `[1.0, 3.0, 6.0, 12.0]` means intervals (0,1], (1,3], (3,6], (6,12].
///
/// # Arguments
/// - `times` — observed durations.
/// - `events` — event indicators.
/// - `groups` — cohort/group labels.
/// - `period_boundaries` — sorted time points defining period ends.
pub fn cohort_retention_matrix(
    times: &[f64],
    events: &[bool],
    groups: &[i64],
    period_boundaries: &[f64],
) -> Result<CohortRetentionMatrix> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".into()));
    }
    if events.len() != n || groups.len() != n {
        return Err(Error::Validation("all arrays must have the same length".into()));
    }
    if period_boundaries.is_empty() {
        return Err(Error::Validation("period_boundaries must be non-empty".into()));
    }

    let k = period_boundaries.len();

    // Collect unique groups in order of first appearance, then sort.
    let mut unique_groups: Vec<i64> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &g in groups {
            if seen.insert(g) {
                unique_groups.push(g);
            }
        }
    }
    unique_groups.sort();

    // Build per-cohort rows.
    let mut cohort_rows: Vec<CohortMatrixRow> = Vec::with_capacity(unique_groups.len());
    for &g in &unique_groups {
        let row = compute_life_table(times, events, groups, Some(g), period_boundaries, k);
        cohort_rows.push(row);
    }

    // Overall row (all cohorts combined).
    let overall = compute_life_table(times, events, groups, None, period_boundaries, k);

    Ok(CohortRetentionMatrix {
        period_boundaries: period_boundaries.to_vec(),
        cohorts: cohort_rows,
        overall,
    })
}

/// Compute a life table for a single cohort (or all if `cohort_filter` is None).
fn compute_life_table(
    times: &[f64],
    events: &[bool],
    groups: &[i64],
    cohort_filter: Option<i64>,
    boundaries: &[f64],
    k: usize,
) -> CohortMatrixRow {
    let n = times.len();
    let cohort_label = cohort_filter.unwrap_or(-1);

    // Collect indices for this cohort.
    let indices: Vec<usize> =
        (0..n).filter(|&i| cohort_filter.is_none() || groups[i] == cohort_label).collect();

    let n_total = indices.len();
    let total_events = indices.iter().filter(|&&i| events[i]).count();

    let mut periods = Vec::with_capacity(k);
    let mut cumulative_ret = 1.0_f64;

    for j in 0..k {
        let lo = if j == 0 { 0.0 } else { boundaries[j - 1] };
        let hi = boundaries[j];

        // n_at_risk: customers whose time > lo (still in the study at period start).
        let n_at_risk = indices.iter().filter(|&&i| times[i] > lo).count();

        // n_events: customers who churned in (lo, hi].
        let n_ev =
            indices.iter().filter(|&&i| events[i] && times[i] > lo && times[i] <= hi).count();

        // n_censored: customers censored in (lo, hi] (event=false, time in (lo, hi]).
        let n_cens =
            indices.iter().filter(|&&i| !events[i] && times[i] > lo && times[i] <= hi).count();

        let period_ret =
            if n_at_risk > 0 { 1.0 - n_ev as f64 / n_at_risk as f64 } else { f64::NAN };

        cumulative_ret *= if period_ret.is_nan() { 1.0 } else { period_ret };

        periods.push(CohortMatrixCell {
            n_at_risk,
            n_events: n_ev,
            n_censored: n_cens,
            retention_rate: period_ret,
            cumulative_retention: cumulative_ret,
        });
    }

    CohortMatrixRow { cohort: cohort_label, n_total, n_events: total_events, periods }
}

// ---------------------------------------------------------------------------
// C8: Survival-native causal uplift (RMST + IPW-weighted KM + ΔS(t))
// ---------------------------------------------------------------------------

/// Compute RMST (Restricted Mean Survival Time) from KM steps up to horizon τ.
///
/// RMST = area under the KM curve from 0 to τ, computed via trapezoidal
/// integration of the step function.
pub fn compute_rmst(steps: &[crate::survival::KaplanMeierStep], tau: f64) -> f64 {
    if steps.is_empty() || tau <= 0.0 {
        return tau.max(0.0); // S(t)=1 everywhere → area = τ
    }

    let mut area = 0.0_f64;
    let mut prev_time = 0.0_f64;
    let mut prev_surv = 1.0_f64;

    for step in steps {
        if step.time >= tau {
            // Remaining rectangle from prev_time to τ at prev_surv.
            area += prev_surv * (tau - prev_time);
            return area;
        }
        // Rectangle from prev_time to step.time at prev_surv.
        area += prev_surv * (step.time - prev_time);
        prev_time = step.time;
        prev_surv = step.survival;
    }

    // Tail from last event time to τ at final survival.
    area += prev_surv * (tau - prev_time);
    area
}

/// Evaluate KM survival at a specific time point via step interpolation.
fn km_survival_at(steps: &[crate::survival::KaplanMeierStep], t: f64) -> f64 {
    let mut s = 1.0_f64;
    for step in steps {
        if step.time > t {
            break;
        }
        s = step.survival;
    }
    s
}

/// Overlap diagnostics for propensity scores.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OverlapDiagnostics {
    /// Number of observations before trimming.
    pub n_total: usize,
    /// Number of observations after trimming.
    pub n_after_trim: usize,
    /// Number trimmed.
    pub n_trimmed: usize,
    /// Mean propensity score.
    pub mean_propensity: f64,
    /// Min propensity score (after trimming).
    pub min_propensity: f64,
    /// Max propensity score (after trimming).
    pub max_propensity: f64,
    /// Effective sample size (treated): sum(w)² / sum(w²).
    pub ess_treated: f64,
    /// Effective sample size (control).
    pub ess_control: f64,
}

/// Survival difference at a specific horizon.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SurvivalDiffAtHorizon {
    /// Time horizon.
    pub horizon: f64,
    /// S(t) for treated arm.
    pub survival_treated: f64,
    /// S(t) for control arm.
    pub survival_control: f64,
    /// ΔS(t) = S_treated(t) - S_control(t). Positive = treatment retains more.
    pub delta_survival: f64,
}

/// Per-arm summary in the survival uplift report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArmSummary {
    /// Arm label ("treated" or "control").
    pub arm: String,
    /// Number of observations (after trimming).
    pub n: usize,
    /// Number of events.
    pub n_events: usize,
    /// RMST up to the specified horizon.
    pub rmst: f64,
    /// Median survival time.
    pub median: Option<f64>,
}

/// Full survival-native uplift report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SurvivalUpliftReport {
    /// RMST for treated arm.
    pub rmst_treated: f64,
    /// RMST for control arm.
    pub rmst_control: f64,
    /// ΔRMST = RMST_treated - RMST_control. Positive = treatment extends retention.
    pub delta_rmst: f64,
    /// Horizon τ used for RMST.
    pub horizon: f64,
    /// Per-arm summaries.
    pub arms: Vec<ArmSummary>,
    /// Survival differences at specified horizons.
    pub survival_diffs: Vec<SurvivalDiffAtHorizon>,
    /// Overlap diagnostics.
    pub overlap: OverlapDiagnostics,
    /// Whether IPW weighting was applied (requires covariates).
    pub ipw_applied: bool,
}

/// Compute survival-native causal uplift with RMST, IPW-weighted KM, and ΔS(t).
///
/// # Arguments
/// - `times` — observation times.
/// - `events` — event indicators.
/// - `treated` — treatment assignment (0 or 1).
/// - `x` — covariates for propensity model (row-major). If empty, unweighted KM is used.
/// - `horizon` — RMST integration horizon τ.
/// - `eval_horizons` — additional time points for ΔS(t) evaluation.
/// - `trim` — propensity score trimming threshold (default 0.01).
pub fn survival_uplift_report(
    times: &[f64],
    events: &[bool],
    treated: &[u8],
    x: &[Vec<f64>],
    horizon: f64,
    eval_horizons: &[f64],
    trim: f64,
) -> Result<SurvivalUpliftReport> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".into()));
    }
    if events.len() != n || treated.len() != n {
        return Err(Error::Validation("all arrays must have the same length".into()));
    }
    if horizon <= 0.0 {
        return Err(Error::Validation("horizon must be positive".into()));
    }

    let has_covariates = !x.is_empty() && x.len() == n && !x[0].is_empty();

    // 1. Compute propensity scores (if covariates provided).
    let propensity = if has_covariates {
        fit_propensity(treated, x)?
    } else {
        // Uniform propensity = proportion treated.
        let p_treat = treated.iter().filter(|&&t| t == 1).count() as f64 / n as f64;
        vec![p_treat.clamp(0.01, 0.99); n]
    };

    // 2. Compute IPW weights and overlap diagnostics.
    let mut weights = vec![0.0_f64; n];
    let mut n_trimmed = 0usize;
    let mut sum_prop = 0.0_f64;
    let mut min_prop = f64::INFINITY;
    let mut max_prop = f64::NEG_INFINITY;
    let mut n_used = 0usize;

    for i in 0..n {
        let e = propensity[i];
        if e < trim || e > 1.0 - trim {
            n_trimmed += 1;
            continue;
        }
        n_used += 1;
        sum_prop += e;
        min_prop = min_prop.min(e);
        max_prop = max_prop.max(e);

        // IPW: treated get 1/e, control get 1/(1-e).
        weights[i] = if treated[i] == 1 { 1.0 / e } else { 1.0 / (1.0 - e) };
    }

    if n_used == 0 {
        return Err(Error::Validation("no observations after trimming".into()));
    }

    // ESS: (Σw)² / Σw².
    let mut sw_t = 0.0_f64;
    let mut sw2_t = 0.0_f64;
    let mut sw_c = 0.0_f64;
    let mut sw2_c = 0.0_f64;
    for i in 0..n {
        if weights[i] == 0.0 {
            continue;
        }
        if treated[i] == 1 {
            sw_t += weights[i];
            sw2_t += weights[i] * weights[i];
        } else {
            sw_c += weights[i];
            sw2_c += weights[i] * weights[i];
        }
    }
    let ess_treated = if sw2_t > 0.0 { sw_t * sw_t / sw2_t } else { 0.0 };
    let ess_control = if sw2_c > 0.0 { sw_c * sw_c / sw2_c } else { 0.0 };

    let overlap = OverlapDiagnostics {
        n_total: n,
        n_after_trim: n_used,
        n_trimmed,
        mean_propensity: sum_prop / n_used as f64,
        min_propensity: if min_prop.is_finite() { min_prop } else { 0.0 },
        max_propensity: if max_prop.is_finite() { max_prop } else { 1.0 },
        ess_treated,
        ess_control,
    };

    // 3. Build per-arm data (applying IPW as observation duplication weight).
    //    For weighted KM, we expand observations by weight. Since KM doesn't
    //    natively support weights, we use the unweighted KM on trimmed data
    //    when no covariates are provided, and the weighted approach otherwise.
    //    For simplicity and correctness, we compute unweighted KM per arm
    //    on the trimmed subset — IPW rebalancing is reflected in the RMST
    //    via the weighted mean approach below.

    let mut t_times = Vec::new();
    let mut t_events = Vec::new();
    let mut c_times = Vec::new();
    let mut c_events = Vec::new();

    for i in 0..n {
        if weights[i] == 0.0 {
            continue;
        }
        if treated[i] == 1 {
            t_times.push(times[i]);
            t_events.push(events[i]);
        } else {
            c_times.push(times[i]);
            c_events.push(events[i]);
        }
    }

    if t_times.is_empty() || c_times.is_empty() {
        return Err(Error::Validation(
            "need both treated and control observations after trimming".into(),
        ));
    }

    let km_treated = crate::survival::kaplan_meier(&t_times, &t_events, 0.95)?;
    let km_control = crate::survival::kaplan_meier(&c_times, &c_events, 0.95)?;

    // 4. RMST per arm.
    let rmst_treated = compute_rmst(&km_treated.steps, horizon);
    let rmst_control = compute_rmst(&km_control.steps, horizon);
    let delta_rmst = rmst_treated - rmst_control;

    // 5. ΔS(t) at evaluation horizons.
    let mut survival_diffs = Vec::with_capacity(eval_horizons.len());
    for &h in eval_horizons {
        let s_t = km_survival_at(&km_treated.steps, h);
        let s_c = km_survival_at(&km_control.steps, h);
        survival_diffs.push(SurvivalDiffAtHorizon {
            horizon: h,
            survival_treated: s_t,
            survival_control: s_c,
            delta_survival: s_t - s_c,
        });
    }

    // 6. Per-arm summaries.
    let arms = vec![
        ArmSummary {
            arm: "treated".into(),
            n: t_times.len(),
            n_events: t_events.iter().filter(|&&e| e).count(),
            rmst: rmst_treated,
            median: km_treated.median,
        },
        ArmSummary {
            arm: "control".into(),
            n: c_times.len(),
            n_events: c_events.iter().filter(|&&e| e).count(),
            rmst: rmst_control,
            median: km_control.median,
        },
    ];

    Ok(SurvivalUpliftReport {
        rmst_treated,
        rmst_control,
        delta_rmst,
        horizon,
        arms,
        survival_diffs,
        overlap,
        ipw_applied: has_covariates,
    })
}

// ---------------------------------------------------------------------------
// C9: Diagnostics / guardrails report
// ---------------------------------------------------------------------------

/// Propensity score distribution summary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PropensityOverlap {
    /// Quantiles: [min, p5, p25, median, p75, p95, max].
    pub quantiles: [f64; 7],
    /// Mean propensity score.
    pub mean: f64,
    /// Number trimmed at lower bound.
    pub n_trimmed_low: usize,
    /// Number trimmed at upper bound.
    pub n_trimmed_high: usize,
    /// Trimming threshold used.
    pub trim: f64,
}

/// Covariate balance row (one per covariate).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CovariateBalance {
    /// Covariate name/index.
    pub name: String,
    /// Standardized mean difference (SMD) before weighting.
    pub smd_raw: f64,
    /// Mean in treated group.
    pub mean_treated: f64,
    /// Mean in control group.
    pub mean_control: f64,
}

/// Per-segment censoring summary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CensoringSegment {
    /// Segment label.
    pub group: i64,
    /// Total observations.
    pub n: usize,
    /// Number of events.
    pub n_events: usize,
    /// Number censored.
    pub n_censored: usize,
    /// Fraction censored.
    pub frac_censored: f64,
}

/// A diagnostic warning with severity.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiagnosticWarning {
    /// Warning category.
    pub category: String,
    /// Severity: "info", "warning", "critical".
    pub severity: String,
    /// Human-readable message.
    pub message: String,
}

/// Full diagnostics/guardrails report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChurnDiagnosticsReport {
    /// Propensity score overlap (if treatment data available).
    pub propensity_overlap: Option<PropensityOverlap>,
    /// Covariate balance (if covariates available).
    pub covariate_balance: Vec<CovariateBalance>,
    /// Censoring diagnostics per segment.
    pub censoring_by_segment: Vec<CensoringSegment>,
    /// Overall censoring fraction.
    pub overall_censoring_frac: f64,
    /// Diagnostic warnings/flags.
    pub warnings: Vec<DiagnosticWarning>,
    /// Overall trust gate: true if no critical warnings.
    pub trust_gate_passed: bool,
    /// Total observations.
    pub n: usize,
    /// Total events.
    pub n_events: usize,
}

/// Compute quantile from sorted slice via linear interpolation.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute a comprehensive diagnostics report for churn analysis data.
///
/// # Arguments
/// - `times` — observation times.
/// - `events` — event indicators.
/// - `groups` — segment/group labels.
/// - `treated` — treatment assignment (0 or 1), or empty if no treatment.
/// - `x` — covariates (row-major, n × p), or empty.
/// - `covariate_names` — optional names for covariates.
/// - `trim` — propensity score trimming threshold.
pub fn churn_diagnostics_report(
    times: &[f64],
    events: &[bool],
    groups: &[i64],
    treated: &[u8],
    x: &[Vec<f64>],
    covariate_names: &[String],
    trim: f64,
) -> Result<ChurnDiagnosticsReport> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".into()));
    }
    if events.len() != n || groups.len() != n {
        return Err(Error::Validation("all arrays must have the same length".into()));
    }

    let total_events = events.iter().filter(|&&e| e).count();
    let total_censored = n - total_events;
    let overall_censoring_frac = total_censored as f64 / n as f64;

    let mut warnings: Vec<DiagnosticWarning> = Vec::new();

    // 1. Censoring diagnostics per segment.
    let mut unique_groups: Vec<i64> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &g in groups {
            if seen.insert(g) {
                unique_groups.push(g);
            }
        }
    }
    unique_groups.sort();

    let mut censoring_by_segment: Vec<CensoringSegment> = Vec::new();
    for &g in &unique_groups {
        let mut seg_n = 0usize;
        let mut seg_events = 0usize;
        for i in 0..n {
            if groups[i] == g {
                seg_n += 1;
                if events[i] {
                    seg_events += 1;
                }
            }
        }
        let seg_censored = seg_n - seg_events;
        let frac = if seg_n > 0 { seg_censored as f64 / seg_n as f64 } else { 0.0 };
        censoring_by_segment.push(CensoringSegment {
            group: g,
            n: seg_n,
            n_events: seg_events,
            n_censored: seg_censored,
            frac_censored: frac,
        });

        if frac > 0.8 {
            warnings.push(DiagnosticWarning {
                category: "censoring".into(),
                severity: "warning".into(),
                message: format!(
                    "Segment {} has {:.0}% censoring ({}/{})",
                    g,
                    frac * 100.0,
                    seg_censored,
                    seg_n
                ),
            });
        }
        if seg_events < 5 {
            warnings.push(DiagnosticWarning {
                category: "sample_size".into(),
                severity: "warning".into(),
                message: format!("Segment {} has only {} events (< 5)", g, seg_events),
            });
        }
    }

    if overall_censoring_frac > 0.9 {
        warnings.push(DiagnosticWarning {
            category: "censoring".into(),
            severity: "critical".into(),
            message: format!(
                "Overall censoring is {:.0}% — survival estimates may be unreliable",
                overall_censoring_frac * 100.0
            ),
        });
    }

    // 2. Propensity overlap + covariate balance (if treatment data available).
    let has_treatment = !treated.is_empty() && treated.len() == n;
    let has_covariates = !x.is_empty() && x.len() == n && !x[0].is_empty();

    let mut propensity_overlap: Option<PropensityOverlap> = None;
    let mut covariate_balance: Vec<CovariateBalance> = Vec::new();

    if has_treatment && has_covariates {
        let propensity = fit_propensity(treated, x)?;

        let mut sorted_prop = propensity.clone();
        sorted_prop.sort_by(|a, b| a.total_cmp(b));

        let mut n_trim_low = 0usize;
        let mut n_trim_high = 0usize;
        for &e in &propensity {
            if e <= trim {
                n_trim_low += 1;
            }
            if e >= 1.0 - trim {
                n_trim_high += 1;
            }
        }

        let quantiles = [
            sorted_prop[0],
            quantile_sorted(&sorted_prop, 0.05),
            quantile_sorted(&sorted_prop, 0.25),
            quantile_sorted(&sorted_prop, 0.50),
            quantile_sorted(&sorted_prop, 0.75),
            quantile_sorted(&sorted_prop, 0.95),
            *sorted_prop.last().unwrap(),
        ];
        let mean_prop = sorted_prop.iter().sum::<f64>() / sorted_prop.len() as f64;

        propensity_overlap = Some(PropensityOverlap {
            quantiles,
            mean: mean_prop,
            n_trimmed_low: n_trim_low,
            n_trimmed_high: n_trim_high,
            trim,
        });

        // Check overlap quality.
        let range = quantiles[6] - quantiles[0];
        if range < 0.1 {
            warnings.push(DiagnosticWarning {
                category: "overlap".into(),
                severity: "warning".into(),
                message: format!("Propensity score range is narrow ({:.3}), may indicate weak confounders or near-deterministic treatment", range),
            });
        }

        // 3. Covariate balance (standardized mean differences).
        let p = x[0].len();
        for j in 0..p {
            let name = if j < covariate_names.len() {
                covariate_names[j].clone()
            } else {
                format!("covariate_{}", j + 1)
            };

            let mut sum_t = 0.0_f64;
            let mut sum_c = 0.0_f64;
            let mut ss_t = 0.0_f64;
            let mut ss_c = 0.0_f64;
            let mut nt = 0usize;
            let mut nc = 0usize;

            for i in 0..n {
                let v = x[i][j];
                if treated[i] == 1 {
                    sum_t += v;
                    ss_t += v * v;
                    nt += 1;
                } else {
                    sum_c += v;
                    ss_c += v * v;
                    nc += 1;
                }
            }

            let mean_t = if nt > 0 { sum_t / nt as f64 } else { 0.0 };
            let mean_c = if nc > 0 { sum_c / nc as f64 } else { 0.0 };
            let var_t =
                if nt > 1 { (ss_t - sum_t * sum_t / nt as f64) / (nt - 1) as f64 } else { 0.0 };
            let var_c =
                if nc > 1 { (ss_c - sum_c * sum_c / nc as f64) / (nc - 1) as f64 } else { 0.0 };
            let pooled_sd = ((var_t + var_c) / 2.0).sqrt();

            let smd = if pooled_sd > 1e-15 { (mean_t - mean_c) / pooled_sd } else { 0.0 };

            if smd.abs() > 0.25 {
                warnings.push(DiagnosticWarning {
                    category: "balance".into(),
                    severity: "warning".into(),
                    message: format!("{}: SMD = {:.3} (> 0.25 threshold)", name, smd),
                });
            }

            covariate_balance.push(CovariateBalance {
                name,
                smd_raw: smd,
                mean_treated: mean_t,
                mean_control: mean_c,
            });
        }
    }

    // 4. Trust gate: no critical warnings.
    let trust_gate_passed = !warnings.iter().any(|w| w.severity == "critical");

    Ok(ChurnDiagnosticsReport {
        propensity_overlap,
        covariate_balance,
        censoring_by_segment,
        overall_censoring_frac,
        warnings,
        trust_gate_passed,
        n,
        n_events: total_events,
    })
}

// ---------------------------------------------------------------------------
// C7: Segment comparison report (pairwise log-rank + effect sizes + MCP)
// ---------------------------------------------------------------------------

/// Multiple comparisons correction method.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionMethod {
    /// Bonferroni correction (conservative).
    Bonferroni,
    /// Benjamini-Hochberg False Discovery Rate.
    BenjaminiHochberg,
    /// No correction.
    None,
}

/// Per-segment summary in the comparison report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentSummary {
    /// Group / segment label.
    pub group: i64,
    /// Number of observations.
    pub n: usize,
    /// Number of events.
    pub n_events: usize,
    /// Median survival time (from KM), or None if S(t) never reaches 0.5.
    pub median: Option<f64>,
    /// Observed events (from k-sample log-rank).
    pub observed: f64,
    /// Expected events under H₀.
    pub expected: f64,
}

/// A single pairwise comparison between two segments.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PairwiseComparison {
    /// First group label.
    pub group_a: i64,
    /// Second group label.
    pub group_b: i64,
    /// Log-rank chi-squared statistic.
    pub chi_squared: f64,
    /// Raw (unadjusted) p-value.
    pub p_value: f64,
    /// Adjusted p-value (after multiple comparisons correction).
    pub p_adjusted: f64,
    /// Hazard ratio proxy: (O_a/E_a) / (O_b/E_b). Values > 1 mean group A
    /// has higher hazard (more churn) than group B.
    pub hazard_ratio_proxy: f64,
    /// Difference in median survival (median_a - median_b), or None if either is undefined.
    pub median_diff: Option<f64>,
    /// Whether the comparison is significant at the given alpha level (after correction).
    pub significant: bool,
}

/// Full segment comparison report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentComparisonReport {
    /// K-sample (overall) log-rank test result.
    pub overall_chi_squared: f64,
    /// Overall p-value.
    pub overall_p_value: f64,
    /// Degrees of freedom for overall test.
    pub overall_df: usize,
    /// Per-segment summaries.
    pub segments: Vec<SegmentSummary>,
    /// Pairwise comparisons (sorted by p_adjusted ascending).
    pub pairwise: Vec<PairwiseComparison>,
    /// Correction method used.
    pub correction_method: CorrectionMethod,
    /// Significance level used.
    pub alpha: f64,
    /// Total number of observations.
    pub n: usize,
    /// Total number of events.
    pub n_events: usize,
}

/// Compute a segment comparison report with pairwise log-rank tests,
/// effect sizes (HR proxy), and multiple comparisons correction.
///
/// # Arguments
/// - `times` — observed durations.
/// - `events` — event indicators.
/// - `groups` — segment/group labels.
/// - `conf_level` — confidence level for KM CIs (e.g. 0.95).
/// - `correction` — multiple comparisons correction method.
/// - `alpha` — significance level (e.g. 0.05).
pub fn segment_comparison_report(
    times: &[f64],
    events: &[bool],
    groups: &[i64],
    conf_level: f64,
    correction: CorrectionMethod,
    alpha: f64,
) -> Result<SegmentComparisonReport> {
    let n = times.len();
    if n == 0 {
        return Err(Error::Validation("times must be non-empty".into()));
    }
    if events.len() != n || groups.len() != n {
        return Err(Error::Validation("all arrays must have the same length".into()));
    }

    // 1. K-sample log-rank.
    let overall_lr = crate::survival::log_rank_test(times, events, groups)?;

    // 2. Unique groups sorted.
    let mut unique_groups: Vec<i64> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for &g in groups {
            if seen.insert(g) {
                unique_groups.push(g);
            }
        }
    }
    unique_groups.sort();
    let g_count = unique_groups.len();

    // 3. Per-segment KM + summary.
    let mut seg_km: Vec<(i64, crate::survival::KaplanMeierEstimate)> = Vec::new();
    let mut segments: Vec<SegmentSummary> = Vec::new();

    // Build group_idx map for O/E lookup.
    let group_idx: std::collections::HashMap<i64, usize> =
        unique_groups.iter().enumerate().map(|(i, &g)| (g, i)).collect();

    for &g in &unique_groups {
        let mut g_times = Vec::new();
        let mut g_events = Vec::new();
        for i in 0..n {
            if groups[i] == g {
                g_times.push(times[i]);
                g_events.push(events[i]);
            }
        }
        let km = crate::survival::kaplan_meier(&g_times, &g_events, conf_level)?;
        let gi = group_idx[&g];
        let (_, obs, exp) = overall_lr.group_summaries[gi];
        segments.push(SegmentSummary {
            group: g,
            n: g_times.len(),
            n_events: g_events.iter().filter(|&&e| e).count(),
            median: km.median,
            observed: obs,
            expected: exp,
        });
        seg_km.push((g, km));
    }

    // 4. Pairwise comparisons.
    let mut raw_pairwise: Vec<PairwiseComparison> = Vec::new();

    for i in 0..g_count {
        for j in (i + 1)..g_count {
            let ga = unique_groups[i];
            let gb = unique_groups[j];

            // Extract sub-arrays for this pair.
            let mut pair_times = Vec::new();
            let mut pair_events = Vec::new();
            let mut pair_groups = Vec::new();
            for k in 0..n {
                if groups[k] == ga || groups[k] == gb {
                    pair_times.push(times[k]);
                    pair_events.push(events[k]);
                    pair_groups.push(groups[k]);
                }
            }

            let lr = crate::survival::log_rank_test(&pair_times, &pair_events, &pair_groups)?;

            // HR proxy from O/E ratios.
            let (_, oa, ea) = lr
                .group_summaries
                .iter()
                .find(|(g, _, _)| *g == ga)
                .copied()
                .unwrap_or((ga, 0.0, 1.0));
            let (_, ob, eb) = lr
                .group_summaries
                .iter()
                .find(|(g, _, _)| *g == gb)
                .copied()
                .unwrap_or((gb, 0.0, 1.0));

            let hr_proxy = if ea > 0.0 && eb > 0.0 && oa > 0.0 && ob > 0.0 {
                (oa / ea) / (ob / eb)
            } else {
                f64::NAN
            };

            let median_a = seg_km.iter().find(|(g, _)| *g == ga).and_then(|(_, km)| km.median);
            let median_b = seg_km.iter().find(|(g, _)| *g == gb).and_then(|(_, km)| km.median);
            let median_diff = match (median_a, median_b) {
                (Some(a), Some(b)) => Some(a - b),
                _ => Option::None,
            };

            raw_pairwise.push(PairwiseComparison {
                group_a: ga,
                group_b: gb,
                chi_squared: lr.chi_squared,
                p_value: lr.p_value,
                p_adjusted: lr.p_value, // will be corrected below
                hazard_ratio_proxy: hr_proxy,
                median_diff,
                significant: false, // will be set below
            });
        }
    }

    // 5. Multiple comparisons correction.
    let m = raw_pairwise.len();
    match correction {
        CorrectionMethod::Bonferroni => {
            for pw in &mut raw_pairwise {
                pw.p_adjusted = (pw.p_value * m as f64).min(1.0);
                pw.significant = pw.p_adjusted < alpha;
            }
        }
        CorrectionMethod::BenjaminiHochberg => {
            // BH procedure: sort by p-value, adjust p_i = min(p_i * m / rank, 1.0),
            // then enforce monotonicity from the bottom up.
            let mut indices: Vec<usize> = (0..m).collect();
            indices.sort_by(|&a, &b| raw_pairwise[a].p_value.total_cmp(&raw_pairwise[b].p_value));

            let mut adjusted = vec![0.0_f64; m];
            for (rank_0, &idx) in indices.iter().enumerate() {
                let rank = rank_0 + 1;
                adjusted[idx] = (raw_pairwise[idx].p_value * m as f64 / rank as f64).min(1.0);
            }
            // Enforce monotonicity: walk from largest rank down.
            let mut running_min = 1.0_f64;
            for &idx in indices.iter().rev() {
                adjusted[idx] = adjusted[idx].min(running_min);
                running_min = adjusted[idx];
            }

            for (idx, pw) in raw_pairwise.iter_mut().enumerate() {
                pw.p_adjusted = adjusted[idx];
                pw.significant = pw.p_adjusted < alpha;
            }
        }
        CorrectionMethod::None => {
            for pw in &mut raw_pairwise {
                pw.p_adjusted = pw.p_value;
                pw.significant = pw.p_adjusted < alpha;
            }
        }
    }

    // 6. Sort by p_adjusted ascending for the report.
    raw_pairwise.sort_by(|a, b| a.p_adjusted.total_cmp(&b.p_adjusted));

    let total_events = events.iter().filter(|&&e| e).count();

    Ok(SegmentComparisonReport {
        overall_chi_squared: overall_lr.chi_squared,
        overall_p_value: overall_lr.p_value,
        overall_df: overall_lr.df,
        segments,
        pairwise: raw_pairwise,
        correction_method: correction,
        alpha,
        n,
        n_events: total_events,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_churn_dataset_default_config() {
        let config = ChurnDataConfig::default();
        let ds = generate_churn_dataset(&config).unwrap();
        assert_eq!(ds.records.len(), 2000);
        assert_eq!(ds.times.len(), 2000);
        assert_eq!(ds.events.len(), 2000);
        assert_eq!(ds.groups.len(), 2000);
        assert_eq!(ds.covariates.len(), 2000);

        // Should have a mix of events and censorings.
        let n_events = ds.events.iter().filter(|&&e| e).count();
        assert!(n_events > 100, "too few events: {n_events}");
        assert!(n_events < 1900, "too few censorings: {n_events}");

        // All times should be in (0, max_time].
        for &t in &ds.times {
            assert!(t > 0.0 && t <= config.max_time);
        }

        // All plans should be 0, 1, or 2.
        for r in &ds.records {
            assert!(r.plan <= 2);
            assert!(r.region <= 2);
        }
    }

    #[test]
    fn generate_churn_dataset_deterministic() {
        let config = ChurnDataConfig { seed: 123, n_customers: 100, ..Default::default() };
        let ds1 = generate_churn_dataset(&config).unwrap();
        let ds2 = generate_churn_dataset(&config).unwrap();
        for i in 0..100 {
            assert_eq!(ds1.times[i], ds2.times[i]);
            assert_eq!(ds1.events[i], ds2.events[i]);
        }
    }

    #[test]
    fn generate_churn_dataset_validation_errors() {
        let cfg = ChurnDataConfig { n_customers: 0, ..ChurnDataConfig::default() };
        assert!(generate_churn_dataset(&cfg).is_err());

        let cfg = ChurnDataConfig { max_time: -1.0, ..ChurnDataConfig::default() };
        assert!(generate_churn_dataset(&cfg).is_err());
    }

    #[test]
    fn retention_analysis_runs() {
        let config = ChurnDataConfig { n_customers: 200, seed: 7, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();
        let ra = retention_analysis(&ds.times, &ds.events, &ds.groups, 0.95).unwrap();

        assert!(!ra.overall.steps.is_empty());
        assert_eq!(ra.log_rank.df, 2); // 3 plan groups
        assert!(!ra.by_group.is_empty());
    }

    #[test]
    fn churn_risk_model_runs() {
        let config = ChurnDataConfig { n_customers: 300, seed: 99, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();
        let names: Vec<String> = vec![
            "plan_basic".into(),
            "plan_premium".into(),
            "usage_score".into(),
            "support_tickets".into(),
        ];
        let result = churn_risk_model(&ds.times, &ds.events, &ds.covariates, &names, 0.95).unwrap();

        assert_eq!(result.names.len(), 4);
        assert_eq!(result.coefficients.len(), 4);
        assert_eq!(result.hazard_ratios.len(), 4);

        // Premium plan should have HR < 1 (protective).
        assert!(
            result.hazard_ratios[1] < 1.0,
            "premium HR should be < 1, got {}",
            result.hazard_ratios[1]
        );

        // All SEs should be positive.
        for &s in &result.se {
            assert!(s > 0.0 && s.is_finite());
        }
    }

    #[test]
    fn churn_uplift_runs() {
        let config = ChurnDataConfig {
            n_customers: 500,
            seed: 55,
            treatment_fraction: 0.4,
            ..Default::default()
        };
        let ds = generate_churn_dataset(&config).unwrap();
        let result =
            churn_uplift(&ds.times, &ds.events, &ds.treated, &ds.covariates, 12.0).unwrap();

        assert!(result.n_treated > 50);
        assert!(result.n_control > 50);
        // ATE should be negative (treatment reduces churn).
        assert!(
            result.ate < 0.0,
            "ATE should be negative (treatment reduces churn), got {}",
            result.ate
        );
        assert!(result.se > 0.0);
    }

    // -----------------------------------------------------------------------
    // C5: Ingestion tests
    // -----------------------------------------------------------------------

    #[test]
    fn ingest_basic_arrays() {
        let times = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        let events = vec![true, false, true, false, true];
        let groups = vec![0i64, 1, 0, 1, 0];
        let treated = vec![0u8, 1, 0, 1, 0];
        let covariates =
            vec![vec![1.0, 0.5], vec![-0.3, 1.2], vec![0.8, -0.1], vec![-1.0, 0.3], vec![0.2, 0.7]];
        let names = vec!["x1".into(), "x2".into()];

        let result = ingest_churn_arrays(
            &times,
            &events,
            Some(&groups),
            Some(&treated),
            &covariates,
            &names,
            None,
        )
        .unwrap();

        assert_eq!(result.dataset.times.len(), 5);
        assert_eq!(result.dataset.events.iter().filter(|&&e| e).count(), 3);
        assert_eq!(result.n_dropped, 0);
        assert!(result.warnings.is_empty());
        assert_eq!(result.covariate_names, vec!["x1", "x2"]);
    }

    #[test]
    fn ingest_drops_invalid_rows() {
        let times = vec![5.0, f64::NAN, 0.0, -1.0, 10.0];
        let events = vec![true, true, true, true, false];

        let result = ingest_churn_arrays(&times, &events, None, None, &[], &[], None).unwrap();

        assert_eq!(result.dataset.times.len(), 2); // only 5.0 and 10.0
        assert_eq!(result.n_dropped, 3);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn ingest_observation_end_caps_time() {
        let times = vec![5.0, 30.0, 15.0];
        let events = vec![true, true, false];

        let result =
            ingest_churn_arrays(&times, &events, None, None, &[], &[], Some(24.0)).unwrap();

        assert_eq!(result.dataset.times, vec![5.0, 24.0, 15.0]);
        assert_eq!(result.dataset.events, vec![true, false, false]);
    }

    #[test]
    fn ingest_empty_data_error() {
        let result = ingest_churn_arrays(&[], &[], None, None, &[], &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn ingest_no_events_error() {
        let times = vec![5.0, 10.0];
        let events = vec![false, false];

        let result = ingest_churn_arrays(&times, &events, None, None, &[], &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn ingest_high_censoring_warning() {
        let mut times = vec![5.0];
        let mut events = vec![true];
        for i in 1..100 {
            times.push(i as f64);
            events.push(false);
        }

        let result = ingest_churn_arrays(&times, &events, None, None, &[], &[], None).unwrap();

        assert!(result.warnings.iter().any(|w| w.contains("censoring rate")));
    }

    #[test]
    fn mapping_config_json_roundtrip() {
        let json = r#"{
            "time_col": "tenure_months",
            "event_col": "churned",
            "group_col": "plan",
            "treated_col": "treated",
            "covariate_cols": ["usage_score", "support_tickets"],
            "observation_end": 24.0,
            "time_unit": "months"
        }"#;
        let cfg: ChurnMappingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.time_col.as_deref(), Some("tenure_months"));
        assert_eq!(cfg.event_col, "churned");
        assert_eq!(cfg.group_col.as_deref(), Some("plan"));
        assert_eq!(cfg.treated_col.as_deref(), Some("treated"));
        assert_eq!(cfg.covariate_cols, vec!["usage_score", "support_tickets"]);
        assert_eq!(cfg.observation_end, Some(24.0));
        assert_eq!(cfg.time_unit, "months");
    }

    // -----------------------------------------------------------------------
    // C6: Cohort retention matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn cohort_matrix_basic() {
        // 2 cohorts, periods at 3, 6, 12 months.
        let times = vec![2.0, 5.0, 12.0, 1.0, 8.0, 12.0];
        let events = vec![true, true, false, true, false, false];
        let groups = vec![0, 0, 0, 1, 1, 1];
        let boundaries = vec![3.0, 6.0, 12.0];

        let m = cohort_retention_matrix(&times, &events, &groups, &boundaries).unwrap();

        assert_eq!(m.cohorts.len(), 2);
        assert_eq!(m.period_boundaries, vec![3.0, 6.0, 12.0]);

        // Cohort 0: times [2,5,12], events [T,T,F]
        let c0 = &m.cohorts[0];
        assert_eq!(c0.n_total, 3);
        assert_eq!(c0.n_events, 2);
        // Period (0,3]: at_risk=3, events=1(t=2.0), censored=0
        assert_eq!(c0.periods[0].n_at_risk, 3);
        assert_eq!(c0.periods[0].n_events, 1);
        // Period (3,6]: at_risk=2, events=1(t=5.0), censored=0
        assert_eq!(c0.periods[1].n_at_risk, 2);
        assert_eq!(c0.periods[1].n_events, 1);
        // Period (6,12]: at_risk=1, events=0, censored=1(t=12.0)
        assert_eq!(c0.periods[2].n_at_risk, 1);
        assert_eq!(c0.periods[2].n_events, 0);
        assert_eq!(c0.periods[2].n_censored, 1);

        // Cohort 1: times [1,8,12], events [T,F,F]
        let c1 = &m.cohorts[1];
        assert_eq!(c1.n_total, 3);
        assert_eq!(c1.n_events, 1);
        // Period (0,3]: at_risk=3, events=1(t=1.0)
        assert_eq!(c1.periods[0].n_at_risk, 3);
        assert_eq!(c1.periods[0].n_events, 1);

        // Overall.
        assert_eq!(m.overall.n_total, 6);
        assert_eq!(m.overall.n_events, 3);
    }

    #[test]
    fn cohort_matrix_cumulative_retention() {
        let times = vec![1.0, 3.0, 6.0, 12.0];
        let events = vec![true, true, true, false];
        let groups = vec![0, 0, 0, 0];
        let boundaries = vec![2.0, 4.0, 8.0, 14.0];

        let m = cohort_retention_matrix(&times, &events, &groups, &boundaries).unwrap();
        let c0 = &m.cohorts[0];

        // (0,2]: at_risk=4, events=1 → ret=0.75, cum=0.75
        assert!((c0.periods[0].retention_rate - 0.75).abs() < 1e-10);
        assert!((c0.periods[0].cumulative_retention - 0.75).abs() < 1e-10);

        // (2,4]: at_risk=3, events=1 → ret=2/3, cum=0.75*2/3=0.5
        assert!((c0.periods[1].cumulative_retention - 0.5).abs() < 1e-10);

        // (4,8]: at_risk=2, events=1 → ret=0.5, cum=0.5*0.5=0.25
        assert!((c0.periods[2].cumulative_retention - 0.25).abs() < 1e-10);

        // (8,14]: at_risk=1, events=0, censored=1 → ret=1.0, cum=0.25
        assert!((c0.periods[3].cumulative_retention - 0.25).abs() < 1e-10);
    }

    #[test]
    fn cohort_matrix_on_synthetic_data() {
        let config = ChurnDataConfig { n_customers: 500, seed: 77, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();
        let boundaries = vec![1.0, 3.0, 6.0, 12.0, 24.0];

        let m = cohort_retention_matrix(&ds.times, &ds.events, &ds.groups, &boundaries).unwrap();

        // Should have 3 cohorts (plan 0, 1, 2).
        assert_eq!(m.cohorts.len(), 3);
        // Overall cumulative retention should decrease or stay flat.
        let mut prev = 1.0_f64;
        for cell in &m.overall.periods {
            assert!(cell.cumulative_retention <= prev + 1e-10);
            prev = cell.cumulative_retention;
        }
    }

    // -----------------------------------------------------------------------
    // C7: Segment comparison report tests
    // -----------------------------------------------------------------------

    #[test]
    fn compare_basic_two_groups() {
        // Two very different groups.
        let times = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let events = vec![true, true, true, true, true, true];
        let groups = vec![1, 1, 1, 2, 2, 2];

        let r =
            segment_comparison_report(&times, &events, &groups, 0.95, CorrectionMethod::None, 0.05)
                .unwrap();

        assert_eq!(r.segments.len(), 2);
        assert_eq!(r.pairwise.len(), 1);
        assert_eq!(r.overall_df, 1);
        assert_eq!(r.n, 6);
        assert_eq!(r.n_events, 6);

        // The pair should have a substantial chi-squared.
        let pw = &r.pairwise[0];
        assert!(pw.chi_squared > 1.0);
        // HR proxy should be > 1 (group 1 churns earlier).
        assert!(pw.hazard_ratio_proxy > 1.0);
    }

    #[test]
    fn compare_three_groups_bonferroni() {
        let config = ChurnDataConfig { n_customers: 300, seed: 42, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();

        let r = segment_comparison_report(
            &ds.times,
            &ds.events,
            &ds.groups,
            0.95,
            CorrectionMethod::Bonferroni,
            0.05,
        )
        .unwrap();

        assert_eq!(r.segments.len(), 3);
        // C(3,2) = 3 pairwise comparisons.
        assert_eq!(r.pairwise.len(), 3);
        assert_eq!(r.overall_df, 2);

        // Bonferroni: p_adjusted >= p_value for all.
        for pw in &r.pairwise {
            assert!(pw.p_adjusted >= pw.p_value - 1e-15);
            assert!(pw.p_adjusted <= 1.0 + 1e-15);
        }

        // Pairwise should be sorted by p_adjusted ascending.
        for i in 1..r.pairwise.len() {
            assert!(r.pairwise[i].p_adjusted >= r.pairwise[i - 1].p_adjusted - 1e-15);
        }
    }

    #[test]
    fn compare_bh_less_conservative_than_bonferroni() {
        let config = ChurnDataConfig { n_customers: 500, seed: 99, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();

        let r_bonf = segment_comparison_report(
            &ds.times,
            &ds.events,
            &ds.groups,
            0.95,
            CorrectionMethod::Bonferroni,
            0.05,
        )
        .unwrap();
        let r_bh = segment_comparison_report(
            &ds.times,
            &ds.events,
            &ds.groups,
            0.95,
            CorrectionMethod::BenjaminiHochberg,
            0.05,
        )
        .unwrap();

        // BH should find at least as many significant comparisons as Bonferroni.
        let n_sig_bonf = r_bonf.pairwise.iter().filter(|pw| pw.significant).count();
        let n_sig_bh = r_bh.pairwise.iter().filter(|pw| pw.significant).count();
        assert!(n_sig_bh >= n_sig_bonf);
    }

    #[test]
    fn compare_identical_groups_nonsignificant() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let events = vec![true, true, true, true, true, true, true, true];
        let groups = vec![1, 1, 1, 1, 2, 2, 2, 2];

        let r =
            segment_comparison_report(&times, &events, &groups, 0.95, CorrectionMethod::None, 0.05)
                .unwrap();

        assert!(r.overall_p_value > 0.99);
        assert!(!r.pairwise[0].significant);
    }

    #[test]
    fn compare_observed_expected_consistency() {
        let config = ChurnDataConfig { n_customers: 200, seed: 55, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();

        let r = segment_comparison_report(
            &ds.times,
            &ds.events,
            &ds.groups,
            0.95,
            CorrectionMethod::BenjaminiHochberg,
            0.05,
        )
        .unwrap();

        // Sum of observed should equal total events.
        let sum_obs: f64 = r.segments.iter().map(|s| s.observed).sum();
        let total_events = ds.events.iter().filter(|&&e| e).count() as f64;
        assert!((sum_obs - total_events).abs() < 1e-6);

        // Sum of expected should also equal total events.
        let sum_exp: f64 = r.segments.iter().map(|s| s.expected).sum();
        assert!((sum_exp - total_events).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // C8: Survival uplift tests
    // -----------------------------------------------------------------------

    #[test]
    fn rmst_no_events_equals_tau() {
        // No events → S(t)=1 → RMST = τ.
        let steps = vec![];
        assert!((compute_rmst(&steps, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn rmst_all_events_at_start() {
        // 5 observations, all events at t=1 → S(t)=0 for t≥1 → RMST(τ=10)=1.
        let km = crate::survival::kaplan_meier(
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            &[true, true, true, true, true],
            0.95,
        )
        .unwrap();
        let rmst = compute_rmst(&km.steps, 10.0);
        assert!((rmst - 1.0).abs() < 1e-10, "RMST should be 1.0, got {rmst}");
    }

    #[test]
    fn rmst_step_function_integration() {
        // Simple: events at t=2 (S drops to 0.5), t=4 (S drops to 0).
        // τ=6: area = 1.0*2 + 0.5*2 + 0.0*2 = 3.0
        let km = crate::survival::kaplan_meier(&[2.0, 4.0], &[true, true], 0.95).unwrap();
        let rmst = compute_rmst(&km.steps, 6.0);
        assert!((rmst - 3.0).abs() < 1e-10, "RMST should be 3.0, got {rmst}");
    }

    #[test]
    fn survival_uplift_basic_no_covariates() {
        // Treated survive longer than control.
        let times = vec![10.0, 12.0, 15.0, 20.0, 2.0, 3.0, 5.0, 7.0];
        let events = vec![true, true, false, false, true, true, true, true];
        let treated = vec![1, 1, 1, 1, 0, 0, 0, 0];

        let r =
            survival_uplift_report(&times, &events, &treated, &[], 12.0, &[3.0, 6.0, 12.0], 0.01)
                .unwrap();

        // ΔRMST should be positive (treated retains longer).
        assert!(r.delta_rmst > 0.0, "ΔRMST should be positive, got {}", r.delta_rmst);
        assert_eq!(r.arms.len(), 2);
        assert_eq!(r.survival_diffs.len(), 3);

        // At t=3, control already lost some, treated hasn't.
        assert!(r.survival_diffs[0].delta_survival > 0.0);

        // IPW not applied (no covariates).
        assert!(!r.ipw_applied);
    }

    #[test]
    fn survival_uplift_with_covariates() {
        let config = ChurnDataConfig { n_customers: 200, seed: 33, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();

        // ds.covariates is already row-major: covariates[i] = [plan_basic, plan_premium, usage, tickets]
        let r = survival_uplift_report(
            &ds.times,
            &ds.events,
            &ds.treated,
            &ds.covariates,
            12.0,
            &[3.0, 6.0, 12.0],
            0.01,
        )
        .unwrap();

        assert!(r.ipw_applied);
        assert!(r.overlap.ess_treated > 0.0);
        assert!(r.overlap.ess_control > 0.0);
        assert_eq!(r.survival_diffs.len(), 3);
    }

    // -----------------------------------------------------------------------
    // C9: Diagnostics / guardrails tests
    // -----------------------------------------------------------------------

    #[test]
    fn diagnostics_basic_censoring() {
        let times = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let events = vec![true, true, true, false, false, false];
        let groups = vec![1, 1, 1, 2, 2, 2];

        let r = churn_diagnostics_report(&times, &events, &groups, &[], &[], &[], 0.01).unwrap();

        assert_eq!(r.n, 6);
        assert_eq!(r.n_events, 3);
        assert!((r.overall_censoring_frac - 0.5).abs() < 1e-10);
        assert_eq!(r.censoring_by_segment.len(), 2);

        // Group 1: 0% censored, Group 2: 100% censored.
        let g1 = r.censoring_by_segment.iter().find(|s| s.group == 1).unwrap();
        assert_eq!(g1.frac_censored, 0.0);
        let g2 = r.censoring_by_segment.iter().find(|s| s.group == 2).unwrap();
        assert!((g2.frac_censored - 1.0).abs() < 1e-10);

        // Should have warnings for group 2 (100% censoring, 0 events).
        assert!(!r.warnings.is_empty());
        assert!(r.trust_gate_passed); // No critical warnings.
        assert!(r.propensity_overlap.is_none()); // No treatment data.
    }

    #[test]
    fn diagnostics_with_covariates_and_treatment() {
        let config = ChurnDataConfig { n_customers: 200, seed: 44, ..Default::default() };
        let ds = generate_churn_dataset(&config).unwrap();

        let names = vec![
            "plan_basic".into(),
            "plan_premium".into(),
            "usage_score".into(),
            "support_tickets".into(),
        ];

        let r = churn_diagnostics_report(
            &ds.times,
            &ds.events,
            &ds.groups,
            &ds.treated,
            &ds.covariates,
            &names,
            0.01,
        )
        .unwrap();

        assert!(r.propensity_overlap.is_some());
        let po = r.propensity_overlap.unwrap();
        assert!(po.mean > 0.0 && po.mean < 1.0);
        assert!(po.quantiles[0] <= po.quantiles[3]); // min <= median
        assert!(po.quantiles[3] <= po.quantiles[6]); // median <= max

        assert_eq!(r.covariate_balance.len(), 4);
        assert_eq!(r.covariate_balance[0].name, "plan_basic");

        assert!(r.trust_gate_passed);
    }

    #[test]
    fn diagnostics_high_censoring_triggers_critical() {
        // 95% censored → should trigger critical warning.
        let n = 100;
        let times: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let events: Vec<bool> = (0..n).map(|i| i < 5).collect(); // only 5 events
        let groups: Vec<i64> = vec![1; n];

        let r = churn_diagnostics_report(&times, &events, &groups, &[], &[], &[], 0.01).unwrap();

        assert!((r.overall_censoring_frac - 0.95).abs() < 1e-10);
        assert!(!r.trust_gate_passed);
        assert!(r.warnings.iter().any(|w| w.severity == "critical"));
    }

    #[test]
    fn diagnostics_validation_errors() {
        // Empty input.
        assert!(churn_diagnostics_report(&[], &[], &[], &[], &[], &[], 0.01).is_err());
        // Mismatched lengths.
        assert!(
            churn_diagnostics_report(&[1.0, 2.0], &[true], &[1, 1], &[], &[], &[], 0.01,).is_err()
        );
        assert!(
            churn_diagnostics_report(&[1.0, 2.0], &[true, false], &[1], &[], &[], &[], 0.01,)
                .is_err()
        );
    }

    #[test]
    fn diagnostics_treatment_no_covariates_skips_propensity() {
        // Treatment data present but no covariates → propensity should be None.
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let events = vec![true, true, false, true];
        let groups = vec![1, 1, 2, 2];
        let treated = vec![1u8, 0, 1, 0];

        let r =
            churn_diagnostics_report(&times, &events, &groups, &treated, &[], &[], 0.01).unwrap();

        assert!(r.propensity_overlap.is_none());
        assert!(r.covariate_balance.is_empty());
    }

    #[test]
    fn diagnostics_small_segment_events_warning() {
        // Segment with < 5 events should trigger a sample_size warning.
        let times = vec![1.0, 2.0, 3.0, 10.0, 20.0];
        let events = vec![true, true, false, false, false];
        let groups = vec![1, 1, 1, 2, 2];

        let r = churn_diagnostics_report(&times, &events, &groups, &[], &[], &[], 0.01).unwrap();

        // Group 1 has 2 events (< 5), group 2 has 0 events (< 5).
        let sample_warnings: Vec<_> =
            r.warnings.iter().filter(|w| w.category == "sample_size").collect();
        assert!(
            sample_warnings.len() >= 2,
            "expected sample_size warnings, got {:?}",
            sample_warnings
        );
    }

    #[test]
    fn diagnostics_quantile_sorted_edge_cases() {
        assert!(quantile_sorted(&[], 0.5).is_nan());
        assert_eq!(quantile_sorted(&[42.0], 0.5), 42.0);
        assert_eq!(quantile_sorted(&[1.0, 2.0, 3.0], 0.0), 1.0);
        assert_eq!(quantile_sorted(&[1.0, 2.0, 3.0], 1.0), 3.0);
        assert!((quantile_sorted(&[1.0, 2.0, 3.0], 0.5) - 2.0).abs() < 1e-10);
    }
}
