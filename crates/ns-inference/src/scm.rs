//! Stepwise Covariate Modeling (SCM) for population PK models.
//!
//! Forward selection + backward elimination of covariate–parameter
//! relationships, using ΔOFV (likelihood ratio test, χ²(1)) as the
//! selection criterion.
//!
//! # Workflow
//!
//! 1. Fit a base (no-covariate) model.
//! 2. **Forward selection**: test each candidate covariate one at a time;
//!    add the one with the largest OFV drop exceeding the forward threshold
//!    (default: 3.84, α = 0.05). Repeat until no more significant additions.
//! 3. **Backward elimination**: from the full forward model, remove each
//!    covariate one at a time; drop the one with the smallest OFV increase
//!    below the backward threshold (default: 6.63, α = 0.01). Repeat until
//!    all remaining covariates are significant.
//!
//! Supported covariate relationships on base PK parameters (CL, V, Ka):
//! - **Power**: TV(P) = θ_P · (COV / center)^θ_cov
//! - **Proportional**: TV(P) = θ_P · (1 + θ_cov · (COV − center))
//! - **Exponential**: TV(P) = θ_P · exp(θ_cov · (COV − center))

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::foce::{FoceConfig, OmegaMatrix};
use crate::pk::{self, ErrorModel};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Covariate–parameter relationship type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovariateRelationship {
    /// TV(P) = θ_P · (COV / center)^θ_cov
    Power,
    /// TV(P) = θ_P · (1 + θ_cov · (COV − center))
    Proportional,
    /// TV(P) = θ_P · exp(θ_cov · (COV − center))
    Exponential,
}

/// A candidate covariate effect to test in SCM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateCandidate {
    /// Human-readable label (e.g. "WT on CL").
    pub name: String,
    /// Which base parameter: 0 = CL, 1 = V, 2 = Ka.
    pub param_index: usize,
    /// Per-subject covariate values (length = n_subjects).
    pub values: Vec<f64>,
    /// Centering value (e.g. median weight = 70 kg).
    pub center: f64,
    /// Relationship type.
    pub relationship: CovariateRelationship,
}

/// Configuration for the SCM algorithm.
#[derive(Debug, Clone)]
pub struct ScmConfig {
    /// p-value threshold for forward selection (default 0.05 → ΔOFV > 3.84).
    pub forward_alpha: f64,
    /// p-value threshold for backward elimination (default 0.01 → ΔOFV > 6.63).
    pub backward_alpha: f64,
    /// FOCE configuration used for each model refit.
    pub foce: FoceConfig,
}

impl Default for ScmConfig {
    fn default() -> Self {
        Self { forward_alpha: 0.05, backward_alpha: 0.01, foce: FoceConfig::default() }
    }
}

/// One step in the SCM forward/backward trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScmStep {
    /// Name of the covariate relationship tested.
    pub name: String,
    /// Parameter index (0 = CL, 1 = V, 2 = Ka).
    pub param_index: usize,
    /// Relationship type.
    pub relationship: CovariateRelationship,
    /// Change in OFV (negative = improvement when adding, positive = cost of removing).
    pub delta_ofv: f64,
    /// p-value from χ²(1) test.
    pub p_value: f64,
    /// Estimated covariate coefficient.
    pub coefficient: f64,
    /// Whether this step kept the covariate in the model.
    pub included: bool,
}

/// Result of the full SCM procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScmResult {
    /// Final selected covariate relationships (after backward elimination).
    pub selected: Vec<ScmStep>,
    /// Forward selection trace (one entry per addition).
    pub forward_trace: Vec<ScmStep>,
    /// Backward elimination trace (one entry per removal attempt).
    pub backward_trace: Vec<ScmStep>,
    /// Final population parameters `[CL, V, Ka, cov_coeff_1, ...]`.
    pub theta: Vec<f64>,
    /// Final Ω matrix.
    pub omega: OmegaMatrix,
    /// Final OFV.
    pub ofv: f64,
    /// Base (no-covariate) OFV for reference.
    pub base_ofv: f64,
}

// ---------------------------------------------------------------------------
// SCM engine
// ---------------------------------------------------------------------------

/// Stepwise Covariate Modeling estimator.
pub struct ScmEstimator {
    config: ScmConfig,
}

impl ScmEstimator {
    /// Create a new SCM estimator with the given configuration.
    pub fn new(config: ScmConfig) -> Self {
        Self { config }
    }

    /// Create an SCM estimator with default thresholds (α_fwd = 0.05, α_bwd = 0.01).
    pub fn with_defaults() -> Self {
        Self::new(ScmConfig::default())
    }

    /// Run stepwise covariate modeling on a 1-compartment oral PK model.
    ///
    /// # Arguments
    /// - `times`, `y`, `subject_idx`, `n_subjects`: observation data
    /// - `dose`, `bioav`: dosing information
    /// - `error_model`: residual error model
    /// - `theta_init`: initial `[CL_pop, V_pop, Ka_pop]`
    /// - `omega_init`: initial Ω matrix (3×3)
    /// - `candidates`: covariate relationships to test
    pub fn run_1cpt_oral(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &OmegaMatrix,
        candidates: &[CovariateCandidate],
    ) -> Result<ScmResult> {
        if theta_init.len() != 3 {
            return Err(Error::Validation("theta_init must have 3 elements".into()));
        }
        if omega_init.dim() != 3 {
            return Err(Error::Validation("omega must be 3×3".into()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".into()));
        }
        error_model.validate()?;
        for c in candidates {
            if c.param_index > 2 {
                return Err(Error::Validation(format!(
                    "param_index {} out of range 0..2 for '{}'",
                    c.param_index, c.name
                )));
            }
            if c.values.len() != n_subjects {
                return Err(Error::Validation(format!(
                    "covariate '{}' has {} values, expected {n_subjects}",
                    c.name,
                    c.values.len()
                )));
            }
        }

        let subj_obs = group_by_subject(times, y, subject_idx, n_subjects);

        // 1. Fit base model (no covariates).
        let base = fit_cov_foce(
            &subj_obs,
            n_subjects,
            dose,
            bioav,
            &error_model,
            theta_init,
            omega_init,
            &[],
            &self.config.foce,
        )?;
        let base_ofv = base.ofv;

        let chi2_fwd = chi2_critical_1df(self.config.forward_alpha);
        let chi2_bwd = chi2_critical_1df(self.config.backward_alpha);

        // Working state.
        let mut included: Vec<usize> = Vec::new();
        let mut current_theta = base.theta.clone();
        let mut current_omega = base.omega.clone();
        let mut current_ofv = base.ofv;
        let mut forward_trace: Vec<ScmStep> = Vec::new();

        // 2. Forward selection.
        loop {
            let mut best_ci: Option<usize> = None;
            let mut best_drop = 0.0_f64; // positive = OFV dropped
            let mut best_fit: Option<CovFitResult> = None;

            for (ci, cand) in candidates.iter().enumerate() {
                if included.contains(&ci) {
                    continue;
                }

                let mut trial_active = active_covariates(candidates, &included);
                trial_active.push(cand);

                let mut trial_theta = current_theta[..3].to_vec();
                // Carry over existing covariate coefficients.
                for &idx in &included {
                    let pos = included.iter().position(|&x| x == idx).unwrap();
                    if 3 + pos < current_theta.len() {
                        trial_theta.push(current_theta[3 + pos]);
                    } else {
                        trial_theta.push(init_coeff(cand.relationship));
                    }
                }
                trial_theta.push(init_coeff(cand.relationship));

                let fit = fit_cov_foce(
                    &subj_obs,
                    n_subjects,
                    dose,
                    bioav,
                    &error_model,
                    &trial_theta,
                    &current_omega,
                    &trial_active,
                    &self.config.foce,
                )?;

                let drop = current_ofv - fit.ofv;
                if drop > best_drop {
                    best_drop = drop;
                    best_ci = Some(ci);
                    best_fit = Some(fit);
                }
            }

            if best_drop > chi2_fwd {
                let ci = best_ci.unwrap();
                let fit = best_fit.unwrap();
                let p = chi2_sf_1df(best_drop);
                let coeff_idx = 3 + included.len();
                let coeff = if coeff_idx < fit.theta.len() { fit.theta[coeff_idx] } else { 0.0 };

                forward_trace.push(ScmStep {
                    name: candidates[ci].name.clone(),
                    param_index: candidates[ci].param_index,
                    relationship: candidates[ci].relationship,
                    delta_ofv: -best_drop,
                    p_value: p,
                    coefficient: coeff,
                    included: true,
                });

                included.push(ci);
                current_theta = fit.theta;
                current_omega = fit.omega;
                current_ofv = fit.ofv;
            } else {
                break;
            }
        }

        // 3. Backward elimination.
        let mut backward_trace: Vec<ScmStep> = Vec::new();
        loop {
            if included.is_empty() {
                break;
            }

            let mut worst_pos: Option<usize> = None;
            let mut worst_increase = f64::MAX;
            let mut worst_fit: Option<CovFitResult> = None;

            for pos in 0..included.len() {
                let mut trial_included = included.clone();
                trial_included.remove(pos);

                let trial_active = active_covariates(candidates, &trial_included);

                let mut trial_theta = current_theta[..3].to_vec();
                for _ in &trial_included {
                    trial_theta.push(init_coeff(CovariateRelationship::Power));
                }

                let fit = fit_cov_foce(
                    &subj_obs,
                    n_subjects,
                    dose,
                    bioav,
                    &error_model,
                    &trial_theta,
                    &current_omega,
                    &trial_active,
                    &self.config.foce,
                )?;

                let increase = fit.ofv - current_ofv;
                if increase < worst_increase {
                    worst_increase = increase;
                    worst_pos = Some(pos);
                    worst_fit = Some(fit);
                }
            }

            if worst_increase < chi2_bwd {
                let pos = worst_pos.unwrap();
                let ci = included[pos];
                let fit = worst_fit.unwrap();
                let p = chi2_sf_1df(worst_increase);

                backward_trace.push(ScmStep {
                    name: candidates[ci].name.clone(),
                    param_index: candidates[ci].param_index,
                    relationship: candidates[ci].relationship,
                    delta_ofv: worst_increase,
                    p_value: p,
                    coefficient: 0.0,
                    included: false,
                });

                included.remove(pos);
                current_theta = fit.theta;
                current_omega = fit.omega;
                current_ofv = fit.ofv;
            } else {
                break;
            }
        }

        // Build selected covariates from forward trace (only those still in `included`).
        let selected: Vec<ScmStep> = included
            .iter()
            .filter_map(|&ci| forward_trace.iter().find(|s| s.name == candidates[ci].name).cloned())
            .collect();

        Ok(ScmResult {
            selected,
            forward_trace,
            backward_trace,
            theta: current_theta,
            omega: current_omega,
            ofv: current_ofv,
            base_ofv,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal: covariate-aware FOCE fitting
// ---------------------------------------------------------------------------

struct CovFitResult {
    theta: Vec<f64>,
    omega: OmegaMatrix,
    ofv: f64,
}

fn group_by_subject(
    times: &[f64],
    y: &[f64],
    subject_idx: &[usize],
    n_subjects: usize,
) -> Vec<Vec<(f64, f64)>> {
    let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
    for i in 0..times.len() {
        subj_obs[subject_idx[i]].push((times[i], y[i]));
    }
    subj_obs
}

fn active_covariates<'a>(
    candidates: &'a [CovariateCandidate],
    included: &[usize],
) -> Vec<&'a CovariateCandidate> {
    included.iter().map(|&i| &candidates[i]).collect()
}

fn init_coeff(rel: CovariateRelationship) -> f64 {
    match rel {
        CovariateRelationship::Power => 0.75, // allometric default
        CovariateRelationship::Proportional => 0.0,
        CovariateRelationship::Exponential => 0.0,
    }
}

/// Compute individual concentration with covariate effects on base PK params.
#[inline]
fn individual_conc_cov(
    theta: &[f64],
    eta: &[f64],
    dose: f64,
    bioav: f64,
    t: f64,
    cov_effects: &[(usize, CovariateRelationship, f64, f64)], // (param_idx, rel, value, center)
) -> f64 {
    let mut tv = [theta[0], theta[1], theta[2]];
    for (i, &(pidx, rel, val, center)) in cov_effects.iter().enumerate() {
        let coeff = theta[3 + i];
        match rel {
            CovariateRelationship::Power => {
                let ratio = (val / center).abs().max(1e-10);
                tv[pidx] *= ratio.powf(coeff);
            }
            CovariateRelationship::Proportional => {
                tv[pidx] *= 1.0 + coeff * (val - center);
            }
            CovariateRelationship::Exponential => {
                tv[pidx] *= (coeff * (val - center)).exp();
            }
        }
    }
    let cl = tv[0] * eta[0].exp();
    let v = tv[1] * eta[1].exp();
    let ka = tv[2] * eta[2].exp();
    pk::conc_oral(dose, bioav, cl, v, ka, t)
}

/// Inner objective for one subject (covariate-aware).
fn inner_obj_cov(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    cov_effects: &[(usize, CovariateRelationship, f64, f64)],
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = individual_conc_cov(theta, eta, dose, bioav, t, cov_effects).max(1e-30);
        obj += em.nll_obs(yobs, c);
    }
    obj += 0.5 * omega.inv_quadratic(eta);
    obj
}

/// Inner optimization (covariate-aware Newton-Raphson).
fn inner_optimize_cov(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta_init: &[f64],
    max_iter: usize,
    cov_effects: &[(usize, CovariateRelationship, f64, f64)],
) -> Vec<f64> {
    let n = eta_init.len();
    let mut eta = eta_init.to_vec();
    let h = 1e-7;

    for _ in 0..max_iter {
        // Numerical gradient.
        let mut grad = vec![0.0; n];
        let mut eta_buf = eta.clone();
        for k in 0..n {
            let orig = eta_buf[k];
            eta_buf[k] = orig + h;
            let fp = inner_obj_cov(obs, theta, omega, em, dose, bioav, &eta_buf, cov_effects);
            eta_buf[k] = orig - h;
            let fm = inner_obj_cov(obs, theta, omega, em, dose, bioav, &eta_buf, cov_effects);
            eta_buf[k] = orig;
            grad[k] = (fp - fm) / (2.0 * h);
        }

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            break;
        }

        // Numerical Hessian (diagonal only for speed).
        let f0 = inner_obj_cov(obs, theta, omega, em, dose, bioav, &eta, cov_effects);
        let hd = 1e-5;
        let mut hess_diag = vec![0.0; n];
        for k in 0..n {
            let orig = eta[k];
            eta[k] = orig + hd;
            let fp = inner_obj_cov(obs, theta, omega, em, dose, bioav, &eta, cov_effects);
            eta[k] = orig - hd;
            let fm = inner_obj_cov(obs, theta, omega, em, dose, bioav, &eta, cov_effects);
            eta[k] = orig;
            hess_diag[k] = ((fp - 2.0 * f0 + fm) / (hd * hd)).max(1e-6);
        }

        // Damped Newton step.
        let obj_cur = f0;
        let mut step = 1.0;
        for _ in 0..5 {
            let trial: Vec<f64> = (0..n).map(|k| eta[k] - step * grad[k] / hess_diag[k]).collect();
            let obj_trial = inner_obj_cov(obs, theta, omega, em, dose, bioav, &trial, cov_effects);
            if obj_trial < obj_cur {
                eta = trial;
                break;
            }
            step *= 0.5;
        }
    }
    eta
}

/// Compute Hessian log-determinant for Laplace (reused from foce module logic).
fn hessian_log_det(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    cov_effects: &[(usize, CovariateRelationship, f64, f64)],
) -> f64 {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_obj_cov(obs, theta, omega, em, dose, bioav, eta, cov_effects);
    let mut hess = vec![vec![0.0; n]; n];
    let mut buf = eta.to_vec();

    for i in 0..n {
        let orig = buf[i];
        buf[i] = orig + h;
        let fp = inner_obj_cov(obs, theta, omega, em, dose, bioav, &buf, cov_effects);
        buf[i] = orig - h;
        let fm = inner_obj_cov(obs, theta, omega, em, dose, bioav, &buf, cov_effects);
        buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);
        for j in (i + 1)..n {
            let oi = buf[i];
            let oj = buf[j];
            buf[i] = oi + h;
            buf[j] = oj + h;
            let fpp = inner_obj_cov(obs, theta, omega, em, dose, bioav, &buf, cov_effects);
            buf[i] = oi + h;
            buf[j] = oj - h;
            let fpm = inner_obj_cov(obs, theta, omega, em, dose, bioav, &buf, cov_effects);
            buf[i] = oi - h;
            buf[j] = oj + h;
            let fmp = inner_obj_cov(obs, theta, omega, em, dose, bioav, &buf, cov_effects);
            buf[i] = oi - h;
            buf[j] = oj - h;
            let fmm = inner_obj_cov(obs, theta, omega, em, dose, bioav, &buf, cov_effects);
            buf[i] = oi;
            buf[j] = oj;
            let val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }

    // log det with ridge regularization.
    let ridge = 1e-8;
    for i in 0..n {
        hess[i][i] += ridge;
    }
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = hess[i][i] - sum;
                if diag <= 0.0 {
                    return hess.iter().enumerate().map(|(k, row)| row[k].max(1e-20).ln()).sum();
                }
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = (hess[i][j] - sum) / l[j][j];
            }
        }
    }
    2.0 * l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>()
}

/// Full covariate-aware FOCE fit.
fn fit_cov_foce(
    subj_obs: &[Vec<(f64, f64)>],
    n_subjects: usize,
    dose: f64,
    bioav: f64,
    em: &ErrorModel,
    theta_init: &[f64],
    omega_init: &OmegaMatrix,
    active_covs: &[&CovariateCandidate],
    foce_cfg: &FoceConfig,
) -> Result<CovFitResult> {
    let n_eta = 3;

    // Build per-subject covariate effect descriptors.
    let cov_descriptors: Vec<Vec<(usize, CovariateRelationship, f64, f64)>> = (0..n_subjects)
        .map(|s| {
            active_covs
                .iter()
                .map(|c| (c.param_index, c.relationship, c.values[s], c.center))
                .collect()
        })
        .collect();

    let mut theta = theta_init.to_vec();
    let n_theta = theta.len();
    let mut om = omega_init.clone();
    let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];
    let mut prev_ofv = f64::MAX;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();

    for iter in 0..foce_cfg.max_outer_iter {
        // Inner: optimize etas.
        for s in 0..n_subjects {
            if subj_obs[s].is_empty() {
                continue;
            }
            etas[s] = inner_optimize_cov(
                &subj_obs[s],
                &theta,
                &om,
                em,
                dose,
                bioav,
                &etas[s],
                foce_cfg.max_inner_iter,
                &cov_descriptors[s],
            );
        }

        // Compute OFV.
        let log_det_omega = om.log_det();
        let mut ofv = 0.0;
        for s in 0..n_subjects {
            if subj_obs[s].is_empty() {
                continue;
            }
            let nll_s = inner_obj_cov(
                &subj_obs[s],
                &theta,
                &om,
                em,
                dose,
                bioav,
                &etas[s],
                &cov_descriptors[s],
            );
            let ld = hessian_log_det(
                &subj_obs[s],
                &theta,
                &om,
                em,
                dose,
                bioav,
                &etas[s],
                &cov_descriptors[s],
            );
            ofv += 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
        }

        // Convergence check.
        if (prev_ofv - ofv).abs() < foce_cfg.tol && iter > 0 {
            prev_ofv = ofv;
            break;
        }
        prev_ofv = ofv;

        // Outer step: update theta via gradient descent.
        let h = 1e-7;
        let cond_nll = |th: &[f64]| -> f64 {
            if th.iter().take(3).any(|v| !v.is_finite() || *v <= 0.0) {
                return f64::MAX;
            }
            let mut nll = 0.0;
            for s in 0..n_subjects {
                for &(t, yobs) in &subj_obs[s] {
                    let c = individual_conc_cov(th, &etas[s], dose, bioav, t, &cov_descriptors[s])
                        .max(1e-30);
                    nll += em.nll_obs(yobs, c);
                }
            }
            nll
        };

        let nll0 = cond_nll(&theta);
        let mut grad = vec![0.0; n_theta];
        let mut th_buf = theta.clone();
        for j in 0..n_theta {
            let orig = th_buf[j];
            th_buf[j] = orig + h;
            let fp = cond_nll(&th_buf);
            th_buf[j] = orig - h;
            let fm = cond_nll(&th_buf);
            th_buf[j] = orig;
            grad[j] = (fp - fm) / (2.0 * h);
        }

        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
        let base_lr = 0.01;
        let mut lr =
            base_lr * theta.iter().map(|v| v.abs()).sum::<f64>() / (n_theta as f64 * grad_norm);

        for _ in 0..15 {
            let trial: Vec<f64> = theta
                .iter()
                .enumerate()
                .zip(grad.iter())
                .map(|((j, &p), &g)| {
                    let v = p - lr * g;
                    if j < 3 { v.max(1e-6) } else { v }
                })
                .collect();
            if cond_nll(&trial) < nll0 {
                theta = trial;
                break;
            }
            lr *= 0.5;
        }

        // Update omega from empirical covariance.
        let active_etas: Vec<Vec<f64>> = etas
            .iter()
            .enumerate()
            .filter(|(s, _)| !subj_obs[*s].is_empty())
            .map(|(_, e)| e.clone())
            .collect();
        if !active_etas.is_empty() {
            om = OmegaMatrix::empirical(&active_etas, n_eta)?;
        }
    }

    Ok(CovFitResult { theta, omega: om, ofv: prev_ofv })
}

// ---------------------------------------------------------------------------
// Chi-squared helpers (df = 1)
// ---------------------------------------------------------------------------

/// Approximate error function (Abramowitz & Stegun 7.1.26, |ε| < 2.5e-5).
fn erf_approx(x: f64) -> f64 {
    if x < 0.0 {
        return -erf_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.47047 * x);
    let poly = t * (0.3480242 + t * (-0.0958798 + t * 0.7478556));
    1.0 - poly * (-x * x).exp()
}

/// χ²(1) CDF: P(X ≤ x) = erf(√(x/2)).
fn chi2_cdf_1df(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    erf_approx((x / 2.0).sqrt())
}

/// χ²(1) survival function: P(X > x) = 1 − CDF(x).
fn chi2_sf_1df(x: f64) -> f64 {
    1.0 - chi2_cdf_1df(x)
}

/// Critical value for χ²(1) at significance level α.
/// Solved via bisection on chi2_cdf_1df.
fn chi2_critical_1df(alpha: f64) -> f64 {
    let target = 1.0 - alpha;
    let mut lo = 0.0_f64;
    let mut hi = 30.0_f64;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if chi2_cdf_1df(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};

    #[test]
    fn chi2_critical_values() {
        let c005 = chi2_critical_1df(0.05);
        assert!((c005 - 3.841).abs() < 0.05, "χ²(1) critical at 0.05: {c005}");
        let c001 = chi2_critical_1df(0.01);
        assert!((c001 - 6.635).abs() < 0.05, "χ²(1) critical at 0.01: {c001}");
    }

    #[test]
    fn chi2_sf_basic() {
        let p = chi2_sf_1df(3.841);
        assert!((p - 0.05).abs() < 0.01, "P(χ²(1) > 3.841) ≈ 0.05, got {p}");
    }

    #[test]
    fn scm_rejects_unrelated_covariate() {
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let n_subjects: usize = 12;

        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let eta_dist = RandNormal::new(0.0, 0.2).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = eta_dist.sample(&mut rng);
            let eta_v: f64 = eta_dist.sample(&mut rng);
            let eta_ka: f64 = eta_dist.sample(&mut rng);
            let cl_i = cl_pop * eta_cl.exp();
            let v_i = v_pop * eta_v.exp();
            let ka_i = ka_pop * eta_ka.exp();

            for &t in &times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        // Random covariate with no true effect on CL.
        let random_cov: Vec<f64> =
            (0..n_subjects).map(|i| 60.0 + 20.0 * (i as f64 / n_subjects as f64)).collect();

        let candidates = vec![CovariateCandidate {
            name: "RANDOM_on_CL".to_string(),
            param_index: 0,
            values: random_cov,
            center: 70.0,
            relationship: CovariateRelationship::Power,
        }];

        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let scm = ScmEstimator::with_defaults();
        let result = scm
            .run_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                dose,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.5],
                &omega,
                &candidates,
            )
            .unwrap();

        // With a random covariate and small N, SCM should not select it.
        assert!(
            result.selected.is_empty(),
            "SCM selected {} covariates for a random covariate, expected 0",
            result.selected.len()
        );
        assert!(result.ofv.is_finite());
        assert!(result.base_ofv.is_finite());
    }

    #[test]
    fn scm_selects_true_weight_effect() {
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let n_subjects: usize = 30;
        let true_wt_exponent = 0.75; // allometric on CL

        let times_per = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let eta_dist = RandNormal::new(0.0, 0.15).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let weights: Vec<f64> =
            (0..n_subjects).map(|i| 40.0 + 60.0 * (i as f64 / (n_subjects - 1) as f64)).collect();
        let wt_center = 70.0;

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = eta_dist.sample(&mut rng);
            let eta_v: f64 = eta_dist.sample(&mut rng);
            let eta_ka: f64 = eta_dist.sample(&mut rng);
            let cl_i = cl_pop * (weights[sid] / wt_center).powf(true_wt_exponent) * eta_cl.exp();
            let v_i = v_pop * eta_v.exp();
            let ka_i = ka_pop * eta_ka.exp();

            for &t in &times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        let candidates = vec![
            CovariateCandidate {
                name: "WT_on_CL".to_string(),
                param_index: 0,
                values: weights.clone(),
                center: wt_center,
                relationship: CovariateRelationship::Power,
            },
            CovariateCandidate {
                name: "WT_on_V".to_string(),
                param_index: 1,
                values: weights,
                center: wt_center,
                relationship: CovariateRelationship::Power,
            },
        ];

        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let scm = ScmEstimator::with_defaults();
        let result = scm
            .run_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                dose,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.5],
                &omega,
                &candidates,
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert!(
            result.ofv <= result.base_ofv,
            "final OFV {} > base {}",
            result.ofv,
            result.base_ofv
        );

        // With 30 subjects and strong weight effect, WT_on_CL should be selected.
        let wt_cl_selected = result.selected.iter().any(|s| s.name == "WT_on_CL");
        assert!(
            wt_cl_selected,
            "WT_on_CL not selected; selected = {:?}",
            result.selected.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn scm_config_defaults() {
        let cfg = ScmConfig::default();
        assert!((cfg.forward_alpha - 0.05).abs() < 1e-10);
        assert!((cfg.backward_alpha - 0.01).abs() < 1e-10);
    }

    #[test]
    fn scm_validates_inputs() {
        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let scm = ScmEstimator::with_defaults();

        let err = scm
            .run_1cpt_oral(
                &[1.0],
                &[2.0],
                &[0],
                1,
                100.0,
                1.0,
                ErrorModel::Additive(0.1),
                &[1.0, 10.0], // wrong length
                &omega,
                &[],
            )
            .unwrap_err();
        assert!(err.to_string().contains("theta_init"));
    }

    #[test]
    fn individual_conc_cov_power() {
        let theta = [1.0, 10.0, 2.0, 0.75]; // base CL=1, V=10, Ka=2, WT exponent=0.75
        let eta = [0.0, 0.0, 0.0];
        let cov = vec![(0, CovariateRelationship::Power, 100.0, 70.0)]; // WT=100, center=70
        let c1 = individual_conc_cov(&theta, &eta, 100.0, 1.0, 1.0, &cov);

        // Without covariate (WT=70):
        let c0 = pk::conc_oral(100.0, 1.0, 1.0, 10.0, 2.0, 1.0);

        // With covariate: CL = 1.0 * (100/70)^0.75
        let cl_adj = 1.0 * (100.0_f64 / 70.0).powf(0.75);
        let c_expected = pk::conc_oral(100.0, 1.0, cl_adj, 10.0, 2.0, 1.0);

        assert!((c1 - c_expected).abs() < 1e-10, "conc: {c1} vs expected {c_expected}");
        assert!((c1 - c0).abs() > 0.01, "covariate should change concentration");
    }

    // -----------------------------------------------------------------------
    // Deterministic reference tests
    // -----------------------------------------------------------------------

    /// Helper: generate synthetic 1-cpt oral data with optional weight effect on CL.
    fn generate_pk_data(
        n_subjects: usize,
        times_per: &[f64],
        cl_pop: f64,
        v_pop: f64,
        ka_pop: f64,
        dose: f64,
        bioav: f64,
        sigma: f64,
        omega_sd: f64,
        weights: &[f64],
        wt_center: f64,
        wt_exponent: f64, // 0.0 = no weight effect
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let eta_dist = RandNormal::new(0.0, omega_sd).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = eta_dist.sample(&mut rng);
            let eta_v: f64 = eta_dist.sample(&mut rng);
            let eta_ka: f64 = eta_dist.sample(&mut rng);

            let wt_effect = if wt_exponent.abs() > 1e-12 {
                (weights[sid] / wt_center).powf(wt_exponent)
            } else {
                1.0
            };
            let cl_i = cl_pop * wt_effect * eta_cl.exp();
            let v_i = v_pop * eta_v.exp();
            let ka_i = ka_pop * eta_ka.exp();

            for &t in times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }
        (times, y, subject_idx)
    }

    #[test]
    fn scm_deterministic_forward_selection() {
        // Synthetic data with a TRUE weight effect on CL (exponent = 0.75).
        // With 40 subjects, wide weight range, and moderate IIV, forward selection
        // must select WT_on_CL.
        let n_subjects: usize = 40;
        let times_per = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        let weights: Vec<f64> =
            (0..n_subjects).map(|i| 40.0 + 60.0 * i as f64 / (n_subjects - 1) as f64).collect();
        let wt_center = 70.0;

        let (times, y, subject_idx) = generate_pk_data(
            n_subjects,
            &times_per,
            1.2,   // cl_pop
            15.0,  // v_pop
            2.0,   // ka_pop
            100.0, // dose
            1.0,   // bioav
            0.04,  // sigma (low noise)
            0.12,  // omega_sd (moderate IIV)
            &weights,
            wt_center,
            0.75, // true wt exponent
            77,   // seed (matches known-good configuration)
        );

        let candidates = vec![CovariateCandidate {
            name: "WT_on_CL".to_string(),
            param_index: 0,
            values: weights,
            center: wt_center,
            relationship: CovariateRelationship::Power,
        }];

        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let scm = ScmEstimator::with_defaults();
        let result = scm
            .run_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                100.0,
                1.0,
                ErrorModel::Additive(0.05),
                &[1.0, 10.0, 1.5],
                &omega,
                &candidates,
            )
            .unwrap();

        // Forward selection should select WT_on_CL.
        assert!(
            !result.forward_trace.is_empty(),
            "forward trace should not be empty when a true effect exists"
        );
        assert!(
            result.forward_trace[0].name == "WT_on_CL",
            "first forward step should be WT_on_CL, got '{}'",
            result.forward_trace[0].name
        );
        assert!(
            result.forward_trace[0].delta_ofv < 0.0,
            "delta_OFV should be negative (improvement), got {}",
            result.forward_trace[0].delta_ofv
        );
        assert!(
            result.forward_trace[0].p_value < 0.05,
            "p-value should be < 0.05, got {}",
            result.forward_trace[0].p_value
        );
        // Should remain in selected after backward elimination.
        assert!(
            result.selected.iter().any(|s| s.name == "WT_on_CL"),
            "WT_on_CL should be in final selected set"
        );
        assert!(result.ofv < result.base_ofv, "final OFV should improve over base");
    }

    #[test]
    fn scm_deterministic_backward_elimination() {
        // Use 12 subjects, small IIV, and a RANDOM (non-significant) covariate.
        // Forward might accidentally add it with a liberal threshold, but backward
        // should remove it with the stricter threshold.
        //
        // We configure a very liberal forward threshold (alpha = 0.50) so the
        // random covariate has a decent chance of sneaking in, then verify that
        // backward elimination (alpha = 0.01) removes it.
        let n_subjects: usize = 12;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

        // No true covariate effect — data generated WITHOUT weight scaling.
        let weights: Vec<f64> =
            (0..n_subjects).map(|i| 50.0 + 30.0 * i as f64 / (n_subjects - 1) as f64).collect();

        let (times, y, subject_idx) = generate_pk_data(
            n_subjects,
            &times_per,
            1.2,
            15.0,
            2.0,
            100.0,
            1.0,
            0.05,
            0.20,
            &weights,
            70.0,
            0.0, // NO weight effect
            555,
        );

        let candidates = vec![CovariateCandidate {
            name: "NOISE_on_CL".to_string(),
            param_index: 0,
            values: weights,
            center: 70.0,
            relationship: CovariateRelationship::Power,
        }];

        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();

        // Very liberal forward, strict backward.
        let config = ScmConfig {
            forward_alpha: 0.50, // delta_OFV > ~0.45
            backward_alpha: 0.01,
            foce: FoceConfig::default(),
        };
        let scm = ScmEstimator::new(config);
        let result = scm
            .run_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                100.0,
                1.0,
                ErrorModel::Additive(0.05),
                &[1.0, 10.0, 1.5],
                &omega,
                &candidates,
            )
            .unwrap();

        // If forward added the noise covariate, backward should have removed it.
        // Either way, the final selected set should be empty for a noise covariate.
        assert!(
            result.selected.is_empty(),
            "backward elimination should remove non-significant covariate; selected = {:?}",
            result.selected.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        assert!(result.ofv.is_finite());
    }

    #[test]
    fn scm_no_covariates_selected() {
        // All covariates are pure noise — nothing should be selected.
        let n_subjects: usize = 15;
        let times_per = [0.5, 1.0, 2.0, 4.0, 8.0];

        // Generate data with NO covariate effects.
        let noise_cov1: Vec<f64> =
            (0..n_subjects).map(|i| 60.0 + 20.0 * (i as f64 / n_subjects as f64)).collect();
        let noise_cov2: Vec<f64> =
            (0..n_subjects).map(|i| 30.0 + 10.0 * ((n_subjects - i) as f64 / n_subjects as f64)).collect();

        let (times, y, subject_idx) = generate_pk_data(
            n_subjects,
            &times_per,
            1.2,
            15.0,
            2.0,
            100.0,
            1.0,
            0.05,
            0.20,
            &noise_cov1, // not used in data generation (exponent = 0)
            70.0,
            0.0,
            999,
        );

        let candidates = vec![
            CovariateCandidate {
                name: "NOISE1_on_CL".to_string(),
                param_index: 0,
                values: noise_cov1,
                center: 70.0,
                relationship: CovariateRelationship::Power,
            },
            CovariateCandidate {
                name: "NOISE2_on_V".to_string(),
                param_index: 1,
                values: noise_cov2,
                center: 35.0,
                relationship: CovariateRelationship::Exponential,
            },
        ];

        let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let scm = ScmEstimator::with_defaults();
        let result = scm
            .run_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                100.0,
                1.0,
                ErrorModel::Additive(0.05),
                &[1.0, 10.0, 1.5],
                &omega,
                &candidates,
            )
            .unwrap();

        assert!(
            result.selected.is_empty(),
            "no covariates should be selected when all are noise; got {:?}",
            result.selected.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        assert!(result.ofv.is_finite());
        assert!(result.base_ofv.is_finite());
        // OFV should not change much (no covariates added).
        assert!(
            (result.ofv - result.base_ofv).abs() < 1e-6,
            "OFV should equal base when no covariates selected: {} vs {}",
            result.ofv,
            result.base_ofv
        );
    }

    #[test]
    fn scm_multiple_covariates() {
        // Two TRUE covariate effects:
        //   1. Weight on CL (power, exponent = 0.75) — strong
        //   2. Weight on V (power, exponent = 1.0) — moderate (isometric)
        // Plus one noise covariate (random values on Ka).
        // Expected: WT_on_CL selected first (strongest), then WT_on_V.
        // NOISE_on_Ka should NOT be selected.
        let n_subjects: usize = 40;
        let times_per = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        let weights: Vec<f64> =
            (0..n_subjects).map(|i| 40.0 + 60.0 * i as f64 / (n_subjects - 1) as f64).collect();
        let wt_center = 70.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let eta_dist = RandNormal::new(0.0, 0.12).unwrap();
        let noise_dist = RandNormal::new(0.0, 0.04).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = eta_dist.sample(&mut rng);
            let eta_v: f64 = eta_dist.sample(&mut rng);
            let eta_ka: f64 = eta_dist.sample(&mut rng);

            // CL has weight effect (power 0.75)
            let cl_i = 1.2 * (weights[sid] / wt_center).powf(0.75) * eta_cl.exp();
            // V has weight effect (power 1.0, isometric)
            let v_i = 15.0 * (weights[sid] / wt_center).powf(1.0) * eta_v.exp();
            let ka_i = 2.0 * eta_ka.exp();

            for &t in &times_per {
                let c = pk::conc_oral(100.0, 1.0, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise_dist.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        // Noise covariate: random values, no relationship to any PK parameter.
        let noise_vals: Vec<f64> = {
            let mut rng2 = rand::rngs::StdRng::seed_from_u64(7777);
            let d = RandNormal::new(50.0, 10.0).unwrap();
            (0..n_subjects).map(|_| d.sample(&mut rng2)).collect()
        };

        let candidates = vec![
            CovariateCandidate {
                name: "WT_on_CL".to_string(),
                param_index: 0,
                values: weights.clone(),
                center: wt_center,
                relationship: CovariateRelationship::Power,
            },
            CovariateCandidate {
                name: "WT_on_V".to_string(),
                param_index: 1,
                values: weights,
                center: wt_center,
                relationship: CovariateRelationship::Power,
            },
            CovariateCandidate {
                name: "NOISE_on_Ka".to_string(),
                param_index: 2,
                values: noise_vals,
                center: 50.0,
                relationship: CovariateRelationship::Power,
            },
        ];

        let omega = OmegaMatrix::from_diagonal(&[0.25, 0.25, 0.25]).unwrap();
        let scm = ScmEstimator::with_defaults();
        let result = scm
            .run_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                100.0,
                1.0,
                ErrorModel::Additive(0.04),
                &[1.0, 10.0, 1.5],
                &omega,
                &candidates,
            )
            .unwrap();

        // WT_on_CL should be selected.
        assert!(
            result.selected.iter().any(|s| s.name == "WT_on_CL"),
            "WT_on_CL should be selected; selected = {:?}",
            result.selected.iter().map(|s| &s.name).collect::<Vec<_>>()
        );

        // NOISE_on_Ka should NOT be selected.
        assert!(
            !result.selected.iter().any(|s| s.name == "NOISE_on_Ka"),
            "NOISE_on_Ka should not be selected; selected = {:?}",
            result.selected.iter().map(|s| &s.name).collect::<Vec<_>>()
        );

        // Forward trace should show WT_on_CL as the first addition
        // (it has the strongest true effect).
        assert!(
            !result.forward_trace.is_empty(),
            "forward trace should not be empty"
        );
        assert_eq!(
            result.forward_trace[0].name, "WT_on_CL",
            "WT_on_CL should be selected first (strongest effect); got '{}'",
            result.forward_trace[0].name
        );

        // Final OFV should be better than base.
        assert!(
            result.ofv < result.base_ofv,
            "final OFV {} should be < base OFV {}",
            result.ofv,
            result.base_ofv
        );
    }
}
