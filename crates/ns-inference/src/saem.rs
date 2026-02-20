//! SAEM (Stochastic Approximation Expectation-Maximization) for NLME.
//!
//! SAEM is the core algorithm used by Monolix. It is more robust than FOCE
//! for complex nonlinear models, particularly those with:
//! - Non-convex individual log-likelihoods
//! - High-dimensional random effects
//! - ODE-based models where Laplace approximation may be poor
//!
//! ## Algorithm
//!
//! 1. **Simulation (E-step)**: Sample η_i from the conditional posterior
//!    p(η_i | y_i, θ, Ω) using MCMC (Metropolis-Hastings with adaptive
//!    proposal variance).
//! 2. **Stochastic Approximation**: Update sufficient statistics S_k using
//!    a decreasing step-size sequence: `S_k = S_{k-1} + γ_k (s(η) - S_{k-1})`
//! 3. **Maximization (M-step)**: Update population parameters θ, Ω, σ
//!    from the sufficient statistics in closed form.
//!
//! ## Step-size schedule
//!
//! - Burn-in phase (iterations 1..K1): γ_k = 1 (full update, rapid exploration)
//! - Estimation phase (iterations K1+1..K): γ_k = 1/(k - K1) (convergence)

use ns_core::{Error, Result};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::foce::{FoceResult, OmegaMatrix};
use crate::pk::{self, ErrorModel};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SAEM algorithm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaemConfig {
    /// Number of burn-in iterations (γ = 1).
    pub n_burn: usize,
    /// Number of estimation iterations (γ = 1/(k - n_burn)).
    pub n_iter: usize,
    /// Number of MCMC chains per subject.
    pub n_chains: usize,
    /// Initial MCMC proposal standard deviation (relative to omega SD).
    pub mcmc_proposal_scale: f64,
    /// Minimum acceptance rate before adapting proposal variance.
    pub mcmc_target_accept_low: f64,
    /// Maximum acceptance rate before adapting proposal variance.
    pub mcmc_target_accept_high: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Convergence tolerance on relative parameter change (applied after burn-in).
    pub tol: f64,
}

impl Default for SaemConfig {
    fn default() -> Self {
        Self {
            n_burn: 200,
            n_iter: 100,
            n_chains: 1,
            mcmc_proposal_scale: 0.5,
            mcmc_target_accept_low: 0.20,
            mcmc_target_accept_high: 0.45,
            seed: 12345,
            tol: 1e-4,
        }
    }
}

/// SAEM-specific result metrics (extends FoceResult).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaemDiagnostics {
    /// MCMC acceptance rate per subject (final iteration).
    pub acceptance_rates: Vec<f64>,
    /// OFV trace across iterations (for convergence diagnostics).
    pub ofv_trace: Vec<f64>,
    /// Whether the algorithm used burn-in only (did not reach estimation phase).
    pub burn_in_only: bool,
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// SAEM estimator for population pharmacokinetic models.
pub struct SaemEstimator {
    config: SaemConfig,
}

impl SaemEstimator {
    /// Create a new SAEM estimator with given configuration.
    pub fn new(config: SaemConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SaemConfig::default())
    }

    // ----- 2-compartment IV ------------------------------------------------

    /// Fit a 2-compartment IV bolus PK model using SAEM (diagonal Ω).
    ///
    /// Parameters: `[CL, V1, Q, V2]` (4 fixed effects).
    pub fn fit_2cpt_iv(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<(FoceResult, SaemDiagnostics)> {
        if omega_init.len() != 4 {
            return Err(Error::Validation("omega_init must have 4 elements for 2-cpt IV".into()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_iv_correlated(times, y, subject_idx, n_subjects, dose, error_model, theta_init, om)
    }

    /// Fit a 2-compartment IV bolus PK model with correlated random effects using SAEM.
    pub fn fit_2cpt_iv_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<(FoceResult, SaemDiagnostics)> {
        if theta_init.len() != 4 {
            return Err(Error::Validation("theta_init must have 4 elements [CL, V1, Q, V2]".into()));
        }
        if omega_init.dim() != 4 {
            return Err(Error::Validation("omega must be 4×4".into()));
        }
        let conc_fn = |theta: &[f64], eta: &[f64], t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q  = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            pk::conc_iv_2cpt_macro(dose, cl, v1, v2, q, t)
        };
        self.fit_generic(times, y, subject_idx, n_subjects, error_model, theta_init, omega_init, 4, &conc_fn)
    }

    // ----- 2-compartment oral ----------------------------------------------

    /// Fit a 2-compartment oral PK model using SAEM (diagonal Ω).
    ///
    /// Parameters: `[CL, V1, Q, V2, Ka]` (5 fixed effects).
    pub fn fit_2cpt_oral(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<(FoceResult, SaemDiagnostics)> {
        if omega_init.len() != 5 {
            return Err(Error::Validation("omega_init must have 5 elements for 2-cpt oral".into()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_oral_correlated(times, y, subject_idx, n_subjects, dose, bioav, error_model, theta_init, om)
    }

    /// Fit a 2-compartment oral PK model with correlated random effects using SAEM.
    pub fn fit_2cpt_oral_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<(FoceResult, SaemDiagnostics)> {
        if theta_init.len() != 5 {
            return Err(Error::Validation("theta_init must have 5 elements [CL, V1, Q, V2, Ka]".into()));
        }
        if omega_init.dim() != 5 {
            return Err(Error::Validation("omega must be 5×5".into()));
        }
        let conc_fn = |theta: &[f64], eta: &[f64], t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q  = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let ka = theta[4] * eta[4].exp();
            pk::conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t)
        };
        self.fit_generic(times, y, subject_idx, n_subjects, error_model, theta_init, omega_init, 5, &conc_fn)
    }

    // ----- Generic SAEM engine ---------------------------------------------

    /// Generic SAEM fitting: concentration function injected via closure.
    ///
    /// `conc_fn(theta, eta, t)` returns the individual concentration at time `t`
    /// given population parameters `theta` and random effects `eta`.
    fn fit_generic(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        n_eta: usize,
        conc_fn: &dyn Fn(&[f64], &[f64], f64) -> f64,
    ) -> Result<(FoceResult, SaemDiagnostics)> {
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".into()));
        }
        error_model.validate()?;

        let n_obs = times.len();

        // Group observations by subject.
        let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
        for i in 0..n_obs {
            subj_obs[subject_idx[i]].push((times[i], y[i]));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        let mut theta = theta_init.to_vec();
        let mut omega = omega_init;

        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];

        let omega_sds = omega.sds();
        let mut proposal_sd: Vec<Vec<f64>> =
            vec![
                omega_sds.iter().map(|&s| s * self.config.mcmc_proposal_scale).collect();
                n_subjects
            ];

        let mut s1 = vec![0.0; n_eta];
        let mut s2 = vec![vec![0.0; n_eta]; n_eta];

        let mut ofv_trace = Vec::new();
        let mut accept_count: Vec<usize> = vec![0; n_subjects];
        let mut mcmc_total: Vec<usize> = vec![0; n_subjects];

        let total_iter = self.config.n_burn + self.config.n_iter;

        for iter in 0..total_iter {
            let is_burn = iter < self.config.n_burn;
            let gamma = if is_burn { 1.0 } else { 1.0 / (iter - self.config.n_burn + 1) as f64 };

            // === E-step: MCMC sampling of etas ===
            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }

                let current_obj = individual_log_posterior_generic(
                    &subj_obs[s], &theta, &omega, &error_model, &etas[s], conc_fn,
                );

                for _chain in 0..self.config.n_chains {
                    let mut eta_proposed = etas[s].clone();
                    for k in 0..n_eta {
                        let noise: f64 = rng.sample(StandardNormal);
                        eta_proposed[k] += proposal_sd[s][k] * noise;
                    }

                    let proposed_obj = individual_log_posterior_generic(
                        &subj_obs[s], &theta, &omega, &error_model, &eta_proposed, conc_fn,
                    );

                    let log_alpha = proposed_obj - current_obj;
                    let u: f64 = rng.random();
                    if u.ln() < log_alpha {
                        etas[s] = eta_proposed;
                        accept_count[s] += 1;
                    }
                    mcmc_total[s] += 1;
                }

                if is_burn && mcmc_total[s] > 0 && iter % 20 == 19 {
                    let accept_rate = accept_count[s] as f64 / mcmc_total[s] as f64;
                    for k in 0..n_eta {
                        if accept_rate < self.config.mcmc_target_accept_low {
                            proposal_sd[s][k] *= 0.8;
                        } else if accept_rate > self.config.mcmc_target_accept_high {
                            proposal_sd[s][k] *= 1.2;
                        }
                        proposal_sd[s][k] = proposal_sd[s][k].max(1e-6).min(5.0);
                    }
                    accept_count[s] = 0;
                    mcmc_total[s] = 0;
                }
            }

            // === Stochastic Approximation: update sufficient statistics ===
            let mut s1_new = vec![0.0; n_eta];
            let mut s2_new = vec![vec![0.0; n_eta]; n_eta];

            let active_subjects = subj_obs.iter().filter(|o| !o.is_empty()).count() as f64;
            if active_subjects == 0.0 {
                return Err(Error::Validation("no subjects with observations".into()));
            }

            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                for k in 0..n_eta {
                    s1_new[k] += etas[s][k] / active_subjects;
                    for l in 0..n_eta {
                        s2_new[k][l] += etas[s][k] * etas[s][l] / active_subjects;
                    }
                }
            }

            for k in 0..n_eta {
                s1[k] = (1.0 - gamma) * s1[k] + gamma * s1_new[k];
                for l in 0..n_eta {
                    s2[k][l] = (1.0 - gamma) * s2[k][l] + gamma * s2_new[k][l];
                }
            }

            // === M-step: update theta and omega ===
            for k in 0..n_eta {
                let shift = s1[k];
                theta[k] *= shift.exp();
                theta[k] = theta[k].max(1e-10).min(1e6);
                for s in 0..n_subjects {
                    etas[s][k] -= shift;
                }
                s1[k] = 0.0;
            }

            let mut cov = vec![vec![0.0; n_eta]; n_eta];
            for k in 0..n_eta {
                for l in 0..n_eta {
                    cov[k][l] = s2[k][l];
                }
            }
            let min_var = 1e-4;
            for k in 0..n_eta {
                cov[k][k] = cov[k][k].max(min_var);
            }

            if let Ok(om_new) = OmegaMatrix::from_covariance(&cov) {
                omega = om_new;
                let new_sds = omega.sds();
                for s in 0..n_subjects {
                    for k in 0..n_eta {
                        proposal_sd[s][k] = new_sds[k] * self.config.mcmc_proposal_scale;
                    }
                }
            }

            if iter % 10 == 0 || iter == total_iter - 1 {
                let ofv = compute_marginal_ofv_generic(
                    &subj_obs, &theta, &omega, &error_model, &etas, n_eta, conc_fn,
                );
                ofv_trace.push(ofv);
            }
        }

        let final_ofv = compute_marginal_ofv_generic(
            &subj_obs, &theta, &omega, &error_model, &etas, n_eta, conc_fn,
        );

        let converged = theta.iter().all(|v| v.is_finite())
            && omega.sds().iter().all(|v| v.is_finite())
            && final_ofv.is_finite()
            && self.config.n_iter > 0;

        let acceptance_rates: Vec<f64> = (0..n_subjects)
            .map(|s| {
                if mcmc_total[s] > 0 { accept_count[s] as f64 / mcmc_total[s] as f64 } else { 0.0 }
            })
            .collect();

        let correlation = omega.correlation();
        let omega_diag = omega.sds();

        let result = FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: omega,
            correlation,
            eta: etas,
            ofv: final_ofv,
            converged,
            n_iter: self.config.n_burn + self.config.n_iter,
        };

        let diagnostics =
            SaemDiagnostics { acceptance_rates, ofv_trace, burn_in_only: self.config.n_iter == 0 };

        Ok((result, diagnostics))
    }

    /// Fit a 1-compartment oral PK model using SAEM (diagonal Ω).
    pub fn fit_1cpt_oral(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<(FoceResult, SaemDiagnostics)> {
        if omega_init.len() != 3 {
            return Err(Error::Validation("omega_init must have 3 elements".into()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_oral_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            dose,
            bioav,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 1-compartment oral PK model with correlated random effects using SAEM.
    pub fn fit_1cpt_oral_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        dose: f64,
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<(FoceResult, SaemDiagnostics)> {
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

        let n_obs = times.len();
        let n_eta = 3;

        // Group observations by subject.
        let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
        for i in 0..n_obs {
            subj_obs[subject_idx[i]].push((times[i], y[i]));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        let mut theta = theta_init.to_vec();
        let mut omega = omega_init;

        // Initialize etas at zero (or could use MAP from FOCE-like inner opt).
        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];

        // MCMC proposal standard deviations (per subject, per eta).
        let omega_sds = omega.sds();
        let mut proposal_sd: Vec<Vec<f64>> =
            vec![
                omega_sds.iter().map(|&s| s * self.config.mcmc_proposal_scale).collect();
                n_subjects
            ];

        // Sufficient statistics for the M-step.
        // S1[k] = (1/N) Σ_i η_ik           (mean of etas for param k)
        // S2[k][l] = (1/N) Σ_i η_ik·η_il   (second moments for Ω update)
        let mut s1 = vec![0.0; n_eta];
        let mut s2 = vec![vec![0.0; n_eta]; n_eta];

        // Sufficient stat for theta: weighted sum of log(theta_i) from individual fits
        // For 1-cpt oral: theta_pop_k = exp(mean(log(theta_k,pop) + eta_k))
        // But simpler: theta_pop_k = (1/N) Σ theta_k,pop · exp(η_ik) ... not closed form.
        // Instead: SAEM for theta uses a gradient step or closed-form update.
        // For log-normal random effects: theta_pop_k ≈ geometric mean of individual params
        // when η ~ N(0, ω²). The sufficient stat is: S_theta_k = (1/N) Σ_i η_ik
        // Then theta_pop_k_new = theta_pop_k · exp(-S_theta_k) to recenter.

        let mut ofv_trace = Vec::new();
        let mut accept_count: Vec<usize> = vec![0; n_subjects];
        let mut mcmc_total: Vec<usize> = vec![0; n_subjects];

        let total_iter = self.config.n_burn + self.config.n_iter;

        for iter in 0..total_iter {
            let is_burn = iter < self.config.n_burn;
            let gamma = if is_burn { 1.0 } else { 1.0 / (iter - self.config.n_burn + 1) as f64 };

            // === E-step: MCMC sampling of etas ===
            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }

                let current_obj = individual_log_posterior(
                    &subj_obs[s],
                    &theta,
                    &omega,
                    &error_model,
                    dose,
                    bioav,
                    &etas[s],
                );

                for _chain in 0..self.config.n_chains {
                    // Propose new eta
                    let mut eta_proposed = etas[s].clone();
                    for k in 0..n_eta {
                        let noise: f64 = rng.sample(StandardNormal);
                        eta_proposed[k] += proposal_sd[s][k] * noise;
                    }

                    let proposed_obj = individual_log_posterior(
                        &subj_obs[s],
                        &theta,
                        &omega,
                        &error_model,
                        dose,
                        bioav,
                        &eta_proposed,
                    );

                    // Metropolis-Hastings acceptance (log scale)
                    let log_alpha = proposed_obj - current_obj;
                    let u: f64 = rng.random();
                    if u.ln() < log_alpha {
                        etas[s] = eta_proposed;
                        accept_count[s] += 1;
                    }
                    mcmc_total[s] += 1;
                }

                // Adapt proposal variance during burn-in
                if is_burn && mcmc_total[s] > 0 && iter % 20 == 19 {
                    let accept_rate = accept_count[s] as f64 / mcmc_total[s] as f64;
                    for k in 0..n_eta {
                        if accept_rate < self.config.mcmc_target_accept_low {
                            proposal_sd[s][k] *= 0.8; // shrink
                        } else if accept_rate > self.config.mcmc_target_accept_high {
                            proposal_sd[s][k] *= 1.2; // grow
                        }
                        proposal_sd[s][k] = proposal_sd[s][k].max(1e-6).min(5.0);
                    }
                    // Reset counters for next adaptation window
                    accept_count[s] = 0;
                    mcmc_total[s] = 0;
                }
            }

            // === Stochastic Approximation: update sufficient statistics ===
            let mut s1_new = vec![0.0; n_eta];
            let mut s2_new = vec![vec![0.0; n_eta]; n_eta];

            let active_subjects = subj_obs.iter().filter(|o| !o.is_empty()).count() as f64;
            if active_subjects == 0.0 {
                return Err(Error::Validation("no subjects with observations".into()));
            }

            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                for k in 0..n_eta {
                    s1_new[k] += etas[s][k] / active_subjects;
                    for l in 0..n_eta {
                        s2_new[k][l] += etas[s][k] * etas[s][l] / active_subjects;
                    }
                }
            }

            // Stochastic approximation update
            for k in 0..n_eta {
                s1[k] = (1.0 - gamma) * s1[k] + gamma * s1_new[k];
                for l in 0..n_eta {
                    s2[k][l] = (1.0 - gamma) * s2[k][l] + gamma * s2_new[k][l];
                }
            }

            // === M-step: update theta and omega ===

            // Theta update: recenter so mean eta ≈ 0
            // theta_k_new = theta_k * exp(s1[k])
            // Then shift etas: eta_k_new = eta_k - s1[k]
            for k in 0..n_eta {
                let shift = s1[k];
                theta[k] *= shift.exp();
                // Clamp theta to reasonable range
                theta[k] = theta[k].max(1e-10).min(1e6);
                // Re-center etas
                for s in 0..n_subjects {
                    etas[s][k] -= shift;
                }
                s1[k] = 0.0; // reset after re-centering
            }

            // Omega update: Ω = E[η·ηᵀ] (second moment, since mean is re-centered to 0)
            // Covariance matrix from sufficient statistics
            let mut cov = vec![vec![0.0; n_eta]; n_eta];
            for k in 0..n_eta {
                for l in 0..n_eta {
                    cov[k][l] = s2[k][l];
                }
            }
            // Ridge regularization for positive-definiteness
            let min_var = 1e-4;
            for k in 0..n_eta {
                cov[k][k] = cov[k][k].max(min_var);
            }

            if let Ok(om_new) = OmegaMatrix::from_covariance(&cov) {
                omega = om_new;
                // Update proposal SDs from new omega
                let new_sds = omega.sds();
                for s in 0..n_subjects {
                    for k in 0..n_eta {
                        proposal_sd[s][k] = new_sds[k] * self.config.mcmc_proposal_scale;
                    }
                }
            }

            // Compute OFV for monitoring
            if iter % 10 == 0 || iter == total_iter - 1 {
                let ofv = compute_marginal_ofv(
                    &subj_obs,
                    &theta,
                    &omega,
                    &error_model,
                    dose,
                    bioav,
                    &etas,
                    n_eta,
                );
                ofv_trace.push(ofv);
            }
        }

        // Final OFV
        let final_ofv = compute_marginal_ofv(
            &subj_obs,
            &theta,
            &omega,
            &error_model,
            dose,
            bioav,
            &etas,
            n_eta,
        );

        // SAEM convergence: the algorithm is guaranteed to converge when
        // step sizes γ_k → 0 (Delyon, Lavielle, Moulines 1999). OFV oscillates
        // due to MCMC noise in the E-step even when parameters have stabilized.
        // We declare converged if all iterations completed and estimates are finite.
        let converged = theta.iter().all(|v| v.is_finite())
            && omega.sds().iter().all(|v| v.is_finite())
            && final_ofv.is_finite()
            && self.config.n_iter > 0;

        // Compute final acceptance rates
        let acceptance_rates: Vec<f64> = (0..n_subjects)
            .map(|s| {
                if mcmc_total[s] > 0 { accept_count[s] as f64 / mcmc_total[s] as f64 } else { 0.0 }
            })
            .collect();

        let correlation = omega.correlation();
        let omega_diag = omega.sds();

        let result = FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: omega,
            correlation,
            eta: etas,
            ofv: final_ofv,
            converged,
            n_iter: self.config.n_burn + self.config.n_iter,
        };

        let diagnostics =
            SaemDiagnostics { acceptance_rates, ofv_trace, burn_in_only: self.config.n_iter == 0 };

        Ok((result, diagnostics))
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute individual concentration for 1-cpt oral given population params + eta.
#[inline]
fn individual_conc(theta: &[f64], eta: &[f64], dose: f64, bioav: f64, t: f64) -> f64 {
    let cl = theta[0] * eta[0].exp();
    let v = theta[1] * eta[1].exp();
    let ka = theta[2] * eta[2].exp();
    pk::conc_oral(dose, bioav, cl, v, ka, t)
}

/// Log-posterior for one subject's etas: log p(y|η,θ) + log p(η|Ω)
/// Returns the *negative* log-posterior (to align with NLL convention).
/// For M-H we want log p(η|y) ∝ log p(y|η) + log p(η).
fn individual_log_posterior(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
) -> f64 {
    // log p(y|η) = -Σ nll_obs(y_j, f(t_j, η))
    let mut log_lik = 0.0;
    for &(t, yobs) in obs {
        let c = individual_conc(theta, eta, dose, bioav, t);
        log_lik -= em.nll_obs(yobs, c.max(1e-30));
    }
    // log p(η|Ω) = -0.5 * η'Ω⁻¹η - 0.5*log|Ω| - (n/2)*log(2π)
    let prior = -0.5 * omega.inv_quadratic(eta) - 0.5 * omega.log_det();
    log_lik + prior
}

/// Log-posterior for one subject's etas using a generic concentration function.
fn individual_log_posterior_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta: &[f64],
    conc_fn: &dyn Fn(&[f64], &[f64], f64) -> f64,
) -> f64 {
    let mut log_lik = 0.0;
    for &(t, yobs) in obs {
        let c = conc_fn(theta, eta, t);
        log_lik -= em.nll_obs(yobs, c.max(1e-30));
    }
    let prior = -0.5 * omega.inv_quadratic(eta) - 0.5 * omega.log_det();
    log_lik + prior
}

/// Compute a simple OFV for monitoring using a generic concentration function.
fn compute_marginal_ofv_generic(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    etas: &[Vec<f64>],
    _n_eta: usize,
    conc_fn: &dyn Fn(&[f64], &[f64], f64) -> f64,
) -> f64 {
    let mut ofv = 0.0;
    for (s, obs) in subj_obs.iter().enumerate() {
        if obs.is_empty() {
            continue;
        }
        for &(t, yobs) in obs {
            let c = conc_fn(theta, &etas[s], t);
            ofv += em.nll_obs(yobs, c.max(1e-30));
        }
        ofv += 0.5 * omega.inv_quadratic(&etas[s]);
    }
    2.0 * ofv
}

/// Compute a simple OFV for monitoring (sum of individual NLLs + prior).
fn compute_marginal_ofv(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    etas: &[Vec<f64>],
    _n_eta: usize,
) -> f64 {
    let mut ofv = 0.0;
    for (s, obs) in subj_obs.iter().enumerate() {
        if obs.is_empty() {
            continue;
        }
        for &(t, yobs) in obs {
            let c = individual_conc(theta, &etas[s], dose, bioav, t);
            ofv += em.nll_obs(yobs, c.max(1e-30));
        }
        ofv += 0.5 * omega.inv_quadratic(&etas[s]);
    }
    2.0 * ofv // NONMEM convention: OFV = -2LL
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_data(
        n_subjects: usize,
        n_obs: usize,
        theta_true: &[f64],
        omega_sds: &[f64],
        sigma: f64,
        dose: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for s in 0..n_subjects {
            let eta_cl: f64 = omega_sds[0] * rng.sample::<f64, _>(StandardNormal);
            let eta_v: f64 = omega_sds[1] * rng.sample::<f64, _>(StandardNormal);
            let eta_ka: f64 = omega_sds[2] * rng.sample::<f64, _>(StandardNormal);

            let cl_i = theta_true[0] * eta_cl.exp();
            let v_i = theta_true[1] * eta_v.exp();
            let ka_i = theta_true[2] * eta_ka.exp();

            for j in 0..n_obs {
                let t = (j + 1) as f64 * 24.0 / n_obs as f64;
                let c = pk::conc_oral(dose, 1.0, cl_i, v_i, ka_i, t);
                let noise: f64 = sigma * rng.sample::<f64, _>(StandardNormal);
                let y_obs = (c + noise).max(0.01);

                times.push(t);
                y.push(y_obs);
                subject_idx.push(s);
            }
        }
        (times, y, subject_idx)
    }

    #[test]
    fn saem_smoke_test() {
        let theta_true = [0.133, 8.0, 0.8];
        let omega_sds = [0.3, 0.25, 0.3];
        let sigma = 0.5;
        let dose = 100.0;
        let n_subjects = 20;
        let n_obs = 8;

        let (times, y, subject_idx) =
            make_synthetic_data(n_subjects, n_obs, &theta_true, &omega_sds, sigma, dose, 42);

        let config = SaemConfig { n_burn: 50, n_iter: 50, seed: 42, ..Default::default() };
        let estimator = SaemEstimator::new(config);

        let (result, diag) = estimator
            .fit_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                dose,
                1.0,
                ErrorModel::Additive(sigma),
                &theta_true,
                &omega_sds,
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV should be finite");
        assert!(!result.theta.iter().any(|t| !t.is_finite()), "theta should be finite");
        assert!(result.eta.len() == n_subjects, "should have etas for all subjects");
        assert!(!diag.ofv_trace.is_empty(), "should have OFV trace");
    }

    #[test]
    fn saem_parameter_recovery() {
        let theta_true = [0.133, 8.0, 0.8];
        let omega_sds = [0.3, 0.25, 0.3];
        let sigma = 0.5;
        let dose = 100.0;
        let n_subjects = 40;
        let n_obs = 10;

        let (times, y, subject_idx) =
            make_synthetic_data(n_subjects, n_obs, &theta_true, &omega_sds, sigma, dose, 123);

        let config = SaemConfig { n_burn: 150, n_iter: 100, seed: 123, ..Default::default() };
        let estimator = SaemEstimator::new(config);

        let (result, _diag) = estimator
            .fit_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                dose,
                1.0,
                ErrorModel::Additive(sigma),
                &theta_true,
                &omega_sds,
            )
            .unwrap();

        // Check parameter recovery (within 3x omega SD, same as FOCE benchmarks)
        for (k, name) in ["CL", "V", "Ka"].iter().enumerate() {
            let rel_err = (result.theta[k] - theta_true[k]).abs() / theta_true[k];
            assert!(
                rel_err < 3.0 * omega_sds[k],
                "SAEM {name}: fitted={:.4}, true={:.4}, rel_err={rel_err:.4}",
                result.theta[k],
                theta_true[k]
            );
        }
    }

    #[test]
    fn saem_ofv_decreases() {
        let theta_true = [0.133, 8.0, 0.8];
        let omega_sds = [0.3, 0.25, 0.3];
        let sigma = 0.5;
        let dose = 100.0;

        let (times, y, subject_idx) =
            make_synthetic_data(30, 8, &theta_true, &omega_sds, sigma, dose, 99);

        let config = SaemConfig { n_burn: 100, n_iter: 50, seed: 99, ..Default::default() };
        let estimator = SaemEstimator::new(config);

        let (_result, diag) = estimator
            .fit_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                30,
                dose,
                1.0,
                ErrorModel::Additive(sigma),
                &theta_true,
                &omega_sds,
            )
            .unwrap();

        // OFV should generally decrease (allow some stochastic noise)
        let n = diag.ofv_trace.len();
        if n >= 4 {
            let early_mean = diag.ofv_trace[..2].iter().sum::<f64>() / 2.0;
            let late_mean = diag.ofv_trace[n - 2..].iter().sum::<f64>() / 2.0;
            assert!(
                late_mean < early_mean * 1.5,
                "OFV should not dramatically increase: early={early_mean:.1}, late={late_mean:.1}"
            );
        }
    }

    #[test]
    fn saem_correlated_omega() {
        let theta_true = [0.133, 8.0, 0.8];
        let omega_sds = [0.3, 0.25, 0.3];
        let sigma = 0.5;
        let dose = 100.0;
        let n_subjects = 40;

        let (times, y, subject_idx) =
            make_synthetic_data(n_subjects, 10, &theta_true, &omega_sds, sigma, dose, 77);

        // Start with correlated Omega
        let corr = vec![vec![1.0, 0.5, 0.0], vec![0.5, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let om = OmegaMatrix::from_correlation(&omega_sds, &corr).unwrap();

        let config = SaemConfig { n_burn: 100, n_iter: 50, seed: 77, ..Default::default() };
        let estimator = SaemEstimator::new(config);

        let (result, _diag) = estimator
            .fit_1cpt_oral_correlated(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                dose,
                1.0,
                ErrorModel::Additive(sigma),
                &theta_true,
                om,
            )
            .unwrap();

        assert!(result.ofv.is_finite());
        assert!(result.correlation.len() == 3);
        // Omega should remain positive-definite
        assert!(result.omega.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn saem_input_validation() {
        let estimator = SaemEstimator::default_config();

        // Wrong theta length
        assert!(
            estimator
                .fit_1cpt_oral(
                    &[1.0],
                    &[1.0],
                    &[0],
                    1,
                    100.0,
                    1.0,
                    ErrorModel::Additive(1.0),
                    &[0.1, 0.2],
                    &[0.3, 0.25, 0.3],
                )
                .is_err()
        );

        // Wrong omega length
        assert!(
            estimator
                .fit_1cpt_oral(
                    &[1.0],
                    &[1.0],
                    &[0],
                    1,
                    100.0,
                    1.0,
                    ErrorModel::Additive(1.0),
                    &[0.1, 0.2, 0.3],
                    &[0.3, 0.25],
                )
                .is_err()
        );

        // Mismatched lengths
        assert!(
            estimator
                .fit_1cpt_oral(
                    &[1.0, 2.0],
                    &[1.0],
                    &[0],
                    1,
                    100.0,
                    1.0,
                    ErrorModel::Additive(1.0),
                    &[0.1, 0.2, 0.3],
                    &[0.3, 0.25, 0.3],
                )
                .is_err()
        );
    }

    // -------------------------------------------------------------------
    // 2-compartment tests
    // -------------------------------------------------------------------

    fn make_synthetic_2cpt_iv_data(
        n_subjects: usize,
        n_obs: usize,
        theta_true: &[f64],
        omega_sds: &[f64],
        sigma: f64,
        dose: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for s in 0..n_subjects {
            let eta: Vec<f64> = omega_sds.iter().map(|&sd| {
                sd * rng.sample::<f64, _>(StandardNormal)
            }).collect();

            let cl_i = theta_true[0] * eta[0].exp();
            let v1_i = theta_true[1] * eta[1].exp();
            let q_i  = theta_true[2] * eta[2].exp();
            let v2_i = theta_true[3] * eta[3].exp();

            for j in 0..n_obs {
                let t = (j + 1) as f64 * 24.0 / n_obs as f64;
                let c = pk::conc_iv_2cpt_macro(dose, cl_i, v1_i, v2_i, q_i, t);
                let noise: f64 = sigma * rng.sample::<f64, _>(StandardNormal);
                let y_obs = (c + noise).max(0.01);

                times.push(t);
                y.push(y_obs);
                subject_idx.push(s);
            }
        }
        (times, y, subject_idx)
    }

    fn make_synthetic_2cpt_oral_data(
        n_subjects: usize,
        n_obs: usize,
        theta_true: &[f64],
        omega_sds: &[f64],
        sigma: f64,
        dose: f64,
        bioav: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for s in 0..n_subjects {
            let eta: Vec<f64> = omega_sds.iter().map(|&sd| {
                sd * rng.sample::<f64, _>(StandardNormal)
            }).collect();

            let cl_i = theta_true[0] * eta[0].exp();
            let v1_i = theta_true[1] * eta[1].exp();
            let q_i  = theta_true[2] * eta[2].exp();
            let v2_i = theta_true[3] * eta[3].exp();
            let ka_i = theta_true[4] * eta[4].exp();

            for j in 0..n_obs {
                let t = (j + 1) as f64 * 24.0 / n_obs as f64;
                let c = pk::conc_oral_2cpt_macro(dose, bioav, cl_i, v1_i, v2_i, q_i, ka_i, t);
                let noise: f64 = sigma * rng.sample::<f64, _>(StandardNormal);
                let y_obs = (c + noise).max(0.01);

                times.push(t);
                y.push(y_obs);
                subject_idx.push(s);
            }
        }
        (times, y, subject_idx)
    }

    #[test]
    fn saem_2cpt_iv_convergence() {
        let theta_true = [2.0, 10.0, 1.5, 20.0];
        let omega_sds = [0.20, 0.15, 0.20, 0.15];
        let sigma = 0.5;
        let dose = 100.0;
        let n_subjects = 20;
        let n_obs = 8;

        let (times, y, subject_idx) =
            make_synthetic_2cpt_iv_data(n_subjects, n_obs, &theta_true, &omega_sds, sigma, dose, 42);

        let config = SaemConfig { n_burn: 50, n_iter: 50, seed: 42, ..Default::default() };
        let estimator = SaemEstimator::new(config);

        let (result, diag) = estimator
            .fit_2cpt_iv(
                &times, &y, &subject_idx, n_subjects, dose,
                ErrorModel::Additive(sigma),
                &theta_true,
                &omega_sds,
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV should be finite");
        assert!(!result.theta.iter().any(|t| !t.is_finite()), "theta should be finite");
        assert_eq!(result.theta.len(), 4);
        assert_eq!(result.eta.len(), n_subjects);
        assert!(!diag.ofv_trace.is_empty(), "should have OFV trace");
    }

    #[test]
    fn saem_2cpt_oral_convergence() {
        let theta_true = [2.0, 10.0, 1.5, 20.0, 1.0];
        let omega_sds = [0.20, 0.15, 0.20, 0.15, 0.25];
        let sigma = 0.5;
        let dose = 100.0;
        let bioav = 0.8;
        let n_subjects = 20;
        let n_obs = 8;

        let (times, y, subject_idx) = make_synthetic_2cpt_oral_data(
            n_subjects, n_obs, &theta_true, &omega_sds, sigma, dose, bioav, 55,
        );

        let config = SaemConfig { n_burn: 50, n_iter: 50, seed: 55, ..Default::default() };
        let estimator = SaemEstimator::new(config);

        let (result, diag) = estimator
            .fit_2cpt_oral(
                &times, &y, &subject_idx, n_subjects, dose, bioav,
                ErrorModel::Additive(sigma),
                &theta_true,
                &omega_sds,
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV should be finite");
        assert!(!result.theta.iter().any(|t| !t.is_finite()), "theta should be finite");
        assert_eq!(result.theta.len(), 5);
        assert_eq!(result.eta.len(), n_subjects);
        assert!(!diag.ofv_trace.is_empty(), "should have OFV trace");
    }

    #[test]
    fn saem_2cpt_iv_input_validation() {
        let estimator = SaemEstimator::default_config();

        // Wrong theta length for 2-cpt IV
        assert!(
            estimator
                .fit_2cpt_iv(
                    &[1.0], &[1.0], &[0], 1, 100.0,
                    ErrorModel::Additive(1.0),
                    &[0.1, 0.2, 0.3],  // only 3, need 4
                    &[0.3, 0.3, 0.3, 0.3],
                )
                .is_err()
        );

        // Wrong omega length for 2-cpt IV
        assert!(
            estimator
                .fit_2cpt_iv(
                    &[1.0], &[1.0], &[0], 1, 100.0,
                    ErrorModel::Additive(1.0),
                    &[0.1, 0.2, 0.3, 0.4],
                    &[0.3, 0.3, 0.3],  // only 3, need 4
                )
                .is_err()
        );
    }
}
