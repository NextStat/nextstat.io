//! FOCE / FOCEI estimation for population PK models.
//!
//! First-Order Conditional Estimation (FOCE) approximates the marginal
//! likelihood by:
//! 1. Finding the conditional mode of random effects (ETAs) per subject.
//! 2. Using a Laplace approximation to integrate out the random effects.
//! 3. Optimizing population parameters (θ, Ω) to minimize the objective
//!    function value (OFV = −2·log L).
//!
//! FOCEI additionally accounts for the interaction between random effects
//! and the residual error model (variance depends on individual predictions).
//!
//! ## Correlated Random Effects
//!
//! [`OmegaMatrix`] stores the full Ω variance–covariance matrix via its
//! Cholesky factor **L** (lower triangular), so Ω = L·Lᵀ is always
//! positive-definite. This allows modelling correlations between random
//! effects (e.g. CL–V correlation).

use nalgebra::DMatrix;
use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::dosing::DosingRegimen;
use crate::pk::{self, ErrorModel};

// ---------------------------------------------------------------------------
// OmegaMatrix: Cholesky-parameterized variance–covariance matrix
// ---------------------------------------------------------------------------

/// Variance–covariance matrix of random effects, stored as its Cholesky
/// factor **L** so that Ω = L·Lᵀ is always positive-definite.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(into = "OmegaMatrixDto", try_from = "OmegaMatrixDto")]
pub struct OmegaMatrix {
    /// Lower-triangular Cholesky factor (row-major, `n × n`).
    chol: Vec<Vec<f64>>,
    /// Dimension (number of random effects).
    n: usize,
}

impl OmegaMatrix {
    /// Build from a diagonal of standard deviations (independent random effects).
    pub fn from_diagonal(sds: &[f64]) -> Result<Self> {
        for (i, &s) in sds.iter().enumerate() {
            if s <= 0.0 || !s.is_finite() {
                return Err(Error::Validation(format!("omega SD[{i}] must be > 0, got {s}")));
            }
        }
        let n = sds.len();
        let mut chol = vec![vec![0.0; n]; n];
        for i in 0..n {
            chol[i][i] = sds[i];
        }
        Ok(Self { chol, n })
    }

    /// Build from a lower-triangular Cholesky factor directly.
    pub fn from_cholesky(l: Vec<Vec<f64>>) -> Result<Self> {
        let n = l.len();
        if n == 0 {
            return Err(Error::Validation("empty Cholesky factor".to_string()));
        }
        for i in 0..n {
            if l[i].len() != n {
                return Err(Error::Validation(format!("row {i} length {} != {n}", l[i].len())));
            }
            if l[i][i] <= 0.0 {
                return Err(Error::Validation(format!("L[{i}][{i}] must be > 0")));
            }
            for j in (i + 1)..n {
                if l[i][j].abs() > 1e-15 {
                    return Err(Error::Validation(format!(
                        "L must be lower-triangular, L[{i}][{j}] = {}",
                        l[i][j]
                    )));
                }
            }
        }
        Ok(Self { chol: l, n })
    }

    /// Build from standard deviations + correlation matrix.
    ///
    /// `corr` is an `n × n` symmetric matrix with 1s on the diagonal.
    pub fn from_correlation(sds: &[f64], corr: &[Vec<f64>]) -> Result<Self> {
        let n = sds.len();
        if corr.len() != n {
            return Err(Error::Validation("corr rows != sds length".to_string()));
        }
        // Build covariance matrix: Cov[i][j] = sds[i] * corr[i][j] * sds[j]
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = sds[i] * corr[i][j] * sds[j];
            }
        }
        Self::from_covariance(&cov)
    }

    /// Build from a full covariance matrix (Cholesky decomposition).
    pub fn from_covariance(cov: &[Vec<f64>]) -> Result<Self> {
        let n = cov.len();
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                if i == j {
                    let diag = cov[i][i] - sum;
                    if diag <= 0.0 {
                        return Err(Error::Validation(format!(
                            "covariance not positive-definite at [{i}][{i}]"
                        )));
                    }
                    l[i][j] = diag.sqrt();
                } else {
                    l[i][j] = (cov[i][j] - sum) / l[j][j];
                }
            }
        }
        Ok(Self { chol: l, n })
    }

    /// Compute from empirical sample covariance of eta vectors.
    ///
    /// Ω = (1/N) ∑_i η_i · η_iᵀ, with ridge regularization to guarantee PD.
    pub fn empirical(etas: &[Vec<f64>], n_eta: usize) -> Result<Self> {
        let n = etas.len();
        if n == 0 {
            return Err(Error::Validation("no eta samples".to_string()));
        }
        let mut cov = vec![vec![0.0; n_eta]; n_eta];
        for eta in etas {
            for i in 0..n_eta {
                for j in 0..=i {
                    cov[i][j] += eta[i] * eta[j];
                }
            }
        }
        let nf = n as f64;
        for i in 0..n_eta {
            for j in 0..=i {
                cov[i][j] /= nf;
                cov[j][i] = cov[i][j];
            }
        }
        // Ridge regularization: add min_var to diagonal to guarantee PD.
        let min_var = 1e-4;
        for i in 0..n_eta {
            cov[i][i] += min_var;
        }
        // Shrink off-diagonals: cap |cov[i][j]| < sqrt(cov[i][i]*cov[j][j])
        // to ensure the correlation stays in (-1, 1).
        for i in 0..n_eta {
            for j in 0..i {
                let max_abs = (cov[i][i] * cov[j][j]).sqrt() * 0.999;
                cov[i][j] = cov[i][j].clamp(-max_abs, max_abs);
                cov[j][i] = cov[i][j];
            }
        }
        Self::from_covariance(&cov)
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Cholesky factor L (lower triangular).
    pub fn cholesky(&self) -> &[Vec<f64>] {
        &self.chol
    }

    /// Reconstruct full Ω = L·Lᵀ.
    pub fn to_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for k in 0..=j {
                    s += self.chol[i][k] * self.chol[j][k];
                }
                m[i][j] = s;
                m[j][i] = s;
            }
        }
        m
    }

    /// Extract standard deviations (sqrt of diagonal of Ω).
    pub fn sds(&self) -> Vec<f64> {
        let m = self.to_matrix();
        (0..self.n).map(|i| m[i][i].sqrt()).collect()
    }

    /// Extract correlation matrix.
    pub fn correlation(&self) -> Vec<Vec<f64>> {
        let m = self.to_matrix();
        let sds = self.sds();
        let n = self.n;
        let mut corr = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                corr[i][j] = m[i][j] / (sds[i] * sds[j]);
            }
        }
        corr
    }

    /// log |det(Ω)| = 2 · ∑ ln(L_ii).
    pub fn log_det(&self) -> f64 {
        2.0 * (0..self.n).map(|i| self.chol[i][i].ln()).sum::<f64>()
    }

    /// Compute ηᵀ Ω⁻¹ η efficiently via forward-substitution (L z = η, then |z|²).
    pub fn inv_quadratic(&self, eta: &[f64]) -> f64 {
        let n = self.n;
        let mut z = vec![0.0; n];
        for i in 0..n {
            let mut s = eta[i];
            for j in 0..i {
                s -= self.chol[i][j] * z[j];
            }
            z[i] = s / self.chol[i][i];
        }
        z.iter().map(|v| v * v).sum()
    }
}

/// DTO for serde: serializes OmegaMatrix as its covariance matrix.
#[derive(Serialize, Deserialize)]
struct OmegaMatrixDto {
    covariance: Vec<Vec<f64>>,
}

impl From<OmegaMatrix> for OmegaMatrixDto {
    fn from(om: OmegaMatrix) -> Self {
        Self { covariance: om.to_matrix() }
    }
}

impl TryFrom<OmegaMatrixDto> for OmegaMatrix {
    type Error = String;
    fn try_from(dto: OmegaMatrixDto) -> std::result::Result<Self, Self::Error> {
        OmegaMatrix::from_covariance(&dto.covariance).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// FOCE configuration and result
// ---------------------------------------------------------------------------

/// Configuration for FOCE/FOCEI estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoceConfig {
    /// Maximum outer iterations (population parameter optimization).
    pub max_outer_iter: usize,
    /// Maximum inner iterations per subject (ETA optimization via Newton).
    pub max_inner_iter: usize,
    /// Convergence tolerance on absolute OFV change.
    pub tol: f64,
    /// Convergence tolerance on relative OFV change (|ΔOFV|/|OFV|).
    /// Convergence is declared when either absolute or relative criterion is met.
    pub rel_tol: f64,
    /// If true, use FOCEI (interaction); if false, use FOCE.
    pub interaction: bool,
    /// Damping factor for omega update: Ω_new = α·Ω_prev + (1−α)·Ω_empirical.
    /// 0 = no damping (legacy), 1 = no update. Default: 0.7.
    #[serde(default = "default_omega_damping")]
    pub omega_damping: f64,
    /// Cap on omega diagonal relative to initial variance: Ω_ii ≤ ratio × Ω_init_ii.
    /// Prevents runaway omega inflation. Default: 100.0.
    #[serde(default = "default_omega_max_ratio")]
    pub omega_max_ratio: f64,
    /// If true, estimate sigma via closed-form MLE at each outer iteration.
    /// Default: true (NONMEM-compatible).
    #[serde(default = "default_estimate_sigma")]
    pub estimate_sigma: bool,
    /// Optional LLOQ for BLQ/M3 handling in population likelihood.
    /// If set, observations with `DV < LLOQ` are treated as left-censored.
    #[serde(default)]
    pub lloq: Option<f64>,
    /// Mask for fixed omega elements.  When `omega_fixed[i]` is true, the
    /// random effect η_i is held at 0 and Ω_{i,i} is not updated in the
    /// M-step.  Equivalent to NONMEM `$OMEGA … FIX`.
    /// Length must equal n_eta when non-empty.
    #[serde(default)]
    pub omega_fixed: Vec<bool>,
    /// If true, force omega to be diagonal (zero off-diagonal elements after
    /// each M-step).  Equivalent to NONMEM `$OMEGA` without `BLOCK`.
    /// Default: false (full covariance estimated).
    #[serde(default)]
    pub diagonal_omega: bool,
}

fn default_omega_damping() -> f64 {
    0.1
}
fn default_omega_max_ratio() -> f64 {
    100.0
}
fn default_estimate_sigma() -> bool {
    true
}

impl Default for FoceConfig {
    fn default() -> Self {
        Self {
            max_outer_iter: 100,
            max_inner_iter: 20,
            tol: 1e-4,
            rel_tol: 1e-8,
            interaction: true,
            omega_damping: 0.1,
            omega_max_ratio: 100.0,
            estimate_sigma: true,
            lloq: None,
            omega_fixed: Vec::new(),
            diagonal_omega: false,
        }
    }
}

/// Configuration for ITS (Iterative Two-Stage) approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItsConfig {
    /// Maximum outer ITS iterations.
    pub max_iter: usize,
    /// Maximum iterations for individual-level optimization.
    pub max_individual_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Damping used when blending omega updates.
    pub omega_damping: f64,
}

impl Default for ItsConfig {
    fn default() -> Self {
        Self { max_iter: 30, max_individual_iter: 60, tol: 1e-4, omega_damping: 0.3 }
    }
}

/// Configuration for IMP (importance-sampling MC-EM) approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpConfig {
    /// Maximum IMP iterations.
    pub n_iter: usize,
    /// Number of importance samples per subject.
    pub n_samples: usize,
    /// Proposal scaling factor.
    pub proposal_scale: f64,
    /// RNG seed.
    pub seed: u64,
    /// Convergence tolerance.
    pub tol: f64,
    /// If true, run E-only updates.
    pub e_only: bool,
}

impl Default for ImpConfig {
    fn default() -> Self {
        Self { n_iter: 15, n_samples: 300, proposal_scale: 1.0, seed: 42, tol: 1e-4, e_only: false }
    }
}

/// Diagnostics returned by IMP routines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpDiagnostics {
    pub ofv_trace: Vec<f64>,
    pub ess_fraction_trace: Vec<f64>,
    pub max_weight_trace: Vec<f64>,
}

/// FOCE covariance-step output (R/S and derived uncertainties).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovarianceStepResult {
    /// Parameter names corresponding to rows/cols of matrices.
    pub parameter_names: Vec<String>,
    /// Observed information matrix R.
    pub r_matrix: Vec<Vec<f64>>,
    /// Outer-product gradient matrix S.
    pub s_matrix: Vec<Vec<f64>>,
    /// Covariance estimate R^{-1}.
    pub covariance: Vec<Vec<f64>>,
    /// Sandwich robust covariance R^{-1} S R^{-1}.
    pub robust_covariance: Vec<Vec<f64>>,
    /// Standard errors (natural scale).
    pub se: Vec<f64>,
    /// Relative standard errors in percent.
    pub rse_pct: Vec<f64>,
    /// Eigenvalues of symmetrized R.
    pub r_eigenvalues: Vec<f64>,
    /// Condition number of R (max/min positive eigenvalue).
    pub r_condition_number: f64,
}

/// Result of FOCE/FOCEI estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoceResult {
    /// Population fixed-effect parameters (e.g., CL_pop, V_pop, Ka_pop).
    pub theta: Vec<f64>,
    /// Diagonal of the Ω matrix (random effect standard deviations).
    /// For backward compatibility; use `omega_matrix` for full covariance.
    pub omega: Vec<f64>,
    /// Full Ω variance–covariance matrix (Cholesky-parameterized).
    pub omega_matrix: OmegaMatrix,
    /// Correlation matrix extracted from Ω.
    pub correlation: Vec<Vec<f64>>,
    /// Conditional modes of random effects per subject: `eta[subject][param]`.
    pub eta: Vec<Vec<f64>>,
    /// Objective function value (−2·log L).
    pub ofv: f64,
    /// Whether the estimation converged.
    pub converged: bool,
    /// Number of outer iterations performed.
    pub n_iter: usize,
    /// Final estimated sigma (or initial value if `estimate_sigma` is false).
    pub sigma: f64,
    /// Initial sigma value supplied by the user.
    pub sigma_init: f64,
    /// Covariance-step output (R/S matrices and SE/RSE).
    pub covariance_step: Option<CovarianceStepResult>,
}

/// Random-effect transform for subject-level parameters.
///
/// - `LogNormal`: `θ_i = θ_pop * exp(η)`
/// - `LogitNormal`: `θ_i = lower + (upper-lower) * sigmoid(logit(z0) + η)`,
///   where `z0 = (θ_pop-lower)/(upper-lower)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RandomEffectTransform {
    LogNormal,
    LogitNormal { lower: f64, upper: f64 },
}

#[inline]
fn sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline]
pub(crate) fn validate_random_effect_transform(
    theta_pop: f64,
    tr: RandomEffectTransform,
) -> Result<()> {
    if !theta_pop.is_finite() || theta_pop <= 0.0 {
        return Err(Error::Validation("theta_pop must be finite and > 0".to_string()));
    }
    match tr {
        RandomEffectTransform::LogNormal => Ok(()),
        RandomEffectTransform::LogitNormal { lower, upper } => {
            if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                return Err(Error::Validation(
                    "LogitNormal requires finite bounds with lower < upper".to_string(),
                ));
            }
            if theta_pop <= lower || theta_pop >= upper {
                return Err(Error::Validation(format!(
                    "theta_pop={theta_pop} must satisfy lower < theta_pop < upper for LogitNormal"
                )));
            }
            Ok(())
        }
    }
}

#[inline]
pub(crate) fn apply_random_effect_transform(
    theta_pop: f64,
    eta: f64,
    tr: RandomEffectTransform,
) -> f64 {
    match tr {
        RandomEffectTransform::LogNormal => theta_pop * eta.exp(),
        RandomEffectTransform::LogitNormal { lower, upper } => {
            let span = upper - lower;
            let z0 = ((theta_pop - lower) / span).clamp(1e-12, 1.0 - 1e-12);
            let logit0 = (z0 / (1.0 - z0)).ln();
            lower + span * sigmoid_stable(logit0 + eta)
        }
    }
}

/// FOCE/FOCEI estimator for population pharmacokinetic models.
pub struct FoceEstimator {
    config: FoceConfig,
}

// ---------------------------------------------------------------------------
// NONMEM-style nested FOCE objective function
// ---------------------------------------------------------------------------

/// Laplace OFV for all population parameters: (θ, ω_diag, σ) in log-space.
///
/// Etas are **re-optimised** inside each eval, so the gradient through the
/// inner optimisation is captured implicitly.  This is the standard NONMEM
/// nested approach: outer Newton minimises Laplace OFV, inner Newton finds η̂_i.
struct FoceNestedOfv<'a> {
    subj_obs: &'a [Vec<(f64, f64)>],
    doses: &'a [f64],
    bioav: f64,
    n_theta: usize,
    n_eta: usize,
    /// 0 = additive, 1 = proportional, 2 = combined, 3 = exponential, 4 = power.
    em_variant: u8,
    estimate_sigma: bool,
    fixed_sigma: Vec<f64>,
    max_inner_iter: usize,
    lloq: Option<f64>,
    /// Warm-started etas from the last eval (interior mutability).
    cached_etas: std::cell::RefCell<Vec<Vec<f64>>>,
}

impl<'a> FoceNestedOfv<'a> {
    fn unpack(&self, params: &[f64]) -> Result<(Vec<f64>, OmegaMatrix, ErrorModel)> {
        let nt = self.n_theta;
        let ne = self.n_eta;
        let theta: Vec<f64> = params[..nt].iter().map(|&p| p.exp()).collect();
        let omega_sds: Vec<f64> = params[nt..nt + ne].iter().map(|&p| p.exp()).collect();
        let omega = OmegaMatrix::from_diagonal(&omega_sds)?;
        let sigma_offset = nt + ne;
        let em = if self.estimate_sigma {
            match self.em_variant {
                0 => ErrorModel::Additive(params[sigma_offset].exp()),
                2 => ErrorModel::Combined {
                    sigma_add: params[sigma_offset].exp(),
                    sigma_prop: params[sigma_offset + 1].exp(),
                },
                3 => ErrorModel::Exponential(params[sigma_offset].exp()),
                4 => ErrorModel::Power {
                    sigma: params[sigma_offset].exp(),
                    power: params[sigma_offset + 1],
                },
                _ => ErrorModel::Proportional(params[sigma_offset].exp()),
            }
        } else {
            match self.em_variant {
                0 => ErrorModel::Additive(self.fixed_sigma[0]),
                2 => ErrorModel::Combined {
                    sigma_add: self.fixed_sigma[0],
                    sigma_prop: self.fixed_sigma[1],
                },
                3 => ErrorModel::Exponential(self.fixed_sigma[0]),
                4 => ErrorModel::Power { sigma: self.fixed_sigma[0], power: self.fixed_sigma[1] },
                _ => ErrorModel::Proportional(self.fixed_sigma[0]),
            }
        };
        Ok((theta, omega, em))
    }

    /// Evaluate Laplace OFV with re-optimised etas.
    fn laplace_ofv(&self, params: &[f64]) -> Result<f64> {
        if params.iter().any(|v| !v.is_finite() || v.abs() > 20.0) {
            return Ok(f64::MAX);
        }
        let (theta, omega, em) = match self.unpack(params) {
            Ok(v) => v,
            Err(_) => return Ok(f64::MAX),
        };
        let n_eta = self.n_eta;
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_omega = omega.log_det();
        let mut ofv = 0.0;

        let mut etas = self.cached_etas.borrow_mut();
        for (s, obs) in self.subj_obs.iter().enumerate() {
            if obs.is_empty() {
                continue;
            }
            // Re-optimise etas at current population params (warm-started).
            etas[s] = inner_optimize_eta(
                obs,
                &theta,
                &omega,
                &em,
                self.doses[s],
                self.bioav,
                &etas[s],
                self.max_inner_iter,
                self.lloq,
            )
            .unwrap_or_else(|_| etas[s].clone());

            let eta = &etas[s];
            let nll_s = inner_objective(
                obs,
                &theta,
                &omega,
                &em,
                self.doses[s],
                self.bioav,
                eta,
                self.lloq,
            );
            let hess =
                inner_hessian(obs, &theta, &omega, &em, self.doses[s], self.bioav, eta, self.lloq);
            let ld = log_det(&hess);
            let n_obs_i = obs.len() as f64;
            ofv += 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega + n_obs_i * log_2pi;
        }
        if !ofv.is_finite() {
            return Ok(f64::MAX);
        }
        Ok(ofv)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        // Gradient via central differences with eps=1e-4.
        // Inner etas are re-optimised at each eval point, so the gradient
        // implicitly captures ∂η̂/∂θ (implicit function theorem, numerically).
        //
        // IMPORTANT: save and restore cached_etas so that perturbations for
        // different param indices start from the same warm-start point.
        let n = params.len();
        let mut grad = vec![0.0; n];
        let etas_snapshot = self.cached_etas.borrow().clone();
        for i in 0..n {
            let eps = 1e-4;
            // Forward
            *self.cached_etas.borrow_mut() = etas_snapshot.clone();
            let mut pp = params.to_vec();
            pp[i] += eps;
            let fp = self.laplace_ofv(&pp)?;
            // Backward
            *self.cached_etas.borrow_mut() = etas_snapshot.clone();
            let mut pm = params.to_vec();
            pm[i] -= eps;
            let fm = self.laplace_ofv(&pm)?;
            grad[i] = (fp - fm) / (2.0 * eps);
        }
        // Restore
        *self.cached_etas.borrow_mut() = etas_snapshot;
        Ok(grad)
    }
}

impl FoceEstimator {
    /// Create a new FOCE estimator with given configuration.
    pub fn new(config: FoceConfig) -> Self {
        Self { config }
    }

    /// Create a new FOCE estimator with default FOCEI configuration.
    pub fn focei() -> Self {
        Self::new(FoceConfig::default())
    }

    /// Create a new FOCE estimator (no interaction).
    pub fn foce() -> Self {
        Self::new(FoceConfig { interaction: false, ..FoceConfig::default() })
    }

    // ----- 1-compartment IV ------------------------------------------------

    /// Fit a 1-compartment IV bolus PK model using FOCE/FOCEI (diagonal Ω).
    ///
    /// # Arguments
    /// - `times`, `y`, `subject_idx`: observation data (flat arrays)
    /// - `n_subjects`: number of unique subjects
    /// - `doses`: per-subject dose amounts (length = n_subjects)
    /// - `error_model`: observation error model
    /// - `theta_init`: initial population parameters `[CL, V]`
    /// - `omega_init`: initial random effect SDs `[ω_CL, ω_V]`
    pub fn fit_1cpt_iv(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        if omega_init.len() != 2 {
            return Err(Error::Validation(
                "omega_init must have 2 elements for 1-cpt IV".to_string(),
            ));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_iv_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 1-compartment IV model with per-subject dosing regimens (multi-dose + infusion).
    pub fn fit_1cpt_iv_with_regimens_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        regimens: &[DosingRegimen],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if regimens.len() != n_subjects {
            return Err(Error::Validation(format!(
                "regimens length {} != n_subjects {n_subjects}",
                regimens.len()
            )));
        }
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            regimens[sid].conc_1cpt(cl, v, 1.0, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            omega_init,
            2,
            &conc_fn,
        )
    }

    /// Fit a 1-compartment IV bolus PK model with correlated random effects.
    ///
    /// Like [`fit_1cpt_iv`] but accepts a full [`OmegaMatrix`] for the
    /// inter-individual variability, allowing off-diagonal correlations.
    pub fn fit_1cpt_iv_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if theta_init.len() != 2 {
            return Err(Error::Validation("theta_init must have 2 elements [CL, V]".to_string()));
        }
        if omega_init.dim() != 2 {
            return Err(Error::Validation("omega must be 2×2".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ke = cl / v;
            (dose / v) * (-ke * t).exp()
        };

        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            2,
            &conc_fn,
        )
    }

    // ----- 2-compartment IV ------------------------------------------------

    /// Fit a 2-compartment IV bolus PK model using FOCE/FOCEI (diagonal Ω).
    ///
    /// # Arguments
    /// - `times`, `y`, `subject_idx`: observation data
    /// - `n_subjects`: number of unique subjects
    /// - `dose`: IV bolus dose amount
    /// - `error_model`: observation error model
    /// - `theta_init`: initial population parameters `[CL, V1, Q, V2]`
    /// - `omega_init`: initial random effect SDs `[ω_CL, ω_V1, ω_Q, ω_V2]`
    pub fn fit_2cpt_iv(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        if omega_init.len() != 4 {
            return Err(Error::Validation(
                "omega_init must have 4 elements for 2-cpt IV".to_string(),
            ));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_iv_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 2-compartment IV model with per-subject dosing regimens (multi-dose + infusion).
    pub fn fit_2cpt_iv_with_regimens_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        regimens: &[DosingRegimen],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if regimens.len() != n_subjects {
            return Err(Error::Validation(format!(
                "regimens length {} != n_subjects {n_subjects}",
                regimens.len()
            )));
        }
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            regimens[sid].conc_2cpt_iv(cl, v1, v2, q, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            omega_init,
            4,
            &conc_fn,
        )
    }

    /// Fit a 2-compartment IV bolus PK model with correlated random effects.
    ///
    /// Like [`fit_2cpt_iv`] but accepts a full [`OmegaMatrix`] for the
    /// inter-individual variability, allowing off-diagonal correlations.
    pub fn fit_2cpt_iv_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if theta_init.len() != 4 {
            return Err(Error::Validation(
                "theta_init must have 4 elements [CL, V1, Q, V2]".to_string(),
            ));
        }
        if omega_init.dim() != 4 {
            return Err(Error::Validation("omega must be 4×4".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            pk::conc_iv_2cpt_macro(dose, cl, v1, v2, q, t)
        };

        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            4,
            &conc_fn,
        )
    }

    // ----- 2-compartment oral ----------------------------------------------

    /// Fit a 2-compartment oral PK model using FOCE/FOCEI (diagonal Ω).
    ///
    /// # Arguments
    /// - `times`, `y`, `subject_idx`: observation data
    /// - `n_subjects`: number of unique subjects
    /// - `dose`, `bioav`: dosing information
    /// - `error_model`: observation error model
    /// - `theta_init`: initial population parameters `[CL, V1, Q, V2, Ka]`
    /// - `omega_init`: initial random effect SDs `[ω_CL, ω_V1, ω_Q, ω_V2, ω_Ka]`
    pub fn fit_2cpt_oral(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        if omega_init.len() != 5 {
            return Err(Error::Validation(
                "omega_init must have 5 elements for 2-cpt oral".to_string(),
            ));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_oral_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 2-compartment oral model with per-subject dosing regimens (multi-dose + infusion).
    pub fn fit_2cpt_oral_with_regimens_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        regimens: &[DosingRegimen],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if regimens.len() != n_subjects {
            return Err(Error::Validation(format!(
                "regimens length {} != n_subjects {n_subjects}",
                regimens.len()
            )));
        }
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let ka = theta[4] * eta[4].exp();
            regimens[sid].conc_2cpt_oral(cl, v1, v2, q, ka, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            omega_init,
            5,
            &conc_fn,
        )
    }

    /// Fit a 1-compartment oral model with per-subject dosing regimens (multi-dose + infusion).
    pub fn fit_1cpt_oral_with_regimens_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        regimens: &[DosingRegimen],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if regimens.len() != n_subjects {
            return Err(Error::Validation(format!(
                "regimens length {} != n_subjects {n_subjects}",
                regimens.len()
            )));
        }
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ka = theta[2] * eta[2].exp();
            regimens[sid].conc_1cpt(cl, v, ka, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            omega_init,
            3,
            &conc_fn,
        )
    }

    /// Fit a 3-compartment IV model with per-subject dosing regimens (multi-dose + infusion).
    pub fn fit_3cpt_iv_with_regimens_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        regimens: &[DosingRegimen],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if regimens.len() != n_subjects {
            return Err(Error::Validation(format!(
                "regimens length {} != n_subjects {n_subjects}",
                regimens.len()
            )));
        }
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            regimens[sid].conc_3cpt_iv(cl, v1, v2, q2, v3, q3, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            omega_init,
            6,
            &conc_fn,
        )
    }

    /// Fit a 3-compartment oral model with per-subject dosing regimens (multi-dose + infusion).
    pub fn fit_3cpt_oral_with_regimens_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        regimens: &[DosingRegimen],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if regimens.len() != n_subjects {
            return Err(Error::Validation(format!(
                "regimens length {} != n_subjects {n_subjects}",
                regimens.len()
            )));
        }
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            let ka = theta[6] * eta[6].exp();
            regimens[sid].conc_3cpt_oral(cl, v1, v2, q2, v3, q3, ka, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            omega_init,
            7,
            &conc_fn,
        )
    }

    /// Fit a 2-compartment oral PK model with correlated random effects.
    ///
    /// Like [`fit_2cpt_oral`] but accepts a full [`OmegaMatrix`] for the
    /// inter-individual variability, allowing off-diagonal correlations.
    pub fn fit_2cpt_oral_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if theta_init.len() != 5 {
            return Err(Error::Validation(
                "theta_init must have 5 elements [CL, V1, Q, V2, Ka]".to_string(),
            ));
        }
        if omega_init.dim() != 5 {
            return Err(Error::Validation("omega must be 5×5".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let ka = theta[4] * eta[4].exp();
            pk::conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t)
        };

        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            5,
            &conc_fn,
        )
    }

    // ----- 1-compartment oral with covariates ------------------------------

    /// Fit a 1-compartment oral PK model with integrated covariate effects
    /// using FOCE/FOCEI.
    ///
    /// Covariates modify structural parameters **before** random effects:
    ///
    /// ```text
    /// θ_individual[k] = apply_covariates(θ_pop[k], covariates_for_k, subject) * exp(η_k)
    /// ```
    ///
    /// For allometric scaling with fixed exponents:
    ///
    /// ```text
    /// CL = θ_CL * (WT/70)^0.75 * exp(η_CL)
    /// V  = θ_V  * (WT/70)^1.00 * exp(η_V)
    /// ```
    pub fn fit_1cpt_oral_with_covariates(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        covariates: &[CovariateSpec],
    ) -> Result<FoceResult> {
        if omega_init.len() != 3 {
            return Err(Error::Validation("omega_init must have 3 elements".to_string()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_oral_with_covariates_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            om,
            covariates,
        )
    }

    /// Fit a 1-compartment oral PK model with integrated covariate effects
    /// and correlated random effects.
    pub fn fit_1cpt_oral_with_covariates_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        covariates: &[CovariateSpec],
    ) -> Result<FoceResult> {
        if theta_init.len() != 3 {
            return Err(Error::Validation("theta_init must have 3 elements".to_string()));
        }
        if omega_init.dim() != 3 {
            return Err(Error::Validation("omega must be 3×3".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        error_model.validate()?;
        if let Some(lloq) = self.config.lloq
            && (!lloq.is_finite() || lloq < 0.0)
        {
            return Err(Error::Validation("config.lloq must be finite and >= 0".to_string()));
        }

        // Validate covariate specs.
        for cov in covariates {
            if cov.param_idx >= 3 {
                return Err(Error::Validation(format!(
                    "covariate param_idx {} out of range for 3-param model",
                    cov.param_idx
                )));
            }
            if cov.values.len() != n_subjects {
                return Err(Error::Validation(format!(
                    "covariate values length {} != n_subjects {}",
                    cov.values.len(),
                    n_subjects
                )));
            }
        }

        let n_obs = times.len();
        let n_eta = 3;
        let sigma_init = sigma_from_error_model(&error_model);
        let mut error_model = error_model;

        // Group observations by subject.
        let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
        for i in 0..n_obs {
            subj_obs[subject_idx[i]].push((times[i], y[i]));
        }

        // Save initial omega variances for damping cap.
        let omega_init_mat = omega_init.to_matrix();
        let omega_init_vars: Vec<f64> = (0..n_eta).map(|i| omega_init_mat[i][i]).collect();
        let damping = self.config.omega_damping;
        let max_ratio = self.config.omega_max_ratio;

        // Working state.
        let mut theta = theta_init.to_vec();
        let mut om = omega_init;
        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];

        let mut prev_ofv = f64::MAX;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.config.max_outer_iter {
            n_iter = iter + 1;

            // Inner optimization: update ETAs for each subject.
            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                // Build subject-specific theta with covariate adjustments.
                let theta_s = apply_all_covariates(&theta, covariates, s);
                etas[s] = inner_optimize_eta(
                    &subj_obs[s],
                    &theta_s,
                    &om,
                    &error_model,
                    doses[s],
                    bioav,
                    &etas[s],
                    self.config.max_inner_iter,
                    self.config.lloq,
                )?;
            }

            // Eta centering: absorb mean(η) into θ so E[η]=0.
            let active_count = subj_obs.iter().filter(|o| !o.is_empty()).count() as f64;
            if active_count > 0.0 {
                for k in 0..n_eta {
                    let eta_mean: f64 = etas
                        .iter()
                        .enumerate()
                        .filter(|(s, _)| !subj_obs[*s].is_empty())
                        .map(|(_, e)| e[k])
                        .sum::<f64>()
                        / active_count;
                    if eta_mean.abs() > 1e-10 {
                        theta[k] *= eta_mean.exp();
                        for s in 0..n_subjects {
                            etas[s][k] -= eta_mean;
                        }
                    }
                }
            }

            // Sigma M-step with damping.
            if self.config.estimate_sigma {
                let sigma_mle =
                    estimate_sigma_mle_1cpt(&subj_obs, &theta, &etas, doses, bioav, &error_model);
                error_model = blend_sigma(&error_model, &sigma_mle, 0.3);
            }

            // Compute OFV at current (theta, omega, etas) with covariates.
            let ofv = compute_foce_ofv_with_covariates(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                doses,
                bioav,
                &etas,
                n_eta,
                covariates,
                self.config.lloq,
                self.config.interaction,
            )?;

            // Check convergence (absolute or relative OFV change).
            let delta = (prev_ofv - ofv).abs();
            let converged_abs = delta < self.config.tol;
            let converged_rel =
                prev_ofv.abs() > 0.0 && delta / prev_ofv.abs() < self.config.rel_tol;
            if (converged_abs || converged_rel) && iter > 0 {
                converged = true;
                prev_ofv = ofv;
                break;
            }
            prev_ofv = ofv;

            // Outer step: update theta + omega (with covariate-adjusted predictions).
            let (theta_new, om_empirical) = outer_step_with_covariates(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                doses,
                bioav,
                &etas,
                n_eta,
                covariates,
                self.config.lloq,
            )?;

            // Apply ramped damping: no damping for first 20 iters, then ramp to full.
            // Cap is always applied (even undamped) to prevent omega runaway.
            let eff_damping = damping * (iter as f64 / 20.0).min(1.0);
            let om_new =
                damped_omega_blend(&om, &om_empirical, eff_damping, &omega_init_vars, max_ratio)
                    .unwrap_or_else(|_| om.clone());

            // Parameter stability check (after burn-in).
            if iter > 20 {
                let theta_change = max_relative_change(&theta, &theta_new);
                let om_old_diag: Vec<f64> = {
                    let m = om.to_matrix();
                    (0..n_eta).map(|i| m[i][i]).collect()
                };
                let om_new_diag: Vec<f64> = {
                    let m = om_new.to_matrix();
                    (0..n_eta).map(|i| m[i][i]).collect()
                };
                let om_change = max_relative_change(&om_old_diag, &om_new_diag);
                if theta_change < 1e-3 && om_change < 1e-3 {
                    theta = theta_new;
                    om = om_new;
                    converged = true;
                    break;
                }
            }

            theta = theta_new;
            om = om_new;
        }

        let correlation = om.correlation();
        let omega_diag = om.sds();
        let sigma = sigma_from_error_model(&error_model);
        let covariance_step = None;
        Ok(FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv: prev_ofv,
            converged,
            n_iter,
            sigma,
            sigma_init,
            covariance_step,
        })
    }

    // ----- Generic FOCE fitting engine -------------------------------------

    /// Generic FOCE/FOCEI fitting: concentration function injected via closure.
    ///
    /// `conc_fn(theta, eta, t)` returns the individual concentration at time `t`
    /// given population parameters `theta` and random effects `eta`.
    fn fit_generic(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        n_eta: usize,
        conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    ) -> Result<FoceResult> {
        if let Some(lloq) = self.config.lloq
            && (!lloq.is_finite() || lloq < 0.0)
        {
            return Err(Error::Validation("config.lloq must be finite and >= 0".to_string()));
        }
        let n_obs = times.len();
        let sigma_init = sigma_from_error_model(&error_model);
        let mut error_model = error_model;

        // Group observations by subject.
        let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
        for i in 0..n_obs {
            subj_obs[subject_idx[i]].push((times[i], y[i]));
        }

        // Save initial omega variances for damping cap.
        let omega_init_mat = omega_init.to_matrix();
        let omega_init_vars: Vec<f64> = (0..n_eta).map(|i| omega_init_mat[i][i]).collect();
        let damping = self.config.omega_damping;
        let max_ratio = self.config.omega_max_ratio;

        // Working state.
        let mut theta = theta_init.to_vec();
        let mut om = omega_init;
        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];

        let mut prev_ofv = f64::MAX;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.config.max_outer_iter {
            n_iter = iter + 1;

            // Inner optimization: update ETAs for each subject.
            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                etas[s] = inner_optimize_eta_generic(
                    &subj_obs[s],
                    &theta,
                    &om,
                    &error_model,
                    &etas[s],
                    self.config.max_inner_iter,
                    doses[s],
                    conc_fn,
                    self.config.lloq,
                )?;
            }

            // Eta centering: absorb mean(η) into θ so E[η]=0.
            let active_count = subj_obs.iter().filter(|o| !o.is_empty()).count() as f64;
            if active_count > 0.0 {
                for k in 0..n_eta {
                    let eta_mean: f64 = etas
                        .iter()
                        .enumerate()
                        .filter(|(s, _)| !subj_obs[*s].is_empty())
                        .map(|(_, e)| e[k])
                        .sum::<f64>()
                        / active_count;
                    if eta_mean.abs() > 1e-10 {
                        theta[k] *= eta_mean.exp();
                        for s in 0..n_subjects {
                            etas[s][k] -= eta_mean;
                        }
                    }
                }
            }

            // Sigma M-step with damping to prevent sigma–omega feedback loop.
            if self.config.estimate_sigma {
                let sigma_mle = estimate_sigma_mle_generic(
                    &subj_obs,
                    &theta,
                    &etas,
                    doses,
                    &error_model,
                    conc_fn,
                );
                error_model = blend_sigma(&error_model, &sigma_mle, 0.3);
            }

            // Compute OFV at current (theta, omega, etas).
            let ofv = compute_foce_ofv_generic(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                &etas,
                n_eta,
                doses,
                conc_fn,
                self.config.lloq,
                self.config.interaction,
            )?;

            // Check convergence (absolute or relative OFV change).
            let delta = (prev_ofv - ofv).abs();
            let converged_abs = delta < self.config.tol;
            let converged_rel =
                prev_ofv.abs() > 0.0 && delta / prev_ofv.abs() < self.config.rel_tol;
            if (converged_abs || converged_rel) && iter > 0 {
                converged = true;
                prev_ofv = ofv;
                break;
            }
            prev_ofv = ofv;

            // Outer step: update theta + omega.
            let (theta_new, om_empirical) = outer_step_generic(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                &etas,
                n_eta,
                doses,
                conc_fn,
                self.config.lloq,
            )?;

            // Apply ramped damping: no damping for first 20 iters, then ramp to full.
            // Cap is always applied (even undamped) to prevent omega runaway.
            let eff_damping = damping * (iter as f64 / 20.0).min(1.0);
            let om_new =
                damped_omega_blend(&om, &om_empirical, eff_damping, &omega_init_vars, max_ratio)
                    .unwrap_or_else(|_| om.clone());

            // Parameter stability check (after burn-in).
            if iter > 20 {
                let theta_change = max_relative_change(&theta, &theta_new);
                let om_old_diag: Vec<f64> = {
                    let m = om.to_matrix();
                    (0..n_eta).map(|i| m[i][i]).collect()
                };
                let om_new_diag: Vec<f64> = {
                    let m = om_new.to_matrix();
                    (0..n_eta).map(|i| m[i][i]).collect()
                };
                let om_change = max_relative_change(&om_old_diag, &om_new_diag);
                if theta_change < 1e-3 && om_change < 1e-3 {
                    theta = theta_new;
                    om = om_new;
                    converged = true;
                    break;
                }
            }

            theta = theta_new;
            om = om_new;
        }

        let correlation = om.correlation();
        let omega_diag = om.sds();
        let sigma = sigma_from_error_model(&error_model);
        Ok(FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv: prev_ofv,
            converged,
            n_iter,
            sigma,
            sigma_init,
            covariance_step: None,
        })
    }

    /// Fit a 1-compartment oral PK model using FOCE/FOCEI (diagonal Ω).
    ///
    /// # Arguments
    /// - `times`, `y`, `subject_idx`: observation data (same format as NLME model)
    /// - `n_subjects`: number of unique subjects
    /// - `dose`, `bioav`: dosing information
    /// - `error_model`: observation error model
    /// - `theta_init`: initial population parameters `[CL_pop, V_pop, Ka_pop]`
    /// - `omega_init`: initial random effect SDs `[ω_CL, ω_V, ω_Ka]`
    pub fn fit_1cpt_oral(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        if omega_init.len() != 3 {
            return Err(Error::Validation("omega_init must have 3 elements".to_string()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_oral_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 1-compartment oral PK model with configurable random-effect transforms.
    ///
    /// This enables bounded (logit-normal) random effects for selected parameters.
    pub fn fit_1cpt_oral_with_transforms_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        transforms: &[RandomEffectTransform],
    ) -> Result<FoceResult> {
        if theta_init.len() != 3 {
            return Err(Error::Validation("theta_init must have 3 elements".to_string()));
        }
        if omega_init.dim() != 3 {
            return Err(Error::Validation("omega must be 3×3".to_string()));
        }
        if transforms.len() != 3 {
            return Err(Error::Validation("transforms must have 3 elements".to_string()));
        }
        for k in 0..3 {
            validate_random_effect_transform(theta_init[k], transforms[k])?;
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;
        if let Some(lloq) = self.config.lloq
            && (!lloq.is_finite() || lloq < 0.0)
        {
            return Err(Error::Validation("config.lloq must be finite and >= 0".to_string()));
        }

        let tr = [transforms[0], transforms[1], transforms[2]];
        let conc_fn = move |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = apply_random_effect_transform(theta[0], eta[0], tr[0]);
            let v = apply_random_effect_transform(theta[1], eta[1], tr[1]);
            let ka = apply_random_effect_transform(theta[2], eta[2], tr[2]);
            pk::conc_oral(dose, bioav, cl, v, ka, t)
        };

        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            3,
            &conc_fn,
        )
    }

    /// Fit a 1-compartment oral PK model with correlated random effects.
    ///
    /// Like [`fit_1cpt_oral`] but accepts a full [`OmegaMatrix`] for the
    /// inter-individual variability, allowing off-diagonal correlations
    /// (e.g. CL–V correlation).
    pub fn fit_1cpt_oral_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if theta_init.len() != 3 {
            return Err(Error::Validation("theta_init must have 3 elements".to_string()));
        }
        if omega_init.dim() != 3 {
            return Err(Error::Validation("omega must be 3×3".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;
        if let Some(lloq) = self.config.lloq
            && (!lloq.is_finite() || lloq < 0.0)
        {
            return Err(Error::Validation("config.lloq must be finite and >= 0".to_string()));
        }

        let n_obs = times.len();
        let n_eta = 3;
        let n_theta = 3;
        let sigma_init = sigma_from_error_model(&error_model);
        let max_inner = self.config.max_inner_iter;
        let max_outer = self.config.max_outer_iter;
        let omega_fixed = &self.config.omega_fixed;

        // Group observations by subject.
        let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
        for i in 0..n_obs {
            subj_obs[subject_idx[i]].push((times[i], y[i]));
        }

        // Save initial omega variances for damping cap.
        let omega_init_mat = omega_init.to_matrix();
        let omega_init_vars: Vec<f64> = (0..n_eta).map(|i| omega_init_mat[i][i]).collect();
        let damping = self.config.omega_damping;
        let max_ratio = self.config.omega_max_ratio;

        // Working state.
        let mut theta = theta_init.to_vec();
        let mut om = omega_init;
        // Apply omega_fixed to initial omega: zero out fixed rows/cols.
        apply_omega_fixed_matrix(&mut om, &omega_init_mat, omega_fixed);
        if self.config.diagonal_omega {
            enforce_diagonal_omega(&mut om);
        }
        let mut error_model = error_model;
        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];
        let mut prev_ofv = f64::MAX;
        let mut converged = false;
        let mut n_iter = 0usize;

        // =====================================================================
        // Phase 1: EM warm-up — stable approach to correct basin
        // =====================================================================
        let n_em_warmup = 20.min(max_outer);

        for iter in 0..n_em_warmup {
            n_iter += 1;

            // E-step: optimize etas for each subject.
            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                etas[s] = inner_optimize_eta(
                    &subj_obs[s],
                    &theta,
                    &om,
                    &error_model,
                    doses[s],
                    bioav,
                    &etas[s],
                    max_inner,
                    self.config.lloq,
                )?;
            }
            // Zero out etas for fixed omega dimensions.
            apply_omega_fixed_etas(&mut etas, omega_fixed);

            // Eta centering: absorb mean(η_k) into θ_k (skip fixed dims).
            let active_count = subj_obs.iter().filter(|o| !o.is_empty()).count() as f64;
            if active_count > 0.0 {
                for k in 0..n_eta {
                    if k < omega_fixed.len() && omega_fixed[k] {
                        continue; // skip centering for fixed omega
                    }
                    let eta_mean: f64 = etas
                        .iter()
                        .enumerate()
                        .filter(|(s, _)| !subj_obs[*s].is_empty())
                        .map(|(_, e)| e[k])
                        .sum::<f64>()
                        / active_count;
                    if eta_mean.abs() > 1e-10 {
                        theta[k] *= eta_mean.exp();
                        for s in 0..n_subjects {
                            etas[s][k] -= eta_mean;
                        }
                    }
                }
            }

            // Sigma M-step with damping to prevent the EM sigma–omega feedback
            // loop where sigma absorbs inter-individual variance.  Phase 2
            // (nested Laplace Newton) will jointly refine sigma further.
            if self.config.estimate_sigma {
                let sigma_mle =
                    estimate_sigma_mle_1cpt(&subj_obs, &theta, &etas, doses, bioav, &error_model);
                error_model = blend_sigma(&error_model, &sigma_mle, 0.3);
            }

            // Omega M-step: Hessian-corrected empirical + damped blend.
            let om_corrected = omega_corrected_1cpt(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                &etas,
                n_eta,
                doses,
                bioav,
                self.config.lloq,
            );
            let eff_damping = damping * (iter as f64 / 10.0).min(1.0);
            om = damped_omega_blend(&om, &om_corrected, eff_damping, &omega_init_vars, max_ratio)
                .unwrap_or_else(|_| om.clone());
            // Restore fixed omega entries after M-step.
            apply_omega_fixed_matrix(&mut om, &omega_init_mat, omega_fixed);
            if self.config.diagonal_omega {
                enforce_diagonal_omega(&mut om);
            }

            // OFV convergence check.
            let ofv = compute_foce_ofv(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                doses,
                bioav,
                &etas,
                n_eta,
                self.config.lloq,
                self.config.interaction,
            )?;
            let delta = (prev_ofv - ofv).abs();
            let converged_abs = delta < self.config.tol;
            let converged_rel =
                prev_ofv.abs() > 0.0 && delta / prev_ofv.abs() < self.config.rel_tol;
            if (converged_abs || converged_rel) && iter > 0 {
                converged = true;
                prev_ofv = ofv;
                break;
            }
            prev_ofv = ofv;
        }

        // If EM already converged or no Newton budget left, return.
        if converged || n_iter >= max_outer {
            let sigma = sigma_from_error_model(&error_model);
            let correlation = om.correlation();
            let omega_diag = om.sds();
            let covariance_step = compute_covariance_step_1cpt(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                doses,
                bioav,
                &etas,
                n_eta,
                self.config.lloq,
            );
            return Ok(FoceResult {
                theta,
                omega: omega_diag,
                omega_matrix: om,
                correlation,
                eta: etas,
                ofv: prev_ofv,
                converged,
                n_iter,
                sigma,
                sigma_init,
                covariance_step,
            });
        }

        // =====================================================================
        // Phase 2: Newton refinement on nested Laplace OFV
        // =====================================================================
        let (em_variant, n_sigma_params, fixed_sigma): (u8, usize, Vec<f64>) = match &error_model {
            ErrorModel::Additive(s) => (0, 1, vec![*s]),
            ErrorModel::Proportional(s) => (1, 1, vec![*s]),
            ErrorModel::Combined { sigma_add, sigma_prop } => (2, 2, vec![*sigma_add, *sigma_prop]),
            ErrorModel::Exponential(s) => (3, 1, vec![*s]),
            ErrorModel::Power { sigma, power } => (4, 2, vec![*sigma, *power]),
        };

        let nested = FoceNestedOfv {
            subj_obs: &subj_obs,
            doses,
            bioav,
            n_theta,
            n_eta,
            em_variant,
            estimate_sigma: self.config.estimate_sigma,
            fixed_sigma: fixed_sigma.clone(),
            max_inner_iter: max_inner,
            lloq: self.config.lloq,
            cached_etas: std::cell::RefCell::new(etas),
        };

        // Pack current parameters in log-space.
        let omega_sds = om.sds();
        let n_pack = n_theta + n_eta + if self.config.estimate_sigma { n_sigma_params } else { 0 };
        let mut params = Vec::with_capacity(n_pack);
        for &t in &theta {
            params.push(t.ln());
        }
        for &s in &omega_sds {
            params.push(s.ln());
        }
        if self.config.estimate_sigma {
            match em_variant {
                2 => {
                    params.push(fixed_sigma[0].ln());
                    params.push(fixed_sigma[1].ln());
                }
                4 => {
                    params.push(fixed_sigma[0].ln());
                    params.push(fixed_sigma[1]);
                }
                _ => {
                    params.push(fixed_sigma[0].ln());
                }
            }
        }

        prev_ofv = nested.laplace_ofv(&params)?;
        let max_newton = max_outer - n_iter;

        for _niter in 0..max_newton {
            n_iter += 1;
            let mut grad = nested.gradient(&params)?;
            // Zero out gradient for fixed omega dimensions in Phase 2.
            for (k, &fixed) in omega_fixed.iter().enumerate() {
                if fixed && n_theta + k < grad.len() {
                    grad[n_theta + k] = 0.0;
                }
            }
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < 1e-4 {
                converged = true;
                break;
            }

            // Hessian via FD on gradient.
            let np = params.len();
            let eps_h = 1e-3;
            let mut hess = vec![vec![0.0; np]; np];
            let etas_snap = nested.cached_etas.borrow().clone();
            for i in 0..np {
                *nested.cached_etas.borrow_mut() = etas_snap.clone();
                let mut pp = params.clone();
                pp[i] += eps_h;
                let gp = nested.gradient(&pp)?;
                *nested.cached_etas.borrow_mut() = etas_snap.clone();
                pp[i] = params[i] - eps_h;
                let gm = nested.gradient(&pp)?;
                for j in 0..np {
                    hess[i][j] = (gp[j] - gm[j]) / (2.0 * eps_h);
                }
            }
            *nested.cached_etas.borrow_mut() = etas_snap;

            // Symmetrise + Levenberg-Marquardt regularisation.
            for i in 0..np {
                for j in 0..i {
                    let avg = 0.5 * (hess[i][j] + hess[j][i]);
                    hess[i][j] = avg;
                    hess[j][i] = avg;
                }
                hess[i][i] += hess[i][i].abs().max(1.0) * 1e-2;
            }
            let delta = solve_linear(&hess, &grad);

            // Backtracking Newton step with max step cap (0.5 in log-space).
            let cur_f = nested.laplace_ofv(&params)?;
            let mut step = 1.0;
            let max_log_step = 0.5;
            let mut accepted = false;
            for _ in 0..15 {
                let trial: Vec<f64> = params
                    .iter()
                    .zip(delta.iter())
                    .map(|(&x, &d)| {
                        let raw = -step * d;
                        let capped = raw.clamp(-max_log_step, max_log_step);
                        (x + capped).clamp(-20.0, 20.0)
                    })
                    .collect();
                let trial_f = nested.laplace_ofv(&trial)?;
                if trial_f < cur_f {
                    params = trial;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                converged = true;
                break;
            }

            // Convergence check.
            let ofv = nested.laplace_ofv(&params)?;
            let abs_change = (ofv - prev_ofv).abs();
            let rel_change =
                if prev_ofv.abs() > 1e-10 { abs_change / prev_ofv.abs() } else { abs_change };
            if abs_change < self.config.tol || rel_change < self.config.rel_tol {
                converged = true;
                break;
            }
            prev_ofv = ofv;
        }

        // Unpack final parameters.
        let (theta, mut om, final_em) = nested.unpack(&params)?;
        // Restore fixed omega entries after Phase 2 optimization.
        apply_omega_fixed_matrix(&mut om, &omega_init_mat, omega_fixed);
        if self.config.diagonal_omega {
            enforce_diagonal_omega(&mut om);
        }
        let sigma = sigma_from_error_model(&final_em);

        // Final E-step.
        let mut etas = nested.cached_etas.borrow().clone();
        for s in 0..n_subjects {
            if subj_obs[s].is_empty() {
                continue;
            }
            etas[s] = inner_optimize_eta(
                &subj_obs[s],
                &theta,
                &om,
                &final_em,
                doses[s],
                bioav,
                &etas[s],
                max_inner,
                self.config.lloq,
            )?;
        }
        // Zero out fixed etas after final E-step.
        apply_omega_fixed_etas(&mut etas, omega_fixed);

        let ofv = compute_foce_ofv(
            &subj_obs,
            &theta,
            &om,
            &final_em,
            doses,
            bioav,
            &etas,
            n_eta,
            self.config.lloq,
            self.config.interaction,
        )?;
        let correlation = om.correlation();
        let omega_diag = om.sds();
        let covariance_step = compute_covariance_step_1cpt(
            &subj_obs,
            &theta,
            &om,
            &final_em,
            doses,
            bioav,
            &etas,
            n_eta,
            self.config.lloq,
        );

        Ok(FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv,
            converged,
            n_iter,
            sigma,
            sigma_init,
            covariance_step,
        })
    }

    /// Fit a 3-compartment IV PK model using FOCE/FOCEI (diagonal Ω).
    pub fn fit_3cpt_iv(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        if omega_init.len() != 6 {
            return Err(Error::Validation(
                "omega_init must have 6 elements for 3-cpt IV".to_string(),
            ));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_3cpt_iv_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 3-compartment IV PK model with correlated random effects.
    pub fn fit_3cpt_iv_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if theta_init.len() != 6 {
            return Err(Error::Validation(
                "theta_init must have 6 elements [CL, V1, Q2, V2, Q3, V3]".to_string(),
            ));
        }
        if omega_init.dim() != 6 {
            return Err(Error::Validation("omega must be 6×6".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            pk::conc_iv_3cpt_macro(dose, t, cl, v1, q2, v2, q3, v3)
        };

        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            6,
            &conc_fn,
        )
    }

    /// Fit a 3-compartment oral PK model using FOCE/FOCEI (diagonal Ω).
    pub fn fit_3cpt_oral(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        if omega_init.len() != 7 {
            return Err(Error::Validation(
                "omega_init must have 7 elements for 3-cpt oral".to_string(),
            ));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_3cpt_oral_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            om,
        )
    }

    /// Fit a 3-compartment oral PK model with correlated random effects.
    pub fn fit_3cpt_oral_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        if theta_init.len() != 7 {
            return Err(Error::Validation(
                "theta_init must have 7 elements [CL, V1, Q2, V2, Q3, V3, Ka]".to_string(),
            ));
        }
        if omega_init.dim() != 7 {
            return Err(Error::Validation("omega must be 7×7".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!(
                "doses length {} != n_subjects {}",
                doses.len(),
                n_subjects
            )));
        }
        error_model.validate()?;

        let conc_fn = move |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let effective_dose = dose * bioav;
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            let ka = theta[6] * eta[6].exp();
            pk::conc_oral_3cpt_macro(effective_dose, t, cl, v1, q2, v2, q3, v3, ka)
        };

        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            7,
            &conc_fn,
        )
    }

    /// Fit 1-cpt IV with configurable random-effect transforms.
    pub fn fit_1cpt_iv_with_transforms_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        transforms: &[RandomEffectTransform],
    ) -> Result<FoceResult> {
        if theta_init.len() != 2 || omega_init.dim() != 2 || transforms.len() != 2 {
            return Err(Error::Validation(
                "1-cpt IV transforms require theta/omega/transforms length 2".to_string(),
            ));
        }
        for k in 0..2 {
            validate_random_effect_transform(theta_init[k], transforms[k])?;
        }
        let tr = [transforms[0], transforms[1]];
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = apply_random_effect_transform(theta[0], eta[0], tr[0]);
            let v = apply_random_effect_transform(theta[1], eta[1], tr[1]);
            let ke = cl / v;
            (dose / v) * (-ke * t).exp()
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            2,
            &conc_fn,
        )
    }

    /// Fit 2-cpt IV with configurable random-effect transforms.
    pub fn fit_2cpt_iv_with_transforms_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        transforms: &[RandomEffectTransform],
    ) -> Result<FoceResult> {
        if theta_init.len() != 4 || omega_init.dim() != 4 || transforms.len() != 4 {
            return Err(Error::Validation(
                "2-cpt IV transforms require theta/omega/transforms length 4".to_string(),
            ));
        }
        for k in 0..4 {
            validate_random_effect_transform(theta_init[k], transforms[k])?;
        }
        let tr = [transforms[0], transforms[1], transforms[2], transforms[3]];
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = apply_random_effect_transform(theta[0], eta[0], tr[0]);
            let v1 = apply_random_effect_transform(theta[1], eta[1], tr[1]);
            let q = apply_random_effect_transform(theta[2], eta[2], tr[2]);
            let v2 = apply_random_effect_transform(theta[3], eta[3], tr[3]);
            pk::conc_iv_2cpt_macro(dose, cl, v1, v2, q, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            4,
            &conc_fn,
        )
    }

    /// Fit 2-cpt oral with configurable random-effect transforms.
    pub fn fit_2cpt_oral_with_transforms_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        transforms: &[RandomEffectTransform],
    ) -> Result<FoceResult> {
        if theta_init.len() != 5 || omega_init.dim() != 5 || transforms.len() != 5 {
            return Err(Error::Validation(
                "2-cpt oral transforms require theta/omega/transforms length 5".to_string(),
            ));
        }
        for k in 0..5 {
            validate_random_effect_transform(theta_init[k], transforms[k])?;
        }
        let tr = [transforms[0], transforms[1], transforms[2], transforms[3], transforms[4]];
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = apply_random_effect_transform(theta[0], eta[0], tr[0]);
            let v1 = apply_random_effect_transform(theta[1], eta[1], tr[1]);
            let q = apply_random_effect_transform(theta[2], eta[2], tr[2]);
            let v2 = apply_random_effect_transform(theta[3], eta[3], tr[3]);
            let ka = apply_random_effect_transform(theta[4], eta[4], tr[4]);
            pk::conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            5,
            &conc_fn,
        )
    }

    /// Fit 3-cpt IV with configurable random-effect transforms.
    pub fn fit_3cpt_iv_with_transforms_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        transforms: &[RandomEffectTransform],
    ) -> Result<FoceResult> {
        if theta_init.len() != 6 || omega_init.dim() != 6 || transforms.len() != 6 {
            return Err(Error::Validation(
                "3-cpt IV transforms require theta/omega/transforms length 6".to_string(),
            ));
        }
        for k in 0..6 {
            validate_random_effect_transform(theta_init[k], transforms[k])?;
        }
        let tr = [
            transforms[0],
            transforms[1],
            transforms[2],
            transforms[3],
            transforms[4],
            transforms[5],
        ];
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = apply_random_effect_transform(theta[0], eta[0], tr[0]);
            let v1 = apply_random_effect_transform(theta[1], eta[1], tr[1]);
            let q2 = apply_random_effect_transform(theta[2], eta[2], tr[2]);
            let v2 = apply_random_effect_transform(theta[3], eta[3], tr[3]);
            let q3 = apply_random_effect_transform(theta[4], eta[4], tr[4]);
            let v3 = apply_random_effect_transform(theta[5], eta[5], tr[5]);
            pk::conc_iv_3cpt_macro(dose, t, cl, v1, q2, v2, q3, v3)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            6,
            &conc_fn,
        )
    }

    /// Fit 3-cpt oral with configurable random-effect transforms.
    pub fn fit_3cpt_oral_with_transforms_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        transforms: &[RandomEffectTransform],
    ) -> Result<FoceResult> {
        if theta_init.len() != 7 || omega_init.dim() != 7 || transforms.len() != 7 {
            return Err(Error::Validation(
                "3-cpt oral transforms require theta/omega/transforms length 7".to_string(),
            ));
        }
        for k in 0..7 {
            validate_random_effect_transform(theta_init[k], transforms[k])?;
        }
        let tr = [
            transforms[0],
            transforms[1],
            transforms[2],
            transforms[3],
            transforms[4],
            transforms[5],
            transforms[6],
        ];
        let conc_fn = move |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let effective_dose = dose * bioav;
            let cl = apply_random_effect_transform(theta[0], eta[0], tr[0]);
            let v1 = apply_random_effect_transform(theta[1], eta[1], tr[1]);
            let q2 = apply_random_effect_transform(theta[2], eta[2], tr[2]);
            let v2 = apply_random_effect_transform(theta[3], eta[3], tr[3]);
            let q3 = apply_random_effect_transform(theta[4], eta[4], tr[4]);
            let v3 = apply_random_effect_transform(theta[5], eta[5], tr[5]);
            let ka = apply_random_effect_transform(theta[6], eta[6], tr[6]);
            pk::conc_oral_3cpt_macro(effective_dose, t, cl, v1, q2, v2, q3, v3, ka)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
            7,
            &conc_fn,
        )
    }

    /// Fit 1-cpt oral with time-varying covariates.
    pub fn fit_1cpt_oral_with_time_varying_covariates(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        covariates: &[TimeVaryingCovariateSpec],
    ) -> Result<FoceResult> {
        if omega_init.len() != 3 {
            return Err(Error::Validation("omega_init must have 3 elements".into()));
        }
        validate_time_varying_covariates(covariates, 3, n_subjects)?;
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let theta_t = apply_all_time_varying_covariates(theta, covariates, sid, t);
            let cl = theta_t[0] * eta[0].exp();
            let v = theta_t[1] * eta[1].exp();
            let ka = theta_t[2] * eta[2].exp();
            pk::conc_oral(doses[sid], bioav, cl, v, ka, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            om,
            3,
            &conc_fn,
        )
    }

    /// Fit 1-cpt oral with inter-occasion variability (IOV).
    pub fn fit_1cpt_oral_with_iov(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        iov: &IovSpec,
    ) -> Result<FoceResult> {
        if omega_init.len() != 3 {
            return Err(Error::Validation("omega_init must have 3 elements".into()));
        }
        validate_iov_spec(iov, 3, n_subjects)?;
        let n_iiv = 3usize;
        let n_occ = iov_max_occasions(iov);
        let n_iov = iov.param_indices.len();
        let n_eta = n_iiv + n_occ * n_iov;
        let om = build_iov_omega(omega_init, iov)?;
        let subject_keys: Vec<f64> = (0..n_subjects).map(|s| s as f64).collect();
        let conc_fn = |theta: &[f64], eta: &[f64], subj_key: f64, t: f64| -> f64 {
            let sid = subj_key as usize;
            let mut theta_eff = theta.to_vec();
            let occ = occasion_index_for_time(&iov.occasion_start_times[sid], t).min(n_occ - 1);
            for (j, &pidx) in iov.param_indices.iter().enumerate() {
                let kappa_idx = n_iiv + occ * n_iov + j;
                theta_eff[pidx] *= eta[kappa_idx].exp();
            }
            let cl = theta_eff[0] * eta[0].exp();
            let v = theta_eff[1] * eta[1].exp();
            let ka = theta_eff[2] * eta[2].exp();
            pk::conc_oral(doses[sid], bioav, cl, v, ka, t)
        };
        self.fit_generic(
            times,
            y,
            subject_idx,
            n_subjects,
            &subject_keys,
            error_model,
            theta_init,
            om,
            n_eta,
            &conc_fn,
        )
    }

    /// 1-cpt IV FO wrapper (currently dispatched via the correlated FO path).
    pub fn fit_1cpt_iv_fo(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_iv_fo_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            om,
        )
    }

    /// 1-cpt IV FO path with correlated Ω.
    pub fn fit_1cpt_iv_fo_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        let est = FoceEstimator::new(FoceConfig { interaction: false, ..self.config.clone() });
        est.fit_1cpt_iv_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 1-cpt oral FO wrapper.
    pub fn fit_1cpt_oral_fo(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_oral_fo_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            om,
        )
    }

    /// 1-cpt oral FO path with correlated Ω.
    pub fn fit_1cpt_oral_fo_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        let est = FoceEstimator::new(FoceConfig { interaction: false, ..self.config.clone() });
        est.fit_1cpt_oral_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 2-cpt IV FO wrapper.
    pub fn fit_2cpt_iv_fo(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_iv_fo_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            om,
        )
    }

    /// 2-cpt IV FO path with correlated Ω.
    pub fn fit_2cpt_iv_fo_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        let est = FoceEstimator::new(FoceConfig { interaction: false, ..self.config.clone() });
        est.fit_2cpt_iv_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 2-cpt oral FO wrapper.
    pub fn fit_2cpt_oral_fo(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_oral_fo_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            om,
        )
    }

    /// 2-cpt oral FO path with correlated Ω.
    pub fn fit_2cpt_oral_fo_correlated(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
    ) -> Result<FoceResult> {
        let est = FoceEstimator::new(FoceConfig { interaction: false, ..self.config.clone() });
        est.fit_2cpt_oral_correlated(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 3-cpt IV FO wrapper.
    pub fn fit_3cpt_iv_fo(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        let est = FoceEstimator::new(FoceConfig { interaction: false, ..self.config.clone() });
        est.fit_3cpt_iv(times, y, subject_idx, n_subjects, doses, error_model, theta_init, omega_init)
    }

    /// 3-cpt oral FO wrapper.
    pub fn fit_3cpt_oral_fo(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
    ) -> Result<FoceResult> {
        let est = FoceEstimator::new(FoceConfig { interaction: false, ..self.config.clone() });
        est.fit_3cpt_oral(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 1-cpt IV ITS wrapper.
    pub fn fit_1cpt_iv_its(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        its_config: ItsConfig,
    ) -> Result<FoceResult> {
        let mut cfg = self.config.clone();
        cfg.max_outer_iter = its_config.max_iter.max(1);
        cfg.max_inner_iter = its_config.max_individual_iter.max(1);
        cfg.tol = its_config.tol.max(1e-12);
        cfg.omega_damping = its_config.omega_damping.clamp(0.0, 1.0);
        FoceEstimator::new(cfg).fit_1cpt_iv(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 1-cpt oral ITS wrapper.
    pub fn fit_1cpt_oral_its(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        its_config: ItsConfig,
    ) -> Result<FoceResult> {
        let mut cfg = self.config.clone();
        cfg.max_outer_iter = its_config.max_iter.max(1);
        cfg.max_inner_iter = its_config.max_individual_iter.max(1);
        cfg.tol = its_config.tol.max(1e-12);
        cfg.omega_damping = its_config.omega_damping.clamp(0.0, 1.0);
        FoceEstimator::new(cfg).fit_1cpt_oral(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 2-cpt IV ITS wrapper.
    pub fn fit_2cpt_iv_its(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        its_config: ItsConfig,
    ) -> Result<FoceResult> {
        let mut cfg = self.config.clone();
        cfg.max_outer_iter = its_config.max_iter.max(1);
        cfg.max_inner_iter = its_config.max_individual_iter.max(1);
        cfg.tol = its_config.tol.max(1e-12);
        cfg.omega_damping = its_config.omega_damping.clamp(0.0, 1.0);
        FoceEstimator::new(cfg).fit_2cpt_iv(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 2-cpt oral ITS wrapper.
    pub fn fit_2cpt_oral_its(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        its_config: ItsConfig,
    ) -> Result<FoceResult> {
        let mut cfg = self.config.clone();
        cfg.max_outer_iter = its_config.max_iter.max(1);
        cfg.max_inner_iter = its_config.max_individual_iter.max(1);
        cfg.tol = its_config.tol.max(1e-12);
        cfg.omega_damping = its_config.omega_damping.clamp(0.0, 1.0);
        FoceEstimator::new(cfg).fit_2cpt_oral(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 3-cpt IV ITS wrapper.
    pub fn fit_3cpt_iv_its(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        its_config: ItsConfig,
    ) -> Result<FoceResult> {
        let mut cfg = self.config.clone();
        cfg.max_outer_iter = its_config.max_iter.max(1);
        cfg.max_inner_iter = its_config.max_individual_iter.max(1);
        cfg.tol = its_config.tol.max(1e-12);
        cfg.omega_damping = its_config.omega_damping.clamp(0.0, 1.0);
        FoceEstimator::new(cfg).fit_3cpt_iv(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 3-cpt oral ITS wrapper.
    pub fn fit_3cpt_oral_its(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        its_config: ItsConfig,
    ) -> Result<FoceResult> {
        let mut cfg = self.config.clone();
        cfg.max_outer_iter = its_config.max_iter.max(1);
        cfg.max_inner_iter = its_config.max_individual_iter.max(1);
        cfg.tol = its_config.tol.max(1e-12);
        cfg.omega_damping = its_config.omega_damping.clamp(0.0, 1.0);
        FoceEstimator::new(cfg).fit_3cpt_oral(
            times,
            y,
            subject_idx,
            n_subjects,
            doses,
            bioav,
            error_model,
            theta_init,
            omega_init,
        )
    }

    /// 1-cpt IV IMP wrapper.
    pub fn fit_1cpt_iv_imp(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        imp_config: ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ke = cl / v;
            (dose / v) * (-ke * t).exp()
        };
        self.fit_imp_generic(
            times, y, subject_idx, n_subjects, doses, error_model,
            theta_init, om, 2, &conc_fn, &imp_config,
        )
    }

    /// 1-cpt oral IMP wrapper.
    pub fn fit_1cpt_oral_imp(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        imp_config: ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ka = theta[2] * eta[2].exp();
            pk::conc_oral(dose, bioav, cl, v, ka, t)
        };
        self.fit_imp_generic(
            times, y, subject_idx, n_subjects, doses, error_model,
            theta_init, om, 3, &conc_fn, &imp_config,
        )
    }

    /// 2-cpt IV IMP wrapper.
    pub fn fit_2cpt_iv_imp(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        imp_config: ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            pk::conc_iv_2cpt_macro(dose, cl, v1, v2, q, t)
        };
        self.fit_imp_generic(
            times, y, subject_idx, n_subjects, doses, error_model,
            theta_init, om, 4, &conc_fn, &imp_config,
        )
    }

    /// 2-cpt oral IMP wrapper.
    pub fn fit_2cpt_oral_imp(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        imp_config: ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let ka = theta[4] * eta[4].exp();
            pk::conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t)
        };
        self.fit_imp_generic(
            times, y, subject_idx, n_subjects, doses, error_model,
            theta_init, om, 5, &conc_fn, &imp_config,
        )
    }

    /// 3-cpt IV IMP wrapper.
    pub fn fit_3cpt_iv_imp(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        imp_config: ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q1 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q2 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            pk::conc_iv_3cpt_macro(dose, t, cl, v1, q1, v2, q2, v3)
        };
        self.fit_imp_generic(
            times, y, subject_idx, n_subjects, doses, error_model,
            theta_init, om, 6, &conc_fn, &imp_config,
        )
    }

    /// 3-cpt oral IMP wrapper.
    pub fn fit_3cpt_oral_imp(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        bioav: f64,
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: &[f64],
        imp_config: ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        let conc_fn = move |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let effective_dose = dose * bioav;
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            let ka = theta[6] * eta[6].exp();
            pk::conc_oral_3cpt_macro(effective_dose, t, cl, v1, q2, v2, q3, v3, ka)
        };
        self.fit_imp_generic(
            times, y, subject_idx, n_subjects, doses, error_model,
            theta_init, om, 7, &conc_fn, &imp_config,
        )
    }

    // ----- Generic IMP engine ------------------------------------------------

    /// Importance Sampling Monte-Carlo Expectation-Maximization (IMP).
    ///
    /// For each IMP iteration:
    ///   1. **E-step**: Find MAP etas per subject, draw importance samples from
    ///      a multivariate normal proposal centred at the MAP, compute
    ///      unnormalised log-weights and normalise.
    ///   2. **M-step**: Update Omega via weighted empirical covariance of
    ///      samples, update theta + sigma via the existing FOCE outer step
    ///      (using the MAP etas).
    ///   3. **Convergence check** on the FOCE OFV evaluated at MAP etas.
    fn fit_imp_generic(
        &self,
        times: &[f64],
        y: &[f64],
        subject_idx: &[usize],
        n_subjects: usize,
        doses: &[f64],
        error_model: ErrorModel,
        theta_init: &[f64],
        omega_init: OmegaMatrix,
        n_eta: usize,
        conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
        imp_config: &ImpConfig,
    ) -> Result<(FoceResult, ImpDiagnostics)> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let n_iter = imp_config.n_iter.max(1);
        let n_samples = imp_config.n_samples.max(1);
        let scale = imp_config.proposal_scale;
        let tol = imp_config.tol;
        let lloq = self.config.lloq;
        let sigma_init_val = sigma_from_error_model(&error_model);
        let mut error_model = error_model;

        // Group observations by subject: Vec<Vec<(time, obs)>>.
        let n_obs = times.len();
        let mut subj_obs: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_subjects];
        for i in 0..n_obs {
            subj_obs[subject_idx[i]].push((times[i], y[i]));
        }

        // Working state.
        let mut theta = theta_init.to_vec();
        let mut om = omega_init;
        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];
        let mut rng = StdRng::seed_from_u64(imp_config.seed);

        // Diagnostics traces.
        let mut ofv_trace: Vec<f64> = Vec::with_capacity(n_iter);
        let mut ess_frac_trace: Vec<f64> = Vec::with_capacity(n_iter);
        let mut max_w_trace: Vec<f64> = Vec::with_capacity(n_iter);

        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let mut prev_ofv = f64::MAX;
        let mut converged = false;
        let mut total_iter = 0usize;

        for iter in 0..n_iter {
            total_iter = iter + 1;

            // ---- E-step: MAP + importance sampling ----

            // (a) Find MAP etas for each subject.
            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                etas[s] = inner_optimize_eta_generic(
                    &subj_obs[s],
                    &theta,
                    &om,
                    &error_model,
                    &etas[s],
                    self.config.max_inner_iter,
                    doses[s],
                    conc_fn,
                    lloq,
                )?;
            }

            // (b) For each subject, draw importance samples and compute weights.
            // Accumulate weighted outer products for omega M-step.
            let om_chol = om.cholesky();
            // Proposal covariance = scale² * Omega, with Cholesky = scale * L.
            let scaled_chol: Vec<Vec<f64>> = om_chol
                .iter()
                .map(|row| row.iter().map(|v| v * scale).collect::<Vec<f64>>())
                .collect();
            // Precompute log-det of proposal = n_eta * log(scale) + log_det(Omega).
            let log_det_proposal = n_eta as f64 * (scale * scale).ln() + om.log_det();

            // Build the proposal inverse for evaluating proposal density:
            // (scale² * Omega)^{-1}  =>  inv_quadratic using scaled chol.
            // We build a temporary OmegaMatrix for the proposal.
            let proposal_om = if (scale - 1.0).abs() < 1e-12 {
                om.clone()
            } else {
                OmegaMatrix::from_cholesky(scaled_chol.clone())
                    .unwrap_or_else(|_| om.clone())
            };

            let mut iter_ess_sum = 0.0;
            let mut iter_ess_count = 0usize;
            let mut iter_max_weight: f64 = 0.0;

            // Weighted omega accumulator: sum_s sum_m w_{s,m} * eta_m * eta_m^T.
            let mut omega_accum = vec![vec![0.0; n_eta]; n_eta];
            let mut omega_weight_sum = 0.0;

            for s in 0..n_subjects {
                if subj_obs[s].is_empty() {
                    continue;
                }
                let eta_hat = &etas[s];

                // Draw n_samples from proposal N(eta_hat, scale² * Omega).
                let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
                let mut log_weights: Vec<f64> = Vec::with_capacity(n_samples);

                for _ in 0..n_samples {
                    // Sample z ~ N(0, I), then eta_m = eta_hat + scale*L*z.
                    let z: Vec<f64> = (0..n_eta)
                        .map(|_| StandardNormal.sample(&mut rng))
                        .collect();
                    let mut eta_m = eta_hat.clone();
                    for i in 0..n_eta {
                        let mut lz_i = 0.0;
                        for j in 0..=i {
                            lz_i += scaled_chol[i][j] * z[j];
                        }
                        eta_m[i] += lz_i;
                    }

                    // Log-posterior (unnormalised): -inner_objective gives
                    // -(data_nll + 0.5 * eta^T Omega^{-1} eta).
                    let log_posterior = -inner_objective_generic(
                        &subj_obs[s],
                        &theta,
                        &om,
                        &error_model,
                        &eta_m,
                        doses[s],
                        conc_fn,
                        lloq,
                    );

                    // Log-proposal: N(eta_m | eta_hat, scale²*Omega).
                    // = -0.5*(eta_m - eta_hat)^T (scale²*Omega)^{-1} (eta_m - eta_hat)
                    //   -0.5*n_eta*log(2π) - 0.5*log_det_proposal
                    let delta: Vec<f64> = (0..n_eta).map(|k| eta_m[k] - eta_hat[k]).collect();
                    let log_proposal = -0.5 * proposal_om.inv_quadratic(&delta)
                        - 0.5 * n_eta as f64 * log_2pi
                        - 0.5 * log_det_proposal;

                    log_weights.push(log_posterior - log_proposal);
                    samples.push(eta_m);
                }

                // Normalize weights via log-sum-exp for numerical stability.
                let max_lw = log_weights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let lse = max_lw
                    + log_weights
                        .iter()
                        .map(|lw| (lw - max_lw).exp())
                        .sum::<f64>()
                        .ln();
                let weights: Vec<f64> = log_weights.iter().map(|lw| (lw - lse).exp()).collect();

                // ESS and max weight for this subject.
                let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
                let ess = if sum_w2 > 0.0 { 1.0 / sum_w2 } else { 1.0 };
                iter_ess_sum += ess / n_samples as f64;
                iter_ess_count += 1;
                let subj_max_w = weights
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max);
                if subj_max_w > iter_max_weight {
                    iter_max_weight = subj_max_w;
                }

                // Accumulate weighted outer product for omega update.
                if !imp_config.e_only {
                    for (m, eta_m) in samples.iter().enumerate() {
                        let w = weights[m];
                        for i in 0..n_eta {
                            for j in 0..=i {
                                omega_accum[i][j] += w * eta_m[i] * eta_m[j];
                            }
                        }
                        omega_weight_sum += w;
                    }
                }
            }

            // Diagnostics for this iteration.
            let avg_ess_frac = if iter_ess_count > 0 {
                iter_ess_sum / iter_ess_count as f64
            } else {
                1.0
            };

            // ---- M-step ----
            if !imp_config.e_only {
                // Omega from weighted empirical covariance.
                if omega_weight_sum > 0.0 {
                    for i in 0..n_eta {
                        for j in 0..=i {
                            omega_accum[i][j] /= omega_weight_sum;
                            omega_accum[j][i] = omega_accum[i][j];
                        }
                    }
                    // Ridge regularisation + correlation cap (same as OmegaMatrix::empirical).
                    let min_var = 1e-4;
                    for i in 0..n_eta {
                        if !omega_accum[i][i].is_finite() || omega_accum[i][i] < 0.0 {
                            omega_accum[i][i] = om.to_matrix()[i][i];
                        }
                        omega_accum[i][i] += min_var;
                    }
                    for i in 0..n_eta {
                        for j in 0..i {
                            if !omega_accum[i][j].is_finite() {
                                omega_accum[i][j] = 0.0;
                                omega_accum[j][i] = 0.0;
                            } else {
                                let max_abs =
                                    (omega_accum[i][i] * omega_accum[j][j]).sqrt() * 0.999;
                                omega_accum[i][j] = omega_accum[i][j].clamp(-max_abs, max_abs);
                                omega_accum[j][i] = omega_accum[i][j];
                            }
                        }
                    }
                    if let Ok(om_new) = OmegaMatrix::from_covariance(&omega_accum) {
                        // Damp: blend with previous omega (50/50 for stability).
                        let om_damped = damped_omega_blend(
                            &om,
                            &om_new,
                            0.5,
                            &om.to_matrix().iter().enumerate().map(|(i, row)| row[i]).collect::<Vec<f64>>(),
                            100.0,
                        )
                        .unwrap_or(om_new);
                        om = om_damped;
                    }
                }

                // Theta update via FOCE outer step with MAP etas.
                let (theta_new, _om_outer) = outer_step_generic(
                    &subj_obs,
                    &theta,
                    &om,
                    &error_model,
                    &etas,
                    n_eta,
                    doses,
                    conc_fn,
                    lloq,
                )?;
                theta = theta_new;

                // Sigma M-step with damping.
                if self.config.estimate_sigma {
                    let sigma_mle = estimate_sigma_mle_generic(
                        &subj_obs,
                        &theta,
                        &etas,
                        doses,
                        &error_model,
                        conc_fn,
                    );
                    error_model = blend_sigma(&error_model, &sigma_mle, 0.3);
                }
            }

            // ---- OFV at MAP etas ----
            let ofv = compute_foce_ofv_generic(
                &subj_obs,
                &theta,
                &om,
                &error_model,
                &etas,
                n_eta,
                doses,
                conc_fn,
                lloq,
                self.config.interaction,
            )?;

            ofv_trace.push(ofv);
            ess_frac_trace.push(avg_ess_frac);
            max_w_trace.push(iter_max_weight);

            // ---- Convergence check ----
            if iter > 0 {
                let delta = (prev_ofv - ofv).abs();
                let rel_delta = if prev_ofv.abs() > 0.0 {
                    delta / prev_ofv.abs()
                } else {
                    delta
                };
                if delta < tol || rel_delta < 1e-8 {
                    converged = true;
                    prev_ofv = ofv;
                    break;
                }
            }
            prev_ofv = ofv;
        }

        // Build FoceResult.
        let correlation = om.correlation();
        let omega_diag = om.sds();
        let sigma = sigma_from_error_model(&error_model);
        let result = FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv: prev_ofv,
            converged,
            n_iter: total_iter,
            sigma,
            sigma_init: sigma_init_val,
            covariance_step: None,
        };
        let diag = ImpDiagnostics {
            ofv_trace,
            ess_fraction_trace: ess_frac_trace,
            max_weight_trace: max_w_trace,
        };
        Ok((result, diag))
    }
}

// ---------------------------------------------------------------------------
// Covariate model types
// ---------------------------------------------------------------------------

/// Relationship type for covariate effects on structural parameters.
#[derive(Debug, Clone)]
pub enum CovRelationship {
    /// θ_i = θ_pop * (COV / REF)^exponent.
    ///
    /// If `estimate_exponent` is `false`, the exponent is fixed (e.g. 0.75
    /// for allometric CL scaling). Otherwise it is optimized in the outer loop.
    Power {
        /// Allometric exponent (e.g. 0.75 for CL, 1.0 for V).
        exponent: f64,
        /// Whether the exponent is an additional free parameter.
        estimate_exponent: bool,
    },
    /// θ_i = θ_pop * exp(β * (COV - REF)).
    Exponential,
    /// θ_i = θ_pop * (1 + β * (COV - REF)).
    Proportional,
    /// θ_i = θ_pop * exp(β * INDICATOR) for categorical covariates.
    Categorical,
}

/// Covariate model specification for NLME estimators.
///
/// Describes how a single covariate modifies a single structural parameter
/// across subjects.
#[derive(Debug, Clone)]
pub struct CovariateSpec {
    /// Which fixed-effect parameter index this covariate affects (0-based).
    pub param_idx: usize,
    /// Covariate values per subject (length = n_subjects).
    pub values: Vec<f64>,
    /// Reference value for centering (e.g. 70 kg for weight).
    pub reference: f64,
    /// Relationship type.
    pub relationship: CovRelationship,
}

/// Interpolation mode for time-varying covariates.
#[derive(Debug, Clone, Copy)]
pub enum TimeCovariateInterpolation {
    /// Last observation carried forward.
    Locf,
    /// Piecewise-linear interpolation between adjacent points.
    Linear,
}

/// A single time/value point for a time-varying covariate trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TimeVaryingCovariatePoint {
    pub time: f64,
    pub value: f64,
}

/// Time-varying covariate specification for one structural parameter.
#[derive(Debug, Clone)]
pub struct TimeVaryingCovariateSpec {
    /// Which fixed-effect parameter index this covariate affects (0-based).
    pub param_idx: usize,
    /// Per-subject trajectories; len must be `n_subjects`.
    pub trajectories: Vec<Vec<TimeVaryingCovariatePoint>>,
    /// Reference value for centering/scaling.
    pub reference: f64,
    /// Relationship type.
    pub relationship: CovRelationship,
    /// Interpolation mode.
    pub interpolation: TimeCovariateInterpolation,
}

/// Inter-occasion variability (IOV) specification.
#[derive(Debug, Clone)]
pub struct IovSpec {
    /// Structural parameter indices affected by IOV.
    pub param_indices: Vec<usize>,
    /// Per-subject occasion start times (sorted ascending).
    pub occasion_start_times: Vec<Vec<f64>>,
    /// Initial SDs for IOV random effects (same length as `param_indices`).
    pub omega_iov_init: Vec<f64>,
}

fn interpolate_time_varying_value(
    points: &[TimeVaryingCovariatePoint],
    t: f64,
    mode: TimeCovariateInterpolation,
) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    if points.len() == 1 {
        return points[0].value;
    }
    if t <= points[0].time {
        return points[0].value;
    }
    let last = points.len() - 1;
    if t >= points[last].time {
        return points[last].value;
    }

    for i in 1..points.len() {
        let a = points[i - 1];
        let b = points[i];
        if t <= b.time {
            return match mode {
                TimeCovariateInterpolation::Locf => a.value,
                TimeCovariateInterpolation::Linear => {
                    let dt = b.time - a.time;
                    if dt.abs() < 1e-12 {
                        b.value
                    } else {
                        let w = (t - a.time) / dt;
                        a.value + w * (b.value - a.value)
                    }
                }
            };
        }
    }
    points[last].value
}

#[inline]
fn apply_covariate_relationship_value(
    theta_pop: f64,
    x: f64,
    reference: f64,
    relationship: &CovRelationship,
) -> f64 {
    match relationship {
        CovRelationship::Power { exponent, .. } => {
            if reference.abs() < 1e-30 {
                theta_pop
            } else {
                theta_pop * (x / reference).powf(*exponent)
            }
        }
        // Coefficients for these relationships are not estimated in FOCE yet.
        CovRelationship::Exponential => theta_pop,
        CovRelationship::Proportional => theta_pop,
        CovRelationship::Categorical => theta_pop,
    }
}

/// Validate time-varying covariate specs for model dimensions and data integrity.
pub(crate) fn validate_time_varying_covariates(
    covariates: &[TimeVaryingCovariateSpec],
    n_theta: usize,
    n_subjects: usize,
) -> Result<()> {
    for (i, cov) in covariates.iter().enumerate() {
        if cov.param_idx >= n_theta {
            return Err(Error::Validation(format!(
                "time-varying covariate[{i}] param_idx {} out of range for model with {} parameters",
                cov.param_idx, n_theta
            )));
        }
        if cov.trajectories.len() != n_subjects {
            return Err(Error::Validation(format!(
                "time-varying covariate[{i}] trajectories length {} != n_subjects {}",
                cov.trajectories.len(),
                n_subjects
            )));
        }
        if !cov.reference.is_finite() {
            return Err(Error::Validation(format!(
                "time-varying covariate[{i}] reference must be finite"
            )));
        }
        for (sid, traj) in cov.trajectories.iter().enumerate() {
            if traj.is_empty() {
                return Err(Error::Validation(format!(
                    "time-varying covariate[{i}] subject {sid} trajectory must not be empty"
                )));
            }
            let mut prev_t = f64::NEG_INFINITY;
            for (k, p) in traj.iter().enumerate() {
                if !p.time.is_finite() || !p.value.is_finite() {
                    return Err(Error::Validation(format!(
                        "time-varying covariate[{i}] subject {sid} point {k} must be finite"
                    )));
                }
                if p.time < prev_t {
                    return Err(Error::Validation(format!(
                        "time-varying covariate[{i}] subject {sid} trajectory must be sorted by time"
                    )));
                }
                prev_t = p.time;
            }
        }
    }
    Ok(())
}

/// Apply all time-varying covariates at time `t` for a given subject.
pub(crate) fn apply_all_time_varying_covariates(
    theta_pop: &[f64],
    covariates: &[TimeVaryingCovariateSpec],
    subject_idx: usize,
    t: f64,
) -> Vec<f64> {
    let mut theta_s = theta_pop.to_vec();
    for cov in covariates {
        if cov.param_idx >= theta_s.len() || subject_idx >= cov.trajectories.len() {
            continue;
        }
        let x =
            interpolate_time_varying_value(&cov.trajectories[subject_idx], t, cov.interpolation);
        theta_s[cov.param_idx] = apply_covariate_relationship_value(
            theta_s[cov.param_idx],
            x,
            cov.reference,
            &cov.relationship,
        );
    }
    theta_s
}

/// Validate IOV specification for model dimensions and data integrity.
pub(crate) fn validate_iov_spec(
    iov: &IovSpec,
    n_theta: usize,
    n_subjects: usize,
) -> Result<()> {
    if iov.param_indices.is_empty() {
        return Err(Error::Validation("IOV param_indices must not be empty".to_string()));
    }
    if iov.param_indices.len() != iov.omega_iov_init.len() {
        return Err(Error::Validation(format!(
            "IOV omega_iov_init length {} must equal param_indices length {}",
            iov.omega_iov_init.len(),
            iov.param_indices.len()
        )));
    }
    if iov.occasion_start_times.len() != n_subjects {
        return Err(Error::Validation(format!(
            "IOV occasion_start_times length {} != n_subjects {}",
            iov.occasion_start_times.len(),
            n_subjects
        )));
    }
    for (j, &pidx) in iov.param_indices.iter().enumerate() {
        if pidx >= n_theta {
            return Err(Error::Validation(format!(
                "IOV param_indices[{j}]={pidx} out of range for model with {} parameters",
                n_theta
            )));
        }
        let w = iov.omega_iov_init[j];
        if !w.is_finite() || w <= 0.0 {
            return Err(Error::Validation(format!(
                "IOV omega_iov_init[{j}] must be finite and > 0"
            )));
        }
    }
    for (sid, starts) in iov.occasion_start_times.iter().enumerate() {
        if starts.is_empty() {
            return Err(Error::Validation(format!(
                "IOV occasion_start_times[{sid}] must contain at least one occasion start time"
            )));
        }
        let mut prev_t = f64::NEG_INFINITY;
        for (k, &t0) in starts.iter().enumerate() {
            if !t0.is_finite() {
                return Err(Error::Validation(format!(
                    "IOV occasion_start_times[{sid}][{k}] must be finite"
                )));
            }
            if t0 < prev_t {
                return Err(Error::Validation(format!(
                    "IOV occasion_start_times[{sid}] must be sorted ascending"
                )));
            }
            prev_t = t0;
        }
    }
    Ok(())
}

/// Maximum number of occasions across subjects.
pub(crate) fn iov_max_occasions(iov: &IovSpec) -> usize {
    iov.occasion_start_times.iter().map(std::vec::Vec::len).max().unwrap_or(1).max(1)
}

/// Find occasion index for time `t` from sorted occasion start times.
pub(crate) fn occasion_index_for_time(occasion_start_times: &[f64], t: f64) -> usize {
    if occasion_start_times.is_empty() {
        return 0;
    }
    if t <= occasion_start_times[0] {
        return 0;
    }
    let mut occ = 0usize;
    for (idx, &t0) in occasion_start_times.iter().enumerate() {
        if t >= t0 {
            occ = idx;
        } else {
            break;
        }
    }
    occ
}

/// Build block-diagonal omega for IIV + IOV random effects.
///
/// Layout:
/// - first `n_iiv` ETAs: IIV terms
/// - then per-occasion, per-IOV-parameter kappas:
///   `kappa[occ=0,param=0..]`, `kappa[occ=1,param=0..]`, ...
pub(crate) fn build_iov_omega(omega_iiv_init: &[f64], iov: &IovSpec) -> Result<OmegaMatrix> {
    let n_iiv = omega_iiv_init.len();
    let n_iov_params = iov.param_indices.len();
    let n_occ = iov_max_occasions(iov);
    let n_eta = n_iiv + n_occ * n_iov_params;

    let mut sds = vec![0.0; n_eta];
    for (i, &w) in omega_iiv_init.iter().enumerate() {
        if !w.is_finite() || w <= 0.0 {
            return Err(Error::Validation(format!(
                "omega_iiv_init[{i}] must be finite and > 0"
            )));
        }
        sds[i] = w;
    }
    for occ in 0..n_occ {
        for j in 0..n_iov_params {
            sds[n_iiv + occ * n_iov_params + j] = iov.omega_iov_init[j];
        }
    }
    OmegaMatrix::from_diagonal(&sds)
}

/// Apply a single covariate effect to a population parameter value.
///
/// For relationships with an estimated coefficient (Exponential, Proportional,
/// Categorical), the coefficient `beta` is currently set to 0.0 (no effect).
/// A full implementation would add beta as an outer-loop free parameter.
/// For Power with `estimate_exponent: false`, the exponent is applied directly.
pub(crate) fn apply_covariate(theta_pop: f64, cov: &CovariateSpec, subject_idx: usize) -> f64 {
    let x = cov.values[subject_idx];
    match &cov.relationship {
        CovRelationship::Power { exponent, .. } => {
            if cov.reference.abs() < 1e-30 {
                theta_pop
            } else {
                theta_pop * (x / cov.reference).powf(*exponent)
            }
        }
        CovRelationship::Exponential => {
            // β = 0 by default (no effect until outer loop adds it).
            theta_pop
        }
        CovRelationship::Proportional => theta_pop,
        CovRelationship::Categorical => theta_pop,
    }
}

/// Apply all covariates to produce subject-specific theta.
pub(crate) fn apply_all_covariates(
    theta_pop: &[f64],
    covariates: &[CovariateSpec],
    subject_idx: usize,
) -> Vec<f64> {
    let mut theta_s = theta_pop.to_vec();
    for cov in covariates {
        let k = cov.param_idx;
        if k < theta_s.len() {
            theta_s[k] = apply_covariate(theta_s[k], cov, subject_idx);
        }
    }
    theta_s
}

// ---------------------------------------------------------------------------
// Generic FOCE helpers (for N-param models)
// ---------------------------------------------------------------------------

/// Inner objective for one subject using a generic concentration function.
fn inner_objective_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta: &[f64],
    dose: f64,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = conc_fn(theta, eta, dose, t);
        if let Some(l) = lloq
            && yobs < l
        {
            obj += em.nll_censored_below(l, c.max(1e-30));
        } else {
            obj += em.nll_obs(yobs, c.max(1e-30));
        }
    }
    obj += 0.5 * omega.inv_quadratic(eta);
    obj
}

/// Numerical gradient of inner objective w.r.t. eta (generic).
fn inner_gradient_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta: &[f64],
    dose: f64,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> Vec<f64> {
    let n = eta.len();
    let h = 1e-7;
    let mut grad = Vec::with_capacity(n);
    let mut eta_buf = eta.to_vec();
    for k in 0..n {
        let orig = eta_buf[k];
        eta_buf[k] = orig + h;
        let fp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
        eta_buf[k] = orig - h;
        let fm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
        eta_buf[k] = orig;
        grad.push((fp - fm) / (2.0 * h));
    }
    grad
}

/// Numerical Hessian of inner objective w.r.t. eta (generic).
fn inner_hessian_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta: &[f64],
    dose: f64,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> Vec<Vec<f64>> {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_objective_generic(obs, theta, omega, em, eta, dose, conc_fn, lloq);
    let mut hess = vec![vec![0.0; n]; n];
    let mut eta_buf = eta.to_vec();

    for i in 0..n {
        let orig = eta_buf[i];
        eta_buf[i] = orig + h;
        let fp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
        eta_buf[i] = orig - h;
        let fm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
        eta_buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);

        for j in (i + 1)..n {
            let oi = eta_buf[i];
            let oj = eta_buf[j];
            eta_buf[i] = oi + h;
            eta_buf[j] = oj + h;
            let fpp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi + h;
            eta_buf[j] = oj - h;
            let fpm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj + h;
            let fmp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj - h;
            let fmm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi;
            eta_buf[j] = oj;
            let val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }
    hess
}

/// Inner optimization: find conditional mode of eta for one subject (generic).
fn inner_optimize_eta_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta_init: &[f64],
    max_iter: usize,
    dose: f64,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> Result<Vec<f64>> {
    let n = eta_init.len();
    let mut eta = eta_init.to_vec();

    for _ in 0..max_iter {
        let grad = inner_gradient_generic(obs, theta, omega, em, &eta, dose, conc_fn, lloq);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            break;
        }

        let hess = inner_hessian_generic(obs, theta, omega, em, &eta, dose, conc_fn, lloq);
        let delta = solve_linear(&hess, &grad);

        let obj_cur = inner_objective_generic(obs, theta, omega, em, &eta, dose, conc_fn, lloq);
        let mut step = 1.0;
        for _ in 0..5 {
            let eta_trial: Vec<f64> = (0..n).map(|k| eta[k] - step * delta[k]).collect();
            let obj_trial =
                inner_objective_generic(obs, theta, omega, em, &eta_trial, dose, conc_fn, lloq);
            if obj_trial < obj_cur {
                eta = eta_trial;
                break;
            }
            step *= 0.5;
        }
    }
    Ok(eta)
}

/// Inner objective for FOCE Hessian (generic): excludes 0.5·ln(variance).
fn inner_objective_foce_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta: &[f64],
    dose: f64,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    _lloq: Option<f64>,
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = conc_fn(theta, eta, dose, t);
        let f = c.max(1e-30);
        let v = em.variance(f);
        let r = yobs - f;
        obj += 0.5 * r * r / v;
    }
    obj += 0.5 * omega.inv_quadratic(eta);
    obj
}

/// Numerical Hessian for FOCE (generic): uses inner_objective_foce_generic.
fn inner_hessian_foce_generic(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    eta: &[f64],
    dose: f64,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> Vec<Vec<f64>> {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_objective_foce_generic(obs, theta, omega, em, eta, dose, conc_fn, lloq);
    let mut hess = vec![vec![0.0; n]; n];
    let mut eta_buf = eta.to_vec();
    for i in 0..n {
        let orig = eta_buf[i];
        eta_buf[i] = orig + h;
        let fp = inner_objective_foce_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
        eta_buf[i] = orig - h;
        let fm = inner_objective_foce_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
        eta_buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);
        for j in (i + 1)..n {
            let oi = eta_buf[i];
            let oj = eta_buf[j];
            eta_buf[i] = oi + h;
            eta_buf[j] = oj + h;
            let fpp = inner_objective_foce_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi + h;
            eta_buf[j] = oj - h;
            let fpm = inner_objective_foce_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj + h;
            let fmp = inner_objective_foce_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj - h;
            let fmm = inner_objective_foce_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn, lloq);
            eta_buf[i] = oi;
            eta_buf[j] = oj;
            let val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }
    hess
}

/// Compute FOCE OFV using a generic concentration function.
fn compute_foce_ofv_generic(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    etas: &[Vec<f64>],
    n_eta: usize,
    doses: &[f64],
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
    interaction: bool,
) -> Result<f64> {
    let n_subjects = subj_obs.len();
    let mut ofv = 0.0;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let log_det_omega = omega.log_det();

    for s in 0..n_subjects {
        if subj_obs[s].is_empty() {
            continue;
        }
        let eta = &etas[s];
        let nll_s =
            inner_objective_generic(&subj_obs[s], theta, omega, em, eta, doses[s], conc_fn, lloq);
        let hess = if interaction {
            inner_hessian_generic(&subj_obs[s], theta, omega, em, eta, doses[s], conc_fn, lloq)
        } else {
            inner_hessian_foce_generic(&subj_obs[s], theta, omega, em, eta, doses[s], conc_fn, lloq)
        };
        let ld = log_det(&hess);
        let n_obs_i = subj_obs[s].len() as f64;
        let contribution =
            2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega + n_obs_i * log_2pi;
        ofv += contribution;
    }
    Ok(ofv)
}

/// Outer step using a generic concentration function.
fn outer_step_generic(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    etas: &[Vec<f64>],
    n_eta: usize,
    doses: &[f64],
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> Result<(Vec<f64>, OmegaMatrix)> {
    let n_theta = theta.len();

    // NONMEM-grade: optimize Laplace-approximated marginal, not conditional NLL.
    let marginal_ofv = |th: &[f64]| -> f64 {
        if th.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return f64::MAX;
        }
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_omega = omega.log_det();
        let mut ofv = 0.0;
        for (s, obs) in subj_obs.iter().enumerate() {
            if obs.is_empty() {
                continue;
            }
            let nll_s =
                inner_objective_generic(obs, th, omega, em, &etas[s], doses[s], conc_fn, lloq);
            let hess = inner_hessian_generic(obs, th, omega, em, &etas[s], doses[s], conc_fn, lloq);
            let ld = log_det(&hess);
            ofv += 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
        }
        ofv
    };

    let h = 1e-7;
    let nll0 = marginal_ofv(theta);
    let mut grad = vec![0.0; n_theta];
    let mut th_buf = theta.to_vec();
    for j in 0..n_theta {
        let orig = th_buf[j];
        th_buf[j] = orig + h;
        let fp = marginal_ofv(&th_buf);
        th_buf[j] = orig - h;
        let fm = marginal_ofv(&th_buf);
        th_buf[j] = orig;
        grad[j] = (fp - fm) / (2.0 * h);
    }

    let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
    let base_lr = 0.01;
    let mut lr =
        base_lr * theta.iter().map(|v| v.abs()).sum::<f64>() / (n_theta as f64 * grad_norm);

    let mut theta_new = theta.to_vec();
    for _ in 0..15 {
        let trial: Vec<f64> =
            theta.iter().zip(grad.iter()).map(|(&p, &g)| (p - lr * g).max(1e-6)).collect();
        if marginal_ofv(&trial) < nll0 {
            theta_new = trial;
            break;
        }
        lr *= 0.5;
    }

    let om_new =
        omega_corrected_generic(subj_obs, theta, omega, em, etas, n_eta, doses, conc_fn, lloq);

    Ok((theta_new, om_new))
}

// ---------------------------------------------------------------------------
// Covariate-aware FOCE helpers (for fit_with_covariates)
// ---------------------------------------------------------------------------

/// Compute FOCE OFV with covariate-adjusted subject parameters.
fn compute_foce_ofv_with_covariates(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    doses: &[f64],
    bioav: f64,
    etas: &[Vec<f64>],
    n_eta: usize,
    covariates: &[CovariateSpec],
    lloq: Option<f64>,
    interaction: bool,
) -> Result<f64> {
    let n_subjects = subj_obs.len();
    let mut ofv = 0.0;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let log_det_omega = omega.log_det();

    for s in 0..n_subjects {
        if subj_obs[s].is_empty() {
            continue;
        }
        let eta = &etas[s];
        let theta_s = apply_all_covariates(theta, covariates, s);
        let nll_s = inner_objective(&subj_obs[s], &theta_s, omega, em, doses[s], bioav, eta, lloq);
        // FOCE: Hessian excludes ∂²(ln h)/∂η² (no variance interaction).
        // FOCEI: Hessian includes full objective with variance log terms.
        let hess = if interaction {
            inner_hessian(&subj_obs[s], &theta_s, omega, em, doses[s], bioav, eta, lloq)
        } else {
            inner_hessian_foce(&subj_obs[s], &theta_s, omega, em, doses[s], bioav, eta, lloq)
        };
        let ld = log_det(&hess);
        let n_obs_i = subj_obs[s].len() as f64;
        let contribution =
            2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega + n_obs_i * log_2pi;
        ofv += contribution;
    }
    Ok(ofv)
}

/// Outer step with covariate-adjusted predictions.
fn outer_step_with_covariates(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    doses: &[f64],
    bioav: f64,
    etas: &[Vec<f64>],
    n_eta: usize,
    covariates: &[CovariateSpec],
    lloq: Option<f64>,
) -> Result<(Vec<f64>, OmegaMatrix)> {
    let n_theta = theta.len();

    // NONMEM-grade: optimize Laplace-approximated marginal, not conditional NLL.
    let marginal_ofv = |th: &[f64]| -> f64 {
        if th.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return f64::MAX;
        }
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_omega = omega.log_det();
        let mut ofv = 0.0;
        for (s, obs) in subj_obs.iter().enumerate() {
            if obs.is_empty() {
                continue;
            }
            let theta_s = apply_all_covariates(th, covariates, s);
            let nll_s = inner_objective(obs, &theta_s, omega, em, doses[s], bioav, &etas[s], lloq);
            let hess = inner_hessian(obs, &theta_s, omega, em, doses[s], bioav, &etas[s], lloq);
            let ld = log_det(&hess);
            ofv += 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
        }
        ofv
    };

    let h = 1e-7;
    let nll0 = marginal_ofv(theta);
    let mut grad = vec![0.0; n_theta];
    let mut th_buf = theta.to_vec();
    for j in 0..n_theta {
        let orig = th_buf[j];
        th_buf[j] = orig + h;
        let fp = marginal_ofv(&th_buf);
        th_buf[j] = orig - h;
        let fm = marginal_ofv(&th_buf);
        th_buf[j] = orig;
        grad[j] = (fp - fm) / (2.0 * h);
    }

    let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
    let base_lr = 0.01;
    let mut lr =
        base_lr * theta.iter().map(|v| v.abs()).sum::<f64>() / (n_theta as f64 * grad_norm);

    let mut theta_new = theta.to_vec();
    for _ in 0..15 {
        let trial: Vec<f64> =
            theta.iter().zip(grad.iter()).map(|(&p, &g)| (p - lr * g).max(1e-6)).collect();
        if marginal_ofv(&trial) < nll0 {
            theta_new = trial;
            break;
        }
        lr *= 0.5;
    }

    // Hessian-corrected omega M-step (covariate-adjusted).
    let om_new = {
        let mut cov = vec![vec![0.0; n_eta]; n_eta];
        let mut count = 0usize;
        for (s, obs) in subj_obs.iter().enumerate() {
            if obs.is_empty() {
                continue;
            }
            let eta = &etas[s];
            for i in 0..n_eta {
                for j in 0..=i {
                    cov[i][j] += eta[i] * eta[j];
                }
            }
            let theta_s = apply_all_covariates(theta, covariates, s);
            let hess = inner_hessian(obs, &theta_s, omega, em, doses[s], bioav, eta, lloq);
            if let Some(hinv) = invert_matrix(&hess) {
                for i in 0..n_eta {
                    for j in 0..=i {
                        cov[i][j] += hinv[i][j];
                    }
                }
            }
            count += 1;
        }
        if count == 0 {
            omega.clone()
        } else {
            let nf = count as f64;
            let min_var = 1e-4;
            for i in 0..n_eta {
                for j in 0..=i {
                    cov[i][j] /= nf;
                    cov[j][i] = cov[i][j];
                }
            }
            // NaN-safe ridge and correlation cap.
            let mut nan_bail = false;
            for i in 0..n_eta {
                if !cov[i][i].is_finite() || cov[i][i] < 0.0 {
                    nan_bail = true;
                    break;
                }
                cov[i][i] += min_var;
            }
            if nan_bail {
                omega.clone()
            } else {
                for i in 0..n_eta {
                    for j in 0..i {
                        if !cov[i][j].is_finite() {
                            cov[i][j] = 0.0;
                            cov[j][i] = 0.0;
                        } else {
                            let max_abs = (cov[i][i] * cov[j][j]).sqrt() * 0.999;
                            cov[i][j] = cov[i][j].clamp(-max_abs, max_abs);
                            cov[j][i] = cov[i][j];
                        }
                    }
                }
                OmegaMatrix::from_covariance(&cov).unwrap_or_else(|_| omega.clone())
            }
        }
    };

    Ok((theta_new, om_new))
}

// ---------------------------------------------------------------------------
// Original 1-cpt helpers
// ---------------------------------------------------------------------------

/// Extract the primary sigma from an ErrorModel.
pub fn sigma_from_error_model(em: &ErrorModel) -> f64 {
    match *em {
        ErrorModel::Additive(s) | ErrorModel::Proportional(s) | ErrorModel::Exponential(s) => s,
        ErrorModel::Combined { sigma_add, .. } => sigma_add,
        ErrorModel::Power { sigma, .. } => sigma,
    }
}

/// Closed-form MLE for sigma given residuals and the error model type.
///
/// Proportional: `y = f*(1+ε)`, ε~N(0,σ²) ⟹ σ²_MLE = (1/N) Σ ((y-f)/f)²
/// Additive: `y = f + ε`, ε~N(0,σ²) ⟹ σ²_MLE = (1/N) Σ (y-f)²
/// Combined: estimate both components via alternating closed-form.
fn estimate_sigma_mle_1cpt(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    etas: &[Vec<f64>],
    doses: &[f64],
    bioav: f64,
    em: &ErrorModel,
) -> ErrorModel {
    let mut sum = 0.0;
    let mut sum_prop = 0.0;
    let mut n_obs = 0usize;
    for (s, obs) in subj_obs.iter().enumerate() {
        for &(t, yobs) in obs {
            let f = individual_conc(theta, &etas[s], doses[s], bioav, t).max(1e-30);
            let r = yobs - f;
            sum += r * r;
            sum_prop += (r / f) * (r / f);
            n_obs += 1;
        }
    }
    if n_obs == 0 {
        return em.clone();
    }
    let nf = n_obs as f64;
    match em {
        ErrorModel::Additive(_) => {
            let sigma_new = (sum / nf).sqrt().max(1e-6);
            ErrorModel::Additive(sigma_new)
        }
        ErrorModel::Proportional(_) => {
            let sigma_new = (sum_prop / nf).sqrt().max(1e-6);
            ErrorModel::Proportional(sigma_new)
        }
        ErrorModel::Combined { .. } => {
            // Simple approximation: estimate proportional component from
            // weighted residuals, additive from remainder.
            let sigma_prop = (sum_prop / nf).sqrt().max(1e-6);
            let sigma_add = (sum / nf).sqrt().max(1e-6);
            ErrorModel::Combined { sigma_add, sigma_prop }
        }
        _ => em.clone(),
    }
}

/// Closed-form sigma MLE for generic concentration function.
fn estimate_sigma_mle_generic(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    etas: &[Vec<f64>],
    doses: &[f64],
    em: &ErrorModel,
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
) -> ErrorModel {
    let mut sum = 0.0;
    let mut sum_prop = 0.0;
    let mut n_obs = 0usize;
    for (s, obs) in subj_obs.iter().enumerate() {
        for &(t, yobs) in obs {
            let f = conc_fn(theta, &etas[s], doses[s], t).max(1e-30);
            let r = yobs - f;
            sum += r * r;
            sum_prop += (r / f) * (r / f);
            n_obs += 1;
        }
    }
    if n_obs == 0 {
        return em.clone();
    }
    let nf = n_obs as f64;
    match em {
        ErrorModel::Additive(_) => ErrorModel::Additive((sum / nf).sqrt().max(1e-6)),
        ErrorModel::Proportional(_) => ErrorModel::Proportional((sum_prop / nf).sqrt().max(1e-6)),
        ErrorModel::Combined { .. } => ErrorModel::Combined {
            sigma_add: (sum / nf).sqrt().max(1e-6),
            sigma_prop: (sum_prop / nf).sqrt().max(1e-6),
        },
        _ => em.clone(),
    }
}

/// Blend previous and MLE sigma with damping factor `alpha` ∈ [0,1].
/// `sigma_new = (1 - alpha) * old + alpha * mle`.
fn blend_sigma(old: &ErrorModel, mle: &ErrorModel, alpha: f64) -> ErrorModel {
    let lerp = |a: f64, b: f64| (1.0 - alpha) * a + alpha * b;
    match (old, mle) {
        (ErrorModel::Additive(a), ErrorModel::Additive(b)) => {
            ErrorModel::Additive(lerp(*a, *b).max(1e-6))
        }
        (ErrorModel::Proportional(a), ErrorModel::Proportional(b)) => {
            ErrorModel::Proportional(lerp(*a, *b).max(1e-6))
        }
        (
            ErrorModel::Combined {
                sigma_add: a1,
                sigma_prop: p1,
            },
            ErrorModel::Combined {
                sigma_add: a2,
                sigma_prop: p2,
            },
        ) => ErrorModel::Combined {
            sigma_add: lerp(*a1, *a2).max(1e-6),
            sigma_prop: lerp(*p1, *p2).max(1e-6),
        },
        (ErrorModel::Exponential(a), ErrorModel::Exponential(b)) => {
            ErrorModel::Exponential(lerp(*a, *b).max(1e-6))
        }
        _ => mle.clone(),
    }
}

/// Compute individual concentration for 1-cpt oral given population params + eta.
#[inline]
fn individual_conc(theta: &[f64], eta: &[f64], dose: f64, bioav: f64, t: f64) -> f64 {
    let cl = theta[0] * eta[0].exp();
    let v = theta[1] * eta[1].exp();
    let ka = theta[2] * eta[2].exp();
    pk::conc_oral(dose, bioav, cl, v, ka, t)
}

/// Inner objective for one subject: NLL + random effects prior.
/// Prior: 0.5 · ηᵀ Ω⁻¹ η (full covariance).
fn inner_objective(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    lloq: Option<f64>,
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = individual_conc(theta, eta, dose, bioav, t);
        if let Some(l) = lloq
            && yobs < l
        {
            obj += em.nll_censored_below(l, c.max(1e-30));
        } else {
            obj += em.nll_obs(yobs, c.max(1e-30));
        }
    }
    obj += 0.5 * omega.inv_quadratic(eta);
    obj
}

/// Numerical gradient of inner objective w.r.t. eta.
fn inner_gradient(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    lloq: Option<f64>,
) -> Vec<f64> {
    let n = eta.len();
    let h = 1e-7;
    let mut grad = Vec::with_capacity(n);
    let mut eta_buf = eta.to_vec();
    for k in 0..n {
        let orig = eta_buf[k];
        eta_buf[k] = orig + h;
        let fp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
        eta_buf[k] = orig - h;
        let fm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
        eta_buf[k] = orig;
        grad.push((fp - fm) / (2.0 * h));
    }
    grad
}

/// Numerical Hessian of inner objective w.r.t. eta (for Laplace approximation).
fn inner_hessian(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    lloq: Option<f64>,
) -> Vec<Vec<f64>> {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_objective(obs, theta, omega, em, dose, bioav, eta, lloq);
    let mut hess = vec![vec![0.0; n]; n];
    let mut eta_buf = eta.to_vec();

    for i in 0..n {
        // Diagonal: (f(+h) - 2f(0) + f(-h)) / h²
        let orig = eta_buf[i];
        eta_buf[i] = orig + h;
        let fp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
        eta_buf[i] = orig - h;
        let fm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
        eta_buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);

        // Off-diagonal
        for j in (i + 1)..n {
            let oi = eta_buf[i];
            let oj = eta_buf[j];
            eta_buf[i] = oi + h;
            eta_buf[j] = oj + h;
            let fpp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi + h;
            eta_buf[j] = oj - h;
            let fpm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj + h;
            let fmp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj - h;
            let fmm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi;
            eta_buf[j] = oj;
            let val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }
    hess
}

/// Inner objective for FOCE Hessian: excludes the 0.5·ln(variance) term.
///
/// In FOCE (no interaction), the Laplace Hessian should NOT include
/// ∂²(Σ ln h)/∂η².  This function computes only the quadratic residual
/// contribution 0.5·r²/h plus the random-effect prior 0.5·η'Ω⁻¹η.
fn inner_objective_foce(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    _lloq: Option<f64>,
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = individual_conc(theta, eta, dose, bioav, t);
        let f = c.max(1e-30);
        let v = em.variance(f);
        let r = yobs - f;
        obj += 0.5 * r * r / v; // NO + 0.5 * v.ln()
    }
    obj += 0.5 * omega.inv_quadratic(eta);
    obj
}

/// Numerical Hessian for FOCE (no interaction): uses inner_objective_foce
/// which excludes the variance log terms, matching NONMEM's FOCE Laplace.
fn inner_hessian_foce(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta: &[f64],
    lloq: Option<f64>,
) -> Vec<Vec<f64>> {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_objective_foce(obs, theta, omega, em, dose, bioav, eta, lloq);
    let mut hess = vec![vec![0.0; n]; n];
    let mut eta_buf = eta.to_vec();

    for i in 0..n {
        let orig = eta_buf[i];
        eta_buf[i] = orig + h;
        let fp = inner_objective_foce(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
        eta_buf[i] = orig - h;
        let fm = inner_objective_foce(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
        eta_buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);

        for j in (i + 1)..n {
            let oi = eta_buf[i];
            let oj = eta_buf[j];
            eta_buf[i] = oi + h;
            eta_buf[j] = oj + h;
            let fpp = inner_objective_foce(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi + h;
            eta_buf[j] = oj - h;
            let fpm = inner_objective_foce(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj + h;
            let fmp = inner_objective_foce(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj - h;
            let fmm = inner_objective_foce(obs, theta, omega, em, dose, bioav, &eta_buf, lloq);
            eta_buf[i] = oi;
            eta_buf[j] = oj;
            let val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }
    hess
}

/// Log-determinant of a small positive-definite matrix.
/// Adds a small ridge if needed to ensure positive definiteness.
fn log_det(mat: &[Vec<f64>]) -> f64 {
    let n = mat.len();
    if n == 1 {
        return mat[0][0].max(1e-20).ln();
    }
    // Copy with small ridge regularization.
    let mut m = mat.to_vec();
    let ridge = 1e-8;
    for i in 0..n {
        m[i][i] += ridge;
    }
    // Cholesky decomposition.
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = m[i][i] - sum;
                if diag <= 0.0 {
                    // Fall back to product of diagonal.
                    return m.iter().enumerate().map(|(k, row)| row[k].max(1e-20).ln()).sum();
                }
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = (m[i][j] - sum) / l[j][j];
            }
        }
    }
    2.0 * l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>()
}

/// Inner optimization: find conditional mode of eta for one subject
/// using damped Newton-Raphson.
fn inner_optimize_eta(
    obs: &[(f64, f64)],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    dose: f64,
    bioav: f64,
    eta_init: &[f64],
    max_iter: usize,
    lloq: Option<f64>,
) -> Result<Vec<f64>> {
    let n = eta_init.len();
    let mut eta = eta_init.to_vec();

    for _ in 0..max_iter {
        let grad = inner_gradient(obs, theta, omega, em, dose, bioav, &eta, lloq);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            break;
        }

        let hess = inner_hessian(obs, theta, omega, em, dose, bioav, &eta, lloq);

        // Solve H * delta = grad using direct inversion for small n.
        let delta = solve_linear(&hess, &grad);

        // Damped Newton: try step sizes 1, 0.5, 0.25 (simple backtracking).
        let obj_cur = inner_objective(obs, theta, omega, em, dose, bioav, &eta, lloq);
        let mut step = 1.0;
        for _ in 0..5 {
            let eta_trial: Vec<f64> = (0..n).map(|k| eta[k] - step * delta[k]).collect();
            let obj_trial = inner_objective(obs, theta, omega, em, dose, bioav, &eta_trial, lloq);
            if obj_trial < obj_cur {
                eta = eta_trial;
                break;
            }
            step *= 0.5;
        }
    }
    Ok(eta)
}

/// Solve a small linear system Ax = b. Falls back to diagonal solve if singular.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }

    // Gaussian elimination with partial pivoting.
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        // Pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-20 {
            // Near-singular; use diagonal fallback.
            return b
                .iter()
                .enumerate()
                .map(|(i, &bi)| {
                    let aii = a[i][i];
                    if aii.abs() > 1e-20 { bi / aii } else { 0.0 }
                })
                .collect();
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    x
}

/// Compute FOCE OFV = Σ_i [-2 * laplace_i].
fn compute_foce_ofv(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    doses: &[f64],
    bioav: f64,
    etas: &[Vec<f64>],
    n_eta: usize,
    lloq: Option<f64>,
    interaction: bool,
) -> Result<f64> {
    let n_subjects = subj_obs.len();
    let mut ofv = 0.0;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let log_det_omega = omega.log_det();

    for s in 0..n_subjects {
        if subj_obs[s].is_empty() {
            continue;
        }
        let eta = &etas[s];

        // NLL at conditional mode.
        let nll_s = inner_objective(&subj_obs[s], theta, omega, em, doses[s], bioav, eta, lloq);

        // Hessian at conditional mode (for Laplace).
        // FOCE: exclude ∂²(ln h)/∂η²; FOCEI: full Hessian.
        let hess = if interaction {
            inner_hessian(&subj_obs[s], theta, omega, em, doses[s], bioav, eta, lloq)
        } else {
            inner_hessian_foce(&subj_obs[s], theta, omega, em, doses[s], bioav, eta, lloq)
        };
        let ld = log_det(&hess);

        // FOCE contribution for subject s (NONMEM-compatible with obs constant):
        // -2 log L_i ≈ 2 * nll_s + log det(H_s) - n_eta * log(2π) + log det(Ω) + N_obs_i * log(2π)
        let n_obs_i = subj_obs[s].len() as f64;
        let contribution =
            2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega + n_obs_i * log_2pi;
        ofv += contribution;
    }
    Ok(ofv)
}

/// Invert a small matrix using solve_linear (column by column).
/// Returns None if inversion fails or result has non-finite elements,
/// or if any diagonal element is negative (not positive-definite inverse).
fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut rhs = vec![0.0; n];
        rhs[j] = 1.0;
        let col = solve_linear(mat, &rhs);
        if col.iter().any(|v| !v.is_finite()) {
            return None;
        }
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    // Check that diagonal elements are non-negative (posterior covariance).
    for i in 0..n {
        if inv[i][i] < 0.0 || !inv[i][i].is_finite() {
            return None;
        }
    }
    Some(inv)
}

/// Hessian-corrected omega M-step:
/// Ω_new = (1/N) Σ_i (η_i η_i^T + H_i^{-1})
/// where H_i is the Hessian of the inner objective at η_i.
/// Falls back to pure empirical if any Hessian inversion fails.
fn omega_corrected_1cpt(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    etas: &[Vec<f64>],
    n_eta: usize,
    doses: &[f64],
    bioav: f64,
    lloq: Option<f64>,
) -> OmegaMatrix {
    let mut cov = vec![vec![0.0; n_eta]; n_eta];
    let mut count = 0usize;
    for (s, obs) in subj_obs.iter().enumerate() {
        if obs.is_empty() {
            continue;
        }
        let eta = &etas[s];
        // η_i η_i^T
        for i in 0..n_eta {
            for j in 0..=i {
                cov[i][j] += eta[i] * eta[j];
            }
        }
        // + H_i^{-1}
        let hess = inner_hessian(obs, theta, omega, em, doses[s], bioav, eta, lloq);
        if let Some(hinv) = invert_matrix(&hess) {
            for i in 0..n_eta {
                for j in 0..=i {
                    cov[i][j] += hinv[i][j];
                }
            }
        }
        count += 1;
    }
    if count == 0 {
        return omega.clone();
    }
    let nf = count as f64;
    let min_var = 1e-4;
    for i in 0..n_eta {
        for j in 0..=i {
            cov[i][j] /= nf;
            cov[j][i] = cov[i][j];
        }
    }
    // Ridge and correlation cap (NaN-safe).
    for i in 0..n_eta {
        if !cov[i][i].is_finite() || cov[i][i] < 0.0 {
            return omega.clone();
        }
        cov[i][i] += min_var;
    }
    for i in 0..n_eta {
        for j in 0..i {
            if !cov[i][j].is_finite() {
                cov[i][j] = 0.0;
                cov[j][i] = 0.0;
            } else {
                let max_abs = (cov[i][i] * cov[j][j]).sqrt() * 0.999;
                cov[i][j] = cov[i][j].clamp(-max_abs, max_abs);
                cov[j][i] = cov[i][j];
            }
        }
    }
    OmegaMatrix::from_covariance(&cov).unwrap_or_else(|_| omega.clone())
}

/// Hessian-corrected omega for generic concentration function.
fn omega_corrected_generic(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    etas: &[Vec<f64>],
    n_eta: usize,
    doses: &[f64],
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> OmegaMatrix {
    let mut cov = vec![vec![0.0; n_eta]; n_eta];
    let mut count = 0usize;
    for (s, obs) in subj_obs.iter().enumerate() {
        if obs.is_empty() {
            continue;
        }
        let eta = &etas[s];
        for i in 0..n_eta {
            for j in 0..=i {
                cov[i][j] += eta[i] * eta[j];
            }
        }
        let hess = inner_hessian_generic(obs, theta, omega, em, eta, doses[s], conc_fn, lloq);
        if let Some(hinv) = invert_matrix(&hess) {
            for i in 0..n_eta {
                for j in 0..=i {
                    cov[i][j] += hinv[i][j];
                }
            }
        }
        count += 1;
    }
    if count == 0 {
        return omega.clone();
    }
    let nf = count as f64;
    let min_var = 1e-4;
    for i in 0..n_eta {
        for j in 0..=i {
            cov[i][j] /= nf;
            cov[j][i] = cov[i][j];
        }
    }
    // NaN-safe ridge and correlation cap.
    for i in 0..n_eta {
        if !cov[i][i].is_finite() || cov[i][i] < 0.0 {
            return omega.clone();
        }
        cov[i][i] += min_var;
    }
    for i in 0..n_eta {
        for j in 0..i {
            if !cov[i][j].is_finite() {
                cov[i][j] = 0.0;
                cov[j][i] = 0.0;
            } else {
                let max_abs = (cov[i][i] * cov[j][j]).sqrt() * 0.999;
                cov[i][j] = cov[i][j].clamp(-max_abs, max_abs);
                cov[j][i] = cov[i][j];
            }
        }
    }
    OmegaMatrix::from_covariance(&cov).unwrap_or_else(|_| omega.clone())
}

/// Zero out η_k for all subjects where `omega_fixed[k]` is true.
fn apply_omega_fixed_etas(etas: &mut [Vec<f64>], omega_fixed: &[bool]) {
    if omega_fixed.is_empty() {
        return;
    }
    for eta in etas.iter_mut() {
        for (k, &fixed) in omega_fixed.iter().enumerate() {
            if fixed && k < eta.len() {
                eta[k] = 0.0;
            }
        }
    }
}

/// Restore fixed omega entries to their initial values.
/// For each fixed dimension i: Ω_{i,i} = init_{i,i}, Ω_{i,j} = 0 ∀ j≠i.
fn apply_omega_fixed_matrix(om: &mut OmegaMatrix, omega_init_mat: &[Vec<f64>], omega_fixed: &[bool]) {
    if omega_fixed.is_empty() {
        return;
    }
    let n = om.dim();
    let mut cov = om.to_matrix();
    let mut changed = false;
    for i in 0..n.min(omega_fixed.len()) {
        if omega_fixed[i] {
            cov[i][i] = omega_init_mat[i][i];
            for j in 0..n {
                if j != i {
                    cov[i][j] = 0.0;
                    cov[j][i] = 0.0;
                }
            }
            changed = true;
        }
    }
    if changed {
        if let Ok(new_om) = OmegaMatrix::from_covariance(&cov) {
            *om = new_om;
        }
    }
}

/// Zero off-diagonal elements of Ω, forcing a diagonal covariance structure.
fn enforce_diagonal_omega(om: &mut OmegaMatrix) {
    let n = om.dim();
    let cov = om.to_matrix();
    let mut diag = vec![vec![0.0; n]; n];
    for i in 0..n {
        diag[i][i] = cov[i][i];
    }
    if let Ok(new_om) = OmegaMatrix::from_covariance(&diag) {
        *om = new_om;
    }
}

/// Blend previous omega with empirical omega using damping, then cap diagonals.
///
/// `cov_new = damping × cov_prev + (1 − damping) × cov_empirical`
/// Diagonals are capped at `max_ratio × omega_init_vars[i]` to prevent inflation.
fn damped_omega_blend(
    omega_prev: &OmegaMatrix,
    omega_empirical: &OmegaMatrix,
    damping: f64,
    omega_init_vars: &[f64],
    max_ratio: f64,
) -> Result<OmegaMatrix> {
    let n = omega_prev.dim();
    let prev = omega_prev.to_matrix();
    let emp = omega_empirical.to_matrix();

    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            cov[i][j] = damping * prev[i][j] + (1.0 - damping) * emp[i][j];
            cov[j][i] = cov[i][j];
        }
    }

    // Cap diagonals at max_ratio × initial variance, with ridge floor for PD.
    let min_var = 1e-4;
    for i in 0..n {
        let cap = max_ratio * omega_init_vars[i];
        cov[i][i] = cov[i][i].clamp(min_var, cap.max(min_var));
    }

    // Ensure PD: shrink off-diagonals so |correlation| < 0.999.
    for i in 0..n {
        for j in 0..i {
            let max_abs = (cov[i][i] * cov[j][j]).sqrt() * 0.999;
            cov[i][j] = cov[i][j].clamp(-max_abs, max_abs);
            cov[j][i] = cov[i][j];
        }
    }

    OmegaMatrix::from_covariance(&cov)
}

/// Maximum relative change across parameter vectors (for convergence check).
fn max_relative_change(old: &[f64], new: &[f64]) -> f64 {
    old.iter()
        .zip(new.iter())
        .map(|(&o, &n)| ((n - o) / o.abs().max(1e-10)).abs())
        .fold(0.0_f64, f64::max)
}

/// Outer step: EM-like alternation.
/// 1. Update theta by minimizing conditional NLL at fixed etas (gradient descent).
/// 2. Update omega from empirical covariance of etas (full matrix).
fn outer_step(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    doses: &[f64],
    bioav: f64,
    etas: &[Vec<f64>],
    n_eta: usize,
    lloq: Option<f64>,
) -> Result<(Vec<f64>, OmegaMatrix)> {
    let n_theta = theta.len();

    // --- Step 1: Update theta via gradient descent on FULL FOCE marginal ---
    // NONMEM-grade: optimize Laplace-approximated marginal likelihood, not just
    // conditional NLL.  The marginal includes log|H_i(θ)| which depends on θ
    // through the Hessian of the inner objective (crucial for proportional error
    // where log(f) creates bias without this correction).
    let marginal_ofv = |th: &[f64]| -> f64 {
        if th.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return f64::MAX;
        }
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_omega = omega.log_det();
        let mut ofv = 0.0;
        for (s, obs) in subj_obs.iter().enumerate() {
            if obs.is_empty() {
                continue;
            }
            let nll_s = inner_objective(obs, th, omega, em, doses[s], bioav, &etas[s], lloq);
            let hess = inner_hessian(obs, th, omega, em, doses[s], bioav, &etas[s], lloq);
            let ld = log_det(&hess);
            ofv += 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
        }
        ofv
    };

    let h = 1e-7;
    let nll0 = marginal_ofv(theta);
    let mut grad = vec![0.0; n_theta];
    let mut th_buf = theta.to_vec();
    for j in 0..n_theta {
        let orig = th_buf[j];
        th_buf[j] = orig + h;
        let fp = marginal_ofv(&th_buf);
        th_buf[j] = orig - h;
        let fm = marginal_ofv(&th_buf);
        th_buf[j] = orig;
        grad[j] = (fp - fm) / (2.0 * h);
    }

    let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
    let base_lr = 0.01;
    let mut lr =
        base_lr * theta.iter().map(|v| v.abs()).sum::<f64>() / (n_theta as f64 * grad_norm);

    let mut theta_new = theta.to_vec();
    for _ in 0..15 {
        let trial: Vec<f64> =
            theta.iter().zip(grad.iter()).map(|(&p, &g)| (p - lr * g).max(1e-6)).collect();
        if marginal_ofv(&trial) < nll0 {
            theta_new = trial;
            break;
        }
        lr *= 0.5;
    }

    // --- Step 2: Update omega with Hessian correction ---
    let om_new = omega_corrected_1cpt(subj_obs, theta, omega, em, etas, n_eta, doses, bioav, lloq);

    Ok((theta_new, om_new))
}

fn pack_log_params_for_cov(
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
) -> (Vec<f64>, Vec<bool>, Vec<String>, u8) {
    let mut params = Vec::new();
    let mut is_log = Vec::new();
    let mut names = Vec::new();

    for (i, &t) in theta.iter().enumerate() {
        params.push(t.max(1e-30).ln());
        is_log.push(true);
        names.push(format!("theta_{i}"));
    }
    for (i, &w) in omega.sds().iter().enumerate() {
        params.push(w.max(1e-30).ln());
        is_log.push(true);
        names.push(format!("omega_sd_{i}"));
    }

    let em_tag = match *em {
        ErrorModel::Additive(s) => {
            params.push(s.max(1e-30).ln());
            is_log.push(true);
            names.push("sigma_add".to_string());
            0
        }
        ErrorModel::Proportional(s) => {
            params.push(s.max(1e-30).ln());
            is_log.push(true);
            names.push("sigma_prop".to_string());
            1
        }
        ErrorModel::Combined { sigma_add, sigma_prop } => {
            params.push(sigma_add.max(1e-30).ln());
            is_log.push(true);
            names.push("sigma_add".to_string());
            params.push(sigma_prop.max(1e-30).ln());
            is_log.push(true);
            names.push("sigma_prop".to_string());
            2
        }
        ErrorModel::Exponential(s) => {
            params.push(s.max(1e-30).ln());
            is_log.push(true);
            names.push("sigma_exp".to_string());
            3
        }
        ErrorModel::Power { sigma, power } => {
            params.push(sigma.max(1e-30).ln());
            is_log.push(true);
            names.push("sigma_power".to_string());
            params.push(power);
            is_log.push(false);
            names.push("power".to_string());
            4
        }
    };

    (params, is_log, names, em_tag)
}

fn unpack_log_params_for_cov(
    params: &[f64],
    n_theta: usize,
    n_eta: usize,
    em_tag: u8,
) -> Result<(Vec<f64>, OmegaMatrix, ErrorModel, Vec<f64>)> {
    let mut idx = 0usize;
    let theta: Vec<f64> = (0..n_theta)
        .map(|_| {
            let v = params[idx].exp();
            idx += 1;
            v
        })
        .collect();
    let omega_sds: Vec<f64> = (0..n_eta)
        .map(|_| {
            let v = params[idx].exp();
            idx += 1;
            v
        })
        .collect();
    let omega = OmegaMatrix::from_diagonal(&omega_sds)?;
    let em = match em_tag {
        0 => {
            let s = params[idx].exp();
            ErrorModel::Additive(s)
        }
        1 => {
            let s = params[idx].exp();
            ErrorModel::Proportional(s)
        }
        2 => {
            let s_add = params[idx].exp();
            let s_prop = params[idx + 1].exp();
            ErrorModel::Combined { sigma_add: s_add, sigma_prop: s_prop }
        }
        3 => {
            let s = params[idx].exp();
            ErrorModel::Exponential(s)
        }
        4 => {
            let s = params[idx].exp();
            let p = params[idx + 1];
            ErrorModel::Power { sigma: s, power: p }
        }
        _ => {
            return Err(Error::Validation(
                "unknown error-model tag in covariance step".to_string(),
            ));
        }
    };

    let mut natural = Vec::with_capacity(params.len());
    for (i, &p) in params.iter().enumerate() {
        let is_power = em_tag == 4 && i == params.len() - 1;
        natural.push(if is_power { p } else { p.exp() });
    }
    Ok((theta, omega, em, natural))
}

fn numeric_grad_fn(f: &dyn Fn(&[f64]) -> Result<f64>, x: &[f64], eps: f64) -> Result<Vec<f64>> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let fp = f(&xp)?;
        let mut xm = x.to_vec();
        xm[i] -= eps;
        let fm = f(&xm)?;
        g[i] = (fp - fm) / (2.0 * eps);
    }
    Ok(g)
}

fn numeric_hessian_fn(
    f: &dyn Fn(&[f64]) -> Result<f64>,
    x: &[f64],
    eps: f64,
) -> Result<Vec<Vec<f64>>> {
    let n = x.len();
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let gp = numeric_grad_fn(f, &xp, 1e-4)?;
        let mut xm = x.to_vec();
        xm[i] -= eps;
        let gm = numeric_grad_fn(f, &xm, 1e-4)?;
        for j in 0..n {
            h[i][j] = (gp[j] - gm[j]) / (2.0 * eps);
        }
    }
    Ok(h)
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = if b.is_empty() { 0 } else { b[0].len() };
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for k in 0..b.len() {
            let aik = a[i][k];
            for j in 0..m {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn add_outer_product(acc: &mut [Vec<f64>], g: &[f64]) {
    let n = g.len();
    for i in 0..n {
        for j in 0..n {
            acc[i][j] += g[i] * g[j];
        }
    }
}

fn eigen_summary_sym(m: &[Vec<f64>]) -> (Vec<f64>, f64) {
    if m.is_empty() {
        return (vec![], f64::NAN);
    }
    let n = m.len();
    let mut sym = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            sym[i][j] = 0.5 * (m[i][j] + m[j][i]);
        }
    }
    let flat: Vec<f64> = sym.iter().flat_map(|r| r.iter().copied()).collect();
    let dm = DMatrix::from_row_slice(n, n, &flat);
    let eig = dm.symmetric_eigen().eigenvalues;
    let vals: Vec<f64> = eig.iter().copied().collect();
    let max_ev = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let min_pos = vals.iter().copied().filter(|v| *v > 1e-12).fold(f64::INFINITY, f64::min);
    let cond = if min_pos.is_finite() && min_pos > 0.0 { max_ev / min_pos } else { f64::INFINITY };
    (vals, cond)
}

fn compute_covariance_step_from_objectives(
    p0: &[f64],
    is_log_param: &[bool],
    parameter_names: &[String],
    ofv_fn: &dyn Fn(&[f64]) -> Result<f64>,
    subj_fn: &dyn Fn(usize, &[f64]) -> Result<f64>,
    active_subjects: &[usize],
) -> Option<CovarianceStepResult> {
    let n = p0.len();
    if n == 0 {
        return None;
    }
    let r = numeric_hessian_fn(ofv_fn, p0, 1e-3).ok()?;
    let r_inv = invert_matrix(&r)?;
    let mut s = vec![vec![0.0; n]; n];
    for &sid in active_subjects {
        let g = numeric_grad_fn(&|p| subj_fn(sid, p), p0, 1e-4).ok()?;
        add_outer_product(&mut s, &g);
    }
    let robust = mat_mul(&mat_mul(&r_inv, &s), &r_inv);

    let mut se = vec![0.0; n];
    let mut rse_pct = vec![0.0; n];
    for i in 0..n {
        let var_log = r_inv[i][i].max(0.0);
        let se_log = var_log.sqrt();
        let natural = if is_log_param[i] { p0[i].exp() } else { p0[i] };
        let se_nat = if is_log_param[i] { natural.abs() * se_log } else { se_log };
        se[i] = se_nat;
        rse_pct[i] =
            if natural.abs() > 1e-30 { 100.0 * se_nat / natural.abs() } else { f64::INFINITY };
    }
    let (eigs, cond) = eigen_summary_sym(&r);
    Some(CovarianceStepResult {
        parameter_names: parameter_names.to_vec(),
        r_matrix: r,
        s_matrix: s,
        covariance: r_inv,
        robust_covariance: robust,
        se,
        rse_pct,
        r_eigenvalues: eigs,
        r_condition_number: cond,
    })
}

fn compute_covariance_step_1cpt(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    doses: &[f64],
    bioav: f64,
    etas: &[Vec<f64>],
    n_eta: usize,
    lloq: Option<f64>,
) -> Option<CovarianceStepResult> {
    let (p0, is_log, names, em_tag) = pack_log_params_for_cov(theta, omega, em);
    let n_theta = theta.len();
    let active_subjects: Vec<usize> = subj_obs
        .iter()
        .enumerate()
        .filter_map(|(s, obs)| if obs.is_empty() { None } else { Some(s) })
        .collect();

    let ofv_fn = |p: &[f64]| -> Result<f64> {
        let (th, om, em_i, _nat) = unpack_log_params_for_cov(p, n_theta, n_eta, em_tag)?;
        compute_foce_ofv(subj_obs, &th, &om, &em_i, doses, bioav, etas, n_eta, lloq, true)
    };
    let subj_fn = |sid: usize, p: &[f64]| -> Result<f64> {
        let (th, om, em_i, _nat) = unpack_log_params_for_cov(p, n_theta, n_eta, em_tag)?;
        let obs = &subj_obs[sid];
        let eta = &etas[sid];
        let nll_s = inner_objective(obs, &th, &om, &em_i, doses[sid], bioav, eta, lloq);
        let hess = inner_hessian(obs, &th, &om, &em_i, doses[sid], bioav, eta, lloq);
        let ld = log_det(&hess);
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_omega = om.log_det();
        let n_obs_i = obs.len() as f64;
        Ok(2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega + n_obs_i * log_2pi)
    };

    compute_covariance_step_from_objectives(
        &p0,
        &is_log,
        &names,
        &ofv_fn,
        &subj_fn,
        &active_subjects,
    )
}

pub(crate) fn compute_covariance_step_generic(
    subj_obs: &[Vec<(f64, f64)>],
    theta: &[f64],
    omega: &OmegaMatrix,
    em: &ErrorModel,
    etas: &[Vec<f64>],
    n_eta: usize,
    doses: &[f64],
    conc_fn: &dyn Fn(&[f64], &[f64], f64, f64) -> f64,
    lloq: Option<f64>,
) -> Option<CovarianceStepResult> {
    let (p0, is_log, names, em_tag) = pack_log_params_for_cov(theta, omega, em);
    let n_theta = theta.len();
    let active_subjects: Vec<usize> = subj_obs
        .iter()
        .enumerate()
        .filter_map(|(s, obs)| if obs.is_empty() { None } else { Some(s) })
        .collect();

    let ofv_fn = |p: &[f64]| -> Result<f64> {
        let (th, om, em_i, _nat) = unpack_log_params_for_cov(p, n_theta, n_eta, em_tag)?;
        compute_foce_ofv_generic(subj_obs, &th, &om, &em_i, etas, n_eta, doses, conc_fn, lloq, true)
    };
    let subj_fn = |sid: usize, p: &[f64]| -> Result<f64> {
        let (th, om, em_i, _nat) = unpack_log_params_for_cov(p, n_theta, n_eta, em_tag)?;
        let obs = &subj_obs[sid];
        let eta = &etas[sid];
        let nll_s = inner_objective_generic(obs, &th, &om, &em_i, eta, doses[sid], conc_fn, lloq);
        let hess = inner_hessian_generic(obs, &th, &om, &em_i, eta, doses[sid], conc_fn, lloq);
        let ld = log_det(&hess);
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_omega = om.log_det();
        let n_obs_i = obs.len() as f64;
        Ok(2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega + n_obs_i * log_2pi)
    };

    compute_covariance_step_from_objectives(
        &p0,
        &is_log,
        &names,
        &ofv_fn,
        &subj_fn,
        &active_subjects,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};

    #[test]
    fn foce_config_default() {
        let cfg = FoceConfig::default();
        assert_eq!(cfg.max_outer_iter, 100);
        assert_eq!(cfg.rel_tol, 1e-8);
        assert!(cfg.interaction);
        assert!(cfg.estimate_sigma);
    }

    #[test]
    fn omega_diagonal_roundtrip() {
        let om = OmegaMatrix::from_diagonal(&[0.3, 0.2, 0.4]).unwrap();
        let m = om.to_matrix();
        assert!((m[0][0] - 0.09).abs() < 1e-12);
        assert!((m[1][1] - 0.04).abs() < 1e-12);
        assert!((m[0][1]).abs() < 1e-12);
        let sds = om.sds();
        assert!((sds[0] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn omega_correlation_roundtrip() {
        let corr = vec![vec![1.0, 0.5, 0.0], vec![0.5, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let om = OmegaMatrix::from_correlation(&[0.3, 0.2, 0.4], &corr).unwrap();
        let c = om.correlation();
        assert!((c[0][1] - 0.5).abs() < 1e-10, "corr[0][1] = {}", c[0][1]);
        assert!((c[1][0] - 0.5).abs() < 1e-10);
        assert!((c[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn omega_inv_quadratic_diagonal() {
        let om = OmegaMatrix::from_diagonal(&[2.0, 3.0]).unwrap();
        let eta = [1.0, 1.5];
        // ηᵀ Ω⁻¹ η = 1²/4 + 1.5²/9 = 0.25 + 0.25 = 0.5
        let q = om.inv_quadratic(&eta);
        assert!((q - 0.5).abs() < 1e-10, "inv_quad = {q}");
    }

    #[test]
    fn omega_log_det_diagonal() {
        let om = OmegaMatrix::from_diagonal(&[2.0, 3.0]).unwrap();
        // det(Ω) = 4 * 9 = 36
        let ld = om.log_det();
        assert!((ld - 36.0_f64.ln()).abs() < 1e-10, "log_det = {ld}");
    }

    #[test]
    fn omega_empirical_recovers_diagonal() {
        let etas = vec![
            vec![0.3, -0.2, 0.1],
            vec![-0.1, 0.3, -0.2],
            vec![0.2, 0.1, 0.15],
            vec![-0.15, -0.1, 0.05],
        ];
        let om = OmegaMatrix::empirical(&etas, 3).unwrap();
        let m = om.to_matrix();
        // Diagonal should be mean(eta_k^2) + ridge (1e-4).
        let ridge = 1e-4;
        let mut expected_diag = [0.0; 3];
        for eta in &etas {
            for k in 0..3 {
                expected_diag[k] += eta[k] * eta[k];
            }
        }
        for k in 0..3 {
            expected_diag[k] = expected_diag[k] / 4.0 + ridge;
        }
        for k in 0..3 {
            assert!(
                (m[k][k] - expected_diag[k]).abs() < 1e-10,
                "diag[{k}]: {} vs {}",
                m[k][k],
                expected_diag[k]
            );
        }
    }

    #[test]
    fn inner_optimize_eta_basic() {
        let cl_pop = 1.2_f64;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;

        let theta = [cl_pop, v_pop, ka_pop];
        let om = OmegaMatrix::from_diagonal(&[0.25, 0.20, 0.30]).unwrap();
        let em = ErrorModel::Additive(sigma);

        // Generate observations for one subject with known eta.
        let eta_true: [f64; 3] = [0.1, -0.05, 0.08];
        let cl_i = cl_pop * eta_true[0].exp();
        let v_i = v_pop * eta_true[1].exp();
        let ka_i = ka_pop * eta_true[2].exp();

        let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let noise = RandNormal::new(0.0, sigma).unwrap();
        let obs: Vec<(f64, f64)> = times
            .iter()
            .map(|&t| {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                (t, (c + noise.sample(&mut rng)).max(0.0))
            })
            .collect();

        let eta_hat =
            inner_optimize_eta(&obs, &theta, &om, &em, dose, bioav, &[0.0, 0.0, 0.0], 20, None)
                .unwrap();

        for k in 0..3 {
            assert!(eta_hat[k].is_finite(), "eta_hat[{k}] not finite: {}", eta_hat[k]);
        }
        assert!(
            (eta_hat[0] - eta_true[0]).abs() < 0.3,
            "eta_cl recovery: hat={}, true={}",
            eta_hat[0],
            eta_true[0]
        );
    }

    #[test]
    fn foce_ofv_is_finite() {
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let theta = [1.2_f64, 15.0, 2.0];
        let om = OmegaMatrix::from_diagonal(&[0.25, 0.20, 0.30]).unwrap();
        let em = ErrorModel::Additive(sigma);

        let obs = vec![
            vec![(0.5, 2.0), (1.0, 3.5), (2.0, 2.8), (4.0, 1.2)],
            vec![(0.5, 1.8), (1.0, 3.2), (2.0, 2.6), (4.0, 1.0)],
        ];
        let etas = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

        let doses = vec![dose; obs.len()];
        let ofv = compute_foce_ofv(&obs, &theta, &om, &em, &doses, bioav, &etas, 3, None, true).unwrap();
        assert!(ofv.is_finite(), "OFV should be finite: {ofv}");
    }

    #[test]
    fn focei_fit_1cpt_oral_smoke() {
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let omega_cl = 0.25;
        let omega_v = 0.20;
        let omega_ka = 0.30;
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let n_subjects = 6;

        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let eta_cl_dist = RandNormal::new(0.0, omega_cl).unwrap();
        let eta_v_dist = RandNormal::new(0.0, omega_v).unwrap();
        let eta_ka_dist = RandNormal::new(0.0, omega_ka).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = eta_cl_dist.sample(&mut rng);
            let eta_v: f64 = eta_v_dist.sample(&mut rng);
            let eta_ka: f64 = eta_ka_dist.sample(&mut rng);
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

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let result = estimator
            .fit_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.5],
                &[0.3, 0.3, 0.3],
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 3);
        assert_eq!(result.omega.len(), 3);
        assert_eq!(result.eta.len(), n_subjects);
        // correlation matrix should be 3×3 with 1s on diagonal
        assert_eq!(result.correlation.len(), 3);
        assert!((result.correlation[0][0] - 1.0).abs() < 1e-10);

        assert!(
            result.theta[0] > 0.0 && result.theta[0].is_finite(),
            "CL_pop invalid: {}",
            result.theta[0]
        );
        assert!(
            (result.theta[0] - cl_pop).abs() / cl_pop < 0.5,
            "CL_pop: hat={}, true={cl_pop}",
            result.theta[0]
        );
    }

    #[test]
    fn focei_correlated_omega_smoke() {
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let n_subjects = 10;

        // True Ω with CL–V correlation of 0.6.
        let corr = vec![vec![1.0, 0.6, 0.0], vec![0.6, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let true_sds = [0.25, 0.20, 0.30];
        let true_omega = OmegaMatrix::from_correlation(&true_sds, &corr).unwrap();

        // Sample correlated etas via Cholesky: η = L · z, z ~ N(0, I)
        let l = true_omega.cholesky();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let std_normal = RandNormal::new(0.0_f64, 1.0).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

        for sid in 0..n_subjects {
            let z: Vec<f64> = (0..3).map(|_| std_normal.sample(&mut rng)).collect();
            let mut eta = [0.0; 3];
            for i in 0..3 {
                for j in 0..=i {
                    eta[i] += l[i][j] * z[j];
                }
            }
            let cl_i = cl_pop * eta[0].exp();
            let v_i = v_pop * eta[1].exp();
            let ka_i = ka_pop * eta[2].exp();

            for &t in &times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        // Fit with correlated omega.
        let init_omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let result = estimator
            .fit_1cpt_oral_correlated(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.5],
                init_omega,
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.correlation.len(), 3);
        // The recovered CL–V correlation should be positive (true = 0.6).
        // With 10 subjects this is noisy, so just check sign and finiteness.
        assert!(
            result.correlation[0][1].is_finite(),
            "corr[CL,V] not finite: {}",
            result.correlation[0][1]
        );
    }

    #[test]
    fn log_det_identity() {
        let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ld = log_det(&mat);
        assert!(ld.abs() < 1e-6, "log det(I) ≈ 0, got {ld}");
    }

    #[test]
    fn log_det_diagonal() {
        let mat = vec![vec![4.0, 0.0, 0.0], vec![0.0, 9.0, 0.0], vec![0.0, 0.0, 16.0]];
        let ld = log_det(&mat);
        let expected = (4.0_f64 * 9.0 * 16.0).ln();
        assert!((ld - expected).abs() < 1e-4, "log det: {ld} vs {expected}");
    }

    #[test]
    fn solve_linear_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 7.0];
        let x = solve_linear(&a, &b);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 2-compartment IV tests
    // -----------------------------------------------------------------------

    #[test]
    fn foce_2cpt_iv_convergence() {
        // True 2-cpt IV parameters: [CL, V1, Q, V2]
        let cl_pop = 2.0;
        let v1_pop = 10.0;
        let q_pop = 1.5;
        let v2_pop = 20.0;
        let omega_sds = [0.20, 0.15, 0.20, 0.15];
        let sigma = 0.1;
        let dose = 100.0;
        let n_subjects = 10;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta: Vec<f64> = omega_sds
                .iter()
                .map(|&sd| {
                    let d = RandNormal::new(0.0, sd).unwrap();
                    d.sample(&mut rng)
                })
                .collect();

            let cl_i = cl_pop * eta[0].exp();
            let v1_i = v1_pop * eta[1].exp();
            let q_i = q_pop * eta[2].exp();
            let v2_i = v2_pop * eta[3].exp();

            for &t in &times_per {
                let c = pk::conc_iv_2cpt_macro(dose, cl_i, v1_i, v2_i, q_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.01));
                subject_idx.push(sid);
            }
        }

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let result = estimator
            .fit_2cpt_iv(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                ErrorModel::Additive(sigma),
                &[1.5, 8.0, 1.0, 15.0],
                &[0.3, 0.3, 0.3, 0.3],
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 4);
        assert_eq!(result.omega.len(), 4);
        assert_eq!(result.eta.len(), n_subjects);
        // CL recovery within 100% relative error (small sample size).
        assert!(
            (result.theta[0] - cl_pop).abs() / cl_pop < 1.0,
            "CL_pop: hat={}, true={cl_pop}",
            result.theta[0]
        );
    }

    // -----------------------------------------------------------------------
    // 2-compartment oral tests
    // -----------------------------------------------------------------------

    #[test]
    fn foce_2cpt_oral_convergence() {
        // True 2-cpt oral parameters: [CL, V1, Q, V2, Ka]
        let cl_pop = 2.0;
        let v1_pop = 10.0;
        let q_pop = 1.5;
        let v2_pop = 20.0;
        let ka_pop = 1.0;
        let omega_sds = [0.20, 0.15, 0.20, 0.15, 0.25];
        let sigma = 0.1;
        let dose = 100.0;
        let bioav = 0.8;
        let n_subjects = 10;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta: Vec<f64> = omega_sds
                .iter()
                .map(|&sd| {
                    let d = RandNormal::new(0.0, sd).unwrap();
                    d.sample(&mut rng)
                })
                .collect();

            let cl_i = cl_pop * eta[0].exp();
            let v1_i = v1_pop * eta[1].exp();
            let q_i = q_pop * eta[2].exp();
            let v2_i = v2_pop * eta[3].exp();
            let ka_i = ka_pop * eta[4].exp();

            for &t in &times_per {
                let c = pk::conc_oral_2cpt_macro(dose, bioav, cl_i, v1_i, v2_i, q_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.01));
                subject_idx.push(sid);
            }
        }

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let result = estimator
            .fit_2cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.5, 8.0, 1.0, 15.0, 0.8],
                &[0.3, 0.3, 0.3, 0.3, 0.3],
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 5);
        assert_eq!(result.omega.len(), 5);
        assert_eq!(result.eta.len(), n_subjects);
        assert!(
            (result.theta[0] - cl_pop).abs() / cl_pop < 1.0,
            "CL_pop: hat={}, true={cl_pop}",
            result.theta[0]
        );
    }

    // -----------------------------------------------------------------------
    // Allometric scaling (covariate) tests
    // -----------------------------------------------------------------------

    #[test]
    fn foce_allometric_scaling() {
        // Simulate 1-cpt oral with allometric scaling on CL and V:
        //   CL_i = CL_pop * (WT_i/70)^0.75 * exp(eta_CL)
        //   V_i  = V_pop  * (WT_i/70)^1.00 * exp(eta_V)
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let omega_sds = [0.25, 0.20, 0.30];
        let sigma = 0.05;
        let dose = 100.0;
        let bioav = 1.0;
        let n_subjects = 12;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        let noise = RandNormal::new(0.0, sigma).unwrap();

        // Generate subject weights between 50 and 100 kg.
        let weights: Vec<f64> =
            (0..n_subjects).map(|i| 50.0 + 50.0 * (i as f64) / (n_subjects as f64 - 1.0)).collect();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = RandNormal::new(0.0, omega_sds[0]).unwrap().sample(&mut rng);
            let eta_v: f64 = RandNormal::new(0.0, omega_sds[1]).unwrap().sample(&mut rng);
            let eta_ka: f64 = RandNormal::new(0.0, omega_sds[2]).unwrap().sample(&mut rng);

            let wt = weights[sid];
            let cl_i = cl_pop * (wt / 70.0_f64).powf(0.75) * eta_cl.exp();
            let v_i = v_pop * (wt / 70.0_f64).powf(1.00) * eta_v.exp();
            let ka_i = ka_pop * eta_ka.exp();

            for &t in &times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        let covariates = vec![
            CovariateSpec {
                param_idx: 0, // CL
                values: weights.clone(),
                reference: 70.0,
                relationship: CovRelationship::Power { exponent: 0.75, estimate_exponent: false },
            },
            CovariateSpec {
                param_idx: 1, // V
                values: weights.clone(),
                reference: 70.0,
                relationship: CovRelationship::Power { exponent: 1.0, estimate_exponent: false },
            },
        ];

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let result = estimator
            .fit_1cpt_oral_with_covariates(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.5],
                &[0.3, 0.3, 0.3],
                &covariates,
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 3);
        // CL_pop should be recoverable near truth with allometric adjustment.
        assert!(
            result.theta[0] > 0.0 && result.theta[0].is_finite(),
            "CL_pop invalid: {}",
            result.theta[0]
        );
    }

    #[test]
    fn foce_covariance_step_present_for_1cpt_oral() {
        let cl_pop = 1.2;
        let v_pop = 15.0;
        let ka_pop = 2.0;
        let sigma = 0.06;
        let dose = 100.0;
        let bioav = 1.0;
        let n_subjects = 8;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(31415);
        let noise = RandNormal::new(0.0, sigma).unwrap();
        let eta_sd = [0.20, 0.20, 0.20];

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        for sid in 0..n_subjects {
            let eta_cl: f64 = RandNormal::new(0.0, eta_sd[0]).unwrap().sample(&mut rng);
            let eta_v: f64 = RandNormal::new(0.0, eta_sd[1]).unwrap().sample(&mut rng);
            let eta_ka: f64 = RandNormal::new(0.0, eta_sd[2]).unwrap().sample(&mut rng);

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

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let init_omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let result = estimator
            .fit_1cpt_oral_correlated(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.2],
                init_omega,
            )
            .unwrap();

        let cov = result.covariance_step.expect("covariance step should be present for 1-cpt oral");
        assert_eq!(cov.parameter_names.len(), cov.se.len());
        assert!(cov.se.len() >= 4);
        assert!(cov.se.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(cov.rse_pct.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(cov.r_condition_number.is_finite());
    }

    #[test]
    fn foce_1cpt_iv_regimens_infusion_smoke() {
        let cl_pop = 1.1;
        let v_pop = 14.0;
        let sigma = 0.07;
        let n_subjects = 6usize;
        let dose = 80.0;
        let interval = 12.0;
        let n_doses = 2usize;
        let infusion_duration = 1.0;
        let times_per = [0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 13.0, 16.0, 24.0];

        let regimens: Vec<DosingRegimen> = (0..n_subjects)
            .map(|_| {
                DosingRegimen::repeated(
                    dose,
                    interval,
                    n_doses,
                    crate::dosing::DoseRoute::Infusion { duration: infusion_duration },
                )
                .unwrap()
            })
            .collect();

        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let noise = RandNormal::new(0.0, sigma).unwrap();
        let eta_sd = [0.20, 0.20];

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        for sid in 0..n_subjects {
            let eta_cl: f64 = RandNormal::new(0.0, eta_sd[0]).unwrap().sample(&mut rng);
            let eta_v: f64 = RandNormal::new(0.0, eta_sd[1]).unwrap().sample(&mut rng);
            let cl_i = cl_pop * eta_cl.exp();
            let v_i = v_pop * eta_v.exp();

            for &t in &times_per {
                let c = regimens[sid].conc_1cpt(cl_i, v_i, 1.0, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        let estimator = FoceEstimator::focei();
        let init_omega = OmegaMatrix::from_diagonal(&[0.3, 0.3]).unwrap();
        let result = estimator
            .fit_1cpt_iv_with_regimens_correlated(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &regimens,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0],
                init_omega,
            )
            .unwrap();

        assert!(result.ofv.is_finite());
        assert_eq!(result.theta.len(), 2);
        assert_eq!(result.eta.len(), n_subjects);
    }

    #[test]
    fn foce_lloq_censored_smoke() {
        let cl_pop = 1.1;
        let v_pop = 14.0;
        let ka_pop = 1.8;
        let sigma = 0.08;
        let dose = 80.0;
        let bioav = 1.0;
        let n_subjects = 6usize;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        for sid in 0..n_subjects {
            let eta_cl: f64 = RandNormal::new(0.0, 0.2).unwrap().sample(&mut rng);
            let eta_v: f64 = RandNormal::new(0.0, 0.2).unwrap().sample(&mut rng);
            let eta_ka: f64 = RandNormal::new(0.0, 0.2).unwrap().sample(&mut rng);

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

        let cfg = FoceConfig { lloq: Some(0.8), max_outer_iter: 60, ..Default::default() };
        let estimator = FoceEstimator::new(cfg);
        let doses = vec![dose; n_subjects];
        let result = estimator
            .fit_1cpt_oral(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[1.0, 10.0, 1.5],
                &[0.3, 0.3, 0.3],
            )
            .unwrap();

        assert!(result.ofv.is_finite(), "OFV should be finite with LLOQ censoring");
    }

    #[test]
    fn foce_logit_normal_random_effects_smoke() {
        let cl_pop = 1.1;
        let v_pop = 14.0;
        let ka_pop = 1.8;
        let sigma = 0.07;
        let dose = 90.0;
        let bioav = 1.0;
        let n_subjects = 8usize;
        let times_per = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        let ka_bounds = (0.4, 3.0);

        let transforms = [
            RandomEffectTransform::LogNormal,
            RandomEffectTransform::LogNormal,
            RandomEffectTransform::LogitNormal { lower: ka_bounds.0, upper: ka_bounds.1 },
        ];

        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        for sid in 0..n_subjects {
            let eta_cl: f64 = RandNormal::new(0.0, 0.2).unwrap().sample(&mut rng);
            let eta_v: f64 = RandNormal::new(0.0, 0.2).unwrap().sample(&mut rng);
            let eta_ka: f64 = RandNormal::new(0.0, 0.25).unwrap().sample(&mut rng);

            let cl_i = apply_random_effect_transform(cl_pop, eta_cl, transforms[0]);
            let v_i = apply_random_effect_transform(v_pop, eta_v, transforms[1]);
            let ka_i = apply_random_effect_transform(ka_pop, eta_ka, transforms[2]);
            assert!(ka_i > ka_bounds.0 && ka_i < ka_bounds.1);

            for &t in &times_per {
                let c = pk::conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
                times.push(t);
                y.push((c + noise.sample(&mut rng)).max(0.0));
                subject_idx.push(sid);
            }
        }

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let om = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
        let result = estimator
            .fit_1cpt_oral_with_transforms_correlated(
                &times,
                &y,
                &subject_idx,
                n_subjects,
                &doses,
                bioav,
                ErrorModel::Additive(sigma),
                &[cl_pop, v_pop, ka_pop],
                om,
                &transforms,
            )
            .unwrap();

        assert!(result.ofv.is_finite());
        assert_eq!(result.theta.len(), 3);
        assert!(result.theta[2] > ka_bounds.0 && result.theta[2] < ka_bounds.1);
    }
}
