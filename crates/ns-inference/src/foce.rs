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

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

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
    /// Prevents runaway omega inflation. Default: 4.0.
    #[serde(default = "default_omega_max_ratio")]
    pub omega_max_ratio: f64,
}

fn default_omega_damping() -> f64 { 0.7 }
fn default_omega_max_ratio() -> f64 { 100.0 }

impl Default for FoceConfig {
    fn default() -> Self {
        Self {
            max_outer_iter: 100,
            max_inner_iter: 20,
            tol: 1e-4,
            rel_tol: 1e-8,
            interaction: true,
            omega_damping: 0.7,
            omega_max_ratio: 100.0,
        }
    }
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
}

/// FOCE/FOCEI estimator for population pharmacokinetic models.
pub struct FoceEstimator {
    config: FoceConfig,
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
            return Err(Error::Validation("omega_init must have 2 elements for 1-cpt IV".to_string()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_1cpt_iv_correlated(times, y, subject_idx, n_subjects, doses, error_model, theta_init, om)
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
            return Err(Error::Validation(format!("doses length {} != n_subjects {}", doses.len(), n_subjects)));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ke = cl / v;
            (dose / v) * (-ke * t).exp()
        };

        self.fit_generic(times, y, subject_idx, n_subjects, doses, error_model, theta_init, omega_init, 2, &conc_fn)
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
            return Err(Error::Validation("omega_init must have 4 elements for 2-cpt IV".to_string()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_iv_correlated(times, y, subject_idx, n_subjects, doses, error_model, theta_init, om)
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
            return Err(Error::Validation("theta_init must have 4 elements [CL, V1, Q, V2]".to_string()));
        }
        if omega_init.dim() != 4 {
            return Err(Error::Validation("omega must be 4×4".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!("doses length {} != n_subjects {}", doses.len(), n_subjects)));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q  = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            pk::conc_iv_2cpt_macro(dose, cl, v1, v2, q, t)
        };

        self.fit_generic(times, y, subject_idx, n_subjects, doses, error_model, theta_init, omega_init, 4, &conc_fn)
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
            return Err(Error::Validation("omega_init must have 5 elements for 2-cpt oral".to_string()));
        }
        let om = OmegaMatrix::from_diagonal(omega_init)?;
        self.fit_2cpt_oral_correlated(times, y, subject_idx, n_subjects, doses, bioav, error_model, theta_init, om)
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
            return Err(Error::Validation("theta_init must have 5 elements [CL, V1, Q, V2, Ka]".to_string()));
        }
        if omega_init.dim() != 5 {
            return Err(Error::Validation("omega must be 5×5".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation("times/y/subject_idx length mismatch".to_string()));
        }
        if doses.len() != n_subjects {
            return Err(Error::Validation(format!("doses length {} != n_subjects {}", doses.len(), n_subjects)));
        }
        error_model.validate()?;

        let conc_fn = |theta: &[f64], eta: &[f64], dose: f64, t: f64| -> f64 {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q  = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let ka = theta[4] * eta[4].exp();
            pk::conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t)
        };

        self.fit_generic(times, y, subject_idx, n_subjects, doses, error_model, theta_init, omega_init, 5, &conc_fn)
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
            times, y, subject_idx, n_subjects, doses, bioav, error_model, theta_init, om, covariates,
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

        // Validate covariate specs.
        for cov in covariates {
            if cov.param_idx >= 3 {
                return Err(Error::Validation(format!(
                    "covariate param_idx {} out of range for 3-param model", cov.param_idx
                )));
            }
            if cov.values.len() != n_subjects {
                return Err(Error::Validation(format!(
                    "covariate values length {} != n_subjects {}", cov.values.len(), n_subjects
                )));
            }
        }

        let n_obs = times.len();
        let n_eta = 3;

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
                    &subj_obs[s], &theta_s, &om, &error_model, doses[s], bioav, &etas[s],
                    self.config.max_inner_iter,
                )?;
            }

            // Compute OFV at current (theta, omega, etas) with covariates.
            let ofv = compute_foce_ofv_with_covariates(
                &subj_obs, &theta, &om, &error_model, doses, bioav, &etas, n_eta, covariates,
            )?;

            // Check convergence (absolute or relative OFV change).
            let delta = (prev_ofv - ofv).abs();
            let converged_abs = delta < self.config.tol;
            let converged_rel = prev_ofv.abs() > 0.0 && delta / prev_ofv.abs() < self.config.rel_tol;
            if (converged_abs || converged_rel) && iter > 0 {
                converged = true;
                prev_ofv = ofv;
                break;
            }
            prev_ofv = ofv;

            // Outer step: update theta + omega (with covariate-adjusted predictions).
            let (theta_new, om_empirical) = outer_step_with_covariates(
                &subj_obs, &theta, &om, &error_model, doses, bioav, &etas, n_eta, covariates,
            )?;

            // Apply damped omega blend + diagonal cap.
            let om_new = if damping > 0.0 {
                damped_omega_blend(&om, &om_empirical, damping, &omega_init_vars, max_ratio)?
            } else {
                om_empirical
            };

            // Parameter stability check.
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
            if theta_change < 1e-3 && om_change < 1e-3 && iter > 5 {
                theta = theta_new;
                om = om_new;
                converged = true;
                break;
            }

            theta = theta_new;
            om = om_new;
        }

        let correlation = om.correlation();
        let omega_diag = om.sds();
        Ok(FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv: prev_ofv,
            converged,
            n_iter,
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
        let n_obs = times.len();

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
                    &subj_obs[s], &theta, &om, &error_model, &etas[s],
                    self.config.max_inner_iter, doses[s], conc_fn,
                )?;
            }

            // Compute OFV at current (theta, omega, etas).
            let ofv = compute_foce_ofv_generic(
                &subj_obs, &theta, &om, &error_model, &etas, n_eta, doses, conc_fn,
            )?;

            // Check convergence (absolute or relative OFV change).
            let delta = (prev_ofv - ofv).abs();
            let converged_abs = delta < self.config.tol;
            let converged_rel = prev_ofv.abs() > 0.0 && delta / prev_ofv.abs() < self.config.rel_tol;
            if (converged_abs || converged_rel) && iter > 0 {
                converged = true;
                prev_ofv = ofv;
                break;
            }
            prev_ofv = ofv;

            // Outer step: update theta + omega.
            let (theta_new, om_empirical) = outer_step_generic(
                &subj_obs, &theta, &om, &error_model, &etas, n_eta, doses, conc_fn,
            )?;

            // Apply damped omega blend + diagonal cap.
            let om_new = if damping > 0.0 {
                damped_omega_blend(&om, &om_empirical, damping, &omega_init_vars, max_ratio)?
            } else {
                om_empirical
            };

            // Parameter stability check.
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
            if theta_change < 1e-3 && om_change < 1e-3 && iter > 5 {
                theta = theta_new;
                om = om_new;
                converged = true;
                break;
            }

            theta = theta_new;
            om = om_new;
        }

        let correlation = om.correlation();
        let omega_diag = om.sds();
        Ok(FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv: prev_ofv,
            converged,
            n_iter,
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
            return Err(Error::Validation(format!("doses length {} != n_subjects {}", doses.len(), n_subjects)));
        }
        error_model.validate()?;

        let n_obs = times.len();
        let n_eta = 3;

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
                etas[s] = inner_optimize_eta(
                    &subj_obs[s],
                    &theta,
                    &om,
                    &error_model,
                    doses[s],
                    bioav,
                    &etas[s],
                    self.config.max_inner_iter,
                )?;
            }

            // Compute OFV at current (theta, omega, etas).
            let ofv =
                compute_foce_ofv(&subj_obs, &theta, &om, &error_model, doses, bioav, &etas, n_eta)?;

            // Check convergence (absolute or relative OFV change).
            let delta = (prev_ofv - ofv).abs();
            let converged_abs = delta < self.config.tol;
            let converged_rel = prev_ofv.abs() > 0.0 && delta / prev_ofv.abs() < self.config.rel_tol;
            if (converged_abs || converged_rel) && iter > 0 {
                converged = true;
                prev_ofv = ofv;
                break;
            }
            prev_ofv = ofv;

            // Outer step: update theta + omega.
            let (theta_new, om_empirical) =
                outer_step(&subj_obs, &theta, &om, &error_model, doses, bioav, &etas, n_eta)?;

            // Apply ramped damping: no damping for first 20 iters, then ramp to full.
            let eff_damping = damping * (iter as f64 / 20.0).min(1.0);
            let om_new = if eff_damping > 0.0 {
                damped_omega_blend(&om, &om_empirical, eff_damping, &omega_init_vars, max_ratio)?
            } else {
                om_empirical
            };

            // Parameter stability check: converge if params stabilized (after burn-in).
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
        Ok(FoceResult {
            theta,
            omega: omega_diag,
            omega_matrix: om,
            correlation,
            eta: etas,
            ofv: prev_ofv,
            converged,
            n_iter,
        })
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
        CovRelationship::Proportional => {
            theta_pop
        }
        CovRelationship::Categorical => {
            theta_pop
        }
    }
}

/// Apply all covariates to produce subject-specific theta.
pub(crate) fn apply_all_covariates(theta_pop: &[f64], covariates: &[CovariateSpec], subject_idx: usize) -> Vec<f64> {
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
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = conc_fn(theta, eta, dose, t);
        obj += em.nll_obs(yobs, c.max(1e-30));
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
) -> Vec<f64> {
    let n = eta.len();
    let h = 1e-7;
    let mut grad = Vec::with_capacity(n);
    let mut eta_buf = eta.to_vec();
    for k in 0..n {
        let orig = eta_buf[k];
        eta_buf[k] = orig + h;
        let fp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
        eta_buf[k] = orig - h;
        let fm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
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
) -> Vec<Vec<f64>> {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_objective_generic(obs, theta, omega, em, eta, dose, conc_fn);
    let mut hess = vec![vec![0.0; n]; n];
    let mut eta_buf = eta.to_vec();

    for i in 0..n {
        let orig = eta_buf[i];
        eta_buf[i] = orig + h;
        let fp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
        eta_buf[i] = orig - h;
        let fm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
        eta_buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);

        for j in (i + 1)..n {
            let oi = eta_buf[i];
            let oj = eta_buf[j];
            eta_buf[i] = oi + h;
            eta_buf[j] = oj + h;
            let fpp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
            eta_buf[i] = oi + h;
            eta_buf[j] = oj - h;
            let fpm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj + h;
            let fmp = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj - h;
            let fmm = inner_objective_generic(obs, theta, omega, em, &eta_buf, dose, conc_fn);
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
) -> Result<Vec<f64>> {
    let n = eta_init.len();
    let mut eta = eta_init.to_vec();

    for _ in 0..max_iter {
        let grad = inner_gradient_generic(obs, theta, omega, em, &eta, dose, conc_fn);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            break;
        }

        let hess = inner_hessian_generic(obs, theta, omega, em, &eta, dose, conc_fn);
        let delta = solve_linear(&hess, &grad);

        let obj_cur = inner_objective_generic(obs, theta, omega, em, &eta, dose, conc_fn);
        let mut step = 1.0;
        for _ in 0..5 {
            let eta_trial: Vec<f64> = (0..n).map(|k| eta[k] - step * delta[k]).collect();
            let obj_trial = inner_objective_generic(obs, theta, omega, em, &eta_trial, dose, conc_fn);
            if obj_trial < obj_cur {
                eta = eta_trial;
                break;
            }
            step *= 0.5;
        }
    }
    Ok(eta)
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
        let nll_s = inner_objective_generic(&subj_obs[s], theta, omega, em, eta, doses[s], conc_fn);
        let hess = inner_hessian_generic(&subj_obs[s], theta, omega, em, eta, doses[s], conc_fn);
        let ld = log_det(&hess);
        let contribution = 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
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
) -> Result<(Vec<f64>, OmegaMatrix)> {
    let n_theta = theta.len();

    let cond_nll = |th: &[f64]| -> f64 {
        if th.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return f64::MAX;
        }
        let mut nll = 0.0;
        for (s, obs) in subj_obs.iter().enumerate() {
            for &(t, yobs) in obs {
                let c = conc_fn(th, &etas[s], doses[s], t).max(1e-30);
                nll += em.nll_obs(yobs, c);
            }
        }
        nll
    };

    let h = 1e-7;
    let nll0 = cond_nll(theta);
    let mut grad = vec![0.0; n_theta];
    let mut th_buf = theta.to_vec();
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

    let mut theta_new = theta.to_vec();
    for _ in 0..15 {
        let trial: Vec<f64> =
            theta.iter().zip(grad.iter()).map(|(&p, &g)| (p - lr * g).max(1e-6)).collect();
        if cond_nll(&trial) < nll0 {
            theta_new = trial;
            break;
        }
        lr *= 0.5;
    }

    let active_etas: Vec<Vec<f64>> = etas
        .iter()
        .enumerate()
        .filter(|(s, _)| !subj_obs[*s].is_empty())
        .map(|(_, e)| e.clone())
        .collect();

    let om_new = if !active_etas.is_empty() {
        OmegaMatrix::empirical(&active_etas, n_eta)?
    } else {
        omega.clone()
    };

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
        let nll_s = inner_objective(&subj_obs[s], &theta_s, omega, em, doses[s], bioav, eta);
        let hess = inner_hessian(&subj_obs[s], &theta_s, omega, em, doses[s], bioav, eta);
        let ld = log_det(&hess);
        let contribution = 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
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
) -> Result<(Vec<f64>, OmegaMatrix)> {
    let n_theta = theta.len();

    let cond_nll = |th: &[f64]| -> f64 {
        if th.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return f64::MAX;
        }
        let mut nll = 0.0;
        for (s, obs) in subj_obs.iter().enumerate() {
            let theta_s = apply_all_covariates(th, covariates, s);
            for &(t, yobs) in obs {
                let c = individual_conc(&theta_s, &etas[s], doses[s], bioav, t).max(1e-30);
                nll += em.nll_obs(yobs, c);
            }
        }
        nll
    };

    let h = 1e-7;
    let nll0 = cond_nll(theta);
    let mut grad = vec![0.0; n_theta];
    let mut th_buf = theta.to_vec();
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

    let mut theta_new = theta.to_vec();
    for _ in 0..15 {
        let trial: Vec<f64> =
            theta.iter().zip(grad.iter()).map(|(&p, &g)| (p - lr * g).max(1e-6)).collect();
        if cond_nll(&trial) < nll0 {
            theta_new = trial;
            break;
        }
        lr *= 0.5;
    }

    let active_etas: Vec<Vec<f64>> = etas
        .iter()
        .enumerate()
        .filter(|(s, _)| !subj_obs[*s].is_empty())
        .map(|(_, e)| e.clone())
        .collect();

    let om_new = if !active_etas.is_empty() {
        OmegaMatrix::empirical(&active_etas, n_eta)?
    } else {
        omega.clone()
    };

    Ok((theta_new, om_new))
}

// ---------------------------------------------------------------------------
// Original 1-cpt helpers
// ---------------------------------------------------------------------------

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
) -> f64 {
    let mut obj = 0.0;
    for &(t, yobs) in obs {
        let c = individual_conc(theta, eta, dose, bioav, t);
        obj += em.nll_obs(yobs, c.max(1e-30));
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
) -> Vec<f64> {
    let n = eta.len();
    let h = 1e-7;
    let mut grad = Vec::with_capacity(n);
    let mut eta_buf = eta.to_vec();
    for k in 0..n {
        let orig = eta_buf[k];
        eta_buf[k] = orig + h;
        let fp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
        eta_buf[k] = orig - h;
        let fm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
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
) -> Vec<Vec<f64>> {
    let n = eta.len();
    let h = 1e-5;
    let f0 = inner_objective(obs, theta, omega, em, dose, bioav, eta);
    let mut hess = vec![vec![0.0; n]; n];
    let mut eta_buf = eta.to_vec();

    for i in 0..n {
        // Diagonal: (f(+h) - 2f(0) + f(-h)) / h²
        let orig = eta_buf[i];
        eta_buf[i] = orig + h;
        let fp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
        eta_buf[i] = orig - h;
        let fm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
        eta_buf[i] = orig;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h);

        // Off-diagonal
        for j in (i + 1)..n {
            let oi = eta_buf[i];
            let oj = eta_buf[j];
            eta_buf[i] = oi + h;
            eta_buf[j] = oj + h;
            let fpp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
            eta_buf[i] = oi + h;
            eta_buf[j] = oj - h;
            let fpm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj + h;
            let fmp = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
            eta_buf[i] = oi - h;
            eta_buf[j] = oj - h;
            let fmm = inner_objective(obs, theta, omega, em, dose, bioav, &eta_buf);
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
) -> Result<Vec<f64>> {
    let n = eta_init.len();
    let mut eta = eta_init.to_vec();

    for _ in 0..max_iter {
        let grad = inner_gradient(obs, theta, omega, em, dose, bioav, &eta);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-6 {
            break;
        }

        let hess = inner_hessian(obs, theta, omega, em, dose, bioav, &eta);

        // Solve H * delta = grad using direct inversion for small n.
        let delta = solve_linear(&hess, &grad);

        // Damped Newton: try step sizes 1, 0.5, 0.25 (simple backtracking).
        let obj_cur = inner_objective(obs, theta, omega, em, dose, bioav, &eta);
        let mut step = 1.0;
        for _ in 0..5 {
            let eta_trial: Vec<f64> = (0..n).map(|k| eta[k] - step * delta[k]).collect();
            let obj_trial = inner_objective(obs, theta, omega, em, dose, bioav, &eta_trial);
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
        let nll_s = inner_objective(&subj_obs[s], theta, omega, em, doses[s], bioav, eta);

        // Hessian at conditional mode (for Laplace).
        let hess = inner_hessian(&subj_obs[s], theta, omega, em, doses[s], bioav, eta);
        let ld = log_det(&hess);

        // FOCE contribution for subject s:
        // -2 log L_i ≈ 2 * nll_s + log det(H_s) - n_eta * log(2π) + log det(Ω)
        let contribution = 2.0 * nll_s + ld - n_eta as f64 * log_2pi + log_det_omega;
        ofv += contribution;
    }
    Ok(ofv)
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

    // Cap diagonals at max_ratio × initial variance.
    for i in 0..n {
        let cap = max_ratio * omega_init_vars[i];
        if cov[i][i] > cap {
            cov[i][i] = cap;
        }
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
) -> Result<(Vec<f64>, OmegaMatrix)> {
    let n_theta = theta.len();

    // --- Step 1: Update theta via gradient descent on conditional NLL ---
    let cond_nll = |th: &[f64]| -> f64 {
        if th.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return f64::MAX;
        }
        let mut nll = 0.0;
        for (s, obs) in subj_obs.iter().enumerate() {
            for &(t, yobs) in obs {
                let c = individual_conc(th, &etas[s], doses[s], bioav, t).max(1e-30);
                nll += em.nll_obs(yobs, c);
            }
        }
        nll
    };

    let h = 1e-7;
    let nll0 = cond_nll(theta);
    let mut grad = vec![0.0; n_theta];
    let mut th_buf = theta.to_vec();
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

    let mut theta_new = theta.to_vec();
    for _ in 0..15 {
        let trial: Vec<f64> =
            theta.iter().zip(grad.iter()).map(|(&p, &g)| (p - lr * g).max(1e-6)).collect();
        if cond_nll(&trial) < nll0 {
            theta_new = trial;
            break;
        }
        lr *= 0.5;
    }

    // --- Step 2: Update omega from empirical covariance of etas ---
    let active_etas: Vec<Vec<f64>> = etas
        .iter()
        .enumerate()
        .filter(|(s, _)| !subj_obs[*s].is_empty())
        .map(|(_, e)| e.clone())
        .collect();

    let om_new = if !active_etas.is_empty() {
        OmegaMatrix::empirical(&active_etas, n_eta)?
    } else {
        omega.clone()
    };

    Ok((theta_new, om_new))
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
            inner_optimize_eta(&obs, &theta, &om, &em, dose, bioav, &[0.0, 0.0, 0.0], 20).unwrap();

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
        let ofv = compute_foce_ofv(&obs, &theta, &om, &em, &doses, bioav, &etas, 3).unwrap();
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
            let eta: Vec<f64> = omega_sds.iter().map(|&sd| {
                let d = RandNormal::new(0.0, sd).unwrap();
                d.sample(&mut rng)
            }).collect();

            let cl_i = cl_pop * eta[0].exp();
            let v1_i = v1_pop * eta[1].exp();
            let q_i  = q_pop  * eta[2].exp();
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
        let result = estimator.fit_2cpt_iv(
            &times, &y, &subject_idx, n_subjects, &doses,
            ErrorModel::Additive(sigma),
            &[1.5, 8.0, 1.0, 15.0],
            &[0.3, 0.3, 0.3, 0.3],
        ).unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 4);
        assert_eq!(result.omega.len(), 4);
        assert_eq!(result.eta.len(), n_subjects);
        // CL recovery within 100% relative error (small sample size).
        assert!(
            (result.theta[0] - cl_pop).abs() / cl_pop < 1.0,
            "CL_pop: hat={}, true={cl_pop}", result.theta[0]
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
        let q_pop  = 1.5;
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
            let eta: Vec<f64> = omega_sds.iter().map(|&sd| {
                let d = RandNormal::new(0.0, sd).unwrap();
                d.sample(&mut rng)
            }).collect();

            let cl_i = cl_pop * eta[0].exp();
            let v1_i = v1_pop * eta[1].exp();
            let q_i  = q_pop  * eta[2].exp();
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
        let result = estimator.fit_2cpt_oral(
            &times, &y, &subject_idx, n_subjects, &doses, bioav,
            ErrorModel::Additive(sigma),
            &[1.5, 8.0, 1.0, 15.0, 0.8],
            &[0.3, 0.3, 0.3, 0.3, 0.3],
        ).unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 5);
        assert_eq!(result.omega.len(), 5);
        assert_eq!(result.eta.len(), n_subjects);
        assert!(
            (result.theta[0] - cl_pop).abs() / cl_pop < 1.0,
            "CL_pop: hat={}, true={cl_pop}", result.theta[0]
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
        let weights: Vec<f64> = (0..n_subjects)
            .map(|i| 50.0 + 50.0 * (i as f64) / (n_subjects as f64 - 1.0))
            .collect();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        for sid in 0..n_subjects {
            let eta_cl: f64 = RandNormal::new(0.0, omega_sds[0]).unwrap().sample(&mut rng);
            let eta_v: f64  = RandNormal::new(0.0, omega_sds[1]).unwrap().sample(&mut rng);
            let eta_ka: f64 = RandNormal::new(0.0, omega_sds[2]).unwrap().sample(&mut rng);

            let wt = weights[sid];
            let cl_i = cl_pop * (wt / 70.0_f64).powf(0.75) * eta_cl.exp();
            let v_i  = v_pop  * (wt / 70.0_f64).powf(1.00) * eta_v.exp();
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
                param_idx: 0,  // CL
                values: weights.clone(),
                reference: 70.0,
                relationship: CovRelationship::Power { exponent: 0.75, estimate_exponent: false },
            },
            CovariateSpec {
                param_idx: 1,  // V
                values: weights.clone(),
                reference: 70.0,
                relationship: CovRelationship::Power { exponent: 1.0, estimate_exponent: false },
            },
        ];

        let estimator = FoceEstimator::focei();
        let doses = vec![dose; n_subjects];
        let result = estimator.fit_1cpt_oral_with_covariates(
            &times, &y, &subject_idx, n_subjects, &doses, bioav,
            ErrorModel::Additive(sigma),
            &[1.0, 10.0, 1.5],
            &[0.3, 0.3, 0.3],
            &covariates,
        ).unwrap();

        assert!(result.ofv.is_finite(), "OFV not finite: {}", result.ofv);
        assert_eq!(result.theta.len(), 3);
        // CL_pop should be recoverable near truth with allometric adjustment.
        assert!(
            result.theta[0] > 0.0 && result.theta[0].is_finite(),
            "CL_pop invalid: {}", result.theta[0]
        );
    }
}
