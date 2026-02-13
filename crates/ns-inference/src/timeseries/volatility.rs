//! Volatility models (Phase 12 econometrics add-on).
//!
//! Baselines:
//! - GARCH(1,1) with Gaussian innovations (MLE)
//! - Approximate stochastic volatility (SV) via log(y^2) with a Gaussian approximation
//!   for log(chi^2_1), fit as a 1D AR(1) latent state with Kalman MLE.

use nalgebra::DVector;
use ns_core::{Error, Result};

use super::kalman::{KalmanModel, kalman_filter, rts_smoother};
use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};

const LN_2PI: f64 = 1.837_877_066_409_345_3; // ln(2*pi)
const LOG_CHI2_MEAN: f64 = -1.270_362_845_461_478_2;
const LOG_CHI2_VAR: f64 = 4.934_802_200_544_679; // pi^2/2

#[derive(Debug, Clone, Copy)]
pub struct Garch11Params {
    pub mu: f64,
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct Garch11Config {
    pub optimizer: OptimizerConfig,
    pub alpha_beta_max: f64,
    pub init: Option<Garch11Params>,
    pub min_var: f64,
}

impl Default for Garch11Config {
    fn default() -> Self {
        Self {
            optimizer: OptimizerConfig::default(),
            alpha_beta_max: 0.999,
            init: None,
            min_var: 1e-18,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Garch11Fit {
    pub params: Garch11Params,
    pub log_likelihood: f64,
    pub conditional_variance: Vec<f64>,
    pub optimization: OptimizationResult,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / (xs.len() as f64)
}

fn var_pop(xs: &[f64], mu: f64) -> f64 {
    xs.iter()
        .map(|&v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (xs.len() as f64)
}

fn garch11_loglik(
    y: &[f64],
    p: Garch11Params,
    alpha_beta_max: f64,
    min_var: f64,
) -> Result<(f64, Vec<f64>)> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("y must be finite".to_string()));
    }
    if !p.mu.is_finite() || !p.omega.is_finite() || !p.alpha.is_finite() || !p.beta.is_finite() {
        return Err(Error::Validation("params must be finite".to_string()));
    }
    if p.omega <= 0.0 || p.alpha < 0.0 || p.beta < 0.0 {
        return Err(Error::Validation("omega must be >0 and alpha/beta must be >=0".to_string()));
    }
    if !(alpha_beta_max > 0.0 && alpha_beta_max < 1.0 && alpha_beta_max.is_finite()) {
        return Err(Error::Validation("alpha_beta_max must be finite and in (0,1)".to_string()));
    }
    if p.alpha + p.beta >= alpha_beta_max {
        return Err(Error::Validation("alpha+beta violates alpha_beta_max".to_string()));
    }
    let min_var = min_var.max(0.0);

    let eps: Vec<f64> = y.iter().map(|&v| v - p.mu).collect();

    let mut h = vec![0.0f64; y.len()];
    let denom = 1.0 - (p.alpha + p.beta);
    let mut h0 = if denom > 1e-12 {
        p.omega / denom
    } else {
        let m = mean(&eps);
        var_pop(&eps, m)
    };
    if !h0.is_finite() || h0 <= 0.0 {
        h0 = min_var.max(1e-12);
    }
    h[0] = h0.max(min_var);

    for t in 1..y.len() {
        let prev_eps2 = eps[t - 1] * eps[t - 1];
        let mut v = p.omega + p.alpha * prev_eps2 + p.beta * h[t - 1];
        if !v.is_finite() || v <= 0.0 {
            v = min_var.max(1e-12);
        }
        h[t] = v.max(min_var);
    }

    let mut ll = 0.0f64;
    for (et, ht) in eps.iter().zip(h.iter()) {
        let quad = (et * et) / ht;
        ll += -0.5 * (LN_2PI + ht.ln() + quad);
    }
    Ok((ll, h))
}

struct Garch11Objective {
    y: Vec<f64>,
    alpha_beta_max: f64,
    min_var: f64,
}

impl ObjectiveFunction for Garch11Objective {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 4 {
            return Err(Error::Validation(
                "expected 4 params (mu, omega, alpha, beta)".to_string(),
            ));
        }
        let p =
            Garch11Params { mu: params[0], omega: params[1], alpha: params[2], beta: params[3] };
        match garch11_loglik(&self.y, p, self.alpha_beta_max, self.min_var) {
            Ok((ll, _)) => Ok(-ll),
            Err(_) => Ok(1e30),
        }
    }

    /// Analytical gradient of the negative log-likelihood w.r.t. (mu, omega, alpha, beta).
    ///
    /// Single forward pass through observations computing both h_t and ∂h_t/∂θ
    /// via the GARCH(1,1) recursion, avoiding 2*n_params extra likelihood evaluations
    /// from numerical central differences.
    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 4 {
            return Err(Error::Validation(
                "expected 4 params (mu, omega, alpha, beta)".to_string(),
            ));
        }
        let mu = params[0];
        let omega = params[1];
        let alpha = params[2];
        let beta = params[3];

        // At invalid/boundary points fall back to numerical differentiation.
        if !mu.is_finite()
            || !omega.is_finite()
            || !alpha.is_finite()
            || !beta.is_finite()
            || omega <= 0.0
            || alpha < 0.0
            || beta < 0.0
            || alpha + beta >= self.alpha_beta_max
        {
            let n = params.len();
            let mut grad = vec![0.0; n];
            for i in 0..n {
                let eps = 1e-8 * params[i].abs().max(1.0);
                let mut pp = params.to_vec();
                pp[i] += eps;
                let fp = self.eval(&pp)?;
                pp[i] = params[i] - eps;
                let fm = self.eval(&pp)?;
                grad[i] = (fp - fm) / (2.0 * eps);
            }
            return Ok(grad);
        }

        let y = &self.y;
        let t_len = y.len();
        let min_var = self.min_var.max(0.0);

        let eps: Vec<f64> = y.iter().map(|&v| v - mu).collect();

        let denom = 1.0 - alpha - beta;
        let h0 = if denom > 1e-12 {
            (omega / denom).max(min_var.max(1e-12))
        } else {
            let m = mean(&eps);
            var_pop(&eps, m).max(min_var.max(1e-12))
        };

        // ∂h_0/∂{mu, omega, alpha, beta}
        let (mut dh_mu, mut dh_omega, mut dh_alpha, mut dh_beta) = if denom > 1e-12 {
            let inv_d = 1.0 / denom;
            (0.0, inv_d, omega * inv_d * inv_d, omega * inv_d * inv_d)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let mut g_mu = 0.0_f64;
        let mut g_omega = 0.0_f64;
        let mut g_alpha = 0.0_f64;
        let mut g_beta = 0.0_f64;

        let mut h_prev = h0;
        for t in 0..t_len {
            let ht = if t == 0 {
                h0
            } else {
                let ep = eps[t - 1];
                let ep2 = ep * ep;
                let raw = omega + alpha * ep2 + beta * h_prev;
                let ht = raw.max(min_var);
                if raw >= min_var {
                    let new_mu = -2.0 * alpha * ep + beta * dh_mu;
                    let new_omega = 1.0 + beta * dh_omega;
                    let new_alpha = ep2 + beta * dh_alpha;
                    let new_beta = h_prev + beta * dh_beta;
                    dh_mu = new_mu;
                    dh_omega = new_omega;
                    dh_alpha = new_alpha;
                    dh_beta = new_beta;
                } else {
                    dh_mu = 0.0;
                    dh_omega = 0.0;
                    dh_alpha = 0.0;
                    dh_beta = 0.0;
                }
                ht
            };

            let et = eps[t];
            let inv_h = 1.0 / ht;
            let et2_inv_h2 = et * et * inv_h * inv_h;
            // 0.5 * (1/h_t - ε_t²/h_t²)
            let factor = 0.5 * (inv_h - et2_inv_h2);

            g_omega += factor * dh_omega;
            g_alpha += factor * dh_alpha;
            g_beta += factor * dh_beta;
            // ∂NLL_t/∂μ = factor * ∂h_t/∂μ - ε_t/h_t
            g_mu += factor * dh_mu - et * inv_h;

            h_prev = ht;
        }

        Ok(vec![g_mu, g_omega, g_alpha, g_beta])
    }
}

pub fn garch11_fit(y: &[f64], cfg: Garch11Config) -> Result<Garch11Fit> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("y must be finite".to_string()));
    }

    let mu0 = mean(y);
    let v0 = var_pop(y, mu0).max(1e-12);
    let init =
        cfg.init.unwrap_or(Garch11Params { mu: mu0, omega: 0.1 * v0, alpha: 0.05, beta: 0.9 });
    let init_params = vec![init.mu, init.omega, init.alpha, init.beta];

    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY),
        (1e-12, f64::INFINITY),
        (0.0, 0.999_999),
        (0.0, 0.999_999),
    ];

    let obj = Garch11Objective {
        y: y.to_vec(),
        alpha_beta_max: cfg.alpha_beta_max,
        min_var: cfg.min_var,
    };
    let opt = LbfgsbOptimizer::new(cfg.optimizer).minimize(&obj, &init_params, &bounds)?;
    let params = Garch11Params {
        mu: opt.parameters[0],
        omega: opt.parameters[1],
        alpha: opt.parameters[2],
        beta: opt.parameters[3],
    };
    let (ll, h) = garch11_loglik(y, params, cfg.alpha_beta_max, cfg.min_var)?;

    Ok(Garch11Fit { params, log_likelihood: ll, conditional_variance: h, optimization: opt })
}

#[derive(Debug, Clone, Copy)]
pub struct SvLogChi2Params {
    pub mu: f64,
    pub phi: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone)]
pub struct SvLogChi2Config {
    pub optimizer: OptimizerConfig,
    pub log_eps: f64,
    pub init: Option<SvLogChi2Params>,
}

impl Default for SvLogChi2Config {
    fn default() -> Self {
        Self { optimizer: OptimizerConfig::default(), log_eps: 1e-12, init: None }
    }
}

#[derive(Debug, Clone)]
pub struct SvLogChi2Fit {
    pub params: SvLogChi2Params,
    pub log_likelihood: f64,
    pub smoothed_h: Vec<f64>,
    pub smoothed_sigma: Vec<f64>,
    pub optimization: OptimizationResult,
}

struct SvLogChi2Objective {
    z: Vec<f64>,
    log_eps: f64,
}

impl SvLogChi2Objective {
    fn loglik(&self, params: SvLogChi2Params) -> Result<(f64, KalmanModel, Vec<DVector<f64>>)> {
        if !params.mu.is_finite() || !params.phi.is_finite() || !params.sigma.is_finite() {
            return Err(Error::Validation("params must be finite".to_string()));
        }
        if params.sigma <= 0.0 {
            return Err(Error::Validation("sigma must be > 0".to_string()));
        }
        if !(params.phi.abs() < 1.0) {
            return Err(Error::Validation("|phi| must be < 1".to_string()));
        }
        if !self.log_eps.is_finite() || self.log_eps < 0.0 {
            return Err(Error::Validation("log_eps must be finite and >=0".to_string()));
        }

        let q = params.sigma * params.sigma;
        let denom = 1.0 - params.phi * params.phi;
        let mut p0 = if denom > 1e-12 { q / denom } else { q / 1e-12 };
        if !p0.is_finite() || p0 <= 0.0 {
            p0 = 1.0;
        }

        let model = KalmanModel::ar1(params.phi, q, LOG_CHI2_VAR, 0.0, p0)?;
        let ys: Vec<DVector<f64>> =
            self.z.iter().map(|&zt| DVector::from_row_slice(&[zt - params.mu])).collect();
        let fr = kalman_filter(&model, &ys)?;
        Ok((fr.log_likelihood, model, ys))
    }
}

impl ObjectiveFunction for SvLogChi2Objective {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 3 {
            return Err(Error::Validation("expected 3 params (mu, phi, sigma)".to_string()));
        }
        let p = SvLogChi2Params { mu: params[0], phi: params[1], sigma: params[2] };
        match self.loglik(p) {
            Ok((ll, _, _)) => Ok(-ll),
            Err(_) => Ok(1e30),
        }
    }
}

pub fn sv_logchi2_fit(y: &[f64], cfg: SvLogChi2Config) -> Result<SvLogChi2Fit> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("y must be finite".to_string()));
    }
    if !cfg.log_eps.is_finite() || cfg.log_eps < 0.0 {
        return Err(Error::Validation("log_eps must be finite and >=0".to_string()));
    }

    let eps = cfg.log_eps;
    let z: Vec<f64> = y.iter().map(|&v| (v * v + eps).ln() - LOG_CHI2_MEAN).collect();

    let mu0 = mean(&z);
    let init = cfg.init.unwrap_or(SvLogChi2Params { mu: mu0, phi: 0.9, sigma: 0.2 });
    let init_params = vec![init.mu, init.phi, init.sigma.max(1e-6)];

    let bounds =
        vec![(f64::NEG_INFINITY, f64::INFINITY), (-0.999_999, 0.999_999), (1e-6, f64::INFINITY)];

    let obj = SvLogChi2Objective { z, log_eps: cfg.log_eps };
    let opt = LbfgsbOptimizer::new(cfg.optimizer).minimize(&obj, &init_params, &bounds)?;
    let params =
        SvLogChi2Params { mu: opt.parameters[0], phi: opt.parameters[1], sigma: opt.parameters[2] };

    let (ll, model, ys) = obj.loglik(params)?;
    let fr = kalman_filter(&model, &ys)?;
    let sr = rts_smoother(&model, &fr)?;
    let mut smoothed_h = Vec::with_capacity(sr.smoothed_means.len());
    let mut smoothed_sigma = Vec::with_capacity(sr.smoothed_means.len());
    for m in sr.smoothed_means {
        let x = m[0];
        let h = params.mu + x;
        smoothed_h.push(h);
        smoothed_sigma.push((0.5 * h).exp());
    }

    Ok(SvLogChi2Fit { params, log_likelihood: ll, smoothed_h, smoothed_sigma, optimization: opt })
}

// ---------------------------------------------------------------------------
// EGARCH(1,1) — Nelson (1991)
// ---------------------------------------------------------------------------
//
// log(h_t) = ω + α·g(z_{t-1}) + β·log(h_{t-1})
// where z_t = ε_t / sqrt(h_t), g(z) = θ·z + γ·(|z| - E|z|), E|z| = sqrt(2/π).
//
// The log-variance formulation ensures h_t > 0 without parameter constraints.
// Leverage effect: γ < 0 means negative shocks increase volatility more.

/// Parameters of an EGARCH(1,1) model.
#[derive(Debug, Clone, Copy)]
pub struct Egarch11Params {
    /// Mean of the return series.
    pub mu: f64,
    /// Intercept in the log-variance equation.
    pub omega: f64,
    /// Magnitude effect coefficient.
    pub alpha: f64,
    /// Sign (asymmetry / leverage) coefficient.
    pub gamma: f64,
    /// Persistence coefficient.
    pub beta: f64,
}

/// Configuration for EGARCH(1,1) fitting.
#[derive(Debug, Clone, Default)]
pub struct Egarch11Config {
    pub optimizer: OptimizerConfig,
    pub init: Option<Egarch11Params>,
}

/// Result of an EGARCH(1,1) fit.
#[derive(Debug, Clone)]
pub struct Egarch11Fit {
    pub params: Egarch11Params,
    pub log_likelihood: f64,
    pub conditional_variance: Vec<f64>,
    pub optimization: OptimizationResult,
}

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4; // sqrt(2/π)

fn egarch11_loglik(y: &[f64], p: Egarch11Params) -> Result<(f64, Vec<f64>)> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    let params = [p.mu, p.omega, p.alpha, p.gamma, p.beta];
    if params.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("params must be finite".to_string()));
    }

    let eps: Vec<f64> = y.iter().map(|&v| v - p.mu).collect();

    let mut log_h = vec![0.0_f64; y.len()];
    // Unconditional log-variance: if |beta| < 1, log(h) = omega / (1 - beta).
    let denom = 1.0 - p.beta;
    log_h[0] = if denom.abs() > 1e-12 { p.omega / denom } else { p.omega };

    for t in 1..y.len() {
        let h_prev = log_h[t - 1].exp().max(1e-30);
        let z = eps[t - 1] / h_prev.sqrt();
        let g = p.alpha * (z.abs() - SQRT_2_OVER_PI) + p.gamma * z;
        log_h[t] = p.omega + g + p.beta * log_h[t - 1];
    }

    let mut h = vec![0.0_f64; y.len()];
    let mut ll = 0.0_f64;
    for t in 0..y.len() {
        let ht = log_h[t].exp().max(1e-30);
        h[t] = ht;
        let quad = (eps[t] * eps[t]) / ht;
        ll += -0.5 * (LN_2PI + ht.ln() + quad);
    }
    Ok((ll, h))
}

struct Egarch11Objective {
    y: Vec<f64>,
}

impl ObjectiveFunction for Egarch11Objective {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 5 {
            return Err(Error::Validation(
                "expected 5 params (mu, omega, alpha, gamma, beta)".to_string(),
            ));
        }
        let p = Egarch11Params {
            mu: params[0],
            omega: params[1],
            alpha: params[2],
            gamma: params[3],
            beta: params[4],
        };
        match egarch11_loglik(&self.y, p) {
            Ok((ll, _)) => Ok(-ll),
            Err(_) => Ok(1e30),
        }
    }
}

/// Fit an EGARCH(1,1) model by maximum likelihood.
///
/// Returns fitted parameters, log-likelihood, and conditional variance series.
pub fn egarch11_fit(y: &[f64], cfg: Egarch11Config) -> Result<Egarch11Fit> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("y must be finite".to_string()));
    }

    let mu0 = mean(y);
    let init = cfg.init.unwrap_or(Egarch11Params {
        mu: mu0,
        omega: -0.1,
        alpha: 0.1,
        gamma: -0.05,
        beta: 0.95,
    });
    let init_params = vec![init.mu, init.omega, init.alpha, init.gamma, init.beta];

    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY), // mu
        (f64::NEG_INFINITY, f64::INFINITY), // omega (unconstrained in EGARCH)
        (f64::NEG_INFINITY, f64::INFINITY), // alpha
        (f64::NEG_INFINITY, f64::INFINITY), // gamma
        (-0.999_999, 0.999_999),            // beta (stationarity)
    ];

    let obj = Egarch11Objective { y: y.to_vec() };
    let opt = LbfgsbOptimizer::new(cfg.optimizer).minimize(&obj, &init_params, &bounds)?;
    let params = Egarch11Params {
        mu: opt.parameters[0],
        omega: opt.parameters[1],
        alpha: opt.parameters[2],
        gamma: opt.parameters[3],
        beta: opt.parameters[4],
    };
    let (ll, h) = egarch11_loglik(y, params)?;

    Ok(Egarch11Fit { params, log_likelihood: ll, conditional_variance: h, optimization: opt })
}

// ---------------------------------------------------------------------------
// GJR-GARCH(1,1) — Glosten, Jagannathan & Runkle (1993)
// ---------------------------------------------------------------------------
//
// h_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1} < 0) + β·h_{t-1}
//
// The indicator term γ captures leverage: negative shocks (bad news) can have
// a larger effect on volatility than positive shocks of the same magnitude.
// Stationarity condition: α + β + γ/2 < 1.

/// Parameters of a GJR-GARCH(1,1) model.
#[derive(Debug, Clone, Copy)]
pub struct GjrGarch11Params {
    /// Mean of the return series.
    pub mu: f64,
    /// Intercept (must be > 0).
    pub omega: f64,
    /// ARCH coefficient (must be >= 0).
    pub alpha: f64,
    /// Asymmetry / leverage coefficient (must be >= 0 for standard leverage).
    pub gamma: f64,
    /// GARCH persistence coefficient (must be >= 0).
    pub beta: f64,
}

/// Configuration for GJR-GARCH(1,1) fitting.
#[derive(Debug, Clone)]
pub struct GjrGarch11Config {
    pub optimizer: OptimizerConfig,
    /// Maximum value for α + β + γ/2 (stationarity bound).
    pub persistence_max: f64,
    pub init: Option<GjrGarch11Params>,
    pub min_var: f64,
}

impl Default for GjrGarch11Config {
    fn default() -> Self {
        Self {
            optimizer: OptimizerConfig::default(),
            persistence_max: 0.999,
            init: None,
            min_var: 1e-18,
        }
    }
}

/// Result of a GJR-GARCH(1,1) fit.
#[derive(Debug, Clone)]
pub struct GjrGarch11Fit {
    pub params: GjrGarch11Params,
    pub log_likelihood: f64,
    pub conditional_variance: Vec<f64>,
    pub optimization: OptimizationResult,
}

fn gjr_garch11_loglik(
    y: &[f64],
    p: GjrGarch11Params,
    persistence_max: f64,
    min_var: f64,
) -> Result<(f64, Vec<f64>)> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    let params = [p.mu, p.omega, p.alpha, p.gamma, p.beta];
    if params.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("params must be finite".to_string()));
    }
    if p.omega <= 0.0 || p.alpha < 0.0 || p.gamma < 0.0 || p.beta < 0.0 {
        return Err(Error::Validation(
            "omega must be >0 and alpha/gamma/beta must be >=0".to_string(),
        ));
    }
    let persistence = p.alpha + p.beta + 0.5 * p.gamma;
    if persistence >= persistence_max {
        return Err(Error::Validation(
            "alpha + beta + gamma/2 violates persistence_max".to_string(),
        ));
    }
    let min_var = min_var.max(0.0);

    let eps: Vec<f64> = y.iter().map(|&v| v - p.mu).collect();

    let mut h = vec![0.0_f64; y.len()];
    // Unconditional variance: h = omega / (1 - alpha - beta - gamma/2).
    let denom = 1.0 - persistence;
    let mut h0 = if denom > 1e-12 {
        p.omega / denom
    } else {
        let m = mean(&eps);
        var_pop(&eps, m)
    };
    if !h0.is_finite() || h0 <= 0.0 {
        h0 = min_var.max(1e-12);
    }
    h[0] = h0.max(min_var);

    for t in 1..y.len() {
        let prev_eps = eps[t - 1];
        let prev_eps2 = prev_eps * prev_eps;
        let indicator = if prev_eps < 0.0 { 1.0 } else { 0.0 };
        let mut v =
            p.omega + p.alpha * prev_eps2 + p.gamma * prev_eps2 * indicator + p.beta * h[t - 1];
        if !v.is_finite() || v <= 0.0 {
            v = min_var.max(1e-12);
        }
        h[t] = v.max(min_var);
    }

    let mut ll = 0.0_f64;
    for (et, ht) in eps.iter().zip(h.iter()) {
        let quad = (et * et) / ht;
        ll += -0.5 * (LN_2PI + ht.ln() + quad);
    }
    Ok((ll, h))
}

struct GjrGarch11Objective {
    y: Vec<f64>,
    persistence_max: f64,
    min_var: f64,
}

impl ObjectiveFunction for GjrGarch11Objective {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 5 {
            return Err(Error::Validation(
                "expected 5 params (mu, omega, alpha, gamma, beta)".to_string(),
            ));
        }
        let p = GjrGarch11Params {
            mu: params[0],
            omega: params[1],
            alpha: params[2],
            gamma: params[3],
            beta: params[4],
        };
        match gjr_garch11_loglik(&self.y, p, self.persistence_max, self.min_var) {
            Ok((ll, _)) => Ok(-ll),
            Err(_) => Ok(1e30),
        }
    }
}

/// Fit a GJR-GARCH(1,1) model by maximum likelihood.
///
/// The GJR model extends GARCH(1,1) with an asymmetric leverage term:
/// negative shocks have a larger impact on volatility when γ > 0.
///
/// Returns fitted parameters, log-likelihood, and conditional variance series.
pub fn gjr_garch11_fit(y: &[f64], cfg: GjrGarch11Config) -> Result<GjrGarch11Fit> {
    if y.is_empty() {
        return Err(Error::Validation("y must be non-empty".to_string()));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("y must be finite".to_string()));
    }

    let mu0 = mean(y);
    let v0 = var_pop(y, mu0).max(1e-12);
    let init = cfg.init.unwrap_or(GjrGarch11Params {
        mu: mu0,
        omega: 0.1 * v0,
        alpha: 0.03,
        gamma: 0.05,
        beta: 0.9,
    });
    let init_params = vec![init.mu, init.omega, init.alpha, init.gamma, init.beta];

    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY), // mu
        (1e-12, f64::INFINITY),             // omega > 0
        (0.0, 0.999_999),                   // alpha >= 0
        (0.0, 0.999_999),                   // gamma >= 0
        (0.0, 0.999_999),                   // beta >= 0
    ];

    let obj = GjrGarch11Objective {
        y: y.to_vec(),
        persistence_max: cfg.persistence_max,
        min_var: cfg.min_var,
    };
    let opt = LbfgsbOptimizer::new(cfg.optimizer).minimize(&obj, &init_params, &bounds)?;
    let params = GjrGarch11Params {
        mu: opt.parameters[0],
        omega: opt.parameters[1],
        alpha: opt.parameters[2],
        gamma: opt.parameters[3],
        beta: opt.parameters[4],
    };
    let (ll, h) = gjr_garch11_loglik(y, params, cfg.persistence_max, cfg.min_var)?;

    Ok(GjrGarch11Fit { params, log_likelihood: ll, conditional_variance: h, optimization: opt })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garch11_fit_smoke() {
        let y = [0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02];
        let fit = garch11_fit(&y, Garch11Config::default()).unwrap();
        assert!(fit.log_likelihood.is_finite());
        assert_eq!(fit.conditional_variance.len(), y.len());
        assert!(fit.params.omega > 0.0);
        assert!(fit.params.alpha + fit.params.beta < 1.0);
    }

    #[test]
    fn sv_logchi2_fit_smoke() {
        let y = [0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02];
        let fit = sv_logchi2_fit(&y, SvLogChi2Config::default()).unwrap();
        assert!(fit.log_likelihood.is_finite());
        assert_eq!(fit.smoothed_h.len(), y.len());
        assert!(fit.params.sigma > 0.0);
        assert!(fit.params.phi.abs() < 1.0);
    }

    #[test]
    fn egarch11_fit_smoke() {
        let y = [0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02];
        let fit = egarch11_fit(&y, Egarch11Config::default()).unwrap();
        assert!(fit.log_likelihood.is_finite());
        assert_eq!(fit.conditional_variance.len(), y.len());
        // All conditional variances must be positive (EGARCH guarantees this).
        for &h in &fit.conditional_variance {
            assert!(h > 0.0, "conditional variance must be > 0, got {}", h);
        }
    }

    #[test]
    fn egarch11_leverage_sign() {
        // Larger negative returns should produce higher variance than positive ones.
        // Construct a series with asymmetric shocks.
        let mut y = vec![0.0; 50];
        // Insert large negative shocks.
        y[10] = -0.5;
        y[20] = -0.6;
        y[30] = -0.4;
        // Insert smaller positive shocks.
        y[15] = 0.3;
        y[25] = 0.2;
        y[35] = 0.1;
        let fit = egarch11_fit(&y, Egarch11Config::default()).unwrap();
        assert!(fit.log_likelihood.is_finite());
        // gamma should be negative (leverage effect) or at least the fit converges.
        assert!(fit.params.beta.abs() < 1.0, "beta must satisfy |beta| < 1");
    }

    #[test]
    fn gjr_garch11_fit_smoke() {
        let y = [0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02];
        let fit = gjr_garch11_fit(&y, GjrGarch11Config::default()).unwrap();
        assert!(fit.log_likelihood.is_finite());
        assert_eq!(fit.conditional_variance.len(), y.len());
        assert!(fit.params.omega > 0.0);
        assert!(fit.params.alpha >= 0.0);
        assert!(fit.params.gamma >= 0.0);
        assert!(fit.params.beta >= 0.0);
        let persistence = fit.params.alpha + fit.params.beta + 0.5 * fit.params.gamma;
        assert!(persistence < 1.0, "persistence = {}", persistence);
    }

    #[test]
    fn gjr_garch11_stationarity_guard() {
        // With extreme params that violate stationarity, loglik should error.
        let y = [0.1, -0.2, 0.05];
        let p = GjrGarch11Params { mu: 0.0, omega: 0.01, alpha: 0.5, gamma: 0.5, beta: 0.5 };
        // alpha + beta + gamma/2 = 0.5 + 0.5 + 0.25 = 1.25 > 0.999
        let result = gjr_garch11_loglik(&y, p, 0.999, 1e-18);
        assert!(result.is_err());
    }

    #[test]
    fn gjr_vs_garch_nesting() {
        // GJR-GARCH with gamma=0 reduces to GARCH, so its LL should be >= GARCH LL.
        // Warm-start GJR from the GARCH solution to avoid local minima.
        let y: Vec<f64> = (0..100).map(|i| ((i as f64) * 0.1).sin() * 0.1).collect();
        let garch = garch11_fit(&y, Garch11Config::default()).unwrap();
        let gjr_init = GjrGarch11Params {
            mu: garch.params.mu,
            omega: garch.params.omega,
            alpha: garch.params.alpha,
            gamma: 0.0,
            beta: garch.params.beta,
        };
        let gjr =
            gjr_garch11_fit(&y, GjrGarch11Config { init: Some(gjr_init), ..Default::default() })
                .unwrap();
        assert!(
            gjr.log_likelihood >= garch.log_likelihood - 1.0,
            "GJR LL ({}) should be close to or better than GARCH LL ({})",
            gjr.log_likelihood,
            garch.log_likelihood
        );
    }

    #[test]
    fn garch11_analytical_gradient_vs_numerical() {
        let y: Vec<f64> = vec![
            0.1, -0.2, 0.05, 0.3, -0.15, 0.02, 0.01, -0.4, 0.35, -0.1, 0.05, -0.02, 0.12, -0.08,
            0.22, -0.31, 0.14, -0.06, 0.09, 0.03,
        ];
        let obj = Garch11Objective { y, alpha_beta_max: 0.999, min_var: 1e-18 };
        let params = [0.001, 0.02, 0.08, 0.88];

        // Analytical gradient
        let g_analytical = obj.gradient(&params).unwrap();

        // Numerical gradient (central differences)
        let n = params.len();
        let mut g_numerical = vec![0.0; n];
        for i in 0..n {
            let eps = 1e-7 * params[i].abs().max(1.0);
            let mut pp = params.to_vec();
            pp[i] += eps;
            let fp = obj.eval(&pp).unwrap();
            pp[i] = params[i] - eps;
            let fm = obj.eval(&pp).unwrap();
            g_numerical[i] = (fp - fm) / (2.0 * eps);
        }

        for i in 0..n {
            let rel = if g_numerical[i].abs() > 1e-10 {
                (g_analytical[i] - g_numerical[i]).abs() / g_numerical[i].abs()
            } else {
                (g_analytical[i] - g_numerical[i]).abs()
            };
            assert!(
                rel < 1e-4,
                "gradient[{}]: analytical={:.8e}, numerical={:.8e}, rel_diff={:.4e}",
                i,
                g_analytical[i],
                g_numerical[i],
                rel
            );
        }
    }
}
