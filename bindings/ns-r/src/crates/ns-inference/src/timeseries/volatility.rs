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
}
