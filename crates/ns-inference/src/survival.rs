//! Survival models (Phase 9 Pack A).
//!
//! Baseline scope:
//! - right-censoring via (t, event) data
//! - intercept-only parametric families
//! - Cox proportional hazards (partial likelihood; covariates supported)
//!
//! Parameterisation:
//! - Exponential: `log_rate`
//! - Weibull: `log_k`, `log_lambda`
//! - LogNormal AFT: `mu`, `log_sigma`

use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::exp_clamped;

const MIN_TAIL: f64 = 1e-300;

#[inline]
fn validate_right_censoring_data(times: &[f64], events: &[bool]) -> Result<()> {
    if times.is_empty() {
        return Err(Error::Validation("times must be non-empty".to_string()));
    }
    if times.len() != events.len() {
        return Err(Error::Validation(format!(
            "times/events length mismatch: {} vs {}",
            times.len(),
            events.len()
        )));
    }
    if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(Error::Validation(
            "times must be finite and >= 0".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn normal_phi(z: f64) -> f64 {
    // phi(z) = exp(-0.5*z^2)/sqrt(2*pi)
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    (-0.5 * z * z).exp() * INV_SQRT_2PI
}

#[inline]
fn normal_cdf(x: f64) -> f64 {
    // Use erfc for better tail behavior:
    // Phi(x) = 0.5 * erfc(-x / sqrt(2))
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
}

// ---------------------------------------------------------------------------
// Exponential survival (right-censoring).
// ---------------------------------------------------------------------------

/// Exponential survival model with right-censoring.
///
/// Model:
/// - `T ~ Exp(rate)` (support: `t >= 0`)
/// - `event[i]=true` indicates an observed event at `t[i]`
/// - `event[i]=false` indicates right-censoring at `t[i]` (contribution `S(t)`)
#[derive(Debug, Clone)]
pub struct ExponentialSurvivalModel {
    sum_t: f64,
    n_events: f64,
}

impl ExponentialSurvivalModel {
    /// Create a new exponential survival model from `times` and `events`.
    pub fn new(times: Vec<f64>, events: Vec<bool>) -> Result<Self> {
        validate_right_censoring_data(&times, &events)?;
        let mut sum_t = 0.0;
        let mut n_events = 0.0;
        for (t, d) in times.iter().zip(events.iter()) {
            sum_t += *t;
            if *d {
                n_events += 1.0;
            }
        }
        Ok(Self { sum_t, n_events })
    }

    #[inline]
    fn rate(log_rate: f64) -> Result<f64> {
        if !log_rate.is_finite() {
            return Err(Error::Validation("log_rate must be finite".to_string()));
        }
        Ok(log_rate.exp())
    }
}

impl LogDensityModel for ExponentialSurvivalModel {
    type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

    fn dim(&self) -> usize {
        1
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["log_rate".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-30.0, 30.0)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 1 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let log_rate = params[0];
        let rate = Self::rate(log_rate)?;
        // loglik = sum_i [d_i * (log_rate - rate*t_i) + (1-d_i) * (-rate*t_i)]
        //        = n_events * log_rate - rate * sum_t
        let nll = -self.n_events * log_rate + rate * self.sum_t;
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 1 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let log_rate = params[0];
        let rate = Self::rate(log_rate)?;
        let g = -self.n_events + rate * self.sum_t;
        Ok(vec![g])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// Weibull survival (right-censoring).
// ---------------------------------------------------------------------------

/// Weibull survival model with right-censoring.
///
/// Parameterisation: `k = exp(log_k)`, `lambda = exp(log_lambda)`.
#[derive(Debug, Clone)]
pub struct WeibullSurvivalModel {
    times: Vec<f64>,
    events: Vec<bool>,
    ln_times: Vec<f64>,
}

impl WeibullSurvivalModel {
    /// Create a new Weibull survival model from `times` and `events`.
    pub fn new(times: Vec<f64>, events: Vec<bool>) -> Result<Self> {
        validate_right_censoring_data(&times, &events)?;
        // For Weibull, t==0 is allowed in theory but leads to singularities for k != 1.
        // Keep baseline strict and avoid special casing: require strictly positive times.
        if times.iter().any(|t| *t <= 0.0) {
            return Err(Error::Validation("weibull requires times > 0".to_string()));
        }
        let ln_times = times.iter().map(|t| t.ln()).collect();
        Ok(Self { times, events, ln_times })
    }
}

impl LogDensityModel for WeibullSurvivalModel {
    type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

    fn dim(&self) -> usize {
        2
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["log_k".to_string(), "log_lambda".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0), (-30.0, 30.0)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 2 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let log_k = params[0];
        let log_lambda = params[1];
        if !(log_k.is_finite() && log_lambda.is_finite()) {
            return Err(Error::Validation("params must be finite".to_string()));
        }

        let k = log_k.exp();

        let mut ll = 0.0;
        for i in 0..self.times.len() {
            let d = self.events[i];
            let ln_t = self.ln_times[i];

            let ln_z = ln_t - log_lambda; // ln(t/lambda)
            let a = (k * ln_z).exp(); // (t/lambda)^k

            // loglik = d * [log_k - log_lambda + (k-1)*ln(t/lambda)] - (t/lambda)^k
            if d {
                ll += log_k - log_lambda + (k - 1.0) * ln_z;
            }
            ll -= a;
        }
        Ok(-ll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 2 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let log_k = params[0];
        let log_lambda = params[1];
        if !(log_k.is_finite() && log_lambda.is_finite()) {
            return Err(Error::Validation("params must be finite".to_string()));
        }

        let k = log_k.exp();
        let mut g_log_k = 0.0;
        let mut g_log_lambda = 0.0;

        for i in 0..self.times.len() {
            let d = self.events[i];
            let ln_z = self.ln_times[i] - log_lambda;
            let a = (k * ln_z).exp();

            let d_f = if d { 1.0 } else { 0.0 };

            // d/d log_k of loglik: d*(1 + k*ln_z) - a*k*ln_z
            let dloglik_dlogk = d_f * (1.0 + k * ln_z) - a * k * ln_z;
            // d/d log_lambda of loglik: k*(a - d)
            let dloglik_dloglambda = k * (a - d_f);

            g_log_k -= dloglik_dlogk;
            g_log_lambda -= dloglik_dloglambda;
        }

        Ok(vec![g_log_k, g_log_lambda])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// LogNormal AFT survival (right-censoring).
// ---------------------------------------------------------------------------

/// LogNormal AFT survival model with right-censoring.
///
/// Model: `ln T ~ Normal(mu, sigma)`, parameterisation: `sigma = exp(log_sigma)`.
#[derive(Debug, Clone)]
pub struct LogNormalAftModel {
    times: Vec<f64>,
    events: Vec<bool>,
    ln_times: Vec<f64>,
}

impl LogNormalAftModel {
    /// Create a new log-normal AFT survival model from `times` and `events`.
    pub fn new(times: Vec<f64>, events: Vec<bool>) -> Result<Self> {
        validate_right_censoring_data(&times, &events)?;
        if times.iter().any(|t| *t <= 0.0) {
            return Err(Error::Validation("lognormal requires times > 0".to_string()));
        }
        let ln_times = times.iter().map(|t| t.ln()).collect();
        Ok(Self { times, events, ln_times })
    }
}

impl LogDensityModel for LogNormalAftModel {
    type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

    fn dim(&self) -> usize {
        2
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["mu".to_string(), "log_sigma".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-100.0, 100.0), (-10.0, 10.0)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 2 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let mu = params[0];
        let log_sigma = params[1];
        if !(mu.is_finite() && log_sigma.is_finite()) {
            return Err(Error::Validation("params must be finite".to_string()));
        }
        let sigma = log_sigma.exp();
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(Error::Validation("sigma must be finite and > 0".to_string()));
        }

        let mut ll = 0.0;
        for i in 0..self.times.len() {
            let y = self.ln_times[i];
            let z = (y - mu) / sigma;
            if self.events[i] {
                // log f(t) = log f_Y(y) - log t
                let lp_y = ns_prob::normal::logpdf(y, mu, sigma)?;
                ll += lp_y - y;
            } else {
                // log S(t) = log P(T > t) = log Phi(-(ln t - mu)/sigma)
                let p = normal_cdf(-z).max(MIN_TAIL);
                ll += p.ln();
            }
        }
        Ok(-ll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 2 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let mu = params[0];
        let log_sigma = params[1];
        if !(mu.is_finite() && log_sigma.is_finite()) {
            return Err(Error::Validation("params must be finite".to_string()));
        }
        let sigma = log_sigma.exp();
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(Error::Validation("sigma must be finite and > 0".to_string()));
        }

        let mut g_mu = 0.0;
        let mut g_log_sigma = 0.0;

        for i in 0..self.times.len() {
            let y = self.ln_times[i];
            let z = (y - mu) / sigma;
            if self.events[i] {
                // logpdf component:
                // d/d mu: z / sigma
                // d/d log_sigma: z^2 - 1
                let dloglik_dmu = z / sigma;
                let dloglik_dlogsig = z * z - 1.0;
                g_mu -= dloglik_dmu;
                g_log_sigma -= dloglik_dlogsig;
            } else {
                // censored: logS = log Phi(-z)
                let p = normal_cdf(-z).max(MIN_TAIL);
                let ratio = normal_phi(z) / p; // phi(z) / Phi(-z)
                let dloglik_dmu = ratio / sigma;
                let dloglik_dlogsig = z * ratio;
                g_mu -= dloglik_dmu;
                g_log_sigma -= dloglik_dlogsig;
            }
        }

        Ok(vec![g_mu, g_log_sigma])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// Cox proportional hazards (partial likelihood; right-censoring).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Ties approximation for the Cox PH partial likelihood.
pub enum CoxTies {
    /// Breslow ties approximation.
    Breslow,
    /// Efron ties approximation.
    Efron,
}

#[inline]
fn row_dot(x_row: &[f64], beta: &[f64]) -> f64 {
    debug_assert_eq!(x_row.len(), beta.len());
    x_row.iter().zip(beta).map(|(&x, &b)| x * b).sum()
}

/// Cox proportional hazards model using partial likelihood (right-censoring).
///
/// Data:
/// - `times[i] >= 0`
/// - `events[i] = true` indicates an observed event at `times[i]`
/// - `events[i] = false` indicates right-censoring at `times[i]`
///
/// Notes:
/// - Intercept is not identifiable in the partial likelihood and is not included.
/// - Ties policy is fixed per-model (Breslow or Efron).
#[derive(Debug, Clone)]
pub struct CoxPhModel {
    n: usize,
    p: usize,
    events: Vec<bool>, // aligned with sorted times
    x: Vec<f64>,      // row-major, aligned with sorted times
    group_starts: Vec<usize>,
    ties: CoxTies,
}

impl CoxPhModel {
    /// Create a new Cox PH model from `times`, `events`, and row-wise covariates `x`.
    pub fn new(times: Vec<f64>, events: Vec<bool>, x: Vec<Vec<f64>>, ties: CoxTies) -> Result<Self> {
        validate_right_censoring_data(&times, &events)?;
        let n = times.len();
        let p = x.first().map(|r| r.len()).unwrap_or(0);
        if p == 0 {
            return Err(Error::Validation("X must have at least 1 feature column".to_string()));
        }
        if x.len() != n {
            return Err(Error::Validation(format!(
                "X must have n rows: expected {}, got {}",
                n,
                x.len()
            )));
        }
        if events.iter().all(|d| !*d) {
            return Err(Error::Validation("need at least one event".to_string()));
        }

        // Validate and pack X row-major.
        let mut x_data = Vec::with_capacity(n * p);
        for (i, row) in x.into_iter().enumerate() {
            if row.len() != p {
                return Err(Error::Validation(format!(
                    "X must be rectangular: row {} has len {}, expected {}",
                    i,
                    row.len(),
                    p
                )));
            }
            for v in row {
                if !v.is_finite() {
                    return Err(Error::Validation("X must contain only finite values".to_string()));
                }
                x_data.push(v);
            }
        }

        // Sort by time descending and build aligned arrays.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| times[j].partial_cmp(&times[i]).unwrap());

        let mut times_s = Vec::with_capacity(n);
        let mut events_s = Vec::with_capacity(n);
        let mut x_s = Vec::with_capacity(n * p);
        for idx in order {
            times_s.push(times[idx]);
            events_s.push(events[idx]);
            let start = idx * p;
            x_s.extend_from_slice(&x_data[start..start + p]);
        }

        // Group boundaries for ties at exact times.
        let mut group_starts = Vec::new();
        group_starts.push(0);
        for i in 1..n {
            if times_s[i] != times_s[i - 1] {
                group_starts.push(i);
            }
        }

        Ok(Self { n, p, events: events_s, x: x_s, group_starts, ties })
    }

    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        let start = i * self.p;
        &self.x[start..start + self.p]
    }
}

impl LogDensityModel for CoxPhModel {
    type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

    fn dim(&self) -> usize {
        self.p
    }

    fn parameter_names(&self) -> Vec<String> {
        (0..self.p).map(|j| format!("beta{}", j + 1)).collect()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.p]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0; self.p]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.p {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must be finite".to_string()));
        }

        let mut eta = vec![0.0; self.n];
        let mut w = vec![0.0; self.n];
        for i in 0..self.n {
            let e = row_dot(self.row(i), params);
            eta[i] = e;
            w[i] = exp_clamped(e);
        }

        let mut ll = 0.0;
        let mut risk0 = 0.0;

        for (g, &start) in self.group_starts.iter().enumerate() {
            let end = self
                .group_starts
                .get(g + 1)
                .copied()
                .unwrap_or(self.n);

            for i in start..end {
                risk0 += w[i];
            }

            let mut m = 0usize;
            let mut sum_eta_events = 0.0;
            let mut d0 = 0.0;
            for i in start..end {
                if self.events[i] {
                    m += 1;
                    sum_eta_events += eta[i];
                    d0 += w[i];
                }
            }
            if m == 0 {
                continue;
            }

            ll += sum_eta_events;
            match self.ties {
                CoxTies::Breslow => {
                    ll -= (m as f64) * risk0.ln();
                }
                CoxTies::Efron => {
                    let m_f = m as f64;
                    for r in 0..m {
                        let frac = (r as f64) / m_f;
                        let denom = (risk0 - frac * d0).max(MIN_TAIL);
                        ll -= denom.ln();
                    }
                }
            }
        }

        Ok(-ll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != self.p {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must be finite".to_string()));
        }

        let mut eta = vec![0.0; self.n];
        let mut w = vec![0.0; self.n];
        for i in 0..self.n {
            let e = row_dot(self.row(i), params);
            eta[i] = e;
            w[i] = exp_clamped(e);
        }

        let mut grad_ll = vec![0.0; self.p];
        let mut risk0 = 0.0;
        let mut risk1 = vec![0.0; self.p];

        for (g, &start) in self.group_starts.iter().enumerate() {
            let end = self
                .group_starts
                .get(g + 1)
                .copied()
                .unwrap_or(self.n);

            for i in start..end {
                let wi = w[i];
                risk0 += wi;
                let row = self.row(i);
                for j in 0..self.p {
                    risk1[j] += wi * row[j];
                }
            }

            let mut m = 0usize;
            let mut x_events = vec![0.0; self.p];
            let mut d0 = 0.0;
            let mut d1 = vec![0.0; self.p];
            for i in start..end {
                if !self.events[i] {
                    continue;
                }
                m += 1;
                let wi = w[i];
                d0 += wi;
                let row = self.row(i);
                for j in 0..self.p {
                    x_events[j] += row[j];
                    d1[j] += wi * row[j];
                }
            }
            if m == 0 {
                continue;
            }

            match self.ties {
                CoxTies::Breslow => {
                    let inv = 1.0 / risk0.max(MIN_TAIL);
                    for j in 0..self.p {
                        grad_ll[j] += x_events[j] - (m as f64) * risk1[j] * inv;
                    }
                }
                CoxTies::Efron => {
                    let m_f = m as f64;
                    let mut tmp = vec![0.0; self.p];
                    for r in 0..m {
                        let frac = (r as f64) / m_f;
                        let denom = (risk0 - frac * d0).max(MIN_TAIL);
                        let inv = 1.0 / denom;
                        for j in 0..self.p {
                            tmp[j] += (risk1[j] - frac * d1[j]) * inv;
                        }
                    }
                    for j in 0..self.p {
                        grad_ll[j] += x_events[j] - tmp[j];
                    }
                }
            }
        }

        Ok(grad_ll.into_iter().map(|v| -v).collect())
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_grad<M: LogDensityModel>(m: &M, params: &[f64], eps: f64) -> Vec<f64> {
        let mut g = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut p_hi = params.to_vec();
            let mut p_lo = params.to_vec();
            p_hi[i] += eps;
            p_lo[i] -= eps;
            let f_hi = m.nll(&p_hi).unwrap();
            let f_lo = m.nll(&p_lo).unwrap();
            g[i] = (f_hi - f_lo) / (2.0 * eps);
        }
        g
    }

    #[test]
    fn exponential_grad_matches_finite_diff() {
        let times = vec![0.5, 1.2, 0.7, 2.0];
        let events = vec![true, false, true, false];
        let m = ExponentialSurvivalModel::new(times, events).unwrap();
        let p = vec![0.3];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        assert!((g[0] - g_fd[0]).abs() < 1e-6);
    }

    #[test]
    fn weibull_k1_matches_exponential_nll() {
        let times = vec![0.2, 1.0, 0.7, 2.3];
        let events = vec![true, false, true, true];
        let exp_m = ExponentialSurvivalModel::new(times.clone(), events.clone()).unwrap();
        let w_m = WeibullSurvivalModel::new(times, events).unwrap();

        // k=1, lambda = 1/rate
        let log_rate: f64 = 0.4;
        let rate = log_rate.exp();
        let log_k = 0.0;
        let log_lambda: f64 = (1.0 / rate).ln();

        let nll_e = exp_m.nll(&[log_rate]).unwrap();
        let nll_w = w_m.nll(&[log_k, log_lambda]).unwrap();
        assert!((nll_e - nll_w).abs() < 1e-9);
    }

    #[test]
    fn weibull_grad_matches_finite_diff() {
        let times = vec![0.5, 1.2, 0.7, 2.0, 0.9];
        let events = vec![true, false, true, false, true];
        let m = WeibullSurvivalModel::new(times, events).unwrap();
        let p = vec![0.2, -0.1];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        assert!((g[0] - g_fd[0]).abs() < 5e-5);
        assert!((g[1] - g_fd[1]).abs() < 5e-5);
    }

    #[test]
    fn lognormal_grad_matches_finite_diff() {
        let times = vec![0.5, 1.2, 0.7, 2.0, 0.9, 3.1];
        let events = vec![true, false, true, false, false, true];
        let m = LogNormalAftModel::new(times, events).unwrap();
        let p = vec![0.1, -0.2];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        assert!((g[0] - g_fd[0]).abs() < 5e-5);
        assert!((g[1] - g_fd[1]).abs() < 5e-5);
    }

    #[test]
    fn cox_grad_matches_finite_diff_efron() {
        let times = vec![2.0, 1.0, 1.0, 0.5, 0.5, 0.2];
        let events = vec![true, true, false, true, false, false];
        let x = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
            vec![0.0, -1.0],
            vec![0.5, 0.5],
        ];
        let m = CoxPhModel::new(times, events, x, CoxTies::Efron).unwrap();
        let p = vec![0.1, -0.2];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        assert!((g[0] - g_fd[0]).abs() < 5e-5);
        assert!((g[1] - g_fd[1]).abs() < 5e-5);
    }
}
