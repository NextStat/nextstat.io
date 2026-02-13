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
        return Err(Error::Validation("times must be finite and >= 0".to_string()));
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
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

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
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

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
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

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
    n_events: usize,
    events: Vec<bool>, // aligned with sorted times
    x: Vec<f64>,       // row-major, aligned with sorted times
    group_starts: Vec<usize>,
    ties: CoxTies,
}

impl CoxPhModel {
    /// Create a new Cox PH model from `times`, `events`, and row-wise covariates `x`.
    pub fn new(
        times: Vec<f64>,
        events: Vec<bool>,
        x: Vec<Vec<f64>>,
        ties: CoxTies,
    ) -> Result<Self> {
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

        // Center covariates for numerical stability (prevents exp(X*beta) overflow
        // at large N). This matches lifelines and R survival::coxph behaviour.
        // Centering does not change the Cox PH partial likelihood optimum because
        // it depends only on covariate differences within risk sets.
        let mut means = vec![0.0_f64; p];
        for i in 0..n {
            for j in 0..p {
                means[j] += x_data[i * p + j];
            }
        }
        let n_f = n as f64;
        for j in 0..p {
            means[j] /= n_f;
        }
        for i in 0..n {
            for j in 0..p {
                x_data[i * p + j] -= means[j];
            }
        }

        // Sort by time descending and build aligned arrays.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| times[j].total_cmp(&times[i]));

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

        let n_events = events_s.iter().filter(|&&e| e).count();

        Ok(Self { n, p, n_events, events: events_s, x: x_s, group_starts, ties })
    }

    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        let start = i * self.p;
        &self.x[start..start + self.p]
    }
}

impl LogDensityModel for CoxPhModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

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
            let end = self.group_starts.get(g + 1).copied().unwrap_or(self.n);

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

        // Normalize by n_events for numerical stability (keeps gradient O(1)).
        let scale = if self.n_events > 0 { 1.0 / self.n_events as f64 } else { 1.0 };
        Ok(-ll * scale)
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
            let end = self.group_starts.get(g + 1).copied().unwrap_or(self.n);

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

        let scale = if self.n_events > 0 { 1.0 / self.n_events as f64 } else { 1.0 };
        Ok(grad_ll.into_iter().map(|v| -v * scale).collect())
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// Kaplan-Meier estimator (non-parametric).
// ---------------------------------------------------------------------------

/// A single step in the Kaplan-Meier survival curve.
#[derive(Debug, Clone)]
pub struct KaplanMeierStep {
    /// Event time.
    pub time: f64,
    /// Number at risk just before this time.
    pub n_risk: usize,
    /// Number of events at this time.
    pub n_events: usize,
    /// Number of censorings at this time.
    pub n_censored: usize,
    /// Kaplan-Meier survival estimate S(t) just after this time.
    pub survival: f64,
    /// Greenwood variance estimate of S(t).
    pub variance: f64,
    /// Lower bound of pointwise confidence interval.
    pub ci_lower: f64,
    /// Upper bound of pointwise confidence interval.
    pub ci_upper: f64,
}

/// Result of the Kaplan-Meier estimator.
#[derive(Debug, Clone)]
pub struct KaplanMeierEstimate {
    /// Ordered steps of the survival curve (one per distinct event time).
    pub steps: Vec<KaplanMeierStep>,
    /// Median survival time (smallest t where S(t) <= 0.5), or `None` if
    /// the survival function never drops to 0.5.
    pub median: Option<f64>,
    /// Confidence level used for intervals (e.g. 0.95).
    pub conf_level: f64,
    /// Total number of observations.
    pub n: usize,
    /// Total number of events.
    pub n_events: usize,
}

/// Compute the Kaplan-Meier survival estimate.
///
/// # Arguments
/// - `times` — observation times (≥ 0, finite).
/// - `events` — `true` = event, `false` = right-censored.
/// - `conf_level` — confidence level for pointwise CIs (default: 0.95).
///   Uses the log-log transform (complementary log-log) for better small-sample
///   coverage, matching R's `survival::survfit` default.
///
/// # Returns
/// A [`KaplanMeierEstimate`] with one step per distinct **event** time.
pub fn kaplan_meier(
    times: &[f64],
    events: &[bool],
    conf_level: f64,
) -> Result<KaplanMeierEstimate> {
    validate_right_censoring_data(times, events)?;
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation("conf_level must be in (0, 1)".to_string()));
    }

    let n = times.len();
    let total_events = events.iter().filter(|&&e| e).count();

    // Sort by time ascending; within ties events come before censorings
    // (so that n_risk is correct at each event time).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a].partial_cmp(&times[b]).unwrap().then_with(|| {
            // events (true) sort before censorings (false)
            events[b].cmp(&events[a])
        })
    });

    // Walk distinct times, accumulate d_j (events) and c_j (censorings).
    struct TimeSlice {
        time: f64,
        d: usize,
        c: usize,
    }
    let mut slices: Vec<TimeSlice> = Vec::new();
    {
        let mut i = 0;
        while i < n {
            let t = times[order[i]];
            let mut d = 0usize;
            let mut c = 0usize;
            while i < n && times[order[i]] == t {
                if events[order[i]] {
                    d += 1;
                } else {
                    c += 1;
                }
                i += 1;
            }
            slices.push(TimeSlice { time: t, d, c });
        }
    }

    let z_alpha = normal_cdf_inv(1.0 - (1.0 - conf_level) / 2.0);

    let mut steps: Vec<KaplanMeierStep> = Vec::new();
    let mut n_risk = n;
    let mut s = 1.0_f64;
    let mut greenwood_sum = 0.0_f64;

    for slice in &slices {
        if slice.d > 0 {
            let nj = n_risk as f64;
            let dj = slice.d as f64;
            s *= 1.0 - dj / nj;
            if nj > dj {
                greenwood_sum += dj / (nj * (nj - dj));
            }
            let variance = s * s * greenwood_sum;

            // Log-log CI (complementary log-log transform).
            let (ci_lo, ci_hi) = if s > 0.0 && s < 1.0 {
                let log_h = (-s.ln()).ln();
                let se_log_h = greenwood_sum.sqrt() / s.ln().abs().max(1e-300);
                let lo = (-(log_h + z_alpha * se_log_h).exp()).exp();
                let hi = (-(log_h - z_alpha * se_log_h).exp()).exp();
                (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0))
            } else if s <= 0.0 {
                (0.0, 0.0)
            } else {
                (1.0, 1.0)
            };

            steps.push(KaplanMeierStep {
                time: slice.time,
                n_risk,
                n_events: slice.d,
                n_censored: slice.c,
                survival: s,
                variance,
                ci_lower: ci_lo,
                ci_upper: ci_hi,
            });
        }
        n_risk -= slice.d + slice.c;
    }

    let median = steps.iter().find(|st| st.survival <= 0.5).map(|st| st.time);

    Ok(KaplanMeierEstimate { steps, median, conf_level, n, n_events: total_events })
}

/// Inverse of the standard normal CDF (probit function).
///
/// Rational approximation by Peter Acklam (relative error < 1.15e-9).
fn normal_cdf_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Log-rank test (Mantel-Cox).
// ---------------------------------------------------------------------------

/// Result of the log-rank test comparing survival distributions across groups.
#[derive(Debug, Clone)]
pub struct LogRankResult {
    /// Chi-squared test statistic.
    pub chi_squared: f64,
    /// Degrees of freedom (number of groups − 1).
    pub df: usize,
    /// p-value from the chi-squared distribution.
    pub p_value: f64,
    /// Per-group summaries: `(group_id, observed_events, expected_events)`.
    pub group_summaries: Vec<(i64, f64, f64)>,
    /// Total number of observations.
    pub n: usize,
}

/// Perform the log-rank (Mantel-Cox) test.
///
/// Compares survival distributions of two or more groups. The test statistic
/// follows a chi-squared distribution with `G − 1` degrees of freedom under
/// H₀ (all groups share the same survival function).
///
/// # Arguments
/// - `times` — observation times (≥ 0, finite).
/// - `events` — `true` = event, `false` = right-censored.
/// - `groups` — integer group labels (same length as `times`).
///
/// # Returns
/// A [`LogRankResult`] with the chi-squared statistic, df, and p-value.
pub fn log_rank_test(times: &[f64], events: &[bool], groups: &[i64]) -> Result<LogRankResult> {
    validate_right_censoring_data(times, events)?;
    let n = times.len();
    if groups.len() != n {
        return Err(Error::Validation(format!(
            "groups length ({}) != times length ({})",
            groups.len(),
            n
        )));
    }

    // Identify unique groups in order of first appearance, then sort.
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
    if g_count < 2 {
        return Err(Error::Validation("log-rank test requires at least 2 groups".to_string()));
    }

    // Map group label → index.
    let group_idx: std::collections::HashMap<i64, usize> =
        unique_groups.iter().enumerate().map(|(i, &g)| (g, i)).collect();

    // Sort by time ascending; within ties events before censorings.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a].partial_cmp(&times[b]).unwrap().then_with(|| events[b].cmp(&events[a]))
    });

    // Per-group total counts.
    let mut n_risk_g: Vec<f64> = vec![0.0; g_count];
    for &g in groups {
        n_risk_g[group_idx[&g]] += 1.0;
    }

    // Accumulators for O − E per group.
    let mut observed = vec![0.0_f64; g_count];
    let mut expected = vec![0.0_f64; g_count];
    // Variance-covariance accumulator (for multi-group chi-sq).
    // Only need (G-1)×(G-1) but accumulate full G×G for clarity, then extract.
    let mut v_mat = vec![0.0_f64; g_count * g_count];

    let mut i = 0;
    while i < n {
        let t = times[order[i]];
        // Count events and censorings at this time, per group.
        let mut d_total = 0.0_f64;
        let mut d_g = vec![0.0_f64; g_count];
        let mut c_g = vec![0.0_f64; g_count];
        while i < n && times[order[i]] == t {
            let gi = group_idx[&groups[order[i]]];
            if events[order[i]] {
                d_total += 1.0;
                d_g[gi] += 1.0;
            } else {
                c_g[gi] += 1.0;
            }
            i += 1;
        }
        if d_total == 0.0 {
            // Only censorings — update risk sets and continue.
            for k in 0..g_count {
                n_risk_g[k] -= d_g[k] + c_g[k];
            }
            continue;
        }

        let n_total: f64 = n_risk_g.iter().sum();
        if n_total < 1.0 {
            break;
        }

        for k in 0..g_count {
            let e_k = d_total * n_risk_g[k] / n_total;
            observed[k] += d_g[k];
            expected[k] += e_k;
        }

        // Hypergeometric variance-covariance contribution.
        if n_total > 1.0 {
            let factor = d_total * (n_total - d_total) / (n_total * n_total * (n_total - 1.0));
            for j in 0..g_count {
                for k in 0..g_count {
                    let v = if j == k {
                        n_risk_g[j] * (n_total - n_risk_g[j])
                    } else {
                        -n_risk_g[j] * n_risk_g[k]
                    };
                    v_mat[j * g_count + k] += factor * v;
                }
            }
        }

        // Remove from risk set.
        for k in 0..g_count {
            n_risk_g[k] -= d_g[k] + c_g[k];
        }
    }

    // Compute chi-squared statistic using the first G-1 groups.
    // For 2 groups, this simplifies to (O₁ - E₁)² / V₁₁.
    let chi_squared = if g_count == 2 {
        let diff = observed[0] - expected[0];
        let var = v_mat[0];
        if var > 0.0 { diff * diff / var } else { 0.0 }
    } else {
        // General case: χ² = (O-E)ₜ V⁻¹ (O-E), using first G-1 components.
        let m = g_count - 1;
        // Extract the upper-left (G-1)×(G-1) block.
        let mut v_sub = vec![0.0_f64; m * m];
        for j in 0..m {
            for k in 0..m {
                v_sub[j * m + k] = v_mat[j * g_count + k];
            }
        }
        let diff_sub: Vec<f64> = (0..m).map(|k| observed[k] - expected[k]).collect();
        // Invert V_sub via nalgebra.
        let v_nalg = nalgebra::DMatrix::from_row_slice(m, m, &v_sub);
        match v_nalg.try_inverse() {
            Some(v_inv) => {
                let mut chi2 = 0.0;
                for j in 0..m {
                    for k in 0..m {
                        chi2 += diff_sub[j] * v_inv[(j, k)] * diff_sub[k];
                    }
                }
                chi2
            }
            None => 0.0,
        }
    };

    let df = g_count - 1;
    // p-value from chi-squared distribution.
    let p_value = 1.0 - chi_squared_cdf(chi_squared, df as f64);

    let group_summaries: Vec<(i64, f64, f64)> =
        unique_groups.iter().enumerate().map(|(i, &g)| (g, observed[i], expected[i])).collect();

    Ok(LogRankResult { chi_squared, df, p_value, group_summaries, n })
}

/// Regularised incomplete upper gamma function P(a, x) = γ(a,x)/Γ(a).
/// Used to compute chi-squared CDF: F(x; k) = P(k/2, x/2).
fn chi_squared_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    statrs::function::gamma::gamma_lr(k / 2.0, x / 2.0)
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

    // -----------------------------------------------------------------------
    // Kaplan-Meier tests — reference values from R survival::survfit
    // -----------------------------------------------------------------------
    //
    // R code:
    //   library(survival)
    //   t <- c(1,1,2,2,3,4,4,5,5,8,8,12,12,15,23,27)
    //   e <- c(1,1,1,0,1,1,0,1,0,1,1, 1, 0, 0, 1, 1)
    //   fit <- survfit(Surv(t, e) ~ 1, conf.type="log-log")
    //   summary(fit)

    fn km_test_data() -> (Vec<f64>, Vec<bool>) {
        let times = vec![
            1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 8.0, 8.0, 12.0, 12.0, 15.0, 23.0, 27.0,
        ];
        let events = vec![
            true, true, true, false, true, true, false, true, false, true, true, true, false,
            false, true, true,
        ];
        (times, events)
    }

    #[test]
    fn km_basic_survival_values() {
        let (times, events) = km_test_data();
        let est = kaplan_meier(&times, &events, 0.95).unwrap();

        assert_eq!(est.n, 16);
        assert_eq!(est.n_events, 11);

        // R survfit: S(1) = 0.875, n.risk=16, n.event=2
        let s1 = &est.steps[0];
        assert_eq!(s1.time, 1.0);
        assert_eq!(s1.n_risk, 16);
        assert_eq!(s1.n_events, 2);
        assert!((s1.survival - 0.875).abs() < 1e-10);

        // R survfit: S(2) = 0.8125 (14 at risk, 1 event after removing 1 censored at t=2)
        // Actually: at t=2, n_risk = 16 - 2 = 14, d=1, c=1 → S=0.875*(1-1/14) = 0.8125
        let s2 = &est.steps[1];
        assert_eq!(s2.time, 2.0);
        assert_eq!(s2.n_risk, 14);
        assert_eq!(s2.n_events, 1);
        assert!((s2.survival - 0.8125).abs() < 1e-10);

        // S(3): n_risk=12, d=1 → 0.8125*(1−1/12) = 0.8125*11/12
        let s3 = &est.steps[2];
        assert_eq!(s3.time, 3.0);
        assert_eq!(s3.n_risk, 12);
        let expected_s3 = 0.8125 * 11.0 / 12.0;
        assert!(
            (s3.survival - expected_s3).abs() < 1e-10,
            "S(3): got {}, expected {}",
            s3.survival,
            expected_s3
        );
    }

    #[test]
    fn km_greenwood_variance() {
        let (times, events) = km_test_data();
        let est = kaplan_meier(&times, &events, 0.95).unwrap();

        // R: Var[S(1)] = S(1)^2 * sum(d/(n*(n-d))) = 0.875^2 * (2/(16*14))
        //   = 0.765625 * 0.00892857 = 0.006835937
        let v1 = est.steps[0].variance;
        assert!(
            (v1 - 0.006835937).abs() < 1e-6,
            "Greenwood var at t=1: got {v1}, expected 0.006835937"
        );

        // Cumulative Greenwood sum at t=2: 2/(16*14) + 1/(14*13)
        let gw_sum = 2.0 / (16.0 * 14.0) + 1.0 / (14.0 * 13.0);
        let v2_expected = 0.8125_f64.powi(2) * gw_sum;
        let v2 = est.steps[1].variance;
        assert!(
            (v2 - v2_expected).abs() < 1e-9,
            "Greenwood var at t=2: got {v2}, expected {v2_expected}"
        );
    }

    #[test]
    fn km_median_survival() {
        let (times, events) = km_test_data();
        let est = kaplan_meier(&times, &events, 0.95).unwrap();
        // Median = first time S(t) <= 0.5. From R: median = 8.
        assert_eq!(est.median, Some(8.0));
    }

    #[test]
    fn km_ci_bounds_monotone_and_bracketing() {
        let (times, events) = km_test_data();
        let est = kaplan_meier(&times, &events, 0.95).unwrap();

        for step in &est.steps {
            assert!(
                step.ci_lower <= step.survival,
                "CI lower ({}) > survival ({}) at t={}",
                step.ci_lower,
                step.survival,
                step.time
            );
            assert!(
                step.ci_upper >= step.survival,
                "CI upper ({}) < survival ({}) at t={}",
                step.ci_upper,
                step.survival,
                step.time
            );
            assert!(
                step.ci_lower >= 0.0 && step.ci_upper <= 1.0,
                "CI out of [0,1] at t={}",
                step.time
            );
        }
    }

    #[test]
    fn km_no_events_returns_empty_steps() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![false, false, false];
        let est = kaplan_meier(&times, &events, 0.95).unwrap();
        assert!(est.steps.is_empty());
        assert_eq!(est.median, None);
        assert_eq!(est.n_events, 0);
    }

    #[test]
    fn km_all_events_no_censoring() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, true, true, true];
        let est = kaplan_meier(&times, &events, 0.95).unwrap();
        assert_eq!(est.steps.len(), 5);
        // S(1) = 4/5 = 0.8, S(2)=3/5=0.6, S(3)=2/5=0.4, S(4)=1/5=0.2, S(5)=0
        assert!((est.steps[0].survival - 0.8).abs() < 1e-10);
        assert!((est.steps[1].survival - 0.6).abs() < 1e-10);
        assert!((est.steps[2].survival - 0.4).abs() < 1e-10);
        assert!((est.steps[3].survival - 0.2).abs() < 1e-10);
        assert!((est.steps[4].survival - 0.0).abs() < 1e-10);
        // Median should be 3.0 (first time S <= 0.5)
        assert_eq!(est.median, Some(3.0));
    }

    #[test]
    fn km_single_event() {
        let times = vec![5.0];
        let events = vec![true];
        let est = kaplan_meier(&times, &events, 0.95).unwrap();
        assert_eq!(est.steps.len(), 1);
        assert!((est.steps[0].survival - 0.0).abs() < 1e-10);
    }

    #[test]
    fn km_validation_errors() {
        assert!(kaplan_meier(&[], &[], 0.95).is_err());
        assert!(kaplan_meier(&[1.0], &[true, false], 0.95).is_err());
        assert!(kaplan_meier(&[-1.0], &[true], 0.95).is_err());
        assert!(kaplan_meier(&[1.0], &[true], 0.0).is_err());
        assert!(kaplan_meier(&[1.0], &[true], 1.0).is_err());
    }

    #[test]
    fn km_conf_level_90() {
        let (times, events) = km_test_data();
        let est90 = kaplan_meier(&times, &events, 0.90).unwrap();
        let est95 = kaplan_meier(&times, &events, 0.95).unwrap();
        // 90% CI should be narrower than 95% CI at every step.
        for (s90, s95) in est90.steps.iter().zip(est95.steps.iter()) {
            assert!(
                s90.ci_upper - s90.ci_lower <= s95.ci_upper - s95.ci_lower + 1e-12,
                "90% CI wider than 95% CI at t={}",
                s90.time
            );
        }
    }

    // -----------------------------------------------------------------------
    // Log-rank tests — reference values from R survival::survdiff
    // -----------------------------------------------------------------------
    //
    // R code:
    //   t <- c(6,6,6,7,10, 13,16,22,23, 6,9,10,11,17,19,20,25,32,32,34,35)
    //   e <- c(1,1,1,1,0,  1, 1, 1, 1,  1,0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0)
    //   g <- c(1,1,1,1,1,  1, 1, 1, 1,  2,2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    //   survdiff(Surv(t, e) ~ g)
    //   # N=21, Chisq=3.4 on 1 df, p=0.0653

    fn logrank_test_data() -> (Vec<f64>, Vec<bool>, Vec<i64>) {
        // R's aml dataset: survdiff(Surv(time, status) ~ x, data=aml)
        // Group 1 = Maintained (11 patients, 7 events)
        // Group 2 = Nonmaintained (12 patients, 10 events)
        // R output: Chisq=3.4 on 1 df, p=0.0653
        let times = vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0, 5.0, 5.0, 8.0, 8.0,
            12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
        ];
        let events = vec![
            true, true, false, true, true, false, true, true, false, true, false, true, true, true,
            true, true, false, true, true, true, true, true, true,
        ];
        let groups = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        (times, events, groups)
    }

    #[test]
    fn logrank_two_groups_chisq() {
        let (times, events, groups) = logrank_test_data();
        let res = log_rank_test(&times, &events, &groups).unwrap();

        assert_eq!(res.df, 1);
        assert_eq!(res.n, 23);
        // R survdiff(Surv(time, status) ~ x, data=aml):
        //   Chisq=3.4 on 1 df, p=0.0653
        //   Maintained:    O=7,  E=10.69
        //   Nonmaintained: O=10, E=6.31
        assert!(
            (res.chi_squared - 3.4).abs() < 0.15,
            "chi_squared: got {}, expected ~3.4",
            res.chi_squared
        );
        assert!(
            res.p_value > 0.05 && res.p_value < 0.10,
            "p_value: got {}, expected ~0.065",
            res.p_value
        );

        // Verify per-group observed/expected against R.
        let (_, o1, e1) = res.group_summaries[0];
        let (_, o2, e2) = res.group_summaries[1];
        assert!((o1 - 7.0).abs() < 1e-10, "O1: got {o1}, expected 7");
        assert!((o2 - 11.0).abs() < 1e-10, "O2: got {o2}, expected 11");
        assert!((e1 - 10.69).abs() < 0.1, "E1: got {e1}, expected ~10.69");
        assert!((e2 - 7.31).abs() < 0.1, "E2: got {e2}, expected ~7.31");
    }

    #[test]
    fn logrank_observed_expected_sums() {
        let (times, events, groups) = logrank_test_data();
        let res = log_rank_test(&times, &events, &groups).unwrap();

        // Sum of observed should equal total events.
        let total_obs: f64 = res.group_summaries.iter().map(|(_, o, _)| o).sum();
        let total_events = events.iter().filter(|&&e| e).count() as f64;
        assert!((total_obs - total_events).abs() < 1e-10);

        // Sum of expected should also equal total events.
        let total_exp: f64 = res.group_summaries.iter().map(|(_, _, e)| e).sum();
        assert!(
            (total_exp - total_events).abs() < 1e-6,
            "sum(E) = {total_exp}, total events = {total_events}"
        );
    }

    #[test]
    fn logrank_identical_groups_nonsignificant() {
        // If both groups have the same survival, chi-sq ≈ 0.
        let times = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let events = vec![true, true, true, true, true, true, true, true];
        let groups = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let res = log_rank_test(&times, &events, &groups).unwrap();
        assert!(res.chi_squared < 1e-10, "chi_squared should be ~0 for identical groups");
        assert!(res.p_value > 0.99);
    }

    #[test]
    fn logrank_very_different_groups() {
        // Group 1: all events early. Group 2: all censored early, events late.
        let times = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let events = vec![true, true, true, true, true, true];
        let groups = vec![1, 1, 1, 2, 2, 2];
        let res = log_rank_test(&times, &events, &groups).unwrap();
        assert!(res.chi_squared > 2.0, "chi_squared should be substantial");
        assert!(res.p_value < 0.20);
    }

    #[test]
    fn logrank_three_groups() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let events = vec![true, true, true, true, true, true, true, true, true];
        let groups = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
        let res = log_rank_test(&times, &events, &groups).unwrap();
        assert_eq!(res.df, 2);
        assert_eq!(res.group_summaries.len(), 3);
    }

    #[test]
    fn logrank_validation_errors() {
        assert!(log_rank_test(&[], &[], &[]).is_err());
        assert!(log_rank_test(&[1.0], &[true], &[1]).is_err()); // only 1 group
        assert!(log_rank_test(&[1.0, 2.0], &[true], &[1, 2]).is_err()); // length mismatch
    }

    #[test]
    fn normal_cdf_inv_basic() {
        // Check a few known quantiles.
        assert!((normal_cdf_inv(0.5) - 0.0).abs() < 1e-8);
        assert!((normal_cdf_inv(0.975) - 1.959964).abs() < 1e-4);
        assert!((normal_cdf_inv(0.025) - (-1.959964)).abs() < 1e-4);
        assert!((normal_cdf_inv(0.99) - 2.326348).abs() < 1e-4);
    }

    #[test]
    fn chi_squared_cdf_basic() {
        // chi-sq CDF(3.841, df=1) ≈ 0.95
        let p = chi_squared_cdf(3.841459, 1.0);
        assert!((p - 0.95).abs() < 1e-4, "chi_squared_cdf(3.84, 1) = {p}, expected ~0.95");
        // chi-sq CDF(5.991, df=2) ≈ 0.95
        let p2 = chi_squared_cdf(5.991465, 2.0);
        assert!((p2 - 0.95).abs() < 1e-4, "chi_squared_cdf(5.99, 2) = {p2}, expected ~0.95");
    }
}
