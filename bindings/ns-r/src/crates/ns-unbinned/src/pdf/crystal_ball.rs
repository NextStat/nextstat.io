use crate::event_store::EventStore;
use crate::math::standard_normal_cdf;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// One-sided Crystal Ball PDF (Gaussian core with a power-law tail), normalized on the observable
/// bounds.
///
/// Parameterization matches RooFit's `RooCBShape` (left tail):
/// - `mu`: location
/// - `sigma`: width (must be > 0)
/// - `alpha`: transition point in units of sigma (must be > 0)
/// - `n`: tail exponent (must be > 1)
#[derive(Debug, Clone)]
pub struct CrystalBallPdf {
    observables: [String; 1],
}

impl CrystalBallPdf {
    /// Create a new Crystal Ball PDF over the given observable.
    pub fn new(observable: impl Into<String>) -> Self {
        Self { observables: [observable.into()] }
    }
}

/// Two-sided Crystal Ball PDF (DoubleCB), normalized on the observable bounds.
///
/// Parameters:
/// - `mu`, `sigma`
/// - `alpha_l`, `n_l` for the left tail (must be `> 0`, `> 1`)
/// - `alpha_r`, `n_r` for the right tail (must be `> 0`, `> 1`)
#[derive(Debug, Clone)]
pub struct DoubleCrystalBallPdf {
    observables: [String; 1],
}

impl DoubleCrystalBallPdf {
    /// Create a new Double Crystal Ball PDF over the given observable.
    pub fn new(observable: impl Into<String>) -> Self {
        Self { observables: [observable.into()] }
    }
}

#[derive(Debug, Clone, Copy)]
struct CbTail {
    alpha: f64,
    n: f64,
    log_a: f64,
    b: f64,
}

impl CbTail {
    fn new(alpha: f64, n: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBall alpha must be finite and > 0, got {alpha}"
            )));
        }
        if !n.is_finite() || n <= 1.0 {
            return Err(Error::Validation(format!(
                "CrystalBall n must be finite and > 1, got {n}"
            )));
        }

        let log_a = n * (n / alpha).ln() - 0.5 * alpha * alpha;
        let b = n / alpha - alpha;

        Ok(Self { alpha, n, log_a, b })
    }

    #[inline]
    fn t_boundary_left(&self) -> f64 {
        -self.alpha
    }

    #[inline]
    fn t_boundary_right(&self) -> f64 {
        self.alpha
    }

    #[inline]
    fn logf_left(&self, t: f64) -> f64 {
        // Tail applies for t <= -alpha.
        self.log_a - self.n * (self.b - t).ln()
    }

    #[inline]
    fn logf_right(&self, t: f64) -> f64 {
        // Tail applies for t >= +alpha.
        self.log_a - self.n * (self.b + t).ln()
    }

    #[inline]
    fn dlogf_dt_left(&self, t: f64) -> f64 {
        self.n / (self.b - t)
    }

    #[inline]
    fn dlogf_dt_right(&self, t: f64) -> f64 {
        -self.n / (self.b + t)
    }

    #[inline]
    fn dlogf_dalpha_left(&self, t: f64) -> f64 {
        let alpha = self.alpha;
        let n = self.n;

        let dln_a = -(n / alpha + alpha);
        let db = -(n / (alpha * alpha) + 1.0);

        dln_a - n * db / (self.b - t)
    }

    #[inline]
    fn dlogf_dn_left(&self, t: f64) -> f64 {
        let alpha = self.alpha;
        let n = self.n;

        let dln_a = 1.0 + (n / alpha).ln();
        let db = 1.0 / alpha;

        dln_a - (self.b - t).ln() - n * db / (self.b - t)
    }

    #[inline]
    fn dlogf_dalpha_right(&self, t: f64) -> f64 {
        let alpha = self.alpha;
        let n = self.n;

        let dln_a = -(n / alpha + alpha);
        let db = -(n / (alpha * alpha) + 1.0);

        dln_a - n * db / (self.b + t)
    }

    #[inline]
    fn dlogf_dn_right(&self, t: f64) -> f64 {
        let alpha = self.alpha;
        let n = self.n;

        let dln_a = 1.0 + (n / alpha).ln();
        let db = 1.0 / alpha;

        dln_a - (self.b + t).ln() - n * db / (self.b + t)
    }

    /// Tail integral on the left side for `t in [t1, t2]` where `t2 <= -alpha`.
    fn integral_left(&self, t1: f64, t2: f64) -> Result<f64> {
        debug_assert!(t1 <= t2);
        let m = self.n - 1.0;
        let a = self.log_a.exp();
        let u2 = (self.b - t2).powf(-m);
        let u1 = (self.b - t1).powf(-m);
        let i = a / m * (u2 - u1);
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBall tail integral (left) is not finite/positive: {i}"
            )));
        }
        Ok(i)
    }

    /// Tail integral on the right side for `t in [t1, t2]` where `t1 >= +alpha`.
    fn integral_right(&self, t1: f64, t2: f64) -> Result<f64> {
        debug_assert!(t1 <= t2);
        let m = self.n - 1.0;
        let a = self.log_a.exp();
        let u1 = (self.b + t1).powf(-m);
        let u2 = (self.b + t2).powf(-m);
        let i = a / m * (u1 - u2);
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBall tail integral (right) is not finite/positive: {i}"
            )));
        }
        Ok(i)
    }

    fn integral_and_derivatives_left(&self, t1: f64, t2: f64) -> Result<(f64, f64, f64)> {
        debug_assert!(t1 <= t2);

        let alpha = self.alpha;
        let n = self.n;
        let m = n - 1.0;

        let a = self.log_a.exp();

        let b1 = self.b - t1;
        let b2 = self.b - t2;
        let u1 = b1.powf(-m);
        let u2 = b2.powf(-m);
        let i = a / m * (u2 - u1);
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBall tail integral (left) is not finite/positive: {i}"
            )));
        }

        let dln_a_dalpha = -(n / alpha + alpha);
        let dln_a_dn = 1.0 + (n / alpha).ln();
        let db_dalpha = -(n / (alpha * alpha) + 1.0);
        let db_dn = 1.0 / alpha;

        let v1 = b1.powf(-n);
        let v2 = b2.powf(-n);

        let di_dalpha = i * dln_a_dalpha - a * db_dalpha * (v2 - v1);

        let du1_dn = u1 * (-b1.ln() - m * db_dn / b1);
        let du2_dn = u2 * (-b2.ln() - m * db_dn / b2);
        let di_dn = i * dln_a_dn - i / m + (a / m) * (du2_dn - du1_dn);

        Ok((i, di_dalpha, di_dn))
    }

    fn integral_and_derivatives_right(&self, t1: f64, t2: f64) -> Result<(f64, f64, f64)> {
        debug_assert!(t1 <= t2);

        let alpha = self.alpha;
        let n = self.n;
        let m = n - 1.0;

        let a = self.log_a.exp();

        let b1 = self.b + t1;
        let b2 = self.b + t2;
        let u1 = b1.powf(-m);
        let u2 = b2.powf(-m);
        let i = a / m * (u1 - u2);
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBall tail integral (right) is not finite/positive: {i}"
            )));
        }

        let dln_a_dalpha = -(n / alpha + alpha);
        let dln_a_dn = 1.0 + (n / alpha).ln();
        let db_dalpha = -(n / (alpha * alpha) + 1.0);
        let db_dn = 1.0 / alpha;

        let v1 = b1.powf(-n);
        let v2 = b2.powf(-n);

        let di_dalpha = i * dln_a_dalpha - a * db_dalpha * (v1 - v2);

        let du1_dn = u1 * (-b1.ln() - m * db_dn / b1);
        let du2_dn = u2 * (-b2.ln() - m * db_dn / b2);
        let di_dn = i * dln_a_dn - i / m + (a / m) * (du1_dn - du2_dn);

        Ok((i, di_dalpha, di_dn))
    }
}

#[inline]
fn gauss_logf(t: f64) -> f64 {
    -0.5 * t * t
}

#[inline]
fn gauss_integral(t1: f64, t2: f64) -> f64 {
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    sqrt_2pi * (standard_normal_cdf(t2) - standard_normal_cdf(t1))
}

impl UnbinnedPdf for CrystalBallPdf {
    fn n_params(&self) -> usize {
        4
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn pdf_tag(&self) -> &'static str {
        "crystal_ball"
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if params.len() != 4 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf expects 4 params (mu, sigma, alpha, n), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        let alpha = params[2];
        let n = params[3];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let tail = CbTail::new(alpha, n)?;

        let n_events = events.n_events();
        if out.len() != n_events {
            return Err(Error::Validation(format!(
                "CrystalBallPdf out length mismatch: expected {n_events}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        if a.partial_cmp(&b) != Some(core::cmp::Ordering::Less) {
            return Err(Error::Validation(format!(
                "invalid bounds for '{obs}': expected low < high, got ({a}, {b})"
            )));
        }

        let inv_sigma = 1.0 / sigma;
        let t_a = (a - mu) * inv_sigma;
        let t_b = (b - mu) * inv_sigma;
        let t0 = tail.t_boundary_left();

        let i = if t_b <= t0 {
            tail.integral_left(t_a, t_b)?
        } else if t_a >= t0 {
            gauss_integral(t_a, t_b)
        } else {
            tail.integral_left(t_a, t0)? + gauss_integral(t0, t_b)
        };
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf normalization integral is not finite/positive: {i}"
            )));
        }
        let log_i = i.ln();

        for (i_evt, &x) in xs.iter().enumerate() {
            let t = (x - mu) * inv_sigma;
            let logf = if t > t0 { gauss_logf(t) } else { tail.logf_left(t) };
            out[i_evt] = logf - sigma.ln() - log_i;
        }

        Ok(())
    }

    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()> {
        if params.len() != 4 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf expects 4 params (mu, sigma, alpha, n), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        let alpha = params[2];
        let n = params[3];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let tail = CbTail::new(alpha, n)?;

        let n_events = events.n_events();
        if out_logp.len() != n_events {
            return Err(Error::Validation(format!(
                "CrystalBallPdf out_logp length mismatch: expected {n_events}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n_events * self.n_params();
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "CrystalBallPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
                out_grad.len()
            )));
        }

        let obs = self.observables[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        if a.partial_cmp(&b) != Some(core::cmp::Ordering::Less) {
            return Err(Error::Validation(format!(
                "invalid bounds for '{obs}': expected low < high, got ({a}, {b})"
            )));
        }

        let inv_sigma = 1.0 / sigma;
        let t_a = (a - mu) * inv_sigma;
        let t_b = (b - mu) * inv_sigma;
        let t0 = tail.t_boundary_left();

        let (i, di_dalpha, di_dn) = if t_b <= t0 {
            tail.integral_and_derivatives_left(t_a, t_b)?
        } else if t_a >= t0 {
            (gauss_integral(t_a, t_b), 0.0, 0.0)
        } else {
            let (it, dit_da, dit_dn) = tail.integral_and_derivatives_left(t_a, t0)?;
            (it + gauss_integral(t0, t_b), dit_da, dit_dn)
        };
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf normalization integral is not finite/positive: {i}"
            )));
        }

        let log_i = i.ln();
        let dlogi_dalpha = di_dalpha / i;
        let dlogi_dn = di_dn / i;

        let logf_a = if t_a > t0 { gauss_logf(t_a) } else { tail.logf_left(t_a) };
        let logf_b = if t_b > t0 { gauss_logf(t_b) } else { tail.logf_left(t_b) };
        let f_a = logf_a.exp();
        let f_b = logf_b.exp();

        // Endpoint derivatives for mu/sigma (Leibniz rule).
        let dlogi_dmu = (f_a - f_b) * inv_sigma / i;
        let dlogi_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma / i;

        for (i_evt, &x) in xs.iter().enumerate() {
            let t = (x - mu) * inv_sigma;
            let is_gauss = t > t0;

            let (logf, dlogf_dt, dlogf_dalpha, dlogf_dn) = if is_gauss {
                (gauss_logf(t), -t, 0.0, 0.0)
            } else {
                (
                    tail.logf_left(t),
                    tail.dlogf_dt_left(t),
                    tail.dlogf_dalpha_left(t),
                    tail.dlogf_dn_left(t),
                )
            };

            out_logp[i_evt] = logf - sigma.ln() - log_i;

            // mu
            let d_mu = -inv_sigma * dlogf_dt - dlogi_dmu;
            // sigma
            let d_sigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlogi_dsigma;
            // alpha, n
            let d_alpha = dlogf_dalpha - dlogi_dalpha;
            let d_n = dlogf_dn - dlogi_dn;

            let base = i_evt * 4;
            out_grad[base] = d_mu;
            out_grad[base + 1] = d_sigma;
            out_grad[base + 2] = d_alpha;
            out_grad[base + 3] = d_n;
        }

        Ok(())
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn rand::RngCore,
    ) -> Result<EventStore> {
        if params.len() != 4 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf expects 4 params (mu, sigma, alpha, n), got {}",
                params.len()
            )));
        }
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        let alpha = params[2];
        let n = params[3];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "CrystalBallPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let tail = CbTail::new(alpha, n)?;
        let (a, b) = support[0];
        if !a.is_finite() || !b.is_finite() || a >= b {
            return Err(Error::Validation(format!(
                "CrystalBallPdf sample requires finite support with low < high, got ({a}, {b})"
            )));
        }

        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            let v = rng.next_u64();
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        let inv_sigma = 1.0 / sigma;
        let t_a = (a - mu) * inv_sigma;
        let t_b = (b - mu) * inv_sigma;
        if t_a >= t_b {
            return Err(Error::Validation(format!(
                "CrystalBallPdf sample has degenerate t-range: t_a={t_a}, t_b={t_b}"
            )));
        }
        let t0 = tail.t_boundary_left();

        // Rejection sampling from uniform on [t_a, t_b] with envelope 1.0. The unnormalized CB
        // shape satisfies f(t) <= 1 (maximum at t=0).
        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let mut tries = 0usize;
            loop {
                let t = t_a + (t_b - t_a) * u01(rng);
                let f = if t > t0 { (-0.5 * t * t).exp() } else { tail.logf_left(t).exp() };
                let u = u01(rng);
                if u <= f {
                    xs.push((mu + sigma * t).clamp(a, b));
                    break;
                }
                tries += 1;
                if tries > 100_000 {
                    return Err(Error::Computation(
                        "CrystalBallPdf sample rejection loop exceeded max iterations".into(),
                    ));
                }
            }
        }

        let obs = crate::event_store::ObservableSpec::branch(self.observables[0].clone(), (a, b));
        EventStore::from_columns(vec![obs], vec![(self.observables[0].clone(), xs)], None)
    }
}

impl UnbinnedPdf for DoubleCrystalBallPdf {
    fn n_params(&self) -> usize {
        6
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if params.len() != 6 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf expects 6 params (mu, sigma, alpha_l, n_l, alpha_r, n_r), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let left = CbTail::new(params[2], params[3])?;
        let right = CbTail::new(params[4], params[5])?;

        let n_events = events.n_events();
        if out.len() != n_events {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf out length mismatch: expected {n_events}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        if a.partial_cmp(&b) != Some(core::cmp::Ordering::Less) {
            return Err(Error::Validation(format!(
                "invalid bounds for '{obs}': expected low < high, got ({a}, {b})"
            )));
        }

        let inv_sigma = 1.0 / sigma;
        let t_a = (a - mu) * inv_sigma;
        let t_b = (b - mu) * inv_sigma;

        let t_l = left.t_boundary_left();
        let t_r = right.t_boundary_right();

        let mut i = 0.0f64;
        if t_a < t_l {
            let t2 = t_b.min(t_l);
            i += left.integral_left(t_a, t2)?;
        }
        // Core: intersection with [t_l, t_r]
        let core_lo = t_a.max(t_l);
        let core_hi = t_b.min(t_r);
        if core_hi > core_lo {
            i += gauss_integral(core_lo, core_hi);
        }
        if t_b > t_r {
            let t1 = t_a.max(t_r);
            i += right.integral_right(t1, t_b)?;
        }

        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf normalization integral is not finite/positive: {i}"
            )));
        }
        let log_i = i.ln();

        for (i_evt, &x) in xs.iter().enumerate() {
            let t = (x - mu) * inv_sigma;
            let logf = if t < t_l {
                left.logf_left(t)
            } else if t > t_r {
                right.logf_right(t)
            } else {
                gauss_logf(t)
            };
            out[i_evt] = logf - sigma.ln() - log_i;
        }

        Ok(())
    }

    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()> {
        if params.len() != 6 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf expects 6 params (mu, sigma, alpha_l, n_l, alpha_r, n_r), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let left = CbTail::new(params[2], params[3])?;
        let right = CbTail::new(params[4], params[5])?;

        let n_events = events.n_events();
        if out_logp.len() != n_events {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf out_logp length mismatch: expected {n_events}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n_events * self.n_params();
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
                out_grad.len()
            )));
        }

        let obs = self.observables[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        if a.partial_cmp(&b) != Some(core::cmp::Ordering::Less) {
            return Err(Error::Validation(format!(
                "invalid bounds for '{obs}': expected low < high, got ({a}, {b})"
            )));
        }

        let inv_sigma = 1.0 / sigma;
        let t_a = (a - mu) * inv_sigma;
        let t_b = (b - mu) * inv_sigma;

        let t_l = left.t_boundary_left();
        let t_r = right.t_boundary_right();

        let mut i = 0.0f64;
        let mut di_dalpha_l = 0.0f64;
        let mut di_dn_l = 0.0f64;
        let mut di_dalpha_r = 0.0f64;
        let mut di_dn_r = 0.0f64;

        if t_a < t_l {
            let t2 = t_b.min(t_l);
            let (it, dit_da, dit_dn) = left.integral_and_derivatives_left(t_a, t2)?;
            i += it;
            di_dalpha_l += dit_da;
            di_dn_l += dit_dn;
        }

        let core_lo = t_a.max(t_l);
        let core_hi = t_b.min(t_r);
        if core_hi > core_lo {
            i += gauss_integral(core_lo, core_hi);
        }

        if t_b > t_r {
            let t1 = t_a.max(t_r);
            let (it, dit_da, dit_dn) = right.integral_and_derivatives_right(t1, t_b)?;
            i += it;
            di_dalpha_r += dit_da;
            di_dn_r += dit_dn;
        }

        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf normalization integral is not finite/positive: {i}"
            )));
        }
        let log_i = i.ln();

        let dlogi_dalpha_l = di_dalpha_l / i;
        let dlogi_dn_l = di_dn_l / i;
        let dlogi_dalpha_r = di_dalpha_r / i;
        let dlogi_dn_r = di_dn_r / i;

        let logf_a = if t_a < t_l {
            left.logf_left(t_a)
        } else if t_a > t_r {
            right.logf_right(t_a)
        } else {
            gauss_logf(t_a)
        };
        let logf_b = if t_b < t_l {
            left.logf_left(t_b)
        } else if t_b > t_r {
            right.logf_right(t_b)
        } else {
            gauss_logf(t_b)
        };
        let f_a = logf_a.exp();
        let f_b = logf_b.exp();
        let dlogi_dmu = (f_a - f_b) * inv_sigma / i;
        let dlogi_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma / i;

        for (i_evt, &x) in xs.iter().enumerate() {
            let t = (x - mu) * inv_sigma;
            let (logf, dlogf_dt, dlogf_dalpha_l, dlogf_dn_l, dlogf_dalpha_r, dlogf_dn_r) =
                if t < t_l {
                    (
                        left.logf_left(t),
                        left.dlogf_dt_left(t),
                        left.dlogf_dalpha_left(t),
                        left.dlogf_dn_left(t),
                        0.0,
                        0.0,
                    )
                } else if t > t_r {
                    (
                        right.logf_right(t),
                        right.dlogf_dt_right(t),
                        0.0,
                        0.0,
                        right.dlogf_dalpha_right(t),
                        right.dlogf_dn_right(t),
                    )
                } else {
                    (gauss_logf(t), -t, 0.0, 0.0, 0.0, 0.0)
                };

            out_logp[i_evt] = logf - sigma.ln() - log_i;

            let d_mu = -inv_sigma * dlogf_dt - dlogi_dmu;
            let d_sigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlogi_dsigma;

            let d_alpha_l = dlogf_dalpha_l - dlogi_dalpha_l;
            let d_n_l = dlogf_dn_l - dlogi_dn_l;
            let d_alpha_r = dlogf_dalpha_r - dlogi_dalpha_r;
            let d_n_r = dlogf_dn_r - dlogi_dn_r;

            let base = i_evt * 6;
            out_grad[base] = d_mu;
            out_grad[base + 1] = d_sigma;
            out_grad[base + 2] = d_alpha_l;
            out_grad[base + 3] = d_n_l;
            out_grad[base + 4] = d_alpha_r;
            out_grad[base + 5] = d_n_r;
        }

        Ok(())
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn rand::RngCore,
    ) -> Result<EventStore> {
        if params.len() != 6 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf expects 6 params (mu, sigma, alpha_l, n_l, alpha_r, n_r), got {}",
                params.len()
            )));
        }
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let left = CbTail::new(params[2], params[3])?;
        let right = CbTail::new(params[4], params[5])?;
        let (a, b) = support[0];
        if !a.is_finite() || !b.is_finite() || a >= b {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf sample requires finite support with low < high, got ({a}, {b})"
            )));
        }

        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            let v = rng.next_u64();
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        let inv_sigma = 1.0 / sigma;
        let t_a = (a - mu) * inv_sigma;
        let t_b = (b - mu) * inv_sigma;
        if t_a >= t_b {
            return Err(Error::Validation(format!(
                "DoubleCrystalBallPdf sample has degenerate t-range: t_a={t_a}, t_b={t_b}"
            )));
        }
        let t_l = left.t_boundary_left();
        let t_r = right.t_boundary_right();

        // Rejection sampling from uniform on [t_a, t_b] with envelope 1.0.
        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let mut tries = 0usize;
            loop {
                let t = t_a + (t_b - t_a) * u01(rng);
                let f = if t < t_l {
                    left.logf_left(t).exp()
                } else if t > t_r {
                    right.logf_right(t).exp()
                } else {
                    (-0.5 * t * t).exp()
                };
                let u = u01(rng);
                if u <= f {
                    xs.push((mu + sigma * t).clamp(a, b));
                    break;
                }
                tries += 1;
                if tries > 100_000 {
                    return Err(Error::Computation(
                        "DoubleCrystalBallPdf sample rejection loop exceeded max iterations".into(),
                    ));
                }
            }
        }

        let obs = crate::event_store::ObservableSpec::branch(self.observables[0].clone(), (a, b));
        EventStore::from_columns(vec![obs], vec![(self.observables[0].clone(), xs)], None)
    }
}
