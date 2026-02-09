use crate::event_store::EventStore;
use crate::math::log_diff_exp;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Exponential-family PDF `p(x) ∝ exp(λ x)` normalized on the observable bounds.
///
/// This is a convenient HEP background shape; unlike the "rate-parameter exponential",
/// `λ` is allowed to be any real number.
#[derive(Debug, Clone)]
pub struct ExponentialPdf {
    observables: [String; 1],
}

impl ExponentialPdf {
    /// Create a new exponential PDF over the given observable.
    pub fn new(observable: impl Into<String>) -> Self {
        Self { observables: [observable.into()] }
    }
}

impl UnbinnedPdf for ExponentialPdf {
    fn n_params(&self) -> usize {
        1
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        let mut tmp_grad = vec![0.0f64; events.n_events() * self.n_params()];
        self.log_prob_grad_batch(events, params, out, &mut tmp_grad)
    }

    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()> {
        if params.len() != 1 {
            return Err(Error::Validation(format!(
                "ExponentialPdf expects 1 param (lambda), got {}",
                params.len()
            )));
        }
        let lambda = params[0];
        if !lambda.is_finite() {
            return Err(Error::Validation(format!(
                "ExponentialPdf parameter must be finite, got lambda={lambda}"
            )));
        }

        let n = events.n_events();
        if out_logp.len() != n {
            return Err(Error::Validation(format!(
                "ExponentialPdf out_logp length mismatch: expected {n}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n * self.n_params();
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "ExponentialPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
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

        // logZ = log ∫_a^b exp(λ x) dx
        let (log_z, ex) = logz_and_ex(lambda, a, b)?;

        for (i, &x) in xs.iter().enumerate() {
            out_logp[i] = lambda * x - log_z;
            out_grad[i] = x - ex;
        }

        Ok(())
    }
}

/// Return `(logZ, E[x])` for the bounded exponential family `p(x) ∝ exp(λ x)` on `[a,b]`.
fn logz_and_ex(lambda: f64, a: f64, b: f64) -> Result<(f64, f64)> {
    debug_assert!(a < b);

    // Small-λ limit: uniform on [a,b].
    if lambda.abs() < 1e-12 {
        let z = b - a;
        if !(z.is_finite() && z > 0.0) {
            return Err(Error::Validation(format!(
                "invalid bounds for ExponentialPdf: ({a}, {b})"
            )));
        }
        let log_z = z.ln();
        let ex = 0.5 * (a + b);
        return Ok((log_z, ex));
    }

    let t_a = lambda * a;
    let t_b = lambda * b;

    // log|exp(t_b) - exp(t_a)|
    let (hi_t, lo_t) = if t_b >= t_a { (t_b, t_a) } else { (t_a, t_b) };
    let log_num = if hi_t == lo_t {
        // Should be unreachable unless a==b or lambda==0 (handled above).
        f64::NEG_INFINITY
    } else {
        log_diff_exp(hi_t, lo_t)
    };
    let log_z = log_num - lambda.abs().ln();

    // E[x] = d/dλ logZ = (x_hi - x_lo*r)/(1-r) - 1/λ, where r=exp(lo-hi).
    let (x_hi, x_lo, r) =
        if t_b >= t_a { (b, a, (t_a - t_b).exp()) } else { (a, b, (t_b - t_a).exp()) };
    let denom = 1.0 - r;
    if denom <= 0.0 {
        // Numerically indistinguishable from uniform.
        return Ok(((b - a).ln(), 0.5 * (a + b)));
    }
    let ratio = (x_hi - x_lo * r) / denom;
    let ex = ratio - 1.0 / lambda;

    Ok((log_z, ex))
}
