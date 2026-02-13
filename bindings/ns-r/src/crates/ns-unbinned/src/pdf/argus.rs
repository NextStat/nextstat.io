use crate::event_store::EventStore;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// ARGUS background PDF, commonly used in B-meson mass spectroscopy.
///
/// The ARGUS function describes the shape of combinatorial background in the invariant
/// mass distribution near a kinematic threshold (e.g., `m_max = m(Υ(4S))/2`).
///
/// Shape: `f(x; m₀, c, p) = x · (1 - (x/m₀)²)^p · exp(c · (1 - (x/m₀)²))`
///
/// where:
/// - `m₀` (cutoff): upper kinematic limit (fixed or floating)
/// - `c` (shape): curvature parameter (typically `c < 0`)
/// - `p` (power): power parameter (default `p = 0.5` for classic ARGUS)
///
/// The PDF is defined on `[0, m₀]` and normalized numerically.
///
/// **Shape parameters (2):** `[c, p]`. The cutoff `m₀` is taken from the EventStore bounds.
pub struct ArgusPdf {
    observable: [String; 1],
}

impl ArgusPdf {
    /// Create an ARGUS PDF over the given observable.
    ///
    /// The upper cutoff `m₀` is taken from the EventStore upper bound at evaluation time.
    pub fn new(observable: impl Into<String>) -> Self {
        Self { observable: [observable.into()] }
    }

    /// Log of the unnormalized ARGUS density, or NEG_INFINITY if out of support.
    #[inline]
    fn log_unnorm(x: f64, m0: f64, c: f64, p: f64) -> f64 {
        let t = x / m0;
        let u = 1.0 - t * t;
        if u <= 0.0 || x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        x.ln() + p * u.ln() + c * u
    }

    /// Numerical normalization via 128-point Gauss-Legendre on [a, b].
    fn log_norm(a: f64, b: f64, c: f64, p: f64) -> Result<f64> {
        // Quick 64-pt GL quadrature.
        let n = 64;
        let (ref_nodes, ref_weights) = gauss_legendre_64();
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);

        // log-sum-exp for stability.
        let mut terms = Vec::with_capacity(n);
        for i in 0..n {
            let x = mid + half * ref_nodes[i];
            let w = ref_weights[i] * half;
            if w <= 0.0 {
                continue;
            }
            let log_f = Self::log_unnorm(x, b, c, p);
            if log_f.is_finite() {
                terms.push(log_f + w.ln());
            }
        }

        if terms.is_empty() {
            return Err(Error::Computation(
                "ArgusPdf: normalization integral is zero (all terms -inf)".into(),
            ));
        }

        let max_val = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = terms.iter().map(|&t| (t - max_val).exp()).sum();
        let log_integral = max_val + sum_exp.ln();

        if !log_integral.is_finite() {
            return Err(Error::Computation(
                "ArgusPdf: normalization integral is not finite".into(),
            ));
        }
        Ok(log_integral)
    }
}

impl UnbinnedPdf for ArgusPdf {
    fn n_params(&self) -> usize {
        2 // c, p
    }

    fn observables(&self) -> &[String] {
        &self.observable
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if params.len() != 2 {
            return Err(Error::Validation(format!(
                "ArgusPdf expects 2 params (c, p), got {}",
                params.len()
            )));
        }
        let c = params[0];
        let p = params[1];
        if !c.is_finite() || !p.is_finite() || p < 0.0 {
            return Err(Error::Validation(format!(
                "ArgusPdf: c must be finite, p must be >= 0, got c={c}, p={p}"
            )));
        }

        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "ArgusPdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observable[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;

        let log_norm = Self::log_norm(a, b, c, p)?;

        for (i, &x) in xs.iter().enumerate() {
            out[i] = Self::log_unnorm(x, b, c, p) - log_norm;
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
        if params.len() != 2 {
            return Err(Error::Validation(format!(
                "ArgusPdf expects 2 params (c, p), got {}",
                params.len()
            )));
        }
        let c = params[0];
        let p = params[1];
        if !c.is_finite() || !p.is_finite() || p < 0.0 {
            return Err(Error::Validation(format!(
                "ArgusPdf: c must be finite, p must be >= 0, got c={c}, p={p}"
            )));
        }

        let n = events.n_events();
        if out_logp.len() != n {
            return Err(Error::Validation(format!(
                "ArgusPdf out_logp length mismatch: expected {n}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n * 2;
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "ArgusPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
                out_grad.len()
            )));
        }

        let obs = self.observable[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;

        let log_norm = Self::log_norm(a, b, c, p)?;

        // Compute d(log_norm)/dc and d(log_norm)/dp via finite differences.
        let eps_c = 1e-6 * (1.0 + c.abs());
        let eps_p = 1e-6 * (1.0 + p.abs());
        let log_norm_dc = Self::log_norm(a, b, c + eps_c, p)?;
        let log_norm_dp = Self::log_norm(a, b, c, p + eps_p)?;
        let dlogn_dc = (log_norm_dc - log_norm) / eps_c;
        let dlogn_dp = (log_norm_dp - log_norm) / eps_p;

        for (i, &x) in xs.iter().enumerate() {
            let t = x / b;
            let u = 1.0 - t * t;
            let log_f = Self::log_unnorm(x, b, c, p);
            out_logp[i] = log_f - log_norm;

            // d(log f)/dc = u  (since log f contains c·u)
            // d(log f)/dp = ln(u) (since log f contains p·ln(u))
            let (df_dc, df_dp) = if u > 0.0 { (u, u.ln()) } else { (0.0, 0.0) };

            let base = i * 2;
            out_grad[base] = df_dc - dlogn_dc;
            out_grad[base + 1] = df_dp - dlogn_dp;
        }

        Ok(())
    }
}

/// 64-point Gauss-Legendre nodes and weights on [-1, 1] (precomputed).
fn gauss_legendre_64() -> (Vec<f64>, Vec<f64>) {
    // Use the normalize.rs implementation path.
    crate::normalize::gauss_legendre_nodes_weights_pub(64)
}
