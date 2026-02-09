use crate::event_store::EventStore;
use crate::math::{standard_normal_cdf, standard_normal_logpdf, standard_normal_pdf};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Truncated Gaussian PDF `N(μ, σ)` normalized on the observable bounds.
#[derive(Debug, Clone)]
pub struct GaussianPdf {
    observables: [String; 1],
}

impl GaussianPdf {
    /// Create a new Gaussian PDF over the given observable.
    pub fn new(observable: impl Into<String>) -> Self {
        Self { observables: [observable.into()] }
    }
}

impl UnbinnedPdf for GaussianPdf {
    fn n_params(&self) -> usize {
        2
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
        if params.len() != 2 {
            return Err(Error::Validation(format!(
                "GaussianPdf expects 2 params (mu, sigma), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "GaussianPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }

        let n = events.n_events();
        if out_logp.len() != n {
            return Err(Error::Validation(format!(
                "GaussianPdf out_logp length mismatch: expected {n}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n * self.n_params();
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "GaussianPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
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
        let z_a = (a - mu) * inv_sigma;
        let z_b = (b - mu) * inv_sigma;

        // Normalization constant Z = Phi(z_b) - Phi(z_a).
        let mut z = standard_normal_cdf(z_b) - standard_normal_cdf(z_a);
        if !z.is_finite() || z <= 0.0 {
            // Underflow/degenerate: keep cost finite; this corresponds to extreme truncation.
            z = f64::MIN_POSITIVE;
        }
        let log_z = z.ln();

        // Derivatives of logZ.
        let phi_a = standard_normal_pdf(z_a);
        let phi_b = standard_normal_pdf(z_b);
        let dlogz_dmu = (phi_a - phi_b) * inv_sigma / z;
        let dlogz_dsigma = (z_a * phi_a - z_b * phi_b) * inv_sigma / z;

        for (i, &x) in xs.iter().enumerate() {
            let z_x = (x - mu) * inv_sigma;
            let lp = standard_normal_logpdf(z_x) - sigma.ln() - log_z;
            out_logp[i] = lp;

            // d/dmu logp = (z/σ) - d/dmu logZ
            let dmu = z_x * inv_sigma - dlogz_dmu;
            // d/dsigma logp = ((z^2 - 1)/σ) - d/dsigma logZ
            let ds = (z_x * z_x - 1.0) * inv_sigma - dlogz_dsigma;

            let base = i * 2;
            out_grad[base] = dmu;
            out_grad[base + 1] = ds;
        }

        Ok(())
    }
}
