use crate::event_store::EventStore;
use crate::math::{standard_normal_cdf, standard_normal_logpdf, standard_normal_pdf};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};
use statrs::distribution::{ContinuousCDF, Normal};

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
        if out.len() != n {
            return Err(Error::Validation(format!(
                "GaussianPdf out length mismatch: expected {n}, got {}",
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
        let z_a = (a - mu) * inv_sigma;
        let z_b = (b - mu) * inv_sigma;

        // Normalization constant Z = Phi(z_b) - Phi(z_a).
        let mut z = standard_normal_cdf(z_b) - standard_normal_cdf(z_a);
        if !z.is_finite() || z <= 0.0 {
            // Underflow/degenerate: keep cost finite; this corresponds to extreme truncation.
            z = f64::MIN_POSITIVE;
        }
        let log_z = z.ln();

        let log_sigma = sigma.ln();
        for (i, &x) in xs.iter().enumerate() {
            let z_x = (x - mu) * inv_sigma;
            out[i] = standard_normal_logpdf(z_x) - log_sigma - log_z;
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

    fn pdf_tag(&self) -> &'static str {
        "gaussian"
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn rand::RngCore,
    ) -> Result<EventStore> {
        if params.len() != 2 {
            return Err(Error::Validation(format!(
                "GaussianPdf expects 2 params (mu, sigma), got {}",
                params.len()
            )));
        }
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "GaussianPdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation(format!(
                "GaussianPdf parameters must be finite with sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        let (a, b) = support[0];
        if !a.is_finite() || !b.is_finite() || a >= b {
            return Err(Error::Validation(format!(
                "GaussianPdf sample requires finite support with low < high, got ({a}, {b})"
            )));
        }

        // Uniform(0,1) from RngCore (open interval).
        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            // Map u64 -> (0,1) excluding endpoints.
            let v = rng.next_u64();
            // Add 0.5 to avoid exact 0; denominator is 2^64.
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        let stdn = Normal::new(0.0, 1.0).map_err(|e| {
            Error::Computation(format!("failed to construct standard normal distribution: {e}"))
        })?;

        let z_a = (a - mu) / sigma;
        let z_b = (b - mu) / sigma;
        let mut u_lo = stdn.cdf(z_a);
        let mut u_hi = stdn.cdf(z_b);
        if !(u_lo.is_finite() && u_hi.is_finite() && u_lo < u_hi) {
            return Err(Error::Validation(format!(
                "GaussianPdf sample has invalid truncated CDF range: Phi(z_a)={u_lo}, Phi(z_b)={u_hi}"
            )));
        }
        // Avoid inverse_cdf(0/1) infinities.
        let eps = 1e-15;
        u_lo = u_lo.clamp(eps, 1.0 - eps);
        u_hi = u_hi.clamp(eps, 1.0 - eps);
        if u_lo >= u_hi {
            return Err(Error::Validation(format!(
                "GaussianPdf sample has degenerate truncated CDF range after clamping: [{u_lo}, {u_hi}]"
            )));
        }

        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let u = u_lo + (u_hi - u_lo) * u01(rng);
            let z = stdn.inverse_cdf(u);
            let x = mu + sigma * z;
            xs.push(x.clamp(a, b));
        }

        let obs = crate::event_store::ObservableSpec::branch(self.observables[0].clone(), (a, b));
        EventStore::from_columns(vec![obs], vec![(self.observables[0].clone(), xs)], None)
    }
}
