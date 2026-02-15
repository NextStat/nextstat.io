use crate::event_store::{EventStore, ObservableSpec};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Voigtian PDF — convolution of a Gaussian and a Cauchy (Breit-Wigner) distribution.
///
/// Commonly used in particle physics for modeling resonance line shapes where both
/// natural width (Breit-Wigner) and detector resolution (Gaussian) contribute.
///
/// `V(x; μ, σ, γ) ∝ Re[w(z)]` where `w(z)` is the Faddeeva function and
/// `z = (x - μ + iγ) / (σ√2)`.
///
/// **Shape parameters (3):** `[μ, σ, γ]`
/// - `μ`: peak position
/// - `σ`: Gaussian width (detector resolution), must be > 0
/// - `γ`: Lorentzian half-width (natural width), must be > 0
///
/// The PDF is normalized on the EventStore bounds via numerical quadrature.
pub struct VoigtianPdf {
    observable: [String; 1],
}

impl VoigtianPdf {
    /// Create a Voigtian PDF over the given observable.
    pub fn new(observable: impl Into<String>) -> Self {
        Self { observable: [observable.into()] }
    }

    /// Evaluate the Voigt profile (unnormalized) at point `x`.
    ///
    /// Uses the pseudo-Voigt approximation (Thompson et al. 1987) which is accurate
    /// to <1% and avoids the expensive Faddeeva function.
    #[inline]
    fn voigt_unnorm(x: f64, mu: f64, sigma: f64, gamma: f64) -> f64 {
        // Full-widths at half-maximum.
        let f_g = 2.0 * sigma * (2.0_f64 * 2.0_f64.ln()).sqrt(); // FWHM Gaussian
        let f_l = 2.0 * gamma; // FWHM Lorentzian

        // Thompson-Cox-Hastings approximation for total FWHM.
        let f5 = f_g.powi(5)
            + 2.69269 * f_g.powi(4) * f_l
            + 2.42843 * f_g.powi(3) * f_l.powi(2)
            + 4.47163 * f_g.powi(2) * f_l.powi(3)
            + 0.07842 * f_g * f_l.powi(4)
            + f_l.powi(5);
        let f_v = f5.powf(0.2);

        // Mixing parameter eta (fraction of Lorentzian).
        let ratio = f_l / f_v;
        let eta = 1.36603 * ratio - 0.47719 * ratio * ratio + 0.11116 * ratio * ratio * ratio;
        let eta = eta.clamp(0.0, 1.0);

        // Pseudo-Voigt: eta * L(x) + (1-eta) * G(x).
        let dx = x - mu;
        let gauss = (-0.5 * dx * dx / (sigma * sigma)).exp()
            / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        let lorentz = gamma / (std::f64::consts::PI * (dx * dx + gamma * gamma));

        eta * lorentz + (1.0 - eta) * gauss
    }

    /// Log of the unnormalized Voigt profile.
    #[inline]
    fn log_voigt_unnorm(x: f64, mu: f64, sigma: f64, gamma: f64) -> f64 {
        let val = Self::voigt_unnorm(x, mu, sigma, gamma);
        if val > 0.0 { val.ln() } else { f64::NEG_INFINITY }
    }

    /// Numerical normalization via Gauss-Legendre quadrature on [a, b].
    fn log_norm(a: f64, b: f64, mu: f64, sigma: f64, gamma: f64) -> Result<f64> {
        let (ref_nodes, ref_weights) = crate::normalize::gauss_legendre_nodes_weights_pub(64);
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);

        let mut terms = Vec::with_capacity(64);
        for i in 0..ref_nodes.len() {
            let x = mid + half * ref_nodes[i];
            let w = ref_weights[i] * half;
            if w <= 0.0 {
                continue;
            }
            let log_f = Self::log_voigt_unnorm(x, mu, sigma, gamma);
            if log_f.is_finite() {
                terms.push(log_f + w.ln());
            }
        }

        if terms.is_empty() {
            return Err(Error::Computation("VoigtianPdf: normalization integral is zero".into()));
        }

        let max_val = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = terms.iter().map(|&t| (t - max_val).exp()).sum();
        let log_integral = max_val + sum_exp.ln();

        if !log_integral.is_finite() {
            return Err(Error::Computation(
                "VoigtianPdf: normalization integral is not finite".into(),
            ));
        }
        Ok(log_integral)
    }
}

impl UnbinnedPdf for VoigtianPdf {
    fn n_params(&self) -> usize {
        3 // mu, sigma, gamma
    }

    fn observables(&self) -> &[String] {
        &self.observable
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if params.len() != 3 {
            return Err(Error::Validation(format!(
                "VoigtianPdf expects 3 params (mu, sigma, gamma), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        let gamma = params[2];
        if !mu.is_finite() || !sigma.is_finite() || !gamma.is_finite() {
            return Err(Error::Validation(format!(
                "VoigtianPdf: all params must be finite, got mu={mu}, sigma={sigma}, gamma={gamma}"
            )));
        }
        if sigma <= 0.0 || gamma <= 0.0 {
            return Err(Error::Validation(format!(
                "VoigtianPdf: sigma and gamma must be > 0, got sigma={sigma}, gamma={gamma}"
            )));
        }

        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "VoigtianPdf out length mismatch: expected {n}, got {}",
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

        let log_norm = Self::log_norm(a, b, mu, sigma, gamma)?;

        for (i, &x) in xs.iter().enumerate() {
            out[i] = Self::log_voigt_unnorm(x, mu, sigma, gamma) - log_norm;
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
        if params.len() != 3 {
            return Err(Error::Validation(format!(
                "VoigtianPdf expects 3 params (mu, sigma, gamma), got {}",
                params.len()
            )));
        }
        let mu = params[0];
        let sigma = params[1];
        let gamma = params[2];
        if !mu.is_finite() || !sigma.is_finite() || !gamma.is_finite() {
            return Err(Error::Validation(format!(
                "VoigtianPdf: all params must be finite, got mu={mu}, sigma={sigma}, gamma={gamma}"
            )));
        }
        if sigma <= 0.0 || gamma <= 0.0 {
            return Err(Error::Validation(format!(
                "VoigtianPdf: sigma and gamma must be > 0, got sigma={sigma}, gamma={gamma}"
            )));
        }

        let n = events.n_events();
        if out_logp.len() != n {
            return Err(Error::Validation(format!(
                "VoigtianPdf out_logp length mismatch: expected {n}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n * 3;
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "VoigtianPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
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

        // Central log_norm and finite-difference derivatives of log_norm.
        let log_norm = Self::log_norm(a, b, mu, sigma, gamma)?;

        let eps_mu = 1e-6 * (1.0 + mu.abs());
        let eps_sig = 1e-6 * sigma;
        let eps_gam = 1e-6 * gamma;

        let dlogn_dmu = (Self::log_norm(a, b, mu + eps_mu, sigma, gamma)? - log_norm) / eps_mu;
        let dlogn_dsig = (Self::log_norm(a, b, mu, sigma + eps_sig, gamma)? - log_norm) / eps_sig;
        let dlogn_dgam = (Self::log_norm(a, b, mu, sigma, gamma + eps_gam)? - log_norm) / eps_gam;

        for (i, &x) in xs.iter().enumerate() {
            let log_f = Self::log_voigt_unnorm(x, mu, sigma, gamma);
            out_logp[i] = log_f - log_norm;

            // Finite-difference gradient of log f w.r.t. each parameter.
            let df_dmu = (Self::log_voigt_unnorm(x, mu + eps_mu, sigma, gamma) - log_f) / eps_mu;
            let df_dsig = (Self::log_voigt_unnorm(x, mu, sigma + eps_sig, gamma) - log_f) / eps_sig;
            let df_dgam = (Self::log_voigt_unnorm(x, mu, sigma, gamma + eps_gam) - log_f) / eps_gam;

            let base = i * 3;
            out_grad[base] = df_dmu - dlogn_dmu;
            out_grad[base + 1] = df_dsig - dlogn_dsig;
            out_grad[base + 2] = df_dgam - dlogn_dgam;
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
        if params.len() != 3 {
            return Err(Error::Validation(format!(
                "VoigtianPdf expects 3 params (mu, sigma, gamma), got {}",
                params.len()
            )));
        }
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "VoigtianPdf sample expects 1D support, got {}D",
                support.len()
            )));
        }

        let mu = params[0];
        let sigma = params[1];
        let gamma = params[2];
        if !mu.is_finite() || !sigma.is_finite() || !gamma.is_finite() {
            return Err(Error::Validation(format!(
                "VoigtianPdf: all params must be finite, got mu={mu}, sigma={sigma}, gamma={gamma}"
            )));
        }
        if sigma <= 0.0 || gamma <= 0.0 {
            return Err(Error::Validation(format!(
                "VoigtianPdf: sigma and gamma must be > 0, got sigma={sigma}, gamma={gamma}"
            )));
        }

        let (a, b) = support[0];
        if !a.is_finite() || !b.is_finite() || a >= b {
            return Err(Error::Validation(format!(
                "VoigtianPdf sample requires finite support with low < high, got ({a}, {b})"
            )));
        }

        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            (rng.next_u64() as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        // Numerical inverse-CDF sampling on finite support.
        const N_GRID: usize = 4096;
        let dx = (b - a) / (N_GRID as f64 - 1.0);
        let mut x_grid = vec![0.0; N_GRID];
        let mut pdf_grid = vec![0.0; N_GRID];
        for i in 0..N_GRID {
            let x = a + dx * i as f64;
            x_grid[i] = x;
            let v = Self::voigt_unnorm(x, mu, sigma, gamma);
            pdf_grid[i] = if v.is_finite() && v > 0.0 { v } else { 0.0 };
        }

        let mut cdf = vec![0.0; N_GRID];
        for i in 1..N_GRID {
            let area = 0.5 * (pdf_grid[i - 1] + pdf_grid[i]) * dx;
            cdf[i] = cdf[i - 1] + area.max(0.0);
        }

        let total = *cdf.last().unwrap_or(&0.0);
        if !total.is_finite() || total <= 0.0 {
            return Err(Error::Computation(
                "VoigtianPdf::sample failed: numerical integral is non-positive".into(),
            ));
        }
        for v in &mut cdf {
            *v /= total;
        }

        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let u = u01(rng);
            let idx = cdf.partition_point(|&v| v < u);
            let x = if idx == 0 {
                x_grid[0]
            } else if idx >= N_GRID {
                x_grid[N_GRID - 1]
            } else {
                let c0 = cdf[idx - 1];
                let c1 = cdf[idx];
                let x0 = x_grid[idx - 1];
                let x1 = x_grid[idx];
                if c1 > c0 { x0 + (u - c0) * (x1 - x0) / (c1 - c0) } else { x0 }
            };
            xs.push(x.clamp(a, b));
        }

        let obs = ObservableSpec::branch(self.observable[0].clone(), (a, b));
        EventStore::from_columns(vec![obs], vec![(self.observable[0].clone(), xs)], None)
    }
}
