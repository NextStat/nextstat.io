use crate::event_store::EventStore;
use crate::math::{standard_normal_cdf, standard_normal_logpdf};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};
use statrs::distribution::{ContinuousCDF, Normal};

/// 1D Gaussian kernel density estimator (KDE) on a bounded support.
///
/// The KDE is normalized on `[low, high]` by using *truncated* Gaussian kernels:
///
/// `p(x) = (1 / Σ w_i) Σ_i w_i · φ((x - x_i)/h) / (h · Z_i)`
///
/// where `Z_i = Φ((high - x_i)/h) - Φ((low - x_i)/h)` is the truncation factor and `w_i >= 0`.
///
/// This is a non-ML baseline for approximating unknown 1D PDFs from MC samples.
#[derive(Debug, Clone)]
pub struct KdePdf {
    observables: [String; 1],
    support: (f64, f64),
    centers: Vec<f64>,
    /// Per-kernel log prefactor: `ln(w_i) - ln(h) - ln(Z_i)`.
    kernel_log_prefactor: Vec<f64>,
    /// CDF for selecting kernels with probability proportional to weight.
    kernel_cdf: Vec<f64>,
    log_sum_w: f64,
    inv_bandwidth: f64,
}

impl KdePdf {
    /// Construct a bounded KDE from samples and an optional non-negative weight per sample.
    pub fn from_samples(
        observable: impl Into<String>,
        support: (f64, f64),
        centers: Vec<f64>,
        weights: Option<Vec<f64>>,
        bandwidth: f64,
    ) -> Result<Self> {
        let (low, high) = support;
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(Error::Validation(format!(
                "KdePdf requires finite support with low < high, got ({low}, {high})"
            )));
        }
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(Error::Validation(format!(
                "KdePdf bandwidth must be finite and > 0, got {bandwidth}"
            )));
        }
        if centers.is_empty() {
            return Err(Error::Validation("KdePdf requires at least one center".into()));
        }
        if centers.iter().any(|x| !x.is_finite()) {
            return Err(Error::Validation("KdePdf centers must be finite".into()));
        }
        if centers.iter().any(|&x| x < low || x > high) {
            return Err(Error::Validation(format!(
                "KdePdf centers must lie within support [{low}, {high}]"
            )));
        }

        if let Some(w) = &weights {
            if w.len() != centers.len() {
                return Err(Error::Validation(format!(
                    "KdePdf weights length mismatch: expected {}, got {}",
                    centers.len(),
                    w.len()
                )));
            }
            if w.iter().any(|x| !x.is_finite()) {
                return Err(Error::Validation("KdePdf weights must be finite".into()));
            }
            if w.iter().any(|x| *x < 0.0) {
                return Err(Error::Validation("KdePdf weights must be >= 0".into()));
            }
        }

        let inv_bandwidth = 1.0 / bandwidth;
        let log_h = bandwidth.ln();

        let mut kernel_log_prefactor = Vec::with_capacity(centers.len());
        let mut kernel_cdf = Vec::with_capacity(centers.len());

        let mut sum_w = 0.0f64;
        for (i, &x0) in centers.iter().enumerate() {
            let w = weights.as_ref().map(|v| v[i]).unwrap_or(1.0);
            sum_w += w;
            kernel_cdf.push(sum_w);

            let log_w = if w > 0.0 { w.ln() } else { f64::NEG_INFINITY };
            let z_low = (low - x0) * inv_bandwidth;
            let z_high = (high - x0) * inv_bandwidth;
            let mut z = standard_normal_cdf(z_high) - standard_normal_cdf(z_low);
            if !z.is_finite() || z <= 0.0 {
                // Underflow/extreme truncation: keep density finite.
                z = f64::MIN_POSITIVE;
            }
            let log_z = z.ln();

            kernel_log_prefactor.push(log_w - log_h - log_z);
        }

        if !(sum_w.is_finite() && sum_w > 0.0) {
            return Err(Error::Validation(format!(
                "KdePdf requires sum(weights) > 0, got {sum_w}"
            )));
        }

        for v in &mut kernel_cdf {
            *v /= sum_w;
        }
        if let Some(last) = kernel_cdf.last_mut() {
            *last = 1.0;
        }

        Ok(Self {
            observables: [observable.into()],
            support,
            centers,
            kernel_log_prefactor,
            kernel_cdf,
            log_sum_w: sum_w.ln(),
            inv_bandwidth,
        })
    }

    fn validate_support_matches(&self, events: &EventStore, obs: &str) -> Result<()> {
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        let (low, high) = self.support;
        // Exact equality is common (comes from the spec). Use a tiny tolerance to avoid false
        // negatives from serialization/rounding.
        let eps = 1e-12;
        if (a - low).abs() > eps || (b - high).abs() > eps {
            return Err(Error::Validation(format!(
                "KdePdf support mismatch: pdf=[{low}, {high}], events=({a}, {b})"
            )));
        }
        Ok(())
    }

    #[inline]
    fn log_sum_kernels_at(&self, x: f64) -> f64 {
        // Online logsumexp: keep (m, s) so that log(Σ exp(t_i)) = m + ln(s).
        let mut m = f64::NEG_INFINITY;
        let mut s = 0.0f64;

        for (x0, lpref) in self.centers.iter().zip(&self.kernel_log_prefactor) {
            if !lpref.is_finite() {
                continue;
            }
            let z = (x - x0) * self.inv_bandwidth;
            let t = lpref + standard_normal_logpdf(z);
            if t > m {
                if m.is_finite() {
                    s = s * (m - t).exp() + 1.0;
                } else {
                    s = 1.0;
                }
                m = t;
            } else {
                s += (t - m).exp();
            }
        }

        if !m.is_finite() {
            return f64::NEG_INFINITY;
        }
        m + s.ln()
    }
}

impl UnbinnedPdf for KdePdf {
    fn n_params(&self) -> usize {
        0
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if !params.is_empty() {
            return Err(Error::Validation(format!(
                "KdePdf expects 0 params, got {}",
                params.len()
            )));
        }

        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "KdePdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        for (i, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::Validation("KdePdf requires finite x".into()));
            }
            out[i] = self.log_sum_kernels_at(x) - self.log_sum_w;
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
        if !out_grad.is_empty() {
            return Err(Error::Validation(format!(
                "KdePdf out_grad must be empty (n_params=0), got len={}",
                out_grad.len()
            )));
        }
        self.log_prob_batch(events, params, out_logp)
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn rand::RngCore,
    ) -> Result<EventStore> {
        if !params.is_empty() {
            return Err(Error::Validation(format!(
                "KdePdf expects 0 params, got {}",
                params.len()
            )));
        }
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "KdePdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let (a, b) = support[0];
        let (low, high) = self.support;
        let eps = 1e-12;
        if (a - low).abs() > eps || (b - high).abs() > eps {
            return Err(Error::Validation(format!(
                "KdePdf support mismatch: pdf=[{low}, {high}], support=({a}, {b})"
            )));
        }

        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            let v = rng.next_u64();
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        if self.centers.is_empty() {
            return Err(Error::Validation("KdePdf requires at least one center".into()));
        }

        let stdn = Normal::new(0.0, 1.0).map_err(|e| {
            Error::Computation(format!("failed to construct standard normal distribution: {e}"))
        })?;
        let h = 1.0 / self.inv_bandwidth;

        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let u = u01(rng);
            let idx = self.kernel_cdf.partition_point(|p| *p < u).min(self.centers.len() - 1);
            let x0 = self.centers[idx];

            // Truncated Gaussian kernel centered at x0 with bandwidth h.
            let z_low = (low - x0) / h;
            let z_high = (high - x0) / h;
            let mut u_lo = stdn.cdf(z_low);
            let mut u_hi = stdn.cdf(z_high);
            if !(u_lo.is_finite() && u_hi.is_finite() && u_lo < u_hi) {
                return Err(Error::Validation(format!(
                    "KdePdf sample kernel has invalid truncated CDF range: Phi(z_low)={u_lo}, Phi(z_high)={u_hi}"
                )));
            }
            let eps = 1e-15;
            u_lo = u_lo.clamp(eps, 1.0 - eps);
            u_hi = u_hi.clamp(eps, 1.0 - eps);
            if u_lo >= u_hi {
                return Err(Error::Validation(format!(
                    "KdePdf sample kernel has degenerate truncated CDF range after clamping: [{u_lo}, {u_hi}]"
                )));
            }

            let uu = u_lo + (u_hi - u_lo) * u01(rng);
            let z = stdn.inverse_cdf(uu);
            let x = x0 + h * z;
            xs.push(x.clamp(low, high));
        }

        let obs =
            crate::event_store::ObservableSpec::branch(self.observables[0].clone(), (low, high));
        EventStore::from_columns(vec![obs], vec![(self.observables[0].clone(), xs)], None)
    }
}
