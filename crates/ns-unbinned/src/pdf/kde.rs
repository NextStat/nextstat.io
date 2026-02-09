use crate::event_store::EventStore;
use crate::math::{standard_normal_cdf, standard_normal_logpdf};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

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

        let mut sum_w = 0.0f64;
        for (i, &x0) in centers.iter().enumerate() {
            let w = weights.as_ref().map(|v| v[i]).unwrap_or(1.0);
            sum_w += w;

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

        Ok(Self {
            observables: [observable.into()],
            support,
            centers,
            kernel_log_prefactor,
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
}
