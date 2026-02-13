use crate::event_store::{EventStore, ObservableSpec};
use crate::interp::{HistoSysInterpCode, histosys_interp};
use crate::math::{standard_normal_cdf, standard_normal_logpdf};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};
use statrs::distribution::{ContinuousCDF, Normal};

/// KDE PDF with HistFactory-style per-kernel weight morphing.
///
/// This provides a baseline for unbinned "WeightSys" systematics on `kde_from_tree` shapes:
///
/// - Build a nominal KDE from centers `x_i` with weights `w_i` (non-negative).
/// - For each nuisance parameter α_j, provide up/down per-kernel weights `w_i^{up}`, `w_i^{down}`.
/// - At evaluation time, interpolate each kernel weight using HistFactory interpolation
///   (Code0 or Code4p) and normalize the resulting KDE by `Σ w_i(α)`.
///
/// The resulting density is normalized on the bounded support `[low, high]` for any α values.
#[derive(Debug, Clone)]
pub struct MorphingKdePdf {
    observables: [String; 1],
    support: (f64, f64),
    centers: Vec<f64>,
    /// Per-kernel term: `-ln(h) - ln(Z_i)` where `Z_i` is the truncation factor for kernel i.
    kernel_log_base: Vec<f64>,
    nominal_weights: Vec<f64>,
    systematics: Vec<KdeWeightSystematic>,
    inv_bandwidth: f64,
}

/// Up/down kernel weights for one nuisance parameter.
#[derive(Debug, Clone)]
pub struct KdeWeightSystematic {
    /// Per-kernel weights at α = -1.
    pub down: Vec<f64>,
    /// Per-kernel weights at α = +1.
    pub up: Vec<f64>,
    /// Interpolation code used to morph between down/nominal/up.
    pub interp_code: HistoSysInterpCode,
}

impl MorphingKdePdf {
    /// Construct a morphing KDE PDF on a bounded support.
    pub fn new(
        observable: impl Into<String>,
        support: (f64, f64),
        centers: Vec<f64>,
        nominal_weights: Vec<f64>,
        systematics: Vec<KdeWeightSystematic>,
        bandwidth: f64,
    ) -> Result<Self> {
        let (low, high) = support;
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(Error::Validation(format!(
                "MorphingKdePdf requires finite support with low < high, got ({low}, {high})"
            )));
        }
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(Error::Validation(format!(
                "MorphingKdePdf bandwidth must be finite and > 0, got {bandwidth}"
            )));
        }
        if centers.is_empty() {
            return Err(Error::Validation("MorphingKdePdf requires at least one center".into()));
        }
        if nominal_weights.len() != centers.len() {
            return Err(Error::Validation(format!(
                "MorphingKdePdf nominal_weights length mismatch: expected {}, got {}",
                centers.len(),
                nominal_weights.len()
            )));
        }
        if centers.iter().any(|x| !x.is_finite()) {
            return Err(Error::Validation("MorphingKdePdf centers must be finite".into()));
        }
        if centers.iter().any(|&x| x < low || x > high) {
            return Err(Error::Validation(format!(
                "MorphingKdePdf centers must lie within support [{low}, {high}]"
            )));
        }
        if nominal_weights.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(Error::Validation(
                "MorphingKdePdf nominal_weights must be finite and >= 0".into(),
            ));
        }
        let sum_nom: f64 = nominal_weights.iter().sum();
        if !(sum_nom.is_finite() && sum_nom > 0.0) {
            return Err(Error::Validation(format!(
                "MorphingKdePdf requires sum(nominal_weights) > 0, got {sum_nom}"
            )));
        }

        if systematics.is_empty() {
            return Err(Error::Validation(
                "MorphingKdePdf requires at least one systematic (n_params > 0)".into(),
            ));
        }
        for (sidx, s) in systematics.iter().enumerate() {
            if s.down.len() != centers.len() || s.up.len() != centers.len() {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf systematic[{sidx}] length mismatch: expected {}, got down={} up={}",
                    centers.len(),
                    s.down.len(),
                    s.up.len()
                )));
            }
            if s.down.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf systematic[{sidx}].down must be finite and >= 0"
                )));
            }
            if s.up.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf systematic[{sidx}].up must be finite and >= 0"
                )));
            }
        }

        let inv_bandwidth = 1.0 / bandwidth;
        let log_h = bandwidth.ln();

        let mut kernel_log_base = Vec::with_capacity(centers.len());
        for &x0 in &centers {
            let z_low = (low - x0) * inv_bandwidth;
            let z_high = (high - x0) * inv_bandwidth;
            let mut z = standard_normal_cdf(z_high) - standard_normal_cdf(z_low);
            if !z.is_finite() || z <= 0.0 {
                // Underflow/extreme truncation: keep density finite.
                z = f64::MIN_POSITIVE;
            }
            kernel_log_base.push(-log_h - z.ln());
        }

        Ok(Self {
            observables: [observable.into()],
            support,
            centers,
            kernel_log_base,
            nominal_weights,
            systematics,
            inv_bandwidth,
        })
    }

    fn validate_support_matches(&self, events: &EventStore, obs: &str) -> Result<()> {
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        let (low, high) = self.support;
        let eps = 1e-12;
        if (a - low).abs() > eps || (b - high).abs() > eps {
            return Err(Error::Validation(format!(
                "MorphingKdePdf support mismatch: pdf=[{low}, {high}], events=({a}, {b})"
            )));
        }
        Ok(())
    }

    fn weights_scratch(&self, params: &[f64]) -> Result<KernelWeightsScratch> {
        let n_params = self.systematics.len();
        if params.len() != n_params {
            return Err(Error::Validation(format!(
                "MorphingKdePdf params length mismatch: expected {n_params}, got {}",
                params.len()
            )));
        }

        let n_kernels = self.centers.len();
        let mut weights = vec![0.0f64; n_kernels];
        let mut logw_plus_base = vec![f64::NEG_INFINITY; n_kernels];
        let mut dw = vec![0.0f64; n_kernels * n_params];

        let eps_w = f64::MIN_POSITIVE;

        for i in 0..n_kernels {
            let nom = self.nominal_weights[i];
            let mut wi = nom;

            for (pidx, (s, &alpha)) in self.systematics.iter().zip(params).enumerate() {
                let (val, dval) = histosys_interp(alpha, s.down[i], nom, s.up[i], s.interp_code)?;
                wi += val - nom;
                dw[i * n_params + pidx] = dval;
            }

            if !wi.is_finite() {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf interpolated weight is not finite (kernel {i})"
                )));
            }

            if wi <= 0.0 {
                // Guardrail: clamp and zero derivatives for stability.
                wi = eps_w;
                for pidx in 0..n_params {
                    dw[i * n_params + pidx] = 0.0;
                }
            }

            weights[i] = wi;
            logw_plus_base[i] = wi.ln() + self.kernel_log_base[i];
        }

        let sum_w: f64 = weights.iter().sum();
        if !(sum_w.is_finite() && sum_w > 0.0) {
            return Err(Error::Validation(format!(
                "MorphingKdePdf requires sum(weights) > 0 for interpolated weights, got {sum_w}"
            )));
        }

        let inv_sum_w = 1.0 / sum_w;
        let log_sum_w = sum_w.ln();

        let mut dlog_sum_w = vec![0.0f64; n_params];
        for i in 0..n_kernels {
            for pidx in 0..n_params {
                dlog_sum_w[pidx] += dw[i * n_params + pidx];
            }
        }
        for v in &mut dlog_sum_w {
            *v *= inv_sum_w;
        }

        // Convert to per-kernel (dw / w) for stable dlogS accumulation.
        let mut dlogw = vec![0.0f64; n_kernels * n_params];
        for i in 0..n_kernels {
            let inv_w = 1.0 / weights[i];
            for pidx in 0..n_params {
                dlogw[i * n_params + pidx] = dw[i * n_params + pidx] * inv_w;
            }
        }

        Ok(KernelWeightsScratch { weights, logw_plus_base, sum_w, log_sum_w, dlog_sum_w, dlogw })
    }
}

struct KernelWeightsScratch {
    weights: Vec<f64>,
    logw_plus_base: Vec<f64>,
    sum_w: f64,
    log_sum_w: f64,
    dlog_sum_w: Vec<f64>,
    /// Per-kernel `dw / w` values (row-major kernel×param).
    dlogw: Vec<f64>,
}

impl UnbinnedPdf for MorphingKdePdf {
    fn n_params(&self) -> usize {
        self.systematics.len()
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "MorphingKdePdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        let scratch = self.weights_scratch(params)?;

        for (evt_idx, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::Validation("MorphingKdePdf requires finite x".into()));
            }

            let mut m = f64::NEG_INFINITY;
            for (x0, lb) in self.centers.iter().zip(&scratch.logw_plus_base) {
                let z = (x - x0) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                if t > m {
                    m = t;
                }
            }
            if !m.is_finite() {
                return Err(Error::Validation(
                    "MorphingKdePdf failed to compute kernel logsumexp (all -inf)".into(),
                ));
            }

            let mut s_sum = 0.0f64;
            for (x0, lb) in self.centers.iter().zip(&scratch.logw_plus_base) {
                let z = (x - x0) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                s_sum += (t - m).exp();
            }
            if !(s_sum.is_finite() && s_sum > 0.0) {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf kernel sum underflow/NaN for x={x}: s_sum={s_sum}"
                )));
            }
            let log_s = m + s_sum.ln();

            out[evt_idx] = log_s - scratch.log_sum_w;
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
        let n_events = events.n_events();
        if out_logp.len() != n_events {
            return Err(Error::Validation(format!(
                "MorphingKdePdf out_logp length mismatch: expected {n_events}, got {}",
                out_logp.len()
            )));
        }

        let n_params = self.n_params();
        if params.len() != n_params {
            return Err(Error::Validation(format!(
                "MorphingKdePdf params length mismatch: expected {n_params}, got {}",
                params.len()
            )));
        }
        if out_grad.len() != n_events * n_params {
            return Err(Error::Validation(format!(
                "MorphingKdePdf out_grad length mismatch: expected {}, got {}",
                n_events * n_params,
                out_grad.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        let scratch = self.weights_scratch(params)?;
        let mut num = vec![0.0f64; n_params];

        for (evt_idx, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::Validation("MorphingKdePdf requires finite x".into()));
            }

            // Pass 1: max log-term for stable logsumexp.
            let mut m = f64::NEG_INFINITY;
            for (x0, lb) in self.centers.iter().zip(&scratch.logw_plus_base) {
                let z = (x - x0) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                if t > m {
                    m = t;
                }
            }
            if !m.is_finite() {
                return Err(Error::Validation(
                    "MorphingKdePdf failed to compute kernel logsumexp (all -inf)".into(),
                ));
            }

            // Pass 2: accumulate Σ exp(t-m) and Σ (dw/w) exp(t-m).
            let mut s_sum = 0.0f64;
            num.fill(0.0);

            for (i, (x0, lb)) in self.centers.iter().zip(&scratch.logw_plus_base).enumerate() {
                let z = (x - x0) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                let e = (t - m).exp();
                s_sum += e;

                let drow = &scratch.dlogw[i * n_params..(i + 1) * n_params];
                for pidx in 0..n_params {
                    num[pidx] += drow[pidx] * e;
                }
            }

            if !(s_sum.is_finite() && s_sum > 0.0) {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf kernel sum underflow/NaN for x={x}: s_sum={s_sum}"
                )));
            }

            let log_s = m + s_sum.ln();
            out_logp[evt_idx] = log_s - scratch.log_sum_w;

            for pidx in 0..n_params {
                let dlog_s = num[pidx] / s_sum;
                out_grad[evt_idx * n_params + pidx] = dlog_s - scratch.dlog_sum_w[pidx];
            }
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
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "MorphingKdePdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let (a, b) = support[0];
        let (low, high) = self.support;
        let eps = 1e-12;
        if (a - low).abs() > eps || (b - high).abs() > eps {
            return Err(Error::Validation(format!(
                "MorphingKdePdf support mismatch: pdf=[{low}, {high}], support=({a}, {b})"
            )));
        }

        let scratch = self.weights_scratch(params)?;
        let sum_w = scratch.sum_w;

        let mut cdf = Vec::with_capacity(scratch.weights.len());
        let mut acc = 0.0f64;
        for &w in &scratch.weights {
            acc += w / sum_w;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            let v = rng.next_u64();
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        let stdn = Normal::new(0.0, 1.0).map_err(|e| {
            Error::Computation(format!("failed to construct standard normal distribution: {e}"))
        })?;
        let h = 1.0 / self.inv_bandwidth;

        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let u = u01(rng);
            let idx = cdf.partition_point(|p| *p < u).min(self.centers.len() - 1);
            let x0 = self.centers[idx];

            // Truncated Gaussian kernel centered at x0 with bandwidth h.
            let z_low = (low - x0) / h;
            let z_high = (high - x0) / h;
            let mut u_lo = stdn.cdf(z_low);
            let mut u_hi = stdn.cdf(z_high);
            if !(u_lo.is_finite() && u_hi.is_finite() && u_lo < u_hi) {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf sample kernel has invalid truncated CDF range: Phi(z_low)={u_lo}, Phi(z_high)={u_hi}"
                )));
            }
            let eps = 1e-15;
            u_lo = u_lo.clamp(eps, 1.0 - eps);
            u_hi = u_hi.clamp(eps, 1.0 - eps);
            if u_lo >= u_hi {
                return Err(Error::Validation(format!(
                    "MorphingKdePdf sample kernel has degenerate truncated CDF range after clamping: [{u_lo}, {u_hi}]"
                )));
            }

            let uu = u_lo + (u_hi - u_lo) * u01(rng);
            let z = stdn.inverse_cdf(uu);
            let x = x0 + h * z;
            xs.push(x.clamp(low, high));
        }

        let obs_name = self.observables[0].clone();
        let observables = vec![ObservableSpec::branch(obs_name.clone(), (low, high))];
        EventStore::from_columns(observables, vec![(obs_name, xs)], None)
    }
}
