use crate::event_store::{EventStore, ObservableSpec};
use crate::interp::{HistoSysInterpCode, histosys_interp};
use crate::math::{standard_normal_cdf, standard_normal_logpdf};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};
use statrs::distribution::{ContinuousCDF, Normal};

use super::morphing_kde::KdeWeightSystematic;

/// Up/down kernel centers for one nuisance parameter.
#[derive(Debug, Clone)]
pub struct KdeHorizontalSystematic {
    /// Per-kernel center locations at α = -1.
    pub down: Vec<f64>,
    /// Per-kernel center locations at α = +1.
    pub up: Vec<f64>,
    /// Interpolation code used to morph between down/nominal/up.
    pub interp_code: HistoSysInterpCode,
}

/// KDE PDF with HistFactory-style per-kernel center morphing ("horizontal systematics").
///
/// This is a baseline for unbinned horizontal morphing on `kde_from_tree` shapes:
///
/// - Build a nominal KDE from centers `c_i` with weights `w_i` (non-negative).
/// - For each nuisance parameter α_j, provide up/down **center** locations
///   `c_i^{up}`, `c_i^{down}` per kernel.
/// - At evaluation time, interpolate each center location using HistFactory interpolation
///   (Code0 or Code4p), and evaluate a truncated-Gaussian KDE with those centers.
///
/// Optional per-kernel weight morphing (WeightSys) can also be applied simultaneously.
///
/// The resulting density is normalized on the bounded support `[low, high]` for any α values.
#[derive(Debug, Clone)]
pub struct HorizontalMorphingKdePdf {
    observables: [String; 1],
    support: (f64, f64),
    nominal_centers: Vec<f64>,
    nominal_weights: Vec<f64>,
    weight_systematics: Vec<KdeWeightSystematic>,
    center_systematics: Vec<KdeHorizontalSystematic>,
    inv_bandwidth: f64,
}

impl HorizontalMorphingKdePdf {
    /// Construct a morphing KDE PDF on a bounded support.
    ///
    /// - `nominal_centers` must be finite and within `support`.
    /// - `nominal_weights` must be finite, >= 0, and sum to > 0.
    /// - `center_systematics` must be non-empty.
    pub fn new(
        observable: impl Into<String>,
        support: (f64, f64),
        nominal_centers: Vec<f64>,
        nominal_weights: Vec<f64>,
        weight_systematics: Vec<KdeWeightSystematic>,
        center_systematics: Vec<KdeHorizontalSystematic>,
        bandwidth: f64,
    ) -> Result<Self> {
        let (low, high) = support;
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf requires finite support with low < high, got ({low}, {high})"
            )));
        }
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf bandwidth must be finite and > 0, got {bandwidth}"
            )));
        }
        if nominal_centers.is_empty() {
            return Err(Error::Validation(
                "HorizontalMorphingKdePdf requires at least one center".into(),
            ));
        }
        if nominal_weights.len() != nominal_centers.len() {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf nominal_weights length mismatch: expected {}, got {}",
                nominal_centers.len(),
                nominal_weights.len()
            )));
        }
        if nominal_centers.iter().any(|x| !x.is_finite()) {
            return Err(Error::Validation(
                "HorizontalMorphingKdePdf nominal_centers must be finite".into(),
            ));
        }
        if nominal_centers.iter().any(|&x| x < low || x > high) {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf nominal_centers must lie within support [{low}, {high}]"
            )));
        }
        if nominal_weights.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(Error::Validation(
                "HorizontalMorphingKdePdf nominal_weights must be finite and >= 0".into(),
            ));
        }
        let sum_nom: f64 = nominal_weights.iter().sum();
        if !(sum_nom.is_finite() && sum_nom > 0.0) {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf requires sum(nominal_weights) > 0, got {sum_nom}"
            )));
        }

        if center_systematics.is_empty() {
            return Err(Error::Validation(
                "HorizontalMorphingKdePdf requires at least one center systematic (n_center_params > 0)".into(),
            ));
        }

        let n = nominal_centers.len();
        for (sidx, s) in center_systematics.iter().enumerate() {
            if s.down.len() != n || s.up.len() != n {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf center_systematics[{sidx}] length mismatch: expected {n}, got down={} up={}",
                    s.down.len(),
                    s.up.len()
                )));
            }
            if s.down.iter().any(|x| !x.is_finite()) || s.up.iter().any(|x| !x.is_finite()) {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf center_systematics[{sidx}] requires finite centers"
                )));
            }
            if s.down.iter().any(|&x| x < low || x > high)
                || s.up.iter().any(|&x| x < low || x > high)
            {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf center_systematics[{sidx}] centers must lie within support [{low}, {high}]"
                )));
            }
        }

        for (sidx, s) in weight_systematics.iter().enumerate() {
            if s.down.len() != n || s.up.len() != n {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf weight_systematics[{sidx}] length mismatch: expected {n}, got down={} up={}",
                    s.down.len(),
                    s.up.len()
                )));
            }
            if s.down.iter().any(|x| !x.is_finite() || *x < 0.0)
                || s.up.iter().any(|x| !x.is_finite() || *x < 0.0)
            {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf weight_systematics[{sidx}] requires finite and >=0 weights"
                )));
            }
        }

        Ok(Self {
            observables: [observable.into()],
            support,
            nominal_centers,
            nominal_weights,
            weight_systematics,
            center_systematics,
            inv_bandwidth: 1.0 / bandwidth,
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
                "HorizontalMorphingKdePdf support mismatch: pdf=[{low}, {high}], events=({a}, {b})"
            )));
        }
        Ok(())
    }

    fn scratch(&self, params: &[f64]) -> Result<KdeScratch> {
        let n_weight = self.weight_systematics.len();
        let n_center = self.center_systematics.len();
        let n_params = n_weight + n_center;
        if params.len() != n_params {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf params length mismatch: expected {n_params}, got {}",
                params.len()
            )));
        }

        let (params_w, params_c) = params.split_at(n_weight);
        let n_kernels = self.nominal_centers.len();

        // ── Interpolate weights (WeightSys) ─────────────────────────────────────
        let mut weights = vec![0.0f64; n_kernels];
        let mut log_w = vec![f64::NEG_INFINITY; n_kernels];
        let mut dw = vec![0.0f64; n_kernels * n_weight];

        let eps_w = f64::MIN_POSITIVE;

        for i in 0..n_kernels {
            let nom = self.nominal_weights[i];
            let mut wi = nom;

            for (pidx, (s, &alpha)) in self.weight_systematics.iter().zip(params_w).enumerate() {
                let (val, dval) = histosys_interp(alpha, s.down[i], nom, s.up[i], s.interp_code)?;
                wi += val - nom;
                dw[i * n_weight + pidx] = dval;
            }

            if !wi.is_finite() {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf interpolated weight is not finite (kernel {i})"
                )));
            }

            if wi <= 0.0 {
                // Guardrail: clamp and zero derivatives for stability.
                wi = eps_w;
                for pidx in 0..n_weight {
                    dw[i * n_weight + pidx] = 0.0;
                }
            }

            weights[i] = wi;
            log_w[i] = wi.ln();
        }

        let sum_w: f64 = weights.iter().sum();
        if !(sum_w.is_finite() && sum_w > 0.0) {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf requires sum(weights) > 0 for interpolated weights, got {sum_w}"
            )));
        }

        let inv_sum_w = 1.0 / sum_w;
        let log_sum_w = sum_w.ln();

        let mut dlog_sum_w = vec![0.0f64; n_weight];
        for i in 0..n_kernels {
            for pidx in 0..n_weight {
                dlog_sum_w[pidx] += dw[i * n_weight + pidx];
            }
        }
        for v in &mut dlog_sum_w {
            *v *= inv_sum_w;
        }

        // Convert to per-kernel (dw / w) for stable dlogS accumulation.
        let mut dlogw = vec![0.0f64; n_kernels * n_weight];
        for i in 0..n_kernels {
            let inv_w = 1.0 / weights[i];
            for pidx in 0..n_weight {
                dlogw[i * n_weight + pidx] = dw[i * n_weight + pidx] * inv_w;
            }
        }

        // ── Interpolate centers (horizontal systematics) ───────────────────────
        let mut centers = vec![0.0f64; n_kernels];
        let mut dcenter = vec![0.0f64; n_kernels * n_center];

        for i in 0..n_kernels {
            let nom = self.nominal_centers[i];
            let mut ci = nom;

            for (pidx, (s, &alpha)) in self.center_systematics.iter().zip(params_c).enumerate() {
                let (val, dval) = histosys_interp(alpha, s.down[i], nom, s.up[i], s.interp_code)?;
                ci += val - nom;
                dcenter[i * n_center + pidx] = dval;
            }

            if !ci.is_finite() {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf interpolated center is not finite (kernel {i})"
                )));
            }
            centers[i] = ci;
        }

        // ── Per-kernel truncation base + derivative w.r.t center ───────────────
        let (low, high) = self.support;
        let h = 1.0 / self.inv_bandwidth;
        let log_h = h.ln();
        let inv_h = self.inv_bandwidth;

        let mut logw_plus_base = vec![f64::NEG_INFINITY; n_kernels];
        let mut dbase_dc = vec![0.0f64; n_kernels];

        for i in 0..n_kernels {
            let c = centers[i];
            let u_low = (low - c) * inv_h;
            let u_high = (high - c) * inv_h;

            let mut z = standard_normal_cdf(u_high) - standard_normal_cdf(u_low);
            if !z.is_finite() || z <= 0.0 {
                // Underflow/extreme truncation: keep density finite and damp gradients.
                z = f64::MIN_POSITIVE;
                dbase_dc[i] = 0.0;
            } else {
                let phi_low = standard_normal_logpdf(u_low).exp();
                let phi_high = standard_normal_logpdf(u_high).exp();
                dbase_dc[i] = (phi_high - phi_low) * (inv_h / z);
            }

            let base = -log_h - z.ln();
            logw_plus_base[i] = log_w[i] + base;
        }

        Ok(KdeScratch {
            weights,
            centers,
            logw_plus_base,
            log_sum_w,
            dlog_sum_w,
            dlogw,
            dcenter,
            dbase_dc,
            n_weight,
            n_center,
        })
    }
}

struct KdeScratch {
    weights: Vec<f64>,
    centers: Vec<f64>,
    logw_plus_base: Vec<f64>,
    log_sum_w: f64,
    dlog_sum_w: Vec<f64>,
    /// Per-kernel `dw / w` values (row-major kernel×param for weight params only).
    dlogw: Vec<f64>,
    /// Per-kernel `dc / dalpha` values (row-major kernel×param for center params only).
    dcenter: Vec<f64>,
    /// Per-kernel `d/dc (-ln Z(c))` values.
    dbase_dc: Vec<f64>,
    n_weight: usize,
    n_center: usize,
}

impl UnbinnedPdf for HorizontalMorphingKdePdf {
    fn n_params(&self) -> usize {
        self.weight_systematics.len() + self.center_systematics.len()
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        let scratch = self.scratch(params)?;

        for (evt_idx, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::Validation("HorizontalMorphingKdePdf requires finite x".into()));
            }

            let mut m = f64::NEG_INFINITY;
            for (c, lb) in scratch.centers.iter().zip(&scratch.logw_plus_base) {
                let z = (x - c) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                if t > m {
                    m = t;
                }
            }
            if !m.is_finite() {
                return Err(Error::Validation(
                    "HorizontalMorphingKdePdf failed to compute kernel logsumexp (all -inf)".into(),
                ));
            }

            let mut s_sum = 0.0f64;
            for (c, lb) in scratch.centers.iter().zip(&scratch.logw_plus_base) {
                let z = (x - c) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                s_sum += (t - m).exp();
            }
            if !(s_sum.is_finite() && s_sum > 0.0) {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf kernel sum underflow/NaN for x={x}: s_sum={s_sum}"
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
                "HorizontalMorphingKdePdf out_logp length mismatch: expected {n_events}, got {}",
                out_logp.len()
            )));
        }

        let n_params = self.n_params();
        if params.len() != n_params {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf params length mismatch: expected {n_params}, got {}",
                params.len()
            )));
        }
        if out_grad.len() != n_events * n_params {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf out_grad length mismatch: expected {}, got {}",
                n_events * n_params,
                out_grad.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        let scratch = self.scratch(params)?;

        let inv_h2 = self.inv_bandwidth * self.inv_bandwidth;
        let mut num_w = vec![0.0f64; scratch.n_weight];
        let mut num_c = vec![0.0f64; scratch.n_center];

        for (evt_idx, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::Validation("HorizontalMorphingKdePdf requires finite x".into()));
            }

            // Pass 1: max log-term for stable logsumexp.
            let mut m = f64::NEG_INFINITY;
            for (c, lb) in scratch.centers.iter().zip(&scratch.logw_plus_base) {
                let z = (x - c) * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                if t > m {
                    m = t;
                }
            }
            if !m.is_finite() {
                return Err(Error::Validation(
                    "HorizontalMorphingKdePdf failed to compute kernel logsumexp (all -inf)".into(),
                ));
            }

            // Pass 2: accumulate Σ exp(t-m) and Σ d(t)/dα · exp(t-m).
            let mut s_sum = 0.0f64;
            num_w.fill(0.0);
            num_c.fill(0.0);

            for i in 0..scratch.centers.len() {
                let c = scratch.centers[i];
                let lb = scratch.logw_plus_base[i];

                let dx = x - c;
                let z = dx * self.inv_bandwidth;
                let t = lb + standard_normal_logpdf(z);
                let e = (t - m).exp();
                s_sum += e;

                if scratch.n_weight > 0 {
                    let drow = &scratch.dlogw[i * scratch.n_weight..(i + 1) * scratch.n_weight];
                    for pidx in 0..scratch.n_weight {
                        num_w[pidx] += drow[pidx] * e;
                    }
                }

                if scratch.n_center > 0 {
                    // d/dc log φ((x-c)/h) = (x-c) / h^2
                    let dlogphi_dc = dx * inv_h2;
                    let dlogk_dc = scratch.dbase_dc[i] + dlogphi_dc;

                    let drow = &scratch.dcenter[i * scratch.n_center..(i + 1) * scratch.n_center];
                    for pidx in 0..scratch.n_center {
                        num_c[pidx] += (dlogk_dc * drow[pidx]) * e;
                    }
                }
            }

            if !(s_sum.is_finite() && s_sum > 0.0) {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf kernel sum underflow/NaN for x={x}: s_sum={s_sum}"
                )));
            }

            let log_s = m + s_sum.ln();
            out_logp[evt_idx] = log_s - scratch.log_sum_w;

            // Weight params first.
            for pidx in 0..scratch.n_weight {
                out_grad[evt_idx * n_params + pidx] =
                    (num_w[pidx] / s_sum) - scratch.dlog_sum_w[pidx];
            }
            // Center params after weight params.
            for pidx in 0..scratch.n_center {
                out_grad[evt_idx * n_params + scratch.n_weight + pidx] = num_c[pidx] / s_sum;
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
                "HorizontalMorphingKdePdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let (a, b) = support[0];
        let (low, high) = self.support;
        let eps = 1e-12;
        if (a - low).abs() > eps || (b - high).abs() > eps {
            return Err(Error::Validation(format!(
                "HorizontalMorphingKdePdf support mismatch: pdf=[{low}, {high}], support=({a}, {b})"
            )));
        }

        let scratch = self.scratch(params)?;
        let sum_w = scratch.weights.iter().sum::<f64>();

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
            let idx = cdf.partition_point(|p| *p < u).min(scratch.centers.len() - 1);
            let c = scratch.centers[idx];

            // Truncated Gaussian kernel centered at c with bandwidth h.
            let z_low = (low - c) / h;
            let z_high = (high - c) / h;
            let mut u_lo = stdn.cdf(z_low);
            let mut u_hi = stdn.cdf(z_high);
            if !(u_lo.is_finite() && u_hi.is_finite() && u_lo < u_hi) {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf sample kernel has invalid truncated CDF range: Phi(z_low)={u_lo}, Phi(z_high)={u_hi}"
                )));
            }
            let eps = 1e-15;
            u_lo = u_lo.clamp(eps, 1.0 - eps);
            u_hi = u_hi.clamp(eps, 1.0 - eps);
            if u_lo >= u_hi {
                return Err(Error::Validation(format!(
                    "HorizontalMorphingKdePdf sample kernel has degenerate truncated CDF range after clamping: [{u_lo}, {u_hi}]"
                )));
            }

            let uu = u_lo + (u_hi - u_lo) * u01(rng);
            let z = stdn.inverse_cdf(uu);
            let x = c + h * z;
            xs.push(x.clamp(low, high));
        }

        let obs_name = self.observables[0].clone();
        let observables = vec![ObservableSpec::branch(obs_name.clone(), (low, high))];
        EventStore::from_columns(observables, vec![(obs_name, xs)], None)
    }
}
