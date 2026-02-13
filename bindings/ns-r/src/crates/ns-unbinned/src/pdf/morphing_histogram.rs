use crate::event_store::{EventStore, ObservableSpec};
use crate::interp::{HistoSysInterpCode, histosys_interp};
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Histogram PDF with HistFactory-style bin-wise template morphing.
///
/// This is the unbinned analogue of a binned HistoSys/WeightSys template: we keep the
/// piecewise-constant histogram density but allow bin **masses** to vary with nuisance
/// parameters via Code0 / Code4p interpolation between nominal/up/down templates.
///
/// The PDF is normalized on the bin support for any nuisance parameter values:
/// `p(x|α) = mass_bin(α) / Σ mass(α) / width_bin`.
#[derive(Debug, Clone)]
pub struct MorphingHistogramPdf {
    observables: [String; 1],
    bin_edges: Vec<f64>,
    log_bin_width: Vec<f64>,
    nominal_bin_content: Vec<f64>,
    systematics: Vec<HistogramSystematic>,
    pseudo_count: f64,
}

#[derive(Debug, Clone)]
/// Up/down histogram templates for a single nuisance parameter.
pub struct HistogramSystematic {
    /// Down-variation bin contents (α = -1).
    pub down: Vec<f64>,
    /// Up-variation bin contents (α = +1).
    pub up: Vec<f64>,
    /// Interpolation code used to morph between down/nominal/up.
    pub interp_code: HistoSysInterpCode,
}

impl MorphingHistogramPdf {
    /// Construct a morphing histogram PDF.
    ///
    /// - `nominal_bin_content` and each systematic `down/up` must have length `n_bins`.
    /// - `bin_edges` must have length `n_bins + 1` and be strictly increasing.
    /// - `pseudo_count` is added to each bin before interpolation/normalization (for stability).
    pub fn new(
        observable: impl Into<String>,
        bin_edges: Vec<f64>,
        nominal_bin_content: Vec<f64>,
        systematics: Vec<HistogramSystematic>,
        pseudo_count: f64,
    ) -> Result<Self> {
        if bin_edges.len() < 2 {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf requires at least 2 bin edges, got {}",
                bin_edges.len()
            )));
        }
        if nominal_bin_content.len() + 1 != bin_edges.len() {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf nominal_bin_content length mismatch: expected {}, got {}",
                bin_edges.len() - 1,
                nominal_bin_content.len()
            )));
        }
        if systematics.is_empty() {
            return Err(Error::Validation(
                "MorphingHistogramPdf requires at least one systematic (n_params > 0)".into(),
            ));
        }
        if !pseudo_count.is_finite() || pseudo_count < 0.0 {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf pseudo_count must be finite and >=0, got {pseudo_count}"
            )));
        }

        for i in 0..bin_edges.len() {
            let e = bin_edges[i];
            if !e.is_finite() {
                return Err(Error::Validation(format!(
                    "MorphingHistogramPdf bin_edges[{i}] must be finite, got {e}"
                )));
            }
            if i > 0 && bin_edges[i - 1] >= e {
                return Err(Error::Validation(
                    "MorphingHistogramPdf bin edges must be strictly increasing".into(),
                ));
            }
        }

        for (i, w) in nominal_bin_content.iter().enumerate() {
            if !w.is_finite() || *w < 0.0 {
                return Err(Error::Validation(format!(
                    "MorphingHistogramPdf nominal_bin_content[{i}] must be finite and >=0, got {w}"
                )));
            }
        }

        let n_bins = nominal_bin_content.len();
        for (sidx, s) in systematics.iter().enumerate() {
            if s.down.len() != n_bins || s.up.len() != n_bins {
                return Err(Error::Validation(format!(
                    "MorphingHistogramPdf systematic[{sidx}] length mismatch: expected {n_bins} bins, got down={} up={}",
                    s.down.len(),
                    s.up.len()
                )));
            }
            for (i, w) in s.down.iter().enumerate() {
                if !w.is_finite() || *w < 0.0 {
                    return Err(Error::Validation(format!(
                        "MorphingHistogramPdf systematic[{sidx}].down[{i}] must be finite and >=0, got {w}"
                    )));
                }
            }
            for (i, w) in s.up.iter().enumerate() {
                if !w.is_finite() || *w < 0.0 {
                    return Err(Error::Validation(format!(
                        "MorphingHistogramPdf systematic[{sidx}].up[{i}] must be finite and >=0, got {w}"
                    )));
                }
            }
        }

        let mut log_bin_width = Vec::with_capacity(n_bins);
        for i in 0..n_bins {
            let width = bin_edges[i + 1] - bin_edges[i];
            if !(width.is_finite() && width > 0.0) {
                return Err(Error::Validation(format!(
                    "MorphingHistogramPdf bin width must be finite and >0, got width={width} for bin {i}"
                )));
            }
            log_bin_width.push(width.ln());
        }

        Ok(Self {
            observables: [observable.into()],
            bin_edges,
            log_bin_width,
            nominal_bin_content,
            systematics,
            pseudo_count,
        })
    }

    fn bin_index(&self, x: f64) -> Result<usize> {
        let x_min = self.bin_edges[0];
        let x_max = *self.bin_edges.last().unwrap_or(&x_min);
        if !(x.is_finite()) {
            return Err(Error::Validation("MorphingHistogramPdf requires finite x".into()));
        }
        if x < x_min || x > x_max {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf x out of range: x={x} not in [{x_min}, {x_max}]"
            )));
        }

        let n_bins = self.nominal_bin_content.len();
        if n_bins == 0 {
            return Err(Error::Validation("MorphingHistogramPdf has 0 bins".into()));
        }
        if x >= x_max {
            return Ok(n_bins - 1);
        }

        let k = self.bin_edges.partition_point(|e| *e <= x);
        if k == 0 {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf x below first edge unexpectedly: x={x}, x_min={x_min}"
            )));
        }
        let idx = k - 1;
        if idx >= n_bins {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf bin lookup failed for x={x}: idx={idx} >= n_bins={n_bins}"
            )));
        }
        Ok(idx)
    }

    fn validate_support_matches(&self, events: &EventStore, obs: &str) -> Result<()> {
        let (a, b) = events
            .bounds(obs)
            .ok_or_else(|| Error::Validation(format!("missing bounds for '{obs}'")))?;
        let x_min = self.bin_edges[0];
        let x_max = *self.bin_edges.last().unwrap_or(&x_min);
        let eps = 1e-12;
        if (a - x_min).abs() > eps || (b - x_max).abs() > eps {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf support mismatch: pdf=[{x_min}, {x_max}], events=({a}, {b})"
            )));
        }
        Ok(())
    }

    fn masses_and_total(&self, params: &[f64]) -> Result<(Vec<f64>, f64)> {
        if params.len() != self.systematics.len() {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf params length mismatch: expected {}, got {}",
                self.systematics.len(),
                params.len()
            )));
        }

        let n_bins = self.nominal_bin_content.len();
        let mut masses = Vec::with_capacity(n_bins);
        let mut total = 0.0f64;

        let eps_mass = f64::MIN_POSITIVE;
        for i in 0..n_bins {
            let nom = self.nominal_bin_content[i] + self.pseudo_count;
            let mut m = nom;

            for (s, &alpha) in self.systematics.iter().zip(params) {
                let down = s.down[i] + self.pseudo_count;
                let up = s.up[i] + self.pseudo_count;
                let (val, _) = histosys_interp(alpha, down, nom, up, s.interp_code)?;
                m += val - nom;
            }

            if !m.is_finite() {
                return Err(Error::Validation(format!(
                    "MorphingHistogramPdf interpolated mass is not finite (bin {i})"
                )));
            }
            if m <= 0.0 {
                // Guardrail: clamp to keep log-density defined.
                m = eps_mass;
            }

            masses.push(m);
            total += m;
        }

        if !(total.is_finite() && total > 0.0) {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf total mass must be finite and >0, got {total}"
            )));
        }
        Ok((masses, total))
    }
}

impl UnbinnedPdf for MorphingHistogramPdf {
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
                "MorphingHistogramPdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        let (masses, total) = self.masses_and_total(params)?;
        let log_total = total.ln();

        for (i, &x) in xs.iter().enumerate() {
            let idx = self.bin_index(x)?;
            out[i] = masses[idx].ln() - log_total - self.log_bin_width[idx];
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
                "MorphingHistogramPdf out_logp length mismatch: expected {n_events}, got {}",
                out_logp.len()
            )));
        }

        let n_params = self.n_params();
        if params.len() != n_params {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf params length mismatch: expected {n_params}, got {}",
                params.len()
            )));
        }
        if out_grad.len() != n_events * n_params {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf out_grad length mismatch: expected {}, got {}",
                n_events * n_params,
                out_grad.len()
            )));
        }

        let obs = self.observables[0].as_str();
        self.validate_support_matches(events, obs)?;

        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        let n_bins = self.nominal_bin_content.len();
        let mut masses = vec![0.0f64; n_bins];
        let mut dmass = vec![0.0f64; n_bins * n_params];

        let eps_mass = f64::MIN_POSITIVE;

        for i in 0..n_bins {
            let nom = self.nominal_bin_content[i] + self.pseudo_count;
            let mut m = nom;

            for (pidx, (s, &alpha)) in self.systematics.iter().zip(params).enumerate() {
                let down = s.down[i] + self.pseudo_count;
                let up = s.up[i] + self.pseudo_count;
                let (val, dval) = histosys_interp(alpha, down, nom, up, s.interp_code)?;
                m += val - nom;
                dmass[i * n_params + pidx] = dval;
            }

            if !m.is_finite() {
                return Err(Error::Validation(format!(
                    "MorphingHistogramPdf interpolated mass is not finite (bin {i})"
                )));
            }
            if m <= 0.0 {
                // Guardrail: clamp and zero derivatives for stability.
                m = eps_mass;
                for pidx in 0..n_params {
                    dmass[i * n_params + pidx] = 0.0;
                }
            }

            masses[i] = m;
        }

        let total: f64 = masses.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf total mass must be finite and >0, got {total}"
            )));
        }
        let inv_total = 1.0 / total;
        let log_total = total.ln();

        let mut dtotal = vec![0.0f64; n_params];
        for i in 0..n_bins {
            for pidx in 0..n_params {
                dtotal[pidx] += dmass[i * n_params + pidx];
            }
        }

        for (evt_idx, &x) in xs.iter().enumerate() {
            let bin_idx = self.bin_index(x)?;
            let m = masses[bin_idx];
            out_logp[evt_idx] = m.ln() - log_total - self.log_bin_width[bin_idx];

            let inv_m = 1.0 / m;
            for pidx in 0..n_params {
                let dm = dmass[bin_idx * n_params + pidx];
                out_grad[evt_idx * n_params + pidx] = dm * inv_m - dtotal[pidx] * inv_total;
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
                "MorphingHistogramPdf expects 1D support, got {} dims",
                support.len()
            )));
        }
        let (low, high) = support[0];
        let x_min = self.bin_edges[0];
        let x_max = *self.bin_edges.last().unwrap_or(&x_min);
        let eps = 1e-12;
        if (low - x_min).abs() > eps || (high - x_max).abs() > eps {
            return Err(Error::Validation(format!(
                "MorphingHistogramPdf support mismatch: pdf=[{x_min}, {x_max}], support=({low}, {high})"
            )));
        }

        let (masses, total) = self.masses_and_total(params)?;
        let mut cdf = Vec::with_capacity(masses.len());
        let mut acc = 0.0f64;
        for m in &masses {
            acc += *m / total;
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

        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let u = u01(rng);
            let idx = cdf.partition_point(|p| *p < u).min(masses.len().saturating_sub(1));
            let a = self.bin_edges[idx];
            let b = self.bin_edges[idx + 1];
            let x = a + (b - a) * u01(rng);
            xs.push(x.clamp(low, high));
        }

        let obs_name = self.observables[0].clone();
        let observables = vec![ObservableSpec::branch(obs_name.clone(), (low, high))];
        EventStore::from_columns(observables, vec![(obs_name, xs)], None)
    }
}
