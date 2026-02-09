//! Unbinned mixture model (extended likelihood) with yields and constraints.

use crate::event_store::EventStore;
use crate::math::logsumexp;
use crate::pdf::UnbinnedPdf;
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel, PreparedModelRef};
use ns_core::{Error, Result};
use rayon::prelude::*;
use std::sync::Arc;

/// A model parameter.
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Stable parameter name.
    pub name: String,
    /// Suggested initial value.
    pub init: f64,
    /// Bounds `(low, high)` (LBFGS-B box constraints).
    pub bounds: (f64, f64),
    /// Optional constraint term (nuisance parameter prior).
    pub constraint: Option<Constraint>,
}

/// Constraint (penalty) term for nuisance parameters.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Gaussian constraint.
    Gaussian {
        /// Constraint mean.
        mean: f64,
        /// Constraint sigma (standard deviation), must be finite and > 0.
        sigma: f64,
    },
}

impl Constraint {
    fn nll_and_grad(&self, x: f64) -> Result<(f64, f64)> {
        match *self {
            Constraint::Gaussian { mean, sigma } => {
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(Error::Validation(format!(
                        "Gaussian constraint requires sigma > 0, got {sigma}"
                    )));
                }
                let z = (x - mean) / sigma;
                let nll = 0.5 * z * z + (sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln());
                let grad = z / sigma;
                Ok((nll, grad))
            }
        }
    }
}

/// Multiplicative rate modifier applied to a process yield.
#[derive(Debug, Clone)]
pub enum RateModifier {
    /// HistFactory-like NormSys modifier: a positive factor `f(α)` such that:
    ///
    /// - `f(0) = 1`
    /// - `f(+1) = hi`
    /// - `f(-1) = lo`
    ///
    /// Phase 2 default interpolation is piecewise-exponential:
    ///
    /// - if `α >= 0`: `f(α) = hi^α`
    /// - if `α < 0`:  `f(α) = lo^{-α}`
    NormSys {
        /// Index of the nuisance parameter `α` in the global parameter vector.
        alpha_index: usize,
        /// Factor at `α = -1` (must be finite and > 0).
        lo: f64,
        /// Factor at `α = +1` (must be finite and > 0).
        hi: f64,
    },
}

impl RateModifier {
    fn validate_param_indices(&self, n_params: usize) -> Result<()> {
        match *self {
            RateModifier::NormSys { alpha_index, .. } => {
                if alpha_index >= n_params {
                    return Err(Error::Validation(format!(
                        "NormSys alpha_index out of range: {alpha_index} >= {n_params}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Return `(factor, param_index, d/dα log(factor))`.
    fn factor_and_dlogf(&self, params: &[f64]) -> Result<(f64, usize, f64)> {
        match *self {
            RateModifier::NormSys { alpha_index, lo, hi } => {
                if !(lo.is_finite() && lo > 0.0 && hi.is_finite() && hi > 0.0) {
                    return Err(Error::Validation(format!(
                        "NormSys requires finite lo/hi > 0, got lo={lo}, hi={hi}"
                    )));
                }
                let alpha = *params.get(alpha_index).ok_or_else(|| {
                    Error::Validation(format!("NormSys alpha_index out of range: {alpha_index}"))
                })?;
                if !alpha.is_finite() {
                    return Err(Error::Validation(format!(
                        "NormSys requires finite alpha, got {alpha} at index {alpha_index}"
                    )));
                }

                let (logf, dlogf) = if alpha >= 0.0 {
                    let ln_hi = hi.ln();
                    (alpha * ln_hi, ln_hi)
                } else {
                    let ln_lo = lo.ln();
                    (-alpha * ln_lo, -ln_lo)
                };

                let f = logf.exp();
                if !f.is_finite() {
                    return Err(Error::Validation(format!(
                        "NormSys factor overflow/NaN: alpha={alpha}, lo={lo}, hi={hi}"
                    )));
                }
                Ok((f, alpha_index, dlogf))
            }
        }
    }
}

/// How a process yield depends on model parameters.
#[derive(Debug, Clone)]
pub enum YieldExpr {
    /// Fixed number of expected events.
    Fixed(f64),
    /// A free yield parameter `ν` (directly optimized).
    Parameter {
        /// Index in the global parameter vector.
        index: usize,
    },
    /// `base_yield × scale_param` (typical HEP signal-strength pattern).
    Scaled {
        /// Base expected yield (non-negative).
        base_yield: f64,
        /// Index of the scale parameter in the global parameter vector.
        scale_index: usize,
    },
    /// Apply multiplicative rate modifiers to a base yield expression.
    Modified {
        /// Base yield expression (must evaluate to a finite, non-negative yield).
        base: Box<YieldExpr>,
        /// Rate modifiers to apply multiplicatively.
        modifiers: Vec<RateModifier>,
    },
}

impl YieldExpr {
    fn validate_param_indices(&self, n_params: usize) -> Result<()> {
        match self {
            YieldExpr::Fixed(_) => Ok(()),
            YieldExpr::Parameter { index } => {
                if *index >= n_params {
                    return Err(Error::Validation(format!(
                        "yield parameter index out of range: {index} >= {n_params}"
                    )));
                }
                Ok(())
            }
            YieldExpr::Scaled { scale_index, .. } => {
                if *scale_index >= n_params {
                    return Err(Error::Validation(format!(
                        "yield scale_index out of range: {scale_index} >= {n_params}"
                    )));
                }
                Ok(())
            }
            YieldExpr::Modified { base, modifiers } => {
                base.validate_param_indices(n_params)?;
                for m in modifiers {
                    m.validate_param_indices(n_params)?;
                }
                Ok(())
            }
        }
    }

    fn value_and_sparse_grad(&self, params: &[f64], out: &mut Vec<(usize, f64)>) -> Result<f64> {
        match *self {
            YieldExpr::Fixed(v) => {
                if !v.is_finite() || v < 0.0 {
                    return Err(Error::Validation(format!(
                        "fixed yield must be finite and >=0, got {v}"
                    )));
                }
                Ok(v)
            }
            YieldExpr::Parameter { index } => {
                let v = *params.get(index).ok_or_else(|| {
                    Error::Validation(format!("yield parameter index out of range: {index}"))
                })?;
                if !v.is_finite() || v < 0.0 {
                    return Err(Error::Validation(format!(
                        "yield parameter must be finite and >=0, got {v} at index {index}"
                    )));
                }
                out.push((index, 1.0));
                Ok(v)
            }
            YieldExpr::Scaled { base_yield, scale_index } => {
                if !base_yield.is_finite() || base_yield < 0.0 {
                    return Err(Error::Validation(format!(
                        "base_yield must be finite and >=0, got {base_yield}"
                    )));
                }
                let s = *params.get(scale_index).ok_or_else(|| {
                    Error::Validation(format!("scale_index out of range: {scale_index}"))
                })?;
                if !s.is_finite() || s < 0.0 {
                    return Err(Error::Validation(format!(
                        "scale parameter must be finite and >=0, got {s} at index {scale_index}"
                    )));
                }
                out.push((scale_index, base_yield));
                Ok(base_yield * s)
            }
            YieldExpr::Modified { ref base, ref modifiers } => {
                let mut base_sparse = Vec::new();
                let nu0 = base.value_and_sparse_grad(params, &mut base_sparse)?;
                debug_assert!(nu0 >= 0.0);

                let mut m = 1.0f64;
                let mut dlogfs: Vec<(usize, f64)> = Vec::with_capacity(modifiers.len());
                for modif in modifiers {
                    let (f, idx, dlogf) = modif.factor_and_dlogf(params)?;
                    m *= f;
                    if dlogf != 0.0 {
                        dlogfs.push((idx, dlogf));
                    }
                }

                let nu = nu0 * m;
                if !nu.is_finite() || nu < 0.0 {
                    return Err(Error::Validation(format!(
                        "modified yield is not finite / negative: base={nu0}, factor={m} => nu={nu}"
                    )));
                }

                for (idx, dnu0) in base_sparse {
                    out.push((idx, dnu0 * m));
                }
                for (idx, dlogf) in dlogfs {
                    // d/dα [nu0 * Π f_i] = nu * d/dα log(f_i) (if multiple share a parameter,
                    // repeated indices are fine; the caller will accumulate them).
                    out.push((idx, nu * dlogf));
                }

                Ok(nu)
            }
        }
    }
}

/// One physics process (signal/background) in an unbinned channel.
pub struct Process {
    /// Process name (stable).
    pub name: String,
    /// Shape model for this process.
    pub pdf: Arc<dyn UnbinnedPdf>,
    /// Global parameter indices used as the PDF shape parameters.
    ///
    /// Length must equal `pdf.n_params()`.
    pub shape_param_indices: Vec<usize>,
    /// Yield model.
    pub yield_expr: YieldExpr,
}

/// An unbinned channel containing observed events and a mixture of processes.
pub struct UnbinnedChannel {
    /// Channel name.
    pub name: String,
    /// Whether this channel contributes to the likelihood (vs validation-only).
    pub include_in_fit: bool,
    /// Observed events.
    pub data: Arc<EventStore>,
    /// Processes (signal + backgrounds).
    pub processes: Vec<Process>,
}

/// An extended unbinned model (multi-channel mixture + constraints).
pub struct UnbinnedModel {
    parameters: Vec<Parameter>,
    poi_index: Option<usize>,
    channels: Vec<UnbinnedChannel>,
}

impl UnbinnedModel {
    /// Create a new unbinned model.
    pub fn new(
        parameters: Vec<Parameter>,
        channels: Vec<UnbinnedChannel>,
        poi_index: Option<usize>,
    ) -> Result<Self> {
        if parameters.is_empty() {
            return Err(Error::Validation("UnbinnedModel requires at least one parameter".into()));
        }
        if channels.is_empty() {
            return Err(Error::Validation("UnbinnedModel requires at least one channel".into()));
        }
        if let Some(poi) = poi_index
            && poi >= parameters.len()
        {
            return Err(Error::Validation(format!(
                "poi_index out of range: {poi} >= {}",
                parameters.len()
            )));
        }

        for p in &parameters {
            if !p.init.is_finite() {
                return Err(Error::Validation(format!(
                    "parameter '{}' init is not finite",
                    p.name
                )));
            }
            if p.bounds.0.is_nan() || p.bounds.1.is_nan() || p.bounds.0 > p.bounds.1 {
                return Err(Error::Validation(format!(
                    "parameter '{}' has invalid bounds {:?}",
                    p.name, p.bounds
                )));
            }
            if p.init < p.bounds.0 || p.init > p.bounds.1 {
                return Err(Error::Validation(format!(
                    "parameter '{}' init {} outside bounds {:?}",
                    p.name, p.init, p.bounds
                )));
            }
            if let Some(Constraint::Gaussian { sigma, .. }) = &p.constraint
                && (!sigma.is_finite() || *sigma <= 0.0)
            {
                return Err(Error::Validation(format!(
                    "parameter '{}' has invalid Gaussian constraint sigma {sigma}",
                    p.name
                )));
            }
        }

        // Validate channel/process param indices.
        for ch in &channels {
            for proc in &ch.processes {
                if proc.shape_param_indices.len() != proc.pdf.n_params() {
                    return Err(Error::Validation(format!(
                        "process '{}' shape_param_indices length {} != pdf.n_params() {}",
                        proc.name,
                        proc.shape_param_indices.len(),
                        proc.pdf.n_params()
                    )));
                }
                for &idx in &proc.shape_param_indices {
                    if idx >= parameters.len() {
                        return Err(Error::Validation(format!(
                            "process '{}' references out-of-range shape parameter index {idx}",
                            proc.name
                        )));
                    }
                }
                proc.yield_expr.validate_param_indices(parameters.len())?;
            }
        }

        Ok(Self { parameters, poi_index, channels })
    }

    /// Access parameters.
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    /// Access channels.
    pub fn channels(&self) -> &[UnbinnedChannel] {
        &self.channels
    }

    fn validate_params_len(&self, len: usize) -> Result<()> {
        if len != self.parameters.len() {
            return Err(Error::Validation(format!(
                "parameter length mismatch: expected {}, got {}",
                self.parameters.len(),
                len
            )));
        }
        Ok(())
    }

    fn nll_and_grad_internal(
        &self,
        params: &[f64],
        want_grad: bool,
    ) -> Result<(f64, Option<Vec<f64>>)> {
        self.validate_params_len(params.len())?;

        let mut nll = 0.0f64;
        let mut grad = if want_grad { Some(vec![0.0f64; params.len()]) } else { None };

        for ch in &self.channels {
            if !ch.include_in_fit {
                continue;
            }
            if ch.data.weights().is_some() {
                return Err(Error::Validation(format!(
                    "weighted observed data is not supported (channel '{}')",
                    ch.name
                )));
            }

            let n_events = ch.data.n_events();
            let n_proc = ch.processes.len();
            if n_proc == 0 {
                return Err(Error::Validation(format!("channel '{}' has no processes", ch.name)));
            }

            // Per-process yields and sparse yield gradients.
            let mut yields = vec![0.0f64; n_proc];
            let mut dyields: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_proc];

            // Per-process logp and (optional) grad buffers.
            let mut logps: Vec<Vec<f64>> = Vec::with_capacity(n_proc);
            let mut dlogps: Vec<Vec<f64>> = Vec::with_capacity(n_proc);

            for (pidx, proc) in ch.processes.iter().enumerate() {
                // Yields
                let mut sparse = Vec::new();
                let nu = proc.yield_expr.value_and_sparse_grad(params, &mut sparse)?;
                yields[pidx] = nu;
                dyields[pidx] = sparse;

                // PDF shape params
                let mut shape_params = Vec::with_capacity(proc.shape_param_indices.len());
                for &idx in &proc.shape_param_indices {
                    shape_params.push(params[idx]);
                }

                let mut lp = vec![0.0f64; n_events];
                if want_grad {
                    let mut dlp = vec![0.0f64; n_events * proc.pdf.n_params()];
                    proc.pdf.log_prob_grad_batch(&ch.data, &shape_params, &mut lp, &mut dlp)?;
                    logps.push(lp);
                    dlogps.push(dlp);
                } else {
                    proc.pdf.log_prob_batch(&ch.data, &shape_params, &mut lp)?;
                    logps.push(lp);
                }
            }

            let nu_tot: f64 = yields.iter().sum();
            nll += nu_tot;

            // Precompute shape offsets for a packed accumulator.
            let mut shape_offsets = vec![0usize; n_proc];
            let mut total_shape = 0usize;
            for (p, proc) in ch.processes.iter().enumerate() {
                shape_offsets[p] = total_shape;
                total_shape += proc.pdf.n_params();
            }

            if let Some(g) = grad.as_mut() {
                // Yield gradient contribution from +nu_tot.
                for sparse in &dyields {
                    for &(idx, dnu) in sparse {
                        g[idx] += dnu;
                    }
                }

                #[derive(Clone)]
                struct Acc {
                    sum_logf: f64,
                    sum_r_over_nu: Vec<f64>,
                    sum_r_dlogp: Vec<f64>,
                    tmp_terms: Vec<f64>,
                }

                let init = || Acc {
                    sum_logf: 0.0,
                    sum_r_over_nu: vec![0.0; n_proc],
                    sum_r_dlogp: vec![0.0; total_shape],
                    tmp_terms: vec![0.0; n_proc],
                };

                let acc = (0..n_events)
                    .into_par_iter()
                    .fold(init, |mut acc, i| {
                        for p in 0..n_proc {
                            let nu = yields[p];
                            acc.tmp_terms[p] =
                                if nu > 0.0 { nu.ln() + logps[p][i] } else { f64::NEG_INFINITY };
                        }
                        let logf = logsumexp(&acc.tmp_terms);
                        acc.sum_logf += logf;

                        for p in 0..n_proc {
                            let nu = yields[p];
                            if nu <= 0.0 {
                                continue;
                            }
                            let r = (acc.tmp_terms[p] - logf).exp();
                            acc.sum_r_over_nu[p] += r / nu;

                            let nsp = ch.processes[p].pdf.n_params();
                            if nsp == 0 {
                                continue;
                            }
                            let off = shape_offsets[p];
                            let base = i * nsp;
                            for j in 0..nsp {
                                acc.sum_r_dlogp[off + j] += r * dlogps[p][base + j];
                            }
                        }

                        acc
                    })
                    .reduce(init, |mut a, b| {
                        a.sum_logf += b.sum_logf;
                        for p in 0..n_proc {
                            a.sum_r_over_nu[p] += b.sum_r_over_nu[p];
                        }
                        for k in 0..total_shape {
                            a.sum_r_dlogp[k] += b.sum_r_dlogp[k];
                        }
                        a
                    });

                nll -= acc.sum_logf;

                // Yield params: - Σ_i (dnu * r/nu) = -dnu * Σ_i r/nu
                for (dy_p, &sum_r_p) in dyields.iter().zip(&acc.sum_r_over_nu) {
                    for &(idx, dnu) in dy_p {
                        g[idx] -= dnu * sum_r_p;
                    }
                }

                // Shape params: - Σ_i r * dlogp/dθ
                for (proc, &off) in ch.processes.iter().zip(&shape_offsets) {
                    if proc.pdf.n_params() == 0 {
                        continue;
                    }
                    for (j, &global_idx) in proc.shape_param_indices.iter().enumerate() {
                        g[global_idx] -= acc.sum_r_dlogp[off + j];
                    }
                }
            } else {
                #[derive(Clone)]
                struct NllAcc {
                    sum_logf: f64,
                    tmp_terms: Vec<f64>,
                }

                let init = || NllAcc { sum_logf: 0.0, tmp_terms: vec![0.0; n_proc] };
                let acc = (0..n_events)
                    .into_par_iter()
                    .fold(init, |mut acc, i| {
                        for p in 0..n_proc {
                            let nu = yields[p];
                            acc.tmp_terms[p] =
                                if nu > 0.0 { nu.ln() + logps[p][i] } else { f64::NEG_INFINITY };
                        }
                        acc.sum_logf += logsumexp(&acc.tmp_terms);
                        acc
                    })
                    .reduce(init, |mut a, b| {
                        a.sum_logf += b.sum_logf;
                        a
                    });

                nll -= acc.sum_logf;
            }
        }

        // Constraint terms.
        for (idx, p) in self.parameters.iter().enumerate() {
            let Some(c) = &p.constraint else { continue };
            let (cnll, cgrad) = c.nll_and_grad(params[idx])?;
            nll += cnll;
            if let Some(g) = grad.as_mut() {
                g[idx] += cgrad;
            }
        }

        Ok((nll, grad))
    }
}

impl LogDensityModel for UnbinnedModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.parameters.len()
    }

    fn parameter_names(&self) -> Vec<String> {
        self.parameters.iter().map(|p| p.name.clone()).collect()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.parameters.iter().map(|p| p.bounds).collect()
    }

    fn parameter_init(&self) -> Vec<f64> {
        self.parameters.iter().map(|p| p.init).collect()
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let (nll, _) = self.nll_and_grad_internal(params, false)?;
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let (_, grad) = self.nll_and_grad_internal(params, true)?;
        Ok(grad.unwrap_or_default())
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

impl PoiModel for UnbinnedModel {
    fn poi_index(&self) -> Option<usize> {
        self.poi_index
    }
}

impl FixedParamModel for UnbinnedModel {
    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        let mut m = self.clone();
        if let Some(p) = m.parameters.get_mut(param_idx) {
            p.init = value;
            p.bounds = (value, value);
        }
        m
    }
}

impl Clone for Process {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            pdf: self.pdf.clone(),
            shape_param_indices: self.shape_param_indices.clone(),
            yield_expr: self.yield_expr.clone(),
        }
    }
}

impl Clone for UnbinnedChannel {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            include_in_fit: self.include_in_fit,
            data: self.data.clone(),
            processes: self.processes.clone(),
        }
    }
}

impl Clone for UnbinnedModel {
    fn clone(&self) -> Self {
        Self {
            parameters: self.parameters.clone(),
            poi_index: self.poi_index,
            channels: self.channels.clone(),
        }
    }
}
