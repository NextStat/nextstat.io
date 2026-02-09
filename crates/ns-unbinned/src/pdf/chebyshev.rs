use crate::event_store::EventStore;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// Chebyshev polynomial PDF on a bounded support, normalized on the observable bounds.
///
/// The unnormalized shape is:
///
/// `f(x) = 1 + Σ_{k=1..m} c_k T_k(x')`
///
/// where `x'` maps the observable bounds `[a,b]` to `[-1,1]`:
///
/// `x' = (2x - (a+b)) / (b-a)`
///
/// and `T_k` is the Chebyshev polynomial of the first kind.
///
/// This PDF requires that `f(x) > 0` on the full support. A lightweight guardrail checks
/// positivity on a fixed grid in `x'` at evaluation time.
#[derive(Debug, Clone)]
pub struct ChebyshevPdf {
    observables: [String; 1],
    order: usize,
}

impl ChebyshevPdf {
    /// Create a Chebyshev PDF of the given order (number of coefficients `c_1..c_order`).
    pub fn new(observable: impl Into<String>, order: usize) -> Result<Self> {
        if order == 0 {
            return Err(Error::Validation(
                "ChebyshevPdf order must be >= 1 (provide at least one coefficient)".into(),
            ));
        }
        Ok(Self { observables: [observable.into()], order })
    }

    #[inline]
    fn validate_params_len(&self, params: &[f64]) -> Result<()> {
        if params.len() != self.order {
            return Err(Error::Validation(format!(
                "ChebyshevPdf expects {} params (c_1..c_{}), got {}",
                self.order,
                self.order,
                params.len()
            )));
        }
        if params.iter().any(|x| !x.is_finite()) {
            return Err(Error::Validation("ChebyshevPdf params must be finite".into()));
        }
        Ok(())
    }

    #[inline]
    fn xprime(x: f64, a: f64, b: f64) -> f64 {
        // Map [a,b] -> [-1,1].
        let denom = b - a;
        let xp = (2.0 * x - (a + b)) / denom;
        xp.clamp(-1.0, 1.0)
    }

    /// Fill `out_t[k-1] = T_k(xp)` for `k=1..=order`.
    #[inline]
    fn fill_t_values(xp: f64, out_t: &mut [f64]) {
        let m = out_t.len();
        if m == 0 {
            return;
        }
        // T1(x) = x
        out_t[0] = xp;
        if m == 1 {
            return;
        }
        // Recurrence: T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
        let mut tkm1 = 1.0f64; // T0
        let mut tk = xp; // T1
        for k in 2..=m {
            let tkp1 = 2.0 * xp * tk - tkm1;
            out_t[k - 1] = tkp1;
            tkm1 = tk;
            tk = tkp1;
        }
    }

    #[inline]
    fn unnorm_f_from_t(params: &[f64], tvals: &[f64]) -> f64 {
        debug_assert_eq!(params.len(), tvals.len());
        let mut f = 1.0f64;
        for (&c, &t) in params.iter().zip(tvals) {
            f += c * t;
        }
        f
    }

    fn normalization_and_dlogi_dc(
        &self,
        params: &[f64],
        a: f64,
        b: f64,
    ) -> Result<(f64, Vec<f64>)> {
        debug_assert_eq!(params.len(), self.order);
        debug_assert!(a < b);

        let w = b - a;
        if !w.is_finite() || w <= 0.0 {
            return Err(Error::Validation(format!(
                "ChebyshevPdf invalid bounds: expected low < high, got ({a}, {b})"
            )));
        }

        // I = ∫_a^b f(x) dx = w + w * Σ_{k even} c_k / (1-k^2)
        // where k is the Chebyshev order (k >= 1) and k even => nonzero integral.
        let mut i = w;
        for (idx, &c) in params.iter().enumerate() {
            let k = idx + 1;
            if k % 2 == 0 {
                let denom = 1.0 - (k as f64) * (k as f64);
                i += w * c / denom;
            }
        }
        if !i.is_finite() || i <= 0.0 {
            return Err(Error::Validation(format!(
                "ChebyshevPdf normalization integral is not finite/positive: {i}"
            )));
        }

        let mut dlogi = vec![0.0f64; self.order];
        for (j, dlogi_j) in dlogi.iter_mut().enumerate() {
            let k = j + 1;
            if k % 2 == 0 {
                let denom = 1.0 - (k as f64) * (k as f64);
                let di_dc = w / denom;
                *dlogi_j = di_dc / i;
            }
        }

        Ok((i, dlogi))
    }

    fn guardrail_check_positive(&self, params: &[f64]) -> Result<()> {
        // Fast conservative check: evaluate f(x') on a small fixed grid in [-1,1].
        // If it goes non-positive anywhere, the PDF is invalid (log undefined).
        let n_grid = 128usize;
        let mut tvals = vec![0.0f64; self.order];
        for i in 0..n_grid {
            let xp = -1.0 + 2.0 * ((i as f64) + 0.5) / (n_grid as f64);
            Self::fill_t_values(xp, &mut tvals);
            let f = Self::unnorm_f_from_t(params, &tvals);
            if !f.is_finite() || f <= 0.0 {
                return Err(Error::Validation(format!(
                    "ChebyshevPdf is non-positive on support (guardrail failed): f(x')={f} at x'={xp}"
                )));
            }
        }
        Ok(())
    }
}

impl UnbinnedPdf for ChebyshevPdf {
    fn n_params(&self) -> usize {
        self.order
    }

    fn observables(&self) -> &[String] {
        &self.observables
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        self.validate_params_len(params)?;
        self.guardrail_check_positive(params)?;

        let n_events = events.n_events();
        if out.len() != n_events {
            return Err(Error::Validation(format!(
                "ChebyshevPdf out length mismatch: expected {n_events}, got {}",
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

        let (i, _dlogi) = self.normalization_and_dlogi_dc(params, a, b)?;
        let log_i = i.ln();

        let mut tvals = vec![0.0f64; self.order];
        for (i_evt, &x) in xs.iter().enumerate() {
            let xp = Self::xprime(x, a, b);
            Self::fill_t_values(xp, &mut tvals);
            let f = Self::unnorm_f_from_t(params, &tvals);
            if !f.is_finite() || f <= 0.0 {
                return Err(Error::Validation(format!(
                    "ChebyshevPdf is non-positive at data event {i_evt}: f={f} (x={x}, x'={xp})"
                )));
            }
            out[i_evt] = f.ln() - log_i;
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
        self.validate_params_len(params)?;
        self.guardrail_check_positive(params)?;

        let n_events = events.n_events();
        if out_logp.len() != n_events {
            return Err(Error::Validation(format!(
                "ChebyshevPdf out_logp length mismatch: expected {n_events}, got {}",
                out_logp.len()
            )));
        }
        let expected_grad_len = n_events * self.n_params();
        if out_grad.len() != expected_grad_len {
            return Err(Error::Validation(format!(
                "ChebyshevPdf out_grad length mismatch: expected {expected_grad_len}, got {}",
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

        let (i, dlogi_dc) = self.normalization_and_dlogi_dc(params, a, b)?;
        let log_i = i.ln();

        let mut tvals = vec![0.0f64; self.order];
        for (i_evt, &x) in xs.iter().enumerate() {
            let xp = Self::xprime(x, a, b);
            Self::fill_t_values(xp, &mut tvals);
            let f = Self::unnorm_f_from_t(params, &tvals);
            if !f.is_finite() || f <= 0.0 {
                return Err(Error::Validation(format!(
                    "ChebyshevPdf is non-positive at data event {i_evt}: f={f} (x={x}, x'={xp})"
                )));
            }

            out_logp[i_evt] = f.ln() - log_i;

            let inv_f = 1.0 / f;
            let base = i_evt * self.order;
            for j in 0..self.order {
                out_grad[base + j] = tvals[j] * inv_f - dlogi_dc[j];
            }
        }

        Ok(())
    }
}
