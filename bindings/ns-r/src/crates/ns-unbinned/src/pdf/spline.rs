use crate::event_store::EventStore;
use crate::pdf::UnbinnedPdf;
use ns_core::{Error, Result};

/// 1-D PDF defined by a monotonic cubic spline interpolation of density values at knots.
///
/// Given knot positions `x₀ < x₁ < … < xₖ` and corresponding non-negative density values
/// `y₀, y₁, …, yₖ`, the spline constructs a smooth, strictly positive interpolant. The
/// resulting curve is normalized analytically over `[x₀, xₖ]` so that `∫ p(x) dx = 1`.
///
/// Uses Fritsch–Carlson monotone cubic interpolation to guarantee positivity when all `yᵢ > 0`.
///
/// This PDF has **no shape parameters** (n_params = 0). It is a fixed template built from data.
pub struct SplinePdf {
    observable: [String; 1],
    /// Knot x-positions (strictly increasing, length K).
    knots_x: Vec<f64>,
    /// Knot density values (positive, length K).
    knots_y: Vec<f64>,
    /// Monotone Hermite slopes at each knot (length K).
    slopes: Vec<f64>,
    /// `ln(∫ spline(x) dx)` over `[x₀, xₖ]`.
    log_norm: f64,
}

impl SplinePdf {
    /// Construct a spline PDF from knot positions and density values.
    ///
    /// All `knots_y` values must be strictly positive (the spline represents a density).
    /// `knots_x` must be strictly increasing with at least 2 knots.
    pub fn from_knots(
        observable: impl Into<String>,
        knots_x: Vec<f64>,
        knots_y: Vec<f64>,
    ) -> Result<Self> {
        let k = knots_x.len();
        if k < 2 {
            return Err(Error::Validation("SplinePdf requires at least 2 knots".into()));
        }
        if knots_y.len() != k {
            return Err(Error::Validation(format!(
                "SplinePdf: knots_x length ({}) != knots_y length ({})",
                k,
                knots_y.len()
            )));
        }
        for i in 0..k {
            if !knots_x[i].is_finite() || !knots_y[i].is_finite() {
                return Err(Error::Validation(format!(
                    "SplinePdf: knot {} has non-finite value (x={}, y={})",
                    i, knots_x[i], knots_y[i]
                )));
            }
            if knots_y[i] <= 0.0 {
                return Err(Error::Validation(format!(
                    "SplinePdf: knot {} has non-positive density y={} (must be > 0)",
                    i, knots_y[i]
                )));
            }
        }
        for i in 1..k {
            if knots_x[i] <= knots_x[i - 1] {
                return Err(Error::Validation(format!(
                    "SplinePdf: knots_x must be strictly increasing, but x[{}]={} >= x[{}]={}",
                    i - 1,
                    knots_x[i - 1],
                    i,
                    knots_x[i]
                )));
            }
        }

        let slopes = fritsch_carlson_slopes(&knots_x, &knots_y);
        let log_norm = integrate_cubic_hermite(&knots_x, &knots_y, &slopes).ln();

        if !log_norm.is_finite() {
            return Err(Error::Validation(
                "SplinePdf: normalization integral is not finite".into(),
            ));
        }

        Ok(Self { observable: [observable.into()], knots_x, knots_y, slopes, log_norm })
    }

    /// Evaluate the (unnormalized) spline density at a single point.
    #[inline]
    fn eval_unnorm(&self, x: f64) -> f64 {
        let k = self.knots_x.len();

        // Clamp to support boundary values.
        if x <= self.knots_x[0] {
            return self.knots_y[0];
        }
        if x >= self.knots_x[k - 1] {
            return self.knots_y[k - 1];
        }

        // Binary search for the interval.
        let i = match self.knots_x.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(idx) => idx.min(k - 2),
            Err(idx) => (idx - 1).min(k - 2),
        };

        let h = self.knots_x[i + 1] - self.knots_x[i];
        let t = (x - self.knots_x[i]) / h;

        // Cubic Hermite basis.
        let h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
        let h10 = t * (1.0 - t) * (1.0 - t);
        let h01 = t * t * (3.0 - 2.0 * t);
        let h11 = t * t * (t - 1.0);

        let val = h00 * self.knots_y[i]
            + h10 * h * self.slopes[i]
            + h01 * self.knots_y[i + 1]
            + h11 * h * self.slopes[i + 1];

        // Clamp to tiny positive value to avoid log(0).
        val.max(f64::MIN_POSITIVE)
    }
}

impl UnbinnedPdf for SplinePdf {
    fn n_params(&self) -> usize {
        0
    }

    fn observables(&self) -> &[String] {
        &self.observable
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        if !params.is_empty() {
            return Err(Error::Validation(format!(
                "SplinePdf expects 0 params, got {}",
                params.len()
            )));
        }

        let n = events.n_events();
        if out.len() != n {
            return Err(Error::Validation(format!(
                "SplinePdf out length mismatch: expected {n}, got {}",
                out.len()
            )));
        }

        let obs = self.observable[0].as_str();
        let xs = events
            .column(obs)
            .ok_or_else(|| Error::Validation(format!("missing column '{obs}'")))?;

        for (i, &x) in xs.iter().enumerate() {
            let val = self.eval_unnorm(x);
            out[i] = val.ln() - self.log_norm;
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
                "SplinePdf out_grad must be empty (n_params=0), got len={}",
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
                "SplinePdf expects 0 params, got {}",
                params.len()
            )));
        }
        if support.len() != 1 {
            return Err(Error::Validation(format!(
                "SplinePdf sample expects 1D support, got {}D",
                support.len()
            )));
        }
        let (a, b) = support[0];

        // Build a piecewise CDF by integrating the spline on each interval.
        let k = self.knots_x.len();
        let norm = self.log_norm.exp();
        let mut cdf_at_knots = Vec::with_capacity(k);
        let mut cumulative = 0.0;
        cdf_at_knots.push(0.0);
        for i in 0..k - 1 {
            let seg_integral = integrate_segment(
                self.knots_x[i],
                self.knots_x[i + 1],
                self.knots_y[i],
                self.knots_y[i + 1],
                self.slopes[i],
                self.slopes[i + 1],
            );
            cumulative += seg_integral / norm;
            cdf_at_knots.push(cumulative);
        }
        // Fix floating-point: ensure last = 1.
        if let Some(last) = cdf_at_knots.last_mut() {
            *last = 1.0;
        }

        // Inverse CDF sampling via bisection.
        #[inline]
        fn u01(rng: &mut dyn rand::RngCore) -> f64 {
            let v = rng.next_u64();
            (v as f64 + 0.5) * (1.0 / 18446744073709551616.0_f64)
        }

        let mut xs = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let u = u01(rng);
            // Find the interval where cdf[i] <= u < cdf[i+1].
            let seg = cdf_at_knots.partition_point(|&c| c <= u).saturating_sub(1).min(k - 2);

            // Bisect within [knots_x[seg], knots_x[seg+1]] to find x where CDF(x) = u.
            let u_local = u - cdf_at_knots[seg];
            let seg_total = cdf_at_knots[seg + 1] - cdf_at_knots[seg];

            // Target fraction within this segment.
            let target = if seg_total > 0.0 { u_local / seg_total } else { 0.5 };

            let x_lo = self.knots_x[seg];
            let x_hi = self.knots_x[seg + 1];

            // Linear approximation as starting point, then refine with bisection.
            let mut lo = x_lo;
            let mut hi = x_hi;
            let seg_integral = integrate_segment(
                x_lo,
                x_hi,
                self.knots_y[seg],
                self.knots_y[seg + 1],
                self.slopes[seg],
                self.slopes[seg + 1],
            );
            let target_area = target * seg_integral;

            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                let area = integrate_segment_partial(
                    x_lo,
                    mid,
                    x_lo,
                    x_hi,
                    self.knots_y[seg],
                    self.knots_y[seg + 1],
                    self.slopes[seg],
                    self.slopes[seg + 1],
                );
                if area < target_area {
                    lo = mid;
                } else {
                    hi = mid;
                }
                if (hi - lo) < 1e-14 * (x_hi - x_lo).abs().max(1.0) {
                    break;
                }
            }

            xs.push(0.5 * (lo + hi));
        }

        let obs = crate::event_store::ObservableSpec::branch(self.observable[0].clone(), (a, b));
        EventStore::from_columns(vec![obs], vec![(self.observable[0].clone(), xs)], None)
    }
}

/// Fritsch–Carlson monotone cubic interpolation slopes.
///
/// Given knots `(x, y)` with strictly increasing `x`, compute slopes `m[i]` at each knot
/// such that the resulting cubic Hermite interpolant is monotone on each interval.
fn fritsch_carlson_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let k = x.len();
    debug_assert!(k >= 2);

    // Step 1: compute secants.
    let mut delta = Vec::with_capacity(k - 1);
    for i in 0..k - 1 {
        let d = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        delta.push(d);
    }

    // Step 2: initial slopes — average of adjacent secants, or one-sided at endpoints.
    let mut m = vec![0.0; k];
    m[0] = delta[0];
    for i in 1..k - 1 {
        if delta[i - 1].signum() != delta[i].signum() {
            m[i] = 0.0;
        } else {
            m[i] = 0.5 * (delta[i - 1] + delta[i]);
        }
    }
    m[k - 1] = delta[k - 2];

    // Step 3: Fritsch–Carlson modification for monotonicity.
    for i in 0..k - 1 {
        if delta[i].abs() < 1e-30 {
            // Flat segment: force zero slope at both endpoints.
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / delta[i];
            let beta = m[i + 1] / delta[i];

            // Check the Fritsch–Carlson condition: α² + β² ≤ 9.
            let phi = alpha * alpha + beta * beta;
            if phi > 9.0 {
                let tau = 3.0 / phi.sqrt();
                m[i] = tau * alpha * delta[i];
                m[i + 1] = tau * beta * delta[i];
            }
        }
    }

    m
}

/// Analytically integrate a cubic Hermite interpolant over the full knot range.
fn integrate_cubic_hermite(x: &[f64], y: &[f64], m: &[f64]) -> f64 {
    let k = x.len();
    let mut total = 0.0;
    for i in 0..k - 1 {
        total += integrate_segment(x[i], x[i + 1], y[i], y[i + 1], m[i], m[i + 1]);
    }
    total
}

/// Analytically integrate the cubic Hermite segment on `[x_i, x_{i+1}]`.
///
/// The Hermite polynomial `p(t) = h00·y₀ + h10·h·m₀ + h01·y₁ + h11·h·m₁` where `t = (x - x_i)/h`.
/// `∫₀¹ p(t) h dt = h · (y₀/2 + y₁/2 + h·(m₀ - m₁)/12)`.
#[inline]
fn integrate_segment(x0: f64, x1: f64, y0: f64, y1: f64, m0: f64, m1: f64) -> f64 {
    let h = x1 - x0;
    h * (0.5 * (y0 + y1) + h * (m0 - m1) / 12.0)
}

/// Integrate the cubic Hermite from `x_start` to `x_end` within a segment defined by
/// `[seg_x0, seg_x1]` with values `y0, y1` and slopes `m0, m1`.
#[allow(clippy::too_many_arguments, clippy::excessive_precision)]
fn integrate_segment_partial(
    x_start: f64,
    x_end: f64,
    seg_x0: f64,
    seg_x1: f64,
    y0: f64,
    y1: f64,
    m0: f64,
    m1: f64,
) -> f64 {
    let h = seg_x1 - seg_x0;
    if h <= 0.0 {
        return 0.0;
    }
    let t0 = ((x_start - seg_x0) / h).clamp(0.0, 1.0);
    let t1 = ((x_end - seg_x0) / h).clamp(0.0, 1.0);

    if (t1 - t0).abs() < 1e-30 {
        return 0.0;
    }

    // Integrate p(t) from t0 to t1: ∫ (h00·y₀ + h10·h·m₀ + h01·y₁ + h11·h·m₁) · h · dt.
    // Antiderivative coefficients for the Hermite basis in t:
    // p(t) = y₀(1+2t)(1-t)² + m₀·h·t(1-t)² + y₁·t²(3-2t) + m₁·h·t²(t-1)
    //
    // Expanding: p(t) = (y₀ - y₁ + h·m₀ + h·m₁)·(-2t³)
    //                  + (3(y₁ - y₀) - 2h·m₀ - h·m₁)·t²
    //                  + h·m₀·t + y₀
    // But simpler: just use 4-point Gauss-Legendre on [t0, t1].
    let mid = 0.5 * (t0 + t1);
    let half = 0.5 * (t1 - t0);

    // 4-point GL nodes and weights on [-1,1].
    const NODES: [f64; 4] = [
        -0.861_136_311_594_052_6,
        -0.339_981_043_584_856_26,
        0.339_981_043_584_856_26,
        0.861_136_311_594_052_6,
    ];
    const WEIGHTS: [f64; 4] = [
        0.347_854_845_137_453_86,
        0.652_145_154_862_546_14,
        0.652_145_154_862_546_14,
        0.347_854_845_137_453_86,
    ];

    let mut integral = 0.0;
    for (&node, &weight) in NODES.iter().zip(&WEIGHTS) {
        let t = mid + half * node;

        let h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
        let h10 = t * (1.0 - t) * (1.0 - t);
        let h01 = t * t * (3.0 - 2.0 * t);
        let h11 = t * t * (t - 1.0);

        let val = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
        integral += weight * val;
    }

    integral * half * h
}
