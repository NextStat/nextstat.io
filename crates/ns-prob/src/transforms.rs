//! Bijective transforms (bijectors) for unconstrained parameterization.
//!
//! Many inference algorithms (e.g., NUTS/HMC) operate in unconstrained space
//! `z ∈ R^n`. These transforms map between unconstrained `z` and constrained
//! parameters `theta`, providing Jacobian terms needed for correct densities.

use crate::math::{log_sigmoid, sigmoid, softplus};

/// A bijective transform from unconstrained `z` to constrained `theta`.
pub trait Bijector: Send + Sync {
    /// Map unconstrained -> constrained: `theta = forward(z)`
    fn forward(&self, z: f64) -> f64;
    /// Map constrained -> unconstrained: `z = inverse(theta)`
    fn inverse(&self, theta: f64) -> f64;
    /// Log absolute determinant of Jacobian: `log|dtheta/dz|`
    fn log_abs_det_jacobian(&self, z: f64) -> f64;
    /// Derivative of log|J| w.r.t. z: `d/dz log|dtheta/dz|`
    fn grad_log_abs_det_jacobian(&self, z: f64) -> f64;
    /// Jacobian element: `dtheta/dz`
    fn jacobian(&self, z: f64) -> f64;
}

/// Which bijector to use for positive parameters `(0, +inf)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositiveBijectorKind {
    /// `theta = exp(z)`.
    Exp,
    /// `theta = softplus(z)`.
    Softplus,
}

/// Options for building a [`ParameterTransform`] from bounds.
#[derive(Debug, Clone, Copy)]
pub struct ParameterTransformOptions {
    /// Mapping used when bounds are exactly `(0, +inf)`.
    pub positive_bijector: PositiveBijectorKind,
}

impl Default for ParameterTransformOptions {
    fn default() -> Self {
        Self { positive_bijector: PositiveBijectorKind::Exp }
    }
}

/// Identity: `(-inf, inf) -> (-inf, inf)`.
pub struct IdentityBijector;

impl Bijector for IdentityBijector {
    #[inline]
    fn forward(&self, z: f64) -> f64 {
        z
    }
    #[inline]
    fn inverse(&self, theta: f64) -> f64 {
        theta
    }
    #[inline]
    fn log_abs_det_jacobian(&self, _z: f64) -> f64 {
        0.0
    }
    #[inline]
    fn grad_log_abs_det_jacobian(&self, _z: f64) -> f64 {
        0.0
    }
    #[inline]
    fn jacobian(&self, _z: f64) -> f64 {
        1.0
    }
}

/// Exp: `(-inf, inf) -> (0, inf)`, `theta = exp(z)`, `log|J| = z`.
pub struct ExpBijector;

impl Bijector for ExpBijector {
    #[inline]
    fn forward(&self, z: f64) -> f64 {
        z.exp()
    }
    #[inline]
    fn inverse(&self, theta: f64) -> f64 {
        theta.ln()
    }
    #[inline]
    fn log_abs_det_jacobian(&self, z: f64) -> f64 {
        z
    }
    #[inline]
    fn grad_log_abs_det_jacobian(&self, _z: f64) -> f64 {
        1.0
    }
    #[inline]
    fn jacobian(&self, z: f64) -> f64 {
        z.exp()
    }
}

/// LowerBounded: `(-inf, inf) -> (a, inf)`, `theta = a + exp(z)`.
pub struct LowerBoundedBijector {
    lower: f64,
}

impl LowerBoundedBijector {
    /// Create a new lower-bounded bijector with given lower bound.
    pub fn new(lower: f64) -> Self {
        Self { lower }
    }
}

impl Bijector for LowerBoundedBijector {
    #[inline]
    fn forward(&self, z: f64) -> f64 {
        self.lower + z.exp()
    }
    #[inline]
    fn inverse(&self, theta: f64) -> f64 {
        (theta - self.lower).ln()
    }
    #[inline]
    fn log_abs_det_jacobian(&self, z: f64) -> f64 {
        // dtheta/dz = exp(z), log|J| = z
        z
    }
    #[inline]
    fn grad_log_abs_det_jacobian(&self, _z: f64) -> f64 {
        1.0
    }
    #[inline]
    fn jacobian(&self, z: f64) -> f64 {
        z.exp()
    }
}

/// UpperBounded: `(-inf, inf) -> (-inf, b)`, `theta = b - exp(z)`.
pub struct UpperBoundedBijector {
    upper: f64,
}

impl UpperBoundedBijector {
    /// Create a new upper-bounded bijector with given upper bound.
    pub fn new(upper: f64) -> Self {
        Self { upper }
    }
}

impl Bijector for UpperBoundedBijector {
    #[inline]
    fn forward(&self, z: f64) -> f64 {
        self.upper - z.exp()
    }
    #[inline]
    fn inverse(&self, theta: f64) -> f64 {
        // z = ln(b - theta)
        // Clamp to keep inverse finite if theta is marginally out of bounds due to FP noise.
        let d = (self.upper - theta).max(1e-15);
        d.ln()
    }
    #[inline]
    fn log_abs_det_jacobian(&self, z: f64) -> f64 {
        // dtheta/dz = -exp(z), log|J| = z
        z
    }
    #[inline]
    fn grad_log_abs_det_jacobian(&self, _z: f64) -> f64 {
        1.0
    }
    #[inline]
    fn jacobian(&self, z: f64) -> f64 {
        -z.exp()
    }
}

/// SoftplusLowerBounded: `(-inf, inf) -> (a, inf)`, `theta = a + softplus(z)`.
///
/// Compared to `LowerBoundedBijector` (exp), softplus has a gentler behavior for
/// large negative `z`, which can improve numerical stability in some models.
pub struct SoftplusBijector {
    lower: f64,
}

impl SoftplusBijector {
    /// Create a new softplus lower-bounded bijector with given lower bound.
    pub fn new(lower: f64) -> Self {
        Self { lower }
    }

    #[inline]
    fn inverse_softplus(y: f64) -> f64 {
        // Inverse of softplus(y) = ln(1 + exp(z)) is z = ln(exp(y) - 1).
        //
        // Use expm1 for stability and short-circuit for large y where exp(y) overflows.
        let y = y.max(1e-15);
        if y > 20.0 {
            // softplus(z) ~= z for large z, so inverse ~= y
            y
        } else {
            y.exp_m1().ln()
        }
    }
}

impl Bijector for SoftplusBijector {
    #[inline]
    fn forward(&self, z: f64) -> f64 {
        self.lower + softplus(z)
    }

    #[inline]
    fn inverse(&self, theta: f64) -> f64 {
        let y = (theta - self.lower).max(1e-15);
        Self::inverse_softplus(y)
    }

    #[inline]
    fn log_abs_det_jacobian(&self, z: f64) -> f64 {
        // d/dz softplus(z) = sigmoid(z)
        log_sigmoid(z)
    }

    #[inline]
    fn grad_log_abs_det_jacobian(&self, z: f64) -> f64 {
        // d/dz log(sigmoid(z)) = 1 - sigmoid(z) = sigmoid(-z)
        sigmoid(-z)
    }

    #[inline]
    fn jacobian(&self, z: f64) -> f64 {
        sigmoid(z)
    }
}

/// Sigmoid: `(-inf, inf) -> (a, b)`, `theta = a + (b-a)*sigmoid(z)`.
pub struct SigmoidBijector {
    lower: f64,
    #[allow(dead_code)]
    upper: f64,
    width: f64,
    log_width: f64,
}

impl SigmoidBijector {
    /// Create a new sigmoid bijector for the interval `[lower, upper]`.
    pub fn new(lower: f64, upper: f64) -> Self {
        let width = upper - lower;
        Self { lower, upper, width, log_width: width.ln() }
    }
}

impl Bijector for SigmoidBijector {
    #[inline]
    fn forward(&self, z: f64) -> f64 {
        self.lower + self.width * sigmoid(z)
    }

    #[inline]
    fn inverse(&self, theta: f64) -> f64 {
        // z = logit((theta - a) / (b - a))
        let p = (theta - self.lower) / self.width;
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        (p / (1.0 - p)).ln()
    }

    #[inline]
    fn log_abs_det_jacobian(&self, z: f64) -> f64 {
        // dtheta/dz = (b-a) * sigmoid(z) * (1 - sigmoid(z))
        // log|J| = log(b-a) + log_sigmoid(z) + log_sigmoid(-z)
        self.log_width + log_sigmoid(z) + log_sigmoid(-z)
    }

    #[inline]
    fn grad_log_abs_det_jacobian(&self, z: f64) -> f64 {
        // d/dz [log_sigmoid(z) + log_sigmoid(-z)]
        // = sigmoid(-z) - sigmoid(z) = 1 - 2*sigmoid(z)
        1.0 - 2.0 * sigmoid(z)
    }

    #[inline]
    fn jacobian(&self, z: f64) -> f64 {
        let s = sigmoid(z);
        self.width * s * (1.0 - s)
    }
}

/// Composite transform for a vector of parameters.
///
/// Each parameter gets its own bijector, selected from bounds.
pub struct ParameterTransform {
    bijectors: Vec<Box<dyn Bijector>>,
}

impl ParameterTransform {
    /// Create transforms from parameter bounds.
    ///
    /// Selection logic:
    /// - `(-inf, inf)` -> Identity
    /// - `(0, inf)` -> Exp (default; see [`Self::from_bounds_with_options`])
    /// - `(a, inf)` where `a > -inf` -> LowerBounded(a)
    /// - `(-inf, b)` where `b < inf` -> UpperBounded(b)
    /// - `(a, b)` where both finite -> Sigmoid(a, b)
    pub fn from_bounds(bounds: &[(f64, f64)]) -> Self {
        Self::from_bounds_with_options(bounds, ParameterTransformOptions::default())
    }

    /// Create transforms from parameter bounds with custom selection options.
    pub fn from_bounds_with_options(bounds: &[(f64, f64)], opts: ParameterTransformOptions) -> Self {
        let bijectors: Vec<Box<dyn Bijector>> = bounds
            .iter()
            .map(|&(lo, hi)| -> Box<dyn Bijector> {
                let lo_finite = lo > f64::NEG_INFINITY;
                let hi_finite = hi < f64::INFINITY;

                match (lo_finite, hi_finite) {
                    (false, false) => Box::new(IdentityBijector),
                    (true, false) => {
                        if !lo.is_finite() {
                            Box::new(IdentityBijector)
                        } else if lo == 0.0 {
                            match opts.positive_bijector {
                                PositiveBijectorKind::Exp => Box::new(ExpBijector),
                                PositiveBijectorKind::Softplus => Box::new(SoftplusBijector::new(0.0)),
                            }
                        } else {
                            Box::new(LowerBoundedBijector::new(lo))
                        }
                    }
                    (true, true) => {
                        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                            Box::new(IdentityBijector)
                        } else {
                            Box::new(SigmoidBijector::new(lo, hi))
                        }
                    }
                    (false, true) => {
                        if !hi.is_finite() {
                            Box::new(IdentityBijector)
                        } else {
                            Box::new(UpperBoundedBijector::new(hi))
                        }
                    }
                }
            })
            .collect();

        Self { bijectors }
    }

    /// Like [`Self::from_bounds`], but uses softplus for positive parameters `(0, +inf)`.
    pub fn from_bounds_softplus(bounds: &[(f64, f64)]) -> Self {
        Self::from_bounds_with_options(
            bounds,
            ParameterTransformOptions {
                positive_bijector: PositiveBijectorKind::Softplus,
            },
        )
    }

    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.bijectors.len()
    }

    /// Map unconstrained -> constrained.
    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        z.iter().zip(&self.bijectors).map(|(&zi, b)| b.forward(zi)).collect()
    }

    /// Map constrained -> unconstrained.
    pub fn inverse(&self, theta: &[f64]) -> Vec<f64> {
        theta.iter().zip(&self.bijectors).map(|(&ti, b)| b.inverse(ti)).collect()
    }

    /// Sum of log|J| over all parameters.
    pub fn log_abs_det_jacobian(&self, z: &[f64]) -> f64 {
        z.iter().zip(&self.bijectors).map(|(&zi, b)| b.log_abs_det_jacobian(zi)).sum()
    }

    /// Gradient of sum(log|J|) w.r.t. z.
    pub fn grad_log_abs_det_jacobian(&self, z: &[f64]) -> Vec<f64> {
        z.iter().zip(&self.bijectors).map(|(&zi, b)| b.grad_log_abs_det_jacobian(zi)).collect()
    }

    /// Diagonal Jacobian: `dtheta_i/dz_i` for each parameter.
    pub fn jacobian_diag(&self, z: &[f64]) -> Vec<f64> {
        z.iter().zip(&self.bijectors).map(|(&zi, b)| b.jacobian(zi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bijector_roundtrip(b: &dyn Bijector, z_values: &[f64], rtol: f64) {
        for &z in z_values {
            let theta = b.forward(z);
            let z_back = b.inverse(theta);
            let diff = (z - z_back).abs();
            let scale = z.abs().max(1.0);
            assert!(
                diff / scale < rtol,
                "Roundtrip failed: z={}, theta={}, z_back={}, diff={}",
                z,
                theta,
                z_back,
                diff
            );
        }
    }

    fn test_bijector_inverse_roundtrip(b: &dyn Bijector, theta_values: &[f64], rtol: f64) {
        for &theta in theta_values {
            let z = b.inverse(theta);
            let theta_back = b.forward(z);
            let diff = (theta - theta_back).abs();
            let scale = theta.abs().max(1.0);
            assert!(
                diff / scale < rtol,
                "Inverse roundtrip failed: theta={}, z={}, theta_back={}, diff={}",
                theta,
                z,
                theta_back,
                diff
            );
        }
    }

    fn test_bijector_grad_log_jac(b: &dyn Bijector, z_values: &[f64], rtol: f64) {
        let eps = 1e-7;
        for &z in z_values {
            let grad = b.grad_log_abs_det_jacobian(z);
            let f_plus = b.log_abs_det_jacobian(z + eps);
            let f_minus = b.log_abs_det_jacobian(z - eps);
            let grad_fd = (f_plus - f_minus) / (2.0 * eps);
            let diff = (grad - grad_fd).abs();
            let scale = grad.abs().max(1.0);
            assert!(
                diff / scale < rtol,
                "Grad log|J| failed: z={}, analytical={}, fd={}, diff={}",
                z,
                grad,
                grad_fd,
                diff
            );
        }
    }

    #[test]
    fn test_identity_roundtrip() {
        let b = IdentityBijector;
        let zs = vec![-3.0, -1.0, 0.0, 0.5, 2.0, 10.0];
        test_bijector_roundtrip(&b, &zs, 1e-15);
    }

    #[test]
    fn test_identity_jacobian() {
        let b = IdentityBijector;
        assert_eq!(b.log_abs_det_jacobian(1.0), 0.0);
        assert_eq!(b.jacobian(1.0), 1.0);
        assert_eq!(b.grad_log_abs_det_jacobian(1.0), 0.0);
    }

    #[test]
    fn test_exp_roundtrip() {
        let b = ExpBijector;
        let zs = vec![-5.0, -1.0, 0.0, 1.0, 3.0];
        test_bijector_roundtrip(&b, &zs, 1e-10);
    }

    #[test]
    fn test_exp_inverse_roundtrip() {
        let b = ExpBijector;
        let thetas = vec![0.01, 0.5, 1.0, 5.0, 100.0];
        test_bijector_inverse_roundtrip(&b, &thetas, 1e-10);
    }

    #[test]
    fn test_exp_grad_log_jac() {
        let b = ExpBijector;
        let zs = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        test_bijector_grad_log_jac(&b, &zs, 1e-7);
    }

    #[test]
    fn test_lower_bounded_roundtrip() {
        let b = LowerBoundedBijector::new(2.5);
        let zs = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        test_bijector_roundtrip(&b, &zs, 1e-10);
    }

    #[test]
    fn test_lower_bounded_inverse_roundtrip() {
        let b = LowerBoundedBijector::new(2.5);
        let thetas = vec![2.6, 3.0, 5.0, 10.0];
        test_bijector_inverse_roundtrip(&b, &thetas, 1e-10);
    }

    #[test]
    fn test_lower_bounded_grad_log_jac() {
        let b = LowerBoundedBijector::new(2.5);
        let zs = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        test_bijector_grad_log_jac(&b, &zs, 1e-7);
    }

    #[test]
    fn test_upper_bounded_roundtrip() {
        let b = UpperBoundedBijector::new(3.0);
        let zs = vec![-10.0, -2.0, 0.0, 1.5, 4.0];
        test_bijector_roundtrip(&b, &zs, 1e-10);
    }

    #[test]
    fn test_upper_bounded_inverse_roundtrip() {
        let b = UpperBoundedBijector::new(3.0);
        let thetas = vec![-100.0, -2.0, 0.0, 2.0, 2.9];
        test_bijector_inverse_roundtrip(&b, &thetas, 1e-10);
    }

    #[test]
    fn test_upper_bounded_grad_log_jac() {
        let b = UpperBoundedBijector::new(3.0);
        let zs = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        test_bijector_grad_log_jac(&b, &zs, 1e-7);
    }

    #[test]
    fn test_softplus_lower_bounded_roundtrip() {
        let b = SoftplusBijector::new(2.5);
        let zs = vec![-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
        test_bijector_roundtrip(&b, &zs, 1e-10);
    }

    #[test]
    fn test_softplus_lower_bounded_extreme_negative_saturates() {
        let b = SoftplusBijector::new(2.5);
        let z = -40.0;
        let theta = b.forward(z);
        assert!(
            (theta - 2.5).abs() < 1e-12,
            "softplus should saturate to lower for very negative z: theta={}",
            theta
        );

        let z_back = b.inverse(theta);
        assert!(
            z_back.is_finite() && z_back < -30.0,
            "inverse(theta) should be finite and very negative: {}",
            z_back
        );
    }

    #[test]
    fn test_softplus_lower_bounded_inverse_roundtrip() {
        let b = SoftplusBijector::new(2.5);
        let thetas = vec![2.5 + 1e-8, 2.6, 3.0, 5.0, 100.0];
        test_bijector_inverse_roundtrip(&b, &thetas, 1e-10);
    }

    #[test]
    fn test_softplus_lower_bounded_grad_log_jac() {
        let b = SoftplusBijector::new(2.5);
        let zs = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        test_bijector_grad_log_jac(&b, &zs, 1e-7);
    }

    #[test]
    fn test_sigmoid_roundtrip() {
        let b = SigmoidBijector::new(-5.0, 5.0);
        let zs = vec![-10.0, -2.0, 0.0, 2.0, 10.0];
        test_bijector_roundtrip(&b, &zs, 1e-10);
    }

    #[test]
    fn test_sigmoid_inverse_roundtrip() {
        let b = SigmoidBijector::new(-5.0, 5.0);
        let thetas = vec![-4.9, -2.0, 0.0, 2.0, 4.9];
        test_bijector_inverse_roundtrip(&b, &thetas, 1e-10);
    }

    #[test]
    fn test_sigmoid_grad_log_jac() {
        let b = SigmoidBijector::new(-5.0, 5.0);
        let zs = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        test_bijector_grad_log_jac(&b, &zs, 1e-6);
    }

    #[test]
    fn test_sigmoid_stays_in_bounds() {
        let b = SigmoidBijector::new(0.0, 10.0);
        for z in [-100.0, -10.0, 0.0, 10.0, 100.0] {
            let theta = b.forward(z);
            // At extreme z, sigmoid saturates to boundary (IEEE-754 rounding)
            assert!(theta >= 0.0 && theta <= 10.0, "theta={} out of bounds for z={}", theta, z);
        }
        // Moderate values should be strictly inside
        for z in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            let theta = b.forward(z);
            assert!(theta > 0.0 && theta < 10.0, "theta={} not strictly inside for z={}", theta, z);
        }
    }

    #[test]
    fn test_parameter_transform_from_bounds_selection() {
        // Typical bounds
        let bounds = vec![
            (0.0, 10.0),                        // Sigmoid
            (1e-10, 10.0),                      // Sigmoid
            (-5.0, 5.0),                        // Sigmoid
            (f64::NEG_INFINITY, f64::INFINITY), // Identity
        ];

        let t = ParameterTransform::from_bounds(&bounds);
        assert_eq!(t.dim(), 4);

        // Roundtrip
        let theta = vec![1.0, 1.0, 0.0, 0.5];
        let z = t.inverse(&theta);
        let theta_back = t.forward(&z);

        for (i, (&a, &b_val)) in theta.iter().zip(theta_back.iter()).enumerate() {
            let diff = (a - b_val).abs();
            assert!(
                diff < 1e-10,
                "ParameterTransform roundtrip failed at [{}]: {} vs {}, diff={}",
                i,
                a,
                b_val,
                diff
            );
        }
    }

    #[test]
    fn test_parameter_transform_jacobian_sum() {
        let bounds = vec![(0.0, 10.0), (-5.0, 5.0), (1e-10, 10.0)];
        let t = ParameterTransform::from_bounds(&bounds);

        let z = vec![0.5, -0.3, 1.2];
        let log_jac = t.log_abs_det_jacobian(&z);
        let grad_log_jac = t.grad_log_abs_det_jacobian(&z);

        // Verify against finite differences
        let eps = 1e-7;
        for (i, &g) in grad_log_jac.iter().enumerate() {
            let mut z_plus = z.clone();
            z_plus[i] += eps;
            let mut z_minus = z.clone();
            z_minus[i] -= eps;
            let g_fd =
                (t.log_abs_det_jacobian(&z_plus) - t.log_abs_det_jacobian(&z_minus)) / (2.0 * eps);
            let diff = (g - g_fd).abs();
            assert!(
                diff < 1e-6,
                "ParameterTransform grad_log_jac[{}]: analytical={}, fd={}, diff={}",
                i,
                g,
                g_fd,
                diff
            );
        }

        assert!(log_jac.is_finite(), "log_jac should be finite: {}", log_jac);
    }

    #[test]
    fn test_parameter_transform_softplus_keeps_positive_finite_for_large_z() {
        let bounds = vec![(0.0, f64::INFINITY)];
        let t = ParameterTransform::from_bounds_softplus(&bounds);

        // exp(1000) overflows, but softplus(1000) ~= 1000 should stay finite.
        let theta = t.forward(&[1000.0]);
        assert!(theta[0].is_finite() && theta[0] > 0.0, "theta={:?}", theta);

        let z_back = t.inverse(&theta);
        assert!((z_back[0] - 1000.0).abs() / 1000.0 < 1e-12, "z_back={:?}", z_back);
    }
}
