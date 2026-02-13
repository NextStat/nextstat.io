//! Forward-mode automatic differentiation via dual numbers.
//!
//! A dual number `Dual { val, dot }` represents a value and its derivative
//! with respect to a single variable. For multi-variable gradients, evaluate
//! the function N times, each time seeding one variable with `dot = 1.0`.

use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A dual number for forward-mode AD.
///
/// `val` holds the primal value, `dot` holds the derivative (tangent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Primal (function) value.
    pub val: f64,
    /// Tangent (derivative) value.
    pub dot: f64,
}

impl Dual {
    /// Create a constant (derivative = 0).
    #[inline]
    pub fn constant(val: f64) -> Self {
        Self { val, dot: 0.0 }
    }

    /// Create an independent variable (derivative = 1).
    #[inline]
    pub fn var(val: f64) -> Self {
        Self { val, dot: 1.0 }
    }

    /// Create a dual with explicit tangent.
    #[inline]
    pub fn new(val: f64, dot: f64) -> Self {
        Self { val, dot }
    }

    /// Natural logarithm: d/dx ln(x) = 1/x.
    #[inline]
    pub fn ln(self) -> Self {
        Self { val: self.val.ln(), dot: self.dot / self.val }
    }

    /// Exponential: d/dx exp(x) = exp(x).
    #[inline]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self { val: e, dot: self.dot * e }
    }

    /// Power with f64 exponent: d/dx x^n = n * x^(n-1).
    #[inline]
    pub fn powf(self, n: f64) -> Self {
        Self { val: self.val.powf(n), dot: self.dot * n * self.val.powf(n - 1.0) }
    }

    /// Integer power: d/dx x^n = n * x^(n-1).
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self { val: self.val.powi(n), dot: self.dot * (n as f64) * self.val.powi(n - 1) }
    }

    /// Square root: d/dx sqrt(x) = 1/(2*sqrt(x)).
    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self { val: s, dot: self.dot / (2.0 * s) }
    }

    /// Absolute value: d/dx |x| = sign(x).
    #[inline]
    pub fn abs(self) -> Self {
        Self { val: self.val.abs(), dot: self.dot * self.val.signum() }
    }

    /// Maximum of two duals. Derivative follows the larger operand.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.val >= other.val { self } else { other }
    }

    /// Clamp value to [min, max], propagating derivative correctly.
    #[inline]
    pub fn clamp(self, min: f64, max: f64) -> Self {
        if self.val < min {
            Self::constant(min)
        } else if self.val > max {
            Self::constant(max)
        } else {
            self
        }
    }
}

// --- Arithmetic: Dual op Dual ---

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { val: self.val + rhs.val, dot: self.dot + rhs.dot }
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { val: self.val - rhs.val, dot: self.dot - rhs.dot }
    }
}

impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { val: self.val * rhs.val, dot: self.dot * rhs.val + self.val * rhs.dot }
    }
}

impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            dot: (self.dot * rhs.val - self.val * rhs.dot) / (rhs.val * rhs.val),
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { val: -self.val, dot: -self.dot }
    }
}

// --- Arithmetic: Dual op f64 ---

impl Add<f64> for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self { val: self.val + rhs, dot: self.dot }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self { val: self.val - rhs, dot: self.dot }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self { val: self.val * rhs, dot: self.dot * rhs }
    }
}

impl Div<f64> for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self { val: self.val / rhs, dot: self.dot / rhs }
    }
}

// --- Arithmetic: f64 op Dual ---

impl Add<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn add(self, rhs: Dual) -> Dual {
        Dual { val: self + rhs.val, dot: rhs.dot }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn sub(self, rhs: Dual) -> Dual {
        Dual { val: self - rhs.val, dot: -rhs.dot }
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: Dual) -> Dual {
        Dual { val: self * rhs.val, dot: self * rhs.dot }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn div(self, rhs: Dual) -> Dual {
        Dual { val: self / rhs.val, dot: -self * rhs.dot / (rhs.val * rhs.val) }
    }
}

// --- Sum ---

impl Sum for Dual {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Dual::constant(0.0), |acc, x| acc + x)
    }
}

// --- From ---

impl From<f64> for Dual {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

// --- PartialOrd ---

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- Basic arithmetic derivatives ---

    #[test]
    fn test_constant_has_zero_derivative() {
        let c = Dual::constant(5.0);
        assert_eq!(c.val, 5.0);
        assert_eq!(c.dot, 0.0);
    }

    #[test]
    fn test_var_has_unit_derivative() {
        let x = Dual::var(3.0);
        assert_eq!(x.val, 3.0);
        assert_eq!(x.dot, 1.0);
    }

    #[test]
    fn test_add_derivative() {
        // d/dx (x + 5) = 1
        let x = Dual::var(3.0);
        let c = Dual::constant(5.0);
        let y = x + c;
        assert_relative_eq!(y.val, 8.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sub_derivative() {
        // d/dx (x - 2) = 1
        let x = Dual::var(3.0);
        let y = x - Dual::constant(2.0);
        assert_relative_eq!(y.val, 1.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mul_derivative() {
        // d/dx (x * x) = 2x
        let x = Dual::var(3.0);
        let y = x * x;
        assert_relative_eq!(y.val, 9.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_div_derivative() {
        // d/dx (1/x) = -1/x^2
        let x = Dual::var(2.0);
        let y = Dual::constant(1.0) / x;
        assert_relative_eq!(y.val, 0.5, epsilon = 1e-12);
        assert_relative_eq!(y.dot, -0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_neg_derivative() {
        // d/dx (-x) = -1
        let x = Dual::var(3.0);
        let y = -x;
        assert_relative_eq!(y.val, -3.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, -1.0, epsilon = 1e-12);
    }

    // --- Transcendental functions ---

    #[test]
    fn test_ln_derivative() {
        // d/dx ln(x) = 1/x
        let x = Dual::var(2.0);
        let y = x.ln();
        assert_relative_eq!(y.val, 2.0_f64.ln(), epsilon = 1e-12);
        assert_relative_eq!(y.dot, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_exp_derivative() {
        // d/dx exp(x) = exp(x)
        let x = Dual::var(1.0);
        let y = x.exp();
        assert_relative_eq!(y.val, 1.0_f64.exp(), epsilon = 1e-12);
        assert_relative_eq!(y.dot, 1.0_f64.exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_sqrt_derivative() {
        // d/dx sqrt(x) = 1/(2*sqrt(x))
        let x = Dual::var(4.0);
        let y = x.sqrt();
        assert_relative_eq!(y.val, 2.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_powf_derivative() {
        // d/dx x^3 = 3*x^2
        let x = Dual::var(2.0);
        let y = x.powf(3.0);
        assert_relative_eq!(y.val, 8.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 12.0, epsilon = 1e-12);
    }

    #[test]
    fn test_powi_derivative() {
        // d/dx x^2 = 2x
        let x = Dual::var(5.0);
        let y = x.powi(2);
        assert_relative_eq!(y.val, 25.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_abs_derivative() {
        // d/dx |x| = sign(x)
        let x = Dual::var(-3.0);
        let y = x.abs();
        assert_relative_eq!(y.val, 3.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, -1.0, epsilon = 1e-12);
    }

    // --- Mixed operations ---

    #[test]
    fn test_f64_add_dual() {
        let x = Dual::var(3.0);
        let y = 5.0 + x;
        assert_relative_eq!(y.val, 8.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_f64_mul_dual() {
        // d/dx (3*x) = 3
        let x = Dual::var(2.0);
        let y = 3.0 * x;
        assert_relative_eq!(y.val, 6.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_f64_div_dual() {
        // d/dx (6/x) = -6/x^2
        let x = Dual::var(3.0);
        let y = 6.0 / x;
        assert_relative_eq!(y.val, 2.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, -6.0 / 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_f64_sub_dual() {
        // d/dx (10 - x) = -1
        let x = Dual::var(3.0);
        let y = 10.0 - x;
        assert_relative_eq!(y.val, 7.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, -1.0, epsilon = 1e-12);
    }

    // --- Composition ---

    #[test]
    fn test_chain_rule_ln_of_square() {
        // f(x) = ln(x^2) = 2*ln(x)
        // f'(x) = 2/x
        let x = Dual::var(3.0);
        let y = (x * x).ln();
        assert_relative_eq!(y.val, (9.0_f64).ln(), epsilon = 1e-12);
        assert_relative_eq!(y.dot, 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chain_rule_exp_of_neg() {
        // f(x) = exp(-x)
        // f'(x) = -exp(-x)
        let x = Dual::var(2.0);
        let y = (-x).exp();
        assert_relative_eq!(y.val, (-2.0_f64).exp(), epsilon = 1e-12);
        assert_relative_eq!(y.dot, -(-2.0_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_poisson_nll_single_bin() {
        // Poisson NLL for single bin: f(lam) = lam - n*ln(lam)
        // f'(lam) = 1 - n/lam
        let n = 10.0;
        let lam = Dual::var(12.0);
        let nll = lam - Dual::constant(n) * lam.ln();
        let expected_deriv = 1.0 - n / 12.0;
        assert_relative_eq!(nll.dot, expected_deriv, epsilon = 1e-12);
    }

    #[test]
    fn test_gaussian_constraint() {
        // Gaussian constraint: f(x) = (x - mu)^2 / (2*sigma^2)
        // f'(x) = (x - mu) / sigma^2
        let mu = 0.0;
        let sigma = 1.0;
        let x = Dual::var(0.5);
        let constraint = (x - Dual::constant(mu)).powi(2) / (2.0 * sigma * sigma);
        let expected_deriv = (0.5 - mu) / (sigma * sigma);
        assert_relative_eq!(constraint.dot, expected_deriv, epsilon = 1e-12);
    }

    // --- Sum ---

    #[test]
    fn test_sum_iterator() {
        // f(x) = x + 2x + 3x = 6x
        // f'(x) = 6
        let x = Dual::var(1.0);
        let terms = vec![x, x * Dual::constant(2.0), x * Dual::constant(3.0)];
        let y: Dual = terms.into_iter().sum();
        assert_relative_eq!(y.val, 6.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 6.0, epsilon = 1e-12);
    }

    // --- Max / Clamp ---

    #[test]
    fn test_max_derivative() {
        let a = Dual::new(3.0, 1.0);
        let b = Dual::new(5.0, 0.0);
        let y = a.max(b);
        assert_relative_eq!(y.val, 5.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 0.0, epsilon = 1e-12); // b is larger, b.dot = 0

        let c = Dual::new(7.0, 1.0);
        let y2 = c.max(b);
        assert_relative_eq!(y2.val, 7.0, epsilon = 1e-12);
        assert_relative_eq!(y2.dot, 1.0, epsilon = 1e-12); // c is larger, c.dot = 1
    }

    #[test]
    fn test_clamp_derivative() {
        // Within range: derivative passes through
        let x = Dual::var(0.5);
        let y = x.clamp(0.0, 1.0);
        assert_relative_eq!(y.dot, 1.0, epsilon = 1e-12);

        // Below min: derivative = 0
        let x = Dual::var(-1.0);
        let y = x.clamp(0.0, 1.0);
        assert_relative_eq!(y.val, 0.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 0.0, epsilon = 1e-12);

        // Above max: derivative = 0
        let x = Dual::var(2.0);
        let y = x.clamp(0.0, 1.0);
        assert_relative_eq!(y.val, 1.0, epsilon = 1e-12);
        assert_relative_eq!(y.dot, 0.0, epsilon = 1e-12);
    }

    // --- Gradient via forward-mode ---

    #[test]
    fn test_gradient_multivariate() {
        // f(x, y) = x^2 * y + y^3
        // df/dx = 2*x*y
        // df/dy = x^2 + 3*y^2

        let x_val = 2.0;
        let y_val = 3.0;

        // df/dx: seed x with dot=1, y with dot=0
        let x = Dual::var(x_val);
        let y = Dual::constant(y_val);
        let f = x.powi(2) * y + y.powi(3);
        let dfdx = f.dot;
        assert_relative_eq!(dfdx, 2.0 * x_val * y_val, epsilon = 1e-12);

        // df/dy: seed x with dot=0, y with dot=1
        let x = Dual::constant(x_val);
        let y = Dual::var(y_val);
        let f = x.powi(2) * y + y.powi(3);
        let dfdy = f.dot;
        assert_relative_eq!(dfdy, x_val * x_val + 3.0 * y_val * y_val, epsilon = 1e-12);
    }

    // --- Validation against finite differences ---

    fn finite_diff<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn test_complex_expression_vs_finite_diff() {
        // f(x) = exp(-x^2/2) * ln(1+x^2)
        let f_scalar = |x: f64| (-x * x / 2.0).exp() * (1.0 + x * x).ln();

        let f_dual = |x: Dual| {
            let neg_half_x2 = (x * x) * Dual::constant(-0.5);
            let gaussian = neg_half_x2.exp();
            let log_term = (Dual::constant(1.0) + x * x).ln();
            gaussian * log_term
        };

        let x_val = 1.5;
        let dual_result = f_dual(Dual::var(x_val));
        let fd_result = finite_diff(f_scalar, x_val, 1e-7);

        assert_relative_eq!(dual_result.val, f_scalar(x_val), epsilon = 1e-12);
        assert_relative_eq!(dual_result.dot, fd_result, epsilon = 1e-6);
    }
}
