//! Forward-mode automatic differentiation via f32-based dual numbers.
//!
//! `Dual32` is identical to [`Dual`](crate::dual::Dual) but uses `f32` internally.
//! This enables testing f32 analytical gradient accuracy (for Metal GPU validation)
//! through the same `nll_generic::<Dual32>()` code path.

use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An f32 dual number for forward-mode AD.
///
/// `val` holds the primal value (f32), `dot` holds the derivative (f32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual32 {
    /// Primal (function) value.
    pub val: f32,
    /// Tangent (derivative) value.
    pub dot: f32,
}

impl Dual32 {
    /// Create a constant (derivative = 0).
    #[inline]
    pub fn constant(val: f32) -> Self {
        Self { val, dot: 0.0 }
    }

    /// Create an independent variable (derivative = 1).
    #[inline]
    pub fn var(val: f32) -> Self {
        Self { val, dot: 1.0 }
    }

    /// Create a dual with explicit tangent.
    #[inline]
    pub fn new(val: f32, dot: f32) -> Self {
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

    /// Power with f32 exponent: d/dx x^n = n * x^(n-1).
    #[inline]
    pub fn powf(self, n: f32) -> Self {
        Self { val: self.val.powf(n), dot: self.dot * n * self.val.powf(n - 1.0) }
    }

    /// Integer power: d/dx x^n = n * x^(n-1).
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self { val: self.val.powi(n), dot: self.dot * (n as f32) * self.val.powi(n - 1) }
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
    pub fn clamp(self, min: f32, max: f32) -> Self {
        if self.val < min {
            Self::constant(min)
        } else if self.val > max {
            Self::constant(max)
        } else {
            self
        }
    }
}

// --- Arithmetic: Dual32 op Dual32 ---

impl Add for Dual32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { val: self.val + rhs.val, dot: self.dot + rhs.dot }
    }
}

impl Sub for Dual32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { val: self.val - rhs.val, dot: self.dot - rhs.dot }
    }
}

impl Mul for Dual32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { val: self.val * rhs.val, dot: self.dot * rhs.val + self.val * rhs.dot }
    }
}

impl Div for Dual32 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            dot: (self.dot * rhs.val - self.val * rhs.dot) / (rhs.val * rhs.val),
        }
    }
}

impl Neg for Dual32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { val: -self.val, dot: -self.dot }
    }
}

// --- Sum ---

impl Sum for Dual32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Dual32::constant(0.0), |acc, x| acc + x)
    }
}

// --- From ---

impl From<f32> for Dual32 {
    fn from(val: f32) -> Self {
        Self::constant(val)
    }
}

// --- PartialOrd ---

impl PartialOrd for Dual32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_has_zero_derivative() {
        let c = Dual32::constant(5.0);
        assert_eq!(c.val, 5.0);
        assert_eq!(c.dot, 0.0);
    }

    #[test]
    fn test_var_has_unit_derivative() {
        let x = Dual32::var(3.0);
        assert_eq!(x.val, 3.0);
        assert_eq!(x.dot, 1.0);
    }

    #[test]
    fn test_ln_derivative() {
        // d/dx ln(x) = 1/x
        let x = Dual32::var(2.0);
        let y = x.ln();
        assert!((y.val - 2.0_f32.ln()).abs() < 1e-6);
        assert!((y.dot - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_exp_derivative() {
        // d/dx exp(x) = exp(x)
        let x = Dual32::var(1.0);
        let y = x.exp();
        assert!((y.val - 1.0_f32.exp()).abs() < 1e-6);
        assert!((y.dot - 1.0_f32.exp()).abs() < 1e-6);
    }

    #[test]
    fn test_mul_derivative() {
        // d/dx (x * x) = 2x
        let x = Dual32::var(3.0);
        let y = x * x;
        assert!((y.val - 9.0).abs() < 1e-6);
        assert!((y.dot - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_poisson_nll() {
        // f(lam) = lam - n*ln(lam), f'(lam) = 1 - n/lam
        let n = Dual32::constant(10.0);
        let lam = Dual32::var(12.0);
        let nll = lam - n * lam.ln();
        let expected_grad = 1.0 - 10.0 / 12.0;
        assert!((nll.dot - expected_grad).abs() < 1e-5);
    }
}
