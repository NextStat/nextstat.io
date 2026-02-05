//! [`Scalar`] trait: abstraction over `f64` and [`Dual`](crate::dual::Dual)
//! that enables writing NLL/expected-data code once, then reusing it
//! for both evaluation **and** forward-mode gradient computation.

use crate::dual::Dual;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A scalar type suitable for likelihood computation.
///
/// Implement this for `f64` (plain evaluation) and `Dual` (forward-mode AD).
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sum
    + PartialOrd
    + Sized
{
    /// Wrap an `f64` constant (derivative = 0 for AD types).
    fn from_f64(v: f64) -> Self;

    /// Extract the primal (function) value.
    fn value(&self) -> f64;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Exponential.
    fn exp(self) -> Self;

    /// Power with f64 exponent.
    fn powf(self, n: f64) -> Self;

    /// Integer power.
    fn powi(self, n: i32) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Maximum of two values (non-smooth; passes derivative of the winner).
    fn max_s(self, other: Self) -> Self;
}

// --- f64 implementation ---

impl Scalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn value(&self) -> f64 {
        *self
    }

    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }

    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }

    #[inline]
    fn powf(self, n: f64) -> Self {
        f64::powf(self, n)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn max_s(self, other: Self) -> Self {
        f64::max(self, other)
    }
}

// --- Dual implementation ---

impl Scalar for Dual {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Dual::constant(v)
    }

    #[inline]
    fn value(&self) -> f64 {
        self.val
    }

    #[inline]
    fn ln(self) -> Self {
        Dual::ln(self)
    }

    #[inline]
    fn exp(self) -> Self {
        Dual::exp(self)
    }

    #[inline]
    fn powf(self, n: f64) -> Self {
        Dual::powf(self, n)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Dual::powi(self, n)
    }

    #[inline]
    fn abs(self) -> Self {
        Dual::abs(self)
    }

    #[inline]
    fn max_s(self, other: Self) -> Self {
        Dual::max(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Generic Poisson NLL for a single bin.
    fn poisson_nll_bin<S: Scalar>(obs: f64, expected: S) -> S {
        let obs_s = S::from_f64(obs);
        let expected = expected.max_s(S::from_f64(1e-10));
        if obs > 0.0 { expected - obs_s * expected.ln() } else { expected }
    }

    #[test]
    fn test_scalar_f64_poisson() {
        let nll = poisson_nll_bin::<f64>(10.0, 12.0);
        let expected = 12.0 - 10.0 * 12.0_f64.ln();
        assert_relative_eq!(nll, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_scalar_dual_poisson_gradient() {
        // d/dlam [lam - n*ln(lam)] = 1 - n/lam
        let lam = Dual::var(12.0);
        let nll = poisson_nll_bin(10.0, lam);
        let expected_grad = 1.0 - 10.0 / 12.0;
        assert_relative_eq!(nll.dot, expected_grad, epsilon = 1e-12);
    }

    #[test]
    fn test_scalar_generic_code_works_for_both() {
        fn quadratic<S: Scalar>(x: S) -> S {
            x * x + S::from_f64(3.0) * x + S::from_f64(2.0)
        }

        // f64: just value
        let val: f64 = quadratic(3.0);
        assert_relative_eq!(val, 20.0, epsilon = 1e-12);

        // Dual: value + derivative (d/dx [x^2 + 3x + 2] = 2x + 3)
        let dual_result = quadratic(Dual::var(3.0));
        assert_relative_eq!(dual_result.val, 20.0, epsilon = 1e-12);
        assert_relative_eq!(dual_result.dot, 9.0, epsilon = 1e-12);
    }
}
