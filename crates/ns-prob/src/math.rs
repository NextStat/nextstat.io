//! Small numerically-stable math utilities used across probability code.

/// Stable `log(1 + exp(x))`.
#[inline]
pub fn log1pexp(x: f64) -> f64 {
    // For large positive x, exp(x) overflows; use x + log1p(exp(-x)).
    // For large negative x, exp(x) underflows to 0; log1p is fine.
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// Stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Stable `log(sigmoid(x))`.
#[inline]
pub fn log_sigmoid(x: f64) -> f64 {
    // log(sigmoid(x)) = -log(1 + exp(-x))
    if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() }
}

/// Stable softplus: `log(1 + exp(x))`.
#[inline]
pub fn softplus(x: f64) -> f64 {
    log1pexp(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log1pexp_matches_naive_moderate_values() {
        for x in [-10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0] {
            let naive = (1.0 + x.exp()).ln();
            let stable = log1pexp(x);
            assert!((naive - stable).abs() < 1e-12, "x={}: {} vs {}", x, naive, stable);
        }
    }

    #[test]
    fn test_log1pexp_is_finite_extremes() {
        for x in [-1e6, -100.0, 100.0, 1e6] {
            let y = log1pexp(x);
            assert!(y.is_finite(), "x={} produced {}", x, y);
        }
        assert!((log1pexp(1e6) - 1e6).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounds_and_symmetry() {
        for x in [-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0] {
            let s = sigmoid(x);
            assert!(s >= 0.0 && s <= 1.0, "sigmoid({})={}", x, s);
            let t = sigmoid(-x);
            assert!((s + t - 1.0).abs() < 1e-15, "sigmoid symmetry failed at {}", x);
        }
    }

    #[test]
    fn test_log_sigmoid_matches_naive_moderate_values() {
        for x in [-10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0] {
            let naive = sigmoid(x).ln();
            let stable = log_sigmoid(x);
            assert!((naive - stable).abs() < 1e-12, "x={}: {} vs {}", x, naive, stable);
        }
    }
}

