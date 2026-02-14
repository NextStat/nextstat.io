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

/// Exponential with a conservative clamp to avoid overflow.
///
/// For `x > 700`, `exp(x)` can overflow to `inf` on some platforms. For count-model
/// likelihoods (Poisson / NegBin) this typically yields `inf` NLL and breaks line
/// searches; clamping keeps the objective finite so optimizers can recover.
#[inline]
pub fn exp_clamped(x: f64) -> f64 {
    // Clamp both sides:
    // - upper bound prevents exp overflow to +inf
    // - lower bound prevents exp underflow to 0, which can turn log(mu) into -inf and
    //   break line searches in count-model objectives (Poisson / NegBin).
    x.clamp(-700.0, 700.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log1pexp_matches_naive_moderate_values() {
        let xs: [f64; 7] = [-10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0];
        for x in xs {
            let naive = (1.0 + x.exp()).ln();
            let stable = log1pexp(x);
            assert!((naive - stable).abs() < 1e-12, "x={}: {} vs {}", x, naive, stable);
        }
    }

    #[test]
    fn test_log1pexp_is_finite_extremes() {
        let xs: [f64; 4] = [-1e6, -100.0, 100.0, 1e6];
        for x in xs {
            let y = log1pexp(x);
            assert!(y.is_finite(), "x={} produced {}", x, y);
        }
        assert!((log1pexp(1e6) - 1e6).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounds_and_symmetry() {
        let xs: [f64; 7] = [-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0];
        for x in xs {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({})={}", x, s);
            let t = sigmoid(-x);
            assert!((s + t - 1.0).abs() < 1e-15, "sigmoid symmetry failed at {}", x);
        }
    }

    #[test]
    fn test_log_sigmoid_matches_naive_moderate_values() {
        let xs: [f64; 7] = [-10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0];
        for x in xs {
            let naive = sigmoid(x).ln();
            let stable = log_sigmoid(x);
            assert!((naive - stable).abs() < 1e-12, "x={}: {} vs {}", x, naive, stable);
        }
    }

    #[test]
    fn test_exp_clamped_is_finite_extremes() {
        let xs: [f64; 4] = [-1e6, -100.0, 100.0, 1e6];
        for x in xs {
            let y = exp_clamped(x);
            assert!(y.is_finite(), "x={} produced {}", x, y);
            assert!(y >= 0.0);
        }
        // Clamp is active at very large x.
        assert!((exp_clamped(1e6).ln() - 700.0).abs() < 1e-12);
    }
}
