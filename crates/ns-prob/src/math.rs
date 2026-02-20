//! Small numerically-stable math utilities used across probability code.

/// Stable `log(1 + exp(x))`.
///
/// Branchless: `log(1+exp(x)) = max(x,0) + log(1+exp(-|x|))`.
/// `f64::max` compiles to `maxsd` (no branch), single unconditional `exp(-|x|)`.
#[inline]
pub fn log1pexp(x: f64) -> f64 {
    let abs_x = x.abs();
    let e = (-abs_x).exp(); // always in (0, 1], no overflow
    x.max(0.0) + e.ln_1p()
}

/// Stable sigmoid: `1 / (1 + exp(-x))`.
///
/// Branchless core: single `exp(-|x|)`, then `cmov` for the sign flip.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    let abs_x = x.abs();
    let e = (-abs_x).exp();
    let recip = 1.0 / (1.0 + e);
    // x >= 0: sigmoid = 1/(1+exp(-x)) = recip
    // x <  0: sigmoid = exp(x)/(1+exp(x)) = e/(1+e) = e*recip
    if x >= 0.0 { recip } else { e * recip }
}

/// Stable `log(sigmoid(x))`.
#[inline]
pub fn log_sigmoid(x: f64) -> f64 {
    // log(sigmoid(x)) = -log(1 + exp(-x))
    if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() }
}

/// Fused `(log(1+exp(x)), sigmoid(x))` — single `exp()` call.
///
/// Equivalent to `(log1pexp(x), sigmoid(x))` but avoids computing the
/// exponential twice.  In logistic-regression inner loops this halves the
/// transcendental-math cost.
///
/// Branchless core: single unconditional `exp(-|x|)`.
/// `log1pexp(x) = max(x,0) + ln(1+exp(-|x|))` is algebraically exact.
/// The sigmoid branch compiles to `cmov` on x86.
#[inline(always)]
pub fn log1pexp_and_sigmoid(x: f64) -> (f64, f64) {
    let abs_x = x.abs();
    let e = (-abs_x).exp(); // always in (0, 1], no overflow
    let log_term = x.max(0.0) + e.ln_1p();
    let recip = 1.0 / (1.0 + e);
    let sigma = if x >= 0.0 { recip } else { e * recip };
    (log_term, sigma)
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
    fn test_log1pexp_and_sigmoid_matches_separate() {
        let xs: [f64; 9] = [-50.0, -10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0, 50.0];
        for x in xs {
            let (l, s) = log1pexp_and_sigmoid(x);
            let l_ref = log1pexp(x);
            let s_ref = sigmoid(x);
            assert!(
                (l - l_ref).abs() < 1e-15,
                "log1pexp mismatch at x={}: {} vs {}",
                x, l, l_ref
            );
            assert!(
                (s - s_ref).abs() < 1e-15,
                "sigmoid mismatch at x={}: {} vs {}",
                x, s, s_ref
            );
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
