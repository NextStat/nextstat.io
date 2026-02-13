use std::f64::consts::SQRT_2;

#[inline]
pub(crate) fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &x in xs {
        s += (x - m).exp();
    }
    m + s.ln()
}

/// Return `log(exp(a) - exp(b))` for `a > b`, computed stably.
#[inline]
pub(crate) fn log_diff_exp(a: f64, b: f64) -> f64 {
    debug_assert!(a > b);
    a + (-(b - a).exp()).ln_1p()
}

#[inline]
pub(crate) fn standard_normal_logpdf(z: f64) -> f64 {
    // -0.5*ln(2π)
    const LOG_INV_SQRT_2PI: f64 = -0.918_938_533_204_672_7;
    LOG_INV_SQRT_2PI - 0.5 * z * z
}

#[inline]
pub(crate) fn standard_normal_pdf(z: f64) -> f64 {
    standard_normal_logpdf(z).exp()
}

#[inline]
pub(crate) fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-z / SQRT_2)
}
