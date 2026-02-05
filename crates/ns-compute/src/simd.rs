//! SIMD-optimized kernels for statistical computations.
//!
//! Uses `wide::f64x4` for 4-wide SIMD operations on f64.
//! Automatically dispatches to AVX2 on x86_64 and NEON on aarch64.
//!
//! Note: `wide::f64x4::ln()` has ~1000 ULP error, so we use lane-by-lane
//! scalar `f64::ln()` for accurate logarithms while still benefiting from
//! SIMD for arithmetic (add, sub, mul, FMA-like patterns).

use wide::f64x4;

/// Accurate lane-by-lane `ln()` using scalar `f64::ln()`.
///
/// `wide::f64x4::ln()` has insufficient precision (~1000 ULP).
/// This function extracts lanes, computes `f64::ln()` on each, and repacks.
#[inline(always)]
fn ln_f64x4(v: f64x4) -> f64x4 {
    let arr: [f64; 4] = v.into();
    f64x4::from([arr[0].ln(), arr[1].ln(), arr[2].ln(), arr[3].ln()])
}

/// Compute Poisson negative log-likelihood using SIMD.
///
/// Uses branchless formula:
///   `nll_i = exp_i + mask_i * (ln_factorial_i - obs_i * ln(exp_i))`
///
/// When `mask_i = 0` (obs == 0): `nll_i = exp_i` (correct Poisson term).
/// When `mask_i = 1` (obs > 0): `nll_i = exp_i - obs_i * ln(exp_i) + ln_factorial_i`.
///
/// # Arguments
/// * `expected` - Expected counts per bin (must be clamped >= 1e-10 by caller)
/// * `observed` - Observed counts per bin
/// * `ln_factorials` - Pre-computed `lgamma(obs + 1)` per bin
/// * `obs_mask` - `1.0` if `obs > 0`, else `0.0`
///
/// # Panics
/// Panics if slice lengths are not equal.
pub fn poisson_nll_simd(
    expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    obs_mask: &[f64],
) -> f64 {
    let n = expected.len();
    assert_eq!(n, observed.len());
    assert_eq!(n, ln_factorials.len());
    assert_eq!(n, obs_mask.len());

    if !use_simd() {
        return poisson_nll_scalar(expected, observed, ln_factorials, obs_mask);
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let exp = f64x4::from(&expected[offset..offset + 4]);
        let obs = f64x4::from(&observed[offset..offset + 4]);
        let lnf = f64x4::from(&ln_factorials[offset..offset + 4]);
        let mask = f64x4::from(&obs_mask[offset..offset + 4]);

        let ln_exp = ln_f64x4(exp);
        // nll = exp + mask * (lnf - obs * ln_exp)
        let obs_ln_exp = obs * ln_exp;
        let bracket = lnf - obs_ln_exp;
        acc += exp + mask * bracket;
    }

    // Horizontal sum of SIMD accumulator
    let mut total: f64 = acc.reduce_add();

    // Scalar remainder
    let start = chunks * 4;
    for i in start..start + remainder {
        total += poisson_nll_bin_scalar(expected[i], observed[i], ln_factorials[i], obs_mask[i]);
    }

    total
}

/// Scalar reference implementation of Poisson NLL (same interface as SIMD version).
pub fn poisson_nll_scalar(
    expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    obs_mask: &[f64],
) -> f64 {
    let n = expected.len();
    assert_eq!(n, observed.len());
    assert_eq!(n, ln_factorials.len());
    assert_eq!(n, obs_mask.len());

    let mut total = 0.0;
    for i in 0..n {
        total += poisson_nll_bin_scalar(expected[i], observed[i], ln_factorials[i], obs_mask[i]);
    }
    total
}

/// Single-bin Poisson NLL (branchless scalar).
#[inline(always)]
fn poisson_nll_bin_scalar(exp: f64, obs: f64, ln_factorial: f64, mask: f64) -> f64 {
    exp + mask * (ln_factorial - obs * exp.ln())
}

/// SIMD scalar multiplication: `data[i] *= factor` for all i.
pub fn vec_scale(data: &mut [f64], factor: f64) {
    if !use_simd() {
        for v in data.iter_mut() {
            *v *= factor;
        }
        return;
    }

    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;
    let factor4 = f64x4::splat(factor);

    for i in 0..chunks {
        let offset = i * 4;
        let mut v = f64x4::from(&data[offset..offset + 4]);
        v *= factor4;
        let arr: [f64; 4] = v.into();
        data[offset..offset + 4].copy_from_slice(&arr);
    }

    let start = chunks * 4;
    for i in start..start + remainder {
        data[i] *= factor;
    }
}

/// SIMD pairwise multiplication: `dst[i] *= src[i]` for all i.
///
/// # Panics
/// Panics if `dst.len() != src.len()`.
pub fn vec_mul_pairwise(dst: &mut [f64], src: &[f64]) {
    assert_eq!(dst.len(), src.len());

    if !use_simd() {
        for (d, s) in dst.iter_mut().zip(src) {
            *d *= s;
        }
        return;
    }

    let n = dst.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let offset = i * 4;
        let mut d = f64x4::from(&dst[offset..offset + 4]);
        let s = f64x4::from(&src[offset..offset + 4]);
        d *= s;
        let arr: [f64; 4] = d.into();
        dst[offset..offset + 4].copy_from_slice(&arr);
    }

    let start = chunks * 4;
    for i in start..start + remainder {
        dst[i] *= src[i];
    }
}

/// Check if SIMD should be used on the current platform.
#[inline(always)]
fn use_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;
    use statrs::function::gamma::ln_gamma;

    fn ln_factorial(n: f64) -> f64 {
        ln_gamma(n + 1.0)
    }

    fn make_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut expected = Vec::with_capacity(n);
        let mut observed = Vec::with_capacity(n);
        let mut ln_facts = Vec::with_capacity(n);
        let mut mask = Vec::with_capacity(n);

        for i in 0..n {
            let exp = 50.0 + (i as f64 * 7.3) % 100.0;
            let obs = if i % 5 == 0 { 0.0 } else { (exp + (i as f64 * 3.1 - 40.0)).max(1.0) };
            expected.push(exp.max(1e-10));
            observed.push(obs);
            ln_facts.push(ln_factorial(obs));
            mask.push(if obs > 0.0 { 1.0 } else { 0.0 });
        }

        (expected, observed, ln_facts, mask)
    }

    #[test]
    fn test_poisson_nll_simd_matches_scalar() {
        for n in [1, 2, 3, 4, 5, 7, 8, 15, 16, 100, 1000] {
            let (exp, obs, lnf, mask) = make_test_data(n);
            let simd_result = poisson_nll_simd(&exp, &obs, &lnf, &mask);
            let scalar_result = poisson_nll_scalar(&exp, &obs, &lnf, &mask);
            assert_relative_eq!(simd_result, scalar_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_poisson_nll_single_bin_obs_zero() {
        let exp = vec![42.0];
        let obs = vec![0.0];
        let lnf = vec![0.0]; // lgamma(1) = 0
        let mask = vec![0.0];

        let result = poisson_nll_simd(&exp, &obs, &lnf, &mask);
        assert_relative_eq!(result, 42.0, epsilon = 1e-15);
    }

    #[test]
    fn test_poisson_nll_single_bin_obs_positive() {
        let exp = vec![50.0];
        let obs = vec![55.0];
        let lnf = vec![ln_factorial(55.0)];
        let mask = vec![1.0];

        let result = poisson_nll_simd(&exp, &obs, &lnf, &mask);
        let expected_nll = 50.0 - 55.0 * 50.0_f64.ln() + ln_factorial(55.0);
        assert_relative_eq!(result, expected_nll, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_nll_not_divisible_by_4() {
        for n in [1, 2, 3, 5, 6, 7, 9, 13, 17] {
            let (exp, obs, lnf, mask) = make_test_data(n);
            let simd_result = poisson_nll_simd(&exp, &obs, &lnf, &mask);
            let scalar_result = poisson_nll_scalar(&exp, &obs, &lnf, &mask);
            assert_relative_eq!(simd_result, scalar_result, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_poisson_nll_exp_near_floor() {
        let exp = vec![1e-10, 1e-10, 1e-10, 1e-10];
        let obs = vec![0.0, 1.0, 0.0, 2.0];
        let lnf: Vec<f64> = obs.iter().map(|&o| ln_factorial(o)).collect();
        let mask: Vec<f64> = obs.iter().map(|&o| if o > 0.0 { 1.0 } else { 0.0 }).collect();

        let simd_result = poisson_nll_simd(&exp, &obs, &lnf, &mask);
        let scalar_result = poisson_nll_scalar(&exp, &obs, &lnf, &mask);
        assert_relative_eq!(simd_result, scalar_result, epsilon = 1e-10);
        assert!(simd_result.is_finite());
    }

    #[test]
    fn test_vec_scale() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected: Vec<f64> = data.iter().map(|&x| x * 2.5).collect();
        vec_scale(&mut data, 2.5);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_vec_mul_pairwise() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let src = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let expected = vec![10.0, 40.0, 90.0, 160.0, 250.0];
        vec_mul_pairwise(&mut dst, &src);
        for (a, b) in dst.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_ln_f64x4_accuracy() {
        // Validate our lane-by-lane ln() is bit-exact with scalar f64::ln()
        let test_values: Vec<f64> = (0..10001)
            .map(|i| {
                let t = i as f64 / 10000.0;
                10.0_f64.powf(-10.0 + 16.0 * t)
            })
            .collect();

        for chunk in test_values.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let v = f64x4::from(chunk);
            let ln_result = ln_f64x4(v);
            let arr: [f64; 4] = ln_result.into();

            for (i, &val) in chunk.iter().enumerate() {
                let scalar_ln = val.ln();
                // Should be bit-exact since we use f64::ln() on each lane
                assert_eq!(arr[i].to_bits(), scalar_ln.to_bits());
            }
        }
    }

    proptest! {
        // Property test: SIMD and scalar implementations must agree for a wide range of inputs.
        //
        // We generate integer observed counts (Poisson domain) and positive expected values.
        #[test]
        fn prop_poisson_nll_simd_matches_scalar_random(
            n in 1usize..=128,
            expected in proptest::collection::vec(1e-10f64..1e3, 128),
            observed in proptest::collection::vec(0u16..2000, 128),
        ) {
            let exp = &expected[..n];
            let obs: Vec<f64> = observed[..n].iter().map(|&x| x as f64).collect();
            let lnf: Vec<f64> = obs.iter().map(|&o| ln_gamma(o + 1.0)).collect();
            let mask: Vec<f64> = obs.iter().map(|&o| if o > 0.0 { 1.0 } else { 0.0 }).collect();

            let simd_result = poisson_nll_simd(exp, &obs, &lnf, &mask);
            let scalar_result = poisson_nll_scalar(exp, &obs, &lnf, &mask);

            prop_assert!((simd_result - scalar_result).abs() < 1e-9);
            prop_assert!(simd_result.is_finite());
        }
    }
}
