//! Apple Accelerate backend for batch vectorized NLL computation.
//!
//! Uses vDSP and vForce from the Apple Accelerate framework for
//! hardware-optimized f64 vector operations on Apple Silicon.
//!
//! # Architecture
//!
//! The key insight: `vvlog()` computes `ln()` for an entire vector in one call,
//! using Apple's hand-tuned SIMD code for NEON on Apple Silicon. This replaces
//! our lane-by-lane `f64::ln()` approach in `simd.rs` with a single framework call.
//!
//! For batch NLL (many toy experiments), we flatten all bins across all toys into
//! one large vector and call `vvlog` once, avoiding per-toy overhead.
//!
//! # Feature gate
//!
//! This module is available only with `--features accelerate` on macOS/aarch64.

use std::cell::RefCell;
use std::os::raw::c_int;

// FFI bindings to Apple Accelerate framework (vForce + vDSP).
// These are always available on macOS without any external crate.
unsafe extern "C" {
    // vForce: vectorized transcendentals (libm replacement)
    /// `vvlog(result, input, &n)` — compute `ln(input[i])` for i in 0..n
    fn vvlog(result: *mut f64, input: *const f64, n: *const c_int);

    // vDSP: vectorized DSP routines
    /// `vDSP_vaddD(a, 1, b, 1, c, 1, n)` — c[i] = a[i] + b[i]
    fn vDSP_vaddD(
        a: *const f64, a_stride: isize,
        b: *const f64, b_stride: isize,
        c: *mut f64, c_stride: isize,
        n: usize,
    );

    /// `vDSP_vsubD(b, 1, a, 1, c, 1, n)` — c[i] = a[i] - b[i]
    /// Note: vDSP_vsubD subtracts first arg FROM second: c = B - A
    fn vDSP_vsubD(
        a: *const f64, a_stride: isize,
        b: *const f64, b_stride: isize,
        c: *mut f64, c_stride: isize,
        n: usize,
    );

    /// `vDSP_vmulD(a, 1, b, 1, c, 1, n)` — c[i] = a[i] * b[i]
    fn vDSP_vmulD(
        a: *const f64, a_stride: isize,
        b: *const f64, b_stride: isize,
        c: *mut f64, c_stride: isize,
        n: usize,
    );

    /// `vDSP_sveD(a, 1, &sum, n)` — sum = Σ a[i]
    fn vDSP_sveD(
        a: *const f64, a_stride: isize,
        result: *mut f64,
        n: usize,
    );

    /// `vDSP_vclipD(a, 1, &lo, &hi, c, 1, n)` — c[i] = clamp(a[i], lo, hi)
    fn vDSP_vclipD(
        a: *const f64, a_stride: isize,
        lo: *const f64,
        hi: *const f64,
        c: *mut f64, c_stride: isize,
        n: usize,
    );
}

#[derive(Default)]
struct PoissonScratch {
    ln_exp: Vec<f64>,
    obs_ln_exp: Vec<f64>,
    bracket: Vec<f64>,
    masked: Vec<f64>,
    terms: Vec<f64>,
}

impl PoissonScratch {
    fn ensure_len(&mut self, n: usize) {
        if self.ln_exp.len() < n {
            self.ln_exp.resize(n, 0.0);
        }
        if self.obs_ln_exp.len() < n {
            self.obs_ln_exp.resize(n, 0.0);
        }
        if self.bracket.len() < n {
            self.bracket.resize(n, 0.0);
        }
        if self.masked.len() < n {
            self.masked.resize(n, 0.0);
        }
        if self.terms.len() < n {
            self.terms.resize(n, 0.0);
        }
    }
}

thread_local! {
    // Thread-local scratch buffers to avoid repeated allocations in the hot NLL path
    // (e.g. inside L-BFGS iterations). Each Rayon worker thread gets its own buffers.
    static POISSON_SCRATCH: RefCell<PoissonScratch> = RefCell::new(PoissonScratch::default());
}

/// Compute Poisson NLL using Apple Accelerate vectorized operations.
///
/// Formula per bin: `nll_i = exp_i + mask_i * (ln_factorial_i - obs_i * ln(exp_i))`
///
/// This replaces the SIMD loop in `simd.rs` with bulk Accelerate calls:
/// 1. `vvlog(ln_exp, expected, n)` — vectorized natural log
/// 2. `vDSP_vmulD` — obs * ln_exp
/// 3. `vDSP_vsubD` — ln_factorial - obs_ln_exp
/// 4. `vDSP_vmulD` — mask * bracket
/// 5. `vDSP_vaddD` — expected + masked_bracket
/// 6. `vDSP_sveD` — horizontal sum
///
/// # Arguments
/// * `expected` - Expected counts per bin (must be clamped >= 1e-10 by caller)
/// * `observed` - Observed counts per bin
/// * `ln_factorials` - Pre-computed `lgamma(obs + 1)` per bin
/// * `obs_mask` - `1.0` if `obs > 0`, else `0.0`
///
/// # Panics
/// Panics if slice lengths are not equal.
pub fn poisson_nll_accelerate(
    expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    obs_mask: &[f64],
) -> f64 {
    let n = expected.len();
    assert_eq!(n, observed.len());
    assert_eq!(n, ln_factorials.len());
    assert_eq!(n, obs_mask.len());

    if n == 0 {
        return 0.0;
    }

    let ni: c_int = n
        .try_into()
        .expect("poisson_nll_accelerate: n does not fit in c_int");

    POISSON_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_len(n);

        unsafe {
            // 1. ln_exp[i] = ln(expected[i])
            vvlog(scratch.ln_exp.as_mut_ptr(), expected.as_ptr(), &ni);

            // 2. obs_ln_exp[i] = observed[i] * ln_exp[i]
            vDSP_vmulD(
                observed.as_ptr(), 1,
                scratch.ln_exp.as_ptr(), 1,
                scratch.obs_ln_exp.as_mut_ptr(), 1,
                n,
            );

            // 3. bracket[i] = ln_factorials[i] - obs_ln_exp[i]
            // vDSP_vsubD(A, ..., B, ..., C, ...) => C = B - A
            vDSP_vsubD(
                scratch.obs_ln_exp.as_ptr(), 1,
                ln_factorials.as_ptr(), 1,
                scratch.bracket.as_mut_ptr(), 1,
                n,
            );

            // 4. masked[i] = obs_mask[i] * bracket[i]
            vDSP_vmulD(
                obs_mask.as_ptr(), 1,
                scratch.bracket.as_ptr(), 1,
                scratch.masked.as_mut_ptr(), 1,
                n,
            );

            // 5. terms[i] = expected[i] + masked[i]
            vDSP_vaddD(
                expected.as_ptr(), 1,
                scratch.masked.as_ptr(), 1,
                scratch.terms.as_mut_ptr(), 1,
                n,
            );

            // 6. sum = Σ terms[i]
            let mut total: f64 = 0.0;
            vDSP_sveD(scratch.terms.as_ptr(), 1, &mut total, n);
            total
        }
    })
}

/// Compute Poisson NLL for a batch of toy experiments using Accelerate.
///
/// All toys share the same `ln_factorials` and `obs_mask`, but each has its own
/// `expected` vector. This function processes the entire batch with minimal overhead.
///
/// # Arguments
/// * `expected_batch` - Flat array: `[n_toys * n_bins]` of expected counts (row-major)
/// * `observed` - Observed counts per bin `[n_bins]` (shared across toys for standard NLL,
///                or `[n_toys * n_bins]` for per-toy observed data)
/// * `ln_factorials` - Pre-computed `lgamma(obs + 1)` per bin `[n_bins]` or `[n_toys * n_bins]`
/// * `obs_mask` - `1.0` if `obs > 0`, else `0.0`, `[n_bins]` or `[n_toys * n_bins]`
/// * `n_bins` - Number of bins per toy
/// * `n_toys` - Number of toys in the batch
///
/// # Returns
/// Vector of NLL values, one per toy.
///
/// # Note
/// For batch toys where each toy has its own observed data (Poisson-fluctuated),
/// pass flat concatenated arrays of size `n_toys * n_bins`.
pub fn batch_poisson_nll_accelerate(
    expected_flat: &[f64],
    observed_flat: &[f64],
    ln_factorials_flat: &[f64],
    obs_mask_flat: &[f64],
    n_bins: usize,
    n_toys: usize,
) -> Vec<f64> {
    assert_eq!(expected_flat.len(), n_toys * n_bins);

    let per_toy_obs = observed_flat.len() == n_toys * n_bins;
    if !per_toy_obs {
        assert_eq!(observed_flat.len(), n_bins);
        assert_eq!(ln_factorials_flat.len(), n_bins);
        assert_eq!(obs_mask_flat.len(), n_bins);
    }

    // For large batches, we can process all at once with a single vvlog call
    // on the entire flat expected array, then slice results per toy.
    let total_bins = n_toys * n_bins;
    let ni: c_int = total_bins
        .try_into()
        .expect("batch_poisson_nll_accelerate: total_bins does not fit in c_int");

    let mut ln_exp = vec![0.0f64; total_bins];
    let mut obs_ln_exp = vec![0.0f64; total_bins];
    let mut bracket = vec![0.0f64; total_bins];
    let mut masked = vec![0.0f64; total_bins];
    let mut terms = vec![0.0f64; total_bins];

    if per_toy_obs {
        // Each toy has its own observed data — fully flat computation
        assert_eq!(ln_factorials_flat.len(), total_bins);
        assert_eq!(obs_mask_flat.len(), total_bins);

        unsafe {
            // 1. Vectorized log over ALL bins of ALL toys at once
            vvlog(ln_exp.as_mut_ptr(), expected_flat.as_ptr(), &ni);

            // 2-5: Same chain, all vectorized over total_bins
            vDSP_vmulD(
                observed_flat.as_ptr(), 1,
                ln_exp.as_ptr(), 1,
                obs_ln_exp.as_mut_ptr(), 1,
                total_bins,
            );
            vDSP_vsubD(
                obs_ln_exp.as_ptr(), 1,
                ln_factorials_flat.as_ptr(), 1,
                bracket.as_mut_ptr(), 1,
                total_bins,
            );
            vDSP_vmulD(
                obs_mask_flat.as_ptr(), 1,
                bracket.as_ptr(), 1,
                masked.as_mut_ptr(), 1,
                total_bins,
            );
            vDSP_vaddD(
                expected_flat.as_ptr(), 1,
                masked.as_ptr(), 1,
                terms.as_mut_ptr(), 1,
                total_bins,
            );
        }
    } else {
        // Shared observed data — compute log over all expected, then tile observed arrays
        unsafe {
            vvlog(ln_exp.as_mut_ptr(), expected_flat.as_ptr(), &ni);
        }

        // For each toy, multiply by the SAME observed/ln_factorials/obs_mask
        for toy_idx in 0..n_toys {
            let offset = toy_idx * n_bins;
            let exp_slice = &expected_flat[offset..offset + n_bins];
            let ln_exp_slice = &ln_exp[offset..offset + n_bins];

            unsafe {
                // obs_ln_exp[offset..] = observed * ln_exp[offset..]
                vDSP_vmulD(
                    observed_flat.as_ptr(), 1,
                    ln_exp_slice.as_ptr(), 1,
                    obs_ln_exp[offset..].as_mut_ptr(), 1,
                    n_bins,
                );
                // bracket = ln_factorials - obs_ln_exp
                vDSP_vsubD(
                    obs_ln_exp[offset..].as_ptr(), 1,
                    ln_factorials_flat.as_ptr(), 1,
                    bracket[offset..].as_mut_ptr(), 1,
                    n_bins,
                );
                // masked = obs_mask * bracket
                vDSP_vmulD(
                    obs_mask_flat.as_ptr(), 1,
                    bracket[offset..].as_ptr(), 1,
                    masked[offset..].as_mut_ptr(), 1,
                    n_bins,
                );
                // terms = expected + masked
                vDSP_vaddD(
                    exp_slice.as_ptr(), 1,
                    masked[offset..].as_ptr(), 1,
                    terms[offset..].as_mut_ptr(), 1,
                    n_bins,
                );
            }
        }
    }

    // 6. Reduce each toy's terms to a single NLL
    let mut results = vec![0.0f64; n_toys];
    for toy_idx in 0..n_toys {
        let offset = toy_idx * n_bins;
        unsafe {
            vDSP_sveD(
                terms[offset..].as_ptr(), 1,
                &mut results[toy_idx],
                n_bins,
            );
        }
    }

    results
}

/// Compute Poisson NLL using Accelerate with Kahan compensated final summation.
///
/// Uses Accelerate for vectorized `vvlog` and arithmetic, but replaces the
/// `vDSP_sveD` horizontal sum with a Kahan compensated loop for higher precision.
/// Used by parity mode for deterministic validation.
pub fn poisson_nll_accelerate_kahan(
    expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    obs_mask: &[f64],
) -> f64 {
    let n = expected.len();
    assert_eq!(n, observed.len());
    assert_eq!(n, ln_factorials.len());
    assert_eq!(n, obs_mask.len());

    if n == 0 {
        return 0.0;
    }

    let ni: c_int = n
        .try_into()
        .expect("poisson_nll_accelerate_kahan: n does not fit in c_int");

    POISSON_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_len(n);

        unsafe {
            vvlog(scratch.ln_exp.as_mut_ptr(), expected.as_ptr(), &ni);

            vDSP_vmulD(
                observed.as_ptr(), 1,
                scratch.ln_exp.as_ptr(), 1,
                scratch.obs_ln_exp.as_mut_ptr(), 1,
                n,
            );

            vDSP_vsubD(
                scratch.obs_ln_exp.as_ptr(), 1,
                ln_factorials.as_ptr(), 1,
                scratch.bracket.as_mut_ptr(), 1,
                n,
            );

            vDSP_vmulD(
                obs_mask.as_ptr(), 1,
                scratch.bracket.as_ptr(), 1,
                scratch.masked.as_mut_ptr(), 1,
                n,
            );

            vDSP_vaddD(
                expected.as_ptr(), 1,
                scratch.masked.as_ptr(), 1,
                scratch.terms.as_mut_ptr(), 1,
                n,
            );
        }

        // Kahan compensated sum instead of vDSP_sveD
        let mut sum = 0.0_f64;
        let mut comp = 0.0_f64;
        for i in 0..n {
            let y = scratch.terms[i] - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    })
}

/// Clamp a vector of expected values to `[floor, +inf)` in-place using Accelerate.
///
/// This replaces the scalar loop `for val in &mut expected { if *val < 1e-10 { *val = 1e-10; } }`
/// with a single vDSP call.
pub fn clamp_expected_inplace(expected: &mut [f64], floor: f64) {
    let n = expected.len();
    if n == 0 {
        return;
    }
    let hi = f64::MAX;
    unsafe {
        vDSP_vclipD(
            expected.as_ptr(), 1,
            &floor,
            &hi,
            expected.as_mut_ptr(), 1,
            n,
        );
    }
}

/// Check if Apple Accelerate is available at runtime.
///
/// On macOS this is always true (Accelerate.framework is a system framework).
/// On other platforms, this returns false.
pub fn is_available() -> bool {
    cfg!(target_os = "macos")
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Reference scalar implementation for validation.
    fn poisson_nll_scalar_ref(
        expected: &[f64],
        observed: &[f64],
        ln_factorials: &[f64],
        obs_mask: &[f64],
    ) -> f64 {
        let mut total = 0.0;
        for i in 0..expected.len() {
            total += expected[i] + obs_mask[i] * (ln_factorials[i] - observed[i] * expected[i].ln());
        }
        total
    }

    #[test]
    fn test_accelerate_matches_scalar() {
        for n in [1, 2, 3, 4, 5, 7, 8, 15, 16, 100, 1000] {
            let (exp, obs, lnf, mask) = make_test_data(n);
            let accel_result = poisson_nll_accelerate(&exp, &obs, &lnf, &mask);
            let scalar_result = poisson_nll_scalar_ref(&exp, &obs, &lnf, &mask);
            let rel_err = (accel_result - scalar_result).abs() / scalar_result.abs().max(1e-15);
            assert!(
                rel_err < 1e-10,
                "n={}: accelerate={} scalar={} rel_err={}",
                n, accel_result, scalar_result, rel_err
            );
        }
    }

    #[test]
    fn test_accelerate_matches_simd() {
        use crate::simd::poisson_nll_simd;

        for n in [4, 16, 100, 1000] {
            let (exp, obs, lnf, mask) = make_test_data(n);
            let accel_result = poisson_nll_accelerate(&exp, &obs, &lnf, &mask);
            let simd_result = poisson_nll_simd(&exp, &obs, &lnf, &mask);
            let rel_err = (accel_result - simd_result).abs() / simd_result.abs().max(1e-15);
            assert!(
                rel_err < 1e-10,
                "n={}: accelerate={} simd={} rel_err={}",
                n, accel_result, simd_result, rel_err
            );
        }
    }

    #[test]
    fn test_accelerate_single_bin_obs_zero() {
        let exp = vec![42.0];
        let obs = vec![0.0];
        let lnf = vec![0.0];
        let mask = vec![0.0];

        let result = poisson_nll_accelerate(&exp, &obs, &lnf, &mask);
        assert!((result - 42.0).abs() < 1e-12, "got {}", result);
    }

    #[test]
    fn test_accelerate_single_bin_obs_positive() {
        let exp = vec![50.0];
        let obs = vec![55.0];
        let lnf = vec![ln_factorial(55.0)];
        let mask = vec![1.0];

        let result = poisson_nll_accelerate(&exp, &obs, &lnf, &mask);
        let expected_nll = 50.0 - 55.0 * 50.0_f64.ln() + ln_factorial(55.0);
        assert!((result - expected_nll).abs() < 1e-10, "got {} expected {}", result, expected_nll);
    }

    #[test]
    fn test_batch_poisson_nll_shared_obs() {
        let n_bins = 100;
        let n_toys = 10;

        let (exp_base, obs, lnf, mask) = make_test_data(n_bins);

        // Create batch: each toy has slightly different expected values
        let mut expected_flat = Vec::with_capacity(n_toys * n_bins);
        for toy_idx in 0..n_toys {
            let scale = 1.0 + toy_idx as f64 * 0.01;
            for &e in &exp_base {
                expected_flat.push(e * scale);
            }
        }

        let batch_results = batch_poisson_nll_accelerate(
            &expected_flat, &obs, &lnf, &mask, n_bins, n_toys,
        );

        assert_eq!(batch_results.len(), n_toys);

        // Verify each toy matches individual computation
        for toy_idx in 0..n_toys {
            let offset = toy_idx * n_bins;
            let single = poisson_nll_accelerate(
                &expected_flat[offset..offset + n_bins],
                &obs, &lnf, &mask,
            );
            let rel_err = (batch_results[toy_idx] - single).abs() / single.abs().max(1e-15);
            assert!(
                rel_err < 1e-12,
                "toy {}: batch={} single={} rel_err={}",
                toy_idx, batch_results[toy_idx], single, rel_err
            );
        }
    }

    #[test]
    fn test_batch_poisson_nll_per_toy_obs() {
        let n_bins = 50;
        let n_toys = 5;

        let (exp_base, obs_base, _, _) = make_test_data(n_bins);

        // Create per-toy data
        let mut expected_flat = Vec::with_capacity(n_toys * n_bins);
        let mut observed_flat = Vec::with_capacity(n_toys * n_bins);
        let mut lnf_flat = Vec::with_capacity(n_toys * n_bins);
        let mut mask_flat = Vec::with_capacity(n_toys * n_bins);

        for toy_idx in 0..n_toys {
            let scale = 1.0 + toy_idx as f64 * 0.02;
            for i in 0..n_bins {
                let exp = exp_base[i] * scale;
                let obs = (obs_base[i] + toy_idx as f64).max(0.0);
                expected_flat.push(exp);
                observed_flat.push(obs);
                lnf_flat.push(ln_factorial(obs));
                mask_flat.push(if obs > 0.0 { 1.0 } else { 0.0 });
            }
        }

        let batch_results = batch_poisson_nll_accelerate(
            &expected_flat, &observed_flat, &lnf_flat, &mask_flat, n_bins, n_toys,
        );

        assert_eq!(batch_results.len(), n_toys);

        for toy_idx in 0..n_toys {
            let offset = toy_idx * n_bins;
            let single = poisson_nll_accelerate(
                &expected_flat[offset..offset + n_bins],
                &observed_flat[offset..offset + n_bins],
                &lnf_flat[offset..offset + n_bins],
                &mask_flat[offset..offset + n_bins],
            );
            let rel_err = (batch_results[toy_idx] - single).abs() / single.abs().max(1e-15);
            assert!(
                rel_err < 1e-12,
                "toy {}: batch={} single={} rel_err={}",
                toy_idx, batch_results[toy_idx], single, rel_err
            );
        }
    }

    #[test]
    fn test_clamp_expected_inplace() {
        let mut data = vec![-1.0, 0.0, 1e-11, 1e-10, 1.0, 100.0];
        clamp_expected_inplace(&mut data, 1e-10);
        assert_eq!(data[0], 1e-10);
        assert_eq!(data[1], 1e-10);
        assert_eq!(data[2], 1e-10);
        assert_eq!(data[3], 1e-10);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[5], 100.0);
    }

    #[test]
    fn test_is_available() {
        // On macOS, Accelerate is always available
        assert!(is_available());
    }

    #[test]
    fn test_empty_input() {
        let result = poisson_nll_accelerate(&[], &[], &[], &[]);
        assert_eq!(result, 0.0);
    }
}
