//! Toy-based frequentist inference (Phase 3.2 baseline).
//!
//! This module implements a toy-based CLs `hypotest` workflow for HistFactory models,
//! conceptually mirroring `pyhf.infer.hypotest(..., calctype="toybased", test_stat="qtilde")`.
//!
//! Notes / current scope:
//! - Toys fluctuate **main** (binned Poisson counts) only.
//! - Auxiliary constraint observations are kept fixed as stored in the model.
//! - Randomness is deterministic via per-toy seeding (`seed + toy_idx`), independent of threading.

use crate::hypotest::NSIGMA_ORDER;
use crate::mle::MaximumLikelihoodEstimator;
use ns_core::traits::LogDensityModel;
use ns_core::{Error, Result};
use ns_translate::pyhf::HistFactoryModel;
use rayon::prelude::*;

const CLB_MIN: f64 = 1e-300;

fn normal_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
}

#[inline]
fn safe_cls(clsb: f64, clb: f64) -> f64 {
    if !(clsb.is_finite() && clb.is_finite()) {
        return 0.0;
    }
    if clb <= CLB_MIN {
        return if clsb <= CLB_MIN { 0.0 } else { 1.0 };
    }
    (clsb / clb).clamp(0.0, 1.0)
}

fn poi_index(model: &HistFactoryModel) -> Result<usize> {
    model.poi_index().ok_or_else(|| Error::Validation("No POI defined".to_string()))
}

fn tail_prob_counts(n_ge: usize, n_valid: usize) -> f64 {
    if n_valid == 0 {
        return 0.0;
    }
    // Add-one smoothing to avoid exact 0/1 tail-probabilities.
    (n_ge as f64 + 1.0) / (n_valid as f64 + 1.0)
}

fn tail_prob_sorted(sorted: &[f64], threshold: f64) -> f64 {
    // sorted ascending.
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = sorted.partition_point(|v| *v < threshold);
    let tail = n - idx;
    tail_prob_counts(tail, n)
}

fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return sorted[0];
    }
    if p >= 1.0 {
        return sorted[n - 1];
    }
    let idx = p * ((n - 1) as f64);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = idx - (lo as f64);
    sorted[lo] + w * (sorted[hi] - sorted[lo])
}

fn qtilde_for_dataset(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    tape: &mut ns_ad::tape::Tape,
) -> Result<(f64, f64, f64, bool, bool)> {
    // Returns: (qtilde, mu_hat, free_nll, free_converged, fixed_converged)
    let free =
        mle.fit_minimum_histfactory_from_with_bounds_with_tape(model, init_free, bounds, tape)?;
    let mu_hat = free.parameters[poi];
    let free_nll = free.fval;

    if mu_hat > mu_test {
        return Ok((0.0, mu_hat, free_nll, free.converged, true));
    }

    let mut init_fixed = free.parameters;
    init_fixed[poi] = mu_test;
    let fixed = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
        model,
        &init_fixed,
        bounds_fixed,
        tape,
    )?;

    let q = (2.0 * (fixed.fval - free_nll)).max(0.0);
    Ok((q, mu_hat, free_nll, free.converged, fixed.converged))
}

#[derive(Debug, Clone)]
struct QEnsemble {
    q: Vec<f64>,
    n_error: usize,
    n_nonconverged: usize,
}

#[derive(Debug, Clone, Copy)]
struct ToyCounts {
    n_ge: usize,
    n_valid: usize,
    n_error: usize,
    n_nonconverged: usize,
}

fn count_q_ge_ensemble(
    mle: &MaximumLikelihoodEstimator,
    base_model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    q_obs: f64,
    expected_main: &[f64],
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    n_toys: usize,
    seed: u64,
) -> ToyCounts {
    let tape_capacity = base_model.n_params() * 20;

    (0..n_toys)
        .into_par_iter()
        .with_min_len(16)
        .map_init(
            || ns_ad::tape::Tape::with_capacity(tape_capacity),
            |tape, toy_idx| {
                let toy_seed = seed.wrapping_add(toy_idx as u64);
                let toy_data = crate::toys::poisson_main_from_expected(expected_main, toy_seed);
                let toy_model = match base_model.with_observed_main(&toy_data) {
                    Ok(m) => m,
                    Err(_) => {
                        return ToyCounts { n_ge: 0, n_valid: 0, n_error: 1, n_nonconverged: 0 };
                    }
                };

                let (q, _mu_hat, _free_nll, free_conv, fixed_conv) = match qtilde_for_dataset(
                    mle,
                    &toy_model,
                    poi,
                    mu_test,
                    init_free,
                    bounds,
                    bounds_fixed,
                    tape,
                ) {
                    Ok(v) => v,
                    Err(_) => {
                        return ToyCounts { n_ge: 0, n_valid: 0, n_error: 1, n_nonconverged: 0 };
                    }
                };

                if !q.is_finite() {
                    return ToyCounts { n_ge: 0, n_valid: 0, n_error: 1, n_nonconverged: 0 };
                }

                ToyCounts {
                    n_ge: usize::from(q >= q_obs),
                    n_valid: 1,
                    n_error: 0,
                    n_nonconverged: usize::from(!(free_conv && fixed_conv)),
                }
            },
        )
        .reduce(
            || ToyCounts { n_ge: 0, n_valid: 0, n_error: 0, n_nonconverged: 0 },
            |a, b| ToyCounts {
                n_ge: a.n_ge + b.n_ge,
                n_valid: a.n_valid + b.n_valid,
                n_error: a.n_error + b.n_error,
                n_nonconverged: a.n_nonconverged + b.n_nonconverged,
            },
        )
}

fn generate_q_ensemble(
    mle: &MaximumLikelihoodEstimator,
    base_model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    expected_main: &[f64],
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    n_toys: usize,
    seed: u64,
) -> QEnsemble {
    let tape_capacity = base_model.n_params() * 20;

    let results: Vec<Result<(f64, bool)>> = (0..n_toys)
        .into_par_iter()
        .with_min_len(16)
        .map_init(
            || ns_ad::tape::Tape::with_capacity(tape_capacity),
            |tape, toy_idx| {
                let toy_seed = seed.wrapping_add(toy_idx as u64);
                let toy_data = crate::toys::poisson_main_from_expected(expected_main, toy_seed);
                let toy_model = base_model.with_observed_main(&toy_data)?;

                let (q, _mu_hat, _free_nll, free_conv, fixed_conv) = qtilde_for_dataset(
                    mle,
                    &toy_model,
                    poi,
                    mu_test,
                    init_free,
                    bounds,
                    bounds_fixed,
                    tape,
                )?;
                Ok((q, free_conv && fixed_conv))
            },
        )
        .collect();

    let mut q: Vec<f64> = Vec::with_capacity(n_toys);
    let mut n_error = 0usize;
    let mut n_nonconverged = 0usize;

    for r in results {
        match r {
            Ok((qq, converged)) => {
                if qq.is_finite() {
                    q.push(qq);
                } else {
                    n_error += 1;
                }
                if !converged {
                    n_nonconverged += 1;
                }
            }
            Err(_) => {
                n_error += 1;
            }
        }
    }

    QEnsemble { q, n_error, n_nonconverged }
}

/// Result of a toy-based CLs `hypotest` at a fixed tested POI value.
#[derive(Debug, Clone)]
pub struct ToyHypotestResult {
    /// Tested POI value.
    pub mu_test: f64,
    /// Observed CLs.
    pub cls: f64,
    /// Observed CLs+b.
    pub clsb: f64,
    /// Observed CLb.
    pub clb: f64,
    /// Observed `qtilde_mu`.
    pub q_obs: f64,
    /// Unconditional best-fit POI on observed data.
    pub mu_hat: f64,
    /// Number of requested b-only toys.
    pub n_toys_b: usize,
    /// Number of requested s+b toys.
    pub n_toys_sb: usize,
    /// Number of toy errors (b-only).
    pub n_error_b: usize,
    /// Number of toy errors (s+b).
    pub n_error_sb: usize,
    /// Number of non-converged toys (b-only).
    pub n_nonconverged_b: usize,
    /// Number of non-converged toys (s+b).
    pub n_nonconverged_sb: usize,
}

/// Observed and expected CLs (Brazil band) from toy-based hypotest.
#[derive(Debug, Clone)]
pub struct ToyHypotestExpectedSet {
    /// Observed CLs.
    pub observed: ToyHypotestResult,
    /// Expected CLs at `n_sigma = [2, 1, 0, -1, -2]` (pyhf ordering).
    pub expected: [f64; 5],
}

/// Toy-based CLs hypotest (qtilde) returning observed p-values.
pub fn hypotest_qtilde_toys(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
) -> Result<ToyHypotestResult> {
    hypotest_qtilde_toys_impl(mle, model, mu_test, n_toys, seed, false).map(|(r, _)| r)
}

/// Toy-based CLs hypotest (qtilde) returning observed and expected-set p-values.
pub fn hypotest_qtilde_toys_expected_set(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
) -> Result<ToyHypotestExpectedSet> {
    let (observed, expected_opt) =
        hypotest_qtilde_toys_impl(mle, model, mu_test, n_toys, seed, true)?;
    let expected = expected_opt.ok_or_else(|| {
        Error::Validation("internal error: expected_set requested but not produced".to_string())
    })?;
    Ok(ToyHypotestExpectedSet { observed, expected })
}

fn hypotest_qtilde_toys_impl(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
    expected_set: bool,
) -> Result<(ToyHypotestResult, Option<[f64; 5]>)> {
    if n_toys == 0 {
        return Err(Error::Validation("n_toys must be > 0".to_string()));
    }

    let poi = poi_index(model)?;

    let bounds = model.parameter_bounds();
    if poi >= bounds.len() {
        return Err(Error::Validation(format!(
            "POI index out of bounds: poi={} bounds_len={}",
            poi,
            bounds.len()
        )));
    }
    let mut bounds_fixed = bounds.clone();
    bounds_fixed[poi] = (mu_test, mu_test);
    let mut bounds_mu0 = bounds.clone();
    bounds_mu0[poi] = (0.0, 0.0);

    // Baseline: observed free fit + conditional fits for generation points.
    let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
    let init0 = model.parameter_init();
    let free_obs =
        mle.fit_minimum_histfactory_from_with_bounds_with_tape(model, &init0, &bounds, &mut tape)?;
    if !free_obs.converged {
        return Err(Error::Validation(format!(
            "Observed-data free fit did not converge: {}",
            free_obs.message
        )));
    }
    let free_nll = free_obs.fval;
    let mu_hat = free_obs.parameters[poi];

    // Conditional fit at mu_test: s+b generation point.
    let mut init_mu = free_obs.parameters.clone();
    init_mu[poi] = mu_test;
    let fixed_mu = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
        model,
        &init_mu,
        &bounds_fixed,
        &mut tape,
    )?;
    if !fixed_mu.converged {
        return Err(Error::Validation(format!(
            "Failed to fit generation point at mu={}: {}",
            mu_test, fixed_mu.message
        )));
    }

    // Conditional fit at mu=0: b-only generation point.
    let mut init_mu0 = free_obs.parameters.clone();
    init_mu0[poi] = 0.0;
    let fixed_0 = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
        model,
        &init_mu0,
        &bounds_mu0,
        &mut tape,
    )?;
    if !fixed_0.converged {
        return Err(Error::Validation(format!(
            "Failed to fit generation point at mu=0: {}",
            fixed_0.message
        )));
    }

    // Observed qtilde(mu_test) uses the same fixed(mu_test) fit; apply qtilde definition.
    let q_obs = if mu_hat > mu_test { 0.0 } else { (2.0 * (fixed_mu.fval - free_nll)).max(0.0) };

    let expected_sb = model.expected_data_pyhf_main(&fixed_mu.parameters)?;
    let expected_b = model.expected_data_pyhf_main(&fixed_0.parameters)?;

    // Use separate deterministic seeds for the two ensembles.
    let seed_b = seed;
    let seed_sb = seed.wrapping_add(1_000_000_000u64);

    if !expected_set {
        let ens_b = count_q_ge_ensemble(
            mle,
            model,
            poi,
            mu_test,
            q_obs,
            &expected_b,
            &fixed_0.parameters,
            &bounds,
            &bounds_fixed,
            n_toys,
            seed_b,
        );
        let ens_sb = count_q_ge_ensemble(
            mle,
            model,
            poi,
            mu_test,
            q_obs,
            &expected_sb,
            &fixed_mu.parameters,
            &bounds,
            &bounds_fixed,
            n_toys,
            seed_sb,
        );

        if ens_b.n_valid == 0 || ens_sb.n_valid == 0 {
            return Err(Error::Validation(format!(
                "All toys failed: b_only_valid={} sb_valid={}",
                ens_b.n_valid, ens_sb.n_valid
            )));
        }

        let clsb = tail_prob_counts(ens_sb.n_ge, ens_sb.n_valid);
        let clb = tail_prob_counts(ens_b.n_ge, ens_b.n_valid);
        let cls = safe_cls(clsb, clb);

        return Ok((
            ToyHypotestResult {
                mu_test,
                cls,
                clsb,
                clb,
                q_obs,
                mu_hat,
                n_toys_b: n_toys,
                n_toys_sb: n_toys,
                n_error_b: ens_b.n_error,
                n_error_sb: ens_sb.n_error,
                n_nonconverged_b: ens_b.n_nonconverged,
                n_nonconverged_sb: ens_sb.n_nonconverged,
            },
            None,
        ));
    }

    let ens_b = generate_q_ensemble(
        mle,
        model,
        poi,
        mu_test,
        &expected_b,
        &fixed_0.parameters,
        &bounds,
        &bounds_fixed,
        n_toys,
        seed_b,
    );
    let ens_sb = generate_q_ensemble(
        mle,
        model,
        poi,
        mu_test,
        &expected_sb,
        &fixed_mu.parameters,
        &bounds,
        &bounds_fixed,
        n_toys,
        seed_sb,
    );

    if ens_b.q.is_empty() || ens_sb.q.is_empty() {
        return Err(Error::Validation(format!(
            "Toy ensembles are empty after filtering errors: b_only={} sb={}",
            ens_b.q.len(),
            ens_sb.q.len()
        )));
    }

    let mut q_b_sorted = ens_b.q.clone();
    q_b_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut q_sb_sorted = ens_sb.q.clone();
    q_sb_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let clsb = tail_prob_sorted(&q_sb_sorted, q_obs);
    let clb = tail_prob_sorted(&q_b_sorted, q_obs);
    let cls = safe_cls(clsb, clb);

    let observed = ToyHypotestResult {
        mu_test,
        cls,
        clsb,
        clb,
        q_obs,
        mu_hat,
        n_toys_b: n_toys,
        n_toys_sb: n_toys,
        n_error_b: ens_b.n_error,
        n_error_sb: ens_sb.n_error,
        n_nonconverged_b: ens_b.n_nonconverged,
        n_nonconverged_sb: ens_sb.n_nonconverged,
    };

    if !expected_set {
        return Ok((observed, None));
    }

    let mut cls_vals: Vec<f64> = Vec::with_capacity(q_b_sorted.len());
    for &q in &q_b_sorted {
        let clsb_q = tail_prob_sorted(&q_sb_sorted, q);
        let clb_q = tail_prob_sorted(&q_b_sorted, q);
        cls_vals.push(safe_cls(clsb_q, clb_q));
    }
    cls_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut expected = [0.0; 5];
    for (i, t) in NSIGMA_ORDER.into_iter().enumerate() {
        let p = normal_cdf(-t);
        expected[i] = quantile_sorted(&cls_vals, p);
    }

    Ok((observed, Some(expected)))
}

// ── GPU-accelerated ensemble functions ──────────────────────────

#[cfg(feature = "metal")]
fn count_q_ge_ensemble_metal(
    model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    q_obs: f64,
    expected_main: &[f64],
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    n_toys: usize,
    seed: u64,
) -> Result<ToyCounts> {
    // GPU batch: free fits
    let free_fits = crate::metal_batch::fit_toys_from_data_metal(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds,
        None,
    )?;
    // GPU batch: fixed fits (POI pinned to mu_test)
    let fixed_fits = crate::metal_batch::fit_toys_from_data_metal(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds_fixed,
        None,
    )?;

    let mut n_ge = 0usize;
    let mut n_valid = 0usize;
    let mut n_error = 0usize;
    let mut n_nonconverged = 0usize;

    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
        match (fr, fxr) {
            (Ok(free), Ok(fixed)) => {
                let mu_hat = free.parameters[poi];
                let q =
                    if mu_hat > mu_test { 0.0 } else { (2.0 * (fixed.nll - free.nll)).max(0.0) };
                if q.is_finite() {
                    n_valid += 1;
                    if q >= q_obs {
                        n_ge += 1;
                    }
                } else {
                    n_error += 1;
                }
                if !(free.converged && fixed.converged) {
                    n_nonconverged += 1;
                }
            }
            _ => {
                n_error += 1;
            }
        }
    }

    Ok(ToyCounts { n_ge, n_valid, n_error, n_nonconverged })
}

#[cfg(feature = "metal")]
fn generate_q_ensemble_metal(
    model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    expected_main: &[f64],
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    n_toys: usize,
    seed: u64,
) -> Result<QEnsemble> {
    let free_fits = crate::metal_batch::fit_toys_from_data_metal(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds,
        None,
    )?;
    let fixed_fits = crate::metal_batch::fit_toys_from_data_metal(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds_fixed,
        None,
    )?;

    let mut q = Vec::with_capacity(n_toys);
    let mut n_error = 0usize;
    let mut n_nonconverged = 0usize;

    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
        match (fr, fxr) {
            (Ok(free), Ok(fixed)) => {
                let mu_hat = free.parameters[poi];
                let qq =
                    if mu_hat > mu_test { 0.0 } else { (2.0 * (fixed.nll - free.nll)).max(0.0) };
                if qq.is_finite() {
                    q.push(qq);
                } else {
                    n_error += 1;
                }
                if !(free.converged && fixed.converged) {
                    n_nonconverged += 1;
                }
            }
            _ => {
                n_error += 1;
            }
        }
    }

    Ok(QEnsemble { q, n_error, n_nonconverged })
}

// ── CUDA GPU-accelerated ensemble functions ─────────────────────

#[cfg(feature = "cuda")]
fn count_q_ge_ensemble_cuda(
    model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    q_obs: f64,
    expected_main: &[f64],
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    n_toys: usize,
    seed: u64,
) -> Result<ToyCounts> {
    // GPU batch: free fits
    let free_fits = crate::gpu_batch::fit_toys_from_data_gpu(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds,
        None,
    )?;
    // GPU batch: fixed fits (POI pinned to mu_test)
    let fixed_fits = crate::gpu_batch::fit_toys_from_data_gpu(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds_fixed,
        None,
    )?;

    let mut n_ge = 0usize;
    let mut n_valid = 0usize;
    let mut n_error = 0usize;
    let mut n_nonconverged = 0usize;

    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
        match (fr, fxr) {
            (Ok(free), Ok(fixed)) => {
                let mu_hat = free.parameters[poi];
                let q =
                    if mu_hat > mu_test { 0.0 } else { (2.0 * (fixed.nll - free.nll)).max(0.0) };
                if q.is_finite() {
                    n_valid += 1;
                    if q >= q_obs {
                        n_ge += 1;
                    }
                } else {
                    n_error += 1;
                }
                if !(free.converged && fixed.converged) {
                    n_nonconverged += 1;
                }
            }
            _ => {
                n_error += 1;
            }
        }
    }

    Ok(ToyCounts { n_ge, n_valid, n_error, n_nonconverged })
}

#[cfg(feature = "cuda")]
fn generate_q_ensemble_cuda(
    model: &HistFactoryModel,
    poi: usize,
    mu_test: f64,
    expected_main: &[f64],
    init_free: &[f64],
    bounds: &[(f64, f64)],
    bounds_fixed: &[(f64, f64)],
    n_toys: usize,
    seed: u64,
) -> Result<QEnsemble> {
    let free_fits = crate::gpu_batch::fit_toys_from_data_gpu(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds,
        None,
    )?;
    let fixed_fits = crate::gpu_batch::fit_toys_from_data_gpu(
        model,
        expected_main,
        n_toys,
        seed,
        init_free,
        bounds_fixed,
        None,
    )?;

    let mut q = Vec::with_capacity(n_toys);
    let mut n_error = 0usize;
    let mut n_nonconverged = 0usize;

    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
        match (fr, fxr) {
            (Ok(free), Ok(fixed)) => {
                let mu_hat = free.parameters[poi];
                let qq =
                    if mu_hat > mu_test { 0.0 } else { (2.0 * (fixed.nll - free.nll)).max(0.0) };
                if qq.is_finite() {
                    q.push(qq);
                } else {
                    n_error += 1;
                }
                if !(free.converged && fixed.converged) {
                    n_nonconverged += 1;
                }
            }
            _ => {
                n_error += 1;
            }
        }
    }

    Ok(QEnsemble { q, n_error, n_nonconverged })
}

// ── GPU dispatch wrapper for hypotest_qtilde_toys_impl ─────────

#[cfg(any(feature = "metal", feature = "cuda"))]
fn hypotest_qtilde_toys_gpu_impl(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
    expected_set: bool,
    gpu_device: &str,
) -> Result<(ToyHypotestResult, Option<[f64; 5]>)> {
    // Validate device early, before any expensive computation.
    match gpu_device {
        #[cfg(feature = "cuda")]
        "cuda" => {}
        #[cfg(feature = "metal")]
        "metal" => {}
        _ => {
            return Err(Error::Validation(format!(
                "Unsupported GPU device for hypotest-toys: '{}'",
                gpu_device
            )));
        }
    }

    if n_toys == 0 {
        return Err(Error::Validation("n_toys must be > 0".to_string()));
    }

    let poi = poi_index(model)?;

    let bounds = model.parameter_bounds();
    if poi >= bounds.len() {
        return Err(Error::Validation(format!(
            "POI index out of bounds: poi={} bounds_len={}",
            poi,
            bounds.len()
        )));
    }
    let mut bounds_fixed = bounds.clone();
    bounds_fixed[poi] = (mu_test, mu_test);
    let mut bounds_mu0 = bounds.clone();
    bounds_mu0[poi] = (0.0, 0.0);

    // Phase A: baseline CPU fits (3 fits, not bottleneck)
    let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
    let init0 = model.parameter_init();
    let free_obs =
        mle.fit_minimum_histfactory_from_with_bounds_with_tape(model, &init0, &bounds, &mut tape)?;
    if !free_obs.converged {
        return Err(Error::Validation(format!(
            "Observed-data free fit did not converge: {}",
            free_obs.message
        )));
    }
    let free_nll = free_obs.fval;
    let mu_hat = free_obs.parameters[poi];

    let mut init_mu = free_obs.parameters.clone();
    init_mu[poi] = mu_test;
    let fixed_mu = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
        model,
        &init_mu,
        &bounds_fixed,
        &mut tape,
    )?;
    if !fixed_mu.converged {
        return Err(Error::Validation(format!(
            "Failed to fit generation point at mu={}: {}",
            mu_test, fixed_mu.message
        )));
    }

    let mut init_mu0 = free_obs.parameters.clone();
    init_mu0[poi] = 0.0;
    let fixed_0 = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
        model,
        &init_mu0,
        &bounds_mu0,
        &mut tape,
    )?;
    if !fixed_0.converged {
        return Err(Error::Validation(format!(
            "Failed to fit generation point at mu=0: {}",
            fixed_0.message
        )));
    }

    let q_obs = if mu_hat > mu_test { 0.0 } else { (2.0 * (fixed_mu.fval - free_nll)).max(0.0) };

    let expected_sb = model.expected_data_pyhf_main(&fixed_mu.parameters)?;
    let expected_b = model.expected_data_pyhf_main(&fixed_0.parameters)?;

    let seed_b = seed;
    let seed_sb = seed.wrapping_add(1_000_000_000u64);

    // Phase B: GPU-accelerated toy ensembles (device already validated at entry)

    if !expected_set {
        // count-only path (no need to collect full q distributions)
        let (eb, esb) = match gpu_device {
            #[cfg(feature = "cuda")]
            "cuda" => {
                let eb = count_q_ge_ensemble_cuda(
                    model,
                    poi,
                    mu_test,
                    q_obs,
                    &expected_b,
                    &fixed_0.parameters,
                    &bounds,
                    &bounds_fixed,
                    n_toys,
                    seed_b,
                )?;
                let esb = count_q_ge_ensemble_cuda(
                    model,
                    poi,
                    mu_test,
                    q_obs,
                    &expected_sb,
                    &fixed_mu.parameters,
                    &bounds,
                    &bounds_fixed,
                    n_toys,
                    seed_sb,
                )?;
                (eb, esb)
            }
            #[cfg(feature = "metal")]
            "metal" => {
                let eb = count_q_ge_ensemble_metal(
                    model,
                    poi,
                    mu_test,
                    q_obs,
                    &expected_b,
                    &fixed_0.parameters,
                    &bounds,
                    &bounds_fixed,
                    n_toys,
                    seed_b,
                )?;
                let esb = count_q_ge_ensemble_metal(
                    model,
                    poi,
                    mu_test,
                    q_obs,
                    &expected_sb,
                    &fixed_mu.parameters,
                    &bounds,
                    &bounds_fixed,
                    n_toys,
                    seed_sb,
                )?;
                (eb, esb)
            }
            _ => unreachable!("device validated at entry"),
        };

        if eb.n_valid == 0 || esb.n_valid == 0 {
            return Err(Error::Validation(format!(
                "All toys failed: b_only_valid={} sb_valid={}",
                eb.n_valid, esb.n_valid
            )));
        }

        let clsb = tail_prob_counts(esb.n_ge, esb.n_valid);
        let clb = tail_prob_counts(eb.n_ge, eb.n_valid);
        let cls = safe_cls(clsb, clb);

        return Ok((
            ToyHypotestResult {
                mu_test,
                cls,
                clsb,
                clb,
                q_obs,
                mu_hat,
                n_toys_b: n_toys,
                n_toys_sb: n_toys,
                n_error_b: eb.n_error,
                n_error_sb: esb.n_error,
                n_nonconverged_b: eb.n_nonconverged,
                n_nonconverged_sb: esb.n_nonconverged,
            },
            None,
        ));
    }

    // Full ensemble path (expected_set = true)
    let (ens_b_result, ens_sb_result) = match gpu_device {
        #[cfg(feature = "cuda")]
        "cuda" => {
            let eb = generate_q_ensemble_cuda(
                model,
                poi,
                mu_test,
                &expected_b,
                &fixed_0.parameters,
                &bounds,
                &bounds_fixed,
                n_toys,
                seed_b,
            )?;
            let esb = generate_q_ensemble_cuda(
                model,
                poi,
                mu_test,
                &expected_sb,
                &fixed_mu.parameters,
                &bounds,
                &bounds_fixed,
                n_toys,
                seed_sb,
            )?;
            (eb, esb)
        }
        #[cfg(feature = "metal")]
        "metal" => {
            let eb = generate_q_ensemble_metal(
                model,
                poi,
                mu_test,
                &expected_b,
                &fixed_0.parameters,
                &bounds,
                &bounds_fixed,
                n_toys,
                seed_b,
            )?;
            let esb = generate_q_ensemble_metal(
                model,
                poi,
                mu_test,
                &expected_sb,
                &fixed_mu.parameters,
                &bounds,
                &bounds_fixed,
                n_toys,
                seed_sb,
            )?;
            (eb, esb)
        }
        _ => unreachable!("device validated at entry"),
    };

    if ens_b_result.q.is_empty() || ens_sb_result.q.is_empty() {
        return Err(Error::Validation(format!(
            "Toy ensembles are empty after filtering errors: b_only={} sb={}",
            ens_b_result.q.len(),
            ens_sb_result.q.len()
        )));
    }

    let mut q_b_sorted = ens_b_result.q.clone();
    q_b_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut q_sb_sorted = ens_sb_result.q.clone();
    q_sb_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let clsb = tail_prob_sorted(&q_sb_sorted, q_obs);
    let clb = tail_prob_sorted(&q_b_sorted, q_obs);
    let cls = safe_cls(clsb, clb);

    let observed = ToyHypotestResult {
        mu_test,
        cls,
        clsb,
        clb,
        q_obs,
        mu_hat,
        n_toys_b: n_toys,
        n_toys_sb: n_toys,
        n_error_b: ens_b_result.n_error,
        n_error_sb: ens_sb_result.n_error,
        n_nonconverged_b: ens_b_result.n_nonconverged,
        n_nonconverged_sb: ens_sb_result.n_nonconverged,
    };

    let mut cls_vals: Vec<f64> = Vec::with_capacity(q_b_sorted.len());
    for &q_val in &q_b_sorted {
        let clsb_q = tail_prob_sorted(&q_sb_sorted, q_val);
        let clb_q = tail_prob_sorted(&q_b_sorted, q_val);
        cls_vals.push(safe_cls(clsb_q, clb_q));
    }
    cls_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut expected = [0.0; 5];
    for (i, t) in NSIGMA_ORDER.into_iter().enumerate() {
        let p = normal_cdf(-t);
        expected[i] = quantile_sorted(&cls_vals, p);
    }

    Ok((observed, Some(expected)))
}

/// GPU-accelerated toy-based CLs hypotest (qtilde) — observed p-values.
#[cfg(any(feature = "metal", feature = "cuda"))]
pub fn hypotest_qtilde_toys_gpu(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
    device: &str,
) -> Result<ToyHypotestResult> {
    hypotest_qtilde_toys_gpu_impl(mle, model, mu_test, n_toys, seed, false, device).map(|(r, _)| r)
}

/// GPU-accelerated toy-based CLs hypotest (qtilde) — observed and expected-set p-values.
#[cfg(any(feature = "metal", feature = "cuda"))]
pub fn hypotest_qtilde_toys_expected_set_gpu(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
    device: &str,
) -> Result<ToyHypotestExpectedSet> {
    let (observed, expected_opt) =
        hypotest_qtilde_toys_gpu_impl(mle, model, mu_test, n_toys, seed, true, device)?;
    let expected = expected_opt.ok_or_else(|| {
        Error::Validation("internal error: expected_set requested but not produced".to_string())
    })?;
    Ok(ToyHypotestExpectedSet { observed, expected })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;

    fn load_simple_model() -> HistFactoryModel {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        HistFactoryModel::from_workspace(&ws).unwrap()
    }

    #[test]
    fn toy_hypotest_reproducible_small() {
        let model = load_simple_model();
        let mle = MaximumLikelihoodEstimator::new();

        let a = hypotest_qtilde_toys(&mle, &model, 1.0, 20, 42).unwrap();
        let b = hypotest_qtilde_toys(&mle, &model, 1.0, 20, 42).unwrap();

        assert_eq!(a.mu_test, b.mu_test);
        assert_eq!(a.n_toys_b, 20);
        assert_eq!(a.n_toys_sb, 20);
        assert_eq!(a.n_error_b, 0);
        assert_eq!(a.n_error_sb, 0);

        assert!(a.q_obs.is_finite());
        assert!(a.clsb.is_finite());
        assert!(a.clb.is_finite());
        assert!(a.cls.is_finite());
        assert!((0.0..=1.0).contains(&a.clsb));
        assert!((0.0..=1.0).contains(&a.clb));
        assert!((0.0..=1.0).contains(&a.cls));

        // Exact determinism for a fixed seed and n_toys.
        assert_eq!(a.q_obs.to_bits(), b.q_obs.to_bits());
        assert_eq!(a.mu_hat.to_bits(), b.mu_hat.to_bits());
        assert_eq!(a.clsb.to_bits(), b.clsb.to_bits());
        assert_eq!(a.clb.to_bits(), b.clb.to_bits());
        assert_eq!(a.cls.to_bits(), b.cls.to_bits());
    }
}
