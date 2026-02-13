//! CPU batch toy fitting for unbinned models with warm-start, retry, and Rayon parallelism.
//!
//! This module provides [`fit_unbinned_toys_batch_cpu`], a centralised entry point for
//! generating and fitting Poisson-fluctuated toys from any [`LogDensityModel`].
//! Both the CLI (`unbinned-fit-toys --gpu none`) and the Python `unbinned_fit_toys()`
//! binding call this function.
//!
//! Key optimisations over the naive sequential loop:
//! - **Warm-start**: each toy starts from the MLE θ̂ of the observed data (not spec defaults).
//! - **Retry with jitter**: non-converged fits are retried up to `max_retries` times with
//!   a random perturbation of ±`jitter_scale` × param_range.
//! - **Rayon parallelism**: `par_iter` across toys (uses the caller's thread-pool).
//! - **Skip Hessian**: `compute_hessian=false` by default — saves N+1 gradient evals per toy.

use ns_core::Result;
use ns_core::traits::LogDensityModel;
use ns_core::types::FitResult;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::Distribution;
use rayon::prelude::*;

use crate::mle::MaximumLikelihoodEstimator;
use crate::optimizer::ToyFitConfig;

/// Result of a batch toy fit run.
#[derive(Debug)]
pub struct UnbinnedToyBatchResult {
    /// Per-toy fit results (one per toy, in order).
    pub fits: Vec<Result<FitResult>>,
    /// The warm-start θ̂ used as init for every toy (None if nominal fit failed).
    pub theta_hat: Option<Vec<f64>>,
    /// Whether the nominal (observed-data) MLE converged.
    pub nominal_converged: bool,
    /// Wall-clock seconds for toy sampling (aggregate across threads).
    pub sample_secs: f64,
    /// Wall-clock seconds for toy fitting (aggregate across threads).
    pub fit_secs: f64,
}

/// Batch-fit Poisson-fluctuated toys from an unbinned model on CPU.
///
/// # Arguments
/// * `model` - The observed-data model (implements `LogDensityModel`).
/// * `gen_params` - Parameters at which toys are generated.
/// * `n_toys` - Number of toys.
/// * `seed` - Base seed; each toy uses `seed.wrapping_add(toy_idx)`.
/// * `theta_hat` - Optional warm-start init. If `None`, fits observed model for θ̂.
/// * `config` - [`ToyFitConfig`] (max_iter, retries, jitter, hessian flag).
/// * `sample_fn` - Closure: `|model, gen_params, seed| -> Result<M>`.
pub fn fit_unbinned_toys_batch_cpu<M, F>(
    model: &M,
    gen_params: &[f64],
    n_toys: usize,
    seed: u64,
    theta_hat: Option<&[f64]>,
    config: ToyFitConfig,
    sample_fn: F,
) -> UnbinnedToyBatchResult
where
    M: LogDensityModel + Clone,
    F: Fn(&M, &[f64], u64) -> Result<M> + Send + Sync,
{
    let mle = MaximumLikelihoodEstimator::with_config(config.optimizer.clone());

    // Resolve warm-start init: use provided θ̂, or fit observed data, or fallback to spec init.
    let (warm_init, nominal_converged) = match theta_hat {
        Some(th) => (th.to_vec(), true),
        None => match mle.fit_minimum(model) {
            Ok(r) => (r.parameters, r.converged),
            Err(_) => (model.parameter_init(), false),
        },
    };

    let bounds: Vec<(f64, f64)> = model.parameter_bounds();

    use std::sync::atomic::{AtomicU64, Ordering};
    let sample_ns = AtomicU64::new(0);
    let fit_ns = AtomicU64::new(0);

    let run_one = |toy_idx: usize| -> Result<FitResult> {
        let toy_seed = seed.wrapping_add(toy_idx as u64);

        // Sample toy.
        let t0 = std::time::Instant::now();
        let toy_model = sample_fn(model, gen_params, toy_seed)?;
        sample_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Fit with warm-start + retry.
        let t1 = std::time::Instant::now();
        let result =
            fit_one_toy_with_retry(&mle, &toy_model, &warm_init, &bounds, &config, toy_seed);
        fit_ns.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        result
    };

    // Heuristic: avoid nested Rayon parallelism. For small toy counts, run toys sequentially
    // and let the model's event-level kernels use Rayon. For large toy counts, parallelise toys.
    let threads = rayon::current_num_threads();
    let parallelise_toys = threads > 1 && n_toys >= threads;

    let fits: Vec<Result<FitResult>> = if parallelise_toys {
        (0..n_toys).into_par_iter().map(run_one).collect()
    } else {
        (0..n_toys).map(run_one).collect()
    };

    let sample_secs = sample_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let fit_secs = fit_ns.load(Ordering::Relaxed) as f64 / 1e9;

    UnbinnedToyBatchResult {
        fits,
        theta_hat: Some(warm_init),
        nominal_converged,
        sample_secs,
        fit_secs,
    }
}

/// Fit a single toy with retry-on-failure logic.
///
/// 1. First attempt from `warm_init` (MLE θ̂) — fast path.
/// 2. On non-convergence, retry up to `config.max_retries` times with jittered init.
/// 3. Returns the attempt with the lowest NLL (even if none converged).
fn fit_one_toy_with_retry<M: LogDensityModel>(
    mle: &MaximumLikelihoodEstimator,
    toy_model: &M,
    warm_init: &[f64],
    bounds: &[(f64, f64)],
    config: &ToyFitConfig,
    toy_seed: u64,
) -> Result<FitResult> {
    // First attempt: warm-start.
    let first = if config.compute_hessian {
        mle.fit_from(toy_model, warm_init)?
    } else {
        let opt = mle.fit_minimum_from(toy_model, warm_init)?;
        fit_result_from_opt(opt)
    };

    if first.converged {
        return Ok(first);
    }

    let mut best = first;

    // Retry with jitter + escalating strategy.
    let mut rng = rand::rngs::StdRng::seed_from_u64(toy_seed ^ 0xDEAD_BEEF);

    for retry in 0..config.max_retries {
        // Escalate: last retry uses smooth bounds (Minuit-like), which is more
        // robust near boundaries at the cost of extra evaluations.
        let use_smooth = retry == config.max_retries.saturating_sub(1) && config.max_retries > 1;

        let jittered = jitter_params(warm_init, bounds, config.jitter_scale, &mut rng);

        let attempt = if use_smooth {
            let smooth_config = crate::optimizer::OptimizerConfig {
                smooth_bounds: true,
                ..config.optimizer.clone()
            };
            let smooth_mle = MaximumLikelihoodEstimator::with_config(smooth_config);
            if config.compute_hessian {
                smooth_mle.fit_from(toy_model, &jittered)?
            } else {
                let opt = smooth_mle.fit_minimum_from(toy_model, &jittered)?;
                fit_result_from_opt(opt)
            }
        } else if config.compute_hessian {
            mle.fit_from(toy_model, &jittered)?
        } else {
            let opt = mle.fit_minimum_from(toy_model, &jittered)?;
            fit_result_from_opt(opt)
        };

        if attempt.converged {
            return Ok(attempt);
        }

        // Keep the best (lowest NLL) attempt.
        if attempt.nll < best.nll {
            best = attempt;
        }
    }

    Ok(best)
}

/// Convert an `OptimizationResult` (no Hessian) into a `FitResult` with zero uncertainties.
fn fit_result_from_opt(opt: crate::optimizer::OptimizationResult) -> FitResult {
    let n = opt.parameters.len();
    FitResult::new(
        opt.parameters,
        vec![0.0; n],
        opt.fval,
        opt.converged,
        opt.n_iter as usize,
        opt.n_fev,
        opt.n_gev,
    )
}

/// Jitter parameters by ±jitter_scale × (hi - lo), clamped to bounds.
fn jitter_params(
    init: &[f64],
    bounds: &[(f64, f64)],
    jitter_scale: f64,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    let uniform = Uniform::new(-1.0_f64, 1.0).unwrap();
    init.iter()
        .zip(bounds.iter())
        .map(|(&x, &(lo, hi))| {
            let range = hi - lo;
            if !range.is_finite() || range <= 0.0 {
                return x;
            }
            let delta = jitter_scale * range * uniform.sample(rng);
            (x + delta).clamp(lo, hi)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_fit_config_defaults() {
        let cfg = ToyFitConfig::default();
        assert_eq!(cfg.optimizer.max_iter, 5000);
        assert_eq!(cfg.max_retries, 3);
        assert!((cfg.jitter_scale - 0.10).abs() < 1e-12);
        assert!(!cfg.compute_hessian);
    }

    #[test]
    fn test_jitter_within_bounds() {
        let init = vec![0.5, 2.0, -1.0];
        let bounds = vec![(0.0, 1.0), (1.0, 3.0), (-2.0, 0.0)];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let j = jitter_params(&init, &bounds, 0.10, &mut rng);
            for (i, (&v, &(lo, hi))) in j.iter().zip(bounds.iter()).enumerate() {
                assert!(v >= lo && v <= hi, "param {} = {} out of bounds [{}, {}]", i, v, lo, hi);
            }
        }
    }

    #[test]
    fn test_jitter_infinite_bounds_unchanged() {
        let init = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let j = jitter_params(&init, &bounds, 0.10, &mut rng);
        assert_eq!(j[0], 1.0);
    }
}
