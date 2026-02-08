//! Profile likelihood utilities (frequentist).
//!
//! Implements the profile likelihood test statistic for upper-limit workflows:
//! `q_mu` / `qtilde_mu` in pyhf terminology (Cowan et al., arXiv:1007.1727).

use crate::MaximumLikelihoodEstimator;
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel};
use ns_core::{Error, Result};
#[cfg(any(feature = "cuda", feature = "metal"))]
use ns_translate::pyhf::HistFactoryModel;

/// Single point in a profile likelihood scan.
#[derive(Debug, Clone)]
pub struct ProfilePoint {
    /// Tested POI value.
    pub mu: f64,
    /// Test statistic value (pyhf `qmu` / `qmu_tilde`).
    pub q_mu: f64,
    /// Conditional NLL at `mu`.
    pub nll_mu: f64,
    /// Conditional fit convergence.
    pub converged: bool,
    /// Conditional fit iterations (argmin iterations).
    pub n_iter: u64,
    /// Optional fit diagnostics (debug/parity workflows only).
    pub diag: Option<ProfilePointDiag>,
}

/// Optional diagnostics for a profile scan point.
#[derive(Debug, Clone)]
pub struct ProfilePointDiag {
    /// Best-fit parameters (NextStat parameter order).
    pub parameters: Vec<f64>,
    /// Optimizer termination message.
    pub message: String,
    /// Objective evaluations.
    pub n_fev: usize,
    /// Gradient evaluations.
    pub n_gev: usize,
    /// Initial objective value at the start point.
    pub initial_cost: f64,
    /// L2 norm of the final gradient vector (if available).
    pub grad_l2: Option<f64>,
}

/// Profile likelihood scan result.
#[derive(Debug, Clone)]
pub struct ProfileLikelihoodScan {
    /// POI index in NextStat parameter order.
    pub poi_index: usize,
    /// Unconditional best-fit POI value.
    pub mu_hat: f64,
    /// Unconditional NLL at the global minimum.
    pub nll_hat: f64,
    /// Per-point results.
    pub points: Vec<ProfilePoint>,
}

fn poi_index(model: &(impl PoiModel + ?Sized)) -> Result<usize> {
    model.poi_index().ok_or_else(|| Error::Validation("No POI defined".to_string()))
}

/// Compute pyhf-style `q_mu`/`qtilde_mu` for a single `mu_test`.
///
/// Note: NextStat's `HistFactoryModel::nll` is `-log L`, while pyhf uses `twice_nll = -2 log L`.
/// Therefore `q_mu = 2 * (nll(mu) - nll_hat)`, clipped at 0.
pub fn qmu_like(
    mle: &MaximumLikelihoodEstimator,
    model: &(impl LogDensityModel + PoiModel + FixedParamModel),
    mu_test: f64,
) -> Result<f64> {
    let poi = poi_index(model)?;

    let free = mle.fit_minimum(model)?;
    let mu_hat = free.parameters[poi];

    let fixed_model = model.with_fixed_param(poi, mu_test);
    let fixed = mle.fit_minimum(&fixed_model)?;

    let llr = 2.0 * (fixed.fval - free.fval);
    let mut q = llr.max(0.0);
    if mu_hat > mu_test {
        q = 0.0;
    }
    Ok(q)
}

/// Run a profile likelihood scan over the provided POI values.
pub fn scan(
    mle: &MaximumLikelihoodEstimator,
    model: &(impl LogDensityModel + PoiModel + FixedParamModel),
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    let poi = poi_index(model)?;

    let free = mle.fit_minimum(model)?;
    let mu_hat = free.parameters[poi];
    let nll_hat = free.fval;

    let mut points = Vec::with_capacity(mu_values.len());
    for &mu in mu_values {
        let fixed_model = model.with_fixed_param(poi, mu);
        let fixed = mle.fit_minimum(&fixed_model)?;

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        points.push(ProfilePoint {
            mu,
            q_mu: q,
            nll_mu: fixed.fval,
            converged: fixed.converged,
            n_iter: fixed.n_iter,
            diag: None,
        });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
}

/// Optimized profile scan for HistFactory models.
///
/// Uses warm-start between consecutive scan points and bounds-clamping
/// instead of model cloning. Reuses a single AD tape across all fits.
pub fn scan_histfactory(
    mle: &MaximumLikelihoodEstimator,
    model: &ns_translate::pyhf::HistFactoryModel,
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    // In parity mode we prefer robustness over speed: run a multi-start conditional fit at each mu.
    // This helps avoid small but systematic q(mu) mismatches vs ROOT when warm-start drifts into a
    // slightly suboptimal nuisance minimum.
    let multistart = ns_compute::eval_mode() == ns_compute::EvalMode::Parity;
    scan_histfactory_impl(mle, model, mu_values, multistart, false)
}

/// Like [`scan_histfactory`], but includes per-point fitted parameter vectors and optimizer diagnostics.
pub fn scan_histfactory_diag(
    mle: &MaximumLikelihoodEstimator,
    model: &ns_translate::pyhf::HistFactoryModel,
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    let multistart = ns_compute::eval_mode() == ns_compute::EvalMode::Parity;
    scan_histfactory_impl(mle, model, mu_values, multistart, true)
}

fn scan_histfactory_impl(
    mle: &MaximumLikelihoodEstimator,
    model: &ns_translate::pyhf::HistFactoryModel,
    mu_values: &[f64],
    multistart: bool,
    include_diag: bool,
) -> Result<ProfileLikelihoodScan> {
    let poi = model.poi_index().ok_or_else(|| Error::Validation("No POI defined".into()))?;

    // Free fit (unconditional MLE)
    let mut tape = ns_ad::tape::Tape::new();
    let free = mle.fit_minimum_histfactory_with_tape(model, &mut tape)?;
    let mu_hat = free.parameters[poi];
    let nll_hat = free.fval;

    let base_bounds = model.parameter_bounds();
    let mut warm_params = free.parameters.clone();

    // Parity robustness knob: deterministic jittered restarts (escape warm-start basins).
    //
    // Defaults are conservative; override via env vars for experiments.
    let n_restarts: usize = if multistart {
        std::env::var("NEXTSTAT_PROFILE_SCAN_RESTARTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
    } else {
        0
    };
    let jitter_scale: f64 = std::env::var("NEXTSTAT_PROFILE_SCAN_JITTER_SCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.05);

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    fn rand_u01(state: &mut u64) -> f64 {
        // 53-ish bits
        let x = lcg_next(state) >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }

    fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
        let mut y = x;
        if lo.is_finite() {
            y = y.max(lo);
        }
        if hi.is_finite() {
            y = y.min(hi);
        }
        y
    }

    let mut points = Vec::with_capacity(mu_values.len());
    for &mu in mu_values {
        // Fix POI via bounds clamping — no model clone
        let mut bounds = base_bounds.clone();
        bounds[poi] = (mu, mu);
        warm_params[poi] = mu;

        let fixed_warm =
            mle.fit_minimum_histfactory_from_with_bounds_with_tape(model, &warm_params, &bounds, &mut tape)?;
        let mut fixed = if multistart {
            // Secondary start: clamp the global MLE to the tested mu. This is deterministic and
            // often provides a better initial point than chaining warm-starts across mu.
            let mut start2 = free.parameters.clone();
            start2[poi] = mu;
            let fixed2 =
                mle.fit_minimum_histfactory_from_with_bounds_with_tape(model, &start2, &bounds, &mut tape)?;
            if fixed2.fval < fixed_warm.fval { fixed2 } else { fixed_warm }
        } else {
            fixed_warm
        };

        if multistart && n_restarts != 0 && jitter_scale.is_finite() && jitter_scale > 0.0 {
            let mut base = free.parameters.clone();
            base[poi] = mu;
            // Seed deterministically from mu and model dimension.
            let mu_bits = mu.to_bits();
            let seed0 = mu_bits ^ ((model.dim() as u64).wrapping_mul(0x9E3779B97F4A7C15));

            for r in 0..n_restarts {
                let mut start = base.clone();
                let mut seed = seed0 ^ ((r as u64).wrapping_mul(0xD1B54A32D192ED03));
                for i in 0..start.len() {
                    if i == poi {
                        continue;
                    }
                    let (lo, hi) = bounds[i];
                    let u = 2.0 * rand_u01(&mut seed) - 1.0; // [-1, 1]
                    let width = if lo.is_finite() && hi.is_finite() && hi > lo {
                        hi - lo
                    } else {
                        start[i].abs().max(1.0)
                    };
                    start[i] = clamp_f64(start[i] + jitter_scale * width * u, lo, hi);
                }
                start[poi] = mu;

                if let Ok(cand) =
                    mle.fit_minimum_histfactory_from_with_bounds_with_tape(model, &start, &bounds, &mut tape)
                {
                    if cand.fval < fixed.fval {
                        fixed = cand;
                    }
                }
            }
        }

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        // Carry forward for warm-start
        warm_params = fixed.parameters.clone();

        let diag = if include_diag {
            let grad_l2 = fixed.final_gradient.as_ref().map(|g| {
                let ss: f64 = g.iter().map(|x| x * x).sum();
                ss.sqrt()
            });
            Some(ProfilePointDiag {
                parameters: fixed.parameters.clone(),
                message: fixed.message.clone(),
                n_fev: fixed.n_fev,
                n_gev: fixed.n_gev,
                initial_cost: fixed.initial_cost,
                grad_l2,
            })
        } else {
            None
        };

        points.push(ProfilePoint { mu, q_mu: q, nll_mu: fixed.fval, converged: fixed.converged, n_iter: fixed.n_iter, diag });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::{HistFactoryModel, Workspace};

    fn load_workspace(json: &str) -> HistFactoryModel {
        let ws: Workspace = serde_json::from_str(json).unwrap();
        HistFactoryModel::from_workspace(&ws).unwrap()
    }

    #[test]
    fn test_scan_histfactory_matches_generic_scan() {
        let model = load_workspace(include_str!("../../../tests/fixtures/simple_workspace.json"));
        let mle = MaximumLikelihoodEstimator::new();
        let mu_values: Vec<f64> = (0..11).map(|i| i as f64 * 0.2).collect();

        let generic = scan(&mle, &model, &mu_values).unwrap();
        let optimized = scan_histfactory_impl(&mle, &model, &mu_values, false, false).unwrap();

        assert_eq!(generic.poi_index, optimized.poi_index);
        assert!(
            (generic.mu_hat - optimized.mu_hat).abs() < 1e-8,
            "mu_hat: generic={}, optimized={}",
            generic.mu_hat,
            optimized.mu_hat
        );
        assert!(
            (generic.nll_hat - optimized.nll_hat).abs() < 1e-8,
            "nll_hat: generic={}, optimized={}",
            generic.nll_hat,
            optimized.nll_hat
        );

        for (g, o) in generic.points.iter().zip(optimized.points.iter()) {
            assert!((g.mu - o.mu).abs() < 1e-15);
            let q_diff = (g.q_mu - o.q_mu).abs();
            let q_rdiff = q_diff / g.q_mu.abs().max(1e-12);
            assert!(
                q_rdiff < 1e-4,
                "q_mu mismatch at mu={}: generic={}, optimized={}, rdiff={}",
                g.mu,
                g.q_mu,
                o.q_mu,
                q_rdiff
            );
        }
    }

    #[test]
    fn test_scan_histfactory_multistart_is_numerically_consistent() {
        let model = load_workspace(include_str!("../../../tests/fixtures/simple_workspace.json"));
        let mle = MaximumLikelihoodEstimator::new();
        let mu_values: Vec<f64> = (0..11).map(|i| i as f64 * 0.2).collect();

        let a = scan_histfactory_impl(&mle, &model, &mu_values, false, false).unwrap();
        let b = scan_histfactory_impl(&mle, &model, &mu_values, true, false).unwrap();

        assert!((a.mu_hat - b.mu_hat).abs() < 1e-10, "mu_hat: a={}, b={}", a.mu_hat, b.mu_hat);
        assert!(
            (a.nll_hat - b.nll_hat).abs() < 1e-10,
            "nll_hat: a={}, b={}",
            a.nll_hat,
            b.nll_hat
        );
        for (pa, pb) in a.points.iter().zip(b.points.iter()) {
            assert!((pa.mu - pb.mu).abs() < 1e-15);
            assert!(
                (pa.q_mu - pb.q_mu).abs() < 1e-10,
                "q_mu mismatch at mu={}: a={}, b={}",
                pa.mu,
                pa.q_mu,
                pb.q_mu
            );
            assert!(
                (pa.nll_mu - pb.nll_mu).abs() < 1e-10,
                "nll_mu mismatch at mu={}: a={}, b={}",
                pa.mu,
                pa.nll_mu,
                pb.nll_mu
            );
        }
    }

    #[test]
    #[ignore = "benchmark; run with `cargo test -p ns-inference --release test_bench_scan -- --ignored --nocapture`"]
    fn test_bench_scan_simple() {
        let model = load_workspace(include_str!("../../../tests/fixtures/simple_workspace.json"));
        let mle = MaximumLikelihoodEstimator::new();
        let mu_values: Vec<f64> = (0..21).map(|i| i as f64 * 0.1).collect();

        let t0 = std::time::Instant::now();
        let generic = scan(&mle, &model, &mu_values).unwrap();
        let t_generic = t0.elapsed();

        let t0 = std::time::Instant::now();
        let optimized = scan_histfactory(&mle, &model, &mu_values).unwrap();
        let t_optimized = t0.elapsed();

        let n_iter_generic: u64 = generic.points.iter().map(|p| p.n_iter).sum();
        let n_iter_optimized: u64 = optimized.points.iter().map(|p| p.n_iter).sum();

        println!(
            "\n=== simple_workspace ({} params, {} points) ===",
            model.n_params(),
            mu_values.len()
        );
        println!(
            "  scan() generic:      {:.3}s  ({} total iters)",
            t_generic.as_secs_f64(),
            n_iter_generic
        );
        println!(
            "  scan_histfactory():  {:.3}s  ({} total iters)",
            t_optimized.as_secs_f64(),
            n_iter_optimized
        );
        println!(
            "  speedup:             {:.1}x",
            t_generic.as_secs_f64() / t_optimized.as_secs_f64()
        );
        println!("  iter reduction:      {:.1}x", n_iter_generic as f64 / n_iter_optimized as f64);
    }

    #[test]
    #[ignore = "benchmark; run with `cargo test -p ns-inference --release test_bench_scan_thu -- --ignored --nocapture`"]
    fn test_bench_scan_thu() {
        let model = load_workspace(include_str!("../../../tests/fixtures/workspace_tHu.json"));
        let mle = MaximumLikelihoodEstimator::new();
        let mu_values: Vec<f64> = (0..21).map(|i| i as f64 * 0.5).collect();

        let t0 = std::time::Instant::now();
        let generic = scan(&mle, &model, &mu_values).unwrap();
        let t_generic = t0.elapsed();

        let t0 = std::time::Instant::now();
        let optimized = scan_histfactory(&mle, &model, &mu_values).unwrap();
        let t_optimized = t0.elapsed();

        let n_iter_generic: u64 = generic.points.iter().map(|p| p.n_iter).sum();
        let n_iter_optimized: u64 = optimized.points.iter().map(|p| p.n_iter).sum();

        println!(
            "\n=== workspace_tHu ({} params, {} points) ===",
            model.n_params(),
            mu_values.len()
        );
        println!(
            "  scan() generic:      {:.3}s  ({} total iters)",
            t_generic.as_secs_f64(),
            n_iter_generic
        );
        println!(
            "  scan_histfactory():  {:.3}s  ({} total iters)",
            t_optimized.as_secs_f64(),
            n_iter_optimized
        );
        println!(
            "  speedup:             {:.1}x",
            t_generic.as_secs_f64() / t_optimized.as_secs_f64()
        );
        println!("  iter reduction:      {:.1}x", n_iter_generic as f64 / n_iter_optimized as f64);

        // Numerical parity
        for (g, o) in generic.points.iter().zip(optimized.points.iter()) {
            let q_diff = (g.q_mu - o.q_mu).abs();
            let q_rdiff = q_diff / g.q_mu.abs().max(1e-12);
            println!(
                "  mu={:.1}: q_generic={:.6}, q_optimized={:.6}, rdiff={:.2e}, iters {}->{}",
                g.mu, g.q_mu, o.q_mu, q_rdiff, g.n_iter, o.n_iter
            );
        }
    }

    #[test]
    #[ignore = "benchmark; run with `cargo test -p ns-inference --release test_bench_scan_tttt -- --ignored --nocapture`"]
    fn test_bench_scan_tttt() {
        let model =
            load_workspace(include_str!("../../../tests/fixtures/tttt-prod_workspace.json"));
        let mle = MaximumLikelihoodEstimator::new();
        let mu_values: Vec<f64> = (0..51).map(|i| i as f64 * 0.1).collect();

        let t0 = std::time::Instant::now();
        let generic = scan(&mle, &model, &mu_values).unwrap();
        let t_generic = t0.elapsed();

        let t0 = std::time::Instant::now();
        let optimized = scan_histfactory(&mle, &model, &mu_values).unwrap();
        let t_optimized = t0.elapsed();

        let n_iter_generic: u64 = generic.points.iter().map(|p| p.n_iter).sum();
        let n_iter_optimized: u64 = optimized.points.iter().map(|p| p.n_iter).sum();

        println!("\n=== tttt-prod ({} params, {} points) ===", model.n_params(), mu_values.len());
        println!(
            "  scan() generic:      {:.3}s  ({} total iters)",
            t_generic.as_secs_f64(),
            n_iter_generic
        );
        println!(
            "  scan_histfactory():  {:.3}s  ({} total iters)",
            t_optimized.as_secs_f64(),
            n_iter_optimized
        );
        println!(
            "  speedup:             {:.1}x",
            t_generic.as_secs_f64() / t_optimized.as_secs_f64()
        );
        println!("  iter reduction:      {:.1}x", n_iter_generic as f64 / n_iter_optimized as f64);
    }
}

/// GPU-accelerated profile likelihood scan.
///
/// One `GpuSession` is shared across all scan points, avoiding repeated
/// model serialization and GPU buffer allocation. Warm-start: each point
/// starts from the previous point's best-fit parameters.
#[cfg(feature = "cuda")]
pub fn scan_gpu(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    let poi = model.poi_index().ok_or_else(|| Error::Validation("No POI defined".to_string()))?;

    let session = crate::gpu_session::cuda_session(model)?;
    let config = mle.config().clone();

    // Free fit (unconditional MLE)
    let free = session.fit_minimum(model, &config)?;
    let mu_hat = free.parameters[poi];
    let nll_hat = free.fval;

    let base_bounds = model.parameter_bounds();
    let mut warm_params = free.parameters.clone();

    let mut points = Vec::with_capacity(mu_values.len());
    for &mu in mu_values {
        // Fix POI at mu via bounds clamping
        let mut bounds = base_bounds.clone();
        bounds[poi] = (mu, mu);
        warm_params[poi] = mu;

        let fixed = session.fit_minimum_from_with_bounds(model, &warm_params, &bounds, &config)?;

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        // Update warm-start for next point
        warm_params = fixed.parameters.clone();

        points.push(ProfilePoint {
            mu,
            q_mu: q,
            nll_mu: fixed.fval,
            converged: fixed.converged,
            n_iter: fixed.n_iter,
            diag: None,
        });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
}

/// Metal GPU-accelerated profile likelihood scan (Apple Silicon, f32).
///
/// Mirrors [`scan_gpu`] but uses the Metal single-model session. Convergence
/// tolerance is clamped to at least `1e-3` for f32 precision.
#[cfg(feature = "metal")]
pub fn scan_metal(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    mu_values: &[f64],
) -> Result<ProfileLikelihoodScan> {
    let poi = model.poi_index().ok_or_else(|| Error::Validation("No POI defined".to_string()))?;

    let session = crate::gpu_session::metal_session(model)?;
    let mut config = mle.config().clone();
    config.tol = config.tol.max(1e-3);

    // Free fit (unconditional MLE)
    let free = session.fit_minimum(model, &config)?;
    let mu_hat = free.parameters[poi];
    let nll_hat = free.fval;

    let base_bounds = model.parameter_bounds();
    let mut warm_params = free.parameters.clone();

    let mut points = Vec::with_capacity(mu_values.len());
    for &mu in mu_values {
        let mut bounds = base_bounds.clone();
        bounds[poi] = (mu, mu);
        warm_params[poi] = mu;

        let fixed = session.fit_minimum_from_with_bounds(model, &warm_params, &bounds, &config)?;

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        warm_params = fixed.parameters.clone();

        points.push(ProfilePoint {
            mu,
            q_mu: q,
            nll_mu: fixed.fval,
            converged: fixed.converged,
            n_iter: fixed.n_iter,
            diag: None,
        });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
}
