//! Profile likelihood utilities (frequentist).
//!
//! Implements the profile likelihood test statistic for upper-limit workflows:
//! `q_mu` / `qtilde_mu` in pyhf terminology (Cowan et al., arXiv:1007.1727).

use crate::MaximumLikelihoodEstimator;
use crate::optimizer::OptimizerConfig;
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel};
use ns_core::{Error, FitResult, Result};
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
#[must_use]
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

/// Test statistic variant selector (arXiv:1007.1727).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestStatistic {
    /// `q̃_μ` (Eq. 14): one-sided, bounded POI (`μ̂ ∈ [0, μ]`).
    /// Used for upper limits. Default in `pyhf` for `test_stat="qtilde"`.
    QMuTilde,
    /// `q_μ` (Eq. 12): one-sided (`q = 0` when `μ̂ > μ`).
    QMu,
    /// `t_μ` (Eq. 8): two-sided. For confidence intervals (not just upper limits).
    TMu,
    /// `t̃_μ` (Eq. 11): two-sided with bounded POI (`μ̂ ≥ 0`).
    TMuTilde,
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
    compute_test_statistic(mle, model, mu_test, TestStatistic::QMuTilde)
}

/// Compute a test statistic for a single `mu_test`.
///
/// Supports all four test statistics from arXiv:1007.1727:
/// - `QMuTilde` (Eq. 14): one-sided, bounded POI — for upper limits (default).
/// - `QMu` (Eq. 12): one-sided — for upper limits (unbounded μ̂).
/// - `TMu` (Eq. 8): two-sided — for confidence intervals.
/// - `TMuTilde` (Eq. 11): two-sided, bounded POI — for confidence intervals.
pub fn compute_test_statistic(
    mle: &MaximumLikelihoodEstimator,
    model: &(impl LogDensityModel + PoiModel + FixedParamModel),
    mu_test: f64,
    test_stat: TestStatistic,
) -> Result<f64> {
    let poi = poi_index(model)?;

    let free = mle.fit_minimum(model)?;
    let mu_hat = free.parameters[poi];

    // Conditional fit at mu_test
    let fixed_model = model.with_fixed_param(poi, mu_test);
    let fixed = mle.fit_minimum(&fixed_model)?;

    eval_test_statistic(test_stat, mu_hat, mu_test, fixed.fval, free.fval, || {
        // Lazy boundary fit: only needed for tilde variants when mu_hat < 0
        let fixed_zero_model = model.with_fixed_param(poi, 0.0);
        mle.fit_minimum(&fixed_zero_model).map(|r| r.fval)
    })
}

/// Pure computation of a test statistic value from pre-computed fit results.
///
/// `nll_mu`: conditional NLL at `mu_test`.
/// `nll_hat`: unconditional NLL at `mu_hat`.
/// `nll_zero_fn`: lazy evaluation of NLL at `mu=0` (only called for tilde variants when needed).
fn eval_test_statistic(
    test_stat: TestStatistic,
    mu_hat: f64,
    mu_test: f64,
    nll_mu: f64,
    nll_hat: f64,
    nll_zero_fn: impl FnOnce() -> Result<f64>,
) -> Result<f64> {
    match test_stat {
        // Eq. 14 (q̃_μ): one-sided, bounded POI
        TestStatistic::QMuTilde => {
            if mu_hat > mu_test {
                Ok(0.0)
            } else if mu_hat < 0.0 {
                let nll_zero = nll_zero_fn()?;
                Ok((2.0 * (nll_mu - nll_zero)).max(0.0))
            } else {
                Ok((2.0 * (nll_mu - nll_hat)).max(0.0))
            }
        }
        // Eq. 12 (q_μ): one-sided, unbounded
        TestStatistic::QMu => {
            if mu_hat > mu_test {
                Ok(0.0)
            } else {
                Ok((2.0 * (nll_mu - nll_hat)).max(0.0))
            }
        }
        // Eq. 8 (t_μ): two-sided
        TestStatistic::TMu => Ok((2.0 * (nll_mu - nll_hat)).max(0.0)),
        // Eq. 11 (t̃_μ): two-sided, bounded POI (μ̂ ≥ 0)
        TestStatistic::TMuTilde => {
            if mu_hat >= 0.0 {
                Ok((2.0 * (nll_mu - nll_hat)).max(0.0))
            } else {
                let nll_zero = nll_zero_fn()?;
                Ok((2.0 * (nll_mu - nll_zero)).max(0.0))
            }
        }
    }
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
    let mut init = model.parameter_init();
    if multistart && !mu_values.is_empty() {
        // Match ROOT-like profile scan workflows that seed the POI near the scan grid center.
        // This can influence which nuisance minimum is reached on large, non-convex exports.
        let mut mu_lo = f64::INFINITY;
        let mut mu_hi = f64::NEG_INFINITY;
        for &mu in mu_values {
            if mu.is_finite() {
                mu_lo = mu_lo.min(mu);
                mu_hi = mu_hi.max(mu);
            }
        }
        if mu_lo.is_finite() && mu_hi.is_finite() {
            init[poi] = 0.5 * (mu_lo + mu_hi);
        }
    }
    let free = mle.fit_minimum_histfactory_from_with_tape(model, &init, &mut tape)?;
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

        let fixed_warm = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
            model,
            &warm_params,
            &bounds,
            &mut tape,
        )?;
        let mut fixed = if multistart {
            // Secondary start: clamp the global MLE to the tested mu. This is deterministic and
            // often provides a better initial point than chaining warm-starts across mu.
            let mut start2 = free.parameters.clone();
            start2[poi] = mu;
            let fixed2 = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
                model, &start2, &bounds, &mut tape,
            )?;
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

                if let Ok(cand) = mle.fit_minimum_histfactory_from_with_bounds_with_tape(
                    model, &start, &bounds, &mut tape,
                ) && cand.fval < fixed.fval
                {
                    fixed = cand;
                }
            }
        }

        let llr = 2.0 * (fixed.fval - nll_hat);
        let mut q = llr.max(0.0);
        if mu_hat > mu {
            q = 0.0;
        }

        // Carry forward for warm-start — reset to global MLE if fit didn't converge.
        if fixed.converged {
            warm_params = fixed.parameters.clone();
        } else {
            log::warn!("Profile scan: fit at mu={mu} did not converge, resetting warm-start");
            warm_params = free.parameters.clone();
        }

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

        points.push(ProfilePoint {
            mu,
            q_mu: q,
            nll_mu: fixed.fval,
            converged: fixed.converged,
            n_iter: fixed.n_iter,
            diag,
        });
    }

    Ok(ProfileLikelihoodScan { poi_index: poi, mu_hat, nll_hat, points })
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

// ---------------------------------------------------------------------------
// Generic Profile Likelihood CI (any LogDensityModel)
// ---------------------------------------------------------------------------

/// Result of a profile likelihood confidence interval for one parameter.
#[derive(Debug, Clone)]
pub struct ProfileCiResult {
    pub param_idx: usize,
    pub mle: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_evals: usize,
}

/// Wrapper that fixes one parameter and optimizes the rest.
///
/// Reduces dimension by 1: the fixed parameter is removed from the parameter vector.
struct FixedParamWrapper<'a, M: LogDensityModel> {
    inner: &'a M,
    fix_idx: usize,
    fix_value: f64,
}

impl<M: LogDensityModel> FixedParamWrapper<'_, M> {
    /// Reconstruct the full parameter vector by inserting the fixed value.
    fn full_params(&self, reduced: &[f64]) -> Vec<f64> {
        let mut full = Vec::with_capacity(reduced.len() + 1);
        full.extend_from_slice(&reduced[..self.fix_idx]);
        full.push(self.fix_value);
        full.extend_from_slice(&reduced[self.fix_idx..]);
        full
    }
}

impl<M: LogDensityModel> LogDensityModel for FixedParamWrapper<'_, M> {
    type Prepared<'b>
        = ns_core::traits::PreparedModelRef<'b, Self>
    where
        Self: 'b;

    fn dim(&self) -> usize {
        self.inner.dim() - 1
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names = self.inner.parameter_names();
        names.remove(self.fix_idx);
        names
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = self.inner.parameter_bounds();
        bounds.remove(self.fix_idx);
        bounds
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut init = self.inner.parameter_init();
        init.remove(self.fix_idx);
        init
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let full = self.full_params(params);
        self.inner.nll(&full)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let full = self.full_params(params);
        let full_grad = self.inner.grad_nll(&full)?;
        // Remove fixed index from gradient.
        let mut grad = Vec::with_capacity(full_grad.len().saturating_sub(1));
        for (j, g) in full_grad.iter().enumerate() {
            if j == self.fix_idx {
                continue;
            }
            grad.push(*g);
        }
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        ns_core::traits::PreparedModelRef::new(self)
    }
}

/// Compute profile likelihood CI for a single parameter via bisection.
///
/// The CI is defined as `{θ : 2*(NLL(θ) - NLL_min) ≤ chi2_level}`.
/// Default `chi2_level = 3.841` corresponds to 95% CI for 1 DOF.
pub fn profile_ci<M: LogDensityModel>(
    model: &M,
    mle_params: &[f64],
    mle_nll: f64,
    param_idx: usize,
    bounds: (f64, f64),
    chi2_level: f64,
    tol: f64,
    config: &OptimizerConfig,
) -> Result<ProfileCiResult> {
    if param_idx >= model.dim() {
        return Err(Error::Validation(format!(
            "param_idx {} out of range (dim={})",
            param_idx,
            model.dim()
        )));
    }

    let mle_val = mle_params[param_idx];
    let mle_est = MaximumLikelihoodEstimator::with_config(config.clone());
    let mut n_evals = 0usize;

    // Reduced init from MLE params (remove fixed parameter).
    let mut reduced_init: Vec<f64> = mle_params.to_vec();
    reduced_init.remove(param_idx);

    // Profile NLL at a given fixed value, warm-started from `init`.
    // Returns (nll, optimized_params) so caller can carry forward.
    let profile_nll_warm =
        |value: f64, init: &[f64], n_evals: &mut usize| -> Result<(f64, Vec<f64>)> {
            let wrapper = FixedParamWrapper { inner: model, fix_idx: param_idx, fix_value: value };
            let opt = mle_est.fit_minimum_from(&wrapper, init)?;
            *n_evals += opt.n_fev;
            Ok((opt.fval, opt.parameters))
        };

    // Warm-start states for lower and upper bisection.
    let mut warm_lo = reduced_init.clone();
    let mut warm_hi = reduced_init;

    // --- Lower bound ---
    let (b_lo, b_hi) = bounds;
    // Check if boundary is already past the CI.
    // Do NOT carry forward boundary params — they can be far from the CI crossing region.
    let (nll_boundary_lo, _params_lo) = profile_nll_warm(b_lo, &warm_lo, &mut n_evals)?;
    let delta_boundary_lo = 2.0 * (nll_boundary_lo - mle_nll) - chi2_level;
    let ci_lower = if delta_boundary_lo <= 0.0 {
        // Entire lower range is within CI — boundary is the lower CI.
        b_lo
    } else {
        // Bisect between b_lo and mle_val.
        let mut lo_outer = b_lo;
        let mut lo_search = mle_val;
        let mut iter = 0;
        while iter < 100 {
            let mid = 0.5 * (lo_outer + lo_search);
            if (lo_search - lo_outer).abs() < tol {
                break;
            }
            let (nll, params) = profile_nll_warm(mid, &warm_lo, &mut n_evals)?;
            warm_lo = params;
            let d = 2.0 * (nll - mle_nll) - chi2_level;
            if d > 0.0 {
                lo_outer = mid;
            } else {
                lo_search = mid;
            }
            iter += 1;
        }
        0.5 * (lo_outer + lo_search)
    };

    // --- Upper bound ---
    // Note: do NOT carry forward boundary params into bisection warm-start.
    // When b_hi is very far from MLE (e.g. 100 vs MLE=1.05 on 277p models),
    // nuisance params optimized at boundary are in a totally different regime
    // and can trap the bisection optimizer in local minima.
    let (nll_boundary_hi, _params_hi) = profile_nll_warm(b_hi, &warm_hi, &mut n_evals)?;
    let delta_boundary_hi = 2.0 * (nll_boundary_hi - mle_nll) - chi2_level;
    let ci_upper = if delta_boundary_hi <= 0.0 {
        b_hi
    } else {
        let mut hi_outer = b_hi;
        let mut hi_search = mle_val;
        let mut iter = 0;
        while iter < 100 {
            let mid = 0.5 * (hi_outer + hi_search);
            if (hi_outer - hi_search).abs() < tol {
                break;
            }
            let (nll, params) = profile_nll_warm(mid, &warm_hi, &mut n_evals)?;
            warm_hi = params;
            let d = 2.0 * (nll - mle_nll) - chi2_level;
            if d > 0.0 {
                hi_outer = mid;
            } else {
                hi_search = mid;
            }
            iter += 1;
        }
        0.5 * (hi_outer + hi_search)
    };

    Ok(ProfileCiResult { param_idx, mle: mle_val, ci_lower, ci_upper, n_evals })
}

/// Compute profile likelihood CI for all parameters.
pub fn profile_ci_all<M: LogDensityModel>(
    model: &M,
    fit_result: &FitResult,
    config: &OptimizerConfig,
) -> Vec<ProfileCiResult> {
    let n = model.dim();
    let chi2_95 = 3.841; // χ²(1, 0.95)
    let tol = 1e-4;

    let model_bounds = model.parameter_bounds();

    (0..n)
        .filter_map(|i| {
            // Use model bounds, widened if needed for search.
            let (lo, hi) = model_bounds[i];
            let lo = if lo.is_finite() {
                lo
            } else {
                fit_result.parameters[i] - 10.0 * fit_result.uncertainties[i].max(1.0)
            };
            let hi = if hi.is_finite() {
                hi
            } else {
                fit_result.parameters[i] + 10.0 * fit_result.uncertainties[i].max(1.0)
            };
            profile_ci(
                model,
                &fit_result.parameters,
                fit_result.nll,
                i,
                (lo, hi),
                chi2_95,
                tol,
                config,
            )
            .ok()
        })
        .collect()
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
    fn test_test_statistic_variants() {
        let model = load_workspace(include_str!("../../../tests/fixtures/simple_workspace.json"));
        let mle = MaximumLikelihoodEstimator::new();
        let mu_test = 1.5;

        let q_tilde =
            compute_test_statistic(&mle, &model, mu_test, TestStatistic::QMuTilde).unwrap();
        let q_mu = compute_test_statistic(&mle, &model, mu_test, TestStatistic::QMu).unwrap();
        let t_mu = compute_test_statistic(&mle, &model, mu_test, TestStatistic::TMu).unwrap();
        let t_tilde =
            compute_test_statistic(&mle, &model, mu_test, TestStatistic::TMuTilde).unwrap();

        // qmu_like should equal QMuTilde
        let q_legacy = qmu_like(&mle, &model, mu_test).unwrap();
        assert!((q_tilde - q_legacy).abs() < 1e-12, "QMuTilde should match legacy qmu_like");

        // All test stats should be non-negative
        assert!(q_tilde >= 0.0);
        assert!(q_mu >= 0.0);
        assert!(t_mu >= 0.0);
        assert!(t_tilde >= 0.0);

        // t_mu >= q_mu (two-sided is always >= one-sided, since one-sided clips to 0 when mu_hat > mu)
        assert!(t_mu >= q_mu - 1e-12, "t_mu={} should be >= q_mu={}", t_mu, q_mu);

        // For mu_test well above mu_hat: one-sided and two-sided should agree
        // (because mu_hat < mu_test, so no clipping)
        let free = mle.fit_minimum(&model).unwrap();
        let mu_hat = free.parameters[model.poi_index().unwrap()];
        if mu_hat < mu_test {
            assert!(
                (q_tilde - t_tilde).abs() < 1e-10,
                "when mu_hat < mu_test, q_tilde should equal t_tilde (both use same denominator)"
            );
        }
    }

    #[test]
    fn test_tmu_nonzero_when_mu_hat_above_mu_test() {
        let model = load_workspace(include_str!("../../../tests/fixtures/simple_workspace.json"));
        let mle = MaximumLikelihoodEstimator::new();

        // mu_test = 0.0 → mu_hat should be > 0, so qmu clips to 0 but tmu should be > 0
        let mu_test = 0.0;
        let q_mu = compute_test_statistic(&mle, &model, mu_test, TestStatistic::QMu).unwrap();
        let t_mu = compute_test_statistic(&mle, &model, mu_test, TestStatistic::TMu).unwrap();

        let free = mle.fit_minimum(&model).unwrap();
        let mu_hat = free.parameters[model.poi_index().unwrap()];

        if mu_hat > mu_test {
            assert_eq!(q_mu, 0.0, "q_mu should be 0 when mu_hat > mu_test");
            assert!(t_mu > 0.0, "t_mu should be >0 when mu_hat > mu_test (two-sided)");
        }
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
        assert!((a.nll_hat - b.nll_hat).abs() < 1e-10, "nll_hat: a={}, b={}", a.nll_hat, b.nll_hat);
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

    // -----------------------------------------------------------------------
    // Profile CI tests (generic LogDensityModel)
    // -----------------------------------------------------------------------

    /// Simple quadratic model: NLL = 0.5 * sum((x_i - center_i)^2 / sigma_i^2)
    /// Known analytical CI: center_i ± sigma_i * sqrt(chi2_level).
    struct QuadraticModel {
        centers: Vec<f64>,
        sigmas: Vec<f64>,
    }

    impl LogDensityModel for QuadraticModel {
        type Prepared<'a>
            = ns_core::traits::PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            self.centers.len()
        }

        fn parameter_names(&self) -> Vec<String> {
            (0..self.centers.len()).map(|i| format!("x{}", i)).collect()
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            self.centers.iter().map(|c| (c - 20.0, c + 20.0)).collect()
        }

        fn parameter_init(&self) -> Vec<f64> {
            self.centers.iter().map(|c| c + 0.1).collect()
        }

        fn nll(&self, params: &[f64]) -> Result<f64> {
            let mut nll = 0.0;
            for i in 0..params.len() {
                let d = (params[i] - self.centers[i]) / self.sigmas[i];
                nll += 0.5 * d * d;
            }
            Ok(nll)
        }

        fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
            let mut grad = vec![0.0; params.len()];
            for i in 0..params.len() {
                let d = (params[i] - self.centers[i]) / self.sigmas[i];
                grad[i] = d / self.sigmas[i];
            }
            Ok(grad)
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            ns_core::traits::PreparedModelRef::new(self)
        }
    }

    #[test]
    fn test_profile_ci_quadratic_matches_wald() {
        // For a quadratic NLL, profile CI should match Wald CI exactly.
        let model = QuadraticModel { centers: vec![2.0, 5.0], sigmas: vec![0.5, 1.0] };
        let config = OptimizerConfig::default();
        let mle_est = MaximumLikelihoodEstimator::with_config(config.clone());
        let fr = mle_est.fit(&model).unwrap();

        let chi2_95 = 3.841;
        for i in 0..2 {
            let ci = profile_ci(
                &model,
                &fr.parameters,
                fr.nll,
                i,
                (model.centers[i] - 20.0, model.centers[i] + 20.0),
                chi2_95,
                1e-5,
                &config,
            )
            .unwrap();

            let expected_half = model.sigmas[i] * chi2_95.sqrt();
            let expected_lo = model.centers[i] - expected_half;
            let expected_hi = model.centers[i] + expected_half;

            assert!(
                (ci.ci_lower - expected_lo).abs() < 0.01,
                "param {}: ci_lower={:.4} expected={:.4}",
                i,
                ci.ci_lower,
                expected_lo
            );
            assert!(
                (ci.ci_upper - expected_hi).abs() < 0.01,
                "param {}: ci_upper={:.4} expected={:.4}",
                i,
                ci.ci_upper,
                expected_hi
            );
        }
    }

    #[test]
    fn test_profile_ci_covers_true_value() {
        let model = QuadraticModel { centers: vec![3.0], sigmas: vec![2.0] };
        let config = OptimizerConfig::default();
        let mle_est = MaximumLikelihoodEstimator::with_config(config.clone());
        let fr = mle_est.fit(&model).unwrap();

        let ci = profile_ci(&model, &fr.parameters, fr.nll, 0, (-20.0, 20.0), 3.841, 1e-5, &config)
            .unwrap();

        assert!(ci.ci_lower < 3.0, "lower={} should be < 3.0", ci.ci_lower);
        assert!(ci.ci_upper > 3.0, "upper={} should be > 3.0", ci.ci_upper);
    }

    #[test]
    fn test_profile_ci_all_quadratic() {
        let model = QuadraticModel { centers: vec![1.0, 2.0, 3.0], sigmas: vec![0.5, 1.0, 2.0] };
        let config = OptimizerConfig::default();
        let mle_est = MaximumLikelihoodEstimator::with_config(config.clone());
        let fr = mle_est.fit(&model).unwrap();

        let cis = profile_ci_all(&model, &fr, &config);
        assert_eq!(cis.len(), 3);

        for ci in &cis {
            let true_val = model.centers[ci.param_idx];
            assert!(
                ci.ci_lower < true_val && ci.ci_upper > true_val,
                "param {}: [{:.3}, {:.3}] should contain {:.3}",
                ci.param_idx,
                ci.ci_lower,
                ci.ci_upper,
                true_val
            );
        }
    }

    #[test]
    fn test_profile_ci_param_at_boundary() {
        // If search bounds clip the CI, lower/upper should be at the boundary.
        let model = QuadraticModel { centers: vec![0.0], sigmas: vec![1.0] };
        let config = OptimizerConfig::default();
        let mle_est = MaximumLikelihoodEstimator::with_config(config.clone());
        let fr = mle_est.fit(&model).unwrap();

        // Narrow bounds so the CI is clipped on the lower side.
        let ci = profile_ci(
            &model,
            &fr.parameters,
            fr.nll,
            0,
            (-0.5, 10.0), // lower bound clips the natural CI
            3.841,
            1e-5,
            &config,
        )
        .unwrap();

        assert!(
            (ci.ci_lower - (-0.5)).abs() < 0.01,
            "lower should be at boundary -0.5, got {}",
            ci.ci_lower
        );
    }

    // -----------------------------------------------------------------------
    // Profile CI benchmarks (warm vs cold, reproducible)
    // Results recorded in docs/benchmarks/phase2_5_benchmarks.md
    // -----------------------------------------------------------------------

    /// Cold-start profile CI: always restart from MLE projection (no warm-start carry-forward).
    fn profile_ci_cold<M: LogDensityModel>(
        model: &M,
        mle_params: &[f64],
        mle_nll: f64,
        param_idx: usize,
        bounds: (f64, f64),
        chi2_level: f64,
        tol: f64,
        config: &OptimizerConfig,
    ) -> Result<ProfileCiResult> {
        if param_idx >= model.dim() {
            return Err(Error::Validation(format!(
                "param_idx {} out of range (dim={})",
                param_idx,
                model.dim()
            )));
        }
        let mle_val = mle_params[param_idx];
        let mle_est = MaximumLikelihoodEstimator::with_config(config.clone());
        let mut n_evals = 0usize;

        let mut reduced_init: Vec<f64> = mle_params.to_vec();
        reduced_init.remove(param_idx);

        // Always cold-start from reduced_init (NO carry-forward).
        let profile_nll = |value: f64, n_evals: &mut usize| -> Result<f64> {
            let wrapper = FixedParamWrapper { inner: model, fix_idx: param_idx, fix_value: value };
            let opt = mle_est.fit_minimum_from(&wrapper, &reduced_init)?;
            *n_evals += opt.n_fev;
            Ok(opt.fval)
        };

        let (b_lo, b_hi) = bounds;
        let nll_lo = profile_nll(b_lo, &mut n_evals)?;
        let delta_lo = 2.0 * (nll_lo - mle_nll) - chi2_level;
        let ci_lower = if delta_lo <= 0.0 {
            b_lo
        } else {
            let mut lo_outer = b_lo;
            let mut lo_search = mle_val;
            for _ in 0..100 {
                let mid = 0.5 * (lo_outer + lo_search);
                if (lo_search - lo_outer).abs() < tol {
                    break;
                }
                let nll = profile_nll(mid, &mut n_evals)?;
                let d = 2.0 * (nll - mle_nll) - chi2_level;
                if d > 0.0 {
                    lo_outer = mid;
                } else {
                    lo_search = mid;
                }
            }
            0.5 * (lo_outer + lo_search)
        };

        let nll_hi = profile_nll(b_hi, &mut n_evals)?;
        let delta_hi = 2.0 * (nll_hi - mle_nll) - chi2_level;
        let ci_upper = if delta_hi <= 0.0 {
            b_hi
        } else {
            let mut hi_outer = b_hi;
            let mut hi_search = mle_val;
            for _ in 0..100 {
                let mid = 0.5 * (hi_outer + hi_search);
                if (hi_outer - hi_search).abs() < tol {
                    break;
                }
                let nll = profile_nll(mid, &mut n_evals)?;
                let d = 2.0 * (nll - mle_nll) - chi2_level;
                if d > 0.0 {
                    hi_outer = mid;
                } else {
                    hi_search = mid;
                }
            }
            0.5 * (hi_outer + hi_search)
        };

        Ok(ProfileCiResult { param_idx, mle: mle_val, ci_lower, ci_upper, n_evals })
    }

    #[test]
    #[ignore = "benchmark; run with `cargo test -p ns-inference --release bench_profile_ci_tchannel -- --ignored --nocapture`"]
    fn bench_profile_ci_tchannel() {
        let model = load_workspace(include_str!("../../../tests/fixtures/tchannel_workspace.json"));
        let config = OptimizerConfig::default();
        let mle = MaximumLikelihoodEstimator::with_config(config.clone());
        let fr = mle.fit(&model).unwrap();

        let poi = model.poi_index().expect("tchannel has POI");
        let model_bounds = model.parameter_bounds()[poi];
        let chi2_95 = 3.841;
        let tol = 1e-4;

        // Tighten bounds like profile_ci_all: MLE ± 10 * uncertainty.
        let unc = fr.uncertainties[poi].max(0.1);
        let b_lo = if model_bounds.0.is_finite() {
            model_bounds.0.max(fr.parameters[poi] - 10.0 * unc)
        } else {
            fr.parameters[poi] - 10.0 * unc
        };
        let b_hi = if model_bounds.1.is_finite() {
            model_bounds.1.min(fr.parameters[poi] + 10.0 * unc)
        } else {
            fr.parameters[poi] + 10.0 * unc
        };
        let bounds = (b_lo, b_hi);

        eprintln!("\n=== Profile CI Benchmark: tchannel ({} params) ===", model.n_params());
        eprintln!("  POI index: {} (\"{}\")", poi, model.parameter_names()[poi]);
        eprintln!("  MLE: {:.6}, NLL: {:.6}", fr.parameters[poi], fr.nll);
        eprintln!("  Bounds: ({:.2}, {:.2})", bounds.0, bounds.1);

        // Warm-start (current implementation).
        let t0 = std::time::Instant::now();
        let ci_warm =
            profile_ci(&model, &fr.parameters, fr.nll, poi, bounds, chi2_95, tol, &config).unwrap();
        let t_warm = t0.elapsed();

        // Cold-start (always restart from MLE projection).
        let t0 = std::time::Instant::now();
        let ci_cold =
            profile_ci_cold(&model, &fr.parameters, fr.nll, poi, bounds, chi2_95, tol, &config)
                .unwrap();
        let t_cold = t0.elapsed();

        eprintln!("\n  --- Warm-start ---");
        eprintln!("    CI: [{:.6}, {:.6}]", ci_warm.ci_lower, ci_warm.ci_upper);
        eprintln!("    n_evals: {}", ci_warm.n_evals);
        eprintln!("    time: {:.1}ms", t_warm.as_secs_f64() * 1000.0);

        eprintln!("  --- Cold-start ---");
        eprintln!("    CI: [{:.6}, {:.6}]", ci_cold.ci_lower, ci_cold.ci_upper);
        eprintln!("    n_evals: {}", ci_cold.n_evals);
        eprintln!("    time: {:.1}ms", t_cold.as_secs_f64() * 1000.0);

        let ci_lo_diff = (ci_warm.ci_lower - ci_cold.ci_lower).abs();
        let ci_hi_diff = (ci_warm.ci_upper - ci_cold.ci_upper).abs();
        let eval_reduction = 1.0 - (ci_warm.n_evals as f64 / ci_cold.n_evals as f64);
        let speedup = t_cold.as_secs_f64() / t_warm.as_secs_f64();

        eprintln!("\n  --- Comparison ---");
        eprintln!("    CI lower diff: {:.2e}", ci_lo_diff);
        eprintln!("    CI upper diff: {:.2e}", ci_hi_diff);
        eprintln!("    Eval reduction: {:.1}%", eval_reduction * 100.0);
        eprintln!("    Speedup: {:.2}x", speedup);

        // CI values should agree within tolerance.
        assert!(
            ci_lo_diff < 0.01,
            "CI lower mismatch: warm={} cold={}",
            ci_warm.ci_lower,
            ci_cold.ci_lower
        );
        assert!(
            ci_hi_diff < 0.01,
            "CI upper mismatch: warm={} cold={}",
            ci_warm.ci_upper,
            ci_cold.ci_upper
        );
    }
}
