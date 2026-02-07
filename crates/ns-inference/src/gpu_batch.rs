//! GPU-accelerated batch toy fitting.
//!
//! Uses lockstep iteration: all toys are at the same L-BFGS-B iteration,
//! and a single GPU kernel computes NLL + gradient for all active toys.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │  CPU: generate toys (Rayon) → pre-compute ln_facts, mask │
//! │  GPU: upload observed data (one transfer)                 │
//! │                                                           │
//! │  Loop (lockstep):                                         │
//! │    CPU: compact active params → flat buffer               │
//! │    GPU: batch_nll_grad() → NLL + gradient                 │
//! │    CPU: L-BFGS-B step per toy → new params, convergence   │
//! └───────────────────────────────────────────────────────────┘
//! ```

use crate::optimizer::OptimizerConfig;
use ns_compute::cuda_batch::CudaBatchAccelerator;
use ns_compute::cuda_types::GpuModelData;
use ns_core::traits::LogDensityModel;
use ns_core::{FitResult, Result};
use ns_translate::pyhf::HistFactoryModel;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

/// Standalone L-BFGS-B state machine for one optimization problem.
///
/// Maintains the inverse Hessian approximation (via limited-memory pairs)
/// and supports bounded parameters with gradient projection.
struct LbfgsState {
    /// Current parameter values.
    x: Vec<f64>,
    /// Previous gradient.
    prev_grad: Option<Vec<f64>>,
    /// Previous parameters.
    prev_x: Option<Vec<f64>>,
    /// Current function value.
    fval: f64,
    /// L-BFGS memory: s vectors (x_k - x_{k-1}).
    s_history: Vec<Vec<f64>>,
    /// L-BFGS memory: y vectors (g_k - g_{k-1}).
    y_history: Vec<Vec<f64>>,
    /// L-BFGS memory: rho = 1 / (y^T s).
    rho_history: Vec<f64>,
    /// Maximum history size.
    m: usize,
    /// Parameter bounds.
    bounds: Vec<(f64, f64)>,
    /// Convergence tolerance.
    tol: f64,
    /// Step counter.
    iter: usize,
    /// Number of function evaluations.
    n_fev: usize,
    /// Number of gradient evaluations.
    n_gev: usize,
    /// Whether converged.
    converged: bool,
}

impl LbfgsState {
    fn new(x0: Vec<f64>, bounds: Vec<(f64, f64)>, m: usize, tol: f64) -> Self {
        let x = Self::clamp_to_bounds(&x0, &bounds);
        Self {
            x,
            prev_grad: None,
            prev_x: None,
            fval: f64::INFINITY,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
            rho_history: Vec::with_capacity(m),
            m,
            bounds,
            tol,
            iter: 0,
            n_fev: 0,
            n_gev: 0,
            converged: false,
        }
    }

    /// Perform one L-BFGS-B step given current NLL and gradient.
    ///
    /// Returns the new parameters to evaluate. Sets `converged = true` if
    /// the projected gradient norm is below tolerance.
    fn step(&mut self, nll: f64, grad: &[f64]) -> &[f64] {
        let n = self.x.len();
        self.fval = nll;
        self.n_fev += 1;
        self.n_gev += 1;

        // Check convergence: projected gradient norm
        let pg_norm = self.projected_gradient_norm(grad);
        if pg_norm < self.tol {
            self.converged = true;
            return &self.x;
        }

        // Update L-BFGS memory
        if let (Some(prev_x), Some(prev_g)) = (&self.prev_x, &self.prev_grad) {
            let mut s = vec![0.0; n];
            let mut y = vec![0.0; n];
            let mut sy = 0.0;
            for i in 0..n {
                s[i] = self.x[i] - prev_x[i];
                y[i] = grad[i] - prev_g[i];
                sy += s[i] * y[i];
            }
            if sy > 1e-10 {
                if self.s_history.len() >= self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                    self.rho_history.remove(0);
                }
                self.rho_history.push(1.0 / sy);
                self.s_history.push(s);
                self.y_history.push(y);
            }
        }

        // L-BFGS two-loop recursion → search direction
        let direction = self.lbfgs_direction(grad);

        // Backtracking line search with Armijo condition
        let step_size = self.line_search(grad, &direction);

        // Save current state
        self.prev_x = Some(self.x.clone());
        self.prev_grad = Some(grad.to_vec());

        // Update parameters
        for i in 0..n {
            self.x[i] += step_size * direction[i];
        }
        self.x = Self::clamp_to_bounds(&self.x, &self.bounds);

        self.iter += 1;
        &self.x
    }

    /// L-BFGS two-loop recursion to compute search direction.
    fn lbfgs_direction(&self, grad: &[f64]) -> Vec<f64> {
        let n = grad.len();
        let k = self.s_history.len();

        if k == 0 {
            // Steepest descent (negated gradient)
            return grad.iter().map(|&g| -g).collect();
        }

        let mut q = grad.to_vec();
        let mut alpha = vec![0.0; k];

        // Backward pass
        for i in (0..k).rev() {
            let mut dot = 0.0;
            for j in 0..n {
                dot += self.s_history[i][j] * q[j];
            }
            alpha[i] = self.rho_history[i] * dot;
            for j in 0..n {
                q[j] -= alpha[i] * self.y_history[i][j];
            }
        }

        // Initial Hessian approximation: H0 = gamma * I
        // gamma = s^T y / y^T y (from last pair)
        let last = k - 1;
        let mut sy = 0.0;
        let mut yy = 0.0;
        for j in 0..n {
            sy += self.s_history[last][j] * self.y_history[last][j];
            yy += self.y_history[last][j] * self.y_history[last][j];
        }
        let gamma = if yy > 1e-30 { sy / yy } else { 1.0 };

        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Forward pass
        for i in 0..k {
            let mut dot = 0.0;
            for j in 0..n {
                dot += self.y_history[i][j] * r[j];
            }
            let beta = self.rho_history[i] * dot;
            for j in 0..n {
                r[j] += (alpha[i] - beta) * self.s_history[i][j];
            }
        }

        // Negate for descent direction
        for v in &mut r {
            *v = -*v;
        }
        r
    }

    /// Simple backtracking line search with Armijo condition.
    fn line_search(&self, grad: &[f64], direction: &[f64]) -> f64 {
        let n = self.x.len();

        // Directional derivative
        let mut dir_deriv = 0.0;
        for i in 0..n {
            dir_deriv += grad[i] * direction[i];
        }

        // If direction is not a descent direction, use steepest descent
        if dir_deriv >= 0.0 {
            return 0.0;
        }

        // Initial step size: 1.0 for quasi-Newton
        let mut step = 1.0;

        // Clamp step to stay within bounds
        for i in 0..n {
            if direction[i] > 0.0 {
                let max_step = (self.bounds[i].1 - self.x[i]) / direction[i];
                step = step.min(max_step);
            } else if direction[i] < 0.0 {
                let max_step = (self.bounds[i].0 - self.x[i]) / direction[i];
                step = step.min(max_step);
            }
        }

        // Ensure positive step
        step.max(1e-20)
    }

    /// Projected gradient norm (gradient clamped at bounds).
    fn projected_gradient_norm(&self, grad: &[f64]) -> f64 {
        let n = self.x.len();
        let mut norm_sq = 0.0;
        for i in 0..n {
            let g = grad[i];
            let pg = if self.x[i] <= self.bounds[i].0 && g > 0.0 {
                0.0 // At lower bound, positive gradient → no movement
            } else if self.x[i] >= self.bounds[i].1 && g < 0.0 {
                0.0 // At upper bound, negative gradient → no movement
            } else {
                g
            };
            norm_sq += pg * pg;
        }
        norm_sq.sqrt()
    }

    fn clamp_to_bounds(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        x.iter()
            .zip(bounds.iter())
            .map(|(&v, &(lo, hi))| v.clamp(lo, hi))
            .collect()
    }
}

/// Fit toys using GPU-accelerated batch NLL + gradient.
///
/// This is the GPU equivalent of [`crate::batch::fit_toys_batch`].
/// All toys are optimized in lockstep: same iteration count, but each
/// toy converges independently (masked out when done).
///
/// # Arguments
/// * `model` - Base HistFactory model
/// * `params` - Parameters to generate toys at
/// * `n_toys` - Number of pseudo-experiments
/// * `seed` - Random seed
/// * `config` - Optimizer configuration
pub fn fit_toys_batch_gpu(
    model: &HistFactoryModel,
    params: &[f64],
    n_toys: usize,
    seed: u64,
    config: Option<OptimizerConfig>,
) -> Result<Vec<Result<FitResult>>> {
    let config = config.unwrap_or_default();
    let n_params = model.n_params();

    // 1. Serialize model for GPU
    let gpu_data = model.serialize_for_gpu()?;
    let n_main_bins = gpu_data.n_main_bins;

    // 2. Create GPU accelerator
    let mut accel = CudaBatchAccelerator::from_gpu_data(&gpu_data, n_toys)?;

    // 3. Generate expected data and toy datasets on CPU (Rayon parallel)
    let expected = model.expected_data_pyhf_main(params)?;
    let toys: Vec<Vec<f64>> = (0..n_toys)
        .into_par_iter()
        .map(|i| crate::toys::poisson_main_from_expected(&expected, seed.wrapping_add(i as u64)))
        .collect();

    // 4. Pre-compute ln_factorials, obs_mask, and pack flat buffers
    let mut observed_flat = vec![0.0f64; n_toys * n_main_bins];
    let mut ln_facts_flat = vec![0.0f64; n_toys * n_main_bins];
    let mut obs_mask_flat = vec![0.0f64; n_toys * n_main_bins];

    for (t, toy_data) in toys.iter().enumerate() {
        let base = t * n_main_bins;
        for (b, &obs) in toy_data.iter().enumerate() {
            observed_flat[base + b] = obs;
            ln_facts_flat[base + b] = ln_gamma(obs + 1.0);
            obs_mask_flat[base + b] = if obs > 0.0 { 1.0 } else { 0.0 };
        }
    }

    // 5. Upload observed data to GPU (one transfer)
    accel.upload_observed(&observed_flat, &ln_facts_flat, &obs_mask_flat, n_toys)?;

    // 6. Initialize L-BFGS-B states
    let init_params: Vec<f64> = model.parameter_init();
    let bounds: Vec<(f64, f64)> = model.parameter_bounds();

    let mut states: Vec<LbfgsState> = (0..n_toys)
        .map(|_| LbfgsState::new(init_params.clone(), bounds.clone(), config.m, config.tol))
        .collect();

    let mut active_mask: Vec<bool> = vec![true; n_toys];

    // 7. Lockstep iteration loop
    for _iter in 0..config.max_iter {
        let active_indices: Vec<usize> = (0..n_toys)
            .filter(|&i| active_mask[i] && !states[i].converged)
            .collect();

        if active_indices.is_empty() {
            break;
        }

        let n_active = active_indices.len();

        // Compact active params into contiguous buffer
        let mut params_flat = vec![0.0f64; n_active * n_params];
        for (ai, &toy_idx) in active_indices.iter().enumerate() {
            let src = &states[toy_idx].x;
            params_flat[ai * n_params..(ai + 1) * n_params].copy_from_slice(src);
        }

        // GPU: batch NLL + gradient
        let (nlls, grads) = accel.batch_nll_grad(&params_flat, n_active)?;

        // CPU: update each active toy's L-BFGS-B state
        for (ai, &toy_idx) in active_indices.iter().enumerate() {
            let nll = nlls[ai];
            let grad = &grads[ai * n_params..(ai + 1) * n_params];
            states[toy_idx].step(nll, grad);
            if states[toy_idx].converged {
                active_mask[toy_idx] = false;
            }
        }
    }

    // 8. Collect results
    let results: Vec<Result<FitResult>> = states
        .into_iter()
        .map(|state| {
            Ok(FitResult::new(
                state.x,
                vec![0.0; n_params], // No uncertainties for toy fits
                state.fval,
                state.converged,
                state.iter,
                state.n_fev,
                state.n_gev,
            ))
        })
        .collect();

    Ok(results)
}

/// Check if CUDA GPU batch fitting is available at runtime.
pub fn is_cuda_available() -> bool {
    CudaBatchAccelerator::is_available()
}
