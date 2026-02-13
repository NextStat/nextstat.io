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

use crate::lbfgs::LbfgsState;
use crate::optimizer::OptimizerConfig;
use ns_compute::cuda_batch::CudaBatchAccelerator;
use ns_core::traits::LogDensityModel;
use ns_core::{FitResult, Result};
use ns_translate::pyhf::HistFactoryModel;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

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

    let effective_m = config.effective_m(n_params);
    let mut states: Vec<LbfgsState> = (0..n_toys)
        .map(|_| LbfgsState::new(init_params.clone(), bounds.clone(), effective_m, config.tol))
        .collect();

    let mut active_mask: Vec<bool> = vec![true; n_toys];

    // Pre-allocate compaction buffer (worst case: all toys active)
    let mut params_flat = vec![0.0f64; n_toys * n_params];

    // 7. Lockstep iteration loop
    for _iter in 0..config.max_iter {
        let active_indices: Vec<usize> =
            (0..n_toys).filter(|&i| active_mask[i] && !states[i].converged).collect();

        if active_indices.is_empty() {
            break;
        }

        let n_active = active_indices.len();

        // Compact active params into preallocated contiguous buffer
        for (ai, &toy_idx) in active_indices.iter().enumerate() {
            let src = &states[toy_idx].x;
            params_flat[ai * n_params..(ai + 1) * n_params].copy_from_slice(src);
        }

        // GPU: batch NLL + gradient
        let (nlls, grads) = accel.batch_nll_grad(&params_flat[..n_active * n_params], n_active)?;

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

/// Fit pre-generated toy datasets on CUDA GPU with custom expected data, init, and bounds.
///
/// This is the lower-level entry point used by GPU-accelerated hypotest-toys.
/// Toys are generated from `expected_main` using deterministic Poisson sampling.
///
/// # Arguments
/// * `model` - Base HistFactory model (for serialization and metadata)
/// * `expected_main` - Expected data for toy generation (e.g., from conditional fit)
/// * `n_toys` - Number of pseudo-experiments
/// * `seed` - Random seed (toy i uses seed + i)
/// * `init_params` - Custom initial parameters for optimization
/// * `bounds` - Custom bounds (e.g., with fixed POI)
/// * `config` - Optimizer configuration
pub fn fit_toys_from_data_gpu(
    model: &HistFactoryModel,
    expected_main: &[f64],
    n_toys: usize,
    seed: u64,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<Vec<Result<FitResult>>> {
    let config = config.unwrap_or_default();
    let n_params = model.n_params();

    let gpu_data = model.serialize_for_gpu()?;
    let n_main_bins = gpu_data.n_main_bins;
    let mut accel = CudaBatchAccelerator::from_gpu_data(&gpu_data, n_toys)?;

    // Generate toys from custom expected data
    let toys: Vec<Vec<f64>> = (0..n_toys)
        .into_par_iter()
        .map(|i| {
            crate::toys::poisson_main_from_expected(expected_main, seed.wrapping_add(i as u64))
        })
        .collect();

    // Pre-compute per-toy observed data buffers
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

    accel.upload_observed(&observed_flat, &ln_facts_flat, &obs_mask_flat, n_toys)?;

    // Use custom init and bounds
    let effective_m = config.effective_m(n_params);
    let mut states: Vec<LbfgsState> = (0..n_toys)
        .map(|_| LbfgsState::new(init_params.to_vec(), bounds.to_vec(), effective_m, config.tol))
        .collect();

    let mut active_mask: Vec<bool> = vec![true; n_toys];
    let mut params_flat = vec![0.0f64; n_toys * n_params];

    for _iter in 0..config.max_iter {
        let active_indices: Vec<usize> =
            (0..n_toys).filter(|&i| active_mask[i] && !states[i].converged).collect();

        if active_indices.is_empty() {
            break;
        }

        let n_active = active_indices.len();

        for (ai, &toy_idx) in active_indices.iter().enumerate() {
            let src = &states[toy_idx].x;
            params_flat[ai * n_params..(ai + 1) * n_params].copy_from_slice(src);
        }

        let (nlls, grads) = accel.batch_nll_grad(&params_flat[..n_active * n_params], n_active)?;

        for (ai, &toy_idx) in active_indices.iter().enumerate() {
            let nll = nlls[ai];
            let grad = &grads[ai * n_params..(ai + 1) * n_params];
            states[toy_idx].step(nll, grad);
            if states[toy_idx].converged {
                active_mask[toy_idx] = false;
            }
        }
    }

    let results: Vec<Result<FitResult>> = states
        .into_iter()
        .map(|state| {
            Ok(FitResult::new(
                state.x,
                vec![0.0; n_params],
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
