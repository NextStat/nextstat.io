//! Metal GPU-accelerated batch toy fitting (Apple Silicon, f32).
//!
//! Uses lockstep iteration: all toys are at the same L-BFGS-B iteration,
//! and a single Metal kernel computes NLL + gradient for all active toys.
//!
//! All GPU compute is in f32. Conversion f64↔f32 happens at the API boundary
//! inside [`MetalBatchAccelerator`]. Tolerance is relaxed to `max(tol, 1e-4)`
//! to account for f32 precision.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │  CPU: generate toys (Rayon) → pre-compute ln_facts, mask │
//! │  GPU: upload observed data (one transfer, f64→f32)        │
//! │                                                           │
//! │  Loop (lockstep):                                         │
//! │    CPU: compact active params → flat buffer               │
//! │    GPU: batch_nll_grad() → NLL + gradient (f32→f64)       │
//! │    CPU: L-BFGS-B step per toy → new params, convergence   │
//! └───────────────────────────────────────────────────────────┘
//! ```

use crate::lbfgs::LbfgsState;
use crate::optimizer::OptimizerConfig;
use ns_compute::metal_batch::MetalBatchAccelerator;
use ns_compute::metal_types::MetalModelData;
use ns_core::traits::LogDensityModel;
use ns_core::{FitResult, Result};
use ns_translate::pyhf::HistFactoryModel;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;

/// Fit toys using Metal GPU-accelerated batch NLL + gradient.
///
/// This is the Metal (Apple Silicon) equivalent of [`crate::gpu_batch::fit_toys_batch_gpu`].
/// All toys are optimized in lockstep: same iteration count, but each
/// toy converges independently (masked out when done).
///
/// Tolerance is clamped to at least `1e-4` because the kernel uses f32 arithmetic.
///
/// # Arguments
/// * `model` - Base HistFactory model
/// * `params` - Parameters to generate toys at
/// * `n_toys` - Number of pseudo-experiments
/// * `seed` - Random seed
/// * `config` - Optimizer configuration
pub fn fit_toys_batch_metal(
    model: &HistFactoryModel,
    params: &[f64],
    n_toys: usize,
    seed: u64,
    config: Option<OptimizerConfig>,
) -> Result<Vec<Result<FitResult>>> {
    let mut config = config.unwrap_or_default();
    // Relax tolerance for f32 precision — below 1e-4 the f32 gradient noise dominates
    config.tol = config.tol.max(1e-4);
    let n_params = model.n_params();

    // 1. Serialize model for GPU (f64 → f32)
    let gpu_data = model.serialize_for_gpu()?;
    let n_main_bins = gpu_data.n_main_bins;
    let metal_data = MetalModelData::from_gpu_data(&gpu_data);

    // 2. Create Metal accelerator
    let mut accel = MetalBatchAccelerator::from_metal_data(&metal_data, n_toys)?;

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

    // 5. Upload observed data to GPU (one transfer, f64→f32 inside accelerator)
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

        // GPU: batch NLL + gradient (f64→f32→f64 round-trip inside accelerator)
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

/// Check if Metal GPU batch fitting is available at runtime.
pub fn is_metal_available() -> bool {
    MetalBatchAccelerator::is_available()
}
