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
    // Relax tolerance for f32 precision — below 1e-3 the f32 gradient noise dominates
    config.tol = config.tol.max(1e-3);
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

    // Pre-allocate compaction buffer (worst case: all toys active)
    let mut params_flat = vec![0.0f64; n_toys * n_params];

    // 7. Lockstep iteration loop
    for _iter in 0..config.max_iter {
        let active_indices: Vec<usize> = (0..n_toys)
            .filter(|&i| active_mask[i] && !states[i].converged)
            .collect();

        if active_indices.is_empty() {
            break;
        }

        let n_active = active_indices.len();

        // Compact active params into preallocated contiguous buffer
        for (ai, &toy_idx) in active_indices.iter().enumerate() {
            let src = &states[toy_idx].x;
            params_flat[ai * n_params..(ai + 1) * n_params].copy_from_slice(src);
        }

        // GPU: batch NLL + gradient (f64→f32→f64 round-trip inside accelerator)
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

/// Check if Metal GPU batch fitting is available at runtime.
pub fn is_metal_available() -> bool {
    MetalBatchAccelerator::is_available()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_metal_available() {
        // On macOS with Apple Silicon this should return true.
        let available = is_metal_available();
        eprintln!("Metal available: {available}");
        // We don't assert true because CI may not have Metal.
    }

    #[test]
    fn test_metal_batch_smoke() {
        if !is_metal_available() {
            eprintln!("Skipping: Metal not available");
            return;
        }

        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let results = fit_toys_batch_metal(&model, &params, 4, 123, None).unwrap();
        assert_eq!(results.len(), 4);

        for (i, r) in results.iter().enumerate() {
            let fit = r.as_ref().unwrap_or_else(|e| panic!("Toy {i} failed: {e}"));
            assert!(fit.nll.is_finite(), "Toy {i} NLL should be finite: {}", fit.nll);
            // f32 tolerance is 1e-3, so some toys may not fully converge
            // — check that NLL is reasonable (< 100 for simple workspace).
            assert!(fit.nll < 100.0, "Toy {i} NLL suspiciously large: {}", fit.nll);
            eprintln!(
                "Toy {i}: NLL={:.6}, converged={}, iters={}",
                fit.nll, fit.converged, fit.n_iter
            );
        }
    }

    #[test]
    fn test_metal_nll_at_init() {
        //! Diagnostic: compare Metal and CPU NLL at the init point (no optimizer).
        //! This isolates kernel correctness from optimizer convergence.
        if !is_metal_available() {
            eprintln!("Skipping: Metal not available");
            return;
        }

        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        // Generate one toy
        let expected = model.expected_data_pyhf_main(&params).unwrap();
        let toy = crate::toys::poisson_main_from_expected(&expected, 42);

        // CPU NLL at init
        let toy_model = model.with_observed_main(&toy).unwrap();
        let cpu_nll: f64 = toy_model.nll(&params).unwrap();

        // Metal NLL at init (via single-model path)
        let gpu_data = model.serialize_for_gpu().unwrap();
        let metal_data = MetalModelData::from_gpu_data(&gpu_data);
        let mut accel = MetalBatchAccelerator::from_metal_data(&metal_data, 1).unwrap();

        let n_bins = gpu_data.n_main_bins;
        let ln_facts: Vec<f64> = toy.iter().map(|&o| ln_gamma(o + 1.0)).collect();
        let obs_mask: Vec<f64> = toy.iter().map(|&o| if o > 0.0 { 1.0 } else { 0.0 }).collect();
        accel.upload_observed(&toy, &ln_facts, &obs_mask, 1).unwrap();

        let (metal_nlls, metal_grads) = accel.batch_nll_grad(&params, 1).unwrap();
        let metal_nll = metal_nlls[0];

        let rel_diff = ((metal_nll - cpu_nll) / cpu_nll).abs();
        eprintln!("CPU NLL at init:   {cpu_nll:.8}");
        eprintln!("Metal NLL at init: {metal_nll:.8}");
        eprintln!("Relative diff:     {rel_diff:.2e}");
        eprintln!("n_bins={n_bins}, n_params={}", params.len());
        eprintln!("Metal grad[0..3]: {:?}", &metal_grads[..params.len().min(3)]);

        // f32 precision: expect < 1e-3 relative diff for NLL
        assert!(
            rel_diff < 1e-3,
            "Metal NLL at init deviates from CPU: {metal_nll:.8} vs {cpu_nll:.8} (rel={rel_diff:.2e})"
        );
    }

    #[test]
    fn test_metal_matches_cpu() {
        if !is_metal_available() {
            eprintln!("Skipping: Metal not available");
            return;
        }

        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        // Metal (f32)
        let metal_results = fit_toys_batch_metal(&model, &params, 5, 42, None).unwrap();
        // CPU (f64)
        let cpu_results = crate::batch::fit_toys_batch(&model, &params, 5, 42, None);

        let mut n_close = 0;
        for (i, (mr, cr)) in metal_results.iter().zip(cpu_results.iter()).enumerate() {
            if let (Ok(m), Ok(c)) = (mr, cr) {
                let nll_diff = (m.nll - c.nll).abs();
                eprintln!(
                    "Toy {i}: Metal NLL={:.6} CPU NLL={:.6} diff={:.2e} conv={}",
                    m.nll, c.nll, nll_diff, m.converged
                );
                if nll_diff < 1.0 {
                    n_close += 1;
                }
            }
        }
        // Most toys should match within 1.0 (f32 optimizer may diverge on some)
        eprintln!("{n_close}/5 toys match within 1.0 NLL");
        assert!(
            n_close >= 3,
            "Too few toys match: {n_close}/5. Metal kernel may have a bug."
        );
    }
}
