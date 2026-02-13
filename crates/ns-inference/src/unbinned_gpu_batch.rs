//! GPU-accelerated batch toy fitting for unbinned (event-level) models.
//!
//! Uses lockstep iteration: all toys are at the same L-BFGS-B iteration,
//! and a single GPU kernel computes NLL + gradient for all toys.

use crate::lbfgs::LbfgsState;
use crate::optimizer::OptimizerConfig;
use ns_core::{FitResult, Result};

trait UnbinnedBatchAccel {
    fn n_params(&self) -> usize;
    fn n_toys(&self) -> usize;
    fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)>;
    fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>>;
}

fn fit_lockstep<A: UnbinnedBatchAccel>(
    mut accel: A,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, A)> {
    let config = config.unwrap_or_default();
    let n_params = accel.n_params();
    let n_toys = accel.n_toys();

    if init_params.len() != n_params {
        return Err(ns_core::Error::Validation(format!(
            "init_params length mismatch: expected {}, got {}",
            n_params,
            init_params.len()
        )));
    }
    if bounds.len() != n_params {
        return Err(ns_core::Error::Validation(format!(
            "bounds length mismatch: expected {}, got {}",
            n_params,
            bounds.len()
        )));
    }

    let effective_m = config.effective_m(n_params);
    let mut states: Vec<LbfgsState> = (0..n_toys)
        .map(|_| LbfgsState::new(init_params.to_vec(), bounds.to_vec(), effective_m, config.tol))
        .collect();

    let mut params_flat = vec![0.0f64; n_toys * n_params];
    let mut params_trial = vec![0.0f64; n_toys * n_params];
    let mut directions: Vec<Option<Vec<f64>>> = vec![None; n_toys];
    let mut step_sizes: Vec<f64> = vec![0.0; n_toys];
    let mut accepted: Vec<bool> = vec![false; n_toys];

    for _outer_iter in 0..config.max_iter {
        let mut any_active = false;
        for t in 0..n_toys {
            let dst = &mut params_flat[t * n_params..(t + 1) * n_params];
            dst.copy_from_slice(&states[t].x);
            if !states[t].converged {
                any_active = true;
            }
        }
        if !any_active {
            break;
        }

        let (nlls, grads) = accel.batch_nll_grad(&params_flat)?;
        debug_assert_eq!(nlls.len(), n_toys);
        debug_assert_eq!(grads.len(), n_toys * n_params);

        directions.fill(None);
        step_sizes.fill(0.0);
        accepted.fill(false);

        for t in 0..n_toys {
            if states[t].converged {
                accepted[t] = true;
                continue;
            }
            let grad = &grads[t * n_params..(t + 1) * n_params];
            if let Some(dir) = states[t].begin_iter(nlls[t], grad) {
                let step0 = states[t].propose_step_size(&dir);
                directions[t] = Some(dir);
                step_sizes[t] = step0;
            } else {
                accepted[t] = true;
            }
        }

        // Backtracking line search in batch: propose per-toy steps, evaluate in one `batch_nll`,
        // shrink steps that do not improve the objective.
        let max_backtracks = 16usize;
        for _bt in 0..max_backtracks {
            let mut any_pending = false;
            params_trial.copy_from_slice(&params_flat);

            for t in 0..n_toys {
                if accepted[t] {
                    continue;
                }
                let Some(dir) = directions[t].as_ref() else {
                    accepted[t] = true;
                    continue;
                };
                let x_next = states[t].propose_x(dir, step_sizes[t]);
                let dst = &mut params_trial[t * n_params..(t + 1) * n_params];
                dst.copy_from_slice(&x_next);
                any_pending = true;
            }
            if !any_pending {
                break;
            }

            let nll_trial = accel.batch_nll(&params_trial)?;
            debug_assert_eq!(nll_trial.len(), n_toys);

            let mut all_accepted = true;
            for t in 0..n_toys {
                if accepted[t] {
                    continue;
                }
                let new_nll = nll_trial[t];
                let old_nll = nlls[t];
                if new_nll.is_finite() && old_nll.is_finite() && new_nll <= old_nll {
                    let x_next = params_trial[t * n_params..(t + 1) * n_params].to_vec();
                    states[t].accept_x(x_next);
                    states[t].fval = new_nll;
                    accepted[t] = true;
                } else {
                    step_sizes[t] *= 0.5;
                    if step_sizes[t] <= 1e-20 {
                        // Give up on this toy for this iteration; keep x unchanged.
                        accepted[t] = true;
                    } else {
                        all_accepted = false;
                    }
                }
            }
            if all_accepted {
                break;
            }
        }
    }

    let results: Vec<Result<FitResult>> = states
        .into_iter()
        .map(|state| {
            if state.failed {
                Err(ns_core::Error::Computation(
                    "L-BFGS-B lockstep optimizer failed (non-finite NLL/gradient)".into(),
                ))
            } else {
                Ok(FitResult::new(
                    state.x,
                    vec![0.0; n_params], // uncertainties not computed in batch toy fits
                    state.fval,
                    state.converged,
                    state.iter,
                    state.n_fev,
                    state.n_gev,
                ))
            }
        })
        .collect();

    Ok((results, accel))
}

/// Fit toy datasets on Metal GPU in batch/lockstep mode.
#[cfg(feature = "metal")]
pub fn fit_unbinned_toys_batch_metal(
    static_model: &ns_compute::unbinned_types::UnbinnedGpuModelData,
    toy_offsets: &[u32],
    obs_flat: &[f64],
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator)>
{
    let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
        static_model,
        toy_offsets,
        obs_flat,
        n_toys,
    )?;

    struct Wrap(ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator);
    impl UnbinnedBatchAccel for Wrap {
        fn n_params(&self) -> usize {
            self.0.n_params()
        }
        fn n_toys(&self) -> usize {
            self.0.n_toys()
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            self.0.batch_nll_grad(params_flat)
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            self.0.batch_nll(params_flat)
        }
    }

    let (results, wrap) = fit_lockstep(Wrap(accel), init_params, bounds, config)?;
    Ok((results, wrap.0))
}

/// Fit toy datasets on Metal GPU in batch/lockstep mode for a multi-channel model.
///
/// The total NLL/grad is the sum of per-channel contributions. Constraints must be attached
/// to exactly one static model (typically the first included channel) to avoid double counting.
#[cfg(feature = "metal")]
pub fn fit_unbinned_toys_batch_metal_multi(
    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
    toy_offsets_by_channel: &[Vec<u32>],
    obs_flat_by_channel: &[Vec<f64>],
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(
    Vec<Result<FitResult>>,
    Vec<ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator>,
)> {
    if static_models.is_empty() {
        return Err(ns_core::Error::Validation("static_models must be non-empty".into()));
    }
    if toy_offsets_by_channel.len() != static_models.len()
        || obs_flat_by_channel.len() != static_models.len()
    {
        return Err(ns_core::Error::Validation(format!(
            "channel length mismatch: static_models={}, toy_offsets_by_channel={}, obs_flat_by_channel={}",
            static_models.len(),
            toy_offsets_by_channel.len(),
            obs_flat_by_channel.len()
        )));
    }

    let mut accels = Vec::with_capacity(static_models.len());
    for (i, ((m, offs), obs)) in static_models
        .iter()
        .zip(toy_offsets_by_channel.iter())
        .zip(obs_flat_by_channel.iter())
        .enumerate()
    {
        let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
            m, offs, obs, n_toys,
        )
        .map_err(|e| ns_core::Error::Validation(format!("channel {i}: {e}")))?;
        accels.push(accel);
    }

    fit_unbinned_toys_batch_metal_with_accels(accels, init_params, bounds, config)
}

/// Fit toy datasets on Metal GPU in batch/lockstep mode, reusing existing per-channel accelerators.
#[cfg(feature = "metal")]
pub fn fit_unbinned_toys_batch_metal_with_accels(
    accels: Vec<ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator>,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(
    Vec<Result<FitResult>>,
    Vec<ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator>,
)> {
    if accels.is_empty() {
        return Err(ns_core::Error::Validation("accels must be non-empty".into()));
    }
    let n_params = accels[0].n_params();
    let n_toys = accels[0].n_toys();
    for (i, a) in accels.iter().enumerate().skip(1) {
        if a.n_params() != n_params || a.n_toys() != n_toys {
            return Err(ns_core::Error::Validation(format!(
                "accelerator shape mismatch at channel {i}: expected (n_params={n_params}, n_toys={n_toys}), got (n_params={}, n_toys={})",
                a.n_params(),
                a.n_toys()
            )));
        }
    }

    struct Multi {
        accels: Vec<ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator>,
        n_params: usize,
        n_toys: usize,
    }
    impl UnbinnedBatchAccel for Multi {
        fn n_params(&self) -> usize {
            self.n_params
        }
        fn n_toys(&self) -> usize {
            self.n_toys
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            let (mut nll, mut grad) = self.accels[0].batch_nll_grad(params_flat)?;
            for a in self.accels.iter_mut().skip(1) {
                let (nll2, grad2) = a.batch_nll_grad(params_flat)?;
                for (x, y) in nll.iter_mut().zip(nll2.into_iter()) {
                    *x += y;
                }
                for (x, y) in grad.iter_mut().zip(grad2.into_iter()) {
                    *x += y;
                }
            }
            Ok((nll, grad))
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            let mut nll = self.accels[0].batch_nll(params_flat)?;
            for a in self.accels.iter_mut().skip(1) {
                let nll2 = a.batch_nll(params_flat)?;
                for (x, y) in nll.iter_mut().zip(nll2.into_iter()) {
                    *x += y;
                }
            }
            Ok(nll)
        }
    }

    let multi = Multi { accels, n_params, n_toys };
    let (results, multi) = fit_lockstep(multi, init_params, bounds, config)?;
    Ok((results, multi.accels))
}

/// Fit toy datasets on Metal GPU using device-resident toy data (Metal Buffer, f32).
///
/// `buf_obs_flat_f32` is a Metal shared buffer from [`MetalUnbinnedToySampler::sample_toys_1d_device`].
/// This avoids the f64→f32 conversion and buffer re-allocation.
#[cfg(feature = "metal")]
pub fn fit_unbinned_toys_batch_metal_device(
    static_model: &ns_compute::unbinned_types::UnbinnedGpuModelData,
    toy_offsets: &[u32],
    buf_obs_flat_f32: ns_compute::metal_rs::Buffer,
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator)>
{
    let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
        static_model,
        toy_offsets,
        buf_obs_flat_f32,
        n_toys,
    )?;

    struct Wrap(ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator);
    impl UnbinnedBatchAccel for Wrap {
        fn n_params(&self) -> usize {
            self.0.n_params()
        }
        fn n_toys(&self) -> usize {
            self.0.n_toys()
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            self.0.batch_nll_grad(params_flat)
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            self.0.batch_nll(params_flat)
        }
    }

    let (results, wrap) = fit_lockstep(Wrap(accel), init_params, bounds, config)?;
    Ok((results, wrap.0))
}

/// Fit toy datasets on Metal GPU using device-resident toy data, multi-channel.
///
/// `buf_obs_flat_by_channel` contains per-channel Metal Buffers (f32) from the toy sampler.
/// Constraints must be attached to exactly one static model to avoid double counting.
#[cfg(feature = "metal")]
pub fn fit_unbinned_toys_batch_metal_device_multi(
    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
    toy_offsets_by_channel: &[Vec<u32>],
    buf_obs_flat_by_channel: Vec<ns_compute::metal_rs::Buffer>,
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(
    Vec<Result<FitResult>>,
    Vec<ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator>,
)> {
    if static_models.is_empty() {
        return Err(ns_core::Error::Validation("static_models must be non-empty".into()));
    }
    if toy_offsets_by_channel.len() != static_models.len() {
        return Err(ns_core::Error::Validation(format!(
            "toy_offsets_by_channel length mismatch: expected {}, got {}",
            static_models.len(),
            toy_offsets_by_channel.len()
        )));
    }
    if buf_obs_flat_by_channel.len() != static_models.len() {
        return Err(ns_core::Error::Validation(format!(
            "buf_obs_flat_by_channel length mismatch: expected {}, got {}",
            static_models.len(),
            buf_obs_flat_by_channel.len()
        )));
    }

    let mut accels = Vec::with_capacity(static_models.len());
    for (i, (m, (offs, buf_obs))) in static_models
        .iter()
        .zip(toy_offsets_by_channel.iter().zip(buf_obs_flat_by_channel.into_iter()))
        .enumerate()
    {
        let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
            m, offs, buf_obs, n_toys,
        )
        .map_err(|e| ns_core::Error::Validation(format!("channel {i}: {e}")))?;
        accels.push(accel);
    }

    fit_unbinned_toys_batch_metal_with_accels(accels, init_params, bounds, config)
}

/// Fit toy datasets entirely on CUDA GPU (GPU-native L-BFGS optimizer).
///
/// Single kernel launch: 1 CUDA block = 1 toy = 1 complete L-BFGS-B optimization.
/// All iterations, line search, and convergence checks run on the GPU with zero
/// host-device roundtrips per iteration. This eliminates the ~15µs/iter PCIe latency
/// that dominates the lockstep path for small-to-medium event counts.
///
/// Falls back to host-side lockstep if the L-BFGS kernel is not available.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_device_resident(
    static_model: &ns_compute::unbinned_types::UnbinnedGpuModelData,
    toy_offsets: &[u32],
    obs_flat: &[f64],
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<Vec<Result<FitResult>>> {
    let config = config.unwrap_or_default();

    let accel =
        ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
            static_model,
            toy_offsets,
            obs_flat,
            n_toys,
        )?;

    let max_iter = config.max_iter as u32;
    let n_params = init_params.len();
    let lbfgs_m = (config.effective_m(n_params) as u32).min(16);
    let tol = config.tol;
    let max_backtracks = 16u32;

    accel.batch_fit_on_device(init_params, bounds, max_iter, lbfgs_m, tol, max_backtracks)
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda(
    static_model: &ns_compute::unbinned_types::UnbinnedGpuModelData,
    toy_offsets: &[u32],
    obs_flat: &[f64],
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator)>
{
    let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
        static_model,
        toy_offsets,
        obs_flat,
        n_toys,
    )?;

    fit_lockstep_cuda(accel, init_params, bounds, config)
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode for a multi-channel model (host toys).
///
/// The total NLL/grad is the sum of per-channel contributions. Constraints must be attached
/// to exactly one static model (typically the first included channel) to avoid double counting.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_multi(
    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
    toy_offsets_by_channel: &[Vec<u32>],
    obs_flat_by_channel: &[Vec<f64>],
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(
    Vec<Result<FitResult>>,
    Vec<ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator>,
)> {
    if static_models.is_empty() {
        return Err(ns_core::Error::Validation("static_models must be non-empty".into()));
    }
    if toy_offsets_by_channel.len() != static_models.len()
        || obs_flat_by_channel.len() != static_models.len()
    {
        return Err(ns_core::Error::Validation(format!(
            "channel length mismatch: static_models={}, toy_offsets_by_channel={}, obs_flat_by_channel={}",
            static_models.len(),
            toy_offsets_by_channel.len(),
            obs_flat_by_channel.len()
        )));
    }

    let mut accels = Vec::with_capacity(static_models.len());
    for (i, ((m, offs), obs)) in static_models
        .iter()
        .zip(toy_offsets_by_channel.iter())
        .zip(obs_flat_by_channel.iter())
        .enumerate()
    {
        let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
            m, offs, obs, n_toys,
        )
        .map_err(|e| ns_core::Error::Validation(format!("channel {i}: {e}")))?;
        accels.push(accel);
    }

    fit_unbinned_toys_batch_cuda_with_accels(accels, init_params, bounds, config)
}

/// Host-toy shard assigned to a specific CUDA device.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaHostToyShard {
    /// Inclusive toy start index in the original toy array.
    pub toy_start: usize,
    /// Exclusive toy end index in the original toy array.
    pub toy_end: usize,
    /// CUDA device id used for this shard.
    pub device_id: usize,
    /// Per-channel toy offsets, rebased to start from 0.
    pub toy_offsets_by_channel: Vec<Vec<u32>>,
    /// Per-channel flattened observable buffers for this shard only.
    pub obs_flat_by_channel: Vec<Vec<f64>>,
}

/// Split host toy buffers across CUDA devices.
///
/// Sharding is contiguous in toy index space and preserves toy order.
#[cfg(feature = "cuda")]
pub fn shard_unbinned_host_toys_multi_channel(
    toy_offsets_by_channel: &[Vec<u32>],
    obs_flat_by_channel: &[Vec<f64>],
    device_ids: &[usize],
) -> Result<Vec<CudaHostToyShard>> {
    if toy_offsets_by_channel.is_empty() {
        return Err(ns_core::Error::Validation("toy_offsets_by_channel must be non-empty".into()));
    }
    if device_ids.is_empty() {
        return Err(ns_core::Error::Validation("device_ids must be non-empty".into()));
    }
    if toy_offsets_by_channel.len() != obs_flat_by_channel.len() {
        return Err(ns_core::Error::Validation(format!(
            "channel length mismatch: toy_offsets_by_channel={}, obs_flat_by_channel={}",
            toy_offsets_by_channel.len(),
            obs_flat_by_channel.len()
        )));
    }

    let n_channels = toy_offsets_by_channel.len();
    let n_toys = toy_offsets_by_channel[0].len().saturating_sub(1);
    let expected_offsets_len = n_toys + 1;

    for ch in 0..n_channels {
        let offs = &toy_offsets_by_channel[ch];
        if offs.len() != expected_offsets_len {
            return Err(ns_core::Error::Validation(format!(
                "channel {ch}: toy_offsets length mismatch: expected {expected_offsets_len}, got {}",
                offs.len()
            )));
        }
        if offs.first().copied().unwrap_or(1) != 0 {
            return Err(ns_core::Error::Validation(format!(
                "channel {ch}: toy_offsets must start with 0"
            )));
        }
        if offs.windows(2).any(|w| w[0] > w[1]) {
            return Err(ns_core::Error::Validation(format!(
                "channel {ch}: toy_offsets must be non-decreasing"
            )));
        }
        let last = *offs.last().unwrap_or(&0) as usize;
        if last != obs_flat_by_channel[ch].len() {
            return Err(ns_core::Error::Validation(format!(
                "channel {ch}: toy_offsets last entry ({last}) must equal obs_flat len ({})",
                obs_flat_by_channel[ch].len()
            )));
        }
    }

    if n_toys == 0 {
        return Ok(Vec::new());
    }

    let n_shards = device_ids.len().min(n_toys);
    let toys_per_shard = n_toys.div_ceil(n_shards);
    let mut shards = Vec::with_capacity(n_shards);

    for shard_idx in 0..n_shards {
        let toy_start = shard_idx * toys_per_shard;
        let toy_end = ((shard_idx + 1) * toys_per_shard).min(n_toys);
        if toy_start >= toy_end {
            break;
        }

        let mut shard_offsets_by_channel = Vec::with_capacity(n_channels);
        let mut shard_obs_by_channel = Vec::with_capacity(n_channels);

        for ch in 0..n_channels {
            let offs = &toy_offsets_by_channel[ch];
            let ev_start = offs[toy_start] as usize;
            let ev_end = offs[toy_end] as usize;
            let rebased_offsets: Vec<u32> =
                offs[toy_start..=toy_end].iter().map(|&x| x - offs[toy_start]).collect();
            let shard_obs = obs_flat_by_channel[ch][ev_start..ev_end].to_vec();
            shard_offsets_by_channel.push(rebased_offsets);
            shard_obs_by_channel.push(shard_obs);
        }

        shards.push(CudaHostToyShard {
            toy_start,
            toy_end,
            device_id: device_ids[shard_idx],
            toy_offsets_by_channel: shard_offsets_by_channel,
            obs_flat_by_channel: shard_obs_by_channel,
        });
    }

    Ok(shards)
}

/// Fit host toy buffers across multiple CUDA devices.
///
/// Toys are split into contiguous shards, each shard is processed on one GPU and
/// final results are merged back in original toy order.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_multi_gpu_host(
    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
    toy_offsets_by_channel: &[Vec<u32>],
    obs_flat_by_channel: &[Vec<f64>],
    device_ids: &[usize],
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<Vec<Result<FitResult>>> {
    if static_models.is_empty() {
        return Err(ns_core::Error::Validation("static_models must be non-empty".into()));
    }
    if toy_offsets_by_channel.len() != static_models.len()
        || obs_flat_by_channel.len() != static_models.len()
    {
        return Err(ns_core::Error::Validation(format!(
            "channel length mismatch: static_models={}, toy_offsets_by_channel={}, obs_flat_by_channel={}",
            static_models.len(),
            toy_offsets_by_channel.len(),
            obs_flat_by_channel.len()
        )));
    }

    let n_toys = toy_offsets_by_channel.first().map(|o| o.len().saturating_sub(1)).unwrap_or(0);
    if n_toys == 0 {
        return Ok(Vec::new());
    }

    let shards = shard_unbinned_host_toys_multi_channel(
        toy_offsets_by_channel,
        obs_flat_by_channel,
        device_ids,
    )?;
    if shards.is_empty() {
        return Ok(Vec::new());
    }

    // Run each shard on its own thread so all GPUs work in parallel.
    let shard_results: Vec<Result<(usize, usize, Vec<Result<FitResult>>)>> = std::thread::scope(
        |scope| {
            let handles: Vec<_> = shards
                .into_iter()
                .map(|shard| {
                    let config = config.clone();
                    scope.spawn(move || {
                        let shard_n_toys = shard.toy_end - shard.toy_start;
                        let mut accels = Vec::with_capacity(static_models.len());
                        for (i, ((m, offs), obs)) in static_models
                            .iter()
                            .zip(shard.toy_offsets_by_channel.iter())
                            .zip(shard.obs_flat_by_channel.iter())
                            .enumerate()
                        {
                            let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_on_device(
                                m,
                                offs,
                                obs,
                                shard_n_toys,
                                shard.device_id,
                            )
                            .map_err(|e| {
                                ns_core::Error::Validation(format!(
                                    "gpu {} channel {i}: {e}",
                                    shard.device_id
                                ))
                            })?;
                            accels.push(accel);
                        }

                        let (results, _accels) =
                            fit_unbinned_toys_batch_cuda_with_accels(accels, init_params, bounds, config)?;
                        if results.len() != shard_n_toys {
                            return Err(ns_core::Error::Computation(format!(
                                "shard result length mismatch on gpu {}: expected {}, got {}",
                                shard.device_id,
                                shard_n_toys,
                                results.len()
                            )));
                        }
                        Ok((shard.toy_start, shard.toy_end, results))
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        },
    );

    let mut merged: Vec<Option<Result<FitResult>>> = (0..n_toys).map(|_| None).collect();
    for shard_result in shard_results {
        let (toy_start, _toy_end, results) = shard_result?;
        for (local_idx, res) in results.into_iter().enumerate() {
            merged[toy_start + local_idx] = Some(res);
        }
    }

    merged
        .into_iter()
        .enumerate()
        .map(|(idx, maybe)| {
            maybe.ok_or_else(|| {
                ns_core::Error::Computation(format!("missing shard result for toy index {idx}"))
            })
        })
        .collect()
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode, using device-resident toy data.
///
/// This is the device-resident path: `d_obs_flat` stays on the GPU (no H2D copy of the large
/// event buffer). Both the sampler and the batch fitter must share the same CUDA context.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_device(
    ctx: std::sync::Arc<ns_compute::cuda_driver::CudaContext>,
    stream: std::sync::Arc<ns_compute::cuda_driver::CudaStream>,
    static_model: &ns_compute::unbinned_types::UnbinnedGpuModelData,
    toy_offsets: &[u32],
    d_obs_flat: ns_compute::cuda_driver::CudaSlice<f64>,
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator)>
{
    let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
        ctx,
        stream,
        static_model,
        toy_offsets,
        d_obs_flat,
        n_toys,
    )?;

    fit_lockstep_cuda(accel, init_params, bounds, config)
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode for a multi-channel model (device toys).
///
/// `d_obs_flat_by_channel` is moved into the per-channel accelerators.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_device_multi(
    ctx: std::sync::Arc<ns_compute::cuda_driver::CudaContext>,
    stream: std::sync::Arc<ns_compute::cuda_driver::CudaStream>,
    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
    toy_offsets_by_channel: &[Vec<u32>],
    d_obs_flat_by_channel: Vec<ns_compute::cuda_driver::CudaSlice<f64>>,
    n_toys: usize,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(
    Vec<Result<FitResult>>,
    Vec<ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator>,
)> {
    if static_models.is_empty() {
        return Err(ns_core::Error::Validation("static_models must be non-empty".into()));
    }
    if toy_offsets_by_channel.len() != static_models.len() {
        return Err(ns_core::Error::Validation(format!(
            "toy_offsets_by_channel length mismatch: expected {}, got {}",
            static_models.len(),
            toy_offsets_by_channel.len()
        )));
    }
    if d_obs_flat_by_channel.len() != static_models.len() {
        return Err(ns_core::Error::Validation(format!(
            "d_obs_flat_by_channel length mismatch: expected {}, got {}",
            static_models.len(),
            d_obs_flat_by_channel.len()
        )));
    }

    let mut accels = Vec::with_capacity(static_models.len());
    for (i, (m, (offs, d_obs))) in static_models
        .iter()
        .zip(toy_offsets_by_channel.iter().zip(d_obs_flat_by_channel.into_iter()))
        .enumerate()
    {
        let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
            ctx.clone(),
            stream.clone(),
            m,
            offs,
            d_obs,
            n_toys,
        )
        .map_err(|e| ns_core::Error::Validation(format!("channel {i}: {e}")))?;
        accels.push(accel);
    }

    fit_unbinned_toys_batch_cuda_with_accels(accels, init_params, bounds, config)
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode, reusing an existing accelerator.
///
/// This is useful when multiple fits over the same toy data are required (e.g. free and
/// fixed-POI fits in toy-based hypothesis tests) while keeping the large `obs_flat` buffer
/// device-resident and avoiding re-upload.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_with_accel(
    accel: ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator)>
{
    fit_lockstep_cuda(accel, init_params, bounds, config)
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode, reusing existing per-channel accelerators.
#[cfg(feature = "cuda")]
pub fn fit_unbinned_toys_batch_cuda_with_accels(
    accels: Vec<ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator>,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(
    Vec<Result<FitResult>>,
    Vec<ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator>,
)> {
    if accels.is_empty() {
        return Err(ns_core::Error::Validation("accels must be non-empty".into()));
    }
    let n_params = accels[0].n_params();
    let n_toys = accels[0].n_toys();
    for (i, a) in accels.iter().enumerate().skip(1) {
        if a.n_params() != n_params || a.n_toys() != n_toys {
            return Err(ns_core::Error::Validation(format!(
                "accelerator shape mismatch at channel {i}: expected (n_params={n_params}, n_toys={n_toys}), got (n_params={}, n_toys={})",
                a.n_params(),
                a.n_toys()
            )));
        }
    }

    struct Multi {
        accels: Vec<ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator>,
        n_params: usize,
        n_toys: usize,
    }
    impl UnbinnedBatchAccel for Multi {
        fn n_params(&self) -> usize {
            self.n_params
        }
        fn n_toys(&self) -> usize {
            self.n_toys
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            let (mut nll, mut grad) = self.accels[0].batch_nll_grad(params_flat)?;
            for a in self.accels.iter_mut().skip(1) {
                let (nll2, grad2) = a.batch_nll_grad(params_flat)?;
                for (x, y) in nll.iter_mut().zip(nll2.into_iter()) {
                    *x += y;
                }
                for (x, y) in grad.iter_mut().zip(grad2.into_iter()) {
                    *x += y;
                }
            }
            Ok((nll, grad))
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            let mut nll = self.accels[0].batch_nll(params_flat)?;
            for a in self.accels.iter_mut().skip(1) {
                let nll2 = a.batch_nll(params_flat)?;
                for (x, y) in nll.iter_mut().zip(nll2.into_iter()) {
                    *x += y;
                }
            }
            Ok(nll)
        }
    }

    let multi = Multi { accels, n_params, n_toys };
    let (results, multi) = fit_lockstep(multi, init_params, bounds, config)?;
    Ok((results, multi.accels))
}

/// Fit toy datasets on CUDA GPU in batch/lockstep mode using flow (externally-computed) PDFs.
///
/// `logp_flat` is `[n_procs × total_events]` row-major: pre-computed log-prob values
/// for all toy events across all processes. This buffer is uploaded to GPU once.
///
/// `toy_offsets` is `[n_toys + 1]`: prefix sums of per-toy event counts.
///
/// Yields are computed from `FlowBatchProcessDesc` using the parameter values at each
/// optimizer iteration. Gradients use central finite differences on the batch NLL kernel.
#[cfg(feature = "cuda")]
pub fn fit_flow_toys_batch_cuda(
    config: &ns_compute::cuda_flow_batch::FlowBatchNllConfig,
    logp_flat: &[f64],
    init_params: &[f64],
    bounds: &[(f64, f64)],
    opt_config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator)> {
    let accel =
        ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator::new(config, logp_flat, 0)?;
    fit_lockstep_flow_cuda(accel, init_params, bounds, opt_config)
}

/// Fit toy datasets on CUDA GPU using flow PDFs, reusing an existing accelerator.
///
/// Useful when multiple fits over the same toy data are required (e.g. free and
/// fixed-POI fits in toy-based hypothesis tests).
#[cfg(feature = "cuda")]
pub fn fit_flow_toys_batch_cuda_with_accel(
    accel: ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator)> {
    fit_lockstep_flow_cuda(accel, init_params, bounds, config)
}

/// Fit toy datasets across multiple CUDA GPUs using flow PDFs.
///
/// Toys are sharded evenly across `n_gpus` devices. Each device gets its own
/// `CudaFlowBatchNllAccelerator` with a slice of the toys. Results are merged.
///
/// `logp_flat_by_shard[g]` is `[n_procs × shard_total_events]` for GPU `g`.
/// `toy_offsets_by_shard[g]` is `[shard_n_toys + 1]` for GPU `g`.
#[cfg(feature = "cuda")]
pub fn fit_flow_toys_batch_multi_gpu(
    configs: Vec<ns_compute::cuda_flow_batch::FlowBatchNllConfig>,
    logp_flat_by_shard: Vec<Vec<f64>>,
    device_ids: &[usize],
    init_params: &[f64],
    bounds: &[(f64, f64)],
    opt_config: Option<OptimizerConfig>,
) -> Result<Vec<Result<FitResult>>> {
    if configs.is_empty() || device_ids.is_empty() {
        return Err(ns_core::Error::Validation("configs and device_ids must be non-empty".into()));
    }
    if configs.len() != device_ids.len() || configs.len() != logp_flat_by_shard.len() {
        return Err(ns_core::Error::Validation(format!(
            "length mismatch: configs={}, device_ids={}, logp_flat_by_shard={}",
            configs.len(),
            device_ids.len(),
            logp_flat_by_shard.len()
        )));
    }

    // Create per-GPU accelerators (in parallel via rayon).
    let accels: Vec<ns_core::Result<ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator>> =
        configs
            .iter()
            .zip(logp_flat_by_shard.iter())
            .zip(device_ids.iter())
            .map(|((cfg, logp), &dev_id)| {
                ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator::new(cfg, logp, dev_id)
            })
            .collect();

    // Check for errors.
    let mut good_accels = Vec::with_capacity(accels.len());
    for (i, a) in accels.into_iter().enumerate() {
        good_accels.push(
            a.map_err(|e| ns_core::Error::Computation(format!("GPU {}: {e}", device_ids[i])))?,
        );
    }

    // Fit each shard independently (sequentially — each runs on its own GPU).
    let mut all_results = Vec::new();
    for accel in good_accels {
        let (shard_results, _accel) =
            fit_lockstep_flow_cuda(accel, init_params, bounds, opt_config.clone())?;
        all_results.extend(shard_results);
    }

    Ok(all_results)
}

/// Helper: shard toys evenly across `n_gpus` devices.
///
/// Returns per-shard `(FlowBatchNllConfig, logp_flat, device_id)`.
/// The `logp_flat` for each shard contains only the events belonging to that shard's toys.
#[cfg(feature = "cuda")]
pub fn shard_flow_toys(
    full_config: &ns_compute::cuda_flow_batch::FlowBatchNllConfig,
    full_logp_flat: &[f64],
    n_gpus: usize,
) -> Vec<(ns_compute::cuda_flow_batch::FlowBatchNllConfig, Vec<f64>, usize)> {
    let n_toys = full_config.n_toys;
    let n_procs = full_config.processes.len();
    let total_events = full_config.total_events;

    if n_gpus == 0 || n_toys == 0 {
        return vec![];
    }

    let toys_per_gpu = (n_toys + n_gpus - 1) / n_gpus;
    let mut shards = Vec::with_capacity(n_gpus);

    for g in 0..n_gpus {
        let toy_start = g * toys_per_gpu;
        let toy_end = ((g + 1) * toys_per_gpu).min(n_toys);
        if toy_start >= toy_end {
            break;
        }
        let shard_n_toys = toy_end - toy_start;

        // Extract event range for this shard's toys.
        let ev_start = full_config.toy_offsets[toy_start] as usize;
        let ev_end = full_config.toy_offsets[toy_end] as usize;
        let shard_total_events = ev_end - ev_start;

        // Re-base toy offsets to start at 0.
        let shard_offsets: Vec<u32> = full_config.toy_offsets[toy_start..=toy_end]
            .iter()
            .map(|&o| o - full_config.toy_offsets[toy_start])
            .collect();

        // Extract logp for this shard: logp_flat is [n_procs × total_events] row-major.
        let mut shard_logp = Vec::with_capacity(n_procs * shard_total_events);
        for p in 0..n_procs {
            let proc_offset = p * total_events;
            shard_logp
                .extend_from_slice(&full_logp_flat[proc_offset + ev_start..proc_offset + ev_end]);
        }

        let shard_config = ns_compute::cuda_flow_batch::FlowBatchNllConfig {
            total_events: shard_total_events,
            n_toys: shard_n_toys,
            toy_offsets: shard_offsets,
            processes: full_config.processes.clone(),
            n_params: full_config.n_params,
            gauss_constraints: full_config.gauss_constraints.clone(),
            constraint_const: full_config.constraint_const,
        };

        shards.push((shard_config, shard_logp, g));
    }

    shards
}

/// Fit toy datasets on CUDA GPU using flow PDFs with device-resident f32 logp (CUDA EP path).
///
/// `d_logp_flat_ptr` is a raw CUDA device pointer to `[n_procs × total_events]` row-major
/// `float` data produced by ONNX Runtime CUDA EP I/O binding (zero H2D copy).
///
/// The accelerator is created via `new_device_f32` (no host logp upload).
#[cfg(feature = "cuda")]
pub fn fit_flow_toys_batch_cuda_device_f32(
    config: &ns_compute::cuda_flow_batch::FlowBatchNllConfig,
    d_logp_flat_ptr: u64,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    opt_config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator)> {
    let accel =
        ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator::new_device_f32(config, 0)?;
    fit_lockstep_flow_cuda_f32(accel, d_logp_flat_ptr, init_params, bounds, opt_config)
}

#[cfg(feature = "cuda")]
fn fit_lockstep_flow_cuda_f32(
    accel: ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator,
    d_logp_flat_ptr: u64,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator)> {
    struct WrapF32 {
        accel: ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator,
        ptr: u64,
    }
    impl UnbinnedBatchAccel for WrapF32 {
        fn n_params(&self) -> usize {
            self.accel.n_params()
        }
        fn n_toys(&self) -> usize {
            self.accel.n_toys()
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            self.accel.batch_nll_grad_device_ptr_f32(self.ptr, params_flat)
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            self.accel.batch_nll_device_ptr_f32(self.ptr, params_flat)
        }
    }

    let wrap = WrapF32 { accel, ptr: d_logp_flat_ptr };
    let (results, wrap) = fit_lockstep(wrap, init_params, bounds, config)?;
    Ok((results, wrap.accel))
}

#[cfg(feature = "cuda")]
fn fit_lockstep_flow_cuda(
    accel: ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator)> {
    struct Wrap(ns_compute::cuda_flow_batch::CudaFlowBatchNllAccelerator);
    impl UnbinnedBatchAccel for Wrap {
        fn n_params(&self) -> usize {
            self.0.n_params()
        }
        fn n_toys(&self) -> usize {
            self.0.n_toys()
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            self.0.batch_nll_grad(params_flat)
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            self.0.batch_nll(params_flat)
        }
    }

    let (results, wrap) = fit_lockstep(Wrap(accel), init_params, bounds, config)?;
    Ok((results, wrap.0))
}

#[cfg(feature = "cuda")]
fn fit_lockstep_cuda(
    accel: ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator,
    init_params: &[f64],
    bounds: &[(f64, f64)],
    config: Option<OptimizerConfig>,
) -> Result<(Vec<Result<FitResult>>, ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator)>
{
    struct Wrap(ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator);
    impl UnbinnedBatchAccel for Wrap {
        fn n_params(&self) -> usize {
            self.0.n_params()
        }
        fn n_toys(&self) -> usize {
            self.0.n_toys()
        }
        fn batch_nll_grad(&mut self, params_flat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
            self.0.batch_nll_grad(params_flat)
        }
        fn batch_nll(&mut self, params_flat: &[f64]) -> Result<Vec<f64>> {
            self.0.batch_nll(params_flat)
        }
    }

    let (results, wrap) = fit_lockstep(Wrap(accel), init_params, bounds, config)?;
    Ok((results, wrap.0))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::shard_unbinned_host_toys_multi_channel;

    #[test]
    fn shard_unbinned_host_toys_preserves_toy_ranges() {
        let toy_offsets_by_channel = vec![vec![0, 2, 5, 6, 8, 9], vec![0, 1, 3, 5, 6, 8]];
        let obs_flat_by_channel = vec![
            vec![10.0, 11.0, 20.0, 21.0, 22.0, 30.0, 40.0, 41.0, 50.0],
            vec![100.0, 200.0, 201.0, 300.0, 301.0, 400.0, 500.0, 501.0],
        ];
        let device_ids = vec![3, 5, 7];

        let shards = shard_unbinned_host_toys_multi_channel(
            &toy_offsets_by_channel,
            &obs_flat_by_channel,
            &device_ids,
        )
        .expect("sharding should succeed");

        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].device_id, 3);
        assert_eq!(shards[1].device_id, 5);
        assert_eq!(shards[2].device_id, 7);

        assert_eq!((shards[0].toy_start, shards[0].toy_end), (0, 2));
        assert_eq!((shards[1].toy_start, shards[1].toy_end), (2, 4));
        assert_eq!((shards[2].toy_start, shards[2].toy_end), (4, 5));

        assert_eq!(shards[0].toy_offsets_by_channel[0], vec![0, 2, 5]);
        assert_eq!(shards[1].toy_offsets_by_channel[0], vec![0, 1, 3]);
        assert_eq!(shards[2].toy_offsets_by_channel[0], vec![0, 1]);

        assert_eq!(shards[0].obs_flat_by_channel[0], vec![10.0, 11.0, 20.0, 21.0, 22.0]);
        assert_eq!(shards[1].obs_flat_by_channel[0], vec![30.0, 40.0, 41.0]);
        assert_eq!(shards[2].obs_flat_by_channel[0], vec![50.0]);
    }

    #[test]
    fn shard_unbinned_host_toys_validates_offsets() {
        let toy_offsets_by_channel = vec![vec![1, 2], vec![0, 1]];
        let obs_flat_by_channel = vec![vec![10.0, 11.0], vec![20.0]];
        let err = shard_unbinned_host_toys_multi_channel(
            &toy_offsets_by_channel,
            &obs_flat_by_channel,
            &[0],
        )
        .expect_err("invalid offsets must fail");
        let msg = format!("{err}");
        assert!(msg.contains("must start with 0"), "unexpected error: {msg}");
    }
}
