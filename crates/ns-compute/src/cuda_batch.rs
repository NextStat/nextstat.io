//! CUDA batch NLL+gradient accelerator.
//!
//! Manages GPU memory, PTX loading, and kernel launch for fused
//! Poisson NLL + analytical gradient computation across a batch of toy experiments.
//!
//! # Architecture
//!
//! ```text
//! 1 CUDA Block = 1 Toy experiment
//! Threads in block = bins (grid-stride loop for >1024 bins)
//! Shared memory: params[n_params] + reduction scratch
//! ```

use crate::cuda_types::*;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/batch_nll_grad.cu` at build time.
const PTX_SRC: &str = include_str!(env!("CUDA_PTX_PATH"));

/// GPU batch accelerator for HistFactory NLL + gradient.
///
/// Holds all GPU buffers (static model data + dynamic per-iteration data)
/// and provides `batch_nll_grad()` for the lockstep optimizer.
pub struct CudaBatchAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: CudaStream,
    kernel_nll_grad: CudaFunction,
    kernel_nll_only: CudaFunction,

    // --- Static model buffers (uploaded once) ---
    d_nominal: CudaSlice<f64>,
    d_samples: CudaSlice<GpuSampleInfo>,
    d_modifier_descs: CudaSlice<GpuModifierDesc>,
    d_modifier_desc_offsets: CudaSlice<u32>,
    d_per_bin_param_indices: CudaSlice<u32>,
    d_modifier_data: CudaSlice<f64>,
    d_aux_poisson: CudaSlice<GpuAuxPoissonEntry>,
    d_gauss_constr: CudaSlice<GpuGaussConstraintEntry>,

    // --- Dynamic buffers (pre-allocated for max_batch) ---
    d_params: CudaSlice<f64>,
    d_observed: CudaSlice<f64>,
    d_ln_facts: CudaSlice<f64>,
    d_obs_mask: CudaSlice<f64>,
    d_nll_out: CudaSlice<f64>,
    d_grad_out: CudaSlice<f64>,

    // --- Metadata ---
    n_params: usize,
    n_main_bins: usize,
    n_samples: usize,
    n_aux_poisson: usize,
    n_gauss_constr: usize,
    constraint_const: f64,
    max_batch: usize,
}

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA: {msg}"))
}

impl CudaBatchAccelerator {
    /// Check if CUDA is available at runtime (driver loaded, GPU present).
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Create accelerator from pre-serialized GPU model data.
    ///
    /// Uploads all static model buffers to device. Pre-allocates dynamic buffers
    /// for up to `max_batch` concurrent toy experiments.
    pub fn from_gpu_data(data: &GpuModelData, max_batch: usize) -> ns_core::Result<Self> {
        let ctx = CudaContext::new(0).map_err(|e| cuda_err(format!("context: {e}")))?;
        let stream = ctx.default_stream();

        // Load PTX module
        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel_nll_grad = module
            .load_function("batch_nll_grad")
            .map_err(|e| cuda_err(format!("load batch_nll_grad: {e}")))?;
        let kernel_nll_only = module
            .load_function("batch_nll_only")
            .map_err(|e| cuda_err(format!("load batch_nll_only: {e}")))?;

        // Upload static model buffers (Host → Device)
        let d_nominal = stream.clone_htod(&data.nominal).map_err(|e| cuda_err(e))?;
        let d_samples = stream.clone_htod(&data.samples).map_err(|e| cuda_err(e))?;
        let d_modifier_descs = stream.clone_htod(&data.modifier_descs).map_err(|e| cuda_err(e))?;
        let d_modifier_desc_offsets =
            stream.clone_htod(&data.modifier_desc_offsets).map_err(|e| cuda_err(e))?;
        let d_per_bin_param_indices =
            stream.clone_htod(&data.per_bin_param_indices).map_err(|e| cuda_err(e))?;
        let d_modifier_data = stream.clone_htod(&data.modifier_data).map_err(|e| cuda_err(e))?;
        let d_aux_poisson = stream.clone_htod(&data.aux_poisson).map_err(|e| cuda_err(e))?;
        let d_gauss_constr =
            stream.clone_htod(&data.gauss_constraints).map_err(|e| cuda_err(e))?;

        // Pre-allocate dynamic buffers (zeroed)
        let d_params = stream
            .alloc_zeros::<f64>(max_batch * data.n_params)
            .map_err(|e| cuda_err(e))?;
        let d_observed = stream
            .alloc_zeros::<f64>(max_batch * data.n_main_bins)
            .map_err(|e| cuda_err(e))?;
        let d_ln_facts = stream
            .alloc_zeros::<f64>(max_batch * data.n_main_bins)
            .map_err(|e| cuda_err(e))?;
        let d_obs_mask = stream
            .alloc_zeros::<f64>(max_batch * data.n_main_bins)
            .map_err(|e| cuda_err(e))?;
        let d_nll_out = stream.alloc_zeros::<f64>(max_batch).map_err(|e| cuda_err(e))?;
        let d_grad_out = stream
            .alloc_zeros::<f64>(max_batch * data.n_params)
            .map_err(|e| cuda_err(e))?;

        Ok(Self {
            ctx,
            stream,
            kernel_nll_grad,
            kernel_nll_only,
            d_nominal,
            d_samples,
            d_modifier_descs,
            d_modifier_desc_offsets,
            d_per_bin_param_indices,
            d_modifier_data,
            d_aux_poisson,
            d_gauss_constr,
            d_params,
            d_observed,
            d_ln_facts,
            d_obs_mask,
            d_nll_out,
            d_grad_out,
            n_params: data.n_params,
            n_main_bins: data.n_main_bins,
            n_samples: data.samples.len(),
            n_aux_poisson: data.aux_poisson.len(),
            n_gauss_constr: data.gauss_constraints.len(),
            constraint_const: data.constraint_const,
            max_batch,
        })
    }

    /// Upload toy observed data to GPU. Called once per batch of toys.
    ///
    /// `observed_flat`, `ln_facts`, `obs_mask` are all `[n_toys × n_main_bins]` row-major.
    pub fn upload_observed(
        &mut self,
        observed_flat: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
        n_toys: usize,
    ) -> ns_core::Result<()> {
        assert!(n_toys <= self.max_batch);
        let n = n_toys * self.n_main_bins;
        assert_eq!(observed_flat.len(), n);
        assert_eq!(ln_facts.len(), n);
        assert_eq!(obs_mask.len(), n);

        self.stream
            .memcpy_htod(&observed_flat[..n], &mut self.d_observed)
            .map_err(|e| cuda_err(e))?;
        self.stream
            .memcpy_htod(&ln_facts[..n], &mut self.d_ln_facts)
            .map_err(|e| cuda_err(e))?;
        self.stream
            .memcpy_htod(&obs_mask[..n], &mut self.d_obs_mask)
            .map_err(|e| cuda_err(e))?;
        Ok(())
    }

    /// Fused NLL + analytical gradient for all active toys.
    ///
    /// `params_flat` is `[n_active × n_params]` row-major.
    /// Returns `(nll[n_active], grad[n_active × n_params])`.
    ///
    /// This is the hot path — called once per optimizer iteration.
    pub fn batch_nll_grad(
        &mut self,
        params_flat: &[f64],
        n_active: usize,
    ) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        assert!(n_active <= self.max_batch);
        assert_eq!(params_flat.len(), n_active * self.n_params);

        // H→D: upload current parameters
        self.stream
            .memcpy_htod(params_flat, &mut self.d_params)
            .map_err(|e| cuda_err(e))?;

        // Zero gradient output by re-uploading zeros
        let zeros = vec![0.0f64; n_active * self.n_params];
        self.stream
            .memcpy_htod(&zeros, &mut self.d_grad_out)
            .map_err(|e| cuda_err(e))?;

        // Kernel launch: 1 block = 1 toy, threads = min(n_main_bins, 256)
        let block_size = self.n_main_bins.min(256) as u32;
        let shared_bytes =
            ((self.n_params + block_size as usize) * std::mem::size_of::<f64>()) as u32;

        let config = LaunchConfig {
            grid_dim: (n_active as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let mut builder = self.stream.launch_builder(&self.kernel_nll_grad);
        self.push_common_args(&mut builder);
        builder.arg(&mut self.d_grad_out);
        self.push_scalar_args(&mut builder);

        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch batch_nll_grad: {e}")))?;
        }

        // D→H: download results
        let mut nll_out = vec![0.0f64; n_active];
        let mut grad_out = vec![0.0f64; n_active * self.n_params];
        self.stream
            .memcpy_dtoh(&self.d_nll_out, &mut nll_out)
            .map_err(|e| cuda_err(e))?;
        self.stream
            .memcpy_dtoh(&self.d_grad_out, &mut grad_out)
            .map_err(|e| cuda_err(e))?;
        self.stream.synchronize().map_err(|e| cuda_err(e))?;

        Ok((nll_out, grad_out))
    }

    /// NLL-only batch (for line search steps where gradient is not needed).
    ///
    /// `params_flat` is `[n_active × n_params]` row-major.
    /// Returns `nll[n_active]`.
    pub fn batch_nll(
        &mut self,
        params_flat: &[f64],
        n_active: usize,
    ) -> ns_core::Result<Vec<f64>> {
        assert!(n_active <= self.max_batch);
        assert_eq!(params_flat.len(), n_active * self.n_params);

        self.stream
            .memcpy_htod(params_flat, &mut self.d_params)
            .map_err(|e| cuda_err(e))?;

        let block_size = self.n_main_bins.min(256) as u32;
        let shared_bytes =
            ((self.n_params + block_size as usize) * std::mem::size_of::<f64>()) as u32;

        let config = LaunchConfig {
            grid_dim: (n_active as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let mut builder = self.stream.launch_builder(&self.kernel_nll_only);
        self.push_common_args(&mut builder);
        self.push_scalar_args(&mut builder);

        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch batch_nll_only: {e}")))?;
        }

        let mut nll_out = vec![0.0f64; n_active];
        self.stream
            .memcpy_dtoh(&self.d_nll_out, &mut nll_out)
            .map_err(|e| cuda_err(e))?;
        self.stream.synchronize().map_err(|e| cuda_err(e))?;

        Ok(nll_out)
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of main bins.
    pub fn n_main_bins(&self) -> usize {
        self.n_main_bins
    }

    /// Maximum batch size this accelerator was allocated for.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Single-model NLL + gradient evaluation (convenience wrapper for n_active=1).
    ///
    /// Calls `batch_nll_grad` with a single set of parameters.
    /// Returns `(nll, gradient)`.
    pub fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        assert_eq!(params.len(), self.n_params);
        let (nlls, grads) = self.batch_nll_grad(params, 1)?;
        Ok((nlls[0], grads[..self.n_params].to_vec()))
    }

    /// Single-model NLL-only evaluation (convenience wrapper for n_active=1).
    pub fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64> {
        assert_eq!(params.len(), self.n_params);
        let nlls = self.batch_nll(params, 1)?;
        Ok(nlls[0])
    }

    /// Upload observed data for a single model (n_toys=1).
    ///
    /// `observed`, `ln_facts`, `obs_mask` are each `[n_main_bins]`.
    pub fn upload_observed_single(
        &mut self,
        observed: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
    ) -> ns_core::Result<()> {
        self.upload_observed(observed, ln_facts, obs_mask, 1)
    }

    // --- Private helpers for kernel argument setup ---

    fn push_common_args(&self, builder: &mut cudarc::driver::LaunchBuilder<'_>) {
        builder.arg(&self.d_params);
        builder.arg(&self.d_observed);
        builder.arg(&self.d_ln_facts);
        builder.arg(&self.d_obs_mask);
        builder.arg(&self.d_nominal);
        builder.arg(&self.d_samples);
        builder.arg(&self.d_modifier_descs);
        builder.arg(&self.d_modifier_desc_offsets);
        builder.arg(&self.d_per_bin_param_indices);
        builder.arg(&self.d_modifier_data);
        builder.arg(&self.d_aux_poisson);
        builder.arg(&self.d_gauss_constr);
        builder.arg(&self.d_nll_out);
    }

    fn push_scalar_args(&self, builder: &mut cudarc::driver::LaunchBuilder<'_>) {
        builder.arg(&(self.n_params as u32));
        builder.arg(&(self.n_main_bins as u32));
        builder.arg(&(self.n_samples as u32));
        builder.arg(&(self.n_aux_poisson as u32));
        builder.arg(&(self.n_gauss_constr as u32));
        builder.arg(&self.constraint_const);
    }
}
