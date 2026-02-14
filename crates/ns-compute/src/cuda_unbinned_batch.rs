//! CUDA unbinned (event-level) batch toy NLL + gradient accelerator.
//!
//! Evaluates NLL/grad for a batch of *independent toy datasets* in one kernel launch:
//! 1 CUDA block = 1 toy dataset, threads iterate over events (grid-stride loop).
//!
//! Toy event counts may vary; datasets are stored as a flattened event array plus offsets.

use crate::unbinned_types::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/unbinned_nll_grad.cu` at build time.
///
/// Contains both single-dataset and batch toy entry points.
const PTX_SRC: &str = include_str!(env!("CUDA_UNBINNED_PTX_PATH"));

/// PTX source compiled from `kernels/unbinned_lbfgs_fit.cu` at build time.
///
/// Contains the persistent GPU-native L-BFGS optimizer kernel.
const LBFGS_PTX_SRC: &str = include_str!(env!("CUDA_LBFGS_PTX_PATH"));

const CUDA_LBFGS_STATUS_MAX_ITER: u32 = 0;
const CUDA_LBFGS_STATUS_CONVERGED: u32 = 1;
const CUDA_LBFGS_STATUS_FAILED: u32 = 2;

fn cuda_lbfgs_status_reason(status: u32) -> &'static str {
    match status {
        CUDA_LBFGS_STATUS_MAX_ITER => "MaxIterReached",
        CUDA_LBFGS_STATUS_CONVERGED => "Converged",
        CUDA_LBFGS_STATUS_FAILED => "ComputationFailed",
        _ => "UnknownKernelStatus",
    }
}

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (unbinned batch): {msg}"))
}

/// CUDA accelerator for unbinned batch toy NLL + gradient.
pub struct CudaUnbinnedBatchAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_nll_grad: CudaFunction,
    kernel_nll_only: CudaFunction,

    // --- Static buffers ---
    d_obs_flat: CudaSlice<f64>,
    d_toy_offsets: CudaSlice<u32>,
    d_obs_lo: CudaSlice<f64>,
    d_obs_hi: CudaSlice<f64>,
    d_procs: CudaSlice<GpuUnbinnedProcessDesc>,
    d_rate_mods: CudaSlice<GpuUnbinnedRateModifierDesc>,
    d_shape_pidx: CudaSlice<u32>,
    d_pdf_aux_f64: CudaSlice<f64>,
    d_gauss: CudaSlice<GpuUnbinnedGaussConstraintEntry>,

    // --- Dynamic buffers ---
    d_params_flat: CudaSlice<f64>,
    d_nll_out: CudaSlice<f64>,
    d_grad_out: CudaSlice<f64>,
    d_eval_to_toy: CudaSlice<u32>,

    // --- Metadata ---
    n_params: usize,
    n_toys: usize,
    n_procs: usize,
    total_rate_mods: usize,
    total_shape_params: usize,
    n_gauss: usize,
    constraint_const: f64,

    // --- CPU scratch (reused per call) ---
    scratch_zeros: Vec<f64>,
    scratch_eval_to_toy: Vec<u32>,
    eval_to_toy_identity: Vec<u32>,
}

impl CudaUnbinnedBatchAccelerator {
    /// Check if CUDA is available at runtime (driver loaded, GPU present).
    pub fn is_available() -> bool {
        // `cudarc` can panic when the CUDA driver shared library is missing.
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Check if a specific CUDA device is available at runtime.
    pub fn is_available_on_device(device_id: usize) -> bool {
        std::panic::catch_unwind(|| CudaContext::new(device_id).is_ok()).unwrap_or(false)
    }

    /// Create batch accelerator from static unbinned model data + toy datasets.
    ///
    /// Phase 1 batch toys assume `n_obs == 1` and store observed toy events as a flattened array
    /// plus `toy_offsets` (prefix sums).
    pub fn from_unbinned_static_and_toys(
        data: &UnbinnedGpuModelData,
        toy_offsets: &[u32],
        obs_flat: &[f64],
        n_toys: usize,
    ) -> ns_core::Result<Self> {
        Self::from_unbinned_static_and_toys_on_device(data, toy_offsets, obs_flat, n_toys, 0)
    }

    /// Same as [`Self::from_unbinned_static_and_toys`] but explicitly selects the CUDA device.
    pub fn from_unbinned_static_and_toys_on_device(
        data: &UnbinnedGpuModelData,
        toy_offsets: &[u32],
        obs_flat: &[f64],
        n_toys: usize,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        if data.n_obs != 1 {
            return Err(ns_core::Error::Validation(format!(
                "CUDA unbinned batch currently supports n_obs=1, got {}",
                data.n_obs
            )));
        }
        if data.processes.is_empty() {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires at least one process".into(),
            ));
        }
        if n_toys == 0 {
            return Err(ns_core::Error::Validation("n_toys must be > 0".into()));
        }
        if toy_offsets.len() != n_toys + 1 {
            return Err(ns_core::Error::Validation(format!(
                "toy_offsets length mismatch: expected {}, got {}",
                n_toys + 1,
                toy_offsets.len()
            )));
        }
        let last = *toy_offsets.last().unwrap_or(&0) as usize;
        if last != obs_flat.len() {
            return Err(ns_core::Error::Validation(format!(
                "toy_offsets last entry must equal obs_flat.len(): got {last}, obs_flat.len()={}",
                obs_flat.len()
            )));
        }

        let ctx = match std::panic::catch_unwind(|| CudaContext::new(device_id)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context (device {device_id}): {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let stream = ctx.default_stream();

        let d_obs_flat = stream.clone_htod(obs_flat).map_err(cuda_err)?;
        let d_toy_offsets = stream.clone_htod(toy_offsets).map_err(cuda_err)?;

        Self::build(ctx, stream, data, d_obs_flat, d_toy_offsets, n_toys)
    }

    /// Create batch accelerator from toy data already resident on the GPU device.
    ///
    /// `d_obs_flat` must have been allocated in the same CUDA context. `toy_offsets` is uploaded
    /// from the host (small buffer: `(n_toys+1) × 4` bytes). This avoids the large H2D copy of
    /// `obs_flat` when it was sampled on the same device by [`CudaUnbinnedToySampler`].
    pub fn from_unbinned_static_and_toys_device(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        data: &UnbinnedGpuModelData,
        toy_offsets: &[u32],
        d_obs_flat: CudaSlice<f64>,
        n_toys: usize,
    ) -> ns_core::Result<Self> {
        if data.n_obs != 1 {
            return Err(ns_core::Error::Validation(format!(
                "CUDA unbinned batch currently supports n_obs=1, got {}",
                data.n_obs
            )));
        }
        if data.processes.is_empty() {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires at least one process".into(),
            ));
        }
        if n_toys == 0 {
            return Err(ns_core::Error::Validation("n_toys must be > 0".into()));
        }
        if toy_offsets.len() != n_toys + 1 {
            return Err(ns_core::Error::Validation(format!(
                "toy_offsets length mismatch: expected {}, got {}",
                n_toys + 1,
                toy_offsets.len()
            )));
        }

        // H2D toy_offsets only (small: (n_toys+1) × 4 bytes).
        let d_toy_offsets = stream.clone_htod(toy_offsets).map_err(cuda_err)?;

        Self::build(ctx, stream, data, d_obs_flat, d_toy_offsets, n_toys)
    }

    /// Shared builder: loads PTX, uploads model-descriptor buffers, allocates work buffers.
    fn build(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        data: &UnbinnedGpuModelData,
        d_obs_flat: CudaSlice<f64>,
        d_toy_offsets: CudaSlice<u32>,
        n_toys: usize,
    ) -> ns_core::Result<Self> {
        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel_nll_grad = module
            .load_function("unbinned_batch_nll_grad")
            .map_err(|e| cuda_err(format!("load unbinned_batch_nll_grad: {e}")))?;
        let kernel_nll_only = module
            .load_function("unbinned_batch_nll_only")
            .map_err(|e| cuda_err(format!("load unbinned_batch_nll_only: {e}")))?;

        let (lo, hi) = data
            .obs_bounds
            .get(0)
            .copied()
            .ok_or_else(|| ns_core::Error::Validation("missing obs_bounds[0]".into()))?;

        // Upload static buffers.
        let d_obs_lo = stream.clone_htod(&[lo]).map_err(cuda_err)?;
        let d_obs_hi = stream.clone_htod(&[hi]).map_err(cuda_err)?;
        let d_procs = stream.clone_htod(&data.processes).map_err(cuda_err)?;
        let d_rate_mods = stream.clone_htod(&data.rate_modifiers).map_err(cuda_err)?;
        let d_shape_pidx = stream.clone_htod(&data.shape_param_indices).map_err(cuda_err)?;
        let d_pdf_aux_f64 = stream.clone_htod(&data.pdf_aux_f64).map_err(cuda_err)?;
        let d_gauss = stream.clone_htod(&data.gauss_constraints).map_err(cuda_err)?;

        // Pre-allocate dynamic buffers.
        let d_params_flat = stream.alloc_zeros::<f64>(n_toys * data.n_params).map_err(cuda_err)?;
        let d_nll_out = stream.alloc_zeros::<f64>(n_toys).map_err(cuda_err)?;
        let d_grad_out = stream.alloc_zeros::<f64>(n_toys * data.n_params).map_err(cuda_err)?;
        let mut d_eval_to_toy = stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;
        let eval_to_toy_identity: Vec<u32> = (0..n_toys as u32).collect();
        stream.memcpy_htod(&eval_to_toy_identity, &mut d_eval_to_toy).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel_nll_grad,
            kernel_nll_only,
            d_obs_flat,
            d_toy_offsets,
            d_obs_lo,
            d_obs_hi,
            d_procs,
            d_rate_mods,
            d_shape_pidx,
            d_pdf_aux_f64,
            d_gauss,
            d_params_flat,
            d_nll_out,
            d_grad_out,
            d_eval_to_toy,
            n_params: data.n_params,
            n_toys,
            n_procs: data.processes.len(),
            total_rate_mods: data.rate_modifiers.len(),
            total_shape_params: data.shape_param_indices.len(),
            n_gauss: data.gauss_constraints.len(),
            constraint_const: data.constraint_const,
            scratch_zeros: vec![0.0; n_toys * data.n_params],
            scratch_eval_to_toy: vec![0u32; n_toys],
            eval_to_toy_identity,
        })
    }

    fn launch_config(&self, n_eval: usize) -> LaunchConfig {
        // Use a fixed, power-of-two block size; toys have variable event counts.
        let block_size = 256u32;
        let shared_bytes =
            ((self.n_params + block_size as usize) * std::mem::size_of::<f64>()) as u32;
        LaunchConfig {
            grid_dim: (n_eval as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        }
    }

    fn upload_eval_to_toy_map(&mut self, active_toys: &[usize]) -> ns_core::Result<()> {
        for (slot, &toy_idx) in active_toys.iter().enumerate() {
            if toy_idx >= self.n_toys {
                return Err(ns_core::Error::Validation(format!(
                    "active toy index out of range at slot {slot}: toy_idx={toy_idx}, n_toys={}",
                    self.n_toys
                )));
            }
            self.scratch_eval_to_toy[slot] = toy_idx as u32;
        }
        self.stream
            .memcpy_htod(&self.scratch_eval_to_toy[..active_toys.len()], &mut self.d_eval_to_toy)
            .map_err(cuda_err)?;
        Ok(())
    }

    fn eval_batch_nll_grad(
        &mut self,
        params_flat: &[f64],
        n_eval: usize,
    ) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params_flat.len() != n_eval * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch: expected {}, got {}",
                n_eval * self.n_params,
                params_flat.len()
            )));
        }
        if n_eval == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        self.stream.memcpy_htod(params_flat, &mut self.d_params_flat).map_err(cuda_err)?;
        self.stream
            .memcpy_htod(&self.scratch_zeros[..n_eval * self.n_params], &mut self.d_grad_out)
            .map_err(cuda_err)?;

        let config = self.launch_config(n_eval);

        let np = self.n_params as u32;
        let nprocs = self.n_procs as u32;
        let tmods = self.total_rate_mods as u32;
        let tshape = self.total_shape_params as u32;
        let ng = self.n_gauss as u32;
        let cc = self.constraint_const;
        let ntoys_total = self.n_toys as u32;
        let neval = n_eval as u32;

        let mut builder = self.stream.launch_builder(&self.kernel_nll_grad);
        builder.arg(&self.d_params_flat);
        builder.arg(&self.d_obs_flat);
        builder.arg(&self.d_toy_offsets);
        builder.arg(&self.d_eval_to_toy);
        builder.arg(&self.d_obs_lo);
        builder.arg(&self.d_obs_hi);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&self.d_shape_pidx);
        builder.arg(&self.d_pdf_aux_f64);
        builder.arg(&self.d_gauss);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&mut self.d_grad_out);
        builder.arg(&np);
        builder.arg(&nprocs);
        builder.arg(&tmods);
        builder.arg(&tshape);
        builder.arg(&ng);
        builder.arg(&cc);
        builder.arg(&ntoys_total);
        builder.arg(&neval);

        // SAFETY: All device pointers are valid CudaSlice allocations owned by `self`,
        // scalar args match the compiled kernel signature, launch config is within limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_batch_nll_grad: {e}")))?;
        }

        let mut nll_out = vec![0.0f64; self.n_toys];
        let mut grad_out = vec![0.0f64; self.n_toys * self.n_params];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_grad_out, &mut grad_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        nll_out.truncate(n_eval);
        grad_out.truncate(n_eval * self.n_params);
        Ok((nll_out, grad_out))
    }

    fn eval_batch_nll(&mut self, params_flat: &[f64], n_eval: usize) -> ns_core::Result<Vec<f64>> {
        if params_flat.len() != n_eval * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch: expected {}, got {}",
                n_eval * self.n_params,
                params_flat.len()
            )));
        }
        if n_eval == 0 {
            return Ok(Vec::new());
        }

        self.stream.memcpy_htod(params_flat, &mut self.d_params_flat).map_err(cuda_err)?;

        let config = self.launch_config(n_eval);

        let np = self.n_params as u32;
        let nprocs = self.n_procs as u32;
        let tmods = self.total_rate_mods as u32;
        let tshape = self.total_shape_params as u32;
        let ng = self.n_gauss as u32;
        let cc = self.constraint_const;
        let ntoys_total = self.n_toys as u32;
        let neval = n_eval as u32;

        let mut builder = self.stream.launch_builder(&self.kernel_nll_only);
        builder.arg(&self.d_params_flat);
        builder.arg(&self.d_obs_flat);
        builder.arg(&self.d_toy_offsets);
        builder.arg(&self.d_eval_to_toy);
        builder.arg(&self.d_obs_lo);
        builder.arg(&self.d_obs_hi);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&self.d_shape_pidx);
        builder.arg(&self.d_pdf_aux_f64);
        builder.arg(&self.d_gauss);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&np);
        builder.arg(&nprocs);
        builder.arg(&tmods);
        builder.arg(&tshape);
        builder.arg(&ng);
        builder.arg(&cc);
        builder.arg(&ntoys_total);
        builder.arg(&neval);

        // SAFETY: Same invariants as batch_nll_grad launch — valid device pointers,
        // matching scalar args, launch config within hardware limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_batch_nll_only: {e}")))?;
        }

        let mut nll_out = vec![0.0f64; self.n_toys];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        nll_out.truncate(n_eval);
        Ok(nll_out)
    }

    /// Fused NLL + analytical gradient for all toys.
    ///
    /// `params_flat` is `[n_toys × n_params]` row-major.
    pub fn batch_nll_grad(&mut self, params_flat: &[f64]) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params_flat.len() != self.n_toys * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch: expected {}, got {}",
                self.n_toys * self.n_params,
                params_flat.len()
            )));
        }
        self.stream
            .memcpy_htod(&self.eval_to_toy_identity, &mut self.d_eval_to_toy)
            .map_err(cuda_err)?;
        self.eval_batch_nll_grad(params_flat, self.n_toys)
    }

    /// Fused NLL + analytical gradient for active toy subset.
    ///
    /// `params_flat` is `[n_active × n_params]` in active toy order.
    pub fn batch_nll_grad_active(
        &mut self,
        params_flat: &[f64],
        active_toys: &[usize],
    ) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params_flat.len() != active_toys.len() * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch for active toys: expected {}, got {}",
                active_toys.len() * self.n_params,
                params_flat.len()
            )));
        }
        if active_toys.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        self.upload_eval_to_toy_map(active_toys)?;
        self.eval_batch_nll_grad(params_flat, active_toys.len())
    }

    /// NLL-only evaluation for all toys.
    ///
    /// `params_flat` is `[n_toys × n_params]` row-major.
    pub fn batch_nll(&mut self, params_flat: &[f64]) -> ns_core::Result<Vec<f64>> {
        if params_flat.len() != self.n_toys * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch: expected {}, got {}",
                self.n_toys * self.n_params,
                params_flat.len()
            )));
        }
        self.stream
            .memcpy_htod(&self.eval_to_toy_identity, &mut self.d_eval_to_toy)
            .map_err(cuda_err)?;
        self.eval_batch_nll(params_flat, self.n_toys)
    }

    /// NLL-only evaluation for active toy subset.
    ///
    /// `params_flat` is `[n_active × n_params]` in active toy order.
    pub fn batch_nll_active(
        &mut self,
        params_flat: &[f64],
        active_toys: &[usize],
    ) -> ns_core::Result<Vec<f64>> {
        if params_flat.len() != active_toys.len() * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch for active toys: expected {}, got {}",
                active_toys.len() * self.n_params,
                params_flat.len()
            )));
        }
        if active_toys.is_empty() {
            return Ok(Vec::new());
        }
        self.upload_eval_to_toy_map(active_toys)?;
        self.eval_batch_nll(params_flat, active_toys.len())
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of toys in this batch.
    pub fn n_toys(&self) -> usize {
        self.n_toys
    }

    /// Number of processes in this channel.
    pub fn n_procs(&self) -> usize {
        self.n_procs
    }

    /// Total rate modifiers in this channel.
    pub fn total_rate_mods(&self) -> usize {
        self.total_rate_mods
    }

    /// Total shape param indices in this channel.
    pub fn total_shape_params(&self) -> usize {
        self.total_shape_params
    }

    /// Number of Gaussian constraints.
    pub fn n_gauss(&self) -> usize {
        self.n_gauss
    }

    /// Constraint constant.
    pub fn constraint_const(&self) -> f64 {
        self.constraint_const
    }

    /// Download obs_lo and obs_hi from device.
    pub fn download_obs_bounds(&self) -> ns_core::Result<(f64, f64)> {
        let mut lo = vec![0.0f64; 1];
        let mut hi = vec![0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_obs_lo, &mut lo).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_obs_hi, &mut hi).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok((lo[0], hi[0]))
    }

    /// Download static model buffers from device for multi-channel concatenation.
    pub fn download_static_buffers(
        &self,
    ) -> ns_core::Result<(
        Vec<f64>,                             // obs_flat
        Vec<u32>,                             // toy_offsets
        Vec<GpuUnbinnedProcessDesc>,          // procs
        Vec<GpuUnbinnedRateModifierDesc>,     // rate_mods
        Vec<u32>,                             // shape_pidx
        Vec<f64>,                             // pdf_aux_f64
        Vec<GpuUnbinnedGaussConstraintEntry>, // gauss
    )> {
        let obs_len = self.d_obs_flat.len();
        let offsets_len = self.d_toy_offsets.len();
        let procs_len = self.d_procs.len();
        let rmods_len = self.d_rate_mods.len();
        let shape_len = self.d_shape_pidx.len();
        let aux_len = self.d_pdf_aux_f64.len();
        let gauss_len = self.d_gauss.len();

        let mut obs = vec![0.0f64; obs_len];
        let mut offsets = vec![0u32; offsets_len];
        let mut procs = vec![
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: 0,
                yield_kind: 0,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 0,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            };
            procs_len
        ];
        let mut rmods = vec![
            GpuUnbinnedRateModifierDesc {
                kind: 0,
                alpha_param_idx: 0,
                interp_code: 0,
                _pad: 0,
                lo: 0.0,
                hi: 0.0,
            };
            rmods_len
        ];
        let mut shape = vec![0u32; shape_len];
        let mut aux = vec![0.0f64; aux_len];
        let mut gauss = vec![
            GpuUnbinnedGaussConstraintEntry {
                center: 0.0,
                inv_width: 0.0,
                param_idx: 0,
                _pad: 0,
            };
            gauss_len
        ];

        self.stream.memcpy_dtoh(&self.d_obs_flat, &mut obs).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_toy_offsets, &mut offsets).map_err(cuda_err)?;
        if procs_len > 0 {
            self.stream.memcpy_dtoh(&self.d_procs, &mut procs).map_err(cuda_err)?;
        }
        if rmods_len > 0 {
            self.stream.memcpy_dtoh(&self.d_rate_mods, &mut rmods).map_err(cuda_err)?;
        }
        if shape_len > 0 {
            self.stream.memcpy_dtoh(&self.d_shape_pidx, &mut shape).map_err(cuda_err)?;
        }
        if aux_len > 0 {
            self.stream.memcpy_dtoh(&self.d_pdf_aux_f64, &mut aux).map_err(cuda_err)?;
        }
        if gauss_len > 0 {
            self.stream.memcpy_dtoh(&self.d_gauss, &mut gauss).map_err(cuda_err)?;
        }
        self.stream.synchronize().map_err(cuda_err)?;

        Ok((obs, offsets, procs, rmods, shape, aux, gauss))
    }

    /// Run the full L-BFGS optimization on device (GPU-native, no host-device roundtrips).
    ///
    /// Single kernel launch: 1 CUDA block = 1 toy = 1 complete L-BFGS-B optimization.
    /// All iterations, line search, and convergence checks run entirely on the GPU.
    ///
    /// Returns `(results, final_params_flat)` where `final_params_flat` is `[n_toys × n_params]`.
    pub fn batch_fit_on_device(
        &self,
        init_params: &[f64],
        bounds: &[(f64, f64)],
        max_iter: u32,
        lbfgs_m: u32,
        tol: f64,
        max_backtracks: u32,
    ) -> ns_core::Result<Vec<ns_core::Result<ns_core::FitResult>>> {
        let n_params = self.n_params;
        let n_toys = self.n_toys;

        if init_params.len() != n_params {
            return Err(ns_core::Error::Validation(format!(
                "init_params length mismatch: expected {n_params}, got {}",
                init_params.len()
            )));
        }
        if bounds.len() != n_params {
            return Err(ns_core::Error::Validation(format!(
                "bounds length mismatch: expected {n_params}, got {}",
                bounds.len()
            )));
        }
        if lbfgs_m > 16 {
            return Err(ns_core::Error::Validation(format!(
                "lbfgs_m must be <= 16 (MAX_LBFGS_M), got {lbfgs_m}"
            )));
        }

        let ptx = Ptx::from_src(LBFGS_PTX_SRC);
        let module =
            self.ctx.load_module(ptx).map_err(|e| cuda_err(format!("load L-BFGS PTX: {e}")))?;
        let kernel = module
            .load_function("unbinned_batch_lbfgs_fit")
            .map_err(|e| cuda_err(format!("load unbinned_batch_lbfgs_fit: {e}")))?;

        let m = lbfgs_m as usize;

        // Replicate init_params for all toys → g_x
        let mut x_flat = Vec::with_capacity(n_toys * n_params);
        for _ in 0..n_toys {
            x_flat.extend_from_slice(init_params);
        }
        let mut d_x = self.stream.clone_htod(&x_flat).map_err(cuda_err)?;

        // Allocate L-BFGS state buffers (all zero-initialized)
        let mut d_prev_x = self.stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;
        let mut d_prev_grad =
            self.stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;
        let mut d_s_hist =
            self.stream.alloc_zeros::<f64>(n_toys * m * n_params).map_err(cuda_err)?;
        let mut d_y_hist =
            self.stream.alloc_zeros::<f64>(n_toys * m * n_params).map_err(cuda_err)?;
        let mut d_rho_hist = self.stream.alloc_zeros::<f64>(n_toys * m).map_err(cuda_err)?;
        let mut d_grad = self.stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;
        let mut d_direction =
            self.stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;

        // Bounds
        let bounds_lo: Vec<f64> = bounds.iter().map(|(lo, _)| *lo).collect();
        let bounds_hi: Vec<f64> = bounds.iter().map(|(_, hi)| *hi).collect();
        let d_bounds_lo = self.stream.clone_htod(&bounds_lo).map_err(cuda_err)?;
        let d_bounds_hi = self.stream.clone_htod(&bounds_hi).map_err(cuda_err)?;

        // Output buffers
        let mut d_nll_out = self.stream.alloc_zeros::<f64>(n_toys).map_err(cuda_err)?;
        let mut d_status = self.stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;
        let mut d_iters = self.stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;
        let mut d_line_search_exhaust = self.stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;

        // Launch config: 1 block = 1 toy, 256 threads
        let block_size = 256u32;
        // Shared memory: params[n_params] + scratch[block_size] + 1 extra double for flags
        let shared_bytes =
            ((n_params + block_size as usize + 1) * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (n_toys as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        // Build single-channel descriptor (obs_lo/obs_hi read from device to host)
        let mut obs_lo_h = vec![0.0f64; 1];
        let mut obs_hi_h = vec![0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_obs_lo, &mut obs_lo_h).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_obs_hi, &mut obs_hi_h).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        let channel_desc = GpuChannelDesc {
            obs_base: 0,
            toy_offsets_base: 0,
            proc_base: 0,
            n_procs: self.n_procs as u32,
            rate_mod_base: 0,
            total_rate_mods: self.total_rate_mods as u32,
            shape_base: 0,
            total_shape_params: self.total_shape_params as u32,
            pdf_aux_base: 0,
            gauss_base: 0,
            n_gauss: self.n_gauss as u32,
            _pad: 0,
            obs_lo: obs_lo_h[0],
            obs_hi: obs_hi_h[0],
            constraint_const: self.constraint_const,
        };
        let d_channels = self.stream.clone_htod(&[channel_desc]).map_err(cuda_err)?;

        let np = n_params as u32;
        let nch = 1u32;
        let ntoys = n_toys as u32;

        let mut builder = self.stream.launch_builder(&kernel);
        builder.arg(&mut d_x);
        builder.arg(&mut d_prev_x);
        builder.arg(&mut d_prev_grad);
        builder.arg(&mut d_s_hist);
        builder.arg(&mut d_y_hist);
        builder.arg(&mut d_rho_hist);
        builder.arg(&mut d_grad);
        builder.arg(&mut d_direction);
        builder.arg(&d_bounds_lo);
        builder.arg(&d_bounds_hi);
        builder.arg(&self.d_obs_flat);
        builder.arg(&self.d_toy_offsets);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&self.d_shape_pidx);
        builder.arg(&self.d_pdf_aux_f64);
        builder.arg(&self.d_gauss);
        builder.arg(&d_channels);
        builder.arg(&mut d_nll_out);
        builder.arg(&mut d_status);
        builder.arg(&mut d_iters);
        builder.arg(&mut d_line_search_exhaust);
        builder.arg(&np);
        builder.arg(&nch);
        builder.arg(&ntoys);
        builder.arg(&max_iter);
        builder.arg(&lbfgs_m);
        builder.arg(&tol);
        builder.arg(&max_backtracks);

        // SAFETY: All device pointers are valid CudaSlice allocations,
        // scalar args match the compiled kernel signature.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_batch_lbfgs_fit: {e}")))?;
        }

        // Download results
        let mut x_out = vec![0.0f64; n_toys * n_params];
        let mut nll_out = vec![0.0f64; n_toys];
        let mut status_out = vec![0u32; n_toys];
        let mut iters_out = vec![0u32; n_toys];
        let mut line_search_exhaust_out = vec![0u32; n_toys];

        self.stream.memcpy_dtoh(&d_x, &mut x_out).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&d_status, &mut status_out).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&d_iters, &mut iters_out).map_err(cuda_err)?;
        self.stream
            .memcpy_dtoh(&d_line_search_exhaust, &mut line_search_exhaust_out)
            .map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        // Convert to FitResults
        let results: Vec<ns_core::Result<ns_core::FitResult>> = (0..n_toys)
            .map(|t| {
                let params = x_out[t * n_params..(t + 1) * n_params].to_vec();
                let nll = nll_out[t];
                let status = status_out[t];
                let iters = iters_out[t] as usize;
                let ls_exhaust = line_search_exhaust_out[t];
                let converged = status == CUDA_LBFGS_STATUS_CONVERGED;
                let failed = status == CUDA_LBFGS_STATUS_FAILED
                    || (status != CUDA_LBFGS_STATUS_MAX_ITER && !converged);

                if failed {
                    Err(ns_core::Error::Computation(format!(
                        "GPU L-BFGS optimizer failed (status={}, reason={})",
                        status,
                        cuda_lbfgs_status_reason(status)
                    )))
                } else {
                    let mut fit = ns_core::FitResult::new(
                        params,
                        vec![0.0; n_params],
                        nll,
                        converged,
                        iters,
                        iters + 1,
                        iters,
                    )
                    .with_diagnostics(
                        cuda_lbfgs_status_reason(status).to_string(),
                        f64::NAN,
                        f64::NAN,
                        0,
                    );
                    fit.warnings.push(format!("cuda_status={status}"));
                    fit.warnings.push("retry_attempts_used=0".to_string());
                    fit.warnings.push(format!("line_search_exhaustions={ls_exhaust}"));
                    Ok(fit)
                }
            })
            .collect();

        Ok(results)
    }
}

/// Run the full L-BFGS optimization on device for a multi-channel unbinned model.
///
/// Concatenates static data from all per-channel accelerators into flat arrays,
/// builds per-channel descriptors with offsets, and launches the persistent
/// mega-kernel. Gaussian constraints and `constraint_const` are taken from
/// channel 0 only to avoid double-counting.
///
/// All accelerators must have the same `n_params` and `n_toys`.
pub fn batch_fit_multi_channel_on_device(
    accels: &[CudaUnbinnedBatchAccelerator],
    init_params: &[f64],
    bounds: &[(f64, f64)],
    max_iter: u32,
    lbfgs_m: u32,
    tol: f64,
    max_backtracks: u32,
) -> ns_core::Result<Vec<ns_core::Result<ns_core::FitResult>>> {
    if accels.is_empty() {
        return Err(ns_core::Error::Validation("accels must be non-empty".into()));
    }
    let n_channels = accels.len();
    let n_params = accels[0].n_params();
    let n_toys = accels[0].n_toys();
    for (i, a) in accels.iter().enumerate().skip(1) {
        if a.n_params() != n_params || a.n_toys() != n_toys {
            return Err(ns_core::Error::Validation(format!(
                "accelerator shape mismatch at channel {i}: expected (n_params={n_params}, n_toys={n_toys}), got ({}, {})",
                a.n_params(),
                a.n_toys()
            )));
        }
    }
    if init_params.len() != n_params {
        return Err(ns_core::Error::Validation(format!(
            "init_params length {}, expected {n_params}",
            init_params.len()
        )));
    }
    if bounds.len() != n_params {
        return Err(ns_core::Error::Validation(format!(
            "bounds length {}, expected {n_params}",
            bounds.len()
        )));
    }
    if lbfgs_m > 16 {
        return Err(ns_core::Error::Validation(format!("lbfgs_m must be <= 16, got {lbfgs_m}")));
    }

    // Use first accelerator's context/stream
    let ctx = &accels[0].ctx;
    let stream = &accels[0].stream;

    // Download static data from each channel and concatenate
    let mut all_obs: Vec<f64> = Vec::new();
    let mut all_offsets: Vec<u32> = Vec::new();
    let mut all_procs: Vec<GpuUnbinnedProcessDesc> = Vec::new();
    let mut all_rmods: Vec<GpuUnbinnedRateModifierDesc> = Vec::new();
    let mut all_shape: Vec<u32> = Vec::new();
    let mut all_aux: Vec<f64> = Vec::new();
    let mut all_gauss: Vec<GpuUnbinnedGaussConstraintEntry> = Vec::new();
    let mut channel_descs: Vec<GpuChannelDesc> = Vec::with_capacity(n_channels);

    for (ch_idx, accel) in accels.iter().enumerate() {
        let (obs, offsets, mut procs, rmods, shape, aux, gauss) =
            accel.download_static_buffers()?;
        let (obs_lo, obs_hi) = accel.download_obs_bounds()?;

        let obs_base = all_obs.len() as u32;
        let toy_offsets_base = all_offsets.len() as u32;
        let proc_base = all_procs.len() as u32;
        let rate_mod_base = all_rmods.len() as u32;
        let shape_base = all_shape.len() as u32;
        let pdf_aux_base = all_aux.len() as u32;
        let gauss_base = all_gauss.len() as u32;

        // Note: process descriptor internal offsets (rate_mod_offset,
        // shape_param_offset, pdf_aux_offset) remain channel-relative because
        // the kernel passes channel-offset pointers (g_rate_mods + cd.rate_mod_base,
        // g_shape_pidx + cd.shape_base, g_pdf_aux_f64 + cd.pdf_aux_base) to
        // compute_nll_and_grad, so no adjustment is needed here.

        // Gaussian constraints only on channel 0 to avoid double-counting
        let (ch_n_gauss, ch_constraint_const) = if ch_idx == 0 {
            (accel.n_gauss() as u32, accel.constraint_const())
        } else {
            (0u32, 0.0)
        };

        channel_descs.push(GpuChannelDesc {
            obs_base,
            toy_offsets_base,
            proc_base,
            n_procs: accel.n_procs() as u32,
            rate_mod_base,
            total_rate_mods: accel.total_rate_mods() as u32,
            shape_base,
            total_shape_params: accel.total_shape_params() as u32,
            pdf_aux_base,
            gauss_base,
            n_gauss: ch_n_gauss,
            _pad: 0,
            obs_lo,
            obs_hi,
            constraint_const: ch_constraint_const,
        });

        all_obs.extend_from_slice(&obs);
        all_offsets.extend_from_slice(&offsets);
        all_procs.extend(procs);
        all_rmods.extend(rmods);
        all_shape.extend_from_slice(&shape);
        all_aux.extend_from_slice(&aux);
        if ch_idx == 0 {
            all_gauss.extend(gauss);
        }
    }

    // Upload concatenated data to device
    let d_obs_flat = stream.clone_htod(&all_obs).map_err(cuda_err)?;
    let d_toy_offsets = stream.clone_htod(&all_offsets).map_err(cuda_err)?;
    let d_procs = stream.clone_htod(&all_procs).map_err(cuda_err)?;
    let d_rate_mods = if all_rmods.is_empty() {
        stream.alloc_zeros::<GpuUnbinnedRateModifierDesc>(1).map_err(cuda_err)?
    } else {
        stream.clone_htod(&all_rmods).map_err(cuda_err)?
    };
    let d_shape_pidx = if all_shape.is_empty() {
        stream.alloc_zeros::<u32>(1).map_err(cuda_err)?
    } else {
        stream.clone_htod(&all_shape).map_err(cuda_err)?
    };
    let d_pdf_aux = if all_aux.is_empty() {
        stream.alloc_zeros::<f64>(1).map_err(cuda_err)?
    } else {
        stream.clone_htod(&all_aux).map_err(cuda_err)?
    };
    let d_gauss = if all_gauss.is_empty() {
        stream.alloc_zeros::<GpuUnbinnedGaussConstraintEntry>(1).map_err(cuda_err)?
    } else {
        stream.clone_htod(&all_gauss).map_err(cuda_err)?
    };
    let d_channels = stream.clone_htod(&channel_descs).map_err(cuda_err)?;

    // Load L-BFGS kernel
    let ptx = Ptx::from_src(LBFGS_PTX_SRC);
    let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load L-BFGS PTX: {e}")))?;
    let kernel = module
        .load_function("unbinned_batch_lbfgs_fit")
        .map_err(|e| cuda_err(format!("load unbinned_batch_lbfgs_fit: {e}")))?;

    let m = lbfgs_m as usize;

    // Replicate init_params for all toys
    let mut x_flat = Vec::with_capacity(n_toys * n_params);
    for _ in 0..n_toys {
        x_flat.extend_from_slice(init_params);
    }
    let mut d_x = stream.clone_htod(&x_flat).map_err(cuda_err)?;

    // Allocate L-BFGS state buffers
    let mut d_prev_x = stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;
    let mut d_prev_grad = stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;
    let mut d_s_hist = stream.alloc_zeros::<f64>(n_toys * m * n_params).map_err(cuda_err)?;
    let mut d_y_hist = stream.alloc_zeros::<f64>(n_toys * m * n_params).map_err(cuda_err)?;
    let mut d_rho_hist = stream.alloc_zeros::<f64>(n_toys * m).map_err(cuda_err)?;
    let mut d_grad = stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;
    let mut d_direction = stream.alloc_zeros::<f64>(n_toys * n_params).map_err(cuda_err)?;

    // Bounds
    let bounds_lo: Vec<f64> = bounds.iter().map(|(lo, _)| *lo).collect();
    let bounds_hi: Vec<f64> = bounds.iter().map(|(_, hi)| *hi).collect();
    let d_bounds_lo = stream.clone_htod(&bounds_lo).map_err(cuda_err)?;
    let d_bounds_hi = stream.clone_htod(&bounds_hi).map_err(cuda_err)?;

    // Output buffers
    let mut d_nll_out = stream.alloc_zeros::<f64>(n_toys).map_err(cuda_err)?;
    let mut d_status = stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;
    let mut d_iters = stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;
    let mut d_line_search_exhaust = stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;

    // Launch config
    let block_size = 256u32;
    let shared_bytes = ((n_params + block_size as usize + 1) * std::mem::size_of::<f64>()) as u32;
    let config = LaunchConfig {
        grid_dim: (n_toys as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let np = n_params as u32;
    let nch = n_channels as u32;
    let ntoys = n_toys as u32;

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut d_x);
    builder.arg(&mut d_prev_x);
    builder.arg(&mut d_prev_grad);
    builder.arg(&mut d_s_hist);
    builder.arg(&mut d_y_hist);
    builder.arg(&mut d_rho_hist);
    builder.arg(&mut d_grad);
    builder.arg(&mut d_direction);
    builder.arg(&d_bounds_lo);
    builder.arg(&d_bounds_hi);
    builder.arg(&d_obs_flat);
    builder.arg(&d_toy_offsets);
    builder.arg(&d_procs);
    builder.arg(&d_rate_mods);
    builder.arg(&d_shape_pidx);
    builder.arg(&d_pdf_aux);
    builder.arg(&d_gauss);
    builder.arg(&d_channels);
    builder.arg(&mut d_nll_out);
    builder.arg(&mut d_status);
    builder.arg(&mut d_iters);
    builder.arg(&mut d_line_search_exhaust);
    builder.arg(&np);
    builder.arg(&nch);
    builder.arg(&ntoys);
    builder.arg(&max_iter);
    builder.arg(&lbfgs_m);
    builder.arg(&tol);
    builder.arg(&max_backtracks);

    unsafe {
        builder.launch(config).map_err(|e| {
            cuda_err(format!("launch unbinned_batch_lbfgs_fit (multi-channel): {e}"))
        })?;
    }

    // Download results
    let mut x_out = vec![0.0f64; n_toys * n_params];
    let mut nll_out = vec![0.0f64; n_toys];
    let mut status_out = vec![0u32; n_toys];
    let mut iters_out = vec![0u32; n_toys];
    let mut line_search_exhaust_out = vec![0u32; n_toys];

    stream.memcpy_dtoh(&d_x, &mut x_out).map_err(cuda_err)?;
    stream.memcpy_dtoh(&d_nll_out, &mut nll_out).map_err(cuda_err)?;
    stream.memcpy_dtoh(&d_status, &mut status_out).map_err(cuda_err)?;
    stream.memcpy_dtoh(&d_iters, &mut iters_out).map_err(cuda_err)?;
    stream.memcpy_dtoh(&d_line_search_exhaust, &mut line_search_exhaust_out).map_err(cuda_err)?;
    stream.synchronize().map_err(cuda_err)?;

    let results: Vec<ns_core::Result<ns_core::FitResult>> = (0..n_toys)
        .map(|t| {
            let params = x_out[t * n_params..(t + 1) * n_params].to_vec();
            let nll = nll_out[t];
            let status = status_out[t];
            let iters = iters_out[t] as usize;
            let ls_exhaust = line_search_exhaust_out[t];
            let converged = status == CUDA_LBFGS_STATUS_CONVERGED;
            let failed = status == CUDA_LBFGS_STATUS_FAILED
                || (status != CUDA_LBFGS_STATUS_MAX_ITER && !converged);

            if failed {
                Err(ns_core::Error::Computation(format!(
                    "GPU L-BFGS optimizer failed (status={}, reason={})",
                    status,
                    cuda_lbfgs_status_reason(status)
                )))
            } else {
                let mut fit = ns_core::FitResult::new(
                    params,
                    vec![0.0; n_params],
                    nll,
                    converged,
                    iters,
                    iters + 1,
                    iters,
                )
                .with_diagnostics(
                    cuda_lbfgs_status_reason(status).to_string(),
                    f64::NAN,
                    f64::NAN,
                    0,
                );
                fit.warnings.push(format!("cuda_status={status}"));
                fit.warnings.push("retry_attempts_used=0".to_string());
                fit.warnings.push(format!("line_search_exhaustions={ls_exhaust}"));
                Ok(fit)
            }
        })
        .collect();

    Ok(results)
}
