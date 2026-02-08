//! Metal batch NLL+gradient accelerator.
//!
//! Manages GPU memory, MSL compilation, and kernel dispatch for fused
//! Poisson NLL + analytical gradient computation across a batch of toy experiments.
//!
//! All computation in f32. Conversion f64↔f32 happens at the API boundary.

use crate::metal_types::*;
use crate::gpu_accel::GpuAccelerator;
use metal::*;
use std::mem;

/// MSL source compiled from `kernels/batch_nll_grad.metal`.
const MSL_SRC: &str = include_str!("../kernels/batch_nll_grad.metal");

/// Scalar arguments passed to the kernel via `set_bytes`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ScalarArgs {
    n_params: u32,
    n_main_bins: u32,
    n_samples: u32,
    n_aux_poisson: u32,
    n_gauss: u32,
    constraint_const: f32,
}

fn metal_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal: {msg}"))
}

/// Metal batch accelerator for HistFactory NLL + gradient.
///
/// Holds all GPU buffers (static model data + dynamic per-iteration data)
/// and provides `batch_nll_grad()` for the lockstep optimizer.
#[allow(dead_code)]
pub struct MetalBatchAccelerator {
    device: Device,
    queue: CommandQueue,
    pipeline_nll_grad: ComputePipelineState,
    pipeline_nll_only: ComputePipelineState,

    // --- Static model buffers (uploaded once) ---
    buf_nominal: Buffer,
    buf_samples: Buffer,
    buf_modifier_descs: Buffer,
    buf_modifier_desc_offsets: Buffer,
    buf_per_bin_param_indices: Buffer,
    buf_modifier_data: Buffer,
    buf_aux_poisson: Buffer,
    buf_gauss_constr: Buffer,

    // --- Dynamic buffers (pre-allocated for max_batch) ---
    buf_params: Buffer,
    buf_observed: Buffer,
    buf_ln_facts: Buffer,
    buf_obs_mask: Buffer,
    buf_nll_out: Buffer,
    buf_grad_out: Buffer,

    // --- Metadata ---
    n_params: usize,
    n_main_bins: usize,
    scalar_args: ScalarArgs,
    max_batch: usize,

    // --- CPU scratch buffers (preallocated for max_batch, reused per call) ---
    scratch_params_f32: Vec<f32>, // max_batch * n_params
    scratch_zeros_f32: Vec<f32>,  // max_batch * n_params
}

impl MetalBatchAccelerator {
    /// Check if Metal is available at runtime.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create accelerator from pre-serialized Metal model data.
    ///
    /// Compiles MSL at runtime, uploads all static model buffers.
    /// Pre-allocates dynamic buffers for up to `max_batch` concurrent toy experiments.
    pub fn from_metal_data(data: &MetalModelData, max_batch: usize) -> ns_core::Result<Self> {
        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        // Compile MSL source at runtime
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SRC, &options)
            .map_err(|e| metal_err(format!("MSL compile: {e}")))?;

        let fn_nll_grad = library
            .get_function("batch_nll_grad", None)
            .map_err(|e| metal_err(format!("get batch_nll_grad: {e}")))?;
        let fn_nll_only = library
            .get_function("batch_nll_only", None)
            .map_err(|e| metal_err(format!("get batch_nll_only: {e}")))?;

        let pipeline_nll_grad = device
            .new_compute_pipeline_state_with_function(&fn_nll_grad)
            .map_err(|e| metal_err(format!("pipeline batch_nll_grad: {e}")))?;
        let pipeline_nll_only = device
            .new_compute_pipeline_state_with_function(&fn_nll_only)
            .map_err(|e| metal_err(format!("pipeline batch_nll_only: {e}")))?;

        let opts = MTLResourceOptions::StorageModeShared;

        // Upload static model buffers (shared memory = zero-copy on Apple Silicon)
        let buf_nominal = Self::create_buffer_from_slice(&device, &data.nominal, opts);
        let buf_samples = Self::create_buffer_from_slice(&device, &data.samples, opts);
        let buf_modifier_descs =
            Self::create_buffer_from_slice(&device, &data.modifier_descs, opts);
        let buf_modifier_desc_offsets =
            Self::create_buffer_from_slice(&device, &data.modifier_desc_offsets, opts);
        let buf_per_bin_param_indices =
            Self::create_buffer_from_slice(&device, &data.per_bin_param_indices, opts);
        let buf_modifier_data = Self::create_buffer_from_slice(&device, &data.modifier_data, opts);
        let buf_aux_poisson = Self::create_buffer_from_slice(&device, &data.aux_poisson, opts);
        let buf_gauss_constr =
            Self::create_buffer_from_slice(&device, &data.gauss_constraints, opts);

        // Pre-allocate dynamic buffers
        let buf_params =
            device.new_buffer((max_batch * data.n_params * mem::size_of::<f32>()) as u64, opts);
        let buf_observed =
            device.new_buffer((max_batch * data.n_main_bins * mem::size_of::<f32>()) as u64, opts);
        let buf_ln_facts =
            device.new_buffer((max_batch * data.n_main_bins * mem::size_of::<f32>()) as u64, opts);
        let buf_obs_mask =
            device.new_buffer((max_batch * data.n_main_bins * mem::size_of::<f32>()) as u64, opts);
        let buf_nll_out = device.new_buffer((max_batch * mem::size_of::<f32>()) as u64, opts);
        let buf_grad_out =
            device.new_buffer((max_batch * data.n_params * mem::size_of::<f32>()) as u64, opts);

        let scalar_args = ScalarArgs {
            n_params: data.n_params as u32,
            n_main_bins: data.n_main_bins as u32,
            n_samples: data.samples.len() as u32,
            n_aux_poisson: data.aux_poisson.len() as u32,
            n_gauss: data.gauss_constraints.len() as u32,
            constraint_const: data.constraint_const,
        };

        // Pre-allocate CPU scratch buffers for f64→f32 conversion (avoids per-call allocation)
        let scratch_size = max_batch * data.n_params;
        let scratch_params_f32 = vec![0.0f32; scratch_size];
        let scratch_zeros_f32 = vec![0.0f32; scratch_size];

        Ok(Self {
            device,
            queue,
            pipeline_nll_grad,
            pipeline_nll_only,
            buf_nominal,
            buf_samples,
            buf_modifier_descs,
            buf_modifier_desc_offsets,
            buf_per_bin_param_indices,
            buf_modifier_data,
            buf_aux_poisson,
            buf_gauss_constr,
            buf_params,
            buf_observed,
            buf_ln_facts,
            buf_obs_mask,
            buf_nll_out,
            buf_grad_out,
            n_params: data.n_params,
            n_main_bins: data.n_main_bins,
            scalar_args,
            max_batch,
            scratch_params_f32,
            scratch_zeros_f32,
        })
    }

    /// Upload toy observed data to GPU. Called once per batch of toys.
    ///
    /// Input arrays are f64 (CPU) — converted to f32 for Metal.
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

        // Convert f64 → f32 and memcpy to shared buffers
        let obs_f32: Vec<f32> = observed_flat.iter().map(|&v| v as f32).collect();
        let lnf_f32: Vec<f32> = ln_facts.iter().map(|&v| v as f32).collect();
        let mask_f32: Vec<f32> = obs_mask.iter().map(|&v| v as f32).collect();

        Self::copy_to_buffer(&self.buf_observed, &obs_f32);
        Self::copy_to_buffer(&self.buf_ln_facts, &lnf_f32);
        Self::copy_to_buffer(&self.buf_obs_mask, &mask_f32);

        Ok(())
    }

    /// Fused NLL + analytical gradient for all active toys.
    ///
    /// `params_flat` is `[n_active × n_params]` row-major (f64 input).
    /// Returns `(nll[n_active], grad[n_active × n_params])` as f64.
    pub fn batch_nll_grad(
        &mut self,
        params_flat: &[f64],
        n_active: usize,
    ) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        assert!(n_active <= self.max_batch);
        assert_eq!(params_flat.len(), n_active * self.n_params);

        // Upload params (f64→f32) — reuse preallocated scratch buffer
        let count = n_active * self.n_params;
        for (i, &v) in params_flat.iter().enumerate() {
            self.scratch_params_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_params, &self.scratch_params_f32[..count]);

        // Zero gradient output — reuse preallocated scratch buffer
        // scratch_zeros_f32 is always zeroed; we just need to ensure the right length
        for i in 0..count {
            self.scratch_zeros_f32[i] = 0.0;
        }
        Self::copy_to_buffer(&self.buf_grad_out, &self.scratch_zeros_f32[..count]);

        // Dispatch kernel
        //
        // IMPORTANT: the Metal kernels use a power-of-two reduction over `block_size` threads.
        // If `block_size` is not a power of two, the reduction drops some lanes and NLL becomes
        // inconsistent with the gradient (which is accumulated via atomics).
        let n_threads = self.n_main_bins.max(1).min(256);
        let block_size = n_threads.next_power_of_two();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_nll_grad);
        self.set_common_buffers(encoder);
        encoder.set_buffer(13, Some(&self.buf_grad_out), 0);
        encoder.set_bytes(
            14,
            mem::size_of::<ScalarArgs>() as u64,
            &self.scalar_args as *const ScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        let grid = MTLSize::new(n_active as u64, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read results (f32→f64)
        let nll_out = Self::read_buffer_f32_to_f64(&self.buf_nll_out, n_active);
        let grad_out = Self::read_buffer_f32_to_f64(&self.buf_grad_out, n_active * self.n_params);

        Ok((nll_out, grad_out))
    }

    /// NLL-only batch (for line search steps where gradient is not needed).
    pub fn batch_nll(&mut self, params_flat: &[f64], n_active: usize) -> ns_core::Result<Vec<f64>> {
        assert!(n_active <= self.max_batch);
        assert_eq!(params_flat.len(), n_active * self.n_params);

        // Upload params (f64→f32) — reuse preallocated scratch buffer
        let count = n_active * self.n_params;
        for (i, &v) in params_flat.iter().enumerate() {
            self.scratch_params_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_params, &self.scratch_params_f32[..count]);

        // See `batch_nll_grad`: kernels assume power-of-two `block_size`.
        let n_threads = self.n_main_bins.max(1).min(256);
        let block_size = n_threads.next_power_of_two();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_nll_only);
        self.set_common_buffers(encoder);
        encoder.set_bytes(
            14,
            mem::size_of::<ScalarArgs>() as u64,
            &self.scalar_args as *const ScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        let grid = MTLSize::new(n_active as u64, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok(Self::read_buffer_f32_to_f64(&self.buf_nll_out, n_active))
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of main bins.
    pub fn n_main_bins(&self) -> usize {
        self.n_main_bins
    }

    /// Maximum batch size.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Single-model NLL + gradient evaluation (convenience wrapper for n_active=1).
    pub fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        assert_eq!(params.len(), self.n_params);
        let (nlls, grads) = self.batch_nll_grad(params, 1)?;
        Ok((nlls[0], grads[..self.n_params].to_vec()))
    }

    /// Single-model NLL-only evaluation.
    pub fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64> {
        assert_eq!(params.len(), self.n_params);
        let nlls = self.batch_nll(params, 1)?;
        Ok(nlls[0])
    }

    /// Upload observed data for a single model (n_toys=1).
    pub fn upload_observed_single(
        &mut self,
        observed: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
    ) -> ns_core::Result<()> {
        self.upload_observed(observed, ln_facts, obs_mask, 1)
    }

    // --- Private helpers ---

    fn set_common_buffers(&self, encoder: &ComputeCommandEncoderRef) {
        encoder.set_buffer(0, Some(&self.buf_params), 0);
        encoder.set_buffer(1, Some(&self.buf_observed), 0);
        encoder.set_buffer(2, Some(&self.buf_ln_facts), 0);
        encoder.set_buffer(3, Some(&self.buf_obs_mask), 0);
        encoder.set_buffer(4, Some(&self.buf_nominal), 0);
        encoder.set_buffer(5, Some(&self.buf_samples), 0);
        encoder.set_buffer(6, Some(&self.buf_modifier_descs), 0);
        encoder.set_buffer(7, Some(&self.buf_modifier_desc_offsets), 0);
        encoder.set_buffer(8, Some(&self.buf_per_bin_param_indices), 0);
        encoder.set_buffer(9, Some(&self.buf_modifier_data), 0);
        encoder.set_buffer(10, Some(&self.buf_aux_poisson), 0);
        encoder.set_buffer(11, Some(&self.buf_gauss_constr), 0);
        encoder.set_buffer(12, Some(&self.buf_nll_out), 0);
    }

    fn create_buffer_from_slice<T>(
        device: &Device,
        data: &[T],
        opts: MTLResourceOptions,
    ) -> Buffer {
        if data.is_empty() {
            // Metal doesn't allow zero-length buffers
            return device.new_buffer(mem::size_of::<T>().max(4) as u64, opts);
        }
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(data) as u64,
            opts,
        )
    }

    fn copy_to_buffer<T>(buffer: &Buffer, data: &[T]) {
        if data.is_empty() {
            return;
        }
        let ptr = buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn read_buffer_f32_to_f64(buffer: &Buffer, count: usize) -> Vec<f64> {
        let ptr = buffer.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.iter().map(|&v| v as f64).collect()
    }
}

impl GpuAccelerator for MetalBatchAccelerator {
    fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        MetalBatchAccelerator::single_nll_grad(self, params)
    }

    fn upload_observed_single(
        &mut self,
        observed: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
    ) -> ns_core::Result<()> {
        MetalBatchAccelerator::upload_observed_single(self, observed, ln_facts, obs_mask)
    }

    fn n_params(&self) -> usize {
        MetalBatchAccelerator::n_params(self)
    }

    fn n_main_bins(&self) -> usize {
        MetalBatchAccelerator::n_main_bins(self)
    }
}
