//! Metal differentiable NLL accelerator for profiled fitting on Apple Silicon.
//!
//! All computation in f32. Conversion f64↔f32 happens at the API boundary.
//! No PyTorch zero-copy (Metal ≠ CUDA) — signal is uploaded from CPU.
//!
//! Mirrors `DifferentiableAccelerator` (CUDA) but for Metal.

use crate::metal_types::*;
use metal::*;
use std::mem;

/// MSL source for the differentiable kernel.
const DIFF_MSL_SRC: &str = include_str!("../kernels/differentiable_nll_grad.metal");

/// Scalar arguments passed to the kernel via `set_bytes`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DiffScalarArgs {
    n_params: u32,
    n_main_bins: u32,
    n_samples: u32,
    n_aux_poisson: u32,
    n_gauss: u32,
    n_signal_entries: u32,
    constraint_const: f32,
}

/// Signal sample location in the flat GPU buffer layout.
#[derive(Debug, Clone, Copy)]
pub struct SignalSampleInfo {
    /// Index of the signal sample in the flat `g_samples[]` array.
    pub sample_idx: u32,
    /// First main-bin index covered by this signal sample.
    pub first_bin: u32,
    /// Number of bins in the signal sample.
    pub n_bins: u32,
}

fn metal_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal differentiable: {msg}"))
}

/// Metal differentiable NLL accelerator for profiled fitting.
///
/// Signal is uploaded from CPU (no zero-copy with PyTorch).
/// All GPU computation in f32; returns f64 at the API boundary.
#[allow(dead_code)]
pub struct MetalDifferentiableAccelerator {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,

    // Static model buffers
    buf_nominal: Buffer,
    buf_samples: Buffer,
    buf_modifier_descs: Buffer,
    buf_modifier_desc_offsets: Buffer,
    buf_per_bin_param_indices: Buffer,
    buf_modifier_data: Buffer,
    buf_aux_poisson: Buffer,
    buf_gauss_constr: Buffer,

    // Dynamic buffers
    buf_params: Buffer,
    buf_observed: Buffer,
    buf_ln_facts: Buffer,
    buf_obs_mask: Buffer,
    buf_nll_out: Buffer,
    buf_grad_params_out: Buffer,

    // Signal buffers
    buf_external_signal: Buffer,
    buf_grad_signal_out: Buffer,

    // Multi-channel signal entry buffers
    buf_signal_indices: Buffer,
    buf_signal_first_bins: Buffer,
    buf_signal_n_bins_arr: Buffer,

    // Scalar args
    scalar_args: DiffScalarArgs,

    // Metadata
    n_params: usize,
    n_main_bins: usize,
    total_signal_bins: usize,

    // Pre-allocated CPU zero buffers
    zeros_params_f32: Vec<f32>,
    zeros_signal_f32: Vec<f32>,
}

impl MetalDifferentiableAccelerator {
    /// Create accelerator from Metal model data + signal entries.
    pub fn from_metal_data(
        data: &MetalModelData,
        signal_entries: &[SignalSampleInfo],
    ) -> ns_core::Result<Self> {
        assert!(!signal_entries.is_empty(), "at least one signal entry required");

        let total_signal_bins: usize = signal_entries.iter().map(|e| e.n_bins as usize).sum();

        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        // Compile/load MSL pipeline via thread-local cache.
        let pipeline = crate::metal_kernel_cache::get_pipeline(
            &device,
            "differentiable",
            DIFF_MSL_SRC,
            "differentiable_nll_grad",
        )?;

        let opts = MTLResourceOptions::StorageModeShared;

        // Upload static model buffers
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

        // Dynamic buffers (single model, not batch)
        let buf_params = device.new_buffer((data.n_params * mem::size_of::<f32>()) as u64, opts);
        let buf_observed =
            device.new_buffer((data.n_main_bins * mem::size_of::<f32>()) as u64, opts);
        let buf_ln_facts =
            device.new_buffer((data.n_main_bins * mem::size_of::<f32>()) as u64, opts);
        let buf_obs_mask =
            device.new_buffer((data.n_main_bins * mem::size_of::<f32>()) as u64, opts);
        let buf_nll_out = device.new_buffer(mem::size_of::<f32>() as u64, opts);
        let buf_grad_params_out =
            device.new_buffer((data.n_params * mem::size_of::<f32>()) as u64, opts);

        // Signal buffers
        let signal_buf_size = total_signal_bins.max(1) * mem::size_of::<f32>();
        let buf_external_signal = device.new_buffer(signal_buf_size as u64, opts);
        let buf_grad_signal_out = device.new_buffer(signal_buf_size as u64, opts);

        // Multi-channel signal entry arrays
        let indices: Vec<u32> = signal_entries.iter().map(|e| e.sample_idx).collect();
        let first_bins: Vec<u32> = signal_entries.iter().map(|e| e.first_bin).collect();
        let n_bins_arr: Vec<u32> = signal_entries.iter().map(|e| e.n_bins).collect();

        let buf_signal_indices = Self::create_buffer_from_slice(&device, &indices, opts);
        let buf_signal_first_bins = Self::create_buffer_from_slice(&device, &first_bins, opts);
        let buf_signal_n_bins_arr = Self::create_buffer_from_slice(&device, &n_bins_arr, opts);

        let scalar_args = DiffScalarArgs {
            n_params: data.n_params as u32,
            n_main_bins: data.n_main_bins as u32,
            n_samples: data.samples.len() as u32,
            n_aux_poisson: data.aux_poisson.len() as u32,
            n_gauss: data.gauss_constraints.len() as u32,
            n_signal_entries: signal_entries.len() as u32,
            constraint_const: data.constraint_const,
        };

        Ok(Self {
            device,
            queue,
            pipeline,
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
            buf_grad_params_out,
            buf_external_signal,
            buf_grad_signal_out,
            buf_signal_indices,
            buf_signal_first_bins,
            buf_signal_n_bins_arr,
            scalar_args,
            n_params: data.n_params,
            n_main_bins: data.n_main_bins,
            total_signal_bins,
            zeros_params_f32: vec![0.0f32; data.n_params],
            zeros_signal_f32: vec![0.0f32; total_signal_bins],
        })
    }

    /// Upload observed data (called once after creation).
    pub fn upload_observed(
        &mut self,
        observed: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
    ) -> ns_core::Result<()> {
        assert_eq!(observed.len(), self.n_main_bins);
        let obs_f32: Vec<f32> = observed.iter().map(|&v| v as f32).collect();
        let lnf_f32: Vec<f32> = ln_facts.iter().map(|&v| v as f32).collect();
        let mask_f32: Vec<f32> = obs_mask.iter().map(|&v| v as f32).collect();

        Self::copy_to_buffer(&self.buf_observed, &obs_f32);
        Self::copy_to_buffer(&self.buf_ln_facts, &lnf_f32);
        Self::copy_to_buffer(&self.buf_obs_mask, &mask_f32);
        Ok(())
    }

    /// Upload signal histogram (f64 → f32 at boundary).
    pub fn upload_signal(&mut self, signal: &[f64]) -> ns_core::Result<()> {
        assert_eq!(signal.len(), self.total_signal_bins);
        let signal_f32: Vec<f32> = signal.iter().map(|&v| v as f32).collect();
        Self::copy_to_buffer(&self.buf_external_signal, &signal_f32);
        Ok(())
    }

    /// Compute NLL + gradient w.r.t. nuisance parameters.
    ///
    /// Signal gradient is also computed but discarded.
    /// Returns `(nll, grad_params)` as f64.
    pub fn nll_and_grad_params(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        self.upload_params_and_zero_grads(params)?;
        self.dispatch_kernel()?;
        self.download_nll_and_grad_params()
    }

    /// Compute NLL + gradient w.r.t. both nuisance parameters and signal bins.
    ///
    /// Returns `(nll, grad_params, grad_signal)` as f64.
    pub fn nll_grad_all(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>, Vec<f64>)> {
        self.upload_params_and_zero_grads(params)?;
        self.dispatch_kernel()?;

        let (nll, grad_params) = self.download_nll_and_grad_params()?;

        // Download signal gradient (f32 → f64)
        let grad_signal =
            Self::read_buffer_f32_to_f64(&self.buf_grad_signal_out, self.total_signal_bins);

        Ok((nll, grad_params, grad_signal))
    }

    /// Total number of signal bins (across all channels).
    pub fn signal_n_bins(&self) -> usize {
        self.total_signal_bins
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    // --- Private helpers ---

    fn upload_params_and_zero_grads(&mut self, params: &[f64]) -> ns_core::Result<()> {
        assert_eq!(params.len(), self.n_params);

        // Upload params (f64 → f32)
        let params_f32: Vec<f32> = params.iter().map(|&v| v as f32).collect();
        Self::copy_to_buffer(&self.buf_params, &params_f32);

        // Zero gradient buffers
        Self::copy_to_buffer(&self.buf_grad_params_out, &self.zeros_params_f32);
        Self::copy_to_buffer(&self.buf_grad_signal_out, &self.zeros_signal_f32);

        Ok(())
    }

    fn dispatch_kernel(&self) -> ns_core::Result<()> {
        // IMPORTANT: kernel uses a power-of-two reduction over `block_size` threads.
        let n_threads = self.n_main_bins.clamp(1, 256);
        let block_size = n_threads.next_power_of_two();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set all buffers
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
        encoder.set_buffer(13, Some(&self.buf_grad_params_out), 0);
        encoder.set_buffer(14, Some(&self.buf_external_signal), 0);
        encoder.set_buffer(15, Some(&self.buf_grad_signal_out), 0);
        encoder.set_buffer(16, Some(&self.buf_signal_indices), 0);
        encoder.set_buffer(17, Some(&self.buf_signal_first_bins), 0);
        encoder.set_buffer(18, Some(&self.buf_signal_n_bins_arr), 0);

        encoder.set_bytes(
            19,
            mem::size_of::<DiffScalarArgs>() as u64,
            &self.scalar_args as *const DiffScalarArgs as *const std::ffi::c_void,
        );

        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        // Single threadgroup (1 model)
        let grid = MTLSize::new(1, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok(())
    }

    fn download_nll_and_grad_params(&self) -> ns_core::Result<(f64, Vec<f64>)> {
        let nll_f32 = Self::read_buffer_f32(&self.buf_nll_out, 1);
        let grad_params = Self::read_buffer_f32_to_f64(&self.buf_grad_params_out, self.n_params);
        Ok((nll_f32[0] as f64, grad_params))
    }

    fn create_buffer_from_slice<T>(
        device: &Device,
        data: &[T],
        opts: MTLResourceOptions,
    ) -> Buffer {
        if data.is_empty() {
            return device.new_buffer(mem::size_of::<T>().max(4) as u64, opts);
        }
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            mem::size_of_val(data) as u64,
            opts,
        )
    }

    fn copy_to_buffer<T>(buffer: &Buffer, data: &[T]) {
        if data.is_empty() {
            return;
        }
        let ptr = buffer.contents() as *mut T;
        // SAFETY: `buffer` was allocated with capacity >= `data.len() * size_of::<T>()`
        // by the caller. `contents()` returns a valid mapped pointer.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn read_buffer_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buffer.contents() as *const f32;
        // SAFETY: `buffer` capacity >= `count * 4`; pointer is valid for buffer lifetime.
        unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
    }

    fn read_buffer_f32_to_f64(buffer: &Buffer, count: usize) -> Vec<f64> {
        let ptr = buffer.contents() as *const f32;
        // SAFETY: `buffer` capacity >= `count * 4`; `count` matches kernel output size.
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.iter().map(|&v| v as f64).collect()
    }
}
