//! Metal unbinned (event-level) batch toy NLL + gradient accelerator.
//!
//! Evaluates NLL/grad for a batch of independent toy datasets:
//! 1 threadgroup = 1 toy dataset, threads iterate over events (grid-stride loop).
//!
//! All computation in f32. Conversion f64↔f32 happens at the API boundary.

use crate::unbinned_types::*;
use metal::*;
use std::mem;

/// MSL source compiled from `kernels/unbinned_nll_grad.metal`.
///
/// Contains both single-dataset and batch toy entry points.
const MSL_SRC: &str = include_str!("../kernels/unbinned_nll_grad.metal");

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BatchScalarArgs {
    n_params: u32,
    n_procs: u32,
    total_rate_mods: u32,
    total_shape_params: u32,
    n_gauss: u32,
    n_toys: u32,
    constraint_const: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalUnbinnedProcessDesc {
    base_yield: f32,
    pdf_kind: u32,
    yield_kind: u32,
    obs_index: u32,
    shape_param_offset: u32,
    n_shape_params: u32,
    yield_param_idx: u32,
    rate_mod_offset: u32,
    n_rate_mods: u32,
    pdf_aux_offset: u32,
    pdf_aux_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalUnbinnedGaussConstraintEntry {
    center: f32,
    inv_width: f32,
    param_idx: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalUnbinnedRateModifierDesc {
    kind: u32,
    alpha_param_idx: u32,
    interp_code: u32,
    _pad: u32,
    lo: f32,
    hi: f32,
}

fn metal_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal (unbinned batch): {msg}"))
}

/// Metal accelerator for unbinned batch toy NLL + gradient.
#[allow(dead_code)]
pub struct MetalUnbinnedBatchAccelerator {
    device: Device,
    queue: CommandQueue,
    pipeline_nll_grad: ComputePipelineState,
    pipeline_nll_only: ComputePipelineState,

    // --- Static buffers ---
    buf_obs_flat: Buffer,
    buf_toy_offsets: Buffer,
    buf_obs_lo: Buffer,
    buf_obs_hi: Buffer,
    buf_procs: Buffer,
    buf_rate_mods: Buffer,
    buf_shape_pidx: Buffer,
    buf_pdf_aux_f32: Buffer,
    buf_gauss: Buffer,

    // --- Dynamic buffers ---
    buf_params_flat: Buffer,
    buf_nll_out: Buffer,
    buf_grad_out: Buffer,

    // --- Metadata ---
    n_params: usize,
    n_toys: usize,
    scalar_args: BatchScalarArgs,

    // --- CPU scratch ---
    scratch_params_f32: Vec<f32>,
    scratch_zeros_f32: Vec<f32>,
}

impl MetalUnbinnedBatchAccelerator {
    /// Check if Metal is available at runtime.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create batch accelerator from static unbinned model data + toy datasets.
    pub fn from_unbinned_static_and_toys(
        data: &UnbinnedGpuModelData,
        toy_offsets: &[u32],
        obs_flat: &[f64],
        n_toys: usize,
    ) -> ns_core::Result<Self> {
        if data.n_obs != 1 {
            return Err(ns_core::Error::Validation(format!(
                "Metal unbinned batch currently supports n_obs=1, got {}",
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

        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        let opts = MTLResourceOptions::StorageModeShared;
        let obs_flat_f32: Vec<f32> = obs_flat.iter().map(|&v| v as f32).collect();
        let buf_obs_flat = Self::create_buffer_from_slice(&device, &obs_flat_f32, opts);
        let buf_toy_offsets = Self::create_buffer_from_slice(&device, toy_offsets, opts);

        Self::build(device, queue, data, buf_obs_flat, buf_toy_offsets, n_toys)
    }

    /// Create batch accelerator from device-resident toy data (Metal Buffer, f32).
    ///
    /// `buf_obs_flat_f32` must be a Metal shared buffer containing f32 observed events
    /// (e.g. returned by [`MetalUnbinnedToySampler::sample_toys_1d_device`]).
    /// This avoids the f64→f32 conversion and buffer re-allocation.
    pub fn from_unbinned_static_and_toys_device(
        data: &UnbinnedGpuModelData,
        toy_offsets: &[u32],
        buf_obs_flat_f32: Buffer,
        n_toys: usize,
    ) -> ns_core::Result<Self> {
        if data.n_obs != 1 {
            return Err(ns_core::Error::Validation(format!(
                "Metal unbinned batch currently supports n_obs=1, got {}",
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

        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        let opts = MTLResourceOptions::StorageModeShared;
        let buf_toy_offsets = Self::create_buffer_from_slice(&device, toy_offsets, opts);

        Self::build(device, queue, data, buf_obs_flat_f32, buf_toy_offsets, n_toys)
    }

    /// Shared builder: compiles MSL, uploads model-descriptor buffers, allocates work buffers.
    fn build(
        device: Device,
        queue: CommandQueue,
        data: &UnbinnedGpuModelData,
        buf_obs_flat: Buffer,
        buf_toy_offsets: Buffer,
        n_toys: usize,
    ) -> ns_core::Result<Self> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SRC, &options)
            .map_err(|e| metal_err(format!("MSL compile: {e}")))?;

        let fn_nll_grad = library
            .get_function("unbinned_batch_nll_grad", None)
            .map_err(|e| metal_err(format!("get unbinned_batch_nll_grad: {e}")))?;
        let fn_nll_only = library
            .get_function("unbinned_batch_nll_only", None)
            .map_err(|e| metal_err(format!("get unbinned_batch_nll_only: {e}")))?;

        let pipeline_nll_grad = device
            .new_compute_pipeline_state_with_function(&fn_nll_grad)
            .map_err(|e| metal_err(format!("pipeline unbinned_batch_nll_grad: {e}")))?;
        let pipeline_nll_only = device
            .new_compute_pipeline_state_with_function(&fn_nll_only)
            .map_err(|e| metal_err(format!("pipeline unbinned_batch_nll_only: {e}")))?;

        let (lo, hi) = data
            .obs_bounds
            .get(0)
            .copied()
            .ok_or_else(|| ns_core::Error::Validation("missing obs_bounds[0]".into()))?;

        let opts = MTLResourceOptions::StorageModeShared;

        let obs_lo_f32 = [lo as f32];
        let obs_hi_f32 = [hi as f32];

        let procs_f32: Vec<MetalUnbinnedProcessDesc> = data
            .processes
            .iter()
            .map(|p| MetalUnbinnedProcessDesc {
                base_yield: p.base_yield as f32,
                pdf_kind: p.pdf_kind,
                yield_kind: p.yield_kind,
                obs_index: p.obs_index,
                shape_param_offset: p.shape_param_offset,
                n_shape_params: p.n_shape_params,
                yield_param_idx: p.yield_param_idx,
                rate_mod_offset: p.rate_mod_offset,
                n_rate_mods: p.n_rate_mods,
                pdf_aux_offset: p.pdf_aux_offset,
                pdf_aux_len: p.pdf_aux_len,
            })
            .collect();

        let rate_mods_f32: Vec<MetalUnbinnedRateModifierDesc> = data
            .rate_modifiers
            .iter()
            .map(|m| MetalUnbinnedRateModifierDesc {
                kind: m.kind,
                alpha_param_idx: m.alpha_param_idx,
                interp_code: m.interp_code,
                _pad: 0,
                lo: m.lo as f32,
                hi: m.hi as f32,
            })
            .collect();

        let gauss_f32: Vec<MetalUnbinnedGaussConstraintEntry> = data
            .gauss_constraints
            .iter()
            .map(|g| MetalUnbinnedGaussConstraintEntry {
                center: g.center as f32,
                inv_width: g.inv_width as f32,
                param_idx: g.param_idx,
                _pad: 0,
            })
            .collect();

        let pdf_aux_f32: Vec<f32> = data.pdf_aux_f64.iter().map(|&v| v as f32).collect();

        let buf_obs_lo = Self::create_buffer_from_slice(&device, &obs_lo_f32, opts);
        let buf_obs_hi = Self::create_buffer_from_slice(&device, &obs_hi_f32, opts);
        let buf_procs = Self::create_buffer_from_slice(&device, &procs_f32, opts);
        let buf_rate_mods = Self::create_buffer_from_slice(&device, &rate_mods_f32, opts);
        let buf_shape_pidx =
            Self::create_buffer_from_slice(&device, &data.shape_param_indices, opts);
        let buf_pdf_aux_f32 = Self::create_buffer_from_slice(&device, &pdf_aux_f32, opts);
        let buf_gauss = Self::create_buffer_from_slice(&device, &gauss_f32, opts);

        let buf_params_flat =
            device.new_buffer((n_toys * data.n_params * mem::size_of::<f32>()) as u64, opts);
        let buf_nll_out = device.new_buffer((n_toys * mem::size_of::<f32>()) as u64, opts);
        let buf_grad_out =
            device.new_buffer((n_toys * data.n_params * mem::size_of::<f32>()) as u64, opts);

        let scalar_args = BatchScalarArgs {
            n_params: data.n_params as u32,
            n_procs: data.processes.len() as u32,
            total_rate_mods: data.rate_modifiers.len() as u32,
            total_shape_params: data.shape_param_indices.len() as u32,
            n_gauss: data.gauss_constraints.len() as u32,
            n_toys: n_toys as u32,
            constraint_const: data.constraint_const as f32,
        };

        Ok(Self {
            device,
            queue,
            pipeline_nll_grad,
            pipeline_nll_only,
            buf_obs_flat,
            buf_toy_offsets,
            buf_obs_lo,
            buf_obs_hi,
            buf_procs,
            buf_rate_mods,
            buf_shape_pidx,
            buf_pdf_aux_f32,
            buf_gauss,
            buf_params_flat,
            buf_nll_out,
            buf_grad_out,
            n_params: data.n_params,
            n_toys,
            scalar_args,
            scratch_params_f32: vec![0.0f32; n_toys * data.n_params],
            scratch_zeros_f32: vec![0.0f32; n_toys * data.n_params],
        })
    }

    fn block_size(&self) -> usize {
        // Fixed, power-of-two block size; toys have variable event counts.
        256
    }

    /// Batch evaluation: NLL + gradient for all toys.
    pub fn batch_nll_grad(&mut self, params_flat: &[f64]) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        if params_flat.len() != self.n_toys * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch: expected {}, got {}",
                self.n_toys * self.n_params,
                params_flat.len()
            )));
        }

        for (i, &v) in params_flat.iter().enumerate() {
            self.scratch_params_f32[i] = v as f32;
            self.scratch_zeros_f32[i] = 0.0f32;
        }
        Self::copy_to_buffer(&self.buf_params_flat, &self.scratch_params_f32);
        Self::copy_to_buffer(&self.buf_grad_out, &self.scratch_zeros_f32);

        let block_size = self.block_size();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_nll_grad);
        encoder.set_buffer(0, Some(&self.buf_params_flat), 0);
        encoder.set_buffer(1, Some(&self.buf_obs_flat), 0);
        encoder.set_buffer(2, Some(&self.buf_toy_offsets), 0);
        encoder.set_buffer(3, Some(&self.buf_obs_lo), 0);
        encoder.set_buffer(4, Some(&self.buf_obs_hi), 0);
        encoder.set_buffer(5, Some(&self.buf_procs), 0);
        encoder.set_buffer(6, Some(&self.buf_rate_mods), 0);
        encoder.set_buffer(7, Some(&self.buf_shape_pidx), 0);
        encoder.set_buffer(8, Some(&self.buf_pdf_aux_f32), 0);
        encoder.set_buffer(9, Some(&self.buf_gauss), 0);
        encoder.set_buffer(10, Some(&self.buf_nll_out), 0);
        encoder.set_buffer(11, Some(&self.buf_grad_out), 0);
        encoder.set_bytes(
            12,
            mem::size_of::<BatchScalarArgs>() as u64,
            &self.scalar_args as *const BatchScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        let grid = MTLSize::new(self.n_toys as u64, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let nll = Self::read_buffer_f32_to_f64(&self.buf_nll_out, self.n_toys);
        let grad = Self::read_buffer_f32_to_f64(&self.buf_grad_out, self.n_toys * self.n_params);
        Ok((nll, grad))
    }

    /// Batch evaluation: NLL-only for all toys.
    pub fn batch_nll(&mut self, params_flat: &[f64]) -> ns_core::Result<Vec<f64>> {
        if params_flat.len() != self.n_toys * self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length mismatch: expected {}, got {}",
                self.n_toys * self.n_params,
                params_flat.len()
            )));
        }

        for (i, &v) in params_flat.iter().enumerate() {
            self.scratch_params_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_params_flat, &self.scratch_params_f32);

        let block_size = self.block_size();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_nll_only);
        encoder.set_buffer(0, Some(&self.buf_params_flat), 0);
        encoder.set_buffer(1, Some(&self.buf_obs_flat), 0);
        encoder.set_buffer(2, Some(&self.buf_toy_offsets), 0);
        encoder.set_buffer(3, Some(&self.buf_obs_lo), 0);
        encoder.set_buffer(4, Some(&self.buf_obs_hi), 0);
        encoder.set_buffer(5, Some(&self.buf_procs), 0);
        encoder.set_buffer(6, Some(&self.buf_rate_mods), 0);
        encoder.set_buffer(7, Some(&self.buf_shape_pidx), 0);
        encoder.set_buffer(8, Some(&self.buf_pdf_aux_f32), 0);
        encoder.set_buffer(9, Some(&self.buf_gauss), 0);
        encoder.set_buffer(10, Some(&self.buf_nll_out), 0);
        encoder.set_bytes(
            11,
            mem::size_of::<BatchScalarArgs>() as u64,
            &self.scalar_args as *const BatchScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        let grid = MTLSize::new(self.n_toys as u64, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok(Self::read_buffer_f32_to_f64(&self.buf_nll_out, self.n_toys))
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of toys in this batch.
    pub fn n_toys(&self) -> usize {
        self.n_toys
    }

    // --- Private helpers (mirrors metal_unbinned.rs) ---

    fn create_buffer_from_slice<T>(
        device: &Device,
        data: &[T],
        opts: MTLResourceOptions,
    ) -> Buffer {
        if data.is_empty() {
            // Metal buffers must have non-zero length.
            return device.new_buffer(4, opts);
        }
        let bytes = (data.len() * mem::size_of::<T>()) as u64;
        let buf = device.new_buffer(bytes, opts);
        let ptr = buf.contents() as *mut T;
        // SAFETY: `buf` was just allocated with `bytes = data.len() * size_of::<T>()`.
        // `contents()` returns a valid mapped pointer; no overlap with `data`.
        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        buf.did_modify_range(NSRange::new(0, bytes));
        buf
    }

    fn copy_to_buffer<T>(buf: &Buffer, data: &[T]) {
        if data.is_empty() {
            return;
        }
        let bytes = (data.len() * mem::size_of::<T>()) as u64;
        let ptr = buf.contents() as *mut T;
        // SAFETY: `buf` capacity >= `bytes`; `contents()` is valid; no overlap with `data`.
        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        buf.did_modify_range(NSRange::new(0, bytes));
    }

    fn read_buffer_f32_to_f64(buf: &Buffer, n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        let ptr = buf.contents() as *const f32;
        // SAFETY: `buf` capacity >= `n * 4`; `n` matches kernel output element count.
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        slice.iter().map(|&v| v as f64).collect()
    }
}
