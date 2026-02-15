//! Metal unbinned (event-level) NLL + gradient accelerator.
//!
//! Manages GPU memory, MSL compilation, and kernel dispatch for fused
//! unbinned mixture NLL + analytical gradient computation.
//!
//! All computation in f32. Conversion f64↔f32 happens at the API boundary.

use crate::unbinned_types::*;
use metal::*;
use std::mem;

/// MSL source compiled from `kernels/unbinned_nll_grad.metal`.
const MSL_SRC: &str = include_str!("../kernels/unbinned_nll_grad.metal");

/// Scalar arguments passed to the kernel via `set_bytes`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ScalarArgs {
    n_params: u32,
    n_obs: u32,
    n_events: u32,
    has_evt_wts: u32,
    n_procs: u32,
    total_rate_mods: u32,
    total_shape_params: u32,
    n_gauss: u32,
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
    ns_core::Error::Computation(format!("Metal (unbinned): {msg}"))
}

/// Metal accelerator for unbinned NLL + gradient.
#[allow(dead_code)]
pub struct MetalUnbinnedAccelerator {
    device: Device,
    queue: CommandQueue,
    pipeline_nll_grad: ComputePipelineState,
    pipeline_nll_only: ComputePipelineState,

    // --- Static buffers (uploaded once) ---
    buf_obs_soa: Buffer,
    buf_obs_lo: Buffer,
    buf_obs_hi: Buffer,
    buf_evt_wts: Buffer,
    buf_procs: Buffer,
    buf_rate_mods: Buffer,
    buf_shape_pidx: Buffer,
    buf_pdf_aux_f32: Buffer,
    buf_gauss: Buffer,

    // --- Dynamic buffers ---
    buf_params: Buffer,
    buf_nll_out: Buffer,
    buf_grad_out: Buffer,

    // --- Metadata ---
    n_params: usize,
    n_events: usize,
    scalar_args: ScalarArgs,

    // --- CPU scratch buffers ---
    scratch_params_f32: Vec<f32>,
    scratch_zeros_f32: Vec<f32>,
}

impl MetalUnbinnedAccelerator {
    /// Check if Metal is available at runtime.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create accelerator from lowered unbinned GPU model data.
    pub fn from_unbinned_data(data: &UnbinnedGpuModelData) -> ns_core::Result<Self> {
        if data.n_obs == 0 {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires n_obs > 0".into(),
            ));
        }
        if data.n_events == 0 {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires n_events > 0".into(),
            ));
        }
        if data.processes.is_empty() {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires at least one process".into(),
            ));
        }

        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        // Compile/load MSL pipelines via thread-local cache.
        let pipeline_nll_grad = crate::metal_kernel_cache::get_pipeline(
            &device,
            "unbinned_single",
            MSL_SRC,
            "unbinned_nll_grad",
        )?;
        let pipeline_nll_only = crate::metal_kernel_cache::get_pipeline(
            &device,
            "unbinned_single",
            MSL_SRC,
            "unbinned_nll_only",
        )?;

        let opts = MTLResourceOptions::StorageModeShared;

        // Convert observed data to f32.
        let obs_soa_f32: Vec<f32> = data.obs_soa.iter().map(|&v| v as f32).collect();
        let (evt_wts_f32, has_evt_wts) = match &data.event_weights {
            Some(w) => {
                if w.len() != data.n_events {
                    return Err(ns_core::Error::Validation(format!(
                        "event_weights length mismatch: expected {}, got {}",
                        data.n_events,
                        w.len()
                    )));
                }
                let mut any_pos = false;
                let mut out = Vec::with_capacity(w.len());
                for &wi in w {
                    if !(wi.is_finite() && wi >= 0.0) {
                        return Err(ns_core::Error::Validation(format!(
                            "invalid event weight (expected finite and >= 0): {wi}"
                        )));
                    }
                    any_pos |= wi > 0.0;
                    out.push(wi as f32);
                }
                if !any_pos {
                    return Err(ns_core::Error::Validation(
                        "event_weights must have sum(weights) > 0".into(),
                    ));
                }
                (out, 1u32)
            }
            None => (vec![1.0f32], 0u32),
        };

        let mut obs_lo_f32 = Vec::<f32>::with_capacity(data.n_obs);
        let mut obs_hi_f32 = Vec::<f32>::with_capacity(data.n_obs);
        for &(lo, hi) in &data.obs_bounds {
            obs_lo_f32.push(lo as f32);
            obs_hi_f32.push(hi as f32);
        }

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

        let buf_obs_soa = Self::create_buffer_from_slice(&device, &obs_soa_f32, opts);
        let buf_obs_lo = Self::create_buffer_from_slice(&device, &obs_lo_f32, opts);
        let buf_obs_hi = Self::create_buffer_from_slice(&device, &obs_hi_f32, opts);
        let buf_evt_wts = Self::create_buffer_from_slice(&device, &evt_wts_f32, opts);
        let buf_procs = Self::create_buffer_from_slice(&device, &procs_f32, opts);
        let buf_rate_mods = Self::create_buffer_from_slice(&device, &rate_mods_f32, opts);
        let buf_shape_pidx =
            Self::create_buffer_from_slice(&device, &data.shape_param_indices, opts);
        let buf_pdf_aux_f32 = Self::create_buffer_from_slice(&device, &pdf_aux_f32, opts);
        let buf_gauss = Self::create_buffer_from_slice(&device, &gauss_f32, opts);

        let buf_params = device.new_buffer((data.n_params * mem::size_of::<f32>()) as u64, opts);
        let buf_nll_out = device.new_buffer(mem::size_of::<f32>() as u64, opts);
        let buf_grad_out = device.new_buffer((data.n_params * mem::size_of::<f32>()) as u64, opts);

        let scalar_args = ScalarArgs {
            n_params: data.n_params as u32,
            n_obs: data.n_obs as u32,
            n_events: data.n_events as u32,
            has_evt_wts,
            n_procs: data.processes.len() as u32,
            total_rate_mods: data.rate_modifiers.len() as u32,
            total_shape_params: data.shape_param_indices.len() as u32,
            n_gauss: data.gauss_constraints.len() as u32,
            constraint_const: data.constraint_const as f32,
        };

        Ok(Self {
            device,
            queue,
            pipeline_nll_grad,
            pipeline_nll_only,
            buf_obs_soa,
            buf_obs_lo,
            buf_obs_hi,
            buf_evt_wts,
            buf_procs,
            buf_rate_mods,
            buf_shape_pidx,
            buf_pdf_aux_f32,
            buf_gauss,
            buf_params,
            buf_nll_out,
            buf_grad_out,
            n_params: data.n_params,
            n_events: data.n_events,
            scalar_args,
            scratch_params_f32: vec![0.0f32; data.n_params],
            scratch_zeros_f32: vec![0.0f32; data.n_params],
        })
    }

    fn block_size(&self) -> usize {
        let n_threads = self.n_events.clamp(1, 256);
        n_threads.next_power_of_two()
    }

    /// Single evaluation: NLL + gradient.
    pub fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "param length mismatch: expected {}, got {}",
                self.n_params,
                params.len()
            )));
        }

        for (i, &v) in params.iter().enumerate() {
            self.scratch_params_f32[i] = v as f32;
            self.scratch_zeros_f32[i] = 0.0f32;
        }
        Self::copy_to_buffer(&self.buf_params, &self.scratch_params_f32);
        Self::copy_to_buffer(&self.buf_grad_out, &self.scratch_zeros_f32);

        let block_size = self.block_size();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_nll_grad);
        encoder.set_buffer(0, Some(&self.buf_params), 0);
        encoder.set_buffer(1, Some(&self.buf_obs_soa), 0);
        encoder.set_buffer(2, Some(&self.buf_obs_lo), 0);
        encoder.set_buffer(3, Some(&self.buf_obs_hi), 0);
        encoder.set_buffer(4, Some(&self.buf_evt_wts), 0);
        encoder.set_buffer(5, Some(&self.buf_procs), 0);
        encoder.set_buffer(6, Some(&self.buf_rate_mods), 0);
        encoder.set_buffer(7, Some(&self.buf_shape_pidx), 0);
        encoder.set_buffer(8, Some(&self.buf_pdf_aux_f32), 0);
        encoder.set_buffer(9, Some(&self.buf_gauss), 0);
        encoder.set_buffer(10, Some(&self.buf_nll_out), 0);
        encoder.set_buffer(11, Some(&self.buf_grad_out), 0);
        encoder.set_bytes(
            12,
            mem::size_of::<ScalarArgs>() as u64,
            &self.scalar_args as *const ScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        let grid = MTLSize::new(1, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let nll = Self::read_buffer_f32_to_f64(&self.buf_nll_out, 1)[0];
        let grad = Self::read_buffer_f32_to_f64(&self.buf_grad_out, self.n_params);
        Ok((nll, grad))
    }

    /// Single evaluation: NLL-only.
    pub fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64> {
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "param length mismatch: expected {}, got {}",
                self.n_params,
                params.len()
            )));
        }

        for (i, &v) in params.iter().enumerate() {
            self.scratch_params_f32[i] = v as f32;
        }
        Self::copy_to_buffer(&self.buf_params, &self.scratch_params_f32);

        let block_size = self.block_size();
        let shared_bytes = (self.n_params + block_size) * mem::size_of::<f32>();

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_nll_only);
        encoder.set_buffer(0, Some(&self.buf_params), 0);
        encoder.set_buffer(1, Some(&self.buf_obs_soa), 0);
        encoder.set_buffer(2, Some(&self.buf_obs_lo), 0);
        encoder.set_buffer(3, Some(&self.buf_obs_hi), 0);
        encoder.set_buffer(4, Some(&self.buf_evt_wts), 0);
        encoder.set_buffer(5, Some(&self.buf_procs), 0);
        encoder.set_buffer(6, Some(&self.buf_rate_mods), 0);
        encoder.set_buffer(7, Some(&self.buf_shape_pidx), 0);
        encoder.set_buffer(8, Some(&self.buf_pdf_aux_f32), 0);
        encoder.set_buffer(9, Some(&self.buf_gauss), 0);
        encoder.set_buffer(10, Some(&self.buf_nll_out), 0);
        encoder.set_bytes(
            11,
            mem::size_of::<ScalarArgs>() as u64,
            &self.scalar_args as *const ScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);

        let grid = MTLSize::new(1, 1, 1);
        let tg = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok(Self::read_buffer_f32_to_f64(&self.buf_nll_out, 1)[0])
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    // --- Private helpers (mirrors metal_batch.rs) ---

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
            std::mem::size_of_val(data) as u64,
            opts,
        )
    }

    fn copy_to_buffer<T>(buffer: &Buffer, data: &[T]) {
        if data.is_empty() {
            return;
        }
        let ptr = buffer.contents() as *mut T;
        // SAFETY: `buffer` was allocated with capacity >= `data.len() * size_of::<T>()`.
        // `contents()` returns a valid mapped pointer for the buffer's lifetime.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn read_buffer_f32_to_f64(buffer: &Buffer, count: usize) -> Vec<f64> {
        let ptr = buffer.contents() as *const f32;
        // SAFETY: `buffer` capacity >= `count * 4`; `count` matches kernel output size.
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.iter().map(|&v| v as f64).collect()
    }
}
