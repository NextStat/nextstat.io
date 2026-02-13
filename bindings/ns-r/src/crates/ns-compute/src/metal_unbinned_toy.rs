//! Metal unbinned (event-level) toy sampling kernels.
//!
//! This is an optional accelerator that generates toy datasets for the conservative unbinned GPU
//! subset (1 channel / 1 observable; PDFs: Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev/Histogram;
//! yields: fixed/parameter/scaled;
//! yield modifiers: NormSys/WeightSys).
//!
//! Two sampling APIs:
//! - `sample_toys_1d` — returns host `(Vec<u32>, Vec<f64>)` with f32→f64 conversion.
//! - `sample_toys_1d_device` — returns `(Vec<u32>, Buffer)` keeping obs_flat as a Metal f32
//!   shared buffer, suitable for zero-copy handoff to `MetalUnbinnedBatchAccelerator`.

use crate::unbinned_types::*;
use metal::*;
use std::mem;

/// MSL source compiled from `kernels/unbinned_nll_grad.metal` at runtime.
const MSL_SRC: &str = include_str!("../kernels/unbinned_nll_grad.metal");

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ToyScalarArgs {
    n_procs: u32,
    total_rate_mods: u32,
    total_shape_params: u32,
    n_toys: u32,
    seed: u64,
    total_pdf_aux_f32: u32,
    _pad0: u32,
    _pad1: u32,
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
struct MetalUnbinnedRateModifierDesc {
    kind: u32,
    alpha_param_idx: u32,
    interp_code: u32,
    _pad: u32,
    lo: f32,
    hi: f32,
}

fn metal_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("Metal (unbinned toy): {msg}"))
}

/// Metal toy sampler for unbinned 1D models.
#[allow(dead_code)]
pub struct MetalUnbinnedToySampler {
    device: Device,
    queue: CommandQueue,
    pipeline_counts: ComputePipelineState,
    pipeline_sample: ComputePipelineState,

    // Static buffers
    buf_obs_lo: Buffer,
    buf_obs_hi: Buffer,
    buf_procs: Buffer,
    buf_rate_mods: Buffer,
    buf_shape_pidx: Buffer,
    buf_pdf_aux_f32: Buffer,

    n_params: usize,
    n_procs: usize,
    total_shape_params: usize,
    total_rate_mods: usize,
    total_pdf_aux_f32: usize,
}

impl MetalUnbinnedToySampler {
    /// Check if Metal is available at runtime.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create a toy sampler from a GPU-lowered static model.
    pub fn from_unbinned_static(data: &UnbinnedGpuModelData) -> ns_core::Result<Self> {
        if data.n_obs != 1 {
            return Err(ns_core::Error::Validation(format!(
                "Metal unbinned toy sampler currently supports n_obs=1, got {}",
                data.n_obs
            )));
        }
        if data.processes.is_empty() {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires at least one process".into(),
            ));
        }

        // The Metal toy sampler only knows how to sample from analytic 1D PDFs.
        for (i, p) in data.processes.iter().enumerate() {
            if p.obs_index != 0 {
                return Err(ns_core::Error::Validation(format!(
                    "Metal unbinned toy sampler requires obs_index=0 for all processes (n_obs=1), got processes[{i}].obs_index={}",
                    p.obs_index
                )));
            }
            match p.pdf_kind {
                pdf_kind::GAUSSIAN
                | pdf_kind::EXPONENTIAL
                | pdf_kind::CRYSTAL_BALL
                | pdf_kind::DOUBLE_CRYSTAL_BALL
                | pdf_kind::CHEBYSHEV
                | pdf_kind::HISTOGRAM => {}
                other => {
                    return Err(ns_core::Error::Validation(format!(
                        "Metal unbinned toy sampler supports only Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev/Histogram PDFs; got processes[{i}].pdf_kind={other}"
                    )));
                }
            }
            match p.yield_kind {
                yield_kind::FIXED | yield_kind::PARAMETER | yield_kind::SCALED => {}
                other => {
                    return Err(ns_core::Error::Validation(format!(
                        "Metal unbinned toy sampler only supports fixed/parameter/scaled yields; got processes[{i}].yield_kind={other}"
                    )));
                }
            }

            match p.pdf_kind {
                pdf_kind::GAUSSIAN => {
                    if p.n_shape_params != 2 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler expects Gaussian n_shape_params=2; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off + 1 >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler Gaussian shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::EXPONENTIAL => {
                    if p.n_shape_params != 1 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler expects Exponential n_shape_params=1; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler Exponential shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::CRYSTAL_BALL => {
                    if p.n_shape_params != 4 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler expects CrystalBall n_shape_params=4; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off + 3 >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler CrystalBall shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::DOUBLE_CRYSTAL_BALL => {
                    if p.n_shape_params != 6 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler DoubleCrystalBall expects n_shape_params=6; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off + 5 >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler DoubleCrystalBall shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::CHEBYSHEV => {
                    if p.n_shape_params == 0 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler expects Chebyshev n_shape_params>=1; got processes[{i}].n_shape_params=0"
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    let order = p.n_shape_params as usize;
                    if off + (order - 1) >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler Chebyshev shape_param_offset out of range: processes[{i}].shape_param_offset={} order={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            order,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::HISTOGRAM => {
                    if p.n_shape_params != 0 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler expects Histogram n_shape_params=0; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.pdf_aux_offset as usize;
                    let len = p.pdf_aux_len as usize;
                    if len < 3 || (len & 1) == 0 {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler Histogram expects odd pdf_aux_len>=3 (2*n_bins+1); got processes[{i}].pdf_aux_len={}",
                            p.pdf_aux_len
                        )));
                    }
                    if off + len > data.pdf_aux_f64.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Metal unbinned toy sampler Histogram pdf_aux out of range: processes[{i}].pdf_aux_offset={} len={} (pdf_aux_f64 len={})",
                            p.pdf_aux_offset,
                            len,
                            data.pdf_aux_f64.len()
                        )));
                    }
                }
                _ => {}
            }

            let mod_off = p.rate_mod_offset as usize;
            let nmods = p.n_rate_mods as usize;
            if mod_off + nmods > data.rate_modifiers.len() {
                return Err(ns_core::Error::Validation(format!(
                    "Metal unbinned toy sampler rate_modifiers range out of bounds: processes[{i}].rate_mod_offset={mod_off}, n_rate_mods={nmods}, total_rate_mods={}",
                    data.rate_modifiers.len()
                )));
            }
        }

        let (lo, hi) = data
            .obs_bounds
            .get(0)
            .copied()
            .ok_or_else(|| ns_core::Error::Validation("missing obs_bounds[0]".into()))?;

        let device = Device::system_default().ok_or_else(|| metal_err("no Metal device found"))?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SRC, &options)
            .map_err(|e| metal_err(format!("MSL compile: {e}")))?;

        let fn_counts = library
            .get_function("unbinned_toy_counts", None)
            .map_err(|e| metal_err(format!("get unbinned_toy_counts: {e}")))?;
        let fn_sample = library
            .get_function("unbinned_toy_sample_obs_1d", None)
            .map_err(|e| metal_err(format!("get unbinned_toy_sample_obs_1d: {e}")))?;

        let pipeline_counts = device
            .new_compute_pipeline_state_with_function(&fn_counts)
            .map_err(|e| metal_err(format!("pipeline unbinned_toy_counts: {e}")))?;
        let pipeline_sample = device
            .new_compute_pipeline_state_with_function(&fn_sample)
            .map_err(|e| metal_err(format!("pipeline unbinned_toy_sample_obs_1d: {e}")))?;

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

        let buf_obs_lo = device.new_buffer_with_data(
            obs_lo_f32.as_ptr() as *const _,
            (obs_lo_f32.len() * mem::size_of::<f32>()) as u64,
            opts,
        );
        let buf_obs_hi = device.new_buffer_with_data(
            obs_hi_f32.as_ptr() as *const _,
            (obs_hi_f32.len() * mem::size_of::<f32>()) as u64,
            opts,
        );
        let buf_procs = device.new_buffer_with_data(
            procs_f32.as_ptr() as *const _,
            (procs_f32.len() * mem::size_of::<MetalUnbinnedProcessDesc>()) as u64,
            opts,
        );
        let buf_rate_mods = if rate_mods_f32.is_empty() {
            let dummy = [MetalUnbinnedRateModifierDesc {
                kind: 0,
                alpha_param_idx: 0,
                interp_code: 0,
                _pad: 0,
                lo: 0.0,
                hi: 0.0,
            }];
            device.new_buffer_with_data(
                dummy.as_ptr() as *const _,
                mem::size_of_val(&dummy) as u64,
                opts,
            )
        } else {
            device.new_buffer_with_data(
                rate_mods_f32.as_ptr() as *const _,
                (rate_mods_f32.len() * mem::size_of::<MetalUnbinnedRateModifierDesc>()) as u64,
                opts,
            )
        };
        let buf_shape_pidx = if data.shape_param_indices.is_empty() {
            let dummy = [0u32];
            device.new_buffer_with_data(
                dummy.as_ptr() as *const _,
                mem::size_of_val(&dummy) as u64,
                opts,
            )
        } else {
            device.new_buffer_with_data(
                data.shape_param_indices.as_ptr() as *const _,
                (data.shape_param_indices.len() * mem::size_of::<u32>()) as u64,
                opts,
            )
        };
        let pdf_aux_f32: Vec<f32> = data.pdf_aux_f64.iter().map(|&v| v as f32).collect();
        let buf_pdf_aux_f32 = if pdf_aux_f32.is_empty() {
            let dummy = [0.0f32];
            device.new_buffer_with_data(
                dummy.as_ptr() as *const _,
                mem::size_of_val(&dummy) as u64,
                opts,
            )
        } else {
            device.new_buffer_with_data(
                pdf_aux_f32.as_ptr() as *const _,
                (pdf_aux_f32.len() * mem::size_of::<f32>()) as u64,
                opts,
            )
        };

        Ok(Self {
            device,
            queue,
            pipeline_counts,
            pipeline_sample,
            buf_obs_lo,
            buf_obs_hi,
            buf_procs,
            buf_rate_mods,
            buf_shape_pidx,
            buf_pdf_aux_f32,
            n_params: data.n_params,
            n_procs: data.processes.len(),
            total_shape_params: data.shape_param_indices.len(),
            total_rate_mods: data.rate_modifiers.len(),
            total_pdf_aux_f32: data.pdf_aux_f64.len(),
        })
    }

    /// Returns the Metal device used by this sampler.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the Metal command queue used by this sampler.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Run Kernel A (Poisson counts) + host prefix sum + Kernel B (sample observables).
    ///
    /// Returns `(toy_offsets, Option<buf_obs_flat_f32>)` where `buf_obs_flat_f32` is a
    /// Metal shared buffer containing f32 observed values. `None` when total events == 0.
    fn sample_toys_1d_inner(
        &self,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> ns_core::Result<(Vec<u32>, Option<Buffer>)> {
        if n_toys == 0 {
            return Err(ns_core::Error::Validation("n_toys must be > 0".into()));
        }
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params length mismatch: expected {}, got {}",
                self.n_params,
                params.len()
            )));
        }

        let opts = MTLResourceOptions::StorageModeShared;

        let params_f32: Vec<f32> = params.iter().map(|&v| v as f32).collect();
        let buf_params = self.device.new_buffer_with_data(
            params_f32.as_ptr() as *const _,
            (params_f32.len() * mem::size_of::<f32>()) as u64,
            opts,
        );

        let buf_counts =
            self.device.new_buffer((n_toys * mem::size_of::<u32>()).max(1) as u64, opts);

        let scalar = ToyScalarArgs {
            n_procs: self.n_procs as u32,
            total_rate_mods: self.total_rate_mods as u32,
            total_shape_params: self.total_shape_params as u32,
            n_toys: n_toys as u32,
            seed,
            total_pdf_aux_f32: self.total_pdf_aux_f32 as u32,
            _pad0: 0,
            _pad1: 0,
        };

        // --- counts kernel ---
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_counts);
        encoder.set_buffer(0, Some(&buf_params), 0);
        encoder.set_buffer(1, Some(&self.buf_procs), 0);
        encoder.set_buffer(2, Some(&self.buf_rate_mods), 0);
        encoder.set_bytes(
            3,
            mem::size_of::<ToyScalarArgs>() as u64,
            &scalar as *const ToyScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(4, Some(&buf_counts), 0);

        let tg: usize =
            (self.pipeline_counts.max_total_threads_per_threadgroup().min(256)) as usize;
        let grid = MTLSize::new(((n_toys + tg - 1) / tg) as u64, 1, 1);
        let tg = MTLSize::new(tg as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // SAFETY: `buf_counts` is a shared Metal buffer we just wrote to via the counts kernel.
        // Length is `n_toys` u32 elements.
        let counts: Vec<u32> = unsafe {
            let ptr = buf_counts.contents() as *const u32;
            std::slice::from_raw_parts(ptr, n_toys).to_vec()
        };

        // Prefix sum on CPU (small buffer).
        let mut toy_offsets = Vec::with_capacity(n_toys + 1);
        toy_offsets.push(0u32);
        let mut cur = 0u32;
        for &c in &counts {
            cur = cur
                .checked_add(c)
                .ok_or_else(|| ns_core::Error::Validation("toy offset overflow (u32)".into()))?;
            toy_offsets.push(cur);
        }
        let n_events = cur as usize;
        if n_events == 0 {
            return Ok((toy_offsets, None));
        }

        let buf_toy_offsets = self.device.new_buffer_with_data(
            toy_offsets.as_ptr() as *const _,
            (toy_offsets.len() * mem::size_of::<u32>()) as u64,
            opts,
        );
        let buf_obs_out =
            self.device.new_buffer((n_events * mem::size_of::<f32>()).max(1) as u64, opts);

        // --- sample kernel ---
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_sample);
        encoder.set_buffer(0, Some(&buf_params), 0);
        encoder.set_buffer(1, Some(&self.buf_obs_lo), 0);
        encoder.set_buffer(2, Some(&self.buf_obs_hi), 0);
        encoder.set_buffer(3, Some(&self.buf_procs), 0);
        encoder.set_buffer(4, Some(&self.buf_rate_mods), 0);
        encoder.set_buffer(5, Some(&self.buf_shape_pidx), 0);
        encoder.set_buffer(6, Some(&buf_toy_offsets), 0);
        encoder.set_buffer(7, Some(&self.buf_pdf_aux_f32), 0);
        encoder.set_bytes(
            8,
            mem::size_of::<ToyScalarArgs>() as u64,
            &scalar as *const ToyScalarArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(9, Some(&buf_obs_out), 0);

        let tg: usize =
            (self.pipeline_sample.max_total_threads_per_threadgroup().min(256)) as usize;
        let grid = MTLSize::new(n_toys as u64, 1, 1);
        let tg = MTLSize::new(tg as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        Ok((toy_offsets, Some(buf_obs_out)))
    }

    /// Sample toys on Metal GPU, returning all data on the host.
    ///
    /// Returns:
    /// - `toy_offsets`: prefix sums (length `n_toys+1`)
    /// - `obs_flat`: concatenated events as f64 (length `toy_offsets[n_toys]`)
    pub fn sample_toys_1d(
        &self,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> ns_core::Result<(Vec<u32>, Vec<f64>)> {
        let (toy_offsets, buf_obs_opt) = self.sample_toys_1d_inner(params, n_toys, seed)?;

        let n_events = *toy_offsets.last().unwrap_or(&0) as usize;
        let obs_flat = match buf_obs_opt {
            Some(buf) => {
                // SAFETY: `buf` is a shared Metal buffer with `n_events` f32 elements.
                let obs_f32: &[f32] = unsafe {
                    let ptr = buf.contents() as *const f32;
                    std::slice::from_raw_parts(ptr, n_events)
                };
                obs_f32.iter().map(|&v| v as f64).collect()
            }
            None => Vec::new(),
        };

        Ok((toy_offsets, obs_flat))
    }

    /// Sample toys on Metal GPU, keeping `obs_flat` as a device-resident Metal Buffer (f32).
    ///
    /// This avoids the f32→f64 host conversion. The returned `Buffer` can be passed directly
    /// to [`MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device`] when both
    /// share the same Metal device (which is always true on Apple Silicon).
    ///
    /// Returns:
    /// - `toy_offsets`: prefix sums on host (length `n_toys+1`)
    /// - `buf_obs_flat`: Metal shared buffer containing f32 events (length `toy_offsets[n_toys]`)
    pub fn sample_toys_1d_device(
        &self,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> ns_core::Result<(Vec<u32>, Buffer)> {
        let (toy_offsets, buf_obs_opt) = self.sample_toys_1d_inner(params, n_toys, seed)?;

        let buf_obs = match buf_obs_opt {
            Some(buf) => buf,
            None => {
                // Empty: allocate a minimal dummy buffer (Metal requires non-zero length).
                let opts = MTLResourceOptions::StorageModeShared;
                self.device.new_buffer(4, opts)
            }
        };

        Ok((toy_offsets, buf_obs))
    }
}
