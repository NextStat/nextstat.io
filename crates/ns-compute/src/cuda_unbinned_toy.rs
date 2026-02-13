//! CUDA unbinned (event-level) toy sampling kernels.
//!
//! This is an optional accelerator that generates toy datasets for the conservative unbinned GPU
//! subset (1 channel / 1 observable; PDFs: Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev/Histogram;
//! yields: fixed/parameter/scaled;
//! yield modifiers: NormSys/WeightSys).
//!
//! Two sampling APIs are provided:
//! - [`CudaUnbinnedToySampler::sample_toys_1d`]: returns `(Vec<u32>, Vec<f64>)` on the host.
//! - [`CudaUnbinnedToySampler::sample_toys_1d_device`]: returns `(Vec<u32>, CudaSlice<f64>)`,
//!   keeping `obs_flat` on the GPU to avoid a D2H + H2D round-trip when the batch fitter
//!   consumes the toys in the same CUDA context.
//!
//! Toy offsets are always computed on the host (prefix sum of per-toy Poisson counts) because the
//! buffer is small (`(n_toys+1) × 4` bytes) and a GPU scan is not worth the kernel launch overhead.

use crate::unbinned_types::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/unbinned_nll_grad.cu` at build time.
const PTX_SRC: &str = include_str!(env!("CUDA_UNBINNED_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (unbinned toy): {msg}"))
}

/// CUDA toy sampler for unbinned 1D models.
pub struct CudaUnbinnedToySampler {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_counts: CudaFunction,
    kernel_sample: CudaFunction,

    // Static buffers
    d_obs_lo: CudaSlice<f64>,
    d_obs_hi: CudaSlice<f64>,
    d_procs: CudaSlice<GpuUnbinnedProcessDesc>,
    d_rate_mods: CudaSlice<GpuUnbinnedRateModifierDesc>,
    d_shape_pidx: CudaSlice<u32>,
    d_pdf_aux_f64: CudaSlice<f64>,

    n_params: usize,
    n_procs: usize,
    total_shape_params: usize,
    total_rate_mods: usize,
    total_pdf_aux_f64: usize,
}

impl CudaUnbinnedToySampler {
    /// Check if CUDA is available at runtime (driver loaded, GPU present).
    pub fn is_available() -> bool {
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Check if a specific CUDA device is available at runtime.
    pub fn is_available_on_device(device_id: usize) -> bool {
        std::panic::catch_unwind(|| CudaContext::new(device_id).is_ok()).unwrap_or(false)
    }

    /// Create a toy sampler from a GPU-lowered static model (creates its own CUDA context).
    pub fn from_unbinned_static(data: &UnbinnedGpuModelData) -> ns_core::Result<Self> {
        Self::from_unbinned_static_on_device(data, 0)
    }

    /// Same as [`Self::from_unbinned_static`] but explicitly selects the CUDA device.
    pub fn from_unbinned_static_on_device(
        data: &UnbinnedGpuModelData,
        device_id: usize,
    ) -> ns_core::Result<Self> {
        let ctx = match std::panic::catch_unwind(|| CudaContext::new(device_id)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context (device {device_id}): {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let stream = ctx.default_stream();
        Self::with_context(ctx, stream, data)
    }

    /// Create a toy sampler that shares an existing CUDA context and stream.
    ///
    /// This is the preferred constructor when the sampler output will be consumed by a batch
    /// fitter in the same CUDA context (device-resident path).
    pub fn with_context(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        data: &UnbinnedGpuModelData,
    ) -> ns_core::Result<Self> {
        if data.n_obs != 1 {
            return Err(ns_core::Error::Validation(format!(
                "CUDA unbinned toy sampler currently supports n_obs=1, got {}",
                data.n_obs
            )));
        }
        if data.processes.is_empty() {
            return Err(ns_core::Error::Validation(
                "UnbinnedGpuModelData requires at least one process".into(),
            ));
        }

        // The CUDA toy sampler only knows how to sample from a limited set of 1D PDFs.
        for (i, p) in data.processes.iter().enumerate() {
            if p.obs_index != 0 {
                return Err(ns_core::Error::Validation(format!(
                    "CUDA unbinned toy sampler requires obs_index=0 for all processes (n_obs=1), got processes[{i}].obs_index={}",
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
                        "CUDA unbinned toy sampler supports only Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev/Histogram PDFs; got processes[{i}].pdf_kind={other}"
                    )));
                }
            }
            match p.yield_kind {
                yield_kind::FIXED | yield_kind::PARAMETER | yield_kind::SCALED => {}
                other => {
                    return Err(ns_core::Error::Validation(format!(
                        "CUDA unbinned toy sampler only supports fixed/parameter/scaled yields; got processes[{i}].yield_kind={other}"
                    )));
                }
            }

            // Guard against silent "midpoint sampling" fallbacks in the kernel.
            match p.pdf_kind {
                pdf_kind::GAUSSIAN => {
                    if p.n_shape_params != 2 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler expects Gaussian n_shape_params=2; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off + 1 >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler Gaussian shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::EXPONENTIAL => {
                    if p.n_shape_params != 1 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler expects Exponential n_shape_params=1; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler Exponential shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::CRYSTAL_BALL => {
                    if p.n_shape_params != 4 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler expects CrystalBall n_shape_params=4; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off + 3 >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler CrystalBall shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::DOUBLE_CRYSTAL_BALL => {
                    if p.n_shape_params != 6 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler DoubleCrystalBall expects n_shape_params=6; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    if off + 5 >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler DoubleCrystalBall shape_param_offset out of range: processes[{i}].shape_param_offset={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::CHEBYSHEV => {
                    if p.n_shape_params == 0 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler expects Chebyshev n_shape_params>=1; got processes[{i}].n_shape_params=0"
                        )));
                    }
                    let off = p.shape_param_offset as usize;
                    let order = p.n_shape_params as usize;
                    if off + (order - 1) >= data.shape_param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler Chebyshev shape_param_offset out of range: processes[{i}].shape_param_offset={} order={} (shape_param_indices len={})",
                            p.shape_param_offset,
                            order,
                            data.shape_param_indices.len()
                        )));
                    }
                }
                pdf_kind::HISTOGRAM => {
                    if p.n_shape_params != 0 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler expects Histogram n_shape_params=0; got processes[{i}].n_shape_params={}",
                            p.n_shape_params
                        )));
                    }
                    let off = p.pdf_aux_offset as usize;
                    let len = p.pdf_aux_len as usize;
                    if len < 3 || (len & 1) == 0 {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler Histogram expects odd pdf_aux_len>=3 (2*n_bins+1); got processes[{i}].pdf_aux_len={}",
                            p.pdf_aux_len
                        )));
                    }
                    if off + len > data.pdf_aux_f64.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "CUDA unbinned toy sampler Histogram pdf_aux out of range: processes[{i}].pdf_aux_offset={} len={} (pdf_aux_f64 len={})",
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
                    "CUDA unbinned toy sampler rate_modifiers range out of bounds: processes[{i}].rate_mod_offset={mod_off}, n_rate_mods={nmods}, total_rate_mods={}",
                    data.rate_modifiers.len()
                )));
            }
        }

        let (lo, hi) = data
            .obs_bounds
            .get(0)
            .copied()
            .ok_or_else(|| ns_core::Error::Validation("missing obs_bounds[0]".into()))?;

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;

        let kernel_counts = module
            .load_function("unbinned_toy_counts")
            .map_err(|e| cuda_err(format!("load unbinned_toy_counts: {e}")))?;
        let kernel_sample = module
            .load_function("unbinned_toy_sample_obs_1d")
            .map_err(|e| cuda_err(format!("load unbinned_toy_sample_obs_1d: {e}")))?;

        let d_obs_lo = stream.clone_htod(&[lo]).map_err(cuda_err)?;
        let d_obs_hi = stream.clone_htod(&[hi]).map_err(cuda_err)?;
        let d_procs = stream.clone_htod(&data.processes).map_err(cuda_err)?;
        let d_rate_mods = if data.rate_modifiers.is_empty() {
            // Allocate a tiny dummy buffer; kernels will not read from it when total_rate_mods=0.
            stream
                .clone_htod(&[GpuUnbinnedRateModifierDesc {
                    kind: 0,
                    alpha_param_idx: 0,
                    interp_code: 0,
                    _pad: 0,
                    lo: 0.0,
                    hi: 0.0,
                }])
                .map_err(cuda_err)?
        } else {
            stream.clone_htod(&data.rate_modifiers).map_err(cuda_err)?
        };
        let d_shape_pidx = stream.clone_htod(&data.shape_param_indices).map_err(cuda_err)?;
        let d_pdf_aux_f64 = if data.pdf_aux_f64.is_empty() {
            // Allocate a tiny dummy buffer; histogram sampling is gated by pdf_aux_len.
            stream.clone_htod(&[0.0f64]).map_err(cuda_err)?
        } else {
            stream.clone_htod(&data.pdf_aux_f64).map_err(cuda_err)?
        };

        Ok(Self {
            ctx,
            stream,
            kernel_counts,
            kernel_sample,
            d_obs_lo,
            d_obs_hi,
            d_procs,
            d_rate_mods,
            d_shape_pidx,
            d_pdf_aux_f64,
            n_params: data.n_params,
            n_procs: data.processes.len(),
            total_shape_params: data.shape_param_indices.len(),
            total_rate_mods: data.rate_modifiers.len(),
            total_pdf_aux_f64: data.pdf_aux_f64.len(),
        })
    }

    /// Returns the CUDA context used by this sampler.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the CUDA stream used by this sampler.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Run Kernel A (Poisson counts) + host prefix sum + Kernel B (sample observables).
    ///
    /// Returns `(toy_offsets, d_obs_flat)` where `d_obs_flat` remains on the GPU device.
    fn sample_toys_1d_inner(
        &mut self,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> ns_core::Result<(Vec<u32>, Option<CudaSlice<f64>>)> {
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params length mismatch: expected {}, got {}",
                self.n_params,
                params.len()
            )));
        }
        if n_toys == 0 {
            return Err(ns_core::Error::Validation("n_toys must be > 0".into()));
        }

        let d_params = self.stream.clone_htod(params).map_err(cuda_err)?;
        let mut d_counts = self.stream.alloc_zeros::<u32>(n_toys).map_err(cuda_err)?;

        // Kernel A: Poisson counts per toy.
        let block = 256u32;
        let grid = ((n_toys as u32) + block - 1) / block;
        let config =
            LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

        let n_procs = self.n_procs as u32;
        let total_rate_mods = self.total_rate_mods as u32;
        let n_toys_u32 = n_toys as u32;
        let seed_u64 = seed as u64;

        let mut builder = self.stream.launch_builder(&self.kernel_counts);
        builder.arg(&d_params);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&n_procs);
        builder.arg(&total_rate_mods);
        builder.arg(&n_toys_u32);
        builder.arg(&seed_u64);
        builder.arg(&mut d_counts);
        // SAFETY: All device pointers are valid CudaSlice allocations, scalar args
        // match the compiled kernel signature, launch config is within hardware limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_toy_counts: {e}")))?;
        }

        // D2H counts (small: n_toys × 4 bytes).
        let mut counts = vec![0u32; n_toys];
        self.stream.memcpy_dtoh(&d_counts, &mut counts).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        // Host prefix sum -> toy_offsets.
        let mut toy_offsets = Vec::<u32>::with_capacity(n_toys + 1);
        toy_offsets.push(0u32);
        let mut total_events: u64 = 0;
        for &c in &counts {
            total_events = total_events.saturating_add(c as u64);
            if total_events > (u32::MAX as u64) {
                return Err(ns_core::Error::Validation(format!(
                    "total toy events overflow u32: total_events={total_events} (n_toys={n_toys})"
                )));
            }
            toy_offsets.push(total_events as u32);
        }

        let total_events_usize = total_events as usize;
        if total_events_usize == 0 {
            return Ok((toy_offsets, None));
        }

        // H2D toy_offsets (small: (n_toys+1) × 4 bytes).
        let d_toy_offsets = self.stream.clone_htod(&toy_offsets).map_err(cuda_err)?;
        let mut d_obs_flat =
            self.stream.alloc_zeros::<f64>(total_events_usize).map_err(cuda_err)?;

        // Kernel B: sample observables for all toys into `obs_flat`.
        let config = LaunchConfig {
            grid_dim: (n_toys_u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let total_shape_params = self.total_shape_params as u32;
        let total_pdf_aux_f64 = self.total_pdf_aux_f64 as u32;

        let mut builder = self.stream.launch_builder(&self.kernel_sample);
        builder.arg(&d_params);
        builder.arg(&self.d_obs_lo);
        builder.arg(&self.d_obs_hi);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&self.d_shape_pidx);
        builder.arg(&self.d_pdf_aux_f64);
        builder.arg(&n_procs);
        builder.arg(&total_rate_mods);
        builder.arg(&total_shape_params);
        builder.arg(&total_pdf_aux_f64);
        builder.arg(&d_toy_offsets);
        builder.arg(&n_toys_u32);
        builder.arg(&seed_u64);
        builder.arg(&mut d_obs_flat);
        // SAFETY: Same invariants — valid device pointers, matching kernel signature,
        // launch config within limits. `d_toy_offsets` computed from `d_counts` above.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_toy_sample_obs_1d: {e}")))?;
        }

        Ok((toy_offsets, Some(d_obs_flat)))
    }

    /// Sample toys on CUDA (1D only), returning all data on the host.
    ///
    /// Returns:
    /// - `toy_offsets`: prefix sums (length `n_toys+1`)
    /// - `obs_flat`: concatenated events (length `toy_offsets[n_toys]`)
    pub fn sample_toys_1d(
        &mut self,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> ns_core::Result<(Vec<u32>, Vec<f64>)> {
        let (toy_offsets, d_obs_flat) = self.sample_toys_1d_inner(params, n_toys, seed)?;

        let total = *toy_offsets.last().unwrap_or(&0) as usize;
        let obs_flat = match d_obs_flat {
            Some(d) => {
                let mut host = vec![0.0f64; total];
                self.stream.memcpy_dtoh(&d, &mut host).map_err(cuda_err)?;
                self.stream.synchronize().map_err(cuda_err)?;
                host
            }
            None => Vec::new(),
        };

        Ok((toy_offsets, obs_flat))
    }

    /// Sample toys on CUDA (1D only), keeping `obs_flat` on the GPU device.
    ///
    /// This avoids the D2H copy of the large `obs_flat` buffer. The returned `CudaSlice` can
    /// be passed directly to [`CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device`]
    /// when both share the same CUDA context.
    ///
    /// Returns:
    /// - `toy_offsets`: prefix sums on host (length `n_toys+1`)
    /// - `d_obs_flat`: concatenated events on GPU (length `toy_offsets[n_toys]`)
    pub fn sample_toys_1d_device(
        &mut self,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> ns_core::Result<(Vec<u32>, CudaSlice<f64>)> {
        let (toy_offsets, d_obs_flat) = self.sample_toys_1d_inner(params, n_toys, seed)?;

        let total = *toy_offsets.last().unwrap_or(&0) as usize;
        let d_obs_flat = match d_obs_flat {
            Some(d) => d,
            None => self.stream.alloc_zeros::<f64>(total.max(1)).map_err(cuda_err)?,
        };

        self.stream.synchronize().map_err(cuda_err)?;
        Ok((toy_offsets, d_obs_flat))
    }
}
