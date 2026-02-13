//! CUDA unbinned (event-level) NLL + gradient accelerator.
//!
//! Manages GPU memory, PTX loading, and kernel launch for fused
//! unbinned mixture NLL + analytical gradient computation.
//!
//! # Architecture
//!
//! ```text
//! 1 CUDA Block = 1 NLL evaluation on one dataset
//! Threads in block = events (grid-stride loop for >256 events)
//! Shared memory: params[n_params] + reduction scratch
//! ```

use crate::unbinned_types::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/unbinned_nll_grad.cu` at build time.
const PTX_SRC: &str = include_str!(env!("CUDA_UNBINNED_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (unbinned): {msg}"))
}

/// CUDA accelerator for unbinned NLL + gradient.
pub struct CudaUnbinnedAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_nll_grad: CudaFunction,
    kernel_nll_only: CudaFunction,

    // --- Static buffers ---
    d_obs_soa: CudaSlice<f64>,
    d_obs_lo: CudaSlice<f64>,
    d_obs_hi: CudaSlice<f64>,
    d_evt_wts: CudaSlice<f64>,
    has_evt_wts: u32,
    d_procs: CudaSlice<GpuUnbinnedProcessDesc>,
    d_rate_mods: CudaSlice<GpuUnbinnedRateModifierDesc>,
    d_shape_pidx: CudaSlice<u32>,
    d_pdf_aux_f64: CudaSlice<f64>,
    d_gauss: CudaSlice<GpuUnbinnedGaussConstraintEntry>,

    // --- Dynamic buffers ---
    d_params: CudaSlice<f64>,
    d_nll_out: CudaSlice<f64>,
    d_grad_out: CudaSlice<f64>,

    // --- Metadata ---
    n_params: usize,
    n_obs: usize,
    n_events: usize,
    n_procs: usize,
    total_rate_mods: usize,
    total_shape_params: usize,
    n_gauss: usize,
    constraint_const: f64,

    // --- CPU scratch (reused per call) ---
    scratch_zeros: Vec<f64>,
}

impl CudaUnbinnedAccelerator {
    /// Check if CUDA is available at runtime (driver loaded, GPU present).
    pub fn is_available() -> bool {
        // `cudarc` can panic when the CUDA driver shared library is missing
        // (e.g. on non-CUDA machines). Treat that as "not available".
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
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

        let ctx = match std::panic::catch_unwind(|| CudaContext::new(0)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context: {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel_nll_grad = module
            .load_function("unbinned_nll_grad")
            .map_err(|e| cuda_err(format!("load unbinned_nll_grad: {e}")))?;
        let kernel_nll_only = module
            .load_function("unbinned_nll_only")
            .map_err(|e| cuda_err(format!("load unbinned_nll_only: {e}")))?;

        let mut obs_lo = Vec::with_capacity(data.n_obs);
        let mut obs_hi = Vec::with_capacity(data.n_obs);
        for &(lo, hi) in &data.obs_bounds {
            obs_lo.push(lo);
            obs_hi.push(hi);
        }
        if obs_lo.len() != data.n_obs || obs_hi.len() != data.n_obs {
            return Err(ns_core::Error::Validation(format!(
                "obs_bounds length mismatch: expected {}, got {}",
                data.n_obs,
                data.obs_bounds.len()
            )));
        }

        // Upload static buffers
        let d_obs_soa = stream.clone_htod(&data.obs_soa).map_err(cuda_err)?;
        let d_obs_lo = stream.clone_htod(&obs_lo).map_err(cuda_err)?;
        let d_obs_hi = stream.clone_htod(&obs_hi).map_err(cuda_err)?;
        let (d_evt_wts, has_evt_wts) = match &data.event_weights {
            Some(w) => {
                if w.len() != data.n_events {
                    return Err(ns_core::Error::Validation(format!(
                        "event_weights length mismatch: expected {}, got {}",
                        data.n_events,
                        w.len()
                    )));
                }
                let mut any_pos = false;
                for &wi in w {
                    if !(wi.is_finite() && wi >= 0.0) {
                        return Err(ns_core::Error::Validation(format!(
                            "invalid event weight (expected finite and >= 0): {wi}"
                        )));
                    }
                    any_pos |= wi > 0.0;
                }
                if !any_pos {
                    return Err(ns_core::Error::Validation(
                        "event_weights must have sum(weights) > 0".into(),
                    ));
                }
                (stream.clone_htod(w).map_err(cuda_err)?, 1u32)
            }
            None => (stream.clone_htod(&[1.0f64]).map_err(cuda_err)?, 0u32),
        };
        let d_procs = stream.clone_htod(&data.processes).map_err(cuda_err)?;
        let d_rate_mods = stream.clone_htod(&data.rate_modifiers).map_err(cuda_err)?;
        let d_shape_pidx = stream.clone_htod(&data.shape_param_indices).map_err(cuda_err)?;
        let d_pdf_aux_f64 = stream.clone_htod(&data.pdf_aux_f64).map_err(cuda_err)?;
        let d_gauss = stream.clone_htod(&data.gauss_constraints).map_err(cuda_err)?;

        // Pre-allocate dynamic buffers
        let d_params = stream.alloc_zeros::<f64>(data.n_params).map_err(cuda_err)?;
        let d_nll_out = stream.alloc_zeros::<f64>(1).map_err(cuda_err)?;
        let d_grad_out = stream.alloc_zeros::<f64>(data.n_params).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel_nll_grad,
            kernel_nll_only,
            d_obs_soa,
            d_obs_lo,
            d_obs_hi,
            d_evt_wts,
            has_evt_wts,
            d_procs,
            d_rate_mods,
            d_shape_pidx,
            d_pdf_aux_f64,
            d_gauss,
            d_params,
            d_nll_out,
            d_grad_out,
            n_params: data.n_params,
            n_obs: data.n_obs,
            n_events: data.n_events,
            n_procs: data.processes.len(),
            total_rate_mods: data.rate_modifiers.len(),
            total_shape_params: data.shape_param_indices.len(),
            n_gauss: data.gauss_constraints.len(),
            constraint_const: data.constraint_const,
            scratch_zeros: vec![0.0; data.n_params],
        })
    }

    fn launch_config(&self) -> (LaunchConfig, u32) {
        let n_threads = self.n_events.clamp(1, 256);
        let block_size = (n_threads as u32).next_power_of_two();
        let shared_bytes =
            ((self.n_params + block_size as usize) * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        (config, block_size)
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

        self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        self.stream.memcpy_htod(&self.scratch_zeros, &mut self.d_grad_out).map_err(cuda_err)?;

        let (config, _block_size) = self.launch_config();

        let np = self.n_params as u32;
        let no = self.n_obs as u32;
        let ne = self.n_events as u32;
        let nprocs = self.n_procs as u32;
        let tmods = self.total_rate_mods as u32;
        let tshape = self.total_shape_params as u32;
        let ng = self.n_gauss as u32;
        let cc = self.constraint_const;

        let mut builder = self.stream.launch_builder(&self.kernel_nll_grad);
        builder.arg(&self.d_params);
        builder.arg(&self.d_obs_soa);
        builder.arg(&self.d_obs_lo);
        builder.arg(&self.d_obs_hi);
        builder.arg(&self.d_evt_wts);
        builder.arg(&self.has_evt_wts);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&self.d_shape_pidx);
        builder.arg(&self.d_pdf_aux_f64);
        builder.arg(&self.d_gauss);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&mut self.d_grad_out);
        builder.arg(&np);
        builder.arg(&no);
        builder.arg(&ne);
        builder.arg(&nprocs);
        builder.arg(&tmods);
        builder.arg(&tshape);
        builder.arg(&ng);
        builder.arg(&cc);

        // SAFETY: All device pointers are valid CudaSlice allocations owned by `self`,
        // scalar args match the compiled kernel signature, launch config is within limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_nll_grad: {e}")))?;
        }

        let mut nll_out = vec![0.0f64; 1];
        let mut grad_out = vec![0.0f64; self.n_params];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.memcpy_dtoh(&self.d_grad_out, &mut grad_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok((nll_out[0], grad_out))
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

        self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;

        let (config, _block_size) = self.launch_config();

        let np = self.n_params as u32;
        let no = self.n_obs as u32;
        let ne = self.n_events as u32;
        let nprocs = self.n_procs as u32;
        let tmods = self.total_rate_mods as u32;
        let tshape = self.total_shape_params as u32;
        let ng = self.n_gauss as u32;
        let cc = self.constraint_const;

        let mut builder = self.stream.launch_builder(&self.kernel_nll_only);
        builder.arg(&self.d_params);
        builder.arg(&self.d_obs_soa);
        builder.arg(&self.d_obs_lo);
        builder.arg(&self.d_obs_hi);
        builder.arg(&self.d_evt_wts);
        builder.arg(&self.has_evt_wts);
        builder.arg(&self.d_procs);
        builder.arg(&self.d_rate_mods);
        builder.arg(&self.d_shape_pidx);
        builder.arg(&self.d_pdf_aux_f64);
        builder.arg(&self.d_gauss);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&np);
        builder.arg(&no);
        builder.arg(&ne);
        builder.arg(&nprocs);
        builder.arg(&tmods);
        builder.arg(&tshape);
        builder.arg(&ng);
        builder.arg(&cc);

        // SAFETY: Same invariants as `unbinned_nll_grad` — valid device pointers,
        // matching scalar args, launch config within hardware limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch unbinned_nll_only: {e}")))?;
        }

        let mut nll_out = vec![0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(nll_out[0])
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }
}
