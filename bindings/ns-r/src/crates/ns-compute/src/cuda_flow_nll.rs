//! CUDA NLL reduction accelerator for externally-computed log-prob values (flow PDFs).
//!
//! Unlike [`CudaUnbinnedAccelerator`] which fuses PDF evaluation + NLL reduction in a
//! single kernel, this module takes pre-computed `log p(x|θ)` per process per event and
//! only performs the extended unbinned likelihood reduction on the GPU:
//!
//! ```text
//! NLL = ν_tot - Σ_events log(Σ_procs ν_p * p_p(x_i)) + constraints
//! ```
//!
//! This enables:
//! - Flow PDFs evaluated on CPU ([`FlowPdf`]) or CUDA EP (ONNX Runtime)
//! - Mixed models: some processes use flows, others use parametric PDFs
//! - The log_prob buffer can come from host upload or GPU-resident ONNX output
//!
//! # Usage
//!
//! 1. Evaluate all process PDFs on CPU → get `logp_flat[n_procs × n_events]`
//! 2. Compute yields on CPU → get `yields[n_procs]`
//! 3. Call [`CudaFlowNllAccelerator::nll`] → uploads + reduces on GPU

use crate::unbinned_types::GpuUnbinnedGaussConstraintEntry;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/flow_nll_reduce.cu` at build time.
const PTX_SRC: &str = include_str!(env!("CUDA_FLOW_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (flow NLL): {msg}"))
}

/// CUDA accelerator for NLL reduction from externally-computed log-prob values.
///
/// Designed for unbinned models where one or more processes use flow PDFs
/// (evaluated via ONNX Runtime) rather than built-in parametric PDFs.
pub struct CudaFlowNllAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_reduce: CudaFunction,
    kernel_reduce_f32: CudaFunction,
    kernel_grad_reduce: CudaFunction,
    kernel_grad_reduce_f32: CudaFunction,

    // --- GPU buffers ---
    d_logp_flat: CudaSlice<f64>,
    d_yields: CudaSlice<f64>,
    d_gauss: CudaSlice<GpuUnbinnedGaussConstraintEntry>,
    d_params: CudaSlice<f64>,
    d_nll_out: CudaSlice<f64>,

    // --- Gradient intermediate buffers ---
    /// `[n_procs]` — per-process responsibility sums `Σᵢ exp(logpₚ)/f(xᵢ)`.
    d_sum_r: CudaSlice<f64>,
    /// `[n_procs × n_context]` — Jacobian-weighted responsibility sums.
    d_sum_jr: CudaSlice<f64>,
    /// `[n_procs × n_events × n_context]` — Jacobian buffer for upload.
    d_jac_flat: CudaSlice<f64>,

    // --- Metadata ---
    n_events: usize,
    n_procs: usize,
    n_params: usize,
    n_gauss: usize,
    n_context: usize,
    constraint_const: f64,
}

/// Configuration for the flow NLL accelerator.
pub struct FlowNllConfig {
    /// Number of events in the dataset.
    pub n_events: usize,
    /// Number of processes (PDF components).
    pub n_procs: usize,
    /// Number of global parameters.
    pub n_params: usize,
    /// Number of context parameters per flow process (for analytical Jacobian gradient).
    /// Set to 0 if no analytical gradient model is available.
    pub n_context: usize,
    /// Gaussian constraint entries.
    pub gauss_constraints: Vec<GpuUnbinnedGaussConstraintEntry>,
    /// Constant term from constraints: Σ (ln σ + 0.5 ln 2π).
    pub constraint_const: f64,
}

impl CudaFlowNllAccelerator {
    /// Check if CUDA is available at runtime.
    pub fn is_available() -> bool {
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Create a new flow NLL accelerator.
    pub fn new(config: &FlowNllConfig) -> ns_core::Result<Self> {
        if config.n_events == 0 {
            return Err(ns_core::Error::Validation("n_events must be > 0".into()));
        }
        if config.n_procs == 0 {
            return Err(ns_core::Error::Validation("n_procs must be > 0".into()));
        }

        let ctx = match std::panic::catch_unwind(|| CudaContext::new(0)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context: {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel_reduce = module
            .load_function("flow_nll_reduce")
            .map_err(|e| cuda_err(format!("load flow_nll_reduce: {e}")))?;
        let kernel_reduce_f32 = module
            .load_function("flow_nll_reduce_f32")
            .map_err(|e| cuda_err(format!("load flow_nll_reduce_f32: {e}")))?;
        let kernel_grad_reduce = module
            .load_function("flow_nll_grad_reduce")
            .map_err(|e| cuda_err(format!("load flow_nll_grad_reduce: {e}")))?;
        let kernel_grad_reduce_f32 = module
            .load_function("flow_nll_grad_reduce_f32")
            .map_err(|e| cuda_err(format!("load flow_nll_grad_reduce_f32: {e}")))?;

        // Allocate GPU buffers.
        let d_logp_flat =
            stream.alloc_zeros::<f64>(config.n_procs * config.n_events).map_err(cuda_err)?;
        let d_yields = stream.alloc_zeros::<f64>(config.n_procs).map_err(cuda_err)?;
        let d_gauss = if config.gauss_constraints.is_empty() {
            // Allocate a dummy single-entry buffer (kernel won't read it when n_gauss=0).
            let dummy = GpuUnbinnedGaussConstraintEntry {
                center: 0.0,
                inv_width: 0.0,
                param_idx: 0,
                _pad: 0,
            };
            stream.clone_htod(&[dummy]).map_err(cuda_err)?
        } else {
            stream.clone_htod(&config.gauss_constraints).map_err(cuda_err)?
        };
        let d_params = stream.alloc_zeros::<f64>(config.n_params.max(1)).map_err(cuda_err)?;
        let d_nll_out = stream.alloc_zeros::<f64>(1).map_err(cuda_err)?;

        // Gradient intermediate buffers.
        let nc = config.n_context;
        let d_sum_r = stream.alloc_zeros::<f64>(config.n_procs.max(1)).map_err(cuda_err)?;
        let d_sum_jr = stream.alloc_zeros::<f64>((config.n_procs * nc).max(1)).map_err(cuda_err)?;
        let d_jac_flat = stream
            .alloc_zeros::<f64>((config.n_procs * config.n_events * nc).max(1))
            .map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel_reduce,
            kernel_reduce_f32,
            kernel_grad_reduce,
            kernel_grad_reduce_f32,
            d_logp_flat,
            d_yields,
            d_gauss,
            d_params,
            d_nll_out,
            d_sum_r,
            d_sum_jr,
            d_jac_flat,
            n_events: config.n_events,
            n_procs: config.n_procs,
            n_params: config.n_params,
            n_gauss: config.gauss_constraints.len(),
            n_context: nc,
            constraint_const: config.constraint_const,
        })
    }

    /// Compute NLL from pre-computed log-prob values.
    ///
    /// - `logp_flat`: `[n_procs × n_events]` row-major — `logp_flat[p * n_events + i]`
    ///   is `log p_p(x_i | θ)` for process `p` and event `i`.
    /// - `yields`: `[n_procs]` — per-process yields `ν_p` (after rate modifiers).
    /// - `params`: `[n_params]` — current parameter values (for constraint evaluation).
    pub fn nll(
        &mut self,
        logp_flat: &[f64],
        yields: &[f64],
        params: &[f64],
    ) -> ns_core::Result<f64> {
        if logp_flat.len() != self.n_procs * self.n_events {
            return Err(ns_core::Error::Validation(format!(
                "logp_flat length {}: expected {}",
                logp_flat.len(),
                self.n_procs * self.n_events
            )));
        }
        if yields.len() != self.n_procs {
            return Err(ns_core::Error::Validation(format!(
                "yields length {}: expected {}",
                yields.len(),
                self.n_procs
            )));
        }
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params length {}: expected {}",
                params.len(),
                self.n_params
            )));
        }

        // H2D transfers.
        self.stream.memcpy_htod(logp_flat, &mut self.d_logp_flat).map_err(cuda_err)?;
        self.stream.memcpy_htod(yields, &mut self.d_yields).map_err(cuda_err)?;
        if self.n_params > 0 {
            self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        }

        // Launch kernel.
        let block_size = 256u32;
        let shared_bytes = (block_size as usize * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let n_ev = self.n_events as u32;
        let n_pr = self.n_procs as u32;
        let n_ga = self.n_gauss as u32;
        let cc = self.constraint_const;

        let mut builder = self.stream.launch_builder(&self.kernel_reduce);
        builder.arg(&self.d_logp_flat);
        builder.arg(&self.d_yields);
        builder.arg(&self.d_gauss);
        builder.arg(&self.d_params);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&n_ev);
        builder.arg(&n_pr);
        builder.arg(&n_ga);
        builder.arg(&cc);

        // SAFETY: All device pointers are valid CudaSlice allocations owned by `self`,
        // scalar args match the kernel signature, and launch config is within hardware limits.
        unsafe {
            builder.launch(config).map_err(|e| cuda_err(format!("launch flow_nll_reduce: {e}")))?;
        }

        // D2H: read NLL scalar.
        let mut nll_out = [0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok(nll_out[0])
    }

    /// Compute NLL from a GPU-resident log-prob buffer (zero-copy from ONNX CUDA EP).
    ///
    /// The `d_logp_flat` slice must be `[n_procs × n_events]` and live in the same CUDA
    /// context. Yields and params are still uploaded from the host (small buffers).
    pub fn nll_device(
        &mut self,
        d_logp_flat: &CudaSlice<f64>,
        yields: &[f64],
        params: &[f64],
    ) -> ns_core::Result<f64> {
        if yields.len() != self.n_procs {
            return Err(ns_core::Error::Validation(format!(
                "yields length {}: expected {}",
                yields.len(),
                self.n_procs
            )));
        }

        self.stream.memcpy_htod(yields, &mut self.d_yields).map_err(cuda_err)?;
        if self.n_params > 0 {
            self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        }

        let block_size = 256u32;
        let shared_bytes = (block_size as usize * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let n_ev = self.n_events as u32;
        let n_pr = self.n_procs as u32;
        let n_ga = self.n_gauss as u32;
        let cc = self.constraint_const;

        let mut builder = self.stream.launch_builder(&self.kernel_reduce);
        builder.arg(d_logp_flat);
        builder.arg(&self.d_yields);
        builder.arg(&self.d_gauss);
        builder.arg(&self.d_params);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&n_ev);
        builder.arg(&n_pr);
        builder.arg(&n_ga);
        builder.arg(&cc);

        // SAFETY: Same invariants as the host-path launch — valid device pointers,
        // correct scalar args, launch config within limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch flow_nll_reduce (device): {e}")))?;
        }

        let mut nll_out = [0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok(nll_out[0])
    }

    /// Compute NLL from a CUDA device pointer to a `float` log-prob buffer.
    ///
    /// This is the zero-copy path for CUDA EP (ONNX Runtime) I/O binding: the `log_prob`
    /// output is typically `float` and resides in CUDA memory owned by ORT.
    ///
    /// `d_logp_flat_ptr` must point to a contiguous `[n_procs × n_events]` buffer in row-major
    /// order: `logp[p * n_events + i]`. The pointer must be valid in the same CUDA context.
    pub fn nll_device_ptr_f32(
        &mut self,
        d_logp_flat_ptr: u64,
        yields: &[f64],
        params: &[f64],
    ) -> ns_core::Result<f64> {
        if yields.len() != self.n_procs {
            return Err(ns_core::Error::Validation(format!(
                "yields length {}: expected {}",
                yields.len(),
                self.n_procs
            )));
        }
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params length {}: expected {}",
                params.len(),
                self.n_params
            )));
        }

        self.stream.memcpy_htod(yields, &mut self.d_yields).map_err(cuda_err)?;
        if self.n_params > 0 {
            self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        }

        let block_size = 256u32;
        let shared_bytes = (block_size as usize * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let n_ev = self.n_events as u32;
        let n_pr = self.n_procs as u32;
        let n_ga = self.n_gauss as u32;
        let cc = self.constraint_const;

        let mut builder = self.stream.launch_builder(&self.kernel_reduce_f32);
        builder.arg(&d_logp_flat_ptr);
        builder.arg(&self.d_yields);
        builder.arg(&self.d_gauss);
        builder.arg(&self.d_params);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&n_ev);
        builder.arg(&n_pr);
        builder.arg(&n_ga);
        builder.arg(&cc);

        // SAFETY: We pass a raw CUDA device pointer (as u64) for the first arg; the kernel
        // expects a `const float*` and the ABI is the same (64-bit). Other args are owned
        // CudaSlice allocations or scalars matching the signature.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch flow_nll_reduce_f32: {e}")))?;
        }

        let mut nll_out = [0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok(nll_out[0])
    }

    /// Compute NLL + analytical gradient intermediates from pre-computed log-prob and Jacobian.
    ///
    /// Returns `(nll, sum_r, sum_jr)`:
    /// - `sum_r[p]`: `Σᵢ exp(logpₚ(xᵢ)) / f(xᵢ)` — per-process responsibility sum
    /// - `sum_jr[p * n_context + k]`: `Σᵢ [exp(logpₚ(xᵢ)) / f(xᵢ)] · ∂logpₚ/∂ctx_k`
    ///
    /// The caller assembles the full gradient on the CPU:
    /// ```text
    /// ∂NLL/∂θⱼ (yield)   = ∂νₚ/∂θⱼ · (1 − sum_r[p])
    /// ∂NLL/∂θⱼ (context) = − Σₚ νₚ · sum_jr[p * n_ctx + k]
    /// ∂NLL/∂θⱼ (constr)  = (θⱼ − center) · inv_width²
    /// ```
    ///
    /// # Arguments
    ///
    /// - `logp_flat`: `[n_procs × n_events]` row-major log-prob values.
    /// - `jac_flat`: `[n_procs × n_events × n_context]` row-major Jacobian. Pass empty if
    ///   `n_context == 0`.
    /// - `yields`: `[n_procs]` per-process yields.
    /// - `params`: `[n_params]` current parameter values (for constraints).
    pub fn nll_grad(
        &mut self,
        logp_flat: &[f64],
        jac_flat: &[f64],
        yields: &[f64],
        params: &[f64],
    ) -> ns_core::Result<(f64, Vec<f64>, Vec<f64>)> {
        let nc = self.n_context;
        let expected_logp = self.n_procs * self.n_events;
        if logp_flat.len() != expected_logp {
            return Err(ns_core::Error::Validation(format!(
                "logp_flat length {}: expected {}",
                logp_flat.len(),
                expected_logp
            )));
        }
        let expected_jac = self.n_procs * self.n_events * nc;
        if nc > 0 && jac_flat.len() != expected_jac {
            return Err(ns_core::Error::Validation(format!(
                "jac_flat length {}: expected {}",
                jac_flat.len(),
                expected_jac
            )));
        }
        if yields.len() != self.n_procs {
            return Err(ns_core::Error::Validation(format!(
                "yields length {}: expected {}",
                yields.len(),
                self.n_procs
            )));
        }
        if params.len() != self.n_params {
            return Err(ns_core::Error::Validation(format!(
                "params length {}: expected {}",
                params.len(),
                self.n_params
            )));
        }

        // H2D transfers.
        self.stream.memcpy_htod(logp_flat, &mut self.d_logp_flat).map_err(cuda_err)?;
        if nc > 0 {
            self.stream.memcpy_htod(jac_flat, &mut self.d_jac_flat).map_err(cuda_err)?;
        }
        self.stream.memcpy_htod(yields, &mut self.d_yields).map_err(cuda_err)?;
        if self.n_params > 0 {
            self.stream.memcpy_htod(params, &mut self.d_params).map_err(cuda_err)?;
        }

        // Shared memory: block_size × (1 + n_procs + n_procs × n_context) × sizeof(f64)
        let block_size = 256u32;
        let smem_elems = block_size as usize * (1 + self.n_procs + self.n_procs * nc);
        let shared_bytes = (smem_elems * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let n_ev = self.n_events as u32;
        let n_pr = self.n_procs as u32;
        let n_ct = nc as u32;
        let n_ga = self.n_gauss as u32;
        let cc = self.constraint_const;

        let mut builder = self.stream.launch_builder(&self.kernel_grad_reduce);
        builder.arg(&self.d_logp_flat);
        builder.arg(&self.d_jac_flat);
        builder.arg(&self.d_yields);
        builder.arg(&self.d_gauss);
        builder.arg(&self.d_params);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&mut self.d_sum_r);
        builder.arg(&mut self.d_sum_jr);
        builder.arg(&n_ev);
        builder.arg(&n_pr);
        builder.arg(&n_ct);
        builder.arg(&n_ga);
        builder.arg(&cc);

        // SAFETY: All device pointers are valid CudaSlice allocations owned by `self`,
        // scalar args match the kernel signature, shared memory within hardware limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch flow_nll_grad_reduce: {e}")))?;
        }

        // D2H: read NLL + intermediates.
        let mut nll_out = [0.0f64; 1];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;

        let mut sum_r = vec![0.0f64; self.n_procs];
        self.stream.memcpy_dtoh(&self.d_sum_r, &mut sum_r).map_err(cuda_err)?;

        let mut sum_jr = vec![0.0f64; (self.n_procs * nc).max(1)];
        if nc > 0 {
            self.stream
                .memcpy_dtoh(&self.d_sum_jr, &mut sum_jr[..self.n_procs * nc])
                .map_err(cuda_err)?;
        }

        self.stream.synchronize().map_err(cuda_err)?;

        Ok((nll_out[0], sum_r, sum_jr))
    }

    /// Number of context parameters per process.
    pub fn n_context(&self) -> usize {
        self.n_context
    }

    /// Number of events.
    pub fn n_events(&self) -> usize {
        self.n_events
    }

    /// Number of processes.
    pub fn n_procs(&self) -> usize {
        self.n_procs
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }
}
