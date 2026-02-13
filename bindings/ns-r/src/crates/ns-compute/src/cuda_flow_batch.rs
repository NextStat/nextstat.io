//! CUDA batch toy NLL accelerator for flow (externally-computed log-prob) PDFs.
//!
//! Wraps the `flow_batch_nll_reduce` kernel: 1 CUDA block = 1 toy dataset.
//! Toy event counts may vary; datasets are stored as a flattened log-prob array
//! plus prefix-sum offsets.
//!
//! For gradient computation, this uses central finite differences on the batch NLL
//! kernel (`2 × n_params + 1` launches per gradient evaluation), where each launch
//! computes NLL for all toys in parallel.

use crate::unbinned_types::GpuUnbinnedGaussConstraintEntry;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/flow_nll_reduce.cu` at build time.
const PTX_SRC: &str = include_str!(env!("CUDA_FLOW_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA (flow batch): {msg}"))
}

/// Process descriptor for batch toy flow NLL.
///
/// Describes how yields are computed from parameters for each process.
/// Identical across all toys (the model structure is shared).
#[derive(Debug, Clone)]
pub struct FlowBatchProcessDesc {
    /// Base yield (before rate modifiers).
    pub base_yield: f64,
    /// Global parameter index for yield (if yield is a parameter or scaled).
    pub yield_param_idx: Option<usize>,
    /// If `true`, yield = `base_yield * params[yield_param_idx]`.
    /// If `false` and `yield_param_idx` is Some, yield = `params[yield_param_idx]`.
    pub yield_is_scaled: bool,
}

/// Configuration for the flow batch NLL accelerator.
pub struct FlowBatchNllConfig {
    /// Total number of events across all toys.
    pub total_events: usize,
    /// Number of toys.
    pub n_toys: usize,
    /// Per-toy event offsets: `toy_offsets[t]..toy_offsets[t+1]` are the event
    /// indices for toy `t`. Length = `n_toys + 1`.
    pub toy_offsets: Vec<u32>,
    /// Process descriptors (shared across all toys).
    pub processes: Vec<FlowBatchProcessDesc>,
    /// Number of global parameters.
    pub n_params: usize,
    /// Gaussian constraint entries.
    pub gauss_constraints: Vec<GpuUnbinnedGaussConstraintEntry>,
    /// Constant term from constraints.
    pub constraint_const: f64,
}

/// CUDA batch toy NLL accelerator for flow PDFs.
///
/// Pre-computed `log p(x|θ)` values are uploaded once; only yields and params
/// change per optimizer iteration. The `flow_batch_nll_reduce` kernel evaluates
/// NLL for all toys in a single launch (1 block = 1 toy).
pub struct CudaFlowBatchNllAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_batch_nll: CudaFunction,

    // --- Static buffers (uploaded once) ---
    d_logp_flat: CudaSlice<f64>,
    d_toy_offsets: CudaSlice<u32>,
    d_gauss: CudaSlice<GpuUnbinnedGaussConstraintEntry>,

    // --- Dynamic buffers (per iteration) ---
    d_yields_flat: CudaSlice<f64>,
    d_params_flat: CudaSlice<f64>,
    d_nll_out: CudaSlice<f64>,

    // --- Model description ---
    processes: Vec<FlowBatchProcessDesc>,
    n_toys: usize,
    n_procs: usize,
    n_params: usize,
    n_gauss: usize,
    constraint_const: f64,
}

impl CudaFlowBatchNllAccelerator {
    /// Check if CUDA is available at runtime.
    pub fn is_available() -> bool {
        std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    /// Create a new batch NLL accelerator on the specified CUDA device.
    ///
    /// `logp_flat` is `[n_procs × total_events]` row-major: all per-process log-prob
    /// values for all toy events concatenated. This buffer is uploaded to GPU once.
    pub fn new(
        config: &FlowBatchNllConfig,
        logp_flat: &[f64],
        device_id: usize,
    ) -> ns_core::Result<Self> {
        let n_procs = config.processes.len();
        let n_toys = config.n_toys;
        let n_params = config.n_params;

        if n_procs == 0 {
            return Err(ns_core::Error::Validation("n_procs must be > 0".into()));
        }
        if n_toys == 0 {
            return Err(ns_core::Error::Validation("n_toys must be > 0".into()));
        }
        if config.toy_offsets.len() != n_toys + 1 {
            return Err(ns_core::Error::Validation(format!(
                "toy_offsets length {}: expected {}",
                config.toy_offsets.len(),
                n_toys + 1
            )));
        }
        let expected_logp = n_procs * config.total_events;
        if logp_flat.len() != expected_logp {
            return Err(ns_core::Error::Validation(format!(
                "logp_flat length {}: expected {}",
                logp_flat.len(),
                expected_logp
            )));
        }

        let ctx = match std::panic::catch_unwind(|| CudaContext::new(device_id)) {
            Ok(Ok(ctx)) => ctx,
            Ok(Err(e)) => return Err(cuda_err(format!("context (device {device_id}): {e}"))),
            Err(_) => return Err(cuda_err("context: CUDA driver library not available")),
        };
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel_batch_nll = module
            .load_function("flow_batch_nll_reduce")
            .map_err(|e| cuda_err(format!("load flow_batch_nll_reduce: {e}")))?;

        // Static buffers — uploaded once.
        let d_logp_flat = stream.clone_htod(logp_flat).map_err(cuda_err)?;
        let d_toy_offsets = stream.clone_htod(&config.toy_offsets).map_err(cuda_err)?;
        let d_gauss = if config.gauss_constraints.is_empty() {
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

        // Dynamic buffers.
        let d_yields_flat = stream.alloc_zeros::<f64>(n_toys * n_procs).map_err(cuda_err)?;
        let d_params_flat =
            stream.alloc_zeros::<f64>((n_toys * n_params).max(1)).map_err(cuda_err)?;
        let d_nll_out = stream.alloc_zeros::<f64>(n_toys).map_err(cuda_err)?;

        Ok(Self {
            ctx,
            stream,
            kernel_batch_nll,
            d_logp_flat,
            d_toy_offsets,
            d_gauss,
            d_yields_flat,
            d_params_flat,
            d_nll_out,
            processes: config.processes.clone(),
            n_toys,
            n_procs,
            n_params,
            n_gauss: config.gauss_constraints.len(),
            constraint_const: config.constraint_const,
        })
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of toys.
    pub fn n_toys(&self) -> usize {
        self.n_toys
    }

    /// Compute yields for all toys from `params_flat[n_toys × n_params]`.
    ///
    /// Returns `yields_flat[n_toys × n_procs]` row-major.
    fn compute_yields_flat(&self, params_flat: &[f64]) -> Vec<f64> {
        let mut yields = Vec::with_capacity(self.n_toys * self.n_procs);
        for t in 0..self.n_toys {
            let p = &params_flat[t * self.n_params..(t + 1) * self.n_params];
            for proc in &self.processes {
                let y = match proc.yield_param_idx {
                    None => proc.base_yield,
                    Some(idx) => {
                        if proc.yield_is_scaled {
                            proc.base_yield * p[idx]
                        } else {
                            p[idx]
                        }
                    }
                };
                yields.push(y);
            }
        }
        yields
    }

    /// Compute batch NLL for all toys.
    ///
    /// `params_flat` is `[n_toys × n_params]` row-major.
    /// Returns `nll_out[n_toys]`.
    pub fn batch_nll(&mut self, params_flat: &[f64]) -> ns_core::Result<Vec<f64>> {
        let expected = self.n_toys * self.n_params;
        if params_flat.len() != expected {
            return Err(ns_core::Error::Validation(format!(
                "params_flat length {}: expected {}",
                params_flat.len(),
                expected
            )));
        }

        let yields_flat = self.compute_yields_flat(params_flat);

        // H2D transfers (dynamic data only).
        self.stream.memcpy_htod(&yields_flat, &mut self.d_yields_flat).map_err(cuda_err)?;
        if self.n_params > 0 {
            self.stream.memcpy_htod(params_flat, &mut self.d_params_flat).map_err(cuda_err)?;
        }

        // Launch: 1 block per toy, 256 threads per block.
        let block_size = 256u32;
        let shared_bytes = (block_size as usize * std::mem::size_of::<f64>()) as u32;
        let config = LaunchConfig {
            grid_dim: (self.n_toys as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let n_pr = self.n_procs as u32;
        let n_pa = self.n_params as u32;
        let n_ga = self.n_gauss as u32;
        let cc = self.constraint_const;
        let n_t = self.n_toys as u32;

        let mut builder = self.stream.launch_builder(&self.kernel_batch_nll);
        builder.arg(&self.d_logp_flat);
        builder.arg(&self.d_toy_offsets);
        builder.arg(&self.d_yields_flat);
        builder.arg(&self.d_gauss);
        builder.arg(&self.d_params_flat);
        builder.arg(&mut self.d_nll_out);
        builder.arg(&n_pr);
        builder.arg(&n_pa);
        builder.arg(&n_ga);
        builder.arg(&cc);
        builder.arg(&n_t);

        // SAFETY: All device pointers are valid CudaSlice allocations owned by `self`,
        // scalar args match the kernel signature, launch config is within hardware limits.
        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch flow_batch_nll_reduce: {e}")))?;
        }

        // D2H: read NLL values for all toys.
        let mut nll_out = vec![0.0f64; self.n_toys];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok(nll_out)
    }

    /// Compute batch NLL + gradient for all toys via central finite differences.
    ///
    /// `params_flat` is `[n_toys × n_params]` row-major.
    /// Returns `(nll[n_toys], grad[n_toys × n_params])` where grad is row-major.
    ///
    /// Performs `2 × n_params + 1` kernel launches (each processes all toys in parallel).
    pub fn batch_nll_grad(&mut self, params_flat: &[f64]) -> ns_core::Result<(Vec<f64>, Vec<f64>)> {
        let nll_center = self.batch_nll(params_flat)?;

        let eps = 1e-4;
        let inv_2eps = 1.0 / (2.0 * eps);
        let n = self.n_toys;
        let np = self.n_params;
        let mut grad = vec![0.0f64; n * np];
        let mut params_work = params_flat.to_vec();

        for j in 0..np {
            // +eps for all toys
            for t in 0..n {
                params_work[t * np + j] += eps;
            }
            let nll_plus = self.batch_nll(&params_work)?;

            // -2*eps (to get -eps from original)
            for t in 0..n {
                params_work[t * np + j] -= 2.0 * eps;
            }
            let nll_minus = self.batch_nll(&params_work)?;

            // Restore
            for t in 0..n {
                params_work[t * np + j] += eps;
            }

            // Compute gradient
            for t in 0..n {
                grad[t * np + j] = (nll_plus[t] - nll_minus[t]) * inv_2eps;
            }
        }

        Ok((nll_center, grad))
    }
}
