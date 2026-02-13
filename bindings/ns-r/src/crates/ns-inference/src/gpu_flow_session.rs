//! GPU-accelerated flow PDF session for unbinned likelihood fits.
//!
//! Orchestrates:
//! 1. Flow PDF evaluation on CPU (via [`ns_unbinned::FlowPdf`]) or CUDA EP
//! 2. NLL reduction on GPU (via [`ns_compute::cuda_flow_nll::CudaFlowNllAccelerator`])
//! 3. Gradient via finite differences (central)
//!
//! # Paths
//!
//! - **CPU flow + GPU reduce (Phase 3 MVP):** Flow `log_prob` evaluated on CPU, uploaded
//!   to GPU for NLL reduction. Already provides speedup for large event counts because
//!   the logsumexp + constraint reduction runs on the GPU.
//! - **CUDA EP flow + GPU reduce (D4 full):** ONNX Runtime CUDA Execution Provider
//!   evaluates the flow on GPU, output stays device-resident, NLL kernel reads directly.
//!   Requires `neural-cuda` feature.

#[cfg(feature = "cuda")]
use ns_compute::cuda_flow_nll::{CudaFlowNllAccelerator, FlowNllConfig};
#[cfg(feature = "cuda")]
use ns_compute::unbinned_types::GpuUnbinnedGaussConstraintEntry;
#[cfg(feature = "cuda")]
use ns_core::Result;

/// A process in the GPU flow session.
///
/// Each process has a PDF (flow or parametric) that evaluates `log p(x|θ)`,
/// a yield expression, and optional rate modifiers.
#[cfg(feature = "cuda")]
pub struct FlowProcessDesc {
    /// Index of this process in the model.
    pub process_index: usize,
    /// Base yield (before rate modifiers).
    pub base_yield: f64,
    /// Global parameter index for yield (if yield is a parameter or scaled).
    pub yield_param_idx: Option<usize>,
    /// If `true`, yield = `base_yield * params[yield_param_idx]`.
    /// If `false` and `yield_param_idx` is Some, yield = `params[yield_param_idx]`.
    pub yield_is_scaled: bool,
    /// Maps context vector positions → global parameter indices.
    /// Length = `n_context`. Entry `k` means `context[k] = params[context_param_indices[k]]`.
    /// Empty for processes without context parameters (parametric PDFs).
    pub context_param_indices: Vec<usize>,
}

/// Configuration for the GPU flow session.
#[cfg(feature = "cuda")]
pub struct GpuFlowSessionConfig {
    /// Process descriptors.
    pub processes: Vec<FlowProcessDesc>,
    /// Number of events in the dataset.
    pub n_events: usize,
    /// Number of global parameters.
    pub n_params: usize,
    /// Number of context parameters per flow process (for analytical Jacobian gradient).
    /// Set to 0 if no analytical gradient model is available.
    pub n_context: usize,
    /// Gaussian constraint entries.
    pub gauss_constraints: Vec<GpuUnbinnedGaussConstraintEntry>,
    /// Constant term from constraints.
    pub constraint_const: f64,
}

/// GPU-accelerated session for unbinned models with flow (external) PDFs.
///
/// The caller is responsible for evaluating `log p(x|θ)` per process per event
/// (e.g., via [`FlowPdf::log_prob_batch`]) and passing the results to [`nll`].
///
/// This session handles:
/// - Yield computation from parameters + rate modifiers
/// - NLL reduction on GPU via [`CudaFlowNllAccelerator`]
/// - Gradient via analytical Jacobian (when available) or central finite differences
#[cfg(feature = "cuda")]
pub struct GpuFlowSession {
    accel: CudaFlowNllAccelerator,
    processes: Vec<FlowProcessDesc>,
    n_events: usize,
    n_procs: usize,
    n_params: usize,
}

#[cfg(feature = "cuda")]
impl GpuFlowSession {
    /// Create a new GPU flow session.
    pub fn new(config: GpuFlowSessionConfig) -> Result<Self> {
        let n_procs = config.processes.len();
        let accel_config = FlowNllConfig {
            n_events: config.n_events,
            n_procs,
            n_params: config.n_params,
            n_context: config.n_context,
            gauss_constraints: config.gauss_constraints,
            constraint_const: config.constraint_const,
        };
        let accel = CudaFlowNllAccelerator::new(&accel_config)?;

        Ok(Self {
            accel,
            processes: config.processes,
            n_events: config.n_events,
            n_procs,
            n_params: config.n_params,
        })
    }

    /// Compute yields from parameters.
    ///
    /// Returns `yields[n_procs]`.
    pub fn compute_yields(&self, params: &[f64]) -> Vec<f64> {
        self.processes
            .iter()
            .map(|proc| {
                let base = proc.base_yield;
                match proc.yield_param_idx {
                    None => base,
                    Some(idx) => {
                        if proc.yield_is_scaled {
                            base * params[idx]
                        } else {
                            params[idx]
                        }
                    }
                }
            })
            .collect()
    }

    /// Compute NLL from pre-computed log-prob values.
    ///
    /// `logp_flat` is `[n_procs × n_events]` row-major.
    /// `params` is `[n_params]` — needed for yield computation + constraints.
    pub fn nll(&mut self, logp_flat: &[f64], params: &[f64]) -> Result<f64> {
        let yields = self.compute_yields(params);
        self.accel.nll(logp_flat, &yields, params)
    }

    /// Compute NLL + gradient via central finite differences.
    ///
    /// `eval_logp` is a closure that takes `params` and returns `logp_flat[n_procs × n_events]`.
    /// This is called `2 * n_params + 1` times (once for NLL, twice per param for gradient).
    pub fn nll_grad<F>(&mut self, params: &[f64], eval_logp: &mut F) -> Result<(f64, Vec<f64>)>
    where
        F: FnMut(&[f64]) -> Result<Vec<f64>>,
    {
        let logp_center = eval_logp(params)?;
        let nll_center = self.nll(&logp_center, params)?;

        let eps = 1e-4;
        let mut grad = vec![0.0; self.n_params];
        let mut params_work = params.to_vec();

        for i in 0..self.n_params {
            let orig = params_work[i];

            params_work[i] = orig + eps;
            let logp_plus = eval_logp(&params_work)?;
            let nll_plus = self.nll(&logp_plus, &params_work)?;

            params_work[i] = orig - eps;
            let logp_minus = eval_logp(&params_work)?;
            let nll_minus = self.nll(&logp_minus, &params_work)?;

            grad[i] = (nll_plus - nll_minus) / (2.0 * eps);
            params_work[i] = orig;
        }

        Ok((nll_center, grad))
    }

    /// Compute NLL + gradient analytically from pre-computed log-prob and Jacobian.
    ///
    /// This replaces `nll_grad` when the analytical Jacobian model is available.
    /// Instead of `2 × n_params + 1` forward passes, it requires exactly **1** forward pass
    /// of the grad model per process + a single fused GPU kernel launch.
    ///
    /// # Arguments
    ///
    /// - `logp_flat`: `[n_procs × n_events]` row-major log-prob values.
    /// - `jac_flat`: `[n_procs × n_events × n_context]` row-major Jacobian
    ///   `∂logpₚ(xᵢ)/∂ctx_k`. Pass empty if `n_context == 0`.
    /// - `params`: `[n_params]` current parameter values.
    /// - `gauss_constraints`: Gaussian constraint entries (for constraint gradient).
    pub fn nll_grad_analytical(
        &mut self,
        logp_flat: &[f64],
        jac_flat: &[f64],
        params: &[f64],
        gauss_constraints: &[GpuUnbinnedGaussConstraintEntry],
    ) -> Result<(f64, Vec<f64>)> {
        let yields = self.compute_yields(params);
        let nc = self.accel.n_context();

        let (nll, sum_r, sum_jr) = self.accel.nll_grad(logp_flat, jac_flat, &yields, params)?;

        // Assemble full gradient on CPU from kernel intermediates.
        let mut grad = vec![0.0f64; self.n_params];

        // 1. Yield parameter contributions:
        //    ∂NLL/∂θⱼ = ∂νₚ/∂θⱼ · (1 − sum_r[p])
        for (p, proc) in self.processes.iter().enumerate() {
            if let Some(yield_idx) = proc.yield_param_idx {
                let dnu_dtheta = if proc.yield_is_scaled { proc.base_yield } else { 1.0 };
                grad[yield_idx] += dnu_dtheta * (1.0 - sum_r[p]);
            }
        }

        // 2. Context parameter contributions:
        //    ∂NLL/∂θⱼ = − Σₚ νₚ · sum_jr[p * n_ctx + k]
        //    where context_param_indices[k] == j
        if nc > 0 {
            for (p, proc) in self.processes.iter().enumerate() {
                let nu_p = yields[p];
                for (k, &global_idx) in proc.context_param_indices.iter().enumerate() {
                    grad[global_idx] -= nu_p * sum_jr[p * nc + k];
                }
            }
        }

        // 3. Gaussian constraint contributions:
        //    ∂NLL/∂θⱼ += (θⱼ − center) · inv_width²
        for gc in gauss_constraints {
            let pidx = gc.param_idx as usize;
            let delta = params[pidx] - gc.center;
            grad[pidx] += delta * gc.inv_width * gc.inv_width;
        }

        Ok((nll, grad))
    }

    /// Compute NLL from a GPU-resident `float` log-prob buffer (zero-copy CUDA EP path).
    ///
    /// `d_logp_flat_ptr` is a raw CUDA device pointer to a contiguous `float`
    /// buffer of shape `[n_procs × n_events]` in row-major order. This is the
    /// direct output of ONNX Runtime CUDA Execution Provider I/O binding —
    /// no device-to-host copy, no f32→f64 conversion.
    ///
    /// `params` is `[n_params]` — needed for yield computation + constraints.
    pub fn nll_device_ptr_f32(&mut self, d_logp_flat_ptr: u64, params: &[f64]) -> Result<f64> {
        let yields = self.compute_yields(params);
        self.accel.nll_device_ptr_f32(d_logp_flat_ptr, &yields, params)
    }

    /// Compute NLL + gradient via central finite differences using the f32 device path.
    ///
    /// `eval_logp_device` is a closure that takes `params` and returns the raw CUDA
    /// device pointer (`u64`) to a `float` log-prob buffer `[n_procs × n_events]`.
    /// The closure is called `2 * n_params + 1` times.
    ///
    /// This is the fully device-resident gradient path: ONNX CUDA EP evaluates
    /// the flow on GPU, the output stays on device, and the NLL kernel reads
    /// the `float*` directly.
    pub fn nll_grad_device_f32<F>(
        &mut self,
        params: &[f64],
        eval_logp_device: &mut F,
    ) -> Result<(f64, Vec<f64>)>
    where
        F: FnMut(&[f64]) -> Result<u64>,
    {
        let ptr_center = eval_logp_device(params)?;
        let nll_center = self.nll_device_ptr_f32(ptr_center, params)?;

        let eps = 1e-4;
        let mut grad = vec![0.0; self.n_params];
        let mut params_work = params.to_vec();

        for i in 0..self.n_params {
            let orig = params_work[i];

            params_work[i] = orig + eps;
            let ptr_plus = eval_logp_device(&params_work)?;
            let nll_plus = self.nll_device_ptr_f32(ptr_plus, &params_work)?;

            params_work[i] = orig - eps;
            let ptr_minus = eval_logp_device(&params_work)?;
            let nll_minus = self.nll_device_ptr_f32(ptr_minus, &params_work)?;

            grad[i] = (nll_plus - nll_minus) / (2.0 * eps);
            params_work[i] = orig;
        }

        Ok((nll_center, grad))
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
