//! Differentiable NLL accelerator for PyTorch zero-copy integration.
//!
//! Provides `DifferentiableAccelerator` — a GPU accelerator that computes
//! NLL + gradient w.r.t. signal bins, reading/writing directly from/to
//! PyTorch CUDA tensors (zero-copy via raw device pointers).
//!
//! # Zero-Copy Architecture
//!
//! ```text
//! PyTorch Training Loop (GPU)
//!   ├── NN output → signal_histogram (torch.Tensor, CUDA)
//!   │                  ↓ tensor.data_ptr() → u64
//!   ├── DifferentiableAccelerator::nll_grad_wrt_signal()
//!   │   ├── kernel reads signal bins from PyTorch memory
//!   │   ├── kernel writes ∂NLL/∂signal into PyTorch grad tensor
//!   │   └── returns NLL scalar + grad_params
//!   └── backward: grad_signal already on GPU (no D→H→D)
//! ```

use crate::cuda_types::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX source compiled from `kernels/differentiable_nll_grad.cu` at build time.
const DIFF_PTX_SRC: &str = include_str!(env!("CUDA_DIFF_PTX_PATH"));

fn cuda_err(msg: impl std::fmt::Display) -> ns_core::Error {
    ns_core::Error::Computation(format!("CUDA differentiable: {msg}"))
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

/// GPU accelerator for differentiable NLL with PyTorch zero-copy.
///
/// Static model buffers are uploaded once at creation. Per-evaluation,
/// only nuisance parameters are uploaded (H→D). Signal histogram and
/// gradient tensor are accessed via raw CUDA pointers from PyTorch.
pub struct DifferentiableAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: CudaStream,
    kernel: CudaFunction,

    // Static model buffers (same as CudaBatchAccelerator)
    d_nominal: CudaSlice<f64>,
    d_samples: CudaSlice<GpuSampleInfo>,
    d_modifier_descs: CudaSlice<GpuModifierDesc>,
    d_modifier_desc_offsets: CudaSlice<u32>,
    d_per_bin_param_indices: CudaSlice<u32>,
    d_modifier_data: CudaSlice<f64>,
    d_aux_poisson: CudaSlice<GpuAuxPoissonEntry>,
    d_gauss_constr: CudaSlice<GpuGaussConstraintEntry>,

    // Dynamic buffers
    d_params: CudaSlice<f64>,
    d_observed: CudaSlice<f64>,
    d_ln_facts: CudaSlice<f64>,
    d_obs_mask: CudaSlice<f64>,
    d_nll_out: CudaSlice<f64>,
    d_grad_params_out: CudaSlice<f64>,

    // Metadata
    n_params: usize,
    n_main_bins: usize,
    n_samples: usize,
    n_aux_poisson: usize,
    n_gauss_constr: usize,
    constraint_const: f64,

    // Signal sample info
    signal_info: SignalSampleInfo,
}

impl DifferentiableAccelerator {
    /// Create accelerator from pre-serialized GPU model data + signal location.
    pub fn from_gpu_data(
        data: &GpuModelData,
        signal_info: SignalSampleInfo,
    ) -> ns_core::Result<Self> {
        let ctx = CudaContext::new(0).map_err(|e| cuda_err(format!("context: {e}")))?;
        let stream = ctx.default_stream();

        let ptx = Ptx::from_src(DIFF_PTX_SRC);
        let module = ctx.load_module(ptx).map_err(|e| cuda_err(format!("load PTX: {e}")))?;
        let kernel = module
            .load_function("differentiable_nll_grad")
            .map_err(|e| cuda_err(format!("load kernel: {e}")))?;

        // Upload static buffers
        let d_nominal = stream.clone_htod(&data.nominal).map_err(|e| cuda_err(e))?;
        let d_samples = stream.clone_htod(&data.samples).map_err(|e| cuda_err(e))?;
        let d_modifier_descs = stream.clone_htod(&data.modifier_descs).map_err(|e| cuda_err(e))?;
        let d_modifier_desc_offsets =
            stream.clone_htod(&data.modifier_desc_offsets).map_err(|e| cuda_err(e))?;
        let d_per_bin_param_indices =
            stream.clone_htod(&data.per_bin_param_indices).map_err(|e| cuda_err(e))?;
        let d_modifier_data = stream.clone_htod(&data.modifier_data).map_err(|e| cuda_err(e))?;
        let d_aux_poisson = stream.clone_htod(&data.aux_poisson).map_err(|e| cuda_err(e))?;
        let d_gauss_constr = stream.clone_htod(&data.gauss_constraints).map_err(|e| cuda_err(e))?;

        // Dynamic buffers (single model, not batch)
        let d_params = stream.alloc_zeros::<f64>(data.n_params).map_err(|e| cuda_err(e))?;
        let d_observed = stream.alloc_zeros::<f64>(data.n_main_bins).map_err(|e| cuda_err(e))?;
        let d_ln_facts = stream.alloc_zeros::<f64>(data.n_main_bins).map_err(|e| cuda_err(e))?;
        let d_obs_mask = stream.alloc_zeros::<f64>(data.n_main_bins).map_err(|e| cuda_err(e))?;
        let d_nll_out = stream.alloc_zeros::<f64>(1).map_err(|e| cuda_err(e))?;
        let d_grad_params_out = stream.alloc_zeros::<f64>(data.n_params).map_err(|e| cuda_err(e))?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            d_nominal,
            d_samples,
            d_modifier_descs,
            d_modifier_desc_offsets,
            d_per_bin_param_indices,
            d_modifier_data,
            d_aux_poisson,
            d_gauss_constr,
            d_params,
            d_observed,
            d_ln_facts,
            d_obs_mask,
            d_nll_out,
            d_grad_params_out,
            n_params: data.n_params,
            n_main_bins: data.n_main_bins,
            n_samples: data.samples.len(),
            n_aux_poisson: data.aux_poisson.len(),
            n_gauss_constr: data.gauss_constraints.len(),
            constraint_const: data.constraint_const,
            signal_info,
        })
    }

    /// Upload observed data. Called once after creation.
    pub fn upload_observed(
        &mut self,
        observed: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
    ) -> ns_core::Result<()> {
        assert_eq!(observed.len(), self.n_main_bins);
        self.stream.memcpy_htod(observed, &mut self.d_observed).map_err(|e| cuda_err(e))?;
        self.stream.memcpy_htod(ln_facts, &mut self.d_ln_facts).map_err(|e| cuda_err(e))?;
        self.stream.memcpy_htod(obs_mask, &mut self.d_obs_mask).map_err(|e| cuda_err(e))?;
        Ok(())
    }

    /// Compute NLL + gradient w.r.t. signal bins (zero-copy).
    ///
    /// # Arguments
    /// * `params` — nuisance parameter values `[n_params]` (host)
    /// * `signal_device_ptr` — raw CUDA pointer to PyTorch signal tensor (`tensor.data_ptr()`)
    /// * `grad_signal_device_ptr` — raw CUDA pointer to PyTorch grad tensor
    ///
    /// # Safety contract
    /// The buffer at `grad_signal_device_ptr` **must be zeroed** before calling.
    /// The kernel uses `atomicAdd` to accumulate gradients, so non-zero initial
    /// values will corrupt the result. Use `torch.zeros_like(signal)` in Python.
    ///
    /// # Returns
    /// `(nll, grad_params)` — NLL scalar and gradient w.r.t. nuisance parameters.
    /// The gradient w.r.t. signal bins is written directly into `grad_signal_device_ptr`.
    pub fn nll_grad_wrt_signal(
        &mut self,
        params: &[f64],
        signal_device_ptr: u64,
        grad_signal_device_ptr: u64,
    ) -> ns_core::Result<(f64, Vec<f64>)> {
        assert_eq!(params.len(), self.n_params);

        // Upload params
        self.stream.memcpy_htod(params, &mut self.d_params).map_err(|e| cuda_err(e))?;

        // Zero gradient output
        let zeros = vec![0.0f64; self.n_params];
        self.stream.memcpy_htod(&zeros, &mut self.d_grad_params_out).map_err(|e| cuda_err(e))?;

        // Kernel launch: single block, threads = next_power_of_two(min(n_main_bins, 256))
        // Block reduction assumes power-of-2 thread count.
        let block_size = (self.n_main_bins.min(256) as u32).next_power_of_two();
        let shared_bytes =
            ((self.n_params + block_size as usize) * std::mem::size_of::<f64>()) as u32;

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        let mut builder = self.stream.launch_builder(&self.kernel);

        // Dynamic per-iteration
        builder.arg(&self.d_params);
        builder.arg(&self.d_observed);
        builder.arg(&self.d_ln_facts);
        builder.arg(&self.d_obs_mask);

        // Static model
        builder.arg(&self.d_nominal);
        builder.arg(&self.d_samples);
        builder.arg(&self.d_modifier_descs);
        builder.arg(&self.d_modifier_desc_offsets);
        builder.arg(&self.d_per_bin_param_indices);
        builder.arg(&self.d_modifier_data);
        builder.arg(&self.d_aux_poisson);
        builder.arg(&self.d_gauss_constr);

        // Output
        builder.arg(&self.d_nll_out);
        builder.arg(&self.d_grad_params_out);

        // PyTorch zero-copy pointers (passed as raw u64 → kernel interprets as double*)
        builder.arg(&signal_device_ptr);
        builder.arg(&grad_signal_device_ptr);

        // Signal metadata
        builder.arg(&self.signal_info.sample_idx);
        builder.arg(&self.signal_info.first_bin);
        builder.arg(&self.signal_info.n_bins);

        // Scalar metadata
        builder.arg(&(self.n_params as u32));
        builder.arg(&(self.n_main_bins as u32));
        builder.arg(&(self.n_samples as u32));
        builder.arg(&(self.n_aux_poisson as u32));
        builder.arg(&(self.n_gauss_constr as u32));
        builder.arg(&self.constraint_const);

        unsafe {
            builder
                .launch(config)
                .map_err(|e| cuda_err(format!("launch differentiable_nll_grad: {e}")))?;
        }

        // D→H: download NLL + param gradient
        let mut nll_out = [0.0f64; 1];
        let mut grad_params = vec![0.0f64; self.n_params];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(|e| cuda_err(e))?;
        self.stream
            .memcpy_dtoh(&self.d_grad_params_out, &mut grad_params)
            .map_err(|e| cuda_err(e))?;
        self.stream.synchronize().map_err(|e| cuda_err(e))?;

        Ok((nll_out[0], grad_params))
    }

    /// Number of signal bins.
    pub fn signal_n_bins(&self) -> usize {
        self.signal_info.n_bins as usize
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }
}
