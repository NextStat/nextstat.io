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
//!
//! # Multi-Channel Signal
//!
//! The signal sample may appear in multiple channels. Each occurrence is
//! described by a `SignalSampleInfo` entry. The external signal buffer is
//! laid out as `[ch0_bins..., ch1_bins..., ...]` (concatenated).

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

/// Where the signal gradient should be written.
enum GradSignalTarget<'a> {
    /// External raw CUDA pointer (PyTorch zero-copy).
    External(u64),
    /// Internal pre-allocated GPU buffer.
    Internal(&'a CudaSlice<f64>),
}

/// Launch the differentiable NLL+gradient kernel (multi-channel signal).
///
/// Free function so callers can pass individual struct fields without
/// `&mut self` / `&self` borrow conflicts.
#[allow(clippy::too_many_arguments)]
fn launch_kernel(
    stream: &CudaStream,
    kernel: &CudaFunction,
    // Dynamic per-iteration
    d_params: &CudaSlice<f64>,
    d_observed: &CudaSlice<f64>,
    d_ln_facts: &CudaSlice<f64>,
    d_obs_mask: &CudaSlice<f64>,
    // Static model
    d_nominal: &CudaSlice<f64>,
    d_samples: &CudaSlice<GpuSampleInfo>,
    d_modifier_descs: &CudaSlice<GpuModifierDesc>,
    d_modifier_desc_offsets: &CudaSlice<u32>,
    d_per_bin_param_indices: &CudaSlice<u32>,
    d_modifier_data: &CudaSlice<f64>,
    d_aux_poisson: &CudaSlice<GpuAuxPoissonEntry>,
    d_gauss_constr: &CudaSlice<GpuGaussConstraintEntry>,
    // Output
    d_nll_out: &CudaSlice<f64>,
    d_grad_params_out: &CudaSlice<f64>,
    // Signal
    signal_device_ptr: u64,
    grad_target: GradSignalTarget<'_>,
    // Multi-channel signal arrays
    d_signal_indices: &CudaSlice<u32>,
    d_signal_first_bins: &CudaSlice<u32>,
    d_signal_n_bins_arr: &CudaSlice<u32>,
    n_signal_entries: usize,
    // Metadata
    n_params: usize,
    n_main_bins: usize,
    n_samples: usize,
    n_aux_poisson: usize,
    n_gauss_constr: usize,
    constraint_const: f64,
) -> ns_core::Result<()> {
    let block_size = (n_main_bins.min(256) as u32).next_power_of_two();
    let shared_bytes = ((n_params + block_size as usize) * std::mem::size_of::<f64>()) as u32;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    // Pre-compute scalar args as local variables (LaunchArgs borrows them)
    let nse = n_signal_entries as u32;
    let np = n_params as u32;
    let nb = n_main_bins as u32;
    let ns = n_samples as u32;
    let nap = n_aux_poisson as u32;
    let ngc = n_gauss_constr as u32;

    // For GradSignalTarget::External, we need the ptr to live long enough
    let mut grad_ptr_val: u64 = 0;

    let mut builder = stream.launch_builder(kernel);

    builder.arg(d_params);
    builder.arg(d_observed);
    builder.arg(d_ln_facts);
    builder.arg(d_obs_mask);

    builder.arg(d_nominal);
    builder.arg(d_samples);
    builder.arg(d_modifier_descs);
    builder.arg(d_modifier_desc_offsets);
    builder.arg(d_per_bin_param_indices);
    builder.arg(d_modifier_data);
    builder.arg(d_aux_poisson);
    builder.arg(d_gauss_constr);

    builder.arg(d_nll_out);
    builder.arg(d_grad_params_out);

    builder.arg(&signal_device_ptr);
    match grad_target {
        GradSignalTarget::External(ptr) => {
            grad_ptr_val = ptr;
            builder.arg(&grad_ptr_val);
        }
        GradSignalTarget::Internal(buf) => {
            builder.arg(buf);
        }
    }

    // Multi-channel signal entry arrays
    builder.arg(d_signal_indices);
    builder.arg(d_signal_first_bins);
    builder.arg(d_signal_n_bins_arr);
    builder.arg(&nse);

    builder.arg(&np);
    builder.arg(&nb);
    builder.arg(&ns);
    builder.arg(&nap);
    builder.arg(&ngc);
    builder.arg(&constraint_const);

    unsafe {
        builder
            .launch(config)
            .map_err(|e| cuda_err(format!("launch differentiable_nll_grad: {e}")))?;
    }

    Ok(())
}

/// GPU accelerator for differentiable NLL with PyTorch zero-copy.
///
/// Static model buffers are uploaded once at creation. Per-evaluation,
/// only nuisance parameters are uploaded (H→D). Signal histogram and
/// gradient tensor are accessed via raw CUDA pointers from PyTorch.
///
/// Supports multi-channel signal: the signal sample may appear in multiple
/// channels, with each occurrence tracked by a `SignalSampleInfo` entry.
pub struct DifferentiableAccelerator {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
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

    // Pre-allocated buffer for signal gradient (used by nll_and_grad_params
    // when we only need NLL + grad_params and no external PyTorch pointer).
    d_grad_signal_buf: CudaSlice<f64>,

    // Multi-channel signal entry GPU buffers
    d_signal_indices: CudaSlice<u32>,
    d_signal_first_bins: CudaSlice<u32>,
    d_signal_n_bins_arr: CudaSlice<u32>,
    n_signal_entries: usize,

    // Pre-allocated host zero buffers (avoids per-iteration allocation)
    zeros_params: Vec<f64>,
    zeros_signal: Vec<f64>,

    // Metadata
    n_params: usize,
    n_main_bins: usize,
    n_samples: usize,
    n_aux_poisson: usize,
    n_gauss_constr: usize,
    constraint_const: f64,

    // Total signal bins (sum of all entries' n_bins)
    total_signal_bins: usize,
}

impl DifferentiableAccelerator {
    /// Create accelerator from pre-serialized GPU model data + signal entries.
    ///
    /// For single-channel models, `signal_entries` has one element.
    /// For multi-channel models, one entry per channel containing the signal sample.
    pub fn from_gpu_data(
        data: &GpuModelData,
        signal_entries: &[SignalSampleInfo],
    ) -> ns_core::Result<Self> {
        assert!(!signal_entries.is_empty(), "at least one signal entry required");

        let total_signal_bins: usize = signal_entries.iter().map(|e| e.n_bins as usize).sum();

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
        let d_grad_signal_buf =
            stream.alloc_zeros::<f64>(total_signal_bins).map_err(|e| cuda_err(e))?;

        // Upload multi-channel signal entry arrays
        let indices: Vec<u32> = signal_entries.iter().map(|e| e.sample_idx).collect();
        let first_bins: Vec<u32> = signal_entries.iter().map(|e| e.first_bin).collect();
        let n_bins_arr: Vec<u32> = signal_entries.iter().map(|e| e.n_bins).collect();

        let d_signal_indices = stream.clone_htod(&indices).map_err(|e| cuda_err(e))?;
        let d_signal_first_bins = stream.clone_htod(&first_bins).map_err(|e| cuda_err(e))?;
        let d_signal_n_bins_arr = stream.clone_htod(&n_bins_arr).map_err(|e| cuda_err(e))?;

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
            d_grad_signal_buf,
            d_signal_indices,
            d_signal_first_bins,
            d_signal_n_bins_arr,
            n_signal_entries: signal_entries.len(),
            zeros_params: vec![0.0f64; data.n_params],
            zeros_signal: vec![0.0f64; total_signal_bins],
            n_params: data.n_params,
            n_main_bins: data.n_main_bins,
            n_samples: data.samples.len(),
            n_aux_poisson: data.aux_poisson.len(),
            n_gauss_constr: data.gauss_constraints.len(),
            constraint_const: data.constraint_const,
            total_signal_bins,
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

    /// Upload params and zero the param gradient buffer.
    fn upload_params_and_zero_grad(&mut self, params: &[f64]) -> ns_core::Result<()> {
        assert_eq!(params.len(), self.n_params);
        self.stream.memcpy_htod(params, &mut self.d_params).map_err(|e| cuda_err(e))?;
        self.stream
            .memcpy_htod(&self.zeros_params, &mut self.d_grad_params_out)
            .map_err(|e| cuda_err(e))?;
        Ok(())
    }

    /// Zero the internal signal gradient buffer.
    fn zero_signal_grad(&mut self) -> ns_core::Result<()> {
        self.stream
            .memcpy_htod(&self.zeros_signal, &mut self.d_grad_signal_buf)
            .map_err(|e| cuda_err(e))?;
        Ok(())
    }

    /// Launch the kernel with the given grad signal target.
    ///
    /// All mutable uploads must be done before calling this.
    fn launch(&self, signal_device_ptr: u64, grad_target: GradSignalTarget<'_>) -> ns_core::Result<()> {
        launch_kernel(
            &self.stream,
            &self.kernel,
            &self.d_params,
            &self.d_observed,
            &self.d_ln_facts,
            &self.d_obs_mask,
            &self.d_nominal,
            &self.d_samples,
            &self.d_modifier_descs,
            &self.d_modifier_desc_offsets,
            &self.d_per_bin_param_indices,
            &self.d_modifier_data,
            &self.d_aux_poisson,
            &self.d_gauss_constr,
            &self.d_nll_out,
            &self.d_grad_params_out,
            signal_device_ptr,
            grad_target,
            &self.d_signal_indices,
            &self.d_signal_first_bins,
            &self.d_signal_n_bins_arr,
            self.n_signal_entries,
            self.n_params,
            self.n_main_bins,
            self.n_samples,
            self.n_aux_poisson,
            self.n_gauss_constr,
            self.constraint_const,
        )
    }

    /// Download NLL + param gradient from GPU.
    fn download_nll_and_grad_params(&self) -> ns_core::Result<(f64, Vec<f64>)> {
        let mut nll_out = [0.0f64; 1];
        let mut grad_params = vec![0.0f64; self.n_params];
        self.stream.memcpy_dtoh(&self.d_nll_out, &mut nll_out).map_err(|e| cuda_err(e))?;
        self.stream
            .memcpy_dtoh(&self.d_grad_params_out, &mut grad_params)
            .map_err(|e| cuda_err(e))?;
        self.stream.synchronize().map_err(|e| cuda_err(e))?;
        Ok((nll_out[0], grad_params))
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
        self.upload_params_and_zero_grad(params)?;
        self.launch(signal_device_ptr, GradSignalTarget::External(grad_signal_device_ptr))?;
        self.download_nll_and_grad_params()
    }

    /// Compute NLL + gradient w.r.t. nuisance parameters only.
    ///
    /// Signal gradient is written into the internal `d_grad_signal_buf` buffer
    /// (discarded). For use during L-BFGS-B fit iterations where only NLL and
    /// grad_params are needed.
    pub fn nll_and_grad_params(
        &mut self,
        params: &[f64],
        signal_device_ptr: u64,
    ) -> ns_core::Result<(f64, Vec<f64>)> {
        self.upload_params_and_zero_grad(params)?;
        self.zero_signal_grad()?;
        self.launch(signal_device_ptr, GradSignalTarget::Internal(&self.d_grad_signal_buf))?;
        self.download_nll_and_grad_params()
    }

    /// Compute NLL + gradient w.r.t. both nuisance parameters and signal bins.
    ///
    /// Like `nll_and_grad_params`, but also downloads the signal gradient
    /// from the internal buffer. Returns `(nll, grad_params, grad_signal)`.
    pub fn nll_grad_all(
        &mut self,
        params: &[f64],
        signal_device_ptr: u64,
    ) -> ns_core::Result<(f64, Vec<f64>, Vec<f64>)> {
        self.upload_params_and_zero_grad(params)?;
        self.zero_signal_grad()?;
        self.launch(signal_device_ptr, GradSignalTarget::Internal(&self.d_grad_signal_buf))?;

        let (nll, grad_params) = self.download_nll_and_grad_params()?;

        let mut grad_signal = vec![0.0f64; self.total_signal_bins];
        self.stream
            .memcpy_dtoh(&self.d_grad_signal_buf, &mut grad_signal)
            .map_err(|e| cuda_err(e))?;
        self.stream.synchronize().map_err(|e| cuda_err(e))?;

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
}
