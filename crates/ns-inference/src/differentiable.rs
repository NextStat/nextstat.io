//! Differentiable NLL session for PyTorch integration.
//!
//! Wraps `DifferentiableAccelerator` from ns-compute with model-level logic
//! (serialization, observed data upload, signal sample lookup).

use ns_compute::differentiable::{DifferentiableAccelerator, SignalSampleInfo};
use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use statrs::function::gamma::ln_gamma;

/// GPU session for differentiable NLL computation with PyTorch zero-copy.
///
/// Created once per model+signal combination. Holds all GPU state.
/// Thread-safety: NOT thread-safe (single CUDA stream).
pub struct DifferentiableSession {
    accel: DifferentiableAccelerator,
    n_params: usize,
    signal_n_bins: usize,
    init_params: Vec<f64>,
}

impl DifferentiableSession {
    /// Create a new session for the given model and signal sample.
    ///
    /// Serializes the model, uploads it to GPU, and uploads observed data.
    pub fn new(model: &HistFactoryModel, signal_sample_name: &str) -> Result<Self> {
        let (flat_idx, first_bin, n_bins) = model.signal_sample_gpu_info(signal_sample_name)?;
        let signal_info = SignalSampleInfo {
            sample_idx: flat_idx,
            first_bin,
            n_bins,
        };

        let gpu_data = model.serialize_for_gpu()?;
        let n_params = gpu_data.n_params;
        let mut accel = DifferentiableAccelerator::from_gpu_data(&gpu_data, signal_info)?;

        // Upload observed data
        let (observed, ln_facts, obs_mask) = Self::prepare_observed(model);
        accel.upload_observed(&observed, &ln_facts, &obs_mask)?;

        let init_params = model.parameter_init();

        Ok(Self {
            accel,
            n_params,
            signal_n_bins: n_bins as usize,
            init_params,
        })
    }

    /// Compute NLL + write signal gradient into PyTorch tensor (zero-copy).
    ///
    /// `signal_ptr` and `grad_signal_ptr` are raw CUDA device pointers
    /// from `torch.Tensor.data_ptr()`.
    ///
    /// Returns NLL scalar. The gradient w.r.t. signal bins is written
    /// directly into `grad_signal_ptr`.
    pub fn nll_grad_signal(
        &mut self,
        params: &[f64],
        signal_ptr: u64,
        grad_signal_ptr: u64,
    ) -> Result<f64> {
        let (nll, _grad_params) =
            self.accel.nll_grad_wrt_signal(params, signal_ptr, grad_signal_ptr)?;
        Ok(nll)
    }

    /// Number of signal bins.
    pub fn signal_n_bins(&self) -> usize {
        self.signal_n_bins
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Default initial parameter values.
    pub fn parameter_init(&self) -> &[f64] {
        &self.init_params
    }

    fn prepare_observed(model: &HistFactoryModel) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut observed = Vec::new();
        let mut ln_facts = Vec::new();
        let mut obs_mask = Vec::new();

        for ch_data in model.observed_main_by_channel() {
            for &obs in &ch_data.y {
                observed.push(obs);
                ln_facts.push(ln_gamma(obs + 1.0));
                obs_mask.push(if obs > 0.0 { 1.0 } else { 0.0 });
            }
        }

        (observed, ln_facts, obs_mask)
    }
}
