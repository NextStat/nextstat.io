//! Device-agnostic GPU accelerator trait.
//!
//! This trait defines the minimal surface required by ns-inference’s single-model
//! GPU fit path (GpuSession + fused NLL+gradient objective), independent of the
//! concrete backend (CUDA f64 vs Metal f32).

use ns_core::Result;

/// Minimal interface required for GPU-accelerated single-model fitting.
pub trait GpuAccelerator {
    /// Evaluate NLL and gradient for one parameter point.
    fn single_nll_grad(&mut self, params: &[f64]) -> Result<(f64, Vec<f64>)>;

    /// Upload observed data buffers for a single model (n_toys = 1).
    fn upload_observed_single(
        &mut self,
        observed: &[f64],
        ln_facts: &[f64],
        obs_mask: &[f64],
    ) -> Result<()>;

    /// Number of model parameters.
    fn n_params(&self) -> usize;

    /// Number of main bins.
    fn n_main_bins(&self) -> usize;
}

