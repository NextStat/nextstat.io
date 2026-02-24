//! MamsAccelerator trait — device-agnostic interface for GPU MAMS samplers.
//!
//! Implemented by `CudaMamsAccelerator` (f64, NVIDIA) and `MetalMamsAccelerator` (f32, Apple Silicon).
//! The trait is always available (no feature gate) so `laps.rs` can be generic over backends.

use ns_core::Result;

/// Diagnostics from one MAMS transition.
pub struct TransitionDiagnostics {
    /// Per-chain acceptance flags.
    pub accepted: Vec<i32>,
    /// Per-chain energy errors (ΔV + ΔK).
    pub energy_error: Vec<f64>,
}

/// Result from a batch of transitions (GPU-side accumulation).
pub struct BatchResult {
    /// Positions `[batch_size][n_report × dim]` (flat per-sample).
    pub positions: Vec<Vec<f64>>,
    /// Potentials `[batch_size][n_report]`.
    pub potentials: Vec<Vec<f64>>,
    /// Acceptance flags `[batch_size][n_report]`.
    pub accepted: Vec<Vec<i32>>,
    /// Energy errors `[batch_size][n_report]`.
    pub energy_error: Vec<Vec<f64>>,
    /// Number of kernel launches in this batch.
    pub n_launches: usize,
}

/// Device-agnostic interface for GPU MAMS acceleration.
///
/// All data crosses the host-device boundary as `f64`. Backends that compute
/// in lower precision (e.g. Metal f32) perform conversion internally.
pub trait MamsAccelerator {
    /// Upload chain state from host.
    ///
    /// - `x`: positions `[n_chains × dim]` (row-major)
    /// - `u`: unit velocities `[n_chains × dim]`
    /// - `potential`: potentials `[n_chains]`
    /// - `grad`: gradients `[n_chains × dim]`
    fn upload_state(&mut self, x: &[f64], u: &[f64], potential: &[f64], grad: &[f64])
    -> Result<()>;

    /// Download positions from GPU. Returns `[n_chains × dim]` (row-major).
    fn download_positions(&self) -> Result<Vec<f64>>;

    /// Download potentials from GPU. Returns `[n_chains]`.
    fn download_potentials(&self) -> Result<Vec<f64>>;

    /// Download diagnostics (accepted flags + energy errors).
    fn download_diagnostics(&self) -> Result<TransitionDiagnostics>;

    /// Upload diagonal inverse mass matrix `[dim]`.
    fn set_inv_mass(&mut self, inv_mass: &[f64]) -> Result<()>;

    /// Upload per-chain step sizes `[n_chains]`.
    fn set_per_chain_eps(&mut self, eps_vec: &[f64]) -> Result<()>;

    /// Fill per-chain eps with a uniform value.
    fn set_uniform_eps(&mut self, eps: f64) -> Result<()>;

    /// Auto-dispatch single transition (warp kernel for data-heavy models, scalar otherwise).
    fn transition_auto(&mut self, l: f64, max_leapfrog: usize, enable_mh: bool) -> Result<()>;

    /// Auto-dispatch batch of transitions.
    fn transition_batch_auto(
        &mut self,
        l: f64,
        max_leapfrog: usize,
        batch_size: usize,
        prefer_fused: bool,
    ) -> Result<BatchResult>;

    /// Optional model-aware pre-solve hook.
    ///
    /// Default implementation is a no-op for backends/models that do not
    /// support specialized initialization.
    ///
    /// Returns `Ok(true)` when a pre-solve was applied, `Ok(false)` otherwise.
    fn glm_presolve(&mut self, _max_iters: usize) -> Result<bool> {
        Ok(false)
    }

    /// Configure GPU-side accumulation buffers for batch sampling.
    fn configure_batch(&mut self, batch_size: usize, n_report: usize) -> Result<()>;

    /// Whether the fused multi-step kernel is available.
    fn supports_fused(&self) -> bool;

    /// Whether the warp/simdgroup-cooperative kernel is available and beneficial.
    fn supports_warp(&self) -> bool;

    /// Whether the high-dim warp kernel (dim ≤ 96) is available and beneficial.
    ///
    /// For models like ComposedLogistic with dim=p+G > 32 but ≤ 96, the _hi
    /// variant uses register spill to L1 but still provides 32× observation
    /// parallelism over the scalar kernel.
    fn supports_warp_hi(&self) -> bool {
        false
    }

    /// Number of chains.
    fn n_chains(&self) -> usize;

    /// Parameter dimensionality.
    fn dim(&self) -> usize;

    /// Set the energy error threshold for divergence detection.
    ///
    /// The default is 1000.0 (Stan-compatible). Can be set to a model-specific
    /// value based on dimension or posterior scale.
    fn set_divergence_threshold(&mut self, _threshold: f64) {}
}
