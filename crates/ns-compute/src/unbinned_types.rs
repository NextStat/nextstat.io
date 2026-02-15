//! GPU data types for unbinned (event-level) likelihood kernels.
//!
//! These types are always available (no CUDA/Metal feature gate) so that higher-level crates
//! can lower supported unbinned models into a kernel-friendly representation without requiring
//! a GPU toolchain at compile time.

/// Supported PDF kinds in the unbinned GPU kernels.
///
/// The GPU path supports a closed set of built-in PDFs (see plan doc section 9.0).
pub mod pdf_kind {
    /// Truncated/normalized Gaussian PDF on `[a,b]`.
    pub const GAUSSIAN: u32 = 0;
    /// Bounded exponential-family PDF `p(x) ∝ exp(lambda * x)` on `[a,b]`.
    pub const EXPONENTIAL: u32 = 1;
    /// One-sided Crystal Ball PDF (left tail), normalized on `[a,b]`.
    pub const CRYSTAL_BALL: u32 = 2;
    /// Two-sided Crystal Ball PDF (DoubleCB), normalized on `[a,b]`.
    pub const DOUBLE_CRYSTAL_BALL: u32 = 3;
    /// Chebyshev polynomial PDF, normalized on `[a,b]`.
    pub const CHEBYSHEV: u32 = 4;
    /// Piecewise-constant histogram PDF (bin_edges + log_density), normalized on `[a,b]`.
    pub const HISTOGRAM: u32 = 5;
}

/// Supported yield expression kinds in the unbinned GPU kernels.
pub mod yield_kind {
    /// Fixed yield `nu = base_yield`.
    pub const FIXED: u32 = 0;
    /// Free yield parameter `nu = params[yield_param_idx]`.
    pub const PARAMETER: u32 = 1;
    /// Scaled yield `nu = base_yield * params[yield_param_idx]`.
    pub const SCALED: u32 = 2;
}

/// Supported rate modifier kinds for yields in the unbinned GPU kernels.
pub mod rate_modifier_kind {
    /// HistFactory-like NormSys modifier (piecewise exponential interpolation).
    pub const NORM_SYS: u32 = 0;
    /// HistFactory-like WeightSys modifier (template interpolation on a scalar yield factor).
    pub const WEIGHT_SYS: u32 = 1;
}

/// One yield rate modifier descriptor for GPU evaluation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuUnbinnedRateModifierDesc {
    /// Modifier kind (`rate_modifier_kind::*`).
    pub kind: u32,
    /// Global nuisance parameter index `α`.
    pub alpha_param_idx: u32,
    /// Interpolation code (WeightSys only; 0=code0, 1=code4p). Ignored for NormSys.
    pub interp_code: u32,
    /// Padding for 8-byte alignment.
    pub _pad: u32,
    /// Down variation factor (at `α=-1`) for WeightSys, or `lo` for NormSys.
    pub lo: f64,
    /// Up variation factor (at `α=+1`) for WeightSys, or `hi` for NormSys.
    pub hi: f64,
}

/// One process descriptor (PDF + yield) for unbinned GPU evaluation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuUnbinnedProcessDesc {
    /// Base yield constant (used by `yield_kind::FIXED` and `yield_kind::SCALED`).
    pub base_yield: f64,
    /// PDF kind (`pdf_kind::*`).
    pub pdf_kind: u32,
    /// Yield kind (`yield_kind::*`).
    pub yield_kind: u32,
    /// Observable index in the SoA input (Phase 1: always 0).
    pub obs_index: u32,
    /// Offset into the `shape_param_indices` array.
    pub shape_param_offset: u32,
    /// Number of shape parameters for this PDF (e.g. 2 for Gaussian).
    pub n_shape_params: u32,
    /// Global parameter index used by the yield expression (if applicable).
    pub yield_param_idx: u32,
    /// Offset into the `rate_modifiers` array.
    pub rate_mod_offset: u32,
    /// Number of rate modifiers applied to this process yield.
    pub n_rate_mods: u32,
    /// Offset into the `pdf_aux_f64` array for this process (PDF-kind-specific).
    pub pdf_aux_offset: u32,
    /// Length in `f64` of this process's aux segment in `pdf_aux_f64`.
    pub pdf_aux_len: u32,
}

/// Gaussian constraint entry for unbinned GPU evaluation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuUnbinnedGaussConstraintEntry {
    /// Constraint mean.
    pub center: f64,
    /// `1/sigma` (must be finite and > 0).
    pub inv_width: f64,
    /// Global parameter index this constraint applies to.
    pub param_idx: u32,
    /// Padding for 8-byte alignment.
    pub _pad: u32,
}

/// Lowered unbinned model data for GPU evaluation (SoA event layout).
#[derive(Debug, Clone)]
pub struct UnbinnedGpuModelData {
    /// Number of global parameters.
    pub n_params: usize,
    /// Number of observables (dimensions). Phase 1 GPU supports 1D only.
    pub n_obs: usize,
    /// Number of events.
    pub n_events: usize,
    /// Observable bounds as `(low, high)` per observable (length `n_obs`).
    pub obs_bounds: Vec<(f64, f64)>,
    /// Observed event data in SoA layout `[n_obs × n_events]`:
    /// `obs[d * n_events + i]`.
    pub obs_soa: Vec<f64>,
    /// Optional per-event weights (length `n_events`).
    ///
    /// When present, kernels interpret these as frequency weights: each event contribution
    /// to `-log L` is multiplied by `w_i`.
    ///
    /// Contract: weights must be finite and >= 0 (and not all-zero).
    pub event_weights: Option<Vec<f64>>,
    /// Process descriptors.
    pub processes: Vec<GpuUnbinnedProcessDesc>,
    /// Rate modifiers (yield systematics), indexed by `GpuUnbinnedProcessDesc::{rate_mod_offset,n_rate_mods}`.
    pub rate_modifiers: Vec<GpuUnbinnedRateModifierDesc>,
    /// Concatenated global parameter indices for process shape params.
    pub shape_param_indices: Vec<u32>,
    /// PDF-kind-specific aux data (read-only), referenced by `GpuUnbinnedProcessDesc::{pdf_aux_offset,pdf_aux_len}`.
    pub pdf_aux_f64: Vec<f64>,
    /// Gaussian constraints (nuisance priors).
    pub gauss_constraints: Vec<GpuUnbinnedGaussConstraintEntry>,
    /// Constant term for constraints: Σ (ln σ + 0.5 ln 2π).
    pub constraint_const: f64,
}

/// Per-channel descriptor for multi-channel GPU-native L-BFGS.
///
/// Must match `struct GpuChannelDesc` in `unbinned_common.cuh`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuChannelDesc {
    /// Device pointer to flattened event buffer for this channel.
    pub obs_flat: *const f64,
    /// Device pointer to toy offsets (prefix sums), length `n_toys + 1`.
    pub toy_offsets: *const u32,
    /// Device pointer to process descriptors, length `n_procs`.
    pub procs: *const GpuUnbinnedProcessDesc,
    /// Device pointer to rate modifier descriptors, length `total_rate_mods`.
    pub rate_mods: *const GpuUnbinnedRateModifierDesc,
    /// Device pointer to shape param index list, length `total_shape_params`.
    pub shape_pidx: *const u32,
    /// Device pointer to PDF aux buffer.
    pub pdf_aux_f64: *const f64,
    /// Device pointer to Gaussian constraints (channel 0 only); `n_gauss=0` for other channels.
    pub gauss: *const GpuUnbinnedGaussConstraintEntry,
    /// Number of processes in this channel.
    pub n_procs: u32,
    /// Total rate modifiers for this channel.
    pub total_rate_mods: u32,
    /// Total shape param indices for this channel.
    pub total_shape_params: u32,
    /// Number of Gaussian constraints for this channel.
    pub n_gauss: u32,
    /// Observable lower bound for this channel.
    pub obs_lo: f64,
    /// Observable upper bound for this channel.
    pub obs_hi: f64,
    /// Constant term for constraints (only non-zero for first channel).
    pub constraint_const: f64,
}

// --- cudarc DeviceRepr impls (required for clone_htod / memcpy) ---
#[cfg(feature = "cuda")]
mod cuda_impls {
    unsafe impl cudarc::driver::DeviceRepr for super::GpuUnbinnedProcessDesc {}
    unsafe impl cudarc::driver::DeviceRepr for super::GpuUnbinnedGaussConstraintEntry {}
    unsafe impl cudarc::driver::DeviceRepr for super::GpuUnbinnedRateModifierDesc {}
    unsafe impl cudarc::driver::DeviceRepr for super::GpuChannelDesc {}

    unsafe impl cudarc::driver::safe::ValidAsZeroBits for super::GpuUnbinnedProcessDesc {}
    unsafe impl cudarc::driver::safe::ValidAsZeroBits for super::GpuUnbinnedGaussConstraintEntry {}
    unsafe impl cudarc::driver::safe::ValidAsZeroBits for super::GpuUnbinnedRateModifierDesc {}
    unsafe impl cudarc::driver::safe::ValidAsZeroBits for super::GpuChannelDesc {}
}
