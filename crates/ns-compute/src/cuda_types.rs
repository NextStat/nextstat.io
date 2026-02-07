//! GPU data types shared between Rust and CUDA kernels.
//!
//! All types are `#[repr(C)]` for bit-exact layout matching the CUDA kernel.

/// Modifier type discriminant (matches CUDA kernel constants).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuModifierType {
    /// Scalar multiplicative factor: `factors[:] *= param[idx]`.
    NormFactor = 0,
    /// Per-bin multiplicative gamma: `factors[bin] *= param[idx]`.
    ShapeSys = 1,
    /// Per-bin multiplicative free factor: `factors[bin] *= param[idx]`.
    ShapeFactor = 2,
    /// Scalar Code4 interpolation: `factors[:] *= normsys_code4(alpha, hi, lo)`.
    NormSys = 3,
    /// Per-bin Code4p additive delta: `deltas[bin] += histosys_delta(alpha, dn, nom, up)`.
    HistoSys = 4,
    /// Per-bin multiplicative gamma (stat error): `factors[bin] *= param[idx]`.
    StatError = 5,
    /// Scalar multiplicative luminosity: `factors[:] *= param[idx]`.
    Lumi = 6,
}

/// Packed modifier entry in the CSR bin-map (8 bytes).
///
/// For a given sample-bin, each entry tells the CUDA kernel which parameter
/// to read and what kind of modifier operation to apply.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuModifierEntry {
    /// Index into `params[]`.
    pub param_idx: u16,
    /// Modifier type (see [`GpuModifierType`]).
    pub modifier_type: u8,
    /// Padding for alignment.
    pub _pad: u8,
    /// Offset into `modifier_data[]` for extra data.
    ///
    /// - NormSys: 8 doubles (6 coefficients + ln_hi + ln_lo)
    /// - HistoSys: 2 doubles per bin (delta_up, delta_dn)
    /// - Others: unused (0)
    pub data_offset: u32,
}

/// Barlow-Beeston auxiliary Poisson constraint (24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuAuxPoissonEntry {
    /// Index of the gamma parameter in `params[]`.
    pub gamma_param_idx: u16,
    /// Padding.
    pub _pad: u16,
    /// Padding.
    pub _pad2: f32,
    /// Barlow-Beeston scale: `tau = (nominal / sigma)^2`.
    pub tau: f64,
    /// Observed auxiliary count (= tau for observed data, gamma_hat * tau for Asimov).
    pub observed_aux: f64,
}

/// Gaussian constraint entry (24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuGaussConstraintEntry {
    /// Index of the constrained parameter in `params[]`.
    pub param_idx: u16,
    /// Padding.
    pub _pad: u16,
    /// Padding.
    pub _pad2: f32,
    /// Constraint center (typically 0.0 for NP, 1.0 for lumi).
    pub center: f64,
    /// Pre-computed `1.0 / sigma`.
    pub inv_width: f64,
}

/// Sample metadata for the kernel (tells each sample-bin range and main-bin offset).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuSampleInfo {
    /// First sample-bin index for this sample.
    pub first_sample_bin: u32,
    /// Number of bins in this sample.
    pub n_bins: u32,
    /// Offset into the main-bin (channel) output array.
    pub main_bin_offset: u32,
    /// Number of modifiers for this sample.
    pub n_modifiers: u32,
}

/// Flat modifier descriptor — one per (sample, modifier) pair.
///
/// Instead of CSR per sample-bin, we use a simpler per-sample modifier list.
/// The kernel iterates modifiers for each sample and dispatches by type.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuModifierDesc {
    /// Index into `params[]` (for scalar modifiers: NormFactor, NormSys, Lumi).
    /// For per-bin modifiers (ShapeSys, StatError, ShapeFactor):
    /// base index — actual idx = param_idx + bin_offset.
    pub param_idx: u32,
    /// Modifier type (see [`GpuModifierType`]).
    pub modifier_type: u8,
    /// Is this a per-bin modifier? (1 = per-bin, 0 = scalar)
    pub is_per_bin: u8,
    /// Padding.
    pub _pad: u16,
    /// Offset into `modifier_data[]` for extra coefficients.
    pub data_offset: u32,
    /// Number of bins for per-bin parameter indices (0 for scalar modifiers).
    pub n_bins: u32,
}

/// Serialized GPU model data — all flat buffers ready for device upload.
///
/// Produced by `HistFactoryModel::serialize_for_gpu()` in ns-translate,
/// consumed by `CudaBatchAccelerator::from_gpu_data()` in ns-compute.
#[derive(Debug, Clone)]
pub struct GpuModelData {
    // --- Model structure ---
    /// Number of parameters in the model.
    pub n_params: usize,
    /// Total number of main bins (sum across all channels).
    pub n_main_bins: usize,
    /// Total number of sample-bins (sum across all samples in all channels).
    pub n_sample_bins: usize,

    // --- Per-sample metadata ---
    /// One entry per sample, in channel-then-sample order.
    pub samples: Vec<GpuSampleInfo>,

    // --- Per-sample nominal values ---
    /// Nominal counts, flat: `[sample0_bin0, sample0_bin1, ..., sample1_bin0, ...]`.
    pub nominal: Vec<f64>,

    // --- Modifier descriptors ---
    /// Per-sample modifier list, flat (ordered: sample0_mod0, sample0_mod1, ..., sample1_mod0, ...).
    pub modifier_descs: Vec<GpuModifierDesc>,
    /// Offset into `modifier_descs` for each sample: `modifier_desc_offsets[i]..modifier_desc_offsets[i+1]`.
    pub modifier_desc_offsets: Vec<u32>,

    // --- Per-bin parameter indices for per-bin modifiers ---
    /// Flat array of parameter indices for per-bin modifiers (ShapeSys, StatError, ShapeFactor).
    pub per_bin_param_indices: Vec<u32>,

    // --- Extra modifier data ---
    /// Auxiliary data for modifiers:
    /// - NormSys: 8 f64 (6 polynomial coeffs + ln(hi) + ln(lo))
    /// - HistoSys: 2 f64 per bin (delta_up, delta_dn)
    pub modifier_data: Vec<f64>,

    // --- Constraints ---
    /// Barlow-Beeston auxiliary Poisson constraints.
    pub aux_poisson: Vec<GpuAuxPoissonEntry>,
    /// Gaussian (normal) constraints.
    pub gauss_constraints: Vec<GpuGaussConstraintEntry>,
    /// Pre-computed sum of Gaussian constraint normalization constants:
    /// `Σ [ln(σ) + 0.5·ln(2π)]`.
    pub constraint_const: f64,
}
