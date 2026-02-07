//! GPU data types for Metal kernels (f32 precision).
//!
//! All types are `#[repr(C)]` for bit-exact layout matching the MSL kernel.
//! Always available (no feature gate) so ns-translate can serialize
//! `MetalModelData` without requiring Metal at compile time.

use crate::cuda_types::{GpuModelData, GpuSampleInfo, GpuModifierDesc};

/// Barlow-Beeston auxiliary Poisson constraint (Metal f32 version).
///
/// Unlike CUDA, Metal has no `lgamma()` — we precompute it on CPU.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetalAuxPoissonEntry {
    /// Index of the gamma parameter in `params[]`.
    pub gamma_param_idx: u16,
    /// Padding.
    pub _pad: u16,
    /// Barlow-Beeston scale: `tau = (nominal / sigma)^2`.
    pub tau: f32,
    /// Observed auxiliary count.
    pub observed_aux: f32,
    /// Pre-computed `lgamma(observed_aux + 1.0)` (Metal has no lgamma).
    pub lgamma_obs: f32,
}

/// Gaussian constraint entry (Metal f32 version).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetalGaussConstraintEntry {
    /// Index of the constrained parameter in `params[]`.
    pub param_idx: u16,
    /// Padding.
    pub _pad: u16,
    /// Constraint center (typically 0.0 for NP, 1.0 for lumi).
    pub center: f32,
    /// Pre-computed `1.0 / sigma`.
    pub inv_width: f32,
}

/// Serialized Metal model data — all f32 flat buffers ready for GPU upload.
///
/// Produced from `GpuModelData` via [`MetalModelData::from_gpu_data`].
#[derive(Debug, Clone)]
pub struct MetalModelData {
    // --- Model structure (same as CUDA, integers unchanged) ---
    /// Number of parameters in the model.
    pub n_params: usize,
    /// Total number of main bins (sum across all channels).
    pub n_main_bins: usize,
    /// Total number of sample-bins.
    pub n_sample_bins: usize,

    // --- Per-sample metadata (reused from CUDA — all u32, no change) ---
    /// One entry per sample, in channel-then-sample order.
    pub samples: Vec<GpuSampleInfo>,

    // --- Per-sample nominal values (f64→f32) ---
    /// Nominal counts, flat.
    pub nominal: Vec<f32>,

    // --- Modifier descriptors (reused from CUDA — all u32/u8, no change) ---
    /// Per-sample modifier list.
    pub modifier_descs: Vec<GpuModifierDesc>,
    /// Offset into `modifier_descs` for each sample.
    pub modifier_desc_offsets: Vec<u32>,

    // --- Per-bin parameter indices (unchanged, all u32) ---
    /// Flat array of parameter indices for per-bin modifiers.
    pub per_bin_param_indices: Vec<u32>,

    // --- Extra modifier data (f64→f32) ---
    /// NormSys coefficients + HistoSys deltas, all f32.
    pub modifier_data: Vec<f32>,

    // --- Constraints (f32 versions) ---
    /// Barlow-Beeston constraints with precomputed lgamma.
    pub aux_poisson: Vec<MetalAuxPoissonEntry>,
    /// Gaussian constraints.
    pub gauss_constraints: Vec<MetalGaussConstraintEntry>,
    /// Pre-computed constraint constant (kept as f32).
    pub constraint_const: f32,
}

impl MetalModelData {
    /// Convert from CUDA model data (f64) to Metal model data (f32).
    ///
    /// Precomputes `lgamma(observed_aux + 1.0)` for auxiliary Poisson entries
    /// since Metal Shading Language has no `lgamma()` built-in.
    pub fn from_gpu_data(data: &GpuModelData) -> Self {
        use statrs::function::gamma::ln_gamma;

        let nominal: Vec<f32> = data.nominal.iter().map(|&v| v as f32).collect();
        let modifier_data: Vec<f32> = data.modifier_data.iter().map(|&v| v as f32).collect();

        let aux_poisson: Vec<MetalAuxPoissonEntry> = data
            .aux_poisson
            .iter()
            .map(|a| {
                let obs_aux = a.observed_aux as f32;
                MetalAuxPoissonEntry {
                    gamma_param_idx: a.gamma_param_idx,
                    _pad: 0,
                    tau: a.tau as f32,
                    observed_aux: obs_aux,
                    lgamma_obs: if obs_aux > 0.0 {
                        ln_gamma(a.observed_aux + 1.0) as f32
                    } else {
                        0.0
                    },
                }
            })
            .collect();

        let gauss_constraints: Vec<MetalGaussConstraintEntry> = data
            .gauss_constraints
            .iter()
            .map(|g| MetalGaussConstraintEntry {
                param_idx: g.param_idx,
                _pad: 0,
                center: g.center as f32,
                inv_width: g.inv_width as f32,
            })
            .collect();

        Self {
            n_params: data.n_params,
            n_main_bins: data.n_main_bins,
            n_sample_bins: data.n_sample_bins,
            samples: data.samples.clone(),
            nominal,
            modifier_descs: data.modifier_descs.clone(),
            modifier_desc_offsets: data.modifier_desc_offsets.clone(),
            per_bin_param_indices: data.per_bin_param_indices.clone(),
            modifier_data,
            aux_poisson,
            gauss_constraints,
            constraint_const: data.constraint_const as f32,
        }
    }
}

