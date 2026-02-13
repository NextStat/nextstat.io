//! Neural DCR (Direct Classifier Ratio) surrogate for template morphing.
//!
//! Feature-gated behind `neural`. A `DcrSurrogate` wraps a conditional
//! [`FlowPdf`] that has been trained to approximate a HistFactory-style
//! morphed template `p(x | α)`, where `α` is the vector of systematic
//! nuisance parameters.
//!
//! # FAIR-HUC Protocol
//!
//! The training protocol (implemented in `scripts/neural/train_dcr.py`):
//!
//! 1. Generate synthetic data from the full HistFactory model at many
//!    nuisance parameter points `α ~ Uniform(-3, 3)`.
//! 2. Train a conditional NSF flow `p_θ(x | α)` to match the morphed
//!    template distributions.
//! 3. Validate: compare NLL surfaces of the neural surrogate vs the
//!    original binned model across a grid of `α` values.
//!
//! At inference time, the DCR surrogate replaces `MorphingHistogramPdf`
//! or `MorphingKdePdf` with smooth, continuous, bin-free morphing.
//!
//! # Usage in spec
//!
//! ```yaml
//! pdf:
//!   type: dcr_surrogate
//!   manifest: models/bkg_dcr/flow_manifest.json
//!   systematics: [jes_alpha, jer_alpha]
//! ```

#[cfg(feature = "neural")]
use std::path::Path;

#[cfg(feature = "neural")]
use ns_core::Result;
#[cfg(feature = "neural")]
use rand::RngCore;

#[cfg(feature = "neural")]
use crate::event_store::EventStore;
#[cfg(feature = "neural")]
use crate::pdf::UnbinnedPdf;
#[cfg(feature = "neural")]
use crate::pdf::flow::FlowPdf;
#[cfg(feature = "neural")]
use crate::pdf::flow_manifest::FlowManifest;

/// A neural DCR surrogate that replaces binned template morphing.
///
/// Wraps a conditional [`FlowPdf`] with DCR-specific semantics:
/// - The context vector maps to HistFactory systematic nuisance parameters.
/// - Normalization is validated against the original binned model.
/// - Provides metadata for comparison against the reference HistFactory process.
///
/// Implements [`UnbinnedPdf`] — drop-in replacement for `MorphingHistogramPdf`.
#[cfg(feature = "neural")]
pub struct DcrSurrogate {
    /// The underlying conditional flow `p(x | α)`.
    flow: FlowPdf,
    /// Names of the HistFactory systematics this DCR replaces.
    /// These must match model parameter names and flow manifest `context_names`.
    systematic_names: Vec<String>,
    /// Name of the process this surrogate replaces (for diagnostics).
    process_name: String,
    /// Tolerance for normalization deviation from 1.0.
    /// Default: 0.01 (1%). If normalization check fails, a warning is emitted
    /// but the surrogate is still usable (correction is applied automatically).
    norm_tolerance: f64,
}

#[cfg(feature = "neural")]
impl DcrSurrogate {
    /// Load a DCR surrogate from a flow manifest.
    ///
    /// `systematic_param_indices` maps each systematic nuisance parameter to
    /// its global parameter index in the unbinned model. These become the
    /// flow's context vector: `context[i] = params[systematic_param_indices[i]]`.
    ///
    /// The flow manifest's `context_features` must match `systematic_param_indices.len()`.
    pub fn from_manifest(
        manifest_path: &Path,
        systematic_param_indices: &[usize],
        systematic_names: Vec<String>,
        process_name: String,
    ) -> anyhow::Result<Self> {
        // Fail fast on duplicates: ordering matters because it defines the flow context vector.
        {
            let mut uniq = std::collections::HashSet::<&str>::with_capacity(systematic_names.len());
            for s in &systematic_names {
                if !uniq.insert(s.as_str()) {
                    anyhow::bail!(
                        "DcrSurrogate '{}': duplicate systematic name '{}'",
                        process_name,
                        s
                    );
                }
            }
        }

        if systematic_names.len() != systematic_param_indices.len() {
            anyhow::bail!(
                "DcrSurrogate '{}': systematic_names length {} != systematic_param_indices length {}",
                process_name,
                systematic_names.len(),
                systematic_param_indices.len()
            );
        }

        // Ensure the manifest declares the same context parameter names in the same order.
        // This makes it hard to accidentally mismatch model nuisance parameters and the trained surrogate.
        let manifest = FlowManifest::from_path(manifest_path)?;
        if manifest.context_names != systematic_names {
            anyhow::bail!(
                "DcrSurrogate '{}': manifest context_names {:?} != declared systematics {:?}",
                process_name,
                manifest.context_names,
                systematic_names
            );
        }

        let flow = FlowPdf::from_manifest(manifest_path, systematic_param_indices)?;

        if flow.n_params() != systematic_names.len() {
            anyhow::bail!(
                "DcrSurrogate '{}': flow context features {} != number of systematics {}",
                process_name,
                flow.n_params(),
                systematic_names.len()
            );
        }

        Ok(Self { flow, systematic_names, process_name, norm_tolerance: 0.01 })
    }

    /// Set the normalization tolerance (default: 0.01 = 1%).
    pub fn with_norm_tolerance(mut self, tol: f64) -> Self {
        self.norm_tolerance = tol;
        self
    }

    /// Recompute normalization correction for current nuisance parameter values.
    ///
    /// Should be called when systematic parameters change significantly.
    /// For well-trained DCR surrogates, the correction should be near zero.
    pub fn update_normalization(&mut self, params: &[f64]) -> Result<()> {
        self.flow.update_normalization(params)
    }

    /// Name of the process this surrogate replaces.
    pub fn process_name(&self) -> &str {
        &self.process_name
    }

    /// Names of the systematic nuisance parameters.
    pub fn systematic_names(&self) -> &[String] {
        &self.systematic_names
    }

    /// Normalization tolerance.
    pub fn norm_tolerance(&self) -> f64 {
        self.norm_tolerance
    }

    /// Access the underlying flow PDF.
    pub fn flow(&self) -> &FlowPdf {
        &self.flow
    }

    /// Validate the surrogate at the nominal point (all systematics = 0).
    ///
    /// Returns `(integral, deviation_from_1)`. Deviation should be < `norm_tolerance`.
    pub fn validate_nominal_normalization(&mut self, params: &[f64]) -> Result<(f64, f64)> {
        self.flow.update_normalization(params)?;
        let correction = self.flow.log_norm_correction();
        let integral = correction.exp();
        let deviation = (integral - 1.0).abs();
        Ok((integral, deviation))
    }
}

#[cfg(feature = "neural")]
impl UnbinnedPdf for DcrSurrogate {
    fn n_params(&self) -> usize {
        self.flow.n_params()
    }

    fn observables(&self) -> &[String] {
        self.flow.observables()
    }

    fn log_prob_batch(&self, events: &EventStore, params: &[f64], out: &mut [f64]) -> Result<()> {
        self.flow.log_prob_batch(events, params, out)
    }

    fn log_prob_grad_batch(
        &self,
        events: &EventStore,
        params: &[f64],
        out_logp: &mut [f64],
        out_grad: &mut [f64],
    ) -> Result<()> {
        self.flow.log_prob_grad_batch(events, params, out_logp, out_grad)
    }

    fn sample(
        &self,
        params: &[f64],
        n_events: usize,
        support: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> Result<EventStore> {
        self.flow.sample(params, n_events, support, rng)
    }
}

#[cfg(feature = "neural")]
impl std::fmt::Debug for DcrSurrogate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DcrSurrogate")
            .field("process_name", &self.process_name)
            .field("systematic_names", &self.systematic_names)
            .field("n_observables", &self.flow.observables().len())
            .field("norm_tolerance", &self.norm_tolerance)
            .finish()
    }
}
