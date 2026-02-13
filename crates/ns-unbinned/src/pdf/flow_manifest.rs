//! Flow manifest (`flow_manifest.json`) schema.
//!
//! Each exported normalizing flow ships with a JSON manifest describing the ONNX
//! model files, observable metadata, and optional training/validation provenance.

#[cfg(feature = "neural")]
use serde::Deserialize;

/// Top-level flow manifest (deserialized from `flow_manifest.json`).
#[cfg(feature = "neural")]
#[derive(Debug, Clone, Deserialize)]
pub struct FlowManifest {
    /// Schema version tag (must be `"nextstat_flow_v0"`).
    pub schema_version: String,
    /// Flow architecture name (e.g. `"nsf"`, `"maf"`, `"realnvp"`).
    #[serde(default)]
    pub flow_type: Option<String>,
    /// Number of observable features the flow models.
    pub features: usize,
    /// Number of context (conditioning) features. 0 for unconditional flows.
    #[serde(default)]
    pub context_features: usize,
    /// Observable names (length must equal `features`).
    pub observable_names: Vec<String>,
    /// Context parameter names (length must equal `context_features`).
    #[serde(default)]
    pub context_names: Vec<String>,
    /// Per-observable support bounds `[[lo, hi], ...]` (length must equal `features`).
    pub support: Vec<[f64; 2]>,
    /// Base distribution of the flow (e.g. `"standard_normal"`).
    #[serde(default = "default_base_distribution")]
    pub base_distribution: String,
    /// Paths to the ONNX model files (relative to the manifest directory).
    pub models: FlowModels,
    /// Optional training provenance metadata.
    #[serde(default)]
    pub training: Option<serde_json::Value>,
    /// Optional validation metrics.
    #[serde(default)]
    pub validation: Option<serde_json::Value>,
}

/// Paths to the ONNX model files.
#[cfg(feature = "neural")]
#[derive(Debug, Clone, Deserialize)]
pub struct FlowModels {
    /// ONNX model for `log_prob(x [, c]) -> [batch]`. Required.
    pub log_prob: String,
    /// ONNX model for `sample(z [, c]) -> [batch, features]`. Optional (needed for toys).
    #[serde(default)]
    pub sample: Option<String>,
    /// ONNX model for analytical Jacobian: `(x [, c]) -> (log_prob [batch], d_log_prob_d_context [batch, n_context])`.
    ///
    /// When present, `FlowPdf` uses a single forward pass through this model to compute
    /// both `log p(x|θ)` and `∂ log p / ∂ context` analytically, replacing the default
    /// central finite-difference gradient (`2 × n_context` extra forward passes).
    ///
    /// The model must have exactly 2 outputs:
    ///   - output 0: `log_prob`  shape `[batch]`
    ///   - output 1: `d_log_prob_d_context`  shape `[batch, n_context]`
    #[serde(default)]
    pub log_prob_grad: Option<String>,
}

#[cfg(feature = "neural")]
fn default_base_distribution() -> String {
    "standard_normal".to_string()
}

#[cfg(feature = "neural")]
/// Expected value of [`FlowManifest::schema_version`] for the current format.
pub const FLOW_MANIFEST_SCHEMA_VERSION: &str = "nextstat_flow_v0";

#[cfg(feature = "neural")]
impl FlowManifest {
    /// Load a manifest from a JSON file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("failed to read flow manifest {}: {e}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|e| {
            anyhow::anyhow!("failed to parse flow manifest {}: {e}", path.display())
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != FLOW_MANIFEST_SCHEMA_VERSION {
            anyhow::bail!(
                "unsupported flow manifest schema_version: '{}' (expected '{}')",
                self.schema_version,
                FLOW_MANIFEST_SCHEMA_VERSION
            );
        }
        if self.features == 0 {
            anyhow::bail!("flow manifest: features must be > 0");
        }
        if self.observable_names.len() != self.features {
            anyhow::bail!(
                "flow manifest: observable_names length {} != features {}",
                self.observable_names.len(),
                self.features
            );
        }
        if self.context_names.len() != self.context_features {
            anyhow::bail!(
                "flow manifest: context_names length {} != context_features {}",
                self.context_names.len(),
                self.context_features
            );
        }
        if self.support.len() != self.features {
            anyhow::bail!(
                "flow manifest: support length {} != features {}",
                self.support.len(),
                self.features
            );
        }
        for (i, bounds) in self.support.iter().enumerate() {
            if bounds[0].is_nan() || bounds[1].is_nan() || bounds[0] >= bounds[1] {
                anyhow::bail!(
                    "flow manifest: invalid support bounds for feature {}: [{}, {}]",
                    i,
                    bounds[0],
                    bounds[1]
                );
            }
        }
        Ok(())
    }
}
