//! Separation plot artifact — signal vs background shape comparison.
//!
//! TRExFitter produces a "separation" plot per channel that overlays the signal
//! and background distributions **normalised to unit area**, allowing visual
//! assessment of the discriminating power of each region.
//!
//! This module computes the numeric artifact; rendering (SVG/PNG) is handled
//! downstream by the report layer.

use std::collections::{HashMap, HashSet};

use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use serde::{Deserialize, Serialize};

/// Top-level separation artifact covering all channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationArtifact {
    pub schema_version: String,
    pub channels: Vec<SeparationChannelArtifact>,
}

/// Per-channel separation data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationChannelArtifact {
    pub channel_name: String,
    /// Bin edges (length = n_bins + 1). Empty if not available.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bin_edges: Vec<f64>,
    /// Signal shape normalised to unit area.
    pub signal_shape: Vec<f64>,
    /// Background shape normalised to unit area.
    pub background_shape: Vec<f64>,
    /// Separation metric ∈ [0, 1].
    ///
    /// Defined as `0.5 * Σ_i (s_i - b_i)^2 / (s_i + b_i)` where `s_i`, `b_i`
    /// are the unit-normalised bin contents. 0 = identical shapes, 1 = fully separated.
    pub separation: f64,
}

/// Normalise a histogram to unit area. Returns zero-vector if integral ≤ 0.
fn normalise(h: &[f64]) -> Vec<f64> {
    let sum: f64 = h.iter().copied().sum();
    if sum <= 0.0 {
        return vec![0.0; h.len()];
    }
    h.iter().map(|&v| v / sum).collect()
}

/// Compute the separation metric between two unit-normalised distributions.
///
/// `separation = 0.5 * Σ (s - b)^2 / (s + b)` for bins where `s + b > 0`.
fn separation_metric(s: &[f64], b: &[f64]) -> f64 {
    let mut metric = 0.0;
    for (&si, &bi) in s.iter().zip(b.iter()) {
        let denom = si + bi;
        if denom > 0.0 {
            let diff = si - bi;
            metric += diff * diff / denom;
        }
    }
    0.5 * metric
}

/// Build a separation artifact from a model, parameter values, and explicit signal sample names.
///
/// `signal_samples` identifies which samples are "signal" — typically the samples
/// that carry a `normfactor` modifier named after the POI. Everything else is "background".
///
/// `bin_edges_by_channel` is optional; pass an empty map if edges are unavailable.
pub fn separation_artifact(
    model: &HistFactoryModel,
    params: &[f64],
    signal_samples: &HashSet<String>,
    bin_edges_by_channel: &HashMap<String, Vec<f64>>,
) -> Result<SeparationArtifact> {
    let per_channel = model.expected_main_by_channel_sample(params)?;

    let mut channels = Vec::with_capacity(per_channel.len());
    for ch in &per_channel {
        let n_bins = ch.total.len();
        let mut sig = vec![0.0; n_bins];
        let mut bkg = vec![0.0; n_bins];

        for s in &ch.samples {
            let target = if signal_samples.contains(&s.sample_name) { &mut sig } else { &mut bkg };
            for (i, &v) in s.y.iter().enumerate() {
                if i < target.len() {
                    target[i] += v;
                }
            }
        }

        let sig_norm = normalise(&sig);
        let bkg_norm = normalise(&bkg);
        let sep = separation_metric(&sig_norm, &bkg_norm);

        let edges = bin_edges_by_channel.get(&ch.channel_name).cloned().unwrap_or_default();

        channels.push(SeparationChannelArtifact {
            channel_name: ch.channel_name.clone(),
            bin_edges: edges,
            signal_shape: sig_norm,
            background_shape: bkg_norm,
            separation: sep,
        });
    }

    Ok(SeparationArtifact { schema_version: "nextstat_separation_v0".to_string(), channels })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalise_basic() {
        let h = vec![2.0, 3.0, 5.0];
        let n = normalise(&h);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!((n[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_zero() {
        let h = vec![0.0, 0.0];
        let n = normalise(&h);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn test_separation_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        assert!((separation_metric(&a, &a)).abs() < 1e-12);
    }

    #[test]
    fn test_separation_fully_separated() {
        let s = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((separation_metric(&s, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_separation_partial() {
        let s = vec![0.8, 0.2];
        let b = vec![0.2, 0.8];
        let m = separation_metric(&s, &b);
        assert!(m > 0.0 && m < 1.0);
    }
}
