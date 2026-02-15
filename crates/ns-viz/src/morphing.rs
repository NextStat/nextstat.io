//! Morphing validation artifact — template interpolation visualisation.
//!
//! TRExFitter allows visual inspection of how signal templates are morphed
//! (interpolated/extrapolated) between discrete mass or coupling points.
//! This module provides the plot-friendly JSON artifact showing the original
//! discrete templates alongside the interpolated result at an arbitrary
//! parameter value.

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// A single discrete template (anchor point) used in morphing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphingTemplate {
    /// Parameter value at which this template was generated.
    pub parameter_value: f64,
    /// Bin contents of the template histogram.
    pub bin_contents: Vec<f64>,
    /// Label (e.g. "m_H = 125 GeV").
    pub label: String,
}

/// Plot-friendly artifact for morphing validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphingArtifact {
    /// Label for the morphing parameter (e.g. "m_H [GeV]").
    pub parameter_label: String,
    /// Observable label (e.g. "m_jj [GeV]").
    pub observable_label: String,
    /// Bin edges of the observable (length = n_bins + 1).
    pub bin_edges: Vec<f64>,
    /// Discrete anchor templates.
    pub templates: Vec<MorphingTemplate>,
    /// Interpolated result at `target_value`.
    pub interpolated: Vec<f64>,
    /// Parameter value at which interpolation was evaluated.
    pub target_value: f64,
    /// Interpolation method used (e.g. "vertical", "horizontal", "moment").
    pub method: String,
}

impl MorphingArtifact {
    /// Build a morphing artifact from anchor templates and an interpolated result.
    ///
    /// # Arguments
    /// * `parameter_label` — label for the morphing parameter axis
    /// * `observable_label` — label for the histogram x-axis
    /// * `bin_edges` — observable bin edges (n_bins + 1)
    /// * `templates` — discrete anchor templates (at least 2)
    /// * `interpolated` — interpolated histogram at `target_value`
    /// * `target_value` — morphing parameter value for interpolation
    /// * `method` — interpolation method name
    pub fn new(
        parameter_label: &str,
        observable_label: &str,
        bin_edges: Vec<f64>,
        templates: Vec<MorphingTemplate>,
        interpolated: Vec<f64>,
        target_value: f64,
        method: &str,
    ) -> Result<Self> {
        let n_bins = bin_edges.len().saturating_sub(1);
        if n_bins == 0 {
            return Err(Error::Validation("bin edges must define at least 1 bin".into()));
        }
        if templates.len() < 2 {
            return Err(Error::Validation("morphing requires at least 2 anchor templates".into()));
        }
        for (i, t) in templates.iter().enumerate() {
            if t.bin_contents.len() != n_bins {
                return Err(Error::Validation(format!(
                    "template {} has {} bins, expected {}",
                    i,
                    t.bin_contents.len(),
                    n_bins,
                )));
            }
        }
        if interpolated.len() != n_bins {
            return Err(Error::Validation(format!(
                "interpolated has {} bins, expected {}",
                interpolated.len(),
                n_bins,
            )));
        }

        Ok(Self {
            parameter_label: parameter_label.to_string(),
            observable_label: observable_label.to_string(),
            bin_edges,
            templates,
            interpolated,
            target_value,
            method: method.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphing_basic() {
        let t1 = MorphingTemplate {
            parameter_value: 120.0,
            bin_contents: vec![10.0, 20.0, 30.0],
            label: "m_H = 120".into(),
        };
        let t2 = MorphingTemplate {
            parameter_value: 130.0,
            bin_contents: vec![15.0, 25.0, 35.0],
            label: "m_H = 130".into(),
        };
        let interp = vec![12.5, 22.5, 32.5]; // midpoint
        let art = MorphingArtifact::new(
            "m_H [GeV]",
            "m_jj [GeV]",
            vec![0.0, 1.0, 2.0, 3.0],
            vec![t1, t2],
            interp,
            125.0,
            "vertical",
        )
        .unwrap();
        assert_eq!(art.templates.len(), 2);
        assert!((art.target_value - 125.0).abs() < 1e-12);
    }

    #[test]
    fn test_morphing_validation_single_template() {
        let t1 = MorphingTemplate {
            parameter_value: 120.0,
            bin_contents: vec![10.0],
            label: "only one".into(),
        };
        let r = MorphingArtifact::new("x", "y", vec![0.0, 1.0], vec![t1], vec![10.0], 120.0, "v");
        assert!(r.is_err());
    }

    #[test]
    fn test_morphing_validation_bin_mismatch() {
        let t1 = MorphingTemplate {
            parameter_value: 1.0,
            bin_contents: vec![1.0, 2.0],
            label: "a".into(),
        };
        let t2 = MorphingTemplate {
            parameter_value: 2.0,
            bin_contents: vec![3.0], // wrong length
            label: "b".into(),
        };
        let r = MorphingArtifact::new(
            "x",
            "y",
            vec![0.0, 1.0, 2.0],
            vec![t1, t2],
            vec![2.0, 3.0],
            1.5,
            "v",
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_morphing_serialization() {
        let t1 =
            MorphingTemplate { parameter_value: 1.0, bin_contents: vec![10.0], label: "a".into() };
        let t2 =
            MorphingTemplate { parameter_value: 2.0, bin_contents: vec![20.0], label: "b".into() };
        let art =
            MorphingArtifact::new("p", "o", vec![0.0, 1.0], vec![t1, t2], vec![15.0], 1.5, "v")
                .unwrap();
        let json = serde_json::to_string(&art).unwrap();
        let _back: MorphingArtifact = serde_json::from_str(&json).unwrap();
    }
}
