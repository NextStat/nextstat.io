//! Unfolding plot artifacts — response matrix and unfolded spectrum.
//!
//! TRExFitter step `x` performs profile-likelihood unfolding and produces:
//! 1. The detector response (migration) matrix visualisation.
//! 2. The unfolded spectrum with uncertainties compared to truth.
//!
//! This module provides the plot-friendly JSON artifacts for both.

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Response (migration) matrix
// ---------------------------------------------------------------------------

/// Plot-friendly artifact for a detector response / migration matrix.
///
/// The matrix element `matrix[i][j]` gives the probability (or event count)
/// of a true-level bin `j` being reconstructed in reco-level bin `i`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMatrixArtifact {
    /// Label for the observable (e.g. "p_T [GeV]").
    pub observable_label: String,
    /// Reco-level bin edges (length = n_reco_bins + 1).
    pub reco_bin_edges: Vec<f64>,
    /// Truth-level bin edges (length = n_truth_bins + 1).
    pub truth_bin_edges: Vec<f64>,
    /// Response matrix in row-major order: `matrix[reco_idx][truth_idx]`.
    /// Values are conditional probabilities P(reco | truth) if normalised,
    /// or raw event counts otherwise.
    pub matrix: Vec<Vec<f64>>,
    /// Whether the matrix is normalised column-wise to probabilities.
    pub normalised: bool,
    /// Purity per reco bin: fraction of events from diagonal truth bin.
    pub purity: Vec<f64>,
    /// Stability per truth bin: fraction of events staying in the same bin.
    pub stability: Vec<f64>,
}

impl ResponseMatrixArtifact {
    /// Build a response matrix artifact from a raw event-count matrix.
    ///
    /// # Arguments
    /// * `observable_label` — axis label
    /// * `reco_bin_edges` — reco bin edges (n_reco + 1)
    /// * `truth_bin_edges` — truth bin edges (n_truth + 1)
    /// * `matrix` — raw counts `[reco_idx][truth_idx]`
    /// * `normalise` — if true, normalise columns to conditional probabilities
    pub fn new(
        observable_label: &str,
        reco_bin_edges: Vec<f64>,
        truth_bin_edges: Vec<f64>,
        matrix: Vec<Vec<f64>>,
        normalise: bool,
    ) -> Result<Self> {
        let n_reco = reco_bin_edges.len().saturating_sub(1);
        let n_truth = truth_bin_edges.len().saturating_sub(1);

        if n_reco == 0 || n_truth == 0 {
            return Err(Error::Validation("bin edges must define at least 1 bin".into()));
        }
        if matrix.len() != n_reco {
            return Err(Error::Validation(format!(
                "matrix row count {} != n_reco {}",
                matrix.len(),
                n_reco
            )));
        }
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n_truth {
                return Err(Error::Validation(format!(
                    "matrix row {} length {} != n_truth {}",
                    i,
                    row.len(),
                    n_truth
                )));
            }
        }

        // Compute column sums for normalisation and stability.
        let mut col_sums = vec![0.0_f64; n_truth];
        for row in &matrix {
            for (j, &v) in row.iter().enumerate() {
                col_sums[j] += v;
            }
        }

        // Normalise if requested.
        let out_matrix = if normalise {
            matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &v)| if col_sums[j] > 0.0 { v / col_sums[j] } else { 0.0 })
                        .collect()
                })
                .collect()
        } else {
            matrix.clone()
        };

        // Purity: for reco bin i, fraction from truth bin i (diagonal).
        let purity: Vec<f64> = (0..n_reco)
            .map(|i| {
                let row_sum: f64 = matrix[i].iter().sum();
                if row_sum > 0.0 && i < n_truth { matrix[i][i] / row_sum } else { 0.0 }
            })
            .collect();

        // Stability: for truth bin j, fraction staying in reco bin j.
        let stability: Vec<f64> = (0..n_truth)
            .map(|j| if col_sums[j] > 0.0 && j < n_reco { matrix[j][j] / col_sums[j] } else { 0.0 })
            .collect();

        Ok(Self {
            observable_label: observable_label.to_string(),
            reco_bin_edges,
            truth_bin_edges,
            matrix: out_matrix,
            normalised: normalise,
            purity,
            stability,
        })
    }
}

// ---------------------------------------------------------------------------
// Unfolded spectrum
// ---------------------------------------------------------------------------

/// A single bin in an unfolded spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnfoldedBin {
    /// Bin centre.
    pub centre: f64,
    /// Bin low edge.
    pub low: f64,
    /// Bin high edge.
    pub high: f64,
    /// Unfolded value (cross-section, events, etc.).
    pub value: f64,
    /// Total uncertainty (stat + syst combined in quadrature).
    pub error_total: f64,
    /// Statistical uncertainty only.
    pub error_stat: f64,
    /// Systematic uncertainty only.
    pub error_syst: f64,
    /// Truth-level value for comparison (NaN if not available).
    #[serde(skip_serializing_if = "is_nan")]
    pub truth: Option<f64>,
}

fn is_nan(v: &Option<f64>) -> bool {
    v.is_none_or(|x| x.is_nan())
}

/// Plot-friendly artifact for an unfolded differential distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnfoldedSpectrumArtifact {
    /// Observable label (e.g. "p_T [GeV]").
    pub observable_label: String,
    /// Y-axis label (e.g. "dσ/dp_T [pb/GeV]").
    pub y_label: String,
    /// Per-bin results.
    pub bins: Vec<UnfoldedBin>,
    /// Bin edges (length = n_bins + 1).
    pub bin_edges: Vec<f64>,
    /// Unfolded central values (length = n_bins).
    pub values: Vec<f64>,
    /// Total errors (length = n_bins).
    pub errors_total: Vec<f64>,
    /// Truth values for overlay (length = n_bins, empty if not available).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub truth_values: Vec<f64>,
    /// Method used (e.g. "profile_likelihood", "svd", "bayesian").
    pub method: String,
    /// Number of iterations (relevant for iterative methods, 0 otherwise).
    pub n_iterations: u32,
}

impl UnfoldedSpectrumArtifact {
    /// Build from bin edges and per-bin data.
    pub fn new(
        observable_label: &str,
        y_label: &str,
        bin_edges: Vec<f64>,
        bins: Vec<UnfoldedBin>,
        method: &str,
        n_iterations: u32,
    ) -> Result<Self> {
        let n_bins = bin_edges.len().saturating_sub(1);
        if n_bins == 0 {
            return Err(Error::Validation("bin edges must define at least 1 bin".into()));
        }
        if bins.len() != n_bins {
            return Err(Error::Validation(format!(
                "bins length {} != expected {}",
                bins.len(),
                n_bins
            )));
        }

        let values: Vec<f64> = bins.iter().map(|b| b.value).collect();
        let errors_total: Vec<f64> = bins.iter().map(|b| b.error_total).collect();
        let truth_values: Vec<f64> = bins.iter().filter_map(|b| b.truth).collect();
        let truth_values = if truth_values.len() == n_bins { truth_values } else { Vec::new() };

        Ok(Self {
            observable_label: observable_label.to_string(),
            y_label: y_label.to_string(),
            bins,
            bin_edges,
            values,
            errors_total,
            truth_values,
            method: method.to_string(),
            n_iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_matrix_basic() {
        // 3x3 identity-ish matrix.
        let matrix = vec![vec![90.0, 5.0, 0.0], vec![5.0, 85.0, 10.0], vec![0.0, 5.0, 90.0]];
        let edges = vec![0.0, 1.0, 2.0, 3.0];
        let art = ResponseMatrixArtifact::new("p_T", edges.clone(), edges.clone(), matrix, false)
            .unwrap();
        assert_eq!(art.purity.len(), 3);
        assert_eq!(art.stability.len(), 3);
        // Purity of bin 0: 90/(90+5+0) ≈ 0.947
        assert!((art.purity[0] - 90.0 / 95.0).abs() < 1e-10);
        // Stability of bin 0: 90/(90+5+0) ≈ 0.947
        assert!((art.stability[0] - 90.0 / 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_response_matrix_normalised() {
        let matrix = vec![vec![80.0, 20.0], vec![20.0, 80.0]];
        let edges = vec![0.0, 1.0, 2.0];
        let art =
            ResponseMatrixArtifact::new("eta", edges.clone(), edges.clone(), matrix, true).unwrap();
        assert!(art.normalised);
        // Column 0 sum should be 1.0.
        let col0_sum: f64 = art.matrix.iter().map(|row| row[0]).sum();
        assert!((col0_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_response_matrix_validation() {
        let r = ResponseMatrixArtifact::new("x", vec![0.0], vec![0.0, 1.0], vec![], false);
        assert!(r.is_err());
    }

    #[test]
    fn test_unfolded_spectrum_basic() {
        let bins = vec![
            UnfoldedBin {
                centre: 0.5,
                low: 0.0,
                high: 1.0,
                value: 10.0,
                error_total: 2.0,
                error_stat: 1.5,
                error_syst: 1.32,
                truth: Some(9.5),
            },
            UnfoldedBin {
                centre: 1.5,
                low: 1.0,
                high: 2.0,
                value: 8.0,
                error_total: 1.5,
                error_stat: 1.0,
                error_syst: 1.12,
                truth: Some(8.2),
            },
        ];
        let art = UnfoldedSpectrumArtifact::new(
            "p_T [GeV]",
            "dN/dp_T",
            vec![0.0, 1.0, 2.0],
            bins,
            "profile_likelihood",
            0,
        )
        .unwrap();
        assert_eq!(art.values.len(), 2);
        assert_eq!(art.truth_values.len(), 2);
    }

    #[test]
    fn test_unfolded_spectrum_no_truth() {
        let bins = vec![UnfoldedBin {
            centre: 0.5,
            low: 0.0,
            high: 1.0,
            value: 10.0,
            error_total: 2.0,
            error_stat: 1.5,
            error_syst: 1.32,
            truth: None,
        }];
        let art = UnfoldedSpectrumArtifact::new("x", "y", vec![0.0, 1.0], bins, "svd", 4).unwrap();
        assert!(art.truth_values.is_empty());
    }

    #[test]
    fn test_unfolded_serialization() {
        let bins = vec![UnfoldedBin {
            centre: 0.5,
            low: 0.0,
            high: 1.0,
            value: 10.0,
            error_total: 2.0,
            error_stat: 1.5,
            error_syst: 1.32,
            truth: Some(9.8),
        }];
        let art = UnfoldedSpectrumArtifact::new("x", "y", vec![0.0, 1.0], bins, "pl", 0).unwrap();
        let json = serde_json::to_string(&art).unwrap();
        let _back: UnfoldedSpectrumArtifact = serde_json::from_str(&json).unwrap();
    }
}
