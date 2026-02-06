//! Histogram filling helpers.
//!
//! This module is currently a small placeholder API surface to keep the crate's
//! public exports stable while the TTree histogram-filling path is implemented
//! incrementally.
//!
//! The intent is to support workflows like:
//! - open a ROOT file
//! - select a `TTree`
//! - evaluate one or more expressions per entry
//! - fill 1D histograms

use crate::{Result, RootError};

/// 1D histogram fill specification.
#[derive(Debug, Clone)]
pub struct HistogramSpec {
    /// Output histogram name/identifier.
    pub name: String,
    /// Expression to evaluate for each entry (e.g. `pt`, `eta`, `mjj`).
    pub expr: String,
    /// Optional weight expression (defaults to 1).
    pub weight: Option<String>,
    /// Number of bins.
    pub n_bins: usize,
    /// Lower edge (inclusive).
    pub lo: f64,
    /// Upper edge (exclusive).
    pub hi: f64,
}

impl HistogramSpec {
    /// Create a new 1D histogram specification.
    pub fn new(name: impl Into<String>, expr: impl Into<String>, n_bins: usize, lo: f64, hi: f64) -> Self {
        Self {
            name: name.into(),
            expr: expr.into(),
            weight: None,
            n_bins,
            lo,
            hi,
        }
    }
}

/// Filled 1D histogram output.
#[derive(Debug, Clone)]
pub struct FilledHistogram {
    /// Histogram name (from [`HistogramSpec::name`]).
    pub name: String,
    /// Bin edges (length `n_bins + 1`).
    pub edges: Vec<f64>,
    /// Bin contents (length `n_bins`).
    pub counts: Vec<f64>,
    /// Underflow weight sum.
    pub underflow: f64,
    /// Overflow weight sum.
    pub overflow: f64,
}

/// Fill one or more histograms from a ROOT `TTree`.
///
/// Not implemented yet; returns a structured error rather than panicking.
pub fn fill_histograms() -> Result<Vec<FilledHistogram>> {
    Err(RootError::Expression(
        "fill_histograms is not implemented yet (placeholder API)".to_string(),
    ))
}

