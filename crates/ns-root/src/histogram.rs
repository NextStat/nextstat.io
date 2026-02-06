//! Public histogram type returned by `RootFile::get_histogram`.

/// A 1D histogram extracted from a ROOT file.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Histogram name.
    pub name: String,
    /// Histogram title.
    pub title: String,
    /// Number of bins (excluding under/overflow).
    pub n_bins: usize,
    /// Lower edge of first bin.
    pub x_min: f64,
    /// Upper edge of last bin.
    pub x_max: f64,
    /// Bin edges (length = n_bins + 1).
    pub bin_edges: Vec<f64>,
    /// Bin contents (length = n_bins, excluding under/overflow).
    pub bin_content: Vec<f64>,
    /// Sum of weights squared per bin (for statistical errors), if stored.
    pub sumw2: Option<Vec<f64>>,
    /// Total number of entries.
    pub entries: f64,
}

/// A 1D histogram plus explicit underflow/overflow bin values.
#[derive(Debug, Clone)]
pub struct HistogramWithFlows {
    /// Main histogram (excluding under/overflow).
    pub histogram: Histogram,
    /// Underflow bin content.
    pub underflow: f64,
    /// Overflow bin content.
    pub overflow: f64,
    /// Underflow sumw2, if stored.
    pub underflow_sumw2: Option<f64>,
    /// Overflow sumw2, if stored.
    pub overflow_sumw2: Option<f64>,
}
