//! Configuration types for the ntuple → Workspace pipeline.

use std::path::PathBuf;

/// Modifier applied to a sample in the ntuple pipeline.
#[derive(Debug, Clone)]
pub enum NtupleModifier {
    /// Free-floating normalization factor (POI or nuisance).
    NormFactor {
        /// Parameter name.
        name: String,
    },
    /// Log-normal normalization systematic.
    NormSys {
        /// Parameter name.
        name: String,
        /// Low scale factor (e.g. 0.9).
        lo: f64,
        /// High scale factor (e.g. 1.1).
        hi: f64,
    },
    /// Weight-based systematic: same events, different weight expression.
    WeightSys {
        /// Parameter name.
        name: String,
        /// Weight expression for +1σ variation.
        weight_up: String,
        /// Weight expression for −1σ variation.
        weight_down: String,
    },
    /// Tree-based systematic: different ROOT file/tree with varied events.
    TreeSys {
        /// Parameter name.
        name: String,
        /// ROOT file for +1σ variation.
        file_up: PathBuf,
        /// ROOT file for −1σ variation.
        file_down: PathBuf,
        /// Override tree name (defaults to channel tree_name).
        tree_name: Option<String>,
    },
    /// MC statistical error (Barlow–Beeston lite).
    StatError,
}

/// Configuration for a single sample within a channel.
#[derive(Debug, Clone)]
pub struct SampleConfig {
    /// Sample name.
    pub name: String,
    /// ROOT file path.
    pub file: PathBuf,
    /// Override tree name (defaults to channel tree_name).
    pub tree_name: Option<String>,
    /// Weight expression (e.g. `"weight_mc * weight_sf"`).
    pub weight: Option<String>,
    /// Modifiers applied to this sample.
    pub modifiers: Vec<NtupleModifier>,
}

/// Configuration for a single channel (analysis region).
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Channel name.
    pub name: String,
    /// Expression for the variable to histogram (e.g. `"mbb"`).
    pub variable: String,
    /// Bin edges for the histogram.
    pub binning: Vec<f64>,
    /// Selection cut expression (e.g. `"njet >= 4 && pt_lead > 25.0"`).
    pub selection: Option<String>,
    /// ROOT file for observed data (None = Asimov data from sum of nominals).
    pub data_file: Option<PathBuf>,
    /// Override tree name for data file.
    pub data_tree_name: Option<String>,
    /// Samples in this channel.
    pub samples: Vec<SampleConfig>,
}

impl SampleConfig {
    /// Create a new sample config.
    pub fn new(name: impl Into<String>, file: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            file: file.into(),
            tree_name: None,
            weight: None,
            modifiers: Vec::new(),
        }
    }

    /// Set the weight expression.
    pub fn weight(mut self, expr: impl Into<String>) -> Self {
        self.weight = Some(expr.into());
        self
    }

    /// Add a normalization factor.
    pub fn normfactor(mut self, name: impl Into<String>) -> Self {
        self.modifiers.push(NtupleModifier::NormFactor { name: name.into() });
        self
    }

    /// Add a normalization systematic.
    pub fn normsys(mut self, name: impl Into<String>, lo: f64, hi: f64) -> Self {
        self.modifiers.push(NtupleModifier::NormSys { name: name.into(), lo, hi });
        self
    }

    /// Add a weight-based systematic.
    pub fn weight_sys(
        mut self,
        name: impl Into<String>,
        weight_up: impl Into<String>,
        weight_down: impl Into<String>,
    ) -> Self {
        self.modifiers.push(NtupleModifier::WeightSys {
            name: name.into(),
            weight_up: weight_up.into(),
            weight_down: weight_down.into(),
        });
        self
    }

    /// Add a tree-based systematic.
    pub fn tree_sys(
        mut self,
        name: impl Into<String>,
        file_up: impl Into<PathBuf>,
        file_down: impl Into<PathBuf>,
    ) -> Self {
        self.modifiers.push(NtupleModifier::TreeSys {
            name: name.into(),
            file_up: file_up.into(),
            file_down: file_down.into(),
            tree_name: None,
        });
        self
    }

    /// Enable MC statistical error.
    pub fn staterror(mut self) -> Self {
        self.modifiers.push(NtupleModifier::StatError);
        self
    }
}

impl ChannelConfig {
    /// Create a new channel config.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variable: String::new(),
            binning: Vec::new(),
            selection: None,
            data_file: None,
            data_tree_name: None,
            samples: Vec::new(),
        }
    }

    /// Set the variable expression.
    pub fn variable(mut self, expr: impl Into<String>) -> Self {
        self.variable = expr.into();
        self
    }

    /// Set the binning.
    pub fn binning(mut self, edges: &[f64]) -> Self {
        self.binning = edges.to_vec();
        self
    }

    /// Set the selection expression.
    pub fn selection(mut self, expr: impl Into<String>) -> Self {
        self.selection = Some(expr.into());
        self
    }

    /// Set the data file.
    pub fn data_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.data_file = Some(path.into());
        self
    }

    /// Add a sample.
    pub fn add_sample(mut self, sample: SampleConfig) -> Self {
        self.samples.push(sample);
        self
    }
}
