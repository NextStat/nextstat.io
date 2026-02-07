//! NtupleWorkspaceBuilder: orchestrates TTree reading, histogram filling,
//! and Workspace assembly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ns_core::{Error, Result};
use ns_root::{
    CompiledExpr, FilledHistogram, FlowPolicy, HistogramSpec, NegativeWeightPolicy, RootFile,
    fill_histograms,
};

use crate::pyhf::schema::{
    Channel, HistoSysData, Measurement, MeasurementConfig, Modifier, NormSysData, Observation,
    Sample, Workspace,
};

use super::config::{ChannelConfig, NtupleModifier, SampleConfig};

/// Builder for constructing a [`Workspace`] from ROOT ntuple files.
///
/// # Example
///
/// ```no_run
/// use ns_translate::ntuple::{NtupleWorkspaceBuilder, ChannelConfig, SampleConfig};
///
/// let ws = NtupleWorkspaceBuilder::new()
///     .tree_name("events")
///     .measurement("meas", "mu")
///     .add_channel(
///         ChannelConfig::new("SR")
///             .variable("mbb")
///             .binning(&[0., 50., 100., 150., 200., 300.])
///             .add_sample(
///                 SampleConfig::new("signal", "ttH.root")
///                     .weight("weight_mc")
///                     .normfactor("mu")
///             )
///     )
///     .build()
///     .unwrap();
/// ```
pub struct NtupleWorkspaceBuilder {
    base_dir: PathBuf,
    tree_name: String,
    channels: Vec<ChannelConfig>,
    measurement_name: String,
    poi: String,
}

impl NtupleWorkspaceBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            base_dir: PathBuf::from("."),
            tree_name: "events".into(),
            channels: Vec::new(),
            measurement_name: "meas".into(),
            poi: "mu".into(),
        }
    }

    /// Set the base directory for resolving ROOT file paths.
    pub fn ntuple_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.base_dir = path.into();
        self
    }

    /// Set the default TTree name.
    pub fn tree_name(mut self, name: impl Into<String>) -> Self {
        self.tree_name = name.into();
        self
    }

    /// Set the measurement name and parameter of interest.
    pub fn measurement(mut self, name: impl Into<String>, poi: impl Into<String>) -> Self {
        self.measurement_name = name.into();
        self.poi = poi.into();
        self
    }

    /// Add a channel configuration.
    pub fn add_channel(mut self, channel: ChannelConfig) -> Self {
        self.channels.push(channel);
        self
    }

    /// Build the [`Workspace`] by reading ntuples and filling histograms.
    pub fn build(&self) -> Result<Workspace> {
        let mut root_cache: HashMap<PathBuf, RootFile> = HashMap::new();
        let mut ws_channels = Vec::new();
        let mut ws_observations = Vec::new();

        for ch_cfg in &self.channels {
            let (channel, observation) = self.process_channel(ch_cfg, &mut root_cache)?;
            ws_channels.push(channel);
            ws_observations.push(observation);
        }

        let measurement = Measurement {
            name: self.measurement_name.clone(),
            config: MeasurementConfig { poi: self.poi.clone(), parameters: Vec::new() },
        };

        Ok(Workspace {
            channels: ws_channels,
            observations: ws_observations,
            measurements: vec![measurement],
            version: Some("1.0.0".into()),
        })
    }

    /// Process a single channel: fill histograms for all samples.
    fn process_channel(
        &self,
        ch: &ChannelConfig,
        root_cache: &mut HashMap<PathBuf, RootFile>,
    ) -> Result<(Channel, Observation)> {
        let mut samples = Vec::new();

        for sample_cfg in &ch.samples {
            let sample = self.process_sample(ch, sample_cfg, root_cache)?;
            samples.push(sample);
        }

        // Observation: from data file or Asimov (sum of nominals)
        let obs_data = if let Some(ref data_file) = ch.data_file {
            let tree_name = ch.data_tree_name.as_deref().unwrap_or(&self.tree_name);
            let hist = self.fill_one_histogram(
                &self.base_dir.join(data_file),
                tree_name,
                &ch.variable,
                &ch.binning,
                ch.selection.as_deref(),
                None, // no weight for data
                root_cache,
            )?;
            hist.bin_content
        } else {
            // Asimov: sum of nominal histograms
            let n_bins = ch.binning.len() - 1;
            let mut asimov = vec![0.0; n_bins];
            for s in &samples {
                for (i, &v) in s.data.iter().enumerate() {
                    if i < n_bins {
                        asimov[i] += v;
                    }
                }
            }
            asimov
        };

        let observation = Observation { name: ch.name.clone(), data: obs_data };

        let channel = Channel { name: ch.name.clone(), samples };

        Ok((channel, observation))
    }

    /// Process a single sample: fill nominal + systematic histograms.
    fn process_sample(
        &self,
        ch: &ChannelConfig,
        sample: &SampleConfig,
        root_cache: &mut HashMap<PathBuf, RootFile>,
    ) -> Result<Sample> {
        let tree_name = sample.tree_name.as_deref().unwrap_or(&self.tree_name);

        // Fill nominal histogram
        let nominal = self.fill_one_histogram(
            &self.base_dir.join(&sample.file),
            tree_name,
            &ch.variable,
            &ch.binning,
            ch.selection.as_deref(),
            sample.weight.as_deref(),
            root_cache,
        )?;

        // Build modifiers
        let mut modifiers = Vec::new();

        for modifier in &sample.modifiers {
            match modifier {
                NtupleModifier::NormFactor { name } => {
                    modifiers.push(Modifier::NormFactor { name: name.clone(), data: None });
                }
                NtupleModifier::NormSys { name, lo, hi } => {
                    modifiers.push(Modifier::NormSys {
                        name: name.clone(),
                        data: NormSysData { hi: *hi, lo: *lo },
                    });
                }
                NtupleModifier::WeightSys { name, weight_up, weight_down } => {
                    // Same file/tree, different weight expression
                    let hist_up = self.fill_one_histogram(
                        &self.base_dir.join(&sample.file),
                        tree_name,
                        &ch.variable,
                        &ch.binning,
                        ch.selection.as_deref(),
                        Some(weight_up),
                        root_cache,
                    )?;
                    let hist_down = self.fill_one_histogram(
                        &self.base_dir.join(&sample.file),
                        tree_name,
                        &ch.variable,
                        &ch.binning,
                        ch.selection.as_deref(),
                        Some(weight_down),
                        root_cache,
                    )?;
                    modifiers.push(Modifier::HistoSys {
                        name: name.clone(),
                        data: HistoSysData {
                            hi_data: hist_up.bin_content,
                            lo_data: hist_down.bin_content,
                        },
                    });
                }
                NtupleModifier::TreeSys { name, file_up, file_down, tree_name: tn_override } => {
                    let tn = tn_override.as_deref().unwrap_or(tree_name);
                    let hist_up = self.fill_one_histogram(
                        &self.base_dir.join(file_up),
                        tn,
                        &ch.variable,
                        &ch.binning,
                        ch.selection.as_deref(),
                        sample.weight.as_deref(),
                        root_cache,
                    )?;
                    let hist_down = self.fill_one_histogram(
                        &self.base_dir.join(file_down),
                        tn,
                        &ch.variable,
                        &ch.binning,
                        ch.selection.as_deref(),
                        sample.weight.as_deref(),
                        root_cache,
                    )?;
                    modifiers.push(Modifier::HistoSys {
                        name: name.clone(),
                        data: HistoSysData {
                            hi_data: hist_up.bin_content,
                            lo_data: hist_down.bin_content,
                        },
                    });
                }
                NtupleModifier::StatError => {
                    // Use sqrt(sumw2) from nominal
                    let stat_data: Vec<f64> = nominal.sumw2.iter().map(|&s| s.sqrt()).collect();
                    modifiers.push(Modifier::StatError {
                        name: format!("staterror_{}", ch.name),
                        data: stat_data,
                    });
                }
            }
        }

        Ok(Sample { name: sample.name.clone(), data: nominal.bin_content, modifiers })
    }

    /// Fill a single histogram from a ROOT file + TTree.
    fn fill_one_histogram(
        &self,
        root_path: &Path,
        tree_name: &str,
        variable: &str,
        bin_edges: &[f64],
        selection: Option<&str>,
        weight: Option<&str>,
        root_cache: &mut HashMap<PathBuf, RootFile>,
    ) -> Result<FilledHistogram> {
        // Open or reuse cached ROOT file
        let canonical = root_path.to_path_buf();
        if !root_cache.contains_key(&canonical) {
            let rf = RootFile::open(root_path)
                .map_err(|e| Error::RootFile(format!("opening {}: {}", root_path.display(), e)))?;
            root_cache.insert(canonical.clone(), rf);
        }
        let rf = root_cache.get(&canonical).unwrap();

        // Read tree
        let tree = rf.get_tree(tree_name).map_err(|e| {
            Error::RootFile(format!(
                "reading tree '{}' from {}: {}",
                tree_name,
                root_path.display(),
                e
            ))
        })?;

        // Compile expressions
        let var_expr = CompiledExpr::compile(variable)
            .map_err(|e| Error::RootFile(format!("compiling variable '{}': {}", variable, e)))?;
        let weight_expr = weight
            .map(CompiledExpr::compile)
            .transpose()
            .map_err(|e| Error::RootFile(format!("compiling weight: {}", e)))?;
        let sel_expr = selection
            .map(CompiledExpr::compile)
            .transpose()
            .map_err(|e| Error::RootFile(format!("compiling selection: {}", e)))?;

        // Collect all required branches
        let mut all_branches: Vec<String> = var_expr.required_branches.clone();
        if let Some(ref we) = weight_expr {
            for b in &we.required_branches {
                if !all_branches.contains(b) {
                    all_branches.push(b.clone());
                }
            }
        }
        if let Some(ref se) = sel_expr {
            for b in &se.required_branches {
                if !all_branches.contains(b) {
                    all_branches.push(b.clone());
                }
            }
        }

        // Read column data
        let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
        for branch_name in &all_branches {
            let data = rf.branch_data(&tree, branch_name).map_err(|e| {
                Error::RootFile(format!(
                    "reading branch '{}' from {}: {}",
                    branch_name,
                    root_path.display(),
                    e
                ))
            })?;
            columns.insert(branch_name.clone(), data);
        }

        // Fill histogram
        let spec = HistogramSpec {
            name: "nominal".into(),
            variable: var_expr,
            weight: weight_expr,
            selection: sel_expr,
            bin_edges: bin_edges.to_vec(),
            flow_policy: FlowPolicy::Drop,
            negative_weight_policy: NegativeWeightPolicy::Allow,
        };

        let mut results = fill_histograms(&[spec], &columns)
            .map_err(|e| Error::RootFile(format!("filling histogram: {}", e)))?;

        results.pop().ok_or_else(|| Error::RootFile("fill_histograms returned empty".into()))
    }
}

impl Default for NtupleWorkspaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
