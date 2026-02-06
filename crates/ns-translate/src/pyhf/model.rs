//! HistFactory Model representation
//!
//! Converts pyhf Workspace into an internal model suitable for inference.

use super::schema::*;
use ns_ad::scalar::Scalar;
use ns_core::Result;
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel, PreparedNll};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// HistFactory model
#[derive(Debug, Clone)]
pub struct HistFactoryModel {
    /// Model parameters
    parameters: Vec<Parameter>,
    /// Parameter of interest index
    poi_index: Option<usize>,
    /// Channels
    channels: Vec<ModelChannel>,
}

/// Model parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Initial value
    pub init: f64,
    /// Bounds (min, max)
    pub bounds: (f64, f64),
    /// Is this a nuisance parameter with constraint?
    pub constrained: bool,
    /// Constraint center (for constrained NP)
    pub constraint_center: Option<f64>,
    /// Constraint width (for constrained NP)
    pub constraint_width: Option<f64>,
}

/// Model channel
#[derive(Debug, Clone)]
struct ModelChannel {
    /// Channel name
    #[allow(dead_code)]
    name: String,
    /// Samples in this channel
    samples: Vec<ModelSample>,
    /// Observed data (main bins only, auxiliary data added during NLL computation)
    observed: Vec<f64>,
    /// Auxiliary data for Barlow-Beeston constraints (per shapesys modifier)
    /// Each constraint corresponds to one ShapeSys modifier and stores both
    /// the fixed `tau` values and the observed auxiliary counts.
    #[allow(dead_code)]
    auxiliary_data: Vec<AuxiliaryPoissonConstraint>,
}

#[derive(Debug, Clone)]
struct AuxiliaryPoissonConstraint {
    sample_idx: usize,
    modifier_idx: usize,
    /// Barlow-Beeston scale (`tau_i = (nominal_i / sigma_i)^2`).
    tau: Vec<f64>,
    /// Observed auxiliary counts. For the nominal (observed) dataset this equals `tau`.
    /// For Asimov datasets this is set to the expected aux counts `gamma_hat * tau`.
    observed: Vec<f64>,
}

/// Model sample
#[derive(Debug, Clone)]
struct ModelSample {
    /// Sample name
    #[allow(dead_code)]
    name: String,
    /// Nominal expected counts
    nominal: Vec<f64>,
    /// Modifiers
    modifiers: Vec<ModelModifier>,
}

/// Model modifier (processed)
#[derive(Debug, Clone)]
enum ModelModifier {
    /// Normalization factor (unconstrained)
    NormFactor { param_idx: usize },
    /// Shape systematic - per-bin multiplicative (Barlow-Beeston)
    /// Stores parameter indices, uncertainties, and nominal values for constraint
    ShapeSys {
        param_indices: Vec<usize>,
        #[allow(dead_code)]
        uncertainties: Vec<f64>,
        #[allow(dead_code)]
        nominal_values: Vec<f64>,
    },
    /// Shape factor - free per-bin multiplicative factors (unconstrained)
    ShapeFactor { param_indices: Vec<usize> },
    /// Normalization systematic - log-normal constraint
    NormSys { param_idx: usize, hi_factor: f64, lo_factor: f64 },
    /// Histogram systematic - shape interpolation
    HistoSys { param_idx: usize, hi_template: Vec<f64>, lo_template: Vec<f64> },
    /// Statistical error - per-bin multiplicative factors (constrained by normal)
    StatError {
        param_indices: Vec<usize>,
        #[allow(dead_code)]
        uncertainties: Vec<f64>,
    },
    /// Luminosity - normalization with constraint
    Lumi { param_idx: usize },
}

impl HistFactoryModel {
    fn validate_params_len(&self, got: usize) -> Result<()> {
        let expected = self.parameters.len();
        if got != expected {
            return Err(ns_core::Error::Validation(format!(
                "Parameter length mismatch: expected {}, got {}",
                expected, got
            )));
        }
        Ok(())
    }

    fn validate_internal_indices(&self) -> Result<()> {
        let n = self.parameters.len();
        for channel in &self.channels {
            for sample in &channel.samples {
                for m in &sample.modifiers {
                    match m {
                        ModelModifier::NormFactor { param_idx }
                        | ModelModifier::NormSys { param_idx, .. }
                        | ModelModifier::HistoSys { param_idx, .. }
                        | ModelModifier::Lumi { param_idx } => {
                            if *param_idx >= n {
                                return Err(ns_core::Error::Validation(format!(
                                    "Modifier param index out of range: idx={} len={}",
                                    param_idx, n
                                )));
                            }
                        }
                        ModelModifier::ShapeSys { param_indices, .. }
                        | ModelModifier::ShapeFactor { param_indices }
                        | ModelModifier::StatError { param_indices, .. } => {
                            for &idx in param_indices {
                                if idx >= n {
                                    return Err(ns_core::Error::Validation(format!(
                                        "Modifier param index out of range: idx={} len={}",
                                        idx, n
                                    )));
                                }
                            }
                        }
                    }
                }
            }

            for constraint in &channel.auxiliary_data {
                let sample = channel
                    .samples
                    .get(constraint.sample_idx)
                    .ok_or_else(|| ns_core::Error::Validation("Aux constraint sample_idx out of range".to_string()))?;
                let modifier = sample
                    .modifiers
                    .get(constraint.modifier_idx)
                    .ok_or_else(|| ns_core::Error::Validation("Aux constraint modifier_idx out of range".to_string()))?;
                if let ModelModifier::ShapeSys { param_indices, .. } = modifier {
                    if constraint.tau.len() != param_indices.len()
                        || constraint.observed.len() != param_indices.len()
                    {
                        return Err(ns_core::Error::Validation(
                            "Aux constraint length mismatch (tau/observed/param_indices)".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute `ln Γ(n+1)` (generalized factorial).
    fn ln_factorial(n: f64) -> f64 {
        ln_gamma(n + 1.0)
    }

    /// Create model from pyhf workspace
    pub fn from_workspace(ws: &Workspace) -> Result<Self> {
        let mut parameters = Vec::new();
        let mut param_map: HashMap<String, usize> = HashMap::new();

        const POS_LO: f64 = 1e-10;
        const POS_HI: f64 = 10.0;

        // Accumulate nominal/uncertainty sums for staterror sigmas (pyhf behavior).
        #[derive(Debug, Clone)]
        struct StatErrorAccum {
            sum_nominal: Vec<f64>,
            sum_uncert_sq: Vec<f64>,
        }
        let mut staterror_accum: HashMap<String, StatErrorAccum> = HashMap::new();

        // Get POI name from first measurement
        let poi_name = ws.measurements.first().map(|m| m.config.poi.as_str()).unwrap_or("mu");

        // Add POI as first parameter
        param_map.insert(poi_name.to_string(), 0);
        parameters.push(Parameter {
            name: poi_name.to_string(),
            init: 1.0,
            bounds: (0.0, 10.0),
            constrained: false,
            constraint_center: None,
            constraint_width: None,
        });
        let poi_index = Some(0);

        // Collect all unique parameters from modifiers
        // First pass: identify all parameters
        for channel in &ws.channels {
            for sample in &channel.samples {
                for modifier in &sample.modifiers {
                    match modifier {
                        Modifier::NormFactor { name, .. } => {
                            // Free-floating normalization (POI or nuisance).
                            if !param_map.contains_key(name) {
                                param_map.insert(name.clone(), parameters.len());
                                parameters.push(Parameter {
                                    name: name.clone(),
                                    init: 1.0,
                                    bounds: (0.0, POS_HI),
                                    constrained: false,
                                    constraint_center: None,
                                    constraint_width: None,
                                });
                            }
                        }
                        Modifier::ShapeSys { name, data } => {
                            // ShapeSys uses Barlow-Beeston approach with Poisson constraints
                            // Create one gamma parameter per bin
                            for (bin_idx, _uncertainty) in data.iter().enumerate() {
                                let param_name = format!("{}[{}]", name, bin_idx);
                                if !param_map.contains_key(&param_name) {
                                    param_map.insert(param_name.clone(), parameters.len());
                                    parameters.push(Parameter {
                                        name: param_name,
                                        init: 1.0,
                                        bounds: (POS_LO, POS_HI),
                                        // ShapeSys has Poisson constraint (Barlow-Beeston) handled in `nll()`.
                                        constrained: false,
                                        constraint_center: None,
                                        constraint_width: None,
                                    });
                                }
                            }
                        }
                        Modifier::NormSys { name, .. } => {
                            if !param_map.contains_key(name) {
                                param_map.insert(name.clone(), parameters.len());
                                parameters.push(Parameter {
                                    name: name.clone(),
                                    init: 0.0, // log-normal: 0 = nominal
                                    bounds: (-5.0, 5.0),
                                    constrained: true,
                                    constraint_center: Some(0.0),
                                    constraint_width: Some(1.0),
                                });
                            }
                        }
                        Modifier::HistoSys { name, .. } => {
                            if !param_map.contains_key(name) {
                                param_map.insert(name.clone(), parameters.len());
                                parameters.push(Parameter {
                                    name: name.clone(),
                                    init: 0.0, // 0 = nominal, +1/-1 = variations
                                    bounds: (-5.0, 5.0),
                                    constrained: true,
                                    constraint_center: Some(0.0),
                                    constraint_width: Some(1.0),
                                });
                            }
                        }
                        Modifier::StatError { name, data } => {
                            // One parameter per bin
                            for (bin_idx, _) in data.iter().enumerate() {
                                let param_name = format!("{}[{}]", name, bin_idx);
                                if !param_map.contains_key(&param_name) {
                                    param_map.insert(param_name.clone(), parameters.len());
                                    parameters.push(Parameter {
                                        name: param_name,
                                        init: 1.0,
                                        bounds: (POS_LO, POS_HI),
                                        constrained: true,
                                        constraint_center: Some(1.0),
                                        constraint_width: Some(1.0), // Placeholder; computed below.
                                    });
                                }
                            }

                            let entry = staterror_accum.entry(name.clone()).or_insert_with(|| {
                                StatErrorAccum {
                                    sum_nominal: vec![0.0; data.len()],
                                    sum_uncert_sq: vec![0.0; data.len()],
                                }
                            });
                            if entry.sum_nominal.len() != data.len() {
                                return Err(ns_core::Error::Validation(format!(
                                    "StatError modifier '{}' bin length mismatch: {} != {}",
                                    name,
                                    entry.sum_nominal.len(),
                                    data.len()
                                )));
                            }

                            for (bin_idx, (sigma_abs, nominal)) in
                                data.iter().zip(&sample.data).enumerate()
                            {
                                entry.sum_nominal[bin_idx] += *nominal;
                                entry.sum_uncert_sq[bin_idx] += sigma_abs * sigma_abs;
                            }
                        }
                        Modifier::Lumi { name, .. } => {
                            if !param_map.contains_key(name) {
                                param_map.insert(name.clone(), parameters.len());
                                parameters.push(Parameter {
                                    name: name.clone(),
                                    init: 1.0,
                                    bounds: (0.0, POS_HI),
                                    constrained: true,
                                    constraint_center: Some(1.0),
                                    constraint_width: Some(0.02), // typical lumi uncertainty
                                });
                            }
                        }
                        Modifier::ShapeFactor { name, .. } => {
                            // Free-floating shape: one parameter per bin (unconstrained).
                            for bin_idx in 0..sample.data.len() {
                                let param_name = format!("{}[{}]", name, bin_idx);
                                if !param_map.contains_key(&param_name) {
                                    param_map.insert(param_name.clone(), parameters.len());
                                    parameters.push(Parameter {
                                        name: param_name,
                                        init: 1.0,
                                        bounds: (0.0, POS_HI),
                                        constrained: false,
                                        constraint_center: None,
                                        constraint_width: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute StatError sigmas (relative uncertainties) and attach constraints to parameters.
        for (name, accum) in staterror_accum {
            for bin_idx in 0..accum.sum_nominal.len() {
                let denom = accum.sum_nominal[bin_idx];
                let sigma_rel =
                    if denom > 0.0 { accum.sum_uncert_sq[bin_idx].sqrt() / denom } else { 0.0 };

                let param_name = format!("{}[{}]", name, bin_idx);
                if let Some(&pidx) = param_map.get(&param_name) {
                    let p = &mut parameters[pidx];
                    if sigma_rel > 0.0 {
                        p.constrained = true;
                        p.constraint_center = Some(1.0);
                        p.constraint_width = Some(sigma_rel);
                    } else {
                        // If sigma==0, pyhf effectively fixes this parameter at 1.0.
                        p.constrained = false;
                        p.constraint_center = None;
                        p.constraint_width = None;
                        p.init = 1.0;
                        p.bounds = (1.0, 1.0);
                    }
                }
            }
        }

        // Apply measurement parameter overrides (bounds/inits/auxdata/sigmas) if present.
        if let Some(measurement) = ws.measurements.first() {
            for cfg in &measurement.config.parameters {
                for param in &mut parameters {
                    // Match scalar name or vector-like "<base>[<i>]" names.
                    let elem_idx = if param.name == cfg.name {
                        Some(0usize)
                    } else if let Some(rest) = param.name.strip_prefix(&cfg.name) {
                        rest.strip_prefix('[')
                            .and_then(|s| s.strip_suffix(']'))
                            .and_then(|s| s.parse::<usize>().ok())
                    } else {
                        None
                    };

                    let Some(elem_idx) = elem_idx else { continue };

                    if !cfg.inits.is_empty() {
                        let init = if cfg.inits.len() == 1 {
                            cfg.inits[0]
                        } else {
                            *cfg.inits.get(elem_idx).unwrap_or(&param.init)
                        };
                        param.init = init;
                    }

                    if !cfg.bounds.is_empty() {
                        let b = if cfg.bounds.len() == 1 {
                            cfg.bounds[0]
                        } else if let Some(b) = cfg.bounds.get(elem_idx) {
                            *b
                        } else {
                            [param.bounds.0, param.bounds.1]
                        };
                        param.bounds = (b[0], b[1]);
                    }

                    if !cfg.auxdata.is_empty() && !cfg.sigmas.is_empty() {
                        let center = if cfg.auxdata.len() == 1 {
                            cfg.auxdata[0]
                        } else if let Some(v) = cfg.auxdata.get(elem_idx) {
                            *v
                        } else {
                            param.constraint_center.unwrap_or(param.init)
                        };
                        let sigma = if cfg.sigmas.len() == 1 {
                            cfg.sigmas[0]
                        } else if let Some(v) = cfg.sigmas.get(elem_idx) {
                            *v
                        } else {
                            param.constraint_width.unwrap_or(1.0)
                        };
                        param.constrained = true;
                        param.constraint_center = Some(center);
                        param.constraint_width = Some(sigma);
                    }
                }
            }
        }

        // Build channels
        let mut channels = Vec::new();
        for ws_channel in &ws.channels {
            let observed = ws
                .observations
                .iter()
                .find(|o| o.name == ws_channel.name)
                .ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "Workspace missing observations for channel '{}'",
                        ws_channel.name
                    ))
                })?
                .data
                .clone();

            // Validate binning consistency within the channel before building the internal model.
            let n_bins = ws_channel.samples.first().map(|s| s.data.len()).unwrap_or(0);
            if n_bins == 0 {
                return Err(ns_core::Error::Validation(format!(
                    "Channel '{}' has no bins (empty sample data)",
                    ws_channel.name
                )));
            }
            if observed.len() != n_bins {
                return Err(ns_core::Error::Validation(format!(
                    "Channel '{}' observations length mismatch: expected {}, got {}",
                    ws_channel.name,
                    n_bins,
                    observed.len()
                )));
            }

            let mut samples = Vec::new();
            let mut auxiliary_data = Vec::new();

            for (sample_idx, ws_sample) in ws_channel.samples.iter().enumerate() {
                if ws_sample.data.len() != n_bins {
                    return Err(ns_core::Error::Validation(format!(
                        "Channel '{}' sample '{}' bin length mismatch: expected {}, got {}",
                        ws_channel.name,
                        ws_sample.name,
                        n_bins,
                        ws_sample.data.len()
                    )));
                }

                let mut modifiers = Vec::new();

                for ws_modifier in &ws_sample.modifiers {
                    match ws_modifier {
                        Modifier::NormFactor { name, .. } => {
                            if let Some(&idx) = param_map.get(name) {
                                modifiers.push(ModelModifier::NormFactor { param_idx: idx });
                            }
                        }
                        Modifier::ShapeSys { name, data } => {
                            if data.len() != n_bins {
                                return Err(ns_core::Error::Validation(format!(
                                    "Channel '{}' sample '{}' ShapeSys '{}' bin length mismatch: expected {}, got {}",
                                    ws_channel.name,
                                    ws_sample.name,
                                    name,
                                    n_bins,
                                    data.len()
                                )));
                            }

                            let mut param_indices = Vec::new();
                            for (bin_idx, _) in data.iter().enumerate() {
                                let param_name = format!("{}[{}]", name, bin_idx);
                                if let Some(&idx) = param_map.get(&param_name) {
                                    param_indices.push(idx);
                                }
                            }

                            // Compute auxiliary data: tau_i = (nominal_i / sigma_i)^2
                            let tau_values: Vec<f64> = data
                                .iter()
                                .zip(&ws_sample.data)
                                .map(|(sigma, nominal)| {
                                    if *sigma > 0.0 && *nominal > 0.0 {
                                        (nominal / sigma).powi(2)
                                    } else {
                                        1.0
                                    }
                                })
                                .collect();

                            // Store auxiliary constraint for this modifier.
                            // For the observed dataset, the aux observation equals tau (pyhf auxdata convention).
                            auxiliary_data.push(AuxiliaryPoissonConstraint {
                                sample_idx,
                                modifier_idx: modifiers.len(),
                                tau: tau_values.clone(),
                                observed: tau_values.clone(),
                            });

                            modifiers.push(ModelModifier::ShapeSys {
                                param_indices,
                                uncertainties: data.clone(),
                                nominal_values: ws_sample.data.clone(),
                            });
                        }
                        Modifier::NormSys { name, data } => {
                            if let Some(&idx) = param_map.get(name) {
                                modifiers.push(ModelModifier::NormSys {
                                    param_idx: idx,
                                    hi_factor: data.hi,
                                    lo_factor: data.lo,
                                });
                            }
                        }
                        Modifier::HistoSys { name, data } => {
                            if data.hi_data.len() != n_bins || data.lo_data.len() != n_bins {
                                return Err(ns_core::Error::Validation(format!(
                                    "Channel '{}' sample '{}' HistoSys '{}' template length mismatch: expected {}, got hi={}, lo={}",
                                    ws_channel.name,
                                    ws_sample.name,
                                    name,
                                    n_bins,
                                    data.hi_data.len(),
                                    data.lo_data.len()
                                )));
                            }

                            if let Some(&idx) = param_map.get(name) {
                                modifiers.push(ModelModifier::HistoSys {
                                    param_idx: idx,
                                    hi_template: data.hi_data.clone(),
                                    lo_template: data.lo_data.clone(),
                                });
                            }
                        }
                        Modifier::StatError { name, data } => {
                            if data.len() != n_bins {
                                return Err(ns_core::Error::Validation(format!(
                                    "Channel '{}' sample '{}' StatError '{}' bin length mismatch: expected {}, got {}",
                                    ws_channel.name,
                                    ws_sample.name,
                                    name,
                                    n_bins,
                                    data.len()
                                )));
                            }

                            let mut param_indices = Vec::new();
                            for (bin_idx, _) in data.iter().enumerate() {
                                let param_name = format!("{}[{}]", name, bin_idx);
                                if let Some(&idx) = param_map.get(&param_name) {
                                    param_indices.push(idx);
                                }
                            }
                            modifiers.push(ModelModifier::StatError {
                                param_indices,
                                uncertainties: data.clone(),
                            });
                        }
                        Modifier::ShapeFactor { name, .. } => {
                            let mut param_indices = Vec::new();
                            for bin_idx in 0..ws_sample.data.len() {
                                let param_name = format!("{}[{}]", name, bin_idx);
                                if let Some(&idx) = param_map.get(&param_name) {
                                    param_indices.push(idx);
                                }
                            }
                            modifiers.push(ModelModifier::ShapeFactor { param_indices });
                        }
                        Modifier::Lumi { name, .. } => {
                            if let Some(&idx) = param_map.get(name) {
                                modifiers.push(ModelModifier::Lumi { param_idx: idx });
                            }
                        }
                    }
                }

                samples.push(ModelSample {
                    name: ws_sample.name.clone(),
                    nominal: ws_sample.data.clone(),
                    modifiers,
                });
            }

            channels.push(ModelChannel {
                name: ws_channel.name.clone(),
                samples,
                observed,
                auxiliary_data,
            });
        }

        // pyhf orders channels lexicographically (see `model.config.channels`), and the flattened
        // main-data vector follows that order. Sort here so `with_observed_main(...)` can accept
        // pyhf-ordered observations without requiring callers to permute.
        channels.sort_by(|a, b| a.name.cmp(&b.name));

        let model = Self { parameters, poi_index, channels };
        model.validate_internal_indices()?;
        Ok(model)
    }

    /// Number of parameters
    pub fn n_params(&self) -> usize {
        self.parameters.len()
    }

    /// Create a copy of the model with overridden observed (main) data.
    ///
    /// `observed_main` must be a flat vector of length equal to the total number of
    /// main bins across all channels (no auxdata). Auxiliary constraints remain fixed.
    pub fn with_observed_main(&self, observed_main: &[f64]) -> Result<Self> {
        let expected_len: usize = self
            .channels
            .iter()
            .map(|channel| channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0))
            .sum();

        if observed_main.len() != expected_len {
            return Err(ns_core::Error::Validation(format!(
                "Observed main data length mismatch: expected {}, got {}",
                expected_len,
                observed_main.len()
            )));
        }

        let mut out = self.clone();
        let mut offset = 0;
        for channel in &mut out.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            channel.observed.clear();
            channel.observed.extend_from_slice(&observed_main[offset..offset + n_bins]);
            offset += n_bins;
        }

        Ok(out)
    }

    /// Get POI index
    pub fn poi_index(&self) -> Option<usize> {
        self.poi_index
    }

    /// Get parameters
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    /// Create a copy with one parameter fixed at a given value.
    ///
    /// The parameter bounds are clamped to that value so the optimizer
    /// cannot move it.
    pub fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        let mut out = self.clone();
        if let Some(p) = out.parameters.get_mut(param_idx) {
            p.init = value;
            p.bounds = (value, value);
        }
        out
    }

    /// Create a copy with updated constraint centers for constrained parameters.
    ///
    /// This is primarily used for building Asimov datasets compatible with pyhf:
    /// for `constrained_by_normal` terms, pyhf treats the auxiliary measurement as
    /// part of the data vector and the expected aux value is the nuisance parameter
    /// itself. In the Asimov dataset this means the auxiliary "observations" equal
    /// the fitted nuisance values, effectively removing pulls while keeping widths.
    pub fn with_constraint_centers(&self, centers: &[f64]) -> Result<Self> {
        if centers.len() != self.parameters.len() {
            return Err(ns_core::Error::Validation(format!(
                "Constraint centers length mismatch: expected {}, got {}",
                self.parameters.len(),
                centers.len()
            )));
        }

        let mut out = self.clone();
        for (i, p) in out.parameters.iter_mut().enumerate() {
            if !p.constrained {
                continue;
            }
            if p.constraint_center.is_some() {
                p.constraint_center = Some(centers[i]);
            }
        }
        Ok(out)
    }

    /// Create a copy with updated **ShapeSys auxiliary observations** derived from parameter values.
    ///
    /// For the Barlow-Beeston (ShapeSys) constraints, pyhf includes auxiliary Poisson terms
    /// as part of the data vector. When building an Asimov dataset, those auxiliary
    /// observations are set to their expected values `gamma_hat * tau`.
    ///
    /// This helper updates the stored auxiliary observations accordingly while keeping `tau`
    /// fixed.
    pub fn with_shapesys_aux_observed_from_params(&self, params: &[f64]) -> Result<Self> {
        self.validate_params_len(params.len())?;

        let mut out = self.clone();
        for channel in &mut out.channels {
            for constraint in &mut channel.auxiliary_data {
                if let Some(sample) = channel.samples.get(constraint.sample_idx)
                    && let Some(ModelModifier::ShapeSys { param_indices, .. }) =
                        sample.modifiers.get(constraint.modifier_idx)
                {
                    if constraint.tau.len() != param_indices.len() {
                        return Err(ns_core::Error::Validation(format!(
                            "Aux constraint length mismatch: tau={} param_indices={}",
                            constraint.tau.len(),
                            param_indices.len()
                        )));
                    }
                    let mut obs = Vec::with_capacity(constraint.tau.len());
                    for (&tau, &gamma_idx) in constraint.tau.iter().zip(param_indices.iter()) {
                        let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                            ns_core::Error::Validation(format!(
                                "ShapeSys gamma index out of range: idx={} len={}",
                                gamma_idx,
                                params.len()
                            ))
                        })?;
                        obs.push(gamma * tau);
                    }
                    constraint.observed = obs;
                }
            }
        }
        Ok(out)
    }

    /// Compute negative log-likelihood (f64 specialisation).
    pub fn nll(&self, params: &[f64]) -> Result<f64> {
        self.validate_params_len(params.len())?;
        self.nll_generic(params)
    }

    /// Compute expected data at given parameter values (f64 specialisation).
    pub fn expected_data(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.validate_params_len(params.len())?;
        self.expected_data_generic(params)
    }

    /// Expected **main** data in pyhf ordering (channels lexicographically), without auxdata.
    pub fn expected_data_pyhf_main(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.validate_params_len(params.len())?;

        let expected_main_flat: Vec<f64> = self.expected_data(params)?;

        let mut per_channel: Vec<(&str, Vec<f64>)> = Vec::with_capacity(self.channels.len());
        let mut offset = 0usize;
        for channel in &self.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            let slice = expected_main_flat
                .get(offset..offset + n_bins)
                .ok_or_else(|| {
                    ns_core::Error::Validation("expected_data length mismatch".to_string())
                })?
                .to_vec();
            offset += n_bins;
            per_channel.push((channel.name.as_str(), slice));
        }
        per_channel.sort_by(|a, b| a.0.cmp(b.0));

        let mut out: Vec<f64> = Vec::with_capacity(expected_main_flat.len());
        for (_name, bins) in per_channel {
            out.extend(bins);
        }
        Ok(out)
    }

    /// Expected data in pyhf ordering: `main_data + auxdata`.
    ///
    /// - Main data is ordered by channel name (lexicographically).
    /// - Auxdata is ordered according to pyhf's parameter-set construction:
    ///   modifier types are scanned in `histfactory_set` order and modifier names are sorted.
    /// - For `constrained_by_normal` parameter sets, the aux expectation is the parameter value.
    /// - For `shapesys` (Barlow-Beeston) Poisson constraints, the aux expectation is `gamma_i * tau_i`.
    pub fn expected_data_pyhf(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.validate_params_len(params.len())?;

        let mut out = self.expected_data_pyhf_main(params)?;
        out.reserve(self.parameters.len());

        // Auxdata in pyhf parameter-set order:
        // - parameter sets are collected by modifier type in `histfactory_set` order
        //   (`histosys`, `lumi`, `normfactor`, `normsys`, `shapefactor`, `shapesys`, `staterror`)
        // - modifier names are sorted within each type
        // - only constrained parameter sets contribute auxdata
        //
        // For constrained_by_normal parameter sets, the aux expectation equals the parameter value.
        // For constrained_by_poisson (shapesys), the aux expectation equals `gamma_i * tau_i`.
        fn base_name(name: &str) -> &str {
            name.split_once('[').map(|(b, _)| b).unwrap_or(name)
        }

        let mut histosys: HashMap<String, usize> = HashMap::new();
        let mut lumi: HashMap<String, usize> = HashMap::new();
        let mut normsys: HashMap<String, usize> = HashMap::new();
        let mut staterror: HashMap<String, Vec<usize>> = HashMap::new();

        // shapesys needs tau values, which are stored in `auxiliary_data`.
        let mut shapesys: HashMap<String, (Vec<usize>, Vec<f64>)> = HashMap::new();

        // Collect parameter-set names and indices from modifiers.
        for channel in &self.channels {
            for sample in &channel.samples {
                for m in &sample.modifiers {
                    match m {
                        ModelModifier::HistoSys { param_idx, .. } => {
                            if let Some(p) = self.parameters.get(*param_idx) {
                                histosys.insert(p.name.clone(), *param_idx);
                            }
                        }
                        ModelModifier::Lumi { param_idx } => {
                            if let Some(p) = self.parameters.get(*param_idx) {
                                lumi.insert(p.name.clone(), *param_idx);
                            }
                        }
                        ModelModifier::NormSys { param_idx, .. } => {
                            if let Some(p) = self.parameters.get(*param_idx) {
                                normsys.insert(p.name.clone(), *param_idx);
                            }
                        }
                        ModelModifier::StatError { param_indices, .. } => {
                            let b = self
                                .parameters
                                .get(*param_indices.first().unwrap_or(&usize::MAX))
                                .map(|p| base_name(p.name.as_str()).to_string())
                                .unwrap_or_default();
                            if !b.is_empty() {
                                match staterror.get(&b) {
                                    None => {
                                        staterror.insert(b, param_indices.clone());
                                    }
                                    Some(prev) => {
                                        if prev != param_indices {
                                            return Err(ns_core::Error::Validation(format!(
                                                "Inconsistent StatError param indices for '{}'",
                                                b
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Build shapesys mapping from stored auxiliary constraints.
        for channel in &self.channels {
            for constraint in &channel.auxiliary_data {
                let Some(sample) = channel.samples.get(constraint.sample_idx) else { continue };
                let Some(ModelModifier::ShapeSys { param_indices, .. }) =
                    sample.modifiers.get(constraint.modifier_idx)
                else {
                    continue;
                };

                let b = self
                    .parameters
                    .get(*param_indices.first().unwrap_or(&usize::MAX))
                    .map(|p| base_name(p.name.as_str()).to_string())
                    .unwrap_or_default();
                if b.is_empty() {
                    continue;
                }

                match shapesys.get(&b) {
                    None => {
                        shapesys.insert(b, (param_indices.clone(), constraint.tau.clone()));
                    }
                    Some((prev_idx, prev_tau)) => {
                        if prev_idx != param_indices || prev_tau != &constraint.tau {
                            return Err(ns_core::Error::Validation(format!(
                                "Inconsistent ShapeSys definition for '{}'",
                                b
                            )));
                        }
                    }
                }
            }
        }

        fn sorted_keys<T>(m: &HashMap<String, T>) -> Vec<String> {
            let mut v: Vec<String> = m.keys().cloned().collect();
            v.sort();
            v
        }

        for name in sorted_keys(&histosys) {
            let idx = *histosys.get(&name).ok_or_else(|| {
                ns_core::Error::Validation(format!("Missing Histosys index for '{}'", name))
            })?;
            let val = params.get(idx).copied().ok_or_else(|| {
                ns_core::Error::Validation(format!(
                    "Histosys param index out of range for '{}': idx={} len={}",
                    name,
                    idx,
                    params.len()
                ))
            })?;
            out.push(val);
        }
        for name in sorted_keys(&lumi) {
            let idx = *lumi.get(&name).ok_or_else(|| {
                ns_core::Error::Validation(format!("Missing Lumi index for '{}'", name))
            })?;
            let val = params.get(idx).copied().ok_or_else(|| {
                ns_core::Error::Validation(format!(
                    "Lumi param index out of range for '{}': idx={} len={}",
                    name,
                    idx,
                    params.len()
                ))
            })?;
            out.push(val);
        }
        for name in sorted_keys(&normsys) {
            let idx = *normsys.get(&name).ok_or_else(|| {
                ns_core::Error::Validation(format!("Missing NormSys index for '{}'", name))
            })?;
            let val = params.get(idx).copied().ok_or_else(|| {
                ns_core::Error::Validation(format!(
                    "NormSys param index out of range for '{}': idx={} len={}",
                    name,
                    idx,
                    params.len()
                ))
            })?;
            out.push(val);
        }
        for name in sorted_keys(&shapesys) {
            let (param_indices, tau) = shapesys.get(&name).ok_or_else(|| {
                ns_core::Error::Validation(format!("Missing ShapeSys definition for '{}'", name))
            })?;
            if param_indices.len() != tau.len() {
                return Err(ns_core::Error::Validation(format!(
                    "ShapeSys aux length mismatch for '{}': params={} tau={}",
                    name,
                    param_indices.len(),
                    tau.len()
                )));
            }
            for (&pidx, &tau_i) in param_indices.iter().zip(tau.iter()) {
                let gamma = params.get(pidx).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "ShapeSys param index out of range for '{}': idx={} len={}",
                        name,
                        pidx,
                        params.len()
                    ))
                })?;
                out.push((gamma * tau_i).max(1e-10));
            }
        }
        for name in sorted_keys(&staterror) {
            let idxs = staterror.get(&name).ok_or_else(|| {
                ns_core::Error::Validation(format!("Missing StatError indices for '{}'", name))
            })?;
            for &pidx in idxs {
                let gamma = params.get(pidx).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "StatError param index out of range for '{}': idx={} len={}",
                        name,
                        pidx,
                        params.len()
                    ))
                })?;
                out.push(gamma);
            }
        }

        Ok(out)
    }

    /// Generic NLL that works with any [`Scalar`] type (f64 or Dual).
    pub fn nll_generic<T: Scalar>(&self, params: &[T]) -> Result<T> {
        self.validate_params_len(params.len())?;
        let expected = self.expected_data_generic(params)?;

        let mut nll = T::from_f64(0.0);

        // Poisson likelihood for main bins
        let mut bin_idx = 0;
        for channel in &self.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            for i in 0..n_bins {
                let obs = channel.observed.get(i).copied().unwrap_or(0.0);
                let exp = expected.get(bin_idx).copied().unwrap_or(T::from_f64(1e-10));
                let exp = exp.max_s(T::from_f64(1e-10));

                if obs > 0.0 {
                    let ln_factorial = T::from_f64(Self::ln_factorial(obs));
                    nll = nll + exp - T::from_f64(obs) * exp.ln() + ln_factorial;
                } else {
                    nll = nll + exp;
                }

                bin_idx += 1;
            }

            // Barlow-Beeston auxiliary data (Poisson constraints)
            for constraint in &channel.auxiliary_data {
                if let Some(sample) = channel.samples.get(constraint.sample_idx)
                    && let Some(ModelModifier::ShapeSys { param_indices, .. }) =
                        sample.modifiers.get(constraint.modifier_idx)
                {
                    for ((&tau, &obs_aux), &gamma_idx) in constraint
                        .tau
                        .iter()
                        .zip(constraint.observed.iter())
                        .zip(param_indices.iter())
                    {
                        let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                            ns_core::Error::Validation(format!(
                                "ShapeSys gamma index out of range in aux constraint: idx={} len={}",
                                gamma_idx,
                                params.len()
                            ))
                        })?;
                        let exp_aux = (gamma * T::from_f64(tau)).max_s(T::from_f64(1e-10));

                        if obs_aux > 0.0 {
                            let ln_factorial = T::from_f64(Self::ln_factorial(obs_aux));
                            nll =
                                nll + exp_aux - T::from_f64(obs_aux) * exp_aux.ln() + ln_factorial;
                        } else {
                            nll = nll + exp_aux;
                        }
                    }
                }
            }
        }

        // Gaussian constraints (pyhf `constrained_by_normal`)
        for (param_idx, param) in self.parameters.iter().enumerate() {
            if !param.constrained {
                continue;
            }
            if let (Some(center), Some(width)) = (param.constraint_center, param.constraint_width)
                && width > 0.0
            {
                let value = params.get(param_idx).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "Constrained parameter index out of range: idx={} len={}",
                        param_idx,
                        params.len()
                    ))
                })?;
                let pull = (value - T::from_f64(center)) * T::from_f64(1.0 / width);
                nll = nll
                    + T::from_f64(0.5) * pull * pull
                    + T::from_f64(width.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln());
            }
        }

        Ok(nll)
    }

    /// Generic expected data computation.
    pub fn expected_data_generic<T: Scalar>(&self, params: &[T]) -> Result<Vec<T>> {
        self.validate_params_len(params.len())?;
        let mut result = Vec::new();

        for channel in &self.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            let mut channel_expected: Vec<T> = vec![T::from_f64(0.0); n_bins];

            for sample in &channel.samples {
                // Match pyhf combination semantics:
                // - "addition" modifiers (e.g. histosys) produce deltas in nominal space
                // - "multiplication" modifiers produce factors (scalar or per-bin)
                // - expected = (nominal + sum(deltas)) * product(factors)
                let sample_nominal: Vec<T> =
                    sample.nominal.iter().map(|&v| T::from_f64(v)).collect();
                let mut sample_deltas: Vec<T> = vec![T::from_f64(0.0); sample_nominal.len()];
                let mut sample_factors: Vec<T> = vec![T::from_f64(1.0); sample_nominal.len()];

                for modifier in &sample.modifiers {
                    match modifier {
                        ModelModifier::NormFactor { param_idx } => {
                            let norm = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "NormFactor param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            for fac in &mut sample_factors {
                                *fac = *fac * norm;
                            }
                        }
                        ModelModifier::ShapeSys { param_indices, .. } => {
                            for (bin_idx, &gamma_idx) in param_indices.iter().enumerate() {
                                if bin_idx < sample_factors.len() {
                                    let gamma_val =
                                        params.get(gamma_idx).copied().ok_or_else(|| {
                                            ns_core::Error::Validation(format!(
                                                "ShapeSys gamma index out of range: idx={} len={}",
                                                gamma_idx,
                                                params.len()
                                            ))
                                        })?;
                                    sample_factors[bin_idx] = sample_factors[bin_idx] * gamma_val;
                                }
                            }
                        }
                        ModelModifier::NormSys { param_idx, hi_factor, lo_factor } => {
                            let alpha = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "NormSys param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            let factor = normsys_code4(alpha, *hi_factor, *lo_factor);
                            for fac in &mut sample_factors {
                                *fac = *fac * factor;
                            }
                        }
                        ModelModifier::HistoSys { param_idx, hi_template, lo_template } => {
                            let alpha = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "HistoSys param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            for (bin_idx, delta_slot) in sample_deltas.iter_mut().enumerate() {
                                let nom = sample_nominal
                                    .get(bin_idx)
                                    .copied()
                                    .unwrap_or(T::from_f64(0.0));
                                let nom_val = nom.value();
                                let hi = T::from_f64(
                                    hi_template.get(bin_idx).copied().unwrap_or(nom_val),
                                );
                                let lo = T::from_f64(
                                    lo_template.get(bin_idx).copied().unwrap_or(nom_val),
                                );
                                let delta = histosys_code4p_delta(alpha, lo, nom, hi);
                                *delta_slot = *delta_slot + delta;
                            }
                        }
                        ModelModifier::StatError { param_indices, .. } => {
                            for (bin_idx, &gamma_idx) in param_indices.iter().enumerate() {
                                if bin_idx < sample_factors.len() {
                                    let gamma_val =
                                        params.get(gamma_idx).copied().ok_or_else(|| {
                                            ns_core::Error::Validation(format!(
                                                "StatError gamma index out of range: idx={} len={}",
                                                gamma_idx,
                                                params.len()
                                            ))
                                        })?;
                                    sample_factors[bin_idx] = sample_factors[bin_idx] * gamma_val;
                                }
                            }
                        }
                        ModelModifier::ShapeFactor { param_indices } => {
                            for (bin_idx, &gamma_idx) in param_indices.iter().enumerate() {
                                if bin_idx < sample_factors.len() {
                                    let gamma_val =
                                        params.get(gamma_idx).copied().ok_or_else(|| {
                                            ns_core::Error::Validation(format!(
                                                "ShapeFactor gamma index out of range: idx={} len={}",
                                                gamma_idx,
                                                params.len()
                                            ))
                                        })?;
                                    sample_factors[bin_idx] = sample_factors[bin_idx] * gamma_val;
                                }
                            }
                        }
                        ModelModifier::Lumi { param_idx } => {
                            let lumi = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "Lumi param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            for fac in &mut sample_factors {
                                *fac = *fac * lumi;
                            }
                        }
                    }
                }

                for (bin_idx, ch_val) in channel_expected.iter_mut().enumerate() {
                    let nom = sample_nominal.get(bin_idx).copied().unwrap_or(T::from_f64(0.0));
                    let delta = sample_deltas.get(bin_idx).copied().unwrap_or(T::from_f64(0.0));
                    let fac = sample_factors.get(bin_idx).copied().unwrap_or(T::from_f64(1.0));
                    *ch_val = *ch_val + (nom + delta) * fac;
                }
            }

            result.extend(channel_expected);
        }

        Ok(result)
    }

    /// Compute gradient of NLL using central finite differences (fallback).
    pub fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        use rayon::prelude::*;

        self.validate_params_len(params.len())?;
        let n = params.len();
        let mut grad = vec![0.0; n];

        let grad_vals: Vec<_> = (0..n)
            .into_par_iter()
            .map(|i| {
                let eps = 1e-8_f64.sqrt() * params[i].abs().max(1.0);

                let mut params_plus = params.to_vec();
                params_plus[i] += eps;
                let f_plus = self.nll(&params_plus)?;

                let mut params_minus = params.to_vec();
                params_minus[i] -= eps;
                let f_minus = self.nll(&params_minus)?;

                Ok((f_plus - f_minus) / (2.0 * eps))
            })
            .collect::<Result<Vec<f64>>>()?;

        grad.copy_from_slice(&grad_vals);
        Ok(grad)
    }

    /// Compute gradient of NLL using forward-mode automatic differentiation.
    ///
    /// More accurate than finite differences and avoids step-size sensitivity.
    /// Cost: N evaluations (one per parameter), same as numerical gradient.
    pub fn gradient_ad(&self, params: &[f64]) -> Result<Vec<f64>> {
        use ns_ad::dual::Dual;

        self.validate_params_len(params.len())?;
        let n = params.len();
        let mut grad = vec![0.0; n];

        for (i, g) in grad.iter_mut().enumerate() {
            let dual_params: Vec<Dual> = params
                .iter()
                .enumerate()
                .map(|(j, &v)| if j == i { Dual::var(v) } else { Dual::constant(v) })
                .collect();

            let nll_dual = self.nll_generic(&dual_params)?;
            *g = nll_dual.dot;
        }

        Ok(grad)
    }

    /// Compute gradient of NLL using reverse-mode automatic differentiation.
    ///
    /// Cost: **one** forward + backward pass for the **full** gradient,
    /// regardless of parameter count.  Preferred for N > ~10 parameters.
    pub fn gradient_reverse(&self, params: &[f64]) -> Result<Vec<f64>> {
        use ns_ad::tape::Tape;

        self.validate_params_len(params.len())?;
        let n = params.len();
        // Pre-allocate tape: rough estimate of nodes per NLL evaluation
        let mut tape = Tape::with_capacity(n * 20);

        // Record input variables
        let param_vars: Vec<_> = params.iter().map(|&v| tape.var(v)).collect();

        // Build NLL on tape
        let nll_var = self.nll_on_tape(&mut tape, &param_vars)?;

        // Single backward pass => all gradients
        tape.backward(nll_var);

        // Collect gradients
        let grad: Vec<f64> = param_vars.iter().map(|&v| tape.adjoint(v)).collect();
        Ok(grad)
    }

    /// Record the full NLL computation on a [`Tape`](ns_ad::tape::Tape).
    fn nll_on_tape(
        &self,
        tape: &mut ns_ad::tape::Tape,
        params: &[ns_ad::tape::Var],
    ) -> Result<ns_ad::tape::Var> {
        self.validate_params_len(params.len())?;
        let expected = self.expected_data_on_tape(tape, params)?;
        let mut nll = tape.constant(0.0);

        // Poisson likelihood for main bins
        let mut bin_idx = 0;
        for channel in &self.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            for i in 0..n_bins {
                let obs = channel.observed.get(i).copied().unwrap_or(0.0);
                let exp = expected.get(bin_idx).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "expected_data length mismatch on tape: bin_idx={} len={}",
                        bin_idx,
                        expected.len()
                    ))
                })?;
                let floor = tape.constant(1e-10);
                let exp = tape.max(exp, floor);

                if obs > 0.0 {
                    let ln_fact = tape.constant(Self::ln_factorial(obs));
                    let ln_exp = tape.ln(exp);
                    let obs_c = tape.constant(obs);
                    let obs_ln_exp = tape.mul(obs_c, ln_exp);
                    let bin_nll = tape.sub(exp, obs_ln_exp);
                    let bin_nll = tape.add(bin_nll, ln_fact);
                    nll = tape.add(nll, bin_nll);
                } else {
                    nll = tape.add(nll, exp);
                }

                bin_idx += 1;
            }

            // Barlow-Beeston auxiliary data
            for constraint in &channel.auxiliary_data {
                if let Some(sample) = channel.samples.get(constraint.sample_idx)
                    && let Some(ModelModifier::ShapeSys { param_indices, .. }) =
                        sample.modifiers.get(constraint.modifier_idx)
                {
                    for ((&tau, &obs_aux), &gamma_idx) in constraint
                        .tau
                        .iter()
                        .zip(constraint.observed.iter())
                        .zip(param_indices.iter())
                    {
                        let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                            ns_core::Error::Validation(format!(
                                "ShapeSys gamma index out of range in aux constraint (tape): idx={} len={}",
                                gamma_idx,
                                params.len()
                            ))
                        })?;
                        let tau_c = tape.constant(tau);
                        let exp_aux = tape.mul(gamma, tau_c);
                        let floor = tape.constant(1e-10);
                        let exp_aux = tape.max(exp_aux, floor);

                        if obs_aux > 0.0 {
                            let ln_fact = tape.constant(Self::ln_factorial(obs_aux));
                            let ln_exp = tape.ln(exp_aux);
                            let obs_c = tape.constant(obs_aux);
                            let obs_ln = tape.mul(obs_c, ln_exp);
                            let bin_nll = tape.sub(exp_aux, obs_ln);
                            let bin_nll = tape.add(bin_nll, ln_fact);
                            nll = tape.add(nll, bin_nll);
                        } else {
                            nll = tape.add(nll, exp_aux);
                        }
                    }
                }
            }
        }

        // Gaussian constraints (pyhf `constrained_by_normal`)
        for (param_idx, param) in self.parameters.iter().enumerate() {
            if !param.constrained {
                continue;
            }
            if let (Some(center), Some(width)) = (param.constraint_center, param.constraint_width)
                && width > 0.0
            {
                let value = params.get(param_idx).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "Constrained parameter index out of range (tape): idx={} len={}",
                        param_idx,
                        params.len()
                    ))
                })?;
                let center_c = tape.constant(center);
                let diff = tape.sub(value, center_c);
                let inv_width = tape.constant(1.0 / width);
                let pull = tape.mul(diff, inv_width);
                let pull2 = tape.mul(pull, pull);
                let half = tape.constant(0.5);
                let constraint = tape.mul(half, pull2);
                nll = tape.add(nll, constraint);

                // Normalization constant: ln(sigma) + 0.5*ln(2*pi)
                // pyhf includes this in the full Gaussian log-pdf.
                let norm_const =
                    tape.constant(width.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln());
                nll = tape.add(nll, norm_const);
            }
        }

        Ok(nll)
    }

    /// Record expected data computation on a tape.
    fn expected_data_on_tape(
        &self,
        tape: &mut ns_ad::tape::Tape,
        params: &[ns_ad::tape::Var],
    ) -> Result<Vec<ns_ad::tape::Var>> {
        type Var = ns_ad::tape::Var;
        self.validate_params_len(params.len())?;
        let mut result = Vec::new();

        for channel in &self.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            let mut channel_expected: Vec<Var> = (0..n_bins).map(|_| tape.constant(0.0)).collect();

            for sample in &channel.samples {
                // Match pyhf: (nominal + sum(deltas)) * product(factors).
                let sample_nominal: Vec<Var> =
                    sample.nominal.iter().map(|&v| tape.constant(v)).collect();
                let mut sample_deltas: Vec<Var> =
                    (0..sample_nominal.len()).map(|_| tape.constant(0.0)).collect();
                let mut sample_factors: Vec<Var> =
                    (0..sample_nominal.len()).map(|_| tape.constant(1.0)).collect();

                for modifier in &sample.modifiers {
                    match modifier {
                        ModelModifier::NormFactor { param_idx } => {
                            let norm = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "NormFactor param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            for fac in &mut sample_factors {
                                *fac = tape.mul(*fac, norm);
                            }
                        }
                        ModelModifier::ShapeSys { param_indices, .. } => {
                            for (bin_idx, &gamma_idx) in param_indices.iter().enumerate() {
                                if bin_idx < sample_factors.len() {
                                    let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                                        ns_core::Error::Validation(format!(
                                            "ShapeSys gamma index out of range: idx={} len={}",
                                            gamma_idx,
                                            params.len()
                                        ))
                                    })?;
                                    sample_factors[bin_idx] =
                                        tape.mul(sample_factors[bin_idx], gamma);
                                }
                            }
                        }
                        ModelModifier::NormSys { param_idx, hi_factor, lo_factor } => {
                            let alpha = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "NormSys param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            let factor = normsys_code4_on_tape(tape, alpha, *hi_factor, *lo_factor);
                            for fac in &mut sample_factors {
                                *fac = tape.mul(*fac, factor);
                            }
                        }
                        ModelModifier::HistoSys { param_idx, hi_template, lo_template } => {
                            let alpha = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "HistoSys param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            for (bin_idx, delta_slot) in sample_deltas.iter_mut().enumerate() {
                                let nom = sample_nominal
                                    .get(bin_idx)
                                    .copied()
                                    .unwrap_or(tape.constant(0.0));
                                let nom_val = tape.val(nom);
                                let hi_val = hi_template.get(bin_idx).copied().unwrap_or(nom_val);
                                let lo_val = lo_template.get(bin_idx).copied().unwrap_or(nom_val);

                                let hi = tape.constant(hi_val);
                                let lo = tape.constant(lo_val);
                                let delta = histosys_code4p_delta_on_tape(tape, alpha, lo, nom, hi);
                                *delta_slot = tape.add(*delta_slot, delta);
                            }
                        }
                        ModelModifier::StatError { param_indices, .. } => {
                            for (bin_idx, &gamma_idx) in param_indices.iter().enumerate() {
                                if bin_idx < sample_factors.len() {
                                    let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                                        ns_core::Error::Validation(format!(
                                            "StatError gamma index out of range: idx={} len={}",
                                            gamma_idx,
                                            params.len()
                                        ))
                                    })?;
                                    sample_factors[bin_idx] =
                                        tape.mul(sample_factors[bin_idx], gamma);
                                }
                            }
                        }
                        ModelModifier::ShapeFactor { param_indices } => {
                            for (bin_idx, &gamma_idx) in param_indices.iter().enumerate() {
                                if bin_idx < sample_factors.len() {
                                    let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                                        ns_core::Error::Validation(format!(
                                            "ShapeFactor gamma index out of range: idx={} len={}",
                                            gamma_idx,
                                            params.len()
                                        ))
                                    })?;
                                    sample_factors[bin_idx] =
                                        tape.mul(sample_factors[bin_idx], gamma);
                                }
                            }
                        }
                        ModelModifier::Lumi { param_idx } => {
                            let lumi = params.get(*param_idx).copied().ok_or_else(|| {
                                ns_core::Error::Validation(format!(
                                    "Lumi param index out of range: idx={} len={}",
                                    param_idx,
                                    params.len()
                                ))
                            })?;
                            for fac in &mut sample_factors {
                                *fac = tape.mul(*fac, lumi);
                            }
                        }
                    }
                }

                for (bin_idx, ch_val) in channel_expected.iter_mut().enumerate() {
                    let nom = sample_nominal.get(bin_idx).copied().unwrap_or(tape.constant(0.0));
                    let delta = sample_deltas.get(bin_idx).copied().unwrap_or(tape.constant(0.0));
                    let fac = sample_factors.get(bin_idx).copied().unwrap_or(tape.constant(1.0));
                    let sum = tape.add(nom, delta);
                    let val = tape.mul(sum, fac);
                    *ch_val = tape.add(*ch_val, val);
                }
            }

            result.extend(channel_expected);
        }

        Ok(result)
    }
}

/// Pre-computed model data for fast f64 NLL evaluation.
///
/// Caches observed data, `lgamma(obs+1)`, observation masks, and Gaussian
/// constraint constants. Uses SIMD-accelerated Poisson NLL accumulation
/// via [`ns_compute::simd::poisson_nll_simd`], with a sparse scalar fast-path
/// that skips `ln(exp)` when `obs == 0`.
///
/// Created via [`HistFactoryModel::prepare`]. The generic AD path
/// ([`HistFactoryModel::nll_generic`]) is unchanged.
pub struct PreparedModel<'a> {
    model: &'a HistFactoryModel,
    /// Flat contiguous main-bin observed data (across all channels).
    observed_flat: Vec<f64>,
    /// `lgamma(obs + 1)` per main bin.
    ln_factorials: Vec<f64>,
    /// `1.0` if obs > 0, else `0.0` per main bin.
    obs_mask: Vec<f64>,
    /// Whether any main-bin observation is zero (enables sparse Poisson fast-path).
    has_zero_obs: bool,
    /// Number of main bins with `obs == 0`.
    n_zero_obs: usize,
    /// Sum of Gaussian constraint normalization constants:
    /// `Σ [ln(σ) + 0.5·ln(2π)]` over all constrained parameters.
    constraint_const: f64,
    /// Total number of main bins (sum across channels).
    n_main_bins: usize,
}

impl HistFactoryModel {
    /// Create a [`PreparedModel`] that caches per-bin constants for fast NLL.
    pub fn prepare(&self) -> PreparedModel<'_> {
        let mut observed_flat = Vec::new();
        let mut ln_factorials = Vec::new();
        let mut obs_mask = Vec::new();

        for channel in &self.channels {
            let n_bins = channel.samples.first().map(|s| s.nominal.len()).unwrap_or(0);
            for i in 0..n_bins {
                let obs = channel.observed.get(i).copied().unwrap_or(0.0);
                observed_flat.push(obs);
                ln_factorials.push(Self::ln_factorial(obs));
                obs_mask.push(if obs > 0.0 { 1.0 } else { 0.0 });
            }
        }

        let n_main_bins = observed_flat.len();
        let n_zero_obs = obs_mask.iter().filter(|&&m| m == 0.0).count();
        let has_zero_obs = n_zero_obs > 0;

        // Pre-compute Gaussian constraint normalization constant
        let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut constraint_const = 0.0;
        for param in &self.parameters {
            if !param.constrained {
                continue;
            }
            if let (Some(_center), Some(width)) = (param.constraint_center, param.constraint_width)
                && width > 0.0
            {
                constraint_const += width.ln() + half_ln_2pi;
            }
        }

        PreparedModel {
            model: self,
            observed_flat,
            ln_factorials,
            obs_mask,
            has_zero_obs,
            n_zero_obs,
            constraint_const,
            n_main_bins,
        }
    }
}

impl PreparedModel<'_> {
    /// Compute NLL using pre-cached data and SIMD-accelerated Poisson accumulation.
    ///
    /// Equivalent to [`HistFactoryModel::nll`] but faster for f64 evaluation.
    pub fn nll(&self, params: &[f64]) -> Result<f64> {
        use ns_compute::simd::{poisson_nll_scalar_sparse, poisson_nll_simd, poisson_nll_simd_sparse};

        // 1. Compute expected data (scalar — modifier application is branchy)
        let mut expected = self.model.expected_data(params)?;

        // 2. Clamp expected >= 1e-10
        for val in &mut expected {
            if *val < 1e-10 {
                *val = 1e-10;
            }
        }

        // expected_data returns exactly n_main_bins values
        debug_assert_eq!(expected.len(), self.n_main_bins);

        // 3. SIMD Poisson NLL for main bins
        let mut nll = if self.has_zero_obs {
            // Sparse observations: avoid computing ln(exp) where obs==0.
            //
            // Heuristic: for very sparse datasets, scalar can still win due to lower SIMD overhead.
            let zero_frac = (self.n_zero_obs as f64) / (self.n_main_bins as f64).max(1.0);
            if zero_frac >= 0.50 {
                poisson_nll_scalar_sparse(
                    &expected,
                    &self.observed_flat,
                    &self.ln_factorials,
                    &self.obs_mask,
                )
            } else {
                poisson_nll_simd_sparse(
                    &expected,
                    &self.observed_flat,
                    &self.ln_factorials,
                    &self.obs_mask,
                )
            }
        } else {
            poisson_nll_simd(&expected, &self.observed_flat, &self.ln_factorials, &self.obs_mask)
        };

        // 4. Barlow-Beeston auxiliary constraints (scalar — same as generic path)
        for channel in &self.model.channels {
            for constraint in &channel.auxiliary_data {
                if let Some(sample) = channel.samples.get(constraint.sample_idx)
                    && let Some(ModelModifier::ShapeSys { param_indices, .. }) =
                        sample.modifiers.get(constraint.modifier_idx)
                {
                    for ((&tau, &obs_aux), &gamma_idx) in constraint
                        .tau
                        .iter()
                        .zip(constraint.observed.iter())
                        .zip(param_indices.iter())
                    {
                        let gamma = params.get(gamma_idx).copied().ok_or_else(|| {
                            ns_core::Error::Validation(format!(
                                "ShapeSys gamma index out of range in aux constraint: idx={} len={}",
                                gamma_idx,
                                params.len()
                            ))
                        })?;
                        let exp_aux = (gamma * tau).max(1e-10);

                        if obs_aux > 0.0 {
                            let ln_factorial = Self::ln_factorial_static(obs_aux);
                            nll += exp_aux - obs_aux * exp_aux.ln() + ln_factorial;
                        } else {
                            nll += exp_aux;
                        }
                    }
                }
            }
        }

        // 5. Gaussian constraints: 0.5 * pull^2 only (constant part pre-computed)
        for (param_idx, param) in self.model.parameters.iter().enumerate() {
            if !param.constrained {
                continue;
            }
            if let (Some(center), Some(width)) = (param.constraint_center, param.constraint_width)
                && width > 0.0
            {
                let value = params.get(param_idx).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "Constrained parameter index out of range: idx={} len={}",
                        param_idx,
                        params.len()
                    ))
                })?;
                let pull = (value - center) / width;
                nll += 0.5 * pull * pull;
            }
        }

        // 6. Add pre-computed constraint normalization constant
        nll += self.constraint_const;

        Ok(nll)
    }

    /// Static version of ln_factorial for use in PreparedModel.
    #[inline]
    fn ln_factorial_static(n: f64) -> f64 {
        ln_gamma(n + 1.0)
    }
}

impl PreparedNll for PreparedModel<'_> {
    fn nll(&self, params: &[f64]) -> Result<f64> {
        PreparedModel::nll(self, params)
    }
}

impl LogDensityModel for HistFactoryModel {
    type Prepared<'a>
        = PreparedModel<'a>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.n_params()
    }

    fn parameter_names(&self) -> Vec<String> {
        self.parameters().iter().map(|p| p.name.clone()).collect()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.parameters().iter().map(|p| p.bounds).collect()
    }

    fn parameter_init(&self) -> Vec<f64> {
        self.parameters().iter().map(|p| p.init).collect()
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        HistFactoryModel::nll(self, params)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        HistFactoryModel::gradient_reverse(self, params)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        self.prepare()
    }
}

impl PoiModel for HistFactoryModel {
    fn poi_index(&self) -> Option<usize> {
        HistFactoryModel::poi_index(self)
    }
}

impl FixedParamModel for HistFactoryModel {
    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        HistFactoryModel::with_fixed_param(self, param_idx, value)
    }
}

fn normsys_code4_coeffs(hi: f64, lo: f64) -> [f64; 6] {
    // From pyhf.interpolators.code4 with alpha0=1.
    const A_INV: [[f64; 6]; 6] = [
        [15.0 / 16.0, -15.0 / 16.0, -7.0 / 16.0, -7.0 / 16.0, 1.0 / 16.0, -1.0 / 16.0],
        [3.0 / 2.0, 3.0 / 2.0, -9.0 / 16.0, 9.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0],
        [-5.0 / 8.0, 5.0 / 8.0, 5.0 / 8.0, 5.0 / 8.0, -1.0 / 8.0, 1.0 / 8.0],
        [-3.0 / 2.0, -3.0 / 2.0, 7.0 / 8.0, -7.0 / 8.0, -1.0 / 8.0, -1.0 / 8.0],
        [3.0 / 16.0, -3.0 / 16.0, -3.0 / 16.0, -3.0 / 16.0, 1.0 / 16.0, -1.0 / 16.0],
        [1.0 / 2.0, 1.0 / 2.0, -5.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0],
    ];

    let b0 = hi - 1.0;
    let b1 = lo - 1.0;
    let b2 = hi * hi.ln();
    let b3 = -lo * lo.ln();
    let b4 = hi * hi.ln().powi(2);
    let b5 = lo * lo.ln().powi(2);
    let b = [b0, b1, b2, b3, b4, b5];

    let mut a = [0.0; 6];
    for r in 0..6 {
        let mut s = 0.0;
        for c in 0..6 {
            s += A_INV[r][c] * b[c];
        }
        a[r] = s;
    }
    a
}

/// pyhf interpolators: normsys `code4` (alpha0=1), specialized for scalar factors.
fn normsys_code4<T: Scalar>(alpha: T, hi: f64, lo: f64) -> T {
    // Guard against invalid factors.
    if hi <= 0.0 || lo <= 0.0 {
        let alpha_val = alpha.value();
        return if alpha_val >= 0.0 {
            T::from_f64(1.0) + alpha * T::from_f64(hi - 1.0)
        } else {
            T::from_f64(1.0) - alpha * T::from_f64(1.0 - lo)
        };
    }

    let alpha_val = alpha.value();
    if alpha_val.abs() >= 1.0 {
        let base = if alpha_val >= 0.0 { hi } else { lo };
        // base^{|alpha|} = exp(|alpha| * ln(base))
        return (alpha.abs() * T::from_f64(base.ln())).exp();
    }

    let coeffs = normsys_code4_coeffs(hi, lo);
    let a1 = alpha;
    let a2 = alpha.powi(2);
    let a3 = alpha.powi(3);
    let a4 = alpha.powi(4);
    let a5 = alpha.powi(5);
    let a6 = alpha.powi(6);

    T::from_f64(1.0)
        + T::from_f64(coeffs[0]) * a1
        + T::from_f64(coeffs[1]) * a2
        + T::from_f64(coeffs[2]) * a3
        + T::from_f64(coeffs[3]) * a4
        + T::from_f64(coeffs[4]) * a5
        + T::from_f64(coeffs[5]) * a6
}

/// pyhf interpolators: histosys `code4p` delta term (added to nominal).
fn histosys_code4p_delta<T: Scalar>(alpha: T, down: T, nom: T, up: T) -> T {
    let alpha_val = alpha.value();
    let delta_up = up - nom;
    let delta_dn = nom - down;

    if alpha_val > 1.0 {
        return delta_up * alpha;
    }
    if alpha_val < -1.0 {
        return delta_dn * alpha;
    }

    let half = T::from_f64(0.5);
    let a_const = T::from_f64(0.0625);
    let s = half * (delta_up + delta_dn);
    let a = a_const * (delta_up - delta_dn);

    let asq = alpha * alpha;
    let tmp1 = asq * T::from_f64(3.0) - T::from_f64(10.0);
    let tmp2 = asq * tmp1 + T::from_f64(15.0);
    let tmp3 = asq * tmp2;

    alpha * s + tmp3 * a
}

fn normsys_code4_on_tape(
    tape: &mut ns_ad::tape::Tape,
    alpha: ns_ad::tape::Var,
    hi: f64,
    lo: f64,
) -> ns_ad::tape::Var {
    type Var = ns_ad::tape::Var;

    if hi <= 0.0 || lo <= 0.0 {
        let alpha_val = tape.val(alpha);
        return if alpha_val >= 0.0 {
            let delta = tape.constant(hi - 1.0);
            let ad = tape.mul(alpha, delta);
            tape.add_f64(ad, 1.0)
        } else {
            let delta = tape.constant(1.0 - lo);
            let ad = tape.mul(alpha, delta);
            tape.f64_sub(1.0, ad)
        };
    }

    let alpha_val = tape.val(alpha);
    if alpha_val.abs() >= 1.0 {
        let base = if alpha_val >= 0.0 { hi } else { lo };
        let ln_base = tape.constant(base.ln());
        let neg_alpha = tape.neg(alpha);
        let abs_alpha = tape.max(alpha, neg_alpha);
        let prod = tape.mul(abs_alpha, ln_base);
        return tape.exp(prod);
    }

    let coeffs = normsys_code4_coeffs(hi, lo);
    let mut out: Var = tape.constant(1.0);

    let a1 = alpha;
    let a2 = tape.powi(alpha, 2);
    let a3 = tape.powi(alpha, 3);
    let a4 = tape.powi(alpha, 4);
    let a5 = tape.powi(alpha, 5);
    let a6 = tape.powi(alpha, 6);

    let t1 = tape.mul_f64(a1, coeffs[0]);
    out = tape.add(out, t1);
    let t2 = tape.mul_f64(a2, coeffs[1]);
    out = tape.add(out, t2);
    let t3 = tape.mul_f64(a3, coeffs[2]);
    out = tape.add(out, t3);
    let t4 = tape.mul_f64(a4, coeffs[3]);
    out = tape.add(out, t4);
    let t5 = tape.mul_f64(a5, coeffs[4]);
    out = tape.add(out, t5);
    let t6 = tape.mul_f64(a6, coeffs[5]);
    out = tape.add(out, t6);

    out
}

fn histosys_code4p_delta_on_tape(
    tape: &mut ns_ad::tape::Tape,
    alpha: ns_ad::tape::Var,
    down: ns_ad::tape::Var,
    nom: ns_ad::tape::Var,
    up: ns_ad::tape::Var,
) -> ns_ad::tape::Var {
    type Var = ns_ad::tape::Var;

    let alpha_val = tape.val(alpha);
    let delta_up: Var = tape.sub(up, nom);
    let delta_dn: Var = tape.sub(nom, down);

    if alpha_val > 1.0 {
        return tape.mul(alpha, delta_up);
    }
    if alpha_val < -1.0 {
        return tape.mul(alpha, delta_dn);
    }

    // S = 0.5 * (delta_up + delta_dn)
    // A = 0.0625 * (delta_up - delta_dn)
    let sum = tape.add(delta_up, delta_dn);
    let s = tape.mul_f64(sum, 0.5);
    let diff = tape.sub(delta_up, delta_dn);
    let a = tape.mul_f64(diff, 0.0625);

    let asq = tape.mul(alpha, alpha);
    let asq3 = tape.mul_f64(asq, 3.0);
    let ten = tape.constant(10.0);
    let tmp1 = tape.sub(asq3, ten);
    let asq_tmp1 = tape.mul(asq, tmp1);
    let fifteen = tape.constant(15.0);
    let tmp2 = tape.add(asq_tmp1, fifteen);
    let tmp3 = tape.mul(asq, tmp2);

    // delta = alpha*S + tmp3*A
    let alpha_s = tape.mul(alpha, s);
    let tmp3_a = tape.mul(tmp3, a);
    tape.add(alpha_s, tmp3_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn test_model_from_simple_workspace() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();

        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // mu + 2 shapesys gamma parameters (2 bins)
        assert_eq!(model.n_params(), 3);
        assert_eq!(model.poi_index(), Some(0));
    }

    #[test]
    fn test_model_validates_parameter_length_and_does_not_panic() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let n = model.n_params();

        // Too short
        let short = vec![1.0; n.saturating_sub(1)];
        assert!(model.nll(&short).is_err());
        assert!(model.expected_data(&short).is_err());
        assert!(model.expected_data_pyhf_main(&short).is_err());
        assert!(model.expected_data_pyhf(&short).is_err());
        assert!(model.gradient(&short).is_err());
        assert!(model.gradient_ad(&short).is_err());
        assert!(model.gradient_reverse(&short).is_err());

        // Too long
        let long = vec![1.0; n + 1];
        assert!(model.nll(&long).is_err());
        assert!(model.expected_data(&long).is_err());
        assert!(model.expected_data_pyhf_main(&long).is_err());
        assert!(model.expected_data_pyhf(&long).is_err());
        assert!(model.gradient(&long).is_err());
        assert!(model.gradient_ad(&long).is_err());
        assert!(model.gradient_reverse(&long).is_err());
    }

    #[test]
    fn test_runtime_rejects_corrupted_internal_param_indices() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // Deliberately corrupt an internal modifier index to ensure runtime paths return
        // a validation error (never panic, never silently fall back).
        let mut corrupted = model.clone();
        let bad = corrupted.n_params() + 10;

        let mut changed = false;
        for channel in &mut corrupted.channels {
            for sample in &mut channel.samples {
                for m in &mut sample.modifiers {
                    if let ModelModifier::ShapeSys { param_indices, .. } = m {
                        if !param_indices.is_empty() {
                            param_indices[0] = bad;
                            changed = true;
                            break;
                        }
                    }
                }
                if changed {
                    break;
                }
            }
            if changed {
                break;
            }
        }
        assert!(changed, "expected to find a ShapeSys modifier to corrupt");

        let params = vec![1.0; model.n_params()];
        assert!(corrupted.expected_data(&params).is_err());
        assert!(corrupted.nll(&params).is_err());
        assert!(corrupted.expected_data_pyhf(&params).is_err());
        assert!(corrupted.prepared().nll(&params).is_err());
    }

    #[test]
    fn test_model_parameter_names() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let names: Vec<_> = model.parameters().iter().map(|p| &p.name).collect();
        assert!(names.contains(&&"mu".to_string()));
    }

    #[test]
    fn test_expected_data_nominal() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // At nominal values (mu=1, all gammas=1)
        let params = vec![1.0; model.n_params()];
        let expected = model.expected_data(&params).unwrap();

        // signal(mu=1) + background = 5+50, 10+60
        assert_eq!(expected.len(), 2);
        assert_relative_eq!(expected[0], 55.0, epsilon = 1e-6);
        assert_relative_eq!(expected[1], 70.0, epsilon = 1e-6);
    }

    #[test]
    fn test_expected_data_varied_poi() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // mu=2.0, gammas=1.0
        let mut params = vec![1.0; model.n_params()];
        params[0] = 2.0; // POI

        let expected = model.expected_data(&params).unwrap();

        // signal(mu=2) + background = 10+50, 20+60
        assert_relative_eq!(expected[0], 60.0, epsilon = 1e-6);
        assert_relative_eq!(expected[1], 80.0, epsilon = 1e-6);
    }

    #[test]
    fn test_complex_workspace_parsing() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // Should have: mu, lumi, ttbar_norm (normsys), bkg_shape (histosys),
        // shapefactor_CR (2 bins), staterror_SR (2 bins)
        assert_eq!(model.n_params(), 8);
        assert_eq!(model.poi_index(), Some(0));

        let names: Vec<_> = model.parameters().iter().map(|p| &p.name).collect();
        assert!(names.iter().any(|&n| n == "shapefactor_CR[0]"));
        assert!(names.iter().any(|&n| n == "shapefactor_CR[1]"));
        assert!(names.iter().any(|&n| n == "staterror_SR[0]"));
        assert!(names.iter().any(|&n| n == "staterror_SR[1]"));
    }

    #[test]
    fn test_complex_expected_data() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // At nominal init values
        let params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();
        let expected = model.expected_data(&params).unwrap();

        // Should have data for SR (2 bins) + CR (2 bins) = 4 bins
        assert_eq!(expected.len(), 4);

        // `HistFactoryModel` uses pyhf ordering for main bins: channels are sorted
        // lexicographically, so CR comes before SR for this fixture.
        assert_eq!(model.channels[0].name, "CR");
        assert_eq!(model.channels[1].name, "SR");

        // CR: background(500,510) at nominal with shapefactor
        assert!(expected[0] > 450.0 && expected[0] < 550.0, "CR bin 0: {}", expected[0]);
        assert!(expected[1] > 450.0 && expected[1] < 550.0, "CR bin 1: {}", expected[1]);

        // SR expected depends on modifiers applied; check reasonable values.
        assert!(expected[2] > 100.0 && expected[2] < 150.0, "SR bin 0: {}", expected[2]);
        assert!(expected[3] > 100.0 && expected[3] < 150.0, "SR bin 1: {}", expected[3]);
    }

    #[test]
    fn test_complex_nll_matches_pyhf_nominal() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();
        let nll = model.nll(&params).unwrap();

        // Reference from `python tests/validate_pyhf_nll.py` (pyhf 0.7.6)
        let pyhf_nll = 10.806852096216474;
        let diff = (nll - pyhf_nll).abs();
        assert!(
            diff < 1e-10,
            "Complex NLL mismatch: NextStat={:.15}, pyhf={:.15}, diff={:.3e}",
            nll,
            pyhf_nll,
            diff
        );
    }

    #[test]
    fn test_complex_nll_matches_pyhf_mu2() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let mut params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();
        params[0] = 2.0; // mu is the POI and stored first
        let nll = model.nll(&params).unwrap();

        // Reference from pyhf 0.7.6 with POI (mu) set to 2.0 (others at suggested_init).
        let pyhf_nll = 12.192112756616737;
        let diff = (nll - pyhf_nll).abs();
        assert!(
            diff < 1e-10,
            "Complex NLL mismatch (mu=2): NextStat={:.15}, pyhf={:.15}, diff={:.3e}",
            nll,
            pyhf_nll,
            diff
        );
    }

    #[test]
    fn test_nll_at_nominal() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // At nominal parameters (mu=1, gammas=1)
        let params = vec![1.0; model.n_params()];
        let nll = model.nll(&params).unwrap();

        // NLL should be finite and positive (with factorial term)
        assert!(nll.is_finite(), "NLL should be finite: {}", nll);
        assert!(nll > 0.0, "NLL should be positive: {}", nll);

        // Validate against pyhf reference with strict tolerance
        let pyhf_nll = 12.577579332147025;
        let diff = (nll - pyhf_nll).abs();
        let tolerance = 1e-10;

        assert!(
            diff < tolerance,
            "NLL mismatch: NextStat={:.10}, pyhf={:.10}, diff={:.10} (tolerance={})",
            nll,
            pyhf_nll,
            diff,
            tolerance
        );
    }

    #[test]
    fn test_nll_varied_poi() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // At mu=0 (no signal)
        let mut params = vec![1.0; model.n_params()];
        params[0] = 0.0;
        let nll_0 = model.nll(&params).unwrap();

        // At mu=1 (nominal signal)
        params[0] = 1.0;
        let nll_1 = model.nll(&params).unwrap();

        // At mu=2 (double signal)
        params[0] = 2.0;
        let nll_2 = model.nll(&params).unwrap();

        // NLL should change with POI
        assert_ne!(nll_0, nll_1);
        assert_ne!(nll_1, nll_2);

        // All should be finite
        assert!(nll_0.is_finite());
        assert!(nll_1.is_finite());
        assert!(nll_2.is_finite());
    }

    #[test]
    fn test_nll_with_constraints() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // At nominal (all params at constraint center/init)
        let params_nominal = model
            .parameters()
            .iter()
            .map(|p| if p.constrained { p.constraint_center.unwrap_or(p.init) } else { p.init })
            .collect::<Vec<_>>();
        let nll_nominal = model.nll(&params_nominal).unwrap();

        // Pull one constrained parameter away from center
        let mut params_pulled = params_nominal.clone();
        // Find first constrained parameter
        if let Some(idx) = model.parameters().iter().position(|p| p.constrained) {
            params_pulled[idx] += 1.0; // +1 sigma pull
            let nll_pulled = model.nll(&params_pulled).unwrap();

            // NLL should increase when pulling constrained parameter
            // (includes both Gaussian constraint and Poisson likelihood change)
            assert!(
                nll_pulled > nll_nominal,
                "NLL should increase with constraint violation: {} > {}",
                nll_pulled,
                nll_nominal
            );

            // Difference should be positive (can be large due to Poisson term)
            let diff = nll_pulled - nll_nominal;
            assert!(diff > 0.0, "NLL difference should be positive: {}", diff);
        }
    }

    #[test]
    fn test_nll_deterministic() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = vec![1.5, 1.0, 1.0];

        // Compute NLL multiple times
        let nll1 = model.nll(&params).unwrap();
        let nll2 = model.nll(&params).unwrap();
        let nll3 = model.nll(&params).unwrap();

        // Should be exactly the same (deterministic)
        assert_eq!(nll1, nll2);
        assert_eq!(nll2, nll3);
    }

    #[test]
    fn test_with_observed_main_updates_nll() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = vec![1.0; model.n_params()];
        let nll_nominal = model.nll(&params).unwrap();

        // Bump observations by +1 per bin.
        let obs =
            ws.observations.first().unwrap().data.iter().map(|&x| x + 1.0).collect::<Vec<_>>();

        // The simple workspace has only main bins in observations.
        let bumped = model.with_observed_main(&obs).unwrap();
        let nll_bumped = bumped.nll(&params).unwrap();

        assert_ne!(nll_nominal, nll_bumped);

        // Original model remains unchanged.
        let nll_nominal_again = model.nll(&params).unwrap();
        assert_eq!(nll_nominal, nll_nominal_again);

        // Sanity: method is deterministic.
        let nll_bumped_again = bumped.nll(&params).unwrap();
        assert_eq!(nll_bumped, nll_bumped_again);
    }

    #[test]
    fn test_with_observed_main_length_mismatch() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let err = model.with_observed_main(&[1.0]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("length mismatch"), "unexpected error: {}", msg);
    }

    #[test]
    fn test_gradient_computation() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = vec![1.0; model.n_params()];
        let grad = model.gradient(&params).unwrap();

        println!("Gradient at nominal: {:?}", grad);

        // Gradient should have correct length
        assert_eq!(grad.len(), model.n_params());

        // All gradient components should be finite
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "Gradient[{}] is not finite: {}", i, g);
        }

        // Test gradient changes with parameters
        let params2 = vec![1.5, 1.0, 1.0];
        let grad2 = model.gradient(&params2).unwrap();

        // Gradient at different point should differ
        assert_ne!(grad[0], grad2[0], "Gradient should change with parameters");
    }

    #[test]
    fn test_gradient_vs_finite_difference() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = vec![1.2, 0.9, 1.1];
        let grad = model.gradient(&params).unwrap();

        // Manual finite difference for first parameter
        let eps = 1e-8_f64.sqrt();
        let mut params_plus = params.clone();
        params_plus[0] += eps;
        let f_plus = model.nll(&params_plus).unwrap();

        let mut params_minus = params.clone();
        params_minus[0] -= eps;
        let f_minus = model.nll(&params_minus).unwrap();

        let grad_manual = (f_plus - f_minus) / (2.0 * eps);

        println!("Gradient[0] computed: {}", grad[0]);
        println!("Gradient[0] manual:   {}", grad_manual);

        // Should match within numerical tolerance
        assert_relative_eq!(grad[0], grad_manual, epsilon = 1e-5);
    }

    #[test]
    fn test_gradient_ad_matches_numerical() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = vec![1.2, 0.9, 1.1];

        let grad_num = model.gradient(&params).unwrap();
        let grad_ad = model.gradient_ad(&params).unwrap();

        println!("Numerical gradient: {:?}", grad_num);
        println!("AD gradient:        {:?}", grad_ad);

        assert_eq!(grad_ad.len(), grad_num.len());
        for (i, (&ad, &num)) in grad_ad.iter().zip(grad_num.iter()).enumerate() {
            let diff = (ad - num).abs();
            println!("  param[{}]: AD={:.10}, num={:.10}, diff={:.2e}", i, ad, num, diff);
            // AD should be more accurate than finite diff, so compare against it loosely
            assert_relative_eq!(ad, num, epsilon = 1e-5,);
        }
    }

    #[test]
    fn test_gradient_reverse_matches_forward() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params = vec![1.2, 0.9, 1.1];

        let grad_fwd = model.gradient_ad(&params).unwrap();
        let grad_rev = model.gradient_reverse(&params).unwrap();

        println!("Forward AD:  {:?}", grad_fwd);
        println!("Reverse AD:  {:?}", grad_rev);

        assert_eq!(grad_rev.len(), grad_fwd.len());
        for (i, (&rev, &fwd)) in grad_rev.iter().zip(grad_fwd.iter()).enumerate() {
            let diff = (rev - fwd).abs();
            println!("  param[{}]: rev={:.12}, fwd={:.12}, diff={:.2e}", i, rev, fwd, diff);
            assert_relative_eq!(rev, fwd, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gradient_reverse_complex_workspace() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        let params: Vec<f64> = model
            .parameters()
            .iter()
            .map(|p| {
                let base =
                    if p.constrained { p.constraint_center.unwrap_or(p.init) } else { p.init };
                base + 0.01
            })
            .collect();

        let grad_fwd = model.gradient_ad(&params).unwrap();
        let grad_rev = model.gradient_reverse(&params).unwrap();

        println!("Complex workspace - {} parameters", params.len());
        for (i, (&rev, &fwd)) in grad_rev.iter().zip(grad_fwd.iter()).enumerate() {
            let diff = (rev - fwd).abs();
            println!(
                "  {}: rev={:.10}, fwd={:.10}, diff={:.2e}",
                model.parameters()[i].name,
                rev,
                fwd,
                diff
            );
            assert_relative_eq!(rev, fwd, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gradient_ad_complex_workspace() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // Slightly perturb from init to avoid non-smooth switching points
        // (NormSys has branch at alpha=0 where finite-diff fails but AD is correct)
        let params: Vec<f64> = model
            .parameters()
            .iter()
            .map(|p| {
                let base =
                    if p.constrained { p.constraint_center.unwrap_or(p.init) } else { p.init };
                base + 0.01 // small offset away from branch points
            })
            .collect();

        let grad_num = model.gradient(&params).unwrap();
        let grad_ad = model.gradient_ad(&params).unwrap();

        println!("Complex workspace - {} parameters", params.len());
        for (i, (&ad, &num)) in grad_ad.iter().zip(grad_num.iter()).enumerate() {
            let diff = (ad - num).abs();
            println!(
                "  {}: AD={:.8}, num={:.8}, diff={:.2e}",
                model.parameters()[i].name,
                ad,
                num,
                diff
            );
            assert_relative_eq!(ad, num, epsilon = 1e-4);
        }
    }

    // ===== PreparedModel tests =====

    #[test]
    fn test_prepared_model_simple_matches_generic() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let prepared = model.prepare();

        let params = vec![1.0; model.n_params()];
        let nll_generic = model.nll(&params).unwrap();
        let nll_prepared = prepared.nll(&params).unwrap();

        let diff = (nll_generic - nll_prepared).abs();
        assert!(
            diff < 1e-12,
            "PreparedModel NLL mismatch: generic={:.15}, prepared={:.15}, diff={:.3e}",
            nll_generic,
            nll_prepared,
            diff
        );
    }

    #[test]
    fn test_prepared_model_complex_matches_generic() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let prepared = model.prepare();

        let params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();
        let nll_generic = model.nll(&params).unwrap();
        let nll_prepared = prepared.nll(&params).unwrap();

        let diff = (nll_generic - nll_prepared).abs();
        assert!(
            diff < 1e-12,
            "PreparedModel NLL mismatch (complex): generic={:.15}, prepared={:.15}, diff={:.3e}",
            nll_generic,
            nll_prepared,
            diff
        );
    }

    #[test]
    fn test_prepared_model_complex_mu2_matches_generic() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let prepared = model.prepare();

        let mut params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();
        params[0] = 2.0;
        let nll_generic = model.nll(&params).unwrap();
        let nll_prepared = prepared.nll(&params).unwrap();

        let diff = (nll_generic - nll_prepared).abs();
        assert!(
            diff < 1e-12,
            "PreparedModel NLL mismatch (mu=2): generic={:.15}, prepared={:.15}, diff={:.3e}",
            nll_generic,
            nll_prepared,
            diff
        );
    }

    #[test]
    fn test_prepared_model_varied_params_matches_generic() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let prepared = model.prepare();

        // Test at multiple parameter points
        let param_sets: Vec<Vec<f64>> = vec![
            model.parameters().iter().map(|p| p.init + 0.1).collect(),
            model.parameters().iter().map(|p| p.init - 0.05).collect(),
            model
                .parameters()
                .iter()
                .map(|p| {
                    let base =
                        if p.constrained { p.constraint_center.unwrap_or(p.init) } else { p.init };
                    base + 0.5
                })
                .collect(),
        ];

        for (idx, params) in param_sets.iter().enumerate() {
            let nll_generic = model.nll(params).unwrap();
            let nll_prepared = prepared.nll(params).unwrap();
            let diff = (nll_generic - nll_prepared).abs();
            assert!(
                diff < 1e-12,
                "PreparedModel NLL mismatch at param set {}: generic={:.15}, prepared={:.15}, diff={:.3e}",
                idx,
                nll_generic,
                nll_prepared,
                diff
            );
        }
    }

    #[test]
    fn test_prepared_model_preserves_pyhf_parity() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let prepared = model.prepare();

        let params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();
        let nll = prepared.nll(&params).unwrap();

        let pyhf_nll = 10.806852096216474;
        let diff = (nll - pyhf_nll).abs();
        assert!(
            diff < 1e-10,
            "PreparedModel pyhf parity: NextStat={:.15}, pyhf={:.15}, diff={:.3e}",
            nll,
            pyhf_nll,
            diff
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

        #[test]
        fn prop_gradient_reverse_matches_forward_simple(
            mu in 0.0f64..5.0,
            gamma0 in 0.1f64..2.0,
            gamma1 in 0.1f64..2.0,
        ) {
            let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
            let ws: Workspace = serde_json::from_str(json).unwrap();
            let model = HistFactoryModel::from_workspace(&ws).unwrap();

            // Simple fixture is: mu + 2 gamma params
            prop_assert_eq!(model.n_params(), 3);

            let params = vec![mu, gamma0, gamma1];
            let grad_fwd = model.gradient_ad(&params).unwrap();
            let grad_rev = model.gradient_reverse(&params).unwrap();

            prop_assert_eq!(grad_fwd.len(), 3);
            prop_assert_eq!(grad_rev.len(), 3);

            for (a, b) in grad_fwd.iter().zip(grad_rev.iter()) {
                prop_assert!((a - b).abs() < 1e-9);
                prop_assert!(a.is_finite());
                prop_assert!(b.is_finite());
            }
        }
    }
}
