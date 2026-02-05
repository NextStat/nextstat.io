# Фаза I: MVP-α Core Engine

> **Execution note (humans + AI agents):** Выполнять задачи последовательно. Каноничные определения/допуски/детерминизм: `docs/plans/standards.md`.

**Goal:** Создать минимальное ядро, способное читать pyhf JSON и выдавать MLE результат, идентичный pyhf.

**Duration:** Месяцы 2-4 (Недели 5-16)

**Architecture:** Rust core с trait Model, pyhf JSON parser, MLE optimizer (Phase 1: numerical gradients; Phase 2B: AD gradients).

**Tech Stack:** Rust, serde_json, ndarray, argmin (опционально).

---

## Содержание

- [Sprint 1.1: HistFactory Parser](#sprint-11-histfactory-parser-недели-5-6)
- [Sprint 1.2: Likelihood Computation](#sprint-12-likelihood-computation-недели-7-8)
- [Sprint 1.3: MLE Optimizer](#sprint-13-mle-optimizer-недели-9-10)
- [Sprint 1.4: CLI и Python API](#sprint-14-cli-и-python-api-недели-11-12)
- [Sprint 1.5: Validation и Documentation](#sprint-15-validation-и-documentation-недели-13-16)
- [Критерии завершения фазы](#критерии-завершения-фазы)

---

## Acceptance Criteria (Фаза)

| Метрика | Target | Measurement |
|---------|--------|-------------|
| MLE μ̂ | ±1e-6 vs pyhf | `np.allclose(ns.bestfit, pyhf.bestfit, rtol=1e-6)` |
| σ(μ̂) | ±1e-5 vs pyhf | `np.allclose(ns.uncertainties, pyhf.uncertainties, rtol=1e-5)` |
| `twice_nll` (deterministic CPU) | rtol=1e-6, atol=1e-8 vs pyhf | `np.allclose(ns.twice_nll, pyhf.twice_nll, rtol=1e-6, atol=1e-8)` |
| Fit time (10 NP) | <100ms | `timeit` |
| Test coverage | ≥80% | `cargo llvm-cov` |

---

## Sprint 1.1: HistFactory Parser (Недели 5-6)

### Epic 1.1.1: pyhf JSON Schema Types

**Цель:** Полностью парсить pyhf JSON workspace в Rust структуры.

---

#### Task 1.1.1.1: Core schema types

**Priority:** P0
**Effort:** 4 часа
**Dependencies:** Phase 0 complete

**Files:**
- Create: `crates/ns-translate/src/lib.rs`
- Create: `crates/ns-translate/src/pyhf/mod.rs`
- Create: `crates/ns-translate/src/pyhf/schema.rs`
- Test: `crates/ns-translate/src/pyhf/tests.rs`

**Acceptance Criteria:**
- [ ] Parse simple_workspace.json без ошибок
- [ ] Parse complex_workspace.json без ошибок
- [ ] Все modifier types поддержаны
- [ ] Serde round-trip работает

**Step 1: Write failing test**

```rust
// crates/ns-translate/src/pyhf/tests.rs
#[cfg(test)]
mod tests {
    use super::schema::*;

    #[test]
    fn test_parse_simple_workspace() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json)
            .expect("Failed to parse simple_workspace.json");

        assert_eq!(ws.channels.len(), 1);
        assert_eq!(ws.channels[0].name, "singlechannel");
        assert_eq!(ws.channels[0].samples.len(), 2);
        assert_eq!(ws.observations.len(), 1);
        assert_eq!(ws.measurements.len(), 1);
        assert_eq!(ws.measurements[0].config.poi, "mu");
    }

    #[test]
    fn test_parse_complex_workspace() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json)
            .expect("Failed to parse complex_workspace.json");

        assert_eq!(ws.channels.len(), 2);
        assert!(ws.channels.iter().any(|c| c.name == "SR"));
        assert!(ws.channels.iter().any(|c| c.name == "CR"));
    }

    #[test]
    fn test_parse_all_modifier_types() {
        let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();

        let all_modifiers: Vec<_> = ws.channels.iter()
            .flat_map(|c| c.samples.iter())
            .flat_map(|s| s.modifiers.iter())
            .collect();

        // Check we have various modifier types
        assert!(all_modifiers.iter().any(|m| matches!(m, Modifier::NormFactor { .. })));
        assert!(all_modifiers.iter().any(|m| matches!(m, Modifier::NormSys { .. })));
        assert!(all_modifiers.iter().any(|m| matches!(m, Modifier::HistoSys { .. })));
        assert!(all_modifiers.iter().any(|m| matches!(m, Modifier::StatError { .. })));
        assert!(all_modifiers.iter().any(|m| matches!(m, Modifier::Lumi { .. })));
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();

        let serialized = serde_json::to_string(&ws).unwrap();
        let ws2: Workspace = serde_json::from_str(&serialized).unwrap();

        assert_eq!(ws.channels.len(), ws2.channels.len());
        assert_eq!(ws.measurements[0].config.poi, ws2.measurements[0].config.poi);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-translate test_parse_simple_workspace
# Expected: FAIL - module not found
```

**Step 3: Implement schema types**

```rust
// crates/ns-translate/src/pyhf/schema.rs
//! pyhf JSON workspace schema types
//!
//! Based on pyhf specification:
//! https://scikit-hep.org/pyhf/likelihood.html

use serde::{Deserialize, Serialize};

/// Root workspace object
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Workspace {
    /// List of channels (regions)
    pub channels: Vec<Channel>,
    /// Observed data per channel
    pub observations: Vec<Observation>,
    /// Measurement configurations
    pub measurements: Vec<Measurement>,
    /// Schema version
    pub version: String,
}

/// A channel (region) in the workspace
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Channel {
    /// Channel name (e.g., "SR", "CR")
    pub name: String,
    /// Samples in this channel
    pub samples: Vec<Sample>,
}

/// A sample (process) in a channel
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    /// Sample name (e.g., "signal", "background")
    pub name: String,
    /// Nominal histogram (bin contents)
    pub data: Vec<f64>,
    /// Modifiers affecting this sample
    pub modifiers: Vec<Modifier>,
}

/// Observed data for a channel
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    /// Channel name (must match a channel)
    pub name: String,
    /// Observed counts per bin
    pub data: Vec<f64>,
}

/// Measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Measurement {
    /// Measurement name
    pub name: String,
    /// Configuration
    pub config: MeasurementConfig,
}

/// Measurement configuration details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeasurementConfig {
    /// Parameter of interest name
    pub poi: String,
    /// Parameter settings
    #[serde(default)]
    pub parameters: Vec<ParameterConfig>,
}

/// Per-parameter configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterConfig {
    /// Parameter name
    pub name: String,
    /// Fixed value (if fixed)
    #[serde(default)]
    pub fixed: bool,
    /// Initial values
    #[serde(default)]
    pub inits: Option<Vec<f64>>,
    /// Bounds [[lo, hi], ...]
    #[serde(default)]
    pub bounds: Option<Vec<[f64; 2]>>,
    /// Auxiliary data (for constrained NP)
    #[serde(default)]
    pub auxdata: Option<Vec<f64>>,
    /// Sigmas for Gaussian constraint
    #[serde(default)]
    pub sigmas: Option<Vec<f64>>,
}

/// Modifier types in HistFactory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Modifier {
    /// Normalization factor (free parameter, often POI)
    #[serde(rename = "normfactor")]
    NormFactor {
        name: String,
        #[serde(default)]
        data: Option<serde_json::Value>,
    },

    /// Statistical uncertainty (Barlow-Beeston lite)
    #[serde(rename = "staterror")]
    StatError {
        name: String,
        /// Absolute uncertainties per bin
        data: Vec<f64>,
    },

    /// Correlated shape systematic (interpolated histograms)
    #[serde(rename = "histosys")]
    HistoSys {
        name: String,
        data: HistoSysData,
    },

    /// Normalization systematic (up/down factors)
    #[serde(rename = "normsys")]
    NormSys {
        name: String,
        data: NormSysData,
    },

    /// Shape systematic (uncorrelated per-bin)
    #[serde(rename = "shapesys")]
    ShapeSys {
        name: String,
        data: Vec<f64>,
    },

    /// Shape factor (free per-bin multipliers)
    #[serde(rename = "shapefactor")]
    ShapeFactor {
        name: String,
        #[serde(default)]
        data: Option<serde_json::Value>,
    },

    /// Luminosity systematic
    #[serde(rename = "lumi")]
    Lumi {
        name: String,
        #[serde(default)]
        data: Option<serde_json::Value>,
    },
}

/// Data for histosys modifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoSysData {
    /// +1σ variation histogram
    pub hi_data: Vec<f64>,
    /// -1σ variation histogram
    pub lo_data: Vec<f64>,
}

/// Data for normsys modifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormSysData {
    /// +1σ normalization factor
    pub hi: f64,
    /// -1σ normalization factor
    pub lo: f64,
}

impl Workspace {
    /// Get channel by name
    pub fn channel(&self, name: &str) -> Option<&Channel> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// Get observation by channel name
    pub fn observation(&self, channel_name: &str) -> Option<&Observation> {
        self.observations.iter().find(|o| o.name == channel_name)
    }

    /// Get default measurement
    pub fn default_measurement(&self) -> Option<&Measurement> {
        self.measurements.first()
    }

    /// Total number of bins across all channels
    pub fn n_bins(&self) -> usize {
        self.channels.iter()
            .map(|c| c.samples.first().map(|s| s.data.len()).unwrap_or(0))
            .sum()
    }
}

impl Channel {
    /// Number of bins in this channel
    pub fn n_bins(&self) -> usize {
        self.samples.first().map(|s| s.data.len()).unwrap_or(0)
    }
}
```

**Step 4: Create module structure**

```rust
// crates/ns-translate/src/pyhf/mod.rs
//! pyhf JSON workspace support

pub mod schema;

#[cfg(test)]
mod tests;

pub use schema::*;
```

```rust
// crates/ns-translate/src/lib.rs
//! NextStat Translate - format converters
//!
//! This crate handles conversion between various statistical model formats:
//! - pyhf JSON workspace (HistFactory)
//! - HistFactory XML (planned)
//! - ROOT histograms via uproot (planned)

#![warn(missing_docs)]

pub mod pyhf;

pub use pyhf::Workspace;
```

**Step 5: Update Cargo.toml**

```toml
# crates/ns-translate/Cargo.toml
[package]
name = "ns-translate"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Format converters for NextStat"

[dependencies]
ns-core = { path = "../ns-core" }
serde.workspace = true
serde_json.workspace = true

[dev-dependencies]
approx.workspace = true
```

**Step 6: Run tests**

```bash
cargo test -p ns-translate
# Expected: All PASS
```

**Step 7: Commit**

```bash
git add crates/ns-translate/
git commit -m "feat(translate): implement pyhf JSON schema parser"
```

---

#### Task 1.1.1.2: Model conversion

**Priority:** P0
**Effort:** 6 часов
**Dependencies:** Task 1.1.1.1

**Files:**
- Create: `crates/ns-translate/src/pyhf/model.rs`
- Modify: `crates/ns-translate/src/pyhf/mod.rs`
- Test: inline in model.rs

**Acceptance Criteria:**
- [ ] Convert Workspace to internal Model representation
- [ ] Track parameter indices correctly
- [ ] Support all modifier types
- [ ] expected_data() matches pyhf

**Step 1: Write failing test**

```rust
// crates/ns-translate/src/pyhf/model.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyhf::schema::Workspace;
    use approx::assert_relative_eq;

    #[test]
    fn test_model_from_simple_workspace() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();

        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // mu + 2 gamma parameters
        assert_eq!(model.n_params(), 3);
        assert_eq!(model.poi_index(), Some(0));
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
        let init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let expected = model.expected_data(&init);

        // signal=[5,10] + background=[50,60] = [55, 70]
        assert_eq!(expected.len(), 2);
        assert_relative_eq!(expected[0], 55.0, epsilon = 1e-10);
        assert_relative_eq!(expected[1], 70.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expected_data_scaled() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // mu=2, gammas=1
        let params = vec![2.0, 1.0, 1.0];
        let expected = model.expected_data(&params);

        // signal*2=[10,20] + background=[50,60] = [60, 80]
        assert_relative_eq!(expected[0], 60.0, epsilon = 1e-10);
        assert_relative_eq!(expected[1], 80.0, epsilon = 1e-10);
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo test -p ns-translate test_model_from_simple_workspace
# Expected: FAIL - HistFactoryModel not defined
```

**Step 3: Implement HistFactoryModel**

```rust
// crates/ns-translate/src/pyhf/model.rs
//! HistFactory model representation

use std::collections::HashMap;
use ns_core::{Parameter, Result, Error, types::Float};
use super::schema::*;

/// HistFactory model converted from pyhf workspace
#[derive(Debug, Clone)]
pub struct HistFactoryModel {
    /// Model parameters (ordered)
    parameters: Vec<Parameter>,
    /// Index of POI in parameters
    poi_idx: Option<usize>,
    /// Channel models
    channels: Vec<ChannelModel>,
    /// Concatenated observed data
    observed: Vec<Float>,
    /// Parameter name to index mapping
    param_map: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct ChannelModel {
    name: String,
    n_bins: usize,
    samples: Vec<SampleModel>,
}

#[derive(Debug, Clone)]
struct SampleModel {
    name: String,
    nominal: Vec<Float>,
    modifiers: Vec<ModifierModel>,
}

#[derive(Debug, Clone)]
enum ModifierModel {
    /// Multiplicative factor applied to whole sample
    NormFactor { param_idx: usize },

    /// Per-bin statistical uncertainty (Barlow-Beeston)
    /// Each bin gets its own gamma parameter
    StatError {
        /// Parameter indices for each bin
        param_indices: Vec<usize>,
        /// Relative uncertainties (sigma/nominal)
        rel_uncerts: Vec<Float>,
    },

    /// Correlated normalization systematic
    NormSys {
        param_idx: usize,
        hi_factor: Float,
        lo_factor: Float,
    },

    /// Correlated shape systematic (histogram interpolation)
    HistoSys {
        param_idx: usize,
        hi_data: Vec<Float>,
        lo_data: Vec<Float>,
    },

    /// Luminosity uncertainty
    Lumi {
        param_idx: usize,
        uncertainty: Float,
    },
}

impl HistFactoryModel {
    /// Convert pyhf Workspace to internal model
    pub fn from_workspace(ws: &Workspace) -> Result<Self> {
        let measurement = ws.default_measurement()
            .ok_or_else(|| Error::ModelSpec("No measurement defined".into()))?;
        let poi_name = &measurement.config.poi;

        let mut parameters = Vec::new();
        let mut param_map = HashMap::new();
        let mut poi_idx = None;

        // First pass: collect all parameters
        // Order: POI first, then other normfactors, then systematics, then gammas

        // 1. Add POI
        param_map.insert(poi_name.clone(), 0);
        poi_idx = Some(0);
        parameters.push(Parameter::new(poi_name, 1.0).bounded(0.0, 10.0));

        // 2. Scan all modifiers to find parameters
        let mut sys_params: Vec<String> = Vec::new();
        let mut gamma_params: Vec<(String, usize, Float)> = Vec::new(); // (name, bin, rel_uncert)

        for channel in &ws.channels {
            for sample in &channel.samples {
                for modifier in &sample.modifiers {
                    match modifier {
                        Modifier::NormFactor { name, .. } => {
                            if name != poi_name && !param_map.contains_key(name) {
                                let idx = parameters.len();
                                param_map.insert(name.clone(), idx);
                                parameters.push(Parameter::new(name, 1.0).bounded(0.0, 10.0));
                            }
                        }
                        Modifier::NormSys { name, .. } |
                        Modifier::HistoSys { name, .. } => {
                            if !sys_params.contains(name) {
                                sys_params.push(name.clone());
                            }
                        }
                        Modifier::StatError { name, data } => {
                            for (i, &sigma) in data.iter().enumerate() {
                                let gamma_name = format!("staterror_{}[{}]", name, i);
                                let nominal = sample.data.get(i).copied().unwrap_or(1.0);
                                let rel_uncert = if nominal > 0.0 { sigma / nominal } else { 0.0 };
                                gamma_params.push((gamma_name, i, rel_uncert));
                            }
                        }
                        Modifier::Lumi { name, .. } => {
                            if !sys_params.contains(name) {
                                sys_params.push(name.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // 3. Add systematic parameters (Gaussian constrained, init=0, bounds=[-5,5])
        for name in sys_params {
            if !param_map.contains_key(&name) {
                let idx = parameters.len();
                param_map.insert(name.clone(), idx);
                parameters.push(Parameter::new(&name, 0.0).bounded(-5.0, 5.0));
            }
        }

        // 4. Add gamma parameters (Gaussian constrained, init=1)
        for (name, _, _) in &gamma_params {
            if !param_map.contains_key(name) {
                let idx = parameters.len();
                param_map.insert(name.clone(), idx);
                parameters.push(Parameter::new(name, 1.0).bounded(0.0, 10.0));
            }
        }

        // Second pass: build channel models
        let mut channels = Vec::new();
        let mut observed = Vec::new();

        for channel in &ws.channels {
            let n_bins = channel.n_bins();
            let mut samples = Vec::new();

            for sample in &channel.samples {
                let mut modifiers = Vec::new();

                for modifier in &sample.modifiers {
                    match modifier {
                        Modifier::NormFactor { name, .. } => {
                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::NormFactor { param_idx: idx });
                        }
                        Modifier::StatError { name, data } => {
                            let indices: Vec<usize> = (0..data.len())
                                .map(|i| {
                                    let gamma_name = format!("staterror_{}[{}]", name, i);
                                    *param_map.get(&gamma_name).unwrap()
                                })
                                .collect();

                            let rel_uncerts: Vec<Float> = data.iter()
                                .zip(sample.data.iter())
                                .map(|(&sigma, &nom)| if nom > 0.0 { sigma / nom } else { 0.0 })
                                .collect();

                            modifiers.push(ModifierModel::StatError {
                                param_indices: indices,
                                rel_uncerts,
                            });
                        }
                        Modifier::NormSys { name, data } => {
                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::NormSys {
                                param_idx: idx,
                                hi_factor: data.hi,
                                lo_factor: data.lo,
                            });
                        }
                        Modifier::HistoSys { name, data } => {
                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::HistoSys {
                                param_idx: idx,
                                hi_data: data.hi_data.clone(),
                                lo_data: data.lo_data.clone(),
                            });
                        }
                        Modifier::Lumi { name, .. } => {
                            // Get lumi uncertainty from measurement config
                            let uncert = measurement.config.parameters.iter()
                                .find(|p| &p.name == name)
                                .and_then(|p| p.sigmas.as_ref())
                                .and_then(|s| s.first().copied())
                                .unwrap_or(0.02); // default 2% lumi

                            let idx = *param_map.get(name).unwrap();
                            modifiers.push(ModifierModel::Lumi {
                                param_idx: idx,
                                uncertainty: uncert,
                            });
                        }
                        _ => {}
                    }
                }

                samples.push(SampleModel {
                    name: sample.name.clone(),
                    nominal: sample.data.clone(),
                    modifiers,
                });
            }

            channels.push(ChannelModel {
                name: channel.name.clone(),
                n_bins,
                samples,
            });

            // Get observed data
            if let Some(obs) = ws.observation(&channel.name) {
                observed.extend(obs.data.iter().copied());
            }
        }

        Ok(Self {
            parameters,
            poi_idx,
            channels,
            observed,
            param_map,
        })
    }

    /// Number of parameters
    pub fn n_params(&self) -> usize {
        self.parameters.len()
    }

    /// Index of POI
    pub fn poi_index(&self) -> Option<usize> {
        self.poi_idx
    }

    /// Get parameters
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    /// Get observed data
    pub fn observed_data(&self) -> &[Float] {
        &self.observed
    }

    /// Compute expected data at given parameter values
    pub fn expected_data(&self, params: &[Float]) -> Vec<Float> {
        let mut result = Vec::new();

        for channel in &self.channels {
            let mut channel_expected = vec![0.0; channel.n_bins];

            for sample in &channel.samples {
                let mut sample_expected = sample.nominal.clone();

                for modifier in &sample.modifiers {
                    Self::apply_modifier(modifier, params, &mut sample_expected);
                }

                for (i, val) in sample_expected.iter().enumerate() {
                    channel_expected[i] += val;
                }
            }

            result.extend(channel_expected);
        }

        result
    }

    fn apply_modifier(modifier: &ModifierModel, params: &[Float], data: &mut [Float]) {
        match modifier {
            ModifierModel::NormFactor { param_idx } => {
                let factor = params[*param_idx];
                for v in data.iter_mut() {
                    *v *= factor;
                }
            }
            ModifierModel::StatError { param_indices, .. } => {
                for (i, &idx) in param_indices.iter().enumerate() {
                    if i < data.len() {
                        data[i] *= params[idx];
                    }
                }
            }
            ModifierModel::NormSys { param_idx, hi_factor, lo_factor } => {
                let alpha = params[*param_idx];
                let factor = interpolate_normsys(alpha, *lo_factor, *hi_factor);
                for v in data.iter_mut() {
                    *v *= factor;
                }
            }
            ModifierModel::HistoSys { param_idx, hi_data, lo_data } => {
                let alpha = params[*param_idx];
                for (i, v) in data.iter_mut().enumerate() {
                    let nominal = *v;
                    let hi = hi_data.get(i).copied().unwrap_or(nominal);
                    let lo = lo_data.get(i).copied().unwrap_or(nominal);
                    *v = interpolate_histosys(alpha, nominal, lo, hi);
                }
            }
            ModifierModel::Lumi { param_idx, .. } => {
                // Lumi is typically alpha * sigma around 1
                // For simplicity, treat as multiplicative
                let alpha = params[*param_idx];
                // alpha is the nuisance parameter, nominal is 1
                // effect is (1 + alpha * sigma), but we use alpha directly
                // since alpha is already the fractional variation
                let factor = 1.0 + alpha;
                for v in data.iter_mut() {
                    *v *= factor.max(0.0);
                }
            }
        }
    }
}

/// Interpolate normsys factor (exponential interpolation)
fn interpolate_normsys(alpha: Float, lo: Float, hi: Float) -> Float {
    // Exponential interpolation: factor = (hi/1)^alpha if alpha > 0
    //                                   = (1/lo)^(-alpha) if alpha < 0
    if alpha >= 0.0 {
        hi.powf(alpha)
    } else {
        lo.powf(-alpha)
    }
}

/// Interpolate histosys (piecewise linear)
fn interpolate_histosys(alpha: Float, nominal: Float, lo: Float, hi: Float) -> Float {
    if alpha >= 0.0 {
        nominal + alpha * (hi - nominal)
    } else {
        nominal + (-alpha) * (lo - nominal)
    }
}
```

**Step 4: Update mod.rs**

```rust
// crates/ns-translate/src/pyhf/mod.rs
pub mod schema;
pub mod model;

#[cfg(test)]
mod tests;

pub use schema::*;
pub use model::HistFactoryModel;
```

**Step 5: Run tests**

```bash
cargo test -p ns-translate
# Expected: All PASS
```

**Step 6: Commit**

```bash
git add crates/ns-translate/
git commit -m "feat(translate): convert pyhf workspace to HistFactoryModel"
```

---

## Sprint 1.2: Likelihood Computation (Недели 7-8)

### Epic 1.2.1: NLL Implementation

---

#### Task 1.2.1.1: Poisson NLL

**Priority:** P0
**Effort:** 2 часа
**Dependencies:** Sprint 1.1

**Files:**
- Create: `crates/ns-compute/src/lib.rs`
- Create: `crates/ns-compute/src/nll.rs`
- Test: inline

**Acceptance Criteria:**
- [ ] `poisson_nll(n, λ) = λ - n*ln(λ) + ln Γ(n+1)` (matches `-poisson.log_prob` in pyhf)
- [ ] Vectorized version
- [ ] Property tests pass

**Step 1: Write failing test**

```rust
// crates/ns-compute/src/nll.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poisson_nll() {
        // λ - n*ln(λ) + ln Γ(n+1)
        // n=10, λ=10: 10 - 10*ln(10) + ln(10!) ≈ 2.07856
        let nll = poisson_nll(10.0, 10.0);
        assert_relative_eq!(nll, 2.0785616431, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_nll_vec() {
        let obs = vec![10.0, 20.0];
        let exp = vec![10.0, 20.0];
        let nll = poisson_nll_vec(&obs, &exp);

        let expected = poisson_nll(10.0, 10.0) + poisson_nll(20.0, 20.0);
        assert_relative_eq!(nll, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_nll_zero_expected() {
        // Should return infinity for λ=0
        let nll = poisson_nll(1.0, 0.0);
        assert!(nll.is_infinite());
    }

    // Property test: NLL is minimized when λ = n
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_minimum_at_mle(n in 1.0f64..1000.0) {
            let nll_at_mle = poisson_nll(n, n);
            let nll_above = poisson_nll(n, n * 1.1);
            let nll_below = poisson_nll(n, n * 0.9);

            prop_assert!(nll_at_mle <= nll_above);
            prop_assert!(nll_at_mle <= nll_below);
        }
    }
}
```

**Step 2: Implement NLL functions**

```rust
// crates/ns-compute/src/nll.rs
//! Negative log-likelihood computations
//!
//! Core NLL functions for statistical fitting.

use ns_core::types::Float;

/// Poisson negative log-likelihood (single observation)
///
/// Computes: λ - n * ln(λ) + ln Γ(n+1)
///
/// # Arguments
/// * `observed` - Observed count (n)
/// * `expected` - Expected count (λ)
///
/// # Returns
/// NLL value (INFINITY if expected ≤ 0)
#[inline]
pub fn poisson_nll(observed: Float, expected: Float) -> Float {
    if expected <= 0.0 {
        return Float::INFINITY;
    }
    // Match pyhf's Poisson log_prob (includes lgamma(n+1))
    expected - observed * expected.ln() + statrs::function::gamma::ln_gamma(observed + 1.0)
}

/// Poisson NLL for multiple bins
///
/// # Arguments
/// * `observed` - Observed counts per bin
/// * `expected` - Expected counts per bin
///
/// # Panics
/// If arrays have different lengths
pub fn poisson_nll_vec(observed: &[Float], expected: &[Float]) -> Float {
    assert_eq!(observed.len(), expected.len(), "Array length mismatch");

    observed.iter()
        .zip(expected.iter())
        .map(|(&n, &lam)| poisson_nll(n, lam))
        .sum()
}

/// Gaussian constraint NLL
///
/// Computes: -log N(x | μ, σ)
///
/// Used for nuisance parameter constraints.
#[inline]
pub fn gaussian_constraint_nll(value: Float, mean: Float, sigma: Float) -> Float {
    if sigma <= 0.0 {
        return Float::INFINITY;
    }
    let z = (value - mean) / sigma;
    // 0.5*z^2 + ln(sigma) + 0.5*ln(2π)
    0.5 * z * z + sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln()
}
```

**Step 3: Create lib.rs**

```rust
// crates/ns-compute/src/lib.rs
//! NextStat Compute - numerical computations
//!
//! This crate provides core numerical functions for likelihood computation.

#![warn(missing_docs)]

pub mod nll;

pub use nll::{poisson_nll, poisson_nll_vec, gaussian_constraint_nll};
```

**Step 4: Update Cargo.toml**

```toml
# crates/ns-compute/Cargo.toml
[package]
name = "ns-compute"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Numerical computations for NextStat"

[dependencies]
ns-core = { path = "../ns-core" }
statrs = "0.18"

[dev-dependencies]
approx.workspace = true
proptest.workspace = true
```

**Step 5: Run tests**

```bash
cargo test -p ns-compute
# Expected: All PASS
```

**Step 6: Commit**

```bash
git add crates/ns-compute/
git commit -m "feat(compute): implement Poisson NLL"
```

---

#### Task 1.2.1.2: HistFactoryModel NLL parity vs pyhf

**Priority:** P0  
**Effort:** 4-6 часов  
**Dependencies:** Task 1.2.1.1

**Files:**
- Modify: `crates/ns-translate/Cargo.toml`
- Modify: `crates/ns-translate/src/pyhf/model.rs`
- Test: `crates/ns-translate/src/pyhf/tests.rs`

**Acceptance Criteria:**
- [ ] `HistFactoryModel::nll()` использует каноничные определения из `docs/plans/standards.md`
- [ ] `nll(init_params)` совпадает с pyhf (simple fixture, deterministic CPU)
- [ ] Constraint terms добавлены (staterror + normsys/histosys/lumi как N(0,1))

**Step 1: Write failing test**

```rust
// crates/ns-translate/src/pyhf/tests.rs
#[test]
fn test_histfactory_nll_matches_pyhf_simple() {
    use approx::assert_relative_eq;
    use ns_core::{Model, types::Float};

    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();
    let model = HistFactoryModel::from_workspace(&ws).unwrap();

    let init: Vec<Float> = model.parameters().iter().map(|p| p.init).collect();
    let nll = model.nll(&init).unwrap();

    // Fill from `pyhf` reference (see step 2)
    let pyhf_ref_nll: Float = 0.0;

    assert_relative_eq!(nll, pyhf_ref_nll, epsilon = 1e-6);
}
```

**Step 2: Get reference from pyhf**

```bash
python3 -c "
import json, pyhf
ws = pyhf.Workspace(json.load(open('tests/fixtures/simple_workspace.json')))
model = ws.model()
data = ws.data(model)
pars = [1.0] + [1.0] * (model.config.npars - 1)
print(float(-model.logpdf(pars, data)[0]))
"
# Paste value into pyhf_ref_nll above
```

**Step 2.1: Add `ns-compute` dependency**

```toml
# crates/ns-translate/Cargo.toml
[dependencies]
ns-core = { path = "../ns-core" }
ns-compute = { path = "../ns-compute" }
serde.workspace = true
serde_json.workspace = true
```

**Step 3: Implement `ns_core::Model` for `HistFactoryModel`**

```rust
// crates/ns-translate/src/pyhf/model.rs
use ns_core::{Model, types::Float};

impl Model for HistFactoryModel {
    fn nll(&self, params: &[Float]) -> ns_core::Result<Float> {
        let expected = self.expected_data(params);
        let observed = self.observed_data();

        // Poisson term (matches pyhf logpdf via standards)
        let mut nll = ns_compute::poisson_nll_vec(observed, &expected);

        // Constraints:
        // - StatError gammas: gamma ~ Normal(1, rel_uncert)
        // - Systematics (normsys/histosys/lumi): alpha ~ Normal(0, 1)
        // IMPORTANT: add each constraint once per parameter (avoid double counting).
        let mut constrained = vec![false; self.parameters.len()];
        for channel in &self.channels {
            for sample in &channel.samples {
                for modifier in &sample.modifiers {
                    match modifier {
                        ModifierModel::StatError { param_indices, rel_uncerts } => {
                            for (&idx, &rel) in param_indices.iter().zip(rel_uncerts.iter()) {
                                if !constrained[idx] {
                                    nll += ns_compute::gaussian_constraint_nll(params[idx], 1.0, rel);
                                    constrained[idx] = true;
                                }
                            }
                        }
                        ModifierModel::NormSys { param_idx, .. }
                        | ModifierModel::HistoSys { param_idx, .. }
                        | ModifierModel::Lumi { param_idx, .. } => {
                            let idx = *param_idx;
                            if !constrained[idx] {
                                nll += ns_compute::gaussian_constraint_nll(params[idx], 0.0, 1.0);
                                constrained[idx] = true;
                            }
                        }
                        ModifierModel::NormFactor { .. } => {}
                    }
                }
            }
        }

        Ok(nll)
    }

    fn parameters(&self) -> &[ns_core::Parameter] {
        &self.parameters
    }
}
```

**Step 4: Run tests**

```bash
cargo test -p ns-translate test_histfactory_nll_matches_pyhf_simple
```

**Step 5: Commit**

```bash
git add crates/ns-translate/
git commit -m "feat(model): implement HistFactoryModel::nll with pyhf parity"
```

---

## Sprint 1.3: MLE Optimizer (Недели 9-10)

### Epic 1.3.1: Minimizer Core

#### Task 1.3.1.1: Bounded Nelder–Mead minimizer

**Priority:** P0  
**Effort:** 6-8 часов  
**Dependencies:** Sprint 1.2

**Files:**
- Create: `crates/ns-inference/Cargo.toml`
- Create: `crates/ns-inference/src/lib.rs`
- Create: `crates/ns-inference/src/minimize.rs`
- Test: inline

**Acceptance Criteria:**
- [ ] Минимизатор сходится на Rosenbrock до `epsilon=1e-4`
- [ ] Поддерживает bounds через projection (`clamp`)
- [ ] Детерминированный результат при фиксированном `x0`

**Step 1: Write failing test**

```rust
// crates/ns-inference/src/minimize.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_minimize_rosenbrock() {
        let f = |x: &[f64]| -> f64 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };

        let res = minimize(
            f,
            &[0.0, 0.0],
            &[(-5.0, 5.0), (-5.0, 5.0)],
            MinimizeOptions::default(),
        )
        .unwrap();

        assert_relative_eq!(res.x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(res.x[1], 1.0, epsilon = 1e-4);
        assert_relative_eq!(res.fun, 0.0, epsilon = 1e-6);
        assert!(res.success);
    }
}
```

**Step 2: Implement minimizer**

```rust
// crates/ns-inference/src/minimize.rs
use ns_core::types::Float;

#[derive(Debug, Clone)]
pub struct MinimizeResult {
    pub x: Vec<Float>,
    pub fun: Float,
    pub nfev: usize,
    pub nit: usize,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    pub maxiter: usize,
    pub ftol: Float,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self { maxiter: 5_000, ftol: 1e-8 }
    }
}

pub fn minimize<F>(
    func: F,
    x0: &[Float],
    bounds: &[(Float, Float)],
    options: MinimizeOptions,
) -> ns_core::Result<MinimizeResult>
where
    F: Fn(&[Float]) -> Float,
{
    let n = x0.len();
    let mut nfev = 0usize;

    let eval = |x: &[Float]| -> Float {
        let bounded: Vec<Float> = x
            .iter()
            .zip(bounds.iter())
            .map(|(&v, &(lo, hi))| v.clamp(lo, hi))
            .collect();
        func(&bounded)
    };

    // Initial simplex
    let mut simplex: Vec<Vec<Float>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += 0.05 * (bounds[i].1 - bounds[i].0).abs().max(1e-6);
        simplex.push(v);
    }

    let mut values: Vec<Float> = simplex
        .iter()
        .map(|v| {
            nfev += 1;
            eval(v)
        })
        .collect();

    // Nelder–Mead coefficients
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    for iter in 0..options.maxiter {
        // Sort simplex by function values
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best = idx[0];
        let worst = idx[n];
        let second_worst = idx[n - 1];

        let spread = (values[worst] - values[best]).abs();
        if spread < options.ftol {
            return Ok(MinimizeResult {
                x: simplex[best].clone(),
                fun: values[best],
                nfev,
                nit: iter,
                success: true,
                message: "Converged".into(),
            });
        }

        // Centroid (exclude worst)
        let mut centroid = vec![0.0; n];
        for &i in &idx[..n] {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as Float;
        }

        // Reflection
        let reflected: Vec<Float> = centroid
            .iter()
            .zip(simplex[worst].iter())
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();
        nfev += 1;
        let f_reflected = eval(&reflected);

        if f_reflected < values[best] {
            // Expansion
            let expanded: Vec<Float> = centroid
                .iter()
                .zip(reflected.iter())
                .map(|(&c, &r)| c + gamma * (r - c))
                .collect();
            nfev += 1;
            let f_expanded = eval(&expanded);

            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                values[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_reflected;
            }
            continue;
        }

        if f_reflected < values[second_worst] {
            simplex[worst] = reflected;
            values[worst] = f_reflected;
            continue;
        }

        // Contraction
        let contracted: Vec<Float> = centroid
            .iter()
            .zip(simplex[worst].iter())
            .map(|(&c, &w)| c + rho * (w - c))
            .collect();
        nfev += 1;
        let f_contracted = eval(&contracted);

        if f_contracted < values[worst] {
            simplex[worst] = contracted;
            values[worst] = f_contracted;
            continue;
        }

        // Shrink
        for &i in &idx[1..] {
            for j in 0..n {
                simplex[i][j] = simplex[best][j] + sigma * (simplex[i][j] - simplex[best][j]);
            }
            nfev += 1;
            values[i] = eval(&simplex[i]);
        }
    }

    // Best effort return
    let (best_idx, best_val) = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap();

    Ok(MinimizeResult {
        x: simplex[best_idx].clone(),
        fun: best_val,
        nfev,
        nit: options.maxiter,
        success: false,
        message: "Max iterations reached".into(),
    })
}
```

**Step 3: Create crate skeleton**

```rust
// crates/ns-inference/src/lib.rs
#![warn(missing_docs)]

pub mod minimize;
pub mod mle;
```

```toml
# crates/ns-inference/Cargo.toml
[package]
name = "ns-inference"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
ns-core = { path = "../ns-core" }
nalgebra = "0.34"

[dev-dependencies]
approx.workspace = true
```

**Step 4: Run tests**

```bash
cargo test -p ns-inference test_minimize_rosenbrock
```

**Step 5: Commit**

```bash
git add crates/ns-inference/
git commit -m "feat(inference): add bounded Nelder–Mead minimizer"
```

---

### Epic 1.3.2: MLE Fit + Uncertainties

#### Task 1.3.2.1: Implement `mle_fit` with numerical Hessian

**Priority:** P0  
**Effort:** 8-12 часов  
**Dependencies:** Task 1.3.1.1

**Files:**
- Create: `crates/ns-inference/src/mle.rs`
- Test: inline

**Acceptance Criteria:**
- [ ] `mle_fit(model)` returns bestfit + uncertainties
- [ ] `twice_nll = 2 * nll(bestfit)`
- [ ] Uncertainties стабильны (Hessian regularization)

**Step 1: Write failing test (quadratic)**

```rust
// crates/ns-inference/src/mle.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ns_core::{Model, Parameter, types::Float};

    struct QuadModel {
        params: Vec<Parameter>,
    }

    impl Model for QuadModel {
        fn nll(&self, x: &[Float]) -> ns_core::Result<Float> {
            // (x-2)^2 + (y+1)^2
            Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
        }
        fn parameters(&self) -> &[Parameter] {
            &self.params
        }
    }

    #[test]
    fn test_mle_fit_quadratic() {
        let model = QuadModel {
            params: vec![
                Parameter::new("x", 0.0).bounded(-10.0, 10.0),
                Parameter::new("y", 0.0).bounded(-10.0, 10.0),
            ],
        };

        let res = mle_fit(&model, FitOptions::default()).unwrap();

        assert_relative_eq!(res.bestfit[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(res.bestfit[1], -1.0, epsilon = 1e-4);
        assert!(res.success);
        assert!(res.twice_nll >= 0.0);
    }
}
```

**Step 2: Implement `mle_fit`**

```rust
// crates/ns-inference/src/mle.rs
use nalgebra::DMatrix;
use ns_core::{Model, Parameter, types::Float};
use crate::minimize::{minimize, MinimizeOptions, MinimizeResult};

#[derive(Debug, Clone)]
pub struct FitOptions {
    pub minimize: MinimizeOptions,
    pub hessian_step: Float,
    pub hessian_damping: Float,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            minimize: MinimizeOptions::default(),
            hessian_step: 1e-4,
            hessian_damping: 1e-9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub bestfit: Vec<Float>,
    pub uncertainties: Vec<Float>,
    pub twice_nll: Float,
    pub success: bool,
    pub minimize_result: MinimizeResult,
}

pub fn mle_fit<M: Model>(model: &M, options: FitOptions) -> ns_core::Result<FitResult> {
    let params: &[Parameter] = model.parameters();
    let x0: Vec<Float> = params.iter().map(|p| p.init).collect();
    let bounds: Vec<(Float, Float)> = params
        .iter()
        .map(|p| (p.lower.unwrap_or(-1e30), p.upper.unwrap_or(1e30)))
        .collect();

    let minres = minimize(|x| model.nll(x).unwrap_or(Float::INFINITY), &x0, &bounds, options.minimize)?;
    let bestfit = minres.x.clone();
    let twice_nll = 2.0 * minres.fun;

    let h = options.hessian_step;
    let n = bestfit.len();
    let f0 = minres.fun;

    let f = |x: &[Float]| -> Float { model.nll(x).unwrap_or(Float::INFINITY) };

    // Numerical Hessian (central differences)
    let mut hess = vec![0.0; n * n];

    for i in 0..n {
        let hi = h * bestfit[i].abs().max(1.0);
        let mut xp = bestfit.clone();
        let mut xm = bestfit.clone();
        xp[i] += hi;
        xm[i] -= hi;
        let fp = f(&xp);
        let fm = f(&xm);
        hess[i * n + i] = (fp - 2.0 * f0 + fm) / (hi * hi);

        for j in (i + 1)..n {
            let hj = h * bestfit[j].abs().max(1.0);
            let mut xpp = bestfit.clone();
            let mut xpm = bestfit.clone();
            let mut xmp = bestfit.clone();
            let mut xmm = bestfit.clone();
            xpp[i] += hi; xpp[j] += hj;
            xpm[i] += hi; xpm[j] -= hj;
            xmp[i] -= hi; xmp[j] += hj;
            xmm[i] -= hi; xmm[j] -= hj;
            let fij = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * hi * hj);
            hess[i * n + j] = fij;
            hess[j * n + i] = fij;
        }
    }

    // Damping for stability
    for i in 0..n {
        hess[i * n + i] += options.hessian_damping;
    }

    let hmat = DMatrix::from_row_slice(n, n, &hess);
    let inv = hmat.try_inverse().unwrap_or_else(|| DMatrix::identity(n, n));

    let uncertainties: Vec<Float> = (0..n)
        .map(|i| inv[(i, i)].abs().sqrt())
        .collect();

    Ok(FitResult {
        bestfit,
        uncertainties,
        twice_nll,
        success: minres.success,
        minimize_result: minres,
    })
}
```

**Step 3: Commit**

```bash
git add crates/ns-inference/src/mle.rs
git commit -m "feat(inference): add MLE fit with numerical Hessian uncertainties"
```

---

## Sprint 1.4: CLI и Python API (Недели 11-12)

### Epic 1.4.1: Command Line Interface

#### Task 1.4.1.1: Базовая CLI

**Priority:** P0  
**Effort:** 4-6 часов  
**Dependencies:** Sprint 1.3

**Files:**
- Create: `crates/ns-cli/Cargo.toml`
- Create: `crates/ns-cli/src/main.rs`

**Step 0: Create Cargo.toml**

```toml
# crates/ns-cli/Cargo.toml
[package]
name = "ns-cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "nextstat"
path = "src/main.rs"

[dependencies]
ns-core = { path = "../ns-core" }
ns-translate = { path = "../ns-translate" }
ns-inference = { path = "../ns-inference" }
clap.workspace = true
serde_json.workspace = true
anyhow.workspace = true
```

**Step 1: Implement CLI**

```rust
// crates/ns-cli/src/main.rs
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "nextstat")]
#[command(about = "High-performance statistical fitting framework")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform MLE fit
    Fit {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Fit { input, output, threads } => cmd_fit(&input, output.as_deref(), threads),
    }
}

fn cmd_fit(input: &PathBuf, output: Option<&PathBuf>, _threads: usize) -> Result<()> {
    use ns_translate::pyhf::{Workspace, HistFactoryModel};
    use ns_inference::mle::{mle_fit, FitOptions};

    let json = std::fs::read_to_string(input)?;
    let workspace: Workspace = serde_json::from_str(&json)?;
    let model = HistFactoryModel::from_workspace(&workspace)?;
    let result = mle_fit(&model, FitOptions::default())?;

    let output_json = serde_json::json!({
        "bestfit": result.bestfit,
        "uncertainties": result.uncertainties,
        "twice_nll": result.twice_nll,
        "success": result.success,
        "nfev": result.minimize_result.nfev,
        "nit": result.minimize_result.nit,
    });

    if let Some(path) = output {
        std::fs::write(path, serde_json::to_string_pretty(&output_json)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&output_json)?);
    }

    Ok(())
}
```

**Step 2: Commit**

```bash
git add crates/ns-cli/
git commit -m "feat(cli): add nextstat fit command"
```

---

### Epic 1.4.2: Python Bindings

#### Task 1.4.2.1: PyO3 bindings

**Priority:** P0  
**Effort:** 6-10 часов  
**Dependencies:** Sprint 1.3

**Files:**
- Create: `bindings/ns-py/Cargo.toml`
- Create: `bindings/ns-py/src/lib.rs`
- Modify: `bindings/ns-py/python/nextstat/__init__.py`

**Step 0: Create Cargo.toml**

```toml
# bindings/ns-py/Cargo.toml
[package]
name = "ns-py"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
ns-core = { path = "../../crates/ns-core" }
ns-translate = { path = "../../crates/ns-translate" }
ns-inference = { path = "../../crates/ns-inference" }
pyo3.workspace = true
serde_json.workspace = true
```

**Step 1: Implement Rust Python module**

```rust
// bindings/ns-py/src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass]
#[derive(Clone)]
struct PyFitResult {
    #[pyo3(get)]
    bestfit: Vec<f64>,
    #[pyo3(get)]
    uncertainties: Vec<f64>,
    #[pyo3(get)]
    twice_nll: f64,
    #[pyo3(get)]
    success: bool,
}

#[pyclass]
struct PyModel {
    inner: ns_translate::pyhf::HistFactoryModel,
}

#[pymethods]
impl PyModel {
    fn n_params(&self) -> usize {
        use ns_core::Model;
        self.inner.n_params()
    }
}

#[pyfunction]
fn from_pyhf(workspace_json: &str) -> PyResult<PyModel> {
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(workspace_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(PyModel { inner: model })
}

#[pyfunction]
#[pyo3(signature = (model, data=None))]
fn fit(model: &PyModel, data: Option<Vec<f64>>) -> PyResult<PyFitResult> {
    // NOTE (Phase 1 decision): toy/ensemble validation requires overriding observations.
    // `data` is treated as **main observations only** (no auxdata); aux constraints remain fixed.
    let inner = if let Some(obs_main) = data {
        model
            .inner
            .with_observed_main(&obs_main)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?
    } else {
        model.inner.clone()
    };

    let result =
        ns_inference::mle::mle_fit(&inner, ns_inference::mle::FitOptions::default())
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(PyFitResult {
        bestfit: result.bestfit,
        uncertainties: result.uncertainties,
        twice_nll: result.twice_nll,
        success: result.success,
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_pyhf, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyFitResult>()?;
    Ok(())
}
```

**Step 1.1: Update Python package API**

```python
# bindings/ns-py/python/nextstat/__init__.py
"""NextStat - High-performance statistical fitting framework"""

from nextstat._core import from_pyhf, fit, PyModel, PyFitResult

__all__ = ["from_pyhf", "fit", "PyModel", "PyFitResult"]
```

**Step 2: Commit**

```bash
git add bindings/ns-py/
git commit -m "feat(python): add PyO3 bindings for nextstat"
```

---

## Sprint 1.5: Validation и Documentation (Недели 13-16)

### Epic 1.5.1: pyhf Parity Suite

#### Task 1.5.1.1: Deterministic CPU validation tests

**Priority:** P0  
**Effort:** 4-6 часов  
**Dependencies:** Sprint 1.4

**Files:**
- Create: `tests/python/test_pyhf_validation.py`

**Step 1: Implement validation tests**

```python
# tests/python/test_pyhf_validation.py
"""Deterministic parity tests vs pyhf (Phase 1 contract)."""

import json
from pathlib import Path

import numpy as np
import pyhf

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def pyhf_model_and_data(workspace: dict, measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def pyhf_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        if name not in dst_index:
            raise AssertionError(f"Destination missing parameter '{name}'")
        out[dst_index[name]] = float(value)
    return out


def test_simple_nll_parity_nominal_and_poi():
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    pyhf_params = model.config.suggested_init()
    pyhf_val = pyhf_nll(model, data, pyhf_params)

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_params = map_params_by_name(
        model.config.par_names,
        pyhf_params,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )
    ns_val = ns_model.nll(ns_params)

    assert abs(ns_val - pyhf_val) < 1e-10

    # POI variations
    poi_idx = model.config.poi_index
    for poi in [0.0, 2.0]:
        pyhf_params_var = list(pyhf_params)
        pyhf_params_var[poi_idx] = poi
        pyhf_val_var = pyhf_nll(model, data, pyhf_params_var)

        ns_params_var = map_params_by_name(
            model.config.par_names,
            pyhf_params_var,
            ns_model.parameter_names(),
            ns_model.suggested_init(),
        )
        ns_val_var = ns_model.nll(ns_params_var)
        assert abs(ns_val_var - pyhf_val_var) < 1e-10


def test_complex_nll_parity_nominal_and_poi():
    workspace = load_fixture("complex_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="measurement")

    pyhf_params = model.config.suggested_init()
    pyhf_val = pyhf_nll(model, data, pyhf_params)

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_names = ns_model.parameter_names()
    assert set(ns_names) == set(model.config.par_names)

    ns_params = map_params_by_name(
        model.config.par_names,
        pyhf_params,
        ns_names,
        ns_model.suggested_init(),
    )
    ns_val = ns_model.nll(ns_params)
    assert abs(ns_val - pyhf_val) < 1e-10

    poi_idx = model.config.poi_index
    for poi in [0.0, 2.0]:
        pyhf_params_var = list(pyhf_params)
        pyhf_params_var[poi_idx] = poi
        pyhf_val_var = pyhf_nll(model, data, pyhf_params_var)

        ns_params_var = map_params_by_name(
            model.config.par_names,
            pyhf_params_var,
            ns_names,
            ns_model.suggested_init(),
        )
        ns_val_var = ns_model.nll(ns_params_var)
        assert abs(ns_val_var - pyhf_val_var) < 1e-10


def test_simple_mle_parity_bestfit_uncertainties():
    workspace = load_fixture("simple_workspace.json")
    model, data = pyhf_model_and_data(workspace, measurement_name="GaussExample")

    pyhf_bestfit = np.asarray(pyhf.infer.mle.fit(data, model), dtype=float)
    pyhf_bestfit_nll = pyhf_nll(model, data, pyhf_bestfit)

    # Numerical Hessian for uncertainties (NLL, not twice_nll)
    def nll_func(x: np.ndarray) -> float:
        return pyhf_nll(model, data, x)

    n = len(pyhf_bestfit)
    h_step = 1e-4
    damping = 1e-9
    f0 = nll_func(pyhf_bestfit)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        hi = h_step * max(abs(pyhf_bestfit[i]), 1.0)
        xp = pyhf_bestfit.copy()
        xm = pyhf_bestfit.copy()
        xp[i] += hi
        xm[i] -= hi
        fp = nll_func(xp)
        fm = nll_func(xm)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (hi * hi)

        for j in range(i + 1, n):
            hj = h_step * max(abs(pyhf_bestfit[j]), 1.0)
            xpp = pyhf_bestfit.copy()
            xpm = pyhf_bestfit.copy()
            xmp = pyhf_bestfit.copy()
            xmm = pyhf_bestfit.copy()
            xpp[i] += hi
            xpp[j] += hj
            xpm[i] += hi
            xpm[j] -= hj
            xmp[i] -= hi
            xmp[j] += hj
            xmm[i] -= hi
            xmm[j] -= hj
            fij = (nll_func(xpp) - nll_func(xpm) - nll_func(xmp) + nll_func(xmm)) / (
                4.0 * hi * hj
            )
            hess[i, j] = fij
            hess[j, i] = fij

    hess = hess + np.eye(n) * damping
    cov = np.linalg.inv(hess)
    pyhf_unc = np.sqrt(np.maximum(np.diag(cov), 0.0))

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    mle = nextstat.MaximumLikelihoodEstimator()
    ns_res = mle.fit(ns_model)

    ns_names = ns_model.parameter_names()
    ns_bestfit_by_name = dict(zip(ns_names, ns_res.parameters))
    pyhf_bestfit_by_name = dict(zip(model.config.par_names, pyhf_bestfit.tolist()))

    for name in model.config.par_names:
        assert abs(ns_bestfit_by_name[name] - pyhf_bestfit_by_name[name]) < 2e-4

    ns_nll = float(ns_res.nll)
    assert abs(ns_nll - pyhf_bestfit_nll) < 1e-6

    ns_unc_by_name = dict(zip(ns_names, ns_res.uncertainties))
    pyhf_unc_by_name = dict(zip(model.config.par_names, pyhf_unc.tolist()))
    for name in model.config.par_names:
        assert abs(ns_unc_by_name[name] - pyhf_unc_by_name[name]) < 5e-4
```

**Step 2: Commit**

```bash
git add tests/python/test_pyhf_validation.py
git commit -m "test: add pyhf parity suite (Phase 1 contract)"
```

#### Task 1.5.1.2: Toy bias/pull smoke tests (regression vs pyhf)

**Priority:** P1  
**Effort:** 2-4 часа  
**Dependencies:** Task 1.5.1.1 (parity suite) + working `nextstat.fit` (Phase 1)

**Goal:** убедиться, что NextStat **не добавляет** дополнительного смещения/undercoverage относительно pyhf на ансамбле псевдо-данных (toys), а не только на одном workspace.

**Files:**
- Create: `tests/python/test_bias_pulls.py`

**Acceptance Criteria:**
- [ ] Тест детерминирован (фиксированный seed) и не flaky.
- [ ] Сравнение выполняется **vs pyhf** (reference), а не “абсолютная unbiasedness”.
- [ ] Для POI (`mu`) метрики совпадают с pyhf в пределах допусков из `docs/plans/standards.md` (раздел 6).

**Step 1: Implement toy smoke test (slow / opt-in)**

> Это **slow** тест. Запускается вручную или в nightly workflow (не на каждый PR).

```python
# tests/python/test_bias_pulls.py
"""Toy pull/bias smoke tests (regression vs pyhf).

These tests are intentionally slow and opt-in.
Run with:
  NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 pytest -v -m slow tests/python/test_bias_pulls.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = "simple_workspace.json"
N_TOYS = int(os.environ.get("NS_TOYS", "200"))
SEED = int(os.environ.get("NS_SEED", "0"))


def load_workspace() -> dict:
    return json.loads((FIXTURES_DIR / FIXTURE).read_text())


def pyhf_model_and_data(workspace: dict, measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = np.asarray(ws.data(model), dtype=float)
    return model, data


def pyhf_nll(model, data: np.ndarray, params: np.ndarray) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def numerical_uncertainties(model, data: np.ndarray, bestfit: np.ndarray) -> np.ndarray:
    n = len(bestfit)
    h_step = 1e-4
    damping = 1e-9

    f0 = pyhf_nll(model, data, bestfit)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        hi = h_step * max(abs(bestfit[i]), 1.0)
        xp = bestfit.copy()
        xm = bestfit.copy()
        xp[i] += hi
        xm[i] -= hi
        fp = pyhf_nll(model, data, xp)
        fm = pyhf_nll(model, data, xm)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (hi * hi)

        for j in range(i + 1, n):
            hj = h_step * max(abs(bestfit[j]), 1.0)
            xpp = bestfit.copy()
            xpm = bestfit.copy()
            xmp = bestfit.copy()
            xmm = bestfit.copy()
            xpp[i] += hi
            xpp[j] += hj
            xpm[i] += hi
            xpm[j] -= hj
            xmp[i] -= hi
            xmp[j] += hj
            xmm[i] -= hi
            xmm[j] -= hj
            fij = (
                pyhf_nll(model, data, xpp)
                - pyhf_nll(model, data, xpm)
                - pyhf_nll(model, data, xmp)
                + pyhf_nll(model, data, xmm)
            ) / (4.0 * hi * hj)
            hess[i, j] = fij
            hess[j, i] = fij

    hess = hess + np.eye(n) * damping
    cov = np.linalg.inv(hess)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def test_pull_mu_regression_vs_pyhf():
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow toy regression tests.")

    import pyhf
    import nextstat

    rng = np.random.default_rng(SEED)
    ws_json = load_workspace()

    # Build pyhf reference objects
    pyhf_ws = pyhf.Workspace(ws_json)
    pyhf_model = pyhf_ws.model(
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )

    # Choose "truth" pars (decision): POI=1, nuisances at init
    pars_true = np.asarray(pyhf_model.config.suggested_init(), dtype=float)
    pars_true[pyhf_model.config.poi_index] = 1.0

    # NOTE: for a smoke test we keep auxdata fixed and only fluctuate main observations.
    # Full toy generation (main + aux constraints) is a Phase 3 task.
    data_nominal = np.asarray(pyhf_ws.data(pyhf_model), dtype=float)
    expected = np.asarray(pyhf_model.expected_data(pars_true), dtype=float)
    n_main = int(pyhf_model.config.nmaindata)

    pulls_pyhf = []
    pulls_ns = []

    # Build NextStat model once; we only override observations per-toy.
    ns_model = nextstat.from_pyhf(json.dumps(ws_json))

    for _ in range(N_TOYS):
        toy = data_nominal.copy()
        toy[:n_main] = rng.poisson(expected[:n_main])

        # Fit in pyhf
        bestfit_pyhf = np.asarray(pyhf.infer.mle.fit(toy, pyhf_model), dtype=float)
        unc_pyhf = numerical_uncertainties(pyhf_model, toy, bestfit_pyhf)
        mu_hat_pyhf = float(bestfit_pyhf[pyhf_model.config.poi_index])
        mu_sig_pyhf = float(unc_pyhf[pyhf_model.config.poi_index])
        pulls_pyhf.append((mu_hat_pyhf - 1.0) / mu_sig_pyhf)

        # Fit in NextStat
        res = nextstat.fit(ns_model, data=toy[:n_main].tolist())
        ns_poi_idx = ns_model.poi_index()
        assert ns_poi_idx is not None
        mu_hat_ns = float(res.bestfit[ns_poi_idx])
        mu_sig_ns = float(res.uncertainties[ns_poi_idx])
        pulls_ns.append((mu_hat_ns - 1.0) / mu_sig_ns)

    pulls_pyhf = np.asarray(pulls_pyhf)
    pulls_ns = np.asarray(pulls_ns)

    # Compare distributions via summary stats (regression vs pyhf)
    d_mean = float(pulls_ns.mean() - pulls_pyhf.mean())
    d_std = float(pulls_ns.std(ddof=1) - pulls_pyhf.std(ddof=1))

    assert abs(d_mean) <= 0.05
    assert abs(d_std) <= 0.05
```

**Step 2: Run locally**

```bash
pip install -e "bindings/ns-py[validation,dev]"
NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 pytest -v -m slow tests/python/test_bias_pulls.py
```

**Step 3: Commit**

```bash
git add tests/python/test_bias_pulls.py
git commit -m "test(validation): add toy pull smoke test (regression vs pyhf)"
```

### Epic 1.5.2: Documentation

#### Task 1.5.2.1: User-facing docs (minimal)

**Priority:** P1  
**Effort:** 4-6 часов  
**Dependencies:** Sprint 1.4

**Files:**
- Create: `README.md`
- Create: `docs/quickstart.md`
- Create: `docs/validation.md`

**Acceptance Criteria:**
- [ ] One-command quickstart for Rust + Python
- [ ] “Numerical contract” section points to `docs/plans/standards.md`

---

## Критерии завершения фазы

### Validation Checklist

- [ ] `pytest tests/python/test_pyhf_validation.py -v` проходит (deterministic CPU)
- [ ] (P1 / opt-in) `NS_TOYS=200 pytest -v -m slow tests/python/test_bias_pulls.py` (regression vs pyhf)
- [ ] `cargo test --all` проходит
- [ ] CLI работает: `nextstat fit --input workspace.json`
- [ ] Python API работает: `nextstat.fit(model)`
- [ ] Coverage: `cargo llvm-cov --workspace --fail-under-lines 80` (или эквивалент) зелёный

### Exit Criteria

Фаза I завершена когда:

1. [ ] Phase 1 parity выполнен по `tests/python/test_pyhf_validation.py`
2. [ ] `cargo test --all` + `pytest` зелёные в CI
3. [ ] CLI + Python API доступны и документированы
4. [ ] Performance: fit < 100ms для 10 NP (CPU)
5. [ ] Test coverage ≥ 80% (Rust: `cargo llvm-cov`, Python: `pytest --cov`)

---

*Следующая фаза: [Phase 2A: CPU Parallelism](./phase-2a-cpu-parallelism.md) + [Phase 2B: Autodiff & Optimizers](./phase-2b-autodiff.md)*
