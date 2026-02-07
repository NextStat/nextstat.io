# Native HS3 (HEP Statistics Serialization Standard) Support

> **Status:** Planning
> **Author:** NextStat core team
> **Created:** 2026-02-07
> **Target HS3 version:** 0.2 (ROOT 6.37 output), forward-compatible with 0.2.9 spec
> **Reference fixture:** `tests/fixtures/workspace-postFit_PTV.json` (17 MB, 36 channels, 3243 params)

---

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. HS3 vs pyhf: Structural Differences](#2-hs3-vs-pyhf-structural-differences)
- [3. Module Architecture](#3-module-architecture)
- [4. Data Flow](#4-data-flow)
- [5. HS3 Type Mapping Tables](#5-hs3-type-mapping-tables)
- [6. Phase 1: HS3 Serde Schema](#6-phase-1-hs3-serde-schema)
- [7. Phase 2: Reference Resolution (2-Pass Parser)](#7-phase-2-reference-resolution-2-pass-parser)
- [8. Phase 3: HS3 to HistFactoryModel Conversion](#8-phase-3-hs3-to-histfactorymodel-conversion)
- [9. Phase 4: CLI and Python Integration](#9-phase-4-cli-and-python-integration)
- [10. Phase 5: Testing and Validation](#10-phase-5-testing-and-validation)
- [11. Phase 6: HS3 Export (Roundtrip)](#11-phase-6-hs3-export-roundtrip)
- [12. Risks and Edge Cases](#12-risks-and-edge-cases)
- [13. HS3 Spec Version Compatibility](#13-hs3-spec-version-compatibility)
- [14. Effort Summary](#14-effort-summary)

---

## 1. Motivation

The HS3 format is the emerging community standard for serializing HEP statistical models. It is produced natively by ROOT 6.37+ and is intended to replace ad-hoc formats like pyhf JSON and HistFactory XML for likelihood publication. NextStat currently supports:

- **pyhf JSON** -- first-class native parser (`ns-translate/src/pyhf/`)
- **HistFactory XML + ROOT** -- converts to pyhf `Workspace` then uses pyhf path (`ns-translate/src/histfactory/`)

Adding native HS3 support means:

1. **Direct ingestion** of ROOT-exported HS3 files without pyhf conversion.
2. **No information loss** -- HS3 carries explicit constraints, domains, parameter points, and multiple analyses that pyhf JSON does not represent.
3. **Roundtrip capability** -- read HS3, fit, export HS3 with updated parameter points.
4. **Community alignment** -- position NextStat as an early native HS3 consumer alongside ROOT and zfit.

The conversion path will be **HS3 JSON -> HistFactoryModel** (direct), not HS3 -> pyhf -> HistFactoryModel. This avoids lossy intermediate representation and enables HS3-specific features (multiple analyses, explicit parameter points, domain-level bounds).

---

## 2. HS3 vs pyhf: Structural Differences

| Aspect | pyhf JSON | HS3 JSON |
|--------|-----------|----------|
| **Object references** | Nested inline | Named objects, cross-referenced by string |
| **Constraint terms** | Inferred from modifier type | Explicit `gaussian_dist` objects in `distributions[]` |
| **Parameter bounds** | In `measurements[].config.parameters[]` | In `domains[]` as named axes |
| **Parameter init values** | In `measurements[].config.parameters[].inits` | In `parameter_points[]` as named sets |
| **Multiple analyses** | Single `measurements[]` | Multiple `analyses[]`, each with own POI list + likelihood ref |
| **Observed data** | `observations[]` with channel name | `data[]` with type `binned`, referenced by name from `likelihoods[]` |
| **Sample nominal data** | `samples[].data` (flat array) | `samples[].data.contents` (nested object, may include `.errors`) |
| **NormSys data** | `{hi: float, lo: float}` | `{hi: float, lo: float}` (same) |
| **HistoSys data** | `{hi_data: [...], lo_data: [...]}` | `{hi: {contents: [...]}, lo: {contents: [...]}}` |
| **NormFactor** | `{name, data: null}` | `{name, parameter: "param_name"}` |
| **NormSys extras** | None | `constraint_name`, `parameter` (named refs) |
| **HistoSys extras** | None | `constraint_name`, `parameter` (named refs) |
| **StatError** | `{name, data: [uncertainties]}` | `{name, parameters: [...names], constraint_type: "Poisson"}` |
| **StatError uncertainties** | In modifier `data` field | In `sample.data.errors` field |
| **Likelihood structure** | Implicit (all channels) | Explicit `likelihoods[]` mapping distribution names to data names |
| **Global observables** | Implicit | Explicit in `domains[]` (global_observables domain) |
| **Metadata** | `version` field | `metadata.hs3_version`, `metadata.packages[]` |
| **Misc/internal** | None | `misc.ROOT_internal` (ModelConfigs, attributes, combined_datasets) |

### NormSys hi/lo Convention

HS3 uses the same convention as pyhf: `hi` is the yield factor at alpha=+1 and `lo` is the yield factor at alpha=-1. The sign of the effect (whether hi > 1 or hi < 1) depends on the physics -- it is NOT systematically swapped. Analysis of the PTV fixture confirms a mix: 58% have hi > 1 (upward shift), 42% have hi < 1 (downward shift at alpha=+1).

### StatError: Key Difference

In pyhf, `staterror` modifiers carry per-bin absolute uncertainties in their `data` field. In HS3:
- The modifier has only `parameters` (list of named gamma parameters) and `constraint_type`.
- Per-bin uncertainties come from `sample.data.errors`.
- The conversion must aggregate `errors` across samples sharing the same staterror parameters (quadrature sum), matching the pyhf relative-sigma calculation.

---

## 3. Module Architecture

```
crates/ns-translate/src/
  hs3/                          # NEW: HS3 module
    mod.rs                      # Module root, public re-exports
    schema.rs                   # Serde structs for HS3 JSON (Phase 1)
    resolve.rs                  # 2-pass reference resolver (Phase 2)
    convert.rs                  # HS3 -> HistFactoryModel conversion (Phase 3)
    export.rs                   # HistFactoryModel -> HS3 JSON (Phase 6)
    detect.rs                   # Format auto-detection (Phase 4)
    tests.rs                    # Unit + integration tests (Phase 5)
  lib.rs                        # Add `pub mod hs3;` and re-exports
  pyhf/
    model.rs                    # (unchanged -- HistFactoryModel is the shared target)
    schema.rs                   # (unchanged)
```

### Dependency Graph (HS3 module internals)

```
schema.rs  (pure Serde types, no logic)
    |
    v
resolve.rs (consumes schema types, produces ResolvedWorkspace)
    |
    v
convert.rs (consumes ResolvedWorkspace, produces HistFactoryModel)
    |
    v
export.rs  (consumes HistFactoryModel, produces schema types)

detect.rs  (reads JSON bytes, returns FormatKind enum)
```

### Cross-crate Integration

```
ns-translate/src/hs3/convert.rs
    --> uses pyhf::model::{HistFactoryModel, Parameter, ModelChannel, ModelSample, ModelModifier, ...}
    --> uses pyhf::model::{NormSysInterpCode, HistoSysInterpCode}

ns-cli/src/main.rs
    --> uses ns_translate::hs3::from_hs3() or auto-detected load_model()

ns-py/src/lib.rs
    --> uses ns_translate::hs3::from_hs3() in PyHistFactoryModel::from_hs3()
```

---

## 4. Data Flow

### Ingestion (Read)

```
HS3 JSON file
    |
    |  serde_json::from_str()
    v
Hs3Workspace (schema.rs)          <-- raw deserialized, all references unresolved
    |
    |  resolve()
    v
ResolvedWorkspace (resolve.rs)    <-- references resolved, distributions indexed by name
    |                                  constraints matched to modifiers
    |                                  domains merged into parameter bounds
    |                                  parameter_points loaded as init values
    |
    |  convert()                     (selects one analysis + one likelihood)
    v
HistFactoryModel (pyhf/model.rs)  <-- canonical internal representation
    |
    |  .prepare()
    v
PreparedModel                     <-- SIMD-ready, cached constants for NLL eval
```

### Export (Write)

```
HistFactoryModel
    |
    |  export_hs3()
    v
Hs3Workspace (schema.rs)
    |
    |  serde_json::to_string_pretty()
    v
HS3 JSON file
```

### Auto-Detection (CLI/Python)

```
JSON file bytes
    |
    |  detect_format()
    v
FormatKind::Hs3 | FormatKind::Pyhf | FormatKind::Unknown
    |
    |  dispatch to appropriate parser
    v
HistFactoryModel
```

Detection heuristic: HS3 files have top-level `"distributions"` + `"metadata"` keys; pyhf files have `"channels"` + `"measurements"` keys.

---

## 5. HS3 Type Mapping Tables

### 5.1 Top-Level Sections

| HS3 Section | HS3 Type | NextStat Target | Notes |
|-------------|----------|-----------------|-------|
| `distributions[]` | `histfactory_dist` | `ModelChannel` + `ModelSample` + `ModelModifier` | One distribution = one channel |
| `distributions[]` | `gaussian_dist` | `Parameter.constrained=true`, `.constraint_center`, `.constraint_width` | Constraint term for NP |
| `data[]` | `binned` | `ModelChannel.observed` | Matched by name via `likelihoods[]` |
| `domains[]` | `product_domain` | `Parameter.bounds` | Axes provide (min, max) per parameter |
| `parameter_points[]` | named set | `Parameter.init` | Select set by name (default: `"default_values"` or first) |
| `analyses[]` | analysis | POI selection + likelihood selection | User selects which analysis to use |
| `likelihoods[]` | likelihood | Channel-to-data mapping | Maps distribution names to data names |
| `metadata` | metadata | Informational (version check) | Validate `hs3_version` |
| `misc` | ROOT_internal | Ignored (or stored opaquely for roundtrip) | ModelConfigs, attributes, combined_datasets |

### 5.2 Distribution Types

| HS3 `type` | NextStat Handling | Fields Used |
|-------------|-------------------|-------------|
| `histfactory_dist` | Becomes a `ModelChannel` with samples and modifiers | `name`, `axes`, `samples` |
| `gaussian_dist` | Constraint for NP parameter | `x` (constrained param), `mean` (global obs), `sigma` |
| `poisson_dist` | Constraint for gamma parameters (future) | `x`, `mean` |
| `lognormal_dist` | Constraint for lumi-like parameters (future) | `x`, `mean`, `sigma` |
| `mixture_dist` | Not needed for HistFactory models | -- |
| `product_dist` | Not needed for HistFactory models | -- |

### 5.3 Modifier Type Mapping (HS3 -> NextStat)

| HS3 Modifier `type` | HS3 Fields | NextStat `ModelModifier` | Conversion Notes |
|----------------------|-----------|--------------------------|------------------|
| `normfactor` | `name`, `parameter` | `NormFactor { param_idx }` | Look up `parameter` in param_map |
| `normsys` | `name`, `parameter`, `constraint_name`, `data.hi`, `data.lo` | `NormSys { param_idx, hi_factor, lo_factor, interp_code }` | `hi_factor = data.hi`, `lo_factor = data.lo`; interp_code from settings |
| `histosys` | `name`, `parameter`, `constraint_name`, `data.hi.contents`, `data.lo.contents` | `HistoSys { param_idx, hi_template, lo_template, interp_code }` | `hi_template = data.hi.contents`, `lo_template = data.lo.contents` |
| `staterror` | `name`, `parameters[]`, `constraint_type` | `StatError { param_indices, uncertainties }` | Uncertainties from `sample.data.errors`; one gamma param per bin |
| `shapesys` | `name`, `parameters[]`, `constraint_type`, `data.vals` | `ShapeSys { param_indices, uncertainties, nominal_values }` | Per-bin Poisson constraint (Barlow-Beeston) |
| `shapefactor` | `name`, `parameters[]` | `ShapeFactor { param_indices }` | One free param per bin |
| `lumi` | `name`, `parameter`, `constraint_name` | `Lumi { param_idx }` | Constraint from gaussian_dist or lognormal_dist |

### 5.4 Constraint Type Mapping

| HS3 Source | NextStat Parameter Fields | Notes |
|------------|--------------------------|-------|
| `gaussian_dist` with `sigma=1.0` | `constrained=true, constraint_center=Some(0.0), constraint_width=Some(1.0)` | Standard NP constraint (alpha parameters) |
| `gaussian_dist` with general sigma | `constrained=true, constraint_center=Some(mean_value), constraint_width=Some(sigma)` | General Gaussian constraint |
| `staterror` with `constraint_type="Poisson"` | `constrained=true, constraint_center=Some(1.0), constraint_width=Some(sigma_rel)` | StatError handled via relative sigma |
| `staterror` with `constraint_type="Gaussian"` | Same as Poisson but with Gaussian NLL term | Rare; default is Poisson |
| No matching constraint | `constrained=false` | Free parameters (normfactor, shapefactor, POI) |

### 5.5 Domain Axis to Parameter Bounds

| HS3 Domain Type | Axes Format | NextStat Mapping |
|-----------------|-------------|-----------------|
| `product_domain` (nuisance_parameters) | `{name, min, max}` per axis | `Parameter.bounds = (min, max)` |
| `product_domain` (global_observables) | `{name, min, max}` per axis | Global obs values (used for constraint centers) |
| `product_domain` (parameters_of_interest) | `{name, min, max}` per axis | POI bounds |

### 5.6 NormSys hi/lo Semantics

In HS3 (identical to pyhf):
- `hi` = yield multiplicative factor when alpha = +1
- `lo` = yield multiplicative factor when alpha = -1
- The interpolation formula depends on `interp_code`:
  - **Code1** (exponential): `factor = hi^alpha` (alpha >= 0) or `lo^(-alpha)` (alpha < 0)
  - **Code4** (polynomial): smooth polynomial interpolation for |alpha| <= 1

HS3 does not specify interpolation codes explicitly in the JSON; the default depends on the producing framework. ROOT HistFactory uses Code1 for NormSys and Code0 for HistoSys by default. NextStat should accept an explicit setting (matching the existing `from_workspace_with_settings` API) or use sensible defaults.

---

## 6. Phase 1: HS3 Serde Schema

**Goal:** Define Rust types that deserialize any valid HS3 v0.2 JSON file.

### Files to Create

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `crates/ns-translate/src/hs3/mod.rs` | Module root with public re-exports | ~20 |
| `crates/ns-translate/src/hs3/schema.rs` | All Serde structs | ~350 |

### Key Structs

```rust
// Top-level workspace
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Workspace {
    pub distributions: Vec<Hs3Distribution>,
    pub data: Vec<Hs3Data>,
    pub domains: Vec<Hs3Domain>,
    pub parameter_points: Vec<Hs3ParameterPointSet>,
    pub analyses: Vec<Hs3Analysis>,
    pub likelihoods: Vec<Hs3Likelihood>,
    pub metadata: Hs3Metadata,
    #[serde(default)]
    pub misc: Option<serde_json::Value>,  // Opaque pass-through for roundtrip
}

// Distribution (tagged union on "type")
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Hs3Distribution {
    #[serde(rename = "histfactory_dist")]
    HistFactory(Hs3HistFactoryDist),
    #[serde(rename = "gaussian_dist")]
    Gaussian(Hs3GaussianDist),
    #[serde(rename = "poisson_dist")]
    Poisson(Hs3PoissonDist),
    #[serde(rename = "lognormal_dist")]
    LogNormal(Hs3LogNormalDist),
    // Catch-all for unsupported types (mixture_dist, product_dist, etc.)
    #[serde(other)]
    Unknown,
}

// HistFactory distribution (= one channel)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3HistFactoryDist {
    pub name: String,
    pub axes: Vec<Hs3Axis>,
    pub samples: Vec<Hs3Sample>,
}

// Sample within a HistFactory distribution
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Sample {
    pub name: String,
    pub data: Hs3SampleData,
    pub modifiers: Vec<Hs3Modifier>,
}

// Sample data (with optional errors for staterror)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3SampleData {
    pub contents: Vec<f64>,
    #[serde(default)]
    pub errors: Option<Vec<f64>>,
}

// Modifier (tagged union on "type")
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Hs3Modifier {
    #[serde(rename = "normfactor")]
    NormFactor {
        name: String,
        parameter: String,
    },
    #[serde(rename = "normsys")]
    NormSys {
        name: String,
        parameter: String,
        constraint_name: String,
        data: Hs3NormSysData,
    },
    #[serde(rename = "histosys")]
    HistoSys {
        name: String,
        parameter: String,
        constraint_name: String,
        data: Hs3HistoSysData,
    },
    #[serde(rename = "staterror")]
    StatError {
        name: String,
        parameters: Vec<String>,
        constraint_type: String,
    },
    #[serde(rename = "shapesys")]
    ShapeSys {
        name: String,
        parameters: Vec<String>,
        #[serde(default)]
        constraint_type: Option<String>,
        #[serde(default)]
        data: Option<Hs3ShapeSysData>,
    },
    #[serde(rename = "shapefactor")]
    ShapeFactor {
        name: String,
        #[serde(default)]
        parameters: Option<Vec<String>>,
        #[serde(default)]
        parameter: Option<String>,
    },
    #[serde(rename = "lumi")]
    Lumi {
        name: String,
        parameter: String,
        #[serde(default)]
        constraint_name: Option<String>,
    },
}

// NormSys data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3NormSysData {
    pub hi: f64,
    pub lo: f64,
}

// HistoSys data (nested contents)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3HistoSysData {
    pub hi: Hs3HistoTemplate,
    pub lo: Hs3HistoTemplate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3HistoTemplate {
    pub contents: Vec<f64>,
}

// ShapeSys data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3ShapeSysData {
    pub vals: Vec<f64>,
}

// Gaussian constraint distribution
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3GaussianDist {
    pub name: String,
    pub x: String,          // constrained parameter name
    pub mean: String,        // global observable name (or literal)
    pub sigma: f64,
}

// Poisson constraint distribution (for shapesys gamma params)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3PoissonDist {
    pub name: String,
    pub x: String,
    pub mean: String,
}

// LogNormal constraint distribution
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3LogNormalDist {
    pub name: String,
    pub x: String,
    pub mean: String,
    pub sigma: f64,
}

// Axis definition (used in distributions and data)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Axis {
    pub name: String,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub nbins: Option<usize>,
    #[serde(default)]
    pub edges: Option<Vec<f64>>,
}

// Binned data entry
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Data {
    pub name: String,
    #[serde(rename = "type")]
    pub data_type: String,   // "binned"
    pub axes: Vec<Hs3Axis>,
    pub contents: Vec<f64>,
}

// Domain (parameter space definition)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Domain {
    pub name: String,
    #[serde(rename = "type")]
    pub domain_type: String,   // "product_domain"
    pub axes: Vec<Hs3DomainAxis>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3DomainAxis {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

// Parameter point set
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3ParameterPointSet {
    pub name: String,
    pub parameters: Vec<Hs3ParameterValue>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3ParameterValue {
    pub name: String,
    pub value: f64,
}

// Analysis
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Analysis {
    pub name: String,
    pub likelihood: String,
    pub parameters_of_interest: Vec<String>,
    pub domains: Vec<String>,
}

// Likelihood (channel-data mapping)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Likelihood {
    pub name: String,
    pub distributions: Vec<String>,   // distribution names
    pub data: Vec<String>,            // data object names
}

// Metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Metadata {
    pub hs3_version: String,
    #[serde(default)]
    pub packages: Option<Vec<Hs3Package>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Package {
    pub name: String,
    pub version: String,
}
```

### Phase 1 Acceptance Criteria

1. `serde_json::from_str::<Hs3Workspace>(&fixture_json)` succeeds for `workspace-postFit_PTV.json`.
2. All 309 distributions deserialized (36 `histfactory_dist`, 273 `gaussian_dist`).
3. All 72 data entries, 7 domains, 3 parameter point sets, 2 analyses, 2 likelihoods parsed.
4. All 4 modifier types present in fixture (`normsys`, `histosys`, `normfactor`, `staterror`) deserialized with correct field access.
5. `misc` field preserved as opaque `serde_json::Value`.

### Risks

- **Serde `#[serde(tag = "type")]` with unknown variants:** The `#[serde(other)]` catch-all on `Hs3Distribution` requires serde's `untagged` or custom deserializer. Alternative: use `#[serde(untagged)]` with explicit type check, or deserialize as `serde_json::Value` first then match. The safest approach is a custom `Deserialize` impl that reads the `type` field first.
- **HS3 spec is pre-1.0:** Field names or structures may change. Mitigate by making optional fields liberal (`#[serde(default)]`).
- **Large file performance:** 17 MB JSON. serde_json handles this fine (< 1 second), but we should profile.

### Estimated Effort

- `schema.rs`: ~350 lines
- `mod.rs`: ~20 lines
- Total Phase 1: **~370 lines**

---

## 7. Phase 2: Reference Resolution (2-Pass Parser)

**Goal:** Resolve all named cross-references in the HS3 workspace into indexed lookups, producing a `ResolvedWorkspace` that can be converted to `HistFactoryModel` without further name resolution.

### Files to Create/Modify

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `crates/ns-translate/src/hs3/resolve.rs` | Two-pass resolver | ~500 |

### Architecture: Two-Pass Resolution

**Pass 1: Index all named objects**

Build `HashMap<String, &T>` for each top-level section:
- `dist_map: HashMap<String, &Hs3Distribution>` -- all distributions by name
- `data_map: HashMap<String, &Hs3Data>` -- all data entries by name
- `domain_map: HashMap<String, &Hs3Domain>` -- all domains by name
- `param_point_map: HashMap<String, &Hs3ParameterPointSet>` -- parameter point sets by name
- `constraint_map: HashMap<String, &Hs3GaussianDist>` -- Gaussian constraints by name (from distributions)
- `analysis_map: HashMap<String, &Hs3Analysis>` -- analyses by name
- `likelihood_map: HashMap<String, &Hs3Likelihood>` -- likelihoods by name

Also build:
- `constraint_by_param: HashMap<String, ConstraintInfo>` -- maps parameter name to its constraint type and properties (derived from gaussian_dist's `x` field)
- `param_bounds: HashMap<String, (f64, f64)>` -- merged bounds from all domains
- `param_inits: HashMap<String, f64>` -- init values from selected parameter_points set

**Pass 2: Resolve references and build ResolvedWorkspace**

For a selected analysis:
1. Look up the analysis's `likelihood` field to find the `Hs3Likelihood`.
2. For each (distribution_name, data_name) pair in the likelihood:
   a. Look up the `Hs3HistFactoryDist` from `dist_map`.
   b. Look up the `Hs3Data` from `data_map`.
   c. For each sample's modifiers:
      - `normsys.constraint_name` -> look up in `constraint_map`, verify it matches the parameter.
      - `normsys.parameter` -> look up in `param_bounds`, `param_inits`.
      - `histosys.constraint_name` -> same as normsys.
      - `staterror.parameters[]` -> each parameter name looked up in `param_bounds`.
      - `normfactor.parameter` -> look up in `param_bounds`.
3. Validate consistency: every referenced name must exist.

### Key Structs

```rust
/// Fully resolved workspace ready for conversion to HistFactoryModel.
pub struct ResolvedWorkspace {
    /// Selected analysis name.
    pub analysis_name: String,
    /// Parameters of interest (names).
    pub pois: Vec<String>,
    /// Resolved channels: each is a (distribution, observed_data) pair.
    pub channels: Vec<ResolvedChannel>,
    /// All parameter constraints (param_name -> ConstraintInfo).
    pub constraints: HashMap<String, ConstraintInfo>,
    /// All parameter bounds (param_name -> (min, max)).
    pub bounds: HashMap<String, (f64, f64)>,
    /// Parameter init values (param_name -> init).
    pub inits: HashMap<String, f64>,
    /// HS3 metadata.
    pub metadata: Hs3Metadata,
    /// Opaque misc for roundtrip.
    pub misc: Option<serde_json::Value>,
}

pub struct ResolvedChannel {
    pub name: String,
    pub n_bins: usize,
    pub observed: Vec<f64>,
    pub samples: Vec<ResolvedSample>,
}

pub struct ResolvedSample {
    pub name: String,
    pub nominal: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    pub modifiers: Vec<ResolvedModifier>,
}

pub enum ResolvedModifier {
    NormFactor { param_name: String },
    NormSys { param_name: String, hi: f64, lo: f64 },
    HistoSys { param_name: String, hi_template: Vec<f64>, lo_template: Vec<f64> },
    StatError { param_names: Vec<String>, constraint_type: StatConstraintType },
    ShapeSys { param_names: Vec<String>, uncertainties: Vec<f64> },
    ShapeFactor { param_names: Vec<String> },
    Lumi { param_name: String },
}

pub enum StatConstraintType {
    Poisson,
    Gaussian,
}

pub struct ConstraintInfo {
    pub constraint_type: ConstraintKind,
    pub center: f64,    // nominal value (from global observable or literal)
    pub width: f64,     // sigma for Gaussian, unused for Poisson
}

pub enum ConstraintKind {
    Gaussian,
    Poisson,
    LogNormal,
}
```

### Public API

```rust
/// Resolve an HS3 workspace for a specific analysis.
///
/// If `analysis_name` is None, selects the first analysis.
/// If `param_point_set` is None, uses "default_values" or the first available set.
pub fn resolve(
    ws: &Hs3Workspace,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
) -> Result<ResolvedWorkspace>
```

### Phase 2 Acceptance Criteria

1. `resolve(&fixture_ws, None, None)` succeeds and selects the first analysis (`combPdf_asimovData`).
2. `resolve(&fixture_ws, Some("combPdf_obsData"), None)` selects the observed data analysis.
3. All 36 channels resolved with correct observed data arrays (8 bins each).
4. All 273 Gaussian constraints resolved and mapped to parameter names.
5. NormSys, HistoSys, NormFactor, StatError modifiers all resolved with correct parameter names.
6. Parameter bounds from domains correctly merged.
7. Duplicate name detection raises an error (e.g., two distributions with the same name).
8. Missing reference detection raises a descriptive error (e.g., modifier references non-existent constraint).

### Risks

- **Global observable values:** Gaussian constraints reference a `mean` that is a *parameter name* (e.g., `nom_alpha_FOO`), not a literal value. The actual value comes from `parameter_points[]`. For standard NP constraints, `mean = nom_alpha_FOO` and the value in parameter_points is `0.0`, so `constraint_center = 0.0`. Must correctly look up global observable values from parameter_points.
- **Multiple domains with overlapping axes:** Different analyses reference different domain sets. Must merge correctly or warn on conflicts.
- **binWidth parameters:** The `default_values` parameter_points set includes 1969 `binWidth_*` parameters not needed for fitting. These should be silently ignored (not added to the model's parameter list).

### Estimated Effort

- `resolve.rs`: ~500 lines
- Total Phase 2: **~500 lines**

---

## 8. Phase 3: HS3 to HistFactoryModel Conversion

**Goal:** Convert a `ResolvedWorkspace` into a `HistFactoryModel`, the same canonical struct used by all fitting code.

### Files to Create/Modify

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `crates/ns-translate/src/hs3/convert.rs` | Conversion logic | ~600 |
| `crates/ns-translate/src/pyhf/model.rs` | Add `from_hs3()` constructor (or make internals accessible) | ~30 (modifications) |

### Conversion Strategy

The conversion must produce a `HistFactoryModel` with the same structure and semantics as `from_workspace()` produces from pyhf JSON. This is critical for correctness -- all downstream code (PreparedModel, SIMD NLL, gradient, CUDA batch) depends on the exact structure of `HistFactoryModel`.

### Key Conversion Steps

1. **Build parameter list:**
   - Add POI(s) first (from `resolved.pois`). If multiple POIs, add all; the first is the primary.
   - Scan all channels/samples/modifiers to discover all unique parameter names.
   - For each parameter:
     - Bounds from `resolved.bounds` (or defaults: `(-5, 5)` for NPs, `(0, 10)` for norms).
     - Init value from `resolved.inits` (or defaults: `0.0` for NPs, `1.0` for norms/gammas).
     - Constraint info from `resolved.constraints`.
   - Build `param_map: HashMap<String, usize>` for index lookups.

2. **Compute StatError relative sigmas:**
   - For each staterror modifier, aggregate `sample.errors` across all samples sharing the same staterror parameters.
   - Compute `sigma_rel[bin] = sqrt(sum_samples(error[bin]^2)) / sum_samples(nominal[bin])`.
   - Assign to the staterror parameter's `constraint_width`.
   - This exactly mirrors the logic in `from_workspace_impl` lines 432-455.

3. **Build channels:**
   - For each `ResolvedChannel`:
     - Create `ModelChannel` with name, observed data.
     - For each `ResolvedSample`:
       - Create `ModelSample` with `nominal = sample.nominal`.
       - Convert each `ResolvedModifier` to `ModelModifier`:
         - `NormFactor` -> `ModelModifier::NormFactor { param_idx }`
         - `NormSys` -> `ModelModifier::NormSys { param_idx, hi_factor, lo_factor, interp_code }`
         - `HistoSys` -> `ModelModifier::HistoSys { param_idx, hi_template, lo_template, interp_code }`
         - `StatError` -> `ModelModifier::StatError { param_indices, uncertainties }`
         - `ShapeSys` -> `ModelModifier::ShapeSys { param_indices, uncertainties, nominal_values }`
         - `ShapeFactor` -> `ModelModifier::ShapeFactor { param_indices }`
         - `Lumi` -> `ModelModifier::Lumi { param_idx }`
     - Build `auxiliary_data` for ShapeSys (Barlow-Beeston tau values).

4. **Apply parameter_points overrides:**
   - After building the model, apply any fixed-parameter settings (bounds clamped to single point).

5. **Validate:**
   - Call `validate_internal_indices()`.
   - Verify all channels have consistent binning.
   - Verify observed data length matches n_bins per channel.

### Public API

```rust
/// Convert an HS3 workspace to a HistFactoryModel.
///
/// `analysis_name`: which analysis to build (None = first).
/// `param_point_set`: which parameter_points set to use for init values (None = "default_values").
/// `normsys_interp`: interpolation code for NormSys (default: Code4 for ROOT HistFactory).
/// `histosys_interp`: interpolation code for HistoSys (default: Code4p for ROOT HistFactory).
pub fn from_hs3(
    json: &str,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
    normsys_interp: NormSysInterpCode,
    histosys_interp: HistoSysInterpCode,
) -> Result<HistFactoryModel>

/// Convenience: defaults for interpolation codes (ROOT HistFactory defaults).
pub fn from_hs3_default(json: &str) -> Result<HistFactoryModel>
```

### Modifications to Existing Code

The `HistFactoryModel` struct and its internal types (`ModelChannel`, `ModelSample`, `ModelModifier`, `Parameter`) are currently defined in `pyhf/model.rs`. The HS3 converter needs to construct these types directly.

**Option A (preferred):** Make the internal types constructable from outside the `pyhf` module. Currently `ModelChannel`, `ModelSample`, `ModelModifier` are private to `pyhf/model.rs`. Move them to a shared location or make their constructors `pub(crate)`.

**Option B:** Add a `from_hs3_resolved()` associated function on `HistFactoryModel` in `pyhf/model.rs` that takes a `ResolvedWorkspace` and builds the model. This keeps the private types encapsulated.

**Recommendation:** Option A. The types are fundamental to the crate and should be `pub(crate)` at minimum. The long-term direction is to move the canonical model types out of `pyhf/` into a shared `model/` module. For now, making them `pub(crate)` is sufficient.

### StatError Conversion: Detailed Algorithm

The most complex part. In HS3:
- Staterror modifier has `parameters: ["gamma_bin_0", "gamma_bin_1", ...]` and `constraint_type: "Poisson"`.
- Per-bin uncertainties are in `sample.data.errors` (absolute).
- Multiple samples in the same channel may share the same staterror modifier name.

Algorithm:
```
For each channel:
    staterror_accum = {}   // name -> { sum_nominal: [f64], sum_uncert_sq: [f64] }
    For each sample:
        For each staterror modifier:
            name = modifier.name
            errors = sample.errors  // from sample.data.errors
            nominal = sample.nominal  // from sample.data.contents
            If name not in staterror_accum:
                staterror_accum[name] = { sum_nominal: [0; n_bins], sum_uncert_sq: [0; n_bins] }
            For each bin:
                staterror_accum[name].sum_nominal[bin] += nominal[bin]
                staterror_accum[name].sum_uncert_sq[bin] += errors[bin]^2

    For each staterror in staterror_accum:
        For each bin:
            sigma_rel = sqrt(sum_uncert_sq[bin]) / sum_nominal[bin]
            Set parameter constraint_width = sigma_rel
```

This mirrors exactly what pyhf `from_workspace_impl` does.

### Phase 3 Acceptance Criteria

1. `from_hs3_default(&fixture_json)` succeeds and produces a `HistFactoryModel`.
2. Model has correct number of parameters (matching parameter count from pyhf conversion of equivalent workspace).
3. Model has 36 channels, each with 8 bins.
4. POI indices correctly identified.
5. `model.nll(init_params)` produces a finite value.
6. For a simple HS3 workspace (hand-crafted), the NLL matches a manually computed expected value.

### Risks

- **Private model types:** Requires making `ModelChannel`, `ModelSample`, `ModelModifier` at least `pub(crate)`. This is a minor API change with no external impact.
- **StatError sigma calculation:** Must exactly match pyhf's aggregation logic. Differences would cause NLL divergence. Mitigate with targeted unit tests comparing sigma_rel values.
- **Interpolation code defaults:** ROOT HistFactory default differs from pyhf default. ROOT uses Code1 for NormSys (exponential) and Code0 for HistoSys (piecewise linear). pyhf also defaults to Code1/Code0 but NextStat's `from_workspace()` uses Code4/Code4p for parity reasons. The HS3 converter should default to ROOT conventions (Code1/Code0) unless overridden, since HS3 files are produced by ROOT.
- **Multiple POIs:** The PTV fixture has 4 POIs. `HistFactoryModel` currently supports one `poi_index`. For multiple POIs, the first in the list is the primary; others are treated as free parameters. The model should support setting any parameter as POI by name.

### Estimated Effort

- `convert.rs`: ~600 lines
- `model.rs` modifications: ~30 lines
- Total Phase 3: **~630 lines**

---

## 9. Phase 4: CLI and Python Integration

**Goal:** Auto-detect HS3 vs pyhf format and expose HS3 loading through CLI and Python.

### Files to Create/Modify

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `crates/ns-translate/src/hs3/detect.rs` | Format auto-detection | ~80 |
| `crates/ns-translate/src/hs3/mod.rs` | Update re-exports | ~10 (modifications) |
| `crates/ns-translate/src/lib.rs` | Add `pub mod hs3;` | ~5 (modifications) |
| `crates/ns-cli/src/main.rs` | Update `load_model()`, `load_workspace_and_model()` | ~80 (modifications) |
| `bindings/ns-py/src/lib.rs` | Add `from_hs3()` method, update `from_workspace()` with auto-detect | ~60 (modifications) |

### Format Auto-Detection

```rust
/// Detected JSON format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceFormat {
    /// pyhf JSON (has "channels" + "measurements" at top level).
    Pyhf,
    /// HS3 JSON (has "distributions" + "metadata" at top level).
    Hs3,
    /// Unknown format.
    Unknown,
}

/// Detect whether a JSON string is pyhf or HS3 format.
///
/// Uses a fast heuristic: checks for presence of key top-level fields
/// without full deserialization.
pub fn detect_format(json: &str) -> WorkspaceFormat {
    // Fast path: check first ~500 bytes for key markers
    // pyhf: "channels" and ("measurements" or "observations")
    // HS3: "distributions" and "metadata"
    // Full parse fallback: deserialize as serde_json::Value and check keys
}
```

### CLI Changes

Update `load_model()` and `load_workspace_and_model()` in `crates/ns-cli/src/main.rs`:

```rust
fn load_model(input: &PathBuf, threads: usize, parity: bool) -> Result<HistFactoryModel> {
    setup_runtime(threads, parity);
    tracing::info!(path = %input.display(), "loading workspace");
    let json = std::fs::read_to_string(input)?;

    let format = ns_translate::hs3::detect_format(&json);
    let model = match format {
        WorkspaceFormat::Hs3 => {
            tracing::info!("detected HS3 format");
            ns_translate::hs3::from_hs3_default(&json)?
        }
        WorkspaceFormat::Pyhf | WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
            ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?
        }
    };

    tracing::info!(parameters = model.parameters().len(), "workspace loaded");
    Ok(model)
}
```

Add optional CLI flags:
- `--format hs3|pyhf|auto` (default: `auto`)
- `--hs3-analysis <name>` (select which HS3 analysis to use)
- `--hs3-param-points <name>` (select parameter points set)

### Python Changes

Add to `PyHistFactoryModel`:

```rust
/// Load model from HS3 JSON string.
#[staticmethod]
fn from_hs3(json_str: &str, analysis: Option<&str>, param_points: Option<&str>) -> PyResult<Self> {
    let model = ns_translate::hs3::from_hs3(
        json_str,
        analysis,
        param_points,
        NormSysInterpCode::Code1,    // ROOT default
        HistoSysInterpCode::Code0,   // ROOT default
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(Self { inner: model })
}
```

Update `from_workspace()` to auto-detect:

```rust
#[staticmethod]
fn from_workspace(json_str: &str) -> PyResult<Self> {
    let format = ns_translate::hs3::detect_format(json_str);
    match format {
        WorkspaceFormat::Hs3 => Self::from_hs3(json_str, None, None),
        _ => { /* existing pyhf path */ }
    }
}
```

### Phase 4 Acceptance Criteria

1. `nextstat fit --input workspace-postFit_PTV.json` auto-detects HS3 and loads successfully.
2. `nextstat fit --input simple_workspace.json` still works (pyhf auto-detection).
3. `--format hs3` forces HS3 parsing, `--format pyhf` forces pyhf parsing.
4. `--hs3-analysis combPdf_obsData` selects the observed-data analysis.
5. Python: `Model.from_workspace(hs3_json)` auto-detects and loads.
6. Python: `Model.from_hs3(hs3_json, analysis="combPdf_obsData")` loads specific analysis.

### Risks

- **Auto-detection false positives:** A non-HS3 JSON with a `"distributions"` key could be misdetected. Mitigate by requiring BOTH `"distributions"` AND `"metadata"` with `"hs3_version"` subfield.
- **Breaking existing workflows:** Auto-detection must be backward-compatible. If detection fails, fall through to pyhf parsing.

### Estimated Effort

- `detect.rs`: ~80 lines
- `mod.rs` + `lib.rs` modifications: ~15 lines
- `main.rs` modifications: ~80 lines
- `ns-py/src/lib.rs` modifications: ~60 lines
- Total Phase 4: **~235 lines**

---

## 10. Phase 5: Testing and Validation

**Goal:** Comprehensive testing proving HS3 ingestion produces correct models.

### Files to Create/Modify

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `crates/ns-translate/src/hs3/tests.rs` | Unit + integration tests | ~500 |
| `tests/test_hs3.rs` | End-to-end integration tests | ~300 |
| `tests/fixtures/simple_hs3.json` | Minimal hand-crafted HS3 fixture | ~120 |
| `tests/fixtures/two_channel_hs3.json` | Two-channel HS3 fixture | ~200 |

### Testing Strategy

#### Tier 1: Schema Deserialization (Unit Tests)

Test that all HS3 types deserialize correctly from JSON fragments.

```rust
#[test] fn test_deserialize_histfactory_dist() { ... }
#[test] fn test_deserialize_gaussian_dist() { ... }
#[test] fn test_deserialize_normsys_modifier() { ... }
#[test] fn test_deserialize_histosys_modifier() { ... }
#[test] fn test_deserialize_staterror_modifier() { ... }
#[test] fn test_deserialize_normfactor_modifier() { ... }
#[test] fn test_deserialize_full_workspace() { ... }
#[test] fn test_deserialize_ptv_fixture() { ... }
```

#### Tier 2: Reference Resolution (Unit Tests)

Test the two-pass resolver with hand-crafted fixtures.

```rust
#[test] fn test_resolve_simple_workspace() { ... }
#[test] fn test_resolve_missing_distribution_error() { ... }
#[test] fn test_resolve_missing_data_error() { ... }
#[test] fn test_resolve_constraint_mapping() { ... }
#[test] fn test_resolve_domain_bounds() { ... }
#[test] fn test_resolve_parameter_points() { ... }
#[test] fn test_resolve_analysis_selection() { ... }
```

#### Tier 3: Conversion Correctness (Integration Tests)

Test that the produced `HistFactoryModel` has correct structure and NLL values.

```rust
#[test] fn test_convert_simple_hs3_nll() { ... }
#[test] fn test_convert_ptv_model_structure() { ... }
#[test] fn test_convert_ptv_n_params() { ... }
#[test] fn test_convert_ptv_n_channels() { ... }
#[test] fn test_convert_ptv_nll_finite() { ... }
```

#### Tier 4: NLL Parity (Critical Validation)

The gold-standard test. For the PTV fixture:

1. **Export to pyhf JSON via ROOT** (or use a known-good pyhf conversion):
   - If a pyhf JSON equivalent of the PTV workspace exists, load both and compare NLL at multiple parameter points.
   - If not, use ROOT to compute the NLL at known parameter points and compare.

2. **Cross-format NLL parity:**
   ```rust
   #[test]
   fn test_hs3_vs_pyhf_nll_parity() {
       // Load same model from pyhf JSON and HS3 JSON
       // Compare NLL at init, at bestfit, at random points
       // Tolerance: 1e-10 relative
   }
   ```

3. **Gradient parity:**
   ```rust
   #[test]
   fn test_hs3_gradient_matches_ad() {
       // Load HS3 model
       // Compare analytical gradient vs reverse-mode AD gradient
       // Tolerance: 1e-8 relative
   }
   ```

#### Tier 5: Roundtrip (Phase 6 Tests)

```rust
#[test] fn test_roundtrip_simple_hs3() { ... }
#[test] fn test_roundtrip_ptv_metadata_preserved() { ... }
```

#### Tier 6: Format Detection

```rust
#[test] fn test_detect_pyhf_format() { ... }
#[test] fn test_detect_hs3_format() { ... }
#[test] fn test_detect_unknown_format() { ... }
```

### Hand-Crafted Simple HS3 Fixture

Create `tests/fixtures/simple_hs3.json` -- a minimal HS3 workspace with:
- 1 channel, 2 bins
- 2 samples (signal + background)
- 1 normfactor (mu, POI)
- 1 normsys (with gaussian constraint)
- 1 staterror (with Poisson constraint)
- Observed data
- 1 analysis, 1 likelihood, domains, parameter_points

This enables exact NLL calculation by hand for verification.

### Phase 5 Acceptance Criteria

1. All Tier 1-4 tests pass.
2. Schema deserialization handles all modifier types in the PTV fixture.
3. NLL at init parameters is finite and positive for PTV fixture.
4. For the simple hand-crafted fixture, NLL matches hand-computed value to 1e-12.
5. Format detection correctly identifies all fixture files.

### Risks

- **No pyhf-equivalent PTV workspace:** The PTV fixture is HS3-only. We may not have a pyhf JSON version for direct NLL comparison. Mitigate by: (a) creating a small HS3 + pyhf pair for exact comparison, (b) using ROOT to compute reference NLL values for the PTV model.
- **Interpolation code mismatch:** If the HS3 file was produced with Code1/Code0 (ROOT defaults) but we use Code4/Code4p, NLL values will differ. Must use matching interpolation codes for parity tests.

### Estimated Effort

- `tests.rs`: ~500 lines
- `test_hs3.rs`: ~300 lines
- `simple_hs3.json`: ~120 lines
- `two_channel_hs3.json`: ~200 lines
- Total Phase 5: **~1120 lines**

---

## 11. Phase 6: HS3 Export (Roundtrip)

**Goal:** Serialize a `HistFactoryModel` back to HS3 JSON, enabling roundtrip workflows (load HS3, fit, export updated HS3 with bestfit parameter points).

### Files to Create/Modify

| File | Purpose | Estimated Lines |
|------|---------|-----------------|
| `crates/ns-translate/src/hs3/export.rs` | Export logic | ~500 |

### Export Strategy

The export must produce valid HS3 JSON that can be re-ingested by NextStat or ROOT.

### Key Functions

```rust
/// Export a HistFactoryModel to HS3 workspace JSON.
///
/// `analysis_name`: name for the analysis object.
/// `bestfit_params`: if provided, included as an additional parameter_points set.
/// `original_hs3`: if provided, preserve metadata and misc from the original.
pub fn export_hs3(
    model: &HistFactoryModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
    original_hs3: Option<&Hs3Workspace>,
) -> Result<Hs3Workspace>

/// Convenience: export to JSON string.
pub fn export_hs3_json(
    model: &HistFactoryModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
    original_hs3: Option<&Hs3Workspace>,
) -> Result<String>
```

### Export Conversion Steps

1. **Build distributions:**
   - For each channel -> `Hs3HistFactoryDist`.
   - For each constrained parameter -> `Hs3GaussianDist` constraint.

2. **Build data entries:**
   - For each channel -> `Hs3Data` with observed values.

3. **Build domains:**
   - Nuisance parameters domain.
   - Parameters of interest domain.
   - Global observables domain.

4. **Build parameter_points:**
   - `"default_values"` set with all parameters at their init values.
   - If `bestfit_params` provided, add as named set (e.g., `"bestfit"`).

5. **Build analyses and likelihoods:**
   - One analysis referencing one likelihood.
   - Likelihood maps distribution names to data names.

6. **Preserve metadata:**
   - If `original_hs3` provided, copy metadata and misc.
   - Otherwise, generate metadata with `hs3_version: "0.2"` and `packages: [{name: "NextStat", version: env!("CARGO_PKG_VERSION")}]`.

### Roundtrip Fidelity

For full roundtrip (load HS3 -> export HS3 -> reload HS3):
- Structural fidelity: same number of channels, samples, modifiers, parameters.
- Numeric fidelity: NLL at any parameter point matches to machine precision.
- Metadata preserved when `original_hs3` is provided.
- `misc` (ROOT_internal) preserved opaquely.

### Phase 6 Acceptance Criteria

1. `export_hs3(model, ...)` produces valid JSON that re-parses as `Hs3Workspace`.
2. Roundtrip: load PTV -> export -> reload -> NLL matches to 1e-14.
3. Bestfit parameter points included when provided.
4. Metadata preserved from original.
5. Exported JSON validates against HS3 v0.2 expectations (correct top-level keys, correct distribution types).

### Risks

- **Information loss in model:** `HistFactoryModel` does not preserve all HS3 information (e.g., explicit constraint names, multiple analyses, global observable names). For full roundtrip, may need to carry HS3-specific metadata through the model or store alongside it.
- **Constraint naming:** When exporting, need to generate constraint names for Gaussian distributions. Convention: `"{param_name}Constraint"`.
- **Global observable naming:** Need to generate global observable parameter names. Convention: `"nom_{param_name}"`.

### Estimated Effort

- `export.rs`: ~500 lines
- Total Phase 6: **~500 lines**

---

## 12. Risks and Edge Cases

### 12.1 HS3 Spec Instability

The HS3 spec is pre-1.0 (`v0.2.9` as of Feb 2026). Field names, structure, or semantics may change. Mitigation:
- Use `#[serde(default)]` liberally for optional fields.
- Version-check `metadata.hs3_version` and warn on unsupported versions.
- Maintain a compatibility matrix (see Section 13).

### 12.2 ROOT Version Differences

Different ROOT versions may produce slightly different HS3 output. The PTV fixture is from ROOT 6.37.01. Earlier/later versions may:
- Use different modifier field names.
- Omit optional fields.
- Include additional distribution types.

Mitigation: test with fixtures from multiple ROOT versions.

### 12.3 Multiple POIs

The PTV fixture has 4 POIs: `mu`, `mu_PTV_150_250`, `mu_PTV_250_`, `mu_PTV_75_150`. The current `HistFactoryModel` has a single `poi_index: Option<usize>`. Options:
- **Short term:** Use first POI as primary, treat others as free parameters. User can switch POI by name.
- **Long term:** Support `poi_indices: Vec<usize>` for multi-POI profiling.

### 12.4 Asimov vs Observed Data

HS3 likelihoods can reference either Asimov or observed data. The two likelihoods in the PTV fixture use different data sets (`asimovData_*` vs `obsData_*`). The resolver must select the correct data based on the chosen analysis/likelihood.

### 12.5 StatError Without Errors Field

If a sample has a `staterror` modifier but no `data.errors` field, the conversion cannot compute uncertainties. This should raise a descriptive error. In the PTV fixture, 1557 out of 1969 samples have errors (those with staterror modifiers all have errors).

### 12.6 Binning Consistency

HS3 distributions have explicit axes with `nbins`. The converter should verify that `sample.data.contents.len() == axes[0].nbins` and that observed data has the same length.

### 12.7 Global Observables as Parameters

In HS3, gaussian_dist `mean` fields reference named parameters (global observables like `nom_alpha_FOO`). These are NOT fitting parameters -- they are fixed. The resolver must:
1. Look up their values in `parameter_points`.
2. NOT add them to the model's parameter list.
3. Use their values as `constraint_center` for the corresponding NP.

### 12.8 Large Model Performance

The PTV fixture has 3243 parameters in `default_values` (including binWidth parameters and global observables), but only ~806 actual fitting parameters. The resolver must efficiently filter to only the parameters referenced by modifiers.

### 12.9 binWidth Parameters

The `default_values` parameter_points set includes ~1969 `binWidth_*` parameters. These are ROOT-internal bookkeeping and should be silently ignored during conversion.

### 12.10 Constraint Name to Parameter Matching

NormSys modifiers reference both `parameter` and `constraint_name`. The resolver should verify that the constraint_name corresponds to a gaussian_dist whose `x` field matches the parameter name. Mismatches should produce warnings.

---

## 13. HS3 Spec Version Compatibility

| HS3 Version | ROOT Version | Status in NextStat | Notes |
|-------------|-------------|-------------------|-------|
| 0.2 | 6.37.x | **Target** (primary support) | PTV fixture version |
| 0.2.9 | -- | Forward-compatible | Latest spec draft |
| 0.1 | 6.30-6.34 | Not supported | Significantly different structure |
| 1.0 (future) | TBD | Planned | Will require migration work |

### Version Detection and Handling

```rust
fn check_hs3_version(version: &str) -> Result<()> {
    let major_minor: Vec<&str> = version.split('.').collect();
    match (major_minor.get(0), major_minor.get(1)) {
        (Some(&"0"), Some(&"2")) => Ok(()),  // Supported
        (Some(&"0"), Some(minor)) => {
            let m: u32 = minor.parse().unwrap_or(0);
            if m > 2 {
                log::warn!("HS3 version {} is newer than supported 0.2; parsing may fail", version);
            }
            Ok(())
        }
        _ => Err(Error::Validation(format!(
            "Unsupported HS3 version: {}. Supported: 0.2.x", version
        ))),
    }
}
```

### Forward Compatibility Strategy

- Unknown distribution types: log a warning and skip (do not fail).
- Unknown modifier types: log a warning and skip.
- Unknown top-level keys: `#[serde(flatten)]` or `deny_unknown_fields` off.
- New modifier fields: `#[serde(default)]` on all optional fields.

---

## 14. Effort Summary

| Phase | Description | New Files | Modified Files | New Lines | Modified Lines | Priority |
|-------|-------------|-----------|----------------|-----------|----------------|----------|
| 1 | HS3 Serde Schema | 2 | 0 | 370 | 0 | P0 (foundation) |
| 2 | Reference Resolution | 1 | 0 | 500 | 0 | P0 (foundation) |
| 3 | HS3 -> HistFactoryModel | 1 | 1 | 600 | 30 | P0 (core feature) |
| 4 | CLI + Python Integration | 1 | 3 | 80 | 155 | P1 (usability) |
| 5 | Testing + Validation | 4 | 0 | 1120 | 0 | P0 (correctness) |
| 6 | HS3 Export | 1 | 0 | 500 | 0 | P2 (roundtrip) |
| **Total** | | **10** | **4** | **3170** | **185** | |

### Suggested Execution Order

1. **Phase 1** (schema) -- no dependencies, enables all other work.
2. **Phase 5 Tier 1** (schema tests) -- validate Phase 1 immediately.
3. **Phase 2** (resolver) -- depends on Phase 1.
4. **Phase 5 Tier 2** (resolver tests) -- validate Phase 2.
5. **Phase 3** (conversion) -- depends on Phases 1+2.
6. **Phase 5 Tiers 3-4** (conversion + NLL parity tests) -- validate Phase 3.
7. **Phase 4** (CLI/Python) -- depends on Phase 3.
8. **Phase 5 Tier 6** (detection tests) -- validate Phase 4.
9. **Phase 6** (export) -- depends on Phase 3, independent of Phase 4.
10. **Phase 5 Tier 5** (roundtrip tests) -- validate Phase 6.

### Timeline Estimate

- Phases 1-3 + corresponding tests: **~2 weeks** (core implementation).
- Phase 4: **~2-3 days** (integration).
- Phase 6: **~1 week** (export + roundtrip tests).
- Total: **~3-4 weeks** for full implementation.

---

## Appendix A: PTV Fixture Statistics

| Metric | Value |
|--------|-------|
| File size | 17 MB |
| Top-level keys | 8 |
| Distributions | 309 (36 histfactory_dist + 273 gaussian_dist) |
| Data entries | 72 (36 asimov + 36 observed) |
| Domains | 7 |
| Parameter point sets | 3 (default_values: 3243, raw: 806, SnSh_AllVars_Nominal: 806) |
| Analyses | 2 (asimov + observed) |
| Likelihoods | 2 |
| Channels | 36 |
| Bins per channel | 8 |
| Total samples | 1969 |
| Samples with errors | 1557 |
| Modifier counts | normsys: 36851, staterror: 1557, histosys: 3541, normfactor: 1539 |
| Unique modifier parameters | 540 |
| POIs | 4 (mu, mu_PTV_150_250, mu_PTV_250_, mu_PTV_75_150) |
| Gaussian constraints | 273 |
| binWidth parameters | 1969 (to ignore) |

## Appendix B: HS3 Spec References

- [HS3 GitHub Repository](https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard)
- [HS3 Documentation Site](https://hep-statistics-serialization-standard.github.io/)
- [HS3 Composite Distributions Spec](https://hep-statistics-serialization-standard.github.io/chapters/2.1.2_composite_distributions/)
- [HS3 Fundamental Distributions Spec](https://hep-statistics-serialization-standard.github.io/chapters/2.1.1_fundamental_distributions/)
- [ROOT HS3 Implementation (RooFit)](https://root.cern/doc/master/group__HistFactory.html)
- [zfit HS3 Module](https://zfit.readthedocs.io/en/0.14.0/_modules/zfit/hs3.html)
