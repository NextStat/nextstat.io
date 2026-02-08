---
title: "NextStat Configuration Format Reference"
status: stable
---

# NextStat Configuration Format Reference

NextStat uses a block-based text configuration format for defining statistical analyses.
The format is compatible with TRExFitter configs (common subset), but is a **first-class NextStat feature** — not a compatibility shim.

## Quick Start

```bash
# NTUP mode: build workspace from ROOT ntuples
nextstat import trex-config --config analysis.config --output workspace.json

# HIST mode: import existing HistFactory export
nextstat import trex-config --config analysis.config --output workspace.json

# One-step: build histograms + workspace
nextstat build-hists --config analysis.config --out-dir output/

# Full pipeline via analysis spec (YAML)
nextstat run --config analysis.yaml
```

## Syntax

```
# Comments: '#', '//', or '%' (quote-aware)
Key: Value
Key: "quoted value with # inside"

Region: SR
  Variable: mbb
  Binning: 0, 50, 100, 200

Sample: signal
  Type: SIGNAL
  File: data/signal.root
```

**Rules:**

- All keys are **case-insensitive** (`ReadFrom`, `readfrom`, `READFROM` are equivalent)
- Values can be quoted (`"value"` or `'value'`) or bare
- Lists accept multiple formats: `a, b, c` / `[a, b, c]` / `a; b; c` / `"a" b "c"`
- Blocks start with `BlockType: name` and extend until the next block header
- Blocks can nest: Sample inside Region, Systematic inside Sample

---

## Modes

The `ReadFrom` key selects the import mode:

| Mode | Description |
|------|-------------|
| **NTUP** (default) | Build histograms from ROOT ntuples (TTrees), then convert to pyhf workspace |
| **HIST** | Import an existing HistFactory XML export (`combination.xml`) with optional filtering |

---

## Global Keys

These keys appear at the top level (outside blocks) or inside a `Job:` block.

| Key | Aliases | Type | Default | Description |
|-----|---------|------|---------|-------------|
| `ReadFrom` | — | `NTUP` \| `HIST` | `NTUP` | Import mode |
| `TreeName` | `Tree`, `NtupleName` | string | `"events"` | Default TTree name in ROOT files |
| `Measurement` | — | string | `"meas"` | Measurement name (maps to pyhf measurement) |
| `POI` | `Poi` | string | `"mu"` | Parameter of interest |
| `HistoPath` | `HistPath`, `ExportDir` | path | — | HistFactory export directory (HIST mode) |
| `CombinationXml` | `HistFactoryXml`, `CombinationXML` | path | — | Explicit path to `combination.xml` (HIST mode) |

---

## Region Block

Defines a channel (histogram) in the analysis.

```
Region: SR
  Variable: mbb
  Binning: 0, 50, 100, 150, 200, 300
  Selection: njet >= 4
```

| Key | Aliases | Type | Required | Description |
|-----|---------|------|----------|-------------|
| `Variable` | `Var` | expression | yes | Variable to histogram. Can include inline binning (see below) |
| `Binning` | `BinEdges` | float list | yes* | Bin edges. Not needed if encoded in `Variable` |
| `Selection` | `Cut` | expression | no | Selection/cut expression (entries pass if > 0) |
| `Weight` | — | expression | no | Per-region weight (multiplied into sample weights) |
| `DataFile` | — | path | no | ROOT file for observed data. If omitted, uses the DATA sample |
| `DataTreeName` | `DataTree` | string | no | TTree name for data file. Defaults to global `TreeName` |

### Variable + Binning formats

**Inline equal-width bins:**
```
Variable: "jet_pt", 10, 0, 100    # 10 bins from 0 to 100
```

**Inline explicit edges:**
```
Variable: "jet_pt", 0, 50, 100, 200, 500
```

**Separate keys:**
```
Variable: jet_pt
Binning: 0, 50, 100, 200, 500
```

---

## Sample Block

Defines a physics sample (signal, background, or data).

```
Sample: ttbar
  Type: BACKGROUND
  File: data/ttbar.root
  Weight: weight_mc * weight_sf
  Regions: SR; CR
  NormFactor: mu_ttbar
  StatError: true
```

| Key | Aliases | Type | Required | Description |
|-----|---------|------|----------|-------------|
| `Type` | — | enum | no | `SIGNAL`, `BACKGROUND`, or `DATA`. Default: inferred (DATA if name matches `data*`/`obs*`, else `BACKGROUND`) |
| `File` | `Path`, `NtupleFile` | path | yes (NTUP) | ROOT file path |
| `NtupleFiles` | — | list | no | Alternative to `File`; first entry is used |
| `TreeName` | `Tree`, `NtupleName` | string | no | TTree name override for this sample |
| `Weight` | `MCweight` | expression | no | Per-sample weight expression |
| `Selection` | `Cut` | expression | no | Per-sample selection (legacy; usually in Region) |
| `Regions` | — | list | no | Region names where this sample contributes. If omitted: all regions |
| `NormFactor` | — | string | no | Free normalization parameter name. Repeatable for multiple factors |
| `NormSys` | — | string | no | Norm systematic: `"name lo hi"` (e.g. `"lumi 0.98 1.02"`). Repeatable |
| `StatError` | — | bool | no | Enable per-bin statistical uncertainties (Barlow-Beeston) |

**Notes:**
- Samples can be nested inside Region blocks to create region-specific overrides
- Multiple `NormFactor` and `NormSys` entries are allowed on the same sample

---

## Systematic Block

Defines a systematic uncertainty.

```
Systematic: jes
  Type: weight
  Samples: signal; ttbar
  Regions: SR
  WeightUp: weight_jes_up
  WeightDown: weight_jes_down
```

### Common keys (all types)

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `Type` | enum | no (inferred) | `norm`, `weight`, `tree`, or `histo` |
| `Samples` | list | no | Target sample names. If omitted and nested in a Sample block, applies to parent |
| `Regions` | list | no | Target region names. If omitted: all regions |

### Type: norm

Rate-only (multiplicative) systematic. No shape variation.

```
Systematic: lumi
  Type: norm
  Samples: all
  Lo: 0.98
  Hi: 1.02
```

| Key | Aliases | Type | Description |
|-----|---------|------|-------------|
| `Lo` | `Down` | float | Down factor (e.g. `0.95` = -5%) |
| `Hi` | `Up` | float | Up factor (e.g. `1.05` = +5%) |
| `OverallUp` | — | float | Alternative: up shift. Values < 0.5 treated as additive shift from 1.0 |
| `OverallDown` | — | float | Alternative: down shift. Same heuristic |

### Type: weight

Weight-based systematic. Up/down variations via different weight expressions.

```
Systematic: btag
  Type: weight
  Samples: ttbar
  WeightUp: weight_btag_up
  WeightDown: weight_btag_down
```

Three ways to specify weights (in priority order):

**1. Direct expressions:**

| Key | Aliases | Description |
|-----|---------|-------------|
| `WeightUp` | `Up` | Full up-weight expression |
| `WeightDown` | `Down` | Full down-weight expression |

**2. Pre-computed expressions:**

| Key | Description |
|-----|-------------|
| `WeightSufUp` | Up-weight expression |
| `WeightSufDown` | Down-weight expression |

**3. Suffix expansion** (base + suffix → concatenated expression):

| Key | Aliases | Description |
|-----|---------|-------------|
| `WeightBase` | `Weight` | Base weight expression |
| `WeightUpSuffix` | `UpSuffix`, `SuffixUp` | Appended to base for up variation |
| `WeightDownSuffix` | `DownSuffix`, `SuffixDown` | Appended to base for down variation |

```
# These are equivalent:
WeightUp: weight_btag_up
WeightDown: weight_btag_down

WeightBase: weight_btag
WeightUpSuffix: _up
WeightDownSuffix: _down
```

### Type: tree

Tree-based systematic. Up/down variations from alternative ROOT files.

```
Systematic: jer
  Type: tree
  Samples: signal
  FileUp: data/signal_jer_up.root
  FileDown: data/signal_jer_down.root
```

| Key | Aliases | Description |
|-----|---------|-------------|
| `FileUp` | `UpFile`, `Up` | ROOT file with up variation |
| `FileDown` | `DownFile`, `Down` | ROOT file with down variation |
| `TreeName` | `Tree` | TTree name in variation files (default: sample's tree) |

### Type: histo

Histogram-based systematic. Up/down variations from TH1 objects.

```
Systematic: model
  Type: histo
  Samples: signal
  HistoNameUp: signal_model_up
  HistoNameDown: signal_model_down
```

| Key | Aliases | Description |
|-----|---------|-------------|
| `HistoNameUp` | `HistoUp`, `NameUp` | TH1 name for up variation |
| `HistoNameDown` | `HistoDown`, `NameDown` | TH1 name for down variation |
| `HistoFileUp` | `HistoPathUp`, `FileUp` | ROOT file (default: sample's file) |
| `HistoFileDown` | `HistoPathDown`, `FileDown` | ROOT file (default: sample's file) |

---

## NormFactor Block

Defines a free-floating normalization parameter.

```
NormFactor: mu_ttbar
  Samples: ttbar
  Nominal: 1.0
  Min: 0.0
  Max: 10.0
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Samples` | list | all MC samples | Samples to apply this factor |
| `Title` | string | — | Display name |
| `Nominal` | float | 1.0 | Initial/nominal value |
| `Min` | float | — | Lower bound |
| `Max` | float | — | Upper bound |
| `Constant` | bool | false | If true, parameter is fixed (not fitted) |

**Note:** NormFactor can also be specified inline in a Sample block: `NormFactor: mu_ttbar`

---

## Expression Language

Variable, Selection, and Weight fields use an expression language supporting:

| Feature | Syntax | Example |
|---------|--------|---------|
| Arithmetic | `+`, `-`, `*`, `/` | `pt * 0.001` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` | `njet >= 4` |
| Logic | `&&`, `\|\|`, `!` | `njet >= 4 && met > 200` |
| Ternary | `cond ? a : b` | `pt > 100 ? 1.0 : 0.5` |
| Functions | `abs`, `sqrt`, `log`, `log10`, `exp`, `sin`, `cos`, `pow`, `min`, `max`, `atan2` | `sqrt(met)` |
| Static index | `branch[N]` | `jet_pt[0]` |
| Dynamic index | `branch[expr]` | `jet_pt[njet - 1]` |
| Branch names | identifiers, dots | `jet_pt`, `el.pt` |

**Indexing semantics:** numeric out-of-range, negative, or non-finite indices yield `0.0` (ROOT/TTreeFormula convention).

---

## Block Nesting

Blocks can nest hierarchically. Nested blocks inherit context from their parent:

```
Region: SR
  Variable: mbb
  Binning: 0, 50, 100, 200

  Sample: signal                    # region-specific override (no File needed)
    Weight: weight_mc * 1.1         # overrides sample weight in SR only

    Systematic: jes_sr_only         # applies to "signal" in "SR" only
      Type: norm
      Lo: 0.95
      Hi: 1.05

Sample: signal                      # top-level definition
  Type: SIGNAL
  File: data/signal.root
  Weight: weight_mc
```

**Scoping rules:**
- Systematic inside Sample → `Samples` defaults to parent sample name
- Sample inside Region → creates region-specific override; merges with top-level Sample
- Systematic inside Region → `Regions` defaults to parent region name

---

## HIST Mode

When `ReadFrom: HIST`, the config imports an existing HistFactory export rather than building from ntuples.

```
ReadFrom: HIST
HistoPath: path/to/histfactory_export

# Optional: filter to specific regions/samples
Region: SR
Region: CR_ttbar

Sample: signal
Sample: ttbar
```

In HIST mode:
- `HistoPath` or `CombinationXml` points to the HistFactory export
- Region and Sample blocks act as **filters** — only listed channels/samples are kept
- No Variable, Binning, File, or Weight keys are needed
- Systematics are imported from the XML, not re-defined

---

## Coverage Reports

NextStat can report which config keys were recognized and which were ignored:

```bash
# Unknown-attribute report
nextstat import trex-config --config analysis.config \
  --output workspace.json \
  --coverage-json coverage.json

# Expression compilation report (branch requirements, errors)
nextstat import trex-config --config analysis.config \
  --output workspace.json \
  --expr-coverage-json expr_coverage.json
```

The **coverage report** lists:
- Every unrecognized key with line number and block scope
- Counts of parsed blocks and attributes

The **expression coverage report** lists:
- Every compiled expression (Variable, Selection, Weight)
- Required branch names extracted from each expression
- Compilation errors (invalid syntax, unknown functions)

Use these to verify that your config is fully understood by NextStat.

---

## Analysis Spec (YAML Wrapper)

For IDE-friendly workflows, wrap the text config in a YAML analysis spec:

```yaml
$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

inputs:
  mode: trex_config_txt
  trex_config_txt:
    config_path: analysis.config
    base_dir: null

execution:
  determinism: { threads: 1, parity: true }
  import:
    enabled: true
    output_json: workspace.json
  fit:
    enabled: true
    output_json: fit.json
  profile_scan:
    enabled: true
    start: 0.0
    stop: 5.0
    points: 21
    output_json: scan.json
```

Run with:
```bash
nextstat run --config analysis.yaml
nextstat validate --config analysis.yaml   # schema check only
```

JSON Schema provides IDE autocomplete and validation.
See `docs/tutorials/trex-analysis-spec.md` for details.

---

## Complete Example (NTUP)

```
ReadFrom: NTUP
TreeName: nominal
Measurement: meas
POI: mu_sig

# ── Regions ──────────────────────────────

Region: SR
  Variable: mbb
  Binning: 0, 50, 100, 150, 200, 300, 500
  Selection: njet >= 4 && nbtag >= 2

Region: CR_ttbar
  Variable: mbb
  Binning: 0, 100, 200, 500
  Selection: njet >= 4 && nbtag == 1

# ── Samples ──────────────────────────────

Sample: data
  Type: DATA
  File: data/data.root

Sample: signal
  Type: SIGNAL
  File: data/signal.root
  Weight: weight_mc * weight_pileup
  NormFactor: mu_sig
  StatError: true

Sample: ttbar
  Type: BACKGROUND
  File: data/ttbar.root
  Weight: weight_mc * weight_pileup
  Regions: SR; CR_ttbar
  NormFactor: mu_ttbar
  StatError: true

# ── Systematics ──────────────────────────

Systematic: lumi
  Type: norm
  Samples: signal; ttbar
  Lo: 0.98
  Hi: 1.02

Systematic: jes
  Type: weight
  Samples: signal; ttbar
  WeightUp: weight_jes_up
  WeightDown: weight_jes_down

Systematic: btag
  Type: weight
  Samples: signal; ttbar
  WeightBase: weight_btag
  WeightUpSuffix: _up
  WeightDownSuffix: _down

Systematic: ttbar_gen
  Type: tree
  Samples: ttbar
  FileUp: data/ttbar_gen_up.root
  FileDown: data/ttbar_gen_down.root

NormFactor: mu_ttbar
  Samples: ttbar
  Nominal: 1.0
  Min: 0.0
  Max: 10.0
```
