# TRExFitter Config Surface Reference

Status: **complete** (all blocks and knobs parsed; actionable fields wired into workspace building)

Source of truth: `crates/ns-translate/src/trex/mod.rs`

## Supported Block Types

| Block | Parse | Workspace Build | Notes |
|-------|-------|-----------------|-------|
| `Job` | ✅ Full | ✅ Globals extracted | Lumi, MCstatThreshold, pruning, blinding, etc. |
| `Fit` | ✅ Full | ℹ️ Stored | FitType, FitRegion, FitBlind, NumCPU, POIAsimov, UseMinos, … |
| `Limit` | ✅ Full | ℹ️ Stored | LimitType, LimitBlind, ConfidenceLevel, … |
| `Significance` | ✅ Full | ℹ️ Stored | SignificanceBlind, … |
| `Region` | ✅ Full | ✅ Channels built | Variable/Binning/Selection + Type/Label/LogScale/Rebin/MCweight/… |
| `Sample` | ✅ Full | ✅ Samples built | File/Weight/Type + NormalizedByTheory/Group/Exclude/IgnoreSelection/… |
| `Systematic` | ✅ Full | ✅ Modifiers built | Norm/Weight/Tree/Histo/Shape + NuisanceParameter/Symmetrisation/Decorrelate/Exclude/… |
| `NormFactor` | ✅ Full | ✅ Modifiers built | Nominal/Min/Max/Samples/Regions/Constant/Expression |

**Legend:**
- ✅ Full — parsed and used in workspace/model building
- ℹ️ Stored — parsed and accessible on `TrexConfig` struct (for downstream tooling) but not directly used in pyhf workspace construction

## Job / Global Keys

| Key | Type | Aliases | Used in Build |
|-----|------|---------|---------------|
| `ReadFrom` | string | — | ✅ mode selection |
| `HistoPath` | path | `HistPath`, `ExportDir` | ✅ HIST mode base dir |
| `CombinationXml` | path | `HistFactoryXml`, `CombinationXML` | ✅ HIST mode XML |
| `TreeName` | string | `Tree`, `NtupleName` | ✅ default TTree name |
| `Measurement` | string | — | ✅ measurement name |
| `POI` | string | `Poi` | ✅ parameter of interest |
| `Lumi` | float | `Luminosity` | ℹ️ stored |
| `LumiRelErr` | float | `LumiErr`, `LumiRelativeError` | ℹ️ stored |
| `MCstatThreshold` | float | `StatErrorThreshold` | ℹ️ stored |
| `SystPruningShape` | float | — | ℹ️ stored |
| `SystPruningNorm` | float | — | ℹ️ stored |
| `DebugLevel` | int | — | ℹ️ stored |
| `BlindingType` | string | — | ℹ️ stored |
| `BlindingThreshold` | float | — | ℹ️ stored |

Plus 20+ cosmetic/presentation globals recognized for coverage (LumiLabel, ImageFormat, AtlasLabel, CmeLabel, PlotOptions, TableOptions, RankingMaxNP, etc.).

## Fit Block Keys

| Key | Type | Notes |
|-----|------|-------|
| `FitType` | string | `SPLUSB` or `BONLY` |
| `FitRegion` | string | `CRSR` or `CRONLY` |
| `FitBlind` | bool | — |
| `NumCPU` | int | — |
| `POIAsimov` | float | — |
| `UseMinos` | list | parameter names |
| `SaturatedModel` | bool | — |
| `doInjection` / `DoInjection` | bool | — |
| `InjectSignal` / `InjectionSignal` | float | — |

## Limit Block Keys

| Key | Type | Notes |
|-----|------|-------|
| `LimitType` | string | `ASYMPTOTIC` or `TOYS` |
| `LimitBlind` | bool | — |
| `POI` / `Poi` | string | — |
| `OutputDir` | string | — |
| `ConfidenceLevel` / `CL` | float | e.g. 0.95 |

## Significance Block Keys

| Key | Type | Notes |
|-----|------|-------|
| `SignificanceBlind` | bool | — |
| `POI` / `Poi` | string | — |
| `OutputDir` | string | — |

## Region Keys

| Key | Type | Aliases | Used in Build |
|-----|------|---------|---------------|
| `Variable` | string | `Var` | ✅ histogram variable |
| `Binning` | edges | `BinEdges` | ✅ bin edges |
| `Selection` | expr | `Cut` | ✅ event selection |
| `Weight` | expr | — | ✅ region weight |
| `DataFile` | path | — | ✅ observed data source |
| `DataTreeName` | string | `DataTree` | ✅ data TTree override |
| `Type` | string | — | ℹ️ SIGNAL/CONTROL/VALIDATION |
| `Label` | string | — | ℹ️ display label |
| `ShortLabel` | string | — | ℹ️ |
| `TexLabel` | string | — | ℹ️ |
| `LogScale` | bool | — | ℹ️ |
| `Rebin` | string | `Rebinning` | ℹ️ |
| `MCweight` | string | — | ℹ️ per-region MC weight |
| `AutomaticDropBins` | bool | — | ℹ️ |

Plus cosmetic keys: YaxisTitle, Ymin, Ymax, Xmin, Xmax, RatioYmin, RatioYmax, DropBins, TransfoD/F/J, etc.

## Sample Keys

| Key | Type | Aliases | Used in Build |
|-----|------|---------|---------------|
| `Type` | string | — | ✅ DATA/SIGNAL/BACKGROUND/GHOST |
| `File` | path | `Path`, `NtupleFile`, `NtupleFiles` | ✅ ROOT file |
| `TreeName` | string | `Tree`, `NtupleName` | ✅ TTree override |
| `Weight` | expr | `MCweight` | ✅ event weight |
| `Regions` | list | — | ✅ region filter |
| `Selection` | expr | `Cut` | ✅ per-sample selection |
| `NormFactor` | string | — | ✅ attach normfactor modifier |
| `NormSys` | string | — | ✅ attach normsys modifier |
| `StatError` | bool | — | ✅ enable staterror |
| `Title` | string | — | ℹ️ |
| `TexTitle` | string | — | ℹ️ |
| `Group` | string | — | ℹ️ |
| `NormalizedByTheory` | bool | `NormByTheory` | ℹ️ triggers lumi modifier |
| `LumiScale` | float | — | ℹ️ |
| `Exclude` | list | — | ✅ exclude from named regions |
| `IgnoreSelection` | bool | — | ℹ️ |
| `FillColor` | int | `FillColour` | ℹ️ |
| `LineColor` | int | `LineColour` | ℹ️ |
| `SeparateGammas` | bool | — | ℹ️ |
| `UseSystematic` | list | — | ℹ️ |
| `Template` | string | — | ✅ `POI:value` anchor for template morphing (ghost samples) |
| `HistoName` | string | — | ✅ histogram path inside ROOT file (ghost/signal samples) |
| `HistoFiles` | list | — | ✅ comma-separated ROOT file names (summed for total yield) |

## Systematic Keys

| Key | Type | Aliases | Used in Build |
|-----|------|---------|---------------|
| `Type` | string | — | ✅ NORM/WEIGHT/TREE/HISTO/SHAPE/OVERALL |
| `Samples` | list | — | ✅ sample filter |
| `Regions` | list | — | ✅ region filter |
| `OverallUp` / `OverallDown` | float | — | ✅ NormSys payload |
| `WeightUp` / `WeightDown` | expr | — | ✅ WeightSys payload |
| `WeightSufUp` / `WeightSufDown` | expr | — | ✅ WeightSys suffix variant |
| `WeightBase` + `WeightUpSuffix`/`WeightDownSuffix` | expr | `UpSuffix`, `DownSuffix`, `SuffixUp`, `SuffixDown` | ✅ WeightSys base+suffix |
| `FileUp` / `FileDown` | path | `UpFile`, `DownFile` | ✅ TreeSys payload |
| `HistoNameUp` / `HistoNameDown` | string | `HistoUp`, `HistoDown`, `NameUp`, `NameDown` | ✅ HistoSys payload |
| `HistoFileUp` / `HistoFileDown` | path | `HistoPathUp`, `HistoPathDown` | ✅ HistoSys file override |
| `NuisanceParameter` | string | — | ✅ custom NP name (default = systematic name) |
| `Exclude` | list | — | ✅ exclude from samples/regions |
| `ExcludeRegion` | list | `ExcludeRegions` | ✅ exclude from regions |
| `Symmetrisation` | string | `Symmetrization` | ℹ️ ONESIDED/TWOSIDED/MAXIMUM/ABSMEAN |
| `IsFreeParameter` | bool | — | ℹ️ unconstrained NP |
| `Decorrelate` | string | — | ℹ️ REGION/SAMPLE/SHAPEACC |
| `Category` | string | — | ℹ️ grouping label |
| `SubCategory` | string | — | ℹ️ |
| `Title` / `TexTitle` | string | — | ℹ️ |
| `ReferenceSample` | string | — | ℹ️ |
| `ScaleUp` / `ScaleDown` | float | — | ℹ️ |
| `Smoothing` | string | — | ℹ️ |
| `SmoothingOption` | string | — | ℹ️ |
| `PreSmoothing` | bool | — | ℹ️ |
| `DropNorm` | string | — | ℹ️ |
| `Pruning` | string | — | ℹ️ |

Plus aliases: TreeNameUp/Down, SampleUp/Down, CombinationType/CombineName, DropShapeIn, NtuplePath variants.

## NormFactor Block Keys

| Key | Type | Used in Build |
|-----|------|---------------|
| `Samples` | list | ✅ target samples (default: `all`) |
| `Regions` | list | ℹ️ region filter (stored for downstream) |
| `Nominal` | float | ℹ️ initial value |
| `Min` | float | ℹ️ lower bound |
| `Max` | float | ℹ️ upper bound |
| `Constant` | bool | ℹ️ |
| `Expression` | string | ℹ️ derived NormFactor |
| `Title` | string | ℹ️ |
| `Category` | string | ℹ️ |

## Template Morphing (Ghost Samples)

TRExFitter uses **ghost samples** to define per-bin polynomial morphing, where
signal yield depends on a parameter of interest (e.g. `Ytsq`). NextStat fully
supports this mechanism in HIST mode.

### How it works

Ghost samples (`Type: GHOST`) carry a `Template: POI:value` key that specifies
an anchor point for polynomial interpolation. For example, 4 samples define a
cubic polynomial:

```
Sample: "Ghost_ttbar_yt_0"
  Type: GHOST
  Template: Ytsq:0
  HistoName: yt_weights_0
  HistoFiles: ttbar_bb4l_mc20a,ttbar_bb4l_mc20d,ttbar_bb4l_mc20e

Sample: "signal_ttbar_yt_1"
  Type: SIGNAL
  Template: Ytsq:1
  HistoName: yt_weights_1

Sample: "Ghost_ttbar_yt_2"
  Type: GHOST
  Template: Ytsq:4
  HistoName: yt_weights_2

Sample: "Ghost_ttbar_yt_3"
  Type: GHOST
  Template: Ytsq:9
  HistoName: yt_weights_3
```

NextStat:
1. Groups ghost + signal samples by their `Template` parameter name
2. Reads histograms from ROOT files and sums across `HistoFiles`
3. Computes Lagrange interpolation coefficients per bin
4. Attaches a `TemplateMorphing` modifier to the signal sample

At each NLL evaluation, the signal nominal is replaced by the polynomial
evaluated at the current POI value: `signal_bin[b] = Σ_k c[b][k] · POI^k`.

### Requirements

- HIST mode only (ghost samples reference ROOT histograms, not ntuples)
- At least 2 anchor points (ghost + signal with `Template:` keys)
- ROOT files must be accessible at `{HistoPath}/{HistoFiles}.root`

## Coverage Report

Run the coverage report to check which keys NextStat recognizes in your config:

```bash
nextstat import trex-config \
  --config your_config.txt \
  --base-dir . \
  --coverage-json coverage.json
```

The report lists any `unknown` attributes that NextStat does not currently recognize, helping you identify config features that may need manual handling.

## BMCP

- **Epic**: `TRExFitter Config Surface: Full Block/Knob Coverage` (`ec2c4b99-e39b-4c08-930c-3639cff309f2`)
- **Parent**: `TREx Replacement: Full .config/.trf Import` (`2a86a7bb-7dce-4433-9e8e-b6d1a04c3410`)
