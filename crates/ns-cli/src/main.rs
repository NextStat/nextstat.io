//! NextStat CLI
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::approx_constant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::suspicious_operation_groupings)]
#![allow(dead_code)]

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use statrs::distribution::ContinuousCDF;
use std::path::{Path, PathBuf};
use std::process::Command;

mod analysis_spec;
mod churn;
mod convert;
mod discover;
mod report;
mod run;
mod survival;
mod unbinned_gpu;
mod unbinned_spec;
mod validation_report;

const SCHEMA_ANALYSIS_SPEC_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/analysis_spec_v0.schema.json"
));
const SCHEMA_BASELINE_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/baseline_v0.schema.json"
));
const SCHEMA_REPORT_DISTRIBUTIONS_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/report_distributions_v0.schema.json"
));
const SCHEMA_REPORT_PULLS_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/report_pulls_v0.schema.json"
));
const SCHEMA_REPORT_CORR_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/report_corr_v0.schema.json"
));
const SCHEMA_REPORT_YIELDS_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/report_yields_v0.schema.json"
));
const SCHEMA_REPORT_UNCERTAINTY_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/trex/report_uncertainty_v0.schema.json"
));
const SCHEMA_VALIDATION_REPORT_V1: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/validation/validation_report_v1.schema.json"
));
const SCHEMA_UNBINNED_SPEC_V0: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/schemas/unbinned/unbinned_spec_v0.schema.json"
));

/// Interpolation defaults for pyhf JSON inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InterpDefaults {
    /// pyhf defaults: NormSys=code1, HistoSys=code0.
    Pyhf,
    /// TREx/ROOT-style smooth polynomials: NormSys=code4, HistoSys=code4p.
    Root,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum UnbinnedToyGenPoint {
    /// Generate toys at the spec's initial parameter point.
    Init,
    /// Generate toys at the best fit of the observed data.
    Mle,
}

#[derive(Parser)]
#[command(name = "nextstat")]
#[command(about = "NextStat - High-performance statistical fitting")]
#[command(version)]
struct Cli {
    /// Log verbosity level (trace, debug, info, warn, error)
    #[arg(long, global = true, default_value = "warn")]
    log_level: tracing::Level,

    /// Write an immutable run bundle (inputs + hashes + outputs) into this empty directory.
    #[arg(long, global = true)]
    bundle: Option<PathBuf>,

    /// Interpolation defaults for pyhf JSON inputs (NormSys/HistoSys).
    ///
    /// - `pyhf`: NormSys=code1, HistoSys=code0
    /// - `root`: NormSys=code4, HistoSys=code4p
    #[arg(long, global = true, value_enum, default_value_t = InterpDefaults::Root)]
    interp_defaults: InterpDefaults,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// One-command workflow: HistFactory export → workspace → fit → artifacts
    Run {
        /// Run config (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,
    },

    /// Validate a run config / analysis spec (no execution)
    Validate {
        /// Run config (legacy) or analysis spec v0 (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,
    },

    /// Build histograms from ntuples (TRExFitter config, ReadFrom=NTUP) and emit `workspace.json`.
    ///
    /// This is the "NTUP pipeline" entrypoint: it reads TTrees, applies selection/weights,
    /// fills histograms, and writes a pyhf JSON workspace.
    BuildHists {
        /// Path to TRExFitter config file (text).
        #[arg(long)]
        config: PathBuf,

        /// Base directory for resolving relative ROOT file paths.
        ///
        /// Defaults to the directory containing `--config`.
        #[arg(long)]
        base_dir: Option<PathBuf>,

        /// Output directory (will contain `workspace.json`).
        #[arg(long)]
        out_dir: PathBuf,

        /// Allow writing into a non-empty `--out-dir` (overwrites known filenames).
        #[arg(long, default_value_t = false)]
        overwrite: bool,

        /// Also emit a JSON coverage report (unknown keys/attrs) to this path.
        #[arg(long)]
        coverage_json: Option<PathBuf>,

        /// Also emit a JSON expression coverage report (selection/weight/variable + weight systematics).
        ///
        /// This is intended to help parity work against legacy TREx configs by flagging unsupported
        /// constructs early and listing required branch names.
        #[arg(long)]
        expr_coverage_json: Option<PathBuf>,
    },

    /// Perform MLE fit
    Fit {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,

        /// Use GPU for NLL+gradient. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,

        /// Parity mode: Kahan summation, single-thread, deterministic.
        /// For bit-exact validation against pyhf NumPy backend.
        #[arg(long)]
        parity: bool,

        /// Regions/channels included in the fit likelihood (comma-separated).
        ///
        /// If provided, only these regions contribute to the main Poisson likelihood.
        /// Mutually exclusive with regions listed in `--validation-regions`.
        #[arg(long, value_delimiter = ',', num_args = 0..)]
        fit_regions: Vec<String>,

        /// Regions/channels excluded from the fit likelihood (comma-separated).
        ///
        /// Intended for validation regions (VR) that should not constrain the fit.
        #[arg(long, value_delimiter = ',', num_args = 0..)]
        validation_regions: Vec<String>,

        /// Fit the Asimov dataset instead of observed data (blind fit).
        ///
        /// Generates the expected (Asimov) main data at the nominal parameter point
        /// and replaces the observed data before fitting. Comparable to TRExFitter `FitBlind`.
        #[arg(long)]
        asimov: bool,
    },

    /// Perform an unbinned (event-level) MLE fit (Phase 1)
    UnbinnedFit {
        /// Unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Use GPU for unbinned NLL+gradient. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,

        /// Max optimizer iterations (L-BFGS-B).
        #[arg(long)]
        opt_max_iter: Option<u64>,

        /// Convergence tolerance for gradient norm (L-BFGS-B).
        #[arg(long)]
        opt_tol: Option<f64>,

        /// Number of L-BFGS corrections (`m`).
        #[arg(long)]
        opt_m: Option<usize>,

        /// Use smooth bounds transform in optimizer.
        #[arg(long)]
        opt_smooth_bounds: bool,
    },

    /// Perform a hybrid (binned+unbinned) MLE fit with shared parameters (Phase 4)
    HybridFit {
        /// Input binned workspace (pyhf JSON or HS3 JSON)
        #[arg(long)]
        binned: PathBuf,

        /// Input unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        unbinned: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Profile likelihood scan for an unbinned (event-level) model (Phase 1)
    UnbinnedScan {
        /// Unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,

        /// Scan start value (mu).
        #[arg(long)]
        start: f64,

        /// Scan stop value (mu).
        #[arg(long)]
        stop: f64,

        /// Number of scan points (>= 2).
        #[arg(long)]
        points: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Use GPU for unbinned NLL+gradient. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,
    },

    /// Generate Poisson toys and fit each for an unbinned (event-level) model (Phase 2)
    UnbinnedFitToys {
        /// Unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,

        /// Number of toy pseudo-experiments to generate and fit.
        #[arg(long, default_value = "100")]
        n_toys: usize,

        /// RNG seed (toy i uses seed + i).
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Parameter point to generate toys at.
        #[arg(long = "gen", value_enum, default_value_t = UnbinnedToyGenPoint::Init)]
        gen_point: UnbinnedToyGenPoint,

        /// Override generation parameters (repeatable).
        ///
        /// Example: `--set mu=1.0 --set alpha_lumi=0.0`
        #[arg(long, value_parser = parse_param_override, value_name = "NAME=VALUE")]
        set: Vec<(String, f64)>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Use GPU for unbinned toy fitting. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,

        /// CUDA device ids used for sharding toys in batch mode (comma-separated).
        ///
        /// Example: `--gpu cuda --gpu-devices 0,1`
        #[arg(long, value_delimiter = ',')]
        gpu_devices: Vec<usize>,

        /// Logical shard count for CUDA toy orchestration.
        ///
        /// When set, shards are assigned to `--gpu-devices` round-robin.
        /// This can be used to emulate multi-shard scheduling on a single GPU.
        #[arg(long)]
        gpu_shards: Option<usize>,

        /// (CUDA only, experimental) Sample toys on GPU instead of CPU. Requires `--gpu cuda`.
        #[arg(long, default_value_t = false)]
        gpu_sample_toys: bool,

        /// (CUDA only, experimental) Run the full L-BFGS optimizer on device (GPU-native).
        ///
        /// Single kernel launch: 1 CUDA block = 1 toy = 1 complete L-BFGS-B optimization.
        /// All iterations run on the GPU with zero host-device roundtrips.
        /// Requires `--gpu cuda`. Falls back to lockstep if not available.
        #[arg(long, default_value_t = false)]
        gpu_native: bool,

        /// CPU toy shard: run only a slice of the toy range for distributed farming.
        ///
        /// Format: `INDEX/TOTAL` (0-based). Example: `--shard 0/4` runs the first quarter.
        /// Each shard uses deterministic seeds: toy `i` in shard uses `seed + shard_start + i`.
        /// Combine shard outputs with `unbinned-merge-toys`.
        #[arg(long, value_parser = parse_shard)]
        shard: Option<(usize, usize)>,

        /// Fail (non-zero exit) if any toy fit errors or does not converge.
        #[arg(long)]
        require_all_converged: bool,

        /// Fail if |mean pull(POI)| exceeds this threshold (computed from converged toys only).
        #[arg(long)]
        max_abs_poi_pull_mean: Option<f64>,

        /// Fail if pull(POI) std is outside [LOW, HIGH] (computed from converged toys only).
        #[arg(long, num_args = 2, value_names = ["LOW", "HIGH"])]
        poi_pull_std_range: Option<Vec<f64>>,
    },

    /// Merge multiple shard outputs from `unbinned-fit-toys --shard` into a single result.
    ///
    /// Reads JSON files produced by individual shards and produces a combined output
    /// with aggregated convergence metrics (n_converged, n_nonconverged, n_validation_error,
    /// n_computation_error) and merged per-toy arrays.
    UnbinnedMergeToys {
        /// Shard result JSON files (at least 2).
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Output file for merged results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Nuisance-parameter ranking (impact on POI) for an unbinned (event-level) model (Phase 2)
    UnbinnedRanking {
        /// Unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "0")]
        threads: usize,
    },

    /// Hypothesis test statistic for an unbinned (event-level) model (Phase 2)
    ///
    /// Computes `q_mu = 2 * (NLL(mu) - NLL_hat)` with one-sided clipping (upper-limit style).
    UnbinnedHypotest {
        /// Unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,

        /// Tested POI value (mu).
        #[arg(long)]
        mu: f64,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Use GPU for unbinned NLL+gradient. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,
    },

    /// Toy-based CLs hypotest (qtilde) for an unbinned (event-level) model (Phase 2)
    UnbinnedHypotestToys {
        /// Unbinned spec (YAML by default; JSON if extension is .json)
        #[arg(long)]
        config: PathBuf,

        /// Tested POI value (mu)
        #[arg(long)]
        mu: f64,

        /// Toys per hypothesis (b-only and s+b).
        #[arg(long, default_value = "1000")]
        n_toys: usize,

        /// RNG seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Also return the expected CLs set (Brazil band).
        #[arg(long)]
        expected_set: bool,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic runs.
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Use GPU for batch toy fits. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,

        /// CUDA device ids used for sharding toys in batch mode (comma-separated).
        ///
        /// Example: `--gpu cuda --gpu-devices 0,1`
        #[arg(long, value_delimiter = ',')]
        gpu_devices: Vec<usize>,

        /// Logical shard count for CUDA toy orchestration.
        ///
        /// When set, shards are assigned to `--gpu-devices` round-robin.
        /// This can be used to emulate multi-shard scheduling on a single GPU.
        #[arg(long)]
        gpu_shards: Option<usize>,

        /// (CUDA only, experimental) Sample toys on GPU instead of CPU. Requires `--gpu cuda`.
        #[arg(long, default_value_t = false)]
        gpu_sample_toys: bool,
    },

    /// Audit a pyhf workspace: list channels, modifiers, unsupported features
    Audit {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: "text" (human-readable) or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file. Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Asymptotic CLs hypotest (qtilde)
    Hypotest {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Tested POI value (mu)
        #[arg(long)]
        mu: f64,

        /// Also return the expected CLs set (Brazil band).
        #[arg(long)]
        expected_set: bool,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Discovery significance (p₀ and Z-value)
    ///
    /// Tests the background-only hypothesis (μ=0) and reports the observed
    /// discovery p-value (p₀) and significance (Z = Φ⁻¹(1−p₀) ≈ √q₀).
    /// Comparable to TRExFitter `GetSignificance`.
    Significance {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Goodness-of-fit test (saturated model)
    ///
    /// Fits the model, then computes the Poisson deviance χ² between the
    /// best-fit expected yields and observed data.  Reports χ², ndof, and
    /// p-value.  Comparable to TRExFitter saturated-model GoF.
    GoodnessOfFit {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Combine multiple pyhf JSON workspaces into a single workspace
    ///
    /// Merges channels, observations, and measurement parameter configs.
    /// Systematics with the same name are automatically shared (correlated).
    /// Channel names are prefixed if there are conflicts across workspaces.
    /// Comparable to TRExFitter `MultiFit` combination and `pyhf combine`.
    Combine {
        /// Input workspace files (pyhf JSON), at least 2
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Output file for combined workspace (pyhf JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Prefix channel names with workspace index to avoid conflicts
        /// (e.g. "SR" becomes "ws0_SR", "ws1_SR").
        /// If omitted, channel names must be unique across all inputs.
        #[arg(long, default_value_t = false)]
        prefix_channels: bool,
    },

    /// Toy-based CLs hypotest (qtilde)
    HypotestToys {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Tested POI value (mu)
        #[arg(long)]
        mu: f64,

        /// Toys per hypothesis (b-only and s+b).
        #[arg(long, default_value = "1000")]
        n_toys: usize,

        /// RNG seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Also return the expected CLs set (Brazil band).
        #[arg(long)]
        expected_set: bool,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "0")]
        threads: usize,

        /// Use GPU for batch toy fits. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,
    },

    /// Observed CLs upper limit via bisection (asymptotics, qtilde)
    UpperLimit {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Target CLs level (alpha), typically 0.05
        #[arg(long, default_value = "0.05")]
        alpha: f64,

        /// Also compute expected (Brazil band) limits.
        ///
        /// In scan-mode this is always available; in bisection-mode this enables
        /// root-finding for the 5 expected curves.
        #[arg(long)]
        expected: bool,

        /// Use scan mode: scan start (mu). Requires `--scan-stop` and `--scan-points`.
        #[arg(long, requires_all = ["scan_stop", "scan_points"])]
        scan_start: Option<f64>,

        /// Use scan mode: scan stop (mu). Requires `--scan-start` and `--scan-points`.
        #[arg(long, requires_all = ["scan_start", "scan_points"])]
        scan_stop: Option<f64>,

        /// Use scan mode: number of scan points (inclusive). Requires `--scan-start` and `--scan-stop`.
        #[arg(long, requires_all = ["scan_start", "scan_stop"])]
        scan_points: Option<usize>,

        /// Lower bracket (mu)
        #[arg(long, default_value = "0.0")]
        lo: f64,

        /// Upper bracket (mu). If omitted, uses POI upper bound from the workspace.
        #[arg(long)]
        hi: Option<f64>,

        /// Relative tolerance for bisection
        #[arg(long, default_value = "0.0001")]
        rtol: f64,

        /// Max bisection iterations
        #[arg(long, default_value = "80")]
        max_iter: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Mass scan: run asymptotic CLs upper limits across multiple workspaces (Type B Brazil Band)
    ///
    /// Reads a directory of workspace JSON files (one per mass/signal hypothesis),
    /// computes 95% CL upper limits for each, and outputs a JSON array with
    /// observed and expected (±1σ/±2σ) limits — the data for an ATLAS/CMS-style
    /// exclusion plot (μ_up vs mass).
    MassScan {
        /// Directory containing workspace JSON files (one per mass point).
        ///
        /// Files are sorted lexicographically; use zero-padded names
        /// (e.g. `mass_100.json`, `mass_200.json`) for correct ordering.
        #[arg(long)]
        workspaces_dir: PathBuf,

        /// Target CLs level (alpha), typically 0.05
        #[arg(long, default_value = "0.05")]
        alpha: f64,

        /// CLs scan start (mu)
        #[arg(long, default_value = "0.0")]
        scan_start: f64,

        /// CLs scan stop (mu)
        #[arg(long, default_value = "5.0")]
        scan_stop: f64,

        /// CLs scan points (per mass point)
        #[arg(long, default_value = "41")]
        scan_points: usize,

        /// Optional labels for each mass point (comma-separated, same order as files).
        ///
        /// If omitted, filenames (without extension) are used as labels.
        #[arg(long, value_delimiter = ',')]
        labels: Vec<String>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Profile likelihood scan over POI values (q_mu)
    Scan {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Scan start (mu)
        #[arg(long, default_value = "0.0")]
        start: f64,

        /// Scan stop (mu)
        #[arg(long, default_value = "5.0")]
        stop: f64,

        /// Number of points (inclusive)
        #[arg(long, default_value = "21")]
        points: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write a standardized metrics JSON (schema `nextstat_metrics_v0`) for experiment tracking.
        ///
        /// Pass `-` to write to stdout.
        #[arg(long)]
        json_metrics: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,

        /// Use GPU for NLL+gradient. Values: "cuda" (NVIDIA, f64) or "metal" (Apple Silicon, f32).
        #[arg(long)]
        gpu: Option<String>,
    },

    /// TREx-like report artifacts (+ optional PDF/SVG rendering)
    Report {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// HistFactory `combination.xml` (used to extract bin edges from ROOT histograms).
        #[arg(long)]
        histfactory_xml: PathBuf,

        /// Fit result JSON (from `nextstat fit`) providing postfit parameters (+ uncertainties/covariance).
        ///
        /// If omitted, `nextstat report` runs an MLE fit itself and writes `fit.json` into `--out-dir`.
        #[arg(long)]
        fit: Option<PathBuf>,

        /// Output directory for artifacts (JSON). Will be created if missing.
        #[arg(long)]
        out_dir: PathBuf,

        /// Allow writing into a non-empty `--out-dir` (overwrites known filenames).
        #[arg(long, default_value_t = false)]
        overwrite: bool,

        /// Also include the raw covariance matrix in the correlation artifact (if available).
        #[arg(long, default_value_t = false)]
        include_covariance: bool,

        /// Render publication-ready PDF + per-plot SVGs via Python (`python -m nextstat.report ...`).
        #[arg(long, default_value_t = false)]
        render: bool,

        /// Path to output PDF (defaults to `--out-dir/report.pdf` when `--render`).
        #[arg(long)]
        pdf: Option<PathBuf>,

        /// Directory for per-plot SVGs (defaults to `--out-dir/svg` when `--render`).
        #[arg(long)]
        svg_dir: Option<PathBuf>,

        /// Python executable used for rendering (defaults to `.venv/bin/python` if it exists, else `python3`).
        #[arg(long)]
        python: Option<PathBuf>,

        /// Skip computing the uncertainty breakdown artifact (ranking-based, can be expensive).
        #[arg(long, default_value_t = false)]
        skip_uncertainty: bool,

        /// Uncertainty grouping policy (currently: `prefix_1`).
        #[arg(long, default_value = "prefix_1")]
        uncertainty_grouping: String,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,

        /// Make JSON output deterministic (stable ordering; normalize timestamps/timings).
        #[arg(long, default_value_t = false)]
        deterministic: bool,

        /// Regions/channels for which observed data should be suppressed in report artifacts (comma-separated).
        ///
        /// Intended for SR blinding workflows: the report will omit data points/tables for these regions.
        #[arg(long, value_delimiter = ',', num_args = 0..)]
        blind_regions: Vec<String>,
    },

    /// Unified validation report pack (Apex2 + workspace fingerprint) (+ optional PDF)
    ValidationReport {
        /// Apex2 master report JSON (from `tests/apex2_master_report.py`).
        #[arg(long)]
        apex2: PathBuf,

        /// Workspace JSON used for dataset/model fingerprinting.
        #[arg(long)]
        workspace: PathBuf,

        /// Output JSON path (`validation_report_v1`).
        #[arg(long)]
        out: PathBuf,

        /// Optional output PDF path (rendered via `python -m nextstat.validation_report ...`).
        #[arg(long)]
        pdf: Option<PathBuf>,

        /// Python executable used for rendering (defaults to `.venv/bin/python` if it exists, else `python3`).
        #[arg(long)]
        python: Option<PathBuf>,

        /// Make JSON/PDF output deterministic (stable ordering; omit timestamps/timings).
        #[arg(long, default_value_t = false)]
        deterministic: bool,
    },

    /// Visualization artifacts (plot-friendly JSON)
    Viz {
        #[command(subcommand)]
        command: VizCommands,
    },

    /// Import external formats into NextStat-compatible inputs
    Import {
        #[command(subcommand)]
        command: ImportCommands,
    },

    /// TRExFitter migration helpers
    Trex {
        #[command(subcommand)]
        command: TrexCommands,
    },

    /// Workspace preprocessing (native Rust, no Python)
    Preprocess {
        #[command(subcommand)]
        command: PreprocessCommands,
    },

    /// Export NextStat inputs into external formats (for cross-checks)
    Export {
        #[command(subcommand)]
        command: ExportCommands,
    },

    /// Time series and state space models (Phase 8)
    Timeseries {
        #[command(subcommand)]
        command: TimeseriesCommands,
    },

    /// Survival analysis (Phase 9)
    Survival {
        #[command(subcommand)]
        command: SurvivalCommands,
    },

    /// Monte Carlo fault-tree simulation (aviation reliability)
    FaultTreeMc {
        /// Input JSON config with fault tree spec.
        #[arg(short, long)]
        config: PathBuf,

        /// Number of MC scenarios.
        #[arg(long, default_value = "10000000")]
        n_scenarios: usize,

        /// RNG seed.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Device: "cpu" or "cuda".
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Scenarios per chunk (0 = auto).
        #[arg(long, default_value = "0")]
        chunk_size: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Subscription / churn analysis workflows
    Churn {
        #[command(subcommand)]
        command: ChurnCommands,
    },

    /// Convert ROOT TTree branches to Parquet (unbinned event-level schema v1).
    ///
    /// Example:
    ///   nextstat convert --input data.root --tree MyTree --output events.parquet \
    ///     --observable mass:100:180 --observable pt:0:500
    Convert {
        /// Input ROOT file.
        #[arg(short, long)]
        input: PathBuf,

        /// TTree name inside the ROOT file.
        #[arg(long)]
        tree: String,

        /// Output Parquet file path.
        #[arg(short, long)]
        output: PathBuf,

        /// Observable specification (repeatable).
        ///
        /// Format: `name:low:high` or `name:expr:low:high`.
        /// Example: `mass:100:180` or `mass:m_inv:100:180`.
        #[arg(long = "observable", value_name = "SPEC")]
        observables: Vec<String>,

        /// Selection expression (ROOT expression language).
        #[arg(long)]
        selection: Option<String>,

        /// Per-event weight expression.
        #[arg(long)]
        weight: Option<String>,

        /// Maximum number of events to write.
        #[arg(long)]
        max_events: Option<usize>,
    },

    /// Configuration helpers (schemas, etc.)
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },

    /// Print version information
    Version,
}

#[derive(Subcommand)]
enum VizCommands {
    /// Profile likelihood curve artifact (q_mu vs mu)
    Profile {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Scan start (mu)
        #[arg(long, default_value = "0.0")]
        start: f64,

        /// Scan stop (mu)
        #[arg(long, default_value = "5.0")]
        stop: f64,

        /// Number of points (inclusive)
        #[arg(long, default_value = "21")]
        points: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// CLs curve artifact with Brazil bands (observed + expected)
    Cls {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Target CLs level (alpha), typically 0.05
        #[arg(long, default_value = "0.05")]
        alpha: f64,

        /// Scan start (mu)
        #[arg(long, default_value = "0.0")]
        scan_start: f64,

        /// Scan stop (mu)
        #[arg(long, default_value = "5.0")]
        scan_stop: f64,

        /// Number of scan points (inclusive)
        #[arg(long, default_value = "201")]
        scan_points: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Nuisance-parameter ranking artifact (impact on POI)
    Ranking {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Pulls + constraints artifact (TREx-style)
    Pulls {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Fit result JSON (from `nextstat fit`) providing postfit parameters + uncertainties.
        #[arg(long)]
        fit: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Gammas (staterror / Barlow-Beeston) artifact
    ///
    /// Shows postfit values of γ parameters relative to their nominal (1.0),
    /// with prefit and postfit uncertainties.  Comparable to TRExFitter gammas plot.
    Gammas {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Fit result JSON (from `nextstat fit`) providing postfit parameters + uncertainties.
        #[arg(long)]
        fit: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Correlation matrix artifact (TREx-style)
    Corr {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Fit result JSON (from `nextstat fit`) providing covariance.
        #[arg(long)]
        fit: PathBuf,

        /// Also include the raw covariance matrix in the output artifact.
        #[arg(long)]
        include_covariance: bool,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// TREx-like stacked distributions artifact (prefit/postfit + ratio)
    Distributions {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// HistFactory `combination.xml` (used to extract bin edges from ROOT histograms).
        #[arg(long)]
        histfactory_xml: PathBuf,

        /// Fit result JSON (from `nextstat fit`) providing postfit parameters.
        ///
        /// If omitted, postfit is set equal to prefit (init parameters).
        #[arg(long)]
        fit: Option<PathBuf>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Separation plot artifact (S vs B shape comparison per channel)
    ///
    /// Shows signal and background shapes normalised to unit area, plus a
    /// numeric separation metric.  Comparable to TRExFitter separation plot.
    Separation {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Names of signal samples (comma-separated). If omitted, the first
        /// sample containing a normfactor modifier with the POI name is used.
        #[arg(long, value_delimiter = ',')]
        signal_samples: Vec<String>,

        /// HistFactory `combination.xml` (optional, for bin edges).
        #[arg(long)]
        histfactory_xml: Option<PathBuf>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },

    /// Summary plot artifact (multi-fit μ comparison)
    ///
    /// Takes multiple fit result JSONs and produces an artifact with POI
    /// central values and uncertainties for each.  Comparable to TRExFitter
    /// summary plot in combination papers.
    Summary {
        /// Fit result JSON files (from `nextstat fit`), at least 1
        #[arg(required = true, num_args = 1..)]
        fits: Vec<PathBuf>,

        /// Labels for each fit (comma-separated, same order as fit files).
        /// If omitted, labels are derived from file names.
        #[arg(long, value_delimiter = ',')]
        labels: Vec<String>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Pie chart artifact (sample composition per channel)
    ///
    /// Shows the fraction of total expected yield contributed by each sample
    /// (process) in every channel.  Comparable to TRExFitter pie chart.
    Pie {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Fit result JSON (from `nextstat fit`).
        /// If omitted, uses prefit (init) parameters.
        #[arg(long)]
        fit: Option<PathBuf>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
    },
}

#[derive(Subcommand)]
enum ImportCommands {
    /// HistFactory `combination.xml` (+ referenced ROOT histograms) → pyhf JSON workspace
    Histfactory {
        /// Path to HistFactory `combination.xml`
        #[arg(long, required_unless_present = "dir", conflicts_with = "dir")]
        xml: Option<PathBuf>,

        /// Directory to scan recursively for `combination.xml` (TREx export dirs).
        #[arg(long, required_unless_present = "xml", conflicts_with = "xml")]
        dir: Option<PathBuf>,

        /// Base directory for resolving relative paths in XML (ROOT files, sub-XMLs).
        ///
        /// Defaults to the parent directory of `--xml`. Needed for pyhf validation
        /// fixtures where paths are relative to the export root, not to the XML file.
        #[arg(long)]
        basedir: Option<PathBuf>,

        /// Output file for the workspace (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// TRExFitter-style config (subset; ReadFrom=NTUP) → pyhf JSON workspace
    TrexConfig {
        /// Path to TRExFitter config file (text).
        #[arg(long)]
        config: PathBuf,

        /// Base directory for resolving relative ROOT file paths.
        ///
        /// Defaults to the directory containing `--config`.
        #[arg(long)]
        base_dir: Option<PathBuf>,

        /// Output file for the workspace (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also emit an analysis spec YAML (mode=trex_config_txt) to this path.
        #[arg(long)]
        analysis_yaml: Option<PathBuf>,

        /// Also emit a JSON coverage report (unknown keys/attrs) to this path.
        #[arg(long)]
        coverage_json: Option<PathBuf>,

        /// Also emit a JSON expression coverage report (selection/weight/variable + weight systematics).
        #[arg(long)]
        expr_coverage_json: Option<PathBuf>,
    },

    /// Apply a pyhf PatchSet (HEPData) to a base workspace JSON (typically background-only).
    Patchset {
        /// Base workspace JSON (e.g., `BkgOnly.json`).
        #[arg(long)]
        workspace: PathBuf,

        /// PatchSet JSON (RFC 6902 patch container).
        #[arg(long)]
        patchset: PathBuf,

        /// Patch name to apply (defaults to the first patch in the PatchSet).
        #[arg(long)]
        patch_name: Option<String>,

        /// Output file for the patched workspace (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ExportCommands {
    /// pyhf workspace.json → HistFactory XML + ROOT hists (via pyhf.writexml)
    Histfactory {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory (will contain `combination.xml`, `data.root`, `HistFactorySchema.dtd`, and `channels/`)
        #[arg(long)]
        out_dir: PathBuf,

        /// Overwrite existing out_dir contents.
        #[arg(long, default_value_t = false)]
        overwrite: bool,

        /// Output prefix used by HistFactory (also names channel XML files).
        #[arg(long, default_value = "results")]
        prefix: String,

        /// Python executable to use (default: python3).
        #[arg(long)]
        python: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum PreprocessCommands {
    /// Smooth histosys templates (353QH,twice — ROOT TH1::Smooth equivalent).
    ///
    /// Applies running-median smoothing to HistoSys up/down deltas (variation − nominal),
    /// preserving the nominal shape.  Writes a new workspace JSON with smoothed templates.
    Smooth {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output workspace (pyhf JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Max relative variation cap (0 = disabled).
        /// Bins where |delta/nominal| > cap are clamped.
        #[arg(long, default_value = "0.0")]
        max_variation: f64,
    },

    /// Prune negligible systematics from a workspace.
    ///
    /// Removes HistoSys/NormSys modifiers whose max relative effect is below a
    /// threshold.  Writes a new workspace JSON with pruned modifiers.
    Prune {
        /// Input workspace (pyhf JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output workspace (pyhf JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Pruning threshold: modifiers with max |delta/nominal| < threshold are removed.
        #[arg(long, default_value = "0.005")]
        threshold: f64,
    },
}

#[derive(Subcommand)]
enum TrexCommands {
    /// Convert a TRExFitter `.config` file into an analysis spec v0 YAML + mapping report.
    ImportConfig {
        /// Path to TRExFitter `.config` file.
        #[arg(long)]
        config: PathBuf,

        /// Output analysis spec path (YAML).
        #[arg(long)]
        out: PathBuf,

        /// Output mapping report path (JSON). Defaults to `<out>.mapping.json`.
        #[arg(long)]
        report: Option<PathBuf>,

        /// Determinism threads in the generated spec.
        #[arg(long, default_value = "1")]
        threads: usize,

        /// `execution.import.output_json` path written into the generated spec.
        #[arg(long, default_value = "tmp/trex_workspace.json")]
        workspace_out: PathBuf,

        /// Python executable to use (default: `.venv/bin/python` if present, else `python3`).
        #[arg(long)]
        python: Option<PathBuf>,

        /// Overwrite existing output files.
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Print a JSON schema to stdout (for IDE integration / validation)
    Schema {
        /// Schema name (default: analysis_spec_v0). Known: analysis_spec_v0, baseline_v0,
        /// report_distributions_v0, report_pulls_v0, report_corr_v0, report_yields_v0, report_uncertainty_v0,
        /// validation_report_v1, unbinned_spec_v0.
        #[arg(long)]
        name: Option<String>,
    },
}

#[derive(Subcommand)]
enum TimeseriesCommands {
    /// Kalman filter (linear-Gaussian SSM)
    KalmanFilter {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// RTS smoother (runs filter + smoother)
    KalmanSmooth {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Fit Q/R with EM (holds F/H/m0/P0 fixed)
    KalmanEm {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Max EM iterations
        #[arg(long, default_value = "50")]
        max_iter: usize,

        /// Relative tolerance on log-likelihood improvement
        #[arg(long, default_value = "1e-6")]
        tol: f64,

        /// Update Q
        #[arg(long, default_value_t = true)]
        estimate_q: bool,

        /// Update R
        #[arg(long, default_value_t = true)]
        estimate_r: bool,

        /// Update F (currently only supports 1D: n_state=1, n_obs=1)
        #[arg(long, default_value_t = false)]
        estimate_f: bool,

        /// Update H (currently only supports 1D: n_state=1, n_obs=1)
        #[arg(long, default_value_t = false)]
        estimate_h: bool,

        /// Minimum diagonal value applied to Q/R
        #[arg(long, default_value = "1e-12")]
        min_diag: f64,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Fit with EM, then run RTS smoother (and optional forecast)
    KalmanFit {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Max EM iterations
        #[arg(long, default_value = "50")]
        max_iter: usize,

        /// Relative tolerance on log-likelihood improvement
        #[arg(long, default_value = "1e-6")]
        tol: f64,

        /// Update Q
        #[arg(long, default_value_t = true)]
        estimate_q: bool,

        /// Update R
        #[arg(long, default_value_t = true)]
        estimate_r: bool,

        /// Update F (currently only supports 1D: n_state=1, n_obs=1)
        #[arg(long, default_value_t = false)]
        estimate_f: bool,

        /// Update H (currently only supports 1D: n_state=1, n_obs=1)
        #[arg(long, default_value_t = false)]
        estimate_h: bool,

        /// Minimum diagonal value applied to Q/R
        #[arg(long, default_value = "1e-12")]
        min_diag: f64,

        /// Number of forecast steps (>0). If 0, omit forecast.
        #[arg(long, default_value = "0")]
        forecast_steps: usize,

        /// Skip running RTS smoother (only returns fitted model + EM trace)
        #[arg(long, default_value_t = false)]
        no_smooth: bool,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Plot-friendly artifact for smoothed states/observations (+ optional forecast bands)
    ///
    /// Runs EM, then Kalman smooth, and computes marginal normal bands using `--level`.
    KalmanViz {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Max EM iterations
        #[arg(long, default_value = "50")]
        max_iter: usize,

        /// Relative tolerance on log-likelihood improvement
        #[arg(long, default_value = "1e-6")]
        tol: f64,

        /// Update Q
        #[arg(long, default_value_t = true)]
        estimate_q: bool,

        /// Update R
        #[arg(long, default_value_t = true)]
        estimate_r: bool,

        /// Update F (currently only supports 1D: n_state=1, n_obs=1)
        #[arg(long, default_value_t = false)]
        estimate_f: bool,

        /// Update H (currently only supports 1D: n_state=1, n_obs=1)
        #[arg(long, default_value_t = false)]
        estimate_h: bool,

        /// Minimum diagonal value applied to Q/R
        #[arg(long, default_value = "1e-12")]
        min_diag: f64,

        /// Two-sided central credible level for normal bands (e.g. 0.95 -> 95%).
        #[arg(long, default_value = "0.95")]
        level: f64,

        /// Number of forecast steps (>0). If 0, omit forecast.
        #[arg(long, default_value = "0")]
        forecast_steps: usize,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Forecast K steps ahead after filtering the provided `ys`
    KalmanForecast {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Number of forecast steps (>0)
        #[arg(long, default_value = "1")]
        steps: usize,

        /// Optional two-sided alpha for marginal normal prediction intervals.
        ///
        /// Example: `--alpha 0.05` computes 95% intervals per observation dimension.
        #[arg(long)]
        alpha: Option<f64>,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Simulate (xs, ys) from the model in the input JSON
    KalmanSimulate {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Number of timesteps (>0)
        #[arg(long)]
        t_max: usize,

        /// RNG seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Fit a Gaussian GARCH(1,1) volatility model by MLE.
    Garch11Fit {
        /// Input JSON file (see docs/tutorials/phase-12-volatility.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Max optimizer iterations
        #[arg(long, default_value = "1000")]
        max_iter: u64,

        /// Convergence tolerance (gradient norm)
        #[arg(long, default_value = "1e-6")]
        tol: f64,

        /// Enforce alpha+beta < alpha_beta_max
        #[arg(long, default_value = "0.999")]
        alpha_beta_max: f64,

        /// Minimum conditional variance clamp
        #[arg(long, default_value = "1e-18")]
        min_var: f64,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Fit an approximate stochastic volatility model (log-chi2 + Kalman MLE).
    SvLogchi2Fit {
        /// Input JSON file (see docs/tutorials/phase-12-volatility.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Max optimizer iterations
        #[arg(long, default_value = "1000")]
        max_iter: u64,

        /// Convergence tolerance (gradient norm)
        #[arg(long, default_value = "1e-6")]
        tol: f64,

        /// epsilon added inside log(y^2 + eps)
        #[arg(long, default_value = "1e-12")]
        log_eps: f64,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum SurvivalCommands {
    /// Cox proportional hazards (partial likelihood) fit (with optional robust/cluster-robust SE)
    CoxPhFit {
        /// Input JSON file (see docs/tutorials/phase-9-survival.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ties approximation: breslow or efron.
        #[arg(long, default_value = "efron", value_parser = ["breslow", "efron"])]
        ties: String,

        /// Disable robust (sandwich) standard errors.
        #[arg(long, default_value_t = false)]
        no_robust: bool,

        /// Disable small-sample correction for clustered robust covariance (G/(G-1)).
        #[arg(long, default_value_t = false)]
        no_cluster_correction: bool,

        /// Disable baseline cumulative hazard output (Breslow/Efron increments).
        #[arg(long, default_value_t = false)]
        no_baseline: bool,
    },

    /// Kaplan-Meier survival estimate (non-parametric)
    Km {
        /// Input JSON file with "times" and "events" arrays.
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Confidence level for pointwise CIs (log-log transform).
        #[arg(long, default_value = "0.95")]
        conf_level: f64,
    },

    /// Log-rank (Mantel-Cox) test comparing survival across groups
    LogRankTest {
        /// Input JSON file with "times", "events", and "groups" arrays.
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for results (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Interval-censored Weibull fit
    IntervalWeibullFit {
        /// Input JSON: {"time_lower": [...], "time_upper": [...], "censor_type": ["exact","right","left","interval",...]}
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Interval-censored LogNormal fit
    IntervalLognormalFit {
        /// Input JSON: {"time_lower": [...], "time_upper": [...], "censor_type": [...]}
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Interval-censored Exponential fit
    IntervalExponentialFit {
        /// Input JSON: {"time_lower": [...], "time_upper": [...], "censor_type": [...]}
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ChurnCommands {
    /// Generate a synthetic SaaS churn dataset (deterministic, seeded).
    GenerateData {
        /// Number of customers.
        #[arg(long, default_value = "2000")]
        n_customers: usize,

        /// Number of signup cohorts.
        #[arg(long, default_value = "6")]
        n_cohorts: usize,

        /// Max observation window (months).
        #[arg(long, default_value = "24")]
        max_time: f64,

        /// Treatment fraction (0.0–1.0).
        #[arg(long, default_value = "0.3")]
        treatment_fraction: f64,

        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Cohort retention analysis: stratified KM + log-rank test.
    Retention {
        /// Input JSON file (output of generate-data, or custom with times/events/groups).
        #[arg(short, long)]
        input: PathBuf,

        /// Confidence level for KM CIs.
        #[arg(long, default_value = "0.95")]
        conf_level: f64,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Cox PH churn risk model: hazard ratios + CIs.
    RiskModel {
        /// Input JSON file (output of generate-data, or custom with times/events/covariates/names).
        #[arg(short, long)]
        input: PathBuf,

        /// Confidence level for HR CIs.
        #[arg(long, default_value = "0.95")]
        conf_level: f64,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Bootstrap hazard ratios (Rayon-parallel Cox PH refitting).
    BootstrapHr {
        /// Input JSON file (output of generate-data, or custom with times/events/covariates/names).
        #[arg(short, long)]
        input: PathBuf,

        /// Number of bootstrap resamples.
        #[arg(long, default_value = "1000")]
        n_bootstrap: usize,

        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Confidence level for bootstrap CIs.
        #[arg(long, default_value = "0.95")]
        conf_level: f64,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Causal uplift: AIPW estimate of intervention impact on churn.
    Uplift {
        /// Input JSON file (output of generate-data, or custom with times/events/treated/covariates).
        #[arg(short, long)]
        input: PathBuf,

        /// Evaluation horizon (months).
        #[arg(long, default_value = "12")]
        horizon: f64,

        /// Output file (pretty JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Ingest real customer data from Parquet or CSV into churn analysis JSON.
    Ingest {
        /// Input file (Parquet or CSV).
        #[arg(short, long)]
        input: PathBuf,

        /// Column mapping YAML file.
        #[arg(short, long)]
        mapping: PathBuf,

        /// Output file (churn JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Cohort retention matrix: life-table per cohort with period retention rates.
    CohortMatrix {
        /// Input JSON file (output of generate-data or ingest).
        #[arg(short, long)]
        input: PathBuf,

        /// Period boundaries (comma-separated, e.g. "1,3,6,12,24").
        /// Defaults to "1,2,3,6,9,12,18,24".
        #[arg(long, default_value = "1,2,3,6,9,12,18,24")]
        periods: String,

        /// Output directory for artifacts (JSON + CSV). Defaults to stdout JSON only.
        #[arg(long)]
        out_dir: Option<PathBuf>,

        /// Output file (JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Segment comparison report: pairwise log-rank + effect sizes + MCP.
    Compare {
        /// Input JSON file (output of generate-data or ingest).
        #[arg(short, long)]
        input: PathBuf,

        /// Multiple comparisons correction: bonferroni, benjamini_hochberg, none.
        #[arg(long, default_value = "benjamini_hochberg")]
        correction: String,

        /// Significance level.
        #[arg(long, default_value = "0.05")]
        alpha: f64,

        /// Confidence level for KM CIs.
        #[arg(long, default_value = "0.95")]
        conf_level: f64,

        /// Output directory for artifacts (JSON + CSV).
        #[arg(long)]
        out_dir: Option<PathBuf>,

        /// Output file (JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Survival-native causal uplift: RMST, IPW-weighted KM, ΔS(t).
    UpliftSurvival {
        /// Input JSON file (output of generate-data or ingest).
        #[arg(short, long)]
        input: PathBuf,

        /// RMST integration horizon (months).
        #[arg(long, default_value = "12.0")]
        horizon: f64,

        /// Evaluation horizons for ΔS(t), comma-separated (e.g. "3,6,12,24").
        #[arg(long, default_value = "1,3,6,12")]
        eval_horizons: String,

        /// Propensity score trimming threshold.
        #[arg(long, default_value = "0.01")]
        trim: f64,

        /// Output directory for artifacts (JSON + CSV).
        #[arg(long)]
        out_dir: Option<PathBuf>,

        /// Output file (JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Diagnostics / guardrails: overlap, balance, censoring, trust gate.
    Diagnostics {
        /// Input JSON file (output of generate-data or ingest).
        #[arg(short, long)]
        input: PathBuf,

        /// Propensity score trimming threshold.
        #[arg(long, default_value = "0.01")]
        trim: f64,

        /// Covariate names (comma-separated). Optional.
        #[arg(long)]
        covariate_names: Option<String>,

        /// Output directory for artifacts (JSON + CSV).
        #[arg(long)]
        out_dir: Option<PathBuf>,

        /// Output file (JSON). Defaults to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let Cli { log_level, bundle, interp_defaults, command } = Cli::parse();

    tracing_subscriber::fmt().with_max_level(log_level).with_target(false).init();

    match command {
        Commands::Run { config } => cmd_run(&config, bundle.as_ref(), interp_defaults),
        Commands::Validate { config } => cmd_validate(&config),
        Commands::BuildHists {
            config,
            base_dir,
            out_dir,
            overwrite,
            coverage_json,
            expr_coverage_json,
        } => cmd_build_hists(
            &config,
            base_dir.as_ref(),
            &out_dir,
            overwrite,
            coverage_json.as_ref(),
            expr_coverage_json.as_ref(),
        ),
        Commands::Fit {
            input,
            output,
            json_metrics,
            threads,
            gpu,
            parity,
            fit_regions,
            validation_regions,
            asimov,
        } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            eprintln!("GPU mode enabled (CUDA single-model)");
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(feature = "metal")]
                        {
                            eprintln!("GPU mode enabled (Metal single-model, f32)");
                        }
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            if parity {
                eprintln!("Parity mode: Kahan summation, threads=1, Accelerate disabled");
            }
            cmd_fit(
                &input,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                interp_defaults,
                gpu.as_deref(),
                bundle.as_ref(),
                parity,
                &fit_regions,
                &validation_regions,
                asimov,
            )
        }
        Commands::UnbinnedFit {
            config,
            output,
            json_metrics,
            threads,
            gpu,
            opt_max_iter,
            opt_tol,
            opt_m,
            opt_smooth_bounds,
        } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            eprintln!("GPU mode enabled (unbinned CUDA, f64)");
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(feature = "metal")]
                        {
                            eprintln!("GPU mode enabled (unbinned Metal, f32)");
                        }
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            cmd_unbinned_fit(
                &config,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                gpu.as_deref(),
                bundle.as_ref(),
                opt_max_iter,
                opt_tol,
                opt_m,
                opt_smooth_bounds,
            )
        }
        Commands::HybridFit { binned, unbinned, output, json_metrics, threads } => cmd_hybrid_fit(
            &binned,
            &unbinned,
            output.as_ref(),
            json_metrics.as_ref(),
            threads,
            interp_defaults,
            bundle.as_ref(),
        ),
        Commands::UnbinnedScan {
            config,
            start,
            stop,
            points,
            output,
            json_metrics,
            threads,
            gpu,
        } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            cmd_unbinned_scan(
                &config,
                start,
                stop,
                points,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                gpu.as_deref(),
                bundle.as_ref(),
            )
        }
        Commands::UnbinnedFitToys {
            config,
            n_toys,
            seed,
            gen_point,
            set,
            output,
            json_metrics,
            threads,
            gpu,
            gpu_devices,
            gpu_shards,
            gpu_sample_toys,
            gpu_native,
            shard,
            require_all_converged,
            max_abs_poi_pull_mean,
            poi_pull_std_range,
        } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            if shard.is_some() && gpu.is_some() {
                anyhow::bail!("--shard is for CPU farming only; do not combine with --gpu");
            }
            cmd_unbinned_fit_toys(
                &config,
                n_toys,
                seed,
                gen_point,
                &set,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                gpu.as_deref(),
                &gpu_devices,
                gpu_shards,
                gpu_sample_toys,
                gpu_native,
                shard,
                require_all_converged,
                max_abs_poi_pull_mean,
                poi_pull_std_range,
                bundle.as_ref(),
            )
        }
        Commands::UnbinnedMergeToys { inputs, output } => {
            cmd_unbinned_merge_toys(&inputs, output.as_ref())
        }
        Commands::UnbinnedRanking { config, output, json_metrics, threads } => {
            cmd_unbinned_ranking(
                &config,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                bundle.as_ref(),
            )
        }
        Commands::UnbinnedHypotest { config, mu, output, json_metrics, threads, gpu } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            cmd_unbinned_hypotest(
                &config,
                mu,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                gpu.as_deref(),
                bundle.as_ref(),
            )
        }
        Commands::UnbinnedHypotestToys {
            config,
            mu,
            n_toys,
            seed,
            expected_set,
            output,
            json_metrics,
            threads,
            gpu,
            gpu_devices,
            gpu_shards,
            gpu_sample_toys,
        } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            cmd_unbinned_hypotest_toys(
                &config,
                mu,
                n_toys,
                seed,
                expected_set,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                gpu.as_deref(),
                &gpu_devices,
                gpu_shards,
                gpu_sample_toys,
                bundle.as_ref(),
            )
        }
        Commands::Audit { input, format, output } => cmd_audit(&input, &format, output.as_ref()),
        Commands::Hypotest { input, mu, expected_set, output, json_metrics, threads } => {
            cmd_hypotest(
                &input,
                mu,
                expected_set,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                interp_defaults,
                bundle.as_ref(),
            )
        }
        Commands::Significance { input, output, json_metrics, threads } => cmd_significance(
            &input,
            output.as_ref(),
            json_metrics.as_ref(),
            threads,
            interp_defaults,
            bundle.as_ref(),
        ),
        Commands::GoodnessOfFit { input, output, json_metrics, threads } => cmd_goodness_of_fit(
            &input,
            output.as_ref(),
            json_metrics.as_ref(),
            threads,
            interp_defaults,
            bundle.as_ref(),
        ),
        Commands::Combine { inputs, output, prefix_channels } => {
            cmd_combine(&inputs, output.as_ref(), prefix_channels)
        }
        Commands::HypotestToys { input, mu, n_toys, seed, expected_set, output, threads, gpu } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            eprintln!("GPU mode enabled (CUDA batch)");
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(feature = "metal")]
                        {
                            eprintln!("GPU mode enabled (Metal batch, f32)");
                        }
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            cmd_hypotest_toys(
                &input,
                mu,
                n_toys,
                seed,
                expected_set,
                output.as_ref(),
                threads,
                interp_defaults,
                gpu.as_deref(),
                bundle.as_ref(),
            )
        }
        Commands::UpperLimit {
            input,
            alpha,
            expected,
            scan_start,
            scan_stop,
            scan_points,
            lo,
            hi,
            rtol,
            max_iter,
            output,
            json_metrics,
            threads,
        } => cmd_upper_limit(
            &input,
            alpha,
            expected,
            scan_start,
            scan_stop,
            scan_points,
            lo,
            hi,
            rtol,
            max_iter,
            output.as_ref(),
            json_metrics.as_ref(),
            threads,
            interp_defaults,
            bundle.as_ref(),
        ),
        Commands::MassScan {
            workspaces_dir,
            alpha,
            scan_start,
            scan_stop,
            scan_points,
            labels,
            output,
            json_metrics,
            threads,
        } => cmd_mass_scan(
            &workspaces_dir,
            alpha,
            scan_start,
            scan_stop,
            scan_points,
            &labels,
            output.as_ref(),
            json_metrics.as_ref(),
            threads,
            interp_defaults,
            bundle.as_ref(),
        ),
        Commands::Scan { input, start, stop, points, output, json_metrics, threads, gpu } => {
            if let Some(ref dev) = gpu {
                match dev.as_str() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            eprintln!("GPU mode enabled (CUDA profile scan)");
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            anyhow::bail!("--gpu cuda requires building with --features cuda");
                        }
                    }
                    "metal" => {
                        #[cfg(feature = "metal")]
                        {
                            eprintln!("GPU mode enabled (Metal profile scan, f32)");
                        }
                        #[cfg(not(feature = "metal"))]
                        {
                            anyhow::bail!("--gpu metal requires building with --features metal");
                        }
                    }
                    other => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
                }
            }
            cmd_scan(
                &input,
                start,
                stop,
                points,
                output.as_ref(),
                json_metrics.as_ref(),
                threads,
                interp_defaults,
                gpu.as_deref(),
                bundle.as_ref(),
            )
        }
        Commands::Report {
            input,
            histfactory_xml,
            fit,
            out_dir,
            overwrite,
            include_covariance,
            render,
            pdf,
            svg_dir,
            python,
            skip_uncertainty,
            uncertainty_grouping,
            threads,
            deterministic,
            blind_regions,
        } => cmd_report(
            &input,
            &histfactory_xml,
            fit.as_ref(),
            &out_dir,
            overwrite,
            include_covariance,
            render,
            pdf.as_ref(),
            svg_dir.as_ref(),
            python.as_ref(),
            skip_uncertainty,
            uncertainty_grouping.as_str(),
            threads,
            interp_defaults,
            deterministic,
            &blind_regions,
        ),
        Commands::ValidationReport { apex2, workspace, out, pdf, python, deterministic } => {
            validation_report::cmd_validation_report(
                &apex2,
                &workspace,
                &out,
                pdf.as_ref(),
                python.as_ref(),
                deterministic,
                interp_defaults,
            )
        }
        Commands::Viz { command } => match command {
            VizCommands::Profile { input, start, stop, points, output, threads } => {
                cmd_viz_profile(
                    &input,
                    start,
                    stop,
                    points,
                    output.as_ref(),
                    threads,
                    interp_defaults,
                    bundle.as_ref(),
                )
            }
            VizCommands::Cls {
                input,
                alpha,
                scan_start,
                scan_stop,
                scan_points,
                output,
                threads,
            } => cmd_viz_cls(
                &input,
                alpha,
                scan_start,
                scan_stop,
                scan_points,
                output.as_ref(),
                threads,
                interp_defaults,
                bundle.as_ref(),
            ),
            VizCommands::Ranking { input, output, threads } => {
                cmd_viz_ranking(&input, output.as_ref(), threads, interp_defaults, bundle.as_ref())
            }
            VizCommands::Pulls { input, fit, output, threads } => cmd_viz_pulls(
                &input,
                &fit,
                output.as_ref(),
                threads,
                interp_defaults,
                bundle.as_ref(),
            ),
            VizCommands::Gammas { input, fit, output, threads } => cmd_viz_gammas(
                &input,
                &fit,
                output.as_ref(),
                threads,
                interp_defaults,
                bundle.as_ref(),
            ),
            VizCommands::Corr { input, fit, include_covariance, output, threads } => cmd_viz_corr(
                &input,
                &fit,
                include_covariance,
                output.as_ref(),
                threads,
                interp_defaults,
                bundle.as_ref(),
            ),
            VizCommands::Distributions { input, histfactory_xml, fit, output, threads } => {
                cmd_viz_distributions(
                    &input,
                    &histfactory_xml,
                    fit.as_ref(),
                    output.as_ref(),
                    threads,
                    interp_defaults,
                    bundle.as_ref(),
                )
            }
            VizCommands::Separation { input, signal_samples, histfactory_xml, output, threads } => {
                cmd_viz_separation(
                    &input,
                    &signal_samples,
                    histfactory_xml.as_ref(),
                    output.as_ref(),
                    threads,
                    interp_defaults,
                    bundle.as_ref(),
                )
            }
            VizCommands::Summary { fits, labels, output } => {
                cmd_viz_summary(&fits, &labels, output.as_ref(), bundle.as_ref())
            }
            VizCommands::Pie { input, fit, output, threads } => cmd_viz_pie(
                &input,
                fit.as_ref(),
                output.as_ref(),
                threads,
                interp_defaults,
                bundle.as_ref(),
            ),
        },
        Commands::Import { command } => match command {
            ImportCommands::Histfactory { xml, dir, basedir, output } => {
                let xml_path = if let Some(xml) = xml {
                    xml
                } else {
                    let dir = dir.expect("clap enforces: --xml or --dir");
                    discover::discover_single_combination_xml(&dir)?
                };
                cmd_import_histfactory(
                    &xml_path,
                    basedir.as_deref(),
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
            ImportCommands::TrexConfig {
                config,
                base_dir,
                output,
                analysis_yaml,
                coverage_json,
                expr_coverage_json,
            } => cmd_import_trex_config(
                &config,
                base_dir.as_ref(),
                output.as_ref(),
                analysis_yaml.as_ref(),
                coverage_json.as_ref(),
                expr_coverage_json.as_ref(),
                bundle.as_ref(),
            ),
            ImportCommands::Patchset { workspace, patchset, patch_name, output } => {
                cmd_import_patchset(
                    &workspace,
                    &patchset,
                    patch_name.as_deref(),
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
        },
        Commands::Trex { command } => match command {
            TrexCommands::ImportConfig {
                config,
                out,
                report,
                threads,
                workspace_out,
                python,
                overwrite,
            } => cmd_trex_import_config(
                &config,
                &out,
                report.as_ref(),
                threads,
                &workspace_out,
                python.as_ref(),
                overwrite,
            ),
        },
        Commands::Preprocess { command } => match command {
            PreprocessCommands::Smooth { input, output, max_variation } => {
                cmd_preprocess_smooth(&input, output.as_ref(), max_variation)
            }
            PreprocessCommands::Prune { input, output, threshold } => {
                cmd_preprocess_prune(&input, output.as_ref(), threshold)
            }
        },
        Commands::Export { command } => match command {
            ExportCommands::Histfactory { input, out_dir, overwrite, prefix, python } => {
                cmd_export_histfactory(
                    &input,
                    &out_dir,
                    overwrite,
                    prefix.as_str(),
                    python.as_ref(),
                )
            }
        },
        Commands::Timeseries { command } => match command {
            TimeseriesCommands::KalmanFilter { input, output } => {
                cmd_ts_kalman_filter(&input, output.as_ref(), bundle.as_ref())
            }
            TimeseriesCommands::KalmanSmooth { input, output } => {
                cmd_ts_kalman_smooth(&input, output.as_ref(), bundle.as_ref())
            }
            TimeseriesCommands::KalmanEm {
                input,
                max_iter,
                tol,
                estimate_q,
                estimate_r,
                estimate_f,
                estimate_h,
                min_diag,
                output,
            } => cmd_ts_kalman_em(
                &input,
                max_iter,
                tol,
                estimate_q,
                estimate_r,
                estimate_f,
                estimate_h,
                min_diag,
                output.as_ref(),
                bundle.as_ref(),
            ),
            TimeseriesCommands::KalmanFit {
                input,
                max_iter,
                tol,
                estimate_q,
                estimate_r,
                estimate_f,
                estimate_h,
                min_diag,
                forecast_steps,
                no_smooth,
                output,
            } => cmd_ts_kalman_fit(
                &input,
                max_iter,
                tol,
                estimate_q,
                estimate_r,
                estimate_f,
                estimate_h,
                min_diag,
                forecast_steps,
                no_smooth,
                output.as_ref(),
                bundle.as_ref(),
            ),
            TimeseriesCommands::KalmanViz {
                input,
                max_iter,
                tol,
                estimate_q,
                estimate_r,
                estimate_f,
                estimate_h,
                min_diag,
                level,
                forecast_steps,
                output,
            } => cmd_ts_kalman_viz(
                &input,
                max_iter,
                tol,
                estimate_q,
                estimate_r,
                estimate_f,
                estimate_h,
                min_diag,
                level,
                forecast_steps,
                output.as_ref(),
                bundle.as_ref(),
            ),
            TimeseriesCommands::KalmanForecast { input, steps, alpha, output } => {
                cmd_ts_kalman_forecast(&input, steps, alpha, output.as_ref(), bundle.as_ref())
            }
            TimeseriesCommands::KalmanSimulate { input, t_max, seed, output } => {
                cmd_ts_kalman_simulate(&input, t_max, seed, output.as_ref(), bundle.as_ref())
            }
            TimeseriesCommands::Garch11Fit {
                input,
                max_iter,
                tol,
                alpha_beta_max,
                min_var,
                output,
            } => cmd_ts_garch11_fit(
                &input,
                max_iter,
                tol,
                alpha_beta_max,
                min_var,
                output.as_ref(),
                bundle.as_ref(),
            ),
            TimeseriesCommands::SvLogchi2Fit { input, max_iter, tol, log_eps, output } => {
                cmd_ts_sv_logchi2_fit(
                    &input,
                    max_iter,
                    tol,
                    log_eps,
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
        },
        Commands::Survival { command } => match command {
            SurvivalCommands::CoxPhFit {
                input,
                output,
                ties,
                no_robust,
                no_cluster_correction,
                no_baseline,
            } => survival::cmd_survival_cox_ph_fit(
                &input,
                output.as_ref(),
                ties.as_str(),
                /*robust*/ !no_robust,
                /*cluster_correction*/ !no_cluster_correction,
                /*baseline*/ !no_baseline,
                bundle.as_ref(),
            ),
            SurvivalCommands::Km { input, output, conf_level } => {
                survival::cmd_survival_km(&input, output.as_ref(), conf_level, bundle.as_ref())
            }
            SurvivalCommands::LogRankTest { input, output } => {
                survival::cmd_survival_log_rank(&input, output.as_ref(), bundle.as_ref())
            }
            SurvivalCommands::IntervalWeibullFit { input, output } => {
                survival::cmd_interval_weibull_fit(&input, output.as_ref(), bundle.as_ref())
            }
            SurvivalCommands::IntervalLognormalFit { input, output } => {
                survival::cmd_interval_lognormal_fit(&input, output.as_ref(), bundle.as_ref())
            }
            SurvivalCommands::IntervalExponentialFit { input, output } => {
                survival::cmd_interval_exponential_fit(&input, output.as_ref(), bundle.as_ref())
            }
        },
        Commands::FaultTreeMc { config, n_scenarios, seed, device, chunk_size, output } => {
            let spec_json = std::fs::read_to_string(&config)?;
            let spec: ns_inference::FaultTreeSpec = serde_json::from_str(&spec_json)?;

            let result = match device.as_str() {
                "cpu" => ns_inference::fault_tree_mc_cpu(&spec, n_scenarios, seed, chunk_size)?,
                #[cfg(feature = "cuda")]
                "cuda" => ns_inference::fault_tree_mc_cuda(&spec, n_scenarios, seed, chunk_size)?,
                other => anyhow::bail!("unsupported device: '{other}'"),
            };

            let json = serde_json::to_string_pretty(&result)?;
            if let Some(path) = &output {
                std::fs::write(path, &json)?;
            } else {
                println!("{json}");
            }
            Ok(())
        }
        Commands::Churn { command } => match command {
            ChurnCommands::GenerateData {
                n_customers,
                n_cohorts,
                max_time,
                treatment_fraction,
                seed,
                output,
            } => churn::cmd_churn_generate_data(
                n_customers,
                n_cohorts,
                max_time,
                treatment_fraction,
                seed,
                output.as_ref(),
                bundle.as_ref(),
            ),
            ChurnCommands::Retention { input, conf_level, output } => {
                churn::cmd_churn_retention(&input, conf_level, output.as_ref(), bundle.as_ref())
            }
            ChurnCommands::RiskModel { input, conf_level, output } => {
                churn::cmd_churn_risk_model(&input, conf_level, output.as_ref(), bundle.as_ref())
            }
            ChurnCommands::BootstrapHr { input, n_bootstrap, seed, conf_level, output } => {
                churn::cmd_churn_bootstrap_hr(
                    &input,
                    n_bootstrap,
                    seed,
                    conf_level,
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
            ChurnCommands::Uplift { input, horizon, output } => {
                churn::cmd_churn_uplift(&input, horizon, output.as_ref(), bundle.as_ref())
            }
            ChurnCommands::Ingest { input, mapping, output } => {
                churn::cmd_churn_ingest(&input, &mapping, output.as_ref(), bundle.as_ref())
            }
            ChurnCommands::CohortMatrix { input, periods, out_dir, output } => {
                churn::cmd_churn_cohort_matrix(
                    &input,
                    &periods,
                    out_dir.as_ref(),
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
            ChurnCommands::Compare { input, correction, alpha, conf_level, out_dir, output } => {
                churn::cmd_churn_compare(
                    &input,
                    &correction,
                    alpha,
                    conf_level,
                    out_dir.as_ref(),
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
            ChurnCommands::UpliftSurvival {
                input,
                horizon,
                eval_horizons,
                trim,
                out_dir,
                output,
            } => churn::cmd_churn_uplift_survival(
                &input,
                horizon,
                &eval_horizons,
                trim,
                out_dir.as_ref(),
                output.as_ref(),
                bundle.as_ref(),
            ),
            ChurnCommands::Diagnostics { input, trim, covariate_names, out_dir, output } => {
                churn::cmd_churn_diagnostics(
                    &input,
                    trim,
                    covariate_names.as_deref(),
                    out_dir.as_ref(),
                    output.as_ref(),
                    bundle.as_ref(),
                )
            }
        },
        Commands::Convert { input, tree, output, observables, selection, weight, max_events } => {
            let obs: Vec<convert::ObsArg> =
                observables.iter().map(|s| convert::ObsArg::parse(s)).collect::<Result<_>>()?;
            convert::cmd_convert(
                &input,
                &tree,
                &output,
                &obs,
                selection.as_deref(),
                weight.as_deref(),
                max_events,
            )
        }
        Commands::Config { command } => match command {
            ConfigCommands::Schema { name } => cmd_config_schema(name.as_deref()),
        },
        Commands::Version => {
            println!("nextstat {}", ns_core::VERSION);
            Ok(())
        }
    }
}

fn cmd_run(
    config_path: &PathBuf,
    bundle: Option<&PathBuf>,
    interp_defaults: InterpDefaults,
) -> Result<()> {
    match analysis_spec::read_any_run_config(config_path.as_path())? {
        analysis_spec::AnyRunConfig::Legacy(cfg) => {
            cmd_run_legacy(config_path, bundle, interp_defaults, &cfg)
        }
        analysis_spec::AnyRunConfig::SpecV0(spec) => {
            cmd_run_spec_v0(config_path, bundle, interp_defaults, &spec)
        }
    }
}

fn cmd_validate(config_path: &PathBuf) -> Result<()> {
    match analysis_spec::read_any_run_config(config_path.as_path())? {
        analysis_spec::AnyRunConfig::Legacy(cfg) => {
            if !cfg.histfactory_xml.is_file() {
                anyhow::bail!("histfactory_xml not found: {}", cfg.histfactory_xml.display());
            }
            let summary = serde_json::json!({
                "config_type": "run_config_legacy",
                "histfactory_xml": cfg.histfactory_xml,
                "out_dir": cfg.out_dir,
                "threads": cfg.threads,
                "deterministic": cfg.deterministic,
                "include_covariance": cfg.include_covariance,
                "skip_uncertainty": cfg.skip_uncertainty,
                "uncertainty_grouping": cfg.uncertainty_grouping,
                "render": {
                    "enabled": cfg.render,
                    "pdf": cfg.pdf,
                    "svg_dir": cfg.svg_dir,
                    "python": cfg.python,
                },
            });
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        analysis_spec::AnyRunConfig::SpecV0(spec) => {
            let plan = spec.to_run_plan(config_path.as_path())?;
            let cfg_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
            let baseline_gate = &spec.gates.baseline_compare;
            let baseline_manifest = if baseline_gate.enabled {
                let baseline_dir = if baseline_gate.baseline_dir.is_absolute() {
                    baseline_gate.baseline_dir.clone()
                } else {
                    cfg_dir.join(&baseline_gate.baseline_dir)
                };
                Some(baseline_dir.join("latest_trex_analysis_spec_manifest.json"))
            } else {
                None
            };
            if let Some(p) = baseline_manifest.as_ref()
                && !p.is_file()
            {
                anyhow::bail!(
                    "analysis spec baseline_compare gate is enabled, but manifest is missing: {}",
                    p.display()
                );
            }

            let summary = serde_json::json!({
                "config_type": "analysis_spec_v0",
                "schema_version": spec.schema_version,
                "inputs": {
                    "mode": spec.inputs.mode,
                },
                "plan": {
                    "threads": plan.threads,
                    "workspace_json": plan.workspace_json,
                    "import": plan.import.as_ref().map(|_| true).unwrap_or(false),
                    "fit": plan.fit.is_some(),
                    "profile_scan": plan.profile_scan.is_some(),
                    "report": plan.report.is_some(),
                },
                "gates": {
                    "baseline_compare": {
                        "enabled": baseline_gate.enabled,
                        "baseline_dir": baseline_gate.baseline_dir,
                        "require_same_host": baseline_gate.require_same_host,
                        "max_slowdown": baseline_gate.max_slowdown,
                        "manifest": baseline_manifest,
                    }
                }
            });
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
    }
}

fn cmd_config_schema(name: Option<&str>) -> Result<()> {
    let key = name.unwrap_or("analysis_spec_v0");
    let schema = match key {
        "analysis_spec_v0" => SCHEMA_ANALYSIS_SPEC_V0,
        "baseline_v0" => SCHEMA_BASELINE_V0,
        "report_distributions_v0" => SCHEMA_REPORT_DISTRIBUTIONS_V0,
        "report_pulls_v0" => SCHEMA_REPORT_PULLS_V0,
        "report_corr_v0" => SCHEMA_REPORT_CORR_V0,
        "report_yields_v0" => SCHEMA_REPORT_YIELDS_V0,
        "report_uncertainty_v0" => SCHEMA_REPORT_UNCERTAINTY_V0,
        "validation_report_v1" => SCHEMA_VALIDATION_REPORT_V1,
        "unbinned_spec_v0" => SCHEMA_UNBINNED_SPEC_V0,
        other => anyhow::bail!(
            "unknown schema name: {other}. Known: analysis_spec_v0, baseline_v0, report_distributions_v0, report_pulls_v0, report_corr_v0, report_yields_v0, report_uncertainty_v0, validation_report_v1, unbinned_spec_v0"
        ),
    };

    print!("{schema}");
    Ok(())
}

fn default_python_executable() -> PathBuf {
    let venv = PathBuf::from(".venv/bin/python");
    if venv.exists() { venv } else { PathBuf::from("python3") }
}

fn discover_repo_root_from_cwd() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..10 {
        if dir.join("tests").is_dir() && dir.join("docs").is_dir() {
            return Some(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn build_repo_pythonpath(repo_root: Option<&Path>) -> std::ffi::OsString {
    let mut pythonpath = std::ffi::OsString::new();
    if let Some(root) = repo_root {
        let cand = root.join("bindings/ns-py/python");
        if cand.join("nextstat").join("__init__.py").is_file() {
            pythonpath.push(cand.as_os_str());
        }
    }
    if let Some(existing) = std::env::var_os("PYTHONPATH")
        && !existing.is_empty()
    {
        if !pythonpath.is_empty() {
            pythonpath.push(":");
        }
        pythonpath.push(existing);
    }
    pythonpath
}

fn cmd_trex_import_config(
    config: &PathBuf,
    out: &PathBuf,
    report: Option<&PathBuf>,
    threads: usize,
    workspace_out: &PathBuf,
    python: Option<&PathBuf>,
    overwrite: bool,
) -> Result<()> {
    let python = python.cloned().unwrap_or_else(default_python_executable);
    let repo_root = discover_repo_root_from_cwd();
    let pythonpath = build_repo_pythonpath(repo_root.as_deref());

    let mut cmd = Command::new(&python);
    cmd.env("PYTHONPATH", pythonpath)
        .arg("-m")
        .arg("nextstat.trex_config.cli")
        .arg("import-config")
        .arg("--config")
        .arg(config)
        .arg("--out")
        .arg(out)
        .arg("--threads")
        .arg(threads.to_string())
        .arg("--workspace-out")
        .arg(workspace_out);

    if let Some(r) = report {
        cmd.arg("--report").arg(r);
    }
    if overwrite {
        cmd.arg("--overwrite");
    }

    let outp = cmd.output().map_err(|e| {
        anyhow::anyhow!(
            "failed to run trex import-config helper (python={}): {}",
            python.display(),
            e
        )
    })?;
    if !outp.status.success() {
        anyhow::bail!(
            "trex import-config helper failed (python={}, status={}):\nstdout:\n{}\nstderr:\n{}",
            python.display(),
            outp.status,
            String::from_utf8_lossy(&outp.stdout),
            String::from_utf8_lossy(&outp.stderr)
        );
    }

    // Pass through the helper's machine-readable JSON summary.
    print!("{}", String::from_utf8_lossy(&outp.stdout));
    Ok(())
}

fn run_analysis_spec_baseline_compare_gate(
    config_path: &Path,
    spec: &analysis_spec::AnalysisSpecV0,
) -> Result<Option<PathBuf>> {
    let gate = &spec.gates.baseline_compare;
    if !gate.enabled {
        return Ok(None);
    }

    let cfg_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let baseline_dir = if gate.baseline_dir.is_absolute() {
        gate.baseline_dir.clone()
    } else {
        cfg_dir.join(&gate.baseline_dir)
    };
    let manifest_path = baseline_dir.join("latest_trex_analysis_spec_manifest.json");
    if !manifest_path.is_file() {
        anyhow::bail!(
            "analysis spec baseline_compare gate is enabled, but manifest is missing: {}",
            manifest_path.display()
        );
    }

    let repo_root = discover_repo_root_from_cwd().ok_or_else(|| {
        anyhow::anyhow!(
            "baseline_compare gate requires repository tools under tests/, but repo root was not found from current directory"
        )
    })?;
    let compare_script = repo_root.join("tests/compare_trex_analysis_spec_with_latest_baseline.py");
    let schema_path = repo_root.join("docs/schemas/trex/analysis_spec_v0.schema.json");
    if !compare_script.is_file() {
        anyhow::bail!("baseline_compare helper script is missing: {}", compare_script.display());
    }
    if !schema_path.is_file() {
        anyhow::bail!("analysis spec schema is missing: {}", schema_path.display());
    }

    let report_path = baseline_dir
        .parent()
        .unwrap_or(baseline_dir.as_path())
        .join("trex_analysis_spec_compare_report.json");
    if let Some(parent) = report_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let python = default_python_executable();
    let pythonpath = build_repo_pythonpath(Some(repo_root.as_path()));
    let mut cmd = Command::new(&python);
    cmd.env("PYTHONPATH", pythonpath)
        .arg(&compare_script)
        .arg("--manifest")
        .arg(&manifest_path)
        .arg("--schema")
        .arg(&schema_path)
        .arg("--out")
        .arg(&report_path);
    if gate.require_same_host {
        cmd.arg("--require-same-host");
    }

    let outp = cmd.output().map_err(|e| {
        anyhow::anyhow!(
            "failed to run baseline_compare gate helper (python={}, script={}): {}",
            python.display(),
            compare_script.display(),
            e
        )
    })?;
    if !outp.status.success() {
        anyhow::bail!(
            "analysis spec baseline_compare gate failed (status={}). Report: {}\nstdout:\n{}\nstderr:\n{}",
            outp.status,
            report_path.display(),
            String::from_utf8_lossy(&outp.stdout),
            String::from_utf8_lossy(&outp.stderr)
        );
    }

    tracing::info!(
        report = %report_path.display(),
        "analysis spec baseline_compare gate passed"
    );
    Ok(Some(report_path))
}

fn cmd_run_legacy(
    config_path: &PathBuf,
    bundle: Option<&PathBuf>,
    interp_defaults: InterpDefaults,
    cfg: &run::RunConfig,
) -> Result<()> {
    setup_runtime(cfg.threads, false);
    let paths = run::derive_paths(&cfg.out_dir);

    ensure_out_dir(&paths.out_dir, cfg.overwrite)?;
    std::fs::create_dir_all(&paths.inputs_dir)?;
    std::fs::create_dir_all(&paths.artifacts_dir)?;

    tracing::info!(path = %cfg.histfactory_xml.display(), "importing HistFactory export");
    let ws = ns_translate::histfactory::from_xml(&cfg.histfactory_xml)?;
    write_json_file(&paths.workspace_json, &serde_json::to_value(&ws)?)?;

    cmd_report(
        &paths.workspace_json,
        &cfg.histfactory_xml,
        None,
        &paths.artifacts_dir,
        /*overwrite*/ true,
        cfg.include_covariance,
        cfg.render,
        cfg.pdf.as_ref(),
        cfg.svg_dir.as_ref(),
        cfg.python.as_ref(),
        cfg.skip_uncertainty,
        cfg.uncertainty_grouping.as_str(),
        cfg.threads,
        interp_defaults,
        cfg.deterministic,
        &[],
    )?;

    if let Some(dir) = bundle {
        run::write_run_bundle(dir, config_path, cfg)?;
    }

    let summary = serde_json::json!({
        "out_dir": paths.out_dir,
        "inputs": {
            "histfactory_xml": cfg.histfactory_xml,
            "workspace_json": paths.workspace_json,
        },
        "artifacts_dir": paths.artifacts_dir,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn cmd_run_spec_v0(
    config_path: &PathBuf,
    bundle: Option<&PathBuf>,
    interp_defaults: InterpDefaults,
    spec: &analysis_spec::AnalysisSpecV0,
) -> Result<()> {
    let plan = spec.to_run_plan(config_path.as_path())?;
    let deterministic = plan.threads == 1;
    setup_runtime(plan.threads, false);

    if let Some(import) = plan.import.as_ref() {
        if let Some(parent) = plan.workspace_json.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match import {
            analysis_spec::ImportPlan::HistfactoryXml { histfactory_xml } => {
                tracing::info!(path = %histfactory_xml.display(), "importing HistFactory export");
                let ws = ns_translate::histfactory::from_xml(histfactory_xml)?;
                write_json_file(&plan.workspace_json, &serde_json::to_value(&ws)?)?;
            }
            analysis_spec::ImportPlan::TrexConfigTxt { config_path, base_dir } => {
                tracing::info!(path = %config_path.display(), "importing TRExFitter config");
                let text = std::fs::read_to_string(config_path)?;
                let ws = ns_translate::trex::workspace_from_str(&text, base_dir)?;
                write_json_file(&plan.workspace_json, &serde_json::to_value(&ws)?)?;
            }
            analysis_spec::ImportPlan::TrexConfigYaml { config_text, base_dir } => {
                tracing::info!(path = %base_dir.display(), "importing TREx YAML config");
                let ws = ns_translate::trex::workspace_from_str(config_text, base_dir)?;
                write_json_file(&plan.workspace_json, &serde_json::to_value(&ws)?)?;
            }
        }
    }

    // Preprocessing (Python subprocess)
    if let Some(preprocess) = plan.preprocess.as_ref() {
        // Write preprocessing config JSON from the spec's steps array.
        let pp_cfg = &spec.execution.preprocessing;
        if let Some(pp) = pp_cfg {
            let config_val = serde_json::json!({
                "enabled": pp.enabled,
                "steps": pp.steps,
            });
            if let Some(parent) = preprocess.config_json.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&preprocess.config_json, serde_json::to_string_pretty(&config_val)?)?;
        }

        tracing::info!("running preprocessing pipeline");
        let mut cmd = std::process::Command::new("python3");
        cmd.arg("-m")
            .arg("nextstat.analysis.preprocess.cli")
            .arg("--input")
            .arg(&plan.workspace_json)
            .arg("--output")
            .arg(&plan.workspace_json)
            .arg("--config")
            .arg(&preprocess.config_json);
        if let Some(ref prov) = preprocess.provenance_json {
            if let Some(parent) = prov.parent() {
                std::fs::create_dir_all(parent)?;
            }
            cmd.arg("--provenance").arg(prov);
        }
        let status = cmd.status()?;
        if !status.success() {
            anyhow::bail!("preprocessing failed with exit code: {:?}", status.code());
        }
    }

    if let Some(fit_out) = plan.fit.as_ref() {
        if let Some(parent) = fit_out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        cmd_fit(
            &plan.workspace_json,
            Some(fit_out),
            /*json_metrics*/ None,
            plan.threads,
            interp_defaults,
            None,
            /*bundle*/ None,
            false,
            &spec.execution.fit.fit_regions,
            &spec.execution.fit.validation_regions,
            false,
        )?;
    }

    if let Some(scan) = plan.profile_scan.as_ref() {
        if let Some(parent) = scan.output_json.parent() {
            std::fs::create_dir_all(parent)?;
        }
        cmd_scan(
            &plan.workspace_json,
            scan.start,
            scan.stop,
            scan.points,
            Some(&scan.output_json),
            /*json_metrics*/ None,
            plan.threads,
            interp_defaults,
            None,
            /*bundle*/ None,
        )?;
    }

    if let Some(report) = plan.report.as_ref() {
        cmd_report(
            &plan.workspace_json,
            &report.histfactory_xml,
            plan.fit.as_ref(),
            &report.out_dir,
            report.overwrite,
            report.include_covariance,
            report.render,
            report.pdf.as_ref(),
            report.svg_dir.as_ref(),
            report.python.as_ref(),
            report.skip_uncertainty,
            report.uncertainty_grouping.as_str(),
            plan.threads,
            interp_defaults,
            deterministic,
            &spec.execution.report.blind_regions,
        )?;
    }

    let baseline_compare_report = run_analysis_spec_baseline_compare_gate(config_path, spec)?;

    if let Some(dir) = bundle {
        run::write_run_bundle_spec_v0(dir, config_path.as_path(), spec, &plan)?;
    }

    let summary = serde_json::json!({
        "schema_version": spec.schema_version,
        "threads": plan.threads,
        "workspace_json": plan.workspace_json,
        "outputs": {
            "fit_json": plan.fit,
            "scan_json": plan.profile_scan.as_ref().map(|s| s.output_json.clone()),
            "report_out_dir": plan.report.as_ref().map(|r| r.out_dir.clone()),
        },
        "gates": {
            "baseline_compare_report": baseline_compare_report,
        }
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn cmd_import_histfactory(
    xml: &PathBuf,
    basedir: Option<&Path>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    tracing::info!(path = %xml.display(), "importing HistFactory combination.xml");
    let ws = ns_translate::histfactory::from_xml_with_basedir(xml, basedir)?;
    let output_json = serde_json::to_value(&ws)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "import_histfactory",
            serde_json::json!({}),
            xml,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_import_trex_config(
    config: &PathBuf,
    base_dir: Option<&PathBuf>,
    output: Option<&PathBuf>,
    analysis_yaml: Option<&PathBuf>,
    coverage_json: Option<&PathBuf>,
    expr_coverage_json: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    tracing::info!(path = %config.display(), "importing TRExFitter config");
    let text = std::fs::read_to_string(config)?;

    let base_dir_override = base_dir.map(|p| p.as_path());
    let resolved_base_dir =
        base_dir_override.or_else(|| config.parent()).unwrap_or_else(|| std::path::Path::new("."));

    if let Some(path) = expr_coverage_json {
        let rep = ns_translate::trex::expr_coverage_from_str(&text)?;
        write_json_file(path, &serde_json::to_value(&rep)?)?;
    }

    let (_cfg, coverage) = ns_translate::trex::TrexConfig::parse_str_with_coverage(&text)?;
    let ws = ns_translate::trex::workspace_from_str(&text, resolved_base_dir)?;
    let output_json = serde_json::to_value(&ws)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "import_trex_config",
            serde_json::json!({
                "base_dir": base_dir_override,
                "analysis_yaml": analysis_yaml,
                "coverage_json": coverage_json,
                "expr_coverage_json": expr_coverage_json,
            }),
            config,
            &output_json,
            false,
        )?;
    }

    if let Some(path) = coverage_json {
        let cov_json = serde_json::to_value(&coverage)?;
        write_json_file(path, &cov_json)?;
    }

    if let Some(path) = analysis_yaml {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let stem = config.file_stem().and_then(|s| s.to_str()).unwrap_or("trex_config");
        let ws_out = format!("tmp/{stem}_workspace.json");
        let fit_out = format!("tmp/{stem}_fit.json");
        let scan_out = format!("tmp/{stem}_scan.json");
        let report_out = format!("tmp/{stem}_report");

        let spec = serde_json::json!({
            "$schema": "https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json",
            "schema_version": "trex_analysis_spec_v0",
            "analysis": {
                "name": format!("TREx Config ({stem})"),
                "description": "Generated from `nextstat import trex-config`.",
                "tags": ["trex-config", "generated"],
            },
            "inputs": {
                "mode": "trex_config_txt",
                "trex_config_txt": {
                    "config_path": config,
                    "base_dir": base_dir_override,
                }
            },
            "execution": {
                "determinism": { "threads": 1 },
                "import": {
                    "enabled": true,
                    "output_json": ws_out,
                },
                "fit": {
                    "enabled": false,
                    "output_json": fit_out,
                },
                "profile_scan": {
                    "enabled": false,
                    "start": 0.0,
                    "stop": 5.0,
                    "points": 21,
                    "output_json": scan_out,
                },
                "report": {
                    "enabled": false,
                    "out_dir": report_out,
                    "overwrite": false,
                    "include_covariance": false,
                    "histfactory_xml": serde_json::Value::Null,
                    "render": {
                        "enabled": false,
                        "pdf": serde_json::Value::Null,
                        "svg_dir": serde_json::Value::Null,
                        "python": serde_json::Value::Null,
                    },
                    "skip_uncertainty": true,
                    "uncertainty_grouping": "prefix_1",
                },
            },
            "gates": {
                "baseline_compare": {
                    "enabled": false,
                    "baseline_dir": "tmp/baselines",
                    "require_same_host": true,
                    "max_slowdown": 1.3,
                }
            }
        });

        std::fs::write(path, serde_yaml_ng::to_string(&spec)?)?;
    }

    Ok(())
}

fn cmd_import_patchset(
    workspace: &PathBuf,
    patchset: &PathBuf,
    patch_name: Option<&str>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    tracing::info!(
        workspace = %workspace.display(),
        patchset = %patchset.display(),
        patch_name = patch_name.unwrap_or("<first>"),
        "applying pyhf PatchSet"
    );

    let base_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(workspace)?)?;
    let ps: ns_translate::pyhf::PatchSet =
        serde_json::from_str(&std::fs::read_to_string(patchset)?)?;

    let out_json = ps.apply_to_value(&base_json, patch_name)?;
    write_json(output, &out_json)?;

    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "import_patchset",
            serde_json::json!({
                "patchset": patchset,
                "patch_name": patch_name,
            }),
            workspace,
            &out_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_build_hists(
    config: &PathBuf,
    base_dir: Option<&PathBuf>,
    out_dir: &PathBuf,
    overwrite: bool,
    coverage_json: Option<&PathBuf>,
    expr_coverage_json: Option<&PathBuf>,
) -> Result<()> {
    // Reuse the report/run convention: require empty dir unless --overwrite.
    ensure_out_dir(out_dir, overwrite)?;

    tracing::info!(path = %config.display(), "building hists from TRExFitter config (NTUP)");
    let text = std::fs::read_to_string(config)?;

    let base_dir_override = base_dir.map(|p| p.as_path());
    let resolved_base_dir =
        base_dir_override.or_else(|| config.parent()).unwrap_or_else(|| std::path::Path::new("."));

    if let Some(path) = expr_coverage_json {
        let rep = ns_translate::trex::expr_coverage_from_str(&text)?;
        write_json_file(path, &serde_json::to_value(&rep)?)?;
    }

    let (_cfg, coverage) = ns_translate::trex::TrexConfig::parse_str_with_coverage(&text)?;
    let ws = ns_translate::trex::workspace_from_str(&text, resolved_base_dir)?;

    let ws_json = serde_json::to_value(&ws)?;
    write_json_file(&out_dir.join("workspace.json"), &ws_json)?;

    if let Some(path) = coverage_json {
        let cov_json = serde_json::to_value(&coverage)?;
        write_json_file(path, &cov_json)?;
    }

    Ok(())
}

fn cmd_export_histfactory(
    input: &PathBuf,
    out_dir: &PathBuf,
    overwrite: bool,
    prefix: &str,
    python: Option<&PathBuf>,
) -> Result<()> {
    if prefix.trim().is_empty() {
        anyhow::bail!("--prefix must be non-empty");
    }

    let input = input.canonicalize().map_err(|e| {
        anyhow::anyhow!("failed to resolve input workspace: {}: {}", input.display(), e)
    })?;
    if !input.is_file() {
        anyhow::bail!("input workspace not found: {}", input.display());
    }

    if out_dir.exists() {
        if !out_dir.is_dir() {
            anyhow::bail!("out_dir exists but is not a directory: {}", out_dir.display());
        }
        if !overwrite {
            let mut rd = std::fs::read_dir(out_dir)?;
            if rd.next().is_some() {
                anyhow::bail!(
                    "out_dir is not empty (pass --overwrite to allow): {}",
                    out_dir.display()
                );
            }
        }
    } else {
        std::fs::create_dir_all(out_dir)?;
    }

    let py = python.map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("python3"));

    let code = r#"
import json
import os
from pathlib import Path

import pyhf.writexml as wx

ws_path = Path(os.environ["NS_WORKSPACE_JSON"])
prefix = os.environ.get("NS_PREFIX", "results")

spec = json.loads(ws_path.read_text(encoding="utf-8"))

# Use relative paths so the export directory is self-contained.
channels_dir = Path("channels")
channels_dir.mkdir(parents=True, exist_ok=True)

# Writes channels/*.xml, data.root, HistFactorySchema.dtd (in cwd), returns combination.xml bytes.
combo = wx.writexml(spec, specdir=str(channels_dir), data_rootdir=".", resultprefix=str(prefix))
Path("combination.xml").write_bytes(combo)
"#;

    let out = Command::new(&py)
        .current_dir(out_dir)
        .arg("-c")
        .arg(code)
        .env("NS_WORKSPACE_JSON", input)
        .env("NS_PREFIX", prefix)
        .output()
        .map_err(|e| anyhow::anyhow!("failed to run python exporter ({}): {}", py.display(), e))?;

    if !out.status.success() {
        anyhow::bail!(
            "python exporter failed ({}):\nstdout:\n{}\nstderr:\n{}",
            py.display(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    Ok(())
}

fn cmd_fit(
    input: &PathBuf,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    gpu: Option<&str>,
    bundle: Option<&PathBuf>,
    parity: bool,
    fit_regions: &[String],
    validation_regions: &[String],
    asimov: bool,
) -> Result<()> {
    if gpu.is_some() && (!fit_regions.is_empty() || !validation_regions.is_empty()) {
        anyhow::bail!(
            "fit/validation region selection is not supported in --gpu mode yet (run on CPU or omit region filters)"
        );
    }
    let model = {
        let base = load_model(input, threads, parity, interp_defaults)?;
        let selected = base.with_fit_channel_selection(
            (!fit_regions.is_empty()).then_some(fit_regions),
            (!validation_regions.is_empty()).then_some(validation_regions),
        )?;
        if asimov {
            tracing::info!("Asimov (blind) fit: replacing observed data with expected at nominal");
            let init_params: Vec<f64> = selected.parameters().iter().map(|p| p.init).collect();
            let asimov_data = ns_inference::asimov_main(&selected, &init_params)?;
            selected.with_observed_main(&asimov_data)?
        } else {
            selected
        }
    };

    let start = std::time::Instant::now();
    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let result = match gpu {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                mle.fit_gpu(&model)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                mle.fit_metal(&model)?
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some(_) => unreachable!("unknown device should have bailed earlier"),
        None => mle.fit(&model)?,
    };
    tracing::info!(nll = result.nll, converged = result.converged, "fit complete");
    let wall_time_s = start.elapsed().as_secs_f64();

    let parameter_names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();
    let poi_idx = model.poi_index();
    let (poi_hat, poi_sigma) = poi_idx
        .and_then(|i| result.parameters.get(i).copied().zip(result.uncertainties.get(i).copied()))
        .map(|(a, b)| (Some(a), Some(b)))
        .unwrap_or((None, None));

    let output_json = serde_json::json!({
        "parameter_names": parameter_names,
        "poi_index": poi_idx,
        "bestfit": result.parameters,
        "uncertainties": result.uncertainties,
        "nll": result.nll,
        "twice_nll": 2.0 * result.nll,
        "converged": result.converged,
        // Back-compat: keep `n_evaluations` as alias for optimizer iterations.
        "n_evaluations": result.n_iter,
        "n_iter": result.n_iter,
        "n_fev": result.n_fev,
        "n_gev": result.n_gev,
        "covariance": result.covariance,
        "termination_reason": result.termination_reason,
        "final_grad_norm": if result.final_grad_norm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.final_grad_norm) },
        "initial_nll": if result.initial_nll.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.initial_nll) },
        "n_active_bounds": result.n_active_bounds,
        "edm": if result.edm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.edm) },
        "fit_regions": if fit_regions.is_empty() { serde_json::Value::Null } else { serde_json::json!(fit_regions) },
        "validation_regions": if validation_regions.is_empty() { serde_json::Value::Null } else { serde_json::json!(validation_regions) },
    });

    write_json(output, &output_json)?;
    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "fit",
            gpu.unwrap_or("cpu"),
            threads,
            parity,
            wall_time_s,
            serde_json::json!({
                "poi_index": poi_idx,
                "poi_hat": poi_hat,
                "poi_sigma": poi_sigma,
                "nll": result.nll,
                "twice_nll": 2.0 * result.nll,
                "converged": result.converged,
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "fit",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_unbinned_fit(
    config: &PathBuf,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    gpu: Option<&str>,
    bundle: Option<&PathBuf>,
    opt_max_iter: Option<u64>,
    opt_tol: Option<f64>,
    opt_m: Option<usize>,
    opt_smooth_bounds: bool,
) -> Result<()> {
    setup_runtime(threads, false);

    let spec = unbinned_spec::read_unbinned_spec(config)?;
    let parameter_names: Vec<String> =
        spec.model.parameters.iter().map(|p| p.name.clone()).collect();
    let poi_idx = if let Some(poi) = &spec.model.poi {
        Some(
            parameter_names
                .iter()
                .position(|n| n == poi)
                .ok_or_else(|| anyhow::anyhow!("unknown POI parameter name: '{poi}'"))?,
        )
    } else {
        None
    };

    let start = std::time::Instant::now();
    let mut opt_cfg = ns_inference::optimizer::OptimizerConfig::default();
    if let Some(v) = opt_max_iter {
        if v == 0 {
            anyhow::bail!("--opt-max-iter must be > 0");
        }
        opt_cfg.max_iter = v;
    }
    if let Some(v) = opt_tol {
        if !v.is_finite() || v <= 0.0 {
            anyhow::bail!("--opt-tol must be finite and > 0");
        }
        opt_cfg.tol = v;
    }
    if let Some(v) = opt_m {
        opt_cfg.m = v;
    }
    if opt_smooth_bounds {
        opt_cfg.smooth_bounds = true;
    }
    let mle = ns_inference::mle::MaximumLikelihoodEstimator::with_config(opt_cfg.clone());
    let result = match gpu {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
                    anyhow::bail!("CUDA is not available at runtime (no device/driver)");
                }
                let model = unbinned_gpu::compile_cuda(&spec, config.as_path())?;
                mle.fit(&model)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
                    anyhow::bail!("Metal is not available at runtime (no Metal device found)");
                }
                let model = unbinned_gpu::compile_metal(&spec, config.as_path())?;
                mle.fit(&model)?
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
        None => {
            let model = unbinned_spec::compile_unbinned_model(&spec, config.as_path())?;
            mle.fit(&model)?
        }
    };
    tracing::info!(nll = result.nll, converged = result.converged, "unbinned fit complete");
    let wall_time_s = start.elapsed().as_secs_f64();

    let (poi_hat, poi_sigma) = poi_idx
        .and_then(|i| result.parameters.get(i).copied().zip(result.uncertainties.get(i).copied()))
        .map(|(a, b)| (Some(a), Some(b)))
        .unwrap_or((None, None));

    let output_json = serde_json::json!({
        "input_schema_version": spec.schema_version,
        "parameter_names": parameter_names,
        "poi_index": poi_idx,
        "bestfit": result.parameters,
        "uncertainties": result.uncertainties,
        "nll": result.nll,
        "twice_nll": 2.0 * result.nll,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "n_fev": result.n_fev,
        "n_gev": result.n_gev,
        "covariance": result.covariance,
        "termination_reason": result.termination_reason,
        "final_grad_norm": if result.final_grad_norm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.final_grad_norm) },
        "initial_nll": if result.initial_nll.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.initial_nll) },
        "n_active_bounds": result.n_active_bounds,
        "edm": if result.edm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.edm) },
    });

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "unbinned_fit",
            gpu.unwrap_or("cpu"),
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "poi_index": poi_idx,
                "poi_hat": poi_hat,
                "poi_sigma": poi_sigma,
                "nll": result.nll,
                "twice_nll": 2.0 * result.nll,
                "converged": result.converged,
                "n_iter": result.n_iter,
                "n_fev": result.n_fev,
                "n_gev": result.n_gev,
                "final_grad_norm": if result.final_grad_norm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.final_grad_norm) },
                "optimizer": {
                    "max_iter": opt_cfg.max_iter,
                    "tol": opt_cfg.tol,
                    "m": opt_cfg.m,
                    "smooth_bounds": opt_cfg.smooth_bounds,
                },
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        // `report::write_bundle` expects JSON inputs (it stores them as inputs/input.json).
        // For Phase 1, require a JSON spec when bundling for clean reproducibility.
        let bytes = std::fs::read(config)?;
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "--bundle requires a JSON unbinned spec (got non-JSON at {}): {e}",
                config.display()
            )
        })?;

        report::write_bundle(
            dir,
            "unbinned_fit",
            serde_json::json!({ "threads": threads, "gpu": gpu.unwrap_or("cpu") }),
            config,
            &output_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_hybrid_fit(
    binned_path: &PathBuf,
    unbinned_path: &PathBuf,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    use ns_core::traits::LogDensityModel;
    use ns_inference::hybrid::{HybridLikelihood, SharedParameterMap};

    let binned_model = load_model(binned_path, threads, false, interp_defaults)?;
    let unbinned_spec = unbinned_spec::read_unbinned_spec(unbinned_path)?;
    let unbinned_model =
        unbinned_spec::compile_unbinned_model(&unbinned_spec, unbinned_path.as_path())?;

    let binned_names = binned_model.parameter_names();
    let unbinned_names = unbinned_model.parameter_names();

    let map = SharedParameterMap::build(&binned_model, &unbinned_model)?;

    // Resolve POI: prefer binned model's POI, fall back to unbinned.
    let map = if let Some(poi) = binned_model.poi_index() {
        map.with_poi_from_a(poi)
    } else if let Some(poi_name) = &unbinned_spec.model.poi {
        let ub_poi = unbinned_names
            .iter()
            .position(|n| n == poi_name)
            .ok_or_else(|| anyhow::anyhow!("unknown POI in unbinned spec: '{poi_name}'"))?;
        map.with_poi_from_b(ub_poi)
    } else {
        map
    };

    let n_shared = map.n_shared();
    let n_global = map.dim();
    let poi_idx = map.poi_index();

    eprintln!(
        "Hybrid model: {} binned params + {} unbinned params → {} global ({} shared)",
        binned_names.len(),
        unbinned_names.len(),
        n_global,
        n_shared,
    );

    let hybrid = HybridLikelihood::new(binned_model, unbinned_model, map);
    let global_names = hybrid.parameter_names();

    let start = std::time::Instant::now();
    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let result = mle.fit(&hybrid)?;
    tracing::info!(nll = result.nll, converged = result.converged, "hybrid fit complete");
    let wall_time_s = start.elapsed().as_secs_f64();

    let (poi_hat, poi_sigma) = poi_idx
        .and_then(|i| result.parameters.get(i).copied().zip(result.uncertainties.get(i).copied()))
        .map(|(a, b)| (Some(a), Some(b)))
        .unwrap_or((None, None));

    let output_json = serde_json::json!({
        "hybrid": true,
        "n_binned_params": binned_names.len(),
        "n_unbinned_params": unbinned_names.len(),
        "n_shared": n_shared,
        "parameter_names": global_names,
        "poi_index": poi_idx,
        "bestfit": result.parameters,
        "uncertainties": result.uncertainties,
        "nll": result.nll,
        "twice_nll": 2.0 * result.nll,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "n_fev": result.n_fev,
        "n_gev": result.n_gev,
        "covariance": result.covariance,
        "termination_reason": result.termination_reason,
        "final_grad_norm": if result.final_grad_norm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.final_grad_norm) },
        "initial_nll": if result.initial_nll.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.initial_nll) },
        "n_active_bounds": result.n_active_bounds,
        "edm": if result.edm.is_nan() { serde_json::Value::Null } else { serde_json::json!(result.edm) },
    });

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "hybrid_fit",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "poi_index": poi_idx,
                "poi_hat": poi_hat,
                "poi_sigma": poi_sigma,
                "nll": result.nll,
                "twice_nll": 2.0 * result.nll,
                "converged": result.converged,
                "n_shared": n_shared,
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "hybrid_fit",
            serde_json::json!({ "threads": threads }),
            binned_path,
            &output_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_unbinned_scan(
    config: &PathBuf,
    start: f64,
    stop: f64,
    points: usize,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    gpu: Option<&str>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if points < 2 {
        anyhow::bail!("points must be >= 2");
    }

    setup_runtime(threads, false);

    let spec = unbinned_spec::read_unbinned_spec(config)?;

    let t0 = std::time::Instant::now();
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let step = (stop - start) / (points as f64 - 1.0);
    let mu_values: Vec<f64> = (0..points).map(|i| start + step * i as f64).collect();
    let scan = match gpu {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
                    anyhow::bail!("CUDA is not available at runtime (no device/driver)");
                }
                let model = unbinned_gpu::compile_cuda(&spec, config.as_path())?;
                ns_inference::profile_likelihood::scan(&mle, &model, &mu_values)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
                    anyhow::bail!("Metal is not available at runtime (no Metal device found)");
                }
                let model = unbinned_gpu::compile_metal(&spec, config.as_path())?;
                ns_inference::profile_likelihood::scan(&mle, &model, &mu_values)?
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
        None => {
            let model = unbinned_spec::compile_unbinned_model(&spec, config.as_path())?;
            ns_inference::profile_likelihood::scan(&mle, &model, &mu_values)?
        }
    };
    let wall_time_s = t0.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "input_schema_version": spec.schema_version,
        "poi_index": scan.poi_index,
        "mu_hat": scan.mu_hat,
        "nll_hat": scan.nll_hat,
        "points": scan.points.iter().map(|p| serde_json::json!({
            "mu": p.mu,
            "q_mu": p.q_mu,
            "nll_mu": p.nll_mu,
            "converged": p.converged,
            "n_iter": p.n_iter,
        })).collect::<Vec<_>>(),
    });

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let poi_hat = Some(scan.mu_hat);

        let q_min = scan.points.iter().fold(f64::INFINITY, |a, p| a.min(p.q_mu));
        let q_max = scan.points.iter().fold(f64::NEG_INFINITY, |a, p| a.max(p.q_mu));

        let metrics_json = metrics_v0(
            "unbinned_scan",
            gpu.unwrap_or("cpu"),
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "poi_index": scan.poi_index,
                "poi_hat": poi_hat.map(|v| serde_json::json!(v)).unwrap_or(serde_json::Value::Null),
                "poi_sigma": serde_json::Value::Null,
                "nll_hat": scan.nll_hat,
                "n_points": scan.points.len(),
                "q_mu_min": if q_min.is_finite() { serde_json::json!(q_min) } else { serde_json::Value::Null },
                "q_mu_max": if q_max.is_finite() { serde_json::json!(q_max) } else { serde_json::Value::Null },
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        // `report::write_bundle` expects JSON inputs (it stores them as inputs/input.json).
        // For Phase 1, require a JSON spec when bundling for clean reproducibility.
        let bytes = std::fs::read(config)?;
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "--bundle requires a JSON unbinned spec (got non-JSON at {}): {e}",
                config.display()
            )
        })?;

        report::write_bundle(
            dir,
            "unbinned_scan",
            serde_json::json!({ "start": start, "stop": stop, "points": points, "threads": threads, "gpu": gpu.unwrap_or("cpu") }),
            config,
            &output_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_unbinned_ranking(
    config: &PathBuf,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    use ns_core::traits::{FixedParamModel, PoiModel};

    setup_runtime(threads, false);

    let spec = unbinned_spec::read_unbinned_spec(config)?;
    let model = unbinned_spec::compile_unbinned_model(&spec, config.as_path())?;

    let poi_idx = model
        .poi_index()
        .ok_or_else(|| anyhow::anyhow!("unbinned-ranking requires model.poi in the spec"))?;

    let t0 = std::time::Instant::now();
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    // Nominal fit (WITH Hessian) to get pull/constraint.
    let nominal = mle.fit(&model)?;
    let mu_hat = nominal.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
    let nll_hat = nominal.nll;

    let base_bounds: Vec<(f64, f64)> = model.parameters().iter().map(|p| p.bounds).collect();

    #[derive(serde::Serialize)]
    struct RankingPoint {
        name: String,
        param_index: usize,
        center: f64,
        sigma: f64,
        theta_hat: f64,
        pull: f64,
        constraint: f64,
        val_up: f64,
        val_down: f64,
        mu_up: Option<f64>,
        mu_down: Option<f64>,
        delta_mu_up: Option<f64>,
        delta_mu_down: Option<f64>,
        converged_up: bool,
        converged_down: bool,
        n_iter_up: Option<usize>,
        n_iter_down: Option<usize>,
        nll_up: Option<f64>,
        nll_down: Option<f64>,
    }

    let mut points = Vec::<RankingPoint>::new();
    let mut entries = Vec::<ns_inference::RankingEntry>::new();

    for (np_idx, p) in model.parameters().iter().enumerate() {
        if np_idx == poi_idx {
            continue;
        }
        let Some(constraint) = p.constraint.clone() else {
            continue;
        };
        if (p.bounds.0 - p.bounds.1).abs() < 1e-12 {
            continue;
        }

        let (center, sigma) = match constraint {
            ns_unbinned::Constraint::Gaussian { mean, sigma } => (mean, sigma),
        };
        if !sigma.is_finite() || sigma <= 0.0 {
            continue;
        }

        let (b_lo, b_hi) = base_bounds[np_idx];
        let val_up = (center + sigma).min(b_hi);
        let val_down = (center - sigma).max(b_lo);

        let theta_hat = nominal.parameters[np_idx];
        let pull = (theta_hat - center) / sigma;
        let constraint = nominal.uncertainties[np_idx] / sigma;

        // --- +1σ refit ---
        let m_up = model.with_fixed_param(np_idx, val_up);
        let mut warm_up = nominal.parameters.clone();
        warm_up[np_idx] = val_up;
        let r_up = mle.fit_minimum_from(&m_up, &warm_up);
        let (mu_up, nll_up, conv_up, n_iter_up) = match r_up {
            Ok(r) => (
                r.parameters.get(poi_idx).copied(),
                Some(r.fval),
                r.converged,
                Some(r.n_iter as usize),
            ),
            Err(_) => (None, None, false, None),
        };

        // --- -1σ refit ---
        let m_down = model.with_fixed_param(np_idx, val_down);
        let mut warm_down = nominal.parameters.clone();
        warm_down[np_idx] = val_down;
        let r_down = mle.fit_minimum_from(&m_down, &warm_down);
        let (mu_down, nll_down, conv_down, n_iter_down) = match r_down {
            Ok(r) => (
                r.parameters.get(poi_idx).copied(),
                Some(r.fval),
                r.converged,
                Some(r.n_iter as usize),
            ),
            Err(_) => (None, None, false, None),
        };

        let delta_mu_up = mu_up.map(|mu| mu - mu_hat);
        let delta_mu_down = mu_down.map(|mu| mu - mu_hat);

        if conv_up
            && conv_down
            && let (Some(d_up), Some(d_down)) = (delta_mu_up, delta_mu_down)
        {
            entries.push(ns_inference::RankingEntry {
                name: p.name.clone(),
                delta_mu_up: d_up,
                delta_mu_down: d_down,
                pull,
                constraint,
            });
        }

        points.push(RankingPoint {
            name: p.name.clone(),
            param_index: np_idx,
            center,
            sigma,
            theta_hat,
            pull,
            constraint,
            val_up,
            val_down,
            mu_up,
            mu_down,
            delta_mu_up,
            delta_mu_down,
            converged_up: conv_up,
            converged_down: conv_down,
            n_iter_up,
            n_iter_down,
            nll_up,
            nll_down,
        });
    }

    // Sort by |impact|
    entries.sort_by(|a, b| {
        let impact_a = a.delta_mu_up.abs().max(a.delta_mu_down.abs());
        let impact_b = b.delta_mu_up.abs().max(b.delta_mu_down.abs());
        impact_b
            .partial_cmp(&impact_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    let artifact: ns_viz::RankingArtifact = entries.clone().into();
    let wall_time_s = t0.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "input_schema_version": spec.schema_version,
        "poi_index": poi_idx,
        "mu_hat": mu_hat,
        "nll_hat": nll_hat,
        "n_points": points.len(),
        "n_converged": artifact.names.len(),
        "points": points,
        "ranking": artifact,
    });
    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let max_impact = entries
            .iter()
            .fold(0.0f64, |a, e| a.max(e.delta_mu_up.abs().max(e.delta_mu_down.abs())));
        let metrics_json = metrics_v0(
            "unbinned_ranking",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "poi_index": poi_idx,
                "poi_hat": mu_hat,
                "poi_sigma": nominal.uncertainties.get(poi_idx).copied().unwrap_or(f64::NAN),
                "nll_hat": nll_hat,
                "n_points": points.len(),
                "n_converged": artifact.names.len(),
                "max_impact": if max_impact.is_finite() { serde_json::json!(max_impact) } else { serde_json::Value::Null },
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        // `report::write_bundle` expects JSON inputs (it stores them as inputs/input.json).
        // For Phase 1, require a JSON spec when bundling for clean reproducibility.
        let bytes = std::fs::read(config)?;
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "--bundle requires a JSON unbinned spec (got non-JSON at {}): {e}",
                config.display()
            )
        })?;

        report::write_bundle(
            dir,
            "unbinned_ranking",
            serde_json::json!({ "threads": threads }),
            config,
            &output_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_unbinned_hypotest(
    config: &PathBuf,
    mu_test: f64,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    gpu: Option<&str>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    use ns_core::traits::{FixedParamModel, LogDensityModel};

    setup_runtime(threads, false);

    let spec = unbinned_spec::read_unbinned_spec(config)?;
    let parameter_names: Vec<String> =
        spec.model.parameters.iter().map(|p| p.name.clone()).collect();
    let poi_name = spec
        .model
        .poi
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("unbinned-hypotest requires model.poi in the spec"))?;
    let poi_idx = parameter_names
        .iter()
        .position(|n| n == poi_name)
        .ok_or_else(|| anyhow::anyhow!("unknown POI parameter name: '{poi_name}'"))?;

    let start = std::time::Instant::now();
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    // Compile model (CPU or GPU) and run fits.
    let (free, fixed_mu, q0, nll_mu0) = match gpu {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
                    anyhow::bail!("CUDA is not available at runtime (no device/driver)");
                }
                let model = unbinned_gpu::compile_cuda(&spec, config.as_path())?;

                let free = mle.fit_minimum(&model)?;
                let fixed_model = model.with_fixed_param(poi_idx, mu_test);
                let mut warm_mu = free.parameters.clone();
                warm_mu[poi_idx] = mu_test;
                let fixed_mu = mle.fit_minimum_from(&fixed_model, &warm_mu)?;

                let (poi_lo, poi_hi) = model.parameter_bounds()[poi_idx];
                let mut q0 = serde_json::Value::Null;
                let mut nll_mu0 = serde_json::Value::Null;
                if poi_lo <= 0.0 && 0.0 <= poi_hi {
                    let fixed0_model = model.with_fixed_param(poi_idx, 0.0);
                    let mut warm0 = free.parameters.clone();
                    warm0[poi_idx] = 0.0;
                    let fixed0 = mle.fit_minimum_from(&fixed0_model, &warm0)?;
                    let llr0 = 2.0 * (fixed0.fval - free.fval);
                    let mu_hat = free.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
                    let q0_val = if mu_hat < 0.0 { 0.0 } else { llr0.max(0.0) };
                    q0 = serde_json::json!(q0_val);
                    nll_mu0 = serde_json::json!(fixed0.fval);
                }

                (free, fixed_mu, q0, nll_mu0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
                    anyhow::bail!("Metal is not available at runtime (no Metal device found)");
                }
                let model = unbinned_gpu::compile_metal(&spec, config.as_path())?;

                let free = mle.fit_minimum(&model)?;
                let fixed_model = model.with_fixed_param(poi_idx, mu_test);
                let mut warm_mu = free.parameters.clone();
                warm_mu[poi_idx] = mu_test;
                let fixed_mu = mle.fit_minimum_from(&fixed_model, &warm_mu)?;

                let (poi_lo, poi_hi) = model.parameter_bounds()[poi_idx];
                let mut q0 = serde_json::Value::Null;
                let mut nll_mu0 = serde_json::Value::Null;
                if poi_lo <= 0.0 && 0.0 <= poi_hi {
                    let fixed0_model = model.with_fixed_param(poi_idx, 0.0);
                    let mut warm0 = free.parameters.clone();
                    warm0[poi_idx] = 0.0;
                    let fixed0 = mle.fit_minimum_from(&fixed0_model, &warm0)?;
                    let llr0 = 2.0 * (fixed0.fval - free.fval);
                    let mu_hat = free.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
                    let q0_val = if mu_hat < 0.0 { 0.0 } else { llr0.max(0.0) };
                    q0 = serde_json::json!(q0_val);
                    nll_mu0 = serde_json::json!(fixed0.fval);
                }

                (free, fixed_mu, q0, nll_mu0)
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
        None => {
            let model = unbinned_spec::compile_unbinned_model(&spec, config.as_path())?;

            let free = mle.fit_minimum(&model)?;
            let fixed_model = model.with_fixed_param(poi_idx, mu_test);
            let mut warm_mu = free.parameters.clone();
            warm_mu[poi_idx] = mu_test;
            let fixed_mu = mle.fit_minimum_from(&fixed_model, &warm_mu)?;

            let (poi_lo, poi_hi) = model.parameter_bounds()[poi_idx];
            let mut q0 = serde_json::Value::Null;
            let mut nll_mu0 = serde_json::Value::Null;
            if poi_lo <= 0.0 && 0.0 <= poi_hi {
                let fixed0_model = model.with_fixed_param(poi_idx, 0.0);
                let mut warm0 = free.parameters.clone();
                warm0[poi_idx] = 0.0;
                let fixed0 = mle.fit_minimum_from(&fixed0_model, &warm0)?;
                let llr0 = 2.0 * (fixed0.fval - free.fval);
                let mu_hat = free.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
                let q0_val = if mu_hat < 0.0 { 0.0 } else { llr0.max(0.0) };
                q0 = serde_json::json!(q0_val);
                nll_mu0 = serde_json::json!(fixed0.fval);
            }

            (free, fixed_mu, q0, nll_mu0)
        }
    };

    let mu_hat = free.parameters.get(poi_idx).copied().unwrap_or(f64::NAN);
    let nll_hat = free.fval;

    let nll_mu = fixed_mu.fval;

    // Upper-limit style q_mu (one-sided).
    let llr = 2.0 * (nll_mu - nll_hat);
    let mut q_mu = llr.max(0.0);
    if mu_hat > mu_test {
        q_mu = 0.0;
    }

    let wall_time_s = start.elapsed().as_secs_f64();
    let output_json = serde_json::json!({
        "input_schema_version": spec.schema_version,
        "poi_index": poi_idx,
        "mu_test": mu_test,
        "mu_hat": mu_hat,
        "nll_hat": nll_hat,
        "nll_mu": nll_mu,
        "q_mu": q_mu,
        "q0": q0,
        "nll_mu0": nll_mu0,
        "converged_hat": free.converged,
        "converged_mu": fixed_mu.converged,
        "n_iter_hat": free.n_iter,
        "n_iter_mu": fixed_mu.n_iter,
    });

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "unbinned_hypotest",
            gpu.unwrap_or("cpu"),
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "poi_index": poi_idx,
                "mu_test": mu_test,
                "mu_hat": mu_hat,
                "nll_hat": nll_hat,
                "nll_mu": nll_mu,
                "q_mu": q_mu,
                "q0": output_json.get("q0").cloned().unwrap_or(serde_json::Value::Null),
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        // `report::write_bundle` expects JSON inputs (it stores them as inputs/input.json).
        // For Phase 1, require a JSON spec when bundling for clean reproducibility.
        let bytes = std::fs::read(config)?;
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "--bundle requires a JSON unbinned spec (got non-JSON at {}): {e}",
                config.display()
            )
        })?;

        report::write_bundle(
            dir,
            "unbinned_hypotest",
            serde_json::json!({ "threads": threads, "mu": mu_test, "gpu": gpu.unwrap_or("cpu") }),
            config,
            &output_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_unbinned_hypotest_toys(
    config: &PathBuf,
    mu_test: f64,
    n_toys: usize,
    seed: u64,
    expected_set: bool,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    gpu: Option<&str>,
    gpu_devices: &[usize],
    gpu_shards: Option<usize>,
    gpu_sample_toys: bool,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    use ns_core::traits::PoiModel;
    #[cfg(any(feature = "metal", feature = "cuda"))]
    use ns_core::traits::{FixedParamModel, LogDensityModel};

    if n_toys == 0 {
        anyhow::bail!("n_toys must be > 0");
    }
    if gpu_sample_toys && gpu != Some("cuda") && gpu != Some("metal") {
        anyhow::bail!("--gpu-sample-toys requires --gpu cuda|metal");
    }
    let cuda_device_ids = normalize_cuda_device_ids(gpu, gpu_devices)?;

    setup_runtime(threads, false);

    let spec = unbinned_spec::read_unbinned_spec(config)?;
    let model = unbinned_spec::compile_unbinned_model(&spec, config.as_path())?;

    let poi_idx = model
        .poi_index()
        .ok_or_else(|| anyhow::anyhow!("unbinned-hypotest-toys requires model.poi in the spec"))?;

    // PF3.1-OPT4: estimate VRAM per toy from expected yields (not observed data size).
    // For hypotest-toys we conservatively take the maximum of the B-only (mu=0) and
    // S+B (mu=mu_test) expected yields to avoid under-sharding in large-yield studies.
    let (estimated_bytes_per_toy, estimated_events_per_toy): (usize, usize) = {
        let mut params_b: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        let mut params_sb = params_b.clone();
        if poi_idx < params_b.len() {
            params_b[poi_idx] = 0.0;
            params_sb[poi_idx] = mu_test.max(0.0);
        }
        let bytes = estimate_unbinned_expected_bytes_per_toy(&model, &params_b)?
            .max(estimate_unbinned_expected_bytes_per_toy(&model, &params_sb)?);
        let events = estimate_unbinned_expected_events_per_toy(&model, &params_b)?
            .max(estimate_unbinned_expected_events_per_toy(&model, &params_sb)?);
        (bytes, events)
    };

    let metal_max_toys_per_batch = if gpu == Some("metal") {
        metal_max_toys_per_batch_for_u32_offsets(n_toys, estimated_events_per_toy)?
    } else {
        n_toys.max(1)
    };

    #[cfg(feature = "cuda")]
    let cuda_device_shard_plan = normalize_cuda_device_shard_plan(
        gpu,
        gpu_sample_toys,
        &cuda_device_ids,
        gpu_shards,
        n_toys,
        estimated_bytes_per_toy,
        estimated_events_per_toy,
    )?;
    #[cfg(not(feature = "cuda"))]
    let _cuda_device_shard_plan = normalize_cuda_device_shard_plan(
        gpu,
        gpu_sample_toys,
        &cuda_device_ids,
        gpu_shards,
        n_toys,
        estimated_bytes_per_toy,
        estimated_events_per_toy,
    )?;

    let start = std::time::Instant::now();
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let mut timing_breakdown = serde_json::Map::<String, serde_json::Value>::new();

    let sample_toy = |m: &ns_unbinned::UnbinnedModel, params: &[f64], seed: u64| {
        m.sample_poisson_toy(params, seed)
    };

    let output_json = match gpu {
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                use rayon::prelude::*;

                const CLB_MIN: f64 = 1e-300;

                fn normal_cdf(x: f64) -> f64 {
                    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
                }

                #[inline]
                fn safe_cls(clsb: f64, clb: f64) -> f64 {
                    if !(clsb.is_finite() && clb.is_finite()) {
                        return 0.0;
                    }
                    if clb <= CLB_MIN {
                        return if clsb <= CLB_MIN { 0.0 } else { 1.0 };
                    }
                    (clsb / clb).clamp(0.0, 1.0)
                }

                #[inline]
                fn tail_prob_counts(n_ge: usize, n_valid: usize) -> f64 {
                    if n_valid == 0 {
                        return 0.0;
                    }
                    // Add-one smoothing to avoid exact 0/1 tail-probabilities.
                    (n_ge as f64 + 1.0) / (n_valid as f64 + 1.0)
                }

                fn tail_prob_sorted(sorted: &[f64], threshold: f64) -> f64 {
                    // sorted ascending.
                    let n = sorted.len();
                    if n == 0 {
                        return 0.0;
                    }
                    let idx = sorted.partition_point(|v| *v < threshold);
                    let tail = n - idx;
                    tail_prob_counts(tail, n)
                }

                fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
                    let n = sorted.len();
                    if n == 0 {
                        return f64::NAN;
                    }
                    if p <= 0.0 {
                        return sorted[0];
                    }
                    if p >= 1.0 {
                        return sorted[n - 1];
                    }
                    let idx = p * ((n - 1) as f64);
                    let lo = idx.floor() as usize;
                    let hi = idx.ceil() as usize;
                    if lo == hi {
                        return sorted[lo];
                    }
                    let w = idx - (lo as f64);
                    sorted[lo] + w * (sorted[hi] - sorted[lo])
                }

                if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available()
                {
                    anyhow::bail!("Metal is not available at runtime (no Metal device found)");
                }

                let (_meta, static_models) =
                    unbinned_gpu::build_gpu_static_datas(&spec, config.as_path())?;

                let included_channels: Vec<usize> = spec
                    .channels
                    .iter()
                    .enumerate()
                    .filter(|(_, ch)| ch.include_in_fit)
                    .map(|(i, _)| i)
                    .collect();
                if included_channels.is_empty() {
                    anyhow::bail!(
                        "unbinned-hypotest-toys requires at least one channel with include_in_fit=true"
                    );
                }
                if included_channels.len() != static_models.len() {
                    anyhow::bail!(
                        "internal error: included_channels ({}) != static_models ({})",
                        included_channels.len(),
                        static_models.len()
                    );
                }
                let obs_names: Vec<String> = included_channels
                    .iter()
                    .map(|&i| spec.channels[i].observables[0].name.clone())
                    .collect();

                // Phase A: baseline CPU fits (3 fits, not bottleneck)
                let t_obs_fits0 = std::time::Instant::now();
                let init0 = model.parameter_init();
                let free_obs = mle.fit_minimum_from(&model, &init0)?;
                if !free_obs.converged {
                    anyhow::bail!("Observed-data free fit did not converge: {}", free_obs.message);
                }
                let free_nll = free_obs.fval;
                let mu_hat = free_obs.parameters[poi_idx];

                let bounds = model.parameter_bounds();
                let mut bounds_fixed = bounds.clone();
                bounds_fixed[poi_idx] = (mu_test, mu_test);
                let mut bounds_mu0 = bounds.clone();
                bounds_mu0[poi_idx] = (0.0, 0.0);
                let toy_fixed_fit_cfg = ns_inference::OptimizerConfig {
                    max_iter: 400,
                    tol: 3e-6,
                    ..ns_inference::OptimizerConfig::default()
                };

                let fixed_mu_model = model.with_fixed_param(poi_idx, mu_test);
                let mut warm_mu = free_obs.parameters.clone();
                warm_mu[poi_idx] = mu_test;
                let fixed_mu = mle.fit_minimum_from(&fixed_mu_model, &warm_mu)?;
                if !fixed_mu.converged {
                    anyhow::bail!(
                        "Failed to fit generation point at mu={}: {}",
                        mu_test,
                        fixed_mu.message
                    );
                }

                let fixed_0_model = model.with_fixed_param(poi_idx, 0.0);
                let mut warm_0 = free_obs.parameters.clone();
                warm_0[poi_idx] = 0.0;
                let fixed_0 = mle.fit_minimum_from(&fixed_0_model, &warm_0)?;
                if !fixed_0.converged {
                    anyhow::bail!("Failed to fit generation point at mu=0: {}", fixed_0.message);
                }
                timing_breakdown.insert(
                    "obs_fits_s".into(),
                    serde_json::json!(t_obs_fits0.elapsed().as_secs_f64()),
                );

                let q_obs = if mu_hat > mu_test {
                    0.0
                } else {
                    (2.0 * (fixed_mu.fval - free_nll)).max(0.0)
                };

                // Phase B: CPU toy generation + GPU batch fits
                let n_channels = static_models.len();
                let metal_batches = contiguous_toy_batches(n_toys, metal_max_toys_per_batch);
                if metal_batches.is_empty() {
                    anyhow::bail!("internal error: empty Metal toy batch plan");
                }

                let metal_samplers = if gpu_sample_toys {
                    if static_models.is_empty() {
                        anyhow::bail!("--gpu-sample-toys requires at least one included channel");
                    }
                    Some(
                        static_models
                            .iter()
                            .map(|m| {
                                ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler::from_unbinned_static(m)
                                    .map_err(|e| anyhow::anyhow!("{e}"))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )
                } else {
                    None
                };

                let sample_device = |gen_params: &[f64],
                                     seed0: u64,
                                     n_toys_local: usize|
                 -> Result<(
                    Vec<Vec<u32>>,
                    Vec<ns_compute::metal_rs::Buffer>,
                    Vec<ns_compute::metal_rs::Buffer>,
                )> {
                    let samplers = metal_samplers.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "internal error: sample_device called without Metal samplers"
                        )
                    })?;
                    let mut toy_offsets_by_channel = Vec::<Vec<u32>>::with_capacity(samplers.len());
                    let mut buf_toy_offsets_by_channel =
                        Vec::<ns_compute::metal_rs::Buffer>::with_capacity(samplers.len());
                    let mut buf_obs_flat_by_channel =
                        Vec::<ns_compute::metal_rs::Buffer>::with_capacity(samplers.len());
                    for (ch_i, sampler) in samplers.iter().enumerate() {
                        let seed_ch =
                            seed0.wrapping_add(1_000_000_003u64.wrapping_mul(ch_i as u64));
                        let (offs, offs_buf, buf) = sampler.sample_toys_1d_device_with_offsets(
                            gen_params,
                            n_toys_local,
                            seed_ch,
                        )?;
                        toy_offsets_by_channel.push(offs);
                        buf_toy_offsets_by_channel.push(offs_buf);
                        buf_obs_flat_by_channel.push(buf);
                    }
                    Ok((
                        toy_offsets_by_channel,
                        buf_toy_offsets_by_channel,
                        buf_obs_flat_by_channel,
                    ))
                };

                let sample_flat = |gen_params: &[f64],
                                   seed0: u64,
                                   n_toys_local: usize|
                 -> Result<(Vec<Vec<u32>>, Vec<Vec<f64>>)> {
                    // CPU sampling (exact semantics), then fit in lockstep on GPU.
                    let per_toy: Vec<Result<Vec<Vec<f64>>>> = (0..n_toys_local)
                        .into_par_iter()
                        .with_min_len(16)
                        .map(|toy_idx| {
                            let toy_seed = seed0.wrapping_add(toy_idx as u64);
                            let toy_model = model.sample_poisson_toy(gen_params, toy_seed)?;
                            let mut out = Vec::<Vec<f64>>::with_capacity(n_channels);
                            for (out_idx, &ch_idx) in included_channels.iter().enumerate() {
                                let ch = toy_model.channels().get(ch_idx).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy model missing channel index {} (unexpected)",
                                        ch_idx
                                    )
                                })?;
                                let obs_name = obs_names.get(out_idx).ok_or_else(|| {
                                    anyhow::anyhow!("internal error: missing obs_names[{out_idx}]")
                                })?;
                                let xs = ch.data.column(obs_name.as_str()).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy channel '{}' data missing observable column '{}'",
                                        ch.name,
                                        obs_name
                                    )
                                })?;
                                out.push(xs.to_vec());
                            }
                            Ok(out)
                        })
                        .collect();

                    let mut toy_offsets_by_channel: Vec<Vec<u32>> =
                        (0..n_channels).map(|_| Vec::with_capacity(n_toys_local + 1)).collect();
                    for offs in &mut toy_offsets_by_channel {
                        offs.push(0u32);
                    }
                    let mut obs_flat_by_channel: Vec<Vec<f64>> =
                        (0..n_channels).map(|_| Vec::new()).collect();

                    for r in per_toy {
                        let xs_by_ch = r?;
                        if xs_by_ch.len() != n_channels {
                            anyhow::bail!(
                                "internal error: per-toy channel count mismatch: expected {n_channels}, got {}",
                                xs_by_ch.len()
                            );
                        }
                        for (ch_i, xs) in xs_by_ch.into_iter().enumerate() {
                            obs_flat_by_channel[ch_i].extend_from_slice(&xs);
                            toy_offsets_by_channel[ch_i].push(checked_len_to_u32(
                                obs_flat_by_channel[ch_i].len(),
                                "cpu toy sampling offset overflow (hypotest metal path)",
                            )?);
                        }
                    }
                    Ok((toy_offsets_by_channel, obs_flat_by_channel))
                };

                #[derive(Clone, Copy)]
                struct ToyCounts {
                    n_ge: usize,
                    n_valid: usize,
                    n_error: usize,
                    n_nonconverged: usize,
                }

                #[derive(Default, Clone, Copy)]
                struct BatchTiming {
                    build_s: f64,
                    free_fit_s: f64,
                    fixed_fit_s: f64,
                }

                fn count_q_ge_ensemble_metal(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    obs_flat_by_channel: &[Vec<f64>],
                    poi: usize,
                    mu_test: f64,
                    q_obs: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                    fixed_fit_cfg: &ns_inference::OptimizerConfig,
                ) -> Result<(ToyCounts, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_build0 = std::time::Instant::now();
                    let mut accels = Vec::with_capacity(static_models.len());
                    for (i, ((m, offs), obs)) in static_models
                        .iter()
                        .zip(toy_offsets_by_channel.iter())
                        .zip(obs_flat_by_channel.iter())
                        .enumerate()
                    {
                        let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
                            m, offs, obs, n_toys,
                        )
                        .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                        accels.push(accel);
                    }
                    let build_s = t_build0.elapsed().as_secs_f64();

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels, init_free, bounds, None,
                        )?;
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let (fixed_fits, _accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels,
                            &init_fixed,
                            bounds_fixed,
                            Some(fixed_fit_cfg.clone()),
                        )?;
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut n_ge = 0usize;
                    let mut n_valid = 0usize;
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let q = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if q.is_finite() {
                                    n_valid += 1;
                                    if q >= q_obs {
                                        n_ge += 1;
                                    }
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        ToyCounts { n_ge, n_valid, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                #[derive(Debug, Clone)]
                struct QEnsemble {
                    q: Vec<f64>,
                    n_error: usize,
                    n_nonconverged: usize,
                }

                fn generate_q_ensemble_metal(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    obs_flat_by_channel: &[Vec<f64>],
                    poi: usize,
                    mu_test: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                    fixed_fit_cfg: &ns_inference::OptimizerConfig,
                ) -> Result<(QEnsemble, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_build0 = std::time::Instant::now();
                    let mut accels = Vec::with_capacity(static_models.len());
                    for (i, ((m, offs), obs)) in static_models
                        .iter()
                        .zip(toy_offsets_by_channel.iter())
                        .zip(obs_flat_by_channel.iter())
                        .enumerate()
                    {
                        let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
                            m, offs, obs, n_toys,
                        )
                        .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                        accels.push(accel);
                    }
                    let build_s = t_build0.elapsed().as_secs_f64();

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels, init_free, bounds, None,
                        )?;
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let (fixed_fits, _accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels,
                            &init_fixed,
                            bounds_fixed,
                            Some(fixed_fit_cfg.clone()),
                        )?;
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut q = Vec::with_capacity(n_toys);
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let qq = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if qq.is_finite() {
                                    q.push(qq);
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        QEnsemble { q, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                fn count_q_ge_ensemble_metal_device(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    buf_toy_offsets_by_channel: Vec<ns_compute::metal_rs::Buffer>,
                    buf_obs_flat_by_channel: Vec<ns_compute::metal_rs::Buffer>,
                    poi: usize,
                    mu_test: f64,
                    q_obs: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                    fixed_fit_cfg: &ns_inference::OptimizerConfig,
                ) -> Result<(ToyCounts, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_build0 = std::time::Instant::now();
                    let mut accels = Vec::with_capacity(static_models.len());
                    for (i, (m, ((offs, offs_buf), buf_obs))) in static_models
                        .iter()
                        .zip(
                            toy_offsets_by_channel
                                .iter()
                                .zip(buf_toy_offsets_by_channel.into_iter())
                                .zip(buf_obs_flat_by_channel.into_iter()),
                        )
                        .enumerate()
                    {
                        let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device_with_offsets(
                            m, offs, offs_buf, buf_obs, n_toys,
                        )
                        .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                        accels.push(accel);
                    }
                    let build_s = t_build0.elapsed().as_secs_f64();

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels, init_free, bounds, None,
                        )?;
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let (fixed_fits, _accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels,
                            &init_fixed,
                            bounds_fixed,
                            Some(fixed_fit_cfg.clone()),
                        )?;
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut n_ge = 0usize;
                    let mut n_valid = 0usize;
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let q = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if q.is_finite() {
                                    n_valid += 1;
                                    if q >= q_obs {
                                        n_ge += 1;
                                    }
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        ToyCounts { n_ge, n_valid, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                fn generate_q_ensemble_metal_device(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    buf_toy_offsets_by_channel: Vec<ns_compute::metal_rs::Buffer>,
                    buf_obs_flat_by_channel: Vec<ns_compute::metal_rs::Buffer>,
                    poi: usize,
                    mu_test: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                    fixed_fit_cfg: &ns_inference::OptimizerConfig,
                ) -> Result<(QEnsemble, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_build0 = std::time::Instant::now();
                    let mut accels = Vec::with_capacity(static_models.len());
                    for (i, (m, ((offs, offs_buf), buf_obs))) in static_models
                        .iter()
                        .zip(
                            toy_offsets_by_channel
                                .iter()
                                .zip(buf_toy_offsets_by_channel.into_iter())
                                .zip(buf_obs_flat_by_channel.into_iter()),
                        )
                        .enumerate()
                    {
                        let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device_with_offsets(
                            m, offs, offs_buf, buf_obs, n_toys,
                        )
                        .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                        accels.push(accel);
                    }
                    let build_s = t_build0.elapsed().as_secs_f64();

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels, init_free, bounds, None,
                        )?;
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let (fixed_fits, _accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                            accels,
                            &init_fixed,
                            bounds_fixed,
                            Some(fixed_fit_cfg.clone()),
                        )?;
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut q = Vec::with_capacity(n_toys);
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let qq = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if qq.is_finite() {
                                    q.push(qq);
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        QEnsemble { q, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                let run_count_ensemble =
                    |seed0: u64,
                     gen_params: &[f64],
                     init_free: &[f64]|
                     -> Result<(ToyCounts, BatchTiming, f64, f64)> {
                        let mut counts =
                            ToyCounts { n_ge: 0, n_valid: 0, n_error: 0, n_nonconverged: 0 };
                        let mut timing = BatchTiming::default();
                        let mut sample_s_total = 0.0f64;
                        let mut ensemble_s_total = 0.0f64;

                        for (toy_start, toy_end) in metal_batches.iter().copied() {
                            let batch_n_toys = toy_end - toy_start;
                            let t_sample0 = std::time::Instant::now();
                            let (c, t) = if gpu_sample_toys {
                                let (toy_offsets, toy_offs_bufs, bufs) = sample_device(
                                    gen_params,
                                    seed0.wrapping_add(toy_start as u64),
                                    batch_n_toys,
                                )?;
                                sample_s_total += t_sample0.elapsed().as_secs_f64();
                                let t_ensemble0 = std::time::Instant::now();
                                let out = count_q_ge_ensemble_metal_device(
                                    &static_models,
                                    &toy_offsets,
                                    toy_offs_bufs,
                                    bufs,
                                    poi_idx,
                                    mu_test,
                                    q_obs,
                                    init_free,
                                    &bounds,
                                    &bounds_fixed,
                                    &toy_fixed_fit_cfg,
                                )?;
                                ensemble_s_total += t_ensemble0.elapsed().as_secs_f64();
                                out
                            } else {
                                let (toy_offsets, obs_flat) = sample_flat(
                                    gen_params,
                                    seed0.wrapping_add(toy_start as u64),
                                    batch_n_toys,
                                )?;
                                sample_s_total += t_sample0.elapsed().as_secs_f64();
                                let t_ensemble0 = std::time::Instant::now();
                                let out = count_q_ge_ensemble_metal(
                                    &static_models,
                                    &toy_offsets,
                                    &obs_flat,
                                    poi_idx,
                                    mu_test,
                                    q_obs,
                                    init_free,
                                    &bounds,
                                    &bounds_fixed,
                                    &toy_fixed_fit_cfg,
                                )?;
                                ensemble_s_total += t_ensemble0.elapsed().as_secs_f64();
                                out
                            };

                            counts.n_ge += c.n_ge;
                            counts.n_valid += c.n_valid;
                            counts.n_error += c.n_error;
                            counts.n_nonconverged += c.n_nonconverged;
                            timing.build_s += t.build_s;
                            timing.free_fit_s += t.free_fit_s;
                            timing.fixed_fit_s += t.fixed_fit_s;
                        }

                        Ok((counts, timing, sample_s_total, ensemble_s_total))
                    };

                let run_generate_ensemble =
                    |seed0: u64,
                     gen_params: &[f64],
                     init_free: &[f64]|
                     -> Result<(QEnsemble, BatchTiming, f64, f64)> {
                        let mut q_all = Vec::<f64>::with_capacity(n_toys);
                        let mut n_error = 0usize;
                        let mut n_nonconverged = 0usize;
                        let mut timing = BatchTiming::default();
                        let mut sample_s_total = 0.0f64;
                        let mut ensemble_s_total = 0.0f64;

                        for (toy_start, toy_end) in metal_batches.iter().copied() {
                            let batch_n_toys = toy_end - toy_start;
                            let t_sample0 = std::time::Instant::now();
                            let (q, t) = if gpu_sample_toys {
                                let (toy_offsets, toy_offs_bufs, bufs) = sample_device(
                                    gen_params,
                                    seed0.wrapping_add(toy_start as u64),
                                    batch_n_toys,
                                )?;
                                sample_s_total += t_sample0.elapsed().as_secs_f64();
                                let t_ensemble0 = std::time::Instant::now();
                                let out = generate_q_ensemble_metal_device(
                                    &static_models,
                                    &toy_offsets,
                                    toy_offs_bufs,
                                    bufs,
                                    poi_idx,
                                    mu_test,
                                    init_free,
                                    &bounds,
                                    &bounds_fixed,
                                    &toy_fixed_fit_cfg,
                                )?;
                                ensemble_s_total += t_ensemble0.elapsed().as_secs_f64();
                                out
                            } else {
                                let (toy_offsets, obs_flat) = sample_flat(
                                    gen_params,
                                    seed0.wrapping_add(toy_start as u64),
                                    batch_n_toys,
                                )?;
                                sample_s_total += t_sample0.elapsed().as_secs_f64();
                                let t_ensemble0 = std::time::Instant::now();
                                let out = generate_q_ensemble_metal(
                                    &static_models,
                                    &toy_offsets,
                                    &obs_flat,
                                    poi_idx,
                                    mu_test,
                                    init_free,
                                    &bounds,
                                    &bounds_fixed,
                                    &toy_fixed_fit_cfg,
                                )?;
                                ensemble_s_total += t_ensemble0.elapsed().as_secs_f64();
                                out
                            };

                            q_all.extend(q.q);
                            n_error += q.n_error;
                            n_nonconverged += q.n_nonconverged;
                            timing.build_s += t.build_s;
                            timing.free_fit_s += t.free_fit_s;
                            timing.fixed_fit_s += t.fixed_fit_s;
                        }

                        Ok((
                            QEnsemble { q: q_all, n_error, n_nonconverged },
                            timing,
                            sample_s_total,
                            ensemble_s_total,
                        ))
                    };

                let seed_b = seed;
                let seed_sb = seed.wrapping_add(1_000_000_000u64);
                let metal_pipeline = if gpu_sample_toys { "metal_device" } else { "metal_host" };

                if !expected_set {
                    let t_phase_b0 = std::time::Instant::now();
                    let (
                        eb,
                        esb,
                        bt_b,
                        bt_sb,
                        sample_b_s,
                        sample_sb_s,
                        ensemble_b_s,
                        ensemble_sb_s,
                    ) = {
                        let (eb, bt_b, sample_b_s, ensemble_b_s) =
                            run_count_ensemble(seed_b, &fixed_0.parameters, &fixed_0.parameters)?;
                        let (esb, bt_sb, sample_sb_s, ensemble_sb_s) = run_count_ensemble(
                            seed_sb,
                            &fixed_mu.parameters,
                            &fixed_mu.parameters,
                        )?;
                        (eb, esb, bt_b, bt_sb, sample_b_s, sample_sb_s, ensemble_b_s, ensemble_sb_s)
                    };

                    if eb.n_valid == 0 || esb.n_valid == 0 {
                        anyhow::bail!(
                            "All toys failed: b_only_valid={} sb_valid={}",
                            eb.n_valid,
                            esb.n_valid
                        );
                    }

                    let clsb = tail_prob_counts(esb.n_ge, esb.n_valid);
                    let clb = tail_prob_counts(eb.n_ge, eb.n_valid);
                    let cls = safe_cls(clsb, clb);
                    timing_breakdown.insert(
                        "toys".into(),
                        serde_json::json!({
                            "pipeline": metal_pipeline,
                            "expected_set": false,
                            "n_batches": metal_batches.len(),
                            "max_toys_per_batch": metal_max_toys_per_batch,
                            "phase_b_s": t_phase_b0.elapsed().as_secs_f64(),
                            "b": {
                                "sample_s": sample_b_s,
                                "ensemble_s": ensemble_b_s,
                                "build_s": bt_b.build_s,
                                "free_fit_s": bt_b.free_fit_s,
                                "fixed_fit_s": bt_b.fixed_fit_s,
                            },
                            "sb": {
                                "sample_s": sample_sb_s,
                                "ensemble_s": ensemble_sb_s,
                                "build_s": bt_sb.build_s,
                                "free_fit_s": bt_sb.free_fit_s,
                                "fixed_fit_s": bt_sb.fixed_fit_s,
                            },
                        }),
                    );

                    serde_json::json!({
                        "input_schema_version": spec.schema_version,
                        "poi_index": poi_idx,
                        "mu_test": mu_test,
                        "cls": cls,
                        "clsb": clsb,
                        "clb": clb,
                        "q_obs": q_obs,
                        "mu_hat": mu_hat,
                        "n_toys": { "b": n_toys, "sb": n_toys },
                        "n_error": { "b": eb.n_error, "sb": esb.n_error },
                        "n_nonconverged": { "b": eb.n_nonconverged, "sb": esb.n_nonconverged },
                        "seed": seed,
                        "expected_set": serde_json::Value::Null,
                    })
                } else {
                    let t_phase_b0 = std::time::Instant::now();
                    let (
                        eb,
                        esb,
                        bt_b,
                        bt_sb,
                        sample_b_s,
                        sample_sb_s,
                        ensemble_b_s,
                        ensemble_sb_s,
                    ) = {
                        let (eb, bt_b, sample_b_s, ensemble_b_s) = run_generate_ensemble(
                            seed_b,
                            &fixed_0.parameters,
                            &fixed_0.parameters,
                        )?;
                        let (esb, bt_sb, sample_sb_s, ensemble_sb_s) = run_generate_ensemble(
                            seed_sb,
                            &fixed_mu.parameters,
                            &fixed_mu.parameters,
                        )?;
                        (eb, esb, bt_b, bt_sb, sample_b_s, sample_sb_s, ensemble_b_s, ensemble_sb_s)
                    };

                    if eb.q.is_empty() || esb.q.is_empty() {
                        anyhow::bail!(
                            "Toy ensembles are empty after filtering errors: b_only={} sb={}",
                            eb.q.len(),
                            esb.q.len()
                        );
                    }

                    let mut q_b_sorted = eb.q.clone();
                    q_b_sorted.sort_by(|a, b| a.total_cmp(b));
                    let mut q_sb_sorted = esb.q.clone();
                    q_sb_sorted.sort_by(|a, b| a.total_cmp(b));

                    let clsb = tail_prob_sorted(&q_sb_sorted, q_obs);
                    let clb = tail_prob_sorted(&q_b_sorted, q_obs);
                    let cls = safe_cls(clsb, clb);

                    // Build CLs distribution for expected-set.
                    let mut cls_vals: Vec<f64> = Vec::with_capacity(q_b_sorted.len());
                    for &q_val in &q_b_sorted {
                        let clsb_q = tail_prob_sorted(&q_sb_sorted, q_val);
                        let clb_q = tail_prob_sorted(&q_b_sorted, q_val);
                        cls_vals.push(safe_cls(clsb_q, clb_q));
                    }
                    cls_vals.sort_by(|a, b| a.total_cmp(b));

                    let mut expected = [0.0f64; 5];
                    for (i, t) in ns_inference::hypotest::NSIGMA_ORDER.into_iter().enumerate() {
                        let p = normal_cdf(-t);
                        expected[i] = quantile_sorted(&cls_vals, p);
                    }
                    timing_breakdown.insert(
                        "toys".into(),
                        serde_json::json!({
                            "pipeline": metal_pipeline,
                            "expected_set": true,
                            "n_batches": metal_batches.len(),
                            "max_toys_per_batch": metal_max_toys_per_batch,
                            "phase_b_s": t_phase_b0.elapsed().as_secs_f64(),
                            "b": {
                                "sample_s": sample_b_s,
                                "ensemble_s": ensemble_b_s,
                                "build_s": bt_b.build_s,
                                "free_fit_s": bt_b.free_fit_s,
                                "fixed_fit_s": bt_b.fixed_fit_s,
                            },
                            "sb": {
                                "sample_s": sample_sb_s,
                                "ensemble_s": ensemble_sb_s,
                                "build_s": bt_sb.build_s,
                                "free_fit_s": bt_sb.free_fit_s,
                                "fixed_fit_s": bt_sb.fixed_fit_s,
                            },
                        }),
                    );

                    serde_json::json!({
                        "input_schema_version": spec.schema_version,
                        "poi_index": poi_idx,
                        "mu_test": mu_test,
                        "cls": cls,
                        "clsb": clsb,
                        "clb": clb,
                        "q_obs": q_obs,
                        "mu_hat": mu_hat,
                        "n_toys": { "b": n_toys, "sb": n_toys },
                        "n_error": { "b": eb.n_error, "sb": esb.n_error },
                        "n_nonconverged": { "b": eb.n_nonconverged, "sb": esb.n_nonconverged },
                        "seed": seed,
                        "expected_set": {
                            "nsigma_order": [2, 1, 0, -1, -2],
                            "cls": expected,
                        },
                    })
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                use rayon::prelude::*;

                const CLB_MIN: f64 = 1e-300;

                fn normal_cdf(x: f64) -> f64 {
                    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
                }

                #[inline]
                fn safe_cls(clsb: f64, clb: f64) -> f64 {
                    if !(clsb.is_finite() && clb.is_finite()) {
                        return 0.0;
                    }
                    if clb <= CLB_MIN {
                        return if clsb <= CLB_MIN { 0.0 } else { 1.0 };
                    }
                    (clsb / clb).clamp(0.0, 1.0)
                }

                #[inline]
                fn tail_prob_counts(n_ge: usize, n_valid: usize) -> f64 {
                    if n_valid == 0 {
                        return 0.0;
                    }
                    (n_ge as f64 + 1.0) / (n_valid as f64 + 1.0)
                }

                fn tail_prob_sorted(sorted: &[f64], threshold: f64) -> f64 {
                    let n = sorted.len();
                    if n == 0 {
                        return 0.0;
                    }
                    let idx = sorted.partition_point(|v| *v < threshold);
                    let tail = n - idx;
                    tail_prob_counts(tail, n)
                }

                fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
                    let n = sorted.len();
                    if n == 0 {
                        return f64::NAN;
                    }
                    if p <= 0.0 {
                        return sorted[0];
                    }
                    if p >= 1.0 {
                        return sorted[n - 1];
                    }
                    let idx = p * ((n - 1) as f64);
                    let lo = idx.floor() as usize;
                    let hi = idx.ceil() as usize;
                    if lo == hi {
                        return sorted[lo];
                    }
                    let w = idx - (lo as f64);
                    sorted[lo] + w * (sorted[hi] - sorted[lo])
                }

                if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
                    anyhow::bail!("CUDA is not available at runtime (no device/driver)");
                }
                for &device_id in &cuda_device_ids {
                    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available_on_device(
                        device_id,
                    ) {
                        anyhow::bail!("CUDA device {device_id} is not available at runtime");
                    }
                }

                #[cfg(feature = "neural")]
                if unbinned_gpu::spec_has_flow_pdfs(&spec) {
                    use ns_unbinned::pdf::UnbinnedPdf;

                    // ── Flow hypotest CUDA path ──────────────────────────────
                    // CPU sampling → CPU logp eval → CUDA paired free+fixed fits
                    // → q_mu computation → CLs

                    let included_ch_idx =
                        spec.channels.iter().position(|ch| ch.include_in_fit).ok_or_else(|| {
                            anyhow::anyhow!(
                                "unbinned-hypotest-toys: no channel with include_in_fit=true"
                            )
                        })?;
                    let ch = &model.channels()[included_ch_idx];
                    let n_procs = ch.processes.len();
                    let obs_names_flow: Vec<String> = ch.processes[0].pdf.observables().to_vec();
                    if obs_names_flow.len() != 1 {
                        anyhow::bail!(
                            "--gpu cuda with flow PDFs currently supports 1 observable, got {}",
                            obs_names_flow.len()
                        );
                    }
                    let obs_name = &obs_names_flow[0];
                    let (obs_lo, obs_hi) = ch.data.bounds(obs_name).ok_or_else(|| {
                        anyhow::anyhow!("missing bounds for observable '{obs_name}'")
                    })?;

                    // Phase A: baseline CPU fits (3 fits)
                    let t_obs_fits0 = std::time::Instant::now();
                    let init0 = model.parameter_init();
                    let free_obs = mle.fit_minimum_from(&model, &init0)?;
                    if !free_obs.converged {
                        anyhow::bail!(
                            "Observed-data free fit did not converge: {}",
                            free_obs.message
                        );
                    }
                    let free_nll = free_obs.fval;
                    let mu_hat = free_obs.parameters[poi_idx];

                    let bounds = model.parameter_bounds();
                    let mut bounds_fixed = bounds.clone();
                    bounds_fixed[poi_idx] = (mu_test, mu_test);

                    let fixed_mu_model = model.with_fixed_param(poi_idx, mu_test);
                    let mut warm_mu = free_obs.parameters.clone();
                    warm_mu[poi_idx] = mu_test;
                    let fixed_mu = mle.fit_minimum_from(&fixed_mu_model, &warm_mu)?;
                    if !fixed_mu.converged {
                        anyhow::bail!(
                            "Failed to fit generation point at mu={}: {}",
                            mu_test,
                            fixed_mu.message
                        );
                    }

                    let fixed_0_model = model.with_fixed_param(poi_idx, 0.0);
                    let mut warm_0 = free_obs.parameters.clone();
                    warm_0[poi_idx] = 0.0;
                    let fixed_0 = mle.fit_minimum_from(&fixed_0_model, &warm_0)?;
                    if !fixed_0.converged {
                        anyhow::bail!(
                            "Failed to fit generation point at mu=0: {}",
                            fixed_0.message
                        );
                    }
                    timing_breakdown.insert(
                        "obs_fits_s".into(),
                        serde_json::json!(t_obs_fits0.elapsed().as_secs_f64()),
                    );

                    let q_obs = if mu_hat > mu_test {
                        0.0
                    } else {
                        (2.0 * (fixed_mu.fval - free_nll)).max(0.0)
                    };

                    // Helper: sample n_toys on CPU, eval logp, build flow config,
                    // run paired free+fixed CUDA fits, compute q distribution.
                    let run_flow_ensemble =
                        |gen_params_ens: &[f64],
                         seed0: u64,
                         init_free: &[f64]|
                         -> Result<(Vec<f64>, usize, usize, f64, f64, f64)> {
                            // Step 1: Sample toys on CPU
                            let t_sample0 = std::time::Instant::now();
                            let mut toy_offsets = vec![0u32];
                            let mut xs_flat: Vec<f64> = Vec::new();
                            for toy_idx in 0..n_toys {
                                let toy_seed = seed0.wrapping_add(toy_idx as u64);
                                let toy_model = model.sample_poisson_toy(gen_params_ens, toy_seed)?;
                                let toy_ch = &toy_model.channels()[included_ch_idx];
                                if let Some(col) = toy_ch.data.column(obs_name) {
                                    xs_flat.extend_from_slice(col);
                                }
                                toy_offsets.push(checked_len_to_u32(
                                    xs_flat.len(),
                                    "cpu flow toy sampling offset overflow (hypotest)",
                                )?);
                            }
                            let total_events = xs_flat.len();
                            let sample_s = t_sample0.elapsed().as_secs_f64();

                            // Step 2: Eval logp per process on concatenated events
                            let obs_spec = ns_unbinned::event_store::ObservableSpec::branch(
                                obs_name.clone(),
                                (obs_lo, obs_hi),
                            );
                            let big_store = ns_unbinned::EventStore::from_columns(
                                vec![obs_spec],
                                vec![(obs_name.clone(), xs_flat)],
                                None,
                            )
                            .map_err(|e| anyhow::anyhow!("building concatenated EventStore: {e}"))?;

                            let mut logp_flat = Vec::with_capacity(n_procs * total_events);
                            for proc in &ch.processes {
                                let shape_params: Vec<f64> =
                                    proc.shape_param_indices.iter().map(|&i| gen_params_ens[i]).collect();
                                let mut logp = vec![0.0f64; total_events];
                                proc.pdf
                                    .log_prob_batch(&big_store, &shape_params, &mut logp)
                                    .map_err(|e| {
                                        anyhow::anyhow!("process '{}' logp eval: {e}", proc.name)
                                    })?;
                                logp_flat.extend_from_slice(&logp);
                            }

                            // Step 3: Build flow config
                            let flow_config = unbinned_gpu::build_flow_batch_config(
                                &spec, n_toys, toy_offsets, total_events,
                            )?;

                            // Step 4: Free fit
                            let t_free0 = std::time::Instant::now();
                            let (free_fits, accel) =
                                ns_inference::unbinned_gpu_batch::fit_flow_toys_batch_cuda(
                                    &flow_config, &logp_flat, init_free, &bounds, None,
                                )?;
                            let free_fit_s = t_free0.elapsed().as_secs_f64();

                            // Step 5: Fixed-mu fit (reuse accelerator)
                            let mut init_fixed = init_free.to_vec();
                            init_fixed[poi_idx] = mu_test;
                            let t_fixed0 = std::time::Instant::now();
                            let (fixed_fits, _accel) =
                                ns_inference::unbinned_gpu_batch::fit_flow_toys_batch_cuda_with_accel(
                                    accel, &init_fixed, &bounds_fixed, None,
                                )?;
                            let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                            // Step 6: Compute q_mu per toy
                            let mut q_vals = Vec::with_capacity(n_toys);
                            let mut n_error = 0usize;
                            let mut n_nonconverged = 0usize;
                            for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                                match (fr, fxr) {
                                    (Ok(free), Ok(fixed)) => {
                                        let mu_hat_toy = free.parameters[poi_idx];
                                        let q = if mu_hat_toy > mu_test {
                                            0.0
                                        } else {
                                            (2.0 * (fixed.nll - free.nll)).max(0.0)
                                        };
                                        if q.is_finite() {
                                            q_vals.push(q);
                                        } else {
                                            n_error += 1;
                                        }
                                        let conv = if mu_hat_toy > mu_test {
                                            free.converged
                                        } else {
                                            free.converged && fixed.converged
                                        };
                                        if !conv {
                                            n_nonconverged += 1;
                                        }
                                    }
                                    _ => {
                                        n_error += 1;
                                    }
                                }
                            }

                            Ok((q_vals, n_error, n_nonconverged, sample_s, free_fit_s, fixed_fit_s))
                        };

                    let seed_b = seed;
                    let seed_sb = seed.wrapping_add(1_000_000_000u64);

                    // Phase B: b-only ensemble
                    let (q_b, n_error_b, n_nonconv_b, sample_b_s, free_b_s, fixed_b_s) =
                        run_flow_ensemble(&fixed_0.parameters, seed_b, &fixed_0.parameters)?;

                    // Phase C: s+b ensemble
                    let (q_sb, n_error_sb, n_nonconv_sb, sample_sb_s, free_sb_s, fixed_sb_s) =
                        run_flow_ensemble(&fixed_mu.parameters, seed_sb, &fixed_mu.parameters)?;

                    timing_breakdown.insert(
                        "toys".into(),
                        serde_json::json!({
                            "pipeline": "cuda_flow",
                            "expected_set": expected_set,
                            "b": {
                                "sample_s": sample_b_s,
                                "fit_free_s": free_b_s,
                                "fit_fixed_s": fixed_b_s,
                            },
                            "sb": {
                                "sample_s": sample_sb_s,
                                "fit_free_s": free_sb_s,
                                "fit_fixed_s": fixed_sb_s,
                            },
                        }),
                    );

                    if q_b.is_empty() || q_sb.is_empty() {
                        anyhow::bail!(
                            "Toy ensembles are empty after filtering errors: b_only={} sb={}",
                            q_b.len(),
                            q_sb.len()
                        );
                    }

                    let mut q_b_sorted = q_b;
                    q_b_sorted.sort_by(|a, b| a.total_cmp(b));
                    let mut q_sb_sorted = q_sb;
                    q_sb_sorted.sort_by(|a, b| a.total_cmp(b));

                    let clsb = tail_prob_sorted(&q_sb_sorted, q_obs);
                    let clb = tail_prob_sorted(&q_b_sorted, q_obs);
                    let cls = safe_cls(clsb, clb);

                    let expected_json = if expected_set {
                        let mut cls_vals: Vec<f64> = Vec::with_capacity(q_b_sorted.len());
                        for &q_val in &q_b_sorted {
                            let clsb_q = tail_prob_sorted(&q_sb_sorted, q_val);
                            let clb_q = tail_prob_sorted(&q_b_sorted, q_val);
                            cls_vals.push(safe_cls(clsb_q, clb_q));
                        }
                        cls_vals.sort_by(|a, b| a.total_cmp(b));
                        let mut expected = [0.0f64; 5];
                        for (i, t) in ns_inference::hypotest::NSIGMA_ORDER.into_iter().enumerate() {
                            let p = normal_cdf(-t);
                            expected[i] = quantile_sorted(&cls_vals, p);
                        }
                        serde_json::json!({
                            "nsigma_order": [2, 1, 0, -1, -2],
                            "cls": expected,
                        })
                    } else {
                        serde_json::Value::Null
                    };

                    let flow_output = serde_json::json!({
                        "input_schema_version": spec.schema_version,
                        "poi_index": poi_idx,
                        "mu_test": mu_test,
                        "cls": cls,
                        "clsb": clsb,
                        "clb": clb,
                        "q_obs": q_obs,
                        "mu_hat": mu_hat,
                        "n_toys": { "b": q_b_sorted.len(), "sb": q_sb_sorted.len() },
                        "n_error": { "b": n_error_b, "sb": n_error_sb },
                        "n_nonconverged": { "b": n_nonconv_b, "sb": n_nonconv_sb },
                        "seed": seed,
                        "expected_set": expected_json,
                    });

                    let wall_time_s = start.elapsed().as_secs_f64();
                    write_json(output, &flow_output)?;
                    if let Some(path) = json_metrics {
                        let timing_extra = if timing_breakdown.is_empty() {
                            serde_json::json!({})
                        } else {
                            serde_json::json!({
                                "breakdown": serde_json::Value::Object(std::mem::take(&mut timing_breakdown)),
                            })
                        };
                        let metrics_json = metrics_v0_with_timing(
                            "unbinned_hypotest_toys",
                            "cuda",
                            threads,
                            false,
                            wall_time_s,
                            timing_extra,
                        );
                        write_json(Some(path), &metrics_json)?;
                    }
                    return Ok(());
                }

                let (_meta, static_models) =
                    unbinned_gpu::build_gpu_static_datas(&spec, config.as_path())?;

                let included_channels: Vec<usize> = spec
                    .channels
                    .iter()
                    .enumerate()
                    .filter(|(_, ch)| ch.include_in_fit)
                    .map(|(i, _)| i)
                    .collect();
                if included_channels.is_empty() {
                    anyhow::bail!(
                        "unbinned-hypotest-toys requires at least one channel with include_in_fit=true"
                    );
                }
                if included_channels.len() != static_models.len() {
                    anyhow::bail!(
                        "internal error: included_channels ({}) != static_models ({})",
                        included_channels.len(),
                        static_models.len()
                    );
                }
                let obs_names: Vec<String> = included_channels
                    .iter()
                    .map(|&i| spec.channels[i].observables[0].name.clone())
                    .collect();

                // Phase A: baseline CPU fits (3 fits, not bottleneck)
                let t_obs_fits0 = std::time::Instant::now();
                let init0 = model.parameter_init();
                let free_obs = mle.fit_minimum_from(&model, &init0)?;
                if !free_obs.converged {
                    anyhow::bail!("Observed-data free fit did not converge: {}", free_obs.message);
                }
                let free_nll = free_obs.fval;
                let mu_hat = free_obs.parameters[poi_idx];

                let bounds = model.parameter_bounds();
                let mut bounds_fixed = bounds.clone();
                bounds_fixed[poi_idx] = (mu_test, mu_test);
                let mut bounds_mu0 = bounds.clone();
                bounds_mu0[poi_idx] = (0.0, 0.0);

                let fixed_mu_model = model.with_fixed_param(poi_idx, mu_test);
                let mut warm_mu = free_obs.parameters.clone();
                warm_mu[poi_idx] = mu_test;
                let fixed_mu = mle.fit_minimum_from(&fixed_mu_model, &warm_mu)?;
                if !fixed_mu.converged {
                    anyhow::bail!(
                        "Failed to fit generation point at mu={}: {}",
                        mu_test,
                        fixed_mu.message
                    );
                }

                let fixed_0_model = model.with_fixed_param(poi_idx, 0.0);
                let mut warm_0 = free_obs.parameters.clone();
                warm_0[poi_idx] = 0.0;
                let fixed_0 = mle.fit_minimum_from(&fixed_0_model, &warm_0)?;
                if !fixed_0.converged {
                    anyhow::bail!("Failed to fit generation point at mu=0: {}", fixed_0.message);
                }
                timing_breakdown.insert(
                    "obs_fits_s".into(),
                    serde_json::json!(t_obs_fits0.elapsed().as_secs_f64()),
                );

                let q_obs = if mu_hat > mu_test {
                    0.0
                } else {
                    (2.0 * (fixed_mu.fval - free_nll)).max(0.0)
                };

                let n_channels = static_models.len();

                // Host sampling path (CPU; exact semantics). Multi-channel toys are returned
                // as per-channel flattened buffers + prefix-sum offsets.
                let sample_flat_host = |gen_params: &[f64],
                                        seed0: u64,
                                        n_toys_local: usize|
                 -> Result<(Vec<Vec<u32>>, Vec<Vec<f64>>)> {
                    let per_toy: Vec<Result<Vec<Vec<f64>>>> = (0..n_toys_local)
                        .into_par_iter()
                        .with_min_len(16)
                        .map(|toy_idx| {
                            let toy_seed = seed0.wrapping_add(toy_idx as u64);
                            let toy_model = model.sample_poisson_toy(gen_params, toy_seed)?;
                            let mut out = Vec::<Vec<f64>>::with_capacity(n_channels);
                            for (out_idx, &ch_idx) in included_channels.iter().enumerate() {
                                let ch = toy_model.channels().get(ch_idx).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy model missing channel index {} (unexpected)",
                                        ch_idx
                                    )
                                })?;
                                let obs_name = obs_names.get(out_idx).ok_or_else(|| {
                                    anyhow::anyhow!("internal error: missing obs_names[{out_idx}]")
                                })?;
                                let xs = ch.data.column(obs_name.as_str()).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy channel '{}' data missing observable column '{}'",
                                        ch.name,
                                        obs_name
                                    )
                                })?;
                                out.push(xs.to_vec());
                            }
                            Ok(out)
                        })
                        .collect();

                    let mut toy_offsets_by_channel: Vec<Vec<u32>> =
                        (0..n_channels).map(|_| Vec::with_capacity(n_toys_local + 1)).collect();
                    for offs in &mut toy_offsets_by_channel {
                        offs.push(0u32);
                    }
                    let mut obs_flat_by_channel: Vec<Vec<f64>> =
                        (0..n_channels).map(|_| Vec::new()).collect();

                    for r in per_toy {
                        let xs_by_ch = r?;
                        if xs_by_ch.len() != n_channels {
                            anyhow::bail!(
                                "internal error: per-toy channel count mismatch: expected {n_channels}, got {}",
                                xs_by_ch.len()
                            );
                        }
                        for (ch_i, xs) in xs_by_ch.into_iter().enumerate() {
                            obs_flat_by_channel[ch_i].extend_from_slice(&xs);
                            toy_offsets_by_channel[ch_i].push(checked_len_to_u32(
                                obs_flat_by_channel[ch_i].len(),
                                "cpu toy sampling offset overflow (hypotest cuda path)",
                            )?);
                        }
                    }
                    Ok((toy_offsets_by_channel, obs_flat_by_channel))
                };

                #[derive(Clone, Copy)]
                struct ToyCounts {
                    n_ge: usize,
                    n_valid: usize,
                    n_error: usize,
                    n_nonconverged: usize,
                }

                #[derive(Default, Clone, Copy)]
                struct BatchTiming {
                    build_s: f64,
                    free_fit_s: f64,
                    fixed_fit_s: f64,
                }

                fn count_q_ge_ensemble_cuda(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    obs_flat_by_channel: &[Vec<f64>],
                    device_ids: &[usize],
                    poi: usize,
                    mu_test: f64,
                    q_obs: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                ) -> Result<(ToyCounts, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, maybe_accels, build_s) = if device_ids.len() == 1 {
                        let t_build0 = std::time::Instant::now();
                        let mut accels = Vec::with_capacity(static_models.len());
                        for (i, ((m, offs), obs)) in static_models
                            .iter()
                            .zip(toy_offsets_by_channel.iter())
                            .zip(obs_flat_by_channel.iter())
                            .enumerate()
                        {
                            let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_on_device(
                                m,
                                offs,
                                obs,
                                n_toys,
                                device_ids[0],
                            )
                            .map_err(|e| {
                                ns_core::Error::Validation(format!(
                                    "gpu {} channel {i}: {e}",
                                    device_ids[0]
                                ))
                            })?;
                            accels.push(accel);
                        }
                        let build_s = t_build0.elapsed().as_secs_f64();
                        let (free_fits, accels) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                accels, init_free, bounds, None,
                            )?;
                        (free_fits, Some(accels), build_s)
                    } else {
                        let free_fits =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_multi_gpu_host(
                                static_models,
                                toy_offsets_by_channel,
                                obs_flat_by_channel,
                                device_ids,
                                init_free,
                                bounds,
                                None,
                            )?;
                        (free_fits, None, 0.0)
                    };
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let fixed_fits = if let Some(accels) = maybe_accels {
                        let (fixed_fits, _accels) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                accels,
                                &init_fixed,
                                bounds_fixed,
                                None,
                            )?;
                        fixed_fits
                    } else {
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_multi_gpu_host(
                            static_models,
                            toy_offsets_by_channel,
                            obs_flat_by_channel,
                            device_ids,
                            &init_fixed,
                            bounds_fixed,
                            None,
                        )?
                    };
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut n_ge = 0usize;
                    let mut n_valid = 0usize;
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let q = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if q.is_finite() {
                                    n_valid += 1;
                                    if q >= q_obs {
                                        n_ge += 1;
                                    }
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        ToyCounts { n_ge, n_valid, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                fn count_q_ge_ensemble_cuda_device(
                    ctx: std::sync::Arc<ns_compute::cuda_driver::CudaContext>,
                    stream: std::sync::Arc<ns_compute::cuda_driver::CudaStream>,
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    d_obs_flat_by_channel: Vec<ns_compute::cuda_driver::CudaSlice<f64>>,
                    poi: usize,
                    mu_test: f64,
                    q_obs: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                ) -> Result<(ToyCounts, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_build0 = std::time::Instant::now();
                    let mut accels = Vec::with_capacity(static_models.len());
                    for (i, (m, (offs, d_obs))) in static_models
                        .iter()
                        .zip(toy_offsets_by_channel.iter().zip(d_obs_flat_by_channel.into_iter()))
                        .enumerate()
                    {
                        let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
                            ctx.clone(),
                            stream.clone(),
                            m,
                            offs,
                            d_obs,
                            n_toys,
                        )
                        .map_err(|e| ns_core::Error::Validation(format!("channel {i}: {e}")))?;
                        accels.push(accel);
                    }
                    let build_s = t_build0.elapsed().as_secs_f64();

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                            accels, init_free, bounds, None,
                        )?;
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let (fixed_fits, _accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                            accels,
                            &init_fixed,
                            bounds_fixed,
                            None,
                        )?;
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut n_ge = 0usize;
                    let mut n_valid = 0usize;
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let q = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if q.is_finite() {
                                    n_valid += 1;
                                    if q >= q_obs {
                                        n_ge += 1;
                                    }
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        ToyCounts { n_ge, n_valid, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                #[derive(Debug, Clone)]
                struct QEnsemble {
                    q: Vec<f64>,
                    n_error: usize,
                    n_nonconverged: usize,
                }

                fn generate_q_ensemble_cuda(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    obs_flat_by_channel: &[Vec<f64>],
                    device_ids: &[usize],
                    poi: usize,
                    mu_test: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                ) -> Result<(QEnsemble, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, maybe_accels, build_s) = if device_ids.len() == 1 {
                        let t_build0 = std::time::Instant::now();
                        let mut accels = Vec::with_capacity(static_models.len());
                        for (i, ((m, offs), obs)) in static_models
                            .iter()
                            .zip(toy_offsets_by_channel.iter())
                            .zip(obs_flat_by_channel.iter())
                            .enumerate()
                        {
                            let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_on_device(
                                m,
                                offs,
                                obs,
                                n_toys,
                                device_ids[0],
                            )
                            .map_err(|e| {
                                ns_core::Error::Validation(format!(
                                    "gpu {} channel {i}: {e}",
                                    device_ids[0]
                                ))
                            })?;
                            accels.push(accel);
                        }
                        let build_s = t_build0.elapsed().as_secs_f64();
                        let (free_fits, accels) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                accels, init_free, bounds, None,
                            )?;
                        (free_fits, Some(accels), build_s)
                    } else {
                        let free_fits =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_multi_gpu_host(
                                static_models,
                                toy_offsets_by_channel,
                                obs_flat_by_channel,
                                device_ids,
                                init_free,
                                bounds,
                                None,
                            )?;
                        (free_fits, None, 0.0)
                    };
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let fixed_fits = if let Some(accels) = maybe_accels {
                        let (fixed_fits, _accels) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                accels,
                                &init_fixed,
                                bounds_fixed,
                                None,
                            )?;
                        fixed_fits
                    } else {
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_multi_gpu_host(
                            static_models,
                            toy_offsets_by_channel,
                            obs_flat_by_channel,
                            device_ids,
                            &init_fixed,
                            bounds_fixed,
                            None,
                        )?
                    };
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut q = Vec::with_capacity(n_toys);
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let qq = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if qq.is_finite() {
                                    q.push(qq);
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        QEnsemble { q, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                fn generate_q_ensemble_cuda_device(
                    ctx: std::sync::Arc<ns_compute::cuda_driver::CudaContext>,
                    stream: std::sync::Arc<ns_compute::cuda_driver::CudaStream>,
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    toy_offsets_by_channel: &[Vec<u32>],
                    d_obs_flat_by_channel: Vec<ns_compute::cuda_driver::CudaSlice<f64>>,
                    poi: usize,
                    mu_test: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                ) -> Result<(QEnsemble, BatchTiming)> {
                    let n_toys = toy_offsets_by_channel
                        .first()
                        .map(|o| o.len().saturating_sub(1))
                        .unwrap_or(0);

                    let t_build0 = std::time::Instant::now();
                    let mut accels = Vec::with_capacity(static_models.len());
                    for (i, (m, (offs, d_obs))) in static_models
                        .iter()
                        .zip(toy_offsets_by_channel.iter().zip(d_obs_flat_by_channel.into_iter()))
                        .enumerate()
                    {
                        let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
                            ctx.clone(),
                            stream.clone(),
                            m,
                            offs,
                            d_obs,
                            n_toys,
                        )
                        .map_err(|e| ns_core::Error::Validation(format!("channel {i}: {e}")))?;
                        accels.push(accel);
                    }
                    let build_s = t_build0.elapsed().as_secs_f64();

                    let t_free0 = std::time::Instant::now();
                    let (free_fits, accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                            accels, init_free, bounds, None,
                        )?;
                    let free_fit_s = t_free0.elapsed().as_secs_f64();

                    let mut init_fixed = init_free.to_vec();
                    init_fixed[poi] = mu_test;
                    let t_fixed0 = std::time::Instant::now();
                    let (fixed_fits, _accels) =
                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                            accels,
                            &init_fixed,
                            bounds_fixed,
                            None,
                        )?;
                    let fixed_fit_s = t_fixed0.elapsed().as_secs_f64();

                    let mut q = Vec::with_capacity(n_toys);
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;

                    for (fr, fxr) in free_fits.iter().zip(fixed_fits.iter()) {
                        match (fr, fxr) {
                            (Ok(free), Ok(fixed)) => {
                                let mu_hat = free.parameters[poi];
                                let qq = if mu_hat > mu_test {
                                    0.0
                                } else {
                                    (2.0 * (fixed.nll - free.nll)).max(0.0)
                                };
                                if qq.is_finite() {
                                    q.push(qq);
                                } else {
                                    n_error += 1;
                                }
                                let conv = if mu_hat > mu_test {
                                    free.converged
                                } else {
                                    free.converged && fixed.converged
                                };
                                if !conv {
                                    n_nonconverged += 1;
                                }
                            }
                            _ => {
                                n_error += 1;
                            }
                        }
                    }

                    Ok((
                        QEnsemble { q, n_error, n_nonconverged },
                        BatchTiming { build_s, free_fit_s, fixed_fit_s },
                    ))
                }

                fn count_q_ge_ensemble_cuda_device_sharded(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    device_shard_plan: &[usize],
                    n_toys: usize,
                    seed0: u64,
                    gen_params: &[f64],
                    poi: usize,
                    mu_test: f64,
                    q_obs: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                ) -> Result<(ToyCounts, BatchTiming, f64)> {
                    // PF3.1-OPT2: one worker per unique device; each worker
                    // processes its assigned shards sequentially with a reused
                    // CUDA context/stream/sampler set.
                    let shards = contiguous_toy_shards(n_toys, device_shard_plan);
                    let mut shards_by_device: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
                    for (toy_start, toy_end, device_id) in shards {
                        if let Some((_, device_shards)) =
                            shards_by_device.iter_mut().find(|(d, _)| *d == device_id)
                        {
                            device_shards.push((toy_start, toy_end));
                        } else {
                            shards_by_device.push((device_id, vec![(toy_start, toy_end)]));
                        }
                    }

                    let device_results: Vec<Result<(ToyCounts, BatchTiming, f64)>> =
                        std::thread::scope(|scope| {
                            let handles: Vec<_> = shards_by_device
                                .into_iter()
                                .map(|(device_id, device_shards)| {
                                    scope.spawn(move || {
                                        let mut samplers = Vec::<
                                            ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler,
                                        >::with_capacity(
                                            static_models.len()
                                        );
                                        let s0 = ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::from_unbinned_static_on_device(
                                            &static_models[0],
                                            device_id,
                                        )?;
                                        let ctx = s0.context().clone();
                                        let stream = s0.stream().clone();
                                        samplers.push(s0);
                                        for ch_i in 1..static_models.len() {
                                            samplers.push(
                                                ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::with_context(
                                                    ctx.clone(),
                                                    stream.clone(),
                                                    &static_models[ch_i],
                                                )?,
                                            );
                                        }

                                        let mut counts = ToyCounts {
                                            n_ge: 0,
                                            n_valid: 0,
                                            n_error: 0,
                                            n_nonconverged: 0,
                                        };
                                        let mut timing = BatchTiming::default();
                                        let mut sample_s_total = 0.0f64;

                                        for (toy_start, toy_end) in device_shards {
                                            let shard_n_toys = toy_end - toy_start;
                                            let t_sample0 = std::time::Instant::now();
                                            let mut toy_offsets_by_channel =
                                                Vec::<Vec<u32>>::with_capacity(samplers.len());
                                            let mut d_obs_flat_by_channel = Vec::<
                                                ns_compute::cuda_driver::CudaSlice<f64>,
                                            >::with_capacity(
                                                samplers.len(),
                                            );
                                            for (ch_i, sampler) in samplers.iter_mut().enumerate() {
                                                let seed_ch = seed0
                                                    .wrapping_add(toy_start as u64)
                                                    .wrapping_add(1_000_000_003u64.wrapping_mul(ch_i as u64));
                                                let (offs, d_obs) = sampler.sample_toys_1d_device(
                                                    gen_params,
                                                    shard_n_toys,
                                                    seed_ch,
                                                )?;
                                                toy_offsets_by_channel.push(offs);
                                                d_obs_flat_by_channel.push(d_obs);
                                            }
                                            sample_s_total += t_sample0.elapsed().as_secs_f64();

                                            let (c, t) = count_q_ge_ensemble_cuda_device(
                                                ctx.clone(),
                                                stream.clone(),
                                                static_models,
                                                &toy_offsets_by_channel,
                                                d_obs_flat_by_channel,
                                                poi,
                                                mu_test,
                                                q_obs,
                                                init_free,
                                                bounds,
                                                bounds_fixed,
                                            )?;
                                            counts.n_ge += c.n_ge;
                                            counts.n_valid += c.n_valid;
                                            counts.n_error += c.n_error;
                                            counts.n_nonconverged += c.n_nonconverged;
                                            timing.build_s += t.build_s;
                                            timing.free_fit_s += t.free_fit_s;
                                            timing.fixed_fit_s += t.fixed_fit_s;
                                        }

                                        Ok((counts, timing, sample_s_total))
                                    })
                                })
                                .collect();

                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        });

                    let mut counts =
                        ToyCounts { n_ge: 0, n_valid: 0, n_error: 0, n_nonconverged: 0 };
                    let mut timing = BatchTiming::default();
                    let mut sample_s_total = 0.0f64;
                    for sr in device_results {
                        let (c, t, ss) = sr?;
                        counts.n_ge += c.n_ge;
                        counts.n_valid += c.n_valid;
                        counts.n_error += c.n_error;
                        counts.n_nonconverged += c.n_nonconverged;
                        timing.build_s += t.build_s;
                        timing.free_fit_s += t.free_fit_s;
                        timing.fixed_fit_s += t.fixed_fit_s;
                        sample_s_total += ss;
                    }

                    Ok((counts, timing, sample_s_total))
                }

                fn generate_q_ensemble_cuda_device_sharded(
                    static_models: &[ns_compute::unbinned_types::UnbinnedGpuModelData],
                    device_shard_plan: &[usize],
                    n_toys: usize,
                    seed0: u64,
                    gen_params: &[f64],
                    poi: usize,
                    mu_test: f64,
                    init_free: &[f64],
                    bounds: &[(f64, f64)],
                    bounds_fixed: &[(f64, f64)],
                ) -> Result<(QEnsemble, BatchTiming, f64)> {
                    // PF3.1-OPT2: one worker per unique device; each worker
                    // processes its assigned shards sequentially with a reused
                    // CUDA context/stream/sampler set.
                    let shards = contiguous_toy_shards(n_toys, device_shard_plan);
                    let mut shards_by_device: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
                    for (toy_start, toy_end, device_id) in shards {
                        if let Some((_, device_shards)) =
                            shards_by_device.iter_mut().find(|(d, _)| *d == device_id)
                        {
                            device_shards.push((toy_start, toy_end));
                        } else {
                            shards_by_device.push((device_id, vec![(toy_start, toy_end)]));
                        }
                    }

                    struct ShardQResult {
                        toy_start: usize,
                        q: QEnsemble,
                        timing: BatchTiming,
                        sample_s: f64,
                    }

                    let device_results: Vec<Result<Vec<ShardQResult>>> = std::thread::scope(
                        |scope| {
                            let handles: Vec<_> = shards_by_device
                                .into_iter()
                                .map(|(device_id, device_shards)| {
                                    scope.spawn(move || {
                                        let mut samplers = Vec::<
                                            ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler,
                                        >::with_capacity(
                                            static_models.len()
                                        );
                                        let s0 = ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::from_unbinned_static_on_device(
                                            &static_models[0],
                                            device_id,
                                        )?;
                                        let ctx = s0.context().clone();
                                        let stream = s0.stream().clone();
                                        samplers.push(s0);
                                        for ch_i in 1..static_models.len() {
                                            samplers.push(
                                                ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::with_context(
                                                    ctx.clone(),
                                                    stream.clone(),
                                                    &static_models[ch_i],
                                                )?,
                                            );
                                        }

                                        let mut shard_results =
                                            Vec::<ShardQResult>::with_capacity(device_shards.len());
                                        for (toy_start, toy_end) in device_shards {
                                            let shard_n_toys = toy_end - toy_start;
                                            let t_sample0 = std::time::Instant::now();
                                            let mut toy_offsets_by_channel =
                                                Vec::<Vec<u32>>::with_capacity(samplers.len());
                                            let mut d_obs_flat_by_channel = Vec::<
                                                ns_compute::cuda_driver::CudaSlice<f64>,
                                            >::with_capacity(
                                                samplers.len(),
                                            );
                                            for (ch_i, sampler) in samplers.iter_mut().enumerate() {
                                                let seed_ch = seed0
                                                    .wrapping_add(toy_start as u64)
                                                    .wrapping_add(1_000_000_003u64.wrapping_mul(ch_i as u64));
                                                let (offs, d_obs) = sampler.sample_toys_1d_device(
                                                    gen_params,
                                                    shard_n_toys,
                                                    seed_ch,
                                                )?;
                                                toy_offsets_by_channel.push(offs);
                                                d_obs_flat_by_channel.push(d_obs);
                                            }
                                            let sample_s = t_sample0.elapsed().as_secs_f64();

                                            let (q, t) = generate_q_ensemble_cuda_device(
                                                ctx.clone(),
                                                stream.clone(),
                                                static_models,
                                                &toy_offsets_by_channel,
                                                d_obs_flat_by_channel,
                                                poi,
                                                mu_test,
                                                init_free,
                                                bounds,
                                                bounds_fixed,
                                            )?;
                                            shard_results.push(ShardQResult {
                                                toy_start,
                                                q,
                                                timing: t,
                                                sample_s,
                                            });
                                        }
                                        Ok(shard_results)
                                    })
                                })
                                .collect();

                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        },
                    );

                    let mut shard_results_flat = Vec::<ShardQResult>::new();
                    for dr in device_results {
                        shard_results_flat.extend(dr?);
                    }
                    shard_results_flat.sort_by_key(|r| r.toy_start);

                    let mut q_all = Vec::<f64>::with_capacity(n_toys);
                    let mut n_error = 0usize;
                    let mut n_nonconverged = 0usize;
                    let mut timing = BatchTiming::default();
                    let mut sample_s_total = 0.0f64;
                    for sr in shard_results_flat {
                        q_all.extend(sr.q.q);
                        n_error += sr.q.n_error;
                        n_nonconverged += sr.q.n_nonconverged;
                        timing.build_s += sr.timing.build_s;
                        timing.free_fit_s += sr.timing.free_fit_s;
                        timing.fixed_fit_s += sr.timing.fixed_fit_s;
                        sample_s_total += sr.sample_s;
                    }

                    Ok((QEnsemble { q: q_all, n_error, n_nonconverged }, timing, sample_s_total))
                }

                let seed_b = seed;
                let seed_sb = seed.wrapping_add(1_000_000_000u64);

                let run_count_cuda_host_sharded =
                    |seed0: u64,
                     gen_params: &[f64],
                     init_free: &[f64]|
                     -> Result<(ToyCounts, BatchTiming, f64)> {
                        let mut counts =
                            ToyCounts { n_ge: 0, n_valid: 0, n_error: 0, n_nonconverged: 0 };
                        let mut timing = BatchTiming::default();
                        let mut sample_s_total = 0.0f64;

                        for (toy_start, toy_end, device_id) in
                            contiguous_toy_shards(n_toys, &cuda_device_shard_plan)
                        {
                            let shard_n_toys = toy_end - toy_start;
                            let t_sample0 = std::time::Instant::now();
                            let (toy_offsets, obs_flat) = sample_flat_host(
                                gen_params,
                                seed0.wrapping_add(toy_start as u64),
                                shard_n_toys,
                            )?;
                            sample_s_total += t_sample0.elapsed().as_secs_f64();

                            let shard_device_ids = [device_id];
                            let (c, t) = count_q_ge_ensemble_cuda(
                                &static_models,
                                &toy_offsets,
                                &obs_flat,
                                &shard_device_ids,
                                poi_idx,
                                mu_test,
                                q_obs,
                                init_free,
                                &bounds,
                                &bounds_fixed,
                            )?;
                            counts.n_ge += c.n_ge;
                            counts.n_valid += c.n_valid;
                            counts.n_error += c.n_error;
                            counts.n_nonconverged += c.n_nonconverged;
                            timing.build_s += t.build_s;
                            timing.free_fit_s += t.free_fit_s;
                            timing.fixed_fit_s += t.fixed_fit_s;
                        }

                        Ok((counts, timing, sample_s_total))
                    };

                let run_generate_cuda_host_sharded =
                    |seed0: u64,
                     gen_params: &[f64],
                     init_free: &[f64]|
                     -> Result<(QEnsemble, BatchTiming, f64)> {
                        let mut q_all = Vec::<f64>::with_capacity(n_toys);
                        let mut n_error = 0usize;
                        let mut n_nonconverged = 0usize;
                        let mut timing = BatchTiming::default();
                        let mut sample_s_total = 0.0f64;

                        for (toy_start, toy_end, device_id) in
                            contiguous_toy_shards(n_toys, &cuda_device_shard_plan)
                        {
                            let shard_n_toys = toy_end - toy_start;
                            let t_sample0 = std::time::Instant::now();
                            let (toy_offsets, obs_flat) = sample_flat_host(
                                gen_params,
                                seed0.wrapping_add(toy_start as u64),
                                shard_n_toys,
                            )?;
                            sample_s_total += t_sample0.elapsed().as_secs_f64();

                            let shard_device_ids = [device_id];
                            let (q, t) = generate_q_ensemble_cuda(
                                &static_models,
                                &toy_offsets,
                                &obs_flat,
                                &shard_device_ids,
                                poi_idx,
                                mu_test,
                                init_free,
                                &bounds,
                                &bounds_fixed,
                            )?;
                            q_all.extend(q.q);
                            n_error += q.n_error;
                            n_nonconverged += q.n_nonconverged;
                            timing.build_s += t.build_s;
                            timing.free_fit_s += t.free_fit_s;
                            timing.fixed_fit_s += t.fixed_fit_s;
                        }

                        Ok((
                            QEnsemble { q: q_all, n_error, n_nonconverged },
                            timing,
                            sample_s_total,
                        ))
                    };

                if !expected_set {
                    let (eb, esb) = if gpu_sample_toys {
                        if static_models.is_empty() {
                            anyhow::bail!(
                                "unbinned-hypotest-toys --gpu-sample-toys requires at least one included channel"
                            );
                        }
                        let (eb, t_b, sample_b_s) = count_q_ge_ensemble_cuda_device_sharded(
                            &static_models,
                            &cuda_device_shard_plan,
                            n_toys,
                            seed_b,
                            &fixed_0.parameters,
                            poi_idx,
                            mu_test,
                            q_obs,
                            &fixed_0.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        let (esb, t_sb, sample_sb_s) = count_q_ge_ensemble_cuda_device_sharded(
                            &static_models,
                            &cuda_device_shard_plan,
                            n_toys,
                            seed_sb,
                            &fixed_mu.parameters,
                            poi_idx,
                            mu_test,
                            q_obs,
                            &fixed_mu.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        timing_breakdown.insert(
                            "toys".into(),
                            serde_json::json!({
                                "pipeline": if cuda_device_shard_plan.len() > 1 { "cuda_device_sharded" } else { "cuda_device" },
                                "device_ids": &cuda_device_ids,
                                "device_shard_plan": &cuda_device_shard_plan,
                                "expected_set": false,
                                "b": {
                                    "sample_s": sample_b_s,
                                    "batch_build_s": t_b.build_s,
                                    "fit_free_s": t_b.free_fit_s,
                                    "fit_fixed_s": t_b.fixed_fit_s,
                                },
                                "sb": {
                                    "sample_s": sample_sb_s,
                                    "batch_build_s": t_sb.build_s,
                                    "fit_free_s": t_sb.free_fit_s,
                                    "fit_fixed_s": t_sb.fixed_fit_s,
                                },
                            }),
                        );
                        (eb, esb)
                    } else if !cuda_device_shard_plan.is_empty() {
                        let (eb, t_b, sample_b_s) = run_count_cuda_host_sharded(
                            seed_b,
                            &fixed_0.parameters,
                            &fixed_0.parameters,
                        )?;
                        let (esb, t_sb, sample_sb_s) = run_count_cuda_host_sharded(
                            seed_sb,
                            &fixed_mu.parameters,
                            &fixed_mu.parameters,
                        )?;
                        timing_breakdown.insert(
                            "toys".into(),
                            serde_json::json!({
                                "pipeline": "cuda_host_sharded",
                                "device_ids": &cuda_device_ids,
                                "device_shard_plan": &cuda_device_shard_plan,
                                "expected_set": false,
                                "b": {
                                    "sample_s": sample_b_s,
                                    "batch_build_s": t_b.build_s,
                                    "fit_free_s": t_b.free_fit_s,
                                    "fit_fixed_s": t_b.fixed_fit_s,
                                },
                                "sb": {
                                    "sample_s": sample_sb_s,
                                    "batch_build_s": t_sb.build_s,
                                    "fit_free_s": t_sb.free_fit_s,
                                    "fit_fixed_s": t_sb.fixed_fit_s,
                                },
                            }),
                        );
                        (eb, esb)
                    } else {
                        let t_sample_b0 = std::time::Instant::now();
                        let (toy_offsets_b, obs_flat_b) =
                            sample_flat_host(&fixed_0.parameters, seed_b, n_toys)?;
                        let sample_b_s = t_sample_b0.elapsed().as_secs_f64();
                        let t_sample_sb0 = std::time::Instant::now();
                        let (toy_offsets_sb, obs_flat_sb) =
                            sample_flat_host(&fixed_mu.parameters, seed_sb, n_toys)?;
                        let sample_sb_s = t_sample_sb0.elapsed().as_secs_f64();

                        let (eb, t_b) = count_q_ge_ensemble_cuda(
                            &static_models,
                            &toy_offsets_b,
                            &obs_flat_b,
                            &cuda_device_ids,
                            poi_idx,
                            mu_test,
                            q_obs,
                            &fixed_0.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        let (esb, t_sb) = count_q_ge_ensemble_cuda(
                            &static_models,
                            &toy_offsets_sb,
                            &obs_flat_sb,
                            &cuda_device_ids,
                            poi_idx,
                            mu_test,
                            q_obs,
                            &fixed_mu.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        timing_breakdown.insert(
                            "toys".into(),
                            serde_json::json!({
                                "pipeline": "host",
                                "device_ids": &cuda_device_ids,
                                "device_shard_plan": serde_json::Value::Null,
                                "expected_set": false,
                                "b": {
                                    "sample_s": sample_b_s,
                                    "batch_build_s": t_b.build_s,
                                    "fit_free_s": t_b.free_fit_s,
                                    "fit_fixed_s": t_b.fixed_fit_s,
                                },
                                "sb": {
                                    "sample_s": sample_sb_s,
                                    "batch_build_s": t_sb.build_s,
                                    "fit_free_s": t_sb.free_fit_s,
                                    "fit_fixed_s": t_sb.fixed_fit_s,
                                },
                            }),
                        );
                        (eb, esb)
                    };

                    if eb.n_valid == 0 || esb.n_valid == 0 {
                        anyhow::bail!(
                            "All toys failed: b_only_valid={} sb_valid={}",
                            eb.n_valid,
                            esb.n_valid
                        );
                    }

                    let clsb = tail_prob_counts(esb.n_ge, esb.n_valid);
                    let clb = tail_prob_counts(eb.n_ge, eb.n_valid);
                    let cls = safe_cls(clsb, clb);

                    serde_json::json!({
                        "input_schema_version": spec.schema_version,
                        "poi_index": poi_idx,
                        "mu_test": mu_test,
                        "cls": cls,
                        "clsb": clsb,
                        "clb": clb,
                        "q_obs": q_obs,
                        "mu_hat": mu_hat,
                        "n_toys": { "b": n_toys, "sb": n_toys },
                        "n_error": { "b": eb.n_error, "sb": esb.n_error },
                        "n_nonconverged": { "b": eb.n_nonconverged, "sb": esb.n_nonconverged },
                        "seed": seed,
                        "expected_set": serde_json::Value::Null,
                    })
                } else {
                    let (eb, esb) = if gpu_sample_toys {
                        if static_models.is_empty() {
                            anyhow::bail!(
                                "unbinned-hypotest-toys --gpu-sample-toys requires at least one included channel"
                            );
                        }
                        let (eb, t_b, sample_b_s) = generate_q_ensemble_cuda_device_sharded(
                            &static_models,
                            &cuda_device_shard_plan,
                            n_toys,
                            seed_b,
                            &fixed_0.parameters,
                            poi_idx,
                            mu_test,
                            &fixed_0.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        let (esb, t_sb, sample_sb_s) = generate_q_ensemble_cuda_device_sharded(
                            &static_models,
                            &cuda_device_shard_plan,
                            n_toys,
                            seed_sb,
                            &fixed_mu.parameters,
                            poi_idx,
                            mu_test,
                            &fixed_mu.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        timing_breakdown.insert(
                            "toys".into(),
                            serde_json::json!({
                                "pipeline": if cuda_device_shard_plan.len() > 1 { "cuda_device_sharded" } else { "cuda_device" },
                                "device_ids": &cuda_device_ids,
                                "device_shard_plan": &cuda_device_shard_plan,
                                "expected_set": true,
                                "b": {
                                    "sample_s": sample_b_s,
                                    "batch_build_s": t_b.build_s,
                                    "fit_free_s": t_b.free_fit_s,
                                    "fit_fixed_s": t_b.fixed_fit_s,
                                },
                                "sb": {
                                    "sample_s": sample_sb_s,
                                    "batch_build_s": t_sb.build_s,
                                    "fit_free_s": t_sb.free_fit_s,
                                    "fit_fixed_s": t_sb.fixed_fit_s,
                                },
                            }),
                        );
                        (eb, esb)
                    } else if !cuda_device_shard_plan.is_empty() {
                        let (eb, t_b, sample_b_s) = run_generate_cuda_host_sharded(
                            seed_b,
                            &fixed_0.parameters,
                            &fixed_0.parameters,
                        )?;
                        let (esb, t_sb, sample_sb_s) = run_generate_cuda_host_sharded(
                            seed_sb,
                            &fixed_mu.parameters,
                            &fixed_mu.parameters,
                        )?;
                        timing_breakdown.insert(
                            "toys".into(),
                            serde_json::json!({
                                "pipeline": "cuda_host_sharded",
                                "device_ids": &cuda_device_ids,
                                "device_shard_plan": &cuda_device_shard_plan,
                                "expected_set": true,
                                "b": {
                                    "sample_s": sample_b_s,
                                    "batch_build_s": t_b.build_s,
                                    "fit_free_s": t_b.free_fit_s,
                                    "fit_fixed_s": t_b.fixed_fit_s,
                                },
                                "sb": {
                                    "sample_s": sample_sb_s,
                                    "batch_build_s": t_sb.build_s,
                                    "fit_free_s": t_sb.free_fit_s,
                                    "fit_fixed_s": t_sb.fixed_fit_s,
                                },
                            }),
                        );
                        (eb, esb)
                    } else {
                        let t_sample_b0 = std::time::Instant::now();
                        let (toy_offsets_b, obs_flat_b) =
                            sample_flat_host(&fixed_0.parameters, seed_b, n_toys)?;
                        let sample_b_s = t_sample_b0.elapsed().as_secs_f64();
                        let t_sample_sb0 = std::time::Instant::now();
                        let (toy_offsets_sb, obs_flat_sb) =
                            sample_flat_host(&fixed_mu.parameters, seed_sb, n_toys)?;
                        let sample_sb_s = t_sample_sb0.elapsed().as_secs_f64();

                        let (eb, t_b) = generate_q_ensemble_cuda(
                            &static_models,
                            &toy_offsets_b,
                            &obs_flat_b,
                            &cuda_device_ids,
                            poi_idx,
                            mu_test,
                            &fixed_0.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        let (esb, t_sb) = generate_q_ensemble_cuda(
                            &static_models,
                            &toy_offsets_sb,
                            &obs_flat_sb,
                            &cuda_device_ids,
                            poi_idx,
                            mu_test,
                            &fixed_mu.parameters,
                            &bounds,
                            &bounds_fixed,
                        )?;
                        timing_breakdown.insert(
                            "toys".into(),
                            serde_json::json!({
                                "pipeline": "host",
                                "device_ids": &cuda_device_ids,
                                "device_shard_plan": serde_json::Value::Null,
                                "expected_set": true,
                                "b": {
                                    "sample_s": sample_b_s,
                                    "batch_build_s": t_b.build_s,
                                    "fit_free_s": t_b.free_fit_s,
                                    "fit_fixed_s": t_b.fixed_fit_s,
                                },
                                "sb": {
                                    "sample_s": sample_sb_s,
                                    "batch_build_s": t_sb.build_s,
                                    "fit_free_s": t_sb.free_fit_s,
                                    "fit_fixed_s": t_sb.fixed_fit_s,
                                },
                            }),
                        );
                        (eb, esb)
                    };

                    if eb.q.is_empty() || esb.q.is_empty() {
                        anyhow::bail!(
                            "Toy ensembles are empty after filtering errors: b_only={} sb={}",
                            eb.q.len(),
                            esb.q.len()
                        );
                    }

                    let mut q_b_sorted = eb.q.clone();
                    q_b_sorted.sort_by(|a, b| a.total_cmp(b));
                    let mut q_sb_sorted = esb.q.clone();
                    q_sb_sorted.sort_by(|a, b| a.total_cmp(b));

                    let clsb = tail_prob_sorted(&q_sb_sorted, q_obs);
                    let clb = tail_prob_sorted(&q_b_sorted, q_obs);
                    let cls = safe_cls(clsb, clb);

                    let mut cls_vals: Vec<f64> = Vec::with_capacity(q_b_sorted.len());
                    for &q_val in &q_b_sorted {
                        let clsb_q = tail_prob_sorted(&q_sb_sorted, q_val);
                        let clb_q = tail_prob_sorted(&q_b_sorted, q_val);
                        cls_vals.push(safe_cls(clsb_q, clb_q));
                    }
                    cls_vals.sort_by(|a, b| a.total_cmp(b));

                    let mut expected = [0.0f64; 5];
                    for (i, t) in ns_inference::hypotest::NSIGMA_ORDER.into_iter().enumerate() {
                        let p = normal_cdf(-t);
                        expected[i] = quantile_sorted(&cls_vals, p);
                    }

                    serde_json::json!({
                        "input_schema_version": spec.schema_version,
                        "poi_index": poi_idx,
                        "mu_test": mu_test,
                        "cls": cls,
                        "clsb": clsb,
                        "clb": clb,
                        "q_obs": q_obs,
                        "mu_hat": mu_hat,
                        "n_toys": { "b": n_toys, "sb": n_toys },
                        "n_error": { "b": eb.n_error, "sb": esb.n_error },
                        "n_nonconverged": { "b": eb.n_nonconverged, "sb": esb.n_nonconverged },
                        "seed": seed,
                        "expected_set": {
                            "nsigma_order": [2, 1, 0, -1, -2],
                            "cls": expected,
                        },
                    })
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
        None => {
            if expected_set {
                let r = ns_inference::hypotest_qtilde_toys_expected_set_with_sampler(
                    &mle, &model, mu_test, n_toys, seed, sample_toy,
                )?;
                let o = &r.observed;
                serde_json::json!({
                    "input_schema_version": spec.schema_version,
                    "poi_index": poi_idx,
                    "mu_test": o.mu_test,
                    "cls": o.cls,
                    "clsb": o.clsb,
                    "clb": o.clb,
                    "q_obs": o.q_obs,
                    "mu_hat": o.mu_hat,
                    "n_toys": { "b": o.n_toys_b, "sb": o.n_toys_sb },
                    "n_error": { "b": o.n_error_b, "sb": o.n_error_sb },
                    "n_nonconverged": { "b": o.n_nonconverged_b, "sb": o.n_nonconverged_sb },
                    "seed": seed,
                    "expected_set": {
                        "nsigma_order": [2, 1, 0, -1, -2],
                        "cls": r.expected,
                    },
                })
            } else {
                let o = ns_inference::hypotest_qtilde_toys_with_sampler(
                    &mle, &model, mu_test, n_toys, seed, sample_toy,
                )?;
                serde_json::json!({
                    "input_schema_version": spec.schema_version,
                    "poi_index": poi_idx,
                    "mu_test": o.mu_test,
                    "cls": o.cls,
                    "clsb": o.clsb,
                    "clb": o.clb,
                    "q_obs": o.q_obs,
                    "mu_hat": o.mu_hat,
                    "n_toys": { "b": o.n_toys_b, "sb": o.n_toys_sb },
                    "n_error": { "b": o.n_error_b, "sb": o.n_error_sb },
                    "n_nonconverged": { "b": o.n_nonconverged_b, "sb": o.n_nonconverged_sb },
                    "seed": seed,
                    "expected_set": serde_json::Value::Null,
                })
            }
        }
    };

    let wall_time_s = start.elapsed().as_secs_f64();

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let timing_extra = if timing_breakdown.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::json!({
                "breakdown": serde_json::Value::Object(std::mem::take(&mut timing_breakdown)),
            })
        };
        let metrics_json = metrics_v0_with_timing(
            "unbinned_hypotest_toys",
            gpu.unwrap_or("cpu"),
            threads,
            false,
            wall_time_s,
            timing_extra,
            serde_json::json!({
                "poi_index": poi_idx,
                "mu_test": mu_test,
                "n_toys": n_toys,
                "seed": seed,
                "cls": output_json.get("cls").cloned().unwrap_or(serde_json::Value::Null),
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        // `report::write_bundle` expects JSON inputs (it stores them as inputs/input.json).
        // For Phase 1, require a JSON spec when bundling for clean reproducibility.
        let bytes = std::fs::read(config)?;
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "--bundle requires a JSON unbinned spec (got non-JSON at {}): {e}",
                config.display()
            )
        })?;

        report::write_bundle(
            dir,
            "unbinned_hypotest_toys",
            serde_json::json!({
                "threads": threads,
                "mu": mu_test,
                "n_toys": n_toys,
                "seed": seed,
                "expected_set": expected_set,
                "gpu": gpu.unwrap_or("cpu"),
            }),
            config,
            &output_json,
            false,
        )?;
    }

    Ok(())
}

fn cmd_unbinned_fit_toys(
    config: &PathBuf,
    n_toys: usize,
    seed: u64,
    gen_point: UnbinnedToyGenPoint,
    set: &[(String, f64)],
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    gpu: Option<&str>,
    gpu_devices: &[usize],
    gpu_shards: Option<usize>,
    gpu_sample_toys: bool,
    gpu_native: bool,
    shard: Option<(usize, usize)>,
    require_all_converged: bool,
    max_abs_poi_pull_mean: Option<f64>,
    poi_pull_std_range: Option<Vec<f64>>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    use ns_core::traits::PoiModel;

    if n_toys == 0 {
        anyhow::bail!("n_toys must be > 0");
    }

    // CPU shard mode: compute toy range for this shard.
    let (shard_start, shard_n_toys) = if let Some((shard_idx, shard_total)) = shard {
        let per = n_toys / shard_total;
        let rem = n_toys % shard_total;
        let start = shard_idx * per + shard_idx.min(rem);
        let count = per + if shard_idx < rem { 1 } else { 0 };
        if count == 0 {
            anyhow::bail!(
                "shard {shard_idx}/{shard_total}: zero toys (n_toys={n_toys} < shard_total={shard_total})"
            );
        }
        tracing::info!(
            shard_idx,
            shard_total,
            start,
            count,
            "CPU farm shard: toys [{start}..{})",
            start + count,
        );
        (start, count)
    } else {
        (0, n_toys)
    };

    if gpu_sample_toys && gpu != Some("cuda") && gpu != Some("metal") {
        anyhow::bail!("--gpu-sample-toys requires --gpu cuda|metal");
    }
    if gpu_native && gpu != Some("cuda") {
        anyhow::bail!("--gpu-native requires --gpu cuda");
    }
    let cuda_device_ids = normalize_cuda_device_ids(gpu, gpu_devices)?;

    setup_runtime(threads, false);

    let spec = unbinned_spec::read_unbinned_spec(config)?;
    let model = unbinned_spec::compile_unbinned_model(&spec, config.as_path())?;

    // PF3.1-OPT4: auto-enable gpu_sample_toys when --gpu cuda
    // and the model uses only analytical PDFs (no flow PDFs).
    // `--gpu-native` remains explicit opt-in (no auto-enable).
    let has_flow_pdfs = model
        .channels()
        .iter()
        .filter(|ch| ch.include_in_fit)
        .any(|ch| ch.processes.iter().any(|p| p.pdf.pdf_tag().is_empty()));
    let gpu_sample_toys = if !gpu_sample_toys && gpu == Some("cuda") && !has_flow_pdfs {
        tracing::info!("OPT4: auto-enabling --gpu-sample-toys (analytical PDFs, CUDA)");
        true
    } else {
        gpu_sample_toys
    };
    #[cfg(not(feature = "cuda"))]
    let _ = gpu_native;

    let parameter_names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();
    let poi_idx = model
        .poi_index()
        .ok_or_else(|| anyhow::anyhow!("unbinned-fit-toys requires model.poi in the spec"))?;

    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let mut gen_params: Vec<f64> = match gen_point {
        UnbinnedToyGenPoint::Init => model.parameters().iter().map(|p| p.init).collect(),
        UnbinnedToyGenPoint::Mle => {
            let r = mle
                .fit(&model)
                .map_err(|e| anyhow::anyhow!("failed to fit observed data for --gen mle: {e}"))?;
            r.parameters
        }
    };

    let mut overrides_json = serde_json::Map::<String, serde_json::Value>::new();
    if !set.is_empty() {
        for (name, value) in set {
            let idx = parameter_names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| anyhow::anyhow!("unknown parameter name in --set: '{name}'"))?;
            let p = &model.parameters()[idx];
            if *value < p.bounds.0 || *value > p.bounds.1 {
                anyhow::bail!(
                    "parameter '{}' override {} is outside bounds {:?}",
                    name,
                    value,
                    p.bounds
                );
            }
            gen_params[idx] = *value;
            overrides_json.insert(name.clone(), serde_json::json!(value));
        }
    }

    // PF3.1-OPT4: estimate VRAM per toy from expected yields at the toy generation point.
    let estimated_bytes_per_toy = estimate_unbinned_expected_bytes_per_toy(&model, &gen_params)?;
    let estimated_events_per_toy = estimate_unbinned_expected_events_per_toy(&model, &gen_params)?;

    let metal_max_toys_per_batch = if gpu == Some("metal") {
        metal_max_toys_per_batch_for_u32_offsets(n_toys, estimated_events_per_toy)?
    } else {
        n_toys.max(1)
    };

    #[cfg(feature = "cuda")]
    let cuda_device_shard_plan = normalize_cuda_device_shard_plan(
        gpu,
        gpu_sample_toys,
        &cuda_device_ids,
        gpu_shards,
        n_toys,
        estimated_bytes_per_toy,
        estimated_events_per_toy,
    )?;
    #[cfg(not(feature = "cuda"))]
    let _cuda_device_shard_plan = normalize_cuda_device_shard_plan(
        gpu,
        gpu_sample_toys,
        &cuda_device_ids,
        gpu_shards,
        n_toys,
        estimated_bytes_per_toy,
        estimated_events_per_toy,
    )?;

    let poi_true = gen_params[poi_idx];

    let start = std::time::Instant::now();
    let mut timing_breakdown = serde_json::Map::<String, serde_json::Value>::new();

    if let Some(v) = max_abs_poi_pull_mean
        && (!v.is_finite() || v < 0.0)
    {
        anyhow::bail!("--max-abs-poi-pull-mean must be finite and >= 0, got {v}");
    }
    if let Some([low, high]) = poi_pull_std_range.as_deref() {
        let (low, high) = (*low, *high);
        if !(low.is_finite() && high.is_finite() && low >= 0.0 && high >= 0.0 && low <= high) {
            anyhow::bail!(
                "--poi-pull-std-range expects finite LOW,HIGH with 0 <= LOW <= HIGH, got [{low}, {high}]"
            );
        }
    }
    let need_poi_sigma = max_abs_poi_pull_mean.is_some() || poi_pull_std_range.is_some();

    let mut n_validation_error = 0usize;
    let mut n_computation_error = 0usize;
    let mut n_nonconverged = 0usize;
    let mut n_converged = 0usize;

    let mut poi_hat: Vec<Option<f64>> = Vec::with_capacity(shard_n_toys);
    let mut poi_sigma: Vec<Option<f64>> = Vec::with_capacity(shard_n_toys);
    let mut nll_hat: Vec<Option<f64>> = Vec::with_capacity(shard_n_toys);
    let mut converged: Vec<Option<bool>> = Vec::with_capacity(shard_n_toys);
    let mut pulls_by_param: Vec<Vec<f64>> = vec![Vec::new(); parameter_names.len()];

    match gpu {
        None => {
            // Warm-start: when gen_point is Mle, gen_params IS θ̂ already.
            // When gen_point is Init, fit observed data for a warm-start init.
            let theta_hat: Option<Vec<f64>> = match gen_point {
                UnbinnedToyGenPoint::Mle => Some(gen_params.clone()),
                UnbinnedToyGenPoint::Init => mle.fit(&model).ok().map(|r| r.parameters),
            };

            let toy_config = ns_inference::ToyFitConfig {
                compute_hessian: need_poi_sigma,
                ..ns_inference::ToyFitConfig::default()
            };

            // Shard-aware seed: shard_start offsets the base seed so each shard
            // produces a disjoint, deterministic toy range.
            let shard_seed = seed.wrapping_add(shard_start as u64);

            let batch_result = ns_inference::fit_unbinned_toys_batch_cpu(
                &model,
                &gen_params,
                shard_n_toys,
                shard_seed,
                theta_hat.as_deref(),
                toy_config,
                |m, p, s| m.sample_poisson_toy(p, s),
            );

            let batch_sample_s = batch_result.sample_secs;
            let batch_fit_s = batch_result.fit_secs;
            let batch_warm = batch_result.theta_hat.is_some();
            let batch_nom_conv = batch_result.nominal_converged;

            for (local_idx, fit_res) in batch_result.fits.into_iter().enumerate() {
                let global_toy_idx = shard_start + local_idx;
                match fit_res {
                    Ok(r) => {
                        if r.converged {
                            n_converged += 1;
                        } else {
                            n_nonconverged += 1;
                        }
                        converged.push(Some(r.converged));
                        let mu = r.parameters.get(poi_idx).copied().filter(|v| v.is_finite());
                        let sig = r.uncertainties.get(poi_idx).copied().filter(|v| v.is_finite());
                        let nll = if r.nll.is_finite() { Some(r.nll) } else { None };
                        poi_hat.push(mu);
                        poi_sigma.push(sig);
                        nll_hat.push(nll);

                        if r.converged {
                            for pidx in 0..parameter_names.len() {
                                let Some(&hat) = r.parameters.get(pidx) else { continue };
                                let Some(&sigma) = r.uncertainties.get(pidx) else { continue };
                                if !(hat.is_finite() && sigma.is_finite() && sigma > 0.0) {
                                    continue;
                                }
                                pulls_by_param[pidx].push((hat - gen_params[pidx]) / sigma);
                            }
                        }
                    }
                    Err(e) => {
                        let toy_seed = seed.wrapping_add(global_toy_idx as u64);
                        tracing::warn!(toy_idx = global_toy_idx, seed = toy_seed, error = %e, "toy fit failed");
                        match &e {
                            ns_core::Error::Validation(_) => n_validation_error += 1,
                            _ => n_computation_error += 1,
                        }
                        poi_hat.push(None);
                        poi_sigma.push(None);
                        nll_hat.push(None);
                        converged.push(None);
                    }
                }
            }
            timing_breakdown.insert(
                "toys".into(),
                serde_json::json!({
                    "pipeline": "cpu_batch",
                    "sample_s": batch_sample_s,
                    "fit_s": batch_fit_s,
                    "warm_start": batch_warm,
                    "nominal_converged": batch_nom_conv,
                }),
            );
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available()
                {
                    anyhow::bail!("Metal is not available at runtime (no Metal device found)");
                }

                let (_meta, static_models) =
                    unbinned_gpu::build_gpu_static_datas(&spec, config.as_path())?;

                let included_channels: Vec<usize> = spec
                    .channels
                    .iter()
                    .enumerate()
                    .filter(|(_, ch)| ch.include_in_fit)
                    .map(|(i, _)| i)
                    .collect();
                if included_channels.is_empty() {
                    anyhow::bail!(
                        "unbinned-fit-toys requires at least one channel with include_in_fit=true"
                    );
                }
                if included_channels.len() != static_models.len() {
                    anyhow::bail!(
                        "internal error: included_channels ({}) != static_models ({})",
                        included_channels.len(),
                        static_models.len()
                    );
                }
                let obs_names: Vec<String> = included_channels
                    .iter()
                    .map(|&i| spec.channels[i].observables[0].name.clone())
                    .collect();

                let pipeline = if gpu_sample_toys { "metal_device" } else { "metal_host" };

                // Generate toys and fit in lockstep on GPU, chunked for large toy workloads
                // to keep 32-bit toy offsets within range on Metal.
                let n_channels = static_models.len();
                let init_params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
                let bounds: Vec<(f64, f64)> = model.parameters().iter().map(|p| p.bounds).collect();
                let n_params = init_params.len();
                let metal_batches = contiguous_toy_batches(n_toys, metal_max_toys_per_batch);
                if metal_batches.is_empty() {
                    anyhow::bail!("internal error: empty Metal toy batch plan");
                }

                let mut toy_sample_s = 0.0f64;
                let mut batch_build_s = 0.0f64;
                let mut batch_fit_s = 0.0f64;
                let mut sampler_init_s = 0.0f64;
                let mut sample_phase_detail = serde_json::Value::Null;
                let mut poi_sigma_s = 0.0f64;
                let mut fit_results_all: Vec<ns_core::Result<ns_core::FitResult>> =
                    Vec::with_capacity(n_toys);
                let mut poi_sigmas: Vec<Option<f64>> = Vec::with_capacity(n_toys);

                let estimate_poi_sigma_batch = |
                    fit_results_batch: &[ns_core::Result<ns_core::FitResult>],
                    accels_batch: &mut [ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator],
                | -> Result<Vec<Option<f64>>> {
                    let batch_n_toys = fit_results_batch.len();
                    let mut out = vec![None; batch_n_toys];
                    if !(need_poi_sigma && poi_idx < n_params) {
                        return Ok(out);
                    }

                    let (lo, hi) = bounds[poi_idx];
                    let span = (hi - lo).abs();
                    if !(span.is_finite() && span > 0.0) {
                        return Ok(out);
                    }
                    let eps = (span * 1e-3).max(1e-6);
                    let mut hat_flat = vec![0.0f64; batch_n_toys * n_params];
                    let mut nll0 = vec![f64::NAN; batch_n_toys];
                    let mut ok_fit = vec![false; batch_n_toys];
                    for t in 0..batch_n_toys {
                        let Ok(r) = fit_results_batch
                            .get(t)
                            .ok_or_else(|| anyhow::anyhow!("missing fit result for batch toy {t}"))?
                            .as_ref()
                        else {
                            continue;
                        };
                        if r.parameters.len() == n_params {
                            hat_flat[t * n_params..(t + 1) * n_params].copy_from_slice(&r.parameters);
                        }
                        nll0[t] = r.nll;
                        ok_fit[t] = r.converged && r.nll.is_finite();
                    }

                    let mut plus = hat_flat.clone();
                    let mut minus = hat_flat.clone();
                    for t in 0..batch_n_toys {
                        let idx = t * n_params + poi_idx;
                        let mu_hat = hat_flat[idx];
                        if !mu_hat.is_finite() {
                            continue;
                        }
                        plus[idx] = (mu_hat + eps).clamp(lo, hi);
                        minus[idx] = (mu_hat - eps).clamp(lo, hi);
                    }

                    let mut nll_plus = vec![0.0f64; batch_n_toys];
                    let mut nll_minus = vec![0.0f64; batch_n_toys];
                    for a in accels_batch.iter_mut() {
                        let v_plus = a.batch_nll(&plus)?;
                        let v_minus = a.batch_nll(&minus)?;
                        for (dst, x) in nll_plus.iter_mut().zip(v_plus.into_iter()) {
                            *dst += x;
                        }
                        for (dst, x) in nll_minus.iter_mut().zip(v_minus.into_iter()) {
                            *dst += x;
                        }
                    }
                    for t in 0..batch_n_toys {
                        if !ok_fit[t] {
                            continue;
                        }
                        let idx = t * n_params + poi_idx;
                        let mu_hat = hat_flat[idx];
                        if !mu_hat.is_finite() || plus[idx] == mu_hat || minus[idx] == mu_hat {
                            continue;
                        }
                        let d2 = (nll_plus[t] - 2.0 * nll0[t] + nll_minus[t]) / (eps * eps);
                        if d2.is_finite() && d2 > 0.0 {
                            out[t] = Some(1.0 / d2.sqrt());
                        }
                    }
                    Ok(out)
                };

                if gpu_sample_toys {
                    if static_models.is_empty() {
                        anyhow::bail!("--gpu-sample-toys requires at least one included channel");
                    }
                    let t_sampler_init0 = std::time::Instant::now();
                    let samplers: Vec<ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler> =
                        static_models
                            .iter()
                            .map(|m| {
                                ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler::from_unbinned_static(m)
                                    .map_err(|e| anyhow::anyhow!("{e}"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                    sampler_init_s = t_sampler_init0.elapsed().as_secs_f64();

                    let mut sample_prepare_s = 0.0f64;
                    let mut sample_counts_kernel_s = 0.0f64;
                    let mut sample_counts_readback_s = 0.0f64;
                    let mut sample_prefix_sum_s = 0.0f64;
                    let mut sample_kernel_s = 0.0f64;
                    let mut sample_host_convert_s = 0.0f64;
                    let mut sample_channel_phase =
                        vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); n_channels];

                    for (batch_start, batch_end) in metal_batches.iter().copied() {
                        let batch_n_toys = batch_end - batch_start;
                        let t_sample0 = std::time::Instant::now();
                        let mut offs_by_ch = Vec::<Vec<u32>>::with_capacity(n_channels);
                        let mut offs_bufs_by_ch =
                            Vec::<ns_compute::metal_rs::Buffer>::with_capacity(n_channels);
                        let mut bufs_by_ch =
                            Vec::<ns_compute::metal_rs::Buffer>::with_capacity(n_channels);
                        for (ch_i, sampler) in samplers.iter().enumerate() {
                            let seed_ch = seed
                                .wrapping_add(batch_start as u64)
                                .wrapping_add(1_000_000_003u64.wrapping_mul(ch_i as u64));
                            let (offs, offs_buf, buf, st) = sampler
                                .sample_toys_1d_device_timed_with_offsets(
                                    &gen_params,
                                    batch_n_toys,
                                    seed_ch,
                                )?;
                            offs_by_ch.push(offs);
                            offs_bufs_by_ch.push(offs_buf);
                            bufs_by_ch.push(buf);
                            sample_prepare_s += st.prepare_s;
                            sample_counts_kernel_s += st.counts_kernel_s;
                            sample_counts_readback_s += st.counts_readback_s;
                            sample_prefix_sum_s += st.prefix_sum_s;
                            sample_kernel_s += st.sample_kernel_s;
                            sample_host_convert_s += st.host_convert_s;
                            sample_channel_phase[ch_i].0 += st.prepare_s;
                            sample_channel_phase[ch_i].1 += st.counts_kernel_s;
                            sample_channel_phase[ch_i].2 += st.counts_readback_s;
                            sample_channel_phase[ch_i].3 += st.prefix_sum_s;
                            sample_channel_phase[ch_i].4 += st.sample_kernel_s;
                            sample_channel_phase[ch_i].5 += st.host_convert_s;
                        }
                        toy_sample_s += t_sample0.elapsed().as_secs_f64();

                        let t_build0 = std::time::Instant::now();
                        let mut accels_batch = Vec::with_capacity(n_channels);
                        for (i, (m, ((offs, offs_buf), buf))) in static_models
                            .iter()
                            .zip(
                                offs_by_ch
                                    .iter()
                                    .zip(offs_bufs_by_ch.into_iter())
                                    .zip(bufs_by_ch.into_iter()),
                            )
                            .enumerate()
                        {
                            let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device_with_offsets(
                                m, offs, offs_buf, buf, batch_n_toys,
                            )
                            .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                            accels_batch.push(accel);
                        }
                        batch_build_s += t_build0.elapsed().as_secs_f64();

                        let t_fit0 = std::time::Instant::now();
                        let (fit_results_batch, mut accels_batch) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                                accels_batch,
                                &init_params,
                                &bounds,
                                None,
                            )?;
                        batch_fit_s += t_fit0.elapsed().as_secs_f64();
                        if fit_results_batch.len() != batch_n_toys {
                            anyhow::bail!(
                                "internal error: Metal batch fit length mismatch: expected {batch_n_toys}, got {}",
                                fit_results_batch.len()
                            );
                        }

                        let t_sigma0 = std::time::Instant::now();
                        let batch_sigmas =
                            estimate_poi_sigma_batch(&fit_results_batch, &mut accels_batch)?;
                        poi_sigma_s += t_sigma0.elapsed().as_secs_f64();
                        poi_sigmas.extend(batch_sigmas);
                        fit_results_all.extend(fit_results_batch);
                    }

                    let sample_channel_detail: Vec<serde_json::Value> = sample_channel_phase
                        .into_iter()
                        .enumerate()
                        .map(
                            |(
                                ch_i,
                                (
                                    prepare_s,
                                    counts_kernel_s,
                                    counts_readback_s,
                                    prefix_sum_s,
                                    sample_kernel_s,
                                    host_convert_s,
                                ),
                            )| {
                                serde_json::json!({
                                    "channel_index": ch_i,
                                    "prepare_s": prepare_s,
                                    "counts_kernel_s": counts_kernel_s,
                                    "counts_readback_s": counts_readback_s,
                                    "prefix_sum_s": prefix_sum_s,
                                    "sample_kernel_s": sample_kernel_s,
                                    "host_convert_s": host_convert_s,
                                })
                            },
                        )
                        .collect();
                    sample_phase_detail = serde_json::json!({
                        "prepare_s": sample_prepare_s,
                        "counts_kernel_s": sample_counts_kernel_s,
                        "counts_readback_s": sample_counts_readback_s,
                        "prefix_sum_s": sample_prefix_sum_s,
                        "sample_kernel_s": sample_kernel_s,
                        "host_convert_s": sample_host_convert_s,
                        "channels": sample_channel_detail,
                    });
                } else {
                    for (batch_start, batch_end) in metal_batches.iter().copied() {
                        let batch_n_toys = batch_end - batch_start;
                        let t_sample0 = std::time::Instant::now();
                        let mut offs_by_ch: Vec<Vec<u32>> =
                            (0..n_channels).map(|_| Vec::with_capacity(batch_n_toys + 1)).collect();
                        for offs in &mut offs_by_ch {
                            offs.push(0u32);
                        }
                        let mut obs_by_ch: Vec<Vec<f64>> =
                            (0..n_channels).map(|_| Vec::new()).collect();

                        for toy_idx in 0..batch_n_toys {
                            let toy_seed =
                                seed.wrapping_add(batch_start as u64).wrapping_add(toy_idx as u64);
                            let toy_model = model.sample_poisson_toy(&gen_params, toy_seed)?;
                            for (out_idx, &ch_idx) in included_channels.iter().enumerate() {
                                let ch = toy_model.channels().get(ch_idx).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy model missing channel index {} (unexpected)",
                                        ch_idx
                                    )
                                })?;
                                let obs_name = obs_names.get(out_idx).ok_or_else(|| {
                                    anyhow::anyhow!("internal error: missing obs_names[{out_idx}]")
                                })?;
                                let xs = ch.data.column(obs_name.as_str()).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy channel '{}' data missing observable column '{}'",
                                        ch.name,
                                        obs_name
                                    )
                                })?;
                                obs_by_ch[out_idx].extend_from_slice(xs);
                                offs_by_ch[out_idx].push(checked_len_to_u32(
                                    obs_by_ch[out_idx].len(),
                                    "cpu toy sampling offset overflow (fit metal path)",
                                )?);
                            }
                        }
                        toy_sample_s += t_sample0.elapsed().as_secs_f64();

                        let t_build0 = std::time::Instant::now();
                        let mut accels_batch = Vec::with_capacity(static_models.len());
                        for (i, ((m, offs), obs)) in static_models
                            .iter()
                            .zip(offs_by_ch.iter())
                            .zip(obs_by_ch.iter())
                            .enumerate()
                        {
                            let accel = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
                                m, offs, obs, batch_n_toys,
                            )
                            .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                            accels_batch.push(accel);
                        }
                        batch_build_s += t_build0.elapsed().as_secs_f64();

                        let t_fit0 = std::time::Instant::now();
                        let (fit_results_batch, mut accels_batch) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_metal_with_accels(
                                accels_batch,
                                &init_params,
                                &bounds,
                                None,
                            )?;
                        batch_fit_s += t_fit0.elapsed().as_secs_f64();
                        if fit_results_batch.len() != batch_n_toys {
                            anyhow::bail!(
                                "internal error: Metal batch fit length mismatch: expected {batch_n_toys}, got {}",
                                fit_results_batch.len()
                            );
                        }

                        let t_sigma0 = std::time::Instant::now();
                        let batch_sigmas =
                            estimate_poi_sigma_batch(&fit_results_batch, &mut accels_batch)?;
                        poi_sigma_s += t_sigma0.elapsed().as_secs_f64();
                        poi_sigmas.extend(batch_sigmas);
                        fit_results_all.extend(fit_results_batch);
                    }
                }

                if fit_results_all.len() != n_toys || poi_sigmas.len() != n_toys {
                    anyhow::bail!(
                        "internal error: Metal merged toy result length mismatch: fits={} sigmas={} expected={n_toys}",
                        fit_results_all.len(),
                        poi_sigmas.len()
                    );
                }

                timing_breakdown.insert(
                    "toys".into(),
                    serde_json::json!({
                        "pipeline": pipeline,
                        "sampler_init_s": sampler_init_s,
                        "sample_s": toy_sample_s,
                        "batch_build_s": batch_build_s,
                        "batch_fit_s": batch_fit_s,
                        "n_batches": metal_batches.len(),
                        "max_toys_per_batch": metal_max_toys_per_batch,
                        "poi_sigma_enabled": need_poi_sigma,
                        "poi_sigma_s": poi_sigma_s,
                        "sample_phase_detail": sample_phase_detail,
                    }),
                );

                for t in 0..n_toys {
                    match fit_results_all
                        .get(t)
                        .ok_or_else(|| anyhow::anyhow!("missing fit result for toy {t}"))?
                    {
                        Ok(r) => {
                            let ok = r.converged && r.nll.is_finite();
                            if ok {
                                n_converged += 1;
                            } else {
                                n_nonconverged += 1;
                            }
                            converged.push(Some(ok));
                            let mu = r.parameters.get(poi_idx).copied().filter(|v| v.is_finite());
                            poi_hat.push(mu);
                            poi_sigma.push(poi_sigmas[t].filter(|v| v.is_finite() && *v > 0.0));
                            nll_hat.push(if r.nll.is_finite() { Some(r.nll) } else { None });

                            if ok {
                                if let (Some(hat), Some(sig)) = (mu, poi_sigmas[t]) {
                                    if sig.is_finite() && sig > 0.0 {
                                        pulls_by_param[poi_idx].push((hat - poi_true) / sig);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(toy_idx = t, error = %e, "toy fit failed (gpu metal)");
                            match &e {
                                ns_core::Error::Validation(_) => n_validation_error += 1,
                                _ => n_computation_error += 1,
                            }
                            poi_hat.push(None);
                            poi_sigma.push(None);
                            nll_hat.push(None);
                            converged.push(None);
                        }
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
                    anyhow::bail!("CUDA is not available at runtime (no device/driver)");
                }
                for &device_id in &cuda_device_ids {
                    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available_on_device(
                        device_id,
                    ) {
                        anyhow::bail!("CUDA device {device_id} is not available at runtime");
                    }
                }

                // ── G2-R1: Flow PDF CUDA path ─────────────────────────────
                let has_flows = unbinned_gpu::spec_has_flow_pdfs(&spec);
                if has_flows {
                    use ns_unbinned::pdf::UnbinnedPdf;

                    if gpu_sample_toys {
                        anyhow::bail!(
                            "--gpu-sample-toys is not supported with flow PDFs (CPU sampling only)"
                        );
                    }
                    if gpu_native {
                        anyhow::bail!(
                            "--gpu-native is not supported with flow PDFs (lockstep L-BFGS only)"
                        );
                    }

                    let included_ch_idx =
                        spec.channels.iter().position(|ch| ch.include_in_fit).ok_or_else(|| {
                            anyhow::anyhow!(
                                "unbinned-fit-toys: no channel with include_in_fit=true"
                            )
                        })?;
                    let ch = &model.channels()[included_ch_idx];
                    let n_procs = ch.processes.len();
                    let obs_names_flow: Vec<String> = ch.processes[0].pdf.observables().to_vec();
                    if obs_names_flow.len() != 1 {
                        anyhow::bail!(
                            "--gpu cuda with flow PDFs currently supports 1 observable, got {}",
                            obs_names_flow.len()
                        );
                    }
                    let obs_name = &obs_names_flow[0];
                    let (obs_lo, obs_hi) = ch.data.bounds(obs_name).ok_or_else(|| {
                        anyhow::anyhow!("missing bounds for observable '{obs_name}'")
                    })?;

                    let init_params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
                    let bounds: Vec<(f64, f64)> =
                        model.parameters().iter().map(|p| p.bounds).collect();

                    // Phase 1: Sample toys on CPU, concatenate events.
                    let t_sample0 = std::time::Instant::now();
                    let mut toy_offsets = vec![0u32];
                    let mut xs_flat: Vec<f64> = Vec::new();

                    for toy_idx in 0..n_toys {
                        let toy_seed = seed.wrapping_add(toy_idx as u64);
                        let toy_model = model.sample_poisson_toy(&gen_params, toy_seed)?;
                        let toy_ch = &toy_model.channels()[included_ch_idx];
                        if let Some(col) = toy_ch.data.column(obs_name) {
                            xs_flat.extend_from_slice(col);
                        }
                        toy_offsets.push(checked_len_to_u32(
                            xs_flat.len(),
                            "cpu flow toy sampling offset overflow (fit)",
                        )?);
                    }
                    let total_events = xs_flat.len();
                    let toy_sample_s = t_sample0.elapsed().as_secs_f64();

                    // Phase 2: Evaluate logp per process on concatenated events.
                    let t_build0 = std::time::Instant::now();
                    let obs_spec = ns_unbinned::event_store::ObservableSpec::branch(
                        obs_name.clone(),
                        (obs_lo, obs_hi),
                    );
                    let big_store = ns_unbinned::EventStore::from_columns(
                        vec![obs_spec],
                        vec![(obs_name.clone(), xs_flat)],
                        None,
                    )
                    .map_err(|e| anyhow::anyhow!("building concatenated EventStore: {e}"))?;

                    let mut logp_flat = Vec::with_capacity(n_procs * total_events);
                    for proc in &ch.processes {
                        let shape_params: Vec<f64> =
                            proc.shape_param_indices.iter().map(|&i| gen_params[i]).collect();
                        let mut logp = vec![0.0f64; total_events];
                        proc.pdf.log_prob_batch(&big_store, &shape_params, &mut logp).map_err(
                            |e| anyhow::anyhow!("process '{}' logp eval: {e}", proc.name),
                        )?;
                        logp_flat.extend_from_slice(&logp);
                    }
                    let batch_build_s = t_build0.elapsed().as_secs_f64();

                    // Phase 3: Build FlowBatchNllConfig and fit.
                    let t_fit0 = std::time::Instant::now();
                    let flow_config = unbinned_gpu::build_flow_batch_config(
                        &spec,
                        n_toys,
                        toy_offsets,
                        total_events,
                    )?;
                    let (fit_results, _accel) =
                        ns_inference::unbinned_gpu_batch::fit_flow_toys_batch_cuda(
                            &flow_config,
                            &logp_flat,
                            &init_params,
                            &bounds,
                            None,
                        )?;
                    let batch_fit_s = t_fit0.elapsed().as_secs_f64();

                    timing_breakdown.insert(
                        "toys".into(),
                        serde_json::json!({
                            "pipeline": "cuda_flow",
                            "sample_s": toy_sample_s,
                            "batch_build_s": batch_build_s,
                            "batch_fit_s": batch_fit_s,
                        }),
                    );

                    for t in 0..n_toys {
                        match fit_results
                            .get(t)
                            .ok_or_else(|| anyhow::anyhow!("missing fit result for toy {t}"))?
                        {
                            Ok(r) => {
                                let ok = r.converged && r.nll.is_finite();
                                if ok {
                                    n_converged += 1;
                                } else {
                                    n_nonconverged += 1;
                                }
                                converged.push(Some(ok));
                                let mu =
                                    r.parameters.get(poi_idx).copied().filter(|v| v.is_finite());
                                poi_hat.push(mu);
                                poi_sigma.push(None);
                                nll_hat.push(if r.nll.is_finite() { Some(r.nll) } else { None });
                            }
                            Err(e) => {
                                tracing::warn!(
                                    toy_idx = t, error = %e,
                                    "toy fit failed (gpu cuda flow)"
                                );
                                match &e {
                                    ns_core::Error::Validation(_) => n_validation_error += 1,
                                    _ => n_computation_error += 1,
                                }
                                poi_hat.push(None);
                                poi_sigma.push(None);
                                nll_hat.push(None);
                                converged.push(None);
                            }
                        }
                    }
                }
                // ── Analytical PDF CUDA path (existing) ───────────────────────
                if !has_flows {
                    let (_meta, static_models) =
                        unbinned_gpu::build_gpu_static_datas(&spec, config.as_path())?;

                    let included_channels: Vec<usize> = spec
                        .channels
                        .iter()
                        .enumerate()
                        .filter(|(_, ch)| ch.include_in_fit)
                        .map(|(i, _)| i)
                        .collect();
                    if included_channels.is_empty() {
                        anyhow::bail!(
                            "unbinned-fit-toys requires at least one channel with include_in_fit=true"
                        );
                    }
                    if included_channels.len() != static_models.len() {
                        anyhow::bail!(
                            "internal error: included_channels ({}) != static_models ({})",
                            included_channels.len(),
                            static_models.len()
                        );
                    }
                    let obs_names: Vec<String> = included_channels
                        .iter()
                        .map(|&i| spec.channels[i].observables[0].name.clone())
                        .collect();

                    // Warm-start parity with CPU path:
                    // - when generating at MLE, use `gen_params` directly;
                    // - otherwise try observed-data MLE and fall back to spec init on failure.
                    let theta_hat: Option<Vec<f64>> = match gen_point {
                        UnbinnedToyGenPoint::Mle => Some(gen_params.clone()),
                        UnbinnedToyGenPoint::Init => mle.fit(&model).ok().map(|r| r.parameters),
                    };
                    let init_params: Vec<f64> = theta_hat
                        .clone()
                        .unwrap_or_else(|| model.parameters().iter().map(|p| p.init).collect());
                    let cuda_warm_start = theta_hat.is_some();
                    let cuda_opt_cfg = ns_inference::OptimizerConfig {
                        max_iter: 5000,
                        ..ns_inference::OptimizerConfig::default()
                    };
                    let native_max_iter = 5000u32;
                    let bounds: Vec<(f64, f64)> =
                        model.parameters().iter().map(|p| p.bounds).collect();
                    let n_params = init_params.len();

                    // Device-resident path: sample on GPU → fit on GPU, obs_flat never leaves device.
                    // Host path: sample on CPU, then H2D to batch fitter.
                    let toy_sample_s: f64;
                    let batch_build_s: f64;
                    let batch_fit_s: f64;
                    let pipeline = if gpu_native {
                        if cuda_device_shard_plan.len() > 1 {
                            "cuda_gpu_native_sharded"
                        } else {
                            "cuda_gpu_native"
                        }
                    } else if gpu_sample_toys {
                        if cuda_device_shard_plan.len() > 1 {
                            "cuda_device_sharded"
                        } else {
                            "cuda_device"
                        }
                    } else if !cuda_device_shard_plan.is_empty() {
                        "cuda_host_sharded"
                    } else if cuda_device_ids.len() > 1 {
                        "cuda_host_multi_gpu"
                    } else {
                        "host"
                    };

                    let mut shard_detail: Vec<serde_json::Value> = Vec::new();

                    let (fit_results, mut accels_opt) = if gpu_sample_toys {
                        if static_models.is_empty() {
                            anyhow::bail!(
                                "--gpu-sample-toys requires at least one included channel"
                            );
                        }
                        if cuda_device_shard_plan.is_empty() {
                            anyhow::bail!(
                                "internal error: empty CUDA shard plan for --gpu-sample-toys"
                            );
                        }

                        if cuda_device_shard_plan.len() == 1 {
                            let device_id = cuda_device_shard_plan[0];
                            let mut samplers = Vec::<
                                ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler,
                            >::with_capacity(
                                static_models.len()
                            );
                            let s0 = ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::from_unbinned_static_on_device(
                            &static_models[0],
                            device_id,
                        )?;
                            let ctx = s0.context().clone();
                            let stream = s0.stream().clone();
                            samplers.push(s0);
                            for ch_i in 1..static_models.len() {
                                samplers.push(
                                ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::with_context(
                                    ctx.clone(),
                                    stream.clone(),
                                    &static_models[ch_i],
                                )?,
                            );
                            }

                            let mut toy_offsets_by_channel =
                                Vec::<Vec<u32>>::with_capacity(samplers.len());
                            let mut d_obs_flat_by_channel =
                                Vec::<ns_compute::cuda_driver::CudaSlice<f64>>::with_capacity(
                                    samplers.len(),
                                );
                            let t_sample0 = std::time::Instant::now();
                            for (ch_i, sampler) in samplers.iter_mut().enumerate() {
                                let seed_ch =
                                    seed.wrapping_add(1_000_000_003u64.wrapping_mul(ch_i as u64));
                                let (offs, d_obs) =
                                    sampler.sample_toys_1d_device(&gen_params, n_toys, seed_ch)?;
                                toy_offsets_by_channel.push(offs);
                                d_obs_flat_by_channel.push(d_obs);
                            }
                            toy_sample_s = t_sample0.elapsed().as_secs_f64();

                            let t_build0 = std::time::Instant::now();
                            let mut accels = Vec::with_capacity(static_models.len());
                            for (i, (m, (offs, d_obs))) in static_models
                                .iter()
                                .zip(
                                    toy_offsets_by_channel
                                        .iter()
                                        .zip(d_obs_flat_by_channel.into_iter()),
                                )
                                .enumerate()
                            {
                                let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
                                ctx.clone(),
                                stream.clone(),
                                m,
                                offs,
                                d_obs,
                                n_toys,
                            )
                            .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                                accels.push(accel);
                            }
                            batch_build_s = t_build0.elapsed().as_secs_f64();

                            let t_fit0 = std::time::Instant::now();
                            let (fit_results, accels) =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                accels,
                                &init_params,
                                &bounds,
                                Some(cuda_opt_cfg.clone()),
                            )?;
                            batch_fit_s = t_fit0.elapsed().as_secs_f64();
                            (fit_results, Some(accels))
                        } else {
                            // PF3.1-OPT2: Parallel device-resident sharded path.
                            // Use one worker thread per unique CUDA device and process that
                            // device's shards sequentially inside the worker. This avoids
                            // creating multiple contexts per device when shard plans repeat
                            // the same device id (e.g. [0,0,0,0] or [0,1,0,1,...]).
                            let shards = contiguous_toy_shards(n_toys, &cuda_device_shard_plan);
                            let mut shards_by_device: Vec<(usize, Vec<(usize, usize)>)> =
                                Vec::new();
                            for (toy_start, toy_end, device_id) in shards {
                                if let Some((_, device_shards)) =
                                    shards_by_device.iter_mut().find(|(d, _)| *d == device_id)
                                {
                                    device_shards.push((toy_start, toy_end));
                                } else {
                                    shards_by_device.push((device_id, vec![(toy_start, toy_end)]));
                                }
                            }
                            // Rebind owned Vecs as &[T] slices — references are Copy,
                            // so the move closures can capture them across map iterations.
                            let static_models = static_models.as_slice();
                            let gen_params = gen_params.as_slice();
                            let init_params = init_params.as_slice();
                            let bounds = bounds.as_slice();

                            struct ShardResult {
                                toy_start: usize,
                                device_id: usize,
                                n_toys: usize,
                                fit_results: Vec<ns_core::Result<ns_core::FitResult>>,
                                sample_s: f64,
                                build_s: f64,
                                fit_s: f64,
                            }

                            let shard_results: Vec<anyhow::Result<Vec<ShardResult>>> =
                                std::thread::scope(|scope| {
                                    let handles: Vec<_> = shards_by_device
                                    .into_iter()
                                    .map(|(device_id, device_shards)| {
                                        let cuda_opt_cfg = cuda_opt_cfg.clone();
                                        scope.spawn(move || {
                                            let mut samplers = Vec::<
                                                ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler,
                                            >::with_capacity(
                                                static_models.len()
                                            );
                                            let s0 = ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::from_unbinned_static_on_device(
                                                &static_models[0],
                                                device_id,
                                            ).map_err(|e| anyhow::anyhow!("gpu {device_id} sampler init: {e}"))?;
                                            let ctx = s0.context().clone();
                                            let stream = s0.stream().clone();
                                            samplers.push(s0);
                                            for ch_i in 1..static_models.len() {
                                                samplers.push(
                                                    ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler::with_context(
                                                        ctx.clone(),
                                                        stream.clone(),
                                                        &static_models[ch_i],
                                                    ).map_err(|e| anyhow::anyhow!("gpu {device_id} ch {ch_i} sampler: {e}"))?,
                                                );
                                            }

                                            let mut device_results =
                                                Vec::<ShardResult>::with_capacity(device_shards.len());
                                            for (toy_start, toy_end) in device_shards {
                                                let shard_n_toys = toy_end - toy_start;

                                                let t_sample0 = std::time::Instant::now();
                                                let mut toy_offsets_by_channel =
                                                    Vec::<Vec<u32>>::with_capacity(samplers.len());
                                                let mut d_obs_flat_by_channel =
                                                    Vec::<ns_compute::cuda_driver::CudaSlice<f64>>::with_capacity(
                                                        samplers.len(),
                                                    );
                                                for (ch_i, sampler) in samplers.iter_mut().enumerate() {
                                                    let seed_ch = seed
                                                        .wrapping_add(toy_start as u64)
                                                        .wrapping_add(1_000_000_003u64.wrapping_mul(ch_i as u64));
                                                    let (offs, d_obs) = sampler.sample_toys_1d_device(
                                                        &gen_params,
                                                        shard_n_toys,
                                                        seed_ch,
                                                    )?;
                                                    toy_offsets_by_channel.push(offs);
                                                    d_obs_flat_by_channel.push(d_obs);
                                                }
                                                let sample_s = t_sample0.elapsed().as_secs_f64();

                                                let t_build0 = std::time::Instant::now();
                                                let mut accels = Vec::with_capacity(static_models.len());
                                                for (i, (m, (offs, d_obs))) in static_models
                                                    .iter()
                                                    .zip(
                                                        toy_offsets_by_channel
                                                            .iter()
                                                            .zip(d_obs_flat_by_channel.into_iter()),
                                                    )
                                                    .enumerate()
                                                {
                                                    let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_device(
                                                        ctx.clone(),
                                                        stream.clone(),
                                                        m,
                                                        offs,
                                                        d_obs,
                                                        shard_n_toys,
                                                    )
                                                    .map_err(|e| anyhow::anyhow!("gpu {device_id} channel {i}: {e}"))?;
                                                    accels.push(accel);
                                                }
                                                let build_s = t_build0.elapsed().as_secs_f64();

                                                let t_fit0 = std::time::Instant::now();
                                                let fit_results_shard = if gpu_native {
                                                    if accels.len() == 1 {
                                                        accels[0].batch_fit_on_device(
                                                            &init_params,
                                                            &bounds,
                                                            native_max_iter,
                                                            10u32.min(16),
                                                            1e-6,
                                                            16,
                                                        )?
                                                    } else {
                                                        ns_compute::cuda_unbinned_batch::batch_fit_multi_channel_on_device(
                                                            &accels,
                                                            &init_params,
                                                            &bounds,
                                                            native_max_iter,
                                                            10u32.min(16),
                                                            1e-6,
                                                            16,
                                                        )?
                                                    }
                                                } else {
                                                    let (fit_results_shard, _accels) =
                                                        ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                                            accels,
                                                            &init_params,
                                                            &bounds,
                                                            Some(cuda_opt_cfg.clone()),
                                                        )?;
                                                    fit_results_shard
                                                };
                                                let fit_s = t_fit0.elapsed().as_secs_f64();

                                                if fit_results_shard.len() != shard_n_toys {
                                                    anyhow::bail!(
                                                        "CUDA shard result length mismatch on device {device_id}: expected {shard_n_toys}, got {}",
                                                        fit_results_shard.len()
                                                    );
                                                }

                                                device_results.push(ShardResult {
                                                    toy_start,
                                                    device_id,
                                                    n_toys: shard_n_toys,
                                                    fit_results: fit_results_shard,
                                                    sample_s,
                                                    build_s,
                                                    fit_s,
                                                });
                                            }

                                            Ok(device_results)
                                        })
                                    })
                                    .collect();

                                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                                });

                            let mut merged: Vec<Option<ns_core::Result<ns_core::FitResult>>> =
                                (0..n_toys).map(|_| None).collect();
                            let mut toy_sample_s_total = 0.0f64;
                            let mut batch_build_s_total = 0.0f64;
                            let mut batch_fit_s_total = 0.0f64;

                            for device_results in shard_results {
                                for sr in device_results? {
                                    toy_sample_s_total += sr.sample_s;
                                    batch_build_s_total += sr.build_s;
                                    batch_fit_s_total += sr.fit_s;
                                    shard_detail.push(serde_json::json!({
                                        "device_id": sr.device_id,
                                        "n_toys": sr.n_toys,
                                        "sample_s": sr.sample_s,
                                        "build_s": sr.build_s,
                                        "fit_s": sr.fit_s,
                                    }));
                                    for (local_idx, res) in sr.fit_results.into_iter().enumerate() {
                                        merged[sr.toy_start + local_idx] = Some(res);
                                    }
                                }
                            }

                            toy_sample_s = toy_sample_s_total;
                            batch_build_s = batch_build_s_total;
                            batch_fit_s = batch_fit_s_total;
                            let fit_results = merged
                                .into_iter()
                                .enumerate()
                                .map(|(toy_idx, maybe)| {
                                    maybe.unwrap_or_else(|| {
                                        Err(ns_core::Error::Computation(format!(
                                            "missing shard fit result for toy {toy_idx}"
                                        )))
                                    })
                                })
                                .collect();
                            (fit_results, None)
                        }
                    } else if !cuda_device_shard_plan.is_empty() {
                        let n_channels = static_models.len();
                        let mut toy_sample_s_total = 0.0f64;
                        let mut batch_fit_s_total = 0.0f64;
                        let mut merged: Vec<Option<ns_core::Result<ns_core::FitResult>>> =
                            (0..n_toys).map(|_| None).collect();

                        for (toy_start, toy_end, device_id) in
                            contiguous_toy_shards(n_toys, &cuda_device_shard_plan)
                        {
                            let shard_n_toys = toy_end - toy_start;
                            let mut toy_offsets_by_channel: Vec<Vec<u32>> = (0..n_channels)
                                .map(|_| Vec::with_capacity(shard_n_toys + 1))
                                .collect();
                            for offs in &mut toy_offsets_by_channel {
                                offs.push(0u32);
                            }
                            let mut obs_flat_by_channel: Vec<Vec<f64>> =
                                (0..n_channels).map(|_| Vec::new()).collect();

                            let t_sample0 = std::time::Instant::now();
                            for local_toy_idx in 0..shard_n_toys {
                                let toy_seed =
                                    seed.wrapping_add((toy_start + local_toy_idx) as u64);
                                let toy_model = model.sample_poisson_toy(&gen_params, toy_seed)?;
                                for (out_idx, &ch_idx) in included_channels.iter().enumerate() {
                                    let ch = toy_model.channels().get(ch_idx).ok_or_else(|| {
                                        anyhow::anyhow!(
                                            "toy model missing channel index {} (unexpected)",
                                            ch_idx
                                        )
                                    })?;
                                    let obs_name = obs_names.get(out_idx).ok_or_else(|| {
                                        anyhow::anyhow!(
                                            "internal error: missing obs_names[{out_idx}]"
                                        )
                                    })?;
                                    let xs = ch.data.column(obs_name.as_str()).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy channel '{}' data missing observable column '{}'",
                                        ch.name,
                                        obs_name
                                    )
                                })?;
                                    obs_flat_by_channel[out_idx].extend_from_slice(xs);
                                    toy_offsets_by_channel[out_idx]
                                        .push(checked_len_to_u32(
                                            obs_flat_by_channel[out_idx].len(),
                                            "cpu toy sampling offset overflow (fit cuda host-sharded path)",
                                        )?);
                                }
                            }
                            toy_sample_s_total += t_sample0.elapsed().as_secs_f64();

                            let t_fit0 = std::time::Instant::now();
                            let shard_device_ids = [device_id];
                            let fit_results_shard =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_multi_gpu_host(
                                &static_models,
                                &toy_offsets_by_channel,
                                &obs_flat_by_channel,
                                &shard_device_ids,
                                &init_params,
                                &bounds,
                                None,
                            )?;
                            batch_fit_s_total += t_fit0.elapsed().as_secs_f64();

                            if fit_results_shard.len() != shard_n_toys {
                                anyhow::bail!(
                                    "CUDA host shard result length mismatch on device {device_id}: expected {shard_n_toys}, got {}",
                                    fit_results_shard.len()
                                );
                            }
                            for (local_idx, res) in fit_results_shard.into_iter().enumerate() {
                                merged[toy_start + local_idx] = Some(res);
                            }
                        }

                        toy_sample_s = toy_sample_s_total;
                        batch_build_s = 0.0;
                        batch_fit_s = batch_fit_s_total;
                        let fit_results = merged
                            .into_iter()
                            .enumerate()
                            .map(|(toy_idx, maybe)| {
                                maybe.unwrap_or_else(|| {
                                    Err(ns_core::Error::Computation(format!(
                                        "missing shard fit result for toy {toy_idx}"
                                    )))
                                })
                            })
                            .collect();
                        (fit_results, None)
                    } else {
                        let n_channels = static_models.len();
                        let mut toy_offsets_by_channel: Vec<Vec<u32>> =
                            (0..n_channels).map(|_| Vec::with_capacity(n_toys + 1)).collect();
                        for offs in &mut toy_offsets_by_channel {
                            offs.push(0u32);
                        }
                        let mut obs_flat_by_channel: Vec<Vec<f64>> =
                            (0..n_channels).map(|_| Vec::new()).collect();

                        let t_sample0 = std::time::Instant::now();
                        for toy_idx in 0..n_toys {
                            let toy_seed = seed.wrapping_add(toy_idx as u64);
                            let toy_model = model.sample_poisson_toy(&gen_params, toy_seed)?;
                            for (out_idx, &ch_idx) in included_channels.iter().enumerate() {
                                let ch = toy_model.channels().get(ch_idx).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy model missing channel index {} (unexpected)",
                                        ch_idx
                                    )
                                })?;
                                let obs_name = obs_names.get(out_idx).ok_or_else(|| {
                                    anyhow::anyhow!("internal error: missing obs_names[{out_idx}]")
                                })?;
                                let xs = ch.data.column(obs_name.as_str()).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "toy channel '{}' data missing observable column '{}'",
                                        ch.name,
                                        obs_name
                                    )
                                })?;
                                obs_flat_by_channel[out_idx].extend_from_slice(xs);
                                toy_offsets_by_channel[out_idx].push(checked_len_to_u32(
                                    obs_flat_by_channel[out_idx].len(),
                                    "cpu toy sampling offset overflow (fit cuda host path)",
                                )?);
                            }
                        }
                        toy_sample_s = t_sample0.elapsed().as_secs_f64();

                        if cuda_device_ids.len() == 1 {
                            let t_build0 = std::time::Instant::now();
                            let mut accels = Vec::with_capacity(static_models.len());
                            for (i, ((m, offs), obs)) in static_models
                                .iter()
                                .zip(toy_offsets_by_channel.iter())
                                .zip(obs_flat_by_channel.iter())
                                .enumerate()
                            {
                                let accel = ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys_on_device(
                                m,
                                offs,
                                obs,
                                n_toys,
                                cuda_device_ids[0],
                            )
                            .map_err(|e| anyhow::anyhow!("channel {i}: {e}"))?;
                                accels.push(accel);
                            }
                            batch_build_s = t_build0.elapsed().as_secs_f64();

                            if gpu_native {
                                let t_fit0 = std::time::Instant::now();
                                let fit_results = if accels.len() == 1 {
                                    accels[0].batch_fit_on_device(
                                        &init_params,
                                        &bounds,
                                        native_max_iter,
                                        10u32.min(16),
                                        1e-6,
                                        16,
                                    )?
                                } else {
                                    ns_compute::cuda_unbinned_batch::batch_fit_multi_channel_on_device(
                                    &accels,
                                    &init_params,
                                    &bounds,
                                    native_max_iter,
                                    10u32.min(16),
                                    1e-6,
                                    16,
                                )?
                                };
                                batch_fit_s = t_fit0.elapsed().as_secs_f64();
                                (fit_results, None)
                            } else {
                                let t_fit0 = std::time::Instant::now();
                                let (fit_results, accels) =
                                ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_with_accels(
                                    accels,
                                    &init_params,
                                    &bounds,
                                    Some(cuda_opt_cfg.clone()),
                                )?;
                                batch_fit_s = t_fit0.elapsed().as_secs_f64();
                                (fit_results, Some(accels))
                            }
                        } else {
                            batch_build_s = 0.0;
                            let t_fit0 = std::time::Instant::now();
                            let fit_results =
                            ns_inference::unbinned_gpu_batch::fit_unbinned_toys_batch_cuda_multi_gpu_host(
                                &static_models,
                                &toy_offsets_by_channel,
                                &obs_flat_by_channel,
                                &cuda_device_ids,
                                &init_params,
                                &bounds,
                                None,
                            )?;
                            batch_fit_s = t_fit0.elapsed().as_secs_f64();
                            (fit_results, None)
                        }
                    };

                    let mut hat_flat = vec![0.0f64; n_toys * n_params];
                    let mut nll0 = vec![f64::NAN; n_toys];
                    let mut ok_fit = vec![false; n_toys];
                    for t in 0..n_toys {
                        let Ok(r) = fit_results
                            .get(t)
                            .ok_or_else(|| anyhow::anyhow!("missing fit result for toy {t}"))?
                            .as_ref()
                        else {
                            continue;
                        };
                        if r.parameters.len() == n_params {
                            hat_flat[t * n_params..(t + 1) * n_params]
                                .copy_from_slice(&r.parameters);
                        }
                        nll0[t] = r.nll;
                        ok_fit[t] = r.converged && r.nll.is_finite();
                    }

                    let mut poi_sigma_s = 0.0f64;
                    let mut poi_sigmas: Vec<Option<f64>> = vec![None; n_toys];
                    if need_poi_sigma
                        && poi_idx < n_params
                        && let Some(accels) = accels_opt.as_mut()
                    {
                        let t_poi_sigma0 = std::time::Instant::now();
                        let (lo, hi) = bounds[poi_idx];
                        let span = (hi - lo).abs();
                        if span.is_finite() && span > 0.0 {
                            let eps = (span * 1e-3).max(1e-6);
                            let mut plus = hat_flat.clone();
                            let mut minus = hat_flat.clone();
                            for t in 0..n_toys {
                                let idx = t * n_params + poi_idx;
                                let mu_hat = hat_flat[idx];
                                if !mu_hat.is_finite() {
                                    continue;
                                }
                                plus[idx] = (mu_hat + eps).clamp(lo, hi);
                                minus[idx] = (mu_hat - eps).clamp(lo, hi);
                            }
                            let mut nll_plus = vec![0.0f64; n_toys];
                            let mut nll_minus = vec![0.0f64; n_toys];
                            for a in accels.iter_mut() {
                                let v_plus = a.batch_nll(&plus)?;
                                let v_minus = a.batch_nll(&minus)?;
                                for (dst, x) in nll_plus.iter_mut().zip(v_plus.into_iter()) {
                                    *dst += x;
                                }
                                for (dst, x) in nll_minus.iter_mut().zip(v_minus.into_iter()) {
                                    *dst += x;
                                }
                            }
                            for t in 0..n_toys {
                                if !ok_fit[t] {
                                    continue;
                                }
                                let idx = t * n_params + poi_idx;
                                let mu_hat = hat_flat[idx];
                                if !mu_hat.is_finite()
                                    || plus[idx] == mu_hat
                                    || minus[idx] == mu_hat
                                {
                                    continue;
                                }
                                let d2 = (nll_plus[t] - 2.0 * nll0[t] + nll_minus[t]) / (eps * eps);
                                if d2.is_finite() && d2 > 0.0 {
                                    poi_sigmas[t] = Some(1.0 / d2.sqrt());
                                }
                            }
                        }
                        poi_sigma_s = t_poi_sigma0.elapsed().as_secs_f64();
                    }

                    timing_breakdown.insert(
                    "toys".into(),
                    {
                        let mut t = serde_json::json!({
                            "pipeline": pipeline,
                            "device_ids": if gpu == Some("cuda") { serde_json::json!(&cuda_device_ids) } else { serde_json::Value::Null },
                            "device_shard_plan": if gpu == Some("cuda") && !cuda_device_shard_plan.is_empty() { serde_json::json!(&cuda_device_shard_plan) } else { serde_json::Value::Null },
                            "warm_start": cuda_warm_start,
                            "retry_max_retries": ns_inference::unbinned_gpu_batch::CUDA_TOY_FIT_MAX_RETRIES,
                            "retry_jitter_scale": ns_inference::unbinned_gpu_batch::CUDA_TOY_FIT_JITTER_SCALE,
                            "sample_s": toy_sample_s,
                            "batch_build_s": batch_build_s,
                            "batch_fit_s": batch_fit_s,
                            "poi_sigma_enabled": need_poi_sigma,
                            "poi_sigma_s": poi_sigma_s,
                        });
                        if !shard_detail.is_empty() {
                            t["shard_detail"] = serde_json::json!(shard_detail);
                        }
                        t
                    },
                );

                    for t in 0..n_toys {
                        match fit_results
                            .get(t)
                            .ok_or_else(|| anyhow::anyhow!("missing fit result for toy {t}"))?
                        {
                            Ok(r) => {
                                let ok = r.converged && r.nll.is_finite();
                                if ok {
                                    n_converged += 1;
                                } else {
                                    n_nonconverged += 1;
                                }
                                converged.push(Some(ok));
                                let mu =
                                    r.parameters.get(poi_idx).copied().filter(|v| v.is_finite());
                                poi_hat.push(mu);
                                poi_sigma.push(poi_sigmas[t].filter(|v| v.is_finite() && *v > 0.0));
                                nll_hat.push(if r.nll.is_finite() { Some(r.nll) } else { None });

                                if ok {
                                    if let (Some(hat), Some(sig)) = (mu, poi_sigmas[t]) {
                                        if sig.is_finite() && sig > 0.0 {
                                            pulls_by_param[poi_idx].push((hat - poi_true) / sig);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(toy_idx = t, error = %e, "toy fit failed (gpu cuda)");
                                match &e {
                                    ns_core::Error::Validation(_) => n_validation_error += 1,
                                    _ => n_computation_error += 1,
                                }
                                poi_hat.push(None);
                                poi_sigma.push(None);
                                nll_hat.push(None);
                                converged.push(None);
                            }
                        }
                    }
                } // end if !has_flows (analytical PDF path)
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
    }

    let wall_time_s = start.elapsed().as_secs_f64();

    let poi_hat_ok: Vec<f64> = poi_hat.iter().copied().flatten().collect();

    let poi_pulls = pulls_by_param.get(poi_idx).cloned().unwrap_or_default();
    let poi_pull_summary = if poi_pulls.is_empty() {
        None
    } else {
        let n = poi_pulls.len() as f64;
        let mean = poi_pulls.iter().sum::<f64>() / n;
        let var = poi_pulls.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        Some((mean, var.sqrt()))
    };

    let n_error = n_validation_error + n_computation_error;
    let mut failures: Vec<String> = Vec::new();
    if require_all_converged && (n_error > 0 || n_nonconverged > 0) {
        failures.push(format!(
            "require_all_converged violated (n_validation_error={n_validation_error}, n_computation_error={n_computation_error}, n_nonconverged={n_nonconverged})"
        ));
    }
    if max_abs_poi_pull_mean.is_some() || poi_pull_std_range.is_some() {
        match poi_pull_summary {
            None => {
                failures.push(
                    "POI pull summary is unavailable (no converged toys with finite uncertainties)"
                        .into(),
                );
            }
            Some((pull_mean, pull_std)) => {
                if let Some(max_abs) = max_abs_poi_pull_mean
                    && pull_mean.abs() > max_abs
                {
                    failures
                        .push(format!("abs(mean pull(POI)) too large: |{pull_mean}| > {max_abs}"));
                }

                if let Some([low, high]) = poi_pull_std_range.as_deref() {
                    let (low, high) = (*low, *high);
                    if pull_std < low || pull_std > high {
                        failures.push(format!(
                            "std(pull(POI)) out of range: {pull_std} not in [{low}, {high}]"
                        ));
                    }
                }
            }
        }
    }

    let pull_summary_by_param = {
        let mut n_ok = Vec::<serde_json::Value>::with_capacity(parameter_names.len());
        let mut mean = Vec::<serde_json::Value>::with_capacity(parameter_names.len());
        let mut std = Vec::<serde_json::Value>::with_capacity(parameter_names.len());
        let mut frac_abs_lt_1 = Vec::<serde_json::Value>::with_capacity(parameter_names.len());
        let mut frac_abs_lt_2 = Vec::<serde_json::Value>::with_capacity(parameter_names.len());

        for pulls in &pulls_by_param {
            if pulls.is_empty() {
                n_ok.push(serde_json::Value::Null);
                mean.push(serde_json::Value::Null);
                std.push(serde_json::Value::Null);
                frac_abs_lt_1.push(serde_json::Value::Null);
                frac_abs_lt_2.push(serde_json::Value::Null);
                continue;
            }

            let n = pulls.len() as f64;
            let m = pulls.iter().sum::<f64>() / n;
            let v = pulls.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
            let s = v.sqrt();
            let f1 = pulls.iter().filter(|x| x.abs() < 1.0).count() as f64 / n;
            let f2 = pulls.iter().filter(|x| x.abs() < 2.0).count() as f64 / n;

            n_ok.push(serde_json::json!(pulls.len()));
            mean.push(serde_json::json!(m));
            std.push(serde_json::json!(s));
            frac_abs_lt_1.push(serde_json::json!(f1));
            frac_abs_lt_2.push(serde_json::json!(f2));
        }

        serde_json::json!({
            "only_converged": true,
            "n_ok": n_ok,
            "mean": mean,
            "std": std,
            "frac_abs_lt_1": frac_abs_lt_1,
            "frac_abs_lt_2": frac_abs_lt_2,
        })
    };

    let summary = if poi_hat_ok.is_empty() {
        serde_json::Value::Null
    } else {
        let n = poi_hat_ok.len() as f64;
        let mean = poi_hat_ok.iter().sum::<f64>() / n;
        let var = poi_hat_ok.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let std = var.sqrt();

        let mut sorted = poi_hat_ok.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| -> f64 {
            if sorted.len() == 1 {
                return sorted[0];
            }
            let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
            sorted[idx]
        };

        let pull_summary = if poi_idx >= pulls_by_param.len() || pulls_by_param[poi_idx].is_empty()
        {
            serde_json::Value::Null
        } else {
            let pulls = pulls_by_param[poi_idx].clone();
            let n = pulls.len() as f64;
            let mean = pulls.iter().sum::<f64>() / n;
            let var = pulls.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
            let std = var.sqrt();

            let mut sorted = pulls;
            sorted.sort_by(|a, b| a.total_cmp(b));
            let pct = |p: f64| -> f64 {
                if sorted.len() == 1 {
                    return sorted[0];
                }
                let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
                sorted[idx]
            };

            serde_json::json!({
                "n_ok": sorted.len(),
                "mean": mean,
                "std": std,
                "q16": pct(0.16),
                "q50": pct(0.50),
                "q84": pct(0.84),
                "frac_abs_lt_1": sorted.iter().filter(|x| x.abs() < 1.0).count() as f64 / n,
                "frac_abs_lt_2": sorted.iter().filter(|x| x.abs() < 2.0).count() as f64 / n,
            })
        };

        serde_json::json!({
            "n_ok": poi_hat_ok.len(),
            "mean": mean,
            "std": std,
            "q16": pct(0.16),
            "q50": pct(0.50),
            "q84": pct(0.84),
            "pull": pull_summary,
            "pulls_by_param": pull_summary_by_param,
        })
    };

    let passed = failures.is_empty();
    let failures_json = failures.clone();

    let shard_meta = shard.map(|(idx, total)| {
        serde_json::json!({
            "shard_index": idx,
            "shard_total": total,
            "shard_start": shard_start,
            "shard_n_toys": shard_n_toys,
        })
    });

    let output_json = serde_json::json!({
        "input_schema_version": spec.schema_version,
        "parameter_names": parameter_names,
        "poi_index": poi_idx,
        "gen": {
            "point": match gen_point { UnbinnedToyGenPoint::Init => "init", UnbinnedToyGenPoint::Mle => "mle" },
            "params": gen_params,
            "overrides": overrides_json,
            "seed": seed,
            "n_toys": n_toys,
        },
        "shard": shard_meta,
        "results": {
            "n_toys": shard_n_toys,
            "n_error": n_error,
            "n_validation_error": n_validation_error,
            "n_computation_error": n_computation_error,
            "n_converged": n_converged,
            "n_nonconverged": n_nonconverged,
            "converged": converged,
            "poi_true": poi_true,
            "poi_hat": poi_hat,
            "poi_sigma": poi_sigma,
            "nll": nll_hat,
        },
        "summary": summary,
        "guardrails": {
            "require_all_converged": require_all_converged,
            "max_abs_poi_pull_mean": max_abs_poi_pull_mean,
            "poi_pull_std_range": poi_pull_std_range,
            "poi_pull_mean": poi_pull_summary.map(|(m, _s)| m),
            "poi_pull_std": poi_pull_summary.map(|(_m, s)| s),
            "passed": passed,
            "failures": failures_json,
        },
    });

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let timing_extra = if timing_breakdown.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::json!({
                "breakdown": serde_json::Value::Object(std::mem::take(&mut timing_breakdown)),
            })
        };
        let metrics_json = metrics_v0_with_timing(
            "unbinned_fit_toys",
            gpu.unwrap_or("cpu"),
            threads,
            false,
            wall_time_s,
            timing_extra,
            serde_json::json!({
                "poi_index": poi_idx,
                "poi_true": poi_true,
                "n_toys": n_toys,
                "n_error": n_error,
                "n_converged": n_converged,
                "n_nonconverged": n_nonconverged,
                "summary": summary,
                "guardrails": output_json.get("guardrails").cloned(),
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        // `report::write_bundle` expects JSON inputs (it stores them as inputs/input.json).
        // For Phase 2, require a JSON spec when bundling for clean reproducibility.
        let bytes = std::fs::read(config)?;
        serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "--bundle requires a JSON unbinned spec (got non-JSON at {}): {e}",
                config.display()
            )
        })?;

        report::write_bundle(
            dir,
            "unbinned_fit_toys",
            serde_json::json!({
                "n_toys": n_toys,
                "seed": seed,
                "gen": match gen_point { UnbinnedToyGenPoint::Init => "init", UnbinnedToyGenPoint::Mle => "mle" },
                "set": overrides_json,
                "threads": threads,
                "gpu": gpu.unwrap_or("cpu"),
            }),
            config,
            &output_json,
            false,
        )?;
    }

    if !passed {
        anyhow::bail!("unbinned-fit-toys guardrails failed: {}", failures.join("; "));
    }

    Ok(())
}

/// Merge multiple shard JSON outputs from `unbinned-fit-toys --shard` into one combined result.
fn cmd_unbinned_merge_toys(inputs: &[PathBuf], output: Option<&PathBuf>) -> Result<()> {
    use anyhow::Context;
    if inputs.len() < 2 {
        anyhow::bail!("unbinned-merge-toys requires at least 2 shard files");
    }

    let mut shards: Vec<serde_json::Value> = Vec::with_capacity(inputs.len());
    for path in inputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading shard file: {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing shard JSON: {}", path.display()))?;
        shards.push(v);
    }

    // Validate: all shards must have the same generation config.
    let gen0 = shards[0]["gen"].clone();
    for (i, s) in shards.iter().enumerate().skip(1) {
        if s["gen"] != gen0 {
            anyhow::bail!("shard {} has different gen config than shard 0", i);
        }
    }

    // Sort by shard_start for deterministic ordering.
    shards.sort_by_key(|s| {
        s["shard"].as_object().and_then(|o| o["shard_start"].as_u64()).unwrap_or(0)
    });

    // Merge results.
    let mut n_converged = 0u64;
    let mut n_nonconverged = 0u64;
    let mut n_validation_error = 0u64;
    let mut n_computation_error = 0u64;
    let mut n_toys_merged = 0usize;
    let mut poi_hat: Vec<serde_json::Value> = Vec::new();
    let mut poi_sigma: Vec<serde_json::Value> = Vec::new();
    let mut nll_hat: Vec<serde_json::Value> = Vec::new();
    let mut converged_arr: Vec<serde_json::Value> = Vec::new();
    let mut shard_details: Vec<serde_json::Value> = Vec::new();

    for s in &shards {
        let r = &s["results"];
        n_converged += r["n_converged"].as_u64().unwrap_or(0);
        n_nonconverged += r["n_nonconverged"].as_u64().unwrap_or(0);
        n_validation_error += r["n_validation_error"].as_u64().unwrap_or(0);
        n_computation_error += r["n_computation_error"].as_u64().unwrap_or(0);
        let shard_n = r["n_toys"].as_u64().unwrap_or(0) as usize;
        n_toys_merged += shard_n;

        if let Some(arr) = r["poi_hat"].as_array() {
            poi_hat.extend(arr.iter().cloned());
        }
        if let Some(arr) = r["poi_sigma"].as_array() {
            poi_sigma.extend(arr.iter().cloned());
        }
        if let Some(arr) = r["nll"].as_array() {
            nll_hat.extend(arr.iter().cloned());
        }
        if let Some(arr) = r["converged"].as_array() {
            converged_arr.extend(arr.iter().cloned());
        }

        shard_details.push(serde_json::json!({
            "shard": s["shard"],
            "n_converged": r["n_converged"],
            "n_nonconverged": r["n_nonconverged"],
            "n_validation_error": r["n_validation_error"],
            "n_computation_error": r["n_computation_error"],
        }));
    }

    let n_error = n_validation_error + n_computation_error;
    let n_fittable = n_toys_merged as u64 - n_validation_error;
    let fittable_convergence =
        if n_fittable > 0 { n_converged as f64 / n_fittable as f64 } else { 0.0 };
    let overall_convergence =
        if n_toys_merged > 0 { n_converged as f64 / n_toys_merged as f64 } else { 0.0 };

    let merged = serde_json::json!({
        "merged": true,
        "n_shards": shards.len(),
        "gen": gen0,
        "results": {
            "n_toys": n_toys_merged,
            "n_error": n_error,
            "n_validation_error": n_validation_error,
            "n_computation_error": n_computation_error,
            "n_converged": n_converged,
            "n_nonconverged": n_nonconverged,
            "overall_convergence": overall_convergence,
            "fittable_convergence": fittable_convergence,
            "converged": converged_arr,
            "poi_true": shards[0]["results"]["poi_true"].clone(),
            "poi_hat": poi_hat,
            "poi_sigma": poi_sigma,
            "nll": nll_hat,
        },
        "shards": shard_details,
    });

    write_json(output, &merged)?;
    Ok(())
}

fn cmd_audit(input: &PathBuf, format: &str, output: Option<&PathBuf>) -> Result<()> {
    let json_str = std::fs::read_to_string(input)?;
    let json: serde_json::Value = serde_json::from_str(&json_str)?;
    let audit = ns_translate::pyhf::audit::workspace_audit(&json);

    let text = match format {
        "json" => serde_json::to_string_pretty(&audit)?,
        _ => {
            let mut s = String::new();
            s.push_str(&format!("Workspace: {}\n", input.display()));
            s.push_str(&format!(
                "  Channels: {}, Bins: {}, Samples: {}, Modifiers: {}\n",
                audit.channel_count, audit.total_bins, audit.total_samples, audit.total_modifiers
            ));
            s.push_str(&format!("  Parameters (est): {}\n", audit.parameter_count_estimate));
            s.push_str("\nMeasurements:\n");
            for m in &audit.measurements {
                s.push_str(&format!(
                    "  - {} (POI: {}, {} fixed params)\n",
                    m.name, m.poi, m.n_fixed_params
                ));
            }
            s.push_str("\nModifier types:\n");
            let mut types: Vec<_> = audit.modifier_types_used.iter().collect();
            types.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
            for (typ, count) in &types {
                let marker =
                    if ns_translate::pyhf::audit::KNOWN_MODIFIER_TYPES.contains(&typ.as_str()) {
                        "+"
                    } else {
                        "!"
                    };
                s.push_str(&format!("  {} {} (x{})\n", marker, typ, count));
            }
            if !audit.unsupported_features.is_empty() {
                s.push_str("\nWarnings:\n");
                for w in &audit.unsupported_features {
                    s.push_str(&format!("  - {}\n", w));
                }
            }
            s
        }
    };

    if let Some(path) = output {
        std::fs::write(path, &text)?;
    } else {
        print!("{}", text);
    }
    Ok(())
}

/// Configure Rayon thread pool and deterministic/parity mode.
///
/// When `threads == 1`, Accelerate is automatically disabled to ensure
/// bit-exact parity with the SIMD/scalar path.
///
/// When `parity` is true, enables Kahan summation and forces threads=1.
fn setup_runtime(threads: usize, parity: bool) {
    let effective_threads = if parity { 1 } else { threads };
    if effective_threads > 0 {
        // Best-effort; if a global pool already exists, keep going.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(effective_threads).build_global();
    }
    if parity {
        ns_compute::set_eval_mode(ns_compute::EvalMode::Parity);
        tracing::info!("parity mode: Kahan summation, Accelerate disabled, threads=1");
    } else if effective_threads == 1 {
        ns_compute::set_accelerate_enabled(false);
        tracing::debug!("deterministic mode: Accelerate disabled");
    }
}

fn normalize_cuda_device_ids(gpu: Option<&str>, gpu_devices: &[usize]) -> Result<Vec<usize>> {
    if gpu_devices.is_empty() {
        return Ok(if gpu == Some("cuda") { vec![0] } else { Vec::new() });
    }
    if gpu != Some("cuda") {
        anyhow::bail!("--gpu-devices requires --gpu cuda");
    }
    let mut out = Vec::with_capacity(gpu_devices.len());
    for &id in gpu_devices {
        if !out.contains(&id) {
            out.push(id);
        }
    }
    if out.is_empty() {
        anyhow::bail!("--gpu-devices must contain at least one device id");
    }
    Ok(out)
}

#[inline]
fn checked_len_to_u32(len: usize, context: &str) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        anyhow::anyhow!("{context}: value {len} exceeds 32-bit toy-offset limit ({})", u32::MAX)
    })
}

fn min_shards_for_u32_toy_offsets(
    n_toys: usize,
    estimated_events_per_toy: usize,
) -> Result<Option<(usize, usize)>> {
    if n_toys == 0 || estimated_events_per_toy == 0 {
        return Ok(None);
    }

    // Keep a small headroom below u32::MAX to absorb Poisson fluctuations.
    const U32_EVENTS_HEADROOM: f64 = 0.95;
    let max_events_per_shard = ((u32::MAX as f64) * U32_EVENTS_HEADROOM).floor() as usize;
    if max_events_per_shard == 0 {
        anyhow::bail!("internal error: invalid 32-bit shard event headroom");
    }
    if estimated_events_per_toy > max_events_per_shard {
        anyhow::bail!(
            "estimated events per toy ({estimated_events_per_toy}) exceed safe 32-bit \
             shard budget ({max_events_per_shard}); current GPU toy-offset kernels use u32"
        );
    }

    let max_toys_per_shard = (max_events_per_shard / estimated_events_per_toy).max(1);
    let min_shards = n_toys.div_ceil(max_toys_per_shard);
    Ok(Some((min_shards, max_toys_per_shard)))
}

fn metal_max_toys_per_batch_for_u32_offsets(
    n_toys: usize,
    estimated_events_per_toy: usize,
) -> Result<usize> {
    if n_toys == 0 {
        return Ok(1);
    }
    let max_toys = min_shards_for_u32_toy_offsets(n_toys, estimated_events_per_toy)?
        .map(|(_, max_toys_per_shard)| max_toys_per_shard.max(1))
        .unwrap_or(n_toys.max(1));
    Ok(max_toys.max(1))
}

fn contiguous_toy_batches(n_toys: usize, max_toys_per_batch: usize) -> Vec<(usize, usize)> {
    if n_toys == 0 || max_toys_per_batch == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_toys.div_ceil(max_toys_per_batch));
    let mut toy_start = 0usize;
    while toy_start < n_toys {
        let toy_end = (toy_start + max_toys_per_batch).min(n_toys);
        out.push((toy_start, toy_end));
        toy_start = toy_end;
    }
    out
}

fn normalize_cuda_device_shard_plan(
    gpu: Option<&str>,
    gpu_sample_toys: bool,
    cuda_device_ids: &[usize],
    gpu_shards: Option<usize>,
    n_toys: usize,
    estimated_bytes_per_toy: usize,
    estimated_events_per_toy: usize,
) -> Result<Vec<usize>> {
    if gpu_shards.is_some() && gpu != Some("cuda") {
        anyhow::bail!("--gpu-shards currently requires --gpu cuda");
    }
    if gpu != Some("cuda") {
        return Ok(Vec::new());
    }
    if cuda_device_ids.is_empty() {
        anyhow::bail!("internal error: CUDA device list is empty for toy sharding");
    }

    let u32_guard = min_shards_for_u32_toy_offsets(n_toys, estimated_events_per_toy)?;
    let mut n_shards = if let Some(v) = gpu_shards {
        v
    } else if gpu_sample_toys && estimated_bytes_per_toy > 0 && n_toys > 0 {
        // PF3.1-OPT4: VRAM-aware auto-shard computation.
        // Query minimum VRAM across target devices, then compute shard count
        // so each batch fits in 70% of available VRAM.
        #[cfg(feature = "cuda")]
        {
            let min_vram = cuda_device_ids
                .iter()
                .filter_map(|&d| ns_compute::cuda_device_total_mem(d).ok())
                .min()
                .unwrap_or(0);
            if min_vram > 0 {
                let usable = (min_vram as f64 * 0.70) as usize;
                // `estimated_bytes_per_toy` tracks only the raw event payload, which tends to
                // *underestimate* peak VRAM during CUDA toy sampling + batch fitting.
                // Cap the per-device toy count to avoid OOMs on realistic workloads.
                //
                // Rationale: in PF3.1, unsharded CUDA sampling can OOM for n_toys=10k on 20GB GPUs,
                // while 2+ shards succeeds. A conservative cap keeps the publication matrix robust.
                const CUDA_AUTO_MAX_TOYS_PER_DEVICE: usize = 4096;
                let max_toys_per_device =
                    (usable / estimated_bytes_per_toy).max(1).min(CUDA_AUTO_MAX_TOYS_PER_DEVICE);
                let shards_needed = n_toys.div_ceil(max_toys_per_device);
                let n = shards_needed.max(cuda_device_ids.len());
                tracing::info!(
                    "OPT4 auto-shard: {min_vram} B VRAM, ~{estimated_bytes_per_toy} B/toy \
                     → max {max_toys_per_device} toys/device → {n} shards"
                );
                n
            } else {
                cuda_device_ids.len()
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            cuda_device_ids.len()
        }
    } else if gpu_sample_toys {
        cuda_device_ids.len()
    } else {
        // Host-toy path: keep legacy behavior (no sharding) unless a u32 toy-offset
        // overflow is predicted for the requested workload.
        if let Some((min_shards, _)) = u32_guard {
            if min_shards > 1 {
                let n = min_shards.max(cuda_device_ids.len());
                tracing::info!(
                    "auto-shard: enabling CUDA host sharding ({n} shards) to satisfy 32-bit toy-offset budget"
                );
                n
            } else {
                return Ok(Vec::new());
            }
        } else {
            return Ok(Vec::new());
        }
    };
    if n_shards == 0 {
        anyhow::bail!("--gpu-shards must be > 0");
    }

    if let Some((min_shards, max_toys_per_shard)) = u32_guard {
        if n_shards < min_shards {
            if gpu_shards.is_some() {
                anyhow::bail!(
                    "--gpu-shards={n_shards} is too small for 32-bit toy offsets: \
                     estimated_events_per_toy={estimated_events_per_toy}, \
                     max_toys_per_shard={max_toys_per_shard}, required_shards={min_shards}"
                );
            }
            tracing::warn!(
                "auto-shard: increasing shards from {n_shards} to {min_shards} \
                 to satisfy 32-bit toy-offset budget"
            );
            n_shards = min_shards;
        }
    }

    let mut plan = Vec::with_capacity(n_shards);
    for i in 0..n_shards {
        plan.push(cuda_device_ids[i % cuda_device_ids.len()]);
    }
    Ok(plan)
}

fn estimate_unbinned_expected_bytes_per_toy(
    model: &ns_unbinned::UnbinnedModel,
    params: &[f64],
) -> Result<usize> {
    // For Poisson toys, the expected event count per toy is determined by the model's yield
    // expressions at the generation point (not by the observed dataset size).
    const SAFETY: f64 = 1.5;

    let mut total_expected_events_x_obs = 0.0f64;
    let mut scratch = Vec::<(usize, f64)>::new();

    for ch in model.channels().iter().filter(|ch| ch.include_in_fit) {
        let n_obs =
            ch.processes.first().map(|p| p.pdf.observables().len()).unwrap_or(1).max(1) as f64;

        let mut nu_total = 0.0f64;
        for proc in &ch.processes {
            scratch.clear();
            let nu = proc.yield_expr.value(params)?;
            // `nu` should be >=0 for valid models; clamp for robustness in estimation.
            nu_total += nu.max(0.0);
        }

        total_expected_events_x_obs += nu_total * n_obs;
    }

    if !total_expected_events_x_obs.is_finite() || total_expected_events_x_obs <= 0.0 {
        return Ok(0);
    }

    // f64 event payload is 8 bytes per observable.
    let bytes = total_expected_events_x_obs * 8.0 * SAFETY;
    Ok(bytes.min(usize::MAX as f64) as usize)
}

fn estimate_unbinned_expected_events_per_toy(
    model: &ns_unbinned::UnbinnedModel,
    params: &[f64],
) -> Result<usize> {
    let mut total_expected_events = 0.0f64;
    for ch in model.channels().iter().filter(|ch| ch.include_in_fit) {
        let mut nu_total = 0.0f64;
        for proc in &ch.processes {
            let nu = proc.yield_expr.value(params)?;
            nu_total += nu.max(0.0);
        }
        total_expected_events += nu_total;
    }

    if !total_expected_events.is_finite() || total_expected_events <= 0.0 {
        return Ok(0);
    }
    Ok(total_expected_events.min(usize::MAX as f64) as usize)
}

fn contiguous_toy_shards(n_toys: usize, shard_device_plan: &[usize]) -> Vec<(usize, usize, usize)> {
    if n_toys == 0 || shard_device_plan.is_empty() {
        return Vec::new();
    }
    let n_shards = shard_device_plan.len().min(n_toys);
    let toys_per_shard = n_toys.div_ceil(n_shards);
    let mut out = Vec::with_capacity(n_shards);
    for (shard_idx, &device_id) in shard_device_plan.iter().take(n_shards).enumerate() {
        let toy_start = shard_idx * toys_per_shard;
        let toy_end = ((shard_idx + 1) * toys_per_shard).min(n_toys);
        if toy_start >= toy_end {
            break;
        }
        out.push((toy_start, toy_end, device_id));
    }
    out
}

fn parse_shard(s: &str) -> Result<(usize, usize), String> {
    let (idx, total) =
        s.split_once('/').ok_or_else(|| "expected INDEX/TOTAL (e.g. 0/4)".to_string())?;
    let idx: usize = idx.trim().parse().map_err(|e| format!("invalid shard INDEX: {e}"))?;
    let total: usize = total.trim().parse().map_err(|e| format!("invalid shard TOTAL: {e}"))?;
    if total == 0 {
        return Err("shard TOTAL must be > 0".to_string());
    }
    if idx >= total {
        return Err(format!("shard INDEX ({idx}) must be < TOTAL ({total})"));
    }
    Ok((idx, total))
}

fn parse_param_override(s: &str) -> Result<(String, f64), String> {
    let (name, value) = s.split_once('=').ok_or_else(|| "expected NAME=VALUE".to_string())?;
    if name.trim().is_empty() {
        return Err("expected NAME=VALUE (NAME is empty)".to_string());
    }
    let v: f64 = value.trim().parse().map_err(|e| format!("invalid VALUE '{value}': {e}"))?;
    Ok((name.trim().to_string(), v))
}

fn dash_means_stdout(path: &PathBuf) -> Option<&PathBuf> {
    if path.as_os_str() == "-" { None } else { Some(path) }
}

fn now_unix_ms() -> Result<u128> {
    use std::time::{SystemTime, UNIX_EPOCH};
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("system time error: {e}"))?
        .as_millis())
}

fn metrics_v0(
    command: &str,
    device: &str,
    threads: usize,
    parity: bool,
    wall_time_s: f64,
    metrics: serde_json::Value,
) -> Result<serde_json::Value> {
    metrics_v0_with_timing(
        command,
        device,
        threads,
        parity,
        wall_time_s,
        serde_json::json!({}),
        metrics,
    )
}

fn metrics_v0_with_timing(
    command: &str,
    device: &str,
    threads: usize,
    parity: bool,
    wall_time_s: f64,
    timing_extra: serde_json::Value,
    metrics: serde_json::Value,
) -> Result<serde_json::Value> {
    let mut timing = serde_json::Map::<String, serde_json::Value>::new();
    timing.insert("wall_time_s".into(), serde_json::json!(wall_time_s));
    if let Some(obj) = timing_extra.as_object() {
        for (k, v) in obj {
            timing.insert(k.clone(), v.clone());
        }
    }

    Ok(serde_json::json!({
        "schema_version": "nextstat_metrics_v0",
        "tool": "nextstat",
        "tool_version": ns_core::VERSION,
        "created_unix_ms": now_unix_ms()?,
        "command": command,
        "device": device,
        "threads": threads,
        "parity": parity,
        "timing": serde_json::Value::Object(timing),
        "status": {
            "ok": true,
        },
        "warnings": [],
        "degraded_flags": [],
        "metrics": metrics,
    }))
}

fn load_model(
    input: &PathBuf,
    threads: usize,
    parity: bool,
    interp_defaults: InterpDefaults,
) -> Result<ns_translate::pyhf::HistFactoryModel> {
    setup_runtime(threads, parity);

    tracing::info!(path = %input.display(), "loading workspace");
    let json = std::fs::read_to_string(input)?;

    let format = ns_translate::hs3::detect::detect_format(&json);
    let model = match format {
        ns_translate::hs3::detect::WorkspaceFormat::Hs3 => {
            tracing::info!("detected HS3 format");
            ns_translate::hs3::convert::from_hs3_default(&json)
                .map_err(|e| anyhow::anyhow!("HS3 loading failed: {e}"))?
        }
        ns_translate::hs3::detect::WorkspaceFormat::Pyhf
        | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
            match interp_defaults {
                InterpDefaults::Pyhf => {
                    ns_translate::pyhf::HistFactoryModel::from_workspace_with_settings(
                        &workspace,
                        ns_translate::pyhf::NormSysInterpCode::Code1,
                        ns_translate::pyhf::HistoSysInterpCode::Code0,
                    )?
                }
                InterpDefaults::Root => {
                    ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?
                }
            }
        }
    };

    tracing::info!(parameters = model.parameters().len(), "workspace loaded");
    Ok(model)
}

fn load_workspace_and_model(
    input: &PathBuf,
    threads: usize,
    interp_defaults: InterpDefaults,
) -> Result<(ns_translate::pyhf::Workspace, ns_translate::pyhf::HistFactoryModel)> {
    setup_runtime(threads, false);

    tracing::info!(path = %input.display(), "loading workspace");
    let json = std::fs::read_to_string(input)?;
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
    let model = match interp_defaults {
        InterpDefaults::Pyhf => ns_translate::pyhf::HistFactoryModel::from_workspace_with_settings(
            &workspace,
            ns_translate::pyhf::NormSysInterpCode::Code1,
            ns_translate::pyhf::HistoSysInterpCode::Code0,
        )?,
        InterpDefaults::Root => ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?,
    };
    tracing::info!(parameters = model.parameters().len(), "workspace loaded");
    Ok((workspace, model))
}

fn normalize_json_for_determinism(mut value: serde_json::Value) -> serde_json::Value {
    fn stable_sort_by_key(
        arr: &mut Vec<serde_json::Value>,
        mut key: impl FnMut(&serde_json::Value) -> Option<&str>,
    ) {
        if arr.len() <= 1 {
            return;
        }
        // Stable deterministic sort: preserve original order for ties and missing keys.
        let mut tmp: Vec<(usize, serde_json::Value)> = arr.drain(..).enumerate().collect();
        tmp.sort_by(|(ia, a), (ib, b)| {
            let ka = key(a).unwrap_or("");
            let kb = key(b).unwrap_or("");
            ka.cmp(kb).then_with(|| ia.cmp(ib))
        });
        arr.extend(tmp.into_iter().map(|(_, v)| v));
    }

    fn norm(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, vv) in map.iter_mut() {
                    match k.as_str() {
                        "created_unix_ms" | "timestamp" => {
                            *vv = serde_json::Value::Number(serde_json::Number::from(0u64));
                        }
                        "wall_s" | "elapsed_s" | "fit_s" | "predict_s" => {
                            *vv = serde_json::json!(0.0);
                        }
                        "stdout_tail" => {
                            *vv = serde_json::Value::String(String::new());
                        }
                        _ => {}
                    }
                    norm(vv);
                }

                // Stable ordering for common "set-like" arrays in report artifacts.
                // NOTE: this is only used behind `--deterministic`.
                let has_aligned_bins = map.contains_key("n_bins_per_channel");
                for (k, vv) in map.iter_mut() {
                    let Some(arr) = vv.as_array_mut() else { continue };
                    match k.as_str() {
                        "channels" => {
                            // Most report artifacts use `channels: [{channel_name, ...}, ...]`.
                            // Avoid reordering string arrays that are aligned with other arrays
                            // (e.g. `channels: [..]` + `n_bins_per_channel: [..]`).
                            stable_sort_by_key(arr, |x| {
                                if let Some(s) = x.as_str() {
                                    if has_aligned_bins { None } else { Some(s) }
                                } else {
                                    x.get("channel_name").and_then(|v| v.as_str())
                                }
                            });
                        }
                        "samples" => {
                            stable_sort_by_key(arr, |x| {
                                if let Some(s) = x.as_str() {
                                    Some(s)
                                } else {
                                    x.get("name")
                                        .or_else(|| x.get("sample_name"))
                                        .and_then(|v| v.as_str())
                                }
                            });
                        }
                        "entries" | "groups" | "observations" => {
                            stable_sort_by_key(arr, |x| {
                                if let Some(s) = x.as_str() {
                                    Some(s)
                                } else {
                                    x.get("name").and_then(|v| v.as_str())
                                }
                            });
                        }
                        _ => {}
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for vv in arr.iter_mut() {
                    norm(vv);
                }
            }
            _ => {}
        }
    }

    norm(&mut value);
    value
}

#[cfg(test)]
mod deterministic_json_tests {
    use super::normalize_json_for_determinism;

    #[test]
    fn normalize_sorts_report_channels_and_samples() {
        let v = serde_json::json!({
            "meta": {"created_unix_ms": 123},
            "channels": [
                {"channel_name": "b", "samples": [{"name":"z"}, {"name":"a"}]},
                {"channel_name": "a", "samples": [{"name":"b"}, {"name":"a"}]},
            ],
        });
        let out = normalize_json_for_determinism(v);
        let channels = out.get("channels").unwrap().as_array().unwrap();
        assert_eq!(channels[0].get("channel_name").unwrap().as_str().unwrap(), "a");
        assert_eq!(channels[1].get("channel_name").unwrap().as_str().unwrap(), "b");
        let samples_a = channels[0].get("samples").unwrap().as_array().unwrap();
        assert_eq!(samples_a[0].get("name").unwrap().as_str().unwrap(), "a");
        assert_eq!(samples_a[1].get("name").unwrap().as_str().unwrap(), "b");
        let meta_created =
            out.get("meta").unwrap().get("created_unix_ms").unwrap().as_u64().unwrap();
        assert_eq!(meta_created, 0);
    }

    #[test]
    fn normalize_does_not_reorder_aligned_channel_arrays() {
        // Some artifacts keep multiple arrays aligned by index (e.g. channels + n_bins_per_channel).
        // Normalization must not reorder `channels` in that case.
        let v = serde_json::json!({
            "dataset_fingerprint": {
                "channels": ["b", "a"],
                "n_bins_per_channel": [2, 1],
            }
        });
        let out = normalize_json_for_determinism(v);
        let fp = out.get("dataset_fingerprint").unwrap();
        let channels = fp.get("channels").unwrap().as_array().unwrap();
        assert_eq!(channels[0].as_str().unwrap(), "b");
        assert_eq!(channels[1].as_str().unwrap(), "a");
    }
}

#[cfg(test)]
mod cli_parse_tests {
    use super::{Cli, InterpDefaults};
    use clap::Parser;

    #[test]
    fn interp_defaults_parses_and_defaults() {
        let cli = Cli::try_parse_from(["nextstat", "fit", "--input", "workspace.json"]).unwrap();
        assert_eq!(cli.interp_defaults, InterpDefaults::Root);

        let cli = Cli::try_parse_from([
            "nextstat",
            "--interp-defaults",
            "pyhf",
            "fit",
            "--input",
            "workspace.json",
        ])
        .unwrap();
        assert_eq!(cli.interp_defaults, InterpDefaults::Pyhf);
    }
}

fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                if let Some(v) = map.get(&k) {
                    out.insert(k, canonicalize_json(v));
                }
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize_json).collect())
        }
        _ => value.clone(),
    }
}

fn write_json(output: Option<&PathBuf>, value: &serde_json::Value) -> Result<()> {
    let value = canonicalize_json(value);
    if let Some(path) = output {
        std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&value)?);
    }
    Ok(())
}

fn ensure_out_dir(out_dir: &std::path::Path, overwrite: bool) -> Result<()> {
    if out_dir.exists() {
        if !out_dir.is_dir() {
            anyhow::bail!("out dir exists but is not a directory: {}", out_dir.display());
        }
        if !overwrite && out_dir.read_dir()?.next().is_some() {
            anyhow::bail!("out dir must be empty (or use --overwrite): {}", out_dir.display());
        }
    } else {
        std::fs::create_dir_all(out_dir)?;
    }
    Ok(())
}

fn write_json_file(path: &std::path::Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let value = canonicalize_json(value);
    std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

fn write_yields_tables(
    yields: &ns_viz::yields::YieldsArtifact,
    out_dir: &std::path::Path,
    deterministic: bool,
) -> Result<()> {
    let mut channels: Vec<_> = yields.channels.iter().collect();
    if deterministic {
        channels.sort_by(|a, b| a.channel_name.cmp(&b.channel_name));
    }

    // CSV: channel,row_kind,name,prefit,postfit
    let mut csv = String::new();
    csv.push_str("channel,row_kind,name,prefit,postfit\n");
    for ch in channels.iter().copied() {
        let mut samples: Vec<_> = ch.samples.iter().collect();
        if deterministic {
            samples.sort_by(|a, b| a.name.cmp(&b.name));
        }
        for s in samples {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                ch.channel_name, "sample", s.name, s.prefit, s.postfit
            ));
        }
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            ch.channel_name, "total", "TOTAL", ch.total_prefit, ch.total_postfit
        ));
        csv.push_str(&format!(
            "{},{},{},{},\n",
            ch.channel_name,
            "data",
            "DATA",
            if ch.data_is_blinded == Some(true) { String::new() } else { ch.data.to_string() }
        ));
    }
    std::fs::write(out_dir.join("yields.csv"), csv)?;

    // LaTeX: one tabular per channel (booktabs-friendly).
    let mut tex = String::new();
    tex.push_str("% Auto-generated by `nextstat report`.\n");
    tex.push_str("% Requires: \\usepackage{booktabs}\n\n");
    for ch in channels.iter().copied() {
        tex.push_str(&format!("% Channel: {}\n", ch.channel_name));
        tex.push_str("\\begin{tabular}{lrr}\n");
        tex.push_str("\\toprule\n");
        tex.push_str("Sample & Prefit & Postfit \\\\\n");
        tex.push_str("\\midrule\n");
        let mut samples: Vec<_> = ch.samples.iter().collect();
        if deterministic {
            samples.sort_by(|a, b| a.name.cmp(&b.name));
        }
        for s in samples {
            tex.push_str(&format!("{} & {} & {} \\\\\n", s.name, s.prefit, s.postfit));
        }
        tex.push_str("\\midrule\n");
        tex.push_str(&format!("Total & {} & {} \\\\\n", ch.total_prefit, ch.total_postfit));
        tex.push_str(&format!(
            "Data & {} & \\\\\n",
            if ch.data_is_blinded == Some(true) {
                "\\textit{blinded}".to_string()
            } else {
                ch.data.to_string()
            }
        ));
        tex.push_str("\\bottomrule\n");
        tex.push_str("\\end{tabular}\n\n");
    }
    std::fs::write(out_dir.join("yields.tex"), tex)?;

    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanModelJson {
    f: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    h: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
    m0: Vec<f64>,
    p0: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanLocalLevelJson {
    q: f64,
    r: f64,
    #[serde(default = "default_m0")]
    m0: f64,
    #[serde(default = "default_p0")]
    p0: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanLocalLinearTrendJson {
    q_level: f64,
    q_slope: f64,
    r: f64,
    #[serde(default = "default_level0")]
    level0: f64,
    #[serde(default = "default_slope0")]
    slope0: f64,
    #[serde(default = "default_p0")]
    p0_level: f64,
    #[serde(default = "default_p0")]
    p0_slope: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanAr1Json {
    phi: f64,
    q: f64,
    r: f64,
    #[serde(default = "default_m0")]
    m0: f64,
    #[serde(default = "default_p0")]
    p0: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanLocalLevelSeasonalJson {
    period: usize,
    q_level: f64,
    q_season: f64,
    r: f64,
    #[serde(default = "default_level0")]
    level0: f64,
    #[serde(default = "default_p0")]
    p0_level: f64,
    #[serde(default = "default_p0")]
    p0_season: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanLocalLinearTrendSeasonalJson {
    period: usize,
    q_level: f64,
    q_slope: f64,
    q_season: f64,
    r: f64,
    #[serde(default = "default_level0")]
    level0: f64,
    #[serde(default = "default_slope0")]
    slope0: f64,
    #[serde(default = "default_p0")]
    p0_level: f64,
    #[serde(default = "default_p0")]
    p0_slope: f64,
    #[serde(default = "default_p0")]
    p0_season: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanArma11Json {
    phi: f64,
    theta: f64,
    sigma2: f64,
    #[serde(default = "default_r_tiny")]
    r: f64,
    #[serde(default = "default_m0")]
    m0_x: f64,
    #[serde(default = "default_m0")]
    m0_eps: f64,
    #[serde(default = "default_p0")]
    p0_x: f64,
    #[serde(default = "default_p0")]
    p0_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct KalmanInputJson {
    #[serde(default)]
    model: Option<KalmanModelJson>,
    #[serde(default)]
    local_level: Option<KalmanLocalLevelJson>,
    #[serde(default)]
    local_linear_trend: Option<KalmanLocalLinearTrendJson>,
    #[serde(default)]
    ar1: Option<KalmanAr1Json>,
    #[serde(default)]
    arma11: Option<KalmanArma11Json>,
    #[serde(default)]
    local_level_seasonal: Option<KalmanLocalLevelSeasonalJson>,
    #[serde(default)]
    local_linear_trend_seasonal: Option<KalmanLocalLinearTrendSeasonalJson>,
    ys: Vec<Vec<Option<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReturnsInputJson {
    returns: Vec<f64>,
}

#[derive(Debug, Clone)]
enum KalmanSpecKind {
    Model,
    LocalLevel,
    LocalLinearTrend,
    Ar1,
    Arma11,
    LocalLevelSeasonal { period: usize },
    LocalLinearTrendSeasonal { period: usize },
}

fn default_m0() -> f64 {
    0.0
}

fn default_level0() -> f64 {
    0.0
}

fn default_slope0() -> f64 {
    0.0
}

fn default_p0() -> f64 {
    1.0
}

fn default_r_tiny() -> f64 {
    1e-12
}

fn load_returns_input(path: &PathBuf) -> Result<Vec<f64>> {
    let s = std::fs::read_to_string(path)?;
    let input: ReturnsInputJson = serde_json::from_str(&s)
        .map_err(|e| anyhow::anyhow!("failed to parse returns JSON: {e}"))?;
    if input.returns.is_empty() {
        anyhow::bail!("returns must be non-empty");
    }
    if input.returns.iter().any(|v| !v.is_finite()) {
        anyhow::bail!("returns must contain only finite values");
    }
    Ok(input.returns)
}

fn dmatrix_from_nested(name: &str, rows: Vec<Vec<f64>>) -> Result<DMatrix<f64>> {
    if rows.is_empty() {
        anyhow::bail!("{name} must be non-empty");
    }
    let nrows = rows.len();
    let ncols = rows[0].len();
    if ncols == 0 {
        anyhow::bail!("{name} rows must be non-empty");
    }
    for (i, r) in rows.iter().enumerate() {
        if r.len() != ncols {
            anyhow::bail!(
                "{name} must be rectangular: row {i} has len {}, expected {}",
                r.len(),
                ncols
            );
        }
        if r.iter().any(|v| !v.is_finite()) {
            anyhow::bail!("{name} must contain only finite values");
        }
    }
    let mut flat = Vec::with_capacity(nrows * ncols);
    for r in rows {
        flat.extend_from_slice(&r);
    }
    Ok(DMatrix::from_row_slice(nrows, ncols, &flat))
}

fn dvector_from_vec(name: &str, v: Vec<f64>) -> Result<DVector<f64>> {
    if v.is_empty() {
        anyhow::bail!("{name} must be non-empty");
    }
    if v.iter().any(|x| !x.is_finite()) {
        anyhow::bail!("{name} must contain only finite values");
    }
    Ok(DVector::from_vec(v))
}

fn dvector_from_opt_vec(name: &str, v: Vec<Option<f64>>) -> Result<DVector<f64>> {
    if v.is_empty() {
        anyhow::bail!("{name} must be non-empty");
    }
    let mut out = Vec::with_capacity(v.len());
    for x in v {
        match x {
            Some(v) if v.is_finite() => out.push(v),
            Some(_) => anyhow::bail!("{name} must contain only finite values or null"),
            None => out.push(f64::NAN),
        }
    }
    Ok(DVector::from_vec(out))
}

fn dvector_to_vec(v: &DVector<f64>) -> Vec<f64> {
    v.iter().copied().collect()
}

fn dmatrix_to_nested(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut out = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let mut row = Vec::with_capacity(ncols);
        for j in 0..ncols {
            row.push(m[(i, j)]);
        }
        out.push(row);
    }
    out
}

fn load_kalman_input(
    path: &PathBuf,
) -> Result<(ns_inference::timeseries::kalman::KalmanModel, Vec<DVector<f64>>)> {
    let bytes = std::fs::read(path)?;
    let input: KalmanInputJson = serde_json::from_slice(&bytes)?;

    let mut model_count = 0usize;
    model_count += input.model.is_some() as usize;
    model_count += input.local_level.is_some() as usize;
    model_count += input.local_linear_trend.is_some() as usize;
    model_count += input.ar1.is_some() as usize;
    model_count += input.arma11.is_some() as usize;
    model_count += input.local_level_seasonal.is_some() as usize;
    model_count += input.local_linear_trend_seasonal.is_some() as usize;
    if model_count != 1 {
        anyhow::bail!(
            "expected exactly one of: model, local_level, local_linear_trend, ar1, arma11, local_level_seasonal, local_linear_trend_seasonal"
        );
    }

    let model = if let Some(mj) = input.model {
        ns_inference::timeseries::kalman::KalmanModel::new(
            dmatrix_from_nested("F", mj.f)?,
            dmatrix_from_nested("Q", mj.q)?,
            dmatrix_from_nested("H", mj.h)?,
            dmatrix_from_nested("R", mj.r)?,
            dvector_from_vec("m0", mj.m0)?,
            dmatrix_from_nested("P0", mj.p0)?,
        )
        .map_err(|e| anyhow::anyhow!("invalid Kalman model: {e}"))?
    } else if let Some(ll) = input.local_level {
        ns_inference::timeseries::kalman::KalmanModel::local_level(ll.q, ll.r, ll.m0, ll.p0)
            .map_err(|e| anyhow::anyhow!("invalid local_level model: {e}"))?
    } else if let Some(lt) = input.local_linear_trend {
        ns_inference::timeseries::kalman::KalmanModel::local_linear_trend(
            lt.q_level,
            lt.q_slope,
            lt.r,
            lt.level0,
            lt.slope0,
            lt.p0_level,
            lt.p0_slope,
        )
        .map_err(|e| anyhow::anyhow!("invalid local_linear_trend model: {e}"))?
    } else if let Some(ar) = input.ar1 {
        ns_inference::timeseries::kalman::KalmanModel::ar1(ar.phi, ar.q, ar.r, ar.m0, ar.p0)
            .map_err(|e| anyhow::anyhow!("invalid ar1 model: {e}"))?
    } else if let Some(arma) = input.arma11 {
        if !arma.phi.is_finite() {
            anyhow::bail!("invalid arma11: phi must be finite");
        }
        if !arma.theta.is_finite() {
            anyhow::bail!("invalid arma11: theta must be finite");
        }
        if !arma.sigma2.is_finite() || arma.sigma2 <= 0.0 {
            anyhow::bail!("invalid arma11: sigma2 must be finite and > 0");
        }
        if !arma.r.is_finite() || arma.r <= 0.0 {
            anyhow::bail!("invalid arma11: r must be finite and > 0");
        }
        if !arma.m0_x.is_finite() || !arma.m0_eps.is_finite() {
            anyhow::bail!("invalid arma11: initial state must be finite");
        }
        if !arma.p0_x.is_finite()
            || arma.p0_x <= 0.0
            || !arma.p0_eps.is_finite()
            || arma.p0_eps <= 0.0
        {
            anyhow::bail!("invalid arma11: p0_x/p0_eps must be finite and > 0");
        }

        ns_inference::timeseries::kalman::KalmanModel::new(
            DMatrix::from_row_slice(2, 2, &[arma.phi, arma.theta, 0.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[arma.sigma2, arma.sigma2, arma.sigma2, arma.sigma2]),
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DMatrix::from_row_slice(1, 1, &[arma.r]),
            DVector::from_row_slice(&[arma.m0_x, arma.m0_eps]),
            DMatrix::from_row_slice(2, 2, &[arma.p0_x, 0.0, 0.0, arma.p0_eps]),
        )
        .map_err(|e| anyhow::anyhow!("invalid arma11 model: {e}"))?
    } else if let Some(seas) = input.local_level_seasonal {
        ns_inference::timeseries::kalman::KalmanModel::local_level_seasonal(
            seas.period,
            seas.q_level,
            seas.q_season,
            seas.r,
            seas.level0,
            seas.p0_level,
            seas.p0_season,
        )
        .map_err(|e| anyhow::anyhow!("invalid local_level_seasonal model: {e}"))?
    } else if let Some(seas) = input.local_linear_trend_seasonal {
        ns_inference::timeseries::kalman::KalmanModel::local_linear_trend_seasonal(
            seas.period,
            seas.q_level,
            seas.q_slope,
            seas.q_season,
            seas.r,
            seas.level0,
            seas.slope0,
            seas.p0_level,
            seas.p0_slope,
            seas.p0_season,
        )
        .map_err(|e| anyhow::anyhow!("invalid local_linear_trend_seasonal model: {e}"))?
    } else {
        anyhow::bail!("unreachable: model spec missing");
    };

    let ys: Vec<DVector<f64>> = input
        .ys
        .into_iter()
        .enumerate()
        .map(|(t, y)| dvector_from_opt_vec(&format!("y[{t}]"), y))
        .collect::<Result<Vec<_>>>()?;

    Ok((model, ys))
}

fn load_kalman_input_with_raw(
    path: &PathBuf,
) -> Result<(
    ns_inference::timeseries::kalman::KalmanModel,
    Vec<DVector<f64>>,
    Vec<Vec<Option<f64>>>,
    KalmanSpecKind,
)> {
    let bytes = std::fs::read(path)?;
    let input: KalmanInputJson = serde_json::from_slice(&bytes)?;

    let mut model_count = 0usize;
    model_count += input.model.is_some() as usize;
    model_count += input.local_level.is_some() as usize;
    model_count += input.local_linear_trend.is_some() as usize;
    model_count += input.ar1.is_some() as usize;
    model_count += input.arma11.is_some() as usize;
    model_count += input.local_level_seasonal.is_some() as usize;
    model_count += input.local_linear_trend_seasonal.is_some() as usize;
    if model_count != 1 {
        anyhow::bail!(
            "expected exactly one of: model, local_level, local_linear_trend, ar1, arma11, local_level_seasonal, local_linear_trend_seasonal"
        );
    }

    let (model, kind) = if let Some(mj) = input.model {
        let model = ns_inference::timeseries::kalman::KalmanModel::new(
            dmatrix_from_nested("F", mj.f)?,
            dmatrix_from_nested("Q", mj.q)?,
            dmatrix_from_nested("H", mj.h)?,
            dmatrix_from_nested("R", mj.r)?,
            dvector_from_vec("m0", mj.m0)?,
            dmatrix_from_nested("P0", mj.p0)?,
        )
        .map_err(|e| anyhow::anyhow!("invalid Kalman model: {e}"))?;
        (model, KalmanSpecKind::Model)
    } else if let Some(ll) = input.local_level {
        let model =
            ns_inference::timeseries::kalman::KalmanModel::local_level(ll.q, ll.r, ll.m0, ll.p0)
                .map_err(|e| anyhow::anyhow!("invalid local_level model: {e}"))?;
        (model, KalmanSpecKind::LocalLevel)
    } else if let Some(lt) = input.local_linear_trend {
        let model = ns_inference::timeseries::kalman::KalmanModel::local_linear_trend(
            lt.q_level,
            lt.q_slope,
            lt.r,
            lt.level0,
            lt.slope0,
            lt.p0_level,
            lt.p0_slope,
        )
        .map_err(|e| anyhow::anyhow!("invalid local_linear_trend model: {e}"))?;
        (model, KalmanSpecKind::LocalLinearTrend)
    } else if let Some(ar) = input.ar1 {
        let model =
            ns_inference::timeseries::kalman::KalmanModel::ar1(ar.phi, ar.q, ar.r, ar.m0, ar.p0)
                .map_err(|e| anyhow::anyhow!("invalid ar1 model: {e}"))?;
        (model, KalmanSpecKind::Ar1)
    } else if let Some(arma) = input.arma11 {
        if !arma.phi.is_finite() {
            anyhow::bail!("invalid arma11: phi must be finite");
        }
        if !arma.theta.is_finite() {
            anyhow::bail!("invalid arma11: theta must be finite");
        }
        if !arma.sigma2.is_finite() || arma.sigma2 <= 0.0 {
            anyhow::bail!("invalid arma11: sigma2 must be finite and > 0");
        }
        if !arma.r.is_finite() || arma.r <= 0.0 {
            anyhow::bail!("invalid arma11: r must be finite and > 0");
        }
        if !arma.m0_x.is_finite() || !arma.m0_eps.is_finite() {
            anyhow::bail!("invalid arma11: initial state must be finite");
        }
        if !arma.p0_x.is_finite()
            || arma.p0_x <= 0.0
            || !arma.p0_eps.is_finite()
            || arma.p0_eps <= 0.0
        {
            anyhow::bail!("invalid arma11: p0_x/p0_eps must be finite and > 0");
        }

        let model = ns_inference::timeseries::kalman::KalmanModel::new(
            DMatrix::from_row_slice(2, 2, &[arma.phi, arma.theta, 0.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[arma.sigma2, arma.sigma2, arma.sigma2, arma.sigma2]),
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DMatrix::from_row_slice(1, 1, &[arma.r]),
            DVector::from_row_slice(&[arma.m0_x, arma.m0_eps]),
            DMatrix::from_row_slice(2, 2, &[arma.p0_x, 0.0, 0.0, arma.p0_eps]),
        )
        .map_err(|e| anyhow::anyhow!("invalid arma11 model: {e}"))?;
        (model, KalmanSpecKind::Arma11)
    } else if let Some(seas) = input.local_level_seasonal {
        let period = seas.period;
        let model = ns_inference::timeseries::kalman::KalmanModel::local_level_seasonal(
            seas.period,
            seas.q_level,
            seas.q_season,
            seas.r,
            seas.level0,
            seas.p0_level,
            seas.p0_season,
        )
        .map_err(|e| anyhow::anyhow!("invalid local_level_seasonal model: {e}"))?;
        (model, KalmanSpecKind::LocalLevelSeasonal { period })
    } else if let Some(seas) = input.local_linear_trend_seasonal {
        let period = seas.period;
        let model = ns_inference::timeseries::kalman::KalmanModel::local_linear_trend_seasonal(
            seas.period,
            seas.q_level,
            seas.q_slope,
            seas.q_season,
            seas.r,
            seas.level0,
            seas.slope0,
            seas.p0_level,
            seas.p0_slope,
            seas.p0_season,
        )
        .map_err(|e| anyhow::anyhow!("invalid local_linear_trend_seasonal model: {e}"))?;
        (model, KalmanSpecKind::LocalLinearTrendSeasonal { period })
    } else {
        anyhow::bail!("unreachable: model spec missing");
    };

    let ys_raw = input.ys.clone();
    let ys: Vec<DVector<f64>> = input
        .ys
        .into_iter()
        .enumerate()
        .map(|(t, y)| dvector_from_opt_vec(&format!("y[{t}]"), y))
        .collect::<Result<Vec<_>>>()?;

    Ok((model, ys, ys_raw, kind))
}

fn cmd_ts_kalman_filter(
    input: &PathBuf,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model, ys) = load_kalman_input(input)?;
    let fr = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
        .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;

    let output_json = serde_json::json!({
        "log_likelihood": fr.log_likelihood,
        "predicted_means": fr.predicted_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "predicted_covs": fr.predicted_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
        "filtered_means": fr.filtered_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "filtered_covs": fr.filtered_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_filter",
            serde_json::json!({}),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_kalman_smooth(
    input: &PathBuf,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model, ys) = load_kalman_input(input)?;
    let fr = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
        .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;
    let sr = ns_inference::timeseries::kalman::rts_smoother(&model, &fr)
        .map_err(|e| anyhow::anyhow!("rts_smoother failed: {e}"))?;

    let output_json = serde_json::json!({
        "log_likelihood": fr.log_likelihood,
        "filtered_means": fr.filtered_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "filtered_covs": fr.filtered_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
        "smoothed_means": sr.smoothed_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "smoothed_covs": sr.smoothed_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_smooth",
            serde_json::json!({}),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_kalman_em(
    input: &PathBuf,
    max_iter: usize,
    tol: f64,
    estimate_q: bool,
    estimate_r: bool,
    estimate_f: bool,
    estimate_h: bool,
    min_diag: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model, ys) = load_kalman_input(input)?;
    let cfg = ns_inference::timeseries::em::KalmanEmConfig {
        max_iter,
        tol,
        estimate_q,
        estimate_r,
        estimate_f,
        estimate_h,
        min_diag,
    };

    let res = ns_inference::timeseries::em::kalman_em(&model, &ys, cfg)
        .map_err(|e| anyhow::anyhow!("kalman_em failed: {e}"))?;

    let output_json = serde_json::json!({
        "converged": res.converged,
        "n_iter": res.n_iter,
        "loglik_trace": res.loglik_trace,
        "f": dmatrix_to_nested(&res.model.f),
        "h": dmatrix_to_nested(&res.model.h),
        "q": dmatrix_to_nested(&res.model.q),
        "r": dmatrix_to_nested(&res.model.r),
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_em",
            serde_json::json!({ "max_iter": max_iter, "tol": tol }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_kalman_fit(
    input: &PathBuf,
    max_iter: usize,
    tol: f64,
    estimate_q: bool,
    estimate_r: bool,
    estimate_f: bool,
    estimate_h: bool,
    min_diag: f64,
    forecast_steps: usize,
    no_smooth: bool,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model, ys) = load_kalman_input(input)?;
    let cfg = ns_inference::timeseries::em::KalmanEmConfig {
        max_iter,
        tol,
        estimate_q,
        estimate_r,
        estimate_f,
        estimate_h,
        min_diag,
    };

    let em = ns_inference::timeseries::em::kalman_em(&model, &ys, cfg)
        .map_err(|e| anyhow::anyhow!("kalman_em failed: {e}"))?;
    let fitted = em.model;

    let smooth_json = if no_smooth {
        serde_json::Value::Null
    } else {
        let fr = ns_inference::timeseries::kalman::kalman_filter(&fitted, &ys)
            .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;
        let sr = ns_inference::timeseries::kalman::rts_smoother(&fitted, &fr)
            .map_err(|e| anyhow::anyhow!("rts_smoother failed: {e}"))?;

        serde_json::json!({
            "log_likelihood": fr.log_likelihood,
            "filtered_means": fr.filtered_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
            "filtered_covs": fr.filtered_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
            "smoothed_means": sr.smoothed_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
            "smoothed_covs": sr.smoothed_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
        })
    };

    let forecast_json = if forecast_steps == 0 {
        serde_json::Value::Null
    } else {
        let fr = ns_inference::timeseries::kalman::kalman_filter(&fitted, &ys)
            .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;
        let fc = ns_inference::timeseries::forecast::kalman_forecast(&fitted, &fr, forecast_steps)
            .map_err(|e| anyhow::anyhow!("kalman_forecast failed: {e}"))?;
        serde_json::json!({
            "state_means": fc.state_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
            "state_covs": fc.state_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
            "obs_means": fc.obs_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
            "obs_covs": fc.obs_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
        })
    };

    let output_json = serde_json::json!({
        "em": {
            "converged": em.converged,
            "n_iter": em.n_iter,
            "loglik_trace": em.loglik_trace,
        },
        "model": {
            "f": dmatrix_to_nested(&fitted.f),
            "h": dmatrix_to_nested(&fitted.h),
            "q": dmatrix_to_nested(&fitted.q),
            "r": dmatrix_to_nested(&fitted.r),
            "m0": dvector_to_vec(&fitted.m0),
            "p0": dmatrix_to_nested(&fitted.p0),
        },
        "smooth": smooth_json,
        "forecast": forecast_json,
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_fit",
            serde_json::json!({ "max_iter": max_iter, "tol": tol, "forecast_steps": forecast_steps, "no_smooth": no_smooth }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn z_for_level(level: f64) -> Result<(f64, f64)> {
    if !(level.is_finite() && 0.0 < level && level < 1.0) {
        anyhow::bail!("level must be finite and in (0, 1)");
    }
    let alpha = 1.0 - level;
    let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let p = 1.0 - 0.5 * alpha;
    let z = normal.inverse_cdf(p);
    if !z.is_finite() || z <= 0.0 {
        anyhow::bail!("failed to compute z-score for level={level}");
    }
    Ok((alpha, z))
}

fn bands_from_means_covs(
    means: &[DVector<f64>],
    covs: &[DMatrix<f64>],
    z: f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    if means.is_empty() {
        anyhow::bail!("means must be non-empty");
    }
    if means.len() != covs.len() {
        anyhow::bail!("means/covs length mismatch");
    }
    let t_max = means.len();
    let dim = means[0].len();
    if dim == 0 {
        anyhow::bail!("means must have dim > 0");
    }

    let mut mean_out = Vec::with_capacity(t_max);
    let mut lo_out = Vec::with_capacity(t_max);
    let mut hi_out = Vec::with_capacity(t_max);

    for t in 0..t_max {
        let m = &means[t];
        let p = &covs[t];
        if m.len() != dim {
            anyhow::bail!("means must be rectangular");
        }
        if p.nrows() != dim || p.ncols() != dim {
            anyhow::bail!("covs must be T x dim x dim");
        }
        let mut lo = Vec::with_capacity(dim);
        let mut hi = Vec::with_capacity(dim);
        for j in 0..dim {
            let mu = m[j];
            let mut var = p[(j, j)];
            if !mu.is_finite() || !var.is_finite() {
                anyhow::bail!("non-finite mean/variance in bands");
            }
            if var < 0.0 {
                // Numerical noise guard.
                if var > -1e-12 {
                    var = 0.0;
                } else {
                    anyhow::bail!("negative marginal variance in bands");
                }
            }
            let sd = var.sqrt();
            if !sd.is_finite() {
                anyhow::bail!("non-finite sd in bands");
            }
            lo.push(mu - z * sd);
            hi.push(mu + z * sd);
        }
        mean_out.push(dvector_to_vec(m));
        lo_out.push(lo);
        hi_out.push(hi);
    }

    Ok((mean_out, lo_out, hi_out))
}

fn obs_bands_from_state(
    model: &ns_inference::timeseries::kalman::KalmanModel,
    means: &[DVector<f64>],
    covs: &[DMatrix<f64>],
    z: f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    if means.is_empty() {
        anyhow::bail!("means must be non-empty");
    }
    if means.len() != covs.len() {
        anyhow::bail!("means/covs length mismatch");
    }

    let t_max = means.len();
    let n_state = model.n_state();
    let n_obs = model.n_obs();
    if n_state == 0 || n_obs == 0 {
        anyhow::bail!("model dimensions must be > 0");
    }

    let mut mean_out = Vec::with_capacity(t_max);
    let mut lo_out = Vec::with_capacity(t_max);
    let mut hi_out = Vec::with_capacity(t_max);

    for t in 0..t_max {
        let m = &means[t];
        let p = &covs[t];
        if m.len() != n_state {
            anyhow::bail!("state mean has wrong dim");
        }
        if p.nrows() != n_state || p.ncols() != n_state {
            anyhow::bail!("state cov has wrong shape");
        }

        let y_mean = &model.h * m;
        let s = &model.h * p * model.h.transpose() + &model.r;
        if y_mean.len() != n_obs || s.nrows() != n_obs || s.ncols() != n_obs {
            anyhow::bail!("computed obs mean/cov has wrong shape");
        }

        let mut lo = Vec::with_capacity(n_obs);
        let mut hi = Vec::with_capacity(n_obs);
        for i in 0..n_obs {
            let mu = y_mean[i];
            let mut var = s[(i, i)];
            if !mu.is_finite() || !var.is_finite() {
                anyhow::bail!("non-finite obs mean/variance in bands");
            }
            if var < 0.0 {
                if var > -1e-12 {
                    var = 0.0;
                } else {
                    anyhow::bail!("negative obs marginal variance in bands");
                }
            }
            let sd = var.sqrt();
            if !sd.is_finite() {
                anyhow::bail!("non-finite obs sd in bands");
            }
            lo.push(mu - z * sd);
            hi.push(mu + z * sd);
        }

        mean_out.push(dvector_to_vec(&y_mean));
        lo_out.push(lo);
        hi_out.push(hi);
    }

    Ok((mean_out, lo_out, hi_out))
}

fn cmd_ts_kalman_viz(
    input: &PathBuf,
    max_iter: usize,
    tol: f64,
    estimate_q: bool,
    estimate_r: bool,
    estimate_f: bool,
    estimate_h: bool,
    min_diag: f64,
    level: f64,
    forecast_steps: usize,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model0, ys, ys_raw, kind) = load_kalman_input_with_raw(input)?;
    let t_max = ys.len();
    if t_max == 0 {
        anyhow::bail!("ys must be non-empty");
    }

    let (alpha, z) = z_for_level(level)?;

    let cfg = ns_inference::timeseries::em::KalmanEmConfig {
        max_iter,
        tol,
        estimate_q,
        estimate_r,
        estimate_f,
        estimate_h,
        min_diag,
    };
    let em = ns_inference::timeseries::em::kalman_em(&model0, &ys, cfg)
        .map_err(|e| anyhow::anyhow!("kalman_em failed: {e}"))?;
    let fitted = em.model;

    let fr = ns_inference::timeseries::kalman::kalman_filter(&fitted, &ys)
        .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;
    let sr = ns_inference::timeseries::kalman::rts_smoother(&fitted, &fr)
        .map_err(|e| anyhow::anyhow!("rts_smoother failed: {e}"))?;

    let (state_mean, state_lo, state_hi) =
        bands_from_means_covs(&sr.smoothed_means, &sr.smoothed_covs, z)?;
    let (obs_mean, obs_lo, obs_hi) =
        obs_bands_from_state(&fitted, &sr.smoothed_means, &sr.smoothed_covs, z)?;

    let default_state_labels: Vec<String> =
        (0..fitted.n_state()).map(|i| format!("x[{i}]")).collect();
    let default_obs_labels: Vec<String> = (0..fitted.n_obs()).map(|i| format!("y[{i}]")).collect();
    let mut state_labels: Vec<String> = match kind {
        KalmanSpecKind::LocalLevel => vec!["level".to_string()],
        KalmanSpecKind::Ar1 => vec!["x".to_string()],
        KalmanSpecKind::Arma11 => vec!["x".to_string(), "eps".to_string()],
        KalmanSpecKind::LocalLinearTrend => vec!["level".to_string(), "slope".to_string()],
        KalmanSpecKind::LocalLevelSeasonal { period } => {
            let mut out = vec!["level".to_string()];
            for k in 1..period {
                out.push(format!("season{k}"));
            }
            out
        }
        KalmanSpecKind::LocalLinearTrendSeasonal { period } => {
            let mut out = vec!["level".to_string(), "slope".to_string()];
            for k in 1..period {
                out.push(format!("season{k}"));
            }
            out
        }
        KalmanSpecKind::Model => default_state_labels.clone(),
    };
    if state_labels.len() != fitted.n_state() {
        state_labels = default_state_labels.clone();
    }
    let obs_labels = default_obs_labels;

    let forecast_json = if forecast_steps == 0 {
        serde_json::Value::Null
    } else {
        let fc = ns_inference::timeseries::forecast::kalman_forecast(&fitted, &fr, forecast_steps)
            .map_err(|e| anyhow::anyhow!("kalman_forecast failed: {e}"))?;
        let (fc_state_mean, fc_state_lo, fc_state_hi) =
            bands_from_means_covs(&fc.state_means, &fc.state_covs, z)?;
        let (fc_obs_mean, fc_obs_lo, fc_obs_hi) =
            bands_from_means_covs(&fc.obs_means, &fc.obs_covs, z)?;

        serde_json::json!({
            "t": (t_max..t_max + forecast_steps).collect::<Vec<_>>(),
            "state_mean": fc_state_mean,
            "state_lo": fc_state_lo,
            "state_hi": fc_state_hi,
            "obs_mean": fc_obs_mean,
            "obs_lo": fc_obs_lo,
            "obs_hi": fc_obs_hi,
        })
    };

    let output_json = serde_json::json!({
        "level": level,
        "alpha": alpha,
        "z": z,
        "t_obs": (0..t_max).collect::<Vec<_>>(),
        "ys": ys_raw,
        "state_labels": state_labels,
        "obs_labels": obs_labels,
        "log_likelihood": fr.log_likelihood,
        "smooth": {
            "state_mean": state_mean,
            "state_lo": state_lo,
            "state_hi": state_hi,
            "obs_mean": obs_mean,
            "obs_lo": obs_lo,
            "obs_hi": obs_hi,
        },
        "forecast": forecast_json,
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_viz",
            serde_json::json!({ "max_iter": max_iter, "tol": tol, "level": level, "forecast_steps": forecast_steps }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_kalman_forecast(
    input: &PathBuf,
    steps: usize,
    alpha: Option<f64>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model, ys) = load_kalman_input(input)?;
    let fr = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
        .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;
    let fc = ns_inference::timeseries::forecast::kalman_forecast(&model, &fr, steps)
        .map_err(|e| anyhow::anyhow!("kalman_forecast failed: {e}"))?;

    let mut output_json = serde_json::json!({
        "state_means": fc.state_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "state_covs": fc.state_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
        "obs_means": fc.obs_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "obs_covs": fc.obs_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    });

    if let Some(alpha) = alpha {
        let iv = ns_inference::timeseries::forecast::kalman_forecast_intervals(&fc, alpha)
            .map_err(|e| anyhow::anyhow!("kalman_forecast_intervals failed: {e}"))?;
        output_json["alpha"] = serde_json::json!(iv.alpha);
        output_json["z"] = serde_json::json!(iv.z);
        output_json["obs_lower"] =
            serde_json::json!(iv.obs_lower.iter().map(dvector_to_vec).collect::<Vec<_>>());
        output_json["obs_upper"] =
            serde_json::json!(iv.obs_upper.iter().map(dvector_to_vec).collect::<Vec<_>>());
    }

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_forecast",
            serde_json::json!({ "steps": steps, "alpha": alpha }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_kalman_simulate(
    input: &PathBuf,
    t_max: usize,
    seed: u64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (model, _ys) = load_kalman_input(input)?;
    let sim = ns_inference::timeseries::simulate::kalman_simulate(&model, t_max, seed)
        .map_err(|e| anyhow::anyhow!("kalman_simulate failed: {e}"))?;

    let output_json = serde_json::json!({
        "xs": sim.xs.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "ys": sim.ys.iter().map(dvector_to_vec).collect::<Vec<_>>(),
    });
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_kalman_simulate",
            serde_json::json!({ "t_max": t_max, "seed": seed }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_garch11_fit(
    input: &PathBuf,
    max_iter: u64,
    tol: f64,
    alpha_beta_max: f64,
    min_var: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let ys = load_returns_input(input)?;
    let mut cfg = ns_inference::timeseries::volatility::Garch11Config::default();
    cfg.optimizer.max_iter = max_iter;
    cfg.optimizer.tol = tol;
    cfg.alpha_beta_max = alpha_beta_max;
    cfg.min_var = min_var;

    let fit = ns_inference::timeseries::volatility::garch11_fit(&ys, cfg)
        .map_err(|e| anyhow::anyhow!("garch11_fit failed: {e}"))?;

    let conditional_sigma: Vec<f64> =
        fit.conditional_variance.iter().map(|v| v.max(0.0).sqrt()).collect();

    let output_json = serde_json::json!({
        "params": {
            "mu": fit.params.mu,
            "omega": fit.params.omega,
            "alpha": fit.params.alpha,
            "beta": fit.params.beta,
        },
        "log_likelihood": fit.log_likelihood,
        "conditional_variance": fit.conditional_variance,
        "conditional_sigma": conditional_sigma,
        "optimization": {
            "converged": fit.optimization.converged,
            "n_iter": fit.optimization.n_iter,
            "n_fev": fit.optimization.n_fev,
            "n_gev": fit.optimization.n_gev,
            "fval": fit.optimization.fval,
            "message": fit.optimization.message,
        }
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_garch11_fit",
            serde_json::json!({ "max_iter": max_iter, "tol": tol, "alpha_beta_max": alpha_beta_max }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_ts_sv_logchi2_fit(
    input: &PathBuf,
    max_iter: u64,
    tol: f64,
    log_eps: f64,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let ys = load_returns_input(input)?;
    let mut cfg = ns_inference::timeseries::volatility::SvLogChi2Config::default();
    cfg.optimizer.max_iter = max_iter;
    cfg.optimizer.tol = tol;
    cfg.log_eps = log_eps;

    let fit = ns_inference::timeseries::volatility::sv_logchi2_fit(&ys, cfg)
        .map_err(|e| anyhow::anyhow!("sv_logchi2_fit failed: {e}"))?;

    let output_json = serde_json::json!({
        "params": {
            "mu": fit.params.mu,
            "phi": fit.params.phi,
            "sigma": fit.params.sigma,
        },
        "log_likelihood": fit.log_likelihood,
        "smoothed_h": fit.smoothed_h,
        "smoothed_sigma": fit.smoothed_sigma,
        "optimization": {
            "converged": fit.optimization.converged,
            "n_iter": fit.optimization.n_iter,
            "n_fev": fit.optimization.n_fev,
            "n_gev": fit.optimization.n_gev,
            "fval": fit.optimization.fval,
            "message": fit.optimization.message,
        }
    });

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "timeseries_sv_logchi2_fit",
            serde_json::json!({ "max_iter": max_iter, "tol": tol, "log_eps": log_eps }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_hypotest(
    input: &PathBuf,
    mu: f64,
    expected_set: bool,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let start = std::time::Instant::now();
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::AsymptoticCLsContext::new(&mle, &model)?;
    let r = ctx.hypotest_qtilde(&mle, mu)?;
    tracing::debug!(mu_test = r.mu_test, cls = r.cls, mu_hat = r.mu_hat, "hypotest result");
    let wall_time_s = start.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "mu_test": r.mu_test,
        "cls": r.cls,
        "clsb": r.clsb,
        "clb": r.clb,
        "teststat": r.teststat,
        "q_mu": r.q_mu,
        "q_mu_a": r.q_mu_a,
        "mu_hat": r.mu_hat,
        "expected_set": if expected_set {
            let s = ctx.hypotest_qtilde_expected_set(&mle, mu)?;
            serde_json::json!({
                "nsigma_order": [2, 1, 0, -1, -2],
                "cls": s.expected,
            })
        } else {
            serde_json::Value::Null
        },
    });

    write_json(output, &output_json)?;
    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "hypotest",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "mu_test": r.mu_test,
                "mu_hat": r.mu_hat,
                "cls": r.cls,
                "clsb": r.clsb,
                "clb": r.clb,
                "q_mu": r.q_mu,
                "q_mu_a": r.q_mu_a,
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "hypotest",
            serde_json::json!({ "mu": mu, "expected_set": expected_set, "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_significance(
    input: &PathBuf,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let start = std::time::Instant::now();
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::AsymptoticCLsContext::new(&mle, &model)?;

    // Discovery test: background-only hypothesis (mu = 0).
    let r = ctx.hypotest_qtilde(&mle, 0.0)?;

    // p₀ = P(q₀ ≥ q₀_obs | μ=0).
    // When mu_test=0, clsb = P(q̃ ≥ q̃_obs | μ_test=0) = p₀.
    let p0 = r.clsb;

    // Observed significance: Z = √q₀ if μ̂ > 0, else 0.
    let z_obs = if r.mu_hat > 0.0 && r.q_mu > 0.0 { r.q_mu.sqrt() } else { 0.0 };

    // Expected significance from Asimov dataset (generated at μ̂):
    // Z_exp = √q₀_A. This is the significance one would expect if μ̂ is the
    // true signal strength (the "expected discovery significance").
    let z_exp = if r.q_mu_a > 0.0 { r.q_mu_a.sqrt() } else { 0.0 };

    let wall_time_s = start.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "command": "significance",
        "p0": p0,
        "z_obs": z_obs,
        "z_exp": z_exp,
        "mu_hat": r.mu_hat,
        "q0": r.q_mu,
        "q0_asimov": r.q_mu_a,
        "wall_time_s": wall_time_s,
    });

    tracing::info!(
        p0 = p0,
        z_obs = z_obs,
        z_exp = z_exp,
        mu_hat = r.mu_hat,
        "discovery significance"
    );

    write_json(output, &output_json)?;
    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "significance",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "p0": p0,
                "z_obs": z_obs,
                "z_exp": z_exp,
                "mu_hat": r.mu_hat,
                "q0": r.q_mu,
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "significance",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_goodness_of_fit(
    input: &PathBuf,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let start = std::time::Instant::now();
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let fit_result = mle.fit(&model)?;

    // Compute expected yields at best-fit parameters.
    let expected = model.expected_data(&fit_result.parameters)?;

    // Collect flat observed main-bin data (same channel ordering as expected_data).
    let observed_by_ch = model.observed_main_by_channel();
    let observed: Vec<f64> = observed_by_ch.iter().flat_map(|ch| ch.y.iter().copied()).collect();

    let n_bins = observed.len();
    if expected.len() != n_bins {
        anyhow::bail!("expected_data length ({}) != observed length ({})", expected.len(), n_bins);
    }

    // Poisson deviance: χ² = 2 Σᵢ [λᵢ - dᵢ + dᵢ·ln(dᵢ/λᵢ)]
    // Convention: 0·ln(0) = 0.
    let mut chi2 = 0.0_f64;
    for i in 0..n_bins {
        let d = observed[i];
        let e = expected[i];
        if e <= 0.0 {
            continue;
        }
        let term = e - d + if d > 0.0 { d * (d / e).ln() } else { 0.0 };
        chi2 += term;
    }
    chi2 *= 2.0;

    // Degrees of freedom: n_bins minus the number of floating parameters.
    // Conservative (standard TRExFitter convention).
    let n_floating = model
        .parameters()
        .iter()
        .enumerate()
        .filter(|(idx, p)| {
            let is_poi = model.poi_index() == Some(*idx);
            let is_fixed = (p.bounds.1 - p.bounds.0).abs() < 1e-15;
            !is_fixed || is_poi
        })
        .count();
    let ndof = if n_bins > n_floating { n_bins - n_floating } else { 1 };

    // p-value from χ² distribution.
    let p_value = if chi2.is_finite() && chi2 >= 0.0 && ndof > 0 {
        let chi2_dist = statrs::distribution::ChiSquared::new(ndof as f64)
            .map_err(|e| anyhow::anyhow!("ChiSquared distribution error: {e}"))?;
        1.0 - chi2_dist.cdf(chi2)
    } else {
        f64::NAN
    };

    let chi2_per_ndof = if ndof > 0 { chi2 / ndof as f64 } else { f64::NAN };
    let wall_time_s = start.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "command": "goodness_of_fit",
        "chi2": chi2,
        "ndof": ndof,
        "chi2_per_ndof": chi2_per_ndof,
        "p_value": p_value,
        "n_bins": n_bins,
        "n_floating": n_floating,
        "nll_fit": fit_result.nll,
        "converged": fit_result.converged,
        "wall_time_s": wall_time_s,
    });

    tracing::info!(
        chi2 = chi2,
        ndof = ndof,
        p_value = p_value,
        chi2_per_ndof = chi2_per_ndof,
        "goodness-of-fit"
    );

    write_json(output, &output_json)?;
    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "goodness_of_fit",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "chi2": chi2,
                "ndof": ndof,
                "p_value": p_value,
                "chi2_per_ndof": chi2_per_ndof,
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "goodness_of_fit",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_combine(inputs: &[PathBuf], output: Option<&PathBuf>, prefix_channels: bool) -> Result<()> {
    use ns_translate::pyhf::schema::{
        Measurement, MeasurementConfig, Observation, ParameterConfig, Workspace,
    };
    use std::collections::{HashMap, HashSet};

    if inputs.len() < 2 {
        anyhow::bail!("combine requires at least 2 input workspaces");
    }

    // Load all workspaces.
    let workspaces: Vec<(usize, Workspace)> = inputs
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let data = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
            let ws: Workspace = serde_json::from_str(&data)
                .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
            Ok((i, ws))
        })
        .collect::<Result<Vec<_>>>()?;

    // Collect all channel names to detect conflicts.
    let mut channel_counts: HashMap<String, usize> = HashMap::new();
    for (_, ws) in &workspaces {
        for ch in &ws.channels {
            *channel_counts.entry(ch.name.clone()).or_default() += 1;
        }
    }
    let has_conflicts = channel_counts.values().any(|&c| c > 1);
    if has_conflicts && !prefix_channels {
        let conflicts: Vec<&str> =
            channel_counts.iter().filter(|(_, c)| **c > 1).map(|(n, _)| n.as_str()).collect();
        anyhow::bail!(
            "channel name conflicts: {:?}. Use --prefix-channels to auto-prefix, or rename channels in inputs.",
            conflicts
        );
    }

    // Merge channels and observations.
    let mut combined_channels = Vec::new();
    let mut combined_observations = Vec::new();
    let mut seen_channels: HashSet<String> = HashSet::new();

    for (ws_idx, ws) in &workspaces {
        for ch in &ws.channels {
            let ch_name =
                if prefix_channels { format!("ws{}_{}", ws_idx, ch.name) } else { ch.name.clone() };
            if !seen_channels.insert(ch_name.clone()) {
                anyhow::bail!(
                    "duplicate channel name '{}' after prefixing (workspace {})",
                    ch_name,
                    ws_idx
                );
            }
            let mut new_ch = ch.clone();
            new_ch.name = ch_name.clone();
            combined_channels.push(new_ch);

            // Find matching observation.
            if let Some(obs) = ws.observations.iter().find(|o| o.name == ch.name) {
                combined_observations.push(Observation { name: ch_name, data: obs.data.clone() });
            }
        }
    }

    // Merge measurement configs: use POI from first workspace.
    // Parameter configs are merged by name (union, first-wins for settings).
    let poi = workspaces
        .first()
        .and_then(|(_, ws)| ws.measurements.first())
        .map(|m| m.config.poi.clone())
        .unwrap_or_else(|| "mu".to_string());

    let mut param_configs: Vec<ParameterConfig> = Vec::new();
    let mut seen_params: HashSet<String> = HashSet::new();

    for (_, ws) in &workspaces {
        for meas in &ws.measurements {
            for pc in &meas.config.parameters {
                if seen_params.insert(pc.name.clone()) {
                    param_configs.push(pc.clone());
                }
                // If already seen, skip (first workspace wins for that param config).
            }
        }
    }

    let combined_measurement = Measurement {
        name: "combined".to_string(),
        config: MeasurementConfig { poi, parameters: param_configs },
    };

    let combined_ws = Workspace {
        channels: combined_channels,
        observations: combined_observations,
        measurements: vec![combined_measurement],
        version: Some("1.0.0".to_string()),
    };

    let output_json = serde_json::to_value(&combined_ws)?;

    let n_ws = workspaces.len();
    let n_ch = combined_ws.channels.len();
    tracing::info!(n_workspaces = n_ws, n_channels = n_ch, "combined workspace");

    write_json(output, &output_json)?;
    Ok(())
}

fn cmd_preprocess_smooth(
    input: &PathBuf,
    output: Option<&PathBuf>,
    max_variation: f64,
) -> Result<()> {
    use ns_translate::pyhf::schema::{Modifier, Workspace};

    let data = std::fs::read_to_string(input)?;
    let mut ws: Workspace = serde_json::from_str(&data)?;

    let mut n_smoothed = 0usize;
    for ch in &mut ws.channels {
        for samp in &mut ch.samples {
            let nominal = &samp.data;
            for modifier in &mut samp.modifiers {
                if let Modifier::HistoSys { data, .. } = modifier {
                    let hi_delta: Vec<f64> =
                        data.hi_data.iter().zip(nominal).map(|(h, n)| h - n).collect();
                    let lo_delta: Vec<f64> =
                        data.lo_data.iter().zip(nominal).map(|(l, n)| l - n).collect();

                    let hi_smoothed = smooth_353qh_twice(&hi_delta);
                    let lo_smoothed = smooth_353qh_twice(&lo_delta);

                    data.hi_data = hi_smoothed
                        .iter()
                        .zip(nominal)
                        .map(|(d, n)| {
                            let mut v = n + d;
                            if max_variation > 0.0 && *n > 0.0 {
                                let cap = n * max_variation;
                                v = v.clamp(n - cap, n + cap);
                            }
                            v
                        })
                        .collect();
                    data.lo_data = lo_smoothed
                        .iter()
                        .zip(nominal)
                        .map(|(d, n)| {
                            let mut v = n + d;
                            if max_variation > 0.0 && *n > 0.0 {
                                let cap = n * max_variation;
                                v = v.clamp(n - cap, n + cap);
                            }
                            v
                        })
                        .collect();
                    n_smoothed += 1;
                }
            }
        }
    }

    tracing::info!(n_smoothed, "smoothing complete");
    let output_json = serde_json::to_value(&ws)?;
    write_json(output, &output_json)?;
    Ok(())
}

/// 353QH,twice smoothing (ROOT TH1::Smooth equivalent).
fn smooth_353qh_twice(x: &[f64]) -> Vec<f64> {
    fn running_median(x: &[f64], k: usize) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return vec![];
        }
        let half = k / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut window = Vec::with_capacity(k);
            for j in (i as isize - half as isize)..=(i as isize + half as isize) {
                let idx = j.clamp(0, n as isize - 1) as usize;
                window.push(x[idx]);
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out.push(window[window.len() / 2]);
        }
        out
    }

    fn hanning(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        if n <= 2 {
            return x.to_vec();
        }
        let mut out = Vec::with_capacity(n);
        out.push(x[0]);
        for i in 1..n - 1 {
            out.push(0.25 * x[i - 1] + 0.5 * x[i] + 0.25 * x[i + 1]);
        }
        out.push(x[n - 1]);
        out
    }

    fn pass_353qh(x: &[f64]) -> Vec<f64> {
        let s3 = running_median(x, 3);
        let s5 = running_median(&s3, 5);
        let s3b = running_median(&s5, 3);
        hanning(&s3b)
    }

    let first = pass_353qh(x);
    let residuals: Vec<f64> = x.iter().zip(&first).map(|(a, b)| a - b).collect();
    let second = pass_353qh(&residuals);
    first.iter().zip(&second).map(|(a, b)| a + b).collect()
}

fn cmd_preprocess_prune(input: &PathBuf, output: Option<&PathBuf>, threshold: f64) -> Result<()> {
    use ns_translate::pyhf::schema::{Modifier, Workspace};

    let data = std::fs::read_to_string(input)?;
    let mut ws: Workspace = serde_json::from_str(&data)?;

    let mut n_pruned = 0usize;
    for ch in &mut ws.channels {
        for samp in &mut ch.samples {
            let nominal = &samp.data;
            let before = samp.modifiers.len();
            samp.modifiers.retain(|modifier| match modifier {
                Modifier::HistoSys { data, .. } => {
                    let max_rel_hi = max_rel_delta(nominal, &data.hi_data);
                    let max_rel_lo = max_rel_delta(nominal, &data.lo_data);
                    max_rel_hi >= threshold || max_rel_lo >= threshold
                }
                Modifier::NormSys { data, .. } => {
                    let effect_hi = (data.hi - 1.0).abs();
                    let effect_lo = (data.lo - 1.0).abs();
                    effect_hi >= threshold || effect_lo >= threshold
                }
                _ => true,
            });
            n_pruned += before - samp.modifiers.len();
        }
    }

    tracing::info!(n_pruned, "pruning complete");
    let output_json = serde_json::to_value(&ws)?;
    write_json(output, &output_json)?;
    Ok(())
}

fn max_rel_delta(nominal: &[f64], variation: &[f64]) -> f64 {
    nominal
        .iter()
        .zip(variation)
        .filter(|(n, _)| **n > 0.0)
        .map(|(n, v)| ((v - n) / n).abs())
        .fold(0.0_f64, f64::max)
}

fn cmd_hypotest_toys(
    input: &PathBuf,
    mu: f64,
    n_toys: usize,
    seed: u64,
    expected_set: bool,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    gpu: Option<&str>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    let output_json = if let Some(device) = gpu {
        // GPU-accelerated path
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            if expected_set {
                let r = ns_inference::hypotest_qtilde_toys_expected_set_gpu(
                    &mle, &model, mu, n_toys, seed, device,
                )?;
                let o = &r.observed;
                serde_json::json!({
                    "mu_test": o.mu_test,
                    "cls": o.cls,
                    "clsb": o.clsb,
                    "clb": o.clb,
                    "q_obs": o.q_obs,
                    "mu_hat": o.mu_hat,
                    "n_toys": { "b": o.n_toys_b, "sb": o.n_toys_sb },
                    "n_error": { "b": o.n_error_b, "sb": o.n_error_sb },
                    "n_nonconverged": { "b": o.n_nonconverged_b, "sb": o.n_nonconverged_sb },
                    "seed": seed,
                    "gpu": device,
                    "expected_set": {
                        "nsigma_order": [2, 1, 0, -1, -2],
                        "cls": r.expected,
                    },
                })
            } else {
                let r =
                    ns_inference::hypotest_qtilde_toys_gpu(&mle, &model, mu, n_toys, seed, device)?;
                serde_json::json!({
                    "mu_test": r.mu_test,
                    "cls": r.cls,
                    "clsb": r.clsb,
                    "clb": r.clb,
                    "q_obs": r.q_obs,
                    "mu_hat": r.mu_hat,
                    "n_toys": { "b": r.n_toys_b, "sb": r.n_toys_sb },
                    "n_error": { "b": r.n_error_b, "sb": r.n_error_sb },
                    "n_nonconverged": { "b": r.n_nonconverged_b, "sb": r.n_nonconverged_sb },
                    "seed": seed,
                    "gpu": device,
                    "expected_set": serde_json::Value::Null,
                })
            }
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            let _ = device;
            anyhow::bail!("GPU hypotest-toys requires --features metal or --features cuda");
        }
    } else {
        // CPU path
        if expected_set {
            let r =
                ns_inference::hypotest_qtilde_toys_expected_set(&mle, &model, mu, n_toys, seed)?;
            let o = &r.observed;
            serde_json::json!({
                "mu_test": o.mu_test,
                "cls": o.cls,
                "clsb": o.clsb,
                "clb": o.clb,
                "q_obs": o.q_obs,
                "mu_hat": o.mu_hat,
                "n_toys": { "b": o.n_toys_b, "sb": o.n_toys_sb },
                "n_error": { "b": o.n_error_b, "sb": o.n_error_sb },
                "n_nonconverged": { "b": o.n_nonconverged_b, "sb": o.n_nonconverged_sb },
                "seed": seed,
                "expected_set": {
                    "nsigma_order": [2, 1, 0, -1, -2],
                    "cls": r.expected,
                },
            })
        } else {
            let r = ns_inference::hypotest_qtilde_toys(&mle, &model, mu, n_toys, seed)?;
            serde_json::json!({
                "mu_test": r.mu_test,
                "cls": r.cls,
                "clsb": r.clsb,
                "clb": r.clb,
                "q_obs": r.q_obs,
                "mu_hat": r.mu_hat,
                "n_toys": { "b": r.n_toys_b, "sb": r.n_toys_sb },
                "n_error": { "b": r.n_error_b, "sb": r.n_error_sb },
                "n_nonconverged": { "b": r.n_nonconverged_b, "sb": r.n_nonconverged_sb },
                "seed": seed,
                "expected_set": serde_json::Value::Null,
            })
        }
    };

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "hypotest_toys",
            serde_json::json!({
                "mu": mu,
                "n_toys": n_toys,
                "seed": seed,
                "expected_set": expected_set,
                "threads": threads,
            }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_upper_limit(
    input: &PathBuf,
    alpha: f64,
    expected: bool,
    scan_start: Option<f64>,
    scan_stop: Option<f64>,
    scan_points: Option<usize>,
    lo: f64,
    hi: Option<f64>,
    rtol: f64,
    max_iter: usize,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let start = std::time::Instant::now();
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::AsymptoticCLsContext::new(&mle, &model)?;

    let poi_hi = hi.unwrap_or_else(|| {
        model
            .poi_index()
            .and_then(|idx| model.parameters().get(idx).map(|p| p.bounds.1))
            .unwrap_or(10.0)
    });

    let output_json = if let (Some(start), Some(stop), Some(points)) =
        (scan_start, scan_stop, scan_points)
    {
        if points < 2 {
            anyhow::bail!("scan_points must be >= 2");
        }
        if !(stop > start) {
            anyhow::bail!("scan_stop must be > scan_start");
        }

        let step = (stop - start) / (points as f64 - 1.0);
        let scan: Vec<f64> = (0..points).map(|i| start + step * i as f64).collect();

        let (obs, exp) = ctx.upper_limits_qtilde_linear_scan(&mle, alpha, &scan)?;
        serde_json::json!({
            "mode": "scan",
            "alpha": alpha,
            "obs_limit": obs,
            "mu_up": obs,
            "exp_limits": exp,
            "scan": { "start": start, "stop": stop, "points": points },
        })
    } else {
        let (obs, exp_limits) = if expected {
            ctx.upper_limits_qtilde_bisection(&mle, alpha, lo, poi_hi, rtol, max_iter)?
        } else {
            (ctx.upper_limit_qtilde(&mle, alpha, lo, poi_hi, rtol, max_iter)?, [0.0; 5])
        };
        serde_json::json!({
            "mode": "bisection",
            "alpha": alpha,
            "obs_limit": obs,
            "mu_up": obs,
            "exp_limits": if expected { serde_json::json!(exp_limits) } else { serde_json::Value::Null },
            "bracket": { "lo": lo, "hi": poi_hi },
            "rtol": rtol,
            "max_iter": max_iter,
        })
    };
    let wall_time_s = start.elapsed().as_secs_f64();

    write_json(output, &output_json)?;
    if let Some(path) = json_metrics {
        let metrics_json = metrics_v0(
            "upper_limit",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "alpha": alpha,
                "mode": output_json.get("mode").cloned().unwrap_or(serde_json::Value::Null),
                "obs_limit": output_json.get("obs_limit").cloned().unwrap_or(serde_json::Value::Null),
                "exp_limits": output_json.get("exp_limits").cloned().unwrap_or(serde_json::Value::Null),
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "upper_limit",
            serde_json::json!({
                "alpha": alpha,
                "expected": expected,
                "scan_start": scan_start,
                "scan_stop": scan_stop,
                "scan_points": scan_points,
                "lo": lo,
                "hi": hi,
                "rtol": rtol,
                "max_iter": max_iter,
                "threads": threads
            }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_mass_scan(
    workspaces_dir: &PathBuf,
    alpha: f64,
    scan_start: f64,
    scan_stop: f64,
    scan_points: usize,
    labels: &[String],
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if scan_points < 2 {
        anyhow::bail!("scan_points must be >= 2");
    }
    if !(scan_stop > scan_start) {
        anyhow::bail!("scan_stop must be > scan_start");
    }
    if !(0.0 < alpha && alpha < 1.0) {
        anyhow::bail!("alpha must be in (0, 1)");
    }

    let mut ws_files: Vec<PathBuf> = std::fs::read_dir(workspaces_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|ext| ext.to_str()).map(|ext| ext == "json").unwrap_or(false)
        })
        .collect();
    ws_files.sort();

    if ws_files.is_empty() {
        anyhow::bail!("No .json workspace files found in {}", workspaces_dir.display());
    }

    tracing::info!(
        n_mass_points = ws_files.len(),
        dir = %workspaces_dir.display(),
        "starting mass scan"
    );

    let step = (scan_stop - scan_start) / (scan_points as f64 - 1.0);
    let scan: Vec<f64> = (0..scan_points).map(|i| scan_start + step * i as f64).collect();

    let start_t = std::time::Instant::now();
    let mut mass_points = Vec::with_capacity(ws_files.len());

    for (idx, ws_path) in ws_files.iter().enumerate() {
        let label = if idx < labels.len() {
            labels[idx].clone()
        } else {
            ws_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
        };

        tracing::info!(
            mass_point = idx + 1,
            total = ws_files.len(),
            label = %label,
            "computing CLs upper limit"
        );

        let model = load_model(ws_path, threads, false, interp_defaults)?;
        let mle = ns_inference::MaximumLikelihoodEstimator::new();
        let ctx = ns_inference::AsymptoticCLsContext::new(&mle, &model)?;

        let result = ctx.upper_limits_qtilde_linear_scan_result(&mle, alpha, &scan);

        match result {
            Ok(r) => {
                mass_points.push(serde_json::json!({
                    "index": idx,
                    "label": label,
                    "workspace": ws_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                    "obs_limit": r.observed_limit,
                    "exp_limits": r.expected_limits,
                    "mu_hat": ctx.mu_hat(),
                    "converged": true,
                }));
            }
            Err(e) => {
                tracing::warn!(label = %label, error = %e, "CLs computation failed for mass point");
                mass_points.push(serde_json::json!({
                    "index": idx,
                    "label": label,
                    "workspace": ws_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                    "obs_limit": serde_json::Value::Null,
                    "exp_limits": serde_json::Value::Null,
                    "error": e.to_string(),
                    "converged": false,
                }));
            }
        }
    }

    let wall_time_s = start_t.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "command": "mass_scan",
        "alpha": alpha,
        "scan": { "start": scan_start, "stop": scan_stop, "points": scan_points },
        "n_mass_points": mass_points.len(),
        "wall_time_s": wall_time_s,
        "mass_points": mass_points,
    });

    write_json(output, &output_json)?;

    if let Some(path) = json_metrics {
        let n_converged = mass_points
            .iter()
            .filter(|p| p.get("converged").and_then(|v| v.as_bool()).unwrap_or(false))
            .count();
        let metrics_json = metrics_v0(
            "mass_scan",
            "cpu",
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "alpha": alpha,
                "n_mass_points": mass_points.len(),
                "n_converged": n_converged,
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }

    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "mass_scan",
            serde_json::json!({
                "alpha": alpha,
                "scan_start": scan_start,
                "scan_stop": scan_stop,
                "scan_points": scan_points,
                "n_mass_points": mass_points.len(),
                "threads": threads,
            }),
            &ws_files[0],
            &output_json,
            false,
        )?;
    }

    tracing::info!(
        n_mass_points = mass_points.len(),
        wall_time_s = format!("{wall_time_s:.2}"),
        "mass scan complete"
    );

    Ok(())
}

fn cmd_scan(
    input: &PathBuf,
    start: f64,
    stop: f64,
    points: usize,
    output: Option<&PathBuf>,
    json_metrics: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    gpu: Option<&str>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if points < 2 {
        anyhow::bail!("points must be >= 2");
    }
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    let t0 = std::time::Instant::now();
    let step = (stop - start) / (points as f64 - 1.0);
    let mu_values: Vec<f64> = (0..points).map(|i| start + step * i as f64).collect();
    let scan = match gpu {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                ns_inference::profile_likelihood::scan_gpu(&mle, &model, &mu_values)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("--gpu cuda check should have bailed earlier")
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                ns_inference::profile_likelihood::scan_metal(&mle, &model, &mu_values)?
            }
            #[cfg(not(feature = "metal"))]
            {
                unreachable!("--gpu metal check should have bailed earlier")
            }
        }
        Some(_) => unreachable!("unknown device should have bailed earlier"),
        None => ns_inference::profile_likelihood::scan(&mle, &model, &mu_values)?,
    };
    let wall_time_s = t0.elapsed().as_secs_f64();

    let output_json = serde_json::json!({
        "poi_index": scan.poi_index,
        "mu_hat": scan.mu_hat,
        "nll_hat": scan.nll_hat,
        "points": scan.points.iter().map(|p| serde_json::json!({
            "mu": p.mu,
            "q_mu": p.q_mu,
            "nll_mu": p.nll_mu,
            "converged": p.converged,
            "n_iter": p.n_iter,
        })).collect::<Vec<_>>(),
    });

    write_json(output, &output_json)?;
    if let Some(path) = json_metrics {
        let q_min = scan.points.iter().fold(f64::INFINITY, |a, p| a.min(p.q_mu));
        let q_max = scan.points.iter().fold(f64::NEG_INFINITY, |a, p| a.max(p.q_mu));
        let metrics_json = metrics_v0(
            "scan",
            gpu.unwrap_or("cpu"),
            threads,
            false,
            wall_time_s,
            serde_json::json!({
                "poi_index": scan.poi_index,
                "mu_hat": scan.mu_hat,
                "nll_hat": scan.nll_hat,
                "n_points": scan.points.len(),
                "q_mu_min": if q_min.is_finite() { serde_json::json!(q_min) } else { serde_json::Value::Null },
                "q_mu_max": if q_max.is_finite() { serde_json::json!(q_max) } else { serde_json::Value::Null },
            }),
        )?;
        write_json(dash_means_stdout(path), &metrics_json)?;
    }
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "scan",
            serde_json::json!({ "start": start, "stop": stop, "points": points, "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_profile(
    input: &PathBuf,
    start: f64,
    stop: f64,
    points: usize,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if points < 2 {
        anyhow::bail!("points must be >= 2");
    }
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    let step = (stop - start) / (points as f64 - 1.0);
    let mu_values: Vec<f64> = (0..points).map(|i| start + step * i as f64).collect();
    let scan = ns_inference::profile_likelihood::scan(&mle, &model, &mu_values)?;
    let artifact: ns_viz::ProfileCurveArtifact = scan.into();

    let output_json = serde_json::to_value(artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_profile",
            serde_json::json!({ "start": start, "stop": stop, "points": points, "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_cls(
    input: &PathBuf,
    alpha: f64,
    scan_start: f64,
    scan_stop: f64,
    scan_points: usize,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if scan_points < 2 {
        anyhow::bail!("scan_points must be >= 2");
    }
    if !(scan_stop > scan_start) {
        anyhow::bail!("scan_stop must be > scan_start");
    }

    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::AsymptoticCLsContext::new(&mle, &model)?;

    let step = (scan_stop - scan_start) / (scan_points as f64 - 1.0);
    let scan: Vec<f64> = (0..scan_points).map(|i| scan_start + step * i as f64).collect();

    let artifact = ns_viz::ClsCurveArtifact::from_scan(&ctx, &mle, alpha, &scan)?;
    let output_json = serde_json::to_value(artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_cls",
            serde_json::json!({
                "alpha": alpha,
                "scan_start": scan_start,
                "scan_stop": scan_stop,
                "scan_points": scan_points,
                "threads": threads
            }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_ranking(
    input: &PathBuf,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    let entries = mle.ranking(&model)?;
    let artifact: ns_viz::RankingArtifact = entries.into();

    let output_json = serde_json::to_value(artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_ranking",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
struct FitResultJson {
    parameter_names: Vec<String>,
    bestfit: Vec<f64>,
    #[serde(default)]
    uncertainties: Vec<f64>,
    #[serde(default)]
    covariance: Option<Vec<f64>>,
    #[serde(default)]
    nll: f64,
    #[serde(default)]
    converged: bool,
    #[serde(default)]
    n_iter: usize,
    #[serde(default)]
    n_fev: usize,
    #[serde(default)]
    n_gev: usize,
}

fn params_from_fit_result(
    model: &ns_translate::pyhf::HistFactoryModel,
    fit: &FitResultJson,
) -> Result<Vec<f64>> {
    if fit.parameter_names.len() != fit.bestfit.len() {
        anyhow::bail!(
            "fit result length mismatch: parameter_names={} bestfit={}",
            fit.parameter_names.len(),
            fit.bestfit.len()
        );
    }

    let mut map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for (n, v) in fit.parameter_names.iter().zip(fit.bestfit.iter()) {
        map.insert(n.as_str(), *v);
    }

    let mut out = Vec::with_capacity(model.parameters().len());
    for p in model.parameters() {
        let v = map.get(p.name.as_str()).copied().ok_or_else(|| {
            ns_core::Error::Validation(format!("fit result missing parameter: {}", p.name))
        })?;
        out.push(v);
    }
    Ok(out)
}

fn cmd_report(
    input: &PathBuf,
    histfactory_xml: &PathBuf,
    fit: Option<&PathBuf>,
    out_dir: &PathBuf,
    overwrite: bool,
    include_covariance: bool,
    render: bool,
    pdf: Option<&PathBuf>,
    svg_dir: Option<&PathBuf>,
    python: Option<&PathBuf>,
    skip_uncertainty: bool,
    uncertainty_grouping: &str,
    threads: usize,
    interp_defaults: InterpDefaults,
    deterministic: bool,
    blind_regions: &[String],
) -> Result<()> {
    ensure_out_dir(out_dir, overwrite)?;

    let (workspace, model) = load_workspace_and_model(input, threads, interp_defaults)?;
    let params_prefit: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

    let (params_postfit, fit_result) = if let Some(fit_path) = fit {
        let bytes = std::fs::read(fit_path)?;
        let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;
        let params_postfit = params_from_fit_result(&model, &fit_json)?;

        let uncs_postfit = if fit_json.uncertainties.is_empty() {
            Vec::new()
        } else {
            if fit_json.parameter_names.len() != fit_json.uncertainties.len() {
                anyhow::bail!(
                    "fit result length mismatch: parameter_names={} uncertainties={}",
                    fit_json.parameter_names.len(),
                    fit_json.uncertainties.len()
                );
            }
            let mut map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
            for (n, v) in fit_json.parameter_names.iter().zip(fit_json.uncertainties.iter()) {
                map.insert(n.as_str(), *v);
            }
            let mut out = Vec::with_capacity(model.parameters().len());
            for p in model.parameters() {
                let v = map.get(p.name.as_str()).copied().ok_or_else(|| {
                    ns_core::Error::Validation(format!("fit result missing parameter: {}", p.name))
                })?;
                out.push(v);
            }
            out
        };

        let fit_result = if uncs_postfit.is_empty() {
            None
        } else {
            Some({
                let mut fr = ns_core::FitResult::new(
                    params_postfit.clone(),
                    uncs_postfit,
                    fit_json.nll,
                    fit_json.converged,
                    fit_json.n_iter,
                    fit_json.n_fev,
                    fit_json.n_gev,
                );
                fr.covariance = fit_json.covariance;
                fr
            })
        };

        (params_postfit, fit_result)
    } else {
        // Default: run the fit here (so `nextstat report` is "one command").
        let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model)?;

        let parameter_names: Vec<String> =
            model.parameters().iter().map(|p| p.name.clone()).collect();
        let fit_json_out = serde_json::json!({
            "parameter_names": parameter_names,
            "poi_index": model.poi_index(),
            "bestfit": result.parameters,
            "uncertainties": result.uncertainties,
            "nll": result.nll,
            "twice_nll": 2.0 * result.nll,
            "converged": result.converged,
            // Back-compat: keep `n_evaluations` as alias for optimizer iterations.
            "n_evaluations": result.n_iter,
            "n_iter": result.n_iter,
            "n_fev": result.n_fev,
            "n_gev": result.n_gev,
            "covariance": result.covariance,
        });
        let fit_json_out =
            if deterministic { normalize_json_for_determinism(fit_json_out) } else { fit_json_out };
        write_json_file(&out_dir.join("fit.json"), &fit_json_out)?;

        let fit_result = Some(result.clone());

        (result.parameters, fit_result)
    };

    // Distributions
    let mut data_by_channel: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for obs in &workspace.observations {
        data_by_channel.insert(obs.name.clone(), obs.data.clone());
    }
    let bin_edges_by_channel =
        ns_translate::histfactory::bin_edges_by_channel_from_xml(histfactory_xml)?;

    let blinded_set: Option<std::collections::HashSet<String>> = if blind_regions.is_empty() {
        None
    } else {
        Some(blind_regions.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
    };
    let blinded_ref = blinded_set.as_ref();

    let dist_artifact = ns_viz::distributions::distributions_artifact(
        &model,
        &data_by_channel,
        &bin_edges_by_channel,
        &params_prefit,
        &params_postfit,
        threads,
        blinded_ref,
    )?;
    let mut dist_json = serde_json::to_value(&dist_artifact)?;
    if deterministic {
        dist_json = normalize_json_for_determinism(dist_json);
    }
    write_json_file(&out_dir.join("distributions.json"), &dist_json)?;

    // Yields
    let yields_artifact = ns_viz::yields::yields_artifact(
        &model,
        &params_prefit,
        &params_postfit,
        threads,
        blinded_ref,
    )?;
    let mut yields_json = serde_json::to_value(&yields_artifact)?;
    if deterministic {
        yields_json = normalize_json_for_determinism(yields_json);
    }
    write_json_file(&out_dir.join("yields.json"), &yields_json)?;
    write_yields_tables(&yields_artifact, out_dir, deterministic)?;

    // Pulls + Corr depend on fit uncertainties/covariance.
    if let Some(fit_result) = fit_result.as_ref() {
        let pulls_artifact = ns_viz::pulls::pulls_artifact(&model, fit_result, threads)?;
        let mut pulls_json = serde_json::to_value(&pulls_artifact)?;
        if deterministic {
            pulls_json = normalize_json_for_determinism(pulls_json);
        }
        write_json_file(&out_dir.join("pulls.json"), &pulls_json)?;

        if fit_result.covariance.is_some() {
            let corr_artifact =
                ns_viz::corr::corr_artifact(&model, fit_result, threads, include_covariance)?;
            let mut corr_json = serde_json::to_value(&corr_artifact)?;
            if deterministic {
                corr_json = normalize_json_for_determinism(corr_json);
            }
            write_json_file(&out_dir.join("corr.json"), &corr_json)?;
        } else {
            tracing::warn!("fit covariance missing; skipping corr.json");
        }
    } else {
        tracing::warn!("fit uncertainties missing/omitted; skipping pulls.json and corr.json");
    }

    if !skip_uncertainty {
        let mle = ns_inference::MaximumLikelihoodEstimator::new();
        let entries = mle.ranking(&model)?;
        let ranking_artifact: ns_viz::RankingArtifact = entries.into();
        let unc = ns_viz::uncertainty::uncertainty_breakdown_from_ranking(
            &ranking_artifact,
            uncertainty_grouping,
            threads,
        )?;
        let mut unc_json = serde_json::to_value(&unc)?;
        if deterministic {
            unc_json = normalize_json_for_determinism(unc_json);
        }
        write_json_file(&out_dir.join("uncertainty.json"), &unc_json)?;
    }

    if render {
        let default_python = {
            let venv = PathBuf::from(".venv/bin/python");
            if venv.exists() { venv } else { PathBuf::from("python3") }
        };
        let python = python.cloned().unwrap_or(default_python);
        let pdf = pdf.cloned().unwrap_or_else(|| out_dir.join("report.pdf"));
        let svg_dir = svg_dir.cloned().unwrap_or_else(|| out_dir.join("svg"));
        std::fs::create_dir_all(&svg_dir)?;

        // Matplotlib may try to write cache/config under $HOME; in restricted environments this
        // can fail or create noisy warnings. Force a writable, repo-local cache dir.
        let mplconfigdir = PathBuf::from("tmp/mplconfig");
        let _ = std::fs::create_dir_all(&mplconfigdir);

        // Only inject repo-local Python sources when the in-tree compiled extension exists.
        // Otherwise this would shadow an installed wheel (CI / end users) and break imports.
        let local_ext_present = {
            let pkg = PathBuf::from("bindings/ns-py/python/nextstat");
            if pkg.is_dir() {
                std::fs::read_dir(pkg)
                    .ok()
                    .and_then(|it| {
                        for e in it.flatten() {
                            let name = e.file_name().to_string_lossy().to_string();
                            if name.starts_with("_core.")
                                && (name.ends_with(".so")
                                    || name.ends_with(".pyd")
                                    || name.ends_with(".dylib")
                                    || name.ends_with(".dll"))
                            {
                                return Some(());
                            }
                        }
                        None
                    })
                    .is_some()
            } else {
                false
            }
        };
        let force_py_path = std::env::var("NEXTSTAT_FORCE_PYTHONPATH").ok().as_deref() == Some("1");

        let mut py = Command::new(&python);
        py.env("MPLCONFIGDIR", &mplconfigdir);
        if local_ext_present || force_py_path {
            let mut pythonpath = std::ffi::OsString::new();
            pythonpath.push("bindings/ns-py/python");
            if let Some(existing) = std::env::var_os("PYTHONPATH")
                && !existing.is_empty()
            {
                pythonpath.push(":");
                pythonpath.push(existing);
            }
            py.env("PYTHONPATH", pythonpath);
        }

        let status = py
            .arg("-m")
            .arg("nextstat.report")
            .arg("render")
            .arg("--input-dir")
            .arg(out_dir)
            .arg("--pdf")
            .arg(&pdf)
            .arg("--svg-dir")
            .arg(&svg_dir)
            .status()?;

        if !status.success() {
            anyhow::bail!(
                "report renderer failed (python={}, status={})",
                python.display(),
                status
            );
        }
    }

    Ok(())
}

fn cmd_viz_distributions(
    input: &PathBuf,
    histfactory_xml: &PathBuf,
    fit: Option<&PathBuf>,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let (workspace, model) = load_workspace_and_model(input, threads, interp_defaults)?;
    let params_prefit: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

    let params_postfit: Vec<f64> = if let Some(fit_path) = fit {
        let bytes = std::fs::read(fit_path)?;
        let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;
        params_from_fit_result(&model, &fit_json)?
    } else {
        params_prefit.clone()
    };

    let mut data_by_channel: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for obs in &workspace.observations {
        data_by_channel.insert(obs.name.clone(), obs.data.clone());
    }

    let bin_edges_by_channel =
        ns_translate::histfactory::bin_edges_by_channel_from_xml(histfactory_xml)?;

    let artifact = ns_viz::distributions::distributions_artifact(
        &model,
        &data_by_channel,
        &bin_edges_by_channel,
        &params_prefit,
        &params_postfit,
        threads,
        None,
    )?;

    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_distributions",
            serde_json::json!({
                "threads": threads,
                "histfactory_xml": histfactory_xml,
                "fit": fit,
            }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_pulls(
    input: &PathBuf,
    fit: &PathBuf,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;

    let bytes = std::fs::read(fit)?;
    let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;
    if fit_json.uncertainties.is_empty() {
        anyhow::bail!("fit result missing `uncertainties` (required for pulls)");
    }

    let params_postfit = params_from_fit_result(&model, &fit_json)?;
    let uncs_postfit = {
        if fit_json.parameter_names.len() != fit_json.uncertainties.len() {
            anyhow::bail!(
                "fit result length mismatch: parameter_names={} uncertainties={}",
                fit_json.parameter_names.len(),
                fit_json.uncertainties.len()
            );
        }
        let mut map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for (n, v) in fit_json.parameter_names.iter().zip(fit_json.uncertainties.iter()) {
            map.insert(n.as_str(), *v);
        }
        let mut out = Vec::with_capacity(model.parameters().len());
        for p in model.parameters() {
            let v = map.get(p.name.as_str()).copied().ok_or_else(|| {
                ns_core::Error::Validation(format!("fit result missing parameter: {}", p.name))
            })?;
            out.push(v);
        }
        out
    };

    let fit_result = {
        let mut fr = ns_core::FitResult::new(
            params_postfit,
            uncs_postfit,
            fit_json.nll,
            fit_json.converged,
            fit_json.n_iter,
            fit_json.n_fev,
            fit_json.n_gev,
        );
        fr.covariance = fit_json.covariance;
        fr
    };
    let artifact = ns_viz::pulls::pulls_artifact(&model, &fit_result, threads)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_pulls",
            serde_json::json!({ "threads": threads, "fit": fit }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_gammas(
    input: &PathBuf,
    fit: &PathBuf,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;

    let bytes = std::fs::read(fit)?;
    let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;
    if fit_json.uncertainties.is_empty() {
        anyhow::bail!("fit result missing `uncertainties` (required for gammas)");
    }

    let params_postfit = params_from_fit_result(&model, &fit_json)?;
    let uncs_postfit = {
        if fit_json.parameter_names.len() != fit_json.uncertainties.len() {
            anyhow::bail!(
                "fit result length mismatch: parameter_names={} uncertainties={}",
                fit_json.parameter_names.len(),
                fit_json.uncertainties.len()
            );
        }
        let mut map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for (n, v) in fit_json.parameter_names.iter().zip(fit_json.uncertainties.iter()) {
            map.insert(n.as_str(), *v);
        }
        let mut out = Vec::with_capacity(model.parameters().len());
        for p in model.parameters() {
            let v = map.get(p.name.as_str()).copied().ok_or_else(|| {
                ns_core::Error::Validation(format!("fit result missing parameter: {}", p.name))
            })?;
            out.push(v);
        }
        out
    };

    let fit_result = {
        let mut fr = ns_core::FitResult::new(
            params_postfit,
            uncs_postfit,
            fit_json.nll,
            fit_json.converged,
            fit_json.n_iter,
            fit_json.n_fev,
            fit_json.n_gev,
        );
        fr.covariance = fit_json.covariance;
        fr
    };
    let artifact = ns_viz::gammas::gammas_artifact(&model, &fit_result)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_gammas",
            serde_json::json!({ "threads": threads, "fit": fit }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_separation(
    input: &PathBuf,
    signal_samples: &[String],
    histfactory_xml: Option<&PathBuf>,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;
    let init_params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

    // Resolve signal sample names: explicit or auto-detect from POI
    let signal_set: std::collections::HashSet<String> = if !signal_samples.is_empty() {
        signal_samples.iter().cloned().collect()
    } else {
        // Auto-detect: any sample whose name appears as the first sample in any channel
        // that has a normfactor modifier named after the POI parameter.
        let poi_name =
            model.poi_index().and_then(|i| model.parameters().get(i)).map(|p| p.name.clone());
        if let Some(ref poi) = poi_name {
            // The first sample in each channel that contains a normfactor with the POI name
            // is the signal. We use the workspace schema to detect this.
            let ws_bytes = std::fs::read(input)?;
            let ws: ns_translate::pyhf::Workspace = serde_json::from_slice(&ws_bytes)?;
            let mut names = std::collections::HashSet::new();
            for ch in &ws.channels {
                for sample in &ch.samples {
                    for modifier in &sample.modifiers {
                        if let ns_translate::pyhf::schema::Modifier::NormFactor { name, .. } =
                            modifier
                            && name == poi
                        {
                            names.insert(sample.name.clone());
                        }
                    }
                }
            }
            names
        } else {
            std::collections::HashSet::new()
        }
    };

    let bin_edges: std::collections::HashMap<String, Vec<f64>> =
        if let Some(xml_path) = histfactory_xml {
            ns_translate::histfactory::bin_edges_by_channel_from_xml(xml_path)?
        } else {
            std::collections::HashMap::new()
        };

    let artifact =
        ns_viz::separation::separation_artifact(&model, &init_params, &signal_set, &bin_edges)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_separation",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_summary(
    fits: &[PathBuf],
    labels: &[String],
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if !labels.is_empty() && labels.len() != fits.len() {
        anyhow::bail!(
            "--labels count ({}) must match number of fit files ({})",
            labels.len(),
            fits.len()
        );
    }

    let mut entries = Vec::with_capacity(fits.len());
    for (i, fit_path) in fits.iter().enumerate() {
        let bytes = std::fs::read(fit_path)?;
        let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;

        let label = if !labels.is_empty() {
            labels[i].clone()
        } else {
            fit_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| format!("fit_{}", i))
        };

        let mu_idx = fit_json.parameter_names.iter().position(|n| n == "mu");
        let poi_hat = mu_idx.and_then(|idx| fit_json.bestfit.get(idx).copied()).unwrap_or(f64::NAN);
        let poi_sigma =
            mu_idx.and_then(|idx| fit_json.uncertainties.get(idx).copied()).unwrap_or(f64::NAN);

        entries.push(ns_viz::summary::SummaryEntry {
            label,
            mu_hat: poi_hat,
            sigma: poi_sigma,
            nll: fit_json.nll,
            converged: fit_json.converged,
        });
    }

    let poi_name = "mu".to_string();
    let artifact = ns_viz::summary::summary_artifact(&poi_name, entries)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_summary",
            serde_json::json!({ "n_fits": fits.len() }),
            &fits[0],
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_pie(
    input: &PathBuf,
    fit: Option<&PathBuf>,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;

    let params: Vec<f64> = if let Some(fit_path) = fit {
        let bytes = std::fs::read(fit_path)?;
        let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;
        params_from_fit_result(&model, &fit_json)?
    } else {
        model.parameters().iter().map(|p| p.init).collect()
    };

    let artifact = ns_viz::pie::pie_artifact(&model, &params)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_pie",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

fn cmd_viz_corr(
    input: &PathBuf,
    fit: &PathBuf,
    include_covariance: bool,
    output: Option<&PathBuf>,
    threads: usize,
    interp_defaults: InterpDefaults,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads, false, interp_defaults)?;

    let bytes = std::fs::read(fit)?;
    let fit_json: FitResultJson = serde_json::from_slice(&bytes)?;
    if fit_json.uncertainties.is_empty() {
        anyhow::bail!("fit result missing `uncertainties` (required for corr)");
    }
    if fit_json.covariance.is_none() {
        anyhow::bail!("fit result missing `covariance` (required for corr)");
    }

    let params_postfit = params_from_fit_result(&model, &fit_json)?;
    let uncs_postfit = {
        if fit_json.parameter_names.len() != fit_json.uncertainties.len() {
            anyhow::bail!(
                "fit result length mismatch: parameter_names={} uncertainties={}",
                fit_json.parameter_names.len(),
                fit_json.uncertainties.len()
            );
        }
        let mut map: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for (n, v) in fit_json.parameter_names.iter().zip(fit_json.uncertainties.iter()) {
            map.insert(n.as_str(), *v);
        }
        let mut out = Vec::with_capacity(model.parameters().len());
        for p in model.parameters() {
            let v = map.get(p.name.as_str()).copied().ok_or_else(|| {
                ns_core::Error::Validation(format!("fit result missing parameter: {}", p.name))
            })?;
            out.push(v);
        }
        out
    };

    let fit_result = {
        let mut fr = ns_core::FitResult::new(
            params_postfit,
            uncs_postfit,
            fit_json.nll,
            fit_json.converged,
            fit_json.n_iter,
            fit_json.n_fev,
            fit_json.n_gev,
        );
        fr.covariance = fit_json.covariance;
        fr
    };

    let artifact = ns_viz::corr::corr_artifact(&model, &fit_result, threads, include_covariance)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "viz_corr",
            serde_json::json!({
                "threads": threads,
                "fit": fit,
                "include_covariance": include_covariance,
            }),
            input,
            &output_json,
            false,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_shard_plan_rejects_too_few_shards_for_u32_offsets() {
        let err = normalize_cuda_device_shard_plan(
            Some("cuda"),
            true,
            &[0],
            Some(4),
            10_000,
            0,
            2_000_000,
        )
        .expect_err("expected u32 offset guard to reject undersharded plan");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("too small for 32-bit toy offsets"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn cuda_shard_plan_auto_enables_host_sharding_for_large_event_budget() {
        let plan = normalize_cuda_device_shard_plan(
            Some("cuda"),
            false,
            &[0, 1],
            None,
            10_000,
            0,
            2_000_000,
        )
        .expect("auto host sharding should be computed");
        assert!(
            !plan.is_empty(),
            "expected non-empty shard plan when u32 offset guard requires sharding"
        );
        assert!(
            plan.len() >= 5,
            "expected at least 5 shards for this event budget, got {}",
            plan.len()
        );
    }

    #[test]
    fn cuda_shard_plan_keeps_host_path_unsharded_when_safe() {
        let plan =
            normalize_cuda_device_shard_plan(Some("cuda"), false, &[0, 1], None, 100, 0, 10_000)
                .expect("safe host path should not require auto-sharding");
        assert!(plan.is_empty(), "expected empty plan for safe host workload, got {plan:?}");
    }

    #[test]
    fn metal_u32_guard_rejects_predicted_overflow() {
        let max_toys = metal_max_toys_per_batch_for_u32_offsets(10_000, 2_000_000)
            .expect("batch planner must compute max toys for Metal");
        assert!(
            max_toys < 10_000,
            "expected chunking to be required for oversized workload, max_toys={max_toys}"
        );
    }

    #[test]
    fn metal_u32_guard_allows_safe_workload() {
        let max_toys = metal_max_toys_per_batch_for_u32_offsets(100, 10_000)
            .expect("batch planner should accept safe workload");
        assert!(max_toys >= 100, "safe workload should fit in one batch, max_toys={max_toys}");
    }
}
