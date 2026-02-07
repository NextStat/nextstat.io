//! NextStat CLI

use anyhow::Result;
use clap::{Parser, Subcommand};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use statrs::distribution::ContinuousCDF;
use std::path::PathBuf;
use std::process::Command;

mod analysis_spec;
mod discover;
mod report;
mod run;

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

    /// Perform MLE fit
    Fit {
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

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
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

        /// Threads (0 = auto). Use 1 for deterministic parity.
        #[arg(long, default_value = "1")]
        threads: usize,
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

        /// Also emit an analysis spec YAML (mode=trex_config_yaml) to this path.
        #[arg(long)]
        analysis_yaml: Option<PathBuf>,

        /// Also emit a JSON coverage report (unknown keys/attrs) to this path.
        #[arg(long)]
        coverage_json: Option<PathBuf>,
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
enum ConfigCommands {
    /// Print a JSON schema to stdout (for IDE integration / validation)
    Schema {
        /// Schema name (default: analysis_spec_v0). Known: analysis_spec_v0, baseline_v0,
        /// report_distributions_v0, report_pulls_v0, report_corr_v0, report_yields_v0, report_uncertainty_v0.
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt().with_max_level(cli.log_level).with_target(false).init();

    match cli.command {
        Commands::Run { config } => cmd_run(&config, cli.bundle.as_ref()),
        Commands::Validate { config } => cmd_validate(&config),
        Commands::Fit { input, output, threads } => {
            cmd_fit(&input, output.as_ref(), threads, cli.bundle.as_ref())
        }
        Commands::Hypotest { input, mu, expected_set, output, threads } => {
            cmd_hypotest(&input, mu, expected_set, output.as_ref(), threads, cli.bundle.as_ref())
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
            threads,
            cli.bundle.as_ref(),
        ),
        Commands::Scan { input, start, stop, points, output, threads } => {
            cmd_scan(&input, start, stop, points, output.as_ref(), threads, cli.bundle.as_ref())
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
            deterministic,
        ),
        Commands::Viz { command } => match command {
            VizCommands::Profile { input, start, stop, points, output, threads } => {
                cmd_viz_profile(
                    &input,
                    start,
                    stop,
                    points,
                    output.as_ref(),
                    threads,
                    cli.bundle.as_ref(),
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
                cli.bundle.as_ref(),
            ),
            VizCommands::Ranking { input, output, threads } => {
                cmd_viz_ranking(&input, output.as_ref(), threads, cli.bundle.as_ref())
            }
            VizCommands::Pulls { input, fit, output, threads } => {
                cmd_viz_pulls(&input, &fit, output.as_ref(), threads, cli.bundle.as_ref())
            }
            VizCommands::Corr { input, fit, include_covariance, output, threads } => cmd_viz_corr(
                &input,
                &fit,
                include_covariance,
                output.as_ref(),
                threads,
                cli.bundle.as_ref(),
            ),
            VizCommands::Distributions { input, histfactory_xml, fit, output, threads } => {
                cmd_viz_distributions(
                    &input,
                    &histfactory_xml,
                    fit.as_ref(),
                    output.as_ref(),
                    threads,
                    cli.bundle.as_ref(),
                )
            }
        },
        Commands::Import { command } => match command {
            ImportCommands::Histfactory { xml, dir, output } => {
                let xml_path = if let Some(xml) = xml {
                    xml
                } else {
                    let dir = dir.expect("clap enforces: --xml or --dir");
                    discover::discover_single_combination_xml(&dir)?
                };
                cmd_import_histfactory(&xml_path, output.as_ref(), cli.bundle.as_ref())
            }
            ImportCommands::TrexConfig {
                config,
                base_dir,
                output,
                analysis_yaml,
                coverage_json,
            } => cmd_import_trex_config(
                &config,
                base_dir.as_ref(),
                output.as_ref(),
                analysis_yaml.as_ref(),
                coverage_json.as_ref(),
                cli.bundle.as_ref(),
            ),
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
                cmd_ts_kalman_filter(&input, output.as_ref(), cli.bundle.as_ref())
            }
            TimeseriesCommands::KalmanSmooth { input, output } => {
                cmd_ts_kalman_smooth(&input, output.as_ref(), cli.bundle.as_ref())
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
                cli.bundle.as_ref(),
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
                cli.bundle.as_ref(),
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
                cli.bundle.as_ref(),
            ),
            TimeseriesCommands::KalmanForecast { input, steps, alpha, output } => {
                cmd_ts_kalman_forecast(&input, steps, alpha, output.as_ref(), cli.bundle.as_ref())
            }
            TimeseriesCommands::KalmanSimulate { input, t_max, seed, output } => {
                cmd_ts_kalman_simulate(&input, t_max, seed, output.as_ref(), cli.bundle.as_ref())
            }
        },
        Commands::Config { command } => match command {
            ConfigCommands::Schema { name } => cmd_config_schema(name.as_deref()),
        },
        Commands::Version => {
            println!("nextstat {}", ns_core::VERSION);
            Ok(())
        }
    }
}

fn cmd_run(config_path: &PathBuf, bundle: Option<&PathBuf>) -> Result<()> {
    match analysis_spec::read_any_run_config(config_path.as_path())? {
        analysis_spec::AnyRunConfig::Legacy(cfg) => cmd_run_legacy(config_path, bundle, &cfg),
        analysis_spec::AnyRunConfig::SpecV0(spec) => cmd_run_spec_v0(config_path, bundle, &spec),
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
        other => anyhow::bail!(
            "unknown schema name: {other}. Known: analysis_spec_v0, baseline_v0, report_distributions_v0, report_pulls_v0, report_corr_v0, report_yields_v0, report_uncertainty_v0"
        ),
    };

    print!("{schema}");
    Ok(())
}

fn cmd_run_legacy(
    config_path: &PathBuf,
    bundle: Option<&PathBuf>,
    cfg: &run::RunConfig,
) -> Result<()> {
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
        cfg.deterministic,
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
    spec: &analysis_spec::AnalysisSpecV0,
) -> Result<()> {
    let plan = spec.to_run_plan(config_path.as_path())?;
    let deterministic = plan.threads == 1;

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
            std::fs::write(
                &preprocess.config_json,
                serde_json::to_string_pretty(&config_val)?,
            )?;
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
        cmd_fit(&plan.workspace_json, Some(fit_out), plan.threads, /*bundle*/ None)?;
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
            plan.threads,
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
            deterministic,
        )?;
    }

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
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn cmd_import_histfactory(
    xml: &PathBuf,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if bundle.is_some() {
        anyhow::bail!("--bundle is not supported for `nextstat import histfactory` yet");
    }

    tracing::info!(path = %xml.display(), "importing HistFactory combination.xml");
    let ws = ns_translate::histfactory::from_xml(xml)?;
    let output_json = serde_json::to_value(&ws)?;
    write_json(output, &output_json)?;
    Ok(())
}

fn cmd_import_trex_config(
    config: &PathBuf,
    base_dir: Option<&PathBuf>,
    output: Option<&PathBuf>,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if bundle.is_some() {
        anyhow::bail!("--bundle is not supported for `nextstat import trex-config` yet");
    }

    tracing::info!(path = %config.display(), "importing TRExFitter config");
    let text = std::fs::read_to_string(config)?;

    let base_dir = base_dir
        .map(|p| p.as_path())
        .or_else(|| config.parent())
        .unwrap_or_else(|| std::path::Path::new("."));

    let ws = ns_translate::trex::workspace_from_str(&text, base_dir)?;
    let output_json = serde_json::to_value(&ws)?;
    write_json(output, &output_json)?;
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

    let input = input
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("failed to resolve input workspace: {}: {}", input.display(), e))?;
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

    let py = python
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("python3"));

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
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads)?;

    let mle = ns_inference::mle::MaximumLikelihoodEstimator::new();
    let result = mle.fit(&model)?;
    tracing::info!(nll = result.nll, converged = result.converged, "fit complete");

    let parameter_names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();

    let output_json = serde_json::json!({
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

    write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "fit",
            serde_json::json!({ "threads": threads }),
            input,
            &output_json,
        )?;
    }
    Ok(())
}

fn load_model(input: &PathBuf, threads: usize) -> Result<ns_translate::pyhf::HistFactoryModel> {
    if threads > 0 {
        // Best-effort; if a global pool already exists, keep going.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }

    tracing::info!(path = %input.display(), "loading workspace");
    let json = std::fs::read_to_string(input)?;
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?;
    tracing::info!(parameters = model.parameters().len(), "workspace loaded");
    Ok(model)
}

fn load_workspace_and_model(
    input: &PathBuf,
    threads: usize,
) -> Result<(ns_translate::pyhf::Workspace, ns_translate::pyhf::HistFactoryModel)> {
    if threads > 0 {
        // Best-effort; if a global pool already exists, keep going.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }

    tracing::info!(path = %input.display(), "loading workspace");
    let json = std::fs::read_to_string(input)?;
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?;
    tracing::info!(parameters = model.parameters().len(), "workspace loaded");
    Ok((workspace, model))
}

fn normalize_json_for_determinism(mut value: serde_json::Value) -> serde_json::Value {
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
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let value = canonicalize_json(value);
    std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

fn write_yields_tables(
    yields: &ns_viz::yields::YieldsArtifact,
    out_dir: &std::path::Path,
) -> Result<()> {
    // CSV: channel,row_kind,name,prefit,postfit
    let mut csv = String::new();
    csv.push_str("channel,row_kind,name,prefit,postfit\n");
    for ch in &yields.channels {
        for s in &ch.samples {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                ch.channel_name,
                "sample",
                s.name,
                s.prefit.to_string(),
                s.postfit.to_string()
            ));
        }
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            ch.channel_name,
            "total",
            "TOTAL",
            ch.total_prefit.to_string(),
            ch.total_postfit.to_string()
        ));
        csv.push_str(&format!(
            "{},{},{},{},\n",
            ch.channel_name,
            "data",
            "DATA",
            ch.data.to_string()
        ));
    }
    std::fs::write(out_dir.join("yields.csv"), csv)?;

    // LaTeX: one tabular per channel (booktabs-friendly).
    let mut tex = String::new();
    tex.push_str("% Auto-generated by `nextstat report`.\n");
    tex.push_str("% Requires: \\usepackage{booktabs}\n\n");
    for ch in &yields.channels {
        tex.push_str(&format!("% Channel: {}\n", ch.channel_name));
        tex.push_str("\\begin{tabular}{lrr}\n");
        tex.push_str("\\toprule\n");
        tex.push_str("Sample & Prefit & Postfit \\\\\n");
        tex.push_str("\\midrule\n");
        for s in &ch.samples {
            tex.push_str(&format!(
                "{} & {} & {} \\\\\n",
                s.name,
                s.prefit.to_string(),
                s.postfit.to_string()
            ));
        }
        tex.push_str("\\midrule\n");
        tex.push_str(&format!(
            "Total & {} & {} \\\\\n",
            ch.total_prefit.to_string(),
            ch.total_postfit.to_string()
        ));
        tex.push_str(&format!("Data & {} & \\\\\n", ch.data.to_string()));
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
        )?;
    }
    Ok(())
}

fn z_for_level(level: f64) -> Result<(f64, f64)> {
    if !level.is_finite() || !(0.0 < level && level < 1.0) {
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
        )?;
    }
    Ok(())
}

fn cmd_hypotest(
    input: &PathBuf,
    mu: f64,
    expected_set: bool,
    output: Option<&PathBuf>,
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let ctx = ns_inference::AsymptoticCLsContext::new(&mle, &model)?;
    let r = ctx.hypotest_qtilde(&mle, mu)?;
    tracing::debug!(mu_test = r.mu_test, cls = r.cls, mu_hat = r.mu_hat, "hypotest result");

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
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "hypotest",
            serde_json::json!({ "mu": mu, "expected_set": expected_set, "threads": threads }),
            input,
            &output_json,
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
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads)?;
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

    write_json(output, &output_json)?;
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
        )?;
    }
    Ok(())
}

fn cmd_scan(
    input: &PathBuf,
    start: f64,
    stop: f64,
    points: usize,
    output: Option<&PathBuf>,
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if points < 2 {
        anyhow::bail!("points must be >= 2");
    }
    let model = load_model(input, threads)?;
    let mle = ns_inference::MaximumLikelihoodEstimator::new();

    let step = (stop - start) / (points as f64 - 1.0);
    let mu_values: Vec<f64> = (0..points).map(|i| start + step * i as f64).collect();
    let scan = ns_inference::profile_likelihood::scan(&mle, &model, &mu_values)?;

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
    if let Some(dir) = bundle {
        report::write_bundle(
            dir,
            "scan",
            serde_json::json!({ "start": start, "stop": stop, "points": points, "threads": threads }),
            input,
            &output_json,
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
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if points < 2 {
        anyhow::bail!("points must be >= 2");
    }
    let model = load_model(input, threads)?;
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
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if scan_points < 2 {
        anyhow::bail!("scan_points must be >= 2");
    }
    if !(scan_stop > scan_start) {
        anyhow::bail!("scan_stop must be > scan_start");
    }

    let model = load_model(input, threads)?;
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
        )?;
    }
    Ok(())
}

fn cmd_viz_ranking(
    input: &PathBuf,
    output: Option<&PathBuf>,
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let model = load_model(input, threads)?;
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
    deterministic: bool,
) -> Result<()> {
    ensure_out_dir(out_dir, overwrite)?;

    let (workspace, model) = load_workspace_and_model(input, threads)?;
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
            Some(ns_core::FitResult {
                parameters: params_postfit.clone(),
                uncertainties: uncs_postfit,
                covariance: fit_json.covariance,
                nll: fit_json.nll,
                converged: fit_json.converged,
                n_iter: fit_json.n_iter,
                n_fev: fit_json.n_fev,
                n_gev: fit_json.n_gev,
            })
        };

        (params_postfit, fit_result)
    } else {
        // Default: run the fit here (so `nextstat report` is “one command”).
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

        let fit_result = Some(ns_core::FitResult {
            parameters: result.parameters.clone(),
            uncertainties: result.uncertainties.clone(),
            covariance: result.covariance.clone(),
            nll: result.nll,
            converged: result.converged,
            n_iter: result.n_iter,
            n_fev: result.n_fev,
            n_gev: result.n_gev,
        });

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

    let dist_artifact = ns_viz::distributions::distributions_artifact(
        &model,
        &data_by_channel,
        &bin_edges_by_channel,
        &params_prefit,
        &params_postfit,
        threads,
    )?;
    let mut dist_json = serde_json::to_value(&dist_artifact)?;
    if deterministic {
        dist_json = normalize_json_for_determinism(dist_json);
    }
    write_json_file(&out_dir.join("distributions.json"), &dist_json)?;

    // Yields
    let yields_artifact =
        ns_viz::yields::yields_artifact(&model, &params_prefit, &params_postfit, threads)?;
    let mut yields_json = serde_json::to_value(&yields_artifact)?;
    if deterministic {
        yields_json = normalize_json_for_determinism(yields_json);
    }
    write_json_file(&out_dir.join("yields.json"), &yields_json)?;
    write_yields_tables(&yields_artifact, out_dir)?;

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

        let mut pythonpath = std::ffi::OsString::new();
        pythonpath.push("bindings/ns-py/python");
        if let Some(existing) = std::env::var_os("PYTHONPATH") {
            if !existing.is_empty() {
                pythonpath.push(":");
                pythonpath.push(existing);
            }
        }

        let status = Command::new(&python)
            .env("PYTHONPATH", pythonpath)
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
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if bundle.is_some() {
        anyhow::bail!("--bundle is not supported for `nextstat viz distributions` yet");
    }

    let (workspace, model) = load_workspace_and_model(input, threads)?;
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
    )?;

    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    Ok(())
}

fn cmd_viz_pulls(
    input: &PathBuf,
    fit: &PathBuf,
    output: Option<&PathBuf>,
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if bundle.is_some() {
        anyhow::bail!("--bundle is not supported for `nextstat viz pulls` yet");
    }

    let model = load_model(input, threads)?;

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

    let fit_result = ns_core::FitResult {
        parameters: params_postfit,
        uncertainties: uncs_postfit,
        covariance: fit_json.covariance,
        nll: fit_json.nll,
        converged: fit_json.converged,
        n_iter: fit_json.n_iter,
        n_fev: fit_json.n_fev,
        n_gev: fit_json.n_gev,
    };
    let artifact = ns_viz::pulls::pulls_artifact(&model, &fit_result, threads)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    Ok(())
}

fn cmd_viz_corr(
    input: &PathBuf,
    fit: &PathBuf,
    include_covariance: bool,
    output: Option<&PathBuf>,
    threads: usize,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if bundle.is_some() {
        anyhow::bail!("--bundle is not supported for `nextstat viz corr` yet");
    }

    let model = load_model(input, threads)?;

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

    let fit_result = ns_core::FitResult {
        parameters: params_postfit,
        uncertainties: uncs_postfit,
        covariance: fit_json.covariance,
        nll: fit_json.nll,
        converged: fit_json.converged,
        n_iter: fit_json.n_iter,
        n_fev: fit_json.n_fev,
        n_gev: fit_json.n_gev,
    };

    let artifact = ns_viz::corr::corr_artifact(&model, &fit_result, threads, include_covariance)?;
    let output_json = serde_json::to_value(&artifact)?;
    write_json(output, &output_json)?;
    Ok(())
}
