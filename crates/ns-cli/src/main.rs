//! NextStat CLI

use anyhow::Result;
use clap::{Parser, Subcommand};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nextstat")]
#[command(about = "NextStat - High-performance statistical fitting")]
#[command(version)]
struct Cli {
    /// Log verbosity level (trace, debug, info, warn, error)
    #[arg(long, global = true, default_value = "warn")]
    log_level: tracing::Level,

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

    /// Visualization artifacts (plot-friendly JSON)
    Viz {
        #[command(subcommand)]
        command: VizCommands,
    },

    /// Time series and state space models (Phase 8)
    Timeseries {
        #[command(subcommand)]
        command: TimeseriesCommands,
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

    /// Forecast K steps ahead after filtering the provided `ys`
    KalmanForecast {
        /// Input JSON file (see docs/tutorials/phase-8-timeseries.md)
        #[arg(short, long)]
        input: PathBuf,

        /// Number of forecast steps (>0)
        #[arg(long, default_value = "1")]
        steps: usize,

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
        Commands::Fit { input, output, threads } => cmd_fit(&input, output.as_ref(), threads),
        Commands::Hypotest { input, mu, expected_set, output, threads } => {
            cmd_hypotest(&input, mu, expected_set, output.as_ref(), threads)
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
        ),
        Commands::Scan { input, start, stop, points, output, threads } => {
            cmd_scan(&input, start, stop, points, output.as_ref(), threads)
        }
        Commands::Viz { command } => match command {
            VizCommands::Profile { input, start, stop, points, output, threads } => {
                cmd_viz_profile(&input, start, stop, points, output.as_ref(), threads)
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
            ),
        },
        Commands::Timeseries { command } => match command {
            TimeseriesCommands::KalmanFilter { input, output } => cmd_ts_kalman_filter(&input, output.as_ref()),
            TimeseriesCommands::KalmanSmooth { input, output } => cmd_ts_kalman_smooth(&input, output.as_ref()),
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
            ),
            TimeseriesCommands::KalmanForecast { input, steps, output } => {
                cmd_ts_kalman_forecast(&input, steps, output.as_ref())
            }
            TimeseriesCommands::KalmanSimulate { input, t_max, seed, output } => {
                cmd_ts_kalman_simulate(&input, t_max, seed, output.as_ref())
            }
        },
        Commands::Version => {
            println!("nextstat {}", ns_core::VERSION);
            Ok(())
        }
    }
}

fn cmd_fit(input: &PathBuf, output: Option<&PathBuf>, threads: usize) -> Result<()> {
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

    write_json(output, output_json)
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

fn write_json(output: Option<&PathBuf>, value: serde_json::Value) -> Result<()> {
    if let Some(path) = output {
        std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&value)?);
    }
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
    local_level_seasonal: Option<KalmanLocalLevelSeasonalJson>,
    #[serde(default)]
    local_linear_trend_seasonal: Option<KalmanLocalLinearTrendSeasonalJson>,
    ys: Vec<Vec<Option<f64>>>,
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
    model_count += input.local_level_seasonal.is_some() as usize;
    model_count += input.local_linear_trend_seasonal.is_some() as usize;
    if model_count != 1 {
        anyhow::bail!(
            "expected exactly one of: model, local_level, local_linear_trend, ar1, local_level_seasonal, local_linear_trend_seasonal"
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

fn cmd_ts_kalman_filter(input: &PathBuf, output: Option<&PathBuf>) -> Result<()> {
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

    write_json(output, output_json)
}

fn cmd_ts_kalman_smooth(input: &PathBuf, output: Option<&PathBuf>) -> Result<()> {
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

    write_json(output, output_json)
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

    write_json(output, output_json)
}

fn cmd_ts_kalman_forecast(input: &PathBuf, steps: usize, output: Option<&PathBuf>) -> Result<()> {
    let (model, ys) = load_kalman_input(input)?;
    let fr = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
        .map_err(|e| anyhow::anyhow!("kalman_filter failed: {e}"))?;
    let fc = ns_inference::timeseries::forecast::kalman_forecast(&model, &fr, steps)
        .map_err(|e| anyhow::anyhow!("kalman_forecast failed: {e}"))?;

    let output_json = serde_json::json!({
        "state_means": fc.state_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "state_covs": fc.state_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
        "obs_means": fc.obs_means.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "obs_covs": fc.obs_covs.iter().map(dmatrix_to_nested).collect::<Vec<_>>(),
    });

    write_json(output, output_json)
}

fn cmd_ts_kalman_simulate(input: &PathBuf, t_max: usize, seed: u64, output: Option<&PathBuf>) -> Result<()> {
    let (model, _ys) = load_kalman_input(input)?;
    let sim = ns_inference::timeseries::simulate::kalman_simulate(&model, t_max, seed)
        .map_err(|e| anyhow::anyhow!("kalman_simulate failed: {e}"))?;

    let output_json = serde_json::json!({
        "xs": sim.xs.iter().map(dvector_to_vec).collect::<Vec<_>>(),
        "ys": sim.ys.iter().map(dvector_to_vec).collect::<Vec<_>>(),
    });
    write_json(output, output_json)
}

fn cmd_hypotest(
    input: &PathBuf,
    mu: f64,
    expected_set: bool,
    output: Option<&PathBuf>,
    threads: usize,
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

    write_json(output, output_json)
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

    write_json(output, output_json)
}

fn cmd_scan(
    input: &PathBuf,
    start: f64,
    stop: f64,
    points: usize,
    output: Option<&PathBuf>,
    threads: usize,
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

    write_json(output, output_json)
}

fn cmd_viz_profile(
    input: &PathBuf,
    start: f64,
    stop: f64,
    points: usize,
    output: Option<&PathBuf>,
    threads: usize,
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

    write_json(output, serde_json::to_value(artifact)?)
}

fn cmd_viz_cls(
    input: &PathBuf,
    alpha: f64,
    scan_start: f64,
    scan_stop: f64,
    scan_points: usize,
    output: Option<&PathBuf>,
    threads: usize,
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
    write_json(output, serde_json::to_value(artifact)?)
}
