//! NextStat CLI

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nextstat")]
#[command(about = "NextStat - High-performance statistical fitting")]
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

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

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

    let parameter_names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();

    let output_json = serde_json::json!({
        "parameter_names": parameter_names,
        "poi_index": model.poi_index(),
        "bestfit": result.parameters,
        "uncertainties": result.uncertainties,
        "nll": result.nll,
        "twice_nll": 2.0 * result.nll,
        "converged": result.converged,
        "n_evaluations": result.n_evaluations,
        "covariance": result.covariance,
    });

    write_json(output, output_json)
}

fn load_model(input: &PathBuf, threads: usize) -> Result<ns_translate::pyhf::HistFactoryModel> {
    if threads > 0 {
        // Best-effort; if a global pool already exists, keep going.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }

    let json = std::fs::read_to_string(input)?;
    let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(&json)?;
    Ok(ns_translate::pyhf::HistFactoryModel::from_workspace(&workspace)?)
}

fn write_json(output: Option<&PathBuf>, value: serde_json::Value) -> Result<()> {
    if let Some(path) = output {
        std::fs::write(path, serde_json::to_string_pretty(&value)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&value)?);
    }
    Ok(())
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
