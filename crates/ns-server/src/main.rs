//! NextStat Server — self-hosted GPU inference API.
//!
//! Provides a JSON REST API for shared GPU statistical fits so that
//! lab clusters can share one GPU across multiple users/clients
//! without per-user CUDA/Metal setup.
//!
//! # Endpoints
//!
//! - `POST /v1/fit`     — workspace JSON → FitResult JSON
//! - `POST /v1/ranking` — workspace JSON → ranked systematics JSON
//! - `GET  /v1/health`  — server status, version, GPU info

mod pool;
mod routes;
mod state;
mod tools;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::Router;
use clap::Parser;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use state::AppState;

/// NextStat inference server — shared GPU statistical fits over HTTP.
#[derive(Parser, Debug)]
#[command(name = "nextstat-server", version = ns_core::VERSION, about)]
struct Cli {
    /// Port to listen on.
    #[arg(short, long, default_value = "3742")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// GPU device to use for inference. Values: "cuda", "metal", or omit for CPU-only.
    #[arg(long)]
    gpu: Option<String>,

    /// Maximum number of CPU threads for non-GPU workloads (0 = auto).
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Maximum request body size in MiB (applies to all endpoints).
    ///
    /// Protects the server from accidental or malicious oversized JSON payloads.
    #[arg(long, default_value = "64")]
    max_body_mb: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug")),
        )
        .init();

    let cli = Cli::parse();

    // Validate GPU selection at startup
    validate_gpu(cli.gpu.as_deref())?;

    // Configure Rayon thread pool
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    let state = Arc::new(AppState::new(cli.gpu.clone()));

    let max_body_bytes = mb_to_bytes(cli.max_body_mb);

    let app = Router::new()
        .merge(routes::router())
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    tracing::info!(
        %addr,
        gpu = cli.gpu.as_deref().unwrap_or("cpu"),
        version = ns_core::VERSION,
        "nextstat-server starting"
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn mb_to_bytes(mb: usize) -> usize {
    // Clamp overflow to usize::MAX to avoid panics in debug builds.
    mb.saturating_mul(1024).saturating_mul(1024)
}

fn validate_gpu(gpu: Option<&str>) -> anyhow::Result<()> {
    match gpu {
        Some("cuda") => {
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("--gpu cuda requires building with --features cuda");
            #[cfg(feature = "cuda")]
            tracing::info!("GPU mode: CUDA (f64)");
        }
        Some("metal") => {
            #[cfg(not(feature = "metal"))]
            anyhow::bail!("--gpu metal requires building with --features metal");
            #[cfg(feature = "metal")]
            tracing::info!("GPU mode: Metal (f32)");
        }
        Some(other) => anyhow::bail!("unknown --gpu device: {other}. Use 'cuda' or 'metal'"),
        None => {
            tracing::info!("GPU mode: disabled (CPU only)");
        }
    }
    Ok(())
}
