pub mod canvas;
pub mod color;
pub mod config;
pub mod font;
pub mod header;
pub mod layout;
pub mod output;
pub mod plots;
pub mod primitives;
pub mod text;
pub mod theme;

use config::VizConfig;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("unknown artifact kind: {0}")]
    UnknownKind(String),
    #[error("deserialization error: {0}")]
    Deserialize(#[from] serde_json::Error),
    #[error("config error: {0}")]
    Config(String),
    #[error("font error: {0}")]
    Font(String),
    #[error("layout error: {0}")]
    Layout(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "png")]
    #[error("PNG encoding error: {0}")]
    Png(String),
    #[cfg(feature = "pdf")]
    #[error("PDF conversion error: {0}")]
    Pdf(String),
}

pub type Result<T> = std::result::Result<T, RenderError>;

/// Render an artifact JSON to SVG string.
pub fn render_svg(artifact_json: &str, kind: &str, config: &VizConfig) -> Result<String> {
    let svg = match kind {
        "pulls" => {
            let art: ns_viz::pulls::PullsArtifact = serde_json::from_str(artifact_json)?;
            plots::pulls::render(&art, config)?
        }
        "ranking" => {
            let art: ns_viz::ranking::RankingArtifact = serde_json::from_str(artifact_json)?;
            plots::ranking::render(&art, config)?
        }
        "corr" => {
            let art: ns_viz::corr::CorrArtifact = serde_json::from_str(artifact_json)?;
            plots::corr::render(&art, config)?
        }
        "profile" => {
            let art: ns_viz::profile::ProfileCurveArtifact = serde_json::from_str(artifact_json)?;
            plots::profile::render(&art, config)?
        }
        "cls" | "cls_curve" => {
            let art: ns_viz::cls::ClsCurveArtifact = serde_json::from_str(artifact_json)?;
            plots::cls_curve::render(&art, config)?
        }
        "distributions" => {
            let art: ns_viz::distributions::DistributionsArtifact =
                serde_json::from_str(artifact_json)?;
            plots::distributions::render(&art, config)?
        }
        "gammas" => {
            let art: ns_viz::gammas::GammasArtifact = serde_json::from_str(artifact_json)?;
            plots::gammas::render(&art, config)?
        }
        "separation" => {
            let art: ns_viz::separation::SeparationArtifact = serde_json::from_str(artifact_json)?;
            plots::separation::render(&art, config)?
        }
        "summary" => {
            let art: ns_viz::summary::SummaryArtifact = serde_json::from_str(artifact_json)?;
            plots::summary::render(&art, config)?
        }
        "uncertainty" => {
            let art: ns_viz::uncertainty::UncertaintyBreakdownArtifact =
                serde_json::from_str(artifact_json)?;
            plots::uncertainty::render(&art, config)?
        }
        "significance" => {
            let art: ns_viz::significance::SignificanceScanArtifact =
                serde_json::from_str(artifact_json)?;
            plots::significance::render(&art, config)?
        }
        "contour" => {
            let art: ns_viz::contour::ContourArtifact = serde_json::from_str(artifact_json)?;
            plots::contour::render(&art, config)?
        }
        "pie" => {
            let art: ns_viz::pie::PieArtifact = serde_json::from_str(artifact_json)?;
            plots::pie::render(&art, config)?
        }
        "yields" => {
            let art: ns_viz::yields::YieldsArtifact = serde_json::from_str(artifact_json)?;
            plots::yields::render(&art, config)?
        }
        "unfolding" | "unfolded_spectrum" => {
            let art: ns_viz::unfolding::UnfoldedSpectrumArtifact =
                serde_json::from_str(artifact_json)?;
            plots::unfolding::render(&art, config)?
        }
        "response_matrix" => {
            let art: ns_viz::unfolding::ResponseMatrixArtifact =
                serde_json::from_str(artifact_json)?;
            plots::unfolding::render_matrix(&art, config)?
        }
        "morphing" => {
            let art: ns_viz::morphing::MorphingArtifact = serde_json::from_str(artifact_json)?;
            plots::morphing::render(&art, config)?
        }
        "injection" => {
            let art: ns_viz::injection::InjectionArtifact = serde_json::from_str(artifact_json)?;
            plots::injection::render(&art, config)?
        }
        other => return Err(RenderError::UnknownKind(other.to_string())),
    };
    Ok(svg)
}

/// Render an artifact JSON to bytes in the specified format.
pub fn render_to_bytes(
    artifact_json: &str,
    kind: &str,
    format: &str,
    config: &VizConfig,
) -> Result<Vec<u8>> {
    let svg = render_svg(artifact_json, kind, config)?;
    match format {
        "svg" => Ok(svg.into_bytes()),
        #[cfg(feature = "png")]
        "png" => output::png::svg_to_png(&svg, config.output.dpi),
        #[cfg(feature = "pdf")]
        "pdf" => output::pdf::svg_to_pdf(&svg),
        other => Err(RenderError::UnknownKind(format!("format: {other}"))),
    }
}

/// Render an artifact JSON to a file (format inferred from extension).
pub fn render_to_file(
    artifact_json: &str,
    kind: &str,
    path: &std::path::Path,
    config: &VizConfig,
) -> Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("svg");
    let bytes = render_to_bytes(artifact_json, kind, ext, config)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
