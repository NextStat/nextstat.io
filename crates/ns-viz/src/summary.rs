//! Summary plot artifact — multi-fit μ values on one canvas.
//!
//! Used in combination papers to compare POI (signal strength) estimates from
//! individual channels/analyses and the combined result.  Each entry is one fit
//! result with a central value and symmetric uncertainty.

use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SummaryArtifact {
    pub schema_version: String,
    pub meta: SummaryMeta,
    pub entries: Vec<SummaryEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub poi_name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryEntry {
    pub label: String,
    pub mu_hat: f64,
    pub sigma: f64,
    pub nll: f64,
    pub converged: bool,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

/// Build a summary artifact from multiple fit result JSONs.
///
/// Each entry is `(label, mu_hat, sigma, nll, converged)`.
pub fn summary_artifact(poi_name: &str, entries: Vec<SummaryEntry>) -> Result<SummaryArtifact> {
    Ok(SummaryArtifact {
        schema_version: "nextstat_summary_v0".to_string(),
        meta: SummaryMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            poi_name: poi_name.to_string(),
        },
        entries,
    })
}
