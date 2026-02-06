//! TREx-like pulls/constraints artifact (numbers-first).

use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::{FitResult, Result};
use serde::Serialize;

use ns_translate::pyhf::HistFactoryModel;

#[derive(Debug, Clone, Serialize)]
pub struct PullsArtifact {
    pub schema_version: String,
    pub meta: PullsMeta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordering_policy: Option<String>,
    pub entries: Vec<PullEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PullsMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub parity_mode: PullsParityMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct PullsParityMode {
    pub threads: usize,
    pub stable_ordering: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PullEntry {
    pub name: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefit_center: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefit_sigma: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postfit_center: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postfit_sigma: Option<f64>,
    pub pull: f64,
    pub constraint: f64,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

/// Build a TREx-like pulls/constraints artifact from a `FitResult` and the model parameter metadata.
///
/// Policy:
/// - Includes POI (if any) and all **constrained** nuisance parameters.
/// - Ordering is lexicographic by parameter name for reproducibility.
pub fn pulls_artifact(
    model: &HistFactoryModel,
    fit: &FitResult,
    threads: usize,
) -> Result<PullsArtifact> {
    if fit.parameters.len() != model.parameters().len() {
        return Err(ns_core::Error::Validation(format!(
            "fit/model parameter length mismatch: fit={} model={}",
            fit.parameters.len(),
            model.parameters().len()
        )));
    }
    if fit.uncertainties.len() != model.parameters().len() {
        return Err(ns_core::Error::Validation(format!(
            "fit uncertainties length mismatch: got={} expected={}",
            fit.uncertainties.len(),
            model.parameters().len()
        )));
    }

    let poi_idx = model.poi_index();

    let mut entries: Vec<PullEntry> = Vec::new();
    for (i, p) in model.parameters().iter().enumerate() {
        let is_poi = poi_idx == Some(i);
        let is_constrained =
            p.constrained && p.constraint_center.is_some() && p.constraint_width.is_some();
        if !(is_poi || is_constrained) {
            continue;
        }

        let post_center = fit.parameters[i];
        let post_sigma = fit.uncertainties[i];

        if is_constrained {
            let pre_center = p.constraint_center.unwrap();
            let pre_sigma = p.constraint_width.unwrap();
            let pull = (post_center - pre_center) / pre_sigma;
            let constraint = post_sigma / pre_sigma;
            entries.push(PullEntry {
                name: p.name.clone(),
                kind: if is_poi { "poi".to_string() } else { "nuisance".to_string() },
                group: None,
                prefit_center: Some(pre_center),
                prefit_sigma: Some(pre_sigma),
                postfit_center: Some(post_center),
                postfit_sigma: Some(post_sigma),
                pull,
                constraint,
            });
        } else {
            // POI: not constrained.
            entries.push(PullEntry {
                name: p.name.clone(),
                kind: "poi".to_string(),
                group: Some("poi".to_string()),
                prefit_center: None,
                prefit_sigma: None,
                postfit_center: Some(post_center),
                postfit_sigma: Some(post_sigma),
                pull: 0.0,
                constraint: 1.0,
            });
        }
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(PullsArtifact {
        schema_version: "trex_report_pulls_v0".to_string(),
        meta: PullsMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: PullsParityMode { threads: threads.max(1), stable_ordering: true },
        },
        ordering_policy: Some("name_lex".to_string()),
        entries,
    })
}
