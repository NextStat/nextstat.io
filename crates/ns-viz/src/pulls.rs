//! Pulls + constraints artifacts (TREx-style, numbers-first).

use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::Result;
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
    pub postfit_center: f64,
    pub postfit_sigma: f64,
    pub pull: f64,
    pub constraint: f64,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

fn base_group(name: &str) -> Option<String> {
    name.split_once('[').map(|(b, _)| b.to_string())
}

pub fn pulls_artifact(
    model: &HistFactoryModel,
    postfit_params: &[f64],
    postfit_uncertainties: &[f64],
    threads: usize,
) -> Result<PullsArtifact> {
    if postfit_params.len() != model.parameters().len() {
        return Err(ns_core::Error::Validation(format!(
            "postfit_params length mismatch: expected {}, got {}",
            model.parameters().len(),
            postfit_params.len()
        )));
    }
    if postfit_uncertainties.len() != model.parameters().len() {
        return Err(ns_core::Error::Validation(format!(
            "postfit_uncertainties length mismatch: expected {}, got {}",
            model.parameters().len(),
            postfit_uncertainties.len()
        )));
    }

    let poi_index = model.poi_index();

    let mut entries = Vec::new();
    for (i, p) in model.parameters().iter().enumerate() {
        if !p.constrained {
            continue;
        }
        let pre_center = p.constraint_center.unwrap_or(p.init);
        let pre_sigma = p.constraint_width.unwrap_or(1.0);
        let post_center = postfit_params[i];
        let post_sigma = postfit_uncertainties[i];

        let pull = if pre_sigma.is_finite() && pre_sigma > 0.0 {
            (post_center - pre_center) / pre_sigma
        } else {
            f64::NAN
        };
        let constraint = if pre_sigma.is_finite() && pre_sigma > 0.0 && post_sigma.is_finite() {
            post_sigma / pre_sigma
        } else {
            f64::NAN
        };

        let kind = if poi_index == Some(i) { "poi" } else { "nuisance" };
        entries.push(PullEntry {
            name: p.name.clone(),
            kind: kind.to_string(),
            group: base_group(&p.name),
            prefit_center: Some(pre_center),
            prefit_sigma: Some(pre_sigma),
            postfit_center: post_center,
            postfit_sigma: post_sigma,
            pull,
            constraint,
        });
    }

    Ok(PullsArtifact {
        schema_version: "trex_report_pulls_v0".to_string(),
        meta: PullsMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: PullsParityMode {
                threads: threads.max(1),
                stable_ordering: true,
            },
        },
        ordering_policy: Some("input".to_string()),
        entries,
    })
}

