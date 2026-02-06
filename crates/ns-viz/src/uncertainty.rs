//! TREx-like uncertainty breakdown artifact (numbers-first).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::Result;
use serde::Serialize;

use crate::ranking::RankingArtifact;

#[derive(Debug, Clone, Serialize)]
pub struct UncertaintyBreakdownArtifact {
    pub schema_version: String,
    pub meta: UncertaintyMeta,
    pub grouping_policy: String,
    pub groups: Vec<UncertaintyGroup>,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UncertaintyMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub parity_mode: UncertaintyParityMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct UncertaintyParityMode {
    pub threads: usize,
    pub stable_ordering: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UncertaintyGroup {
    pub name: String,
    pub impact: f64,
    pub n_parameters: usize,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

fn group_name_prefix_1(param_name: &str) -> String {
    let name = param_name;
    if name.starts_with("gamma_") || name.starts_with("gammaStat") || name.starts_with("gammaStat_") {
        return "stat".to_string();
    }
    if name.contains("lumi") || name.contains("Lumi") {
        return "lumi".to_string();
    }
    name.split('_').next().unwrap_or(name).to_string()
}

/// Build an uncertainty breakdown artifact from a ranking artifact.
///
/// Definition:
/// - Per-parameter impact is `max(|delta_mu_up|, |delta_mu_down|)`.
/// - Per-group impact is quadrature: `sqrt(sum_i impact_i^2)`.
pub fn uncertainty_breakdown_from_ranking(
    ranking: &RankingArtifact,
    grouping_policy: &str,
    threads: usize,
) -> Result<UncertaintyBreakdownArtifact> {
    let n = ranking.names.len();
    if ranking.delta_mu_up.len() != n || ranking.delta_mu_down.len() != n {
        return Err(ns_core::Error::Validation(
            "ranking arrays length mismatch".to_string(),
        ));
    }

    let mut sumsq_by_group: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut total_sumsq = 0.0_f64;

    for i in 0..n {
        let up = ranking.delta_mu_up[i];
        let down = ranking.delta_mu_down[i];
        let impact = up.abs().max(down.abs());
        total_sumsq += impact * impact;

        let g = match grouping_policy {
            "prefix_1" => group_name_prefix_1(&ranking.names[i]),
            _ => group_name_prefix_1(&ranking.names[i]),
        };
        let e = sumsq_by_group.entry(g).or_insert((0.0, 0));
        e.0 += impact * impact;
        e.1 += 1;
    }

    let mut groups: Vec<UncertaintyGroup> = sumsq_by_group
        .into_iter()
        .map(|(name, (sumsq, n_parameters))| UncertaintyGroup {
            name,
            impact: sumsq.sqrt(),
            n_parameters,
        })
        .collect();

    groups.sort_by(|a, b| {
        b.impact
            .partial_cmp(&a.impact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(UncertaintyBreakdownArtifact {
        schema_version: "trex_report_uncertainty_v0".to_string(),
        meta: UncertaintyMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: UncertaintyParityMode {
                threads: threads.max(1),
                stable_ordering: true,
            },
        },
        grouping_policy: grouping_policy.to_string(),
        groups,
        total: total_sumsq.sqrt(),
    })
}

