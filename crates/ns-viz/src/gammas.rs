//! Gammas (staterror / Barlow-Beeston) artifact for TREx-style gamma plots.
//!
//! Shows postfit values of γ parameters (staterror bins) relative to their
//! nominal value of 1.0, with prefit and postfit uncertainties.

use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::{FitResult, Result};
use serde::Serialize;

use ns_translate::pyhf::HistFactoryModel;

#[derive(Debug, Clone, Serialize)]
pub struct GammasArtifact {
    pub schema_version: String,
    pub meta: GammasMeta,
    pub entries: Vec<GammaEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GammasMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub n_gamma_params: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct GammaEntry {
    pub name: String,
    pub channel: Option<String>,
    pub bin_index: Option<usize>,
    pub prefit_value: f64,
    pub prefit_sigma: f64,
    pub postfit_value: f64,
    pub postfit_sigma: f64,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

fn is_gamma_param(name: &str) -> bool {
    name.starts_with("gamma_") || name.starts_with("gammaStat") || name.starts_with("staterror_")
}

fn parse_gamma_channel_bin(name: &str) -> (Option<String>, Option<usize>) {
    // Common naming patterns:
    //   gamma_{channel}_bin_{N}
    //   gammaStat_{channel}_bin_{N}
    //   staterror_{channel}[{N}]
    //   gamma_{channel}_{N}
    let stripped = name
        .strip_prefix("gammaStat_")
        .or_else(|| name.strip_prefix("gamma_"))
        .or_else(|| name.strip_prefix("staterror_"));

    let Some(rest) = stripped else {
        return (None, None);
    };

    // Try pattern: ..._bin_N
    if let Some(pos) = rest.rfind("_bin_") {
        let channel = &rest[..pos];
        let bin_str = &rest[pos + 5..];
        let bin = bin_str.parse::<usize>().ok();
        return (Some(channel.to_string()), bin);
    }

    // Try pattern: channel[N]
    if let Some(bracket_pos) = rest.rfind('[')
        && rest.ends_with(']')
    {
        let channel = &rest[..bracket_pos];
        let bin_str = &rest[bracket_pos + 1..rest.len() - 1];
        let bin = bin_str.parse::<usize>().ok();
        return (Some(channel.to_string()), bin);
    }

    // Try pattern: channel_N (last segment is numeric)
    if let Some(pos) = rest.rfind('_') {
        let maybe_bin = &rest[pos + 1..];
        if let Ok(bin) = maybe_bin.parse::<usize>() {
            let channel = &rest[..pos];
            return (Some(channel.to_string()), Some(bin));
        }
    }

    (Some(rest.to_string()), None)
}

/// Build a gammas artifact from a fit result and model.
///
/// Includes only gamma/staterror parameters (identified by name prefix).
/// Prefit value is 1.0 (nominal), prefit sigma from Poisson constraint sigmas.
pub fn gammas_artifact(model: &HistFactoryModel, fit: &FitResult) -> Result<GammasArtifact> {
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

    let poisson_sigmas = model.poisson_constraint_sigmas();

    let mut entries: Vec<GammaEntry> = Vec::new();
    for (i, p) in model.parameters().iter().enumerate() {
        if !is_gamma_param(&p.name) {
            continue;
        }

        let prefit_value = p.init;
        let prefit_sigma = poisson_sigmas.get(&i).copied().or(p.constraint_width).unwrap_or(0.0);
        let postfit_value = fit.parameters[i];
        let postfit_sigma = fit.uncertainties[i];

        let (channel, bin_index) = parse_gamma_channel_bin(&p.name);

        entries.push(GammaEntry {
            name: p.name.clone(),
            channel,
            bin_index,
            prefit_value,
            prefit_sigma,
            postfit_value,
            postfit_sigma,
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));

    let n_gamma_params = entries.len();

    Ok(GammasArtifact {
        schema_version: "nextstat_gammas_v0".to_string(),
        meta: GammasMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            n_gamma_params,
        },
        entries,
    })
}
