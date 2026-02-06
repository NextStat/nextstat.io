//! TREx-like correlation matrix artifact (numbers-first).

use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::{FitResult, Result};
use serde::Serialize;

use ns_translate::pyhf::HistFactoryModel;

#[derive(Debug, Clone, Serialize)]
pub struct CorrArtifact {
    pub schema_version: String,
    pub meta: CorrMeta,
    pub parameter_names: Vec<String>,
    pub corr: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covariance: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub parity_mode: CorrParityMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrParityMode {
    pub threads: usize,
    pub stable_ordering: bool,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

/// Build a correlation-matrix artifact from a `FitResult`.
///
/// Requires `fit.covariance` to be present (correlations are undefined otherwise).
pub fn corr_artifact(
    model: &HistFactoryModel,
    fit: &FitResult,
    threads: usize,
    include_covariance: bool,
) -> Result<CorrArtifact> {
    let n = model.parameters().len();
    if fit.parameters.len() != n {
        return Err(ns_core::Error::Validation(format!(
            "fit/model parameter length mismatch: fit={} model={}",
            fit.parameters.len(),
            n
        )));
    }
    let cov = fit.covariance.as_ref().ok_or_else(|| {
        ns_core::Error::Validation("fit result covariance is required for correlation artifact".to_string())
    })?;
    if cov.len() != n * n {
        return Err(ns_core::Error::Validation(format!(
            "covariance length mismatch: got={} expected={}",
            cov.len(),
            n * n
        )));
    }

    let names: Vec<String> = model.parameters().iter().map(|p| p.name.clone()).collect();

    let mut corr: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut cov_out: Option<Vec<Vec<f64>>> = if include_covariance {
        Some(vec![vec![0.0; n]; n])
    } else {
        None
    };

    for i in 0..n {
        for j in 0..n {
            let cij = cov[i * n + j];
            let si = fit.uncertainties[i];
            let sj = fit.uncertainties[j];
            if si.is_finite() && sj.is_finite() && si > 0.0 && sj > 0.0 && cij.is_finite() {
                corr[i][j] = cij / (si * sj);
            } else {
                corr[i][j] = 0.0;
            }
            if let Some(m) = cov_out.as_mut() {
                m[i][j] = cij;
            }
        }
    }

    Ok(CorrArtifact {
        schema_version: "trex_report_corr_v0".to_string(),
        meta: CorrMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: CorrParityMode {
                threads: threads.max(1),
                stable_ordering: true,
            },
        },
        parameter_names: names,
        corr,
        covariance: cov_out,
    })
}
