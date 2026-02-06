//! TREx-like stacked distributions artifacts (numbers-first).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::Result;
use serde::Serialize;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use ns_translate::pyhf::{ExpectedChannelSampleYields, HistFactoryModel};

#[derive(Debug, Clone, Serialize)]
pub struct DistributionsArtifact {
    pub schema_version: String,
    pub meta: DistributionsMeta,
    pub channels: Vec<DistributionsChannelArtifact>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionsMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub parity_mode: ParityMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<DistributionsInputMeta>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParityMode {
    pub threads: usize,
    pub stable_ordering: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionsInputMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_result_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RatioPolicy {
    pub numerator: String,
    pub denominator: String,
    pub zero_policy: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionsChannelArtifact {
    pub channel_name: String,
    pub bin_edges: Vec<f64>,
    pub data_y: Vec<f64>,
    pub data_yerr_lo: Vec<f64>,
    pub data_yerr_hi: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_error_model: Option<String>,
    pub samples: Vec<DistributionsSampleSeries>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack_order: Option<Vec<String>>,
    pub total_prefit_y: Vec<f64>,
    pub total_postfit_y: Vec<f64>,
    pub ratio_policy: RatioPolicy,
    pub ratio_y: Vec<f64>,
    pub ratio_yerr_lo: Vec<f64>,
    pub ratio_yerr_hi: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionsSampleSeries {
    pub name: String,
    pub prefit_y: Vec<f64>,
    pub postfit_y: Vec<f64>,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

fn is_near_integer_nonneg(x: f64) -> Option<u64> {
    if !(x.is_finite() && x >= 0.0) {
        return None;
    }
    let r = x.round();
    if (x - r).abs() <= 1e-9 {
        Some(r as u64)
    } else {
        None
    }
}

fn garwood_68_interval(n: u64) -> (f64, f64) {
    // Central 68.2689% interval -> alpha = 1 - CL
    let alpha = 0.31731_f64;
    // Chi-square quantiles:
    // lo = n - 0.5 * chi2_{alpha/2, 2n}
    // hi = 0.5 * chi2_{1-alpha/2, 2(n+1)} - n
    let lo = if n == 0 {
        0.0
    } else {
        let dist = ChiSquared::new(2.0 * (n as f64)).unwrap();
        let q = dist.inverse_cdf(alpha / 2.0);
        (n as f64) - 0.5 * q
    };
    let dist_hi = ChiSquared::new(2.0 * ((n + 1) as f64)).unwrap();
    let q_hi = dist_hi.inverse_cdf(1.0 - alpha / 2.0);
    let hi = 0.5 * q_hi - (n as f64);
    (lo, hi)
}

fn data_errors(y: &[f64]) -> (Vec<f64>, Vec<f64>, Option<String>) {
    let mut lo = Vec::with_capacity(y.len());
    let mut hi = Vec::with_capacity(y.len());

    let mut all_poisson = true;
    for &v in y {
        if let Some(n) = is_near_integer_nonneg(v) {
            let (dl, dh) = garwood_68_interval(n);
            lo.push(dl);
            hi.push(dh);
        } else {
            all_poisson = false;
            let e = if v.is_finite() && v > 0.0 { v.sqrt() } else { f64::NAN };
            lo.push(e);
            hi.push(e);
        }
    }
    let model = if all_poisson {
        Some("garwood_poisson_68".to_string())
    } else {
        Some("sqrt_y_fallback".to_string())
    };
    (lo, hi, model)
}

fn ratio_from_data_over_mc(
    data: &[f64],
    data_lo: &[f64],
    data_hi: &[f64],
    mc: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let mut y = Vec::with_capacity(n);
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for i in 0..n {
        let denom = mc[i];
        if denom.is_finite() && denom != 0.0 {
            y.push(data[i] / denom);
            lo.push(data_lo[i] / denom);
            hi.push(data_hi[i] / denom);
        } else {
            y.push(f64::NAN);
            lo.push(f64::NAN);
            hi.push(f64::NAN);
        }
    }
    (y, lo, hi)
}

pub fn distributions_artifact(
    model: &HistFactoryModel,
    data_by_channel: &HashMap<String, Vec<f64>>,
    bin_edges_by_channel: &HashMap<String, Vec<f64>>,
    params_prefit: &[f64],
    params_postfit: &[f64],
    threads: usize,
) -> Result<DistributionsArtifact> {
    let pre = model.expected_main_by_channel_sample(params_prefit)?;
    let post = model.expected_main_by_channel_sample(params_postfit)?;

    let mut post_map: HashMap<&str, &ExpectedChannelSampleYields> = HashMap::new();
    for ch in &post {
        post_map.insert(ch.channel_name.as_str(), ch);
    }

    let mut channels_out: Vec<DistributionsChannelArtifact> = Vec::with_capacity(pre.len());
    for ch_pre in &pre {
        let ch_post = post_map.get(ch_pre.channel_name.as_str()).ok_or_else(|| {
            ns_core::Error::Validation(format!(
                "postfit channel missing: {}",
                ch_pre.channel_name
            ))
        })?;
        let data = data_by_channel.get(&ch_pre.channel_name).ok_or_else(|| {
            ns_core::Error::Validation(format!(
                "data missing for channel: {}",
                ch_pre.channel_name
            ))
        })?;
        let edges = bin_edges_by_channel.get(&ch_pre.channel_name).ok_or_else(|| {
            ns_core::Error::Validation(format!(
                "bin edges missing for channel: {}",
                ch_pre.channel_name
            ))
        })?;

        let (data_err_lo, data_err_hi, data_error_model) = data_errors(data);
        let (ratio_y, ratio_yerr_lo, ratio_yerr_hi) =
            ratio_from_data_over_mc(data, &data_err_lo, &data_err_hi, &ch_post.total);

        let mut samples = Vec::with_capacity(ch_pre.samples.len());
        for s_pre in &ch_pre.samples {
            let s_post = ch_post
                .samples
                .iter()
                .find(|s| s.sample_name == s_pre.sample_name)
                .ok_or_else(|| {
                    ns_core::Error::Validation(format!(
                        "postfit sample missing: channel={} sample={}",
                        ch_pre.channel_name,
                        s_pre.sample_name
                    ))
                })?;
            samples.push(DistributionsSampleSeries {
                name: s_pre.sample_name.clone(),
                prefit_y: s_pre.y.clone(),
                postfit_y: s_post.y.clone(),
            });
        }

        channels_out.push(DistributionsChannelArtifact {
            channel_name: ch_pre.channel_name.clone(),
            bin_edges: edges.clone(),
            data_y: data.clone(),
            data_yerr_lo: data_err_lo,
            data_yerr_hi: data_err_hi,
            data_error_model,
            samples,
            stack_order: None,
            total_prefit_y: ch_pre.total.clone(),
            total_postfit_y: ch_post.total.clone(),
            ratio_policy: RatioPolicy {
                numerator: "data".to_string(),
                denominator: "mc_total_postfit".to_string(),
                zero_policy: "nan".to_string(),
            },
            ratio_y,
            ratio_yerr_lo,
            ratio_yerr_hi,
        });
    }

    Ok(DistributionsArtifact {
        schema_version: "trex_report_distributions_v0".to_string(),
        meta: DistributionsMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: ParityMode {
                threads: threads.max(1),
                stable_ordering: true,
            },
            input: None,
        },
        channels: channels_out,
    })
}
