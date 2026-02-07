//! TREx-like yields tables artifact (numbers-first).

use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use ns_core::Result;
use serde::Serialize;

use ns_translate::pyhf::HistFactoryModel;

#[derive(Debug, Clone, Serialize)]
pub struct YieldsArtifact {
    pub schema_version: String,
    pub meta: YieldsMeta,
    pub channels: Vec<YieldsChannel>,
}

#[derive(Debug, Clone, Serialize)]
pub struct YieldsMeta {
    pub tool: String,
    pub tool_version: String,
    pub created_unix_ms: u128,
    pub parity_mode: YieldsParityMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct YieldsParityMode {
    pub threads: usize,
    pub stable_ordering: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct YieldsChannel {
    pub channel_name: String,
    pub data: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_is_blinded: Option<bool>,
    pub samples: Vec<YieldsSample>,
    pub total_prefit: f64,
    pub total_postfit: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct YieldsSample {
    pub name: String,
    pub prefit: f64,
    pub postfit: f64,
}

fn now_unix_ms() -> Result<u128> {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ns_core::Error::Computation(format!("system time error: {}", e)))?;
    Ok(d.as_millis())
}

pub fn yields_artifact(
    model: &HistFactoryModel,
    params_prefit: &[f64],
    params_postfit: &[f64],
    threads: usize,
    blinded_channels: Option<&HashSet<String>>,
) -> Result<YieldsArtifact> {
    let pre = model.expected_main_by_channel_sample(params_prefit)?;
    let post = model.expected_main_by_channel_sample(params_postfit)?;
    let obs = model.observed_main_by_channel();

    if pre.len() != post.len() || pre.len() != obs.len() {
        return Err(ns_core::Error::Validation(
            "channel count mismatch between prefit/postfit/observed".to_string(),
        ));
    }

    let mut out_channels: Vec<YieldsChannel> = Vec::with_capacity(pre.len());

    for ((ch_pre, ch_post), ch_obs) in pre.iter().zip(post.iter()).zip(obs.iter()) {
        if ch_pre.channel_name != ch_post.channel_name || ch_pre.channel_name != ch_obs.channel_name
        {
            return Err(ns_core::Error::Validation(
                "channel ordering mismatch (expected stable ordering)".to_string(),
            ));
        }

        let blinded =
            blinded_channels.as_ref().map(|xs| xs.contains(&ch_pre.channel_name)).unwrap_or(false);
        let data_sum: f64 = if blinded { 0.0 } else { ch_obs.y.iter().copied().sum() };
        let total_prefit: f64 = ch_pre.total.iter().copied().sum();
        let total_postfit: f64 = ch_post.total.iter().copied().sum();

        let mut samples: Vec<YieldsSample> = Vec::with_capacity(ch_pre.samples.len());
        for s_pre in &ch_pre.samples {
            let s_post =
                ch_post.samples.iter().find(|s| s.sample_name == s_pre.sample_name).ok_or_else(
                    || {
                        ns_core::Error::Validation(format!(
                            "postfit sample missing: channel={} sample={}",
                            ch_pre.channel_name, s_pre.sample_name
                        ))
                    },
                )?;
            let pre_sum: f64 = s_pre.y.iter().copied().sum();
            let post_sum: f64 = s_post.y.iter().copied().sum();
            samples.push(YieldsSample {
                name: s_pre.sample_name.clone(),
                prefit: pre_sum,
                postfit: post_sum,
            });
        }

        out_channels.push(YieldsChannel {
            channel_name: ch_pre.channel_name.clone(),
            data: data_sum,
            data_is_blinded: if blinded { Some(true) } else { None },
            samples,
            total_prefit,
            total_postfit,
        });
    }

    Ok(YieldsArtifact {
        schema_version: "trex_report_yields_v0".to_string(),
        meta: YieldsMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: YieldsParityMode { threads: threads.max(1), stable_ordering: true },
        },
        channels: out_channels,
    })
}
