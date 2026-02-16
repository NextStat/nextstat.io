//! Pie chart artifact — sample composition per channel.
//!
//! For each channel, computes the fraction of total expected yield contributed
//! by each sample (process).  Comparable to TRExFitter pie chart plots.

use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PieArtifact {
    pub schema_version: String,
    pub channels: Vec<PieChannelArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PieChannelArtifact {
    pub channel_name: String,
    pub total_yield: f64,
    pub slices: Vec<PieSlice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PieSlice {
    pub sample_name: String,
    pub yield_sum: f64,
    pub fraction: f64,
}

/// Build a pie chart artifact at the given parameter point.
///
/// Typically called with prefit (init) or postfit parameters.
pub fn pie_artifact(model: &HistFactoryModel, params: &[f64]) -> Result<PieArtifact> {
    let yields_by_channel = model.expected_main_by_channel_sample(params)?;

    let mut channels = Vec::with_capacity(yields_by_channel.len());
    for ch_yields in &yields_by_channel {
        let mut slices = Vec::new();
        let mut total: f64 = 0.0;

        for sample_yields in &ch_yields.samples {
            let sample_sum: f64 = sample_yields.y.iter().copied().sum();
            total += sample_sum;
            slices.push(PieSlice {
                sample_name: sample_yields.sample_name.clone(),
                yield_sum: sample_sum,
                fraction: 0.0,
            });
        }

        if total > 0.0 {
            for s in &mut slices {
                s.fraction = s.yield_sum / total;
            }
        }

        slices.sort_by(|a, b| {
            b.fraction.partial_cmp(&a.fraction).unwrap_or(std::cmp::Ordering::Equal)
        });

        channels.push(PieChannelArtifact {
            channel_name: ch_yields.channel_name.clone(),
            total_yield: total,
            slices,
        });
    }

    Ok(PieArtifact { schema_version: "nextstat_pie_v0".to_string(), channels })
}
