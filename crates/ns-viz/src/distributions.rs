//! TREx-like stacked distributions artifacts (numbers-first).

use std::collections::HashMap;
use std::collections::HashSet;
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
pub struct BandEnvelope {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_is_blinded: Option<bool>,
    pub samples: Vec<DistributionsSampleSeries>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack_order: Option<Vec<String>>,
    pub total_prefit_y: Vec<f64>,
    pub total_postfit_y: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mc_band_prefit: Option<BandEnvelope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mc_band_postfit: Option<BandEnvelope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mc_band_postfit_stat: Option<BandEnvelope>,
    pub ratio_policy: RatioPolicy,
    pub ratio_y: Vec<f64>,
    pub ratio_yerr_lo: Vec<f64>,
    pub ratio_yerr_hi: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio_band: Option<BandEnvelope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio_band_stat: Option<BandEnvelope>,
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
    if (x - r).abs() <= 1e-9 { Some(r as u64) } else { None }
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

fn prefit_sigmas(model: &HistFactoryModel) -> Vec<f64> {
    model
        .parameters()
        .iter()
        .map(|p| p.constraint_width.unwrap_or(0.0))
        .map(|s| if s.is_finite() && s > 0.0 { s } else { 0.0 })
        .collect()
}

fn is_stat_like_parameter(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.starts_with("gamma_")
        || n.starts_with("gamma_stat_")
        || n.contains("staterror")
        || n.contains("mcstat")
}

fn lookup_total_by_channel<'a>(
    shifted: &'a [ExpectedChannelSampleYields],
    idx: usize,
    name: &str,
) -> Option<&'a [f64]> {
    if let Some(ch) = shifted.get(idx)
        && ch.channel_name == name
    {
        return Some(ch.total.as_slice());
    }
    shifted.iter().find(|ch| ch.channel_name == name).map(|ch| ch.total.as_slice())
}

fn channel_offsets(channels: &[ExpectedChannelSampleYields]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(channels.len());
    let mut total = 0usize;
    for ch in channels {
        offsets.push(total);
        total += ch.total.len();
    }
    (offsets, total)
}

fn compute_total_gradients(
    model: &HistFactoryModel,
    center_params: &[f64],
    center: &[ExpectedChannelSampleYields],
    steps: &[f64],
) -> Vec<Vec<f64>> {
    let n_params = center_params.len();
    let n_steps = steps.len().min(n_params);
    let (offsets, n_bins_total) = channel_offsets(center);
    let mut grads: Vec<Vec<f64>> = vec![vec![0.0; n_params]; n_bins_total];

    for pidx in 0..n_steps {
        let step = steps[pidx];
        if !(step.is_finite() && step > 0.0) {
            continue;
        }

        let mut p_up = center_params.to_vec();
        p_up[pidx] += step;
        let mut p_dn = center_params.to_vec();
        p_dn[pidx] -= step;

        let up = model.expected_main_by_channel_sample(&p_up).ok();
        let dn = model.expected_main_by_channel_sample(&p_dn).ok();

        if up.is_none() && dn.is_none() {
            continue;
        }

        for (cidx, ch0) in center.iter().enumerate() {
            let off = offsets[cidx];
            let base = ch0.total.as_slice();
            let up_tot = up
                .as_ref()
                .and_then(|ys| lookup_total_by_channel(ys.as_slice(), cidx, &ch0.channel_name));
            let dn_tot = dn
                .as_ref()
                .and_then(|ys| lookup_total_by_channel(ys.as_slice(), cidx, &ch0.channel_name));

            for b in 0..base.len() {
                let g = match (up_tot, dn_tot) {
                    (Some(u), Some(d)) if b < u.len() && b < d.len() => {
                        let num = u[b] - d[b];
                        if num.is_finite() { num / (2.0 * step) } else { 0.0 }
                    }
                    (Some(u), None) if b < u.len() => {
                        let num = u[b] - base[b];
                        if num.is_finite() { num / step } else { 0.0 }
                    }
                    (None, Some(d)) if b < d.len() => {
                        let num = base[b] - d[b];
                        if num.is_finite() { num / step } else { 0.0 }
                    }
                    _ => 0.0,
                };
                grads[off + b][pidx] = if g.is_finite() { g } else { 0.0 };
            }
        }
    }
    grads
}

fn sigma_from_gradients_diag(grads: &[Vec<f64>], steps: &[f64], mask: Option<&[bool]>) -> Vec<f64> {
    let mut out = vec![0.0; grads.len()];
    for (i, g) in grads.iter().enumerate() {
        let mut var = 0.0;
        for (pidx, &gp) in g.iter().enumerate() {
            let step = steps.get(pidx).copied().unwrap_or(0.0);
            if let Some(m) = mask
                && !m.get(pidx).copied().unwrap_or(false)
            {
                continue;
            }
            if !(gp.is_finite() && step.is_finite() && step > 0.0) {
                continue;
            }
            let d = gp * step;
            var += d * d;
        }
        out[i] = var.max(0.0).sqrt();
    }
    out
}

fn sigma_from_gradients_cov(
    grads: &[Vec<f64>],
    cov: &[f64],
    n_params: usize,
    mask: Option<&[bool]>,
) -> Option<Vec<f64>> {
    if cov.len() != n_params * n_params {
        return None;
    }
    let mut out = vec![0.0; grads.len()];
    for (k, g) in grads.iter().enumerate() {
        let mut var = 0.0;
        for i in 0..n_params {
            if let Some(m) = mask
                && !m.get(i).copied().unwrap_or(false)
            {
                continue;
            }
            let gi = g.get(i).copied().unwrap_or(0.0);
            if !gi.is_finite() || gi == 0.0 {
                continue;
            }
            let row = &cov[i * n_params..(i + 1) * n_params];
            let mut tmp = 0.0;
            for j in 0..n_params {
                if let Some(m) = mask
                    && !m.get(j).copied().unwrap_or(false)
                {
                    continue;
                }
                let gj = g.get(j).copied().unwrap_or(0.0);
                if !gj.is_finite() || gj == 0.0 {
                    continue;
                }
                let cij = row[j];
                if !cij.is_finite() {
                    continue;
                }
                tmp += cij * gj;
            }
            var += gi * tmp;
        }
        out[k] = var.max(0.0).sqrt();
    }
    Some(out)
}

fn envelope_from_total_and_sigma(total: &[f64], sigma: &[f64]) -> BandEnvelope {
    let mut lo = Vec::with_capacity(total.len());
    let mut hi = Vec::with_capacity(total.len());
    for i in 0..total.len() {
        let y = total[i];
        let s = sigma.get(i).copied().unwrap_or(0.0);
        if y.is_finite() && s.is_finite() {
            lo.push((y - s).max(0.0));
            hi.push(y + s);
        } else {
            lo.push(f64::NAN);
            hi.push(f64::NAN);
        }
    }
    BandEnvelope { lo, hi }
}

fn ratio_band_from_total_and_sigma(total: &[f64], sigma: &[f64]) -> BandEnvelope {
    let mut lo = Vec::with_capacity(total.len());
    let mut hi = Vec::with_capacity(total.len());
    for i in 0..total.len() {
        let y = total[i];
        let s = sigma.get(i).copied().unwrap_or(0.0);
        if y.is_finite() && s.is_finite() && y > 0.0 {
            lo.push(((y - s).max(0.0)) / y);
            hi.push((y + s) / y);
        } else {
            lo.push(f64::NAN);
            hi.push(f64::NAN);
        }
    }
    BandEnvelope { lo, hi }
}

pub fn distributions_artifact(
    model: &HistFactoryModel,
    data_by_channel: &HashMap<String, Vec<f64>>,
    bin_edges_by_channel: &HashMap<String, Vec<f64>>,
    params_prefit: &[f64],
    params_postfit: &[f64],
    postfit_uncertainties: Option<&[f64]>,
    postfit_covariance: Option<&[f64]>,
    threads: usize,
    blinded_channels: Option<&HashSet<String>>,
) -> Result<DistributionsArtifact> {
    let pre = model.expected_main_by_channel_sample(params_prefit)?;
    let post = model.expected_main_by_channel_sample(params_postfit)?;
    let n_params = model.parameters().len();
    let stat_mask: Vec<bool> =
        model.parameters().iter().map(|p| is_stat_like_parameter(p.name.as_str())).collect();
    let prefit_steps = prefit_sigmas(model);
    let postfit_steps = if let Some(xs) = postfit_uncertainties {
        if xs.len() == n_params {
            xs.iter().map(|v| if v.is_finite() && *v > 0.0 { *v } else { 0.0 }).collect()
        } else {
            prefit_steps.clone()
        }
    } else {
        prefit_steps.clone()
    };
    let pre_grads = compute_total_gradients(model, params_prefit, &pre, &prefit_steps);
    let post_grads = compute_total_gradients(model, params_postfit, &post, &postfit_steps);
    let pre_sigma_flat = sigma_from_gradients_diag(&pre_grads, &prefit_steps, None);
    let post_sigma_flat = if let Some(cov) = postfit_covariance {
        sigma_from_gradients_cov(&post_grads, cov, n_params, None)
            .unwrap_or_else(|| sigma_from_gradients_diag(&post_grads, &postfit_steps, None))
    } else {
        sigma_from_gradients_diag(&post_grads, &postfit_steps, None)
    };
    let post_sigma_flat_stat = if let Some(cov) = postfit_covariance {
        sigma_from_gradients_cov(&post_grads, cov, n_params, Some(stat_mask.as_slice()))
            .unwrap_or_else(|| {
                sigma_from_gradients_diag(&post_grads, &postfit_steps, Some(stat_mask.as_slice()))
            })
    } else {
        sigma_from_gradients_diag(&post_grads, &postfit_steps, Some(stat_mask.as_slice()))
    };
    let (pre_offsets, _) = channel_offsets(&pre);
    let (post_offsets, _) = channel_offsets(&post);

    let mut post_map: HashMap<&str, &ExpectedChannelSampleYields> = HashMap::new();
    for ch in &post {
        post_map.insert(ch.channel_name.as_str(), ch);
    }

    let mut channels_out: Vec<DistributionsChannelArtifact> = Vec::with_capacity(pre.len());
    for (cidx, ch_pre) in pre.iter().enumerate() {
        let ch_post = post_map.get(ch_pre.channel_name.as_str()).ok_or_else(|| {
            ns_core::Error::Validation(format!("postfit channel missing: {}", ch_pre.channel_name))
        })?;
        let data = data_by_channel.get(&ch_pre.channel_name).ok_or_else(|| {
            ns_core::Error::Validation(format!("data missing for channel: {}", ch_pre.channel_name))
        })?;
        let edges = bin_edges_by_channel.get(&ch_pre.channel_name).ok_or_else(|| {
            ns_core::Error::Validation(format!(
                "bin edges missing for channel: {}",
                ch_pre.channel_name
            ))
        })?;

        let blinded =
            blinded_channels.as_ref().map(|xs| xs.contains(&ch_pre.channel_name)).unwrap_or(false);

        let n_bins = edges.len().saturating_sub(1);
        let (
            data_y,
            data_err_lo,
            data_err_hi,
            data_error_model,
            ratio_y,
            ratio_yerr_lo,
            ratio_yerr_hi,
        ) = if blinded {
            (
                vec![0.0; n_bins],
                vec![0.0; n_bins],
                vec![0.0; n_bins],
                None,
                vec![0.0; n_bins],
                vec![0.0; n_bins],
                vec![0.0; n_bins],
            )
        } else {
            let (data_err_lo, data_err_hi, data_error_model) = data_errors(data);
            let (ratio_y, ratio_yerr_lo, ratio_yerr_hi) =
                ratio_from_data_over_mc(data, &data_err_lo, &data_err_hi, &ch_post.total);
            (
                data.clone(),
                data_err_lo,
                data_err_hi,
                data_error_model,
                ratio_y,
                ratio_yerr_lo,
                ratio_yerr_hi,
            )
        };

        let mut samples = Vec::with_capacity(ch_pre.samples.len());
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
            samples.push(DistributionsSampleSeries {
                name: s_pre.sample_name.clone(),
                prefit_y: s_pre.y.clone(),
                postfit_y: s_post.y.clone(),
            });
        }

        let mc_band_prefit = pre_offsets.get(cidx).and_then(|off| {
            let n = ch_pre.total.len();
            if *off + n <= pre_sigma_flat.len() {
                Some(envelope_from_total_and_sigma(&ch_pre.total, &pre_sigma_flat[*off..*off + n]))
            } else {
                None
            }
        });

        let mc_band_postfit = post_offsets.get(cidx).and_then(|off| {
            let n = ch_post.total.len();
            if *off + n <= post_sigma_flat.len() {
                Some(envelope_from_total_and_sigma(
                    &ch_post.total,
                    &post_sigma_flat[*off..*off + n],
                ))
            } else {
                None
            }
        });

        let ratio_band = post_offsets.get(cidx).and_then(|off| {
            let n = ch_post.total.len();
            if *off + n <= post_sigma_flat.len() {
                Some(ratio_band_from_total_and_sigma(
                    &ch_post.total,
                    &post_sigma_flat[*off..*off + n],
                ))
            } else {
                None
            }
        });
        let mc_band_postfit_stat = post_offsets.get(cidx).and_then(|off| {
            let n = ch_post.total.len();
            if *off + n <= post_sigma_flat_stat.len() {
                Some(envelope_from_total_and_sigma(
                    &ch_post.total,
                    &post_sigma_flat_stat[*off..*off + n],
                ))
            } else {
                None
            }
        });
        let ratio_band_stat = post_offsets.get(cidx).and_then(|off| {
            let n = ch_post.total.len();
            if *off + n <= post_sigma_flat_stat.len() {
                Some(ratio_band_from_total_and_sigma(
                    &ch_post.total,
                    &post_sigma_flat_stat[*off..*off + n],
                ))
            } else {
                None
            }
        });

        channels_out.push(DistributionsChannelArtifact {
            channel_name: ch_pre.channel_name.clone(),
            bin_edges: edges.clone(),
            data_y,
            data_yerr_lo: data_err_lo,
            data_yerr_hi: data_err_hi,
            data_error_model,
            data_is_blinded: if blinded { Some(true) } else { None },
            samples,
            stack_order: None,
            total_prefit_y: ch_pre.total.clone(),
            total_postfit_y: ch_post.total.clone(),
            mc_band_prefit,
            mc_band_postfit,
            mc_band_postfit_stat,
            ratio_policy: RatioPolicy {
                numerator: "data".to_string(),
                denominator: "mc_total_postfit".to_string(),
                zero_policy: "nan".to_string(),
            },
            ratio_y,
            ratio_yerr_lo,
            ratio_yerr_hi,
            ratio_band,
            ratio_band_stat,
        });
    }

    Ok(DistributionsArtifact {
        schema_version: "trex_report_distributions_v0".to_string(),
        meta: DistributionsMeta {
            tool: "nextstat".to_string(),
            tool_version: ns_core::VERSION.to_string(),
            created_unix_ms: now_unix_ms()?,
            parity_mode: ParityMode { threads: threads.max(1), stable_ordering: true },
            input: None,
        },
        channels: channels_out,
    })
}
