use anyhow::Result;
use nalgebra::DMatrix;
use ns_core::traits::LogDensityModel;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
struct CoxPhInputJson {
    times: Vec<f64>,
    events: Vec<bool>,
    x: Vec<Vec<f64>>,
    #[serde(default)]
    groups: Option<Vec<i64>>,
}

#[inline]
fn exp_clamped(x: f64) -> f64 {
    // Avoid overflow while keeping behavior close to the rest of the codebase.
    if x > 700.0 {
        700.0_f64.exp()
    } else if x < -700.0 {
        (-700.0_f64).exp()
    } else {
        x.exp()
    }
}

fn sorted_desc(
    times: &[f64],
    events: &[bool],
    x: &[Vec<f64>],
    groups: Option<&[i64]>,
) -> Result<(Vec<f64>, Vec<bool>, Vec<Vec<f64>>, Option<Vec<i64>>)> {
    let n = times.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| times[j].partial_cmp(&times[i]).unwrap().then_with(|| i.cmp(&j)));

    let mut times_s = Vec::with_capacity(n);
    let mut events_s = Vec::with_capacity(n);
    let mut x_s = Vec::with_capacity(n);
    let mut groups_s: Option<Vec<i64>> = groups.map(|_| Vec::with_capacity(n));

    for idx in order {
        times_s.push(times[idx]);
        events_s.push(events[idx]);
        x_s.push(x[idx].clone());
        if let (Some(gs), Some(gv)) = (groups_s.as_mut(), groups) {
            gs.push(gv[idx]);
        }
    }

    Ok((times_s, events_s, x_s, groups_s))
}

fn group_starts(times_s: &[f64]) -> Vec<usize> {
    let n = times_s.len();
    let mut starts = Vec::new();
    if n == 0 {
        return starts;
    }
    starts.push(0);
    for i in 1..n {
        if times_s[i] != times_s[i - 1] {
            starts.push(i);
        }
    }
    starts
}

fn hessian_from_grad<M: LogDensityModel>(model: &M, beta: &[f64]) -> Result<DMatrix<f64>> {
    let p = beta.len();
    if p == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    let base: Vec<f64> = beta.to_vec();
    let mut h = DMatrix::zeros(p, p);

    for j in 0..p {
        let eps = 1e-5 * (base[j].abs() + 1.0);
        let mut bp = base.clone();
        let mut bm = base.clone();
        bp[j] += eps;
        bm[j] -= eps;
        let gp = model.grad_nll(&bp).map_err(|e| anyhow::anyhow!("grad_nll failed: {e}"))?;
        let gm = model.grad_nll(&bm).map_err(|e| anyhow::anyhow!("grad_nll failed: {e}"))?;
        if gp.len() != p || gm.len() != p {
            anyhow::bail!("grad_nll returned length mismatch");
        }
        for i in 0..p {
            h[(i, j)] = (gp[i] - gm[i]) / (2.0 * eps);
        }
    }

    // Symmetrize to reduce FD noise.
    for i in 0..p {
        for j in (i + 1)..p {
            let v = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }

    Ok(h)
}

fn cox_score_residual_outer(
    times: &[f64],
    events: &[bool],
    x: &[Vec<f64>],
    beta: &[f64],
    ties: &str,
    groups: Option<&[i64]>,
) -> Result<DMatrix<f64>> {
    let n = times.len();
    if n == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    let p = x[0].len();
    if p == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    let (times_s, events_s, x_s, groups_s) = sorted_desc(times, events, x, groups)?;
    let starts = group_starts(&times_s);

    let mut w: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += x_s[i][j] * beta[j];
        }
        w.push(exp_clamped(eta));
    }

    let min_tail = 1e-300_f64;
    let mut risk0 = 0.0_f64;
    let mut risk1 = vec![0.0_f64; p];

    let mut b = DMatrix::zeros(p, p);
    let mut cluster_sums: Option<HashMap<i64, Vec<f64>>> =
        groups_s.as_ref().map(|_| HashMap::new());

    for (g, &start) in starts.iter().enumerate() {
        let end = starts.get(g + 1).copied().unwrap_or(n);

        // Add this time-slice into the risk set.
        for i in start..end {
            let wi = w[i];
            risk0 += wi;
            for j in 0..p {
                risk1[j] += wi * x_s[i][j];
            }
        }

        // Collect events at this time.
        let mut event_idx: Vec<usize> = Vec::new();
        let mut d0 = 0.0_f64;
        let mut d1 = vec![0.0_f64; p];
        for i in start..end {
            if !events_s[i] {
                continue;
            }
            event_idx.push(i);
            let wi = w[i];
            d0 += wi;
            for j in 0..p {
                d1[j] += wi * x_s[i][j];
            }
        }
        let m = event_idx.len();
        if m == 0 {
            continue;
        }

        let xbar: Vec<f64> = match ties {
            "breslow" => {
                let denom = risk0.max(min_tail);
                (0..p).map(|j| risk1[j] / denom).collect()
            }
            "efron" => {
                let mf = m as f64;
                let mut sum_xbar = vec![0.0_f64; p];
                for r in 0..m {
                    let frac = (r as f64) / mf;
                    let denom = (risk0 - frac * d0).max(min_tail);
                    for j in 0..p {
                        let num = risk1[j] - frac * d1[j];
                        sum_xbar[j] += num / denom;
                    }
                }
                sum_xbar.into_iter().map(|v| v / mf).collect()
            }
            other => anyhow::bail!("ties must be 'breslow' or 'efron', got: {other}"),
        };

        for &i in &event_idx {
            let mut r = vec![0.0_f64; p];
            for j in 0..p {
                r[j] = x_s[i][j] - xbar[j];
            }

            if let (Some(ref mut map), Some(gs)) = (cluster_sums.as_mut(), groups_s.as_ref()) {
                let gid = gs[i];
                match map.get_mut(&gid) {
                    Some(acc) => {
                        for j in 0..p {
                            acc[j] += r[j];
                        }
                    }
                    None => {
                        map.insert(gid, r);
                    }
                }
            } else {
                for a in 0..p {
                    for c in 0..p {
                        b[(a, c)] += r[a] * r[c];
                    }
                }
            }
        }
    }

    if let Some(map) = cluster_sums {
        let mut b2 = DMatrix::zeros(p, p);
        for v in map.values() {
            for a in 0..p {
                for c in 0..p {
                    b2[(a, c)] += v[a] * v[c];
                }
            }
        }
        return Ok(b2);
    }

    Ok(b)
}

fn cox_baseline_cumhaz(
    times: &[f64],
    events: &[bool],
    x: &[Vec<f64>],
    beta: &[f64],
    ties: &str,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = times.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    let p = x[0].len();
    if p == 0 {
        return Ok((vec![], vec![]));
    }

    let (times_s, events_s, x_s, _groups_s) = sorted_desc(times, events, x, None)?;
    let starts = group_starts(&times_s);

    let mut w: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += x_s[i][j] * beta[j];
        }
        w.push(exp_clamped(eta));
    }

    let min_tail = 1e-300_f64;
    let mut risk0 = 0.0_f64;
    let mut deltas_desc: Vec<(f64, f64)> = Vec::new();

    for (g, &start) in starts.iter().enumerate() {
        let end = starts.get(g + 1).copied().unwrap_or(n);
        for i in start..end {
            risk0 += w[i];
        }

        let mut m = 0usize;
        let mut d0 = 0.0_f64;
        for i in start..end {
            if events_s[i] {
                m += 1;
                d0 += w[i];
            }
        }
        if m == 0 {
            continue;
        }

        let t0 = times_s[start];
        let inc = match ties {
            "breslow" => (m as f64) / risk0.max(min_tail),
            "efron" => {
                let mf = m as f64;
                let mut s = 0.0_f64;
                for r in 0..m {
                    let frac = (r as f64) / mf;
                    let denom = (risk0 - frac * d0).max(min_tail);
                    s += 1.0 / denom;
                }
                s
            }
            other => anyhow::bail!("ties must be 'breslow' or 'efron', got: {other}"),
        };
        deltas_desc.push((t0, inc));
    }

    if deltas_desc.is_empty() {
        return Ok((vec![], vec![]));
    }

    // We swept times in descending order; reverse to ascending for output.
    deltas_desc.reverse();
    let mut ts = Vec::with_capacity(deltas_desc.len());
    let mut hs = Vec::with_capacity(deltas_desc.len());
    let mut cum = 0.0_f64;
    for (t0, delta) in deltas_desc {
        ts.push(t0);
        cum += delta;
        hs.push(cum);
    }
    Ok((ts, hs))
}

pub fn cmd_survival_cox_ph_fit(
    input: &PathBuf,
    output: Option<&PathBuf>,
    ties: &str,
    robust: bool,
    cluster_correction: bool,
    baseline: bool,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let bytes = std::fs::read(input)?;
    let injson: CoxPhInputJson = serde_json::from_slice(&bytes)?;

    if injson.times.is_empty() {
        anyhow::bail!("times must be non-empty");
    }
    if injson.times.len() != injson.events.len() {
        anyhow::bail!(
            "times/events length mismatch: {} vs {}",
            injson.times.len(),
            injson.events.len()
        );
    }
    if injson.times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        anyhow::bail!("times must be finite and >= 0");
    }
    if injson.x.len() != injson.times.len() {
        anyhow::bail!(
            "x must have n rows: expected {}, got {}",
            injson.times.len(),
            injson.x.len()
        );
    }
    if injson.x.is_empty() || injson.x[0].is_empty() {
        anyhow::bail!("x must have at least 1 feature column");
    }
    let p = injson.x[0].len();
    for (i, row) in injson.x.iter().enumerate() {
        if row.len() != p {
            anyhow::bail!("x must be rectangular: row {i} has len {}, expected {}", row.len(), p);
        }
        if row.iter().any(|v| !v.is_finite()) {
            anyhow::bail!("x must contain only finite values");
        }
    }
    if injson.events.iter().all(|d| !*d) {
        anyhow::bail!("need at least one event");
    }

    let groups: Option<Vec<i64>> = injson.groups;
    if let Some(ref g) = groups
        && g.len() != injson.times.len() {
            anyhow::bail!("groups must have length n");
        }

    let ties_enum = match ties {
        "breslow" => ns_inference::CoxTies::Breslow,
        "efron" => ns_inference::CoxTies::Efron,
        other => anyhow::bail!("ties must be 'breslow' or 'efron', got: {other}"),
    };

    let model = ns_inference::CoxPhModel::new(
        injson.times.clone(),
        injson.events.clone(),
        injson.x.clone(),
        ties_enum,
    )
    .map_err(|e| anyhow::anyhow!("invalid Cox PH input: {e}"))?;

    let mle = ns_inference::MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).map_err(|e| anyhow::anyhow!("fit failed: {e}"))?;
    let beta = fit.parameters.clone();
    if beta.len() != p {
        anyhow::bail!("internal error: fitted dim mismatch");
    }

    // Covariance (inverse observed information). Prefer the estimator output; fall back to FD Hessian inversion.
    let cov_mat: DMatrix<f64> = if let Some(cov_flat) = fit.covariance.as_ref() {
        if cov_flat.len() != p * p {
            anyhow::bail!("internal error: covariance has wrong length");
        }
        DMatrix::from_row_slice(p, p, cov_flat)
    } else {
        let h = hessian_from_grad(&model, &beta)?;
        h.try_inverse().ok_or_else(|| anyhow::anyhow!("failed to invert Hessian (covariance)"))?
    };

    let mut se = Vec::with_capacity(p);
    for i in 0..p {
        let v = cov_mat[(i, i)];
        se.push(v.max(0.0).sqrt());
    }

    let (robust_cov, robust_se, robust_kind, robust_meta) = if robust {
        let bmat = cox_score_residual_outer(
            &injson.times,
            &injson.events,
            &injson.x,
            &beta,
            ties,
            groups.as_deref(),
        )?;
        let mut rcov = &cov_mat * bmat * &cov_mat;

        let mut meta = serde_json::json!({ "enabled": true });
        let kind = if groups.is_some() { "cluster" } else { "hc0" };
        meta["kind"] = serde_json::json!(kind);

        if let Some(ref g) = groups {
            let mut uniq = BTreeSet::new();
            for v in g {
                uniq.insert(*v);
            }
            let g_unique = uniq.len();
            meta["n_groups"] = serde_json::json!(g_unique);
            meta["cluster_correction"] = serde_json::json!(cluster_correction);
            if cluster_correction {
                if g_unique <= 1 {
                    anyhow::bail!("cluster_correction requires at least 2 unique groups");
                }
                let factor = (g_unique as f64) / ((g_unique as f64) - 1.0);
                rcov *= factor;
                meta["cluster_correction_factor"] = serde_json::json!(factor);
            }
        }

        let mut rse = Vec::with_capacity(p);
        for i in 0..p {
            let v = rcov[(i, i)];
            rse.push(v.max(0.0).sqrt());
        }
        (Some(rcov), Some(rse), Some(kind.to_string()), meta)
    } else {
        (None, None, None, serde_json::json!({ "enabled": false }))
    };

    let (baseline_times, baseline_cumhaz) = if baseline {
        let (t, h) = cox_baseline_cumhaz(&injson.times, &injson.events, &injson.x, &beta, ties)?;
        (serde_json::json!(t), serde_json::json!(h))
    } else {
        (serde_json::Value::Null, serde_json::Value::Null)
    };

    let param_names = model.parameter_names();
    let cov_nested = crate::dmatrix_to_nested(&cov_mat);
    let robust_cov_nested = robust_cov.as_ref().map(crate::dmatrix_to_nested);

    let output_json = serde_json::json!({
        "model": "cox_ph",
        "ties": ties,
        "n": injson.times.len(),
        "p": p,
        "parameter_names": param_names,
        "coef": beta,
        "nll": fit.nll,
        "converged": fit.converged,
        "n_iter": fit.n_iter,
        "se": se,
        "cov": cov_nested,
        "robust_kind": robust_kind,
        "robust_se": robust_se,
        "robust_cov": robust_cov_nested,
        "robust_meta": robust_meta,
        "baseline_times": baseline_times,
        "baseline_cumhaz": baseline_cumhaz,
    });

    crate::write_json(output, &output_json)?;
    if let Some(dir) = bundle {
        crate::report::write_bundle(
            dir,
            "survival_cox_ph_fit",
            serde_json::json!({
                "ties": ties,
                "robust": robust,
                "cluster_correction": cluster_correction,
                "baseline": baseline,
            }),
            input,
            &output_json,
        )?;
    }

    Ok(())
}
