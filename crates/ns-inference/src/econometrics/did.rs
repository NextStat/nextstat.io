//! Difference-in-Differences (DiD) and event-study estimators.
//!
//! Implements the canonical two-period / two-group DiD estimator and
//! a multi-period event-study specification with leads and lags around
//! treatment onset.
//!
//! # References
//!
//! - Angrist & Pischke, *Mostly Harmless Econometrics*, Ch. 5.
//! - Callaway & Sant'Anna (2021), "Difference-in-Differences with multiple
//!   time periods." *Journal of Econometrics*.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

use super::panel::cluster_robust_se;

/// Result of a canonical (2×2) DiD estimator.
#[derive(Debug, Clone)]
pub struct DidResult {
    /// ATT estimate (Average Treatment effect on the Treated).
    pub att: f64,
    /// Standard error of the ATT (OLS).
    pub se: f64,
    /// Cluster-robust standard error of the ATT.
    pub se_cluster: f64,
    /// t-statistic (ATT / se_cluster).
    pub t_stat: f64,
    /// Mean outcome: treated-post.
    pub mean_treated_post: f64,
    /// Mean outcome: treated-pre.
    pub mean_treated_pre: f64,
    /// Mean outcome: control-post.
    pub mean_control_post: f64,
    /// Mean outcome: control-pre.
    pub mean_control_pre: f64,
    /// Number of observations.
    pub n_obs: usize,
}

/// Canonical two-period DiD estimator.
///
/// Estimates ATT = (Ȳ_treat,post − Ȳ_treat,pre) − (Ȳ_ctrl,post − Ȳ_ctrl,pre)
/// via OLS: `y = α + β₁·treat + β₂·post + δ·(treat×post) + ε`,
/// where δ is the DiD estimate.
///
/// # Arguments
///
/// - `y` — outcome variable (length n).
/// - `treat` — treatment indicator: 1 = treated group, 0 = control (length n).
/// - `post` — post-treatment indicator: 1 = post, 0 = pre (length n).
/// - `cluster_ids` — clustering variable for robust SE (typically entity id).
pub fn did_canonical(
    y: &[f64],
    treat: &[u8],
    post: &[u8],
    cluster_ids: &[u64],
) -> Result<DidResult> {
    let n = y.len();
    if n == 0 {
        return Err(Error::Validation("y must be non-empty".into()));
    }
    if treat.len() != n || post.len() != n || cluster_ids.len() != n {
        return Err(Error::Validation(
            "treat, post, cluster_ids must have same length as y".into(),
        ));
    }

    // Compute group means
    let mut sum_tp = 0.0_f64;
    let mut n_tp = 0usize;
    let mut sum_t0 = 0.0_f64;
    let mut n_t0 = 0usize;
    let mut sum_cp = 0.0_f64;
    let mut n_cp = 0usize;
    let mut sum_c0 = 0.0_f64;
    let mut n_c0 = 0usize;

    for i in 0..n {
        match (treat[i], post[i]) {
            (1, 1) => {
                sum_tp += y[i];
                n_tp += 1;
            }
            (1, 0) => {
                sum_t0 += y[i];
                n_t0 += 1;
            }
            (0, 1) => {
                sum_cp += y[i];
                n_cp += 1;
            }
            (0, 0) => {
                sum_c0 += y[i];
                n_c0 += 1;
            }
            _ => {
                return Err(Error::Validation("treat and post must be 0 or 1".into()));
            }
        }
    }

    if n_tp == 0 || n_t0 == 0 || n_cp == 0 || n_c0 == 0 {
        return Err(Error::Validation("All four cells (treat×post) must have observations".into()));
    }

    let mean_tp = sum_tp / n_tp as f64;
    let mean_t0 = sum_t0 / n_t0 as f64;
    let mean_cp = sum_cp / n_cp as f64;
    let mean_c0 = sum_c0 / n_c0 as f64;

    // OLS: y = α + β₁·D + β₂·P + δ·(D×P) + ε
    // Columns: [intercept, treat, post, treat*post]
    let k = 4usize;
    let mut x_data = Vec::with_capacity(n * k);
    for i in 0..n {
        let d = treat[i] as f64;
        let p = post[i] as f64;
        x_data.push(1.0);
        x_data.push(d);
        x_data.push(p);
        x_data.push(d * p);
    }

    let x_mat = DMatrix::from_row_slice(n, k, &x_data);
    let y_vec = DVector::from_column_slice(y);

    let xtx = x_mat.transpose() * &x_mat;
    let xty = x_mat.transpose() * &y_vec;
    let xtx_inv =
        xtx.try_inverse().ok_or_else(|| Error::Computation("X'X singular in DiD OLS".into()))?;
    let beta = &xtx_inv * &xty;

    let att = beta[3]; // coefficient on treat×post

    // Residuals
    let y_hat = &x_mat * &beta;
    let resid = &y_vec - &y_hat;
    let rss: f64 = resid.iter().map(|r| r * r).sum();

    // OLS SE
    let dof = n as f64 - k as f64;
    let sigma2 = if dof > 0.0 { rss / dof } else { f64::NAN };
    let se = (sigma2 * xtx_inv[(3, 3)]).sqrt();

    // Cluster-robust SE
    let se_cluster_vec = cluster_robust_se(&x_mat, &resid, &xtx_inv, cluster_ids)?;
    let se_cluster = se_cluster_vec[3];

    let t_stat = if se_cluster > 0.0 { att / se_cluster } else { f64::NAN };

    Ok(DidResult {
        att,
        se,
        se_cluster,
        t_stat,
        mean_treated_post: mean_tp,
        mean_treated_pre: mean_t0,
        mean_control_post: mean_cp,
        mean_control_pre: mean_c0,
        n_obs: n,
    })
}

/// Result of an event-study specification.
#[derive(Debug, Clone)]
pub struct EventStudyResult {
    /// Relative time indices (e.g. -3, -2, -1, 0, 1, 2).
    pub relative_times: Vec<i64>,
    /// Point estimates for each lead/lag.
    pub coefficients: Vec<f64>,
    /// Cluster-robust standard errors for each lead/lag.
    pub se_cluster: Vec<f64>,
    /// 95% CI lower bounds.
    pub ci_lower: Vec<f64>,
    /// 95% CI upper bounds.
    pub ci_upper: Vec<f64>,
    /// Number of observations.
    pub n_obs: usize,
    /// Reference (omitted) period.
    pub reference_period: i64,
}

/// Event-study regression with leads and lags.
///
/// Estimates `y = α_i + λ_t + Σ_k δ_k · 1{t - t*_i = k} + ε`,
/// where `t*_i` is the treatment onset time for entity i.
///
/// # Arguments
///
/// - `y` — outcome (length n).
/// - `entity_ids` — entity identifier (length n).
/// - `time_ids` — time period (length n).
/// - `relative_time` — time relative to treatment onset (length n).
///   For never-treated units, use a large positive value (e.g. 9999).
/// - `min_lag`, `max_lag` — range of leads/lags to include (inclusive).
/// - `reference_period` — omitted period (typically -1).
/// - `cluster_ids` — clustering variable for robust SE.
pub fn event_study(
    y: &[f64],
    entity_ids: &[u64],
    time_ids: &[u64],
    relative_time: &[i64],
    min_lag: i64,
    max_lag: i64,
    reference_period: i64,
    cluster_ids: &[u64],
) -> Result<EventStudyResult> {
    let n = y.len();
    if n == 0 {
        return Err(Error::Validation("y must be non-empty".into()));
    }
    if entity_ids.len() != n
        || time_ids.len() != n
        || relative_time.len() != n
        || cluster_ids.len() != n
    {
        return Err(Error::Validation("All input arrays must have length n".into()));
    }
    if min_lag > max_lag {
        return Err(Error::Validation("min_lag must be <= max_lag".into()));
    }
    if reference_period < min_lag || reference_period > max_lag {
        return Err(Error::Validation("reference_period must be within [min_lag, max_lag]".into()));
    }

    // Build entity and time FE dummies via demeaning
    // Step 1: entity demean
    let mut entity_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &eid) in entity_ids.iter().enumerate() {
        entity_map.entry(eid).or_default().push(i);
    }

    let mut time_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &tid) in time_ids.iter().enumerate() {
        time_map.entry(tid).or_default().push(i);
    }

    // Build lead/lag indicator columns
    let mut rel_times: Vec<i64> = (min_lag..=max_lag).filter(|&k| k != reference_period).collect();
    rel_times.sort();
    let n_lags = rel_times.len();

    // Design matrix: entity dummies + time dummies + lead/lag dummies
    // Use demeaning approach: demean by entity, then include time dummies + lag dummies
    // Simpler approach for robustness: build full OLS with entity FE absorbed via demeaning

    // Demean y and lag indicators by entity
    let mut y_dm = vec![0.0_f64; n];
    let mut lag_indicators = vec![0.0_f64; n * n_lags]; // row-major

    // First, fill lag indicators
    for i in 0..n {
        let rt = relative_time[i];
        if let Ok(pos) = rel_times.binary_search(&rt) {
            lag_indicators[i * n_lags + pos] = 1.0;
        }
    }

    // Demean y and lag indicators by entity
    for indices in entity_map.values() {
        let ni = indices.len() as f64;
        let mut y_mean = 0.0;
        let mut lag_means = vec![0.0; n_lags];
        for &i in indices {
            y_mean += y[i];
            for j in 0..n_lags {
                lag_means[j] += lag_indicators[i * n_lags + j];
            }
        }
        y_mean /= ni;
        for j in 0..n_lags {
            lag_means[j] /= ni;
        }
        for &i in indices {
            y_dm[i] = y[i] - y_mean;
            for j in 0..n_lags {
                lag_indicators[i * n_lags + j] -= lag_means[j];
            }
        }
    }

    // Also demean by time (two-way FE via iterative demeaning — one pass is
    // sufficient for the typical case where time FE are not collinear with lags)
    for indices in time_map.values() {
        let ni = indices.len() as f64;
        let mut y_mean = 0.0;
        let mut lag_means = vec![0.0; n_lags];
        for &i in indices {
            y_mean += y_dm[i];
            for j in 0..n_lags {
                lag_means[j] += lag_indicators[i * n_lags + j];
            }
        }
        y_mean /= ni;
        for j in 0..n_lags {
            lag_means[j] /= ni;
        }
        for &i in indices {
            y_dm[i] -= y_mean;
            for j in 0..n_lags {
                lag_indicators[i * n_lags + j] -= lag_means[j];
            }
        }
    }

    // OLS on demeaned data
    let x_mat = DMatrix::from_row_slice(n, n_lags, &lag_indicators);
    let y_vec = DVector::from_column_slice(&y_dm);

    let xtx = x_mat.transpose() * &x_mat;
    let xty = x_mat.transpose() * &y_vec;
    let xtx_inv = xtx
        .try_inverse()
        .ok_or_else(|| Error::Computation("X'X singular in event study OLS".into()))?;
    let beta = &xtx_inv * &xty;
    let coefficients: Vec<f64> = beta.iter().copied().collect();

    // Residuals
    let y_hat = &x_mat * &beta;
    let resid = &y_vec - &y_hat;

    // Cluster-robust SE
    let se_cluster_vec = cluster_robust_se(&x_mat, &resid, &xtx_inv, cluster_ids)?;

    let ci_lower: Vec<f64> =
        coefficients.iter().zip(&se_cluster_vec).map(|(b, s)| b - 1.96 * s).collect();
    let ci_upper: Vec<f64> =
        coefficients.iter().zip(&se_cluster_vec).map(|(b, s)| b + 1.96 * s).collect();

    Ok(EventStudyResult {
        relative_times: rel_times,
        coefficients,
        se_cluster: se_cluster_vec,
        ci_lower,
        ci_upper,
        n_obs: n,
        reference_period,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_did_canonical_exact() {
        // Control: pre=10, post=12 (trend = +2)
        // Treated: pre=10, post=15 (trend = +5, treatment effect = 3)
        let y = vec![10.0, 10.0, 12.0, 12.0, 10.0, 10.0, 15.0, 15.0];
        let treat = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let post = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let cluster = vec![1, 2, 1, 2, 3, 4, 3, 4];

        let res = did_canonical(&y, &treat, &post, &cluster).unwrap();
        assert!((res.att - 3.0).abs() < 1e-10, "ATT={}, expected 3.0", res.att);
        assert_eq!(res.n_obs, 8);
        assert!((res.mean_treated_post - 15.0).abs() < 1e-10);
        assert!((res.mean_treated_pre - 10.0).abs() < 1e-10);
        assert!((res.mean_control_post - 12.0).abs() < 1e-10);
        assert!((res.mean_control_pre - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_did_validation() {
        assert!(did_canonical(&[], &[], &[], &[]).is_err());
        assert!(did_canonical(&[1.0], &[2], &[0], &[1]).is_err()); // treat=2 invalid
    }

    #[test]
    fn test_event_study_basic() {
        // 4 entities, 5 time periods, treatment at t=3 for entities 3,4
        let n = 20;
        let mut y = Vec::with_capacity(n);
        let mut entity_ids = Vec::with_capacity(n);
        let mut time_ids = Vec::with_capacity(n);
        let mut relative_time = Vec::with_capacity(n);
        let mut cluster_ids = Vec::with_capacity(n);

        for eid in 1..=4u64 {
            let treated = eid >= 3;
            for t in 1..=5u64 {
                entity_ids.push(eid);
                time_ids.push(t);
                cluster_ids.push(eid);

                let rt = if treated { t as i64 - 3 } else { 9999 };
                relative_time.push(rt);

                // y = entity_fe + time_fe + treatment_effect * post
                let entity_fe = eid as f64 * 2.0;
                let time_fe = t as f64;
                let effect = if treated && t >= 3 { 5.0 } else { 0.0 };
                y.push(entity_fe + time_fe + effect);
            }
        }

        let res = event_study(&y, &entity_ids, &time_ids, &relative_time, -2, 2, -1, &cluster_ids)
            .unwrap();

        assert_eq!(res.n_obs, n);
        assert_eq!(res.reference_period, -1);
        // Should have lags: -2, 0, 1, 2 (reference -1 omitted)
        assert_eq!(res.relative_times, vec![-2, 0, 1, 2]);
        assert_eq!(res.coefficients.len(), 4);
    }
}
