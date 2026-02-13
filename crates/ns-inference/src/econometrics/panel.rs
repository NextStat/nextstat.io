//! Panel linear regression with fixed effects and cluster-robust standard errors.
//!
//! Implements entity-demeaned ("within") OLS estimator for balanced and
//! unbalanced panels, plus Liang–Zeger cluster-robust (HC0) sandwich
//! covariance estimator.
//!
//! # References
//!
//! - Wooldridge, *Econometric Analysis of Cross Section and Panel Data*, Ch. 10.
//! - Arellano (1987), "Computing robust standard errors for within-groups estimators."

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

use super::hdfe::FixedEffectsSolver;

/// Result of a panel fixed-effects regression.
#[derive(Debug, Clone)]
pub struct PanelFEResult {
    /// Coefficient estimates (length = p, excludes intercept which is absorbed).
    pub coefficients: Vec<f64>,
    /// OLS (homoskedastic) standard errors.
    pub se_ols: Vec<f64>,
    /// Cluster-robust standard errors (if `cluster_ids` was provided).
    pub se_cluster: Option<Vec<f64>>,
    /// R² (within).
    pub r_squared_within: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of entities (groups).
    pub n_entities: usize,
    /// Number of time periods (0 if 1-way entity FE only).
    pub n_time_periods: usize,
    /// Number of regressors (excluding absorbed FE).
    pub n_regressors: usize,
    /// Degrees of freedom absorbed by FE.
    pub df_absorbed: usize,
    /// Residual sum of squares.
    pub rss: f64,
}

/// Fit a panel fixed-effects ("within") regression.
///
/// Supports **one-way** (entity only) and **two-way** (entity + time) fixed
/// effects. Two-way FE uses the HDFE solver (convergent MAP with Aitken
/// acceleration), exact for both balanced and unbalanced panels.
///
/// # Arguments
///
/// - `entity_ids` — group identifier for each observation (length n).
/// - `time_ids` — optional time-period identifier (length n). When `Some`,
///   two-way (entity + time) FE are absorbed via HDFE. When `None`, only
///   entity FE are absorbed (backward-compatible behavior).
/// - `x` — design matrix, row-major, shape (n, p). No intercept column needed.
/// - `y` — dependent variable (length n).
/// - `cluster_ids` — optional clustering variable for robust SE. If `None`,
///   entities are used as clusters.
///
/// The estimator demeans X and y by the specified FE dimensions, then runs
/// OLS on the demeaned data.
pub fn panel_fe_fit(
    entity_ids: &[u64],
    time_ids: Option<&[u64]>,
    x: &[f64],
    y: &[f64],
    p: usize,
    cluster_ids: Option<&[u64]>,
) -> Result<PanelFEResult> {
    let n = y.len();
    if n == 0 {
        return Err(Error::Validation("y must be non-empty".into()));
    }
    if entity_ids.len() != n {
        return Err(Error::Validation(format!(
            "entity_ids length ({}) != n ({})",
            entity_ids.len(),
            n
        )));
    }
    if x.len() != n * p {
        return Err(Error::Validation(format!("x length ({}) != n*p ({})", x.len(), n * p)));
    }
    if p == 0 {
        return Err(Error::Validation("p must be >= 1".into()));
    }
    if let Some(tids) = time_ids
        && tids.len() != n
    {
        return Err(Error::Validation(format!("time_ids length ({}) != n ({})", tids.len(), n)));
    }

    // Map u64 IDs to dense 0-based indices.
    let mut entity_id_map: HashMap<u64, usize> = HashMap::new();
    let mut entity_dense = Vec::with_capacity(n);
    for &eid in entity_ids {
        let len = entity_id_map.len();
        let idx = *entity_id_map.entry(eid).or_insert(len);
        entity_dense.push(idx);
    }
    let n_entities = entity_id_map.len();

    let (n_time_periods, df_fe, y_dm, x_dm) = if let Some(tids) = time_ids {
        // Two-way FE via HDFE solver.
        let mut time_id_map: HashMap<u64, usize> = HashMap::new();
        let mut time_dense = Vec::with_capacity(n);
        for &tid in tids {
            let len = time_id_map.len();
            let idx = *time_id_map.entry(tid).or_insert(len);
            time_dense.push(idx);
        }
        let n_tp = time_id_map.len();

        let hdfe = FixedEffectsSolver::new(vec![entity_dense, time_dense])?;
        let df = hdfe.degrees_of_freedom_absorbed();

        // Partial out y
        let yd = hdfe.partial_out(y)?;

        // Partial out each column of X
        let mut xd = vec![0.0_f64; n * p];
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
            let col_dm = hdfe.partial_out(&col)?;
            for i in 0..n {
                xd[i * p + j] = col_dm[i];
            }
        }

        (n_tp, df, yd, xd)
    } else {
        // One-way entity FE: single-pass demeaning (exact).
        let hdfe = FixedEffectsSolver::new(vec![entity_dense])?;
        let df = hdfe.degrees_of_freedom_absorbed();

        let yd = hdfe.partial_out(y)?;

        let mut xd = vec![0.0_f64; n * p];
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
            let col_dm = hdfe.partial_out(&col)?;
            for i in 0..n {
                xd[i * p + j] = col_dm[i];
            }
        }

        (0, df, yd, xd)
    };

    // OLS on demeaned data: β = (X'X)⁻¹ X'y
    let x_mat = DMatrix::from_row_slice(n, p, &x_dm);
    let y_vec = DVector::from_column_slice(&y_dm);

    let xtx = x_mat.transpose() * &x_mat;
    let xty = x_mat.transpose() * &y_vec;

    let xtx_inv = xtx
        .clone()
        .try_inverse()
        .ok_or_else(|| Error::Computation("X'X is singular after demeaning".into()))?;

    let beta = &xtx_inv * &xty;
    let coefficients: Vec<f64> = beta.iter().copied().collect();

    // Residuals and RSS
    let y_hat = &x_mat * &beta;
    let resid = &y_vec - &y_hat;
    let rss: f64 = resid.iter().map(|r| r * r).sum();

    // R² within
    let tss: f64 = y_dm.iter().map(|v| v * v).sum();
    let r_squared_within = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    // OLS SE: σ² = RSS / (n − df_absorbed − p)
    let dof = n as f64 - df_fe as f64 - p as f64;
    let sigma2 = if dof > 0.0 { rss / dof } else { f64::NAN };
    let se_ols: Vec<f64> = (0..p).map(|j| (sigma2 * xtx_inv[(j, j)]).sqrt()).collect();

    // Cluster-robust SE
    let clust = cluster_ids.unwrap_or(entity_ids);
    if clust.len() != n {
        return Err(Error::Validation(format!(
            "cluster_ids length ({}) != n ({})",
            clust.len(),
            n
        )));
    }

    let se_cluster = Some(cluster_robust_se(&x_mat, &resid, &xtx_inv, clust, df_fe)?);

    Ok(PanelFEResult {
        coefficients,
        se_ols,
        se_cluster,
        r_squared_within,
        n_obs: n,
        n_entities,
        n_time_periods,
        n_regressors: p,
        df_absorbed: df_fe,
        rss,
    })
}

/// Compute Liang–Zeger cluster-robust (HC0 sandwich) standard errors.
///
/// `V_CR = (X'X)^{-1} B (X'X)^{-1}` where `B = Σ_g X_g' e_g e_g' X_g`.
///
/// # Arguments
///
/// - `df_absorbed` — degrees of freedom consumed by absorbed fixed effects
///   (e.g. from [`FixedEffectsSolver::degrees_of_freedom_absorbed`]). Pass 0
///   when no FE have been absorbed. This enters the small-sample correction
///   factor: `G/(G-1) · (N-1)/(N - p - df_absorbed)`.
pub fn cluster_robust_se(
    x: &DMatrix<f64>,
    residuals: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    cluster_ids: &[u64],
    df_absorbed: usize,
) -> Result<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();

    // Build cluster index
    let mut cluster_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &cid) in cluster_ids.iter().enumerate() {
        cluster_map.entry(cid).or_default().push(i);
    }
    let g = cluster_map.len() as f64;

    // Meat of the sandwich: B = Σ_g X_g' e_g e_g' X_g
    let mut meat = DMatrix::zeros(p, p);
    for indices in cluster_map.values() {
        // Score for cluster g: s_g = X_g' e_g (p × 1)
        let mut s_g = vec![0.0_f64; p];
        for &i in indices {
            let e_i = residuals[i];
            for j in 0..p {
                s_g[j] += x[(i, j)] * e_i;
            }
        }
        // Outer product: s_g s_g'
        for a in 0..p {
            for b in 0..p {
                let val: f64 = s_g[a] * s_g[b];
                meat[(a, b)] += val;
            }
        }
    }

    // Small-sample correction: G/(G-1) * (N-1)/(N-K)
    // K = p (explicit regressors) + df_absorbed (FE degrees of freedom).
    let n_f = n as f64;
    let k_f = p as f64 + df_absorbed as f64;
    let correction =
        if g > 1.0 && n_f > k_f { (g / (g - 1.0)) * ((n_f - 1.0) / (n_f - k_f)) } else { 1.0 };

    let vcr = (xtx_inv * &meat) * xtx_inv * correction;

    let se: Vec<f64> = (0..p).map(|j| vcr[(j, j)].max(0.0).sqrt()).collect();
    Ok(se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_fe_two_entities() {
        // Two entities, 3 obs each, single regressor
        // Entity 1: x=[1,2,3], y=[2,4,6] => within: x_dm=[-1,0,1], y_dm=[-2,0,2] => beta=2
        // Entity 2: x=[10,20,30], y=[20,40,60] => within: x_dm=[-10,0,10], y_dm=[-20,0,20] => beta=2
        let entity_ids = vec![1, 1, 1, 2, 2, 2];
        let x = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let y = vec![2.0, 4.0, 6.0, 20.0, 40.0, 60.0];

        let res = panel_fe_fit(&entity_ids, None, &x, &y, 1, None).unwrap();
        assert_eq!(res.n_obs, 6);
        assert_eq!(res.n_entities, 2);
        assert_eq!(res.n_regressors, 1);
        assert!((res.coefficients[0] - 2.0).abs() < 1e-10, "beta={}", res.coefficients[0]);
        assert!(res.r_squared_within > 0.999);
        assert!(res.rss < 1e-20);
    }

    #[test]
    fn test_panel_fe_with_noise() {
        // Entity 1: y = 5 + 3*x + noise, Entity 2: y = 10 + 3*x + noise
        // FE absorbs entity intercepts, should recover beta ≈ 3
        let entity_ids = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let x = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![
            8.1, 11.0, 13.9, 17.1, // entity 1: ~5 + 3x
            13.0, 16.1, 18.9, 22.0, // entity 2: ~10 + 3x
        ];

        let res = panel_fe_fit(&entity_ids, None, &x, &y, 1, None).unwrap();
        assert!(
            (res.coefficients[0] - 3.0).abs() < 0.2,
            "beta={}, expected ~3",
            res.coefficients[0]
        );
        assert!(res.se_ols[0] > 0.0);
        assert!(res.se_cluster.is_some());
    }

    #[test]
    fn test_cluster_se_larger_than_ols() {
        // With clustered data, cluster SE should generally differ from OLS SE
        let entity_ids = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let res = panel_fe_fit(&entity_ids, None, &x, &y, 1, None).unwrap();
        assert!(res.se_cluster.is_some());
        // Both should be non-NaN
        assert!(res.se_ols[0].is_finite());
        assert!(res.se_cluster.as_ref().unwrap()[0].is_finite());
    }

    #[test]
    fn test_panel_fe_two_way() {
        // 3 entities × 4 time periods
        // x = entity * time (interaction — not collinear with entity + time main effects)
        // y = 10*entity + 2*time + 3*x + 0
        // Two-way FE should recover beta = 3.
        let mut entity_ids = Vec::new();
        let mut time_ids = Vec::new();
        let mut x = Vec::new();
        let mut y = Vec::new();

        for eid in 1..=3u64 {
            for t in 1..=4u64 {
                entity_ids.push(eid);
                time_ids.push(t);
                let xi = eid as f64 * t as f64;
                x.push(xi);
                y.push(10.0 * eid as f64 + 2.0 * t as f64 + 3.0 * xi);
            }
        }

        let res = panel_fe_fit(&entity_ids, Some(&time_ids), &x, &y, 1, None).unwrap();

        assert_eq!(res.n_obs, 12);
        assert_eq!(res.n_entities, 3);
        assert_eq!(res.n_time_periods, 4);
        // df_absorbed = 3 + 4 - 1 = 6 (fully connected)
        assert_eq!(res.df_absorbed, 6);
        assert!(
            (res.coefficients[0] - 3.0).abs() < 1e-8,
            "beta={}, expected 3.0",
            res.coefficients[0]
        );
        assert!(res.r_squared_within > 0.99);
    }

    #[test]
    fn test_panel_fe_two_way_unbalanced() {
        // Unbalanced: entity 1 has t={1,2,3,4,5}, entity 2 has t={2,3,4,5,6},
        // entity 3 has t={1,3,4,5,6}. 15 obs, 3 entities, 6 times.
        // df_absorbed = 3+6-1 = 8, residual df = 15-8-1 = 6. Enough for estimation.
        // x values are arbitrary (no entity/time structure), y = 5*eid + 1.5*t + 2*x
        let ranges: &[(u64, &[u64])] =
            &[(1, &[1, 2, 3, 4, 5]), (2, &[2, 3, 4, 5, 6]), (3, &[1, 3, 4, 5, 6])];
        // Pseudo-random x values with no entity/time pattern
        let x_vals: &[f64] = &[
            0.3, 1.7, 0.8, 2.5, 1.1, // entity 1
            3.2, 0.6, 1.9, 2.8, 0.4, // entity 2
            1.5, 2.1, 0.9, 3.0, 1.3, // entity 3
        ];
        let mut entity_ids = Vec::new();
        let mut time_ids = Vec::new();
        let mut x = Vec::new();
        let mut y = Vec::new();

        let mut k = 0;
        for &(eid, times) in ranges {
            for &t in times {
                entity_ids.push(eid);
                time_ids.push(t);
                let xi = x_vals[k];
                x.push(xi);
                y.push(5.0 * eid as f64 + 1.5 * t as f64 + 2.0 * xi);
                k += 1;
            }
        }

        let res = panel_fe_fit(&entity_ids, Some(&time_ids), &x, &y, 1, None).unwrap();

        assert_eq!(res.n_entities, 3);
        assert_eq!(res.n_time_periods, 6);
        assert!(
            (res.coefficients[0] - 2.0).abs() < 0.01,
            "beta={}, expected ~2.0",
            res.coefficients[0]
        );
    }

    #[test]
    fn test_panel_fe_validation() {
        assert!(panel_fe_fit(&[], None, &[], &[], 1, None).is_err());
        assert!(panel_fe_fit(&[1], None, &[1.0], &[1.0, 2.0], 1, None).is_err());
    }
}
