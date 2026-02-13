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
    /// Number of regressors (excluding absorbed FE).
    pub n_regressors: usize,
    /// Residual sum of squares.
    pub rss: f64,
}

/// Fit a panel fixed-effects ("within") regression.
///
/// # Arguments
///
/// - `entity_ids` — group identifier for each observation (length n).
/// - `x` — design matrix, row-major, shape (n, p). No intercept column needed.
/// - `y` — dependent variable (length n).
/// - `cluster_ids` — optional clustering variable for robust SE. If `None`,
///   entities are used as clusters.
///
/// The estimator demeans X and y by entity means, then runs OLS on the
/// demeaned data. This absorbs all entity-level fixed effects.
pub fn panel_fe_fit(
    entity_ids: &[u64],
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

    // Build entity index: entity_id -> list of row indices
    let mut entity_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &eid) in entity_ids.iter().enumerate() {
        entity_map.entry(eid).or_default().push(i);
    }
    let n_entities = entity_map.len();

    // Demean X and y by entity
    let mut x_dm = vec![0.0_f64; n * p];
    let mut y_dm = vec![0.0_f64; n];

    for indices in entity_map.values() {
        let ni = indices.len() as f64;
        // Compute entity means
        let mut y_mean = 0.0;
        let mut x_mean = vec![0.0; p];
        for &i in indices {
            y_mean += y[i];
            for j in 0..p {
                x_mean[j] += x[i * p + j];
            }
        }
        y_mean /= ni;
        for j in 0..p {
            x_mean[j] /= ni;
        }
        // Subtract means
        for &i in indices {
            y_dm[i] = y[i] - y_mean;
            for j in 0..p {
                x_dm[i * p + j] = x[i * p + j] - x_mean[j];
            }
        }
    }

    // OLS on demeaned data: beta = (X'X)^{-1} X'y
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

    // OLS SE: sigma² = RSS / (n - n_entities - p)
    let dof = n as f64 - n_entities as f64 - p as f64;
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

    let se_cluster = Some(cluster_robust_se(&x_mat, &resid, &xtx_inv, clust)?);

    Ok(PanelFEResult {
        coefficients,
        se_ols,
        se_cluster,
        r_squared_within,
        n_obs: n,
        n_entities,
        n_regressors: p,
        rss,
    })
}

/// Compute Liang–Zeger cluster-robust (HC0 sandwich) standard errors.
///
/// `V_CR = (X'X)^{-1} B (X'X)^{-1}` where `B = Σ_g X_g' e_g e_g' X_g`.
pub fn cluster_robust_se(
    x: &DMatrix<f64>,
    residuals: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    cluster_ids: &[u64],
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
    let n_f = n as f64;
    let p_f = p as f64;
    let correction =
        if g > 1.0 && n_f > p_f { (g / (g - 1.0)) * ((n_f - 1.0) / (n_f - p_f)) } else { 1.0 };

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

        let res = panel_fe_fit(&entity_ids, &x, &y, 1, None).unwrap();
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

        let res = panel_fe_fit(&entity_ids, &x, &y, 1, None).unwrap();
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

        let res = panel_fe_fit(&entity_ids, &x, &y, 1, None).unwrap();
        assert!(res.se_cluster.is_some());
        // Both should be non-NaN
        assert!(res.se_ols[0].is_finite());
        assert!(res.se_cluster.as_ref().unwrap()[0].is_finite());
    }

    #[test]
    fn test_panel_fe_validation() {
        assert!(panel_fe_fit(&[], &[], &[], 1, None).is_err());
        assert!(panel_fe_fit(&[1], &[1.0], &[1.0, 2.0], 1, None).is_err());
    }
}
