//! Instrumental Variables / Two-Stage Least Squares (2SLS).
//!
//! Implements the standard 2SLS estimator with first-stage F-statistic and
//! weak-instrument diagnostics (Stock–Yogo critical values).
//!
//! # References
//!
//! - Wooldridge, *Econometric Analysis of Cross Section and Panel Data*, Ch. 5.
//! - Stock & Yogo (2005), "Testing for weak instruments in linear IV regression."

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

use super::panel::cluster_robust_se;

/// First-stage regression diagnostics.
#[derive(Debug, Clone)]
pub struct FirstStageResult {
    /// First-stage F-statistic (joint significance of excluded instruments).
    pub f_stat: f64,
    /// First-stage R².
    pub r_squared: f64,
    /// Partial R² of excluded instruments.
    pub partial_r_squared: f64,
    /// Whether the instrument passes the Stock–Yogo 10% maximal IV size
    /// critical value (F > 16.38 for single endogenous regressor, single instrument).
    pub passes_stock_yogo_10: bool,
}

/// Result of a 2SLS regression.
#[derive(Debug, Clone)]
pub struct IvResult {
    /// Second-stage coefficient estimates.
    pub coefficients: Vec<f64>,
    /// Parameter names.
    pub names: Vec<String>,
    /// Standard errors (2SLS, homoskedastic).
    pub se: Vec<f64>,
    /// Cluster-robust standard errors (if cluster_ids provided).
    pub se_cluster: Option<Vec<f64>>,
    /// First-stage diagnostics (one per endogenous regressor).
    pub first_stage: Vec<FirstStageResult>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of instruments (excluded).
    pub n_instruments: usize,
}

/// Two-Stage Least Squares (2SLS) estimator.
///
/// # Arguments
///
/// - `y` — dependent variable (length n).
/// - `x_exog` — exogenous regressors, row-major (n × k₁). Include intercept column if desired.
/// - `k_exog` — number of exogenous columns.
/// - `x_endog` — endogenous regressors, row-major (n × k₂).
/// - `k_endog` — number of endogenous columns.
/// - `z` — excluded instruments, row-major (n × m). Must have m ≥ k₂.
/// - `m` — number of excluded instrument columns.
/// - `endog_names` — names for endogenous regressors.
/// - `exog_names` — names for exogenous regressors.
/// - `cluster_ids` — optional clustering variable for robust SE.
pub fn iv_2sls(
    y: &[f64],
    x_exog: &[f64],
    k_exog: usize,
    x_endog: &[f64],
    k_endog: usize,
    z: &[f64],
    m: usize,
    exog_names: &[String],
    endog_names: &[String],
    cluster_ids: Option<&[u64]>,
) -> Result<IvResult> {
    let n = y.len();
    if n == 0 {
        return Err(Error::Validation("y must be non-empty".into()));
    }
    if x_exog.len() != n * k_exog {
        return Err(Error::Validation(format!(
            "x_exog length ({}) != n*k_exog ({})",
            x_exog.len(),
            n * k_exog
        )));
    }
    if x_endog.len() != n * k_endog {
        return Err(Error::Validation(format!(
            "x_endog length ({}) != n*k_endog ({})",
            x_endog.len(),
            n * k_endog
        )));
    }
    if z.len() != n * m {
        return Err(Error::Validation(format!("z length ({}) != n*m ({})", z.len(), n * m)));
    }
    if k_endog == 0 {
        return Err(Error::Validation("Must have at least 1 endogenous regressor".into()));
    }
    if m < k_endog {
        return Err(Error::Validation(format!(
            "Under-identified: {} instruments < {} endogenous regressors",
            m, k_endog
        )));
    }

    let y_vec = DVector::from_column_slice(y);

    // Build full instrument matrix: [X_exog | Z] (n × (k₁ + m))
    let k_full_z = k_exog + m;
    let mut z_full_data = Vec::with_capacity(n * k_full_z);
    for i in 0..n {
        for j in 0..k_exog {
            z_full_data.push(x_exog[i * k_exog + j]);
        }
        for j in 0..m {
            z_full_data.push(z[i * m + j]);
        }
    }
    let z_full = DMatrix::from_row_slice(n, k_full_z, &z_full_data);

    // Projection matrix: P_Z = Z (Z'Z)^{-1} Z'
    let ztz = z_full.transpose() * &z_full;
    let ztz_inv =
        ztz.try_inverse().ok_or_else(|| Error::Computation("Z'Z singular in 2SLS".into()))?;
    let pz = &z_full * &ztz_inv * z_full.transpose();

    // ---- First stage: regress each endogenous var on Z_full ----
    let mut first_stage_results = Vec::with_capacity(k_endog);
    let x_exog_mat =
        if k_exog > 0 { Some(DMatrix::from_row_slice(n, k_exog, x_exog)) } else { None };

    for e in 0..k_endog {
        let mut endog_col = Vec::with_capacity(n);
        for i in 0..n {
            endog_col.push(x_endog[i * k_endog + e]);
        }
        let endog_vec = DVector::from_column_slice(&endog_col);

        // Full first-stage regression
        let zty_fs = z_full.transpose() * &endog_vec;
        let gamma = &ztz_inv * &zty_fs;
        let fitted = &z_full * &gamma;
        let resid_fs = &endog_vec - &fitted;

        let tss_fs: f64 = {
            let mean = endog_col.iter().sum::<f64>() / n as f64;
            endog_col.iter().map(|&v| (v - mean).powi(2)).sum()
        };
        let rss_fs: f64 = resid_fs.iter().map(|r| r * r).sum();
        let r_squared = if tss_fs > 0.0 { 1.0 - rss_fs / tss_fs } else { 0.0 };

        // Partial F-stat: test joint significance of excluded instruments
        // F = ((RSS_restricted - RSS_unrestricted) / m) / (RSS_unrestricted / (n - k_full_z))
        let rss_restricted = if let Some(ref x_exog_m) = x_exog_mat {
            let xtx_r = x_exog_m.transpose() * x_exog_m;
            let xty_r = x_exog_m.transpose() * &endog_vec;
            if let Some(inv) = xtx_r.try_inverse() {
                let beta_r = &inv * &xty_r;
                let fitted_r = x_exog_m * &beta_r;
                let resid_r = &endog_vec - &fitted_r;
                resid_r.iter().map(|r| r * r).sum()
            } else {
                tss_fs // fallback: restricted model is just the mean
            }
        } else {
            tss_fs
        };

        let f_stat = if rss_fs > 0.0 && n > k_full_z {
            ((rss_restricted - rss_fs) / m as f64) / (rss_fs / (n - k_full_z) as f64)
        } else {
            f64::NAN
        };

        let partial_r_squared =
            if rss_restricted > 0.0 { (rss_restricted - rss_fs) / rss_restricted } else { 0.0 };

        // Stock–Yogo 10% critical value for single endogenous regressor
        // With 1 endog and m instruments: CV₁₀ ≈ 16.38 (m=1), 19.93 (m=2), etc.
        // Simplified: use 16.38 for m=1, 10.0 as conservative threshold otherwise
        let stock_yogo_cv = if k_endog == 1 && m == 1 {
            16.38
        } else if k_endog == 1 && m == 2 {
            19.93
        } else {
            10.0 // conservative
        };

        first_stage_results.push(FirstStageResult {
            f_stat,
            r_squared,
            partial_r_squared,
            passes_stock_yogo_10: f_stat > stock_yogo_cv,
        });
    }

    // ---- Second stage: regress y on [X_exog | X̂_endog] ----
    // X̂_endog = P_Z × X_endog
    let x_endog_mat = DMatrix::from_row_slice(n, k_endog, x_endog);
    let x_endog_hat = &pz * &x_endog_mat;

    // Build second-stage design: [X_exog | X̂_endog]
    let k_total = k_exog + k_endog;
    let mut x2_data = Vec::with_capacity(n * k_total);
    for i in 0..n {
        for j in 0..k_exog {
            x2_data.push(x_exog[i * k_exog + j]);
        }
        for j in 0..k_endog {
            x2_data.push(x_endog_hat[(i, j)]);
        }
    }
    let x2 = DMatrix::from_row_slice(n, k_total, &x2_data);

    let xtx2 = x2.transpose() * &x2;
    let xty2 = x2.transpose() * &y_vec;
    let xtx2_inv = xtx2
        .try_inverse()
        .ok_or_else(|| Error::Computation("X'X singular in 2SLS second stage".into()))?;
    let beta2 = &xtx2_inv * &xty2;
    let coefficients: Vec<f64> = beta2.iter().copied().collect();

    // Residuals using ORIGINAL X_endog (not fitted values)
    let mut x_orig_data = Vec::with_capacity(n * k_total);
    for i in 0..n {
        for j in 0..k_exog {
            x_orig_data.push(x_exog[i * k_exog + j]);
        }
        for j in 0..k_endog {
            x_orig_data.push(x_endog[i * k_endog + j]);
        }
    }
    let x_orig = DMatrix::from_row_slice(n, k_total, &x_orig_data);
    let resid = &y_vec - &x_orig * &beta2;
    let rss: f64 = resid.iter().map(|r| r * r).sum();

    // 2SLS SE (using residuals from original regressors, variance from projected X)
    let dof = n as f64 - k_total as f64;
    let sigma2 = if dof > 0.0 { rss / dof } else { f64::NAN };
    let se: Vec<f64> = (0..k_total).map(|j| (sigma2 * xtx2_inv[(j, j)]).max(0.0).sqrt()).collect();

    // Cluster-robust SE
    let se_cluster = if let Some(clust) = cluster_ids {
        Some(cluster_robust_se(&x2, &resid, &xtx2_inv, clust, 0)?)
    } else {
        None
    };

    // Build names
    let mut names = Vec::with_capacity(k_total);
    if exog_names.len() == k_exog {
        names.extend_from_slice(exog_names);
    } else {
        for j in 0..k_exog {
            names.push(format!("exog_{}", j));
        }
    }
    if endog_names.len() == k_endog {
        names.extend_from_slice(endog_names);
    } else {
        for j in 0..k_endog {
            names.push(format!("endog_{}", j));
        }
    }

    Ok(IvResult {
        coefficients,
        names,
        se,
        se_cluster,
        first_stage: first_stage_results,
        n_obs: n,
        n_instruments: m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_2sls_exact_identification() {
        // True model: y = 1 + 2*x_endog + ε
        // x_endog is correlated with ε, but instrument z is valid
        // Use a simple deterministic example where IV recovers the true coefficient
        let n = 100;
        let mut y = Vec::with_capacity(n);
        let mut x_exog = Vec::with_capacity(n); // intercept
        let mut x_endog = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);

        for i in 0..n {
            let zi = (i as f64) / 10.0;
            // x_endog = 0.5 * z + 0 (no endogeneity in this deterministic case)
            let xi = 0.5 * zi;
            // y = 1 + 2*x + 0 (no error)
            let yi = 1.0 + 2.0 * xi;
            x_exog.push(1.0); // intercept
            x_endog.push(xi);
            z.push(zi);
            y.push(yi);
        }

        let res = iv_2sls(
            &y,
            &x_exog,
            1,
            &x_endog,
            1,
            &z,
            1,
            &["const".to_string()],
            &["x".to_string()],
            None,
        )
        .unwrap();

        assert_eq!(res.n_obs, n);
        assert_eq!(res.n_instruments, 1);
        assert_eq!(res.coefficients.len(), 2);
        assert!(
            (res.coefficients[0] - 1.0).abs() < 1e-6,
            "intercept={}, expected 1.0",
            res.coefficients[0]
        );
        assert!(
            (res.coefficients[1] - 2.0).abs() < 1e-6,
            "beta={}, expected 2.0",
            res.coefficients[1]
        );
        assert!(res.first_stage[0].f_stat > 0.0);
        assert!(res.first_stage[0].r_squared > 0.99);
    }

    #[test]
    fn test_iv_under_identified() {
        // 2 endogenous, 1 instrument → should fail
        let y = vec![1.0, 2.0, 3.0];
        let x_exog = vec![1.0, 1.0, 1.0];
        let x_endog = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]; // n=3, k=2
        let z = vec![0.5, 1.0, 1.5]; // m=1

        let res = iv_2sls(&y, &x_exog, 1, &x_endog, 2, &z, 1, &[], &[], None);
        assert!(res.is_err());
    }

    #[test]
    fn test_iv_validation() {
        assert!(iv_2sls(&[], &[], 0, &[], 1, &[], 1, &[], &[], None).is_err());
    }

    #[test]
    fn test_iv_with_cluster_se() {
        let n = 60;
        let mut y = Vec::with_capacity(n);
        let mut x_exog = Vec::with_capacity(n);
        let mut x_endog = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut cluster = Vec::with_capacity(n);

        for i in 0..n {
            let zi = (i as f64) / 10.0;
            let xi = 0.5 * zi + 0.1;
            let yi = 3.0 + 1.5 * xi;
            x_exog.push(1.0);
            x_endog.push(xi);
            z.push(zi);
            y.push(yi);
            cluster.push((i / 10) as u64);
        }

        let res = iv_2sls(
            &y,
            &x_exog,
            1,
            &x_endog,
            1,
            &z,
            1,
            &["const".into()],
            &["x".into()],
            Some(&cluster),
        )
        .unwrap();

        assert!(res.se_cluster.is_some());
        let se_c = res.se_cluster.unwrap();
        assert!(se_c[0].is_finite());
        assert!(se_c[1].is_finite());
    }
}
