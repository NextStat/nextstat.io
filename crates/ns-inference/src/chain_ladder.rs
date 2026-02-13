//! Chain Ladder and Mack method for insurance loss reserving (Phase 9 Cross-Vertical).
//!
//! Implements deterministic and stochastic reserving methods for run-off triangles,
//! the cornerstone of actuarial loss reserving.
//!
//! ## Methods
//!
//! - **Chain Ladder**: deterministic method that estimates development factors from
//!   a cumulative claims triangle and projects ultimates for incomplete origin periods.
//!   Development factor `f_j = Σ C_{i,j+1} / Σ C_{i,j}` (volume-weighted average).
//!
//! - **Mack (1993)**: distribution-free stochastic model that attaches prediction
//!   standard errors to Chain Ladder reserves. Provides:
//!   - Per-origin and total IBNR (Incurred But Not Reported)
//!   - Process variance + estimation variance → prediction SE
//!   - Normal approximation prediction intervals
//!
//! ## Vertical applications
//!
//! - **Insurance / reinsurance**: IBNR reserves, reserving adequacy
//! - **Actuarial reporting**: Solvency II, IFRS 17 compliance
//!
//! ## References
//!
//! - Mack T (1993). Distribution-free calculation of the standard error of chain
//!   ladder reserve estimates. *ASTIN Bulletin* 23(2):213–225.
//! - Taylor GC, Ashe FR (1983). Second moments of estimates of outstanding claims.
//!   *J Econometrics* 23(1):37–61.

use ns_core::{Error, Result};

// ---------------------------------------------------------------------------
// Triangle representation
// ---------------------------------------------------------------------------

/// A cumulative claims triangle (upper-left triangle of an n×n matrix).
///
/// `triangle[i][j]` = cumulative claims for origin period `i` at development period `j`.
/// The triangle is "upper-left": for origin `i`, data is available for `j = 0..n-i-1`
/// (or up to the diagonal). `n` is the number of origin periods.
#[derive(Debug, Clone)]
pub struct ClaimsTriangle {
    /// Number of origin periods (rows).
    pub n: usize,
    /// Cumulative data: `data[i]` has length `n - i` (decreasing by row).
    pub data: Vec<Vec<f64>>,
}

impl ClaimsTriangle {
    /// Create a triangle from a vector of rows.
    ///
    /// Row `i` must have exactly `n - i` elements (the upper-left triangle shape).
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self> {
        let n = data.len();
        if n == 0 {
            return Err(Error::Validation(
                "triangle must have at least 1 origin period".to_string(),
            ));
        }
        for (i, row) in data.iter().enumerate() {
            let expected = n - i;
            if row.len() != expected {
                return Err(Error::Validation(format!(
                    "row {} has {} elements, expected {} (upper-left triangle)",
                    i,
                    row.len(),
                    expected
                )));
            }
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(Error::Validation(format!("triangle[{}][{}] is not finite", i, j)));
                }
                if v < 0.0 {
                    return Err(Error::Validation(format!(
                        "triangle[{}][{}] = {} is negative (cumulative claims must be >= 0)",
                        i, j, v
                    )));
                }
            }
        }
        Ok(Self { n, data })
    }

    /// Create a triangle from a full square matrix (upper-left values only).
    ///
    /// Values below the anti-diagonal are ignored.
    pub fn from_square(matrix: &[Vec<f64>]) -> Result<Self> {
        let n = matrix.len();
        if n == 0 {
            return Err(Error::Validation("matrix must be non-empty".to_string()));
        }
        let mut data = Vec::with_capacity(n);
        for (i, row) in matrix.iter().enumerate() {
            let cols = n - i;
            if row.len() < cols {
                return Err(Error::Validation(format!(
                    "row {} has {} columns, need at least {}",
                    i,
                    row.len(),
                    cols
                )));
            }
            data.push(row[..cols].to_vec());
        }
        Self::new(data)
    }

    /// Get a value from the triangle. Returns `None` if out of bounds.
    #[inline]
    pub fn get(&self, origin: usize, dev: usize) -> Option<f64> {
        self.data.get(origin).and_then(|row| row.get(dev).copied())
    }
}

// ---------------------------------------------------------------------------
// Chain Ladder result
// ---------------------------------------------------------------------------

/// Per-origin row in the Chain Ladder result.
#[derive(Debug, Clone)]
pub struct ChainLadderRow {
    /// Origin period index (0-based).
    pub origin: usize,
    /// Latest observed cumulative value (on the diagonal).
    pub latest: f64,
    /// Projected ultimate cumulative value.
    pub ultimate: f64,
    /// IBNR reserve (ultimate - latest).
    pub ibnr: f64,
}

/// Result of a Chain Ladder computation.
#[derive(Debug, Clone)]
pub struct ChainLadderResult {
    /// Development factors `f_j` for `j = 0..n-2`.
    pub development_factors: Vec<f64>,
    /// Cumulative development factors (tail to ultimate) for each dev period.
    /// `cum_factors[j]` = product of `f_k` for `k = j..n-2`.
    pub cumulative_factors: Vec<f64>,
    /// Per-origin results.
    pub rows: Vec<ChainLadderRow>,
    /// Total IBNR across all origins.
    pub total_ibnr: f64,
    /// Completed (projected) triangle: `projected[i]` has `n` elements.
    pub projected: Vec<Vec<f64>>,
}

/// Compute Chain Ladder development factors and project ultimates.
///
/// # Arguments
/// - `triangle`: a cumulative claims triangle.
///
/// # Returns
/// `ChainLadderResult` with development factors, ultimates, and IBNR.
pub fn chain_ladder(triangle: &ClaimsTriangle) -> Result<ChainLadderResult> {
    let n = triangle.n;
    if n < 2 {
        return Err(Error::Validation(
            "need at least 2 origin periods for chain ladder".to_string(),
        ));
    }

    // Development factors: f_j = Σ_{i: has col j+1} C_{i,j+1} / Σ_{i: has col j+1} C_{i,j}.
    let mut dev_factors = Vec::with_capacity(n - 1);
    for j in 0..(n - 1) {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        // Origin periods that have data at both j and j+1.
        for i in 0..n {
            if let (Some(c_j), Some(c_j1)) = (triangle.get(i, j), triangle.get(i, j + 1)) {
                num += c_j1;
                den += c_j;
            }
        }
        if den <= 0.0 {
            return Err(Error::Computation(format!(
                "development factor f_{} has zero denominator",
                j
            )));
        }
        dev_factors.push(num / den);
    }

    // Cumulative factors: cum_factors[j] = f_j * f_{j+1} * ... * f_{n-2}.
    let mut cum_factors = vec![1.0_f64; n];
    // cum_factors[n-1] = 1.0 (fully developed).
    for j in (0..(n - 1)).rev() {
        cum_factors[j] = dev_factors[j] * cum_factors[j + 1];
    }

    // Project the triangle and compute ultimates.
    let mut projected = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);
    let mut total_ibnr = 0.0_f64;

    for i in 0..n {
        let observed_len = triangle.data[i].len(); // n - i
        let latest = triangle.data[i][observed_len - 1];

        // Fill in projected values.
        let mut proj_row = Vec::with_capacity(n);
        for j in 0..observed_len {
            proj_row.push(triangle.data[i][j]);
        }
        // Project forward using dev factors.
        for j in observed_len..n {
            let prev = proj_row[j - 1];
            proj_row.push(prev * dev_factors[j - 1]);
        }

        let ultimate = proj_row[n - 1];
        let ibnr = ultimate - latest;
        total_ibnr += ibnr;

        rows.push(ChainLadderRow { origin: i, latest, ultimate, ibnr });
        projected.push(proj_row);
    }

    Ok(ChainLadderResult {
        development_factors: dev_factors,
        cumulative_factors: cum_factors,
        rows,
        total_ibnr,
        projected,
    })
}

// ---------------------------------------------------------------------------
// Mack model result
// ---------------------------------------------------------------------------

/// Per-origin Mack prediction result.
#[derive(Debug, Clone)]
pub struct MackRow {
    /// Origin period index.
    pub origin: usize,
    /// Latest observed cumulative value.
    pub latest: f64,
    /// Projected ultimate.
    pub ultimate: f64,
    /// IBNR reserve.
    pub ibnr: f64,
    /// Mack prediction standard error of the IBNR.
    pub se: f64,
    /// Coefficient of variation (SE / IBNR) if IBNR > 0.
    pub cv: f64,
    /// Lower prediction interval bound (normal approximation).
    pub pi_lower: f64,
    /// Upper prediction interval bound.
    pub pi_upper: f64,
}

/// Result of a Mack Chain Ladder computation.
#[derive(Debug, Clone)]
pub struct MackResult {
    /// Chain Ladder development factors.
    pub development_factors: Vec<f64>,
    /// Mack σ² estimates for each development period.
    pub sigma_sq: Vec<f64>,
    /// Per-origin results with prediction SE.
    pub rows: Vec<MackRow>,
    /// Total IBNR.
    pub total_ibnr: f64,
    /// Total Mack SE (accounting for correlation).
    pub total_se: f64,
    /// Confidence level used for prediction intervals.
    pub conf_level: f64,
}

/// Compute Mack Chain Ladder with prediction standard errors.
///
/// # Arguments
/// - `triangle`: a cumulative claims triangle.
/// - `conf_level`: confidence level for prediction intervals (e.g. 0.95).
///
/// # Returns
/// `MackResult` with per-origin SE, IBNR, and prediction intervals.
pub fn mack_chain_ladder(triangle: &ClaimsTriangle, conf_level: f64) -> Result<MackResult> {
    let n = triangle.n;
    if n < 3 {
        return Err(Error::Validation("Mack method needs at least 3 origin periods".to_string()));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation("conf_level must be in (0, 1)".to_string()));
    }

    // First get chain ladder results.
    let cl = chain_ladder(triangle)?;

    // Estimate σ²_j (Mack 1993, eq. 4):
    // σ²_j = (1/(n-j-2)) * Σ_{i=0}^{n-j-2} C_{i,j} * (C_{i,j+1}/C_{i,j} - f_j)²
    let mut sigma_sq = vec![0.0_f64; n - 1];
    for j in 0..(n - 1) {
        let mut s = 0.0_f64;
        let mut count = 0usize;
        for i in 0..n {
            if let (Some(c_j), Some(c_j1)) = (triangle.get(i, j), triangle.get(i, j + 1))
                && c_j > 0.0
            {
                let ratio = c_j1 / c_j;
                let diff = ratio - cl.development_factors[j];
                s += c_j * diff * diff;
                count += 1;
            }
        }
        if count > 1 {
            sigma_sq[j] = s / (count as f64 - 1.0);
        } else if count == 1 {
            // Mack's extrapolation for the last period: σ²_{n-2} = min(σ²_{n-3}, σ²_{n-3}²/σ²_{n-4}).
            // For interior periods with only 1 observation, use neighbor average.
            if j >= 2 && sigma_sq[j - 1] > 0.0 && sigma_sq[j - 2] > 0.0 {
                sigma_sq[j] = (sigma_sq[j - 1] * sigma_sq[j - 1] / sigma_sq[j - 2])
                    .min(sigma_sq[j - 1])
                    .min(sigma_sq[j - 2]);
            } else if j >= 1 {
                sigma_sq[j] = sigma_sq[j - 1];
            } else {
                sigma_sq[j] = s.max(1e-30);
            }
        }
    }
    // Handle last period extrapolation.
    let last = n - 2;
    if sigma_sq[last] == 0.0 && last >= 2 && sigma_sq[last - 1] > 0.0 && sigma_sq[last - 2] > 0.0 {
        sigma_sq[last] = (sigma_sq[last - 1] * sigma_sq[last - 1] / sigma_sq[last - 2])
            .min(sigma_sq[last - 1])
            .min(sigma_sq[last - 2]);
    }

    let z_alpha = normal_quantile(1.0 - (1.0 - conf_level) / 2.0);

    // Per-origin Mack SE (eq. 9 in Mack 1993):
    // MSE(R_i) = C_{i,n}² · Σ_{j=n-i}^{n-2} (σ²_j / f²_j) · (1/C_{i,j}^hat + 1/S_j)
    // where S_j = Σ_{k=0}^{n-j-2} C_{k,j}.
    //
    // S_j is the sum of column j for rows that have data at j.
    let mut s_col = vec![0.0_f64; n];
    for j in 0..n {
        for i in 0..n {
            if let Some(v) = triangle.get(i, j) {
                s_col[j] += v;
            }
        }
    }

    let mut mack_rows = Vec::with_capacity(n);
    let mut total_ibnr = 0.0_f64;

    for i in 0..n {
        let latest = cl.rows[i].latest;
        let ultimate = cl.rows[i].ultimate;
        let ibnr = cl.rows[i].ibnr;
        total_ibnr += ibnr;

        if ibnr.abs() < 1e-15 {
            // Fully developed: no prediction error.
            mack_rows.push(MackRow {
                origin: i,
                latest,
                ultimate,
                ibnr,
                se: 0.0,
                cv: 0.0,
                pi_lower: ultimate,
                pi_upper: ultimate,
            });
            continue;
        }

        let start_j = n - i - 1; // first projected development period
        let mut mse = 0.0_f64;

        for j in start_j..(n - 1) {
            let fj = cl.development_factors[j];
            if fj.abs() < 1e-30 {
                continue;
            }
            let c_ij_hat = cl.projected[i][j];
            if c_ij_hat <= 0.0 {
                continue;
            }
            let sj = s_col[j];
            let term =
                sigma_sq[j] / (fj * fj) * (1.0 / c_ij_hat + if sj > 0.0 { 1.0 / sj } else { 0.0 });
            mse += term;
        }

        let se = (ultimate * ultimate * mse).sqrt();
        let cv = if ibnr.abs() > 1e-15 { se / ibnr.abs() } else { 0.0 };
        let pi_lower = (ibnr - z_alpha * se).max(0.0);
        let pi_upper = ibnr + z_alpha * se;

        mack_rows.push(MackRow { origin: i, latest, ultimate, ibnr, se, cv, pi_lower, pi_upper });
    }

    // Total SE: correlated sum (Mack 1993, eq. 12).
    // Var(R_total) ≈ Σ_i Var(R_i) + 2·Σ_{i<k} C_{i,n}·C_{k,n}·Σ_j (σ²_j/f²_j)·(1/S_j)
    let mut var_total = 0.0_f64;
    for row in &mack_rows {
        var_total += row.se * row.se;
    }
    // Cross-terms.
    for i in 1..n {
        for k in (i + 1)..n {
            let ui = cl.rows[i].ultimate;
            let uk = cl.rows[k].ultimate;
            let start = (n - i - 1).max(n - k - 1);
            let mut cross = 0.0_f64;
            for j in start..(n - 1) {
                let fj = cl.development_factors[j];
                if fj.abs() < 1e-30 {
                    continue;
                }
                let sj = s_col[j];
                if sj > 0.0 {
                    cross += sigma_sq[j] / (fj * fj * sj);
                }
            }
            var_total += 2.0 * ui * uk * cross;
        }
    }
    let total_se = var_total.max(0.0).sqrt();

    Ok(MackResult {
        development_factors: cl.development_factors,
        sigma_sq,
        rows: mack_rows,
        total_ibnr,
        total_se,
        conf_level,
    })
}

// ---------------------------------------------------------------------------
// Bootstrap reserves (optional stochastic method)
// ---------------------------------------------------------------------------

/// Result of bootstrap reserve estimation.
#[derive(Debug, Clone)]
pub struct BootstrapReserveResult {
    /// Mean IBNR from bootstrap samples.
    pub mean_ibnr: f64,
    /// Standard deviation of IBNR across bootstrap samples.
    pub sd_ibnr: f64,
    /// Percentile-based prediction interval (lower).
    pub pi_lower: f64,
    /// Percentile-based prediction interval (upper).
    pub pi_upper: f64,
    /// All bootstrap IBNR samples (sorted).
    pub samples: Vec<f64>,
}

/// Bootstrap the total IBNR reserve using residual resampling.
///
/// # Arguments
/// - `triangle`: cumulative claims triangle.
/// - `n_boot`: number of bootstrap resamples.
/// - `conf_level`: confidence level for prediction intervals.
/// - `seed`: random seed for reproducibility.
pub fn bootstrap_reserves(
    triangle: &ClaimsTriangle,
    n_boot: usize,
    conf_level: f64,
    seed: u64,
) -> Result<BootstrapReserveResult> {
    if n_boot == 0 {
        return Err(Error::Validation("n_boot must be > 0".to_string()));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation("conf_level must be in (0, 1)".to_string()));
    }

    let n = triangle.n;
    let cl = chain_ladder(triangle)?;

    // Compute Pearson residuals from the Chain Ladder fit.
    let mut residuals = Vec::new();
    for i in 0..n {
        for j in 0..(triangle.data[i].len().saturating_sub(1)) {
            if let (Some(c_j), Some(c_j1)) = (triangle.get(i, j), triangle.get(i, j + 1)) {
                let expected = c_j * cl.development_factors[j];
                if expected > 0.0 {
                    let resid = (c_j1 - expected) / expected.sqrt();
                    if resid.is_finite() {
                        residuals.push(resid);
                    }
                }
            }
        }
    }

    if residuals.is_empty() {
        return Err(Error::Computation("no valid residuals for bootstrap".to_string()));
    }

    let n_resid = residuals.len();

    // Simple LCG-based RNG for reproducibility without external deps.
    let mut rng_state = seed;
    let mut next_u64 = || -> u64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng_state
    };
    let mut next_usize = |max: usize| -> usize { (next_u64() % (max as u64)) as usize };

    let mut samples = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        // Resample residuals and reconstruct a pseudo-triangle.
        let mut pseudo_data: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let row_len = triangle.data[i].len();
            let mut row = Vec::with_capacity(row_len);
            row.push(triangle.data[i][0]); // Keep the first column.
            for j in 1..row_len {
                let prev = row[j - 1];
                let expected = prev * cl.development_factors[j - 1];
                let r_idx = next_usize(n_resid);
                let boot_val = expected + residuals[r_idx] * expected.abs().sqrt();
                row.push(boot_val.max(0.0));
            }
            pseudo_data.push(row);
        }

        // Refit chain ladder on pseudo-triangle.
        if let Ok(pseudo_tri) = ClaimsTriangle::new(pseudo_data)
            && let Ok(pseudo_cl) = chain_ladder(&pseudo_tri)
        {
            samples.push(pseudo_cl.total_ibnr);
        }
    }

    if samples.is_empty() {
        return Err(Error::Computation("all bootstrap samples failed".to_string()));
    }

    samples.sort_by(|a, b| a.total_cmp(b));

    let mean_ibnr = samples.iter().sum::<f64>() / samples.len() as f64;
    let var = samples.iter().map(|&s| (s - mean_ibnr).powi(2)).sum::<f64>()
        / (samples.len() as f64 - 1.0).max(1.0);
    let sd_ibnr = var.sqrt();

    let lo_idx = ((1.0 - conf_level) / 2.0 * samples.len() as f64).floor() as usize;
    let hi_idx = ((1.0 - (1.0 - conf_level) / 2.0) * samples.len() as f64).ceil() as usize;
    let pi_lower = samples[lo_idx.min(samples.len() - 1)];
    let pi_upper = samples[hi_idx.min(samples.len() - 1)];

    Ok(BootstrapReserveResult { mean_ibnr, sd_ibnr, pi_lower, pi_upper, samples })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Standard normal quantile (inverse CDF).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let (sign, pp) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let t = (-2.0 * (1.0 - pp).ln()).sqrt();

    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    let num = C0 + t * (C1 + t * C2);
    let den = 1.0 + t * (D1 + t * (D2 + t * D3));
    sign * (t - num / den)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Taylor-Ashe triangle (classic actuarial test case).
    fn taylor_ashe_triangle() -> ClaimsTriangle {
        ClaimsTriangle::new(vec![
            vec![
                357848.0, 1124788.0, 1735330.0, 2218270.0, 2745596.0, 3319994.0, 3466336.0,
                3606286.0, 3833515.0, 3901463.0,
            ],
            vec![
                352118.0, 1236139.0, 2170033.0, 3353322.0, 3799067.0, 4120063.0, 4647867.0,
                4914039.0, 5339085.0,
            ],
            vec![
                290507.0, 1292306.0, 2218525.0, 3235179.0, 3985995.0, 4132918.0, 4628910.0,
                4909315.0,
            ],
            vec![310608.0, 1418858.0, 2195047.0, 3757447.0, 4029929.0, 4381982.0, 4588268.0],
            vec![443160.0, 1136350.0, 2128333.0, 2897821.0, 3402672.0, 3873311.0],
            vec![396132.0, 1333217.0, 2180715.0, 2985752.0, 3691712.0],
            vec![440832.0, 1288463.0, 2419861.0, 3483130.0],
            vec![359480.0, 1421128.0, 2864498.0],
            vec![376686.0, 1363294.0],
            vec![344014.0],
        ])
        .unwrap()
    }

    #[test]
    fn chain_ladder_taylor_ashe() {
        let tri = taylor_ashe_triangle();
        let cl = chain_ladder(&tri).unwrap();

        assert_eq!(cl.development_factors.len(), 9);
        assert_eq!(cl.rows.len(), 10);

        // First origin (fully developed): IBNR = 0.
        assert!((cl.rows[0].ibnr).abs() < 1e-6, "row 0 ibnr = {}", cl.rows[0].ibnr);

        // Last origin (only 1 data point): large IBNR.
        assert!(cl.rows[9].ibnr > 0.0, "row 9 ibnr = {}", cl.rows[9].ibnr);

        // Total IBNR should be positive.
        assert!(cl.total_ibnr > 0.0, "total ibnr = {}", cl.total_ibnr);

        // Dev factors should all be > 1 (cumulative claims grow).
        for (j, &f) in cl.development_factors.iter().enumerate() {
            assert!(f >= 1.0, "dev factor f_{} = {} should be >= 1", j, f);
        }

        // Projected triangle should be complete.
        for (i, row) in cl.projected.iter().enumerate() {
            assert_eq!(row.len(), 10, "projected row {} has {} cols", i, row.len());
        }
    }

    #[test]
    fn mack_taylor_ashe() {
        let tri = taylor_ashe_triangle();
        let mack = mack_chain_ladder(&tri, 0.95).unwrap();

        assert_eq!(mack.rows.len(), 10);
        assert_eq!(mack.sigma_sq.len(), 9);

        // First origin: SE = 0 (fully developed).
        assert!((mack.rows[0].se).abs() < 1e-6);

        // Later origins should have positive SE.
        for row in &mack.rows[2..] {
            assert!(row.se >= 0.0, "origin {} SE = {} should be >= 0", row.origin, row.se);
        }

        // Total SE should be positive.
        assert!(mack.total_se > 0.0, "total SE = {}", mack.total_se);

        // IBNR should match chain ladder.
        let cl = chain_ladder(&tri).unwrap();
        assert!(
            (mack.total_ibnr - cl.total_ibnr).abs() < 1e-6,
            "Mack total IBNR {} != CL total IBNR {}",
            mack.total_ibnr,
            cl.total_ibnr
        );
    }

    #[test]
    fn mack_prediction_intervals() {
        let tri = taylor_ashe_triangle();
        let mack = mack_chain_ladder(&tri, 0.95).unwrap();

        for row in &mack.rows {
            if row.ibnr > 0.0 {
                assert!(
                    row.pi_lower <= row.ibnr,
                    "PI lower {} > IBNR {} for origin {}",
                    row.pi_lower,
                    row.ibnr,
                    row.origin
                );
                assert!(
                    row.pi_upper >= row.ibnr,
                    "PI upper {} < IBNR {} for origin {}",
                    row.pi_upper,
                    row.ibnr,
                    row.origin
                );
            }
        }
    }

    #[test]
    fn bootstrap_reserves_smoke() {
        let tri = taylor_ashe_triangle();
        let boot = bootstrap_reserves(&tri, 500, 0.95, 42).unwrap();
        assert!(boot.mean_ibnr > 0.0);
        assert!(boot.sd_ibnr > 0.0);
        assert!(boot.pi_lower < boot.pi_upper);
        assert_eq!(boot.samples.len(), 500);
        // Bootstrap mean should be in the same ballpark as deterministic.
        let cl = chain_ladder(&tri).unwrap();
        let ratio = boot.mean_ibnr / cl.total_ibnr;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "bootstrap mean {} vs CL {}: ratio = {}",
            boot.mean_ibnr,
            cl.total_ibnr,
            ratio
        );
    }

    #[test]
    fn triangle_validation() {
        // Empty triangle.
        assert!(ClaimsTriangle::new(vec![]).is_err());
        // Wrong row length.
        assert!(ClaimsTriangle::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).is_err());
        // Negative value.
        assert!(ClaimsTriangle::new(vec![vec![-1.0]]).is_err());
        // NaN.
        assert!(ClaimsTriangle::new(vec![vec![f64::NAN]]).is_err());
    }

    #[test]
    fn triangle_from_square() {
        let matrix =
            vec![vec![100.0, 200.0, 300.0], vec![110.0, 220.0, 999.0], vec![120.0, 999.0, 999.0]];
        let tri = ClaimsTriangle::from_square(&matrix).unwrap();
        assert_eq!(tri.n, 3);
        assert_eq!(tri.data[0], vec![100.0, 200.0, 300.0]);
        assert_eq!(tri.data[1], vec![110.0, 220.0]);
        assert_eq!(tri.data[2], vec![120.0]);
    }

    #[test]
    fn small_triangle_chain_ladder() {
        let tri =
            ClaimsTriangle::new(vec![vec![100.0, 200.0, 300.0], vec![110.0, 220.0], vec![120.0]])
                .unwrap();
        let cl = chain_ladder(&tri).unwrap();
        assert_eq!(cl.development_factors.len(), 2);
        // f_0 = (200 + 220) / (100 + 110) = 420/210 = 2.0
        assert!((cl.development_factors[0] - 2.0).abs() < 1e-10);
        // f_1 = 300 / 200 = 1.5
        assert!((cl.development_factors[1] - 1.5).abs() < 1e-10);
        // Ultimate for row 2: 120 * 2.0 * 1.5 = 360
        assert!((cl.rows[2].ultimate - 360.0).abs() < 1e-6);
    }
}
