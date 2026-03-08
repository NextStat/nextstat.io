//! Shared CUPED/CURE variance-reduction primitives.
//!
//! Architectural rule: CUPED is treated as the one-covariate case of the
//! general CURE regression-adjustment layer.

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

const EPS: f64 = 1e-12;
const MAX_ACCEPTABLE_CONDITION_NUMBER: f64 = 1e8;
const RIDGE_BASE_SCALE: f64 = 1e-8;
const RIDGE_MAX_STEPS: usize = 6;

fn default_pre_treatment_only() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarianceReductionMethod {
    None,
    Cuped,
    Cure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarianceReductionSolver {
    Svd,
    Ridge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovariateTiming {
    PreTreatment,
    Unknown,
    PostTreatment,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovariateProvenance {
    pub name: String,
    pub timing: CovariateTiming,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_dataset: Option<String>,
}

/// Input data for one-arm CUPED adjustment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmData {
    /// Post-treatment outcome values.
    pub outcomes: Vec<f64>,
    /// Single pre-treatment covariate aligned with `outcomes`.
    pub covariates: Vec<f64>,
    /// Optional human-readable covariate name for artifact/log surfaces.
    #[serde(default)]
    pub covariate_name: Option<String>,
    /// Optional typed provenance metadata for fail-fast leakage validation.
    #[serde(default)]
    pub covariate_provenance: Option<CovariateProvenance>,
    /// Must remain true: CUPED/CURE only support pre-treatment covariates.
    #[serde(default = "default_pre_treatment_only")]
    pub pre_treatment_only: bool,
}

/// Input data for one-arm CURE adjustment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiCovariateArmData {
    /// Post-treatment outcome values.
    pub outcomes: Vec<f64>,
    /// Row-major pre-treatment covariate matrix.
    pub covariates: Vec<Vec<f64>>,
    /// Optional covariate names for logging/artifact surfaces.
    #[serde(default)]
    pub covariate_names: Vec<String>,
    /// Optional typed provenance metadata for fail-fast leakage validation.
    #[serde(default)]
    pub covariate_provenance: Vec<CovariateProvenance>,
    /// Must remain true: CURE only supports pre-treatment covariates.
    #[serde(default = "default_pre_treatment_only")]
    pub pre_treatment_only: bool,
}

/// Result of CUPED adjustment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CupedResult {
    pub method: VarianceReductionMethod,
    pub mean_control: f64,
    pub mean_variant: f64,
    pub adjusted_mean_control: f64,
    pub adjusted_mean_variant: f64,
    pub effect: f64,
    pub theta: f64,
    pub rho: f64,
    pub r_squared: f64,
    pub variance_reduction_factor: f64,
    pub original_variance: f64,
    pub adjusted_variance: f64,
    pub effective_sample_multiplier: f64,
    pub num_covariates: usize,
    pub selected_covariates: Vec<String>,
    pub covariate_provenance: Vec<CovariateProvenance>,
    pub provenance_validated: bool,
    pub solver: VarianceReductionSolver,
    pub regression_rank: usize,
    pub condition_number: Option<f64>,
    pub ridge_lambda: Option<f64>,
    pub pre_treatment_only: bool,
}

/// Result of CURE adjustment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CureResult {
    pub method: VarianceReductionMethod,
    pub mean_control: f64,
    pub mean_variant: f64,
    pub adjusted_mean_control: f64,
    pub adjusted_mean_variant: f64,
    pub effect: f64,
    /// OLS coefficients, one per covariate.
    pub theta: Vec<f64>,
    pub r_squared: f64,
    pub variance_reduction_factor: f64,
    pub original_variance: f64,
    pub adjusted_variance: f64,
    pub effective_sample_multiplier: f64,
    pub num_covariates: usize,
    pub selected_covariates: Vec<String>,
    pub covariate_provenance: Vec<CovariateProvenance>,
    pub provenance_validated: bool,
    pub solver: VarianceReductionSolver,
    pub regression_rank: usize,
    pub condition_number: Option<f64>,
    pub ridge_lambda: Option<f64>,
    pub pre_treatment_only: bool,
}

struct RegressionSolveDiagnostics {
    theta: DVector<f64>,
    solver: VarianceReductionSolver,
    regression_rank: usize,
    condition_number: Option<f64>,
    ridge_lambda: Option<f64>,
}

/// CUPED adjustment with an explicit pre-treatment covariate.
pub fn cuped_adjust(control: &ArmData, variant: &ArmData) -> Result<CupedResult> {
    validate_pre_treatment_only(control.pre_treatment_only, variant.pre_treatment_only)?;
    validate_single_covariate_arm("control", control)?;
    validate_single_covariate_arm("variant", variant)?;

    let control_multi = MultiCovariateArmData {
        outcomes: control.outcomes.clone(),
        covariates: control.covariates.iter().map(|&x| vec![x]).collect(),
        covariate_names: normalize_single_covariate_names(
            control.covariate_name.clone(),
            variant.covariate_name.clone(),
        )?,
        covariate_provenance: normalize_single_covariate_provenance(control, variant)?
            .into_iter()
            .collect(),
        pre_treatment_only: true,
    };
    let variant_multi = MultiCovariateArmData {
        outcomes: variant.outcomes.clone(),
        covariates: variant.covariates.iter().map(|&x| vec![x]).collect(),
        covariate_names: control_multi.covariate_names.clone(),
        covariate_provenance: control_multi.covariate_provenance.clone(),
        pre_treatment_only: true,
    };

    let cure = cure_adjust(&control_multi, &variant_multi)?;

    let all_y: Vec<f64> = control.outcomes.iter().chain(variant.outcomes.iter()).copied().collect();
    let all_x: Vec<f64> =
        control.covariates.iter().chain(variant.covariates.iter()).copied().collect();
    let mean_y = mean(&all_y)?;
    let mean_x = mean(&all_x)?;
    let cov_xy = sample_covariance(&all_y, &all_x, mean_y, mean_x)?;
    let var_x = sample_variance(&all_x, mean_x)?;
    let var_y = sample_variance(&all_y, mean_y)?;
    let rho = if var_x > EPS && var_y > EPS {
        (cov_xy / (var_y.sqrt() * var_x.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Ok(CupedResult {
        method: VarianceReductionMethod::Cuped,
        mean_control: cure.mean_control,
        mean_variant: cure.mean_variant,
        adjusted_mean_control: cure.adjusted_mean_control,
        adjusted_mean_variant: cure.adjusted_mean_variant,
        effect: cure.effect,
        theta: cure.theta[0],
        rho,
        r_squared: cure.r_squared,
        variance_reduction_factor: cure.variance_reduction_factor,
        original_variance: cure.original_variance,
        adjusted_variance: cure.adjusted_variance,
        effective_sample_multiplier: cure.effective_sample_multiplier,
        num_covariates: 1,
        selected_covariates: cure.selected_covariates,
        covariate_provenance: cure.covariate_provenance,
        provenance_validated: cure.provenance_validated,
        solver: cure.solver,
        regression_rank: cure.regression_rank,
        condition_number: cure.condition_number,
        ridge_lambda: cure.ridge_lambda,
        pre_treatment_only: true,
    })
}

/// CURE adjustment using multiple pre-treatment covariates.
pub fn cure_adjust(
    control: &MultiCovariateArmData,
    variant: &MultiCovariateArmData,
) -> Result<CureResult> {
    validate_pre_treatment_only(control.pre_treatment_only, variant.pre_treatment_only)?;
    validate_multi_covariate_arm("control", control)?;
    validate_multi_covariate_arm("variant", variant)?;

    let p = control
        .covariates
        .first()
        .map(|row| row.len())
        .ok_or_else(|| Error::Validation("control covariates must be non-empty".to_string()))?;
    if p == 0 {
        return Err(Error::Validation(
            "CURE requires at least one pre-treatment covariate".to_string(),
        ));
    }

    for row in &variant.covariates {
        if row.len() != p {
            return Err(Error::Validation(format!(
                "variant covariate row has {} columns, expected {}",
                row.len(),
                p
            )));
        }
    }

    let covariate_provenance = normalize_multi_covariate_provenance(control, variant, p)?;
    let selected_covariates = if !covariate_provenance.is_empty() {
        covariate_provenance.iter().map(|item| item.name.clone()).collect()
    } else {
        normalize_multi_covariate_names(&control.covariate_names, &variant.covariate_names, p)?
    };
    let provenance_validated = covariate_provenance.len() == p;

    let n_c = control.outcomes.len();
    let n_v = variant.outcomes.len();
    let n = n_c + n_v;
    if n <= p {
        return Err(Error::Validation(format!(
            "CURE needs pooled observations > covariates, got n={} and p={}",
            n, p
        )));
    }

    let all_y: Vec<f64> = control.outcomes.iter().chain(variant.outcomes.iter()).copied().collect();
    let all_covariates: Vec<&Vec<f64>> =
        control.covariates.iter().chain(variant.covariates.iter()).collect();

    let mean_y = mean(&all_y)?;
    let mut mean_x = vec![0.0; p];
    for row in &all_covariates {
        for (j, value) in row.iter().enumerate() {
            mean_x[j] += *value;
        }
    }
    for value in &mut mean_x {
        *value /= n as f64;
    }

    let y_centered = DVector::from_iterator(n, all_y.iter().map(|y| *y - mean_y));
    let x_centered = DMatrix::from_fn(n, p, |row, col| all_covariates[row][col] - mean_x[col]);

    let solve = solve_regression_with_guardrails(&x_centered, &y_centered)?;
    let theta = solve.theta;
    let fitted = &x_centered * &theta;
    let residual = &y_centered - fitted;
    let ss_total = y_centered.dot(&y_centered);
    let ss_residual = residual.dot(&residual);
    let r_squared =
        if ss_total > EPS { (1.0 - ss_residual / ss_total).clamp(0.0, 1.0) } else { 0.0 };
    let variance_reduction_factor = (1.0 - r_squared).clamp(0.0, 1.0);
    let effective_sample_multiplier = if variance_reduction_factor > EPS {
        1.0 / variance_reduction_factor
    } else {
        f64::INFINITY
    };

    let mean_control = mean(&control.outcomes)?;
    let mean_variant = mean(&variant.outcomes)?;
    let mean_x_control = column_means(&control.covariates, p)?;
    let mean_x_variant = column_means(&variant.covariates, p)?;

    let mut adjusted_mean_control = mean_control;
    let mut adjusted_mean_variant = mean_variant;
    for j in 0..p {
        adjusted_mean_control -= theta[j] * (mean_x_control[j] - mean_x[j]);
        adjusted_mean_variant -= theta[j] * (mean_x_variant[j] - mean_x[j]);
    }

    let effect = adjusted_mean_variant - adjusted_mean_control;
    let original_variance = sample_variance(&control.outcomes, mean_control)? / n_c as f64
        + sample_variance(&variant.outcomes, mean_variant)? / n_v as f64;
    let adjusted_variance = original_variance * variance_reduction_factor;

    Ok(CureResult {
        method: if p == 1 { VarianceReductionMethod::Cuped } else { VarianceReductionMethod::Cure },
        mean_control,
        mean_variant,
        adjusted_mean_control,
        adjusted_mean_variant,
        effect,
        theta: theta.iter().copied().collect(),
        r_squared,
        variance_reduction_factor,
        original_variance,
        adjusted_variance,
        effective_sample_multiplier,
        num_covariates: p,
        selected_covariates,
        covariate_provenance,
        provenance_validated,
        solver: solve.solver,
        regression_rank: solve.regression_rank,
        condition_number: solve.condition_number,
        ridge_lambda: solve.ridge_lambda,
        pre_treatment_only: true,
    })
}

fn solve_regression_with_guardrails(
    x_centered: &DMatrix<f64>,
    y_centered: &DVector<f64>,
) -> Result<RegressionSolveDiagnostics> {
    let svd = x_centered.clone().svd(true, true);
    let singular_values = &svd.singular_values;
    let sigma_max = singular_values.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    if sigma_max <= EPS {
        return Err(Error::Validation(
            "pre-treatment covariates have zero pooled variance".to_string(),
        ));
    }
    let tol =
        ((x_centered.nrows().max(x_centered.ncols()) as f64) * f64::EPSILON * sigma_max * 16.0)
            .max(EPS);
    let regression_rank = singular_values.iter().filter(|&&s| s > tol).count();
    let sigma_min =
        singular_values.iter().copied().filter(|s| *s > tol).fold(f64::INFINITY, f64::min);
    let condition_number =
        if sigma_min.is_finite() && sigma_min > 0.0 { Some(sigma_max / sigma_min) } else { None };

    let should_use_ridge = regression_rank < x_centered.ncols()
        || condition_number.map(|cond| cond > MAX_ACCEPTABLE_CONDITION_NUMBER).unwrap_or(true);

    if !should_use_ridge {
        let theta = svd
            .solve(y_centered, tol)
            .map_err(|err| Error::Computation(format!("SVD solve failed for CURE: {err}")))?;
        if theta.iter().all(|value| value.is_finite()) {
            return Ok(RegressionSolveDiagnostics {
                theta,
                solver: VarianceReductionSolver::Svd,
                regression_rank,
                condition_number,
                ridge_lambda: None,
            });
        }
    }

    let xtx = x_centered.transpose() * x_centered;
    let xty = x_centered.transpose() * y_centered;
    let mut ridge_lambda = (sigma_max * sigma_max).max(1.0) * RIDGE_BASE_SCALE;
    for _ in 0..RIDGE_MAX_STEPS {
        let mut regularized = xtx.clone();
        for i in 0..regularized.nrows() {
            regularized[(i, i)] += ridge_lambda;
        }
        if let Some(cholesky) = regularized.cholesky() {
            let theta = cholesky.solve(&xty);
            if theta.iter().all(|value| value.is_finite()) {
                return Ok(RegressionSolveDiagnostics {
                    theta,
                    solver: VarianceReductionSolver::Ridge,
                    regression_rank,
                    condition_number,
                    ridge_lambda: Some(ridge_lambda),
                });
            }
        }
        ridge_lambda *= 10.0;
    }

    Err(Error::Computation("failed to solve CURE regression after ridge fallback".to_string()))
}

fn validate_pre_treatment_only(control: bool, variant: bool) -> Result<()> {
    if control && variant {
        Ok(())
    } else {
        Err(Error::Validation("CUPED/CURE covariates must be pre-treatment only".to_string()))
    }
}

fn validate_single_covariate_arm(label: &str, arm: &ArmData) -> Result<()> {
    if arm.outcomes.len() != arm.covariates.len() {
        return Err(Error::Validation(format!(
            "{label} outcomes/covariates length mismatch: {} vs {}",
            arm.outcomes.len(),
            arm.covariates.len()
        )));
    }
    if arm.outcomes.len() < 2 {
        return Err(Error::Validation(format!(
            "{label} arm needs at least 2 observations for CUPED"
        )));
    }
    validate_finite_slice(&arm.outcomes, &format!("{label} outcomes"))?;
    validate_finite_slice(&arm.covariates, &format!("{label} covariates"))?;
    Ok(())
}

fn validate_multi_covariate_arm(label: &str, arm: &MultiCovariateArmData) -> Result<()> {
    if arm.outcomes.len() != arm.covariates.len() {
        return Err(Error::Validation(format!(
            "{label} outcomes/covariates length mismatch: {} vs {}",
            arm.outcomes.len(),
            arm.covariates.len()
        )));
    }
    if arm.outcomes.len() < 2 {
        return Err(Error::Validation(format!(
            "{label} arm needs at least 2 observations for CURE"
        )));
    }
    validate_finite_slice(&arm.outcomes, &format!("{label} outcomes"))?;
    for (row_idx, row) in arm.covariates.iter().enumerate() {
        if row.is_empty() {
            return Err(Error::Validation(format!(
                "{label} covariate row {row_idx} must be non-empty"
            )));
        }
        validate_finite_slice(row, &format!("{label} covariate row {row_idx}"))?;
    }
    Ok(())
}

fn validate_finite_slice(values: &[f64], label: &str) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::Validation(format!("{label} must contain only finite values")))
    }
}

fn normalize_single_covariate_names(
    control_name: Option<String>,
    variant_name: Option<String>,
) -> Result<Vec<String>> {
    match (control_name, variant_name) {
        (Some(control), Some(variant)) if control != variant => Err(Error::Validation(
            "control/variant covariate names must match for CUPED".to_string(),
        )),
        (Some(name), _) | (_, Some(name)) => Ok(vec![name]),
        (None, None) => Ok(Vec::new()),
    }
}

fn normalize_single_covariate_provenance(
    control: &ArmData,
    variant: &ArmData,
) -> Result<Vec<CovariateProvenance>> {
    let provenance = match (&control.covariate_provenance, &variant.covariate_provenance) {
        (Some(control_item), Some(variant_item)) if control_item != variant_item => {
            return Err(Error::Validation(
                "control/variant covariate provenance must match for CUPED".to_string(),
            ));
        }
        (Some(item), _) | (_, Some(item)) => Some(item.clone()),
        (None, None) => None,
    };

    let name = normalize_single_covariate_names(
        control.covariate_name.clone(),
        variant.covariate_name.clone(),
    )?
    .into_iter()
    .next();

    let item = match provenance {
        Some(item) => {
            validate_covariate_provenance("covariate", &item)?;
            if let Some(expected_name) = name.as_deref()
                && item.name != expected_name
            {
                return Err(Error::Validation(format!(
                    "covariate provenance name '{}' must match selected covariate '{}'",
                    item.name, expected_name
                )));
            }
            item
        }
        None => {
            if let Some(name) = name {
                CovariateProvenance {
                    name,
                    timing: CovariateTiming::PreTreatment,
                    source_dataset: None,
                }
            } else {
                return Ok(Vec::new());
            }
        }
    };

    Ok(vec![item])
}

fn normalize_multi_covariate_names(
    control_names: &[String],
    variant_names: &[String],
    p: usize,
) -> Result<Vec<String>> {
    let names = if !control_names.is_empty() && !variant_names.is_empty() {
        if control_names != variant_names {
            return Err(Error::Validation(
                "control/variant selected covariates must match for CURE".to_string(),
            ));
        }
        control_names.to_vec()
    } else if !control_names.is_empty() {
        control_names.to_vec()
    } else {
        variant_names.to_vec()
    };

    if !names.is_empty() && names.len() != p {
        return Err(Error::Validation(format!(
            "selected_covariates length ({}) must match covariate columns ({})",
            names.len(),
            p
        )));
    }

    Ok(names)
}

fn normalize_multi_covariate_provenance(
    control: &MultiCovariateArmData,
    variant: &MultiCovariateArmData,
    p: usize,
) -> Result<Vec<CovariateProvenance>> {
    let provenance = if !control.covariate_provenance.is_empty()
        && !variant.covariate_provenance.is_empty()
    {
        if control.covariate_provenance != variant.covariate_provenance {
            return Err(Error::Validation(
                "control/variant covariate provenance must match for CURE".to_string(),
            ));
        }
        control.covariate_provenance.clone()
    } else if !control.covariate_provenance.is_empty() {
        control.covariate_provenance.clone()
    } else if !variant.covariate_provenance.is_empty() {
        variant.covariate_provenance.clone()
    } else {
        let names =
            normalize_multi_covariate_names(&control.covariate_names, &variant.covariate_names, p)?;
        if names.is_empty() {
            return Ok(Vec::new());
        }
        names
            .into_iter()
            .map(|name| CovariateProvenance {
                name,
                timing: CovariateTiming::PreTreatment,
                source_dataset: None,
            })
            .collect()
    };

    if provenance.len() != p {
        return Err(Error::Validation(format!(
            "covariate_provenance length ({}) must match covariate columns ({})",
            provenance.len(),
            p
        )));
    }

    let expected_names =
        normalize_multi_covariate_names(&control.covariate_names, &variant.covariate_names, p)?;
    if !expected_names.is_empty() {
        let actual_names: Vec<&str> = provenance.iter().map(|item| item.name.as_str()).collect();
        if actual_names != expected_names.iter().map(|item| item.as_str()).collect::<Vec<_>>() {
            return Err(Error::Validation(
                "covariate_provenance names must match selected_covariates".to_string(),
            ));
        }
    }

    for (idx, item) in provenance.iter().enumerate() {
        validate_covariate_provenance(&format!("covariate_provenance[{idx}]"), item)?;
    }

    Ok(provenance)
}

fn validate_covariate_provenance(label: &str, item: &CovariateProvenance) -> Result<()> {
    if item.name.trim().is_empty() {
        return Err(Error::Validation(format!("{label}.name must be non-empty")));
    }
    match item.timing {
        CovariateTiming::PreTreatment => Ok(()),
        CovariateTiming::Unknown => Err(Error::Validation(format!(
            "{label} has unknown timing; CUPED/CURE require explicit pre-treatment covariates"
        ))),
        CovariateTiming::PostTreatment => Err(Error::Validation(format!(
            "{label} is post-treatment; leakage-prone covariates are not allowed in CUPED/CURE"
        ))),
        CovariateTiming::Mixed => Err(Error::Validation(format!(
            "{label} mixes pre- and post-treatment data; CUPED/CURE require pre-treatment covariates only"
        ))),
    }
}

fn column_means(rows: &[Vec<f64>], p: usize) -> Result<Vec<f64>> {
    if rows.is_empty() {
        return Err(Error::Validation(
            "covariate matrix must contain at least one row".to_string(),
        ));
    }
    let mut means = vec![0.0; p];
    for row in rows {
        for (j, value) in row.iter().enumerate() {
            means[j] += *value;
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }
    Ok(means)
}

fn mean(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(Error::Validation("mean requires non-empty input".to_string()));
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn sample_variance(values: &[f64], mean: f64) -> Result<f64> {
    if values.len() < 2 {
        return Err(Error::Validation("variance requires at least two observations".to_string()));
    }
    Ok(values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0))
}

fn sample_covariance(xs: &[f64], ys: &[f64], mean_x: f64, mean_y: f64) -> Result<f64> {
    if xs.len() != ys.len() {
        return Err(Error::Validation(format!(
            "covariance length mismatch: {} vs {}",
            xs.len(),
            ys.len()
        )));
    }
    if xs.len() < 2 {
        return Err(Error::Validation("covariance requires at least two observations".to_string()));
    }
    Ok(xs.iter().zip(ys.iter()).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>()
        / (xs.len() as f64 - 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuped_reduces_variance_with_correlated_covariate() {
        let control = ArmData {
            outcomes: vec![10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5, 12.5, 13.5],
            covariates: vec![9.5, 11.5, 10.5, 12.5, 8.5, 13.5, 10.0, 11.0, 12.0, 13.0],
            covariate_name: Some("pre_period_metric".to_string()),
            covariate_provenance: Some(CovariateProvenance {
                name: "pre_period_metric".to_string(),
                timing: CovariateTiming::PreTreatment,
                source_dataset: Some("ads_pre_period_daily".to_string()),
            }),
            pre_treatment_only: true,
        };
        let variant = ArmData {
            outcomes: vec![11.0, 13.0, 12.0, 14.0, 10.0, 15.0, 11.5, 12.5, 13.5, 14.5],
            covariates: vec![9.5, 11.5, 10.5, 12.5, 8.5, 13.5, 10.0, 11.0, 12.0, 13.0],
            covariate_name: Some("pre_period_metric".to_string()),
            covariate_provenance: Some(CovariateProvenance {
                name: "pre_period_metric".to_string(),
                timing: CovariateTiming::PreTreatment,
                source_dataset: Some("ads_pre_period_daily".to_string()),
            }),
            pre_treatment_only: true,
        };

        let result = cuped_adjust(&control, &variant).unwrap();

        assert_eq!(result.method, VarianceReductionMethod::Cuped);
        assert_eq!(result.num_covariates, 1);
        assert_eq!(result.selected_covariates, vec!["pre_period_metric".to_string()]);
        assert!(result.provenance_validated);
        assert_eq!(result.covariate_provenance.len(), 1);
        assert!(result.rho > 0.9);
        assert!(result.variance_reduction_factor < 0.2);
        assert!(result.adjusted_variance < result.original_variance);
        assert!(result.effective_sample_multiplier > 5.0);
        assert!((result.effect - 1.0).abs() < 0.1);
    }

    #[test]
    fn cure_with_two_covariates() {
        let n = 20;
        let control = MultiCovariateArmData {
            outcomes: (0..n)
                .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.7).sin() * 2.0)
                .collect(),
            covariates: (0..n)
                .map(|i| vec![9.5 + i as f64 * 0.5, 3.0 + (i as f64 * 0.7).sin() * 2.5])
                .collect(),
            covariate_names: vec!["pre_clicks".to_string(), "pre_device_mix".to_string()],
            covariate_provenance: vec![
                CovariateProvenance {
                    name: "pre_clicks".to_string(),
                    timing: CovariateTiming::PreTreatment,
                    source_dataset: Some("account_history".to_string()),
                },
                CovariateProvenance {
                    name: "pre_device_mix".to_string(),
                    timing: CovariateTiming::PreTreatment,
                    source_dataset: Some("device_breakdown".to_string()),
                },
            ],
            pre_treatment_only: true,
        };
        let variant = MultiCovariateArmData {
            outcomes: (0..n)
                .map(|i| 11.0 + i as f64 * 0.5 + (i as f64 * 0.7).sin() * 2.0)
                .collect(),
            covariates: (0..n)
                .map(|i| vec![9.5 + i as f64 * 0.5, 3.0 + (i as f64 * 0.7).sin() * 2.5])
                .collect(),
            covariate_names: vec!["pre_clicks".to_string(), "pre_device_mix".to_string()],
            covariate_provenance: control.covariate_provenance.clone(),
            pre_treatment_only: true,
        };

        let result = cure_adjust(&control, &variant).unwrap();
        assert_eq!(result.method, VarianceReductionMethod::Cure);
        assert_eq!(result.num_covariates, 2);
        assert_eq!(
            result.selected_covariates,
            vec!["pre_clicks".to_string(), "pre_device_mix".to_string()]
        );
        assert!(result.provenance_validated);
        assert_eq!(result.covariate_provenance.len(), 2);
        assert!(result.r_squared > 0.5);
        assert!(result.variance_reduction_factor < 0.5);
        assert!((result.effect - 1.0).abs() < 0.5);
    }

    #[test]
    fn cure_single_covariate_matches_cuped() {
        let control_cuped = ArmData {
            outcomes: vec![10.0, 12.0, 11.0, 13.0, 9.0],
            covariates: vec![9.5, 11.5, 10.5, 12.5, 8.5],
            covariate_name: Some("pre_conversions".to_string()),
            covariate_provenance: Some(CovariateProvenance {
                name: "pre_conversions".to_string(),
                timing: CovariateTiming::PreTreatment,
                source_dataset: Some("pre_period_conversions".to_string()),
            }),
            pre_treatment_only: true,
        };
        let variant_cuped = ArmData {
            outcomes: vec![11.0, 13.0, 12.0, 14.0, 10.0],
            covariates: vec![9.5, 11.5, 10.5, 12.5, 8.5],
            covariate_name: Some("pre_conversions".to_string()),
            covariate_provenance: Some(CovariateProvenance {
                name: "pre_conversions".to_string(),
                timing: CovariateTiming::PreTreatment,
                source_dataset: Some("pre_period_conversions".to_string()),
            }),
            pre_treatment_only: true,
        };

        let control_cure = MultiCovariateArmData {
            outcomes: control_cuped.outcomes.clone(),
            covariates: control_cuped.covariates.iter().map(|&x| vec![x]).collect(),
            covariate_names: vec!["pre_conversions".to_string()],
            covariate_provenance: vec![control_cuped.covariate_provenance.clone().unwrap()],
            pre_treatment_only: true,
        };
        let variant_cure = MultiCovariateArmData {
            outcomes: variant_cuped.outcomes.clone(),
            covariates: variant_cuped.covariates.iter().map(|&x| vec![x]).collect(),
            covariate_names: vec!["pre_conversions".to_string()],
            covariate_provenance: vec![variant_cuped.covariate_provenance.clone().unwrap()],
            pre_treatment_only: true,
        };

        let cuped = cuped_adjust(&control_cuped, &variant_cuped).unwrap();
        let cure = cure_adjust(&control_cure, &variant_cure).unwrap();

        assert!((cuped.effect - cure.effect).abs() < 1e-6);
        assert!((cuped.r_squared - cure.r_squared).abs() < 1e-10);
        assert_eq!(cure.method, VarianceReductionMethod::Cuped);
    }

    #[test]
    fn cure_uses_ridge_fallback_for_collinear_covariates() {
        let control = MultiCovariateArmData {
            outcomes: vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            covariates: vec![
                vec![1.0, 2.0],
                vec![2.0, 4.0],
                vec![3.0, 6.0],
                vec![4.0, 8.0],
                vec![5.0, 10.0],
                vec![6.0, 12.0],
            ],
            covariate_names: vec!["pre_clicks".to_string(), "pre_clicks_x2".to_string()],
            covariate_provenance: vec![
                CovariateProvenance {
                    name: "pre_clicks".to_string(),
                    timing: CovariateTiming::PreTreatment,
                    source_dataset: Some("campaign_history".to_string()),
                },
                CovariateProvenance {
                    name: "pre_clicks_x2".to_string(),
                    timing: CovariateTiming::PreTreatment,
                    source_dataset: Some("campaign_history".to_string()),
                },
            ],
            pre_treatment_only: true,
        };
        let variant = MultiCovariateArmData {
            outcomes: vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            covariates: control.covariates.clone(),
            covariate_names: control.covariate_names.clone(),
            covariate_provenance: control.covariate_provenance.clone(),
            pre_treatment_only: true,
        };

        let result = cure_adjust(&control, &variant).unwrap();
        assert_eq!(result.solver, VarianceReductionSolver::Ridge);
        assert!(result.ridge_lambda.is_some());
        assert!(result.condition_number.is_none() || result.condition_number.unwrap().is_finite());
        assert!(result.effect.is_finite());
    }

    #[test]
    fn reject_non_pre_treatment_covariates() {
        let control = ArmData {
            outcomes: vec![1.0, 2.0],
            covariates: vec![1.0, 2.0],
            covariate_name: None,
            covariate_provenance: None,
            pre_treatment_only: false,
        };
        let variant = ArmData {
            outcomes: vec![2.0, 3.0],
            covariates: vec![1.0, 2.0],
            covariate_name: None,
            covariate_provenance: None,
            pre_treatment_only: true,
        };

        assert!(cuped_adjust(&control, &variant).is_err());
    }

    #[test]
    fn reject_post_treatment_provenance() {
        let control = ArmData {
            outcomes: vec![1.0, 2.0, 3.0],
            covariates: vec![1.0, 2.0, 3.0],
            covariate_name: Some("post_clicks".to_string()),
            covariate_provenance: Some(CovariateProvenance {
                name: "post_clicks".to_string(),
                timing: CovariateTiming::PostTreatment,
                source_dataset: Some("daily_experiment_exports".to_string()),
            }),
            pre_treatment_only: true,
        };
        let variant = ArmData {
            outcomes: vec![1.5, 2.5, 3.5],
            covariates: vec![1.0, 2.0, 3.0],
            covariate_name: Some("post_clicks".to_string()),
            covariate_provenance: control.covariate_provenance.clone(),
            pre_treatment_only: true,
        };

        let error = cuped_adjust(&control, &variant).unwrap_err().to_string();
        assert!(error.contains("post-treatment"));
    }
}
