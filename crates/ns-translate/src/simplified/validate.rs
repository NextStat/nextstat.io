use super::schema::{
    SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN,
    SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE, SIMPLIFIED_LIKELIHOOD_SCHEMA_V0,
    SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_BASIS, SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_COVARIANCE,
    SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_DERIVED_FROM_WORKSPACE,
    SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_HS3,
    SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_PYHF, SimplifiedDiagnostics,
    SimplifiedFidelityDiagnostics, SimplifiedLikelihoodDerivation, SimplifiedLikelihoodWorkspace,
    SimplifiedUncertaintyModel,
};
use nalgebra::DMatrix;
use ns_core::{Error, Result};

const MATRIX_SYMM_TOL: f64 = 1e-9;
const PSD_EIGEN_TOL: f64 = 1e-10;
const EXPECTED_FLOOR: f64 = 0.0;

pub fn validate_simplified_likelihood(spec: &SimplifiedLikelihoodWorkspace) -> Result<()> {
    if spec.schema_version != SIMPLIFIED_LIKELIHOOD_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "expected schema_version '{}' but got '{}'",
            SIMPLIFIED_LIKELIHOOD_SCHEMA_V0, spec.schema_version
        )));
    }
    validate_nonempty("metadata.experiment", &spec.metadata.experiment)?;
    validate_nonempty("metadata.analysis_id", &spec.metadata.analysis_id)?;
    validate_nonempty("metadata.reference", &spec.metadata.reference)?;
    validate_source_format(&spec.metadata.source_format)?;

    if spec.poi.name.trim().is_empty() {
        return Err(Error::Validation("poi.name must not be empty".to_string()));
    }
    if !spec.poi.init.is_finite() {
        return Err(Error::Validation("poi.init must be finite".to_string()));
    }
    let [lo, hi] = spec.poi.bounds;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(Error::Validation(format!(
            "poi.bounds must be finite and ordered, got [{lo}, {hi}]"
        )));
    }

    let n_bins = spec.bins.len();
    if n_bins == 0 {
        return Err(Error::Validation("bins must contain at least one entry".to_string()));
    }
    for (idx, bin) in spec.bins.iter().enumerate() {
        if bin.channel.trim().is_empty() {
            return Err(Error::Validation(format!("bins[{idx}].channel must not be empty")));
        }
        if bin.name.trim().is_empty() {
            return Err(Error::Validation(format!("bins[{idx}].name must not be empty")));
        }
    }

    validate_nonnegative_vector("observed", &spec.observed, n_bins)?;
    validate_nonnegative_vector("background_nominal", &spec.background_nominal, n_bins)?;
    if let Some(signal_nominal) = &spec.signal_nominal {
        validate_nonnegative_vector("signal_nominal", signal_nominal, n_bins)?;
    }

    match &spec.uncertainty_model {
        SimplifiedUncertaintyModel::Basis { components } => {
            for (idx, component) in components.iter().enumerate() {
                if component.name.trim().is_empty() {
                    return Err(Error::Validation(format!(
                        "uncertainty_model.components[{idx}].name must not be empty"
                    )));
                }
                validate_nonnegative_vector(
                    &format!("uncertainty_model.components[{idx}].hi"),
                    &component.hi,
                    n_bins,
                )?;
                validate_nonnegative_vector(
                    &format!("uncertainty_model.components[{idx}].lo"),
                    &component.lo,
                    n_bins,
                )?;
            }
        }
        SimplifiedUncertaintyModel::Covariance { total_covariance, stat_covariance } => {
            validate_covariance_matrix(
                "uncertainty_model.total_covariance",
                total_covariance,
                n_bins,
            )?;
            validate_positive_semidefinite("uncertainty_model.total_covariance", total_covariance)?;
            if let Some(stat_covariance) = stat_covariance {
                validate_covariance_matrix(
                    "uncertainty_model.stat_covariance",
                    stat_covariance,
                    n_bins,
                )?;
                validate_positive_semidefinite(
                    "uncertainty_model.stat_covariance",
                    stat_covariance,
                )?;

                let shared_covariance = subtract_matrices(total_covariance, stat_covariance);
                validate_covariance_matrix(
                    "uncertainty_model.shared_systematic_covariance",
                    &shared_covariance,
                    n_bins,
                )?;
                validate_positive_semidefinite(
                    "uncertainty_model.shared_systematic_covariance",
                    &shared_covariance,
                )?;
            }
        }
    }

    if let Some(diagnostics) = &spec.diagnostics {
        validate_diagnostics(diagnostics)?;
    }

    if spec.metadata.source_format == SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_DERIVED_FROM_WORKSPACE {
        let derivation = spec.derivation.as_ref().ok_or_else(|| {
            Error::Validation(
                "derived_from_workspace artifacts must include derivation provenance".to_string(),
            )
        })?;
        validate_derivation(derivation)?;

        let diagnostics = spec.diagnostics.as_ref().ok_or_else(|| {
            Error::Validation(
                "derived_from_workspace artifacts must include diagnostics with fidelity metadata"
                    .to_string(),
            )
        })?;
        if matches!(&spec.uncertainty_model, SimplifiedUncertaintyModel::Basis { .. })
            && diagnostics.factorization.is_none()
        {
            return Err(Error::Validation(
                "derived_from_workspace basis artifacts must include factorization diagnostics"
                    .to_string(),
            ));
        }
        let fidelity = diagnostics.fidelity.as_ref().ok_or_else(|| {
            Error::Validation(
                "derived_from_workspace artifacts must include fidelity diagnostics".to_string(),
            )
        })?;
        validate_required_derived_fidelity(fidelity, n_bins)?;
    }

    Ok(())
}

fn validate_nonnegative_vector(label: &str, values: &[f64], expected_len: usize) -> Result<()> {
    if values.len() != expected_len {
        return Err(Error::Validation(format!(
            "{label} length mismatch: got {} expected {}",
            values.len(),
            expected_len
        )));
    }

    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::Validation(format!("{label}[{idx}] must be finite")));
        }
        if *value < EXPECTED_FLOOR {
            return Err(Error::Validation(format!(
                "{label}[{idx}] must be >= {EXPECTED_FLOOR}, got {value}"
            )));
        }
    }

    Ok(())
}

fn validate_nonempty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Validation(format!("{label} must not be empty")));
    }
    Ok(())
}

fn validate_source_format(source_format: &str) -> Result<()> {
    match source_format {
        SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_BASIS
        | SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_COVARIANCE
        | SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_DERIVED_FROM_WORKSPACE => Ok(()),
        other => Err(Error::Validation(format!(
            "metadata.source_format must be one of 'basis', 'covariance', 'derived_from_workspace'; got '{other}'"
        ))),
    }
}

fn validate_covariance_matrix(label: &str, matrix: &[Vec<f64>], expected_dim: usize) -> Result<()> {
    if matrix.len() != expected_dim {
        return Err(Error::Validation(format!(
            "{label} dimension mismatch: got {} expected {}",
            matrix.len(),
            expected_dim
        )));
    }

    for (i, row) in matrix.iter().enumerate() {
        if row.len() != expected_dim {
            return Err(Error::Validation(format!(
                "{label}[{i}] length mismatch: got {} expected {}",
                row.len(),
                expected_dim
            )));
        }
        for (j, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::Validation(format!("{label}[{i}][{j}] must be finite")));
            }
            if i == j && *value < 0.0 {
                return Err(Error::Validation(format!(
                    "{label}[{i}][{j}] must have non-negative diagonal, got {value}"
                )));
            }
        }
    }

    for i in 0..expected_dim {
        for j in 0..expected_dim {
            if (matrix[i][j] - matrix[j][i]).abs() > MATRIX_SYMM_TOL {
                return Err(Error::Validation(format!(
                    "{label} must be symmetric within tolerance at ({i}, {j})"
                )));
            }
        }
    }

    Ok(())
}

fn validate_positive_semidefinite(label: &str, matrix: &[Vec<f64>]) -> Result<()> {
    let matrix = matrix_from_rows(matrix)?;
    let matrix = symmetrize_matrix(matrix);
    let eigen = matrix.symmetric_eigen();
    let min_eigenvalue = eigen.eigenvalues.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    if min_eigenvalue < -PSD_EIGEN_TOL {
        return Err(Error::Validation(format!(
            "{label} must be positive semidefinite within tolerance, min eigenvalue={min_eigenvalue}"
        )));
    }
    Ok(())
}

fn matrix_from_rows(rows: &[Vec<f64>]) -> Result<DMatrix<f64>> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, |row| row.len());
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(Error::Validation("matrix rows must have equal length".to_string()));
    }

    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in rows {
        flat.extend_from_slice(row);
    }
    Ok(DMatrix::from_row_slice(nrows, ncols, &flat))
}

fn symmetrize_matrix(matrix: DMatrix<f64>) -> DMatrix<f64> {
    let transpose = matrix.transpose();
    (matrix + transpose).scale(0.5)
}

fn subtract_matrices(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_row, rhs_row)| {
            lhs_row
                .iter()
                .zip(rhs_row.iter())
                .map(|(lhs_value, rhs_value)| lhs_value - rhs_value)
                .collect()
        })
        .collect()
}

fn validate_diagnostics(diagnostics: &SimplifiedDiagnostics) -> Result<()> {
    if let Some(factorization) = &diagnostics.factorization {
        validate_nonempty("diagnostics.factorization.method", &factorization.method)?;
        if !factorization.explained_variance_fraction.is_finite()
            || factorization.explained_variance_fraction < 0.0
            || factorization.explained_variance_fraction > 1.0 + 1e-9
        {
            return Err(Error::Validation(format!(
                "diagnostics.factorization.explained_variance_fraction must be finite and within [0, 1], got {}",
                factorization.explained_variance_fraction
            )));
        }
        validate_finite_nonnegative(
            "diagnostics.factorization.frobenius_residual",
            factorization.frobenius_residual,
        )?;
        validate_finite_nonnegative(
            "diagnostics.factorization.max_clipped_negative_eigenvalue_magnitude",
            factorization.max_clipped_negative_eigenvalue_magnitude,
        )?;
        validate_finite_nonnegative(
            "diagnostics.factorization.input_trace",
            factorization.input_trace,
        )?;
        validate_finite_nonnegative(
            "diagnostics.factorization.retained_trace",
            factorization.retained_trace,
        )?;
        if let Some(stat_covariance_trace) = factorization.stat_covariance_trace {
            validate_finite_nonnegative(
                "diagnostics.factorization.stat_covariance_trace",
                stat_covariance_trace,
            )?;
        }
        if let Some(shared_systematic_trace) = factorization.shared_systematic_trace {
            validate_finite_nonnegative(
                "diagnostics.factorization.shared_systematic_trace",
                shared_systematic_trace,
            )?;
        }
    }

    if let Some(fidelity) = &diagnostics.fidelity {
        validate_optional_fidelity(fidelity)?;
    }

    Ok(())
}

fn validate_optional_fidelity(fidelity: &SimplifiedFidelityDiagnostics) -> Result<()> {
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.nuisance_count_full",
        fidelity.nuisance_count_full.map(|value| value as f64),
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.nuisance_count_reduced",
        fidelity.nuisance_count_reduced.map(|value| value as f64),
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.bins_count",
        fidelity.bins_count.map(|value| value as f64),
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.relative_background_cov_residual",
        fidelity.relative_background_cov_residual,
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.max_abs_expected_delta_at_nominal",
        fidelity.max_abs_expected_delta_at_nominal,
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.max_abs_expected_delta_random_draws",
        fidelity.max_abs_expected_delta_random_draws,
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.qmu_delta_smoke",
        fidelity.qmu_delta_smoke,
    )?;
    if let Some(upper_limit_ratio_smoke) = fidelity.upper_limit_ratio_smoke
        && (!upper_limit_ratio_smoke.is_finite() || upper_limit_ratio_smoke <= 0.0)
    {
        return Err(Error::Validation(format!(
            "diagnostics.fidelity.upper_limit_ratio_smoke must be finite and > 0, got {upper_limit_ratio_smoke}"
        )));
    }
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.max_abs_yield_delta",
        fidelity.max_abs_yield_delta,
    )?;
    validate_optional_finite_nonnegative(
        "diagnostics.fidelity.max_rel_yield_delta",
        fidelity.max_rel_yield_delta,
    )?;
    Ok(())
}

fn validate_derivation(derivation: &SimplifiedLikelihoodDerivation) -> Result<()> {
    match derivation.source_workspace_format.as_str() {
        SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_PYHF
        | SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_HS3 => {}
        other => {
            return Err(Error::Validation(format!(
                "derivation.source_workspace_format must be 'pyhf' or 'hs3', got '{other}'"
            )));
        }
    }
    if let Some(source_workspace_schema_version) =
        derivation.source_workspace_schema_version.as_deref()
    {
        validate_nonempty(
            "derivation.source_workspace_schema_version",
            source_workspace_schema_version,
        )?;
    }
    if let Some(fit_result_schema_version) = derivation.fit_result_schema_version.as_deref() {
        validate_nonempty("derivation.fit_result_schema_version", fit_result_schema_version)?;
    }
    validate_nonempty_vec("derivation.selected_channels", &derivation.selected_channels)?;
    if let Some(selected_bins) = derivation.selected_bins.as_deref() {
        validate_nonempty_vec("derivation.selected_bins", selected_bins)?;
    }
    if derivation.basis_method != SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN {
        return Err(Error::Validation(format!(
            "derivation.basis_method must be '{}', got '{}'",
            SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN, derivation.basis_method
        )));
    }
    if !derivation.explained_variance_target.is_finite()
        || derivation.explained_variance_target <= 0.0
        || derivation.explained_variance_target > 1.0
    {
        return Err(Error::Validation(format!(
            "derivation.explained_variance_target must be in (0, 1], got {}",
            derivation.explained_variance_target
        )));
    }
    if derivation.jacobian_method != SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE {
        return Err(Error::Validation(format!(
            "derivation.jacobian_method must be '{}', got '{}'",
            SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE, derivation.jacobian_method
        )));
    }
    Ok(())
}

fn validate_nonempty_vec(label: &str, values: &[String]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::Validation(format!("{label} must contain at least one entry")));
    }
    for (idx, value) in values.iter().enumerate() {
        validate_nonempty(&format!("{label}[{idx}]"), value)?;
    }
    Ok(())
}

fn validate_required_derived_fidelity(
    fidelity: &SimplifiedFidelityDiagnostics,
    expected_bins: usize,
) -> Result<()> {
    validate_optional_fidelity(fidelity)?;

    let nuisance_count_full = fidelity.nuisance_count_full.ok_or_else(|| {
        Error::Validation(
            "derived_from_workspace artifacts must include diagnostics.fidelity.nuisance_count_full"
                .to_string(),
        )
    })?;
    let nuisance_count_reduced = fidelity.nuisance_count_reduced.ok_or_else(|| {
        Error::Validation(
            "derived_from_workspace artifacts must include diagnostics.fidelity.nuisance_count_reduced"
                .to_string(),
        )
    })?;
    let bins_count = fidelity.bins_count.ok_or_else(|| {
        Error::Validation(
            "derived_from_workspace artifacts must include diagnostics.fidelity.bins_count"
                .to_string(),
        )
    })?;
    if bins_count != expected_bins {
        return Err(Error::Validation(format!(
            "diagnostics.fidelity.bins_count mismatch: got {bins_count} expected {expected_bins}"
        )));
    }
    if nuisance_count_reduced > nuisance_count_full {
        return Err(Error::Validation(format!(
            "diagnostics.fidelity.nuisance_count_reduced must be <= nuisance_count_full, got {nuisance_count_reduced} > {nuisance_count_full}"
        )));
    }

    require_fidelity_field(
        "diagnostics.fidelity.relative_background_cov_residual",
        fidelity.relative_background_cov_residual,
    )?;
    require_fidelity_field(
        "diagnostics.fidelity.max_abs_expected_delta_at_nominal",
        fidelity.max_abs_expected_delta_at_nominal,
    )?;
    require_fidelity_field(
        "diagnostics.fidelity.max_abs_expected_delta_random_draws",
        fidelity.max_abs_expected_delta_random_draws,
    )?;
    require_fidelity_field("diagnostics.fidelity.qmu_delta_smoke", fidelity.qmu_delta_smoke)?;
    let upper_limit_ratio_smoke = fidelity.upper_limit_ratio_smoke.ok_or_else(|| {
        Error::Validation(
            "derived_from_workspace artifacts must include diagnostics.fidelity.upper_limit_ratio_smoke"
                .to_string(),
        )
    })?;
    if !upper_limit_ratio_smoke.is_finite() || upper_limit_ratio_smoke <= 0.0 {
        return Err(Error::Validation(format!(
            "diagnostics.fidelity.upper_limit_ratio_smoke must be finite and > 0, got {upper_limit_ratio_smoke}"
        )));
    }

    Ok(())
}

fn require_fidelity_field(label: &str, value: Option<f64>) -> Result<f64> {
    let value = value.ok_or_else(|| {
        Error::Validation(format!("derived_from_workspace artifacts must include {label}"))
    })?;
    validate_finite_nonnegative(label, value)?;
    Ok(value)
}

fn validate_optional_finite_nonnegative(label: &str, value: Option<f64>) -> Result<()> {
    if let Some(value) = value {
        validate_finite_nonnegative(label, value)?;
    }
    Ok(())
}

fn validate_finite_nonnegative(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::Validation(format!("{label} must be finite and >= 0, got {value}")));
    }
    Ok(())
}
