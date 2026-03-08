use super::schema::{
    SimplifiedBasisComponent, SimplifiedFactorizationDiagnostics, SimplifiedLikelihoodWorkspace,
    SimplifiedUncertaintyModel,
};
use super::validate::validate_simplified_likelihood;
use nalgebra::DMatrix;
use ns_core::{Error, Result};

const PSD_EIGEN_TOL: f64 = 1e-10;
const RETAIN_ABS_EIGEN_TOL: f64 = PSD_EIGEN_TOL;
const RETAIN_REL_EIGEN_TOL: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct CovarianceFactorizationResult {
    pub components: Vec<SimplifiedBasisComponent>,
    pub diagnostics: SimplifiedFactorizationDiagnostics,
    pub retained_covariance: Vec<Vec<f64>>,
}

pub fn factorize_covariance_workspace(
    spec: &SimplifiedLikelihoodWorkspace,
) -> Result<CovarianceFactorizationResult> {
    validate_simplified_likelihood(spec)?;

    let (total_covariance, stat_covariance) = match &spec.uncertainty_model {
        SimplifiedUncertaintyModel::Covariance { total_covariance, stat_covariance } => {
            (total_covariance, stat_covariance.as_ref())
        }
        SimplifiedUncertaintyModel::Basis { .. } => {
            return Err(Error::Validation(
                "covariance factorization requires uncertainty_model.kind = covariance".to_string(),
            ));
        }
    };

    factorize_covariance_matrix(
        &spec.background_nominal,
        total_covariance,
        stat_covariance.map(|matrix| matrix.as_slice()),
        1.0,
        None,
        "sl_cov_",
    )
}

pub fn factorize_covariance_matrix(
    nominal: &[f64],
    total_covariance: &[Vec<f64>],
    stat_covariance: Option<&[Vec<f64>]>,
    explained_variance_target: f64,
    max_components: Option<usize>,
    component_prefix: &str,
) -> Result<CovarianceFactorizationResult> {
    if !explained_variance_target.is_finite()
        || explained_variance_target <= 0.0
        || explained_variance_target > 1.0
    {
        return Err(Error::Validation(format!(
            "explained_variance_target must be in (0, 1], got {explained_variance_target}"
        )));
    }
    if matches!(max_components, Some(0)) {
        return Err(Error::Validation("max_components must be >= 1 when provided".to_string()));
    }
    if nominal.len() != total_covariance.len() {
        return Err(Error::Validation(format!(
            "nominal length mismatch: nominal={} covariance_dim={}",
            nominal.len(),
            total_covariance.len()
        )));
    }

    let covariance = symmetrize_matrix(matrix_from_rows(total_covariance)?);
    let eigen = covariance.clone().symmetric_eigen();
    let max_eigenvalue =
        eigen.eigenvalues.iter().copied().fold(0.0_f64, |acc, value| acc.max(value.max(0.0)));
    let input_trace = covariance.trace();

    let mut clipped_negative_eigenvalues = 0usize;
    let mut max_clipped_negative_eigenvalue_magnitude = 0.0_f64;
    let mut original_rank = 0usize;
    let mut retained_rank = 0usize;
    let mut retained_trace = 0.0_f64;
    let mut retained_eigenpairs = Vec::<(usize, f64)>::new();

    for (idx, eigenvalue) in eigen.eigenvalues.iter().copied().enumerate() {
        if eigenvalue < -PSD_EIGEN_TOL {
            return Err(Error::Validation(format!(
                "uncertainty_model.total_covariance must be positive semidefinite within tolerance, min eigenvalue={eigenvalue}"
            )));
        }

        if eigenvalue > PSD_EIGEN_TOL {
            original_rank += 1;
        } else if eigenvalue < 0.0 {
            clipped_negative_eigenvalues += 1;
            max_clipped_negative_eigenvalue_magnitude =
                max_clipped_negative_eigenvalue_magnitude.max(-eigenvalue);
        }

        let clipped = eigenvalue.max(0.0);
        if should_retain_eigenmode(clipped, max_eigenvalue) {
            retained_eigenpairs.push((idx, clipped));
        }
    }

    retained_eigenpairs.sort_by(|(idx_a, eigen_a), (idx_b, eigen_b)| {
        eigen_b.partial_cmp(eigen_a).unwrap_or(std::cmp::Ordering::Equal).then(idx_a.cmp(idx_b))
    });

    let component_budget = max_components.unwrap_or(retained_eigenpairs.len());
    let mut retained_columns = Vec::<(String, Vec<f64>)>::new();
    for (idx, clipped) in retained_eigenpairs {
        if retained_rank >= component_budget {
            break;
        }
        if retained_trace < input_trace * explained_variance_target || retained_rank == 0 {
            let component_idx = retained_rank;
            retained_rank += 1;
            retained_trace += clipped;

            let scale = clipped.sqrt();
            let mut shift = Vec::with_capacity(nominal.len());
            for row in 0..covariance.nrows() {
                shift.push(eigen.eigenvectors[(row, idx)] * scale);
            }
            retained_columns.push((format!("{component_prefix}{component_idx:03}"), shift));
        }
    }

    let mut components = Vec::with_capacity(retained_columns.len());
    for (name, shift) in retained_columns {
        let hi = apply_shift("hi", nominal, &shift, 1.0)?;
        let lo = apply_shift("lo", nominal, &shift, -1.0)?;
        components.push(SimplifiedBasisComponent { name, hi, lo });
    }

    let rebuilt_covariance = covariance_from_components(nominal, &components);
    let residual = frobenius_norm(&(covariance.clone() - rebuilt_covariance.clone()));
    let explained_variance_fraction =
        if input_trace > 0.0 { retained_trace / input_trace } else { 1.0 };

    let stat_covariance_trace = stat_covariance.map(matrix_trace);
    let shared_systematic_trace =
        stat_covariance.map(|stat| matrix_trace(total_covariance) - matrix_trace(stat));

    Ok(CovarianceFactorizationResult {
        components,
        retained_covariance: matrix_to_rows(&rebuilt_covariance),
        diagnostics: SimplifiedFactorizationDiagnostics {
            method: "symmetric_eigendecomposition".to_string(),
            original_rank,
            retained_rank,
            explained_variance_fraction,
            frobenius_residual: residual,
            clipped_negative_eigenvalues,
            max_clipped_negative_eigenvalue_magnitude,
            input_trace,
            retained_trace,
            stat_covariance_trace,
            shared_systematic_trace,
        },
    })
}

fn should_retain_eigenmode(eigenvalue: f64, max_eigenvalue: f64) -> bool {
    if eigenvalue <= RETAIN_ABS_EIGEN_TOL {
        return false;
    }
    if max_eigenvalue <= 0.0 {
        return false;
    }
    eigenvalue / max_eigenvalue > RETAIN_REL_EIGEN_TOL
}

fn apply_shift(label: &str, nominal: &[f64], shift: &[f64], sign: f64) -> Result<Vec<f64>> {
    nominal
        .iter()
        .zip(shift.iter())
        .enumerate()
        .map(|(idx, (nominal_value, shift_value))| {
            let shifted = nominal_value + sign * shift_value;
            if shifted < 0.0 {
                return Err(Error::Validation(format!(
                    "factorized covariance produced negative {label} template at bin {idx}: {shifted}"
                )));
            }
            Ok(shifted)
        })
        .collect()
}

fn covariance_from_components(
    nominal: &[f64],
    components: &[SimplifiedBasisComponent],
) -> DMatrix<f64> {
    let n = nominal.len();
    let mut covariance = DMatrix::zeros(n, n);

    for component in components {
        let shift: Vec<f64> =
            component.hi.iter().zip(nominal.iter()).map(|(hi, nominal)| hi - nominal).collect();
        for i in 0..n {
            for j in 0..n {
                covariance[(i, j)] += shift[i] * shift[j];
            }
        }
    }

    covariance
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

fn matrix_trace(matrix: &[Vec<f64>]) -> f64 {
    matrix.iter().enumerate().map(|(idx, row)| row[idx]).sum()
}

fn frobenius_norm(matrix: &DMatrix<f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn matrix_to_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(matrix.nrows());
    for row in 0..matrix.nrows() {
        let mut values = Vec::with_capacity(matrix.ncols());
        for col in 0..matrix.ncols() {
            values.push(matrix[(row, col)]);
        }
        rows.push(values);
    }
    rows
}
