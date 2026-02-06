//! Parameterization utilities for time series models.
//!
//! Many time-series models need constrained parameters:
//! - variances must be strictly positive
//! - some dynamics parameters are naturally bounded (e.g. AR(1) phi in (-1, 1))
//!
//! This module provides canonical bounds and bijective transforms that map from an
//! unconstrained vector `z ∈ R^n` to constrained parameters.

use crate::transforms::ParameterTransform;

/// Bounds for AR(1) parameters: `(phi, q, r)`.
///
/// - `phi ∈ (-1, 1)` (stationarity-friendly)
/// - `q > 0` (state noise variance)
/// - `r > 0` (observation noise variance)
pub const AR1_PARAMETER_BOUNDS: [(f64, f64); 3] =
    [(-1.0, 1.0), (0.0, f64::INFINITY), (0.0, f64::INFINITY)];

/// Build a transform for AR(1) parameters `(phi, q, r)`.
///
/// Uses softplus for positive parameters to avoid `exp(z)` overflow for large `z`.
pub fn ar1_parameter_transform() -> ParameterTransform {
    ParameterTransform::from_bounds_softplus(&AR1_PARAMETER_BOUNDS)
}

/// Stable parameter names for AR(1) parameters.
pub fn ar1_parameter_names() -> Vec<String> {
    vec!["phi".to_string(), "q".to_string(), "r".to_string()]
}

/// Suggested initial values for AR(1) parameters.
pub fn ar1_parameter_init() -> Vec<f64> {
    vec![0.5, 0.1, 0.2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar1_transform_roundtrip_and_grad_log_jacobian() {
        let t = ar1_parameter_transform();
        assert_eq!(t.dim(), 3);

        let z = vec![0.2, -1.0, 3.0];
        let theta = t.forward(&z);

        assert!(
            theta[0].is_finite() && theta[0] > -1.0 && theta[0] < 1.0,
            "phi out of bounds: {:?}",
            theta
        );
        assert!(theta[1].is_finite() && theta[1] > 0.0, "q not positive: {:?}", theta);
        assert!(theta[2].is_finite() && theta[2] > 0.0, "r not positive: {:?}", theta);

        let z_back = t.inverse(&theta);
        for (i, (&a, &b)) in z.iter().zip(z_back.iter()).enumerate() {
            let diff = (a - b).abs();
            let scale = a.abs().max(1.0);
            assert!(
                diff / scale < 1e-10,
                "roundtrip failed (i={}): z={} z_back={} theta={:?}",
                i,
                a,
                b,
                theta
            );
        }

        // Gradient sanity: compare grad(log|J|) vs finite differences.
        let g = t.grad_log_abs_det_jacobian(&z);
        let eps = 1e-7;
        for i in 0..z.len() {
            let mut zp = z.clone();
            zp[i] += eps;
            let mut zm = z.clone();
            zm[i] -= eps;
            let fd = (t.log_abs_det_jacobian(&zp) - t.log_abs_det_jacobian(&zm)) / (2.0 * eps);

            let diff = (g[i] - fd).abs();
            let scale = g[i].abs().max(1.0);
            assert!(
                diff / scale < 1e-6,
                "grad log|J| failed (i={}): analytical={} fd={} z={:?}",
                i,
                g[i],
                fd,
                z
            );
        }
    }
}
