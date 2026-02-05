use ns_core::{Error, Result};
use ns_inference::{AsymptoticCLsContext, MaximumLikelihoodEstimator};
use serde::{Deserialize, Serialize};

/// Canonical expected-set ordering in `-muhat/sigma` space.
///
/// Matches pyhf: `[2, 1, 0, -1, -2]`.
pub type NsSigmaOrder = [i32; 5];

const NSIGMA_ORDER: NsSigmaOrder = [2, 1, 0, -1, -2];

/// Single point in a CLs curve artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClsCurvePoint {
    /// Tested POI value.
    pub mu: f64,
    /// Observed CLs.
    pub cls: f64,
    /// Expected CLs at `nsigma_order=[2,1,0,-1,-2]`.
    pub expected: [f64; 5],
}

/// Plot-friendly artifact for CLs curves and Brazil bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClsCurveArtifact {
    /// Target CLs level (alpha), typically 0.05.
    pub alpha: f64,
    /// Canonical ordering of expected curves.
    pub nsigma_order: NsSigmaOrder,
    /// Per-point curve values.
    pub points: Vec<ClsCurvePoint>,
    /// Observed upper limit from scan interpolation.
    pub obs_limit: f64,
    /// Expected upper limits from scan interpolation.
    pub exp_limits: [f64; 5],
}

impl ClsCurveArtifact {
    /// Compute CLs curves (observed + expected band) over a scan grid.
    ///
    /// This mirrors pyhf's scan-based `upper_limit` behavior: compute CLs at each
    /// `mu` then interpolate `CLs(mu)=alpha` to get observed and expected limits.
    pub fn from_scan(
        ctx: &AsymptoticCLsContext,
        mle: &MaximumLikelihoodEstimator,
        alpha: f64,
        scan: &[f64],
    ) -> Result<Self> {
        if scan.len() < 2 {
            return Err(Error::Validation("scan must have at least 2 points".to_string()));
        }

        let mut points = Vec::with_capacity(scan.len());
        for &mu in scan {
            let r = ctx.hypotest_qtilde(mle, mu)?;
            let expected = ns_inference::hypotest::expected_cls_band_from_sqrtq_a(r.q_mu_a.sqrt());
            points.push(ClsCurvePoint { mu, cls: r.cls, expected });
        }

        let (obs_limit, exp_limits) = ctx.upper_limits_qtilde_linear_scan(mle, alpha, scan)?;

        Ok(Self { alpha, nsigma_order: NSIGMA_ORDER, points, obs_limit, exp_limits })
    }
}
