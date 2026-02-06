//! Asymptotic CLs hypothesis tests (frequentist).
//!
//! This follows the pyhf `infer.hypotest(..., calctype="asymptotics", test_stat="qtilde")`
//! implementation, matching the test statistic transformation used in
//! `pyhf.infer.calculators.AsymptoticCalculator` (pyhf 0.7.x).

use crate::MaximumLikelihoodEstimator;
use ns_core::traits::LogDensityModel;
use ns_core::{Error, Result};
use ns_translate::pyhf::HistFactoryModel;

/// Canonical expected-set ordering in `-muhat/sigma` space.
///
/// Matches pyhf: `n_sigma = [2, 1, 0, -1, -2]`.
pub const NSIGMA_ORDER: [f64; 5] = [2.0, 1.0, 0.0, -1.0, -2.0];

fn poi_index(model: &HistFactoryModel) -> Result<usize> {
    model.poi_index().ok_or_else(|| Error::Validation("No POI defined".to_string()))
}

const CLB_MIN: f64 = 1e-300;

fn normal_cdf(x: f64) -> f64 {
    // Use erfc for better numerical behavior in the tails:
    // Φ(x) = 0.5 * erfc(-x / sqrt(2))
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
}

#[inline]
fn safe_cls(clsb: f64, clb: f64) -> f64 {
    // CLs = CLsb / CLb, but CLb can underflow to 0 in the far tails.
    // In that regime CLsb also underflows, and the physically meaningful ratio tends to 0.
    if !(clsb.is_finite() && clb.is_finite()) {
        return 0.0;
    }
    if clb <= CLB_MIN {
        return if clsb <= CLB_MIN { 0.0 } else { 1.0 };
    }
    (clsb / clb).clamp(0.0, 1.0)
}

fn interp_limit(alpha: f64, xs: &[f64], ys: &[f64]) -> Result<f64> {
    // Robust linear interpolation for the limit, matching numpy.interp behavior when ys is monotone.
    //
    // For typical upper-limit scans, `xs` are mu scan points in ascending order and `ys` (CLs)
    // is (approximately) non-increasing. Numerical noise can break monotonicity; in that case we
    // pick the *first* crossing of alpha as mu increases.
    if xs.len() != ys.len() {
        return Err(Error::Validation("interp input length mismatch".to_string()));
    }
    let n = xs.len();
    if n < 2 {
        return Err(Error::Validation("interp requires >=2 points".to_string()));
    }

    // Clamp outside the observed range in the scan direction (mirrors numpy.interp clamp).
    let decreasing = ys[0] >= ys[n - 1];
    if decreasing {
        if alpha >= ys[0] {
            return Ok(xs[0]);
        }
        if alpha <= ys[n - 1] {
            return Ok(xs[n - 1]);
        }
        // First crossing from above to below.
        for i in 0..(n - 1) {
            let y0 = ys[i];
            let y1 = ys[i + 1];
            if (y0 - alpha).abs() < 1e-18 {
                return Ok(xs[i]);
            }
            if y0 >= alpha && y1 <= alpha && (y1 - y0).abs() >= 1e-18 {
                let x0 = xs[i];
                let x1 = xs[i + 1];
                let t = (alpha - y0) / (y1 - y0);
                return Ok(x0 + t * (x1 - x0));
            }
        }
    } else {
        if alpha <= ys[0] {
            return Ok(xs[0]);
        }
        if alpha >= ys[n - 1] {
            return Ok(xs[n - 1]);
        }
        for i in 0..(n - 1) {
            let y0 = ys[i];
            let y1 = ys[i + 1];
            if (y0 - alpha).abs() < 1e-18 {
                return Ok(xs[i]);
            }
            if y0 <= alpha && y1 >= alpha && (y1 - y0).abs() >= 1e-18 {
                let x0 = xs[i];
                let x1 = xs[i + 1];
                let t = (alpha - y0) / (y1 - y0);
                return Ok(x0 + t * (x1 - x0));
            }
        }
    }

    // If we didn't find a bracket due to non-monotonic noise or flats, fall back to any sign change.
    log::warn!("hypotest: non-monotonic CLs scan; using first sign-change fallback");
    for i in 0..(n - 1) {
        let y0 = ys[i] - alpha;
        let y1 = ys[i + 1] - alpha;
        if y0 == 0.0 {
            return Ok(xs[i]);
        }
        if (y0 > 0.0 && y1 < 0.0) || (y0 < 0.0 && y1 > 0.0) {
            let x0 = xs[i];
            let x1 = xs[i + 1];
            let yy0 = ys[i];
            let yy1 = ys[i + 1];
            if (yy1 - yy0).abs() < 1e-18 {
                return Ok(x0);
            }
            let t = (alpha - yy0) / (yy1 - yy0);
            return Ok(x0 + t * (x1 - x0));
        }
    }

    // Total failure: return the end closest to alpha.
    let mut best_i = 0usize;
    let mut best_d = (ys[0] - alpha).abs();
    for i in 1..n {
        let d = (ys[i] - alpha).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    Ok(xs[best_i])
}

/// Compute the expected CLs "Brazil band" for a given Asimov `sqrt(q_mu,A)`.
///
/// Output ordering is `NSIGMA_ORDER` in `-muhat/sigma` space (pyhf-compatible).
pub fn expected_cls_band_from_sqrtq_a(sqrtq_a: f64) -> [f64; 5] {
    let mut out = [0.0; 5];
    for (i, t) in NSIGMA_ORDER.into_iter().enumerate() {
        let clsb = normal_cdf(-(t + sqrtq_a));
        let clb = normal_cdf(-t);
        out[i] = safe_cls(clsb, clb);
    }
    out
}

/// Result of an asymptotic `hypotest` at a single tested POI value.
#[derive(Debug, Clone)]
pub struct HypotestResult {
    /// Tested POI value.
    pub mu_test: f64,
    /// Observed CLs value.
    pub cls: f64,
    /// Observed CLs+b value.
    pub clsb: f64,
    /// Observed CLb value.
    pub clb: f64,
    /// The test statistic in `-muhat/sigma` space (pyhf `AsymptoticCalculator.teststatistic`).
    pub teststat: f64,
    /// Observed `q_mu`/`qtilde_mu` value.
    pub q_mu: f64,
    /// Asimov `q_mu,A` value.
    pub q_mu_a: f64,
    /// Unconditional best-fit POI on observed data.
    pub mu_hat: f64,
}

/// Observed and expected CLs values (Brazil band) for a fixed tested POI.
///
/// The expected ordering matches pyhf: `n_sigma = [2, 1, 0, -1, -2]` in the
/// `-muhat/sigma` space (see `pyhf.infer.calculators.AsymptoticCalculator.expected_pvalues`).
#[derive(Debug, Clone)]
pub struct HypotestExpectedSet {
    /// Observed CLs.
    pub observed: f64,
    /// Expected CLs at `n_sigma = [2, 1, 0, -1, -2]`.
    pub expected: [f64; 5],
}

fn qmu_like_with_free(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
    free_nll: f64,
    mu_hat: f64,
    poi: usize,
    mu_test: f64,
) -> Result<(f64, f64, u64, bool)> {
    let fixed_model = model.with_fixed_param(poi, mu_test);
    let fixed = mle.fit_minimum(&fixed_model)?;
    if !fixed.converged {
        log::warn!(
            "hypotest: fixed fit did not converge for mu_test={}: {} (continuing with best-found)",
            mu_test,
            fixed.message
        );
    }

    let llr = 2.0 * (fixed.fval - free_nll);
    let mut q = llr.max(0.0);
    if mu_hat > mu_test {
        q = 0.0;
    }
    Ok((q, fixed.fval, fixed.n_iter, fixed.converged))
}

/// Context for repeated asymptotic CLs evaluations (caches free fits and Asimov model).
#[derive(Debug, Clone)]
pub struct AsymptoticCLsContext {
    poi: usize,
    data_model: HistFactoryModel,
    asimov_model: HistFactoryModel,
    free_data_nll: f64,
    free_data_mu_hat: f64,
    free_asimov_nll: f64,
    free_asimov_mu_hat: f64,
}

impl AsymptoticCLsContext {
    /// Build an asymptotic calculator context for `test_stat="qtilde"`.
    ///
    /// - Free fit to observed data is cached.
    /// - Asimov dataset is built with `asimov_mu = 0.0` (pyhf behavior for qtilde).
    /// - Free fit to the Asimov dataset is cached.
    pub fn new(mle: &MaximumLikelihoodEstimator, model: &HistFactoryModel) -> Result<Self> {
        let poi = poi_index(model)?;

        let free_data = mle.fit_minimum(model)?;
        if !free_data.converged {
            return Err(Error::Validation(format!(
                "Free fit on observed data did not converge: {}",
                free_data.message
            )));
        }
        let free_data_nll = free_data.fval;
        let free_data_mu_hat = free_data.parameters[poi];

        // Asimov: fit nuisances with POI fixed at 0.0, then set observed main to expected.
        let asimov_mu = 0.0;
        let fixed0 = mle.fit_minimum(&model.with_fixed_param(poi, asimov_mu))?;
        if !fixed0.converged {
            return Err(Error::Validation(format!(
                "Fit for Asimov nuisances (mu=0) did not converge: {}",
                fixed0.message
            )));
        }
        let expected_main = model.expected_data(&fixed0.parameters)?;
        let asimov_model = model
            .with_observed_main(&expected_main)?
            .with_constraint_centers(&fixed0.parameters)?
            .with_shapesys_aux_observed_from_params(&fixed0.parameters)?;

        let free_asimov = mle.fit_minimum(&asimov_model)?;
        if !free_asimov.converged {
            return Err(Error::Validation(format!(
                "Free fit on Asimov data did not converge: {}",
                free_asimov.message
            )));
        }
        let free_asimov_nll = free_asimov.fval;
        let free_asimov_mu_hat = free_asimov.parameters[poi];

        Ok(Self {
            poi,
            data_model: model.clone(),
            asimov_model,
            free_data_nll,
            free_data_mu_hat,
            free_asimov_nll,
            free_asimov_mu_hat,
        })
    }

    /// Evaluate asymptotic `hypotest` with `test_stat="qtilde"` and `calc_base_dist="normal"`.
    pub fn hypotest_qtilde(
        &self,
        mle: &MaximumLikelihoodEstimator,
        mu_test: f64,
    ) -> Result<HypotestResult> {
        // Observed q_mu
        let (q_mu, _nll_mu, _n_iter_mu, _conv_mu) = qmu_like_with_free(
            mle,
            &self.data_model,
            self.free_data_nll,
            self.free_data_mu_hat,
            self.poi,
            mu_test,
        )?;

        // Asimov q_mu,A
        let (q_mu_a, _nll_mu_a, _n_iter_mu_a, _conv_mu_a) = qmu_like_with_free(
            mle,
            &self.asimov_model,
            self.free_asimov_nll,
            self.free_asimov_mu_hat,
            self.poi,
            mu_test,
        )?;

        let sqrtq = q_mu.sqrt();
        let sqrtq_a = q_mu_a.sqrt();

        // Match pyhf's AsymptoticCalculator transformation for qtilde.
        let teststat = if sqrtq <= sqrtq_a {
            sqrtq - sqrtq_a
        } else {
            let denom = 2.0 * sqrtq_a.max(1e-16);
            (q_mu - q_mu_a) / denom
        };

        // Distributions in -muhat/sigma space:
        // sb: shift = -sqrtq_a, b: shift = 0, and right-tail pvalues as normal_cdf(-(...)).
        let clsb = normal_cdf(-(teststat + sqrtq_a));
        let clb = normal_cdf(-teststat);
        let cls = safe_cls(clsb, clb);

        Ok(HypotestResult {
            mu_test,
            cls,
            clsb,
            clb,
            teststat,
            q_mu,
            q_mu_a,
            mu_hat: self.free_data_mu_hat,
        })
    }

    /// Compute observed CLs and expected CLs band (Brazil band) for a single `mu_test`.
    pub fn hypotest_qtilde_expected_set(
        &self,
        mle: &MaximumLikelihoodEstimator,
        mu_test: f64,
    ) -> Result<HypotestExpectedSet> {
        let r = self.hypotest_qtilde(mle, mu_test)?;
        let expected = expected_cls_band_from_sqrtq_a(r.q_mu_a.sqrt());
        Ok(HypotestExpectedSet { observed: r.cls, expected })
    }

    /// Compute observed and expected upper limits from a linear scan + interpolation.
    ///
    /// Mirrors `pyhf.infer.intervals.upper_limits.upper_limit(..., scan=..., test_stat="qtilde")`.
    pub fn upper_limits_qtilde_linear_scan(
        &self,
        mle: &MaximumLikelihoodEstimator,
        alpha: f64,
        scan: &[f64],
    ) -> Result<(f64, [f64; 5])> {
        if scan.len() < 2 {
            return Err(Error::Validation("scan must have at least 2 points".to_string()));
        }
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(Error::Validation(format!("alpha must be in (0,1), got {}", alpha)));
        }

        let mut observed_cls: Vec<f64> = Vec::with_capacity(scan.len());
        let mut expected_cls: Vec<[f64; 5]> = Vec::with_capacity(scan.len());

        // Warm-start fixed fits across scan points (huge speedup vs restarting from parameter_init).
        let mut last_fixed_data_params: Option<Vec<f64>> = None;
        let mut last_fixed_asimov_params: Option<Vec<f64>> = None;

        for &mu in scan {
            // Observed: q_mu
            let fixed_data_model = self.data_model.with_fixed_param(self.poi, mu);
            let init_data = if let Some(mut p) = last_fixed_data_params.clone() {
                if self.poi < p.len() {
                    p[self.poi] = mu;
                }
                p
            } else {
                fixed_data_model.parameter_init()
            };
            let fixed_data = mle.fit_minimum_from(&fixed_data_model, &init_data)?;
            if !fixed_data.converged {
                log::warn!(
                    "hypotest scan: fixed fit did not converge for mu_test={}: {} (continuing with best-found)",
                    mu,
                    fixed_data.message
                );
            } else {
                // Only warm-start from a converged fit. Using non-converged parameters can
                // destabilize later scan points and make the scan much slower.
                last_fixed_data_params = Some(fixed_data.parameters.clone());
            }

            let llr = 2.0 * (fixed_data.fval - self.free_data_nll);
            let mut q_mu = llr.max(0.0);
            if self.free_data_mu_hat > mu {
                q_mu = 0.0;
            }

            // Asimov: q_mu,A
            let fixed_asimov_model = self.asimov_model.with_fixed_param(self.poi, mu);
            let init_asimov = if let Some(mut p) = last_fixed_asimov_params.clone() {
                if self.poi < p.len() {
                    p[self.poi] = mu;
                }
                p
            } else {
                fixed_asimov_model.parameter_init()
            };
            let fixed_asimov = mle.fit_minimum_from(&fixed_asimov_model, &init_asimov)?;
            if !fixed_asimov.converged {
                log::warn!(
                    "hypotest scan: fixed fit (Asimov) did not converge for mu_test={}: {} (continuing with best-found)",
                    mu,
                    fixed_asimov.message
                );
            } else {
                // Same warm-start policy as observed scan.
                last_fixed_asimov_params = Some(fixed_asimov.parameters.clone());
            }

            let llr_a = 2.0 * (fixed_asimov.fval - self.free_asimov_nll);
            let mut q_mu_a = llr_a.max(0.0);
            if self.free_asimov_mu_hat > mu {
                q_mu_a = 0.0;
            }

            let sqrtq = q_mu.sqrt();
            let sqrtq_a = q_mu_a.sqrt();

            let teststat = if sqrtq <= sqrtq_a {
                sqrtq - sqrtq_a
            } else {
                let denom = 2.0 * sqrtq_a.max(1e-16);
                (q_mu - q_mu_a) / denom
            };

            let clsb = normal_cdf(-(teststat + sqrtq_a));
            let clb = normal_cdf(-teststat);
            let cls = safe_cls(clsb, clb);

            observed_cls.push(cls);
            expected_cls.push(expected_cls_band_from_sqrtq_a(sqrtq_a));
        }

        let obs_limit = interp_limit(alpha, scan, &observed_cls)?;
        let mut exp_limits = [0.0; 5];
        for j in 0..5 {
            let band: Vec<f64> = expected_cls.iter().map(|v| v[j]).collect();
            exp_limits[j] = interp_limit(alpha, scan, &band)?;
        }

        Ok((obs_limit, exp_limits))
    }

    /// Find an observed upper limit `mu_up` such that `CLs(mu_up) = alpha` via bisection.
    ///
    /// This is the Phase 3 baseline: observed limit only (no expected bands yet).
    pub fn upper_limit_qtilde(
        &self,
        mle: &MaximumLikelihoodEstimator,
        alpha: f64,
        mut lo: f64,
        mut hi: f64,
        rtol: f64,
        max_iter: usize,
    ) -> Result<f64> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(Error::Validation(format!("alpha must be in (0,1), got {}", alpha)));
        }
        if lo < 0.0 {
            lo = 0.0;
        }
        if hi <= lo {
            return Err(Error::Validation(format!("Invalid bracket: lo={} hi={}", lo, hi)));
        }

        let cls_lo = self.hypotest_qtilde(mle, lo)?.cls;
        if cls_lo < alpha {
            return Err(Error::Validation(format!(
                "Lower bracket does not satisfy CLs(lo) >= alpha: CLs({})={} < {}",
                lo, cls_lo, alpha
            )));
        }

        let mut cls_hi = self.hypotest_qtilde(mle, hi)?.cls;
        let mut expand = 0;
        while cls_hi > alpha && expand < 50 {
            hi *= 2.0;
            cls_hi = self.hypotest_qtilde(mle, hi)?.cls;
            expand += 1;
        }
        if cls_hi > alpha {
            return Err(Error::Validation(format!(
                "Failed to bracket limit: CLs(hi)={} still > alpha={} after expansions",
                cls_hi, alpha
            )));
        }

        for _ in 0..max_iter {
            let mid = 0.5 * (lo + hi);
            let cls_mid = self.hypotest_qtilde(mle, mid)?.cls;
            if cls_mid > alpha {
                lo = mid;
            } else {
                hi = mid;
            }

            let denom = hi.abs().max(1.0);
            if ((hi - lo).abs() / denom) < rtol {
                break;
            }
        }

        Ok(0.5 * (lo + hi))
    }

    fn cls_expected_at(
        &self,
        mle: &MaximumLikelihoodEstimator,
        mu_test: f64,
        expected_idx: usize,
    ) -> Result<f64> {
        if expected_idx >= 5 {
            return Err(Error::Validation(format!("expected_idx out of range: {}", expected_idx)));
        }

        let (q_mu_a, _nll_mu_a, _n_iter_mu_a, _conv_mu_a) = qmu_like_with_free(
            mle,
            &self.asimov_model,
            self.free_asimov_nll,
            self.free_asimov_mu_hat,
            self.poi,
            mu_test,
        )?;
        let band = expected_cls_band_from_sqrtq_a(q_mu_a.sqrt());
        Ok(band[expected_idx])
    }

    fn bisection_limit<F: Fn(f64) -> Result<f64>>(
        alpha: f64,
        mut lo: f64,
        mut hi: f64,
        rtol: f64,
        max_iter: usize,
        f: F,
    ) -> Result<f64> {
        if lo < 0.0 {
            lo = 0.0;
        }
        if hi <= lo {
            return Err(Error::Validation(format!("Invalid bracket: lo={} hi={}", lo, hi)));
        }

        let flo = f(lo)? - alpha;
        if flo < 0.0 {
            return Err(Error::Validation(format!(
                "Lower bracket does not satisfy f(lo) >= alpha: f({})={} < {}",
                lo,
                flo + alpha,
                alpha
            )));
        }

        let mut fhi = f(hi)? - alpha;
        let mut expand = 0usize;
        while fhi > 0.0 && expand < 50 {
            hi *= 2.0;
            fhi = f(hi)? - alpha;
            expand += 1;
        }
        if fhi > 0.0 {
            return Err(Error::Validation("Failed to bracket limit after expansions".to_string()));
        }

        for _ in 0..max_iter {
            let mid = 0.5 * (lo + hi);
            let fmid = f(mid)? - alpha;
            if fmid > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }

            let denom = hi.abs().max(1.0);
            if ((hi - lo).abs() / denom) < rtol {
                break;
            }
        }

        Ok(0.5 * (lo + hi))
    }

    /// Compute observed and expected upper limits using bisection root-finding.
    ///
    /// This mirrors `pyhf.infer.intervals.upper_limits.upper_limit(..., scan=None)` behavior
    /// (root-finding for observed and each expected band), though NextStat uses bisection.
    pub fn upper_limits_qtilde_bisection(
        &self,
        mle: &MaximumLikelihoodEstimator,
        alpha: f64,
        lo: f64,
        hi: f64,
        rtol: f64,
        max_iter: usize,
    ) -> Result<(f64, [f64; 5])> {
        let obs = self.upper_limit_qtilde(mle, alpha, lo, hi, rtol, max_iter)?;

        let mut exp = [0.0; 5];
        for idx in 0..5 {
            let f = |mu: f64| self.cls_expected_at(mle, mu, idx);
            exp[idx] = Self::bisection_limit(alpha, lo, hi, rtol, max_iter, f)?;
        }

        Ok((obs, exp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_hypotest_qtilde_matches_pyhf_golden_simple() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let mle = MaximumLikelihoodEstimator::new();
        let ctx = AsymptoticCLsContext::new(&mle, &model).unwrap();

        // Golden values from pyhf 0.7.6:
        // pyhf.infer.hypotest(mu, data, model, test_stat="qtilde", calctype="asymptotics")
        let cases = [
            (0.0, 1.0),
            (0.5, 0.6972458106165782),
            (1.0, 0.40567153849506016),
            (2.0, 0.06802288789117743),
        ];

        for (mu, expected_cls) in cases {
            let r = ctx.hypotest_qtilde(&mle, mu).unwrap();
            let diff = (r.cls - expected_cls).abs();
            assert!(diff < 5e-6, "mu={} cls={} expected={} diff={}", mu, r.cls, expected_cls, diff);
        }
    }

    #[test]
    fn test_safe_cls_underflow_is_finite() {
        // Force clb=0 underflow and ensure we don't produce NaN/inf for CLs.
        let clb = normal_cdf(-1e6);
        let clsb = normal_cdf(-1e6 - 1.0);
        assert_eq!(clb, 0.0);
        assert_eq!(clsb, 0.0);
        let cls = safe_cls(clsb, clb);
        assert!(cls.is_finite());
        assert_eq!(cls, 0.0);
    }

    #[test]
    fn test_interp_limit_nonmonotonic_picks_first_crossing() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 0.2, 0.8, 0.0];
        let alpha = 0.5;
        let x = interp_limit(alpha, &xs, &ys).unwrap();
        // First crossing is between (0,1): y=1 -> 0.2
        let expected = 0.0 + (alpha - 1.0) / (0.2 - 1.0) * (1.0 - 0.0);
        assert!((x - expected).abs() < 1e-12, "x={} expected={}", x, expected);
    }
}
