//! Ads-native observation and response models.
//!
//! This module focuses on the parts of the ads DGP that are easy to get wrong
//! if we reuse overly simple count assumptions:
//!
//! - overdispersed conversion rates
//! - delayed conversions / right-censoring
//! - segment-level shrinkage under weak per-segment power
//! - saturating spend response primitives

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::meta_analysis::{StudyEffect, meta_random};
pub use crate::variance_reduction::{
    ArmData as CupedArmData, CovariateProvenance, CovariateTiming, CupedResult, CureResult,
    MultiCovariateArmData, VarianceReductionMethod, VarianceReductionSolver, cuped_adjust,
    cure_adjust,
};

const EPS: f64 = 1e-12;
const MIN_LOG_LAMBDA: f64 = -12.0;
const MAX_LOG_LAMBDA: f64 = 6.0;
const GRID_POINTS: usize = 256;
const GRID_PASSES: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialModel {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaBinomialModel {
    /// Construct a Beta prior from pseudo-count parameters.
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(Error::Validation("alpha must be finite and > 0".to_string()));
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(Error::Validation("beta must be finite and > 0".to_string()));
        }
        Ok(Self { alpha, beta })
    }

    pub fn fit(conversion_rates: &[f64], sample_sizes: &[usize]) -> Result<Self> {
        if conversion_rates.is_empty() {
            return Err(Error::Validation("conversion_rates must be non-empty".to_string()));
        }
        if conversion_rates.len() != sample_sizes.len() {
            return Err(Error::Validation(format!(
                "conversion_rates/sample_sizes mismatch: {} vs {}",
                conversion_rates.len(),
                sample_sizes.len()
            )));
        }
        if sample_sizes.contains(&0) {
            return Err(Error::Validation("sample_sizes must be > 0".to_string()));
        }
        if conversion_rates.iter().any(|rate| !rate.is_finite() || *rate < 0.0 || *rate > 1.0) {
            return Err(Error::Validation(
                "conversion_rates must be finite and inside [0, 1]".to_string(),
            ));
        }

        let total_n = sample_sizes.iter().map(|n| *n as f64).sum::<f64>();
        let mean = conversion_rates
            .iter()
            .zip(sample_sizes.iter())
            .map(|(rate, n)| rate * *n as f64)
            .sum::<f64>()
            / total_n;
        let mean = mean.clamp(EPS, 1.0 - EPS);

        let baseline_var = mean * (1.0 - mean);
        let mut rho = if baseline_var <= EPS {
            EPS
        } else {
            // Weighted method-of-moments on campaign rates:
            // E[(r_i - mu)^2] = mu(1-mu) * [1/n_i + ((n_i-1)/n_i) * rho].
            let weights: Vec<f64> = sample_sizes.iter().map(|n| *n as f64 / total_n).collect();
            let weighted_sq_resid = conversion_rates
                .iter()
                .zip(weights.iter())
                .map(|(rate, weight)| weight * (rate - mean).powi(2))
                .sum::<f64>();
            let weighted_binomial_noise = sample_sizes
                .iter()
                .zip(weights.iter())
                .map(|(n, weight)| weight * baseline_var / *n as f64)
                .sum::<f64>();
            let weighted_overdisp_scale = sample_sizes
                .iter()
                .zip(weights.iter())
                .map(|(n, weight)| weight * baseline_var * (*n as f64 - 1.0) / *n as f64)
                .sum::<f64>();

            if weighted_overdisp_scale <= EPS {
                EPS
            } else {
                ((weighted_sq_resid - weighted_binomial_noise) / weighted_overdisp_scale).max(EPS)
            }
        };
        if !rho.is_finite() {
            rho = EPS;
        }
        rho = rho.clamp(EPS, 1.0 - 1e-6);

        let concentration_cap = total_n.max(2.0);
        let concentration = (1.0 / rho - 1.0).clamp(2.0, concentration_cap);
        Self::new(mean * concentration, (1.0 - mean) * concentration)
    }

    /// Fit an empirical-Bayes beta prior from campaign counts.
    ///
    /// This is a counts-first convenience wrapper around [`Self::fit`], and is
    /// the preferred API when historical data arrives as `(successes, trials)`.
    pub fn fit_from_counts(successes: &[u64], trials: &[u64]) -> Result<Self> {
        if successes.is_empty() {
            return Err(Error::Validation("successes must be non-empty".to_string()));
        }
        if successes.len() != trials.len() {
            return Err(Error::Validation(format!(
                "successes/trials mismatch: {} vs {}",
                successes.len(),
                trials.len()
            )));
        }

        let mut rates = Vec::with_capacity(successes.len());
        let mut sample_sizes = Vec::with_capacity(successes.len());
        for (idx, (&y, &n)) in successes.iter().zip(trials.iter()).enumerate() {
            if n == 0 {
                return Err(Error::Validation(format!("trials[{idx}] must be > 0")));
            }
            if y > n {
                return Err(Error::Validation(format!(
                    "successes[{idx}] must be <= trials[{idx}], got {y} > {n}"
                )));
            }
            let n_usize = usize::try_from(n).map_err(|_| {
                Error::Validation(format!(
                    "trials[{idx}]={n} exceeds supported platform usize range"
                ))
            })?;
            rates.push(y as f64 / n as f64);
            sample_sizes.push(n_usize);
        }

        Self::fit(&rates, &sample_sizes)
    }

    /// Mean of the latent conversion-rate prior.
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the latent conversion-rate prior.
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }

    /// Beta-binomial overdispersion parameter `rho = 1 / (alpha + beta + 1)`.
    pub fn overdispersion(&self) -> f64 {
        1.0 / (self.alpha + self.beta + 1.0)
    }

    /// Conjugate posterior after observing `successes` out of `trials`.
    pub fn posterior(&self, successes: u64, trials: u64) -> Result<Self> {
        if successes > trials {
            return Err(Error::Validation(format!(
                "successes must be <= trials, got {} > {}",
                successes, trials
            )));
        }
        Self::new(self.alpha + successes as f64, self.beta + (trials - successes) as f64)
    }

    pub fn predictive_mean(&self, trials: u64) -> f64 {
        trials as f64 * self.mean()
    }

    pub fn predictive_variance(&self, trials: u64) -> f64 {
        let n = trials as f64;
        let sum = self.alpha + self.beta;
        n * self.alpha * self.beta * (sum + n) / (sum * sum * (sum + 1.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DelayCorrectionModel {
    pub lambda: f64,
    pub lambda_se: Option<f64>,
}

impl DelayCorrectionModel {
    pub fn new(lambda: f64, lambda_se: Option<f64>) -> Result<Self> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(Error::Validation("lambda must be finite and > 0".to_string()));
        }
        if let Some(se) = lambda_se
            && (!se.is_finite() || se < 0.0)
        {
            return Err(Error::Validation("lambda_se must be finite and >= 0".to_string()));
        }
        Ok(Self { lambda, lambda_se })
    }

    pub fn observed_fraction(&self, window_days: f64) -> Result<f64> {
        if !window_days.is_finite() || window_days < 0.0 {
            return Err(Error::Validation("window_days must be finite and >= 0".to_string()));
        }
        if window_days == 0.0 {
            return Ok(0.0);
        }
        Ok((1.0 - (-self.lambda * window_days).exp()).clamp(0.0, 1.0))
    }

    pub fn correct(&self, observed_count: f64, window_days: f64) -> Result<(f64, f64)> {
        if !observed_count.is_finite() || observed_count < 0.0 {
            return Err(Error::Validation("observed_count must be finite and >= 0".to_string()));
        }
        let fraction = self.observed_fraction(window_days)?;
        if fraction <= 0.0 {
            return Err(Error::Validation(
                "window_days must be > 0 so that an observed fraction exists".to_string(),
            ));
        }
        let corrected = observed_count / fraction;
        let poisson_var = observed_count / (fraction * fraction);

        let lambda_var = self.lambda_se.unwrap_or(0.0).powi(2);
        let f_prime = window_days * (-self.lambda * window_days).exp();
        let inv_f_prime = -f_prime / (fraction * fraction);
        let delay_var = (observed_count * inv_f_prime).powi(2) * lambda_var;
        let uncertainty = (poisson_var + delay_var).sqrt();

        Ok((corrected, uncertainty))
    }

    pub fn fit_from_lag_buckets(buckets: &[(f64, usize)]) -> Result<Self> {
        let weighted_buckets: Vec<_> =
            buckets.iter().map(|(upper, count)| (*upper, *count as f64)).collect();
        Self::fit_from_weighted_lag_buckets(&weighted_buckets)
    }

    pub fn fit_from_weighted_lag_buckets(buckets: &[(f64, f64)]) -> Result<Self> {
        validate_weighted_lag_buckets(buckets)?;
        let (log_lambda, nll) = fit_log_lambda_interval_grid(buckets)?;
        let lambda = log_lambda.exp();
        let lambda_se = approximate_lambda_se(buckets, log_lambda, nll);
        Self::new(lambda, lambda_se)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentLiftObservation {
    pub segment_id: String,
    pub observed_lift: f64,
    pub standard_error: f64,
}

impl SegmentLiftObservation {
    pub fn new(
        segment_id: impl Into<String>,
        observed_lift: f64,
        standard_error: f64,
    ) -> Result<Self> {
        let segment_id = segment_id.into();
        if segment_id.trim().is_empty() {
            return Err(Error::Validation("segment_id must be non-empty".to_string()));
        }
        if !observed_lift.is_finite() {
            return Err(Error::Validation("observed_lift must be finite".to_string()));
        }
        if !standard_error.is_finite() || standard_error <= 0.0 {
            return Err(Error::Validation("standard_error must be finite and > 0".to_string()));
        }
        Ok(Self { segment_id, observed_lift, standard_error })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentLiftPosterior {
    pub segment_id: String,
    pub observed_lift: f64,
    pub standard_error: f64,
    pub posterior_mean: f64,
    pub posterior_sd: f64,
    pub shrinkage_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HierarchicalSegmentLiftSummary {
    pub global_estimate: f64,
    pub global_se: f64,
    pub tau_squared: f64,
    pub conf_level: f64,
    pub segments: Vec<SegmentLiftPosterior>,
}

pub fn hierarchical_segment_lift_summary(
    observations: &[SegmentLiftObservation],
    conf_level: f64,
) -> Result<HierarchicalSegmentLiftSummary> {
    if observations.len() < 2 {
        return Err(Error::Validation(
            "hierarchical segment shrinkage requires at least 2 observations".to_string(),
        ));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation("conf_level must be in (0, 1)".to_string()));
    }

    let studies: Vec<StudyEffect> = observations
        .iter()
        .map(|obs| StudyEffect {
            label: obs.segment_id.clone(),
            estimate: obs.observed_lift,
            se: obs.standard_error,
        })
        .collect();
    let meta = meta_random(&studies, conf_level)?;
    let tau_squared = meta.heterogeneity.tau_squared.max(0.0);

    let segments = observations
        .iter()
        .map(|obs| {
            if tau_squared <= EPS {
                SegmentLiftPosterior {
                    segment_id: obs.segment_id.clone(),
                    observed_lift: obs.observed_lift,
                    standard_error: obs.standard_error,
                    posterior_mean: meta.estimate,
                    posterior_sd: 0.0,
                    shrinkage_weight: 0.0,
                }
            } else {
                let obs_var = obs.standard_error * obs.standard_error;
                let denom = 1.0 / obs_var + 1.0 / tau_squared;
                let posterior_var = 1.0 / denom;
                let posterior_mean =
                    posterior_var * (obs.observed_lift / obs_var + meta.estimate / tau_squared);
                let shrinkage_weight = tau_squared / (tau_squared + obs_var);

                SegmentLiftPosterior {
                    segment_id: obs.segment_id.clone(),
                    observed_lift: obs.observed_lift,
                    standard_error: obs.standard_error,
                    posterior_mean,
                    posterior_sd: posterior_var.sqrt(),
                    shrinkage_weight,
                }
            }
        })
        .collect();

    Ok(HierarchicalSegmentLiftSummary {
        global_estimate: meta.estimate,
        global_se: meta.se,
        tau_squared,
        conf_level,
        segments,
    })
}

pub fn hill(x: f64, ec: f64, slope: f64) -> Result<f64> {
    if !x.is_finite() || x < 0.0 {
        return Err(Error::Validation("x must be finite and >= 0".to_string()));
    }
    if !ec.is_finite() || ec <= 0.0 {
        return Err(Error::Validation("ec must be finite and > 0".to_string()));
    }
    if !slope.is_finite() || slope <= 0.0 {
        return Err(Error::Validation("slope must be finite and > 0".to_string()));
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    Ok(1.0 / (1.0 + (x / ec).powf(-slope)))
}

pub fn adstock_geometric(spend: &[f64], decay: f64) -> Result<Vec<f64>> {
    if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
        return Err(Error::Validation("decay must be finite and inside [0, 1]".to_string()));
    }
    if spend.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(Error::Validation("spend values must be finite and >= 0".to_string()));
    }
    if spend.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = vec![0.0; spend.len()];
    result[0] = spend[0];
    for idx in 1..spend.len() {
        result[idx] = spend[idx] + decay * result[idx - 1];
    }
    Ok(result)
}

fn validate_weighted_lag_buckets(buckets: &[(f64, f64)]) -> Result<()> {
    if buckets.is_empty() {
        return Err(Error::Validation("lag buckets must be non-empty".to_string()));
    }
    let mut prev = 0.0;
    let mut total = 0.0;
    for (idx, (upper, count)) in buckets.iter().enumerate() {
        if !upper.is_finite() || *upper <= prev {
            return Err(Error::Validation(format!(
                "bucket {idx} upper bound must be finite and strictly increasing"
            )));
        }
        if !count.is_finite() || *count < 0.0 {
            return Err(Error::Validation(format!("bucket {idx} weight must be finite and >= 0")));
        }
        total += *count;
        prev = *upper;
    }
    if total <= 0.0 {
        return Err(Error::Validation(
            "lag buckets must contain at least one observation".to_string(),
        ));
    }
    Ok(())
}

fn interval_bucket_probability(lambda: f64, lower: f64, upper: f64) -> f64 {
    ((-lambda * lower).exp() - (-lambda * upper).exp()).max(EPS)
}

fn interval_nll_from_log_lambda(log_lambda: f64, buckets: &[(f64, f64)]) -> f64 {
    let lambda = log_lambda.exp();
    let mut lower = 0.0;
    let mut nll = 0.0;
    for (upper, count) in buckets {
        if *count > 0.0 {
            let prob = interval_bucket_probability(lambda, lower, *upper);
            nll -= *count * prob.ln();
        }
        lower = *upper;
    }
    nll
}

fn fit_log_lambda_interval_grid(buckets: &[(f64, f64)]) -> Result<(f64, f64)> {
    let mut lo = MIN_LOG_LAMBDA;
    let mut hi = MAX_LOG_LAMBDA;
    let mut best_log = 0.0;
    let mut best_nll = f64::INFINITY;

    for _ in 0..GRID_PASSES {
        let step = (hi - lo) / (GRID_POINTS as f64 - 1.0);
        for idx in 0..GRID_POINTS {
            let candidate = lo + idx as f64 * step;
            let nll = interval_nll_from_log_lambda(candidate, buckets);
            if nll < best_nll {
                best_nll = nll;
                best_log = candidate;
            }
        }
        lo = best_log - step;
        hi = best_log + step;
    }

    if !best_nll.is_finite() {
        return Err(Error::Computation(
            "failed to fit delay correction model from lag buckets".to_string(),
        ));
    }
    Ok((best_log, best_nll))
}

fn approximate_lambda_se(
    buckets: &[(f64, f64)],
    best_log_lambda: f64,
    center_nll: f64,
) -> Option<f64> {
    let h = 1e-3;
    let left = interval_nll_from_log_lambda(best_log_lambda - h, buckets);
    let right = interval_nll_from_log_lambda(best_log_lambda + h, buckets);
    let second = (left - 2.0 * center_nll + right) / (h * h);
    if !second.is_finite() || second <= EPS {
        return None;
    }
    let se_log_lambda = (1.0 / second).sqrt();
    Some(best_log_lambda.exp() * se_log_lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_binomial_posterior_updates_counts_exactly() {
        let prior = BetaBinomialModel::new(2.0, 18.0).unwrap();
        let posterior = prior.posterior(5, 20).unwrap();

        assert!((posterior.alpha - 7.0).abs() < 1e-12);
        assert!((posterior.beta - 33.0).abs() < 1e-12);
    }

    #[test]
    fn beta_binomial_fit_recovers_mean_and_detects_overdispersion() {
        let rates = [0.012, 0.019, 0.010, 0.024, 0.021, 0.015];
        let sample_sizes = [900, 1200, 800, 1100, 950, 1050];

        let model = BetaBinomialModel::fit(&rates, &sample_sizes).unwrap();
        let weighted_mean =
            rates.iter().zip(sample_sizes.iter()).map(|(rate, n)| rate * *n as f64).sum::<f64>()
                / sample_sizes.iter().map(|n| *n as f64).sum::<f64>();

        assert!((model.mean() - weighted_mean).abs() < 0.01);
        assert!(model.overdispersion() > 0.0);
    }

    #[test]
    fn beta_binomial_fit_downweights_tiny_noisy_campaigns() {
        let rates = [0.50, 0.01];
        let sample_sizes = [10, 10_000];

        let model = BetaBinomialModel::fit(&rates, &sample_sizes).unwrap();

        assert!(model.mean() < 0.02);
        assert!(
            model.overdispersion() < 0.1,
            "tiny noisy campaign should not force massive overdispersion; got rho={}",
            model.overdispersion()
        );
    }

    #[test]
    fn beta_binomial_fit_from_counts_matches_rate_api() {
        let successes = [12_u64, 19, 10, 24];
        let trials = [900_u64, 1200, 800, 1100];
        let rates = [12.0 / 900.0, 19.0 / 1200.0, 10.0 / 800.0, 24.0 / 1100.0];
        let sample_sizes = [900_usize, 1200, 800, 1100];

        let by_counts = BetaBinomialModel::fit_from_counts(&successes, &trials).unwrap();
        let by_rates = BetaBinomialModel::fit(&rates, &sample_sizes).unwrap();

        assert!((by_counts.alpha - by_rates.alpha).abs() < 1e-12);
        assert!((by_counts.beta - by_rates.beta).abs() < 1e-12);
    }

    #[test]
    fn beta_binomial_fit_from_counts_rejects_invalid_counts() {
        let err = BetaBinomialModel::fit_from_counts(&[11], &[10]).unwrap_err();
        match err {
            Error::Validation(message) => assert!(message.contains("successes[0]")),
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn delay_correction_fit_from_lag_buckets_recovers_rate() {
        let lambda = 0.35;
        let buckets = [(1.0, 296usize), (2.0, 193), (4.0, 252), (7.0, 173), (14.0, 80)];

        let fitted = DelayCorrectionModel::fit_from_lag_buckets(&buckets).unwrap();

        assert!((fitted.lambda - lambda).abs() < 0.08, "expected ~{lambda}, got {}", fitted.lambda);
        assert!(fitted.lambda_se.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn delay_correction_fit_from_weighted_lag_buckets_handles_fractional_counts() {
        let lambda = 0.35;
        let buckets = [(1.0, 296.5), (2.0, 193.25), (4.0, 252.0), (7.0, 173.75), (14.0, 80.5)];

        let fitted = DelayCorrectionModel::fit_from_weighted_lag_buckets(&buckets).unwrap();

        assert!((fitted.lambda - lambda).abs() < 0.08, "expected ~{lambda}, got {}", fitted.lambda);
        assert!(fitted.lambda_se.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn delay_correction_inflates_observed_counts_for_short_windows() {
        let model = DelayCorrectionModel::new(0.4, Some(0.05)).unwrap();

        let (corrected, uncertainty) = model.correct(42.0, 3.0).unwrap();

        assert!(corrected > 42.0);
        assert!(uncertainty > 0.0);
    }

    #[test]
    fn delay_observed_fraction_is_zero_at_time_zero_and_cannot_be_corrected() {
        let model = DelayCorrectionModel::new(0.4, Some(0.05)).unwrap();

        assert_eq!(model.observed_fraction(0.0).unwrap(), 0.0);
        assert!(model.correct(42.0, 0.0).is_err());
    }

    #[test]
    fn hierarchical_shrinkage_pulls_noisy_segments_toward_global_mean() {
        let observations = vec![
            SegmentLiftObservation::new("desktop_us", 0.08, 0.02).unwrap(),
            SegmentLiftObservation::new("mobile_us", 0.06, 0.03).unwrap(),
            SegmentLiftObservation::new("tablet_us", 0.35, 0.20).unwrap(),
        ];

        let summary = hierarchical_segment_lift_summary(&observations, 0.95).unwrap();
        let tablet =
            summary.segments.iter().find(|segment| segment.segment_id == "tablet_us").unwrap();

        assert!(tablet.posterior_mean < tablet.observed_lift);
        assert!(tablet.posterior_mean > summary.global_estimate - 0.05);
    }

    #[test]
    fn hill_is_monotone_and_adstock_carries_memory() {
        let low = hill(10.0, 50.0, 1.2).unwrap();
        let high = hill(100.0, 50.0, 1.2).unwrap();
        let transformed = adstock_geometric(&[100.0, 0.0, 0.0], 0.5).unwrap();

        assert!(high > low);
        assert_eq!(transformed, vec![100.0, 50.0, 25.0]);
    }
}
