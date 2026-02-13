//! Pharmacodynamic (PD) models.
//!
//! # Direct-effect models (algebraic)
//!
//! - [`EmaxModel`] — `E = E0 + Emax·C/(EC50 + C)`
//! - [`SigmoidEmaxModel`] — `E = E0 + Emax·C^γ/(EC50^γ + C^γ)` (Hill equation)
//!
//! # Indirect-response models (ODE-based)
//!
//! Four canonical types ([Dayneka et al., 1993](https://doi.org/10.1007/BF01062336)):
//!
//! - [`IndirectResponseType::InhibitProduction`] (Type I) — drug inhibits kin
//! - [`IndirectResponseType::InhibitLoss`] (Type II) — drug inhibits kout
//! - [`IndirectResponseType::StimulateProduction`] (Type III) — drug stimulates kin
//! - [`IndirectResponseType::StimulateLoss`] (Type IV) — drug stimulates kout
//!
//! All IDR models use the adaptive ODE solver ([`rk45`](crate::ode_adaptive::rk45))
//! and accept a drug concentration time-course as input.
//!
//! # PK/PD linking
//!
//! Use [`PkPdLink`] to connect a PK concentration profile to a PD model.
//! The link interpolates the PK profile at arbitrary times for the ODE solver.

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::ode_adaptive::{OdeOptions, OdeSystem, rk45};
use crate::pk::ErrorModel;

// ---------------------------------------------------------------------------
// Direct-effect models
// ---------------------------------------------------------------------------

/// Emax (maximum effect) model.
///
/// `E(C) = E0 + Emax · C / (EC50 + C)`
///
/// - `E0`: baseline effect (no drug)
/// - `Emax`: maximum drug-induced effect
/// - `EC50`: concentration producing 50% of Emax
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmaxModel {
    pub e0: f64,
    pub emax: f64,
    pub ec50: f64,
}

impl EmaxModel {
    pub fn new(e0: f64, emax: f64, ec50: f64) -> Result<Self> {
        if !ec50.is_finite() || ec50 <= 0.0 {
            return Err(Error::Validation("EC50 must be finite and > 0".into()));
        }
        Ok(Self { e0, emax, ec50 })
    }

    /// Predict effect at concentration `c`.
    #[inline]
    pub fn predict(&self, c: f64) -> f64 {
        let c = c.max(0.0);
        self.e0 + self.emax * c / (self.ec50 + c)
    }

    /// Gradient `dE/d(E0, Emax, EC50)` at concentration `c`.
    #[inline]
    pub fn gradient(&self, c: f64) -> [f64; 3] {
        let c = c.max(0.0);
        let denom = self.ec50 + c;
        let frac = c / denom;
        [
            1.0,                              // dE/dE0
            frac,                             // dE/dEmax
            -self.emax * c / (denom * denom), // dE/dEC50
        ]
    }

    /// Predict effects for a vector of concentrations.
    pub fn predict_vec(&self, conc: &[f64]) -> Vec<f64> {
        conc.iter().map(|&c| self.predict(c)).collect()
    }

    /// NLL for observations `obs` at concentrations `conc` under error model.
    pub fn nll(&self, conc: &[f64], obs: &[f64], error: &ErrorModel) -> Result<f64> {
        if conc.len() != obs.len() {
            return Err(Error::Validation("conc.len() != obs.len()".into()));
        }
        let mut nll = 0.0;
        for (&c, &y) in conc.iter().zip(obs.iter()) {
            let pred = self.predict(c);
            nll += error.nll_obs(y, pred);
        }
        Ok(nll)
    }
}

/// Sigmoid Emax (Hill) model.
///
/// `E(C) = E0 + Emax · C^γ / (EC50^γ + C^γ)`
///
/// - `gamma` (γ, Hill coefficient): controls steepness.
///   - γ = 1: standard Emax
///   - γ > 1: steeper (switch-like)
///   - 0 < γ < 1: shallower
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SigmoidEmaxModel {
    pub e0: f64,
    pub emax: f64,
    pub ec50: f64,
    pub gamma: f64,
}

impl SigmoidEmaxModel {
    pub fn new(e0: f64, emax: f64, ec50: f64, gamma: f64) -> Result<Self> {
        if !ec50.is_finite() || ec50 <= 0.0 {
            return Err(Error::Validation("EC50 must be finite and > 0".into()));
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(Error::Validation("gamma must be finite and > 0".into()));
        }
        Ok(Self { e0, emax, ec50, gamma })
    }

    /// Predict effect at concentration `c`.
    #[inline]
    pub fn predict(&self, c: f64) -> f64 {
        let c = c.max(0.0);
        let cg = c.powf(self.gamma);
        let ecg = self.ec50.powf(self.gamma);
        self.e0 + self.emax * cg / (ecg + cg)
    }

    /// Gradient `dE/d(E0, Emax, EC50, gamma)` at concentration `c`.
    pub fn gradient(&self, c: f64) -> [f64; 4] {
        let c = c.max(1e-30);
        let g = self.gamma;
        let cg = c.powf(g);
        let ecg = self.ec50.powf(g);
        let denom = ecg + cg;
        let frac = cg / denom;

        let de_de0 = 1.0;
        let de_demax = frac;
        // dE/dEC50 = Emax * cg * (-g * EC50^(g-1)) / denom^2
        let de_dec50 = -self.emax * cg * g * self.ec50.powf(g - 1.0) / (denom * denom);
        // dE/dgamma: d(cg)/dg = cg*ln(c), d(ecg)/dg = ecg*ln(ec50)
        let dcg = cg * c.ln();
        let decg = ecg * self.ec50.ln();
        let de_dgamma = self.emax * (dcg * ecg - cg * decg) / (denom * denom);

        [de_de0, de_demax, de_dec50, de_dgamma]
    }

    /// Predict effects for a vector of concentrations.
    pub fn predict_vec(&self, conc: &[f64]) -> Vec<f64> {
        conc.iter().map(|&c| self.predict(c)).collect()
    }

    /// NLL for observations `obs` at concentrations `conc` under error model.
    pub fn nll(&self, conc: &[f64], obs: &[f64], error: &ErrorModel) -> Result<f64> {
        if conc.len() != obs.len() {
            return Err(Error::Validation("conc.len() != obs.len()".into()));
        }
        let mut nll = 0.0;
        for (&c, &y) in conc.iter().zip(obs.iter()) {
            let pred = self.predict(c);
            nll += error.nll_obs(y, pred);
        }
        Ok(nll)
    }
}

// ---------------------------------------------------------------------------
// Indirect-response models
// ---------------------------------------------------------------------------

/// Indirect response model type.
///
/// All four types share the baseline equation `dR/dt = kin - kout·R`
/// with steady state `R0 = kin/kout`. Drug action modifies either
/// production (kin) or loss (kout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndirectResponseType {
    /// Type I: Drug **inhibits production** (reduces kin).
    /// `dR/dt = kin·(1 - Imax·C/(IC50+C)) - kout·R`
    InhibitProduction,
    /// Type II: Drug **inhibits loss** (reduces kout).
    /// `dR/dt = kin - kout·(1 - Imax·C/(IC50+C))·R`
    InhibitLoss,
    /// Type III: Drug **stimulates production** (increases kin).
    /// `dR/dt = kin·(1 + Emax·C/(EC50+C)) - kout·R`
    StimulateProduction,
    /// Type IV: Drug **stimulates loss** (increases kout).
    /// `dR/dt = kin - kout·(1 + Emax·C/(EC50+C))·R`
    StimulateLoss,
}

/// Indirect response model configuration.
///
/// At steady state without drug: `R0 = kin / kout`.
/// The user specifies `kin` and `kout` (or equivalently `R0` and `kout`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndirectResponseModel {
    pub idr_type: IndirectResponseType,
    /// Zero-order production rate.
    pub kin: f64,
    /// First-order loss rate constant (h⁻¹).
    pub kout: f64,
    /// Maximum drug effect (Imax for inhibition, Emax for stimulation).
    /// For inhibition types: 0 < imax ≤ 1 (complete inhibition at 1).
    /// For stimulation types: emax > 0 (unbounded).
    pub max_effect: f64,
    /// Concentration at 50% of max effect (IC50 or EC50).
    pub c50: f64,
}

impl IndirectResponseModel {
    pub fn new(
        idr_type: IndirectResponseType,
        kin: f64,
        kout: f64,
        max_effect: f64,
        c50: f64,
    ) -> Result<Self> {
        if !kin.is_finite() || kin <= 0.0 {
            return Err(Error::Validation("kin must be finite and > 0".into()));
        }
        if !kout.is_finite() || kout <= 0.0 {
            return Err(Error::Validation("kout must be finite and > 0".into()));
        }
        if !c50.is_finite() || c50 <= 0.0 {
            return Err(Error::Validation("C50 must be finite and > 0".into()));
        }
        if !max_effect.is_finite() || max_effect <= 0.0 {
            return Err(Error::Validation("max_effect must be finite and > 0".into()));
        }
        match idr_type {
            IndirectResponseType::InhibitProduction | IndirectResponseType::InhibitLoss => {
                if max_effect > 1.0 {
                    return Err(Error::Validation(
                        "Imax must be in (0, 1] for inhibition models".into(),
                    ));
                }
            }
            _ => {}
        }
        Ok(Self { idr_type, kin, kout, max_effect, c50 })
    }

    /// Baseline response (steady state without drug).
    #[inline]
    pub fn baseline(&self) -> f64 {
        self.kin / self.kout
    }

    /// Drug effect factor at concentration `c`.
    #[inline]
    fn drug_factor(&self, c: f64) -> f64 {
        let c = c.max(0.0);
        self.max_effect * c / (self.c50 + c)
    }

    /// Compute `dR/dt` given current response `r` and drug concentration `c`.
    #[inline]
    pub fn drdt(&self, r: f64, c: f64) -> f64 {
        let h = self.drug_factor(c);
        match self.idr_type {
            IndirectResponseType::InhibitProduction => self.kin * (1.0 - h) - self.kout * r,
            IndirectResponseType::InhibitLoss => self.kin - self.kout * (1.0 - h) * r,
            IndirectResponseType::StimulateProduction => self.kin * (1.0 + h) - self.kout * r,
            IndirectResponseType::StimulateLoss => self.kin - self.kout * (1.0 + h) * r,
        }
    }

    /// Simulate the response time-course given a drug concentration profile.
    ///
    /// `conc_profile` is a sorted list of `(time, concentration)` pairs.
    /// Returns the response at each time point in `output_times`.
    pub fn simulate(
        &self,
        conc_profile: &[(f64, f64)],
        output_times: &[f64],
        r0: Option<f64>,
        opts: Option<&OdeOptions>,
    ) -> Result<Vec<f64>> {
        if conc_profile.is_empty() {
            return Err(Error::Validation("conc_profile must be non-empty".into()));
        }
        if output_times.is_empty() {
            return Ok(Vec::new());
        }

        let baseline = r0.unwrap_or_else(|| self.baseline());
        let default_opts = OdeOptions::default();
        let opts = opts.unwrap_or(&default_opts);

        let link = PkPdLink::new(conc_profile);
        let sys = IdrOdeSystem { model: self, link: &link };

        let t0 = conc_profile[0].0;
        let t1 = *output_times.last().unwrap();
        let y0 = [baseline];

        let sol = rk45(&sys, &y0, t0, t1, opts)?;

        // Interpolate at output times
        let mut result = Vec::with_capacity(output_times.len());
        let mut idx = 0;
        for &tq in output_times {
            while idx + 1 < sol.t.len() && sol.t[idx + 1] < tq {
                idx += 1;
            }
            if idx + 1 >= sol.t.len() {
                result.push(sol.y.last().unwrap()[0]);
                continue;
            }
            let ta = sol.t[idx];
            let tb = sol.t[idx + 1];
            let frac = if (tb - ta).abs() < 1e-30 { 0.0 } else { (tq - ta) / (tb - ta) };
            result.push(sol.y[idx][0] + frac * (sol.y[idx + 1][0] - sol.y[idx][0]));
        }
        Ok(result)
    }

    /// NLL for response observations at given times.
    pub fn nll(
        &self,
        conc_profile: &[(f64, f64)],
        obs_times: &[f64],
        obs_values: &[f64],
        error: &ErrorModel,
        r0: Option<f64>,
    ) -> Result<f64> {
        if obs_times.len() != obs_values.len() {
            return Err(Error::Validation("obs_times.len() != obs_values.len()".into()));
        }
        let predicted = self.simulate(conc_profile, obs_times, r0, None)?;
        let mut nll = 0.0;
        for ((&y, &pred), _) in obs_values.iter().zip(predicted.iter()).zip(obs_times.iter()) {
            nll += error.nll_obs(y, pred);
        }
        Ok(nll)
    }
}

// ---------------------------------------------------------------------------
// PK/PD linking (concentration interpolation)
// ---------------------------------------------------------------------------

/// Linear-interpolation link between a PK concentration profile and a PD model.
///
/// Accepts sorted `(time, concentration)` pairs and provides `conc_at(t)`.
#[derive(Debug, Clone)]
pub struct PkPdLink {
    times: Vec<f64>,
    concs: Vec<f64>,
}

impl PkPdLink {
    pub fn new(profile: &[(f64, f64)]) -> Self {
        let (times, concs): (Vec<_>, Vec<_>) = profile.iter().copied().unzip();
        Self { times, concs }
    }

    /// Interpolate concentration at time `t`.
    #[inline]
    pub fn conc_at(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        if t <= self.times[0] {
            return self.concs[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.concs.last().unwrap();
        }
        // Binary search
        let idx = match self
            .times
            .binary_search_by(|a| a.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => return self.concs[i],
            Err(i) => i - 1,
        };
        let ta = self.times[idx];
        let tb = self.times[idx + 1];
        let frac = (t - ta) / (tb - ta);
        self.concs[idx] + frac * (self.concs[idx + 1] - self.concs[idx])
    }
}

// ---------------------------------------------------------------------------
// ODE system adapter for IDR models
// ---------------------------------------------------------------------------

struct IdrOdeSystem<'a> {
    model: &'a IndirectResponseModel,
    link: &'a PkPdLink,
}

impl<'a> OdeSystem for IdrOdeSystem<'a> {
    fn ndim(&self) -> usize {
        1
    }

    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        let c = self.link.conc_at(t);
        dydt[0] = self.model.drdt(y[0], c);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Emax ---

    #[test]
    fn emax_at_zero_concentration() {
        let m = EmaxModel::new(5.0, 10.0, 2.0).unwrap();
        assert!((m.predict(0.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn emax_at_ec50() {
        let m = EmaxModel::new(0.0, 10.0, 5.0).unwrap();
        // At C = EC50: E = Emax/2 = 5.0
        assert!((m.predict(5.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn emax_at_high_concentration() {
        let m = EmaxModel::new(0.0, 10.0, 5.0).unwrap();
        // At C >> EC50: E → Emax = 10.0
        assert!((m.predict(1e6) - 10.0).abs() < 0.01);
    }

    #[test]
    fn emax_gradient() {
        let m = EmaxModel::new(1.0, 8.0, 3.0).unwrap();
        let c = 5.0;
        let g = m.gradient(c);
        // Numerical check
        let eps = 1e-7;
        let de0 = (EmaxModel::new(1.0 + eps, 8.0, 3.0).unwrap().predict(c)
            - EmaxModel::new(1.0 - eps, 8.0, 3.0).unwrap().predict(c))
            / (2.0 * eps);
        assert!((g[0] - de0).abs() < 1e-5, "dE/dE0: {} vs {}", g[0], de0);
    }

    #[test]
    fn emax_nll() {
        let m = EmaxModel::new(0.0, 10.0, 5.0).unwrap();
        let conc = vec![0.0, 5.0, 100.0];
        let obs = vec![0.1, 5.2, 9.8];
        let err = ErrorModel::Additive(1.0);
        let nll = m.nll(&conc, &obs, &err).unwrap();
        assert!(nll.is_finite() && nll > 0.0);
    }

    #[test]
    fn emax_invalid_ec50() {
        assert!(EmaxModel::new(0.0, 10.0, 0.0).is_err());
        assert!(EmaxModel::new(0.0, 10.0, -1.0).is_err());
    }

    // --- Sigmoid Emax ---

    #[test]
    fn sigmoid_emax_gamma_one_equals_emax() {
        let emax = EmaxModel::new(1.0, 10.0, 5.0).unwrap();
        let sig = SigmoidEmaxModel::new(1.0, 10.0, 5.0, 1.0).unwrap();
        for c in [0.0, 1.0, 5.0, 10.0, 100.0] {
            assert!(
                (emax.predict(c) - sig.predict(c)).abs() < 1e-10,
                "c={c}: {} vs {}",
                emax.predict(c),
                sig.predict(c)
            );
        }
    }

    #[test]
    fn sigmoid_emax_steepness() {
        let shallow = SigmoidEmaxModel::new(0.0, 10.0, 5.0, 0.5).unwrap();
        let steep = SigmoidEmaxModel::new(0.0, 10.0, 5.0, 3.0).unwrap();
        // At C = 2 (below EC50): steep curve should give less effect
        assert!(steep.predict(2.0) < shallow.predict(2.0));
        // At C = 10 (above EC50): steep curve should give more effect
        assert!(steep.predict(10.0) > shallow.predict(10.0));
    }

    #[test]
    fn sigmoid_emax_at_ec50() {
        // At C = EC50, effect should be Emax/2 regardless of gamma
        for gamma in [0.5, 1.0, 2.0, 5.0] {
            let m = SigmoidEmaxModel::new(0.0, 10.0, 5.0, gamma).unwrap();
            assert!(
                (m.predict(5.0) - 5.0).abs() < 1e-10,
                "gamma={gamma}: predict(EC50)={} expected 5.0",
                m.predict(5.0)
            );
        }
    }

    #[test]
    fn sigmoid_emax_invalid() {
        assert!(SigmoidEmaxModel::new(0.0, 10.0, 5.0, 0.0).is_err());
        assert!(SigmoidEmaxModel::new(0.0, 10.0, 0.0, 1.0).is_err());
    }

    // --- Indirect response ---

    #[test]
    fn idr_baseline() {
        let m = IndirectResponseModel::new(
            IndirectResponseType::StimulateProduction,
            2.0,
            0.1,
            3.0,
            5.0,
        )
        .unwrap();
        assert!((m.baseline() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn idr_type1_inhibit_production() {
        let m =
            IndirectResponseModel::new(IndirectResponseType::InhibitProduction, 1.0, 0.1, 0.8, 5.0)
                .unwrap();
        // With drug at high concentration, production is inhibited
        // dR/dt = kin*(1 - 0.8) - kout*R = 0.2 - 0.1*R → R_ss = 2.0 (baseline = 10)
        // So response should decrease
        let conc_profile: Vec<(f64, f64)> = (0..=100)
            .map(|i| (i as f64, 100.0)) // constant high drug
            .collect();
        let times: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let response = m.simulate(&conc_profile, &times, None, None).unwrap();

        // Should start at baseline (10) and decrease toward 2.0
        assert!((response[0] - 10.0).abs() < 0.5, "should start at baseline");
        let final_r = *response.last().unwrap();
        assert!(final_r < 5.0, "should decrease from baseline: got {final_r}");
    }

    #[test]
    fn idr_type3_stimulate_production() {
        let m = IndirectResponseModel::new(
            IndirectResponseType::StimulateProduction,
            1.0,
            0.1,
            2.0,
            5.0,
        )
        .unwrap();
        // With drug at high concentration, production is stimulated
        // dR/dt = kin*(1 + 2.0) - kout*R = 3.0 - 0.1*R → R_ss = 30 (baseline = 10)
        let conc_profile: Vec<(f64, f64)> = (0..=200).map(|i| (i as f64, 100.0)).collect();
        let times: Vec<f64> = (0..=200).map(|i| i as f64).collect();
        let response = m.simulate(&conc_profile, &times, None, None).unwrap();

        let final_r = *response.last().unwrap();
        assert!(final_r > 20.0, "should increase from baseline 10 toward 30: got {final_r}");
    }

    #[test]
    fn idr_type2_inhibit_loss() {
        let m = IndirectResponseModel::new(IndirectResponseType::InhibitLoss, 1.0, 0.1, 0.9, 5.0)
            .unwrap();
        // Drug inhibits loss → response should increase
        let conc_profile: Vec<(f64, f64)> = (0..=200).map(|i| (i as f64, 100.0)).collect();
        let times: Vec<f64> = (0..=200).map(|i| i as f64).collect();
        let response = m.simulate(&conc_profile, &times, None, None).unwrap();

        let final_r = *response.last().unwrap();
        assert!(final_r > 15.0, "inhibit loss should increase response: got {final_r}");
    }

    #[test]
    fn idr_type4_stimulate_loss() {
        let m = IndirectResponseModel::new(IndirectResponseType::StimulateLoss, 1.0, 0.1, 2.0, 5.0)
            .unwrap();
        // Drug stimulates loss → response should decrease
        let conc_profile: Vec<(f64, f64)> = (0..=200).map(|i| (i as f64, 100.0)).collect();
        let times: Vec<f64> = (0..=200).map(|i| i as f64).collect();
        let response = m.simulate(&conc_profile, &times, None, None).unwrap();

        let final_r = *response.last().unwrap();
        assert!(final_r < 5.0, "stimulate loss should decrease response: got {final_r}");
    }

    #[test]
    fn idr_return_to_baseline() {
        // Drug washes out → response returns to baseline
        let m = IndirectResponseModel::new(
            IndirectResponseType::StimulateProduction,
            1.0,
            0.1,
            2.0,
            5.0,
        )
        .unwrap();

        // Drug present for 0-24h, then zero
        let mut conc_profile = Vec::new();
        for i in 0..=24 {
            conc_profile.push((i as f64, 50.0));
        }
        for i in 25..=200 {
            conc_profile.push((i as f64, 0.0));
        }
        let times: Vec<f64> = (0..=200).map(|i| i as f64).collect();
        let response = m.simulate(&conc_profile, &times, None, None).unwrap();

        // At t=200 with no drug for 176h (~17 time constants), should be near baseline
        let final_r = *response.last().unwrap();
        assert!((final_r - 10.0).abs() < 1.0, "should return to baseline 10.0: got {final_r}");
    }

    #[test]
    fn idr_invalid_params() {
        assert!(IndirectResponseModel::new(
            IndirectResponseType::InhibitProduction, 1.0, 0.1, 1.5, 5.0,
        ).is_err(), "Imax > 1 should fail for inhibition");

        assert!(
            IndirectResponseModel::new(
                IndirectResponseType::StimulateProduction,
                0.0,
                0.1,
                2.0,
                5.0,
            )
            .is_err(),
            "kin=0 should fail"
        );

        assert!(
            IndirectResponseModel::new(
                IndirectResponseType::StimulateProduction,
                1.0,
                0.1,
                2.0,
                0.0,
            )
            .is_err(),
            "C50=0 should fail"
        );
    }

    #[test]
    fn idr_nll_computation() {
        let m = IndirectResponseModel::new(
            IndirectResponseType::StimulateProduction,
            1.0,
            0.1,
            2.0,
            5.0,
        )
        .unwrap();

        let conc_profile: Vec<(f64, f64)> = (0..=50).map(|i| (i as f64, 10.0)).collect();
        let obs_times = vec![0.0, 12.0, 24.0, 48.0];
        let obs_values = vec![10.0, 15.0, 18.0, 20.0];
        let err = ErrorModel::Additive(2.0);

        let nll = m.nll(&conc_profile, &obs_times, &obs_values, &err, None).unwrap();
        assert!(nll.is_finite() && nll > 0.0);
    }

    // --- PkPdLink ---

    #[test]
    fn pkpd_link_interpolation() {
        let profile = vec![(0.0, 0.0), (1.0, 10.0), (2.0, 5.0)];
        let link = PkPdLink::new(&profile);

        assert!((link.conc_at(0.0) - 0.0).abs() < 1e-12);
        assert!((link.conc_at(0.5) - 5.0).abs() < 1e-12);
        assert!((link.conc_at(1.0) - 10.0).abs() < 1e-12);
        assert!((link.conc_at(1.5) - 7.5).abs() < 1e-12);
        assert!((link.conc_at(2.0) - 5.0).abs() < 1e-12);
        // Extrapolation: clamp
        assert!((link.conc_at(-1.0) - 0.0).abs() < 1e-12);
        assert!((link.conc_at(5.0) - 5.0).abs() < 1e-12);
    }
}
