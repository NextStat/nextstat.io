//! Pharmacometrics models (Phase 13).
//!
//! Currently implemented:
//! - 1-compartment PK model (oral dosing, first-order absorption)
//!
//! # LLOQ policy
//! Observations below the lower limit of quantification (LLOQ) can be handled as:
//! - `Ignore`: drop those observations from the likelihood.
//! - `ReplaceHalf`: replace `y < LLOQ` with `LLOQ/2` (simple heuristic).
//! - `Censored`: left-censored likelihood term `P(Y < LLOQ)` under the observation model.
//!
//! Baseline observation model: additive Normal noise on concentration:
//! `y_i ~ Normal(C(t_i), sigma)` with fixed `sigma` provided in the model config.

use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

#[inline]
fn conc_oral(dose: f64, bioavailability: f64, cl: f64, v: f64, ka: f64, t: f64) -> f64 {
    let ke = cl / v;
    let d = ka - ke;
    let d_amt = dose * bioavailability;
    let pref = d_amt / v;

    let eke = (-ke * t).exp();
    // Stable form for (eke - eka) / (ka - ke)
    // eke - eka = eke * (1 - exp(-(ka-ke)t)) = eke * (-expm1(-d t))
    let s = if d.abs() < 1e-10 { t } else { (-(-d * t).exp_m1()) / d };
    // C(t) = (D/V) * ka * eke * s
    pref * ka * eke * s
}

#[inline]
fn conc_oral_and_grad(
    dose: f64,
    bioavailability: f64,
    cl: f64,
    v: f64,
    ka: f64,
    t: f64,
) -> (f64, f64, f64, f64) {
    // Use the standard closed form (not the expm1 form) for derivatives;
    // avoid testing cases near Ka ~= Ke in the baseline acceptance tests.
    let ke = cl / v;
    let d_amt = dose * bioavailability;
    let pref = d_amt / v;

    let eke = (-ke * t).exp();
    let eka = (-ka * t).exp();
    let denom = ka - ke;
    if denom.abs() < 1e-8 {
        // Limit Ka -> Ke = K:
        // C(t) = (D/V) * K * t * exp(-K t)
        let k = 0.5 * (ka + ke);
        let ek = (-k * t).exp();
        let c = pref * k * t * ek;
        // Gradient is not used in our synthetic fixture (we keep Ka far from Ke),
        // but return finite values.
        let eps = 1e-6;
        let c_cl = (conc_oral(dose, bioavailability, cl + eps, v, ka, t)
            - conc_oral(dose, bioavailability, cl - eps, v, ka, t))
            / (2.0 * eps);
        let c_v = (conc_oral(dose, bioavailability, cl, v + eps, ka, t)
            - conc_oral(dose, bioavailability, cl, v - eps, ka, t))
            / (2.0 * eps);
        let c_ka = (conc_oral(dose, bioavailability, cl, v, ka + eps, t)
            - conc_oral(dose, bioavailability, cl, v, ka - eps, t))
            / (2.0 * eps);
        return (c, c_cl, c_v, c_ka);
    }

    let frac = ka / denom;
    let diff = eke - eka;
    let c = pref * frac * diff;

    let denom2 = denom * denom;
    let dfrac_dka = -ke / denom2;
    let dfrac_dke = ka / denom2;
    let ddiff_dka = t * eka;
    let ddiff_dke = -t * eke;

    let dc_dka = pref * (dfrac_dka * diff + frac * ddiff_dka);
    let dc_dke = pref * (dfrac_dke * diff + frac * ddiff_dke);

    let dke_dcl = 1.0 / v;
    let dke_dv = -ke / v;
    let dpref_dv = -pref / v;

    let dc_dcl = dc_dke * dke_dcl;
    let dc_dv = dpref_dv * frac * diff + dc_dke * dke_dv;

    (c, dc_dcl, dc_dv, dc_dka)
}

/// Policy for handling observations below LLOQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LloqPolicy {
    /// Drop points below LLOQ.
    Ignore,
    /// Replace points below LLOQ with LLOQ/2.
    ReplaceHalf,
    /// Treat points below LLOQ as left-censored.
    Censored,
}

/// 1-compartment PK model (oral, first-order absorption).
///
/// State-space (amounts):
/// - `dA_gut/dt = -Ka * A_gut`
/// - `dA_cent/dt = Ka * A_gut - Ke * A_cent`, `Ke = CL / V`
///
/// Concentration:
/// - `C(t) = A_cent(t) / V`
#[derive(Debug, Clone)]
pub struct OneCompartmentOralPkModel {
    times: Vec<f64>,
    y: Vec<f64>,
    dose: f64,
    bioavailability: f64,
    sigma: f64,
    lloq: Option<f64>,
    lloq_policy: LloqPolicy,
}

impl OneCompartmentOralPkModel {
    /// Create a PK model instance.
    pub fn new(
        times: Vec<f64>,
        y: Vec<f64>,
        dose: f64,
        bioavailability: f64,
        sigma: f64,
        lloq: Option<f64>,
        lloq_policy: LloqPolicy,
    ) -> Result<Self> {
        if times.is_empty() {
            return Err(Error::Validation("times must be non-empty".to_string()));
        }
        if times.len() != y.len() {
            return Err(Error::Validation(format!(
                "times/y length mismatch: {} vs {}",
                times.len(),
                y.len()
            )));
        }
        if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(Error::Validation("times must be finite and >= 0".to_string()));
        }
        if y.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::Validation("y must be finite and >= 0".to_string()));
        }
        if !dose.is_finite() || dose <= 0.0 {
            return Err(Error::Validation("dose must be finite and > 0".to_string()));
        }
        if !bioavailability.is_finite() || bioavailability <= 0.0 {
            return Err(Error::Validation("bioavailability must be finite and > 0".to_string()));
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation("sigma must be finite and > 0".to_string()));
        }
        if let Some(lloq) = lloq {
            if !lloq.is_finite() || lloq < 0.0 {
                return Err(Error::Validation("lloq must be finite and >= 0".to_string()));
            }
        }
        Ok(Self { times, y, dose, bioavailability, sigma, lloq, lloq_policy })
    }

    /// Predicted concentration at time `t` for parameters `(cl, v, ka)`.
    #[inline]
    fn conc(&self, cl: f64, v: f64, ka: f64, t: f64) -> f64 {
        conc_oral(self.dose, self.bioavailability, cl, v, ka, t)
    }

    /// Concentration and partial derivatives wrt (cl, v, ka).
    #[inline]
    fn conc_and_grad(&self, cl: f64, v: f64, ka: f64, t: f64) -> (f64, f64, f64, f64) {
        conc_oral_and_grad(self.dose, self.bioavailability, cl, v, ka, t)
    }
}

/// NLME baseline for the 1-compartment oral PK model with log-normal random effects.
///
/// Individual parameters:
/// - `cl_i = cl_pop * exp(eta_cl_i)`
/// - `v_i  = v_pop  * exp(eta_v_i)`
/// - `ka_i = ka_pop * exp(eta_ka_i)`
///
/// Random effects priors:
/// - `eta_*_i ~ Normal(0, omega_*)` (independent; diagonal covariance)
///
/// Parameter vector:
/// - population: `cl_pop, v_pop, ka_pop`
/// - random effects scales: `omega_cl, omega_v, omega_ka`
/// - per-subject random effects: `eta_cl[0..n_subjects), eta_v[...], eta_ka[...]`
#[derive(Debug, Clone)]
pub struct OneCompartmentOralPkNlmeModel {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    n_subjects: usize,
    dose: f64,
    bioavailability: f64,
    sigma: f64,
    lloq: Option<f64>,
    lloq_policy: LloqPolicy,
}

impl OneCompartmentOralPkNlmeModel {
    /// Create a new NLME PK model instance.
    pub fn new(
        times: Vec<f64>,
        y: Vec<f64>,
        subject_idx: Vec<usize>,
        n_subjects: usize,
        dose: f64,
        bioavailability: f64,
        sigma: f64,
        lloq: Option<f64>,
        lloq_policy: LloqPolicy,
    ) -> Result<Self> {
        if times.is_empty() {
            return Err(Error::Validation("times must be non-empty".to_string()));
        }
        if times.len() != y.len() || times.len() != subject_idx.len() {
            return Err(Error::Validation(format!(
                "times/y/subject_idx length mismatch: {}, {}, {}",
                times.len(),
                y.len(),
                subject_idx.len()
            )));
        }
        if n_subjects == 0 {
            return Err(Error::Validation("n_subjects must be > 0".to_string()));
        }
        if subject_idx.iter().any(|&s| s >= n_subjects) {
            return Err(Error::Validation("subject_idx must be in [0, n_subjects)".to_string()));
        }
        if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(Error::Validation("times must be finite and >= 0".to_string()));
        }
        if y.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::Validation("y must be finite and >= 0".to_string()));
        }
        if !dose.is_finite() || dose <= 0.0 {
            return Err(Error::Validation("dose must be finite and > 0".to_string()));
        }
        if !bioavailability.is_finite() || bioavailability <= 0.0 {
            return Err(Error::Validation("bioavailability must be finite and > 0".to_string()));
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(Error::Validation("sigma must be finite and > 0".to_string()));
        }
        if let Some(lloq) = lloq {
            if !lloq.is_finite() || lloq < 0.0 {
                return Err(Error::Validation("lloq must be finite and >= 0".to_string()));
            }
        }
        Ok(Self {
            times,
            y,
            subject_idx,
            n_subjects,
            dose,
            bioavailability,
            sigma,
            lloq,
            lloq_policy,
        })
    }

    #[inline]
    fn unpack<'a>(
        &self,
        params: &'a [f64],
    ) -> Result<(f64, f64, f64, f64, f64, f64, &'a [f64], &'a [f64], &'a [f64])> {
        let expect = self.dim();
        if params.len() != expect {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                expect,
                params.len()
            )));
        }
        let cl_pop = params[0];
        let v_pop = params[1];
        let ka_pop = params[2];
        let omega_cl = params[3];
        let omega_v = params[4];
        let omega_ka = params[5];
        if [cl_pop, v_pop, ka_pop, omega_cl, omega_v, omega_ka]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::Validation(
                "population/omega params must be finite and > 0".to_string(),
            ));
        }
        let n = self.n_subjects;
        let eta_cl = &params[6..6 + n];
        let eta_v = &params[6 + n..6 + 2 * n];
        let eta_ka = &params[6 + 2 * n..6 + 3 * n];
        Ok((cl_pop, v_pop, ka_pop, omega_cl, omega_v, omega_ka, eta_cl, eta_v, eta_ka))
    }

    #[inline]
    fn conc_subject(
        &self,
        cl_pop: f64,
        v_pop: f64,
        ka_pop: f64,
        eta_cl: f64,
        eta_v: f64,
        eta_ka: f64,
        t: f64,
    ) -> f64 {
        let cl = cl_pop * eta_cl.exp();
        let v = v_pop * eta_v.exp();
        let ka = ka_pop * eta_ka.exp();
        conc_oral(self.dose, self.bioavailability, cl, v, ka, t)
    }

    #[inline]
    fn conc_subject_and_grad(
        &self,
        cl_pop: f64,
        v_pop: f64,
        ka_pop: f64,
        eta_cl: f64,
        eta_v: f64,
        eta_ka: f64,
        t: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64) {
        let ecl = eta_cl.exp();
        let ev = eta_v.exp();
        let eka = eta_ka.exp();
        let cl = cl_pop * ecl;
        let v = v_pop * ev;
        let ka = ka_pop * eka;
        let (c, dc_dcl, dc_dv, dc_dka) =
            conc_oral_and_grad(self.dose, self.bioavailability, cl, v, ka, t);
        (c, dc_dcl, dc_dv, dc_dka, cl, v, ka)
    }

    /// Predict population curve at observation times (ignores random effects).
    pub fn predict_population(&self, cl_pop: f64, v_pop: f64, ka_pop: f64) -> Result<Vec<f64>> {
        if [cl_pop, v_pop, ka_pop].iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::Validation(
                "cl_pop/v_pop/ka_pop must be finite and > 0".to_string(),
            ));
        }
        Ok(self
            .times
            .iter()
            .map(|&t| conc_oral(self.dose, self.bioavailability, cl_pop, v_pop, ka_pop, t))
            .collect())
    }

    /// Predict for a single subject at observation times.
    pub fn predict_subject(&self, params: &[f64], subject: usize) -> Result<Vec<f64>> {
        let (cl_pop, v_pop, ka_pop, _ocl, _ov, _oka, eta_cl, eta_v, eta_ka) =
            self.unpack(params)?;
        if subject >= self.n_subjects {
            return Err(Error::Validation("subject out of range".to_string()));
        }
        Ok(self
            .times
            .iter()
            .zip(self.subject_idx.iter())
            .filter_map(|(&t, &s)| {
                if s == subject {
                    Some(
                        self.conc_subject(cl_pop, v_pop, ka_pop, eta_cl[s], eta_v[s], eta_ka[s], t),
                    )
                } else {
                    None
                }
            })
            .collect())
    }
}

impl LogDensityModel for OneCompartmentOralPkNlmeModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        6 + 3 * self.n_subjects
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.dim());
        names.push("cl_pop".to_string());
        names.push("v_pop".to_string());
        names.push("ka_pop".to_string());
        names.push("omega_cl".to_string());
        names.push("omega_v".to_string());
        names.push("omega_ka".to_string());
        for i in 0..self.n_subjects {
            names.push(format!("eta_cl[{}]", i));
        }
        for i in 0..self.n_subjects {
            names.push(format!("eta_v[{}]", i));
        }
        for i in 0..self.n_subjects {
            names.push(format!("eta_ka[{}]", i));
        }
        names
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut b = Vec::with_capacity(self.dim());
        // population
        b.push((1e-12, f64::INFINITY));
        b.push((1e-12, f64::INFINITY));
        b.push((1e-12, f64::INFINITY));
        // omegas
        b.push((1e-12, f64::INFINITY));
        b.push((1e-12, f64::INFINITY));
        b.push((1e-12, f64::INFINITY));
        // etas (unbounded)
        for _ in 0..(3 * self.n_subjects) {
            b.push((f64::NEG_INFINITY, f64::INFINITY));
        }
        b
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.dim());
        p.extend_from_slice(&[1.0, 10.0, 1.0]);
        p.extend_from_slice(&[0.2, 0.2, 0.2]);
        p.resize(self.dim(), 0.0);
        p
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let (cl_pop, v_pop, ka_pop, omega_cl, omega_v, omega_ka, eta_cl, eta_v, eta_ka) =
            self.unpack(params)?;

        let s = self.sigma;
        let inv_s2 = 1.0 / (s * s);
        let normal = Normal::new(0.0, 1.0).map_err(|e| Error::Validation(e.to_string()))?;

        let mut nll = 0.0;
        for ((&t, &yobs), &subj) in
            self.times.iter().zip(self.y.iter()).zip(self.subject_idx.iter())
        {
            let c = self.conc_subject(
                cl_pop,
                v_pop,
                ka_pop,
                eta_cl[subj],
                eta_v[subj],
                eta_ka[subj],
                t,
            );

            if let Some(lloq) = self.lloq {
                if yobs < lloq {
                    match self.lloq_policy {
                        LloqPolicy::Ignore => continue,
                        LloqPolicy::ReplaceHalf => {
                            let y = 0.5 * lloq;
                            let r = y - c;
                            nll += 0.5 * r * r * inv_s2 + s.ln();
                        }
                        LloqPolicy::Censored => {
                            let z = (lloq - c) / s;
                            let p = normal.cdf(z).max(1e-300);
                            nll += -p.ln();
                        }
                    }
                    continue;
                }
            }

            let r = yobs - c;
            nll += 0.5 * r * r * inv_s2 + s.ln();
        }

        // Random effects priors, up to an additive constant.
        for &e in eta_cl {
            nll += 0.5 * (e * e) / (omega_cl * omega_cl) + omega_cl.ln();
        }
        for &e in eta_v {
            nll += 0.5 * (e * e) / (omega_v * omega_v) + omega_v.ln();
        }
        for &e in eta_ka {
            nll += 0.5 * (e * e) / (omega_ka * omega_ka) + omega_ka.ln();
        }

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let (cl_pop, v_pop, ka_pop, omega_cl, omega_v, omega_ka, eta_cl, eta_v, eta_ka) =
            self.unpack(params)?;

        let n = self.n_subjects;
        let s = self.sigma;
        let inv_s2 = 1.0 / (s * s);
        let normal = Normal::new(0.0, 1.0).map_err(|e| Error::Validation(e.to_string()))?;

        let mut g = vec![0.0_f64; self.dim()];

        for ((&t, &yobs), &subj) in
            self.times.iter().zip(self.y.iter()).zip(self.subject_idx.iter())
        {
            let (c, dc_dcl, dc_dv, dc_dka, cl_i, v_i, ka_i) = self.conc_subject_and_grad(
                cl_pop,
                v_pop,
                ka_pop,
                eta_cl[subj],
                eta_v[subj],
                eta_ka[subj],
                t,
            );

            if let Some(lloq) = self.lloq {
                if yobs < lloq {
                    match self.lloq_policy {
                        LloqPolicy::Ignore => continue,
                        LloqPolicy::ReplaceHalf => {
                            let y = 0.5 * lloq;
                            let r = c - y;
                            let w = r * inv_s2;
                            g[0] += w * dc_dcl * (cl_i / cl_pop);
                            g[1] += w * dc_dv * (v_i / v_pop);
                            g[2] += w * dc_dka * (ka_i / ka_pop);
                            g[6 + subj] += w * dc_dcl * cl_i;
                            g[6 + n + subj] += w * dc_dv * v_i;
                            g[6 + 2 * n + subj] += w * dc_dka * ka_i;
                        }
                        LloqPolicy::Censored => {
                            let z = (lloq - c) / s;
                            let p = normal.cdf(z).max(1e-300);
                            let pdf = normal.pdf(z);
                            let ratio = pdf / p;
                            let w = ratio / s;
                            g[0] += w * dc_dcl * (cl_i / cl_pop);
                            g[1] += w * dc_dv * (v_i / v_pop);
                            g[2] += w * dc_dka * (ka_i / ka_pop);
                            g[6 + subj] += w * dc_dcl * cl_i;
                            g[6 + n + subj] += w * dc_dv * v_i;
                            g[6 + 2 * n + subj] += w * dc_dka * ka_i;
                        }
                    }
                    continue;
                }
            }

            let r = c - yobs;
            let w = r * inv_s2;
            g[0] += w * dc_dcl * (cl_i / cl_pop);
            g[1] += w * dc_dv * (v_i / v_pop);
            g[2] += w * dc_dka * (ka_i / ka_pop);
            g[6 + subj] += w * dc_dcl * cl_i;
            g[6 + n + subj] += w * dc_dv * v_i;
            g[6 + 2 * n + subj] += w * dc_dka * ka_i;
        }

        // Prior gradients:
        let ocl2 = omega_cl * omega_cl;
        let ov2 = omega_v * omega_v;
        let oka2 = omega_ka * omega_ka;

        let mut sum_eta2 = 0.0;
        for (i, &e) in eta_cl.iter().enumerate() {
            g[6 + i] += e / ocl2;
            sum_eta2 += e * e;
        }
        g[3] += -sum_eta2 / (omega_cl * ocl2) + (n as f64) / omega_cl;

        let mut sum_eta2 = 0.0;
        for (i, &e) in eta_v.iter().enumerate() {
            g[6 + n + i] += e / ov2;
            sum_eta2 += e * e;
        }
        g[4] += -sum_eta2 / (omega_v * ov2) + (n as f64) / omega_v;

        let mut sum_eta2 = 0.0;
        for (i, &e) in eta_ka.iter().enumerate() {
            g[6 + 2 * n + i] += e / oka2;
            sum_eta2 += e * e;
        }
        g[5] += -sum_eta2 / (omega_ka * oka2) + (n as f64) / omega_ka;

        Ok(g)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

impl LogDensityModel for OneCompartmentOralPkModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        3
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["cl".to_string(), "v".to_string(), "ka".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(1e-12, f64::INFINITY), (1e-12, f64::INFINITY), (1e-12, f64::INFINITY)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![1.0, 10.0, 1.0]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 3 {
            return Err(Error::Validation(format!("expected 3 parameters, got {}", params.len())));
        }
        if params.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::Validation("params must be finite and > 0".to_string()));
        }
        let cl = params[0];
        let v = params[1];
        let ka = params[2];

        let s = self.sigma;
        let inv_s2 = 1.0 / (s * s);
        let normal = Normal::new(0.0, 1.0).map_err(|e| Error::Validation(e.to_string()))?;

        let mut nll = 0.0;
        for (&t, &yobs) in self.times.iter().zip(self.y.iter()) {
            let c = self.conc(cl, v, ka, t);

            if let Some(lloq) = self.lloq {
                if yobs < lloq {
                    match self.lloq_policy {
                        LloqPolicy::Ignore => continue,
                        LloqPolicy::ReplaceHalf => {
                            let y = 0.5 * lloq;
                            let r = y - c;
                            nll += 0.5 * r * r * inv_s2 + s.ln();
                        }
                        LloqPolicy::Censored => {
                            let z = (lloq - c) / s;
                            let p = normal.cdf(z).max(1e-300);
                            nll += -p.ln();
                        }
                    }
                    continue;
                }
            }

            let r = yobs - c;
            nll += 0.5 * r * r * inv_s2 + s.ln();
        }
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 3 {
            return Err(Error::Validation(format!("expected 3 parameters, got {}", params.len())));
        }
        if params.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::Validation("params must be finite and > 0".to_string()));
        }
        let cl = params[0];
        let v = params[1];
        let ka = params[2];

        let s = self.sigma;
        let inv_s2 = 1.0 / (s * s);
        let normal = Normal::new(0.0, 1.0).map_err(|e| Error::Validation(e.to_string()))?;

        let mut g = vec![0.0_f64; 3];
        for (&t, &yobs) in self.times.iter().zip(self.y.iter()) {
            let (c, dc_dcl, dc_dv, dc_dka) = self.conc_and_grad(cl, v, ka, t);

            if let Some(lloq) = self.lloq {
                if yobs < lloq {
                    match self.lloq_policy {
                        LloqPolicy::Ignore => continue,
                        LloqPolicy::ReplaceHalf => {
                            let y = 0.5 * lloq;
                            let r = c - y;
                            let w = r * inv_s2;
                            g[0] += w * dc_dcl;
                            g[1] += w * dc_dv;
                            g[2] += w * dc_dka;
                        }
                        LloqPolicy::Censored => {
                            let z = (lloq - c) / s;
                            let p = normal.cdf(z).max(1e-300);
                            let pdf = normal.pdf(z);
                            let ratio = pdf / p; // φ/Φ
                            let w = ratio / s;
                            g[0] += w * dc_dcl;
                            g[1] += w * dc_dv;
                            g[2] += w * dc_dka;
                        }
                    }
                    continue;
                }
            }

            let r = c - yobs;
            let w = r * inv_s2;
            g[0] += w * dc_dcl;
            g[1] += w * dc_dv;
            g[2] += w * dc_dka;
        }

        Ok(g)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplace::laplace_log_marginal;
    use crate::mle::MaximumLikelihoodEstimator;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};

    #[test]
    fn pk_fit_recovers_params_synthetic_smoke() {
        let cl_true = 1.2;
        let v_true = 15.0;
        let ka_true = 2.0;
        let dose = 100.0;
        let f = 1.0;
        let sigma = 0.05;

        let times: Vec<f64> = (0..30).map(|i| i as f64 * 0.25).collect();
        let model = OneCompartmentOralPkModel::new(
            times.clone(),
            vec![0.0; times.len()],
            dose,
            f,
            sigma,
            None,
            LloqPolicy::Censored,
        )
        .unwrap();

        // Generate synthetic observations
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let noise = RandNormal::new(0.0, sigma).unwrap();
        let mut y = Vec::with_capacity(times.len());
        for &t in &times {
            let c = model.conc(cl_true, v_true, ka_true, t);
            y.push((c + noise.sample(&mut rng)).max(0.0));
        }

        let model =
            OneCompartmentOralPkModel::new(times, y, dose, f, sigma, None, LloqPolicy::Censored)
                .unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let fit = mle.fit(&model).unwrap();
        assert!(fit.converged, "fit did not converge: {:?}", fit);

        let cl_hat = fit.parameters[0];
        let v_hat = fit.parameters[1];
        let ka_hat = fit.parameters[2];

        // Loose tolerances: this is a smoke test.
        assert!((cl_hat - cl_true).abs() / cl_true < 0.15);
        assert!((v_hat - v_true).abs() / v_true < 0.15);
        assert!((ka_hat - ka_true).abs() / ka_true < 0.20);
    }

    #[test]
    fn nlme_pk_map_and_laplace_smoke() {
        let cl_pop_true = 1.2;
        let v_pop_true = 15.0;
        let ka_pop_true = 2.0;
        let omega_cl_true = 0.25;
        let omega_v_true = 0.20;
        let omega_ka_true = 0.30;

        let dose = 100.0;
        let f = 1.0;
        let sigma = 0.05;
        let n_subjects = 4usize;
        let times_per = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let eta_cl_dist = RandNormal::new(0.0, omega_cl_true).unwrap();
        let eta_v_dist = RandNormal::new(0.0, omega_v_true).unwrap();
        let eta_ka_dist = RandNormal::new(0.0, omega_ka_true).unwrap();
        let noise = RandNormal::new(0.0, sigma).unwrap();

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();

        // Reuse the same closed-form concentration as the model implementation.
        let base = OneCompartmentOralPkModel::new(
            vec![0.25],
            vec![0.0],
            dose,
            f,
            sigma,
            None,
            LloqPolicy::Censored,
        )
        .unwrap();

        for sid in 0..n_subjects {
            let eta_cl: f64 = eta_cl_dist.sample(&mut rng);
            let eta_v: f64 = eta_v_dist.sample(&mut rng);
            let eta_ka: f64 = eta_ka_dist.sample(&mut rng);

            let cl_i = cl_pop_true * eta_cl.exp();
            let v_i = v_pop_true * eta_v.exp();
            let ka_i = ka_pop_true * eta_ka.exp();

            for &t in &times_per {
                let c = base.conc(cl_i, v_i, ka_i, t);
                let obs = (c + noise.sample(&mut rng)).max(0.0);
                times.push(t);
                y.push(obs);
                subject_idx.push(sid);
            }
        }

        let model = OneCompartmentOralPkNlmeModel::new(
            times,
            y,
            subject_idx,
            n_subjects,
            dose,
            f,
            sigma,
            None,
            LloqPolicy::Censored,
        )
        .unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let fit = mle.fit(&model).unwrap();
        assert_eq!(fit.parameters.len(), model.dim());
        assert!(fit.nll.is_finite());

        // Basic sanity (avoid overfitting exact recovery in a baseline smoke test).
        assert!(fit.parameters[0].is_finite() && fit.parameters[0] > 0.0);
        assert!(fit.parameters[1].is_finite() && fit.parameters[1] > 0.0);
        assert!(fit.parameters[2].is_finite() && fit.parameters[2] > 0.0);
        assert!(fit.parameters[3].is_finite() && fit.parameters[3] > 0.0);
        assert!(fit.parameters[4].is_finite() && fit.parameters[4] > 0.0);
        assert!(fit.parameters[5].is_finite() && fit.parameters[5] > 0.0);

        // Laplace approximation at the MAP mode should be finite.
        let lap = laplace_log_marginal(&model, &fit.parameters).unwrap();
        assert!(lap.log_marginal.is_finite());
    }
}
