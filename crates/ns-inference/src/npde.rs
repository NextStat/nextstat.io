use ns_core::{Error, Result};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::foce::OmegaMatrix;
use crate::pk::{self, ErrorModel};
use crate::vpc::PkModelKind;

/// NPDE configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpdeConfig {
    /// Number of Monte Carlo simulations per observation.
    pub n_sim: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for NpdeConfig {
    fn default() -> Self {
        Self { n_sim: 1000, seed: 42 }
    }
}

/// NPDE record for one observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpdeRecord {
    pub subject: usize,
    pub time: f64,
    pub dv: f64,
    pub percentile: f64,
    pub npde: f64,
}

/// NPDE result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpdeResult {
    pub records: Vec<NpdeRecord>,
    pub mean: f64,
    pub variance: f64,
}

fn expected_n_theta(model: PkModelKind) -> usize {
    match model {
        PkModelKind::OneCptOral => 3,
        PkModelKind::OneCptIv => 2,
        PkModelKind::TwoCptIv => 4,
        PkModelKind::TwoCptOral => 5,
        PkModelKind::ThreeCptIv => 6,
        PkModelKind::ThreeCptOral => 7,
    }
}

#[inline]
fn conc_model(
    model: PkModelKind,
    dose: f64,
    bioav: f64,
    theta: &[f64],
    eta: &[f64],
    t: f64,
) -> f64 {
    match model {
        PkModelKind::OneCptOral => {
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ka = theta[2] * eta[2].exp();
            pk::conc_oral(dose, bioav, cl, v, ka, t)
        }
        PkModelKind::OneCptIv => {
            let cl = theta[0] * eta[0].exp();
            let v = theta[1] * eta[1].exp();
            let ke = cl / v;
            (dose / v) * (-ke * t).exp()
        }
        PkModelKind::TwoCptIv => {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            pk::conc_iv_2cpt_macro(dose, cl, v1, v2, q, t)
        }
        PkModelKind::TwoCptOral => {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let ka = theta[4] * eta[4].exp();
            pk::conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t)
        }
        PkModelKind::ThreeCptIv => {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            pk::conc_iv_3cpt_macro(dose, t, cl, v1, q2, v2, q3, v3)
        }
        PkModelKind::ThreeCptOral => {
            let cl = theta[0] * eta[0].exp();
            let v1 = theta[1] * eta[1].exp();
            let q2 = theta[2] * eta[2].exp();
            let v2 = theta[3] * eta[3].exp();
            let q3 = theta[4] * eta[4].exp();
            let v3 = theta[5] * eta[5].exp();
            let ka = theta[6] * eta[6].exp();
            pk::conc_oral_3cpt_macro(dose * bioav, t, cl, v1, q2, v2, q3, v3, ka)
        }
    }
}

/// Compute NPDE diagnostics via simulation.
pub fn npde_pk(
    model: PkModelKind,
    times: &[f64],
    y: &[f64],
    subject_idx: &[usize],
    n_subjects: usize,
    doses: &[f64],
    bioav: f64,
    theta: &[f64],
    omega: &OmegaMatrix,
    error_model: &ErrorModel,
    config: &NpdeConfig,
) -> Result<NpdeResult> {
    if config.n_sim < 10 {
        return Err(Error::Validation("n_sim must be >= 10".into()));
    }
    let n_theta = expected_n_theta(model);
    if theta.len() != n_theta {
        return Err(Error::Validation(format!(
            "theta must have {n_theta} elements for model {:?}",
            model
        )));
    }
    if omega.dim() != n_theta {
        return Err(Error::Validation(format!("omega must be {n_theta}×{n_theta}")));
    }
    if times.len() != y.len() || times.len() != subject_idx.len() {
        return Err(Error::Validation("times/y/subject_idx length mismatch".into()));
    }
    if doses.len() != n_subjects {
        return Err(Error::Validation(format!(
            "doses length {} != n_subjects {}",
            doses.len(),
            n_subjects
        )));
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let std_normal = RandNormal::new(0.0_f64, 1.0).unwrap();
    let normal = Normal::new(0.0, 1.0).map_err(|e| Error::Validation(e.to_string()))?;
    let chol = omega.cholesky();
    let n_eta = n_theta;

    let mut records = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        let s = subject_idx[i];
        let t = times[i];
        let dv = y[i];
        let dose = doses[s];

        let mut n_le = 0usize;
        for _ in 0..config.n_sim {
            let z: Vec<f64> = (0..n_eta).map(|_| std_normal.sample(&mut rng)).collect();
            let mut eta = vec![0.0; n_eta];
            for ii in 0..n_eta {
                for jj in 0..=ii {
                    eta[ii] += chol[ii][jj] * z[jj];
                }
            }
            let c = conc_model(model, dose, bioav, theta, &eta, t);
            let sd = error_model.variance(c.max(1e-30)).sqrt().max(1e-30);
            let sim_y = c + sd * std_normal.sample(&mut rng);
            if sim_y <= dv {
                n_le += 1;
            }
        }

        let p = (n_le as f64 + 0.5) / (config.n_sim as f64 + 1.0);
        let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
        let npde = normal.inverse_cdf(p_clamped);
        records.push(NpdeRecord { subject: s, time: t, dv, percentile: p_clamped, npde });
    }

    let n = records.len() as f64;
    let mean = if n > 0.0 { records.iter().map(|r| r.npde).sum::<f64>() / n } else { 0.0 };
    let variance = if n > 1.0 {
        records.iter().map(|r| (r.npde - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };

    Ok(NpdeResult { records, mean, variance })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npde_basic_smoke() {
        let n_subjects = 8usize;
        let times_per = [0.5, 1.0, 2.0, 4.0];
        let theta = [1.2, 15.0, 2.0];
        let omega = OmegaMatrix::from_diagonal(&[0.2, 0.2, 0.2]).unwrap();
        let em = ErrorModel::Additive(0.05);
        let doses = vec![100.0; n_subjects];

        let mut times = Vec::new();
        let mut y = Vec::new();
        let mut subject_idx = Vec::new();
        for s in 0..n_subjects {
            for &t in &times_per {
                let c = pk::conc_oral(doses[s], 1.0, theta[0], theta[1], theta[2], t);
                times.push(t);
                y.push(c);
                subject_idx.push(s);
            }
        }

        let cfg = NpdeConfig { n_sim: 200, seed: 7 };
        let res = npde_pk(
            PkModelKind::OneCptOral,
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            &theta,
            &omega,
            &em,
            &cfg,
        )
        .unwrap();

        assert_eq!(res.records.len(), times.len());
        assert!(res.mean.is_finite());
        assert!(res.variance.is_finite());
    }
}
