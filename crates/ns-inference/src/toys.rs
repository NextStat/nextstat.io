//! Toy data generation utilities (Asimov + Poisson).
//!
//! This module focuses on generating **main** (binned count) observations for HistFactory models.
//! Auxiliary constraints remain as stored in the model unless explicitly overridden elsewhere.

use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

/// Generate the Asimov (deterministic expected) main dataset at the given parameter point.
///
/// This returns a flat vector of main-bin expectations in pyhf channel ordering
/// (lexicographic channel name), suitable for `HistFactoryModel::with_observed_main(...)`.
pub fn asimov_main(model: &HistFactoryModel, params: &[f64]) -> Result<Vec<f64>> {
    model.expected_data_pyhf_main(params)
}

/// Sample one Poisson-fluctuated main dataset from a vector of expectations.
pub fn poisson_main_from_expected(expected_main: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    expected_main
        .iter()
        .map(|&lam| {
            if !lam.is_finite() || lam <= 0.0 {
                // Correct limit: Poisson(0) is deterministically 0, and negative/NaN/inf
                // expected yields are treated as 0 for toy generation.
                return 0.0;
            }
            let pois = Poisson::new(lam).expect("Poisson::new(lambda>0)");
            pois.sample(&mut rng)
        })
        .collect()
}

/// Generate Poisson toy main datasets at the given parameter point.
///
/// Sampling is deterministic: toy `i` uses seed `seed + i`.
pub fn poisson_main_toys(
    model: &HistFactoryModel,
    params: &[f64],
    n_toys: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    let expected_main = model.expected_data_pyhf_main(params)?;
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(n_toys);
    for toy_idx in 0..n_toys {
        out.push(poisson_main_from_expected(&expected_main, seed.wrapping_add(toy_idx as u64)));
    }
    Ok(out)
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
    fn test_asimov_main_matches_expected_data_pyhf_main() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let a = asimov_main(&model, &params).unwrap();
        let e = model.expected_data_pyhf_main(&params).unwrap();
        assert_eq!(a, e);
    }

    #[test]
    fn test_poisson_main_toys_reproducible() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let toys1 = poisson_main_toys(&model, &params, 5, 123).unwrap();
        let toys2 = poisson_main_toys(&model, &params, 5, 123).unwrap();
        assert_eq!(toys1, toys2);
    }
}
