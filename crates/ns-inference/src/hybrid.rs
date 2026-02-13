//! Hybrid likelihood: combine two [`LogDensityModel`] implementations with a shared parameter map.
//!
//! This enables simultaneous fitting of binned (HistFactory) and unbinned (event-level)
//! likelihoods that share nuisance parameters (luminosity, signal strength, systematics).
//!
//! # Architecture
//!
//! `HybridLikelihood<A, B>` is generic over any two `LogDensityModel` implementations.
//! Parameters are merged by name: shared parameters get a single global index, while
//! model-specific parameters are appended. The global NLL is the sum of the two
//! component NLLs evaluated at their respective parameter subsets.
//!
//! # Example
//!
//! ```ignore
//! use ns_inference::hybrid::{HybridLikelihood, SharedParameterMap};
//!
//! let map = SharedParameterMap::build(&binned_model, &unbinned_model)?;
//! let hybrid = HybridLikelihood::new(binned_model, unbinned_model, map);
//! // hybrid implements LogDensityModel — use with MLE, profile scans, CLs, etc.
//! ```

use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel, PreparedModelRef};
use ns_core::{Error, Result};
use std::collections::HashMap;

/// Mapping between a merged (global) parameter vector and two component models.
///
/// Shared parameters (matched by name) get one global slot. Model-specific parameters
/// are appended after the shared ones. Bounds for shared parameters are intersected;
/// init values are taken from model A (the "primary" model, typically binned).
#[derive(Debug, Clone)]
pub struct SharedParameterMap {
    /// Global parameter names (stable order: shared first, then A-only, then B-only).
    pub names: Vec<String>,
    /// Global initial values.
    pub init: Vec<f64>,
    /// Global bounds.
    pub bounds: Vec<(f64, f64)>,
    /// For each global param: index in model A's parameter vector, or `None`.
    map_a: Vec<Option<usize>>,
    /// For each global param: index in model B's parameter vector, or `None`.
    map_b: Vec<Option<usize>>,
    /// Dimension of model A.
    dim_a: usize,
    /// Dimension of model B.
    dim_b: usize,
    /// POI index in the global vector (if either model declares one).
    poi_global: Option<usize>,
}

impl SharedParameterMap {
    /// Build a shared parameter map from two models.
    ///
    /// Parameters are matched by name (case-sensitive). For shared parameters:
    /// - Bounds are intersected: `(max(lo_a, lo_b), min(hi_a, hi_b))`.
    /// - Init value is taken from model A if within the intersected bounds,
    ///   otherwise clamped.
    ///
    /// Ordering: shared parameters first (in model A's order), then A-only
    /// (in A's order), then B-only (in B's order).
    pub fn build<A: LogDensityModel, B: LogDensityModel>(a: &A, b: &B) -> Result<Self> {
        let names_a = a.parameter_names();
        let bounds_a = a.parameter_bounds();
        let init_a = a.parameter_init();
        let names_b = b.parameter_names();
        let bounds_b = b.parameter_bounds();
        let init_b = b.parameter_init();

        // Index B's parameters by name for O(1) lookup.
        let b_by_name: HashMap<&str, usize> =
            names_b.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();

        let mut global_names = Vec::new();
        let mut global_init = Vec::new();
        let mut global_bounds = Vec::new();
        let mut map_a = Vec::new();
        let mut map_b = Vec::new();

        // Track which B params have been consumed (shared).
        let mut b_used = vec![false; names_b.len()];

        // Pass 1: iterate A's parameters in order.
        for (ia, name) in names_a.iter().enumerate() {
            if let Some(&ib) = b_by_name.get(name.as_str()) {
                // Shared parameter: intersect bounds.
                let lo = bounds_a[ia].0.max(bounds_b[ib].0);
                let hi = bounds_a[ia].1.min(bounds_b[ib].1);
                if lo > hi {
                    return Err(Error::Validation(format!(
                        "shared parameter '{}' has incompatible bounds: A={:?} B={:?} → empty intersection",
                        name, bounds_a[ia], bounds_b[ib]
                    )));
                }
                let init_val = init_a[ia].clamp(lo, hi);
                global_names.push(name.clone());
                global_init.push(init_val);
                global_bounds.push((lo, hi));
                map_a.push(Some(ia));
                map_b.push(Some(ib));
                b_used[ib] = true;
            } else {
                // A-only parameter.
                global_names.push(name.clone());
                global_init.push(init_a[ia]);
                global_bounds.push(bounds_a[ia]);
                map_a.push(Some(ia));
                map_b.push(None);
            }
        }

        // Pass 2: append B-only parameters.
        for (ib, name) in names_b.iter().enumerate() {
            if b_used[ib] {
                continue;
            }
            global_names.push(name.clone());
            global_init.push(init_b[ib]);
            global_bounds.push(bounds_b[ib]);
            map_a.push(None);
            map_b.push(Some(ib));
        }

        // Resolve POI: check both models (prefer A if both declare one).
        let poi_a = Self::find_poi_index_dyn(a);
        let poi_b = Self::find_poi_index_dyn(b);

        let poi_global = if let Some(ia) = poi_a {
            // Find global index where map_a[g] == Some(ia).
            map_a.iter().position(|&v| v == Some(ia))
        } else if let Some(ib) = poi_b {
            map_b.iter().position(|&v| v == Some(ib))
        } else {
            None
        };

        Ok(Self {
            names: global_names,
            init: global_init,
            bounds: global_bounds,
            map_a,
            map_b,
            dim_a: names_a.len(),
            dim_b: names_b.len(),
            poi_global,
        })
    }

    /// Try to extract POI index if the model implements `PoiModel`.
    ///
    /// We use a blanket approach: check if A or B has a `poi_index()` method.
    /// Since we can't do runtime trait checks in Rust, this is handled via
    /// explicit `build_with_poi` or the caller sets it after construction.
    fn find_poi_index_dyn<M: LogDensityModel>(_m: &M) -> Option<usize> {
        // Cannot dynamically check PoiModel without Any/downcast.
        // Caller should use `with_poi_from_a` / `with_poi_from_b`.
        None
    }

    /// Set the global POI index from model A's local POI index.
    pub fn with_poi_from_a(mut self, local_poi: usize) -> Self {
        self.poi_global = self.map_a.iter().position(|&v| v == Some(local_poi));
        self
    }

    /// Set the global POI index from model B's local POI index.
    pub fn with_poi_from_b(mut self, local_poi: usize) -> Self {
        self.poi_global = self.map_b.iter().position(|&v| v == Some(local_poi));
        self
    }

    /// Global POI index (if either model declares one).
    pub fn poi_index(&self) -> Option<usize> {
        self.poi_global
    }

    /// Number of global parameters.
    pub fn dim(&self) -> usize {
        self.names.len()
    }

    /// Number of shared parameters.
    pub fn n_shared(&self) -> usize {
        self.map_a.iter().zip(&self.map_b).filter(|(a, b)| a.is_some() && b.is_some()).count()
    }

    /// Extract model A's parameter vector from the global vector.
    pub fn extract_a(&self, global: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.dim_a];
        for (g, &maybe_a) in self.map_a.iter().enumerate() {
            if let Some(ia) = maybe_a {
                out[ia] = global[g];
            }
        }
        out
    }

    /// Extract model B's parameter vector from the global vector.
    pub fn extract_b(&self, global: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.dim_b];
        for (g, &maybe_b) in self.map_b.iter().enumerate() {
            if let Some(ib) = maybe_b {
                out[ib] = global[g];
            }
        }
        out
    }

    /// Scatter model A's gradient into the global gradient vector (additive).
    fn scatter_grad_a(&self, local_grad: &[f64], global_grad: &mut [f64]) {
        for (g, &maybe_a) in self.map_a.iter().enumerate() {
            if let Some(ia) = maybe_a {
                global_grad[g] += local_grad[ia];
            }
        }
    }

    /// Scatter model B's gradient into the global gradient vector (additive).
    fn scatter_grad_b(&self, local_grad: &[f64], global_grad: &mut [f64]) {
        for (g, &maybe_b) in self.map_b.iter().enumerate() {
            if let Some(ib) = maybe_b {
                global_grad[g] += local_grad[ib];
            }
        }
    }
}

/// Combined likelihood of two models with shared parameters.
///
/// `NLL_hybrid(θ) = NLL_A(θ_A) + NLL_B(θ_B)`
///
/// where `θ_A` and `θ_B` are extracted from the global parameter vector `θ`
/// via [`SharedParameterMap`]. Shared parameters contribute to both terms.
///
/// Implements [`LogDensityModel`] so it can be used with all existing inference
/// machinery (MLE, profile scans, CLs, toy generation, etc.).
#[derive(Clone)]
pub struct HybridLikelihood<A, B> {
    model_a: A,
    model_b: B,
    map: SharedParameterMap,
}

impl<A: LogDensityModel, B: LogDensityModel> HybridLikelihood<A, B> {
    /// Create a new hybrid likelihood.
    pub fn new(model_a: A, model_b: B, map: SharedParameterMap) -> Self {
        Self { model_a, model_b, map }
    }

    /// Access model A.
    pub fn model_a(&self) -> &A {
        &self.model_a
    }

    /// Access model B.
    pub fn model_b(&self) -> &B {
        &self.model_b
    }

    /// Access the shared parameter map.
    pub fn parameter_map(&self) -> &SharedParameterMap {
        &self.map
    }

    fn validate_len(&self, params: &[f64]) -> Result<()> {
        let expected = self.map.dim();
        if params.len() != expected {
            return Err(Error::Validation(format!(
                "HybridLikelihood: parameter length mismatch: expected {expected}, got {}",
                params.len()
            )));
        }
        Ok(())
    }
}

impl<A: LogDensityModel, B: LogDensityModel> LogDensityModel for HybridLikelihood<A, B> {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.map.dim()
    }

    fn parameter_names(&self) -> Vec<String> {
        self.map.names.clone()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.map.bounds.clone()
    }

    fn parameter_init(&self) -> Vec<f64> {
        self.map.init.clone()
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        self.validate_len(params)?;
        let pa = self.map.extract_a(params);
        let pb = self.map.extract_b(params);
        let nll_a = self.model_a.nll(&pa)?;
        let nll_b = self.model_b.nll(&pb)?;
        Ok(nll_a + nll_b)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.validate_len(params)?;
        let pa = self.map.extract_a(params);
        let pb = self.map.extract_b(params);
        let ga = self.model_a.grad_nll(&pa)?;
        let gb = self.model_b.grad_nll(&pb)?;
        let mut global_grad = vec![0.0; self.map.dim()];
        self.map.scatter_grad_a(&ga, &mut global_grad);
        self.map.scatter_grad_b(&gb, &mut global_grad);
        Ok(global_grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }

    fn prefer_fused_eval_grad(&self) -> bool {
        self.model_a.prefer_fused_eval_grad() || self.model_b.prefer_fused_eval_grad()
    }

    fn nll_grad_prepared(
        &self,
        _prepared: &Self::Prepared<'_>,
        params: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        self.validate_len(params)?;
        let pa = self.map.extract_a(params);
        let pb = self.map.extract_b(params);

        let nll_a = self.model_a.nll(&pa)?;
        let ga = self.model_a.grad_nll(&pa)?;
        let nll_b = self.model_b.nll(&pb)?;
        let gb = self.model_b.grad_nll(&pb)?;

        let mut global_grad = vec![0.0; self.map.dim()];
        self.map.scatter_grad_a(&ga, &mut global_grad);
        self.map.scatter_grad_b(&gb, &mut global_grad);
        Ok((nll_a + nll_b, global_grad))
    }
}

impl<A: LogDensityModel, B: LogDensityModel> PoiModel for HybridLikelihood<A, B> {
    fn poi_index(&self) -> Option<usize> {
        self.map.poi_global
    }
}

impl<A: LogDensityModel + FixedParamModel, B: LogDensityModel + FixedParamModel> FixedParamModel
    for HybridLikelihood<A, B>
{
    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        let mut new_map = self.map.clone();
        new_map.init[param_idx] = value;
        new_map.bounds[param_idx] = (value, value);

        let new_a = if let Some(ia) = self.map.map_a[param_idx] {
            self.model_a.with_fixed_param(ia, value)
        } else {
            // A doesn't have this param — clone via identity fix on param 0 at its current init.
            // Since FixedParamModel requires Sized, we need a concrete copy. Use a dummy fix.
            self.model_a.with_fixed_param(0, self.model_a.parameter_init()[0])
        };

        let new_b = if let Some(ib) = self.map.map_b[param_idx] {
            self.model_b.with_fixed_param(ib, value)
        } else {
            self.model_b.with_fixed_param(0, self.model_b.parameter_init()[0])
        };

        Self { model_a: new_a, model_b: new_b, map: new_map }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_core::traits::PreparedModelRef;

    /// Minimal test model: quadratic NLL with named parameters.
    struct QuadModel {
        names: Vec<String>,
        centers: Vec<f64>,
    }

    impl QuadModel {
        fn new(names: &[&str], centers: &[f64]) -> Self {
            Self { names: names.iter().map(|s| s.to_string()).collect(), centers: centers.to_vec() }
        }
    }

    impl LogDensityModel for QuadModel {
        type Prepared<'a>
            = PreparedModelRef<'a, Self>
        where
            Self: 'a;

        fn dim(&self) -> usize {
            self.names.len()
        }

        fn parameter_names(&self) -> Vec<String> {
            self.names.clone()
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(-10.0, 10.0); self.names.len()]
        }

        fn parameter_init(&self) -> Vec<f64> {
            self.centers.clone()
        }

        fn nll(&self, params: &[f64]) -> Result<f64> {
            Ok(params.iter().zip(&self.centers).map(|(&x, &c)| 0.5 * (x - c).powi(2)).sum())
        }

        fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
            Ok(params.iter().zip(&self.centers).map(|(&x, &c)| x - c).collect())
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    impl FixedParamModel for QuadModel {
        fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
            let mut out = Self::new(
                &self.names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                &self.centers,
            );
            out.centers[param_idx] = value;
            out
        }
    }

    #[test]
    fn test_shared_parameter_map_no_overlap() {
        let a = QuadModel::new(&["mu", "alpha"], &[1.0, 0.0]);
        let b = QuadModel::new(&["beta", "gamma"], &[2.0, 3.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();

        assert_eq!(map.dim(), 4);
        assert_eq!(map.n_shared(), 0);
        assert_eq!(map.names, vec!["mu", "alpha", "beta", "gamma"]);

        let global = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(map.extract_a(&global), vec![1.0, 2.0]);
        assert_eq!(map.extract_b(&global), vec![3.0, 4.0]);
    }

    #[test]
    fn test_shared_parameter_map_with_overlap() {
        let a = QuadModel::new(&["mu", "lumi", "alpha"], &[1.0, 1.0, 0.0]);
        let b = QuadModel::new(&["lumi", "beta"], &[1.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();

        assert_eq!(map.dim(), 4); // mu, lumi(shared), alpha, beta
        assert_eq!(map.n_shared(), 1);
        assert_eq!(map.names, vec!["mu", "lumi", "alpha", "beta"]);

        let global = vec![2.0, 1.5, 0.5, 3.0];
        assert_eq!(map.extract_a(&global), vec![2.0, 1.5, 0.5]); // mu, lumi, alpha
        assert_eq!(map.extract_b(&global), vec![1.5, 3.0]); // lumi, beta
    }

    #[test]
    fn test_shared_parameter_map_incompatible_bounds() {
        struct BoundedModel {
            names: Vec<String>,
            bounds: Vec<(f64, f64)>,
        }
        impl LogDensityModel for BoundedModel {
            type Prepared<'a>
                = PreparedModelRef<'a, Self>
            where
                Self: 'a;
            fn dim(&self) -> usize {
                self.names.len()
            }
            fn parameter_names(&self) -> Vec<String> {
                self.names.clone()
            }
            fn parameter_bounds(&self) -> Vec<(f64, f64)> {
                self.bounds.clone()
            }
            fn parameter_init(&self) -> Vec<f64> {
                vec![0.0; self.names.len()]
            }
            fn nll(&self, _: &[f64]) -> Result<f64> {
                Ok(0.0)
            }
            fn grad_nll(&self, _: &[f64]) -> Result<Vec<f64>> {
                Ok(vec![0.0; self.names.len()])
            }
            fn prepared(&self) -> Self::Prepared<'_> {
                PreparedModelRef::new(self)
            }
        }

        let a = BoundedModel { names: vec!["x".into()], bounds: vec![(0.0, 1.0)] };
        let b = BoundedModel { names: vec!["x".into()], bounds: vec![(2.0, 3.0)] };
        let err = SharedParameterMap::build(&a, &b).unwrap_err();
        assert!(err.to_string().contains("incompatible bounds"));
    }

    #[test]
    fn test_hybrid_nll_no_overlap() {
        let a = QuadModel::new(&["mu"], &[0.0]);
        let b = QuadModel::new(&["beta"], &[0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();
        let hybrid = HybridLikelihood::new(a, b, map);

        assert_eq!(hybrid.dim(), 2);
        let nll = hybrid.nll(&[3.0, 4.0]).unwrap();
        // 0.5*3^2 + 0.5*4^2 = 4.5 + 8.0 = 12.5
        assert!((nll - 12.5).abs() < 1e-12);
    }

    #[test]
    fn test_hybrid_nll_with_shared_param() {
        let a = QuadModel::new(&["mu", "lumi"], &[0.0, 0.0]);
        let b = QuadModel::new(&["lumi", "sigma"], &[0.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();
        let hybrid = HybridLikelihood::new(a, b, map);

        assert_eq!(hybrid.dim(), 3); // mu, lumi, sigma
        // global = [mu=1, lumi=2, sigma=3]
        // A sees [1, 2] → NLL_A = 0.5*1 + 0.5*4 = 2.5
        // B sees [2, 3] → NLL_B = 0.5*4 + 0.5*9 = 6.5
        // Total = 9.0
        let nll = hybrid.nll(&[1.0, 2.0, 3.0]).unwrap();
        assert!((nll - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_hybrid_gradient_shared_param() {
        let a = QuadModel::new(&["mu", "lumi"], &[0.0, 0.0]);
        let b = QuadModel::new(&["lumi", "sigma"], &[0.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();
        let hybrid = HybridLikelihood::new(a, b, map);

        // global = [mu=1, lumi=2, sigma=3]
        // grad_A = [1, 2], grad_B = [2, 3]
        // global_grad:
        //   mu:    grad_A[0] = 1
        //   lumi:  grad_A[1] + grad_B[0] = 2 + 2 = 4  (shared!)
        //   sigma: grad_B[1] = 3
        let grad = hybrid.grad_nll(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(grad.len(), 3);
        assert!((grad[0] - 1.0).abs() < 1e-12);
        assert!((grad[1] - 4.0).abs() < 1e-12);
        assert!((grad[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_hybrid_poi_from_a() {
        let a = QuadModel::new(&["mu", "alpha"], &[1.0, 0.0]);
        let b = QuadModel::new(&["alpha", "beta"], &[0.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap().with_poi_from_a(0);
        let hybrid = HybridLikelihood::new(a, b, map);

        assert_eq!(hybrid.poi_index(), Some(0)); // mu is global[0]
    }

    #[test]
    fn test_hybrid_poi_from_b() {
        let a = QuadModel::new(&["alpha"], &[0.0]);
        let b = QuadModel::new(&["mu", "beta"], &[1.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap().with_poi_from_b(0);
        let hybrid = HybridLikelihood::new(a, b, map);

        assert_eq!(hybrid.poi_index(), Some(1)); // mu is global[1] (after alpha)
    }

    #[test]
    fn test_hybrid_length_mismatch() {
        let a = QuadModel::new(&["mu"], &[0.0]);
        let b = QuadModel::new(&["beta"], &[0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();
        let hybrid = HybridLikelihood::new(a, b, map);

        let err = hybrid.nll(&[1.0]).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn test_fixed_param_model() {
        let a = QuadModel::new(&["mu", "lumi"], &[0.0, 0.0]);
        let b = QuadModel::new(&["lumi", "sigma"], &[0.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap().with_poi_from_a(0);
        let hybrid = HybridLikelihood::new(a, b, map);

        // Fix mu (global[0]) at 5.0
        let fixed = hybrid.with_fixed_param(0, 5.0);
        assert_eq!(fixed.parameter_map().bounds[0], (5.0, 5.0));
        assert_eq!(fixed.parameter_map().init[0], 5.0);

        // NLL should use fixed value: mu=5 contributes 0.5*25 from A
        // lumi=0, sigma=0 → 0
        let nll = fixed.nll(&[5.0, 0.0, 0.0]).unwrap();
        // A sees [5, 0] with centers [5, 0] → 0.5*0 + 0.5*0 = 0 (center shifted to 5)
        assert!((nll - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_scatter_gradient_correctness() {
        let a = QuadModel::new(&["x", "y", "z"], &[0.0, 0.0, 0.0]);
        let b = QuadModel::new(&["y", "w"], &[0.0, 0.0]);
        let map = SharedParameterMap::build(&a, &b).unwrap();
        // global: x, y(shared), z, w

        let mut g = vec![0.0; 4];
        let local_a = vec![10.0, 20.0, 30.0]; // dx, dy, dz
        map.scatter_grad_a(&local_a, &mut g);
        assert_eq!(g, vec![10.0, 20.0, 30.0, 0.0]);

        let local_b = vec![5.0, 7.0]; // dy, dw
        map.scatter_grad_b(&local_b, &mut g);
        assert_eq!(g, vec![10.0, 25.0, 30.0, 7.0]); // y accumulated
    }
}
