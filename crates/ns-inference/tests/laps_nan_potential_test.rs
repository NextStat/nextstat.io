//! Regression test: first-call NaN potential_old must not trigger divergence.
//!
//! The CUDA/Metal MAMS kernels initialize `potential_old = NaN` as a sentinel.
//! On first call, `!isfinite(potential_old)` triggers unconditional accept.
//! If this guard is broken, all chains diverge on the first transition.
//!
//! Run: `cargo test -p ns-inference --test laps_nan_potential_test --features cuda -- --ignored`

#![cfg(any(feature = "cuda", feature = "metal"))]

use ns_inference::laps::{LapsConfig, LapsModel, sample_laps};

/// With n_warmup=1, n_samples=1 the first kernel launch always has potential_old=NaN.
/// If the isfinite guard is broken, all chains diverge and positions are NaN.
#[test]
#[ignore] // Requires GPU — run with --ignored
fn test_first_call_nan_potential_accepted() {
    let model = LapsModel::StdNormal { dim: 2 };
    let config = LapsConfig {
        n_chains: 64,
        n_warmup: 1,
        n_samples: 1,
        report_chains: 64,
        seed: 42,
        ..Default::default()
    };
    let result = sample_laps(&model, config).expect("LAPS sampling failed");

    // First transition must accept (not diverge)
    let n_div: usize = result
        .sampler_result
        .chains
        .iter()
        .flat_map(|c| c.divergences.iter())
        .filter(|&&d| d)
        .count();
    assert_eq!(n_div, 0, "first-call NaN potential must not trigger divergence");

    // Positions must be finite
    for (ci, chain) in result.sampler_result.chains.iter().enumerate() {
        for (di, &v) in chain.draws_constrained[0].iter().enumerate() {
            assert!(v.is_finite(), "chain {ci} dim {di}: first-call draw must be finite, got {v}");
        }
    }
}
