//! Re-export bijectors/transforms used by inference.
//!
//! This keeps the historical `ns-inference::transforms::*` API stable while the
//! implementation lives in `ns-prob` (Phase 5 shared modeling layer).

pub use ns_prob::transforms::{
    Bijector,
    ExpBijector,
    IdentityBijector,
    LowerBoundedBijector,
    ParameterTransform,
    SigmoidBijector,
};

