//! Re-export of the unbinned spec compiler from `ns-unbinned`.
//!
//! Kept as a thin wrapper to avoid churn in the CLI codebase (`mod unbinned_spec;`).

pub use ns_unbinned::spec::*;
