//! # ns-core
//!
//! Core types, traits, and error handling for NextStat.
//!
//! This crate provides:
//! - Common error types
//! - Core traits (ComputeBackend, Model, etc.)
//! - Shared data structures
//! - Numerical precision utilities

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod traits;
pub mod types;

pub use error::{Error, Result};
pub use traits::{
    ComputeBackend, FixedParamModel, LogDensityModel, PoiModel, PreparedModelRef, PreparedNll,
};
pub use types::FitResult;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
