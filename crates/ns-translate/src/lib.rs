//! # ns-translate
//!
//! Format translators for NextStat.
//!
//! Supports:
//! - pyhf JSON format - Phase 1
//! - HistFactory XML - Phase 3
//! - ROOT files - Phase 3

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod pyhf;

pub use pyhf::*;
