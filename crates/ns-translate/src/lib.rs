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

pub mod histfactory;
pub mod ntuple;
pub mod pyhf;
pub mod trex;

pub use ntuple::NtupleWorkspaceBuilder;
pub use pyhf::*;
