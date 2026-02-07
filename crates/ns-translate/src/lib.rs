//! # ns-translate
//!
//! Format translators for NextStat.
//!
//! Supports:
//! - pyhf JSON format - Phase 1
//! - HistFactory XML - Phase 3
//! - ROOT files - Phase 3

#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

#[cfg(feature = "root-io")]
pub mod histfactory;
#[cfg(feature = "root-io")]
pub mod ntuple;
pub mod pyhf;
#[cfg(feature = "root-io")]
pub mod trex;

#[cfg(feature = "root-io")]
pub use ntuple::NtupleWorkspaceBuilder;
pub use pyhf::*;
