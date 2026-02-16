//! Cabinetry configuration format support.
//!
//! Parses cabinetry YAML config files and converts the structure to a pyhf-compatible
//! `Workspace`. This module handles config structure only — histogram construction
//! (normally done by cabinetry via uproot) must be provided externally.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ns_translate::cabinetry::{parse_cabinetry_config, to_workspace};
//!
//! let yaml = std::fs::read_to_string("config.yml").unwrap();
//! let config = parse_cabinetry_config(&yaml).unwrap();
//! println!("Measurement: {}", config.general.measurement);
//! println!("Regions: {}", config.regions.len());
//! println!("Samples: {}", config.samples.len());
//! ```

pub mod convert;
pub mod schema;

pub use convert::*;
pub use schema::*;

#[cfg(test)]
mod tests;
