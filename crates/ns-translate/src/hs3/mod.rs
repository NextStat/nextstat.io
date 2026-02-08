//! HS3 (HEP Statistics Serialization Standard) format support.
//!
//! Provides native ingestion of HS3 v0.2 JSON files as produced by ROOT 6.37+.
//!
//! # Modules
//!
//! - [`schema`] — Serde types for HS3 JSON deserialization/serialization.
//! - [`resolve`] — Two-pass reference resolver.
//! - [`convert`] — HS3 → HistFactoryModel conversion.
//! - [`detect`] — Format auto-detection (pyhf vs HS3).
//! - [`export`] — HistFactoryModel → HS3 JSON export (roundtrip).

pub mod convert;
pub mod detect;
pub mod export;
pub mod resolve;
pub mod schema;

#[cfg(test)]
mod tests;

pub use convert::*;
pub use resolve::*;
pub use schema::*;
