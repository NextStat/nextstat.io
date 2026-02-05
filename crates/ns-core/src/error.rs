//! Error types for NextStat

use thiserror::Error;

/// NextStat error type
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Computation error
    #[error("Computation error: {0}")]
    Computation(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
