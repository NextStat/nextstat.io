//! Error types for ROOT file reading.

use thiserror::Error;

/// Errors that can occur reading ROOT files.
#[derive(Error, Debug)]
pub enum RootError {
    /// I/O error reading the file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid ROOT file magic bytes.
    #[error("not a ROOT file (bad magic)")]
    BadMagic,

    /// Unsupported ROOT file version.
    #[error("unsupported ROOT version: {0}")]
    UnsupportedVersion(u32),

    /// Buffer underflow (tried to read past end).
    #[error("unexpected end of buffer at offset {offset}, need {need} bytes, have {have}")]
    BufferUnderflow {
        /// Current offset in buffer.
        offset: usize,
        /// Bytes requested.
        need: usize,
        /// Bytes remaining.
        have: usize,
    },

    /// Key not found in directory.
    #[error("key not found: {0}")]
    KeyNotFound(String),

    /// Unsupported object class.
    #[error("unsupported class: {0}")]
    UnsupportedClass(String),

    /// Decompression failure.
    #[error("decompression error: {0}")]
    Decompression(String),

    /// Object deserialization error.
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Branch not found in tree.
    #[error("branch not found: {0}")]
    BranchNotFound(String),

    /// Tree not found in file.
    #[error("tree not found: {0}")]
    TreeNotFound(String),

    /// Expression parse or evaluation error.
    #[error("expression error: {0}")]
    Expression(String),

    /// Type mismatch (e.g. requesting f32 from an i64 branch).
    #[error("type mismatch: {0}")]
    TypeMismatch(String),

    /// Histogram filling error (policy violation, invalid binning, etc).
    #[error("histogram fill error: {0}")]
    HistogramFill(String),
}

/// Result alias for ROOT operations.
pub type Result<T> = std::result::Result<T, RootError>;
