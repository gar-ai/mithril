//! Error types for Mithril.

use thiserror::Error;

/// Result type alias for Mithril operations.
pub type Result<T> = std::result::Result<T, MithrilError>;

/// Errors that can occur in Mithril operations.
#[derive(Error, Debug)]
pub enum MithrilError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Compression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Invalid data format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Key not found
    #[error("Key not found: {0}")]
    NotFound(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}
