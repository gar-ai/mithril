//! Storage backend abstraction.
//!
//! Provides a unified interface for storing and retrieving data from various backends:
//! - Local filesystem (always available)
//! - S3-compatible storage (with `s3` feature)
//! - Google Cloud Storage (with `gcs` feature)

mod local;

#[cfg(feature = "s3")]
mod s3;

#[cfg(feature = "gcs")]
mod gcs;

pub use local::LocalStorage;

#[cfg(feature = "s3")]
pub use s3::S3Storage;

#[cfg(feature = "gcs")]
pub use gcs::GcsStorage;

use crate::error::Result;
use bytes::Bytes;

/// Trait for storage backends.
///
/// All methods are async to support both local and remote storage.
#[allow(async_fn_in_trait)]
pub trait StorageBackend: Send + Sync {
    /// Get data by key.
    async fn get(&self, key: &str) -> Result<Bytes>;

    /// Put data at key.
    async fn put(&self, key: &str, data: Bytes) -> Result<()>;

    /// Delete data at key.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Check if key exists.
    async fn exists(&self, key: &str) -> Result<bool>;

    /// List keys with given prefix.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;
}
