//! # mithril-core
//!
//! Core infrastructure for Mithril ML tools.
//!
//! Provides shared abstractions for:
//! - Storage backends (local, S3, GCS)
//! - Compression (zstd, lz4)
//! - Hashing (xxhash, blake3)
//! - Common types (DType, TensorMeta)
//!
//! ## Status: NOT STARTED
//!
//! See `STATUS.md` for current progress.

pub mod compression;
pub mod error;
pub mod hashing;
pub mod storage;
pub mod types;

pub use compression::{Compressor, ZstdCompressor};
pub use error::{MithrilError, Result};
pub use hashing::{HashFunction, XxHash3};
#[cfg(feature = "gcs")]
pub use storage::GcsStorage;
#[cfg(feature = "s3")]
pub use storage::S3Storage;
pub use storage::{LocalStorage, StorageBackend};
pub use types::{DType, TensorMeta};
