# Mithril Style Guide

This guide ensures consistency across all three product crates. All agents must follow these patterns.

## Rust Conventions

### Error Handling

Always use the `?` operator with proper error context:

```rust
// Good
let data = storage.get(key)
    .await
    .map_err(|e| MithrilError::Storage(e))?;

// Better - with context
use anyhow::Context;
let data = storage.get(key)
    .await
    .context(format!("Failed to get key: {}", key))?;

// Bad - losing error information
let data = storage.get(key).await.unwrap();
```

### Async Patterns

```rust
// Good - async function
pub async fn process(&self, data: &[u8]) -> Result<Vec<u8>> {
    let compressed = self.compress(data).await?;
    self.storage.put("key", compressed.into()).await?;
    Ok(compressed)
}

// Good - CPU-bound work offloaded to rayon
pub async fn compress_large(&self, data: &[u8]) -> Result<Vec<u8>> {
    let data = data.to_vec();
    let compressor = self.compressor.clone();
    
    tokio::task::spawn_blocking(move || {
        compressor.compress(&data)
    }).await?
}

// Bad - blocking in async context
pub async fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
    // This blocks the async runtime!
    std::thread::sleep(Duration::from_secs(1));
    self.compressor.compress(data)
}
```

### Builder Pattern

Use builders for complex configurations:

```rust
// Good
pub struct CheckpointConfig {
    compression_level: CompressionLevel,
    delta_enabled: bool,
    // ... more fields
}

impl CheckpointConfig {
    pub fn builder() -> CheckpointConfigBuilder {
        CheckpointConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct CheckpointConfigBuilder {
    compression_level: Option<CompressionLevel>,
    delta_enabled: Option<bool>,
}

impl CheckpointConfigBuilder {
    pub fn compression_level(mut self, level: CompressionLevel) -> Self {
        self.compression_level = Some(level);
        self
    }
    
    pub fn delta_enabled(mut self, enabled: bool) -> Self {
        self.delta_enabled = Some(enabled);
        self
    }
    
    pub fn build(self) -> CheckpointConfig {
        CheckpointConfig {
            compression_level: self.compression_level.unwrap_or(CompressionLevel::Default),
            delta_enabled: self.delta_enabled.unwrap_or(true),
        }
    }
}

// Usage
let config = CheckpointConfig::builder()
    .compression_level(CompressionLevel::Best)
    .delta_enabled(true)
    .build();
```

### Trait Objects vs Generics

```rust
// Use generics when performance matters and types are known at compile time
pub struct Store<S: StorageBackend> {
    storage: S,
}

// Use trait objects when you need runtime polymorphism
pub struct DynamicStore {
    storage: Box<dyn StorageBackend>,
}

// Rule of thumb:
// - Internal implementation: generics
// - Public API boundaries: either (prefer generics)
// - Python bindings: trait objects (need runtime dispatch anyway)
```

## Naming Conventions

### Crate Names
- `mithril-core` (shared)
- `mithril-checkpoint` (product)
- `mithril-dedup` (product)
- `mithril-cache` (product)
- `mithril-python` (bindings)

### Module Names
```
src/
├── lib.rs              # Public exports only
├── config.rs           # Configuration types
├── error.rs            # Error types (if product-specific)
├── types.rs            # Shared types
├── feature/            # Feature modules as directories
│   ├── mod.rs
│   └── impl.rs
```

### Function Names
```rust
// Async functions: verb or verb_noun
async fn compress() -> Result<()>
async fn get_checkpoint() -> Result<Checkpoint>
async fn store_artifact() -> Result<String>

// Sync functions: same pattern
fn calculate_hash() -> Vec<u8>
fn estimate_size() -> usize

// Conversions
fn to_bytes(&self) -> Vec<u8>
fn from_bytes(data: &[u8]) -> Result<Self>
fn into_inner(self) -> T

// Getters (no get_ prefix in Rust)
fn size(&self) -> usize  // Not get_size()
fn name(&self) -> &str   // Not get_name()

// Predicates
fn is_empty(&self) -> bool
fn has_data(&self) -> bool
fn contains(&self, key: &str) -> bool
```

### Type Names
```rust
// Structs: PascalCase, noun
pub struct CheckpointWriter { }
pub struct MinHashSignature { }
pub struct CacheEntry { }

// Traits: PascalCase, adjective or capability
pub trait Compressible { }
pub trait StorageBackend { }
pub trait ContentAddressable { }

// Enums: PascalCase
pub enum CompressionLevel { }
pub enum StorageError { }

// Type aliases: PascalCase
pub type Result<T> = std::result::Result<T, MithrilError>;
```

## Documentation

### Module Documentation
```rust
//! # Checkpoint Compression
//!
//! This module provides checkpoint compression for ML training.
//!
//! ## Example
//!
//! ```rust
//! use mithril_checkpoint::Compressor;
//!
//! let compressor = Compressor::new()?;
//! let compressed = compressor.compress(&checkpoint)?;
//! ```
//!
//! ## Features
//!
//! - Delta encoding between checkpoints
//! - Adaptive quantization
//! - bfloat16 byte grouping
```

### Function Documentation
```rust
/// Compress a checkpoint with delta encoding.
///
/// This function compresses the given checkpoint data, optionally using
/// delta encoding against a previous checkpoint.
///
/// # Arguments
///
/// * `data` - The checkpoint data to compress
/// * `previous` - Optional previous checkpoint for delta encoding
///
/// # Returns
///
/// The compressed checkpoint data.
///
/// # Errors
///
/// Returns `CompressionError` if compression fails.
///
/// # Example
///
/// ```rust
/// let compressed = compressor.compress(&data, None)?;
/// ```
pub fn compress(&self, data: &[u8], previous: Option<&[u8]>) -> Result<Vec<u8>> {
    // ...
}
```

### Inline Comments
```rust
// Good - explains WHY
// Use XxHash3 because it's 2x faster than XxHash64 on modern CPUs
let hasher = XxHash3::new();

// Bad - explains WHAT (obvious from code)
// Create a new hasher
let hasher = XxHash3::new();

// Good - explains non-obvious behavior
// Chunk size must be power of 2 for SIMD alignment
let chunk_size = 1 << 16;
```

## Testing

### Test Organization
```rust
// Unit tests at bottom of file
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compress_empty() {
        let compressor = ZstdCompressor::new(CompressionLevel::Default);
        let result = compressor.compress(&[]);
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_storage_roundtrip() {
        let storage = MemoryStorage::new();
        storage.put("key", Bytes::from("value")).await.unwrap();
        let data = storage.get("key").await.unwrap();
        assert_eq!(data.as_ref(), b"value");
    }
}

// Integration tests in tests/ directory
// tests/integration/checkpoint_test.rs
```

### Test Naming
```rust
#[test]
fn test_<function>_<scenario>_<expected>() { }

// Examples
fn test_compress_empty_input_returns_empty() { }
fn test_compress_large_data_succeeds() { }
fn test_decompress_invalid_data_returns_error() { }
```

### Test Helpers
```rust
// Create test fixtures module
#[cfg(test)]
mod fixtures {
    pub fn sample_checkpoint() -> Vec<u8> {
        // Return realistic test data
    }
    
    pub fn temp_storage() -> LocalStorage {
        LocalStorage::temp().unwrap()
    }
}
```

## Python Bindings (PyO3)

### Class Definitions
```rust
use pyo3::prelude::*;

/// Checkpoint compressor.
///
/// Example:
///     >>> compressor = Compressor()
///     >>> compressed = compressor.compress(data)
#[pyclass]
pub struct Compressor {
    inner: mithril_checkpoint::Compressor,
}

#[pymethods]
impl Compressor {
    /// Create a new compressor.
    ///
    /// Args:
    ///     level: Compression level (1-22, default 3)
    #[new]
    #[pyo3(signature = (level=3))]
    pub fn new(level: i32) -> PyResult<Self> {
        let inner = mithril_checkpoint::Compressor::new(level)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }
    
    /// Compress checkpoint data.
    ///
    /// Args:
    ///     data: Checkpoint data as bytes
    ///
    /// Returns:
    ///     Compressed data as bytes
    pub fn compress<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<&'py PyBytes> {
        let compressed = self.inner.compress(data)
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &compressed))
    }
}
```

### Async Functions
```rust
// Use pyo3-asyncio for async methods
use pyo3_asyncio::tokio::future_into_py;

#[pymethods]
impl Storage {
    pub fn get<'py>(&self, py: Python<'py>, key: String) -> PyResult<&'py PyAny> {
        let storage = self.inner.clone();
        future_into_py(py, async move {
            let data = storage.get(&key).await.map_err(to_py_err)?;
            Ok(data.to_vec())
        })
    }
}
```

### Module Organization
```rust
// mithril-python/src/lib.rs
use pyo3::prelude::*;

mod checkpoint;
mod dedup;
mod cache;

#[pymodule]
fn mithril(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add submodules
    m.add_submodule(checkpoint::module(_py)?)?;
    m.add_submodule(dedup::module(_py)?)?;
    m.add_submodule(cache::module(_py)?)?;
    Ok(())
}
```

## Logging

### Use tracing
```rust
use tracing::{debug, info, warn, error, instrument, Span};

#[instrument(skip(data), fields(size = data.len()))]
pub async fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
    info!("Starting compression");
    
    let result = self.inner_compress(data)?;
    
    debug!(
        original_size = data.len(),
        compressed_size = result.len(),
        ratio = data.len() as f64 / result.len() as f64,
        "Compression complete"
    );
    
    Ok(result)
}
```

### Log Levels
- `error!` - Operation failed, action required
- `warn!` - Unexpected but handled, worth attention
- `info!` - Significant events (start/complete operations)
- `debug!` - Detailed information for debugging
- `trace!` - Very verbose, rarely enabled

## Performance

### Avoid Unnecessary Allocations
```rust
// Good - reuse buffer
let mut buffer = Vec::with_capacity(expected_size);
for chunk in chunks {
    buffer.clear();
    process_into(&mut buffer, chunk)?;
}

// Bad - allocate every iteration
for chunk in chunks {
    let buffer = Vec::new();
    process(&chunk)?;
}
```

### Use Bytes Crate for Shared Data
```rust
use bytes::{Bytes, BytesMut};

// Bytes is cheaply cloneable (reference counted)
pub async fn store(&self, data: Bytes) -> Result<()> {
    // data can be shared without copying
}

// BytesMut for building data
let mut buffer = BytesMut::with_capacity(1024);
buffer.extend_from_slice(&header);
buffer.extend_from_slice(&payload);
let data = buffer.freeze();  // Convert to Bytes
```

### Parallelize CPU Work
```rust
use rayon::prelude::*;

// Good - parallel processing
let results: Vec<_> = chunks
    .par_iter()
    .map(|chunk| process(chunk))
    .collect();

// Good - with progress
let results: Vec<_> = chunks
    .par_iter()
    .progress_count(chunks.len() as u64)
    .map(|chunk| process(chunk))
    .collect();
```

## Dependencies

### Approved Dependencies
```toml
# Async runtime
tokio = { version = "1", features = ["full"] }

# Parallelism
rayon = "1"

# Bytes handling
bytes = "1"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Compression
zstd = "0.13"
lz4 = "1"

# Hashing
xxhash-rust = { version = "0.8", features = ["xxh3", "xxh64"] }
blake3 = "1"

# Storage
object_store = { version = "0.10", features = ["aws", "gcp"] }

# Python bindings
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"

# Error handling
thiserror = "1"
anyhow = "1"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# CLI (if needed)
clap = { version = "4", features = ["derive"] }

# Testing
criterion = "0.5"  # Benchmarks
proptest = "1"     # Property testing
```

### Adding New Dependencies
1. Check if functionality exists in approved deps
2. Prefer well-maintained, widely-used crates
3. Check license compatibility (MIT, Apache-2.0)
4. Document why it's needed in commit message
