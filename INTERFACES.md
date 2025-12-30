# Mithril Core Interfaces

This document defines the API contracts provided by `mithril-core`. All product crates depend on these interfaces. Changes to these interfaces require coordination across all agents.

## Storage Layer

### StorageBackend Trait

The primary abstraction for all storage operations.

```rust
use async_trait::async_trait;
use bytes::Bytes;

#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Get an object by key
    async fn get(&self, key: &str) -> Result<Bytes, StorageError>;
    
    /// Get an object, returning None if not found (no error)
    async fn get_opt(&self, key: &str) -> Result<Option<Bytes>, StorageError>;
    
    /// Put an object
    async fn put(&self, key: &str, data: Bytes) -> Result<(), StorageError>;
    
    /// Delete an object
    async fn delete(&self, key: &str) -> Result<(), StorageError>;
    
    /// Check if an object exists
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;
    
    /// List objects with a prefix
    async fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError>;
    
    /// Get object metadata without fetching content
    async fn head(&self, key: &str) -> Result<ObjectMeta, StorageError>;
    
    /// Stream get for large objects
    async fn get_stream(&self, key: &str) -> Result<BoxStream<'_, Result<Bytes, StorageError>>, StorageError>;
    
    /// Stream put for large objects
    async fn put_stream(&self, key: &str, stream: BoxStream<'_, Bytes>) -> Result<(), StorageError>;
}

#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub key: String,
    pub size: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub etag: Option<String>,
    pub content_type: Option<String>,
}

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Object not found: {0}")]
    NotFound(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Connection error: {0}")]
    Connection(String),
    
    #[error("Invalid key: {0}")]
    InvalidKey(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Other: {0}")]
    Other(String),
}
```

### Storage Implementations

```rust
// Local filesystem storage
pub struct LocalStorage {
    root: PathBuf,
}

impl LocalStorage {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, StorageError>;
    pub fn temp() -> Result<Self, StorageError>;  // Creates temp directory
}

// S3-compatible storage
pub struct S3Storage {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: Option<String>,
}

impl S3Storage {
    pub async fn new(bucket: &str, prefix: Option<&str>) -> Result<Self, StorageError>;
    pub async fn from_env() -> Result<Self, StorageError>;  // Uses AWS_* env vars
}

// GCS storage
pub struct GcsStorage {
    client: cloud_storage::Client,
    bucket: String,
    prefix: Option<String>,
}

impl GcsStorage {
    pub async fn new(bucket: &str, prefix: Option<&str>) -> Result<Self, StorageError>;
}

// In-memory storage (for testing)
pub struct MemoryStorage {
    data: Arc<RwLock<HashMap<String, Bytes>>>,
}

impl MemoryStorage {
    pub fn new() -> Self;
}
```

### Storage Factory

```rust
pub enum StorageConfig {
    Local { path: PathBuf },
    S3 { bucket: String, prefix: Option<String> },
    Gcs { bucket: String, prefix: Option<String> },
    Memory,
}

pub fn create_storage(config: StorageConfig) -> Result<Box<dyn StorageBackend>, StorageError>;
```

## Compression Layer

### Compressor Trait

```rust
pub trait Compressor: Send + Sync {
    /// Compress data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    
    /// Decompress data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    
    /// Compress with streaming (for large data)
    fn compress_stream<R: Read, W: Write>(&self, reader: R, writer: W) -> Result<u64, CompressionError>;
    
    /// Decompress with streaming
    fn decompress_stream<R: Read, W: Write>(&self, reader: R, writer: W) -> Result<u64, CompressionError>;
    
    /// Get compression ratio estimate for data
    fn estimate_ratio(&self, sample: &[u8]) -> f32;
    
    /// Name of the compressor (for metadata)
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy)]
pub enum CompressionLevel {
    Fastest,
    Fast,
    Default,
    Best,
    Custom(i32),
}

#[derive(Error, Debug)]
pub enum CompressionError {
    #[error("Compression failed: {0}")]
    CompressionFailed(String),
    
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
    
    #[error("Invalid data")]
    InvalidData,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

### Compressor Implementations

```rust
// Zstd compressor (default, best ratio)
pub struct ZstdCompressor {
    level: CompressionLevel,
}

impl ZstdCompressor {
    pub fn new(level: CompressionLevel) -> Self;
    pub fn with_dictionary(dict: &[u8], level: CompressionLevel) -> Self;
}

// LZ4 compressor (fastest)
pub struct Lz4Compressor {
    level: CompressionLevel,
}

impl Lz4Compressor {
    pub fn new(level: CompressionLevel) -> Self;
}

// Adaptive compressor (chooses based on data)
pub struct AdaptiveCompressor {
    compressors: Vec<Box<dyn Compressor>>,
}

impl AdaptiveCompressor {
    pub fn new() -> Self;
    pub fn with_compressors(compressors: Vec<Box<dyn Compressor>>) -> Self;
}
```

## Hashing Layer

### HashFunction Trait

```rust
pub trait HashFunction: Send + Sync {
    /// Hash data to bytes
    fn hash(&self, data: &[u8]) -> Vec<u8>;
    
    /// Hash data to hex string
    fn hash_hex(&self, data: &[u8]) -> String;
    
    /// Hash data to u64 (for bloom filters, etc.)
    fn hash_u64(&self, data: &[u8]) -> u64;
    
    /// Create incremental hasher
    fn hasher(&self) -> Box<dyn Hasher>;
    
    /// Output size in bytes
    fn output_size(&self) -> usize;
    
    /// Name of the hash function
    fn name(&self) -> &'static str;
}

pub trait Hasher: Send {
    fn update(&mut self, data: &[u8]);
    fn finalize(self: Box<Self>) -> Vec<u8>;
    fn finalize_u64(self: Box<Self>) -> u64;
}
```

### Hash Implementations

```rust
// XXHash (fast, good distribution)
pub struct XxHash;
impl XxHash {
    pub fn new() -> Self;
}

// XXHash3 (faster on modern CPUs)
pub struct XxHash3;

// Blake3 (cryptographic but fast)
pub struct Blake3Hash;

// SHA256 (when cryptographic security needed)
pub struct Sha256Hash;
```

### Content-Addressable Storage

```rust
pub trait ContentAddressable: Send + Sync {
    /// Store content and return its address (hash)
    async fn store(&self, data: Bytes) -> Result<String, CasError>;
    
    /// Retrieve content by address
    async fn retrieve(&self, address: &str) -> Result<Bytes, CasError>;
    
    /// Check if content exists
    async fn contains(&self, address: &str) -> Result<bool, CasError>;
    
    /// Delete content (if reference count allows)
    async fn delete(&self, address: &str) -> Result<(), CasError>;
    
    /// Get stats about stored content
    async fn stats(&self) -> Result<CasStats, CasError>;
}

pub struct CasStats {
    pub total_objects: u64,
    pub total_bytes: u64,
    pub deduplicated_bytes: u64,
}

pub struct ContentAddressedStore<S: StorageBackend, H: HashFunction> {
    storage: S,
    hasher: H,
}

impl<S: StorageBackend, H: HashFunction> ContentAddressedStore<S, H> {
    pub fn new(storage: S, hasher: H) -> Self;
}
```

### MinHash (for Dedup)

```rust
pub struct MinHasher {
    num_permutations: usize,
    seed: u64,
}

impl MinHasher {
    pub fn new(num_permutations: usize) -> Self;
    pub fn with_seed(num_permutations: usize, seed: u64) -> Self;
    
    /// Compute MinHash signature for a set of tokens
    pub fn hash_tokens(&self, tokens: &[u64]) -> MinHashSignature;
    
    /// Compute MinHash signature for text (tokenizes internally)
    pub fn hash_text(&self, text: &str, ngram_size: usize) -> MinHashSignature;
    
    /// Estimate Jaccard similarity between two signatures
    pub fn similarity(sig1: &MinHashSignature, sig2: &MinHashSignature) -> f64;
}

#[derive(Clone, Debug)]
pub struct MinHashSignature {
    pub values: Vec<u64>,
}

impl MinHashSignature {
    /// Get LSH bands for bucketing
    pub fn bands(&self, num_bands: usize) -> Vec<u64>;
}
```

## Async Utilities

### Runtime Helpers

```rust
/// Get or create the global tokio runtime
pub fn runtime() -> &'static tokio::runtime::Runtime;

/// Spawn a task on the runtime
pub fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static;

/// Spawn blocking work on rayon
pub fn spawn_blocking<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static;

/// Run CPU-intensive work on rayon thread pool
pub fn par_compute<T, F>(items: Vec<T>, f: F) -> Vec<T::Output>
where
    T: Send,
    F: Fn(T) -> T::Output + Sync,
    T::Output: Send;
```

### Progress Tracking

```rust
pub trait ProgressReporter: Send + Sync {
    fn set_total(&self, total: u64);
    fn increment(&self, amount: u64);
    fn set_message(&self, message: &str);
    fn finish(&self);
}

pub struct NoopProgress;
pub struct ConsoleProgress { /* uses indicatif */ }
pub struct CallbackProgress<F: Fn(u64, u64, &str)> { callback: F }
```

## Types

### Tensor Metadata

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
}

impl DType {
    pub fn size_bytes(&self) -> usize;
    pub fn from_numpy_str(s: &str) -> Option<Self>;
    pub fn to_numpy_str(&self) -> &'static str;
}
```

### Common Result Type

```rust
pub type Result<T> = std::result::Result<T, MithrilError>;

#[derive(Error, Debug)]
pub enum MithrilError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Compression error: {0}")]
    Compression(#[from] CompressionError),
    
    #[error("Hash error: {0}")]
    Hash(String),
    
    #[error("CAS error: {0}")]
    Cas(#[from] CasError),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    #[error("{0}")]
    Other(String),
}
```

## PyO3 Helpers

### Python Conversion Utilities

```rust
use pyo3::prelude::*;
use numpy::PyArray1;

/// Convert numpy array to Rust bytes (zero-copy when possible)
pub fn numpy_to_bytes<'py>(py: Python<'py>, arr: &'py PyAny) -> PyResult<&'py [u8]>;

/// Convert Rust bytes to numpy array
pub fn bytes_to_numpy<'py>(py: Python<'py>, data: &[u8], dtype: DType) -> PyResult<&'py PyAny>;

/// Async Python integration - run Rust future from Python
pub fn run_async<F, T>(py: Python<'_>, future: F) -> PyResult<T>
where
    F: Future<Output = Result<T>>,
    T: IntoPy<PyObject>;
```

### Common Python Exceptions

```rust
pyo3::create_exception!(mithril, MithrilException, pyo3::exceptions::PyException);
pyo3::create_exception!(mithril, StorageException, MithrilException);
pyo3::create_exception!(mithril, CompressionException, MithrilException);

/// Convert MithrilError to appropriate Python exception
pub fn to_py_err(err: MithrilError) -> PyErr;
```

## Configuration

### Config Loading

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    pub storage: StorageConfig,
    pub temp_dir: Option<PathBuf>,
    pub log_level: Option<String>,
}

impl CoreConfig {
    /// Load from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self>;
    
    /// Load from environment variables (MITHRIL_*)
    pub fn from_env() -> Result<Self>;
    
    /// Merge with another config (other takes precedence)
    pub fn merge(self, other: Self) -> Self;
}
```

## Usage Examples

### Storage

```rust
use mithril_core::{StorageBackend, LocalStorage, S3Storage};

// Local storage
let storage = LocalStorage::new("/tmp/mithril")?;
storage.put("key", Bytes::from("value")).await?;
let data = storage.get("key").await?;

// S3 storage (same interface)
let storage = S3Storage::new("my-bucket", Some("prefix/")).await?;
storage.put("key", Bytes::from("value")).await?;
```

### Compression

```rust
use mithril_core::{Compressor, ZstdCompressor, CompressionLevel};

let compressor = ZstdCompressor::new(CompressionLevel::Default);
let compressed = compressor.compress(&data)?;
let decompressed = compressor.decompress(&compressed)?;
assert_eq!(data, decompressed);
```

### Content-Addressed Storage

```rust
use mithril_core::{ContentAddressedStore, LocalStorage, XxHash3};

let storage = LocalStorage::new("/tmp/cas")?;
let cas = ContentAddressedStore::new(storage, XxHash3::new());

let address = cas.store(Bytes::from("content")).await?;
let retrieved = cas.retrieve(&address).await?;
```

### MinHash

```rust
use mithril_core::MinHasher;

let hasher = MinHasher::new(128);  // 128 permutations
let sig1 = hasher.hash_text("hello world", 3);  // 3-gram
let sig2 = hasher.hash_text("hello there world", 3);
let similarity = MinHasher::similarity(&sig1, &sig2);
```
