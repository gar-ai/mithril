//! # mithril-cache
//!
//! Compilation caching for torch.compile.
//!
//! This crate provides a caching layer for PyTorch/Triton compilation artifacts,
//! enabling faster cold starts by reusing previously compiled kernels.
//!
//! ## Features
//!
//! - **Content-Addressable Storage (CAS)**: Store artifacts by content hash
//! - **LRU Eviction**: Automatically manage cache size with LRU eviction
//! - **Framework Hooks**: Intercept PyTorch/Triton cache directories
//! - **Portable Cache Keys**: Machine-independent cache key generation
//!
//! ## Quick Start
//!
//! ```no_run
//! use mithril_cache::hooks::{CacheConfig, init};
//!
//! # fn main() -> mithril_cache::hooks::Result<()> {
//! // Initialize cache with default config
//! let config = CacheConfig::new("/tmp/mithril-cache");
//! let manager = init(config)?;
//!
//! // Now PyTorch will use mithril-managed cache directories
//! // When you call torch.compile(), artifacts will be stored here
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     CACHE ARCHITECTURE                          │
//! │                                                                 │
//! │  ┌──────────────┐                         ┌──────────────────┐ │
//! │  │   PyTorch    │                         │   Cache Store    │ │
//! │  │  Inductor    │───────┐                 │                  │ │
//! │  └──────────────┘       │                 │  ┌────────────┐  │ │
//! │                         │   ┌─────────┐   │  │   Local    │  │ │
//! │  ┌──────────────┐       ├──▶│ Mithril │──▶│  │   Disk     │  │ │
//! │  │   Triton     │───────┘   │  Cache  │   │  └────────────┘  │ │
//! │  │   Kernels    │           │  Layer  │   │        │         │ │
//! │  └──────────────┘           └─────────┘   │        ▼         │ │
//! │                                           │  ┌────────────┐  │ │
//! │                                           │  │   Remote   │  │ │
//! │                                           │  │  (S3/GCS)  │  │ │
//! │                                           │  └────────────┘  │ │
//! │                                           └──────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Modules
//!
//! - [`cas`]: Content-addressable storage for artifacts
//! - [`keys`]: Cache key generation
//! - [`eviction`]: LRU cache eviction
//! - [`hooks`]: Framework integration via environment variables
//! - [`vllm`]: vLLM inference framework caching
//! - [`sglang`]: SGLang inference framework caching

pub mod cas;
pub mod compat;
pub mod environment;
pub mod eviction;
pub mod hooks;
pub mod keys;
pub mod metrics;
pub mod remote;
pub mod sglang;
pub mod vllm;

// Re-export commonly used types
pub use cas::ContentStore;
pub use compat::{CompatibilityChecker, CompatibilityRule};
pub use environment::{
    CompatibilityResult, CompatibilityStatus, ComputeCapability, Environment, RuleResult, Version,
};
pub use eviction::{CacheEntry, CacheStats, LruCache};
pub use hooks::{init, CacheConfig, CacheManager};
pub use keys::{CacheKey, DeviceClass, InputSpec};
pub use metrics::{CacheMetrics, MetricsSnapshot, MetricsSummary};
pub use remote::{PullStats, PushStats, RemoteCache, RemoteCacheConfig, SyncStats};
pub use sglang::{SglangCacheConfig, SglangCacheManager, SglangCacheStats};
pub use vllm::{VllmCacheConfig, VllmCacheManager, VllmCacheStats};
