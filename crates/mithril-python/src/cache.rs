//! Python bindings for mithril-cache.

use mithril_cache::hooks::{CacheConfig, CacheManager, CacheStats};
use mithril_cache::ContentStore;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Python wrapper for CacheConfig.
#[pyclass(name = "CacheConfig")]
#[derive(Clone)]
pub struct PyCacheConfig {
    inner: CacheConfig,
}

#[pymethods]
impl PyCacheConfig {
    /// Create a new cache config.
    ///
    /// Args:
    ///     root_dir: Root directory for the cache
    #[new]
    fn new(root_dir: &str) -> Self {
        Self {
            inner: CacheConfig::new(root_dir),
        }
    }

    /// Set maximum cache size in gigabytes.
    fn with_max_size_gb(&self, size_gb: u64) -> Self {
        Self {
            inner: self.inner.clone().with_max_size_gb(size_gb),
        }
    }

    /// Enable/disable Inductor cache integration.
    fn with_inductor(&self, enable: bool) -> Self {
        Self {
            inner: self.inner.clone().with_inductor(enable),
        }
    }

    /// Enable/disable Triton cache integration.
    fn with_triton(&self, enable: bool) -> Self {
        Self {
            inner: self.inner.clone().with_triton(enable),
        }
    }

    /// Create config from environment variables.
    #[staticmethod]
    fn from_env() -> Self {
        Self {
            inner: CacheConfig::from_env(),
        }
    }

    #[getter]
    fn root_dir(&self) -> String {
        self.inner.root_dir.display().to_string()
    }

    #[getter]
    fn max_size_bytes(&self) -> u64 {
        self.inner.max_size_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheConfig(root_dir='{}', max_size_gb={})",
            self.inner.root_dir.display(),
            self.inner.max_size_bytes / (1024 * 1024 * 1024)
        )
    }
}

/// Python wrapper for CacheStats.
#[pyclass(name = "CacheStats")]
pub struct PyCacheStats {
    #[pyo3(get)]
    pub root_dir: String,
    #[pyo3(get)]
    pub lru_entry_count: usize,
    #[pyo3(get)]
    pub lru_size_bytes: u64,
    #[pyo3(get)]
    pub max_size_bytes: u64,
    #[pyo3(get)]
    pub utilization: f64,
    #[pyo3(get)]
    pub total_hits: u64,
}

impl From<CacheStats> for PyCacheStats {
    fn from(stats: CacheStats) -> Self {
        Self {
            root_dir: stats.root_dir.display().to_string(),
            lru_entry_count: stats.lru_entry_count,
            lru_size_bytes: stats.lru_size_bytes,
            max_size_bytes: stats.max_size_bytes,
            utilization: stats.utilization,
            total_hits: stats.total_hits,
        }
    }
}

#[pymethods]
impl PyCacheStats {
    fn __repr__(&self) -> String {
        format!(
            "CacheStats(entries={}, size_bytes={}, utilization={:.1}%)",
            self.lru_entry_count,
            self.lru_size_bytes,
            self.utilization * 100.0
        )
    }
}

/// Python wrapper for ScanResult.
#[pyclass(name = "ScanResult")]
pub struct PyScanResult {
    #[pyo3(get)]
    pub files_found: usize,
    #[pyo3(get)]
    pub total_size_bytes: u64,
}

#[pymethods]
impl PyScanResult {
    fn __repr__(&self) -> String {
        format!(
            "ScanResult(files={}, size_bytes={})",
            self.files_found, self.total_size_bytes
        )
    }
}

/// Python wrapper for CacheManager.
#[pyclass(name = "CacheManager")]
pub struct PyCacheManager {
    inner: std::sync::Mutex<CacheManager>,
}

#[pymethods]
impl PyCacheManager {
    /// Create a new cache manager.
    ///
    /// Args:
    ///     config: Cache configuration
    #[new]
    fn new(py: Python<'_>, config: PyCacheConfig) -> PyResult<Self> {
        let enable_inductor = config.inner.enable_inductor;
        let enable_triton = config.inner.enable_triton;

        let manager = mithril_cache::hooks::init(config.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Update Python's os.environ to reflect the env vars set by Rust.
        // This is necessary because Python's os.environ doesn't automatically
        // see changes made by native extensions via std::env::set_var().
        let os = py.import("os")?;
        let environ = os.getattr("environ")?;

        if enable_inductor {
            environ.set_item(
                "TORCHINDUCTOR_CACHE_DIR",
                manager.inductor_dir().to_string_lossy().as_ref(),
            )?;
        }

        if enable_triton {
            environ.set_item(
                "TRITON_CACHE_DIR",
                manager.triton_dir().to_string_lossy().as_ref(),
            )?;
        }

        Ok(Self {
            inner: std::sync::Mutex::new(manager),
        })
    }

    /// Get cache statistics.
    fn stats(&self) -> PyResult<PyCacheStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCacheStats::from(guard.stats()))
    }

    /// Clear all cache entries.
    fn clear(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        guard
            .clear()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Record a cache entry.
    ///
    /// Args:
    ///     key: Cache key
    ///     size: Size in bytes
    ///
    /// Returns:
    ///     List of evicted keys (if any)
    fn record_entry(&self, key: String, size: u64) -> PyResult<Vec<String>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(guard.record_entry(key, size))
    }

    /// Record a cache access (hit).
    ///
    /// Args:
    ///     key: Cache key
    ///
    /// Returns:
    ///     True if key was found, False otherwise
    fn record_access(&self, key: &str) -> PyResult<bool> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(guard.record_access(key))
    }

    /// Get the Inductor cache directory.
    fn inductor_dir(&self) -> PyResult<String> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(guard.inductor_dir().display().to_string())
    }

    /// Get the Triton cache directory.
    fn triton_dir(&self) -> PyResult<String> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(guard.triton_dir().display().to_string())
    }

    /// Scan cache directories and update LRU tracking.
    ///
    /// Returns:
    ///     ScanResult with files_found and total_size_bytes
    fn scan(&self) -> PyResult<PyScanResult> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let result = guard
            .scan_and_update()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyScanResult {
            files_found: result.files_found,
            total_size_bytes: result.total_size_bytes,
        })
    }

    fn __repr__(&self) -> String {
        match self.inner.lock() {
            Ok(guard) => {
                let stats = guard.stats();
                format!(
                    "CacheManager(entries={}, utilization={:.1}%)",
                    stats.lru_entry_count,
                    stats.utilization * 100.0
                )
            }
            Err(_) => "CacheManager(<locked>)".to_string(),
        }
    }
}

/// Python wrapper for ContentStore with native async support.
///
/// This is an async-capable content-addressable store.
#[pyclass(name = "ContentStore")]
pub struct PyContentStore {
    inner: Arc<ContentStore>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyContentStore {
    /// Create a new content store.
    ///
    /// Args:
    ///     root: Root directory for storage
    #[new]
    fn new(root: &str) -> PyResult<Self> {
        let runtime =
            Runtime::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let store = ContentStore::new(root)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(store),
            runtime: Arc::new(runtime),
        })
    }

    /// Store content and return its address.
    ///
    /// Args:
    ///     content: Raw bytes to store
    ///
    /// Returns:
    ///     Content address (hex string)
    fn put(&self, content: Vec<u8>) -> PyResult<String> {
        let store = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { store.put(&content).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Retrieve content by address.
    ///
    /// Args:
    ///     address: Content address (hex string)
    ///
    /// Returns:
    ///     Content bytes, or None if not found
    fn get(&self, address: String) -> PyResult<Option<Vec<u8>>> {
        let store = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { store.get(&address).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if content exists.
    ///
    /// Args:
    ///     address: Content address
    ///
    /// Returns:
    ///     True if content exists
    fn exists(&self, address: String) -> PyResult<bool> {
        let store = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { store.exists(&address).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Delete content by address.
    ///
    /// Args:
    ///     address: Content address
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete(&self, address: String) -> PyResult<bool> {
        let store = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { store.delete(&address).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Compute content address without storing.
    ///
    /// This is a synchronous method.
    ///
    /// Args:
    ///     content: Raw bytes
    ///
    /// Returns:
    ///     Content address (hex string)
    fn compute_address(&self, content: &[u8]) -> String {
        self.inner.compute_address(content)
    }

    /// Get the size of content by address.
    ///
    /// Args:
    ///     address: Content address
    ///
    /// Returns:
    ///     Size in bytes, or None if not found
    fn size(&self, address: String) -> PyResult<Option<u64>> {
        let store = Arc::clone(&self.inner);
        self.runtime
            .block_on(async move { store.size(&address).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get root directory path.
    fn root(&self) -> String {
        self.inner.root().display().to_string()
    }

    fn __repr__(&self) -> String {
        format!("ContentStore(root='{}')", self.inner.root().display())
    }
}

// ============================================================================
// Remote Cache (S3/GCS)
// ============================================================================

/// Python wrapper for PushStats.
#[pyclass(name = "PushStats")]
#[derive(Clone)]
pub struct PyPushStats {
    #[pyo3(get)]
    pub uploaded: usize,
    #[pyo3(get)]
    pub skipped: usize,
    #[pyo3(get)]
    pub bytes: u64,
    #[pyo3(get)]
    pub failed: usize,
}

#[pymethods]
impl PyPushStats {
    fn __repr__(&self) -> String {
        format!(
            "PushStats(uploaded={}, skipped={}, bytes={}, failed={})",
            self.uploaded, self.skipped, self.bytes, self.failed
        )
    }
}

/// Python wrapper for PullStats.
#[pyclass(name = "PullStats")]
#[derive(Clone)]
pub struct PyPullStats {
    #[pyo3(get)]
    pub downloaded: usize,
    #[pyo3(get)]
    pub skipped: usize,
    #[pyo3(get)]
    pub bytes: u64,
    #[pyo3(get)]
    pub failed: usize,
}

#[pymethods]
impl PyPullStats {
    fn __repr__(&self) -> String {
        format!(
            "PullStats(downloaded={}, skipped={}, bytes={}, failed={})",
            self.downloaded, self.skipped, self.bytes, self.failed
        )
    }
}

/// Python wrapper for SyncStats.
#[pyclass(name = "SyncStats")]
pub struct PySyncStats {
    #[pyo3(get)]
    pub push: PyPushStats,
    #[pyo3(get)]
    pub pull: PyPullStats,
}

#[pymethods]
impl PySyncStats {
    fn __repr__(&self) -> String {
        format!(
            "SyncStats(uploaded={}, downloaded={})",
            self.push.uploaded, self.pull.downloaded
        )
    }
}

/// Python wrapper for RemoteCacheConfig.
#[pyclass(name = "RemoteCacheConfig")]
#[derive(Clone)]
pub struct PyRemoteCacheConfig {
    pub auto_push: bool,
    pub lazy_pull: bool,
    pub max_concurrent: usize,
}

#[pymethods]
impl PyRemoteCacheConfig {
    /// Create a new remote cache config.
    ///
    /// Args:
    ///     auto_push: Automatically push to remote after local writes
    ///     lazy_pull: Pull from remote on cache miss
    ///     max_concurrent: Maximum concurrent transfers
    #[new]
    #[pyo3(signature = (auto_push=false, lazy_pull=true, max_concurrent=8))]
    fn new(auto_push: bool, lazy_pull: bool, max_concurrent: usize) -> Self {
        Self {
            auto_push,
            lazy_pull,
            max_concurrent,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RemoteCacheConfig(auto_push={}, lazy_pull={}, max_concurrent={})",
            self.auto_push, self.lazy_pull, self.max_concurrent
        )
    }
}

/// Python wrapper for RemoteCache with S3 backend.
///
/// Provides hybrid local+remote caching for cross-machine cache sharing.
#[cfg(feature = "s3")]
#[pyclass(name = "S3RemoteCache")]
pub struct PyS3RemoteCache {
    inner: std::sync::Mutex<mithril_cache::RemoteCache<mithril_core::S3Storage>>,
    runtime: Arc<Runtime>,
}

#[cfg(feature = "s3")]
#[pymethods]
impl PyS3RemoteCache {
    /// Create a new S3-backed remote cache.
    ///
    /// Args:
    ///     local_path: Path to local cache directory
    ///     bucket: S3 bucket name
    ///     prefix: S3 key prefix (e.g., "cache/")
    ///     region: AWS region (default: "us-east-1")
    ///     endpoint: Optional custom endpoint (for MinIO, LocalStack)
    ///     config: Optional RemoteCacheConfig
    #[new]
    #[pyo3(signature = (local_path, bucket, prefix, region="us-east-1", endpoint=None, config=None))]
    fn new(
        local_path: &str,
        bucket: &str,
        prefix: &str,
        region: &str,
        endpoint: Option<&str>,
        config: Option<PyRemoteCacheConfig>,
    ) -> PyResult<Self> {
        let runtime =
            Runtime::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let storage = runtime
            .block_on(mithril_core::S3Storage::new(
                bucket, region, endpoint, prefix,
            ))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let cache = if let Some(cfg) = config {
            let remote_config = mithril_cache::RemoteCacheConfig {
                auto_push: cfg.auto_push,
                lazy_pull: cfg.lazy_pull,
                max_concurrent: cfg.max_concurrent,
            };
            mithril_cache::RemoteCache::with_config(local_path, storage, remote_config)
        } else {
            mithril_cache::RemoteCache::new(local_path, storage)
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner: std::sync::Mutex::new(cache),
            runtime: Arc::new(runtime),
        })
    }

    /// Get an artifact, trying local first, then remote.
    ///
    /// Args:
    ///     address: Content address (hex string)
    ///
    /// Returns:
    ///     Content bytes, or None if not found
    fn get(&self, address: &str) -> PyResult<Option<Vec<u8>>> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.runtime
            .block_on(guard.get(address))
            .map(|opt| opt.map(|b| b.to_vec()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Store content locally.
    ///
    /// Args:
    ///     content: Raw bytes to store
    ///
    /// Returns:
    ///     Content address (hex string)
    fn put(&self, content: Vec<u8>) -> PyResult<String> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.runtime
            .block_on(guard.put(&content))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Push all local artifacts to remote storage.
    ///
    /// Returns:
    ///     PushStats with upload statistics
    fn push(&self) -> PyResult<PyPushStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let stats = self
            .runtime
            .block_on(guard.push())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyPushStats {
            uploaded: stats.uploaded,
            skipped: stats.skipped,
            bytes: stats.bytes,
            failed: stats.failed,
        })
    }

    /// Pull all remote artifacts to local storage.
    ///
    /// Returns:
    ///     PullStats with download statistics
    fn pull(&self) -> PyResult<PyPullStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let stats = self
            .runtime
            .block_on(guard.pull())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyPullStats {
            downloaded: stats.downloaded,
            skipped: stats.skipped,
            bytes: stats.bytes,
            failed: stats.failed,
        })
    }

    /// Sync local and remote caches bidirectionally.
    ///
    /// Returns:
    ///     SyncStats with push and pull statistics
    fn sync(&self) -> PyResult<PySyncStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let stats = self
            .runtime
            .block_on(guard.sync())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PySyncStats {
            push: PyPushStats {
                uploaded: stats.push.uploaded,
                skipped: stats.push.skipped,
                bytes: stats.push.bytes,
                failed: stats.push.failed,
            },
            pull: PyPullStats {
                downloaded: stats.pull.downloaded,
                skipped: stats.pull.skipped,
                bytes: stats.pull.bytes,
                failed: stats.pull.failed,
            },
        })
    }

    /// Warm local cache by pulling from remote.
    ///
    /// Alias for pull() - useful for CI/CD cache warming.
    fn warmup(&self) -> PyResult<PyPullStats> {
        self.pull()
    }

    fn __repr__(&self) -> String {
        "S3RemoteCache(...)".to_string()
    }
}

/// Register cache module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "cache")?;
    m.add_class::<PyCacheConfig>()?;
    m.add_class::<PyCacheStats>()?;
    m.add_class::<PyScanResult>()?;
    m.add_class::<PyCacheManager>()?;
    m.add_class::<PyContentStore>()?;

    // Remote cache classes
    m.add_class::<PyRemoteCacheConfig>()?;
    m.add_class::<PyPushStats>()?;
    m.add_class::<PyPullStats>()?;
    m.add_class::<PySyncStats>()?;

    #[cfg(feature = "s3")]
    m.add_class::<PyS3RemoteCache>()?;

    parent.add_submodule(&m)?;
    Ok(())
}
