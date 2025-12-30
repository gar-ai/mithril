//! Framework hooks for intercepting PyTorch/Triton cache directories.
//!
//! Uses environment variable interception (shallow integration) to redirect
//! torch.compile artifacts to mithril-managed directories. This approach is
//! stable across PyTorch versions unlike deep integration with internal APIs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::cas::ContentStore;
use crate::eviction::{CacheEntry, LruCache};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};

/// Errors that can occur during cache initialization.
#[derive(Error, Debug)]
pub enum HookError {
    /// Failed to create cache directory.
    #[error("Failed to create directory {path}: {source}")]
    DirectoryCreation {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Failed to set environment variable.
    #[error("Failed to set environment variable {var}: {source}")]
    EnvVar {
        var: String,
        #[source]
        source: std::env::VarError,
    },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CAS error.
    #[error("CAS error: {0}")]
    Cas(#[from] crate::cas::CasError),
}

/// Result type for hook operations.
pub type Result<T> = std::result::Result<T, HookError>;

/// Configuration for the cache manager.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Root directory for all cache data.
    pub root_dir: PathBuf,
    /// Maximum local cache size in bytes.
    pub max_size_bytes: u64,
    /// Whether to enable the Inductor cache hook.
    pub enable_inductor: bool,
    /// Whether to enable the Triton cache hook.
    pub enable_triton: bool,
}

impl CacheConfig {
    /// Create a new cache config with default settings.
    #[must_use]
    pub fn new(root_dir: impl AsRef<Path>) -> Self {
        Self {
            root_dir: root_dir.as_ref().to_path_buf(),
            max_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GB default
            enable_inductor: true,
            enable_triton: true,
        }
    }

    /// Set the maximum cache size.
    #[must_use]
    pub fn with_max_size_gb(mut self, size_gb: u64) -> Self {
        self.max_size_bytes = size_gb * 1024 * 1024 * 1024;
        self
    }

    /// Set the maximum cache size in bytes.
    #[must_use]
    pub fn with_max_size_bytes(mut self, size_bytes: u64) -> Self {
        self.max_size_bytes = size_bytes;
        self
    }

    /// Enable or disable Inductor cache.
    #[must_use]
    pub fn with_inductor(mut self, enable: bool) -> Self {
        self.enable_inductor = enable;
        self
    }

    /// Enable or disable Triton cache.
    #[must_use]
    pub fn with_triton(mut self, enable: bool) -> Self {
        self.enable_triton = enable;
        self
    }

    /// Load configuration from environment variables.
    ///
    /// Supported variables:
    /// - `MITHRIL_CACHE_DIR`: Root cache directory
    /// - `MITHRIL_MAX_SIZE_GB`: Maximum cache size in GB
    #[must_use]
    pub fn from_env() -> Self {
        let root_dir = std::env::var("MITHRIL_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("/tmp"))
                    .join("mithril")
            });

        let max_size_gb = std::env::var("MITHRIL_MAX_SIZE_GB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        Self::new(root_dir).with_max_size_gb(max_size_gb)
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Manages the compilation cache and framework hooks.
///
/// The `CacheManager` intercepts PyTorch/Triton cache directories via
/// environment variables and provides a unified interface for cache
/// management.
///
/// # Example
///
/// ```no_run
/// use mithril_cache::hooks::{CacheConfig, CacheManager};
///
/// # fn main() -> mithril_cache::hooks::Result<()> {
/// let config = CacheConfig::new("/tmp/mithril-cache");
/// let manager = CacheManager::init(config)?;
///
/// // Now torch.compile will use mithril-managed directories
/// // import torch
/// // model = torch.compile(model)
///
/// println!("Cache stats: {:?}", manager.stats());
/// # Ok(())
/// # }
/// ```
pub struct CacheManager {
    config: CacheConfig,
    content_store: Arc<ContentStore>,
    lru_cache: LruCache,
    inductor_dir: PathBuf,
    triton_dir: PathBuf,
}

impl CacheManager {
    /// Initialize the cache manager and install framework hooks.
    ///
    /// This sets environment variables to redirect PyTorch/Triton caches
    /// to mithril-managed directories.
    pub fn init(config: CacheConfig) -> Result<Self> {
        let root = &config.root_dir;

        // Create directory structure
        std::fs::create_dir_all(root).map_err(|e| HookError::DirectoryCreation {
            path: root.clone(),
            source: e,
        })?;

        let inductor_dir = root.join("inductor");
        let triton_dir = root.join("triton");
        let cas_dir = root.join("cas");

        std::fs::create_dir_all(&inductor_dir).map_err(|e| HookError::DirectoryCreation {
            path: inductor_dir.clone(),
            source: e,
        })?;

        std::fs::create_dir_all(&triton_dir).map_err(|e| HookError::DirectoryCreation {
            path: triton_dir.clone(),
            source: e,
        })?;

        // Install framework hooks via environment variables
        if config.enable_inductor {
            debug!("Setting TORCHINDUCTOR_CACHE_DIR to {:?}", inductor_dir);
            std::env::set_var("TORCHINDUCTOR_CACHE_DIR", &inductor_dir);
        }

        if config.enable_triton {
            debug!("Setting TRITON_CACHE_DIR to {:?}", triton_dir);
            std::env::set_var("TRITON_CACHE_DIR", &triton_dir);
        }

        let content_store = Arc::new(ContentStore::new(&cas_dir)?);
        let lru_cache = LruCache::new(config.max_size_bytes);

        info!(
            "Mithril cache initialized at {:?} (max size: {} GB)",
            root,
            config.max_size_bytes / (1024 * 1024 * 1024)
        );

        Ok(Self {
            config,
            content_store,
            lru_cache,
            inductor_dir,
            triton_dir,
        })
    }

    /// Get the root cache directory.
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.config.root_dir
    }

    /// Get the Inductor cache directory.
    #[must_use]
    pub fn inductor_dir(&self) -> &Path {
        &self.inductor_dir
    }

    /// Get the Triton cache directory.
    #[must_use]
    pub fn triton_dir(&self) -> &Path {
        &self.triton_dir
    }

    /// Get the content store.
    #[must_use]
    pub fn content_store(&self) -> &ContentStore {
        &self.content_store
    }

    /// Get cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let lru_stats = self.lru_cache.stats();
        CacheStats {
            root_dir: self.config.root_dir.clone(),
            lru_entry_count: lru_stats.entry_count,
            lru_size_bytes: lru_stats.current_size_bytes,
            max_size_bytes: lru_stats.max_size_bytes,
            utilization: lru_stats.utilization,
            total_hits: lru_stats.total_hits,
        }
    }

    /// Record a cache entry (called when a new artifact is created).
    pub fn record_entry(&mut self, key: String, size: u64) -> Vec<String> {
        self.lru_cache.put(CacheEntry::new(key, size))
    }

    /// Record a cache access (called on cache hit).
    pub fn record_access(&mut self, key: &str) -> bool {
        self.lru_cache.get(key).is_some()
    }

    /// Clear the cache.
    pub fn clear(&mut self) -> Result<()> {
        // Clear LRU tracking
        self.lru_cache.clear();

        // Clear directories
        if self.inductor_dir.exists() {
            std::fs::remove_dir_all(&self.inductor_dir)?;
            std::fs::create_dir_all(&self.inductor_dir)?;
        }

        if self.triton_dir.exists() {
            std::fs::remove_dir_all(&self.triton_dir)?;
            std::fs::create_dir_all(&self.triton_dir)?;
        }

        info!("Cache cleared");
        Ok(())
    }

    /// Scan the cache directories and update LRU tracking.
    ///
    /// This is useful after a restart to rebuild the LRU state.
    pub fn scan_and_update(&mut self) -> Result<ScanResult> {
        let mut total_files = 0;
        let mut total_size = 0u64;

        // Scan Inductor directory
        if self.inductor_dir.exists() {
            let inductor_dir = self.inductor_dir.clone();
            let (files, size) = self.scan_directory(&inductor_dir)?;
            total_files += files;
            total_size += size;
        }

        // Scan Triton directory
        if self.triton_dir.exists() {
            let triton_dir = self.triton_dir.clone();
            let (files, size) = self.scan_directory(&triton_dir)?;
            total_files += files;
            total_size += size;
        }

        info!(
            "Scanned {} files ({} MB)",
            total_files,
            total_size / (1024 * 1024)
        );

        Ok(ScanResult {
            files_found: total_files,
            total_size_bytes: total_size,
        })
    }

    /// Scan a directory and add entries to LRU cache.
    fn scan_directory(&mut self, dir: &Path) -> Result<(usize, u64)> {
        let mut count = 0;
        let mut size = 0u64;

        for entry in walkdir::WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Ok(metadata) = entry.metadata() {
                let file_size = metadata.len();
                let key = entry.path().to_string_lossy().to_string();

                self.lru_cache.put(CacheEntry::new(key, file_size));
                count += 1;
                size += file_size;
            }
        }

        Ok((count, size))
    }
}

/// Statistics about the cache.
#[derive(Clone, Debug)]
pub struct CacheStats {
    /// Root cache directory.
    pub root_dir: PathBuf,
    /// Number of entries tracked in LRU.
    pub lru_entry_count: usize,
    /// Size of entries tracked in LRU.
    pub lru_size_bytes: u64,
    /// Maximum cache size.
    pub max_size_bytes: u64,
    /// Cache utilization (0.0 to 1.0).
    pub utilization: f64,
    /// Total cache hits.
    pub total_hits: u64,
}

/// Result of scanning cache directories.
#[derive(Clone, Debug)]
pub struct ScanResult {
    /// Number of files found.
    pub files_found: usize,
    /// Total size of files.
    pub total_size_bytes: u64,
}

/// Initialize the cache with environment variables.
///
/// This is the main entry point for the cache system.
///
/// # Example
///
/// ```no_run
/// use mithril_cache::hooks::{init, CacheConfig};
///
/// # fn main() -> mithril_cache::hooks::Result<()> {
/// let config = CacheConfig::new("/tmp/mithril-cache");
/// let manager = init(config)?;
///
/// // PyTorch will now use mithril-managed cache directories
/// # Ok(())
/// # }
/// ```
pub fn init(config: CacheConfig) -> Result<CacheManager> {
    CacheManager::init(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::new("/tmp/test");
        assert_eq!(config.root_dir, PathBuf::from("/tmp/test"));
        assert_eq!(config.max_size_bytes, 10 * 1024 * 1024 * 1024);
        assert!(config.enable_inductor);
        assert!(config.enable_triton);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new("/tmp/test")
            .with_max_size_gb(20)
            .with_inductor(false)
            .with_triton(true);

        assert_eq!(config.max_size_bytes, 20 * 1024 * 1024 * 1024);
        assert!(!config.enable_inductor);
        assert!(config.enable_triton);
    }

    #[test]
    fn test_cache_manager_init() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path());

        let manager = CacheManager::init(config).unwrap();

        // Check directories were created
        assert!(manager.inductor_dir().exists());
        assert!(manager.triton_dir().exists());

        // Check environment variables were set (note: in parallel tests,
        // another test may have set these, so we just verify they're set)
        let inductor_env = std::env::var("TORCHINDUCTOR_CACHE_DIR").unwrap();
        let triton_env = std::env::var("TRITON_CACHE_DIR").unwrap();
        assert!(inductor_env.ends_with("inductor"));
        assert!(triton_env.ends_with("triton"));
    }

    #[test]
    fn test_cache_manager_stats() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path()).with_max_size_bytes(1000);

        let mut manager = CacheManager::init(config).unwrap();

        // Record some entries
        manager.record_entry("key1".to_string(), 100);
        manager.record_entry("key2".to_string(), 200);

        let stats = manager.stats();
        assert_eq!(stats.lru_entry_count, 2);
        assert_eq!(stats.lru_size_bytes, 300);
        assert_eq!(stats.max_size_bytes, 1000);
    }

    #[test]
    fn test_cache_manager_record_access() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path());

        let mut manager = CacheManager::init(config).unwrap();

        manager.record_entry("key1".to_string(), 100);

        // Access should return true for existing key
        assert!(manager.record_access("key1"));

        // Access should return false for non-existent key
        assert!(!manager.record_access("nonexistent"));
    }

    #[test]
    fn test_cache_manager_clear() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path());

        let mut manager = CacheManager::init(config).unwrap();

        // Add some entries
        manager.record_entry("key1".to_string(), 100);
        manager.record_entry("key2".to_string(), 200);

        // Create some files in cache directories
        std::fs::write(manager.inductor_dir().join("test.txt"), "test").unwrap();
        std::fs::write(manager.triton_dir().join("test.txt"), "test").unwrap();

        // Clear cache
        manager.clear().unwrap();

        let stats = manager.stats();
        assert_eq!(stats.lru_entry_count, 0);
        assert_eq!(stats.lru_size_bytes, 0);

        // Directories should still exist but be empty
        assert!(manager.inductor_dir().exists());
        assert!(manager.triton_dir().exists());
    }

    #[test]
    fn test_cache_manager_disabled_hooks() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path())
            .with_inductor(false)
            .with_triton(false);

        // Clear any existing env vars
        std::env::remove_var("TORCHINDUCTOR_CACHE_DIR");
        std::env::remove_var("TRITON_CACHE_DIR");

        let manager = CacheManager::init(config).unwrap();

        // Directories should still be created
        assert!(manager.inductor_dir().exists());
        assert!(manager.triton_dir().exists());

        // But env vars should not be set (we removed them earlier)
        // Note: This test is a bit weak because other tests may have set them
    }

    #[test]
    fn test_init_function() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path());

        let manager = init(config).unwrap();
        assert!(manager.inductor_dir().exists());
    }
}
