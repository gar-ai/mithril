//! Remote cache for sharing torch.compile artifacts across machines.
//!
//! This module provides a hybrid local+remote cache that:
//! - Uses local disk for fast access (ContentStore)
//! - Syncs to remote storage (S3/GCS) for cross-machine sharing
//!
//! # Example
//!
//! ```rust,ignore
//! use mithril_cache::remote::{RemoteCache, RemoteCacheConfig};
//! use mithril_core::S3Storage;  // Requires "s3" feature
//!
//! # async fn example() -> mithril_cache::remote::Result<()> {
//! // Create remote cache backed by S3
//! let storage = S3Storage::from_env("my-bucket", "cache/").await?;
//! let cache = RemoteCache::new("/tmp/local-cache", storage)?;
//!
//! // Push local cache to remote
//! let stats = cache.push().await?;
//! println!("Pushed {} artifacts ({} bytes)", stats.uploaded, stats.bytes);
//!
//! // Pull from remote to local
//! let stats = cache.pull().await?;
//! println!("Pulled {} artifacts ({} bytes)", stats.downloaded, stats.bytes);
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use bytes::Bytes;
use mithril_core::storage::StorageBackend;
use mithril_core::MithrilError;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::cas::ContentStore;

/// Errors that can occur in remote cache operations.
#[derive(Error, Debug)]
pub enum RemoteCacheError {
    /// Local CAS error
    #[error("CAS error: {0}")]
    Cas(#[from] crate::cas::CasError),

    /// Remote storage error
    #[error("Storage error: {0}")]
    Storage(#[from] MithrilError),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for remote cache operations.
pub type Result<T> = std::result::Result<T, RemoteCacheError>;

/// Statistics from a push operation.
#[derive(Debug, Clone, Default)]
pub struct PushStats {
    /// Number of artifacts uploaded
    pub uploaded: usize,
    /// Number of artifacts skipped (already remote)
    pub skipped: usize,
    /// Total bytes uploaded
    pub bytes: u64,
    /// Number of failures
    pub failed: usize,
}

/// Statistics from a pull operation.
#[derive(Debug, Clone, Default)]
pub struct PullStats {
    /// Number of artifacts downloaded
    pub downloaded: usize,
    /// Number of artifacts skipped (already local)
    pub skipped: usize,
    /// Total bytes downloaded
    pub bytes: u64,
    /// Number of failures
    pub failed: usize,
}

/// Statistics from a sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    /// Push statistics
    pub push: PushStats,
    /// Pull statistics
    pub pull: PullStats,
}

/// Configuration for remote cache.
#[derive(Debug, Clone)]
pub struct RemoteCacheConfig {
    /// Enable auto-push after local cache writes
    pub auto_push: bool,
    /// Enable lazy pull (pull on cache miss)
    pub lazy_pull: bool,
    /// Max concurrent transfers
    pub max_concurrent: usize,
}

impl Default for RemoteCacheConfig {
    fn default() -> Self {
        Self {
            auto_push: false,
            lazy_pull: true,
            max_concurrent: 8,
        }
    }
}

/// Remote cache for cross-machine artifact sharing.
///
/// Combines a local `ContentStore` with remote `StorageBackend` to enable:
/// - Fast local cache access
/// - Cross-machine cache sharing via S3/GCS
/// - Lazy pull on cache miss
/// - Background push after local writes
pub struct RemoteCache<S: StorageBackend> {
    local: ContentStore,
    remote: S,
    config: RemoteCacheConfig,
}

impl<S: StorageBackend> RemoteCache<S> {
    /// Create a new remote cache.
    ///
    /// # Arguments
    /// * `local_path` - Path to local cache directory
    /// * `remote` - Remote storage backend (S3, GCS, etc.)
    pub fn new(local_path: impl AsRef<Path>, remote: S) -> Result<Self> {
        let local = ContentStore::new(local_path)?;
        Ok(Self {
            local,
            remote,
            config: RemoteCacheConfig::default(),
        })
    }

    /// Create with custom configuration.
    pub fn with_config(
        local_path: impl AsRef<Path>,
        remote: S,
        config: RemoteCacheConfig,
    ) -> Result<Self> {
        let local = ContentStore::new(local_path)?;
        Ok(Self {
            local,
            remote,
            config,
        })
    }

    /// Get the local content store.
    pub fn local(&self) -> &ContentStore {
        &self.local
    }

    /// Get the configuration.
    pub fn config(&self) -> &RemoteCacheConfig {
        &self.config
    }

    /// Get an artifact, trying local first, then remote.
    ///
    /// If `lazy_pull` is enabled and the artifact is found remotely,
    /// it will be stored locally for future access.
    pub async fn get(&self, address: &str) -> Result<Option<Bytes>> {
        // Try local first
        if let Some(data) = self.local.get_bytes(address).await? {
            debug!(address = %address, "Cache hit (local)");
            return Ok(Some(data));
        }

        // Try remote if lazy pull enabled
        if self.config.lazy_pull {
            match self.remote.get(address).await {
                Ok(data) => {
                    debug!(address = %address, bytes = data.len(), "Cache hit (remote), storing locally");
                    // Store locally for future access
                    self.local.put(&data).await?;
                    return Ok(Some(data));
                }
                Err(MithrilError::NotFound(_)) => {
                    debug!(address = %address, "Cache miss (remote)");
                }
                Err(e) => {
                    warn!(address = %address, error = %e, "Remote fetch failed");
                    return Err(RemoteCacheError::Storage(e));
                }
            }
        }

        Ok(None)
    }

    /// Store an artifact locally, optionally pushing to remote.
    pub async fn put(&self, content: &[u8]) -> Result<String> {
        let address = self.local.put(content).await?;

        if self.config.auto_push {
            if let Err(e) = self
                .remote
                .put(&address, Bytes::from(content.to_vec()))
                .await
            {
                warn!(address = %address, error = %e, "Auto-push failed");
            }
        }

        Ok(address)
    }

    /// Check if an artifact exists locally or remotely.
    pub async fn exists(&self, address: &str) -> Result<bool> {
        // Check local first
        if self.local.exists(address).await? {
            return Ok(true);
        }

        // Check remote
        match self.remote.exists(address).await {
            Ok(exists) => Ok(exists),
            Err(e) => {
                warn!(address = %address, error = %e, "Remote exists check failed");
                Ok(false) // Assume not found on error
            }
        }
    }

    /// Push all local artifacts to remote storage.
    ///
    /// Only uploads artifacts that don't already exist remotely.
    pub async fn push(&self) -> Result<PushStats> {
        let mut stats = PushStats::default();

        let local_addresses = self.local.list_all().await?;
        info!(
            count = local_addresses.len(),
            "Pushing local artifacts to remote"
        );

        for address in local_addresses {
            // Check if already remote
            match self.remote.exists(&address).await {
                Ok(true) => {
                    stats.skipped += 1;
                    continue;
                }
                Ok(false) => {}
                Err(e) => {
                    warn!(address = %address, error = %e, "Remote exists check failed, skipping");
                    stats.failed += 1;
                    continue;
                }
            }

            // Get local content
            let content = match self.local.get_bytes(&address).await? {
                Some(data) => data,
                None => {
                    warn!(address = %address, "Local content disappeared");
                    stats.failed += 1;
                    continue;
                }
            };

            // Upload to remote
            match self.remote.put(&address, content.clone()).await {
                Ok(()) => {
                    stats.uploaded += 1;
                    stats.bytes += content.len() as u64;
                    debug!(address = %address, bytes = content.len(), "Uploaded artifact");
                }
                Err(e) => {
                    warn!(address = %address, error = %e, "Upload failed");
                    stats.failed += 1;
                }
            }
        }

        info!(
            uploaded = stats.uploaded,
            skipped = stats.skipped,
            failed = stats.failed,
            bytes = stats.bytes,
            "Push complete"
        );

        Ok(stats)
    }

    /// Pull all remote artifacts to local storage.
    ///
    /// Only downloads artifacts that don't already exist locally.
    pub async fn pull(&self) -> Result<PullStats> {
        let mut stats = PullStats::default();

        let remote_addresses = self.remote.list("").await?;
        info!(
            count = remote_addresses.len(),
            "Pulling remote artifacts to local"
        );

        for address in remote_addresses {
            // Check if already local
            if self.local.exists(&address).await? {
                stats.skipped += 1;
                continue;
            }

            // Download from remote
            match self.remote.get(&address).await {
                Ok(data) => {
                    // Store locally
                    if let Err(e) = self.local.put(&data).await {
                        warn!(address = %address, error = %e, "Local store failed");
                        stats.failed += 1;
                        continue;
                    }

                    stats.downloaded += 1;
                    stats.bytes += data.len() as u64;
                    debug!(address = %address, bytes = data.len(), "Downloaded artifact");
                }
                Err(e) => {
                    warn!(address = %address, error = %e, "Download failed");
                    stats.failed += 1;
                }
            }
        }

        info!(
            downloaded = stats.downloaded,
            skipped = stats.skipped,
            failed = stats.failed,
            bytes = stats.bytes,
            "Pull complete"
        );

        Ok(stats)
    }

    /// Sync local and remote caches bidirectionally.
    ///
    /// Pulls remote artifacts not present locally, then pushes
    /// local artifacts not present remotely.
    pub async fn sync(&self) -> Result<SyncStats> {
        info!("Starting bidirectional sync");

        // Pull first (get any new remote artifacts)
        let pull = self.pull().await?;

        // Then push (share any new local artifacts)
        let push = self.push().await?;

        let stats = SyncStats { push, pull };

        info!(
            downloaded = stats.pull.downloaded,
            uploaded = stats.push.uploaded,
            "Sync complete"
        );

        Ok(stats)
    }

    /// Warm the local cache by pulling all remote artifacts.
    ///
    /// Alias for `pull()` - useful for CI/CD where you want to
    /// populate a fresh cache before compilation.
    pub async fn warmup(&self) -> Result<PullStats> {
        info!("Warming local cache from remote");
        self.pull().await
    }

    /// Export specific artifacts to remote.
    ///
    /// Useful for CI/CD where you want to push only the artifacts
    /// generated in this build.
    pub async fn export(&self, addresses: &[String]) -> Result<PushStats> {
        let mut stats = PushStats::default();

        info!(count = addresses.len(), "Exporting artifacts to remote");

        for address in addresses {
            // Get local content
            let content = match self.local.get_bytes(address).await? {
                Some(data) => data,
                None => {
                    warn!(address = %address, "Local content not found");
                    stats.failed += 1;
                    continue;
                }
            };

            // Upload to remote
            match self.remote.put(address, content.clone()).await {
                Ok(()) => {
                    stats.uploaded += 1;
                    stats.bytes += content.len() as u64;
                }
                Err(e) => {
                    warn!(address = %address, error = %e, "Export failed");
                    stats.failed += 1;
                }
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mithril_core::storage::LocalStorage;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_remote_cache_put_get() {
        let local_dir = TempDir::new().unwrap();
        let remote_dir = TempDir::new().unwrap();

        let remote = LocalStorage::new(remote_dir.path()).unwrap();
        let cache = RemoteCache::new(local_dir.path(), remote).unwrap();

        let content = b"hello world";
        let address = cache.put(content).await.unwrap();

        // Should find locally
        let data = cache.get(&address).await.unwrap();
        assert_eq!(data.unwrap().as_ref(), content);
    }

    #[tokio::test]
    async fn test_remote_cache_lazy_pull() {
        let local_dir = TempDir::new().unwrap();
        let remote_dir = TempDir::new().unwrap();

        // First, create a ContentStore to compute the content hash
        let content = b"remote content";
        let temp_store = ContentStore::new(TempDir::new().unwrap().path()).unwrap();
        let address = temp_store.compute_address(content);

        // Put content in remote storage using the content hash as key
        let remote = LocalStorage::new(remote_dir.path()).unwrap();
        remote
            .put(&address, Bytes::from_static(content))
            .await
            .unwrap();

        // Create cache with lazy pull enabled
        let config = RemoteCacheConfig {
            lazy_pull: true,
            ..Default::default()
        };
        let remote = LocalStorage::new(remote_dir.path()).unwrap();
        let cache = RemoteCache::with_config(local_dir.path(), remote, config).unwrap();

        // Should find remotely and cache locally
        let data = cache.get(&address).await.unwrap();
        assert_eq!(data.unwrap().as_ref(), content);

        // Should now be in local cache (stored under same content hash)
        assert!(cache.local().exists(&address).await.unwrap());
    }

    #[tokio::test]
    async fn test_push_pull_sync() {
        let local1_dir = TempDir::new().unwrap();
        let local2_dir = TempDir::new().unwrap();
        let remote_dir = TempDir::new().unwrap();

        // Create first cache and add content
        let remote1 = LocalStorage::new(remote_dir.path()).unwrap();
        let cache1 = RemoteCache::new(local1_dir.path(), remote1).unwrap();
        cache1.put(b"content 1").await.unwrap();
        cache1.put(b"content 2").await.unwrap();

        // Push to remote
        let push_stats = cache1.push().await.unwrap();
        assert_eq!(push_stats.uploaded, 2);
        assert_eq!(push_stats.skipped, 0);

        // Create second cache and pull
        let remote2 = LocalStorage::new(remote_dir.path()).unwrap();
        let cache2 = RemoteCache::new(local2_dir.path(), remote2).unwrap();
        let pull_stats = cache2.pull().await.unwrap();
        assert_eq!(pull_stats.downloaded, 2);
        assert_eq!(pull_stats.skipped, 0);

        // Verify content is the same
        let addresses1 = cache1.local().list_all().await.unwrap();
        let addresses2 = cache2.local().list_all().await.unwrap();
        assert_eq!(addresses1.len(), addresses2.len());
    }

    #[tokio::test]
    async fn test_push_skips_existing() {
        let local_dir = TempDir::new().unwrap();
        let remote_dir = TempDir::new().unwrap();

        let remote = LocalStorage::new(remote_dir.path()).unwrap();
        let cache = RemoteCache::new(local_dir.path(), remote).unwrap();

        // Add and push content
        cache.put(b"content").await.unwrap();
        let stats1 = cache.push().await.unwrap();
        assert_eq!(stats1.uploaded, 1);

        // Push again - should skip
        let stats2 = cache.push().await.unwrap();
        assert_eq!(stats2.uploaded, 0);
        assert_eq!(stats2.skipped, 1);
    }
}
