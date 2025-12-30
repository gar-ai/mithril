//! Content-Addressable Storage (CAS) for cache artifacts.
//!
//! Stores content by its Blake3 hash, enabling deduplication and
//! fast retrieval. Target: <10ms local lookup.

use std::path::{Path, PathBuf};

use bytes::Bytes;
use mithril_core::hashing::Blake3Hasher;
use mithril_core::hashing::HashFunction;
use thiserror::Error;
use tokio::io::AsyncWriteExt;

/// Errors that can occur in CAS operations.
#[derive(Error, Debug)]
pub enum CasError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Content not found
    #[error("Content not found: {0}")]
    NotFound(String),

    /// Invalid address format
    #[error("Invalid address format: {0}")]
    InvalidAddress(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for CAS operations.
pub type Result<T> = std::result::Result<T, CasError>;

/// Content-addressable storage.
///
/// Stores artifacts by their content hash (Blake3). This enables:
/// - Deduplication (same content = same address)
/// - Integrity verification (address is the hash)
/// - Fast lookups via filesystem
///
/// # Example
///
/// ```no_run
/// use mithril_cache::cas::ContentStore;
///
/// # async fn example() -> mithril_cache::cas::Result<()> {
/// let store = ContentStore::new("/tmp/cache")?;
///
/// // Store content
/// let address = store.put(b"hello world").await?;
///
/// // Retrieve by address
/// let content = store.get(&address).await?;
/// assert_eq!(content.unwrap(), b"hello world");
/// # Ok(())
/// # }
/// ```
pub struct ContentStore {
    root: PathBuf,
    hasher: Blake3Hasher,
}

impl ContentStore {
    /// Create a new content store at the given root directory.
    ///
    /// Creates the directory if it doesn't exist.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            hasher: Blake3Hasher::new(),
        })
    }

    /// Get the root directory of this store.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Store content and return its address (Blake3 hash).
    ///
    /// If content with this hash already exists, this is a no-op
    /// (content-addressable = idempotent puts).
    pub async fn put(&self, content: &[u8]) -> Result<String> {
        let address = self.compute_address(content);
        let path = self.address_to_path(&address);

        // Skip if already exists (CAS is idempotent)
        if path.exists() {
            return Ok(address);
        }

        // Create parent directory
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Write atomically via temp file + rename
        let temp_path = path.with_extension("tmp");
        let mut file = tokio::fs::File::create(&temp_path).await?;
        file.write_all(content).await?;
        file.sync_all().await?;
        drop(file);

        tokio::fs::rename(&temp_path, &path).await?;

        Ok(address)
    }

    /// Store content from bytes and return its address.
    pub async fn put_bytes(&self, content: Bytes) -> Result<String> {
        self.put(&content).await
    }

    /// Retrieve content by its address.
    ///
    /// Returns `None` if the content doesn't exist.
    pub async fn get(&self, address: &str) -> Result<Option<Vec<u8>>> {
        let path = self.address_to_path(address);

        match tokio::fs::read(&path).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CasError::Io(e)),
        }
    }

    /// Retrieve content as Bytes.
    pub async fn get_bytes(&self, address: &str) -> Result<Option<Bytes>> {
        Ok(self.get(address).await?.map(Bytes::from))
    }

    /// Check if content with the given address exists.
    pub async fn exists(&self, address: &str) -> Result<bool> {
        let path = self.address_to_path(address);
        Ok(path.exists())
    }

    /// Delete content by address.
    ///
    /// Returns `true` if content was deleted, `false` if it didn't exist.
    pub async fn delete(&self, address: &str) -> Result<bool> {
        let path = self.address_to_path(address);

        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(CasError::Io(e)),
        }
    }

    /// Compute the address (hash) for given content.
    #[must_use]
    pub fn compute_address(&self, content: &[u8]) -> String {
        self.hasher.hash_hex(content)
    }

    /// Get the size of content at address, if it exists.
    pub async fn size(&self, address: &str) -> Result<Option<u64>> {
        let path = self.address_to_path(address);

        match tokio::fs::metadata(&path).await {
            Ok(meta) => Ok(Some(meta.len())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CasError::Io(e)),
        }
    }

    /// Convert an address to a filesystem path.
    ///
    /// Uses a two-level directory structure to avoid too many files
    /// in a single directory: `root/ab/cd/abcdef...`
    fn address_to_path(&self, address: &str) -> PathBuf {
        // Use first 2 chars for first level, next 2 for second level
        // This spreads files across 256 * 256 = 65536 directories
        if address.len() >= 4 {
            self.root
                .join(&address[0..2])
                .join(&address[2..4])
                .join(address)
        } else {
            // Fallback for short addresses (shouldn't happen with Blake3)
            self.root.join(address)
        }
    }

    /// List all addresses in the store.
    ///
    /// Note: This can be slow for large stores.
    pub async fn list_all(&self) -> Result<Vec<String>> {
        let mut addresses = Vec::new();
        self.list_recursive(&self.root, &mut addresses).await?;
        Ok(addresses)
    }

    /// Recursively list addresses.
    #[allow(clippy::only_used_in_recursion)]
    fn list_recursive<'a>(
        &'a self,
        dir: &'a Path,
        addresses: &'a mut Vec<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut entries = match tokio::fs::read_dir(dir).await {
                Ok(entries) => entries,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(CasError::Io(e)),
            };

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                let file_type = entry.file_type().await?;

                if file_type.is_dir() {
                    self.list_recursive(&path, addresses).await?;
                } else if file_type.is_file() {
                    // The filename is the address
                    if let Some(name) = path.file_name() {
                        let name_str = name.to_string_lossy();
                        // Skip temp files
                        if !name_str.ends_with(".tmp") {
                            addresses.push(name_str.into_owned());
                        }
                    }
                }
            }

            Ok(())
        })
    }

    /// Get total size of all content in the store.
    pub async fn total_size(&self) -> Result<u64> {
        self.total_size_recursive(&self.root).await
    }

    /// Recursively compute total size.
    fn total_size_recursive(
        &self,
        dir: &Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64>> + Send + '_>> {
        let dir = dir.to_path_buf();
        Box::pin(async move {
            let mut total = 0u64;

            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(entries) => entries,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
                Err(e) => return Err(CasError::Io(e)),
            };

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                let file_type = entry.file_type().await?;

                if file_type.is_dir() {
                    total += self.total_size_recursive(&path).await?;
                } else if file_type.is_file() {
                    let name = path.file_name().map(|n| n.to_string_lossy());
                    // Skip temp files
                    if !name.map_or(false, |n| n.ends_with(".tmp")) {
                        let meta = entry.metadata().await?;
                        total += meta.len();
                    }
                }
            }

            Ok(total)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_put_get_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let content = b"hello world";
        let address = store.put(content).await.unwrap();

        // Address should be deterministic
        let address2 = store.put(content).await.unwrap();
        assert_eq!(address, address2);

        // Retrieve content
        let retrieved = store.get(&address).await.unwrap();
        assert_eq!(retrieved.unwrap(), content);
    }

    #[tokio::test]
    async fn test_exists() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let content = b"test content";
        let address = store.put(content).await.unwrap();

        assert!(store.exists(&address).await.unwrap());
        assert!(!store.exists("nonexistent").await.unwrap());
    }

    #[tokio::test]
    async fn test_delete() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let content = b"to be deleted";
        let address = store.put(content).await.unwrap();

        assert!(store.exists(&address).await.unwrap());
        assert!(store.delete(&address).await.unwrap());
        assert!(!store.exists(&address).await.unwrap());

        // Deleting again returns false
        assert!(!store.delete(&address).await.unwrap());
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let result = store.get("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_size() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let content = b"hello world";
        let address = store.put(content).await.unwrap();

        let size = store.size(&address).await.unwrap();
        assert_eq!(size, Some(content.len() as u64));

        let nonexistent = store.size("nonexistent").await.unwrap();
        assert!(nonexistent.is_none());
    }

    #[tokio::test]
    async fn test_list_all() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let addr1 = store.put(b"content1").await.unwrap();
        let addr2 = store.put(b"content2").await.unwrap();
        let addr3 = store.put(b"content3").await.unwrap();

        let mut addresses = store.list_all().await.unwrap();
        addresses.sort();

        let mut expected = vec![addr1, addr2, addr3];
        expected.sort();

        assert_eq!(addresses, expected);
    }

    #[tokio::test]
    async fn test_total_size() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        store.put(b"aaaa").await.unwrap(); // 4 bytes
        store.put(b"bbbbbb").await.unwrap(); // 6 bytes
        store.put(b"cc").await.unwrap(); // 2 bytes

        let total = store.total_size().await.unwrap();
        assert_eq!(total, 12);
    }

    #[tokio::test]
    async fn test_content_addressing_dedup() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        // Same content should produce same address
        let addr1 = store.put(b"same content").await.unwrap();
        let addr2 = store.put(b"same content").await.unwrap();
        assert_eq!(addr1, addr2);

        // Different content should produce different address
        let addr3 = store.put(b"different content").await.unwrap();
        assert_ne!(addr1, addr3);
    }

    #[tokio::test]
    async fn test_compute_address() {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let address = store.compute_address(b"test");

        // Blake3 produces 64 hex characters (32 bytes)
        assert_eq!(address.len(), 64);

        // Should be deterministic
        let address2 = store.compute_address(b"test");
        assert_eq!(address, address2);
    }
}
