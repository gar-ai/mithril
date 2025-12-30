//! Local filesystem storage backend.

use super::StorageBackend;
use crate::error::{MithrilError, Result};
use bytes::Bytes;
use std::path::{Path, PathBuf};

/// Local filesystem storage backend.
pub struct LocalStorage {
    root: PathBuf,
}

impl LocalStorage {
    /// Create a new local storage backend.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn key_to_path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }
}

impl StorageBackend for LocalStorage {
    async fn get(&self, key: &str) -> Result<Bytes> {
        let path = self.key_to_path(key);
        let data = tokio::fs::read(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                MithrilError::NotFound(key.to_string())
            } else {
                MithrilError::Io(e)
            }
        })?;
        Ok(Bytes::from(data))
    }

    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let path = self.key_to_path(key);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, &data).await?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.key_to_path(key);
        tokio::fs::remove_file(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                MithrilError::NotFound(key.to_string())
            } else {
                MithrilError::Io(e)
            }
        })
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let path = self.key_to_path(key);
        Ok(path.exists())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        // Determine the directory to search:
        // - Empty prefix or prefix ending with "/" -> search that directory
        // - Otherwise -> search parent directory
        let search_dir = if prefix.is_empty() {
            self.root.clone()
        } else if prefix.ends_with('/') {
            self.key_to_path(prefix)
        } else {
            let prefix_path = self.key_to_path(prefix);
            prefix_path.parent().unwrap_or(&self.root).to_path_buf()
        };

        if !search_dir.exists() {
            return Ok(vec![]);
        }

        let mut keys = Vec::new();
        let mut entries = tokio::fs::read_dir(&search_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Ok(relative) = path.strip_prefix(&self.root) {
                let key = relative.to_string_lossy().to_string();
                if key.starts_with(prefix) {
                    keys.push(key);
                }
            }
        }

        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_local_storage_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let storage = LocalStorage::new(tmp.path()).unwrap();

        let key = "test/data.bin";
        let data = Bytes::from_static(b"hello world");

        storage.put(key, data.clone()).await.unwrap();
        assert!(storage.exists(key).await.unwrap());

        let retrieved = storage.get(key).await.unwrap();
        assert_eq!(retrieved, data);

        storage.delete(key).await.unwrap();
        assert!(!storage.exists(key).await.unwrap());
    }

    #[tokio::test]
    async fn test_local_storage_list() {
        let tmp = TempDir::new().unwrap();
        let storage = LocalStorage::new(tmp.path()).unwrap();

        storage
            .put("prefix/a.bin", Bytes::from_static(b"a"))
            .await
            .unwrap();
        storage
            .put("prefix/b.bin", Bytes::from_static(b"b"))
            .await
            .unwrap();
        storage
            .put("other/c.bin", Bytes::from_static(b"c"))
            .await
            .unwrap();

        let keys = storage.list("prefix/").await.unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_local_storage_not_found() {
        let tmp = TempDir::new().unwrap();
        let storage = LocalStorage::new(tmp.path()).unwrap();

        let result = storage.get("nonexistent").await;
        assert!(matches!(result, Err(MithrilError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_local_storage_list_empty_prefix() {
        let tmp = TempDir::new().unwrap();
        let storage = LocalStorage::new(tmp.path()).unwrap();

        storage
            .put("file1.bin", Bytes::from_static(b"1"))
            .await
            .unwrap();
        storage
            .put("file2.bin", Bytes::from_static(b"2"))
            .await
            .unwrap();

        // List all files with empty prefix
        let keys = storage.list("").await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"file1.bin".to_string()));
        assert!(keys.contains(&"file2.bin".to_string()));
    }
}
