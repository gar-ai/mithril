//! Google Cloud Storage backend.
//!
//! Works with GCS and the fake-gcs-server emulator for local testing.
//!
//! # Example
//!
//! ```rust,no_run
//! use mithril_core::storage::GcsStorage;
//!
//! # async fn example() -> mithril_core::Result<()> {
//! // Connect to GCS (uses GOOGLE_APPLICATION_CREDENTIALS)
//! let storage = GcsStorage::new("my-bucket", "prefix/").await?;
//!
//! // Or connect to emulator (local testing)
//! // Set STORAGE_EMULATOR_HOST=http://localhost:4443
//! let storage = GcsStorage::from_emulator("my-bucket", "test/").await?;
//! # Ok(())
//! # }
//! ```

use super::StorageBackend;
use crate::error::{MithrilError, Result};
use bytes::Bytes;
use cloud_storage::object::ObjectList;
use cloud_storage::Client;
use futures::TryStreamExt;

/// Google Cloud Storage backend.
pub struct GcsStorage {
    client: Client,
    bucket: String,
    prefix: String,
}

impl GcsStorage {
    /// Create a new GCS storage backend.
    ///
    /// Uses GOOGLE_APPLICATION_CREDENTIALS for authentication.
    ///
    /// # Arguments
    /// * `bucket` - Name of the GCS bucket
    /// * `prefix` - Key prefix for all operations
    pub async fn new(bucket: &str, prefix: &str) -> Result<Self> {
        let client = Client::default();

        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
        })
    }

    /// Create from emulator environment.
    ///
    /// Reads STORAGE_EMULATOR_HOST environment variable.
    pub async fn from_emulator(bucket: &str, prefix: &str) -> Result<Self> {
        // cloud-storage crate automatically uses STORAGE_EMULATOR_HOST if set
        Self::new(bucket, prefix).await
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    fn strip_prefix(&self, key: &str) -> String {
        key.strip_prefix(&self.prefix).unwrap_or(key).to_string()
    }
}

impl StorageBackend for GcsStorage {
    async fn get(&self, key: &str) -> Result<Bytes> {
        let full_key = self.full_key(key);
        let data = self
            .client
            .object()
            .download(&self.bucket, &full_key)
            .await
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("No such object") {
                    MithrilError::NotFound(key.to_string())
                } else {
                    MithrilError::Storage(format!("GCS get failed: {}", e))
                }
            })?;

        Ok(Bytes::from(data))
    }

    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let full_key = self.full_key(key);
        self.client
            .object()
            .create(
                &self.bucket,
                data.to_vec(),
                &full_key,
                "application/octet-stream",
            )
            .await
            .map_err(|e| MithrilError::Storage(format!("GCS put failed: {}", e)))?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let full_key = self.full_key(key);
        self.client
            .object()
            .delete(&self.bucket, &full_key)
            .await
            .map_err(|e| MithrilError::Storage(format!("GCS delete failed: {}", e)))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let full_key = self.full_key(key);
        match self.client.object().read(&self.bucket, &full_key).await {
            Ok(_) => Ok(true),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("No such object") {
                    Ok(false)
                } else {
                    Err(MithrilError::Storage(format!("GCS head failed: {}", e)))
                }
            }
        }
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        let stream = self
            .client
            .object()
            .list(
                &self.bucket,
                cloud_storage::ListRequest {
                    prefix: Some(full_prefix),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| MithrilError::Storage(format!("GCS list failed: {}", e)))?;

        // Collect all ObjectLists from the stream
        let object_lists: Vec<ObjectList> = stream
            .try_collect()
            .await
            .map_err(|e| MithrilError::Storage(format!("GCS list stream failed: {}", e)))?;

        // Flatten all objects and strip prefixes
        let keys: Vec<String> = object_lists
            .into_iter()
            .flat_map(|list| list.items)
            .map(|obj| self.strip_prefix(&obj.name))
            .collect();

        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests require fake-gcs-server running locally
    // Run with: cargo test --features gcs -p mithril-core -- --ignored

    #[tokio::test]
    #[ignore]
    async fn test_gcs_storage_with_emulator() {
        // Requires fake-gcs-server running:
        // docker run -p 4443:4443 fsouza/fake-gcs-server -scheme http -port 4443

        std::env::set_var("STORAGE_EMULATOR_HOST", "http://localhost:4443");

        let storage = GcsStorage::from_emulator("test-bucket", "test/")
            .await
            .unwrap();

        let key = "hello.txt";
        let data = Bytes::from_static(b"Hello, GCS!");

        // Put
        storage.put(key, data.clone()).await.unwrap();

        // Exists
        assert!(storage.exists(key).await.unwrap());

        // Get
        let retrieved = storage.get(key).await.unwrap();
        assert_eq!(retrieved, data);

        // List
        let keys = storage.list("").await.unwrap();
        assert!(keys.contains(&key.to_string()));

        // Delete
        storage.delete(key).await.unwrap();
        assert!(!storage.exists(key).await.unwrap());
    }
}
