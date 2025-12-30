//! S3-compatible storage backend.
//!
//! Works with AWS S3, MinIO, and other S3-compatible services.
//!
//! # Example
//!
//! ```rust,no_run
//! use mithril_core::storage::S3Storage;
//!
//! # async fn example() -> mithril_core::Result<()> {
//! // Connect to AWS S3
//! let storage = S3Storage::from_env("my-bucket", "prefix/").await?;
//!
//! // Or connect to MinIO (local testing)
//! let storage = S3Storage::new(
//!     "my-bucket",
//!     "us-east-1",
//!     Some("http://localhost:9000"),
//!     "mithril/",
//! ).await?;
//! # Ok(())
//! # }
//! ```

use super::StorageBackend;
use crate::error::{MithrilError, Result};
use bytes::Bytes;
use s3::creds::Credentials;
use s3::{Bucket, Region};

/// S3-compatible storage backend.
pub struct S3Storage {
    bucket: Box<Bucket>,
    prefix: String,
}

impl S3Storage {
    /// Create a new S3 storage backend.
    ///
    /// # Arguments
    /// * `bucket_name` - Name of the S3 bucket
    /// * `region` - AWS region (e.g., "us-east-1")
    /// * `endpoint` - Optional custom endpoint URL (for MinIO, LocalStack, etc.)
    /// * `prefix` - Key prefix for all operations (e.g., "mithril/cache/")
    pub async fn new(
        bucket_name: &str,
        region: &str,
        endpoint: Option<&str>,
        prefix: &str,
    ) -> Result<Self> {
        let region = if let Some(endpoint) = endpoint {
            Region::Custom {
                region: region.to_string(),
                endpoint: endpoint.to_string(),
            }
        } else {
            region
                .parse()
                .map_err(|e| MithrilError::Config(format!("Invalid region: {}", e)))?
        };

        let credentials = Credentials::from_env()
            .map_err(|e| MithrilError::Config(format!("Failed to load credentials: {}", e)))?;

        let bucket = Bucket::new(bucket_name, region, credentials)
            .map_err(|e| MithrilError::Config(format!("Failed to create bucket: {}", e)))?;

        // For custom endpoints, disable virtual-hosted style (use path style)
        let bucket = if endpoint.is_some() {
            bucket.with_path_style()
        } else {
            bucket
        };

        Ok(Self {
            bucket,
            prefix: prefix.to_string(),
        })
    }

    /// Create from environment variables.
    ///
    /// Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and optionally
    /// AWS_ENDPOINT_URL for custom endpoints.
    pub async fn from_env(bucket_name: &str, prefix: &str) -> Result<Self> {
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let endpoint = std::env::var("AWS_ENDPOINT_URL").ok();

        Self::new(bucket_name, &region, endpoint.as_deref(), prefix).await
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    fn strip_prefix(&self, key: &str) -> String {
        key.strip_prefix(&self.prefix).unwrap_or(key).to_string()
    }
}

impl StorageBackend for S3Storage {
    async fn get(&self, key: &str) -> Result<Bytes> {
        let full_key = self.full_key(key);
        let response = self.bucket.get_object(&full_key).await.map_err(|e| {
            if e.to_string().contains("404") || e.to_string().contains("NoSuchKey") {
                MithrilError::NotFound(key.to_string())
            } else {
                MithrilError::Storage(format!("S3 get failed: {}", e))
            }
        })?;

        Ok(Bytes::from(response.to_vec()))
    }

    async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let full_key = self.full_key(key);
        self.bucket
            .put_object(&full_key, &data)
            .await
            .map_err(|e| MithrilError::Storage(format!("S3 put failed: {}", e)))?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let full_key = self.full_key(key);
        self.bucket
            .delete_object(&full_key)
            .await
            .map_err(|e| MithrilError::Storage(format!("S3 delete failed: {}", e)))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let full_key = self.full_key(key);
        match self.bucket.head_object(&full_key).await {
            Ok(_) => Ok(true),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("NoSuchKey") {
                    Ok(false)
                } else {
                    Err(MithrilError::Storage(format!("S3 head failed: {}", e)))
                }
            }
        }
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        let results = self
            .bucket
            .list(full_prefix, None)
            .await
            .map_err(|e| MithrilError::Storage(format!("S3 list failed: {}", e)))?;

        let keys: Vec<String> = results
            .into_iter()
            .flat_map(|r| r.contents)
            .map(|obj| self.strip_prefix(&obj.key))
            .collect();

        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests require MinIO running locally
    // Run with: cargo test --features s3 -p mithril-core -- --ignored

    #[tokio::test]
    #[ignore]
    async fn test_s3_storage_with_minio() {
        // Requires MinIO running:
        // docker run -p 9000:9000 -e MINIO_ROOT_USER=mithril -e MINIO_ROOT_PASSWORD=mithril123 minio/minio server /data

        std::env::set_var("AWS_ACCESS_KEY_ID", "mithril");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "mithril123");

        let storage = S3Storage::new(
            "test-bucket",
            "us-east-1",
            Some("http://localhost:9000"),
            "test/",
        )
        .await
        .unwrap();

        let key = "hello.txt";
        let data = Bytes::from_static(b"Hello, MinIO!");

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

    #[tokio::test]
    #[ignore]
    async fn test_s3_not_found() {
        std::env::set_var("AWS_ACCESS_KEY_ID", "mithril");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "mithril123");

        let storage = S3Storage::new(
            "test-bucket",
            "us-east-1",
            Some("http://localhost:9000"),
            "test/",
        )
        .await
        .unwrap();

        let result = storage.get("nonexistent-key").await;
        assert!(matches!(result, Err(MithrilError::NotFound(_))));
    }
}
