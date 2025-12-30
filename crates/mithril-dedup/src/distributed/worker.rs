//! Worker for distributed deduplication.
//!
//! Workers process documents locally and periodically sync with the coordinator.

use super::messages::{
    LshParams, MergeRequest, RegisterRequest, SignatureBatch, WorkerCapabilities, WorkerId,
    WorkerStats,
};
use crate::lsh_bloom::LshBloomIndex;
use crate::minhash::{MinHashSignature, MinHasher};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Worker hostname.
    pub hostname: String,
    /// Worker port (if applicable).
    pub port: Option<u16>,
    /// Batch size for processing.
    pub batch_size: usize,
    /// Whether to compress data when sending to coordinator.
    pub compress_transfers: bool,
    /// Local buffer size before syncing.
    pub sync_threshold: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            hostname: gethostname(),
            port: None,
            batch_size: 10_000,
            compress_transfers: true,
            sync_threshold: 100_000,
        }
    }
}

impl WorkerConfig {
    /// Create config with custom hostname.
    pub fn with_hostname(hostname: impl Into<String>) -> Self {
        Self {
            hostname: hostname.into(),
            ..Default::default()
        }
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
}

/// Result of document processing.
#[derive(Debug, Clone, Default)]
pub struct WorkerResult {
    /// Documents processed.
    pub documents_processed: usize,
    /// Duplicates found locally.
    pub local_duplicates: usize,
    /// Unique documents.
    pub unique_documents: usize,
    /// Processing time in seconds.
    pub processing_time_secs: f64,
}

/// Worker for distributed deduplication.
///
/// Processes documents locally and maintains a local index that can be
/// synced with the coordinator.
pub struct Worker {
    /// Configuration.
    config: WorkerConfig,
    /// Worker ID (assigned by coordinator).
    worker_id: Option<WorkerId>,
    /// LSH parameters (from coordinator).
    lsh_params: LshParams,
    /// Local LSH Bloom index.
    local_index: LshBloomIndex,
    /// MinHash hasher.
    hasher: MinHasher,
    /// Stats.
    documents_processed: usize,
    local_duplicates: usize,
    start_time: Instant,
}

impl Worker {
    /// Create a new worker with default LSH parameters.
    pub fn new(config: WorkerConfig) -> Self {
        let lsh_params = LshParams::default();
        Self::with_params(config, lsh_params)
    }

    /// Create a new worker with specific LSH parameters.
    pub fn with_params(config: WorkerConfig, lsh_params: LshParams) -> Self {
        let local_index = LshBloomIndex::new(
            lsh_params.num_bands,
            lsh_params.rows_per_band,
            lsh_params.expected_items,
            lsh_params.fp_rate,
        );
        let hasher = MinHasher::new(lsh_params.num_permutations);

        Self {
            config,
            worker_id: None,
            lsh_params,
            local_index,
            hasher,
            documents_processed: 0,
            local_duplicates: 0,
            start_time: Instant::now(),
        }
    }

    /// Create a registration request.
    pub fn registration_request(&self) -> RegisterRequest {
        RegisterRequest {
            hostname: self.config.hostname.clone(),
            port: self.config.port,
            capabilities: WorkerCapabilities {
                max_batch_size: self.config.batch_size,
                available_memory: get_available_memory(),
                cpu_cores: num_cpus(),
            },
        }
    }

    /// Set worker ID (after registration with coordinator).
    pub fn set_worker_id(&mut self, id: WorkerId) {
        self.worker_id = Some(id);
    }

    /// Update LSH parameters (from coordinator).
    pub fn update_params(&mut self, params: LshParams) {
        if params.num_bands != self.lsh_params.num_bands
            || params.rows_per_band != self.lsh_params.rows_per_band
        {
            // Recreate index with new params
            self.local_index = LshBloomIndex::new(
                params.num_bands,
                params.rows_per_band,
                params.expected_items,
                params.fp_rate,
            );
            self.hasher = MinHasher::new(params.num_permutations);
        }
        self.lsh_params = params;
    }

    /// Get worker ID.
    pub fn worker_id(&self) -> Option<WorkerId> {
        self.worker_id
    }

    /// Process a single document.
    ///
    /// Returns true if the document is a duplicate (already in local index).
    pub fn process_document(&mut self, text: &str) -> bool {
        let signature = self.hasher.signature_from_text(text);
        let is_duplicate = self.local_index.check_and_insert(&signature);

        self.documents_processed += 1;
        if is_duplicate {
            self.local_duplicates += 1;
        }

        is_duplicate
    }

    /// Process multiple documents.
    ///
    /// Returns result with duplicate/unique counts.
    pub fn process_documents(&mut self, texts: &[&str]) -> WorkerResult {
        let start = Instant::now();

        let mut local_dups = 0;
        for text in texts {
            if self.process_document(text) {
                local_dups += 1;
            }
        }

        WorkerResult {
            documents_processed: texts.len(),
            local_duplicates: local_dups,
            unique_documents: texts.len() - local_dups,
            processing_time_secs: start.elapsed().as_secs_f64(),
        }
    }

    /// Process documents with IDs.
    ///
    /// Returns (duplicate_ids, unique_ids).
    pub fn process_documents_with_ids(&mut self, docs: &[(u64, &str)]) -> (Vec<u64>, Vec<u64>) {
        let mut duplicate_ids = Vec::new();
        let mut unique_ids = Vec::new();

        for (id, text) in docs {
            if self.process_document(text) {
                duplicate_ids.push(*id);
            } else {
                unique_ids.push(*id);
            }
        }

        (duplicate_ids, unique_ids)
    }

    /// Create a batch of signatures for submission to coordinator.
    pub fn create_signature_batch(&self, docs: &[(u64, &str)], batch_id: u64) -> SignatureBatch {
        let doc_ids: Vec<u64> = docs.iter().map(|(id, _)| *id).collect();
        let signatures: Vec<MinHashSignature> = docs
            .iter()
            .map(|(_, text)| self.hasher.signature_from_text(text))
            .collect();

        let signatures_bytes = bincode::serialize(&signatures).unwrap_or_default();

        SignatureBatch {
            worker_id: self.worker_id.unwrap_or(0),
            batch_id,
            doc_ids,
            signatures: signatures_bytes,
            is_final: false,
        }
    }

    /// Get the local index data for merging with coordinator.
    pub fn index_data(&self) -> Result<Vec<u8>, WorkerError> {
        let data = self
            .local_index
            .to_bytes()
            .map_err(|e| WorkerError::SerializationError(e.to_string()))?;

        if self.config.compress_transfers {
            zstd::encode_all(data.as_slice(), 3)
                .map_err(|e| WorkerError::CompressionError(e.to_string()))
        } else {
            Ok(data)
        }
    }

    /// Create a merge request for the coordinator.
    pub fn merge_request(&self) -> Result<MergeRequest, WorkerError> {
        let index_data = self.index_data()?;

        Ok(MergeRequest {
            worker_id: self.worker_id.unwrap_or(0),
            index_data,
            items_count: self.local_index.items_count(),
            compressed: self.config.compress_transfers,
        })
    }

    /// Get worker statistics.
    pub fn stats(&self) -> WorkerStats {
        WorkerStats {
            worker_id: self.worker_id.unwrap_or(0),
            documents_processed: self.documents_processed,
            local_duplicates: self.local_duplicates,
            global_duplicates: 0, // Set by caller after coordinator response
            processing_time_secs: self.start_time.elapsed().as_secs_f64(),
            memory_usage: self.local_index.memory_usage() as u64,
        }
    }

    /// Check if sync threshold is reached.
    pub fn should_sync(&self) -> bool {
        self.documents_processed >= self.config.sync_threshold
    }

    /// Reset the local index after syncing.
    pub fn reset(&mut self) {
        self.local_index.clear();
        self.documents_processed = 0;
        self.local_duplicates = 0;
        self.start_time = Instant::now();
    }

    /// Get the number of documents processed.
    pub fn documents_processed(&self) -> usize {
        self.documents_processed
    }

    /// Get the number of local duplicates found.
    pub fn local_duplicates(&self) -> usize {
        self.local_duplicates
    }

    /// Get reference to the local index.
    pub fn local_index(&self) -> &LshBloomIndex {
        &self.local_index
    }
}

/// Errors that can occur in a worker.
#[derive(Debug)]
pub enum WorkerError {
    /// Serialization failed.
    SerializationError(String),
    /// Compression failed.
    CompressionError(String),
    /// Not registered with coordinator.
    NotRegistered,
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            Self::NotRegistered => write!(f, "Worker not registered with coordinator"),
        }
    }
}

impl std::error::Error for WorkerError {}

/// Get hostname.
fn gethostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".into())
}

/// Get available memory (approximation).
fn get_available_memory() -> u64 {
    // Return 0 as a placeholder - actual implementation would use platform-specific APIs
    0
}

/// Get number of CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_creation() {
        let config = WorkerConfig::default();
        let worker = Worker::new(config);

        assert_eq!(worker.documents_processed(), 0);
        assert_eq!(worker.local_duplicates(), 0);
        assert!(worker.worker_id().is_none());
    }

    #[test]
    fn test_process_document() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);

        // First document is not a duplicate
        assert!(!worker.process_document("Hello world"));

        // Same document is a duplicate
        assert!(worker.process_document("Hello world"));

        // Different document is not a duplicate
        assert!(!worker.process_document("Different content"));

        assert_eq!(worker.documents_processed(), 3);
        assert_eq!(worker.local_duplicates(), 1);
    }

    #[test]
    fn test_process_documents() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);

        let texts = vec!["doc one", "doc two", "doc one", "doc three"];
        let result = worker.process_documents(&texts);

        assert_eq!(result.documents_processed, 4);
        assert_eq!(result.local_duplicates, 1);
        assert_eq!(result.unique_documents, 3);
    }

    #[test]
    fn test_process_documents_with_ids() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);

        let docs = vec![
            (1, "document one"),
            (2, "document two"),
            (3, "document one"), // duplicate
            (4, "document three"),
        ];

        let (dups, uniques) = worker.process_documents_with_ids(&docs);

        assert_eq!(dups, vec![3]);
        assert_eq!(uniques, vec![1, 2, 4]);
    }

    #[test]
    fn test_worker_registration_request() {
        let config = WorkerConfig::with_hostname("test-worker");
        let worker = Worker::new(config);

        let request = worker.registration_request();
        assert_eq!(request.hostname, "test-worker");
    }

    #[test]
    fn test_worker_set_id() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);

        assert!(worker.worker_id().is_none());

        worker.set_worker_id(42);
        assert_eq!(worker.worker_id(), Some(42));
    }

    #[test]
    fn test_worker_stats() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);
        worker.set_worker_id(1);

        worker.process_document("test");
        worker.process_document("test"); // duplicate

        let stats = worker.stats();
        assert_eq!(stats.worker_id, 1);
        assert_eq!(stats.documents_processed, 2);
        assert_eq!(stats.local_duplicates, 1);
    }

    #[test]
    fn test_worker_reset() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);

        worker.process_document("test");
        assert_eq!(worker.documents_processed(), 1);

        worker.reset();
        assert_eq!(worker.documents_processed(), 0);
        assert_eq!(worker.local_duplicates(), 0);
    }

    #[test]
    fn test_should_sync() {
        let mut config = WorkerConfig::default();
        config.sync_threshold = 5;
        let mut worker = Worker::new(config);

        for i in 0..4 {
            worker.process_document(&format!("doc {}", i));
        }
        assert!(!worker.should_sync());

        worker.process_document("doc 4");
        assert!(worker.should_sync());
    }

    #[test]
    fn test_create_signature_batch() {
        let config = WorkerConfig::default();
        let mut worker = Worker::new(config);
        worker.set_worker_id(1);

        let docs = vec![(1, "doc one"), (2, "doc two")];
        let batch = worker.create_signature_batch(&docs, 0);

        assert_eq!(batch.worker_id, 1);
        assert_eq!(batch.batch_id, 0);
        assert_eq!(batch.doc_ids, vec![1, 2]);
        assert!(!batch.signatures.is_empty());
    }

    #[test]
    fn test_merge_request() {
        let mut config = WorkerConfig::default();
        config.compress_transfers = false; // Easier to test
        let mut worker = Worker::new(config);
        worker.set_worker_id(1);

        worker.process_document("test document");

        let request = worker.merge_request().unwrap();
        assert_eq!(request.worker_id, 1);
        assert_eq!(request.items_count, 1);
        assert!(!request.compressed);
    }
}
