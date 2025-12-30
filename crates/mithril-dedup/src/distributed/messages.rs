//! Message types for distributed deduplication.
//!
//! These structures define the protocol between coordinator and workers.

use serde::{Deserialize, Serialize};

/// Unique identifier for a worker.
pub type WorkerId = u64;

/// Request to register a worker with the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    /// Worker hostname or identifier.
    pub hostname: String,
    /// Worker port if applicable.
    pub port: Option<u16>,
    /// Worker capabilities.
    pub capabilities: WorkerCapabilities,
}

/// Worker capabilities for task assignment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Maximum documents per batch.
    pub max_batch_size: usize,
    /// Available memory in bytes.
    pub available_memory: u64,
    /// Number of CPU cores.
    pub cpu_cores: usize,
}

/// Response to a register request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    /// Assigned worker ID.
    pub worker_id: WorkerId,
    /// LSH parameters to use.
    pub lsh_params: LshParams,
    /// Whether registration was successful.
    pub success: bool,
    /// Error message if registration failed.
    pub error: Option<String>,
}

/// LSH parameters for index compatibility.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LshParams {
    /// Number of bands.
    pub num_bands: usize,
    /// Rows per band.
    pub rows_per_band: usize,
    /// Number of permutations (signature length).
    pub num_permutations: usize,
    /// Similarity threshold.
    pub threshold: f64,
    /// Expected items per worker.
    pub expected_items: usize,
    /// False positive rate.
    pub fp_rate: f64,
}

impl Default for LshParams {
    fn default() -> Self {
        Self {
            num_bands: 20,
            rows_per_band: 6,
            num_permutations: 128,
            threshold: 0.85,
            expected_items: 1_000_000,
            fp_rate: 0.001,
        }
    }
}

/// A batch of MinHash signatures for submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureBatch {
    /// Worker ID that generated these signatures.
    pub worker_id: WorkerId,
    /// Batch sequence number.
    pub batch_id: u64,
    /// Document IDs.
    pub doc_ids: Vec<u64>,
    /// Serialized signatures (compact format).
    pub signatures: Vec<u8>,
    /// Whether this is the final batch.
    pub is_final: bool,
}

/// Result of deduplication for a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationResult {
    /// Batch ID this result corresponds to.
    pub batch_id: u64,
    /// Document IDs that are duplicates.
    pub duplicate_ids: Vec<u64>,
    /// Document IDs that are unique.
    pub unique_ids: Vec<u64>,
    /// Whether processing was successful.
    pub success: bool,
    /// Error message if processing failed.
    pub error: Option<String>,
}

/// Request to query if a signature exists in the global index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Worker making the query.
    pub worker_id: WorkerId,
    /// Serialized MinHash signature.
    pub signature: Vec<u8>,
    /// Optional document ID for tracking.
    pub doc_id: Option<u64>,
}

/// Response to a duplicate query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Whether a potential duplicate was found.
    pub is_duplicate: bool,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
}

/// Request to merge a worker's index into the global index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRequest {
    /// Worker ID submitting the index.
    pub worker_id: WorkerId,
    /// Serialized LshBloomIndex.
    pub index_data: Vec<u8>,
    /// Number of items in the index.
    pub items_count: usize,
    /// Whether this is compressed.
    pub compressed: bool,
}

/// Response to a merge request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResponse {
    /// Whether merge was successful.
    pub success: bool,
    /// Items in global index before merge.
    pub items_before: usize,
    /// Items added from this merge.
    pub items_added: usize,
    /// Items in global index after merge.
    pub items_after: usize,
    /// Error message if merge failed.
    pub error: Option<String>,
}

/// Statistics from a worker.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerStats {
    /// Worker ID.
    pub worker_id: WorkerId,
    /// Documents processed.
    pub documents_processed: usize,
    /// Duplicates found locally.
    pub local_duplicates: usize,
    /// Duplicates confirmed by coordinator.
    pub global_duplicates: usize,
    /// Processing time in seconds.
    pub processing_time_secs: f64,
    /// Current memory usage in bytes.
    pub memory_usage: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_request_serialization() {
        let req = RegisterRequest {
            hostname: "worker-1".into(),
            port: Some(8080),
            capabilities: WorkerCapabilities {
                max_batch_size: 10000,
                available_memory: 16 * 1024 * 1024 * 1024,
                cpu_cores: 8,
            },
        };

        let json = serde_json::to_string(&req).unwrap();
        let req2: RegisterRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(req.hostname, req2.hostname);
        assert_eq!(req.port, req2.port);
    }

    #[test]
    fn test_lsh_params_default() {
        let params = LshParams::default();
        assert_eq!(params.num_bands * params.rows_per_band, 120);
        assert!(params.threshold > 0.0 && params.threshold <= 1.0);
    }

    #[test]
    fn test_deduplication_result() {
        let result = DeduplicationResult {
            batch_id: 1,
            duplicate_ids: vec![1, 2, 3],
            unique_ids: vec![4, 5, 6, 7],
            success: true,
            error: None,
        };

        assert_eq!(result.duplicate_ids.len(), 3);
        assert_eq!(result.unique_ids.len(), 4);
    }
}
