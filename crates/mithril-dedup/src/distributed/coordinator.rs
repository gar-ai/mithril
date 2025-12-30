//! Coordinator for distributed deduplication.
//!
//! The coordinator maintains the global LSH Bloom index and coordinates
//! deduplication across multiple workers.

use super::messages::{
    DeduplicationResult, LshParams, MergeRequest, MergeResponse, QueryRequest, QueryResponse,
    RegisterRequest, RegisterResponse, SignatureBatch, WorkerId,
};
use crate::lsh_bloom::LshBloomIndex;
use crate::minhash::MinHashSignature;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Configuration for the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// LSH parameters for the global index.
    pub lsh_params: LshParams,
    /// Path to persist the global index.
    pub index_path: Option<String>,
    /// Maximum workers allowed.
    pub max_workers: usize,
    /// Whether to compress index data in transit.
    pub compress_transfers: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            lsh_params: LshParams::default(),
            index_path: None,
            max_workers: 100,
            compress_transfers: true,
        }
    }
}

impl CoordinatorConfig {
    /// Create config with custom LSH parameters.
    pub fn with_lsh_params(lsh_params: LshParams) -> Self {
        Self {
            lsh_params,
            ..Default::default()
        }
    }

    /// Set index persistence path.
    pub fn with_index_path(mut self, path: impl Into<String>) -> Self {
        self.index_path = Some(path.into());
        self
    }
}

/// Information about a registered worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID.
    pub id: WorkerId,
    /// Worker hostname.
    pub hostname: String,
    /// Worker port.
    pub port: Option<u16>,
    /// Registration time.
    pub registered_at: u64,
    /// Last activity time.
    pub last_seen: u64,
    /// Documents processed by this worker.
    pub documents_processed: usize,
    /// Whether worker is active.
    pub is_active: bool,
}

/// Global statistics from the coordinator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinatorStats {
    /// Total documents indexed globally.
    pub total_documents: usize,
    /// Number of registered workers.
    pub worker_count: usize,
    /// Number of active workers.
    pub active_workers: usize,
    /// Number of merge operations performed.
    pub merges_performed: usize,
    /// Number of queries handled.
    pub queries_handled: usize,
    /// Global index memory usage in bytes.
    pub index_memory_bytes: usize,
    /// Coordinator uptime in seconds.
    pub uptime_secs: f64,
}

/// Coordinator for distributed deduplication.
///
/// Maintains the global LSH Bloom index and coordinates workers.
pub struct Coordinator {
    /// Configuration.
    config: CoordinatorConfig,
    /// Global LSH Bloom index.
    global_index: Arc<RwLock<LshBloomIndex>>,
    /// Registered workers.
    workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
    /// Next worker ID.
    next_worker_id: AtomicU64,
    /// Stats counters.
    merges_performed: AtomicU64,
    queries_handled: AtomicU64,
    /// Start time.
    start_time: Instant,
}

impl Coordinator {
    /// Create a new coordinator with the given configuration.
    pub fn new(config: CoordinatorConfig) -> Self {
        let global_index = LshBloomIndex::new(
            config.lsh_params.num_bands,
            config.lsh_params.rows_per_band,
            config.lsh_params.expected_items,
            config.lsh_params.fp_rate,
        );

        Self {
            config,
            global_index: Arc::new(RwLock::new(global_index)),
            workers: Arc::new(RwLock::new(HashMap::new())),
            next_worker_id: AtomicU64::new(1),
            merges_performed: AtomicU64::new(0),
            queries_handled: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Load coordinator state from disk.
    pub fn load<P: AsRef<Path>>(
        path: P,
        config: CoordinatorConfig,
    ) -> Result<Self, CoordinatorError> {
        let data = std::fs::read(path.as_ref())?;
        let global_index = LshBloomIndex::from_bytes(&data)
            .map_err(|e| CoordinatorError::IndexError(e.to_string()))?;

        Ok(Self {
            config,
            global_index: Arc::new(RwLock::new(global_index)),
            workers: Arc::new(RwLock::new(HashMap::new())),
            next_worker_id: AtomicU64::new(1),
            merges_performed: AtomicU64::new(0),
            queries_handled: AtomicU64::new(0),
            start_time: Instant::now(),
        })
    }

    /// Save coordinator state to disk.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), CoordinatorError> {
        let index = self
            .global_index
            .read()
            .map_err(|_| CoordinatorError::LockError)?;
        let data = index
            .to_bytes()
            .map_err(|e| CoordinatorError::IndexError(e.to_string()))?;
        std::fs::write(path.as_ref(), data)?;
        Ok(())
    }

    /// Register a new worker.
    pub fn register(&self, request: RegisterRequest) -> RegisterResponse {
        let worker_id = self.next_worker_id.fetch_add(1, Ordering::SeqCst);

        let worker_info = WorkerInfo {
            id: worker_id,
            hostname: request.hostname.clone(),
            port: request.port,
            registered_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            documents_processed: 0,
            is_active: true,
        };

        let mut workers = match self.workers.write() {
            Ok(w) => w,
            Err(_) => {
                return RegisterResponse {
                    worker_id: 0,
                    lsh_params: self.config.lsh_params,
                    success: false,
                    error: Some("Failed to acquire lock".into()),
                }
            }
        };

        if workers.len() >= self.config.max_workers {
            return RegisterResponse {
                worker_id: 0,
                lsh_params: self.config.lsh_params,
                success: false,
                error: Some("Maximum workers reached".into()),
            };
        }

        workers.insert(worker_id, worker_info);

        RegisterResponse {
            worker_id,
            lsh_params: self.config.lsh_params,
            success: true,
            error: None,
        }
    }

    /// Unregister a worker.
    pub fn unregister(&self, worker_id: WorkerId) -> bool {
        if let Ok(mut workers) = self.workers.write() {
            if let Some(worker) = workers.get_mut(&worker_id) {
                worker.is_active = false;
                return true;
            }
        }
        false
    }

    /// Query if a signature exists in the global index.
    pub fn query(&self, request: QueryRequest) -> QueryResponse {
        self.queries_handled.fetch_add(1, Ordering::Relaxed);

        // Deserialize signature
        let signature = match bincode::deserialize::<MinHashSignature>(&request.signature) {
            Ok(sig) => sig,
            Err(_) => {
                return QueryResponse {
                    is_duplicate: false,
                    confidence: 0.0,
                };
            }
        };

        let index = match self.global_index.read() {
            Ok(i) => i,
            Err(_) => {
                return QueryResponse {
                    is_duplicate: false,
                    confidence: 0.0,
                };
            }
        };

        let is_duplicate = index.query(&signature);

        QueryResponse {
            is_duplicate,
            confidence: if is_duplicate { 0.85 } else { 0.0 }, // Based on threshold
        }
    }

    /// Query and insert a signature atomically.
    pub fn query_and_insert(&self, signature_data: &[u8]) -> Result<bool, CoordinatorError> {
        let signature: MinHashSignature = bincode::deserialize(signature_data)
            .map_err(|e| CoordinatorError::SerializationError(e.to_string()))?;

        let mut index = self
            .global_index
            .write()
            .map_err(|_| CoordinatorError::LockError)?;

        Ok(index.check_and_insert(&signature))
    }

    /// Merge a worker's index into the global index.
    pub fn merge(&self, request: MergeRequest) -> MergeResponse {
        self.merges_performed.fetch_add(1, Ordering::Relaxed);

        // Update worker last_seen
        if let Ok(mut workers) = self.workers.write() {
            if let Some(worker) = workers.get_mut(&request.worker_id) {
                worker.last_seen = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                worker.documents_processed += request.items_count;
            }
        }

        // Decompress if needed
        let index_data = if request.compressed {
            match zstd::decode_all(request.index_data.as_slice()) {
                Ok(data) => data,
                Err(e) => {
                    return MergeResponse {
                        success: false,
                        items_before: 0,
                        items_added: 0,
                        items_after: 0,
                        error: Some(format!("Decompression failed: {}", e)),
                    };
                }
            }
        } else {
            request.index_data
        };

        // Deserialize worker index
        let worker_index = match LshBloomIndex::from_bytes(&index_data) {
            Ok(idx) => idx,
            Err(e) => {
                return MergeResponse {
                    success: false,
                    items_before: 0,
                    items_added: 0,
                    items_after: 0,
                    error: Some(format!("Deserialization failed: {}", e)),
                };
            }
        };

        // Merge into global index
        let mut global = match self.global_index.write() {
            Ok(g) => g,
            Err(_) => {
                return MergeResponse {
                    success: false,
                    items_before: 0,
                    items_added: 0,
                    items_after: 0,
                    error: Some("Failed to acquire lock".into()),
                };
            }
        };

        match global.merge(&worker_index) {
            Ok(stats) => MergeResponse {
                success: true,
                items_before: stats.items_before,
                items_added: stats.items_added,
                items_after: stats.items_after,
                error: None,
            },
            Err(e) => MergeResponse {
                success: false,
                items_before: 0,
                items_added: 0,
                items_after: 0,
                error: Some(e.to_string()),
            },
        }
    }

    /// Process a batch of signatures.
    pub fn process_batch(&self, batch: SignatureBatch) -> DeduplicationResult {
        let mut duplicate_ids = Vec::new();
        let mut unique_ids = Vec::new();

        // Deserialize all signatures
        let signatures: Vec<MinHashSignature> = match bincode::deserialize(&batch.signatures) {
            Ok(sigs) => sigs,
            Err(e) => {
                return DeduplicationResult {
                    batch_id: batch.batch_id,
                    duplicate_ids: vec![],
                    unique_ids: vec![],
                    success: false,
                    error: Some(format!("Failed to deserialize signatures: {}", e)),
                };
            }
        };

        let mut index = match self.global_index.write() {
            Ok(i) => i,
            Err(_) => {
                return DeduplicationResult {
                    batch_id: batch.batch_id,
                    duplicate_ids: vec![],
                    unique_ids: vec![],
                    success: false,
                    error: Some("Failed to acquire lock".into()),
                };
            }
        };

        for (doc_id, signature) in batch.doc_ids.iter().zip(signatures.iter()) {
            if index.check_and_insert(signature) {
                duplicate_ids.push(*doc_id);
            } else {
                unique_ids.push(*doc_id);
            }
        }

        DeduplicationResult {
            batch_id: batch.batch_id,
            duplicate_ids,
            unique_ids,
            success: true,
            error: None,
        }
    }

    /// Get coordinator statistics.
    pub fn stats(&self) -> CoordinatorStats {
        let index = self.global_index.read().ok();
        let workers = self.workers.read().ok();

        let (total_documents, index_memory_bytes) = index
            .map(|i| (i.items_count(), i.memory_usage()))
            .unwrap_or((0, 0));

        let (worker_count, active_workers) = workers
            .map(|w| {
                let total = w.len();
                let active = w.values().filter(|info| info.is_active).count();
                (total, active)
            })
            .unwrap_or((0, 0));

        CoordinatorStats {
            total_documents,
            worker_count,
            active_workers,
            merges_performed: self.merges_performed.load(Ordering::Relaxed) as usize,
            queries_handled: self.queries_handled.load(Ordering::Relaxed) as usize,
            index_memory_bytes,
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }

    /// Get list of registered workers.
    pub fn workers(&self) -> Vec<WorkerInfo> {
        self.workers
            .read()
            .map(|w| w.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the global index for direct access (e.g., for persistence).
    pub fn global_index(&self) -> &Arc<RwLock<LshBloomIndex>> {
        &self.global_index
    }

    /// Clear the global index.
    pub fn clear(&self) -> Result<(), CoordinatorError> {
        let mut index = self
            .global_index
            .write()
            .map_err(|_| CoordinatorError::LockError)?;
        index.clear();
        Ok(())
    }
}

/// Errors that can occur in the coordinator.
#[derive(Debug)]
pub enum CoordinatorError {
    /// IO error.
    Io(std::io::Error),
    /// Index operation error.
    IndexError(String),
    /// Serialization error.
    SerializationError(String),
    /// Lock acquisition failed.
    LockError,
}

impl std::fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::IndexError(msg) => write!(f, "Index error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::LockError => write!(f, "Lock acquisition failed"),
        }
    }
}

impl std::error::Error for CoordinatorError {}

impl From<std::io::Error> for CoordinatorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config);

        let stats = coordinator.stats();
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.worker_count, 0);
    }

    #[test]
    fn test_worker_registration() {
        let coordinator = Coordinator::new(CoordinatorConfig::default());

        let request = RegisterRequest {
            hostname: "worker-1".into(),
            port: Some(8080),
            capabilities: Default::default(),
        };

        let response = coordinator.register(request);
        assert!(response.success);
        assert_eq!(response.worker_id, 1);

        let stats = coordinator.stats();
        assert_eq!(stats.worker_count, 1);
        assert_eq!(stats.active_workers, 1);
    }

    #[test]
    fn test_worker_unregistration() {
        let coordinator = Coordinator::new(CoordinatorConfig::default());

        let request = RegisterRequest {
            hostname: "worker-1".into(),
            port: None,
            capabilities: Default::default(),
        };

        let response = coordinator.register(request);
        let worker_id = response.worker_id;

        assert!(coordinator.unregister(worker_id));

        let stats = coordinator.stats();
        assert_eq!(stats.worker_count, 1);
        assert_eq!(stats.active_workers, 0);
    }

    #[test]
    fn test_max_workers() {
        let mut config = CoordinatorConfig::default();
        config.max_workers = 2;
        let coordinator = Coordinator::new(config);

        // Register max workers
        for i in 0..2 {
            let request = RegisterRequest {
                hostname: format!("worker-{}", i),
                port: None,
                capabilities: Default::default(),
            };
            let response = coordinator.register(request);
            assert!(response.success);
        }

        // Third registration should fail
        let request = RegisterRequest {
            hostname: "worker-3".into(),
            port: None,
            capabilities: Default::default(),
        };
        let response = coordinator.register(request);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_coordinator_clear() {
        let coordinator = Coordinator::new(CoordinatorConfig::default());
        coordinator.clear().unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.total_documents, 0);
    }
}
