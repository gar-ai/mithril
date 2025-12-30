//! Semantic deduplication using text embeddings.
//!
//! This module provides semantic (meaning-based) deduplication that can detect
//! paraphrases and semantically similar documents that MinHash might miss.
//!
//! # Example (with MockBackend)
//!
//! ```no_run
//! use mithril_dedup::semantic::{SemanticDeduplicator, SemanticConfig, MockBackend};
//!
//! let backend = MockBackend::new(384);
//! let config = SemanticConfig::default();
//! let mut dedup = SemanticDeduplicator::new(backend, config);
//!
//! let texts = vec![
//!     "The cat sat on the mat",
//!     "A feline was resting on the rug",  // semantically similar
//!     "Machine learning is transforming AI",
//! ];
//!
//! let result = dedup.deduplicate(&texts);
//! ```
//!
//! # Example (with CandleBackend - requires `candle` feature)
//!
//! ```ignore
//! use mithril_dedup::semantic::{SemanticDeduplicator, SemanticConfig, CandleBackend};
//!
//! // Load a real sentence-transformers model
//! let backend = CandleBackend::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();
//! let config = SemanticConfig::default();
//! let mut dedup = SemanticDeduplicator::new(backend, config);
//!
//! // Now it can detect paraphrases!
//! let texts = vec![
//!     "The cat sat on the mat",
//!     "A feline was resting on the rug",  // will be detected as similar
//! ];
//!
//! let result = dedup.deduplicate(&texts);
//! ```

mod backend;
#[cfg(feature = "candle")]
mod candle;
mod hnsw;
mod index;

pub use backend::{cosine_similarity, EmbeddingBackend, MockBackend};
#[cfg(feature = "candle")]
pub use candle::CandleBackend;
pub use hnsw::{HnswConfig, HnswIndex, HnswMatch};
pub use index::{SemanticIndex, SemanticMatch};

use crate::{DedupResult, DedupStats};
use std::collections::HashSet;
use thiserror::Error;

/// Errors that can occur during semantic deduplication.
#[derive(Debug, Error)]
pub enum SemanticError {
    /// Embedding computation failed.
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Index operation failed.
    #[error("Index error: {0}")]
    Index(String),

    /// Invalid configuration.
    #[error("Invalid config: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, SemanticError>;

/// Configuration for semantic deduplication.
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Cosine similarity threshold (0.0-1.0). Documents with similarity >= threshold are duplicates.
    pub threshold: f32,
    /// Batch size for embedding computation.
    pub batch_size: usize,
    /// Number of neighbors to search in ANN (higher = more accurate but slower).
    pub num_neighbors: usize,
    /// EF parameter for HNSW search (higher = more accurate but slower).
    pub ef_search: usize,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            threshold: 0.9,
            batch_size: 64,
            num_neighbors: 10,
            ef_search: 100,
        }
    }
}

impl SemanticConfig {
    /// Create a config with the specified similarity threshold.
    #[must_use]
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Create a fast config (less accurate but faster).
    #[must_use]
    pub fn fast() -> Self {
        Self {
            threshold: 0.9,
            batch_size: 128,
            num_neighbors: 5,
            ef_search: 50,
        }
    }

    /// Create a precise config (more accurate but slower).
    #[must_use]
    pub fn precise() -> Self {
        Self {
            threshold: 0.9,
            batch_size: 32,
            num_neighbors: 20,
            ef_search: 200,
        }
    }
}

/// Semantic deduplicator using text embeddings.
pub struct SemanticDeduplicator<B: EmbeddingBackend> {
    backend: B,
    config: SemanticConfig,
    index: SemanticIndex,
}

impl<B: EmbeddingBackend> SemanticDeduplicator<B> {
    /// Create a new semantic deduplicator.
    pub fn new(backend: B, config: SemanticConfig) -> Self {
        let dim = backend.embedding_dim();
        Self {
            backend,
            config,
            index: SemanticIndex::new(dim),
        }
    }

    /// Deduplicate a batch of texts.
    ///
    /// Returns a deduplication result with unique document indices.
    pub fn deduplicate(&mut self, texts: &[&str]) -> Result<DedupResult> {
        use std::collections::HashMap;

        if texts.is_empty() {
            return Ok(DedupResult {
                keep_indices: vec![],
                remove_indices: vec![],
                clusters: HashMap::new(),
                stats: DedupStats::default(),
            });
        }

        // Compute embeddings in batches
        let embeddings = self.compute_embeddings(texts)?;

        // Build index and find duplicates
        let mut keep_indices = Vec::new();
        let mut remove_indices = Vec::new();
        let mut duplicate_count = 0;
        let duplicate_set: HashSet<usize> = HashSet::new();

        for (i, embedding) in embeddings.iter().enumerate() {
            if duplicate_set.contains(&i) {
                continue;
            }

            // Search for similar documents already in the index
            let matches = self.index.search(embedding, self.config.num_neighbors);

            let mut is_duplicate = false;
            for m in &matches {
                if m.similarity >= self.config.threshold {
                    is_duplicate = true;
                    duplicate_count += 1;
                    remove_indices.push(i);
                    break;
                }
            }

            if !is_duplicate {
                keep_indices.push(i);
                self.index.add(i as u64, embedding.clone());
            }
        }

        let unique_count = keep_indices.len();
        Ok(DedupResult {
            keep_indices,
            remove_indices,
            clusters: HashMap::new(),
            stats: DedupStats {
                total_documents: texts.len(),
                unique_documents: unique_count,
                duplicate_count,
                duplicate_ratio: duplicate_count as f64 / texts.len().max(1) as f64,
                cluster_count: 0,
                candidate_pairs: 0,
                verified_pairs: 0,
                processing_time_secs: 0.0,
            },
        })
    }

    /// Find all semantically similar pairs above the threshold.
    pub fn find_similar_pairs(&mut self, texts: &[&str]) -> Result<Vec<(usize, usize, f32)>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let embeddings = self.compute_embeddings(texts)?;

        // Build index with all documents
        for (i, embedding) in embeddings.iter().enumerate() {
            self.index.add(i as u64, embedding.clone());
        }

        // Find pairs
        let mut pairs = Vec::new();
        for (i, embedding) in embeddings.iter().enumerate() {
            let matches = self.index.search(embedding, self.config.num_neighbors);

            for m in matches {
                let j = m.doc_id as usize;
                if j > i && m.similarity >= self.config.threshold {
                    pairs.push((i, j, m.similarity));
                }
            }
        }

        Ok(pairs)
    }

    /// Compute embeddings for a batch of texts.
    fn compute_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.config.batch_size) {
            let embeddings = self
                .backend
                .embed_batch(batch)
                .map_err(SemanticError::Embedding)?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Get the current index size.
    #[must_use]
    pub fn index_size(&self) -> usize {
        self.index.len()
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_dedup_with_mock() {
        let backend = MockBackend::new(384);
        let config = SemanticConfig::with_threshold(0.99);
        let mut dedup = SemanticDeduplicator::new(backend, config);

        let texts = vec![
            "Hello world",
            "Hello world", // exact duplicate
            "Something different",
        ];

        let result = dedup
            .deduplicate(&texts.iter().map(|s| *s).collect::<Vec<_>>())
            .unwrap();

        // With mock backend, same texts should have same embeddings
        assert_eq!(result.keep_indices.len(), 2);
        assert_eq!(result.stats.duplicate_count, 1);
    }

    #[test]
    fn test_find_similar_pairs() {
        let backend = MockBackend::new(384);
        let config = SemanticConfig::with_threshold(0.99);
        let mut dedup = SemanticDeduplicator::new(backend, config);

        let texts = vec!["A", "B", "A", "C"];

        let pairs = dedup
            .find_similar_pairs(&texts.iter().map(|s| *s).collect::<Vec<_>>())
            .unwrap();

        // "A" appears at index 0 and 2
        assert!(pairs.iter().any(|(i, j, _)| *i == 0 && *j == 2));
    }
}
