//! Hybrid deduplication combining MinHash and semantic similarity.
//!
//! This module provides a two-stage deduplication approach:
//! 1. **Stage 1 (MinHash)**: Fast syntactic filtering to find candidate duplicates
//! 2. **Stage 2 (Semantic)**: Accurate verification using embeddings
//!
//! This combines the speed of MinHash (100K+ docs/sec) with the accuracy of
//! semantic similarity (catches paraphrases).
//!
//! # Example
//!
//! ```no_run
//! use mithril_dedup::hybrid::{HybridDeduplicator, HybridConfig};
//! use mithril_dedup::semantic::MockBackend;
//!
//! let backend = MockBackend::new(384);
//! let config = HybridConfig::default();
//! let mut dedup = HybridDeduplicator::new(backend, config);
//!
//! let texts = vec![
//!     "The quick brown fox",
//!     "A fast brown fox",  // semantic duplicate
//!     "Something else",
//! ];
//!
//! let result = dedup.deduplicate(&texts);
//! ```

use crate::minhash::{MinHashSignature, MinHasher};
use crate::semantic::{cosine_similarity, EmbeddingBackend, SemanticIndex};
use crate::{DedupResult, DedupStats};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during hybrid deduplication.
#[derive(Debug, Error)]
pub enum HybridError {
    /// MinHash error.
    #[error("MinHash error: {0}")]
    MinHash(String),

    /// Semantic error.
    #[error("Semantic error: {0}")]
    Semantic(String),
}

pub type Result<T> = std::result::Result<T, HybridError>;

/// Configuration for hybrid deduplication.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// MinHash similarity threshold for initial filtering (lower = more candidates).
    pub minhash_threshold: f64,
    /// Semantic similarity threshold for final verification (higher = stricter).
    pub semantic_threshold: f32,
    /// Number of MinHash permutations.
    pub num_permutations: usize,
    /// N-gram size for shingling.
    pub ngram_size: usize,
    /// Batch size for embedding computation.
    pub embed_batch_size: usize,
    /// Whether to use semantic verification for MinHash candidates.
    pub verify_with_semantic: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            minhash_threshold: 0.7,  // Lower to catch more candidates
            semantic_threshold: 0.9, // Higher for accurate verification
            num_permutations: 128,
            ngram_size: 5,
            embed_batch_size: 64,
            verify_with_semantic: true,
        }
    }
}

impl HybridConfig {
    /// Create a fast configuration (more MinHash filtering, less semantic).
    #[must_use]
    pub fn fast() -> Self {
        Self {
            minhash_threshold: 0.8,
            semantic_threshold: 0.85,
            num_permutations: 64,
            ngram_size: 5,
            embed_batch_size: 128,
            verify_with_semantic: true,
        }
    }

    /// Create a precise configuration (more semantic verification).
    #[must_use]
    pub fn precise() -> Self {
        Self {
            minhash_threshold: 0.5, // Very low to catch all potential duplicates
            semantic_threshold: 0.95,
            num_permutations: 256,
            ngram_size: 3,
            embed_batch_size: 32,
            verify_with_semantic: true,
        }
    }
}

/// Statistics from hybrid deduplication.
#[derive(Debug, Clone, Default)]
pub struct HybridStats {
    /// Total documents processed.
    pub total_documents: usize,
    /// Documents remaining after deduplication.
    pub unique_documents: usize,
    /// Duplicates found.
    pub duplicate_count: usize,
    /// Candidates from MinHash stage.
    pub minhash_candidates: usize,
    /// Candidates verified by semantic stage.
    pub semantic_verified: usize,
}

/// Hybrid deduplicator combining MinHash and semantic similarity.
pub struct HybridDeduplicator<B: EmbeddingBackend> {
    backend: B,
    config: HybridConfig,
    minhash: MinHasher,
    /// Stored signatures for comparison.
    signatures: Vec<MinHashSignature>,
    /// Stored embeddings for semantic comparison.
    semantic_index: SemanticIndex,
}

impl<B: EmbeddingBackend> HybridDeduplicator<B> {
    /// Create a new hybrid deduplicator.
    pub fn new(backend: B, config: HybridConfig) -> Self {
        let minhash = MinHasher::new(config.num_permutations).with_ngram_size(config.ngram_size);
        let semantic_index = SemanticIndex::new(backend.embedding_dim());

        Self {
            backend,
            config,
            minhash,
            signatures: Vec::new(),
            semantic_index,
        }
    }

    /// Deduplicate a batch of texts.
    pub fn deduplicate(&mut self, texts: &[&str]) -> Result<DedupResult> {
        if texts.is_empty() {
            return Ok(DedupResult {
                keep_indices: vec![],
                remove_indices: vec![],
                clusters: HashMap::new(),
                stats: DedupStats::default(),
            });
        }

        // Compute MinHash signatures for all texts
        let new_signatures: Vec<MinHashSignature> = texts
            .iter()
            .map(|text| self.minhash.signature_from_text(text))
            .collect();

        let mut keep_indices = Vec::new();
        let mut remove_indices = Vec::new();
        let mut minhash_candidates = 0;
        let mut semantic_verified = 0;

        // Process each document
        for (i, sig) in new_signatures.iter().enumerate() {
            // Stage 1: Check MinHash similarity against existing documents
            let mut has_minhash_match = false;
            let mut matched_idx: Option<usize> = None;

            for (j, existing_sig) in self.signatures.iter().enumerate() {
                let sim = MinHasher::similarity(sig, existing_sig);
                if sim >= self.config.minhash_threshold {
                    has_minhash_match = true;
                    matched_idx = Some(j);
                    minhash_candidates += 1;
                    break;
                }
            }

            let mut is_duplicate = false;

            // Stage 2: Semantic verification (if enabled and MinHash matched)
            if self.config.verify_with_semantic {
                // Compute embedding for this document
                let embedding = self
                    .backend
                    .embed_batch(&[texts[i]])
                    .map_err(HybridError::Semantic)?
                    .pop()
                    .ok_or_else(|| HybridError::Semantic("No embedding returned".to_string()))?;

                if has_minhash_match {
                    // Verify with semantic similarity
                    if let Some(idx) = matched_idx {
                        if let Some(candidate_vec) = self.semantic_index.get(idx as u64) {
                            let semantic_sim = cosine_similarity(&embedding, candidate_vec);
                            if semantic_sim >= self.config.semantic_threshold {
                                is_duplicate = true;
                                semantic_verified += 1;
                            }
                        }
                    }
                }

                // Also check semantic index for near-duplicates missed by MinHash
                if !is_duplicate {
                    let semantic_matches = self.semantic_index.search(&embedding, 5);
                    for m in semantic_matches {
                        if m.similarity >= self.config.semantic_threshold {
                            is_duplicate = true;
                            semantic_verified += 1;
                            break;
                        }
                    }
                }

                // Add to semantic index if unique
                if !is_duplicate {
                    self.semantic_index
                        .add(self.signatures.len() as u64, embedding);
                }
            } else {
                // Just use MinHash threshold
                is_duplicate = has_minhash_match;
            }

            if is_duplicate {
                remove_indices.push(i);
            } else {
                keep_indices.push(i);
                self.signatures.push(sig.clone());
            }
        }

        let unique_count = keep_indices.len();
        let duplicate_count = remove_indices.len();

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
                candidate_pairs: minhash_candidates,
                verified_pairs: semantic_verified,
                processing_time_secs: 0.0,
            },
        })
    }

    /// Get hybrid-specific statistics.
    pub fn stats(&self) -> HybridStats {
        HybridStats {
            total_documents: 0,
            unique_documents: 0,
            duplicate_count: 0,
            minhash_candidates: 0,
            semantic_verified: 0,
        }
    }

    /// Clear both indices.
    pub fn clear(&mut self) {
        self.signatures.clear();
        self.semantic_index.clear();
    }

    /// Get the number of documents in the index.
    #[must_use]
    pub fn index_size(&self) -> usize {
        self.signatures.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::MockBackend;

    #[test]
    fn test_hybrid_dedup_basic() {
        let backend = MockBackend::new(128);
        let config = HybridConfig::default();
        let mut dedup = HybridDeduplicator::new(backend, config);

        let texts = vec![
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog", // exact duplicate
            "A completely different sentence about cats",
        ];

        let result = dedup
            .deduplicate(&texts.iter().map(|s| *s).collect::<Vec<_>>())
            .unwrap();

        assert_eq!(result.keep_indices.len(), 2);
        assert_eq!(result.stats.duplicate_count, 1);
    }

    #[test]
    fn test_hybrid_empty() {
        let backend = MockBackend::new(128);
        let config = HybridConfig::default();
        let mut dedup = HybridDeduplicator::new(backend, config);

        let texts: Vec<&str> = vec![];
        let result = dedup.deduplicate(&texts).unwrap();

        assert!(result.keep_indices.is_empty());
    }

    #[test]
    fn test_hybrid_all_unique() {
        let backend = MockBackend::new(128);
        let config = HybridConfig::default();
        let mut dedup = HybridDeduplicator::new(backend, config);

        let texts = vec![
            "Document A about topic one",
            "Document B about topic two",
            "Document C about topic three",
        ];

        let result = dedup
            .deduplicate(&texts.iter().map(|s| *s).collect::<Vec<_>>())
            .unwrap();

        assert_eq!(result.keep_indices.len(), 3);
        assert_eq!(result.stats.duplicate_count, 0);
    }

    #[test]
    fn test_hybrid_config_variants() {
        let _fast = HybridConfig::fast();
        let _precise = HybridConfig::precise();
        let _default = HybridConfig::default();

        // Just ensure they compile and have sensible values
        assert!(_fast.minhash_threshold > 0.0);
        assert!(_precise.semantic_threshold > 0.0);
    }
}
