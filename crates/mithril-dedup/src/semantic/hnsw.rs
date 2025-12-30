//! HNSW-based semantic index for fast approximate nearest neighbor search.
//!
//! This module provides an HNSW (Hierarchical Navigable Small World) index
//! for efficient semantic similarity search on large datasets.

use instant_distance::{Builder, HnswMap, Search};
use std::collections::HashMap;

/// A point in the embedding space.
#[derive(Clone)]
pub struct EmbeddingPoint {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Document ID.
    pub doc_id: u64,
}

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        // For normalized vectors, this is equivalent to (2 - 2*dot_product) / 2
        let dot: f32 = self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        1.0 - dot
    }
}

/// HNSW-based semantic index configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Number of neighbors to consider during construction (M parameter).
    /// Higher = more accurate but slower construction and more memory.
    pub m: usize,
    /// Size of the dynamic candidate list during construction (ef_construction).
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search (ef_search).
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        }
    }
}

impl HnswConfig {
    /// Fast configuration (less accurate but faster).
    #[must_use]
    pub fn fast() -> Self {
        Self {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
        }
    }

    /// Precise configuration (more accurate but slower).
    #[must_use]
    pub fn precise() -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

/// A match result from HNSW search.
#[derive(Debug, Clone)]
pub struct HnswMatch {
    /// Document ID.
    pub doc_id: u64,
    /// Cosine similarity score (0.0 to 1.0 for normalized vectors).
    pub similarity: f32,
}

/// HNSW-based semantic index for fast approximate nearest neighbor search.
///
/// This provides O(log n) search time compared to O(n) for brute-force.
pub struct HnswIndex {
    /// Embedding dimensionality.
    dim: usize,
    /// Configuration.
    config: HnswConfig,
    /// Points stored before building the index.
    points: Vec<EmbeddingPoint>,
    /// Document ID to point index mapping.
    doc_id_to_idx: HashMap<u64, usize>,
    /// Built HNSW index (None until build() is called).
    hnsw: Option<HnswMap<EmbeddingPoint, u64>>,
    /// Whether the index needs rebuilding.
    needs_rebuild: bool,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self::with_config(dim, HnswConfig::default())
    }

    /// Create an index with custom configuration.
    #[must_use]
    pub fn with_config(dim: usize, config: HnswConfig) -> Self {
        Self {
            dim,
            config,
            points: Vec::new(),
            doc_id_to_idx: HashMap::new(),
            hnsw: None,
            needs_rebuild: false,
        }
    }

    /// Add a vector to the index.
    ///
    /// Note: After adding vectors, call `build()` to construct the HNSW graph.
    pub fn add(&mut self, doc_id: u64, embedding: Vec<f32>) {
        debug_assert_eq!(embedding.len(), self.dim, "Embedding dimension mismatch");

        let idx = self.points.len();
        self.points.push(EmbeddingPoint { embedding, doc_id });
        self.doc_id_to_idx.insert(doc_id, idx);
        self.needs_rebuild = true;
    }

    /// Build the HNSW index from added points.
    ///
    /// This must be called after adding points and before searching.
    pub fn build(&mut self) {
        if self.points.is_empty() {
            self.hnsw = None;
            self.needs_rebuild = false;
            return;
        }

        // Build values (doc_ids) for the map
        let values: Vec<u64> = self.points.iter().map(|p| p.doc_id).collect();

        // Build the HNSW index
        let hnsw = Builder::default()
            .ef_construction(self.config.ef_construction)
            .build(self.points.clone(), values);

        self.hnsw = Some(hnsw);
        self.needs_rebuild = false;
    }

    /// Search for the k nearest neighbors.
    ///
    /// Returns matches sorted by similarity (highest first).
    /// If the index hasn't been built, this will build it first.
    #[must_use]
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<HnswMatch> {
        debug_assert_eq!(query.len(), self.dim, "Query dimension mismatch");

        if self.needs_rebuild {
            self.build();
        }

        let hnsw = match &self.hnsw {
            Some(h) => h,
            None => return Vec::new(),
        };

        let query_point = EmbeddingPoint {
            embedding: query.to_vec(),
            doc_id: 0, // Unused for query
        };

        let mut search = Search::default();
        let results: Vec<_> = hnsw
            .search(&query_point, &mut search)
            .take(k)
            .map(|item| {
                // Convert distance back to similarity
                let similarity = 1.0 - item.distance;
                HnswMatch {
                    doc_id: *item.value,
                    similarity,
                }
            })
            .collect();

        results
    }

    /// Search and return only matches above a similarity threshold.
    #[must_use]
    pub fn search_threshold(
        &mut self,
        query: &[f32],
        threshold: f32,
        max_k: usize,
    ) -> Vec<HnswMatch> {
        self.search(query, max_k)
            .into_iter()
            .filter(|m| m.similarity >= threshold)
            .collect()
    }

    /// Get the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear all vectors from the index.
    pub fn clear(&mut self) {
        self.points.clear();
        self.doc_id_to_idx.clear();
        self.hnsw = None;
        self.needs_rebuild = false;
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the vector for a document ID.
    #[must_use]
    pub fn get(&self, doc_id: u64) -> Option<&[f32]> {
        self.doc_id_to_idx
            .get(&doc_id)
            .map(|&idx| self.points[idx].embedding.as_slice())
    }

    /// Check if the index needs rebuilding.
    #[must_use]
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }

    #[test]
    fn test_hnsw_basic() {
        let mut index = HnswIndex::new(3);

        let mut v1 = vec![1.0, 0.0, 0.0];
        let mut v2 = vec![0.0, 1.0, 0.0];
        let mut v3 = vec![0.9, 0.1, 0.0];

        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        index.add(0, v1.clone());
        index.add(1, v2);
        index.add(2, v3);
        index.build();

        assert_eq!(index.len(), 3);

        // Search for vector similar to v1
        let matches = index.search(&v1, 2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].doc_id, 0); // Exact match should be first
        assert!((matches[0].similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hnsw_search_threshold() {
        let mut index = HnswIndex::new(3);

        let mut v1 = vec![1.0, 0.0, 0.0];
        let mut v2 = vec![0.0, 1.0, 0.0];
        let mut v3 = vec![0.9, 0.1, 0.0];

        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        index.add(0, v1.clone());
        index.add(1, v2);
        index.add(2, v3);
        index.build();

        // High threshold should only return exact/near matches
        let matches = index.search_threshold(&v1, 0.9, 10);
        assert!(matches.len() <= 2);
        assert!(matches.iter().any(|m| m.doc_id == 0));
    }

    #[test]
    fn test_hnsw_empty() {
        let mut index = HnswIndex::new(3);
        index.build();

        let query = vec![1.0, 0.0, 0.0];
        let matches = index.search(&query, 10);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_hnsw_auto_rebuild() {
        let mut index = HnswIndex::new(3);

        let mut v1 = vec![1.0, 0.0, 0.0];
        normalize(&mut v1);

        index.add(0, v1.clone());
        // Don't explicitly call build()

        // Search should auto-build
        let matches = index.search(&v1, 1);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].doc_id, 0);
    }

    #[test]
    fn test_hnsw_config_variants() {
        let _default = HnswConfig::default();
        let _fast = HnswConfig::fast();
        let _precise = HnswConfig::precise();

        // Ensure configs have sensible values
        assert!(_default.m > 0);
        assert!(_fast.ef_search < _precise.ef_search);
    }
}
