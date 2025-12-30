//! Semantic vector index for nearest neighbor search.
//!
//! This module provides a vector index for approximate nearest neighbor (ANN) search.
//! Currently uses brute-force search which is accurate but O(n) per query.
//! For large datasets, this can be upgraded to HNSW or other ANN algorithms.

use super::backend::cosine_similarity;
use std::cmp::Ordering;

/// A match result from semantic search.
#[derive(Debug, Clone)]
pub struct SemanticMatch {
    /// Document ID.
    pub doc_id: u64,
    /// Cosine similarity score (0.0 to 1.0 for normalized vectors).
    pub similarity: f32,
}

impl PartialEq for SemanticMatch {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && (self.similarity - other.similarity).abs() < f32::EPSILON
    }
}

impl Eq for SemanticMatch {}

impl PartialOrd for SemanticMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemanticMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher similarity = better match, so reverse order
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

/// A semantic vector index.
///
/// Stores embedding vectors and supports nearest neighbor search.
/// Currently uses brute-force O(n) search per query.
pub struct SemanticIndex {
    /// Embedding dimensionality.
    dim: usize,
    /// Document IDs.
    doc_ids: Vec<u64>,
    /// Embedding vectors (flattened for cache efficiency).
    vectors: Vec<f32>,
}

impl SemanticIndex {
    /// Create a new empty index.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            doc_ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Create an index with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            doc_ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * dim),
        }
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u64, embedding: Vec<f32>) {
        debug_assert_eq!(embedding.len(), self.dim, "Embedding dimension mismatch");
        self.doc_ids.push(doc_id);
        self.vectors.extend(embedding);
    }

    /// Add multiple vectors to the index.
    pub fn add_batch(&mut self, doc_ids: &[u64], embeddings: &[Vec<f32>]) {
        debug_assert_eq!(doc_ids.len(), embeddings.len());
        for (id, emb) in doc_ids.iter().zip(embeddings.iter()) {
            self.add(*id, emb.clone());
        }
    }

    /// Search for the k nearest neighbors.
    ///
    /// Returns matches sorted by similarity (highest first).
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SemanticMatch> {
        debug_assert_eq!(query.len(), self.dim, "Query dimension mismatch");

        if self.doc_ids.is_empty() {
            return Vec::new();
        }

        let mut matches: Vec<SemanticMatch> = self
            .doc_ids
            .iter()
            .enumerate()
            .map(|(i, &doc_id)| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors[start..end];
                let similarity = cosine_similarity(query, vec);
                SemanticMatch { doc_id, similarity }
            })
            .collect();

        // Sort by similarity (descending)
        matches.sort();

        // Return top k
        matches.truncate(k);
        matches
    }

    /// Search and return only matches above a similarity threshold.
    #[must_use]
    pub fn search_threshold(&self, query: &[f32], threshold: f32) -> Vec<SemanticMatch> {
        debug_assert_eq!(query.len(), self.dim, "Query dimension mismatch");

        self.doc_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &doc_id)| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors[start..end];
                let similarity = cosine_similarity(query, vec);
                if similarity >= threshold {
                    Some(SemanticMatch { doc_id, similarity })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find all pairs with similarity above threshold.
    ///
    /// Returns (i, j, similarity) tuples where i < j.
    #[must_use]
    pub fn find_all_pairs(&self, threshold: f32) -> Vec<(u64, u64, f32)> {
        let n = self.doc_ids.len();
        let mut pairs = Vec::new();

        for i in 0..n {
            let start_i = i * self.dim;
            let vec_i = &self.vectors[start_i..start_i + self.dim];

            for j in (i + 1)..n {
                let start_j = j * self.dim;
                let vec_j = &self.vectors[start_j..start_j + self.dim];

                let sim = cosine_similarity(vec_i, vec_j);
                if sim >= threshold {
                    pairs.push((self.doc_ids[i], self.doc_ids[j], sim));
                }
            }
        }

        pairs
    }

    /// Get the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Clear all vectors from the index.
    pub fn clear(&mut self) {
        self.doc_ids.clear();
        self.vectors.clear();
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the vector for a document ID.
    #[must_use]
    pub fn get(&self, doc_id: u64) -> Option<&[f32]> {
        self.doc_ids.iter().position(|&id| id == doc_id).map(|i| {
            let start = i * self.dim;
            &self.vectors[start..start + self.dim]
        })
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
    fn test_index_basic() {
        let mut index = SemanticIndex::new(3);

        let mut v1 = vec![1.0, 0.0, 0.0];
        let mut v2 = vec![0.0, 1.0, 0.0];
        let mut v3 = vec![1.0, 0.1, 0.0];

        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        index.add(0, v1.clone());
        index.add(1, v2);
        index.add(2, v3);

        assert_eq!(index.len(), 3);

        // Search for vector similar to v1
        let matches = index.search(&v1, 2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].doc_id, 0); // Exact match
        assert!((matches[0].similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_search_threshold() {
        let mut index = SemanticIndex::new(3);

        let mut v1 = vec![1.0, 0.0, 0.0];
        let mut v2 = vec![0.0, 1.0, 0.0];
        let mut v3 = vec![0.9, 0.1, 0.0];

        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        index.add(0, v1.clone());
        index.add(1, v2);
        index.add(2, v3);

        // High threshold should only return exact/near matches
        let matches = index.search_threshold(&v1, 0.9);
        assert!(matches.len() <= 2);
        assert!(matches.iter().any(|m| m.doc_id == 0));
    }

    #[test]
    fn test_find_all_pairs() {
        let mut index = SemanticIndex::new(3);

        // Two similar vectors, one different
        let mut v1 = vec![1.0, 0.0, 0.0];
        let mut v2 = vec![0.95, 0.05, 0.0];
        let mut v3 = vec![0.0, 1.0, 0.0];

        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        index.add(0, v1);
        index.add(1, v2);
        index.add(2, v3);

        let pairs = index.find_all_pairs(0.9);

        // v1 and v2 should be similar
        assert!(pairs.iter().any(|(a, b, _)| *a == 0 && *b == 1));
        // v3 should not be similar to others
        assert!(!pairs.iter().any(|(a, b, _)| *a == 2 || *b == 2));
    }

    #[test]
    fn test_empty_index() {
        let index = SemanticIndex::new(3);
        assert!(index.is_empty());

        let query = vec![1.0, 0.0, 0.0];
        let matches = index.search(&query, 10);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_get_vector() {
        let mut index = SemanticIndex::new(3);
        let v = vec![1.0, 2.0, 3.0];
        index.add(42, v.clone());

        let retrieved = index.get(42).unwrap();
        assert_eq!(retrieved, &v[..]);
        assert!(index.get(99).is_none());
    }
}
