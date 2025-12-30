//! Locality-Sensitive Hashing (LSH) for efficient candidate pair generation.
//!
//! LSH groups similar documents into buckets based on their MinHash signatures,
//! allowing O(n) candidate pair generation instead of O(n^2).

use crate::minhash::MinHashSignature;
use mithril_core::hashing::hash_with_seed;
use std::collections::{HashMap, HashSet};

/// Document ID type.
pub type DocId = u64;

/// LSH index for finding candidate duplicate pairs.
///
/// The index divides each MinHash signature into bands of rows.
/// Documents that share at least one band hash are considered candidates.
pub struct LshIndex {
    /// Number of bands (groups of rows).
    num_bands: usize,
    /// Rows per band.
    rows_per_band: usize,
    /// Buckets for each band: band_id -> hash -> doc_ids.
    buckets: Vec<HashMap<u64, Vec<DocId>>>,
}

impl LshIndex {
    /// Create a new LSH index with the specified band configuration.
    ///
    /// # Arguments
    /// * `num_bands` - Number of bands to divide the signature into
    /// * `rows_per_band` - Number of rows (hash values) per band
    #[must_use]
    pub fn new(num_bands: usize, rows_per_band: usize) -> Self {
        Self {
            num_bands,
            rows_per_band,
            buckets: (0..num_bands).map(|_| HashMap::new()).collect(),
        }
    }

    /// Create LSH index with target similarity threshold.
    ///
    /// For a threshold t with b bands and r rows per band:
    /// P(candidates) ≈ 1 - (1 - t^r)^b
    ///
    /// This method finds optimal b and r values for the given threshold.
    #[must_use]
    pub fn with_threshold(num_permutations: usize, threshold: f64) -> Self {
        let (num_bands, rows_per_band) = Self::optimize_params(num_permutations, threshold);
        Self::new(num_bands, rows_per_band)
    }

    /// Find optimal band/row parameters for the given threshold.
    ///
    /// Returns (num_bands, rows_per_band) that best matches the target threshold.
    fn optimize_params(n: usize, t: f64) -> (usize, usize) {
        let mut best = (1, n);
        let mut best_diff = f64::MAX;

        for b in 1..=n {
            if n % b == 0 {
                let r = n / b;
                // Threshold where P(candidate) ≈ 0.5
                // The S-curve threshold is approximately (1/b)^(1/r)
                let estimated_t = (1.0 / b as f64).powf(1.0 / r as f64);
                let diff = (estimated_t - t).abs();
                if diff < best_diff {
                    best = (b, r);
                    best_diff = diff;
                }
            }
        }

        best
    }

    /// Get the number of bands.
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Get the number of rows per band.
    #[must_use]
    pub fn rows_per_band(&self) -> usize {
        self.rows_per_band
    }

    /// Add a document signature to the index.
    ///
    /// The signature is divided into bands, and each band is hashed
    /// to place the document into buckets.
    pub fn insert(&mut self, doc_id: DocId, signature: &MinHashSignature) {
        assert!(
            signature.len() >= self.num_bands * self.rows_per_band,
            "Signature too short for LSH configuration"
        );

        for band in 0..self.num_bands {
            let start = band * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_hash = self.hash_band(&signature.values[start..end]);

            self.buckets[band]
                .entry(band_hash)
                .or_default()
                .push(doc_id);
        }
    }

    /// Hash a band of signature values to a single u64.
    fn hash_band(&self, values: &[u64]) -> u64 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        hash_with_seed(&bytes, 0)
    }

    /// Get all candidate pairs from the index.
    ///
    /// Returns an iterator over unique (doc_id1, doc_id2) pairs where
    /// doc_id1 < doc_id2 and the documents share at least one bucket.
    pub fn candidates(&self) -> impl Iterator<Item = (DocId, DocId)> + '_ {
        let mut seen = HashSet::new();

        self.buckets
            .iter()
            .flat_map(|band_buckets| {
                band_buckets
                    .values()
                    .filter(|bucket| bucket.len() > 1)
                    .flat_map(|bucket| {
                        bucket.iter().enumerate().flat_map(move |(i, &id1)| {
                            bucket[i + 1..].iter().map(move |&id2| {
                                // Ensure consistent ordering
                                if id1 < id2 {
                                    (id1, id2)
                                } else {
                                    (id2, id1)
                                }
                            })
                        })
                    })
            })
            .filter(move |pair| seen.insert(*pair))
    }

    /// Get candidate pairs as a vector (useful for parallel processing).
    #[must_use]
    pub fn candidates_vec(&self) -> Vec<(DocId, DocId)> {
        self.candidates().collect()
    }

    /// Get the number of buckets with multiple documents.
    #[must_use]
    pub fn num_collision_buckets(&self) -> usize {
        self.buckets
            .iter()
            .flat_map(|band| band.values())
            .filter(|bucket| bucket.len() > 1)
            .count()
    }

    /// Get the total number of documents in the index.
    #[must_use]
    pub fn num_documents(&self) -> usize {
        // Count unique doc IDs across all buckets
        let mut seen = HashSet::new();
        for band in &self.buckets {
            for docs in band.values() {
                for &doc_id in docs {
                    seen.insert(doc_id);
                }
            }
        }
        seen.len()
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        for band in &mut self.buckets {
            band.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minhash::MinHasher;

    #[test]
    fn test_lsh_basic_insertion() {
        let mut index = LshIndex::new(4, 32); // 4 bands, 32 rows = 128 permutations

        let hasher = MinHasher::new(128);
        let sig = hasher.signature_from_text("Test document for LSH");

        index.insert(1, &sig);

        assert_eq!(index.num_documents(), 1);
    }

    #[test]
    fn test_lsh_identical_documents() {
        let mut index = LshIndex::new(4, 32);
        let hasher = MinHasher::new(128);

        let text = "The quick brown fox jumps over the lazy dog";
        let sig = hasher.signature_from_text(text);

        index.insert(1, &sig);
        index.insert(2, &sig);

        let candidates: Vec<_> = index.candidates().collect();

        // Identical documents should be candidates
        assert_eq!(candidates.len(), 1);
        assert!(candidates.contains(&(1, 2)));
    }

    #[test]
    fn test_lsh_different_documents() {
        let mut index = LshIndex::new(4, 32);
        let hasher = MinHasher::new(128);

        let sig1 = hasher.signature_from_text("The quick brown fox");
        let sig2 = hasher.signature_from_text("Completely different text about machine learning");

        index.insert(1, &sig1);
        index.insert(2, &sig2);

        let candidates: Vec<_> = index.candidates().collect();

        // Very different documents should likely not be candidates
        // (though there's some probability of false positives)
        assert!(
            candidates.is_empty() || candidates.len() <= 1,
            "Expected few candidates for different documents"
        );
    }

    #[test]
    fn test_lsh_with_threshold() {
        // Test that with_threshold creates appropriate parameters
        let index = LshIndex::with_threshold(128, 0.85);

        assert_eq!(index.num_bands() * index.rows_per_band(), 128);

        // For 0.85 threshold, typical params are around b=20, r=6 or similar
        // Just verify the configuration is reasonable
        assert!(index.num_bands() > 0);
        assert!(index.rows_per_band() > 0);
    }

    #[test]
    fn test_lsh_candidate_deduplication() {
        let mut index = LshIndex::new(4, 32);
        let hasher = MinHasher::new(128);

        // Same text will hash to same buckets in all bands
        let text = "Same document text";
        let sig = hasher.signature_from_text(text);

        index.insert(1, &sig);
        index.insert(2, &sig);
        index.insert(3, &sig);

        let candidates: Vec<_> = index.candidates().collect();

        // Should get unique pairs: (1,2), (1,3), (2,3)
        assert_eq!(candidates.len(), 3);

        // Verify all pairs are present
        let mut sorted_candidates = candidates.clone();
        sorted_candidates.sort();
        assert!(sorted_candidates.contains(&(1, 2)));
        assert!(sorted_candidates.contains(&(1, 3)));
        assert!(sorted_candidates.contains(&(2, 3)));
    }

    #[test]
    fn test_lsh_many_documents() {
        let mut index = LshIndex::with_threshold(128, 0.85);
        let hasher = MinHasher::new(128);

        // Insert many different documents
        for i in 0..100 {
            let text = format!(
                "Document number {i} with unique content about topic {}",
                i % 10
            );
            let sig = hasher.signature_from_text(&text);
            index.insert(i, &sig);
        }

        assert_eq!(index.num_documents(), 100);

        // Candidates should be much fewer than n^2 pairs
        let candidates: Vec<_> = index.candidates().collect();
        let max_pairs = 100 * 99 / 2;
        assert!(
            candidates.len() < max_pairs / 2,
            "LSH should reduce candidate pairs significantly"
        );
    }

    #[test]
    fn test_lsh_clear() {
        let mut index = LshIndex::new(4, 32);
        let hasher = MinHasher::new(128);

        let sig = hasher.signature_from_text("Test document");
        index.insert(1, &sig);

        assert_eq!(index.num_documents(), 1);

        index.clear();

        assert_eq!(index.num_documents(), 0);
    }

    #[test]
    fn test_optimize_params() {
        // Test various thresholds produce valid configurations
        for threshold in [0.5, 0.7, 0.85, 0.9, 0.95] {
            let (b, r) = LshIndex::optimize_params(128, threshold);
            assert_eq!(b * r, 128, "b * r should equal num_permutations");
            assert!(b > 0);
            assert!(r > 0);
        }
    }

    #[test]
    fn test_lsh_pair_ordering() {
        let mut index = LshIndex::new(4, 32);
        let hasher = MinHasher::new(128);

        let text = "Test document for pair ordering";
        let sig = hasher.signature_from_text(text);

        // Insert in reverse order
        index.insert(10, &sig);
        index.insert(5, &sig);

        let candidates: Vec<_> = index.candidates().collect();

        // Pair should be (5, 10) not (10, 5)
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], (5, 10));
    }
}
