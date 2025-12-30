//! LSH with Bloom filter optimization for memory-efficient deduplication.
//!
//! Uses Bloom filters instead of HashMaps for storing band hashes,
//! reducing memory usage by 10-50x at the cost of small false positive rate.
//!
//! This is ideal for large-scale deduplication (10M+ documents) where
//! exact candidate pairs are less important than finding potential duplicates.

use crate::minhash::MinHashSignature;
use bloomfilter::Bloom;
use mithril_core::hashing::hash_with_seed;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Document ID type.
pub type DocId = u64;

/// Memory-efficient LSH index using Bloom filters.
///
/// Unlike the standard `LshIndex` which stores all document IDs in buckets,
/// `LshBloomIndex` uses Bloom filters to test if a signature has been seen.
/// This trades some accuracy for significantly reduced memory usage.
///
/// Use this for:
/// - Processing 10M+ documents
/// - Memory-constrained environments
/// - Scenarios where some false positives are acceptable
///
/// # Example
/// ```ignore
/// use mithril_dedup::lsh_bloom::LshBloomIndex;
/// use mithril_dedup::minhash::MinHasher;
///
/// let hasher = MinHasher::new(128);
/// let mut index = LshBloomIndex::new(20, 6, 1_000_000, 0.001);
///
/// // Insert documents
/// for (id, text) in documents {
///     let sig = hasher.signature_from_text(text);
///     index.insert(id, &sig);
/// }
///
/// // Query for potential duplicates
/// let sig = hasher.signature_from_text(new_doc);
/// if index.query(&sig) {
///     println!("Potential duplicate detected");
/// }
/// ```
#[derive(Serialize, Deserialize)]
pub struct LshBloomIndex {
    /// Number of bands
    num_bands: usize,
    /// Rows per band
    rows_per_band: usize,
    /// Bloom filters for each band
    filters: Vec<Bloom<u64>>,
    /// Number of items inserted
    items_count: usize,
}

/// Default seed for reproducible Bloom filter creation.
pub const DEFAULT_BLOOM_SEED: [u8; 32] = [
    0x4d, 0x69, 0x74, 0x68, 0x72, 0x69, 0x6c, 0x20, // "Mithril "
    0x44, 0x65, 0x64, 0x75, 0x70, 0x20, 0x76, 0x31, // "Dedup v1"
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

impl LshBloomIndex {
    /// Create a new LSH Bloom index with deterministic seed.
    ///
    /// Uses a fixed seed for reproducible behavior. This is the recommended
    /// constructor for distributed deduplication where indices need to be
    /// merged from multiple workers.
    ///
    /// # Arguments
    /// * `num_bands` - Number of bands to divide signature into
    /// * `rows_per_band` - Number of rows per band
    /// * `expected_items` - Expected number of documents to insert
    /// * `fp_rate` - False positive rate per band (0.001 = 0.1%)
    #[must_use]
    pub fn new(
        num_bands: usize,
        rows_per_band: usize,
        expected_items: usize,
        fp_rate: f64,
    ) -> Self {
        Self::new_with_seed(
            num_bands,
            rows_per_band,
            expected_items,
            fp_rate,
            &DEFAULT_BLOOM_SEED,
        )
    }

    /// Create a new LSH Bloom index with custom seed.
    ///
    /// Use the same seed across all indices that will be merged together.
    ///
    /// # Arguments
    /// * `num_bands` - Number of bands to divide signature into
    /// * `rows_per_band` - Number of rows per band
    /// * `expected_items` - Expected number of documents to insert
    /// * `fp_rate` - False positive rate per band (0.001 = 0.1%)
    /// * `seed` - 32-byte seed for deterministic Bloom filter creation
    #[must_use]
    pub fn new_with_seed(
        num_bands: usize,
        rows_per_band: usize,
        expected_items: usize,
        fp_rate: f64,
        seed: &[u8; 32],
    ) -> Self {
        // Create a Bloom filter for each band with deterministic seeds
        let filters = (0..num_bands)
            .map(|band_idx| {
                // Derive per-band seed
                let mut band_seed = *seed;
                band_seed[0] ^= (band_idx & 0xFF) as u8;
                band_seed[1] ^= ((band_idx >> 8) & 0xFF) as u8;
                Bloom::new_for_fp_rate_with_seed(expected_items, fp_rate, &band_seed)
            })
            .collect();

        Self {
            num_bands,
            rows_per_band,
            filters,
            items_count: 0,
        }
    }

    /// Create index with target similarity threshold.
    ///
    /// # Arguments
    /// * `num_permutations` - Number of hash permutations in signatures
    /// * `threshold` - Similarity threshold (0.0 to 1.0)
    /// * `expected_items` - Expected number of documents
    /// * `fp_rate` - False positive rate per band
    #[must_use]
    pub fn with_threshold(
        num_permutations: usize,
        threshold: f64,
        expected_items: usize,
        fp_rate: f64,
    ) -> Self {
        let (num_bands, rows_per_band) = Self::optimize_params(num_permutations, threshold);
        Self::new(num_bands, rows_per_band, expected_items, fp_rate)
    }

    /// Find optimal band/row parameters for the given threshold.
    fn optimize_params(n: usize, t: f64) -> (usize, usize) {
        let mut best = (1, n);
        let mut best_diff = f64::MAX;

        for b in 1..=n {
            if n % b == 0 {
                let r = n / b;
                // S-curve threshold: (1/b)^(1/r)
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

    /// Get the number of items inserted.
    #[must_use]
    pub fn items_count(&self) -> usize {
        self.items_count
    }

    /// Hash a band of signature values to a single u64.
    fn hash_band(&self, values: &[u64]) -> u64 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        hash_with_seed(&bytes, 0)
    }

    /// Insert a signature into the index.
    ///
    /// After insertion, `query` will return true for this signature
    /// and any other signatures that share at least one band.
    pub fn insert(&mut self, signature: &MinHashSignature) {
        assert!(
            signature.len() >= self.num_bands * self.rows_per_band,
            "Signature too short for LSH configuration"
        );

        for band in 0..self.num_bands {
            let start = band * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_hash = self.hash_band(&signature.values[start..end]);

            self.filters[band].set(&band_hash);
        }

        self.items_count += 1;
    }

    /// Query if any signature with at least one matching band has been inserted.
    ///
    /// Returns true if the signature matches any previously inserted signature
    /// in at least one band. This indicates a potential duplicate.
    ///
    /// Note: Due to Bloom filter false positives, this may return true even
    /// for non-duplicates. Always verify with exact similarity computation.
    #[must_use]
    pub fn query(&self, signature: &MinHashSignature) -> bool {
        assert!(
            signature.len() >= self.num_bands * self.rows_per_band,
            "Signature too short for LSH configuration"
        );

        for band in 0..self.num_bands {
            let start = band * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_hash = self.hash_band(&signature.values[start..end]);

            if self.filters[band].check(&band_hash) {
                return true;
            }
        }

        false
    }

    /// Check and insert a signature atomically.
    ///
    /// Returns true if the signature was already present (potential duplicate).
    /// The signature is inserted regardless.
    pub fn check_and_insert(&mut self, signature: &MinHashSignature) -> bool {
        let is_present = self.query(signature);
        self.insert(signature);
        is_present
    }

    /// Estimate memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        // Each Bloom filter uses bitmap_bits / 8 bytes
        self.filters.iter().map(|f| f.bitmap().len() / 8).sum()
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        for filter in &mut self.filters {
            filter.clear();
        }
        self.items_count = 0;
    }

    /// Serialize the index to bytes.
    ///
    /// Uses bincode format for efficient serialization.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MergeError> {
        bincode::serialize(self).map_err(|e| MergeError::Serialization(e.to_string()))
    }

    /// Deserialize an index from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, MergeError> {
        bincode::deserialize(data).map_err(|e| MergeError::Deserialization(e.to_string()))
    }

    /// Merge another index into this one.
    ///
    /// The other index must have compatible parameters (same num_bands and rows_per_band).
    /// After merging, this index contains all entries from both indices.
    ///
    /// # Returns
    /// Statistics about the merge operation.
    pub fn merge(&mut self, other: &LshBloomIndex) -> Result<MergeStats, MergeError> {
        // Validate compatibility
        if self.num_bands != other.num_bands {
            return Err(MergeError::IncompatibleParams(format!(
                "num_bands mismatch: {} vs {}",
                self.num_bands, other.num_bands
            )));
        }
        if self.rows_per_band != other.rows_per_band {
            return Err(MergeError::IncompatibleParams(format!(
                "rows_per_band mismatch: {} vs {}",
                self.rows_per_band, other.rows_per_band
            )));
        }

        let items_before = self.items_count;

        // Merge Bloom filters using bitwise OR
        for (i, (self_filter, other_filter)) in self
            .filters
            .iter_mut()
            .zip(other.filters.iter())
            .enumerate()
        {
            // Get bitmaps and OR them together
            let self_bitmap = self_filter.bitmap();
            let other_bitmap = other_filter.bitmap();

            if self_bitmap.len() != other_bitmap.len() {
                return Err(MergeError::IncompatibleParams(format!(
                    "Bloom filter {} size mismatch: {} vs {}",
                    i,
                    self_bitmap.len(),
                    other_bitmap.len()
                )));
            }

            // Create merged bitmap
            let merged: Vec<u8> = self_bitmap
                .iter()
                .zip(other_bitmap.iter())
                .map(|(a, b)| a | b)
                .collect();

            // Reconstruct the filter with merged bitmap
            // Note: we lose exact count accuracy, but this is acceptable for Bloom filters
            *self_filter = Bloom::from_existing(
                &merged,
                self_filter.number_of_bits(),
                self_filter.number_of_hash_functions(),
                self_filter.sip_keys(),
            );
        }

        // Update item count (approximate, since we can't know exact overlap)
        self.items_count += other.items_count;

        Ok(MergeStats {
            items_before,
            items_added: other.items_count,
            items_after: self.items_count,
        })
    }

    /// Get configuration parameters for creating a compatible index.
    pub fn params(&self) -> LshBloomParams {
        LshBloomParams {
            num_bands: self.num_bands,
            rows_per_band: self.rows_per_band,
        }
    }
}

/// Errors that can occur during merge operations.
#[derive(Debug, Clone)]
pub enum MergeError {
    /// Index parameters are not compatible for merging.
    IncompatibleParams(String),
    /// Serialization failed.
    Serialization(String),
    /// Deserialization failed.
    Deserialization(String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompatibleParams(msg) => write!(f, "Incompatible parameters: {}", msg),
            Self::Serialization(msg) => write!(f, "Serialization failed: {}", msg),
            Self::Deserialization(msg) => write!(f, "Deserialization failed: {}", msg),
        }
    }
}

impl std::error::Error for MergeError {}

/// Statistics from a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of items before merge.
    pub items_before: usize,
    /// Number of items added from the other index.
    pub items_added: usize,
    /// Total items after merge (may include duplicates).
    pub items_after: usize,
}

/// Parameters for creating a compatible LshBloomIndex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LshBloomParams {
    /// Number of bands.
    pub num_bands: usize,
    /// Rows per band.
    pub rows_per_band: usize,
}

/// Builder for creating LshBloomIndex with sensible defaults.
pub struct LshBloomBuilder {
    num_permutations: usize,
    threshold: f64,
    expected_items: usize,
    fp_rate: f64,
}

impl LshBloomBuilder {
    /// Create a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_permutations: 128,
            threshold: 0.85,
            expected_items: 1_000_000,
            fp_rate: 0.001,
        }
    }

    /// Set the number of permutations (signature length).
    #[must_use]
    pub fn num_permutations(mut self, n: usize) -> Self {
        self.num_permutations = n;
        self
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Set the expected number of items.
    #[must_use]
    pub fn expected_items(mut self, n: usize) -> Self {
        self.expected_items = n;
        self
    }

    /// Set the false positive rate per band.
    #[must_use]
    pub fn fp_rate(mut self, rate: f64) -> Self {
        self.fp_rate = rate;
        self
    }

    /// Build the index.
    #[must_use]
    pub fn build(self) -> LshBloomIndex {
        LshBloomIndex::with_threshold(
            self.num_permutations,
            self.threshold,
            self.expected_items,
            self.fp_rate,
        )
    }
}

impl Default for LshBloomBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory comparison between LshIndex and LshBloomIndex.
pub struct MemoryStats {
    /// Estimated HashMap-based memory usage
    pub hashmap_bytes: usize,
    /// Bloom filter memory usage
    pub bloom_bytes: usize,
    /// Memory savings ratio
    pub savings_ratio: f64,
}

/// Estimate memory usage for both index types.
///
/// Useful for deciding which index type to use based on dataset size.
#[must_use]
pub fn estimate_memory(num_docs: usize, num_bands: usize, fp_rate: f64) -> MemoryStats {
    // HashMap estimate: each entry is roughly 48 bytes (key + Vec overhead)
    // Plus entries for each doc in each band
    let hashmap_bytes = num_docs * num_bands * 48;

    // Bloom filter: uses formula -n*ln(p) / (ln(2)^2) bits
    let bits_per_item = -(fp_rate.ln()) / (2.0_f64.ln().powi(2));
    let bloom_bits = (num_docs as f64 * bits_per_item) as usize;
    let bloom_bytes = (bloom_bits / 8 + 1) * num_bands;

    let savings_ratio = if bloom_bytes > 0 {
        hashmap_bytes as f64 / bloom_bytes as f64
    } else {
        0.0
    };

    MemoryStats {
        hashmap_bytes,
        bloom_bytes,
        savings_ratio,
    }
}

/// Streaming deduplicator using LSH Bloom filters.
///
/// Processes documents one at a time, returning whether each is a duplicate.
/// Much more memory efficient than batch processing for large datasets.
pub struct StreamingDeduplicator {
    index: LshBloomIndex,
    hasher: crate::minhash::MinHasher,
    seen: HashSet<DocId>,
}

impl StreamingDeduplicator {
    /// Create a new streaming deduplicator.
    ///
    /// # Arguments
    /// * `num_permutations` - Number of hash permutations
    /// * `threshold` - Similarity threshold
    /// * `expected_items` - Expected number of documents
    #[must_use]
    pub fn new(num_permutations: usize, threshold: f64, expected_items: usize) -> Self {
        let index =
            LshBloomIndex::with_threshold(num_permutations, threshold, expected_items, 0.001);
        let hasher = crate::minhash::MinHasher::new(num_permutations);

        Self {
            index,
            hasher,
            seen: HashSet::new(),
        }
    }

    /// Process a document and return whether it's a duplicate.
    ///
    /// Returns `true` if the document is a potential duplicate of a
    /// previously seen document.
    pub fn is_duplicate(&mut self, doc_id: DocId, text: &str) -> bool {
        let signature = self.hasher.signature_from_text(text);
        let is_dup = self.index.check_and_insert(&signature);

        if !is_dup {
            self.seen.insert(doc_id);
        }

        is_dup
    }

    /// Get the number of unique documents seen.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.seen.len()
    }

    /// Get estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index.memory_usage() + self.seen.len() * 16 // u64 + overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minhash::MinHasher;

    #[test]
    fn test_lsh_bloom_basic() {
        let mut index = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let hasher = MinHasher::new(128);

        let sig = hasher.signature_from_text("Test document for LSH Bloom");
        index.insert(&sig);

        assert_eq!(index.items_count(), 1);
        assert!(index.query(&sig));
    }

    #[test]
    fn test_lsh_bloom_identical_documents() {
        let mut index = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let hasher = MinHasher::new(128);

        let text = "The quick brown fox jumps over the lazy dog";
        let sig = hasher.signature_from_text(text);

        // First query should return false (not inserted yet)
        assert!(!index.query(&sig));

        // Insert
        index.insert(&sig);

        // Second query should return true
        assert!(index.query(&sig));

        // Same signature should still match
        let sig2 = hasher.signature_from_text(text);
        assert!(index.query(&sig2));
    }

    #[test]
    fn test_lsh_bloom_different_documents() {
        let mut index = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let hasher = MinHasher::new(128);

        let sig1 = hasher.signature_from_text("The quick brown fox");
        let sig2 = hasher.signature_from_text("Completely different text about machine learning");

        index.insert(&sig1);

        // Very different documents should likely not match
        // (though there's some probability of false positives)
        // We can't guarantee this due to Bloom filter FP, so just verify no crash
        let _ = index.query(&sig2);
    }

    #[test]
    fn test_lsh_bloom_with_threshold() {
        let index = LshBloomIndex::with_threshold(128, 0.85, 100_000, 0.001);

        assert_eq!(index.num_bands() * index.rows_per_band(), 128);
        assert!(index.num_bands() > 0);
        assert!(index.rows_per_band() > 0);
    }

    #[test]
    fn test_lsh_bloom_check_and_insert() {
        let mut index = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let hasher = MinHasher::new(128);

        let sig = hasher.signature_from_text("Test document");

        // First check_and_insert should return false
        assert!(!index.check_and_insert(&sig));

        // Second should return true
        assert!(index.check_and_insert(&sig));
    }

    #[test]
    fn test_lsh_bloom_memory_usage() {
        let index = LshBloomIndex::new(20, 6, 1_000_000, 0.001);
        let memory = index.memory_usage();

        // Should be much less than HashMap equivalent
        // HashMap would need ~20 * 1M * 48 bytes = ~960 MB
        // Bloom should be around 10-20 MB depending on FP rate
        assert!(memory < 100_000_000, "Memory usage should be reasonable");
    }

    #[test]
    fn test_lsh_bloom_clear() {
        let mut index = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let hasher = MinHasher::new(128);

        let sig = hasher.signature_from_text("Test document");
        index.insert(&sig);

        assert_eq!(index.items_count(), 1);
        assert!(index.query(&sig));

        index.clear();

        assert_eq!(index.items_count(), 0);
        // After clear, query should return false (no false negatives in cleared filter)
        assert!(!index.query(&sig));
    }

    #[test]
    fn test_memory_estimation() {
        let stats = estimate_memory(1_000_000, 20, 0.001);

        assert!(stats.hashmap_bytes > 0);
        assert!(stats.bloom_bytes > 0);
        assert!(stats.savings_ratio > 1.0, "Bloom should save memory");

        // Bloom should be at least 10x smaller
        assert!(
            stats.savings_ratio > 10.0,
            "Expected >10x savings, got {:.1}x",
            stats.savings_ratio
        );
    }

    #[test]
    fn test_builder_pattern() {
        let index = LshBloomBuilder::new()
            .num_permutations(256)
            .threshold(0.9)
            .expected_items(500_000)
            .fp_rate(0.0001)
            .build();

        assert!(index.num_bands() * index.rows_per_band() == 256);
    }

    #[test]
    fn test_streaming_deduplicator() {
        let mut dedup = StreamingDeduplicator::new(128, 0.85, 10_000);

        // First occurrence is not a duplicate
        assert!(!dedup.is_duplicate(1, "The quick brown fox"));

        // Exact copy is a duplicate
        assert!(dedup.is_duplicate(2, "The quick brown fox"));

        // Different text is not a duplicate
        assert!(!dedup.is_duplicate(3, "Completely different content here"));

        assert_eq!(dedup.unique_count(), 2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut index = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let hasher = MinHasher::new(128);

        // Insert some documents
        for text in &["doc one", "doc two", "doc three"] {
            let sig = hasher.signature_from_text(text);
            index.insert(&sig);
        }

        // Serialize
        let bytes = index.to_bytes().unwrap();

        // Deserialize
        let index2 = LshBloomIndex::from_bytes(&bytes).unwrap();

        // Verify
        assert_eq!(index.num_bands(), index2.num_bands());
        assert_eq!(index.rows_per_band(), index2.rows_per_band());
        assert_eq!(index.items_count(), index2.items_count());

        // Query should work the same
        let sig = hasher.signature_from_text("doc one");
        assert_eq!(index.query(&sig), index2.query(&sig));
    }

    #[test]
    fn test_merge_indices() {
        let hasher = MinHasher::new(128);

        // Create two indices with the same params
        let mut index1 = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let mut index2 = LshBloomIndex::new(4, 32, 10_000, 0.01);

        // Insert different documents
        let sig1 = hasher.signature_from_text("Document in index one");
        let sig2 = hasher.signature_from_text("Document in index two");

        index1.insert(&sig1);
        index2.insert(&sig2);

        // Before merge, index1 can find sig1 but not sig2
        assert!(index1.query(&sig1));
        // sig2 might or might not match due to Bloom FP - we can't assert !query

        // Merge
        let stats = index1.merge(&index2).unwrap();
        assert_eq!(stats.items_before, 1);
        assert_eq!(stats.items_added, 1);
        assert_eq!(stats.items_after, 2);

        // After merge, index1 should find both
        assert!(index1.query(&sig1));
        assert!(index1.query(&sig2));
    }

    #[test]
    fn test_merge_incompatible_params() {
        let index1 = LshBloomIndex::new(4, 32, 10_000, 0.01);
        let index2 = LshBloomIndex::new(8, 16, 10_000, 0.01); // Different bands

        let mut index1_mut = index1;
        let result = index1_mut.merge(&index2);

        assert!(result.is_err());
        match result {
            Err(MergeError::IncompatibleParams(msg)) => {
                assert!(msg.contains("num_bands"));
            }
            _ => panic!("Expected IncompatibleParams error"),
        }
    }

    #[test]
    fn test_params() {
        let index = LshBloomIndex::new(5, 25, 10_000, 0.01);
        let params = index.params();

        assert_eq!(params.num_bands, 5);
        assert_eq!(params.rows_per_band, 25);
    }

    #[test]
    fn test_merge_multiple_indices() {
        let hasher = MinHasher::new(128);

        // Create coordinator index
        let mut global = LshBloomIndex::new(4, 32, 100_000, 0.01);

        // Simulate multiple workers sending indices
        for worker_id in 0..5 {
            let mut worker_index = LshBloomIndex::new(4, 32, 100_000, 0.01);

            // Each worker processes some documents
            for doc_id in 0..100 {
                let text = format!(
                    "Worker {} document {} with unique content",
                    worker_id, doc_id
                );
                let sig = hasher.signature_from_text(&text);
                worker_index.insert(&sig);
            }

            // Merge worker index into global
            global.merge(&worker_index).unwrap();
        }

        // Global should have all items (500 total)
        assert_eq!(global.items_count(), 500);

        // Should be able to find documents from any worker
        let test_sig = hasher.signature_from_text("Worker 2 document 50 with unique content");
        assert!(global.query(&test_sig));
    }
}
