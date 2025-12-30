//! Delta encoding for checkpoint compression.
//!
//! During training, consecutive checkpoints are highly similar (~99% identical).
//! XOR encoding converts unchanged bytes to zeros, which compress extremely well.
//!
//! Expected compression ratios:
//! - Without delta: 1.3-1.5x
//! - With delta: 39-70x (after warmup)
//!
//! ## Usage
//!
//! ```rust
//! use mithril_checkpoint::delta::{DeltaCompressor, DeltaEncoder};
//! use mithril_checkpoint::pipeline::CompressionConfig;
//!
//! // Low-level: Use DeltaEncoder for raw XOR encoding
//! let previous = vec![1u8, 2, 3, 4];
//! let current = vec![1u8, 2, 5, 4];  // Byte 2 changed
//! let delta = DeltaEncoder::encode(&current, Some(&previous));
//! assert_eq!(delta, vec![0, 0, 6, 0]);  // XOR result
//!
//! // High-level: Use DeltaCompressor for reference-tracked compression
//! let mut compressor = DeltaCompressor::new(CompressionConfig::default());
//! let data = vec![0u8; 10000];
//! let (compressed, stats) = compressor.compress_checkpoint("step_100", &data).unwrap();
//! ```

use mithril_core::hashing::hash_with_seed;
use rayon::prelude::*;
use std::collections::HashMap;

/// Hash bytes using xxhash3 with seed 0.
#[inline]
fn hash_bytes(data: &[u8]) -> u64 {
    hash_with_seed(data, 0)
}

/// Threshold for using parallel XOR (1 MB).
const PARALLEL_THRESHOLD: usize = 1024 * 1024;

/// Delta encoder using XOR encoding.
///
/// # Example
///
/// ```
/// use mithril_checkpoint::delta::DeltaEncoder;
///
/// let previous = vec![1, 2, 3, 4];
/// let current = vec![1, 2, 5, 4];  // Only byte 2 changed
///
/// let delta = DeltaEncoder::encode(&current, Some(&previous));
/// assert_eq!(delta, vec![0, 0, 6, 0]);  // XOR: unchanged bytes become 0
///
/// let recovered = DeltaEncoder::decode(&delta, &previous);
/// assert_eq!(recovered, current);
/// ```
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// XOR encode current data against previous checkpoint.
    ///
    /// If `previous` is None, returns a copy of the current data.
    /// If `previous` is Some, returns XOR of current with previous.
    ///
    /// # Panics
    ///
    /// Panics if `current` and `previous` have different lengths.
    pub fn encode(current: &[u8], previous: Option<&[u8]>) -> Vec<u8> {
        match previous {
            Some(prev) => {
                assert_eq!(
                    current.len(),
                    prev.len(),
                    "Checkpoint sizes must match for delta encoding: current={}, previous={}",
                    current.len(),
                    prev.len()
                );

                if current.len() >= PARALLEL_THRESHOLD {
                    Self::encode_parallel(current, prev)
                } else {
                    Self::encode_sequential(current, prev)
                }
            }
            None => current.to_vec(),
        }
    }

    /// XOR decode delta back to full checkpoint.
    ///
    /// # Panics
    ///
    /// Panics if `delta` and `previous` have different lengths.
    pub fn decode(delta: &[u8], previous: &[u8]) -> Vec<u8> {
        assert_eq!(
            delta.len(),
            previous.len(),
            "Delta and previous sizes must match: delta={}, previous={}",
            delta.len(),
            previous.len()
        );

        if delta.len() >= PARALLEL_THRESHOLD {
            Self::decode_parallel(delta, previous)
        } else {
            Self::decode_sequential(delta, previous)
        }
    }

    /// Sequential XOR encoding.
    #[inline]
    fn encode_sequential(current: &[u8], previous: &[u8]) -> Vec<u8> {
        current
            .iter()
            .zip(previous.iter())
            .map(|(&c, &p)| c ^ p)
            .collect()
    }

    /// Sequential XOR decoding.
    #[inline]
    fn decode_sequential(delta: &[u8], previous: &[u8]) -> Vec<u8> {
        delta
            .iter()
            .zip(previous.iter())
            .map(|(&d, &p)| d ^ p)
            .collect()
    }

    /// Parallel XOR encoding using rayon with chunked processing.
    ///
    /// Uses pre-allocated output buffer with parallel chunk processing
    /// for better cache locality and reduced synchronization overhead.
    fn encode_parallel(current: &[u8], previous: &[u8]) -> Vec<u8> {
        const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks for cache efficiency

        let mut delta = vec![0u8; current.len()];

        delta
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let offset = chunk_idx * CHUNK_SIZE;
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = current[offset + i] ^ previous[offset + i];
                }
            });

        delta
    }

    /// Parallel XOR decoding using rayon with chunked processing.
    fn decode_parallel(delta: &[u8], previous: &[u8]) -> Vec<u8> {
        const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks for cache efficiency

        let mut output = vec![0u8; delta.len()];

        output
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let offset = chunk_idx * CHUNK_SIZE;
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = delta[offset + i] ^ previous[offset + i];
                }
            });

        output
    }

    /// Compute the sparsity of a delta (fraction of zero bytes).
    ///
    /// Higher sparsity means better compression. A sparsity of 0.99 means
    /// only 1% of bytes changed between checkpoints.
    pub fn sparsity(delta: &[u8]) -> f64 {
        if delta.is_empty() {
            return 1.0;
        }

        let zeros = if delta.len() >= PARALLEL_THRESHOLD {
            delta.par_iter().filter(|&&b| b == 0).count()
        } else {
            delta.iter().filter(|&&b| b == 0).count()
        };

        zeros as f64 / delta.len() as f64
    }
}

// =============================================================================
// DeltaCompressor: Reference-tracked delta compression
// =============================================================================

use crate::pipeline::{CheckpointCompressor, CompressionConfig};
use mithril_core::types::DType;
use mithril_core::Result;

/// A checkpoint signature for reference tracking.
///
/// Stores metadata about a checkpoint for delta encoding decisions.
#[derive(Debug, Clone)]
pub struct CheckpointSignature {
    /// Content hash of the checkpoint data.
    pub hash: u64,
    /// Size in bytes.
    pub size: usize,
    /// The raw checkpoint data (kept for delta encoding).
    data: Vec<u8>,
}

impl CheckpointSignature {
    /// Create a new signature from checkpoint data.
    #[must_use]
    pub fn from_data(data: &[u8]) -> Self {
        Self {
            hash: hash_bytes(data),
            size: data.len(),
            data: data.to_vec(),
        }
    }

    /// Get the checkpoint data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

/// Statistics from delta compression.
#[derive(Debug, Clone)]
pub struct DeltaStats {
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Compression ratio (original / compressed).
    pub ratio: f64,
    /// Sparsity of delta (fraction of zero bytes).
    pub sparsity: f64,
    /// Whether delta encoding was used.
    pub used_delta: bool,
    /// Reference checkpoint key (if delta was used).
    pub reference_key: Option<String>,
}

/// Delta compressor with automatic reference management.
///
/// Tracks previous checkpoints and automatically uses delta encoding
/// when beneficial. This is the high-level API for checkpoint compression
/// during training.
///
/// # Example
///
/// ```rust
/// use mithril_checkpoint::delta::DeltaCompressor;
/// use mithril_checkpoint::pipeline::CompressionConfig;
///
/// let mut compressor = DeltaCompressor::new(CompressionConfig::default());
///
/// // First checkpoint (no delta available)
/// let data_step1: Vec<u8> = (0..50000).map(|i| (i % 256) as u8).collect();
/// let (_, stats1) = compressor.compress_checkpoint("step_100", &data_step1).unwrap();
/// assert!(!stats1.used_delta);
///
/// // Second checkpoint with only 1% changed (uses delta from step_100)
/// let mut data_step2 = data_step1.clone();
/// for i in (0..data_step2.len()).step_by(100) {
///     data_step2[i] = data_step2[i].wrapping_add(1);
/// }
/// let (_, stats2) = compressor.compress_checkpoint("step_200", &data_step2).unwrap();
/// assert!(stats2.used_delta);
/// assert!(stats2.sparsity > 0.9);  // High sparsity due to delta
/// ```
pub struct DeltaCompressor {
    config: CompressionConfig,
    compressor: CheckpointCompressor,
    /// Reference store: maps checkpoint key to signature.
    references: HashMap<String, CheckpointSignature>,
    /// Most recent checkpoint key for automatic delta chaining.
    latest_key: Option<String>,
    /// Maximum number of references to keep (for memory management).
    max_references: usize,
}

impl DeltaCompressor {
    /// Create a new delta compressor with the given config.
    #[must_use]
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            compressor: CheckpointCompressor::new(config.clone()),
            config,
            references: HashMap::new(),
            latest_key: None,
            max_references: 10,
        }
    }

    /// Set the maximum number of references to keep.
    ///
    /// Older references are evicted when this limit is exceeded.
    #[must_use]
    pub fn with_max_references(mut self, max: usize) -> Self {
        self.max_references = max;
        self
    }

    /// Get the compression configuration.
    #[must_use]
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }

    /// Compress a checkpoint with automatic delta encoding.
    ///
    /// Uses the most recent stored checkpoint as the delta reference.
    /// The compressed checkpoint is stored for future delta encoding.
    ///
    /// # Arguments
    /// * `key` - Unique identifier for this checkpoint (e.g., "step_100")
    /// * `data` - Raw checkpoint bytes
    ///
    /// # Returns
    /// Tuple of (compressed_data, stats).
    pub fn compress_checkpoint(&mut self, key: &str, data: &[u8]) -> Result<(Vec<u8>, DeltaStats)> {
        self.compress_checkpoint_with_dtype(key, data, DType::BFloat16)
    }

    /// Compress a checkpoint with specified dtype.
    pub fn compress_checkpoint_with_dtype(
        &mut self,
        key: &str,
        data: &[u8],
        dtype: DType,
    ) -> Result<(Vec<u8>, DeltaStats)> {
        // Try to find a suitable reference
        let reference = self
            .latest_key
            .as_ref()
            .and_then(|k| self.references.get(k));

        let (compressed, stats) = if let Some(ref_sig) = reference {
            // Check if sizes match (required for delta encoding)
            if ref_sig.size == data.len() {
                let compressed =
                    self.compressor
                        .compress_with_delta(data, dtype, Some(ref_sig.data()))?;

                let delta = DeltaEncoder::encode(data, Some(ref_sig.data()));
                let sparsity = DeltaEncoder::sparsity(&delta);

                let stats = DeltaStats {
                    original_size: data.len(),
                    compressed_size: compressed.len(),
                    ratio: data.len() as f64 / compressed.len() as f64,
                    sparsity,
                    used_delta: true,
                    reference_key: self.latest_key.clone(),
                };

                (compressed, stats)
            } else {
                // Size mismatch, compress without delta
                self.compress_standalone(data, dtype)?
            }
        } else {
            // No reference available
            self.compress_standalone(data, dtype)?
        };

        // Store this checkpoint as a reference
        self.store_reference(key, data);

        Ok((compressed, stats))
    }

    /// Compress a checkpoint with a specific reference.
    ///
    /// Use this when you want explicit control over the delta reference.
    pub fn compress_with_reference(
        &mut self,
        key: &str,
        data: &[u8],
        reference_key: &str,
        dtype: DType,
    ) -> Result<(Vec<u8>, DeltaStats)> {
        let reference = self.references.get(reference_key);

        let (compressed, stats) = if let Some(ref_sig) = reference {
            if ref_sig.size == data.len() {
                let compressed =
                    self.compressor
                        .compress_with_delta(data, dtype, Some(ref_sig.data()))?;

                let delta = DeltaEncoder::encode(data, Some(ref_sig.data()));
                let sparsity = DeltaEncoder::sparsity(&delta);

                let stats = DeltaStats {
                    original_size: data.len(),
                    compressed_size: compressed.len(),
                    ratio: data.len() as f64 / compressed.len() as f64,
                    sparsity,
                    used_delta: true,
                    reference_key: Some(reference_key.to_string()),
                };

                (compressed, stats)
            } else {
                self.compress_standalone(data, dtype)?
            }
        } else {
            self.compress_standalone(data, dtype)?
        };

        // Store this checkpoint
        self.store_reference(key, data);

        Ok((compressed, stats))
    }

    /// Decompress a checkpoint.
    ///
    /// # Arguments
    /// * `data` - Compressed checkpoint bytes
    /// * `original_size` - Expected decompressed size
    /// * `reference_key` - Key of reference checkpoint (if delta was used)
    /// * `dtype` - Data type of the tensor
    pub fn decompress_checkpoint(
        &self,
        data: &[u8],
        original_size: usize,
        reference_key: Option<&str>,
        dtype: DType,
    ) -> Result<Vec<u8>> {
        let reference = reference_key.and_then(|k| self.references.get(k));

        self.compressor.decompress_with_delta(
            data,
            dtype,
            original_size,
            reference.map(|r| r.data()),
        )
    }

    /// Store a reference checkpoint for future delta encoding.
    pub fn store_reference(&mut self, key: &str, data: &[u8]) {
        // Evict if at capacity (unless we're updating an existing key)
        if self.references.len() >= self.max_references && !self.references.contains_key(key) {
            // Find a key to evict that isn't the latest
            let key_to_evict = self
                .references
                .keys()
                .find(|k| Some(*k) != self.latest_key.as_ref())
                .cloned();

            if let Some(evict_key) = key_to_evict {
                self.references.remove(&evict_key);
            }
        }

        let signature = CheckpointSignature::from_data(data);
        self.references.insert(key.to_string(), signature);
        self.latest_key = Some(key.to_string());
    }

    /// Get a stored reference by key.
    #[must_use]
    pub fn get_reference(&self, key: &str) -> Option<&CheckpointSignature> {
        self.references.get(key)
    }

    /// Check if a reference exists.
    #[must_use]
    pub fn has_reference(&self, key: &str) -> bool {
        self.references.contains_key(key)
    }

    /// Get the number of stored references.
    #[must_use]
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    /// Clear all stored references.
    pub fn clear_references(&mut self) {
        self.references.clear();
        self.latest_key = None;
    }

    /// Get the latest reference key.
    #[must_use]
    pub fn latest_reference(&self) -> Option<&str> {
        self.latest_key.as_deref()
    }

    /// Compress without delta encoding.
    fn compress_standalone(&self, data: &[u8], dtype: DType) -> Result<(Vec<u8>, DeltaStats)> {
        let compressed = self.compressor.compress(data, dtype)?;

        let stats = DeltaStats {
            original_size: data.len(),
            compressed_size: compressed.len(),
            ratio: data.len() as f64 / compressed.len() as f64,
            sparsity: 0.0,
            used_delta: false,
            reference_key: None,
        };

        Ok((compressed, stats))
    }
}

impl Default for DeltaCompressor {
    fn default() -> Self {
        Self::new(CompressionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_no_previous() {
        let data = vec![1, 2, 3, 4, 5];
        let encoded = DeltaEncoder::encode(&data, None);
        assert_eq!(encoded, data);
    }

    #[test]
    fn test_encode_with_previous() {
        let previous = vec![1, 2, 3, 4];
        let current = vec![1, 2, 5, 4]; // Byte 2 changed: 3 ^ 5 = 6

        let delta = DeltaEncoder::encode(&current, Some(&previous));
        assert_eq!(delta, vec![0, 0, 6, 0]);
    }

    #[test]
    fn test_decode() {
        let previous = vec![1, 2, 3, 4];
        let delta = vec![0, 0, 6, 0];

        let recovered = DeltaEncoder::decode(&delta, &previous);
        assert_eq!(recovered, vec![1, 2, 5, 4]);
    }

    #[test]
    fn test_roundtrip() {
        let previous = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let current = vec![10, 21, 30, 41, 50, 61, 70, 81];

        let delta = DeltaEncoder::encode(&current, Some(&previous));
        let recovered = DeltaEncoder::decode(&delta, &previous);

        assert_eq!(recovered, current, "Roundtrip must be bit-exact");
    }

    #[test]
    fn test_identical_data_produces_zeros() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let delta = DeltaEncoder::encode(&data, Some(&data));

        assert!(
            delta.iter().all(|&b| b == 0),
            "XOR of identical data should be all zeros"
        );
    }

    #[test]
    fn test_empty_data() {
        let empty: Vec<u8> = vec![];
        let delta = DeltaEncoder::encode(&empty, Some(&empty));
        assert!(delta.is_empty());

        let recovered = DeltaEncoder::decode(&delta, &empty);
        assert!(recovered.is_empty());
    }

    #[test]
    #[should_panic(expected = "Checkpoint sizes must match")]
    fn test_encode_size_mismatch() {
        let previous = vec![1, 2, 3];
        let current = vec![1, 2, 3, 4];
        let _ = DeltaEncoder::encode(&current, Some(&previous));
    }

    #[test]
    #[should_panic(expected = "Delta and previous sizes must match")]
    fn test_decode_size_mismatch() {
        let previous = vec![1, 2, 3];
        let delta = vec![0, 0, 0, 0];
        let _ = DeltaEncoder::decode(&delta, &previous);
    }

    #[test]
    fn test_sparsity_all_zeros() {
        let delta = vec![0, 0, 0, 0, 0];
        assert_eq!(DeltaEncoder::sparsity(&delta), 1.0);
    }

    #[test]
    fn test_sparsity_no_zeros() {
        let delta = vec![1, 2, 3, 4, 5];
        assert_eq!(DeltaEncoder::sparsity(&delta), 0.0);
    }

    #[test]
    fn test_sparsity_half_zeros() {
        let delta = vec![0, 1, 0, 2];
        assert_eq!(DeltaEncoder::sparsity(&delta), 0.5);
    }

    #[test]
    fn test_sparsity_empty() {
        let delta: Vec<u8> = vec![];
        assert_eq!(DeltaEncoder::sparsity(&delta), 1.0);
    }

    #[test]
    fn test_large_data_parallel() {
        // Test with data larger than PARALLEL_THRESHOLD
        let size = PARALLEL_THRESHOLD + 1000;
        let previous: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let current: Vec<u8> = (0..size).map(|i| ((i + 1) % 256) as u8).collect();

        let delta = DeltaEncoder::encode(&current, Some(&previous));
        let recovered = DeltaEncoder::decode(&delta, &previous);

        assert_eq!(recovered, current, "Parallel roundtrip must be bit-exact");
    }

    #[test]
    fn test_realistic_checkpoint_simulation() {
        // Simulate realistic checkpoint: mostly unchanged, few bytes differ
        let size = 10000;
        let previous: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        // Change only 1% of bytes
        let mut current = previous.clone();
        for i in (0..size).step_by(100) {
            current[i] = current[i].wrapping_add(1);
        }

        let delta = DeltaEncoder::encode(&current, Some(&previous));
        let sparsity = DeltaEncoder::sparsity(&delta);

        // Should have ~99% zeros
        assert!(
            sparsity > 0.98,
            "Expected sparsity > 0.98, got {}",
            sparsity
        );

        // Verify roundtrip
        let recovered = DeltaEncoder::decode(&delta, &previous);
        assert_eq!(recovered, current);
    }

    // =========================================================================
    // DeltaCompressor tests
    // =========================================================================

    #[test]
    fn test_delta_compressor_first_checkpoint() {
        let mut compressor = DeltaCompressor::default();
        let data = vec![0u8; 10000];

        let (compressed, stats) = compressor.compress_checkpoint("step_100", &data).unwrap();

        assert!(!stats.used_delta);
        assert!(stats.reference_key.is_none());
        assert!(compressed.len() < data.len());
        assert_eq!(compressor.reference_count(), 1);
        assert!(compressor.has_reference("step_100"));
    }

    #[test]
    fn test_delta_compressor_second_checkpoint() {
        let mut compressor = DeltaCompressor::default();

        // First checkpoint
        let data1: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let (_, stats1) = compressor.compress_checkpoint("step_100", &data1).unwrap();
        assert!(!stats1.used_delta);

        // Second checkpoint with small changes
        let mut data2 = data1.clone();
        data2[0] = 255;
        data2[100] = 255;

        let (_, stats2) = compressor.compress_checkpoint("step_200", &data2).unwrap();
        assert!(stats2.used_delta);
        assert_eq!(stats2.reference_key, Some("step_100".to_string()));
        assert!(stats2.ratio > stats1.ratio, "Delta should compress better");
    }

    #[test]
    fn test_delta_compressor_identical_checkpoints() {
        let mut compressor = DeltaCompressor::default();

        let data: Vec<u8> = (0..50000).map(|i| (i % 256) as u8).collect();

        // First checkpoint
        let (_, stats1) = compressor.compress_checkpoint("step_100", &data).unwrap();

        // Second identical checkpoint
        let (_, stats2) = compressor.compress_checkpoint("step_200", &data).unwrap();

        assert!(stats2.used_delta);
        assert!(
            stats2.sparsity > 0.99,
            "Identical data should have ~100% sparsity"
        );
        assert!(
            stats2.ratio > stats1.ratio * 5.0,
            "Identical data should compress 5x+ better with delta"
        );
    }

    #[test]
    fn test_delta_compressor_roundtrip() {
        let mut compressor = DeltaCompressor::default();

        // First checkpoint
        let data1: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let (compressed1, _) = compressor.compress_checkpoint("step_100", &data1).unwrap();

        // Decompress without delta
        let decompressed1 = compressor
            .decompress_checkpoint(&compressed1, data1.len(), None, DType::BFloat16)
            .unwrap();
        assert_eq!(data1, decompressed1);

        // Second checkpoint
        let mut data2 = data1.clone();
        data2[0] = 255;
        let (compressed2, stats2) = compressor.compress_checkpoint("step_200", &data2).unwrap();

        // Decompress with delta reference
        let decompressed2 = compressor
            .decompress_checkpoint(
                &compressed2,
                data2.len(),
                stats2.reference_key.as_deref(),
                DType::BFloat16,
            )
            .unwrap();
        assert_eq!(data2, decompressed2);
    }

    #[test]
    fn test_delta_compressor_explicit_reference() {
        let mut compressor = DeltaCompressor::default();

        // Store a reference
        let data1: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        compressor.store_reference("base", &data1);

        // Compress with explicit reference
        let mut data2 = data1.clone();
        data2[0] = 255;
        let (_, stats) = compressor
            .compress_with_reference("current", &data2, "base", DType::BFloat16)
            .unwrap();

        assert!(stats.used_delta);
        assert_eq!(stats.reference_key, Some("base".to_string()));
    }

    #[test]
    fn test_delta_compressor_max_references() {
        let mut compressor = DeltaCompressor::default().with_max_references(3);

        let data = vec![0u8; 1000];

        // Add 4 checkpoints, should evict oldest
        compressor.store_reference("step_1", &data);
        compressor.store_reference("step_2", &data);
        compressor.store_reference("step_3", &data);

        assert_eq!(compressor.reference_count(), 3);

        compressor.store_reference("step_4", &data);

        // Should still have 3 references, oldest evicted
        assert_eq!(compressor.reference_count(), 3);
        assert!(compressor.has_reference("step_4"));
        assert_eq!(compressor.latest_reference(), Some("step_4"));
    }

    #[test]
    fn test_delta_compressor_clear() {
        let mut compressor = DeltaCompressor::default();

        let data = vec![0u8; 1000];
        compressor.store_reference("step_1", &data);
        compressor.store_reference("step_2", &data);

        assert_eq!(compressor.reference_count(), 2);

        compressor.clear_references();

        assert_eq!(compressor.reference_count(), 0);
        assert!(compressor.latest_reference().is_none());
    }

    #[test]
    fn test_checkpoint_signature() {
        let data = vec![1, 2, 3, 4, 5];
        let sig = CheckpointSignature::from_data(&data);

        assert_eq!(sig.size, 5);
        assert_eq!(sig.data(), &data);
        assert!(sig.hash != 0);

        // Different data should have different hash
        let sig2 = CheckpointSignature::from_data(&[1, 2, 3, 4, 6]);
        assert_ne!(sig.hash, sig2.hash);
    }
}
