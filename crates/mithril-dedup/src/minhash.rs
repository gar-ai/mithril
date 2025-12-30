//! MinHash signature generation for document similarity.
//!
//! MinHash is a locality-sensitive hashing technique that approximates
//! the Jaccard similarity between sets.

use mithril_core::hashing::hash_with_seed;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// Default number of permutations for MinHash signatures.
pub const DEFAULT_NUM_PERMUTATIONS: usize = 128;

/// Default n-gram size for shingling.
pub const DEFAULT_NGRAM_SIZE: usize = 5;

/// MinHash signature - a compact representation of a document's shingle set.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MinHashSignature {
    /// The minimum hash values for each permutation.
    pub values: Vec<u64>,
}

impl MinHashSignature {
    /// Create a new signature with the given values.
    #[must_use]
    pub fn new(values: Vec<u64>) -> Self {
        Self { values }
    }

    /// Get the number of permutations in this signature.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the signature is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// MinHash signature generator.
///
/// Uses multiple hash functions (simulated via seeds) to generate
/// compact signatures that preserve Jaccard similarity.
pub struct MinHasher {
    /// Number of hash permutations.
    num_permutations: usize,
    /// Seeds for each hash permutation.
    seeds: Vec<u64>,
    /// N-gram size for shingling.
    ngram_size: usize,
}

impl MinHasher {
    /// Create a new MinHasher with the specified number of permutations.
    ///
    /// Uses a fixed seed for reproducibility.
    #[must_use]
    pub fn new(num_permutations: usize) -> Self {
        Self::with_seed(num_permutations, 42)
    }

    /// Create a new MinHasher with a specific random seed.
    #[must_use]
    pub fn with_seed(num_permutations: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let seeds: Vec<u64> = (0..num_permutations).map(|_| rng.gen()).collect();

        Self {
            num_permutations,
            seeds,
            ngram_size: DEFAULT_NGRAM_SIZE,
        }
    }

    /// Set the n-gram size for shingling.
    #[must_use]
    pub fn with_ngram_size(mut self, ngram_size: usize) -> Self {
        self.ngram_size = ngram_size;
        self
    }

    /// Get the number of permutations.
    #[must_use]
    pub fn num_permutations(&self) -> usize {
        self.num_permutations
    }

    /// Get the n-gram size.
    #[must_use]
    pub fn ngram_size(&self) -> usize {
        self.ngram_size
    }

    /// Generate a MinHash signature from a set of token hashes.
    ///
    /// Each token hash is hashed with each seed, and the minimum
    /// hash value is kept for each permutation.
    #[must_use]
    pub fn signature(&self, tokens: &HashSet<u64>) -> MinHashSignature {
        if tokens.is_empty() {
            return MinHashSignature::new(vec![u64::MAX; self.num_permutations]);
        }

        let mut min_hashes = vec![u64::MAX; self.num_permutations];

        for &token in tokens {
            let token_bytes = token.to_le_bytes();
            for (i, &seed) in self.seeds.iter().enumerate() {
                let hash = hash_with_seed(&token_bytes, seed);
                min_hashes[i] = min_hashes[i].min(hash);
            }
        }

        MinHashSignature::new(min_hashes)
    }

    /// Generate a MinHash signature directly from text.
    ///
    /// This convenience method tokenizes the text into n-gram shingles
    /// and computes the signature.
    #[must_use]
    pub fn signature_from_text(&self, text: &str) -> MinHashSignature {
        let tokens = self.tokenize(text);
        self.signature(&tokens)
    }

    /// Tokenize text into n-gram shingle hashes.
    #[must_use]
    pub fn tokenize(&self, text: &str) -> HashSet<u64> {
        // Normalize: lowercase and split on whitespace
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.len() < self.ngram_size {
            // If fewer words than ngram_size, use the whole text as one shingle
            if words.is_empty() {
                return HashSet::new();
            }
            let shingle = words.join(" ");
            let hash = hash_with_seed(shingle.as_bytes(), 0);
            let mut tokens = HashSet::new();
            tokens.insert(hash);
            return tokens;
        }

        let mut tokens = HashSet::new();
        for i in 0..=words.len() - self.ngram_size {
            let shingle = words[i..i + self.ngram_size].join(" ");
            let hash = hash_with_seed(shingle.as_bytes(), 0);
            tokens.insert(hash);
        }

        tokens
    }

    /// Estimate Jaccard similarity from two MinHash signatures.
    ///
    /// The similarity is approximated by the fraction of hash values
    /// that match between the two signatures.
    #[must_use]
    pub fn similarity(sig1: &MinHashSignature, sig2: &MinHashSignature) -> f64 {
        assert_eq!(
            sig1.values.len(),
            sig2.values.len(),
            "Signatures must have the same length"
        );

        if sig1.values.is_empty() {
            return 0.0;
        }

        let matches = sig1
            .values
            .iter()
            .zip(sig2.values.iter())
            .filter(|(a, b)| a == b)
            .count();

        matches as f64 / sig1.values.len() as f64
    }
}

impl Default for MinHasher {
    fn default() -> Self {
        Self::new(DEFAULT_NUM_PERMUTATIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_identical_documents() {
        let hasher = MinHasher::new(128);
        let text = "The quick brown fox jumps over the lazy dog";

        let sig1 = hasher.signature_from_text(text);
        let sig2 = hasher.signature_from_text(text);

        let similarity = MinHasher::similarity(&sig1, &sig2);
        assert!(
            (similarity - 1.0).abs() < f64::EPSILON,
            "Identical documents should have similarity 1.0, got {similarity}"
        );
    }

    #[test]
    fn test_minhash_similar_documents() {
        let hasher = MinHasher::new(128);
        let text1 = "The quick brown fox jumps over the lazy dog in the park";
        let text2 = "The quick brown fox leaps over the lazy dog in the garden";

        let sig1 = hasher.signature_from_text(text1);
        let sig2 = hasher.signature_from_text(text2);

        let similarity = MinHasher::similarity(&sig1, &sig2);
        // These documents share some n-grams; with 5-gram shingling,
        // one word difference creates many different shingles
        assert!(
            similarity > 0.05,
            "Similar documents should have some similarity, got {similarity}"
        );
    }

    #[test]
    fn test_minhash_different_documents() {
        let hasher = MinHasher::new(128);
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "Completely different text about machine learning algorithms";

        let sig1 = hasher.signature_from_text(text1);
        let sig2 = hasher.signature_from_text(text2);

        let similarity = MinHasher::similarity(&sig1, &sig2);
        assert!(
            similarity < 0.3,
            "Different documents should have low similarity, got {similarity}"
        );
    }

    #[test]
    fn test_minhash_empty_document() {
        let hasher = MinHasher::new(128);
        let sig = hasher.signature_from_text("");

        assert_eq!(sig.len(), 128);
        // All values should be u64::MAX for empty document
        assert!(sig.values.iter().all(|&v| v == u64::MAX));
    }

    #[test]
    fn test_minhash_signature_length() {
        for num_perms in [64, 128, 256] {
            let hasher = MinHasher::new(num_perms);
            let sig = hasher.signature_from_text("Some test document");
            assert_eq!(sig.len(), num_perms);
        }
    }

    #[test]
    fn test_minhash_reproducibility() {
        let hasher1 = MinHasher::with_seed(128, 12345);
        let hasher2 = MinHasher::with_seed(128, 12345);
        let text = "Reproducibility test document";

        let sig1 = hasher1.signature_from_text(text);
        let sig2 = hasher2.signature_from_text(text);

        assert_eq!(sig1, sig2, "Same seed should produce same signature");
    }

    #[test]
    fn test_tokenize_basic() {
        let hasher = MinHasher::new(128).with_ngram_size(3);
        let text = "one two three four five";
        let tokens = hasher.tokenize(text);

        // Should have 3 trigrams: "one two three", "two three four", "three four five"
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_tokenize_short_text() {
        let hasher = MinHasher::new(128).with_ngram_size(5);
        let text = "one two three";
        let tokens = hasher.tokenize(text);

        // Text has fewer words than ngram_size, should use whole text as one shingle
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_jaccard_approximation() {
        // Test that MinHash gives a reasonable approximation of Jaccard similarity
        let hasher = MinHasher::new(256); // More permutations for better accuracy

        // Create two documents with known overlap
        let text1 = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let text2 = "word1 word2 word3 word4 word5 wordA wordB wordC wordD wordE";

        let tokens1 = hasher.tokenize(text1);
        let tokens2 = hasher.tokenize(text2);

        // Compute exact Jaccard
        let intersection: HashSet<_> = tokens1.intersection(&tokens2).collect();
        let union: HashSet<_> = tokens1.union(&tokens2).collect();
        let exact_jaccard = intersection.len() as f64 / union.len() as f64;

        // Compute MinHash approximation
        let sig1 = hasher.signature(&tokens1);
        let sig2 = hasher.signature(&tokens2);
        let minhash_similarity = MinHasher::similarity(&sig1, &sig2);

        // Should be within reasonable margin (depends on number of permutations)
        let diff = (exact_jaccard - minhash_similarity).abs();
        assert!(
            diff < 0.2,
            "MinHash should approximate Jaccard. Exact: {exact_jaccard}, MinHash: {minhash_similarity}"
        );
    }

    #[test]
    fn test_signature_from_token_set() {
        let hasher = MinHasher::new(128);
        let mut tokens = HashSet::new();
        tokens.insert(1u64);
        tokens.insert(2u64);
        tokens.insert(3u64);

        let sig = hasher.signature(&tokens);
        assert_eq!(sig.len(), 128);
        // Values should not all be u64::MAX since we have tokens
        assert!(!sig.values.iter().all(|&v| v == u64::MAX));
    }
}
