//! Embedding backend abstractions.
//!
//! This module defines the `EmbeddingBackend` trait which can be implemented
//! for different embedding providers (local models, APIs, etc.).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Trait for text embedding backends.
///
/// Implementations can range from local models (via candle or ONNX)
/// to remote APIs (OpenAI, Cohere, etc.).
pub trait EmbeddingBackend: Send + Sync {
    /// Compute embeddings for a batch of texts.
    ///
    /// # Arguments
    /// * `texts` - Slice of text strings to embed.
    ///
    /// # Returns
    /// Vector of embedding vectors, one per input text.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String>;

    /// Get the dimensionality of the embedding vectors.
    fn embedding_dim(&self) -> usize;

    /// Optional: Get the model name/identifier.
    fn model_name(&self) -> &str {
        "unknown"
    }
}

/// A mock embedding backend for testing.
///
/// Generates deterministic pseudo-embeddings based on text hash.
/// Identical texts produce identical embeddings.
pub struct MockBackend {
    dim: usize,
}

impl MockBackend {
    /// Create a mock backend with the specified embedding dimension.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbeddingBackend for MockBackend {
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|text| self.embed_text(text)).collect())
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "mock"
    }
}

impl MockBackend {
    /// Generate a deterministic embedding for a text.
    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate pseudo-random embedding from seed
        let mut embedding = Vec::with_capacity(self.dim);
        let mut state = seed;

        for _ in 0..self.dim {
            // LCG for deterministic pseudo-random
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
        }

        // Normalize to unit length
        normalize(&mut embedding);
        embedding
    }
}

/// Normalize a vector to unit length.
fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute cosine similarity between two vectors.
///
/// Assumes vectors are already normalized to unit length.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute euclidean distance between two vectors.
#[inline]
#[allow(dead_code)]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_deterministic() {
        let backend = MockBackend::new(384);

        let texts = &["hello world", "test"];
        let embeddings1 = backend.embed_batch(texts).unwrap();
        let embeddings2 = backend.embed_batch(texts).unwrap();

        // Same input should produce same output
        assert_eq!(embeddings1, embeddings2);
    }

    #[test]
    fn test_mock_backend_normalized() {
        let backend = MockBackend::new(384);
        let embeddings = backend.embed_batch(&["test text"]).unwrap();

        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_same_text_same_embedding() {
        let backend = MockBackend::new(128);

        let emb1 = backend.embed_batch(&["hello"]).unwrap();
        let emb2 = backend.embed_batch(&["hello"]).unwrap();

        let sim = cosine_similarity(&emb1[0], &emb2[0]);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same text should have similarity 1.0"
        );
    }

    #[test]
    fn test_different_text_different_embedding() {
        let backend = MockBackend::new(128);

        let embeddings = backend.embed_batch(&["hello", "world"]).unwrap();

        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        // Different texts should not be perfectly similar
        assert!(sim < 0.99, "Different texts should have similarity < 0.99");
    }
}
