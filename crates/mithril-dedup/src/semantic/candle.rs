//! Candle-based embedding backend for native Rust inference.
//!
//! This module provides a real embedding backend using the Candle ML framework.
//! It supports sentence-transformers models like all-MiniLM-L6-v2.
//!
//! # Example
//!
//! ```no_run
//! use mithril_dedup::semantic::CandleBackend;
//!
//! let backend = CandleBackend::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();
//! let embeddings = backend.embed_batch(&["Hello world", "Goodbye world"]).unwrap();
//! ```

use super::EmbeddingBackend;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Candle-based embedding backend using sentence-transformers models.
pub struct CandleBackend {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    embedding_dim: usize,
    normalize: bool,
}

impl CandleBackend {
    /// Create a new Candle backend with the specified model.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded.
    pub fn new(model_id: &str) -> Result<Self, String> {
        Self::with_device(model_id, Device::Cpu)
    }

    /// Create a new Candle backend with a specific device.
    pub fn with_device(model_id: &str, device: Device) -> Result<Self, String> {
        // Download model files from HuggingFace
        let api = Api::new().map_err(|e| format!("Failed to create HF API: {}", e))?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .map_err(|e| format!("Failed to get config.json: {}", e))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| format!("Failed to get tokenizer.json: {}", e))?;
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| format!("Failed to get model weights: {}", e))?;

        Self::from_files(config_path, tokenizer_path, weights_path, device)
    }

    /// Create a backend from local files.
    pub fn from_files(
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        weights_path: PathBuf,
        device: Device,
    ) -> Result<Self, String> {
        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        let embedding_dim = config.hidden_size;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = if weights_path
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
                    .map_err(|e| format!("Failed to load safetensors: {}", e))?
            }
        } else {
            VarBuilder::from_pth(&weights_path, DTYPE, &device)
                .map_err(|e| format!("Failed to load pytorch weights: {}", e))?
        };

        // Build model
        let model = BertModel::load(vb, &config)
            .map_err(|e| format!("Failed to load BERT model: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_dim,
            normalize: true,
        })
    }

    /// Set whether to L2-normalize embeddings (recommended for cosine similarity).
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Encode a batch of texts into embeddings using mean pooling.
    fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build input tensors
        let mut input_ids_vec = Vec::new();
        let mut attention_mask_vec = Vec::new();
        let mut token_type_ids_vec = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            // Pad to max_len
            let mut padded_ids = ids.to_vec();
            let mut padded_attention = attention.to_vec();
            let mut padded_type_ids = type_ids.to_vec();

            padded_ids.resize(max_len, 0);
            padded_attention.resize(max_len, 0);
            padded_type_ids.resize(max_len, 0);

            input_ids_vec.extend(padded_ids);
            attention_mask_vec.extend(padded_attention);
            token_type_ids_vec.extend(padded_type_ids);
        }

        let batch_size = texts.len();

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)
            .map_err(|e| format!("Failed to create input_ids tensor: {}", e))?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {}", e))?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device)
                .map_err(|e| format!("Failed to create token_type_ids tensor: {}", e))?;

        // Forward pass
        let embeddings = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("Model forward pass failed: {}", e))?;

        // Mean pooling over sequence dimension (with attention mask)
        let attention_mask_expanded = attention_mask
            .unsqueeze(2)
            .map_err(|e| format!("Failed to expand attention mask: {}", e))?
            .to_dtype(DType::F32)
            .map_err(|e| format!("Failed to convert attention mask dtype: {}", e))?;

        let masked_embeddings = embeddings
            .broadcast_mul(&attention_mask_expanded)
            .map_err(|e| format!("Failed to apply attention mask: {}", e))?;

        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| format!("Failed to sum embeddings: {}", e))?;

        let sum_mask = attention_mask_expanded
            .sum(1)
            .map_err(|e| format!("Failed to sum mask: {}", e))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| format!("Failed to clamp mask: {}", e))?;

        let mean_embeddings = sum_embeddings
            .broadcast_div(&sum_mask)
            .map_err(|e| format!("Failed to compute mean: {}", e))?;

        // L2 normalize if requested
        let final_embeddings = if self.normalize {
            let norms = mean_embeddings
                .sqr()
                .map_err(|e| format!("Failed to square: {}", e))?
                .sum_keepdim(1)
                .map_err(|e| format!("Failed to sum for norm: {}", e))?
                .sqrt()
                .map_err(|e| format!("Failed to sqrt: {}", e))?
                .clamp(1e-12, f64::MAX)
                .map_err(|e| format!("Failed to clamp norm: {}", e))?;

            mean_embeddings
                .broadcast_div(&norms)
                .map_err(|e| format!("Failed to normalize: {}", e))?
        } else {
            mean_embeddings
        };

        // Convert to Vec<Vec<f32>>
        let flat: Vec<f32> = final_embeddings
            .to_vec2()
            .map_err(|e| format!("Failed to convert to vec: {}", e))?
            .into_iter()
            .flatten()
            .collect();

        // Reshape into individual embeddings
        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * self.embedding_dim;
            let end = start + self.embedding_dim;
            result.push(flat[start..end].to_vec());
        }

        Ok(result)
    }
}

impl EmbeddingBackend for CandleBackend {
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.encode(texts)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require network access and model downloads.
    // Run with: cargo test --features candle -- --ignored

    #[test]
    #[ignore = "Requires model download"]
    fn test_candle_backend_creation() {
        let backend = CandleBackend::new("sentence-transformers/all-MiniLM-L6-v2");
        assert!(backend.is_ok());
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_candle_embed_single() {
        let backend = CandleBackend::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        let embeddings = backend.embed_batch(&["Hello world"]).unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), backend.embedding_dim());
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_candle_embed_batch() {
        let backend = CandleBackend::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        let texts = vec!["Hello world", "Goodbye world", "How are you?"];
        let embeddings = backend.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), backend.embedding_dim());
        }
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_candle_similar_texts() {
        use super::super::cosine_similarity;

        let backend = CandleBackend::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();

        let texts = vec![
            "The cat sat on the mat",
            "A feline was resting on the rug",
            "Machine learning is transforming AI",
        ];

        let embeddings = backend.embed_batch(&texts).unwrap();

        // Cat texts should be more similar to each other than to ML text
        let sim_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
        let sim_02 = cosine_similarity(&embeddings[0], &embeddings[2]);

        assert!(
            sim_01 > sim_02,
            "Similar texts should have higher similarity: {} vs {}",
            sim_01,
            sim_02
        );
    }
}
