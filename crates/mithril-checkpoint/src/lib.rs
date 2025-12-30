//! # mithril-checkpoint
//!
//! Checkpoint compression for PyTorch models.
//!
//! Achieves 10x+ lossless compression on bf16/fp16 model weights through:
//! 1. **Byte grouping** - Separates high/low bytes for better locality
//! 2. **Zstd compression** - High-ratio dictionary-based compression
//!
//! ## Quick Start
//!
//! ```rust
//! use mithril_checkpoint::pipeline::{CheckpointCompressor, CompressionConfig};
//! use mithril_core::types::DType;
//!
//! // Create compressor with default settings
//! let compressor = CheckpointCompressor::default();
//!
//! // Compress bf16 tensor data
//! let tensor_data: Vec<u8> = vec![0u8; 10000]; // Your tensor bytes
//! let compressed = compressor.compress(&tensor_data, DType::BFloat16).unwrap();
//!
//! // Decompress
//! let decompressed = compressor.decompress(
//!     &compressed,
//!     DType::BFloat16,
//!     tensor_data.len()
//! ).unwrap();
//!
//! assert_eq!(tensor_data, decompressed);
//! ```
//!
//! ## Modules
//!
//! - [`bytegroup`] - Byte grouping transforms for floating-point data
//! - [`delta`] - Delta encoding for consecutive checkpoints
//! - [`pipeline`] - Main compression pipeline
//! - [`formats`] - Checkpoint format readers (safetensors, etc.)
//!
//! ## Status: IN PROGRESS
//!
//! See `STATUS.md` for current progress.

pub mod bytegroup;
pub mod compressed_safetensors;
pub mod delta;
pub mod formats;
pub mod gguf;
pub mod onnx;
pub mod orbax;
pub mod pipeline;
pub mod quantize;

// Re-export main types
pub use compressed_safetensors::{CompressedSafetensors, MstHeader, MstReader, MstWriter};
pub use delta::{CheckpointSignature, DeltaCompressor, DeltaEncoder, DeltaStats};
pub use formats::{SafetensorsHeader, SafetensorsReader, SafetensorsWriter, TensorData};
pub use gguf::{
    is_gguf, GgufDType, GgufMetadata, GgufReader, GgufTensorInfo, GgufValue, GgufWriteStats,
    GgufWriter,
};
pub use onnx::{
    is_onnx, OnnxDataType, OnnxInitializer, OnnxMetadata, OnnxReader, OnnxWriteStats, OnnxWriter,
};
pub use orbax::{
    is_orbax_checkpoint, ArrayInfo, OrbaxCheckpoint, OrbaxMetadata, OrbaxReader, OrbaxWriteStats,
    OrbaxWriter,
};
pub use pipeline::{CheckpointCompressor, CompressionConfig, CompressionStats};
pub use quantize::{
    QuantizeConfig, QuantizeError, QuantizeMethod, QuantizeStats, QuantizedTensor, Quantizer,
};
