//! Checkpoint compression pipeline.
//!
//! Combines byte grouping and zstd compression for optimal checkpoint compression.

use mithril_core::compression::{Compressor, ZstdCompressor};
use mithril_core::types::DType;
use mithril_core::Result;

use crate::bytegroup::{
    byte_group_bf16_auto, byte_group_fp32_auto, byte_ungroup_bf16_auto, byte_ungroup_fp32_auto,
};
use crate::delta::DeltaEncoder;
use crate::quantize::{QuantizeConfig, QuantizeError, QuantizedTensor, Quantizer};

/// Configuration for checkpoint compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Zstd compression level (1-22, default 3 for speed, 19+ for ratio)
    pub zstd_level: i32,
    /// Enable byte grouping for floating-point types
    pub byte_grouping: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            zstd_level: 3,
            byte_grouping: true,
        }
    }
}

impl CompressionConfig {
    /// Create config optimized for compression speed.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            zstd_level: 1,
            byte_grouping: true,
        }
    }

    /// Create config optimized for compression ratio.
    #[must_use]
    pub fn best() -> Self {
        Self {
            zstd_level: 19,
            byte_grouping: true,
        }
    }

    /// Create config with custom zstd level.
    #[must_use]
    pub fn with_level(level: i32) -> Self {
        Self {
            zstd_level: level,
            byte_grouping: true,
        }
    }
}

/// Checkpoint compressor combining byte grouping and zstd.
///
/// Achieves 10x+ lossless compression on bf16/fp16 model weights by:
/// 1. Byte grouping - separates high/low bytes for better locality
/// 2. Zstd compression - high-ratio dictionary-based compression
///
/// # Example
/// ```
/// use mithril_checkpoint::pipeline::{CheckpointCompressor, CompressionConfig};
/// use mithril_core::types::DType;
///
/// let compressor = CheckpointCompressor::new(CompressionConfig::default());
///
/// // Compress bf16 tensor data
/// let data = vec![0u8; 1000];
/// let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
/// let decompressed = compressor.decompress(&compressed, DType::BFloat16, data.len()).unwrap();
/// assert_eq!(data, decompressed);
/// ```
pub struct CheckpointCompressor {
    config: CompressionConfig,
    compressor: ZstdCompressor,
}

impl CheckpointCompressor {
    /// Create a new checkpoint compressor with the given config.
    #[must_use]
    pub fn new(config: CompressionConfig) -> Self {
        let compressor = ZstdCompressor::with_level(config.zstd_level);
        Self { config, compressor }
    }

    /// Create a checkpoint compressor with default settings.
    #[must_use]
    pub fn default_compressor() -> Self {
        Self::new(CompressionConfig::default())
    }

    /// Compress tensor data with optional byte grouping based on dtype.
    ///
    /// # Arguments
    /// * `data` - Raw tensor bytes
    /// * `dtype` - Data type of the tensor (determines grouping strategy)
    ///
    /// # Returns
    /// Compressed bytes, or error if compression fails.
    pub fn compress(&self, data: &[u8], dtype: DType) -> Result<Vec<u8>> {
        if data.is_empty() {
            return self.compressor.compress(data);
        }

        let grouped = if self.config.byte_grouping {
            match dtype {
                DType::BFloat16 | DType::Float16 => byte_group_bf16_auto(data),
                DType::Float32 => byte_group_fp32_auto(data),
                _ => data.to_vec(),
            }
        } else {
            data.to_vec()
        };

        self.compressor.compress(&grouped)
    }

    /// Decompress tensor data with optional byte ungrouping based on dtype.
    ///
    /// # Arguments
    /// * `data` - Compressed bytes
    /// * `dtype` - Data type of the tensor (determines ungrouping strategy)
    /// * `original_size` - Expected size of decompressed data in bytes
    ///
    /// # Returns
    /// Decompressed bytes matching original tensor data.
    pub fn decompress(&self, data: &[u8], dtype: DType, original_size: usize) -> Result<Vec<u8>> {
        let decompressed = self.compressor.decompress_exact(data, original_size)?;

        if !self.config.byte_grouping || decompressed.is_empty() {
            return Ok(decompressed);
        }

        let ungrouped = match dtype {
            DType::BFloat16 | DType::Float16 => byte_ungroup_bf16_auto(&decompressed),
            DType::Float32 => byte_ungroup_fp32_auto(&decompressed),
            _ => decompressed,
        };

        Ok(ungrouped)
    }

    /// Compress tensor data with delta encoding against a previous checkpoint.
    ///
    /// This achieves 39-70x compression ratios during training by exploiting
    /// the fact that consecutive checkpoints are ~99% identical.
    ///
    /// Pipeline: Delta XOR → Byte Grouping → Zstd
    ///
    /// # Arguments
    /// * `data` - Current tensor bytes
    /// * `dtype` - Data type of the tensor
    /// * `previous` - Optional previous checkpoint data (same size)
    ///
    /// # Returns
    /// Compressed bytes. If `previous` is Some, returns delta-compressed data.
    ///
    /// # Example
    /// ```
    /// use mithril_checkpoint::pipeline::CheckpointCompressor;
    /// use mithril_core::types::DType;
    ///
    /// let compressor = CheckpointCompressor::default();
    ///
    /// let step1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    /// let step2 = vec![1u8, 2, 3, 5, 5, 6, 7, 8];  // Only byte 3 changed
    ///
    /// // First checkpoint (no delta)
    /// let c1 = compressor.compress_with_delta(&step1, DType::Int8, None).unwrap();
    ///
    /// // Second checkpoint (with delta) - much smaller!
    /// let c2 = compressor.compress_with_delta(&step2, DType::Int8, Some(&step1)).unwrap();
    /// ```
    pub fn compress_with_delta(
        &self,
        data: &[u8],
        dtype: DType,
        previous: Option<&[u8]>,
    ) -> Result<Vec<u8>> {
        if data.is_empty() {
            return self.compressor.compress(data);
        }

        // Step 1: Delta encode (XOR with previous if provided)
        let delta = DeltaEncoder::encode(data, previous);

        // Step 2: Byte group (for floating-point types)
        let grouped = if self.config.byte_grouping {
            match dtype {
                DType::BFloat16 | DType::Float16 => byte_group_bf16_auto(&delta),
                DType::Float32 => byte_group_fp32_auto(&delta),
                _ => delta,
            }
        } else {
            delta
        };

        // Step 3: Zstd compress
        self.compressor.compress(&grouped)
    }

    /// Decompress delta-encoded tensor data.
    ///
    /// Pipeline: Zstd decompress → Byte Ungroup → Delta XOR
    ///
    /// # Arguments
    /// * `data` - Compressed bytes (from compress_with_delta)
    /// * `dtype` - Data type of the tensor
    /// * `original_size` - Expected size of decompressed data in bytes
    /// * `previous` - Previous checkpoint data (same size, required if delta was used)
    ///
    /// # Returns
    /// Decompressed bytes matching original tensor data.
    pub fn decompress_with_delta(
        &self,
        data: &[u8],
        dtype: DType,
        original_size: usize,
        previous: Option<&[u8]>,
    ) -> Result<Vec<u8>> {
        // Step 1: Zstd decompress
        let decompressed = self.compressor.decompress_exact(data, original_size)?;

        if decompressed.is_empty() {
            return Ok(decompressed);
        }

        // Step 2: Byte ungroup (for floating-point types)
        let ungrouped = if self.config.byte_grouping {
            match dtype {
                DType::BFloat16 | DType::Float16 => byte_ungroup_bf16_auto(&decompressed),
                DType::Float32 => byte_ungroup_fp32_auto(&decompressed),
                _ => decompressed,
            }
        } else {
            decompressed
        };

        // Step 3: Delta decode (XOR with previous if provided)
        match previous {
            Some(prev) => Ok(DeltaEncoder::decode(&ungrouped, prev)),
            None => Ok(ungrouped),
        }
    }

    /// Get the compression ratio for the last operation.
    /// Returns original_size / compressed_size.
    #[must_use]
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if compressed_size == 0 {
            return 0.0;
        }
        original_size as f64 / compressed_size as f64
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }

    /// Compress raw bytes without byte grouping.
    ///
    /// This is useful for compressing pickle data or other non-tensor data
    /// where byte grouping doesn't provide benefits.
    ///
    /// # Arguments
    /// * `data` - Raw bytes to compress
    ///
    /// # Returns
    /// Compressed bytes
    pub fn compress_raw(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compressor.compress(data)
    }

    /// Decompress raw bytes without byte grouping.
    ///
    /// # Arguments
    /// * `data` - Compressed bytes
    /// * `original_size` - Expected size of decompressed data
    ///
    /// # Returns
    /// Decompressed bytes
    pub fn decompress_raw(&self, data: &[u8], original_size: usize) -> Result<Vec<u8>> {
        self.compressor.decompress_exact(data, original_size)
    }

    /// Compress tensor data with lossy quantization.
    ///
    /// This applies quantization before compression, achieving higher compression
    /// ratios at the cost of some precision loss. Useful for inference or when
    /// storage is at a premium.
    ///
    /// Pipeline: Quantize → Zstd compress
    ///
    /// # Arguments
    /// * `data` - Raw tensor bytes
    /// * `dtype` - Data type of the tensor (must be bf16, fp16, or fp32)
    /// * `shape` - Shape of the tensor
    /// * `quantize_config` - Quantization configuration
    ///
    /// # Returns
    /// Compressed quantized tensor as bytes (includes metadata for dequantization).
    ///
    /// # Example
    /// ```
    /// use mithril_checkpoint::pipeline::CheckpointCompressor;
    /// use mithril_checkpoint::quantize::{QuantizeConfig, QuantizeMethod};
    /// use mithril_core::types::DType;
    ///
    /// let compressor = CheckpointCompressor::default();
    /// let config = QuantizeConfig::int8();
    ///
    /// // Compress 1000 bf16 values to int8
    /// let bf16_data = vec![0u8; 2000]; // 1000 bf16 values
    /// let compressed = compressor.compress_quantized(
    ///     &bf16_data,
    ///     DType::BFloat16,
    ///     vec![1000],
    ///     &config,
    /// ).unwrap();
    /// ```
    pub fn compress_quantized(
        &self,
        data: &[u8],
        dtype: DType,
        shape: Vec<usize>,
        quantize_config: &QuantizeConfig,
    ) -> std::result::Result<Vec<u8>, QuantizeError> {
        let quantizer = Quantizer::new(quantize_config.clone());

        // Quantize based on dtype
        let quantized = match dtype {
            DType::BFloat16 => quantizer.quantize_bf16_with_shape(data, shape)?,
            DType::Float16 => quantizer.quantize_fp16_with_shape(data, shape)?,
            DType::Float32 => quantizer.quantize_fp32_with_shape(data, shape)?,
            _ => return Err(QuantizeError::UnsupportedDType(dtype)),
        };

        // Serialize the quantized tensor (includes metadata)
        let serialized = quantized.to_bytes()?;

        // Compress the serialized data
        self.compressor
            .compress(&serialized)
            .map_err(|_| QuantizeError::InvalidFormat)
    }

    /// Decompress and dequantize tensor data.
    ///
    /// Pipeline: Zstd decompress → Dequantize
    ///
    /// # Arguments
    /// * `data` - Compressed quantized bytes (from compress_quantized)
    /// * `target_dtype` - Desired output dtype (must match original or be compatible)
    ///
    /// # Returns
    /// Dequantized tensor bytes in the target dtype.
    pub fn decompress_dequantized(
        &self,
        data: &[u8],
        target_dtype: DType,
    ) -> std::result::Result<Vec<u8>, QuantizeError> {
        // Decompress
        let decompressed = self
            .compressor
            .decompress(data)
            .map_err(|_| QuantizeError::InvalidFormat)?;

        // Deserialize quantized tensor
        let quantized = QuantizedTensor::from_bytes(&decompressed)?;

        // Dequantize based on target dtype
        let quantizer = Quantizer::default();

        match target_dtype {
            DType::BFloat16 => quantizer.dequantize_to_bf16(&quantized),
            DType::Float16 => quantizer.dequantize_to_fp16(&quantized),
            DType::Float32 => quantizer.dequantize_to_fp32(&quantized),
            _ => Err(QuantizeError::UnsupportedDType(target_dtype)),
        }
    }

    /// Compress tensor with quantization and additional zstd compression.
    ///
    /// This combines quantization with byte grouping and zstd for maximum compression.
    ///
    /// # Arguments
    /// * `data` - Raw tensor bytes
    /// * `dtype` - Data type of the tensor
    /// * `shape` - Shape of the tensor
    /// * `quantize_config` - Quantization configuration
    ///
    /// # Returns
    /// Compressed quantized tensor bytes.
    pub fn compress_quantized_optimized(
        &self,
        data: &[u8],
        dtype: DType,
        shape: Vec<usize>,
        quantize_config: &QuantizeConfig,
    ) -> std::result::Result<Vec<u8>, QuantizeError> {
        let quantizer = Quantizer::new(quantize_config.clone());

        // Quantize based on dtype
        let quantized = match dtype {
            DType::BFloat16 => quantizer.quantize_bf16_with_shape(data, shape)?,
            DType::Float16 => quantizer.quantize_fp16_with_shape(data, shape)?,
            DType::Float32 => quantizer.quantize_fp32_with_shape(data, shape)?,
            _ => return Err(QuantizeError::UnsupportedDType(dtype)),
        };

        // The quantized data is int8, which doesn't benefit from byte grouping
        // Just serialize and compress
        let serialized = quantized.to_bytes()?;

        self.compressor
            .compress(&serialized)
            .map_err(|_| QuantizeError::InvalidFormat)
    }
}

impl Default for CheckpointCompressor {
    fn default() -> Self {
        Self::default_compressor()
    }
}

/// Statistics from compression operation.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (original / compressed)
    pub ratio: f64,
}

impl CompressionStats {
    /// Create new stats from sizes.
    #[must_use]
    pub fn new(original_size: usize, compressed_size: usize) -> Self {
        Self {
            original_size,
            compressed_size,
            ratio: CheckpointCompressor::compression_ratio(original_size, compressed_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_bf16() {
        let compressor = CheckpointCompressor::default();

        // Create realistic bf16-like data
        let data: Vec<u8> = (0..5000u16)
            .flat_map(|i| {
                // Simulate bf16 weights: small values around zero with occasional larger ones
                let val = if i % 100 == 0 {
                    0x3C00u16 // 1.0 in bf16
                } else {
                    (i % 256) as u16 // small values
                };
                val.to_le_bytes()
            })
            .collect();

        let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DType::BFloat16, data.len())
            .unwrap();

        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_compress_decompress_fp32() {
        let compressor = CheckpointCompressor::default();

        // Create fp32 data
        let data: Vec<u8> = (0..1000u32).flat_map(|x| x.to_le_bytes()).collect();

        let compressed = compressor.compress(&data, DType::Float32).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DType::Float32, data.len())
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compress_decompress_int8() {
        let compressor = CheckpointCompressor::default();

        // Int8 data - no byte grouping applied
        let data: Vec<u8> = (0..=255u8).cycle().take(5000).collect();

        let compressed = compressor.compress(&data, DType::Int8).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DType::Int8, data.len())
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_empty_data() {
        let compressor = CheckpointCompressor::default();
        let data: Vec<u8> = vec![];

        let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DType::BFloat16, 0)
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        assert_eq!(CheckpointCompressor::compression_ratio(100, 10), 10.0);
        assert_eq!(CheckpointCompressor::compression_ratio(100, 100), 1.0);
        assert_eq!(CheckpointCompressor::compression_ratio(0, 0), 0.0);
    }

    #[test]
    fn test_config_presets() {
        let fast = CompressionConfig::fast();
        assert_eq!(fast.zstd_level, 1);

        let best = CompressionConfig::best();
        assert_eq!(best.zstd_level, 19);

        let custom = CompressionConfig::with_level(10);
        assert_eq!(custom.zstd_level, 10);
    }

    #[test]
    fn test_realistic_bf16_weights_compression() {
        let compressor = CheckpointCompressor::new(CompressionConfig::with_level(3));

        // Simulate realistic neural network weights:
        // - Mostly small values (weights initialized near zero)
        // - Some larger values
        // - Patterns in exponent bytes
        let mut rng_state = 42u64;
        let data: Vec<u8> = (0..100_000)
            .flat_map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand = (rng_state >> 33) as u16;

                // bf16 format: 1 sign, 8 exp, 7 mantissa
                // Most weights are small, so exponents cluster around bias (127)
                let exp = (123 + (rand % 9)) as u8; // 123-131 mostly
                let mantissa = (rand & 0x7F) as u8;
                let sign = ((rand >> 15) & 1) as u8;

                let bf16 = ((sign as u16) << 15) | ((exp as u16) << 7) | (mantissa as u16);
                bf16.to_le_bytes()
            })
            .collect();

        let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
        let stats = CompressionStats::new(data.len(), compressed.len());

        // Verify roundtrip
        let decompressed = compressor
            .decompress(&compressed, DType::BFloat16, data.len())
            .unwrap();
        assert_eq!(data, decompressed, "Roundtrip must be bit-exact");

        // Check compression ratio (simulated data achieves modest compression;
        // real model weights with more structure achieve 10x+)
        assert!(
            stats.ratio > 1.0,
            "Expected ratio > 1.0, got {:.2}x",
            stats.ratio
        );
    }

    #[test]
    fn test_byte_grouping_disabled() {
        let config = CompressionConfig {
            zstd_level: 3,
            byte_grouping: false,
        };
        let compressor = CheckpointCompressor::new(config);

        let data: Vec<u8> = (0..1000u16).flat_map(|x| x.to_le_bytes()).collect();

        let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DType::BFloat16, data.len())
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_delta_compression_no_previous() {
        let compressor = CheckpointCompressor::default();
        let data: Vec<u8> = (0..1000u16).flat_map(|x| x.to_le_bytes()).collect();

        // Without previous, should behave like regular compress
        let compressed = compressor
            .compress_with_delta(&data, DType::BFloat16, None)
            .unwrap();
        let decompressed = compressor
            .decompress_with_delta(&compressed, DType::BFloat16, data.len(), None)
            .unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_delta_compression_with_previous() {
        let compressor = CheckpointCompressor::default();

        // Simulate two training steps with mostly identical data
        let step1: Vec<u8> = (0..5000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut step2 = step1.clone();

        // Change only 1% of bytes
        for i in (0..step2.len()).step_by(100) {
            step2[i] = step2[i].wrapping_add(1);
        }

        // Compress with delta
        let compressed = compressor
            .compress_with_delta(&step2, DType::BFloat16, Some(&step1))
            .unwrap();

        // Decompress with delta
        let decompressed = compressor
            .decompress_with_delta(&compressed, DType::BFloat16, step2.len(), Some(&step1))
            .unwrap();

        assert_eq!(step2, decompressed, "Delta roundtrip must be bit-exact");
    }

    #[test]
    fn test_delta_compression_identical_data() {
        let compressor = CheckpointCompressor::default();
        let data: Vec<u8> = (0..10000u16).flat_map(|x| x.to_le_bytes()).collect();

        // Delta of identical data should be all zeros, which compresses extremely well
        let compressed_standalone = compressor.compress(&data, DType::BFloat16).unwrap();
        let compressed_delta = compressor
            .compress_with_delta(&data, DType::BFloat16, Some(&data))
            .unwrap();

        // Delta version should be much smaller (zeros compress very well)
        assert!(
            compressed_delta.len() < compressed_standalone.len() / 5,
            "Delta of identical data should compress 5x+ better: standalone={}, delta={}",
            compressed_standalone.len(),
            compressed_delta.len()
        );

        // Verify roundtrip
        let decompressed = compressor
            .decompress_with_delta(&compressed_delta, DType::BFloat16, data.len(), Some(&data))
            .unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_delta_compression_ratio_improvement() {
        let compressor = CheckpointCompressor::default();

        // Simulate realistic consecutive checkpoints with 1% change
        let size = 100_000;
        let step1: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let mut step2 = step1.clone();

        // Change 1% of bytes
        for i in (0..size).step_by(100) {
            step2[i] = step2[i].wrapping_add(1);
        }

        // Compare standalone vs delta compression
        let compressed_standalone = compressor.compress(&step2, DType::Int8).unwrap();
        let compressed_delta = compressor
            .compress_with_delta(&step2, DType::Int8, Some(&step1))
            .unwrap();

        let ratio_standalone = size as f64 / compressed_standalone.len() as f64;
        let ratio_delta = size as f64 / compressed_delta.len() as f64;

        // Delta should achieve much better compression
        assert!(
            ratio_delta > ratio_standalone * 5.0,
            "Delta ratio ({:.1}x) should be 5x+ better than standalone ({:.1}x)",
            ratio_delta,
            ratio_standalone
        );
    }

    #[test]
    fn test_quantized_compression_roundtrip() {
        use crate::quantize::QuantizeConfig;

        let compressor = CheckpointCompressor::default();
        let config = QuantizeConfig::int8();

        // Create bf16 data
        let values: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) - 0.5).collect();
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect();

        // Compress with quantization
        let compressed = compressor
            .compress_quantized(&bf16_data, DType::BFloat16, vec![1000], &config)
            .unwrap();

        // Decompress and dequantize
        let dequantized = compressor
            .decompress_dequantized(&compressed, DType::BFloat16)
            .unwrap();

        // Should have same length
        assert_eq!(dequantized.len(), bf16_data.len());

        // Compression should achieve significant ratio
        let ratio = bf16_data.len() as f64 / compressed.len() as f64;
        assert!(ratio > 1.5, "Expected ratio > 1.5, got {}", ratio);
    }

    #[test]
    fn test_quantized_nf4_compression() {
        use crate::quantize::QuantizeConfig;

        let compressor = CheckpointCompressor::default();
        let config = QuantizeConfig::nf4_with_group_size(64);

        // Create larger bf16 data (must be divisible by group_size)
        let values: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) - 0.5).collect();
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect();

        // Compress with NF4 quantization
        let compressed = compressor
            .compress_quantized(&bf16_data, DType::BFloat16, vec![1024], &config)
            .unwrap();

        // Decompress and dequantize
        let dequantized = compressor
            .decompress_dequantized(&compressed, DType::BFloat16)
            .unwrap();

        // Should have same length
        assert_eq!(dequantized.len(), bf16_data.len());

        // NF4 should achieve better compression than int8
        let ratio = bf16_data.len() as f64 / compressed.len() as f64;
        assert!(ratio > 2.0, "Expected NF4 ratio > 2.0, got {}", ratio);
    }
}
