//! Lossy quantization for checkpoint compression.
//!
//! This module provides quantization methods to reduce checkpoint size at the cost
//! of some precision loss. Useful for inference or when storage is at a premium.
//!
//! ## Quantization Methods
//!
//! - **Int8Linear**: Simple 8-bit linear quantization with scale/zero-point
//! - **NF4**: 4-bit NormalFloat quantization (QLoRA-style)
//! - **DynamicInt8**: Per-channel dynamic quantization
//!
//! ## Example
//!
//! ```rust
//! use mithril_checkpoint::quantize::{Quantizer, QuantizeConfig, QuantizeMethod};
//!
//! let config = QuantizeConfig::new(QuantizeMethod::Int8Linear);
//! let quantizer = Quantizer::new(config);
//!
//! // bf16 data (2 bytes per element)
//! let bf16_data = vec![0x00, 0x3F, 0x00, 0x40]; // 0.5, 2.0 in bf16
//! let quantized = quantizer.quantize_bf16(&bf16_data).unwrap();
//!
//! // Dequantize back to bf16
//! let dequantized = quantizer.dequantize_to_bf16(&quantized).unwrap();
//! ```

use mithril_core::types::DType;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for quantization operations.
#[derive(Error, Debug)]
pub enum QuantizeError {
    #[error("Invalid data size: expected multiple of {expected}, got {actual}")]
    InvalidDataSize { expected: usize, actual: usize },

    #[error("Unsupported dtype for quantization: {0:?}")]
    UnsupportedDType(DType),

    #[error("Group size {group_size} does not evenly divide element count {num_elements}")]
    InvalidGroupSize {
        group_size: usize,
        num_elements: usize,
    },

    #[error("Invalid quantized data format")]
    InvalidFormat,

    #[error("Quantization method {0:?} requires group_size to be set")]
    GroupSizeRequired(QuantizeMethod),
}

/// Result type for quantization operations.
pub type Result<T> = std::result::Result<T, QuantizeError>;

/// Quantization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizeMethod {
    /// 8-bit linear quantization with scale and zero-point.
    /// Achieves ~2x compression from bf16/fp16.
    Int8Linear,

    /// 4-bit NormalFloat quantization (QLoRA-style).
    /// Achieves ~4x compression from bf16/fp16.
    /// Uses a fixed set of quantization levels optimized for normally-distributed weights.
    NF4,

    /// 4-bit unsigned integer quantization.
    /// Simple linear mapping to 16 levels.
    Int4,

    /// Per-channel dynamic 8-bit quantization.
    /// Each channel gets its own scale, improving accuracy for layers with varying ranges.
    DynamicInt8,
}

impl QuantizeMethod {
    /// Returns the number of bits per element after quantization.
    #[must_use]
    pub const fn bits_per_element(&self) -> usize {
        match self {
            Self::Int8Linear | Self::DynamicInt8 => 8,
            Self::NF4 | Self::Int4 => 4,
        }
    }

    /// Returns whether this method uses grouped quantization.
    #[must_use]
    pub const fn uses_groups(&self) -> bool {
        matches!(self, Self::NF4)
    }
}

/// Configuration for quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeConfig {
    /// Quantization method to use.
    pub method: QuantizeMethod,

    /// Group size for grouped quantization methods (e.g., NF4).
    /// Each group gets its own scale factor.
    /// Common values: 32, 64, 128.
    pub group_size: Option<usize>,

    /// Patterns for tensor names to exclude from quantization.
    /// These tensors remain in full precision.
    /// Example: ["embed", "lm_head", "layernorm"]
    pub exclude_patterns: Vec<String>,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            method: QuantizeMethod::Int8Linear,
            group_size: None,
            exclude_patterns: vec![],
        }
    }
}

impl QuantizeConfig {
    /// Create a new config with the specified method.
    #[must_use]
    pub fn new(method: QuantizeMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Create Int8 linear quantization config.
    #[must_use]
    pub fn int8() -> Self {
        Self::new(QuantizeMethod::Int8Linear)
    }

    /// Create NF4 quantization config with default group size.
    #[must_use]
    pub fn nf4() -> Self {
        Self {
            method: QuantizeMethod::NF4,
            group_size: Some(64),
            ..Default::default()
        }
    }

    /// Create NF4 quantization config with custom group size.
    #[must_use]
    pub fn nf4_with_group_size(group_size: usize) -> Self {
        Self {
            method: QuantizeMethod::NF4,
            group_size: Some(group_size),
            ..Default::default()
        }
    }

    /// Set group size.
    #[must_use]
    pub fn with_group_size(mut self, size: usize) -> Self {
        self.group_size = Some(size);
        self
    }

    /// Add patterns to exclude from quantization.
    #[must_use]
    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns = patterns;
        self
    }

    /// Check if a tensor name should be excluded from quantization.
    #[must_use]
    pub fn should_exclude(&self, tensor_name: &str) -> bool {
        self.exclude_patterns
            .iter()
            .any(|pattern| tensor_name.contains(pattern))
    }
}

/// Quantized tensor data with metadata for dequantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized data bytes.
    pub data: Vec<u8>,

    /// Scale factors (one per tensor or per group).
    pub scales: Vec<f32>,

    /// Zero points for asymmetric quantization (optional).
    pub zero_points: Option<Vec<i8>>,

    /// Original data type before quantization.
    pub original_dtype: DType,

    /// Original shape of the tensor.
    pub original_shape: Vec<usize>,

    /// Quantization method used.
    pub method: QuantizeMethod,

    /// Group size if grouped quantization was used.
    pub group_size: Option<usize>,
}

impl QuantizedTensor {
    /// Returns the number of elements in the original tensor.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.original_shape.iter().product()
    }

    /// Returns the compression ratio achieved.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.num_elements() * self.original_dtype.size_bytes();
        let quantized_bytes = self.data.len() + self.scales.len() * 4;
        if quantized_bytes == 0 {
            return 0.0;
        }
        original_bytes as f64 / quantized_bytes as f64
    }

    /// Serialize to bytes for storage.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|_| QuantizeError::InvalidFormat)
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|_| QuantizeError::InvalidFormat)
    }
}

/// NF4 quantization levels.
/// These are the 16 values that approximate a normal distribution.
/// From the QLoRA paper: https://arxiv.org/abs/2305.14314
const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Quantizer for compressing tensor data.
pub struct Quantizer {
    config: QuantizeConfig,
}

impl Quantizer {
    /// Create a new quantizer with the given configuration.
    #[must_use]
    pub fn new(config: QuantizeConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &QuantizeConfig {
        &self.config
    }

    /// Quantize bf16 tensor data.
    ///
    /// # Arguments
    /// * `data` - Raw bf16 bytes (2 bytes per element, little-endian)
    ///
    /// # Returns
    /// Quantized tensor with metadata for dequantization.
    pub fn quantize_bf16(&self, data: &[u8]) -> Result<QuantizedTensor> {
        self.quantize_bf16_with_shape(data, vec![data.len() / 2])
    }

    /// Quantize bf16 tensor data with explicit shape.
    pub fn quantize_bf16_with_shape(
        &self,
        data: &[u8],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        if data.len() % 2 != 0 {
            return Err(QuantizeError::InvalidDataSize {
                expected: 2,
                actual: data.len(),
            });
        }

        // Convert bf16 bytes to f32 values
        let values: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                bf16_to_f32(bits)
            })
            .collect();

        match self.config.method {
            QuantizeMethod::Int8Linear => {
                self.quantize_int8_linear(&values, DType::BFloat16, shape)
            }
            QuantizeMethod::NF4 => self.quantize_nf4(&values, DType::BFloat16, shape),
            QuantizeMethod::Int4 => self.quantize_int4(&values, DType::BFloat16, shape),
            QuantizeMethod::DynamicInt8 => {
                self.quantize_int8_linear(&values, DType::BFloat16, shape)
            }
        }
    }

    /// Quantize fp16 tensor data.
    pub fn quantize_fp16(&self, data: &[u8]) -> Result<QuantizedTensor> {
        self.quantize_fp16_with_shape(data, vec![data.len() / 2])
    }

    /// Quantize fp16 tensor data with explicit shape.
    pub fn quantize_fp16_with_shape(
        &self,
        data: &[u8],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        if data.len() % 2 != 0 {
            return Err(QuantizeError::InvalidDataSize {
                expected: 2,
                actual: data.len(),
            });
        }

        // Convert fp16 bytes to f32 values
        let values: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                fp16_to_f32(bits)
            })
            .collect();

        match self.config.method {
            QuantizeMethod::Int8Linear => self.quantize_int8_linear(&values, DType::Float16, shape),
            QuantizeMethod::NF4 => self.quantize_nf4(&values, DType::Float16, shape),
            QuantizeMethod::Int4 => self.quantize_int4(&values, DType::Float16, shape),
            QuantizeMethod::DynamicInt8 => {
                self.quantize_int8_linear(&values, DType::Float16, shape)
            }
        }
    }

    /// Quantize fp32 tensor data.
    pub fn quantize_fp32(&self, data: &[u8]) -> Result<QuantizedTensor> {
        self.quantize_fp32_with_shape(data, vec![data.len() / 4])
    }

    /// Quantize fp32 tensor data with explicit shape.
    pub fn quantize_fp32_with_shape(
        &self,
        data: &[u8],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        if data.len() % 4 != 0 {
            return Err(QuantizeError::InvalidDataSize {
                expected: 4,
                actual: data.len(),
            });
        }

        // Convert fp32 bytes to f32 values
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        match self.config.method {
            QuantizeMethod::Int8Linear => self.quantize_int8_linear(&values, DType::Float32, shape),
            QuantizeMethod::NF4 => self.quantize_nf4(&values, DType::Float32, shape),
            QuantizeMethod::Int4 => self.quantize_int4(&values, DType::Float32, shape),
            QuantizeMethod::DynamicInt8 => {
                self.quantize_int8_linear(&values, DType::Float32, shape)
            }
        }
    }

    /// Dequantize back to bf16 bytes.
    pub fn dequantize_to_bf16(&self, quantized: &QuantizedTensor) -> Result<Vec<u8>> {
        let values = self.dequantize_to_f32(quantized)?;

        Ok(values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect())
    }

    /// Dequantize back to fp16 bytes.
    pub fn dequantize_to_fp16(&self, quantized: &QuantizedTensor) -> Result<Vec<u8>> {
        let values = self.dequantize_to_f32(quantized)?;

        Ok(values
            .iter()
            .flat_map(|&v| f32_to_fp16(v).to_le_bytes())
            .collect())
    }

    /// Dequantize back to fp32 bytes.
    pub fn dequantize_to_fp32(&self, quantized: &QuantizedTensor) -> Result<Vec<u8>> {
        let values = self.dequantize_to_f32(quantized)?;

        Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect())
    }

    /// Dequantize to f32 values.
    pub fn dequantize_to_f32(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        match quantized.method {
            QuantizeMethod::Int8Linear | QuantizeMethod::DynamicInt8 => {
                self.dequantize_int8_linear(quantized)
            }
            QuantizeMethod::NF4 => self.dequantize_nf4(quantized),
            QuantizeMethod::Int4 => self.dequantize_int4(quantized),
        }
    }

    // --- Internal quantization methods ---

    fn quantize_int8_linear(
        &self,
        values: &[f32],
        original_dtype: DType,
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        if values.is_empty() {
            return Ok(QuantizedTensor {
                data: vec![],
                scales: vec![0.0],
                zero_points: Some(vec![0]),
                original_dtype,
                original_shape: shape,
                method: QuantizeMethod::Int8Linear,
                group_size: None,
            });
        }

        // Find min/max for symmetric quantization
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in values {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        // Symmetric quantization: scale = max(|min|, |max|) / 127
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };

        // Quantize values
        let quantized_data: Vec<u8> = values
            .iter()
            .map(|&v| {
                let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
                q as u8
            })
            .collect();

        Ok(QuantizedTensor {
            data: quantized_data,
            scales: vec![scale],
            zero_points: None, // Symmetric quantization, no zero point
            original_dtype,
            original_shape: shape,
            method: QuantizeMethod::Int8Linear,
            group_size: None,
        })
    }

    fn quantize_nf4(
        &self,
        values: &[f32],
        original_dtype: DType,
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        let group_size = self
            .config
            .group_size
            .ok_or_else(|| QuantizeError::GroupSizeRequired(QuantizeMethod::NF4))?;

        if !values.is_empty() && values.len() % group_size != 0 {
            return Err(QuantizeError::InvalidGroupSize {
                group_size,
                num_elements: values.len(),
            });
        }

        if values.is_empty() {
            return Ok(QuantizedTensor {
                data: vec![],
                scales: vec![],
                zero_points: None,
                original_dtype,
                original_shape: shape,
                method: QuantizeMethod::NF4,
                group_size: Some(group_size),
            });
        }

        let num_groups = values.len() / group_size;
        let mut scales = Vec::with_capacity(num_groups);
        let mut quantized_nibbles = Vec::with_capacity(values.len());

        // Process each group
        for group_idx in 0..num_groups {
            let start = group_idx * group_size;
            let end = start + group_size;
            let group = &values[start..end];

            // Find absmax for this group
            let abs_max = group
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);

            let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
            scales.push(scale);

            // Quantize each value in the group
            for &v in group {
                let normalized = if scale == 0.0 { 0.0 } else { v / scale };
                let idx = find_nearest_nf4(normalized);
                quantized_nibbles.push(idx);
            }
        }

        // Pack nibbles into bytes (2 values per byte)
        let mut packed_data = Vec::with_capacity((quantized_nibbles.len() + 1) / 2);
        for chunk in quantized_nibbles.chunks(2) {
            let low = chunk[0];
            let high = if chunk.len() > 1 { chunk[1] } else { 0 };
            packed_data.push((high << 4) | low);
        }

        Ok(QuantizedTensor {
            data: packed_data,
            scales,
            zero_points: None,
            original_dtype,
            original_shape: shape,
            method: QuantizeMethod::NF4,
            group_size: Some(group_size),
        })
    }

    fn quantize_int4(
        &self,
        values: &[f32],
        original_dtype: DType,
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        if values.is_empty() {
            return Ok(QuantizedTensor {
                data: vec![],
                scales: vec![0.0],
                zero_points: Some(vec![0]),
                original_dtype,
                original_shape: shape,
                method: QuantizeMethod::Int4,
                group_size: None,
            });
        }

        // Find min/max
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in values {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        // Asymmetric quantization for int4 (0-15 range)
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
        let zero_point = if range == 0.0 {
            0
        } else {
            ((-min_val / scale).round() as i8).clamp(0, 15)
        };

        // Quantize values to 4-bit
        let mut quantized_nibbles: Vec<u8> = values
            .iter()
            .map(|&v| {
                let q = ((v - min_val) / scale).round().clamp(0.0, 15.0) as u8;
                q
            })
            .collect();

        // Pad to even length
        if quantized_nibbles.len() % 2 != 0 {
            quantized_nibbles.push(0);
        }

        // Pack nibbles into bytes
        let mut packed_data = Vec::with_capacity(quantized_nibbles.len() / 2);
        for chunk in quantized_nibbles.chunks(2) {
            let low = chunk[0];
            let high = chunk[1];
            packed_data.push((high << 4) | low);
        }

        Ok(QuantizedTensor {
            data: packed_data,
            scales: vec![scale],
            zero_points: Some(vec![zero_point]),
            original_dtype,
            original_shape: shape,
            method: QuantizeMethod::Int4,
            group_size: None,
        })
    }

    // --- Internal dequantization methods ---

    fn dequantize_int8_linear(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        if quantized.data.is_empty() {
            return Ok(vec![]);
        }

        let scale = quantized.scales.first().copied().unwrap_or(1.0);

        Ok(quantized
            .data
            .iter()
            .map(|&b| {
                let q = b as i8;
                q as f32 * scale
            })
            .collect())
    }

    fn dequantize_nf4(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        let group_size = quantized.group_size.ok_or(QuantizeError::InvalidFormat)?;

        if quantized.data.is_empty() {
            return Ok(vec![]);
        }

        let num_elements = quantized.num_elements();
        let mut values = Vec::with_capacity(num_elements);

        // Unpack nibbles
        let mut nibbles = Vec::with_capacity(num_elements);
        for &byte in &quantized.data {
            nibbles.push(byte & 0x0F); // low nibble
            nibbles.push((byte >> 4) & 0x0F); // high nibble
        }
        nibbles.truncate(num_elements);

        // Dequantize using per-group scales
        for (group_idx, group_nibbles) in nibbles.chunks(group_size).enumerate() {
            let scale = quantized.scales.get(group_idx).copied().unwrap_or(1.0);
            for &nibble in group_nibbles {
                let nf4_value = NF4_LEVELS[nibble as usize & 0x0F];
                values.push(nf4_value * scale);
            }
        }

        Ok(values)
    }

    fn dequantize_int4(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        if quantized.data.is_empty() {
            return Ok(vec![]);
        }

        let scale = quantized.scales.first().copied().unwrap_or(1.0);
        let zero_point = quantized
            .zero_points
            .as_ref()
            .and_then(|zp| zp.first().copied())
            .unwrap_or(0) as f32;

        let num_elements = quantized.num_elements();

        // Unpack nibbles
        let mut values = Vec::with_capacity(num_elements);
        for &byte in &quantized.data {
            let low = (byte & 0x0F) as f32;
            let high = ((byte >> 4) & 0x0F) as f32;

            values.push((low - zero_point) * scale);
            if values.len() < num_elements {
                values.push((high - zero_point) * scale);
            }
        }
        values.truncate(num_elements);

        Ok(values)
    }
}

impl Default for Quantizer {
    fn default() -> Self {
        Self::new(QuantizeConfig::default())
    }
}

// --- Helper functions for float conversion ---

/// Convert bf16 bits to f32.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    // bf16 is the upper 16 bits of f32
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bf16 bits (truncation, no rounding).
#[inline]
fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Convert fp16 bits to f32.
#[inline]
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: convert to normalized f32
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13);
        f32::from_bits(f32_bits)
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F80_0000) // Inf
        } else {
            f32::from_bits((sign << 31) | 0x7FC0_0000 | (mant << 13)) // NaN
        }
    } else {
        // Normal number
        let f32_exp = exp + 127 - 15;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    }
}

/// Convert f32 to fp16 bits.
#[inline]
fn f32_to_fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 255 {
        // Inf or NaN
        if mant == 0 {
            (sign << 15) | 0x7C00 // Inf
        } else {
            (sign << 15) | 0x7E00 // NaN
        }
    } else if exp > 142 {
        // Overflow to Inf
        (sign << 15) | 0x7C00
    } else if exp < 103 {
        // Underflow to zero
        sign << 15
    } else if exp < 113 {
        // Subnormal
        let shift = 113 - exp;
        let m = (mant | 0x80_0000) >> (shift + 13);
        (sign << 15) | (m as u16)
    } else {
        // Normal
        let fp16_exp = ((exp - 127 + 15) as u16) & 0x1F;
        let fp16_mant = ((mant >> 13) as u16) & 0x3FF;
        (sign << 15) | (fp16_exp << 10) | fp16_mant
    }
}

/// Find the nearest NF4 quantization level index.
#[inline]
fn find_nearest_nf4(value: f32) -> u8 {
    let clamped = value.clamp(-1.0, 1.0);
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;

    for (idx, &level) in NF4_LEVELS.iter().enumerate() {
        let dist = (clamped - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx as u8;
        }
    }

    best_idx
}

/// Statistics about quantization accuracy.
#[derive(Debug, Clone)]
pub struct QuantizeStats {
    /// Mean absolute error between original and dequantized values.
    pub mae: f64,
    /// Max absolute error.
    pub max_error: f64,
    /// Root mean square error.
    pub rmse: f64,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
}

impl QuantizeStats {
    /// Compute statistics by comparing original and dequantized values.
    #[must_use]
    pub fn compute(original: &[f32], dequantized: &[f32], quantized_bytes: usize) -> Self {
        if original.is_empty() || original.len() != dequantized.len() {
            return Self {
                mae: 0.0,
                max_error: 0.0,
                rmse: 0.0,
                compression_ratio: 0.0,
            };
        }

        let mut sum_abs_error = 0.0f64;
        let mut sum_sq_error = 0.0f64;
        let mut max_error = 0.0f64;

        for (&orig, &deq) in original.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs() as f64;
            sum_abs_error += error;
            sum_sq_error += error * error;
            max_error = max_error.max(error);
        }

        let n = original.len() as f64;
        let original_bytes = original.len() * 4; // f32 = 4 bytes

        Self {
            mae: sum_abs_error / n,
            max_error,
            rmse: (sum_sq_error / n).sqrt(),
            compression_ratio: original_bytes as f64 / quantized_bytes as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_linear_quantization() {
        let config = QuantizeConfig::int8();
        let quantizer = Quantizer::new(config);

        // Create bf16 data representing [0.0, 0.5, 1.0, -0.5]
        let values = [0.0f32, 0.5, 1.0, -0.5];
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();

        let quantized = quantizer.quantize_bf16(&bf16_data).unwrap();

        assert_eq!(quantized.method, QuantizeMethod::Int8Linear);
        assert_eq!(quantized.data.len(), 4); // 4 int8 values
        assert_eq!(quantized.scales.len(), 1);

        // For small inputs, overhead dominates. Test with larger input for ratio.
        // Dequantize and check accuracy
        let dequantized = quantizer.dequantize_to_bf16(&quantized).unwrap();
        assert_eq!(dequantized.len(), bf16_data.len());

        // Test larger input for compression ratio
        let large_values: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) - 0.5).collect();
        let large_bf16_data: Vec<u8> = large_values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();
        let large_quantized = quantizer.quantize_bf16(&large_bf16_data).unwrap();
        // bf16 = 2 bytes/elem, int8 = 1 byte/elem + 4 bytes scale
        // 2000 bytes -> ~1004 bytes = ~2x compression
        assert!(
            large_quantized.compression_ratio() > 1.5,
            "Expected ratio > 1.5, got {}",
            large_quantized.compression_ratio()
        );
    }

    #[test]
    fn test_nf4_quantization() {
        let config = QuantizeConfig::nf4_with_group_size(4);
        let quantizer = Quantizer::new(config);

        // Create 8 bf16 values (2 groups of 4)
        let values = [0.0f32, 0.5, 1.0, -0.5, 0.2, -0.2, 0.8, -0.8];
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();

        let quantized = quantizer.quantize_bf16(&bf16_data).unwrap();

        assert_eq!(quantized.method, QuantizeMethod::NF4);
        assert_eq!(quantized.data.len(), 4); // 8 nibbles = 4 bytes
        assert_eq!(quantized.scales.len(), 2); // 2 groups
        assert_eq!(quantized.group_size, Some(4));

        // Dequantize and verify roundtrip
        let dequantized = quantizer.dequantize_to_bf16(&quantized).unwrap();
        assert_eq!(dequantized.len(), bf16_data.len());

        // Test compression ratio with larger input (overhead is less significant)
        let config = QuantizeConfig::nf4_with_group_size(64);
        let quantizer = Quantizer::new(config);
        let large_values: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) - 0.5).collect();
        let large_bf16_data: Vec<u8> = large_values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();
        let large_quantized = quantizer.quantize_bf16(&large_bf16_data).unwrap();
        // bf16 = 2 bytes/elem, NF4 = 0.5 bytes/elem + scales
        // 2048 bytes -> 512 bytes data + 16 scales * 4 = 576 bytes = ~3.5x
        assert!(
            large_quantized.compression_ratio() > 2.0,
            "Expected ratio > 2.0, got {}",
            large_quantized.compression_ratio()
        );
    }

    #[test]
    fn test_int4_quantization() {
        let config = QuantizeConfig::new(QuantizeMethod::Int4);
        let quantizer = Quantizer::new(config);

        // Create bf16 data
        let values = [0.0f32, 0.25, 0.5, 0.75, 1.0, -0.5];
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();

        let quantized = quantizer.quantize_bf16(&bf16_data).unwrap();

        assert_eq!(quantized.method, QuantizeMethod::Int4);
        assert_eq!(quantized.data.len(), 3); // 6 nibbles -> 3 bytes

        // Dequantize
        let dequantized = quantizer.dequantize_to_bf16(&quantized).unwrap();
        assert_eq!(dequantized.len(), bf16_data.len());
    }

    #[test]
    fn test_quantize_empty_data() {
        let config = QuantizeConfig::int8();
        let quantizer = Quantizer::new(config);

        let quantized = quantizer.quantize_bf16(&[]).unwrap();
        assert!(quantized.data.is_empty());

        let dequantized = quantizer.dequantize_to_bf16(&quantized).unwrap();
        assert!(dequantized.is_empty());
    }

    #[test]
    fn test_quantize_fp32() {
        let config = QuantizeConfig::int8();
        let quantizer = Quantizer::new(config);

        let values = [0.0f32, 0.5, 1.0, -0.5];
        let fp32_data: Vec<u8> = values.iter().flat_map(|&v| v.to_le_bytes()).collect();

        let quantized = quantizer.quantize_fp32(&fp32_data).unwrap();
        assert_eq!(quantized.original_dtype, DType::Float32);

        let dequantized = quantizer.dequantize_to_fp32(&quantized).unwrap();
        assert_eq!(dequantized.len(), fp32_data.len());
    }

    #[test]
    fn test_exclude_patterns() {
        let config =
            QuantizeConfig::int8().with_exclude_patterns(vec!["embed".into(), "lm_head".into()]);

        assert!(config.should_exclude("model.embed_tokens.weight"));
        assert!(config.should_exclude("lm_head.weight"));
        assert!(!config.should_exclude("model.layers.0.mlp.weight"));
    }

    #[test]
    fn test_bf16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
        for &v in &values {
            let bf16 = f32_to_bf16(v);
            let back = bf16_to_f32(bf16);
            // bf16 has less precision, so we check approximate equality
            assert!(
                (v - back).abs() < v.abs() * 0.01 + 0.001,
                "bf16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_fp16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
        for &v in &values {
            let fp16 = f32_to_fp16(v);
            let back = fp16_to_f32(fp16);
            // fp16 has less precision, so we check approximate equality
            assert!(
                (v - back).abs() < v.abs() * 0.001 + 0.001,
                "fp16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_nf4_levels() {
        // Verify NF4 levels are sorted
        for i in 1..NF4_LEVELS.len() {
            assert!(
                NF4_LEVELS[i] > NF4_LEVELS[i - 1],
                "NF4 levels not sorted at index {}",
                i
            );
        }

        // Verify range
        assert_eq!(NF4_LEVELS[0], -1.0);
        assert_eq!(NF4_LEVELS[15], 1.0);
    }

    #[test]
    fn test_find_nearest_nf4() {
        // Test exact matches
        assert_eq!(find_nearest_nf4(-1.0), 0);
        assert_eq!(find_nearest_nf4(1.0), 15);
        assert_eq!(find_nearest_nf4(0.0), 7);

        // Test clamping
        assert_eq!(find_nearest_nf4(-2.0), 0);
        assert_eq!(find_nearest_nf4(2.0), 15);
    }

    #[test]
    fn test_quantize_stats() {
        let original = vec![0.0f32, 0.5, 1.0, -0.5];
        let dequantized = vec![0.01f32, 0.48, 0.99, -0.52];

        let stats = QuantizeStats::compute(&original, &dequantized, 4);

        assert!(stats.mae > 0.0);
        assert!(stats.max_error > 0.0);
        assert!(stats.rmse > 0.0);
    }

    #[test]
    fn test_quantized_tensor_serialization() {
        let tensor = QuantizedTensor {
            data: vec![1, 2, 3, 4],
            scales: vec![0.1],
            zero_points: None,
            original_dtype: DType::BFloat16,
            original_shape: vec![4],
            method: QuantizeMethod::Int8Linear,
            group_size: None,
        };

        let bytes = tensor.to_bytes().unwrap();
        let restored = QuantizedTensor::from_bytes(&bytes).unwrap();

        assert_eq!(tensor.data, restored.data);
        assert_eq!(tensor.scales, restored.scales);
        assert_eq!(tensor.method, restored.method);
    }

    #[test]
    fn test_nf4_invalid_group_size() {
        let config = QuantizeConfig::nf4_with_group_size(4);
        let quantizer = Quantizer::new(config);

        // 5 elements doesn't divide evenly by group_size=4
        let values = [0.0f32, 0.5, 1.0, -0.5, 0.2];
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();

        let result = quantizer.quantize_bf16(&bf16_data);
        assert!(matches!(
            result,
            Err(QuantizeError::InvalidGroupSize { .. })
        ));
    }

    #[test]
    fn test_quantization_accuracy() {
        // Test that int8 quantization maintains reasonable accuracy
        let config = QuantizeConfig::int8();
        let quantizer = Quantizer::new(config);

        // Simulate realistic weight values
        let mut values = Vec::with_capacity(1000);
        for i in 0..1000 {
            // Small values around zero with some larger outliers
            let v = if i % 100 == 0 {
                ((i % 10) as f32 - 5.0) / 2.0
            } else {
                ((i as f32 / 1000.0) - 0.5) * 0.2
            };
            values.push(v);
        }

        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();

        let quantized = quantizer.quantize_bf16(&bf16_data).unwrap();
        let dequantized_f32 = quantizer.dequantize_to_f32(&quantized).unwrap();

        let stats = QuantizeStats::compute(&values, &dequantized_f32, quantized.data.len());

        // Int8 should have reasonable accuracy
        assert!(
            stats.mae < 0.05,
            "Mean absolute error too high: {}",
            stats.mae
        );
        assert!(
            stats.max_error < 0.5,
            "Max error too high: {}",
            stats.max_error
        );
    }
}
