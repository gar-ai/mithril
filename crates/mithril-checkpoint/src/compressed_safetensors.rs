//! Compressed Safetensors format (`.mst` - Mithril Safetensors).
//!
//! This module provides a compressed variant of the safetensors format that uses
//! Mithril's byte grouping and zstd compression to achieve 10x+ compression ratios
//! on model weights while maintaining compatibility with standard safetensors readers
//! after decompression.
//!
//! ## Format Specification
//!
//! ```text
//! MST File Format:
//! ┌─────────────────────────────────────────┐
//! │ Magic: "MST\x00" (4 bytes)              │
//! ├─────────────────────────────────────────┤
//! │ Version: u32 (4 bytes, little-endian)   │
//! ├─────────────────────────────────────────┤
//! │ Header Size: u64 (8 bytes, LE)          │
//! ├─────────────────────────────────────────┤
//! │ JSON Header (compressed metadata)       │
//! │ - tensor names, shapes, dtypes          │
//! │ - compression settings                  │
//! │ - offsets into compressed data          │
//! ├─────────────────────────────────────────┤
//! │ Compressed Tensor Data                  │
//! │ - byte-grouped + zstd compressed        │
//! │ - stored per-tensor or concatenated     │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use mithril_checkpoint::compressed_safetensors::{MstWriter, MstReader};
//! use mithril_core::types::DType;
//!
//! // Write compressed safetensors
//! let mut writer = MstWriter::new();
//! writer.add_tensor("model.weight", tensor_data, DType::BFloat16, vec![512, 768]);
//! writer.write_file("model.mst")?;
//!
//! // Read compressed safetensors
//! let mut reader = MstReader::open("model.mst")?;
//! let data = reader.read_tensor("model.weight")?;
//! ```

use mithril_core::types::DType;
use mithril_core::{MithrilError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::pipeline::{CheckpointCompressor, CompressionConfig};

/// Magic bytes for MST format: "MST\0"
const MST_MAGIC: [u8; 4] = [b'M', b'S', b'T', 0];

/// Current MST format version
const MST_VERSION: u32 = 1;

/// Header for compressed safetensors format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MstHeader {
    /// Tensor metadata indexed by name
    pub tensors: HashMap<String, MstTensorInfo>,
    /// Compression configuration used
    pub compression: MstCompressionInfo,
    /// Optional user metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Total uncompressed size of all tensors
    pub total_uncompressed_size: usize,
    /// Total compressed size of all tensors
    pub total_compressed_size: usize,
}

/// Tensor metadata for MST format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MstTensorInfo {
    /// Data type of the tensor
    pub dtype: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Offset in compressed data section
    pub compressed_offset: usize,
    /// Size in compressed data section
    pub compressed_size: usize,
    /// Original uncompressed size
    pub uncompressed_size: usize,
}

/// Compression settings stored in header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MstCompressionInfo {
    /// Compression algorithm ("zstd")
    pub algorithm: String,
    /// Zstd compression level
    pub level: i32,
    /// Whether byte grouping was used
    pub byte_grouping: bool,
}

/// High-level interface for working with compressed safetensors.
pub struct CompressedSafetensors;

impl CompressedSafetensors {
    /// Compress a standard safetensors file to MST format.
    ///
    /// This reads a `.safetensors` file and writes a compressed `.mst` file.
    pub fn compress_file<P: AsRef<Path>>(
        input: P,
        output: P,
        config: CompressionConfig,
    ) -> Result<CompressionStats> {
        use crate::formats::SafetensorsReader;

        let mut reader = SafetensorsReader::open(input.as_ref())?;
        let header = reader.header().clone();

        let mut writer = MstWriter::with_config(config);

        for name in header.tensors.keys() {
            let meta = header.tensors.get(name).unwrap();
            let data = reader.read_tensor(name)?;
            let dtype = Self::dtype_to_string(meta.dtype);
            writer.add_tensor_bytes(name, data, &dtype, meta.shape.clone());
        }

        writer.write_file(output)?;

        Ok(CompressionStats {
            original_size: header.data_size,
            compressed_size: writer.compressed_size(),
            tensor_count: header.tensors.len(),
        })
    }

    /// Decompress an MST file to standard safetensors format.
    ///
    /// This reads a `.mst` file and writes a standard `.safetensors` file.
    pub fn decompress_file<P: AsRef<Path>>(input: P, output: P) -> Result<DecompressionStats> {
        use crate::formats::SafetensorsWriter;

        let mut reader = MstReader::open(input.as_ref())?;
        let header = reader.header().clone();

        let mut writer = SafetensorsWriter::new();

        // Copy metadata
        for (key, value) in &header.metadata {
            writer.add_metadata(key, value);
        }

        // Decompress and add each tensor
        for (name, info) in &header.tensors {
            let data = reader.read_tensor(name)?;
            let dtype = Self::parse_dtype(&info.dtype)?;
            writer.add_tensor(name, data, dtype, info.shape.clone());
        }

        writer.write_file(output)?;

        Ok(DecompressionStats {
            compressed_size: header.total_compressed_size,
            decompressed_size: header.total_uncompressed_size,
            tensor_count: header.tensors.len(),
        })
    }

    fn dtype_to_string(dtype: DType) -> String {
        match dtype {
            DType::Float32 => "F32",
            DType::Float16 => "F16",
            DType::BFloat16 => "BF16",
            DType::Float64 => "F64",
            DType::Int32 => "I32",
            DType::Int64 => "I64",
            DType::Int8 => "I8",
            DType::UInt8 => "U8",
            DType::Bool => "BOOL",
        }
        .to_string()
    }

    fn parse_dtype(dtype: &str) -> Result<DType> {
        match dtype {
            "F32" | "float32" => Ok(DType::Float32),
            "F16" | "float16" => Ok(DType::Float16),
            "BF16" | "bfloat16" => Ok(DType::BFloat16),
            "F64" | "float64" => Ok(DType::Float64),
            "I32" | "int32" => Ok(DType::Int32),
            "I64" | "int64" => Ok(DType::Int64),
            "I8" | "int8" => Ok(DType::Int8),
            "U8" | "uint8" => Ok(DType::UInt8),
            "BOOL" | "bool" => Ok(DType::Bool),
            _ => Err(MithrilError::InvalidFormat(format!(
                "Unknown dtype: {}",
                dtype
            ))),
        }
    }
}

/// Statistics from compression operation.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original uncompressed size
    pub original_size: usize,
    /// Final compressed size
    pub compressed_size: usize,
    /// Number of tensors processed
    pub tensor_count: usize,
}

impl CompressionStats {
    /// Get compression ratio (original / compressed).
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            0.0
        } else {
            self.original_size as f64 / self.compressed_size as f64
        }
    }
}

/// Statistics from decompression operation.
#[derive(Debug, Clone)]
pub struct DecompressionStats {
    /// Compressed size read
    pub compressed_size: usize,
    /// Decompressed size written
    pub decompressed_size: usize,
    /// Number of tensors processed
    pub tensor_count: usize,
}

/// Internal tensor info for writer.
#[derive(Debug, Clone)]
struct TensorWriteInfo {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    uncompressed_size: usize,
    compressed_data: Vec<u8>,
}

/// Writer for compressed safetensors (MST) format.
pub struct MstWriter {
    tensors: Vec<TensorWriteInfo>,
    metadata: HashMap<String, String>,
    compressor: CheckpointCompressor,
    config: CompressionConfig,
}

impl MstWriter {
    /// Create a new MST writer with default compression settings.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CompressionConfig::default())
    }

    /// Create a new MST writer with custom compression settings.
    #[must_use]
    pub fn with_config(config: CompressionConfig) -> Self {
        let compressor = CheckpointCompressor::new(config.clone());
        Self {
            tensors: Vec::new(),
            metadata: HashMap::new(),
            compressor,
            config,
        }
    }

    /// Add optional metadata.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Add a tensor to be written.
    ///
    /// # Arguments
    /// * `name` - Tensor name
    /// * `data` - Raw tensor bytes
    /// * `dtype` - Data type as DType enum
    /// * `shape` - Tensor shape
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        dtype: DType,
        shape: Vec<usize>,
    ) {
        let dtype_str = match dtype {
            DType::Float32 => "F32",
            DType::Float16 => "F16",
            DType::BFloat16 => "BF16",
            DType::Float64 => "F64",
            DType::Int32 => "I32",
            DType::Int64 => "I64",
            DType::Int8 => "I8",
            DType::UInt8 => "U8",
            DType::Bool => "BOOL",
        };
        self.add_tensor_bytes(name, data, dtype_str, shape);
    }

    /// Add a tensor with dtype as string.
    pub fn add_tensor_bytes(
        &mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        dtype: &str,
        shape: Vec<usize>,
    ) {
        let dtype_enum = match dtype {
            "F32" | "float32" => DType::Float32,
            "F16" | "float16" => DType::Float16,
            "BF16" | "bfloat16" => DType::BFloat16,
            "F64" | "float64" => DType::Float64,
            "I32" | "int32" => DType::Int32,
            "I64" | "int64" => DType::Int64,
            "I8" | "int8" => DType::Int8,
            "U8" | "uint8" => DType::UInt8,
            "BOOL" | "bool" => DType::Bool,
            _ => DType::UInt8, // Default fallback
        };

        let uncompressed_size = data.len();
        let compressed_data = self
            .compressor
            .compress(&data, dtype_enum)
            .unwrap_or_else(|_| data.clone());

        self.tensors.push(TensorWriteInfo {
            name: name.into(),
            dtype: dtype.to_string(),
            shape,
            uncompressed_size,
            compressed_data,
        });
    }

    /// Get total compressed size of all tensors.
    #[must_use]
    pub fn compressed_size(&self) -> usize {
        self.tensors.iter().map(|t| t.compressed_data.len()).sum()
    }

    /// Get total uncompressed size of all tensors.
    #[must_use]
    pub fn uncompressed_size(&self) -> usize {
        self.tensors.iter().map(|t| t.uncompressed_size).sum()
    }

    /// Get number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Write to the given writer.
    pub fn write<W: Write>(&self, writer: W) -> Result<()> {
        let mut writer = BufWriter::new(writer);

        // Build header with offsets
        let mut tensor_infos = HashMap::new();
        let mut current_offset = 0usize;

        for tensor in &self.tensors {
            tensor_infos.insert(
                tensor.name.clone(),
                MstTensorInfo {
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    compressed_offset: current_offset,
                    compressed_size: tensor.compressed_data.len(),
                    uncompressed_size: tensor.uncompressed_size,
                },
            );
            current_offset += tensor.compressed_data.len();
        }

        let header = MstHeader {
            tensors: tensor_infos,
            compression: MstCompressionInfo {
                algorithm: "zstd".to_string(),
                level: self.config.zstd_level,
                byte_grouping: self.config.byte_grouping,
            },
            metadata: self.metadata.clone(),
            total_uncompressed_size: self.uncompressed_size(),
            total_compressed_size: self.compressed_size(),
        };

        // Serialize header to JSON
        let header_json = serde_json::to_string(&header).map_err(|e| {
            MithrilError::InvalidFormat(format!("Failed to serialize header: {}", e))
        })?;
        let header_bytes = header_json.as_bytes();

        // Write magic
        writer.write_all(&MST_MAGIC)?;

        // Write version
        writer.write_all(&MST_VERSION.to_le_bytes())?;

        // Write header size
        let header_size = header_bytes.len() as u64;
        writer.write_all(&header_size.to_le_bytes())?;

        // Write header
        writer.write_all(header_bytes)?;

        // Write compressed tensor data
        for tensor in &self.tensors {
            writer.write_all(&tensor.compressed_data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Write to a file.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        self.write(file)
    }
}

impl Default for MstWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader for compressed safetensors (MST) format.
pub struct MstReader {
    file: BufReader<File>,
    header: MstHeader,
    data_offset: u64,
    compressor: CheckpointCompressor,
}

impl MstReader {
    /// Open an MST file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MST_MAGIC {
            return Err(MithrilError::InvalidFormat(
                "Invalid MST magic bytes".to_string(),
            ));
        }

        // Read version
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version > MST_VERSION {
            return Err(MithrilError::InvalidFormat(format!(
                "Unsupported MST version: {} (max supported: {})",
                version, MST_VERSION
            )));
        }

        // Read header size
        let mut header_size_buf = [0u8; 8];
        reader.read_exact(&mut header_size_buf)?;
        let header_size = u64::from_le_bytes(header_size_buf) as usize;

        // Read header JSON
        let mut header_buf = vec![0u8; header_size];
        reader.read_exact(&mut header_buf)?;

        let header: MstHeader = serde_json::from_slice(&header_buf)
            .map_err(|e| MithrilError::InvalidFormat(format!("Invalid header JSON: {}", e)))?;

        // Calculate data offset
        let data_offset = 4 + 4 + 8 + header_size as u64;

        // Create compressor matching the header settings
        let config = CompressionConfig {
            zstd_level: header.compression.level,
            byte_grouping: header.compression.byte_grouping,
        };
        let compressor = CheckpointCompressor::new(config);

        Ok(Self {
            file: reader,
            header,
            data_offset,
            compressor,
        })
    }

    /// Get the parsed header.
    #[must_use]
    pub fn header(&self) -> &MstHeader {
        &self.header
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.keys().map(|s| s.as_str())
    }

    /// Get info for a specific tensor.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&MstTensorInfo> {
        self.header.tensors.get(name)
    }

    /// Read and decompress a tensor by name.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| MithrilError::NotFound(name.to_string()))?
            .clone();

        self.read_tensor_data(&info)
    }

    /// Read and decompress tensor using info.
    pub fn read_tensor_data(&mut self, info: &MstTensorInfo) -> Result<Vec<u8>> {
        // Seek to compressed data
        let offset = self.data_offset + info.compressed_offset as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        // Read compressed data
        let mut compressed = vec![0u8; info.compressed_size];
        self.file.read_exact(&mut compressed)?;

        // Decompress
        let dtype = CompressedSafetensors::parse_dtype(&info.dtype)?;
        self.compressor
            .decompress(&compressed, dtype, info.uncompressed_size)
    }

    /// Read all tensors into a map.
    pub fn read_all(&mut self) -> Result<HashMap<String, Vec<u8>>> {
        let names: Vec<String> = self.header.tensors.keys().cloned().collect();
        let mut result = HashMap::new();

        for name in names {
            let data = self.read_tensor(&name)?;
            result.insert(name, data);
        }

        Ok(result)
    }
}

/// Check if a file appears to be in MST format.
pub fn is_mst<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();

    // Check extension
    if let Some(ext) = path.extension() {
        if ext == "mst" {
            return true;
        }
    }

    // Try to read magic bytes
    if let Ok(mut file) = File::open(path) {
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() && magic == MST_MAGIC {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mst_roundtrip_single_tensor() {
        let mut writer = MstWriter::new();

        // Create test tensor data
        let original_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        writer.add_tensor("weight", original_data.clone(), DType::BFloat16, vec![500]);

        // Write to temp file
        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        // Read back
        let mut reader = MstReader::open(temp_file.path()).unwrap();

        // Verify header
        assert_eq!(reader.header().tensors.len(), 1);

        // Verify tensor info
        let info = reader.tensor_info("weight").unwrap();
        assert_eq!(info.dtype, "BF16");
        assert_eq!(info.shape, vec![500]);
        assert_eq!(info.uncompressed_size, 1000);

        // Verify data
        let data = reader.read_tensor("weight").unwrap();
        assert_eq!(data, original_data);
    }

    #[test]
    fn test_mst_roundtrip_multiple_tensors() {
        let mut writer = MstWriter::new();

        let tensor1: Vec<u8> = (0..200).map(|i| i as u8).collect();
        let tensor2: Vec<u8> = (0..400).map(|i| (i * 2) as u8).collect();
        let tensor3: Vec<u8> = (0..100).map(|i| (i * 3) as u8).collect();

        writer.add_tensor("encoder.embed", tensor1.clone(), DType::Int8, vec![200]);
        writer.add_tensor("decoder.weight", tensor2.clone(), DType::Float32, vec![100]);
        writer.add_tensor("output.bias", tensor3.clone(), DType::Float16, vec![50]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        let mut reader = MstReader::open(temp_file.path()).unwrap();

        assert_eq!(reader.header().tensors.len(), 3);

        assert_eq!(reader.read_tensor("encoder.embed").unwrap(), tensor1);
        assert_eq!(reader.read_tensor("decoder.weight").unwrap(), tensor2);
        assert_eq!(reader.read_tensor("output.bias").unwrap(), tensor3);
    }

    #[test]
    fn test_mst_with_metadata() {
        let mut writer = MstWriter::new();
        writer.add_metadata("format", "pytorch");
        writer.add_metadata("model_name", "test_model");

        let data: Vec<u8> = vec![0; 100];
        writer.add_tensor("weight", data, DType::Float32, vec![25]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        let reader = MstReader::open(temp_file.path()).unwrap();
        assert_eq!(
            reader.header().metadata.get("format"),
            Some(&"pytorch".to_string())
        );
        assert_eq!(
            reader.header().metadata.get("model_name"),
            Some(&"test_model".to_string())
        );
    }

    #[test]
    fn test_mst_compression_stats() {
        let mut writer = MstWriter::new();

        // Create compressible data (repeated patterns)
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        writer.add_tensor("weight", data.clone(), DType::BFloat16, vec![5000]);

        assert_eq!(writer.tensor_count(), 1);
        assert_eq!(writer.uncompressed_size(), 10000);

        // Compressed size should be smaller
        assert!(writer.compressed_size() < 10000);
    }

    #[test]
    fn test_mst_empty_tensor() {
        let mut writer = MstWriter::new();
        writer.add_tensor("empty", Vec::new(), DType::Float32, vec![0]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        let mut reader = MstReader::open(temp_file.path()).unwrap();
        let info = reader.tensor_info("empty").unwrap();
        assert_eq!(info.shape, vec![0]);
        assert_eq!(info.uncompressed_size, 0);

        let data = reader.read_tensor("empty").unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_is_mst_by_extension() {
        assert!(is_mst(Path::new("model.mst")));
        assert!(!is_mst(Path::new("model.safetensors")));
        assert!(!is_mst(Path::new("model.pt")));
    }

    #[test]
    fn test_is_mst_by_magic() {
        let temp_file = NamedTempFile::new().unwrap();

        let mut writer = MstWriter::new();
        writer.add_tensor("test", vec![0, 1, 2, 3], DType::UInt8, vec![4]);
        writer.write_file(temp_file.path()).unwrap();

        assert!(is_mst(temp_file.path()));
    }

    #[test]
    fn test_mst_invalid_magic() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), b"INVALID").unwrap();

        let result = MstReader::open(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_mst_read_all() {
        let mut writer = MstWriter::new();

        let tensor1: Vec<u8> = vec![1, 2, 3, 4];
        let tensor2: Vec<u8> = vec![5, 6, 7, 8];

        writer.add_tensor("a", tensor1.clone(), DType::UInt8, vec![4]);
        writer.add_tensor("b", tensor2.clone(), DType::UInt8, vec![4]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        let mut reader = MstReader::open(temp_file.path()).unwrap();
        let all = reader.read_all().unwrap();

        assert_eq!(all.len(), 2);
        assert_eq!(all.get("a").unwrap(), &tensor1);
        assert_eq!(all.get("b").unwrap(), &tensor2);
    }

    #[test]
    fn test_mst_custom_compression() {
        let config = CompressionConfig::best();
        let mut writer = MstWriter::with_config(config);

        let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        writer.add_tensor("weight", data.clone(), DType::BFloat16, vec![2500]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        let reader = MstReader::open(temp_file.path()).unwrap();
        assert_eq!(reader.header().compression.level, 19);
        assert!(reader.header().compression.byte_grouping);
    }

    #[test]
    fn test_mst_bf16_compression_ratio() {
        // Simulate realistic bf16 weights with patterns
        let mut writer = MstWriter::new();

        // Create bf16-like data with clustered exponents (realistic for NN weights)
        let data: Vec<u8> = (0..50000u16)
            .flat_map(|i| {
                // Simulate bf16: sign=0, exp=127-130, mantissa=random-ish
                let exp = 127 + (i % 4) as u8;
                let mantissa = (i.wrapping_mul(7)) as u8 & 0x7F;
                let bf16 = ((exp as u16) << 7) | (mantissa as u16);
                bf16.to_le_bytes()
            })
            .collect();

        writer.add_tensor("weights", data.clone(), DType::BFloat16, vec![50000]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        // Check compression ratio
        let file_size = std::fs::metadata(temp_file.path()).unwrap().len() as usize;
        let ratio = data.len() as f64 / file_size as f64;

        // Should achieve meaningful compression on realistic bf16 data
        assert!(ratio > 1.5, "Expected ratio > 1.5, got {:.2}x", ratio);

        // Verify roundtrip
        let mut reader = MstReader::open(temp_file.path()).unwrap();
        let decompressed = reader.read_tensor("weights").unwrap();
        assert_eq!(decompressed, data, "Roundtrip must be lossless");
    }
}
