//! GGUF checkpoint format support for quantized LLMs.
//!
//! GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp
//! for storing quantized large language models.
//!
//! ## Format Overview
//!
//! GGUF files have the following structure:
//! - Header: Magic ("GGUF"), version, tensor count, metadata count
//! - Metadata: Key-value pairs with type information
//! - Tensor infos: Name, dimensions, type, offset for each tensor
//! - Tensor data: Raw tensor data (aligned)
//!
//! ## Reading GGUF Files
//!
//! ```rust,ignore
//! use mithril_checkpoint::gguf::GgufReader;
//!
//! let reader = GgufReader::open("model.gguf")?;
//! println!("Model: {}", reader.metadata().get_string("general.name").unwrap_or("unknown"));
//! println!("Tensors: {}", reader.tensor_count());
//!
//! for info in reader.tensor_infos() {
//!     println!("  {} {:?} {:?}", info.name, info.dims, info.dtype);
//! }
//! ```
//!
//! ## Writing GGUF Files
//!
//! ```rust,ignore
//! use mithril_checkpoint::gguf::{GgufWriter, GgufDType};
//!
//! let mut writer = GgufWriter::new();
//! writer.add_metadata("general.name", "my-model");
//! writer.add_tensor("token_embd.weight", &data, GgufDType::F16, &[4096, 32000]);
//! writer.write("output.gguf")?;
//! ```

use mithril_core::{MithrilError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// GGUF magic number: "GGUF" in little-endian
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// Current GGUF version
const GGUF_VERSION: u32 = 3;

/// Alignment for tensor data (32 bytes)
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGUF data types for tensors.
///
/// These names match the official llama.cpp GGUF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgufDType {
    /// 32-bit float
    F32 = 0,
    /// 16-bit float
    F16 = 1,
    /// 4-bit quantization (32 weights per block)
    Q4_0 = 2,
    /// 4-bit quantization with 16-bit scales
    Q4_1 = 3,
    /// Legacy 4-bit quantization
    Q4_2 = 4,
    /// Legacy 4-bit quantization
    Q4_3 = 5,
    /// 5-bit quantization
    Q5_0 = 6,
    /// 5-bit quantization with 16-bit scales
    Q5_1 = 7,
    /// 8-bit quantization
    Q8_0 = 8,
    /// 8-bit quantization with 16-bit scales
    Q8_1 = 9,
    /// K-quant 2-bit
    Q2_K = 10,
    /// K-quant 3-bit
    Q3_K = 11,
    /// K-quant 4-bit
    Q4_K = 12,
    /// K-quant 5-bit
    Q5_K = 13,
    /// K-quant 6-bit
    Q6_K = 14,
    /// 8-bit quantization (K-quant compatible)
    Q8_K = 15,
    /// IQ quantization types
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    /// 8-bit integer
    I8 = 24,
    /// 16-bit integer
    I16 = 25,
    /// 32-bit integer
    I32 = 26,
    /// 64-bit integer
    I64 = 27,
    /// 64-bit float
    F64 = 28,
    /// BFloat16
    BF16 = 29,
}

impl GgufDType {
    /// Get the number of bytes per element for non-quantized types.
    /// For quantized types, this returns the block size.
    pub fn type_size(&self) -> usize {
        match self {
            GgufDType::F32 => 4,
            GgufDType::F16 => 2,
            GgufDType::BF16 => 2,
            GgufDType::F64 => 8,
            GgufDType::I8 => 1,
            GgufDType::I16 => 2,
            GgufDType::I32 => 4,
            GgufDType::I64 => 8,
            // Quantized types have block sizes
            GgufDType::Q4_0 => 18, // 32 weights, 2 bytes header + 16 bytes data
            GgufDType::Q4_1 => 20,
            GgufDType::Q5_0 => 22,
            GgufDType::Q5_1 => 24,
            GgufDType::Q8_0 => 34,
            GgufDType::Q8_1 => 36,
            GgufDType::Q2_K => 84,
            GgufDType::Q3_K => 110,
            GgufDType::Q4_K => 144,
            GgufDType::Q5_K => 176,
            GgufDType::Q6_K => 210,
            GgufDType::Q8_K => 292,
            _ => 1, // Default for unknown/IQ types
        }
    }

    /// Get the block size (number of elements per block).
    pub fn block_size(&self) -> usize {
        match self {
            GgufDType::F32 | GgufDType::F16 | GgufDType::BF16 | GgufDType::F64 => 1,
            GgufDType::I8 | GgufDType::I16 | GgufDType::I32 | GgufDType::I64 => 1,
            GgufDType::Q4_0 | GgufDType::Q4_1 | GgufDType::Q5_0 | GgufDType::Q5_1 => 32,
            GgufDType::Q8_0 | GgufDType::Q8_1 => 32,
            GgufDType::Q2_K
            | GgufDType::Q3_K
            | GgufDType::Q4_K
            | GgufDType::Q5_K
            | GgufDType::Q6_K => 256,
            GgufDType::Q8_K => 256,
            _ => 1,
        }
    }

    /// Check if this is a quantized type.
    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            GgufDType::F32
                | GgufDType::F16
                | GgufDType::BF16
                | GgufDType::F64
                | GgufDType::I8
                | GgufDType::I16
                | GgufDType::I32
                | GgufDType::I64
        )
    }

    /// Parse from u32 value.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GgufDType::F32),
            1 => Some(GgufDType::F16),
            2 => Some(GgufDType::Q4_0),
            3 => Some(GgufDType::Q4_1),
            4 => Some(GgufDType::Q4_2),
            5 => Some(GgufDType::Q4_3),
            6 => Some(GgufDType::Q5_0),
            7 => Some(GgufDType::Q5_1),
            8 => Some(GgufDType::Q8_0),
            9 => Some(GgufDType::Q8_1),
            10 => Some(GgufDType::Q2_K),
            11 => Some(GgufDType::Q3_K),
            12 => Some(GgufDType::Q4_K),
            13 => Some(GgufDType::Q5_K),
            14 => Some(GgufDType::Q6_K),
            15 => Some(GgufDType::Q8_K),
            16 => Some(GgufDType::IQ2_XXS),
            17 => Some(GgufDType::IQ2_XS),
            18 => Some(GgufDType::IQ3_XXS),
            19 => Some(GgufDType::IQ1_S),
            20 => Some(GgufDType::IQ4_NL),
            21 => Some(GgufDType::IQ3_S),
            22 => Some(GgufDType::IQ2_S),
            23 => Some(GgufDType::IQ4_XS),
            24 => Some(GgufDType::I8),
            25 => Some(GgufDType::I16),
            26 => Some(GgufDType::I32),
            27 => Some(GgufDType::I64),
            28 => Some(GgufDType::F64),
            29 => Some(GgufDType::BF16),
            _ => None,
        }
    }
}

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetaType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetaType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GgufMetaType::UInt8),
            1 => Some(GgufMetaType::Int8),
            2 => Some(GgufMetaType::UInt16),
            3 => Some(GgufMetaType::Int16),
            4 => Some(GgufMetaType::UInt32),
            5 => Some(GgufMetaType::Int32),
            6 => Some(GgufMetaType::Float32),
            7 => Some(GgufMetaType::Bool),
            8 => Some(GgufMetaType::String),
            9 => Some(GgufMetaType::Array),
            10 => Some(GgufMetaType::UInt64),
            11 => Some(GgufMetaType::Int64),
            12 => Some(GgufMetaType::Float64),
            _ => None,
        }
    }
}

/// GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Get as string if this is a string value.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as u64 if this is a numeric value.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::UInt8(v) => Some(*v as u64),
            GgufValue::UInt16(v) => Some(*v as u64),
            GgufValue::UInt32(v) => Some(*v as u64),
            GgufValue::UInt64(v) => Some(*v),
            GgufValue::Int8(v) => Some(*v as u64),
            GgufValue::Int16(v) => Some(*v as u64),
            GgufValue::Int32(v) => Some(*v as u64),
            GgufValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Get as f64 if this is a numeric value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::Float32(v) => Some(*v as f64),
            GgufValue::Float64(v) => Some(*v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    /// Get the metadata type for this value.
    pub fn meta_type(&self) -> GgufMetaType {
        match self {
            GgufValue::UInt8(_) => GgufMetaType::UInt8,
            GgufValue::Int8(_) => GgufMetaType::Int8,
            GgufValue::UInt16(_) => GgufMetaType::UInt16,
            GgufValue::Int16(_) => GgufMetaType::Int16,
            GgufValue::UInt32(_) => GgufMetaType::UInt32,
            GgufValue::Int32(_) => GgufMetaType::Int32,
            GgufValue::UInt64(_) => GgufMetaType::UInt64,
            GgufValue::Int64(_) => GgufMetaType::Int64,
            GgufValue::Float32(_) => GgufMetaType::Float32,
            GgufValue::Float64(_) => GgufMetaType::Float64,
            GgufValue::Bool(_) => GgufMetaType::Bool,
            GgufValue::String(_) => GgufMetaType::String,
            GgufValue::Array(_) => GgufMetaType::Array,
        }
    }
}

/// GGUF file metadata.
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    /// Key-value pairs.
    pub kv: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    /// Get a string value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| v.as_string())
    }

    /// Get a u64 value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv.get(key).and_then(|v| v.as_u64())
    }

    /// Get a f64 value.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.kv.get(key).and_then(|v| v.as_f64())
    }
}

/// Information about a tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Tensor dimensions.
    pub dims: Vec<u64>,
    /// Data type.
    pub dtype: GgufDType,
    /// Offset in the data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Calculate the size of the tensor data in bytes.
    pub fn data_size(&self) -> usize {
        let num_elements: u64 = self.dims.iter().product();
        let block_size = self.dtype.block_size() as u64;
        let type_size = self.dtype.type_size() as u64;

        if self.dtype.is_quantized() {
            // For quantized types: (num_elements / block_size) * type_size
            let num_blocks = num_elements.div_ceil(block_size);
            (num_blocks * type_size) as usize
        } else {
            // For non-quantized types: num_elements * type_size
            (num_elements * type_size) as usize
        }
    }
}

/// Reader for GGUF files.
pub struct GgufReader {
    file: BufReader<File>,
    metadata: GgufMetadata,
    tensors: Vec<GgufTensorInfo>,
    data_offset: u64,
}

impl GgufReader {
    /// Open a GGUF file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read header
        let magic = Self::read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(MithrilError::InvalidFormat(format!(
                "Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, magic
            )));
        }

        let version = Self::read_u32(&mut reader)?;
        if !(2..=3).contains(&version) {
            return Err(MithrilError::InvalidFormat(format!(
                "Unsupported GGUF version: {}",
                version
            )));
        }

        let tensor_count = Self::read_u64(&mut reader)?;
        let metadata_count = Self::read_u64(&mut reader)?;

        // Read metadata
        let mut metadata = GgufMetadata::default();
        for _ in 0..metadata_count {
            let (key, value) = Self::read_kv(&mut reader)?;
            metadata.kv.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = Self::read_tensor_info(&mut reader)?;
            tensors.push(info);
        }

        // Calculate data section offset (aligned)
        let current_pos = reader.stream_position()? as usize;
        let alignment = metadata
            .get_u64("general.alignment")
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u64) as usize;
        let data_offset = current_pos.div_ceil(alignment) * alignment;

        Ok(Self {
            file: reader,
            metadata,
            tensors,
            data_offset: data_offset as u64,
        })
    }

    /// Get metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Get tensor count.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get tensor infos.
    pub fn tensor_infos(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Read tensor data.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| MithrilError::NotFound(name.to_string()))?
            .clone();

        let size = info.data_size();
        let offset = self.data_offset + info.offset;

        self.file.seek(SeekFrom::Start(offset))?;

        let mut data = vec![0u8; size];
        self.file.read_exact(&mut data)?;

        Ok(data)
    }

    /// Read all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    // Helper functions for reading

    fn read_u32(reader: &mut BufReader<File>) -> Result<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(reader: &mut BufReader<File>) -> Result<u64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(reader: &mut BufReader<File>) -> Result<String> {
        let len = Self::read_u64(reader)? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| MithrilError::InvalidFormat(e.to_string()))
    }

    fn read_kv(reader: &mut BufReader<File>) -> Result<(String, GgufValue)> {
        let key = Self::read_string(reader)?;
        let value = Self::read_value(reader)?;
        Ok((key, value))
    }

    fn read_value(reader: &mut BufReader<File>) -> Result<GgufValue> {
        let type_id = Self::read_u32(reader)?;
        let meta_type = GgufMetaType::from_u32(type_id).ok_or_else(|| {
            MithrilError::InvalidFormat(format!("Unknown metadata type: {}", type_id))
        })?;

        match meta_type {
            GgufMetaType::UInt8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::UInt8(buf[0]))
            }
            GgufMetaType::Int8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int8(buf[0] as i8))
            }
            GgufMetaType::UInt16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::UInt16(u16::from_le_bytes(buf)))
            }
            GgufMetaType::Int16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
            }
            GgufMetaType::UInt32 => Ok(GgufValue::UInt32(Self::read_u32(reader)?)),
            GgufMetaType::Int32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
            }
            GgufMetaType::UInt64 => Ok(GgufValue::UInt64(Self::read_u64(reader)?)),
            GgufMetaType::Int64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
            }
            GgufMetaType::Float32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
            }
            GgufMetaType::Float64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
            }
            GgufMetaType::Bool => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Bool(buf[0] != 0))
            }
            GgufMetaType::String => Ok(GgufValue::String(Self::read_string(reader)?)),
            GgufMetaType::Array => {
                let element_type = Self::read_u32(reader)?;
                let count = Self::read_u64(reader)? as usize;
                let mut values = Vec::with_capacity(count);

                // Temporarily write the type back so read_value can parse it
                for _ in 0..count {
                    let value = Self::read_array_element(reader, element_type)?;
                    values.push(value);
                }

                Ok(GgufValue::Array(values))
            }
        }
    }

    fn read_array_element(reader: &mut BufReader<File>, type_id: u32) -> Result<GgufValue> {
        let meta_type = GgufMetaType::from_u32(type_id).ok_or_else(|| {
            MithrilError::InvalidFormat(format!("Unknown array element type: {}", type_id))
        })?;

        match meta_type {
            GgufMetaType::UInt8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::UInt8(buf[0]))
            }
            GgufMetaType::Int8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int8(buf[0] as i8))
            }
            GgufMetaType::UInt16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::UInt16(u16::from_le_bytes(buf)))
            }
            GgufMetaType::Int16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
            }
            GgufMetaType::UInt32 => Ok(GgufValue::UInt32(Self::read_u32(reader)?)),
            GgufMetaType::Int32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
            }
            GgufMetaType::UInt64 => Ok(GgufValue::UInt64(Self::read_u64(reader)?)),
            GgufMetaType::Int64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
            }
            GgufMetaType::Float32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
            }
            GgufMetaType::Float64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
            }
            GgufMetaType::Bool => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Bool(buf[0] != 0))
            }
            GgufMetaType::String => Ok(GgufValue::String(Self::read_string(reader)?)),
            GgufMetaType::Array => {
                // Nested arrays
                let inner_type = Self::read_u32(reader)?;
                let count = Self::read_u64(reader)? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(Self::read_array_element(reader, inner_type)?);
                }
                Ok(GgufValue::Array(values))
            }
        }
    }

    fn read_tensor_info(reader: &mut BufReader<File>) -> Result<GgufTensorInfo> {
        let name = Self::read_string(reader)?;
        let n_dims = Self::read_u32(reader)? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(Self::read_u64(reader)?);
        }

        let dtype_id = Self::read_u32(reader)?;
        let dtype = GgufDType::from_u32(dtype_id)
            .ok_or_else(|| MithrilError::InvalidFormat(format!("Unknown dtype: {}", dtype_id)))?;

        let offset = Self::read_u64(reader)?;

        Ok(GgufTensorInfo {
            name,
            dims,
            dtype,
            offset,
        })
    }
}

/// Statistics from writing a GGUF file.
#[derive(Debug, Clone, Default)]
pub struct GgufWriteStats {
    /// Number of metadata entries written.
    pub metadata_count: usize,
    /// Number of tensors written.
    pub tensor_count: usize,
    /// Total bytes written.
    pub total_bytes: usize,
}

/// Internal tensor info for writer.
#[derive(Debug, Clone)]
struct TensorWriteInfo {
    name: String,
    dims: Vec<u64>,
    dtype: GgufDType,
    data: Vec<u8>,
}

/// Writer for GGUF files.
pub struct GgufWriter {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<TensorWriteInfo>,
}

impl GgufWriter {
    /// Create a new GGUF writer.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Add a metadata entry.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<GgufValue>) {
        self.metadata.push((key.into(), value.into()));
    }

    /// Add a string metadata entry.
    pub fn add_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata
            .push((key.into(), GgufValue::String(value.into())));
    }

    /// Add a u32 metadata entry.
    pub fn add_u32(&mut self, key: impl Into<String>, value: u32) {
        self.metadata.push((key.into(), GgufValue::UInt32(value)));
    }

    /// Add a tensor.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        data: &[u8],
        dtype: GgufDType,
        dims: &[u64],
    ) {
        self.tensors.push(TensorWriteInfo {
            name: name.into(),
            dims: dims.to_vec(),
            dtype,
            data: data.to_vec(),
        });
    }

    /// Write to a file.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<GgufWriteStats> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        let mut stats = GgufWriteStats::default();

        // Write header
        writer.write_all(&GGUF_MAGIC.to_le_bytes())?;
        writer.write_all(&GGUF_VERSION.to_le_bytes())?;
        writer.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.metadata.len() as u64).to_le_bytes())?;

        // Write metadata
        for (key, value) in &self.metadata {
            Self::write_string(&mut writer, key)?;
            Self::write_value(&mut writer, value)?;
            stats.metadata_count += 1;
        }

        // Calculate data offsets
        let header_end = writer.stream_position()? as usize;
        let mut tensor_info_size = 0;
        for tensor in &self.tensors {
            // name length (8) + name + n_dims (4) + dims (8 * n) + dtype (4) + offset (8)
            tensor_info_size += 8 + tensor.name.len() + 4 + 8 * tensor.dims.len() + 4 + 8;
        }

        let alignment = GGUF_DEFAULT_ALIGNMENT;
        let data_start = (header_end + tensor_info_size).div_ceil(alignment) * alignment;

        // Write tensor infos with calculated offsets
        let mut current_offset: u64 = 0;
        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());

        for tensor in &self.tensors {
            // Align offset
            let aligned_offset = (current_offset as usize).div_ceil(alignment) * alignment;
            tensor_offsets.push(aligned_offset as u64);

            Self::write_string(&mut writer, &tensor.name)?;
            writer.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
            for dim in &tensor.dims {
                writer.write_all(&dim.to_le_bytes())?;
            }
            writer.write_all(&(tensor.dtype as u32).to_le_bytes())?;
            writer.write_all(&(aligned_offset as u64).to_le_bytes())?;

            current_offset = aligned_offset as u64 + tensor.data.len() as u64;
            stats.tensor_count += 1;
        }

        // Pad to data start
        let current_pos = writer.stream_position()? as usize;
        let padding_needed = data_start - current_pos;
        writer.write_all(&vec![0u8; padding_needed])?;

        // Write tensor data
        for (tensor, offset) in self.tensors.iter().zip(tensor_offsets.iter()) {
            // Pad to tensor offset
            let current_pos = writer.stream_position()? - data_start as u64;
            if current_pos < *offset {
                let padding = (*offset - current_pos) as usize;
                writer.write_all(&vec![0u8; padding])?;
            }
            writer.write_all(&tensor.data)?;
        }

        writer.flush()?;
        stats.total_bytes = writer.stream_position()? as usize;

        Ok(stats)
    }

    fn write_string(writer: &mut BufWriter<File>, s: &str) -> Result<()> {
        writer.write_all(&(s.len() as u64).to_le_bytes())?;
        writer.write_all(s.as_bytes())?;
        Ok(())
    }

    fn write_value(writer: &mut BufWriter<File>, value: &GgufValue) -> Result<()> {
        writer.write_all(&(value.meta_type() as u32).to_le_bytes())?;

        match value {
            GgufValue::UInt8(v) => writer.write_all(&[*v])?,
            GgufValue::Int8(v) => writer.write_all(&[*v as u8])?,
            GgufValue::UInt16(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int16(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::UInt32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::UInt64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Bool(v) => writer.write_all(&[if *v { 1 } else { 0 }])?,
            GgufValue::String(s) => Self::write_string(writer, s)?,
            GgufValue::Array(arr) => {
                if arr.is_empty() {
                    writer.write_all(&(GgufMetaType::UInt8 as u32).to_le_bytes())?;
                    writer.write_all(&0u64.to_le_bytes())?;
                } else {
                    let element_type = arr[0].meta_type();
                    writer.write_all(&(element_type as u32).to_le_bytes())?;
                    writer.write_all(&(arr.len() as u64).to_le_bytes())?;
                    for v in arr {
                        Self::write_array_element(writer, v)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn write_array_element(writer: &mut BufWriter<File>, value: &GgufValue) -> Result<()> {
        match value {
            GgufValue::UInt8(v) => writer.write_all(&[*v])?,
            GgufValue::Int8(v) => writer.write_all(&[*v as u8])?,
            GgufValue::UInt16(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int16(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::UInt32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::UInt64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Bool(v) => writer.write_all(&[if *v { 1 } else { 0 }])?,
            GgufValue::String(s) => Self::write_string(writer, s)?,
            GgufValue::Array(arr) => {
                if arr.is_empty() {
                    writer.write_all(&(GgufMetaType::UInt8 as u32).to_le_bytes())?;
                    writer.write_all(&0u64.to_le_bytes())?;
                } else {
                    let element_type = arr[0].meta_type();
                    writer.write_all(&(element_type as u32).to_le_bytes())?;
                    writer.write_all(&(arr.len() as u64).to_le_bytes())?;
                    for v in arr {
                        Self::write_array_element(writer, v)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get the number of metadata entries.
    pub fn metadata_count(&self) -> usize {
        self.metadata.len()
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

// Implement From for common types
impl From<String> for GgufValue {
    fn from(s: String) -> Self {
        GgufValue::String(s)
    }
}

impl From<&str> for GgufValue {
    fn from(s: &str) -> Self {
        GgufValue::String(s.to_string())
    }
}

impl From<u32> for GgufValue {
    fn from(v: u32) -> Self {
        GgufValue::UInt32(v)
    }
}

impl From<u64> for GgufValue {
    fn from(v: u64) -> Self {
        GgufValue::UInt64(v)
    }
}

impl From<i32> for GgufValue {
    fn from(v: i32) -> Self {
        GgufValue::Int32(v)
    }
}

impl From<f32> for GgufValue {
    fn from(v: f32) -> Self {
        GgufValue::Float32(v)
    }
}

impl From<bool> for GgufValue {
    fn from(v: bool) -> Self {
        GgufValue::Bool(v)
    }
}

/// Check if a file is a GGUF file.
pub fn is_gguf<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();

    // Check extension
    if let Some(ext) = path.extension() {
        if ext == "gguf" {
            return true;
        }
    }

    // Try to read magic
    if let Ok(mut file) = File::open(path) {
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() {
            return u32::from_le_bytes(magic) == GGUF_MAGIC;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_gguf_dtype_sizes() {
        assert_eq!(GgufDType::F32.type_size(), 4);
        assert_eq!(GgufDType::F16.type_size(), 2);
        assert_eq!(GgufDType::BF16.type_size(), 2);
        assert_eq!(GgufDType::Q4_0.type_size(), 18);
    }

    #[test]
    fn test_gguf_dtype_block_sizes() {
        assert_eq!(GgufDType::F32.block_size(), 1);
        assert_eq!(GgufDType::Q4_0.block_size(), 32);
        assert_eq!(GgufDType::Q4_K.block_size(), 256);
    }

    #[test]
    fn test_gguf_dtype_is_quantized() {
        assert!(!GgufDType::F32.is_quantized());
        assert!(!GgufDType::F16.is_quantized());
        assert!(GgufDType::Q4_0.is_quantized());
        assert!(GgufDType::Q8_K.is_quantized());
    }

    #[test]
    fn test_gguf_value_conversions() {
        let s: GgufValue = "test".into();
        assert_eq!(s.as_string(), Some("test"));

        let n: GgufValue = 42u32.into();
        assert_eq!(n.as_u64(), Some(42));

        let f: GgufValue = 3.14f32.into();
        assert!(f.as_f64().is_some());
    }

    #[test]
    fn test_gguf_writer_basic() {
        let mut writer = GgufWriter::new();
        writer.add_string("general.name", "test-model");
        writer.add_u32("general.quantization_version", 2);

        let data: Vec<u8> = vec![0; 1024];
        writer.add_tensor("test.weight", &data, GgufDType::F16, &[32, 16]);

        assert_eq!(writer.metadata_count(), 2);
        assert_eq!(writer.tensor_count(), 1);
    }

    #[test]
    fn test_gguf_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");

        // Write
        let mut writer = GgufWriter::new();
        writer.add_string("general.name", "test-model");
        writer.add_u32("general.version", 1);

        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor("layer.weight", &tensor_bytes, GgufDType::F32, &[2, 2]);

        let stats = writer.write(&path).unwrap();
        assert_eq!(stats.tensor_count, 1);
        assert_eq!(stats.metadata_count, 2);

        // Read
        let mut reader = GgufReader::open(&path).unwrap();
        assert_eq!(
            reader.metadata().get_string("general.name"),
            Some("test-model")
        );
        assert_eq!(reader.metadata().get_u64("general.version"), Some(1));
        assert_eq!(reader.tensor_count(), 1);

        let info = reader.tensor_info("layer.weight").unwrap();
        assert_eq!(info.dims, vec![2, 2]);
        assert_eq!(info.dtype, GgufDType::F32);

        let read_data = reader.read_tensor("layer.weight").unwrap();
        assert_eq!(read_data, tensor_bytes);
    }

    #[test]
    fn test_is_gguf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");

        let mut writer = GgufWriter::new();
        writer.add_string("general.name", "test");
        writer.write(&path).unwrap();

        assert!(is_gguf(&path));
        assert!(!is_gguf(dir.path().join("nonexistent.bin")));
    }

    #[test]
    fn test_tensor_data_size() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            dims: vec![1024, 768],
            dtype: GgufDType::F16,
            offset: 0,
        };
        assert_eq!(info.data_size(), 1024 * 768 * 2);

        let q4_info = GgufTensorInfo {
            name: "test_q4".to_string(),
            dims: vec![1024],
            dtype: GgufDType::Q4_0,
            offset: 0,
        };
        // 1024 elements / 32 per block * 18 bytes per block = 576 bytes
        assert_eq!(q4_info.data_size(), 32 * 18);
    }
}
