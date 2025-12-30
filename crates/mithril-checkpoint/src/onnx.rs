//! ONNX format reader/writer for model compression.
//!
//! This module provides read/write support for ONNX models (.onnx files),
//! allowing extraction and compression of model weights (initializers).
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mithril_checkpoint::onnx::{OnnxReader, OnnxWriter, is_onnx};
//! use std::path::Path;
//!
//! // Check if file is ONNX
//! assert!(is_onnx(Path::new("model.onnx")).unwrap());
//!
//! // Read ONNX model
//! let reader = OnnxReader::open("model.onnx")?;
//! println!("Initializers: {}", reader.initializer_count());
//!
//! // Iterate over initializers (weights)
//! for init in reader.initializers() {
//!     println!("{}: {:?} {:?}", init.name, init.dtype, init.dims);
//! }
//!
//! // Create compressed version
//! let writer = OnnxWriter::from_reader(&reader)?;
//! writer.write("model_compressed.onnx")?;
//! ```
//!
//! ## ONNX Wire Format
//!
//! ONNX uses Protocol Buffers encoding. The key structures are:
//! - ModelProto: Top-level container
//!   - graph: GraphProto containing the computation graph
//!     - initializer: List of TensorProto with model weights
//!
//! This implementation manually parses the protobuf wire format to extract
//! initializers without requiring the full ONNX proto definitions.

use mithril_core::types::DType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use thiserror::Error;

/// ONNX-related errors.
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid ONNX file: {0}")]
    InvalidFormat(String),

    #[error("Unsupported ONNX version: {0}")]
    UnsupportedVersion(i64),

    #[error("Protobuf parse error: {0}")]
    ProtobufError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

/// Result type for ONNX operations.
pub type Result<T> = std::result::Result<T, OnnxError>;

/// ONNX data types (from onnx.proto TensorProto.DataType).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum OnnxDataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
    Float8E4M3Fn = 17,
    Float8E4M3FnUz = 18,
    Float8E5M2 = 19,
    Float8E5M2FnUz = 20,
}

impl OnnxDataType {
    /// Create from i32 value.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Undefined),
            1 => Some(Self::Float),
            2 => Some(Self::Uint8),
            3 => Some(Self::Int8),
            4 => Some(Self::Uint16),
            5 => Some(Self::Int16),
            6 => Some(Self::Int32),
            7 => Some(Self::Int64),
            8 => Some(Self::String),
            9 => Some(Self::Bool),
            10 => Some(Self::Float16),
            11 => Some(Self::Double),
            12 => Some(Self::Uint32),
            13 => Some(Self::Uint64),
            14 => Some(Self::Complex64),
            15 => Some(Self::Complex128),
            16 => Some(Self::BFloat16),
            17 => Some(Self::Float8E4M3Fn),
            18 => Some(Self::Float8E4M3FnUz),
            19 => Some(Self::Float8E5M2),
            20 => Some(Self::Float8E5M2FnUz),
            _ => None,
        }
    }

    /// Get element size in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Undefined => 0,
            Self::Float => 4,
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::Uint32 => 4,
            Self::Int64 | Self::Uint64 | Self::Double | Self::Complex64 => 8,
            Self::Complex128 => 16,
            Self::String => 0, // Variable
            Self::Float8E4M3Fn | Self::Float8E4M3FnUz | Self::Float8E5M2 | Self::Float8E5M2FnUz => {
                1
            }
        }
    }

    /// Convert to mithril DType if possible.
    pub fn to_dtype(&self) -> Option<DType> {
        match self {
            Self::Float => Some(DType::Float32),
            Self::Float16 => Some(DType::Float16),
            Self::BFloat16 => Some(DType::BFloat16),
            Self::Double => Some(DType::Float64),
            Self::Int8 => Some(DType::Int8),
            Self::Int32 => Some(DType::Int32),
            Self::Int64 => Some(DType::Int64),
            Self::Uint8 => Some(DType::UInt8),
            Self::Bool => Some(DType::Bool),
            _ => None,
        }
    }
}

impl std::fmt::Display for OnnxDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undefined => write!(f, "undefined"),
            Self::Float => write!(f, "float32"),
            Self::Uint8 => write!(f, "uint8"),
            Self::Int8 => write!(f, "int8"),
            Self::Uint16 => write!(f, "uint16"),
            Self::Int16 => write!(f, "int16"),
            Self::Int32 => write!(f, "int32"),
            Self::Int64 => write!(f, "int64"),
            Self::String => write!(f, "string"),
            Self::Bool => write!(f, "bool"),
            Self::Float16 => write!(f, "float16"),
            Self::Double => write!(f, "float64"),
            Self::Uint32 => write!(f, "uint32"),
            Self::Uint64 => write!(f, "uint64"),
            Self::Complex64 => write!(f, "complex64"),
            Self::Complex128 => write!(f, "complex128"),
            Self::BFloat16 => write!(f, "bfloat16"),
            Self::Float8E4M3Fn => write!(f, "float8e4m3fn"),
            Self::Float8E4M3FnUz => write!(f, "float8e4m3fnuz"),
            Self::Float8E5M2 => write!(f, "float8e5m2"),
            Self::Float8E5M2FnUz => write!(f, "float8e5m2fnuz"),
        }
    }
}

/// Information about an ONNX initializer (tensor).
#[derive(Debug, Clone)]
pub struct OnnxInitializer {
    /// Tensor name.
    pub name: String,
    /// Data type.
    pub dtype: OnnxDataType,
    /// Tensor dimensions.
    pub dims: Vec<i64>,
    /// Raw tensor data.
    pub raw_data: Vec<u8>,
    /// Offset in file (for external data).
    pub file_offset: Option<u64>,
    /// Size in file.
    pub file_size: Option<u64>,
}

impl OnnxInitializer {
    /// Get total element count.
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Get data size in bytes.
    pub fn byte_size(&self) -> usize {
        if !self.raw_data.is_empty() {
            self.raw_data.len()
        } else {
            self.numel() * self.dtype.element_size()
        }
    }
}

/// ONNX model metadata.
#[derive(Debug, Clone, Default)]
pub struct OnnxMetadata {
    /// IR version.
    pub ir_version: i64,
    /// Producer name.
    pub producer_name: String,
    /// Producer version.
    pub producer_version: String,
    /// Model domain.
    pub domain: String,
    /// Model version.
    pub model_version: i64,
    /// Documentation string.
    pub doc_string: String,
    /// Opset imports (domain -> version).
    pub opset_imports: HashMap<String, i64>,
}

/// ONNX file reader.
pub struct OnnxReader {
    metadata: OnnxMetadata,
    initializers: Vec<OnnxInitializer>,
    raw_model: Vec<u8>,
}

impl OnnxReader {
    /// Open an ONNX file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Self::from_bytes(&data)
    }

    /// Parse ONNX from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // Verify it's a valid protobuf (basic check)
        if data.is_empty() {
            return Err(OnnxError::InvalidFormat("Empty file".into()));
        }

        let mut metadata = OnnxMetadata::default();
        let mut initializers = Vec::new();

        // Parse the ModelProto
        let mut pos = 0;
        while pos < data.len() {
            let (field_number, wire_type, bytes_read) = read_tag(&data[pos..])?;
            pos += bytes_read;

            match (field_number, wire_type) {
                // ir_version (field 1, varint)
                (1, 0) => {
                    let (value, bytes_read) = read_varint(&data[pos..])?;
                    metadata.ir_version = value as i64;
                    pos += bytes_read;
                }
                // producer_name (field 2, length-delimited)
                (2, 2) => {
                    let (value, bytes_read) = read_string(&data[pos..])?;
                    metadata.producer_name = value;
                    pos += bytes_read;
                }
                // producer_version (field 3, length-delimited)
                (3, 2) => {
                    let (value, bytes_read) = read_string(&data[pos..])?;
                    metadata.producer_version = value;
                    pos += bytes_read;
                }
                // domain (field 4, length-delimited)
                (4, 2) => {
                    let (value, bytes_read) = read_string(&data[pos..])?;
                    metadata.domain = value;
                    pos += bytes_read;
                }
                // model_version (field 5, varint)
                (5, 0) => {
                    let (value, bytes_read) = read_varint(&data[pos..])?;
                    metadata.model_version = value as i64;
                    pos += bytes_read;
                }
                // doc_string (field 6, length-delimited)
                (6, 2) => {
                    let (value, bytes_read) = read_string(&data[pos..])?;
                    metadata.doc_string = value;
                    pos += bytes_read;
                }
                // graph (field 7, length-delimited) - contains initializers
                (7, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let graph_data = &data[pos..pos + len as usize];
                    Self::parse_graph(graph_data, &mut initializers)?;
                    pos += len as usize;
                }
                // opset_import (field 8, length-delimited)
                (8, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let opset_data = &data[pos..pos + len as usize];
                    if let Ok((domain, version)) = Self::parse_opset_import(opset_data) {
                        metadata.opset_imports.insert(domain, version);
                    }
                    pos += len as usize;
                }
                // Unknown field - skip
                (_, wire_type) => {
                    pos += skip_field(&data[pos..], wire_type)?;
                }
            }
        }

        Ok(Self {
            metadata,
            initializers,
            raw_model: data.to_vec(),
        })
    }

    /// Parse GraphProto to extract initializers.
    fn parse_graph(data: &[u8], initializers: &mut Vec<OnnxInitializer>) -> Result<()> {
        let mut pos = 0;
        while pos < data.len() {
            let (field_number, wire_type, bytes_read) = read_tag(&data[pos..])?;
            pos += bytes_read;

            match (field_number, wire_type) {
                // initializer (field 5, length-delimited) - repeated TensorProto
                (5, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let tensor_data = &data[pos..pos + len as usize];
                    if let Ok(init) = Self::parse_tensor_proto(tensor_data) {
                        initializers.push(init);
                    }
                    pos += len as usize;
                }
                // Skip other fields
                (_, wire_type) => {
                    pos += skip_field(&data[pos..], wire_type)?;
                }
            }
        }
        Ok(())
    }

    /// Parse TensorProto to extract initializer data.
    fn parse_tensor_proto(data: &[u8]) -> Result<OnnxInitializer> {
        let mut name = String::new();
        let mut dtype = OnnxDataType::Undefined;
        let mut dims = Vec::new();
        let mut raw_data = Vec::new();
        let mut float_data = Vec::new();
        let mut int32_data = Vec::new();
        let mut int64_data = Vec::new();
        let mut double_data = Vec::new();

        let mut pos = 0;
        while pos < data.len() {
            let (field_number, wire_type, bytes_read) = read_tag(&data[pos..])?;
            pos += bytes_read;

            match (field_number, wire_type) {
                // dims (field 1, packed repeated int64)
                (1, 0) => {
                    let (value, bytes_read) = read_varint(&data[pos..])?;
                    dims.push(value as i64);
                    pos += bytes_read;
                }
                (1, 2) => {
                    // Packed repeated
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let packed_data = &data[pos..pos + len as usize];
                    let mut packed_pos = 0;
                    while packed_pos < packed_data.len() {
                        let (value, bytes_read) = read_varint(&packed_data[packed_pos..])?;
                        dims.push(value as i64);
                        packed_pos += bytes_read;
                    }
                    pos += len as usize;
                }
                // data_type (field 2, varint)
                (2, 0) => {
                    let (value, bytes_read) = read_varint(&data[pos..])?;
                    dtype = OnnxDataType::from_i32(value as i32).unwrap_or(OnnxDataType::Undefined);
                    pos += bytes_read;
                }
                // float_data (field 4, packed repeated float)
                (4, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let packed_data = &data[pos..pos + len as usize];
                    for chunk in packed_data.chunks_exact(4) {
                        float_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    pos += len as usize;
                }
                // int32_data (field 5, packed repeated int32)
                (5, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let packed_data = &data[pos..pos + len as usize];
                    for chunk in packed_data.chunks_exact(4) {
                        int32_data.push(i32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    pos += len as usize;
                }
                // int64_data (field 7, packed repeated int64)
                (7, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let packed_data = &data[pos..pos + len as usize];
                    for chunk in packed_data.chunks_exact(8) {
                        int64_data.push(i64::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    pos += len as usize;
                }
                // name (field 8, string)
                (8, 2) => {
                    let (value, bytes_read) = read_string(&data[pos..])?;
                    name = value;
                    pos += bytes_read;
                }
                // raw_data (field 9, bytes)
                (9, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    raw_data = data[pos..pos + len as usize].to_vec();
                    pos += len as usize;
                }
                // double_data (field 10, packed repeated double)
                (10, 2) => {
                    let (len, bytes_read) = read_varint(&data[pos..])?;
                    pos += bytes_read;
                    let packed_data = &data[pos..pos + len as usize];
                    for chunk in packed_data.chunks_exact(8) {
                        double_data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    pos += len as usize;
                }
                // Skip other fields
                (_, wire_type) => {
                    pos += skip_field(&data[pos..], wire_type)?;
                }
            }
        }

        // If raw_data is empty, reconstruct from typed arrays
        if raw_data.is_empty() {
            if !float_data.is_empty() {
                raw_data = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            } else if !int32_data.is_empty() {
                raw_data = int32_data.iter().flat_map(|i| i.to_le_bytes()).collect();
            } else if !int64_data.is_empty() {
                raw_data = int64_data.iter().flat_map(|i| i.to_le_bytes()).collect();
            } else if !double_data.is_empty() {
                raw_data = double_data.iter().flat_map(|d| d.to_le_bytes()).collect();
            }
        }

        Ok(OnnxInitializer {
            name,
            dtype,
            dims,
            raw_data,
            file_offset: None,
            file_size: None,
        })
    }

    /// Parse OperatorSetIdProto.
    fn parse_opset_import(data: &[u8]) -> Result<(String, i64)> {
        let mut domain = String::new();
        let mut version: i64 = 0;

        let mut pos = 0;
        while pos < data.len() {
            let (field_number, wire_type, bytes_read) = read_tag(&data[pos..])?;
            pos += bytes_read;

            match (field_number, wire_type) {
                // domain (field 1, string)
                (1, 2) => {
                    let (value, bytes_read) = read_string(&data[pos..])?;
                    domain = value;
                    pos += bytes_read;
                }
                // version (field 2, varint)
                (2, 0) => {
                    let (value, bytes_read) = read_varint(&data[pos..])?;
                    version = value as i64;
                    pos += bytes_read;
                }
                (_, wire_type) => {
                    pos += skip_field(&data[pos..], wire_type)?;
                }
            }
        }

        Ok((domain, version))
    }

    /// Get model metadata.
    pub fn metadata(&self) -> &OnnxMetadata {
        &self.metadata
    }

    /// Get all initializers.
    pub fn initializers(&self) -> &[OnnxInitializer] {
        &self.initializers
    }

    /// Get initializer count.
    pub fn initializer_count(&self) -> usize {
        self.initializers.len()
    }

    /// Get initializer by name.
    pub fn get_initializer(&self, name: &str) -> Option<&OnnxInitializer> {
        self.initializers.iter().find(|i| i.name == name)
    }

    /// Get total weight size in bytes.
    pub fn total_weight_size(&self) -> usize {
        self.initializers.iter().map(|i| i.byte_size()).sum()
    }

    /// Get raw model bytes.
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_model
    }
}

/// Statistics from ONNX write operation.
#[derive(Debug, Clone, Default)]
pub struct OnnxWriteStats {
    /// Number of initializers compressed.
    pub tensors_written: usize,
    /// Original size in bytes.
    pub original_size: u64,
    /// Compressed size in bytes.
    pub compressed_size: u64,
}

impl OnnxWriteStats {
    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            0.0
        } else {
            self.original_size as f64 / self.compressed_size as f64
        }
    }
}

/// ONNX file writer.
///
/// Creates new ONNX files, optionally with compressed initializers.
pub struct OnnxWriter {
    metadata: OnnxMetadata,
    initializers: Vec<OnnxInitializer>,
    #[allow(dead_code)]
    other_data: Vec<u8>, // Reserved for preserving other graph data
}

impl OnnxWriter {
    /// Create a new empty writer.
    pub fn new() -> Self {
        Self {
            metadata: OnnxMetadata::default(),
            initializers: Vec::new(),
            other_data: Vec::new(),
        }
    }

    /// Create writer from an existing reader.
    pub fn from_reader(reader: &OnnxReader) -> Result<Self> {
        Ok(Self {
            metadata: reader.metadata.clone(),
            initializers: reader.initializers.clone(),
            other_data: Vec::new(), // TODO: preserve other graph data
        })
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: OnnxMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add an initializer.
    pub fn add_initializer(&mut self, init: OnnxInitializer) {
        self.initializers.push(init);
    }

    /// Write ONNX file.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<OnnxWriteStats> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let mut stats = OnnxWriteStats::default();

        // Build ModelProto
        let mut model_data = Vec::new();

        // ir_version (field 1)
        write_field_varint(&mut model_data, 1, self.metadata.ir_version as u64);

        // producer_name (field 2)
        if !self.metadata.producer_name.is_empty() {
            write_field_string(&mut model_data, 2, &self.metadata.producer_name);
        }

        // producer_version (field 3)
        if !self.metadata.producer_version.is_empty() {
            write_field_string(&mut model_data, 3, &self.metadata.producer_version);
        }

        // domain (field 4)
        if !self.metadata.domain.is_empty() {
            write_field_string(&mut model_data, 4, &self.metadata.domain);
        }

        // model_version (field 5)
        if self.metadata.model_version != 0 {
            write_field_varint(&mut model_data, 5, self.metadata.model_version as u64);
        }

        // doc_string (field 6)
        if !self.metadata.doc_string.is_empty() {
            write_field_string(&mut model_data, 6, &self.metadata.doc_string);
        }

        // graph (field 7)
        let graph_data = self.build_graph_proto(&mut stats)?;
        write_field_bytes(&mut model_data, 7, &graph_data);

        // opset_import (field 8)
        for (domain, version) in &self.metadata.opset_imports {
            let opset_data = build_opset_import(domain, *version);
            write_field_bytes(&mut model_data, 8, &opset_data);
        }

        writer.write_all(&model_data)?;
        stats.compressed_size = model_data.len() as u64;

        Ok(stats)
    }

    /// Build GraphProto with initializers.
    fn build_graph_proto(&self, stats: &mut OnnxWriteStats) -> Result<Vec<u8>> {
        let mut graph_data = Vec::new();

        // initializer (field 5)
        for init in &self.initializers {
            let tensor_data = self.build_tensor_proto(init)?;
            write_field_bytes(&mut graph_data, 5, &tensor_data);
            stats.tensors_written += 1;
            stats.original_size += init.byte_size() as u64;
        }

        Ok(graph_data)
    }

    /// Build TensorProto for an initializer.
    fn build_tensor_proto(&self, init: &OnnxInitializer) -> Result<Vec<u8>> {
        let mut tensor_data = Vec::new();

        // dims (field 1, packed)
        if !init.dims.is_empty() {
            let mut packed_dims = Vec::new();
            for &dim in &init.dims {
                write_varint(&mut packed_dims, dim as u64);
            }
            write_field_bytes(&mut tensor_data, 1, &packed_dims);
        }

        // data_type (field 2)
        write_field_varint(&mut tensor_data, 2, init.dtype as i32 as u64);

        // name (field 8)
        if !init.name.is_empty() {
            write_field_string(&mut tensor_data, 8, &init.name);
        }

        // raw_data (field 9)
        if !init.raw_data.is_empty() {
            write_field_bytes(&mut tensor_data, 9, &init.raw_data);
        }

        Ok(tensor_data)
    }
}

impl Default for OnnxWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Protobuf Wire Format Helpers
// ============================================================================

/// Read a varint from bytes, return (value, bytes_read).
fn read_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in data {
        bytes_read += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, bytes_read));
        }
        shift += 7;
        if shift > 63 {
            return Err(OnnxError::ProtobufError("Varint too long".into()));
        }
    }

    Err(OnnxError::ProtobufError("Unexpected end of varint".into()))
}

/// Read a tag (field number + wire type).
fn read_tag(data: &[u8]) -> Result<(u32, u32, usize)> {
    let (value, bytes_read) = read_varint(data)?;
    let field_number = (value >> 3) as u32;
    let wire_type = (value & 0x7) as u32;
    Ok((field_number, wire_type, bytes_read))
}

/// Read a length-delimited string.
fn read_string(data: &[u8]) -> Result<(String, usize)> {
    let (len, bytes_read) = read_varint(data)?;
    let len = len as usize;
    if data.len() < bytes_read + len {
        return Err(OnnxError::ProtobufError("String too long".into()));
    }
    let s = String::from_utf8_lossy(&data[bytes_read..bytes_read + len]).into_owned();
    Ok((s, bytes_read + len))
}

/// Skip a field based on wire type.
fn skip_field(data: &[u8], wire_type: u32) -> Result<usize> {
    match wire_type {
        0 => {
            // Varint
            let (_, bytes_read) = read_varint(data)?;
            Ok(bytes_read)
        }
        1 => {
            // 64-bit
            Ok(8)
        }
        2 => {
            // Length-delimited
            let (len, bytes_read) = read_varint(data)?;
            Ok(bytes_read + len as usize)
        }
        5 => {
            // 32-bit
            Ok(4)
        }
        _ => Err(OnnxError::ProtobufError(format!(
            "Unknown wire type: {}",
            wire_type
        ))),
    }
}

/// Write a varint to buffer.
fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Write a field tag.
fn write_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    write_varint(buf, ((field_number as u64) << 3) | (wire_type as u64));
}

/// Write a varint field.
fn write_field_varint(buf: &mut Vec<u8>, field_number: u32, value: u64) {
    write_tag(buf, field_number, 0);
    write_varint(buf, value);
}

/// Write a string field.
fn write_field_string(buf: &mut Vec<u8>, field_number: u32, value: &str) {
    write_tag(buf, field_number, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

/// Write a bytes field.
fn write_field_bytes(buf: &mut Vec<u8>, field_number: u32, value: &[u8]) {
    write_tag(buf, field_number, 2);
    write_varint(buf, value.len() as u64);
    buf.extend_from_slice(value);
}

/// Build OperatorSetIdProto.
fn build_opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut data = Vec::new();
    if !domain.is_empty() {
        write_field_string(&mut data, 1, domain);
    }
    write_field_varint(&mut data, 2, version as u64);
    data
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a file is an ONNX model.
///
/// Performs basic validation by checking if the file can be parsed as ONNX.
pub fn is_onnx<P: AsRef<Path>>(path: P) -> Result<bool> {
    let path = path.as_ref();

    // Check extension first
    if let Some(ext) = path.extension() {
        if ext != "onnx" {
            return Ok(false);
        }
    } else {
        return Ok(false);
    }

    // Try to parse header
    let mut file = File::open(path)?;
    let mut header = [0u8; 64];
    let bytes_read = file.read(&mut header)?;

    if bytes_read < 4 {
        return Ok(false);
    }

    // ONNX is protobuf, first byte should be a valid tag
    // Field 1 (ir_version) with wire type 0 = tag 0x08
    // Field 2 (producer_name) with wire type 2 = tag 0x12
    // etc.
    let first_byte = header[0];
    let wire_type = first_byte & 0x7;
    let field_number = first_byte >> 3;

    // Valid first fields for ModelProto: 1-8
    Ok(wire_type <= 2 && (1..=8).contains(&field_number))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_onnx_data_types() {
        assert_eq!(OnnxDataType::Float.element_size(), 4);
        assert_eq!(OnnxDataType::Float16.element_size(), 2);
        assert_eq!(OnnxDataType::BFloat16.element_size(), 2);
        assert_eq!(OnnxDataType::Int8.element_size(), 1);
        assert_eq!(OnnxDataType::Int64.element_size(), 8);
    }

    #[test]
    fn test_onnx_data_type_display() {
        assert_eq!(format!("{}", OnnxDataType::Float), "float32");
        assert_eq!(format!("{}", OnnxDataType::Float16), "float16");
        assert_eq!(format!("{}", OnnxDataType::BFloat16), "bfloat16");
    }

    #[test]
    fn test_onnx_data_type_to_dtype() {
        assert_eq!(OnnxDataType::Float.to_dtype(), Some(DType::Float32));
        assert_eq!(OnnxDataType::Float16.to_dtype(), Some(DType::Float16));
        assert_eq!(OnnxDataType::BFloat16.to_dtype(), Some(DType::BFloat16));
        assert_eq!(OnnxDataType::Int32.to_dtype(), Some(DType::Int32));
        assert_eq!(OnnxDataType::String.to_dtype(), None);
    }

    #[test]
    fn test_varint_roundtrip() {
        let test_values = [0u64, 1, 127, 128, 16383, 16384, u64::MAX];
        for value in test_values {
            let mut buf = Vec::new();
            write_varint(&mut buf, value);
            let (decoded, _) = read_varint(&buf).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_write_and_read_simple_onnx() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.onnx");

        // Create a simple ONNX model
        let mut writer = OnnxWriter::new();
        writer.metadata = OnnxMetadata {
            ir_version: 8,
            producer_name: "mithril-test".into(),
            producer_version: "1.0".into(),
            ..Default::default()
        };

        // Add a simple initializer
        writer.add_initializer(OnnxInitializer {
            name: "weight".into(),
            dtype: OnnxDataType::Float,
            dims: vec![2, 3],
            raw_data: vec![0u8; 24], // 6 floats * 4 bytes
            file_offset: None,
            file_size: None,
        });

        // Write
        let stats = writer.write(&path).unwrap();
        assert_eq!(stats.tensors_written, 1);

        // Read back
        let reader = OnnxReader::open(&path).unwrap();
        assert_eq!(reader.metadata().ir_version, 8);
        assert_eq!(reader.metadata().producer_name, "mithril-test");
        assert_eq!(reader.initializer_count(), 1);

        let init = reader.get_initializer("weight").unwrap();
        assert_eq!(init.dtype, OnnxDataType::Float);
        assert_eq!(init.dims, vec![2, 3]);
        assert_eq!(init.raw_data.len(), 24);
    }

    #[test]
    fn test_onnx_initializer_numel() {
        let init = OnnxInitializer {
            name: "test".into(),
            dtype: OnnxDataType::Float,
            dims: vec![2, 3, 4],
            raw_data: Vec::new(),
            file_offset: None,
            file_size: None,
        };
        assert_eq!(init.numel(), 24);
    }

    #[test]
    fn test_is_onnx() {
        let dir = tempdir().unwrap();

        // Create a valid ONNX file
        let onnx_path = dir.path().join("model.onnx");
        let writer = OnnxWriter::new().with_metadata(OnnxMetadata {
            ir_version: 8,
            ..Default::default()
        });
        writer.write(&onnx_path).unwrap();
        assert!(is_onnx(&onnx_path).unwrap());

        // Non-ONNX file with wrong extension
        let txt_path = dir.path().join("model.txt");
        std::fs::write(&txt_path, "not onnx").unwrap();
        assert!(!is_onnx(&txt_path).unwrap());
    }

    #[test]
    fn test_onnx_write_stats() {
        let stats = OnnxWriteStats {
            tensors_written: 10,
            original_size: 1000,
            compressed_size: 500,
        };
        assert_eq!(stats.compression_ratio(), 2.0);
    }

    #[test]
    fn test_onnx_multiple_initializers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.onnx");

        let mut writer = OnnxWriter::new().with_metadata(OnnxMetadata {
            ir_version: 9,
            producer_name: "test".into(),
            ..Default::default()
        });

        // Add multiple initializers
        for i in 0..5 {
            writer.add_initializer(OnnxInitializer {
                name: format!("layer_{}", i),
                dtype: OnnxDataType::Float16,
                dims: vec![100, 100],
                raw_data: vec![0u8; 20000], // 10000 fp16 values
                file_offset: None,
                file_size: None,
            });
        }

        let stats = writer.write(&path).unwrap();
        assert_eq!(stats.tensors_written, 5);

        let reader = OnnxReader::open(&path).unwrap();
        assert_eq!(reader.initializer_count(), 5);
        assert_eq!(reader.total_weight_size(), 100000);

        for i in 0..5 {
            let init = reader.get_initializer(&format!("layer_{}", i)).unwrap();
            assert_eq!(init.dtype, OnnxDataType::Float16);
            assert_eq!(init.dims, vec![100, 100]);
        }
    }

    #[test]
    fn test_onnx_opset_imports() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("opset.onnx");

        let mut metadata = OnnxMetadata {
            ir_version: 8,
            ..Default::default()
        };
        metadata.opset_imports.insert("".into(), 18);
        metadata.opset_imports.insert("ai.onnx.ml".into(), 3);

        let writer = OnnxWriter::new().with_metadata(metadata);
        writer.write(&path).unwrap();

        let reader = OnnxReader::open(&path).unwrap();
        assert_eq!(reader.metadata().opset_imports.get(""), Some(&18));
        assert_eq!(reader.metadata().opset_imports.get("ai.onnx.ml"), Some(&3));
    }
}
