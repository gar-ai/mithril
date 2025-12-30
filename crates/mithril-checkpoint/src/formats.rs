//! Checkpoint format readers and writers for PyTorch models.
//!
//! Supports reading and writing tensor metadata and data from:
//! - Safetensors (HuggingFace format, preferred)
//! - PyTorch state_dict (pickle format, future)

use mithril_core::types::{DType, TensorMeta};
use mithril_core::{MithrilError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Header information from a safetensors file.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Tensor metadata indexed by name
    pub tensors: HashMap<String, TensorMeta>,
    /// Total size of the header in bytes
    pub header_size: usize,
    /// Total size of the data section in bytes
    pub data_size: usize,
}

/// Reader for safetensors format files.
///
/// Safetensors format:
/// - 8 bytes: header size (little-endian u64)
/// - N bytes: JSON header with tensor metadata
/// - Remaining: tensor data (contiguous, aligned)
pub struct SafetensorsReader {
    file: BufReader<File>,
    header: SafetensorsHeader,
}

impl SafetensorsReader {
    /// Open a safetensors file and parse its header.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len() as usize;
        let mut reader = BufReader::new(file);

        // Read header size (8 bytes, little-endian)
        let mut header_size_buf = [0u8; 8];
        reader.read_exact(&mut header_size_buf)?;
        let header_size = u64::from_le_bytes(header_size_buf) as usize;

        if header_size > file_size - 8 {
            return Err(MithrilError::InvalidFormat(
                "Header size exceeds file size".to_string(),
            ));
        }

        // Read header JSON
        let mut header_buf = vec![0u8; header_size];
        reader.read_exact(&mut header_buf)?;

        let header_json: serde_json::Value = serde_json::from_slice(&header_buf)
            .map_err(|e| MithrilError::InvalidFormat(format!("Invalid JSON header: {}", e)))?;

        // Parse tensor metadata
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = header_json {
            for (name, value) in map {
                // Skip __metadata__ key
                if name == "__metadata__" {
                    continue;
                }

                let tensor_meta = Self::parse_tensor_meta(&name, &value)?;
                tensors.insert(name, tensor_meta);
            }
        }

        let data_size = file_size - 8 - header_size;

        Ok(Self {
            file: reader,
            header: SafetensorsHeader {
                tensors,
                header_size,
                data_size,
            },
        })
    }

    /// Parse tensor metadata from JSON value.
    fn parse_tensor_meta(name: &str, value: &serde_json::Value) -> Result<TensorMeta> {
        let obj = value.as_object().ok_or_else(|| {
            MithrilError::InvalidFormat(format!("Expected object for tensor '{}'", name))
        })?;

        // Parse dtype
        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MithrilError::InvalidFormat(format!("Missing dtype for '{}'", name)))?;

        let dtype = Self::parse_dtype(dtype_str)?;

        // Parse shape
        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| MithrilError::InvalidFormat(format!("Missing shape for '{}'", name)))?
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();

        // Parse data offsets
        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                MithrilError::InvalidFormat(format!("Missing data_offsets for '{}'", name))
            })?;

        let start = offsets.first().and_then(|v| v.as_u64()).ok_or_else(|| {
            MithrilError::InvalidFormat(format!("Invalid start offset for '{}'", name))
        })? as usize;

        let end = offsets.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
            MithrilError::InvalidFormat(format!("Invalid end offset for '{}'", name))
        })? as usize;

        Ok(TensorMeta {
            name: name.to_string(),
            shape,
            dtype,
            offset: start,
            size: end - start,
        })
    }

    /// Parse dtype string to DType enum.
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

    /// Get the parsed header information.
    #[must_use]
    pub fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    /// Get metadata for a specific tensor.
    #[must_use]
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.header.tensors.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.keys().map(|s| s.as_str())
    }

    /// Read tensor data by name.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let meta = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| MithrilError::NotFound(name.to_string()))?
            .clone();

        self.read_tensor_data(&meta)
    }

    /// Read tensor data using metadata.
    pub fn read_tensor_data(&mut self, meta: &TensorMeta) -> Result<Vec<u8>> {
        // Seek to tensor data (after 8-byte header size + header)
        let data_start = 8 + self.header.header_size + meta.offset;
        self.file.seek(SeekFrom::Start(data_start as u64))?;

        let mut data = vec![0u8; meta.size];
        self.file.read_exact(&mut data)?;

        Ok(data)
    }

    /// Read all tensor data as a contiguous block.
    pub fn read_all_data(&mut self) -> Result<Vec<u8>> {
        let data_start = 8 + self.header.header_size;
        self.file.seek(SeekFrom::Start(data_start as u64))?;

        let mut data = vec![0u8; self.header.data_size];
        self.file.read_exact(&mut data)?;

        Ok(data)
    }
}

/// Check if a file appears to be in safetensors format.
pub fn is_safetensors<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();

    // Check extension
    if let Some(ext) = path.extension() {
        if ext == "safetensors" {
            return true;
        }
    }

    // Try to parse header
    if let Ok(mut file) = File::open(path) {
        let mut header_size_buf = [0u8; 8];
        if file.read_exact(&mut header_size_buf).is_ok() {
            let header_size = u64::from_le_bytes(header_size_buf);
            // Sanity check: header shouldn't be too large
            if header_size > 0 && header_size < 100_000_000 {
                return true;
            }
        }
    }

    false
}

/// Raw tensor data with metadata.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor metadata
    pub meta: TensorMeta,
    /// Raw tensor bytes
    pub data: Vec<u8>,
}

/// Internal tensor info for writer.
#[derive(Debug, Clone)]
struct TensorWriteInfo {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Writer for safetensors format files.
///
/// Builds a safetensors file by collecting tensors and writing them
/// in the correct format with proper header and alignment.
pub struct SafetensorsWriter {
    tensors: Vec<TensorWriteInfo>,
    metadata: HashMap<String, String>,
}

impl SafetensorsWriter {
    /// Create a new safetensors writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add optional metadata to the file.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Add a tensor to be written.
    ///
    /// # Arguments
    /// * `name` - Tensor name (e.g., "model.layer.weight")
    /// * `data` - Raw tensor bytes
    /// * `dtype` - Data type of the tensor
    /// * `shape` - Shape of the tensor
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        dtype: DType,
        shape: Vec<usize>,
    ) {
        self.tensors.push(TensorWriteInfo {
            name: name.into(),
            dtype,
            shape,
            data,
        });
    }

    /// Convert DType to safetensors dtype string.
    fn dtype_to_string(dtype: DType) -> &'static str {
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
    }

    /// Write all tensors to the given writer.
    ///
    /// Format:
    /// - 8 bytes: header size (little-endian u64)
    /// - N bytes: JSON header with tensor metadata
    /// - Remaining: tensor data (contiguous)
    pub fn write<W: Write>(&self, writer: W) -> Result<()> {
        let mut writer = BufWriter::new(writer);

        // Build header JSON with offsets
        let mut header_map = serde_json::Map::new();

        // Add metadata if present
        if !self.metadata.is_empty() {
            let metadata_obj: serde_json::Value = self
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect::<serde_json::Map<_, _>>()
                .into();
            header_map.insert("__metadata__".to_string(), metadata_obj);
        }

        // Calculate offsets for each tensor
        let mut current_offset: usize = 0;
        for tensor in &self.tensors {
            let start = current_offset;
            let end = start + tensor.data.len();
            current_offset = end;

            let tensor_info = serde_json::json!({
                "dtype": Self::dtype_to_string(tensor.dtype),
                "shape": tensor.shape,
                "data_offsets": [start, end]
            });

            header_map.insert(tensor.name.clone(), tensor_info);
        }

        // Serialize header to JSON
        let header_json =
            serde_json::to_string(&serde_json::Value::Object(header_map)).map_err(|e| {
                MithrilError::InvalidFormat(format!("Failed to serialize header: {}", e))
            })?;

        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Write header size (8 bytes, little-endian)
        writer.write_all(&header_size.to_le_bytes())?;

        // Write header JSON
        writer.write_all(header_bytes)?;

        // Write tensor data in order
        for tensor in &self.tensors {
            writer.write_all(&tensor.data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Write to a file at the given path.
    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        self.write(file)
    }

    /// Get the number of tensors added.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get the total size of tensor data.
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.tensors.iter().map(|t| t.data.len()).sum()
    }
}

impl Default for SafetensorsWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert_eq!(
            SafetensorsReader::parse_dtype("BF16").unwrap(),
            DType::BFloat16
        );
        assert_eq!(
            SafetensorsReader::parse_dtype("F32").unwrap(),
            DType::Float32
        );
        assert_eq!(SafetensorsReader::parse_dtype("I8").unwrap(), DType::Int8);
        assert!(SafetensorsReader::parse_dtype("UNKNOWN").is_err());
    }

    #[test]
    fn test_parse_tensor_meta() {
        let json = serde_json::json!({
            "dtype": "BF16",
            "shape": [512, 768],
            "data_offsets": [0, 786432]
        });

        let meta = SafetensorsReader::parse_tensor_meta("test_tensor", &json).unwrap();
        assert_eq!(meta.name, "test_tensor");
        assert_eq!(meta.dtype, DType::BFloat16);
        assert_eq!(meta.shape, vec![512, 768]);
        assert_eq!(meta.offset, 0);
        assert_eq!(meta.size, 786432);
    }

    #[test]
    fn test_is_safetensors_by_extension() {
        assert!(is_safetensors(Path::new("model.safetensors")));
        assert!(!is_safetensors(Path::new("model.pt")));
    }

    #[test]
    fn test_safetensors_writer_basic() {
        let mut writer = SafetensorsWriter::new();

        // Add a simple f32 tensor (2x3 = 6 elements, 24 bytes)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor("test_tensor", bytes, DType::Float32, vec![2, 3]);

        assert_eq!(writer.tensor_count(), 1);
        assert_eq!(writer.data_size(), 24);
    }

    #[test]
    fn test_safetensors_roundtrip_single_tensor() {
        use tempfile::NamedTempFile;

        let mut writer = SafetensorsWriter::new();

        // Create test tensor data (4x4 bf16 = 32 bytes)
        let original_data: Vec<u8> = (0..32).collect();
        writer.add_tensor(
            "layer.weight",
            original_data.clone(),
            DType::BFloat16,
            vec![4, 4],
        );

        // Write to temp file
        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        // Read back
        let mut reader = SafetensorsReader::open(temp_file.path()).unwrap();

        // Verify header
        let header = reader.header();
        assert_eq!(header.tensors.len(), 1);

        // Verify tensor metadata
        let meta = reader.tensor_meta("layer.weight").unwrap();
        assert_eq!(meta.dtype, DType::BFloat16);
        assert_eq!(meta.shape, vec![4, 4]);
        assert_eq!(meta.size, 32);

        // Verify tensor data
        let read_data = reader.read_tensor("layer.weight").unwrap();
        assert_eq!(read_data, original_data);
    }

    #[test]
    fn test_safetensors_roundtrip_multiple_tensors() {
        use tempfile::NamedTempFile;

        let mut writer = SafetensorsWriter::new();

        // Add multiple tensors of different types
        let tensor1: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 4x2 int8
        let tensor2: Vec<u8> = (0..48).collect(); // 3x4 fp32 (12 * 4 bytes)
        let tensor3: Vec<u8> = (0..16).collect(); // 4x2 fp16 (8 * 2 bytes)

        writer.add_tensor("encoder.embed", tensor1.clone(), DType::Int8, vec![4, 2]);
        writer.add_tensor(
            "decoder.weight",
            tensor2.clone(),
            DType::Float32,
            vec![3, 4],
        );
        writer.add_tensor("output.bias", tensor3.clone(), DType::Float16, vec![4, 2]);

        // Write to temp file
        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        // Read back
        let mut reader = SafetensorsReader::open(temp_file.path()).unwrap();

        // Verify all tensors
        assert_eq!(reader.header().tensors.len(), 3);

        let names: Vec<&str> = reader.tensor_names().collect();
        assert!(names.contains(&"encoder.embed"));
        assert!(names.contains(&"decoder.weight"));
        assert!(names.contains(&"output.bias"));

        // Verify data for each tensor
        assert_eq!(reader.read_tensor("encoder.embed").unwrap(), tensor1);
        assert_eq!(reader.read_tensor("decoder.weight").unwrap(), tensor2);
        assert_eq!(reader.read_tensor("output.bias").unwrap(), tensor3);
    }

    #[test]
    fn test_safetensors_writer_with_metadata() {
        use tempfile::NamedTempFile;

        let mut writer = SafetensorsWriter::new();
        writer.add_metadata("format", "pt");
        writer.add_metadata("model_name", "test_model");

        let data: Vec<u8> = vec![0; 16];
        writer.add_tensor("weight", data, DType::Float32, vec![2, 2]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        // Read back and verify we can still read tensors
        // (metadata is in __metadata__ key which is skipped by reader)
        let reader = SafetensorsReader::open(temp_file.path()).unwrap();
        assert_eq!(reader.header().tensors.len(), 1);
        assert!(reader.tensor_meta("weight").is_some());
    }

    #[test]
    fn test_safetensors_empty_tensor() {
        use tempfile::NamedTempFile;

        let mut writer = SafetensorsWriter::new();

        // Empty tensor (e.g., placeholder)
        writer.add_tensor("empty", Vec::new(), DType::Float32, vec![0]);

        let temp_file = NamedTempFile::new().unwrap();
        writer.write_file(temp_file.path()).unwrap();

        let mut reader = SafetensorsReader::open(temp_file.path()).unwrap();
        let meta = reader.tensor_meta("empty").unwrap();
        assert_eq!(meta.shape, vec![0]);
        assert_eq!(meta.size, 0);

        let data = reader.read_tensor("empty").unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_safetensors_dtype_roundtrip() {
        use tempfile::NamedTempFile;

        // Test all dtype conversions
        let dtypes = [
            (DType::Float32, "F32"),
            (DType::Float16, "F16"),
            (DType::BFloat16, "BF16"),
            (DType::Float64, "F64"),
            (DType::Int32, "I32"),
            (DType::Int64, "I64"),
            (DType::Int8, "I8"),
            (DType::UInt8, "U8"),
            (DType::Bool, "BOOL"),
        ];

        for (dtype, _name) in dtypes {
            let mut writer = SafetensorsWriter::new();
            writer.add_tensor("test", vec![0, 1, 2, 3], dtype, vec![4]);

            let temp_file = NamedTempFile::new().unwrap();
            writer.write_file(temp_file.path()).unwrap();

            let reader = SafetensorsReader::open(temp_file.path()).unwrap();
            let meta = reader.tensor_meta("test").unwrap();
            assert_eq!(meta.dtype, dtype, "dtype mismatch for {:?}", dtype);
        }
    }
}
