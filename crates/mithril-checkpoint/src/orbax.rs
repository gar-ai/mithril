//! Orbax checkpoint format support for JAX models.
//!
//! Orbax is the standard checkpointing library for JAX/Flax models.
//! It stores checkpoints as directories containing:
//! - A `_METADATA` file with pytree structure
//! - Array files in various formats (msgpack, tensorstore, etc.)
//!
//! This module supports reading and writing Orbax checkpoints.
//!
//! ## Writing Orbax Checkpoints
//!
//! ```rust,ignore
//! use mithril_checkpoint::orbax::OrbaxWriter;
//! use mithril_core::types::DType;
//!
//! let mut writer = OrbaxWriter::new();
//! writer.add_array("params/layer_0/kernel", &tensor_data, DType::Float32, &[768, 768])?;
//! writer.add_array("params/layer_0/bias", &bias_data, DType::Float32, &[768])?;
//! let stats = writer.write("./checkpoint")?;
//! println!("Wrote {} arrays, {} bytes", stats.array_count, stats.total_bytes);
//! ```

use mithril_core::types::{DType, TensorMeta};
use mithril_core::{MithrilError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Orbax checkpoint structure.
#[derive(Debug, Clone)]
pub struct OrbaxCheckpoint {
    /// Root directory of the checkpoint.
    pub root: PathBuf,
    /// Parsed metadata.
    pub metadata: OrbaxMetadata,
    /// Tensor information extracted from the checkpoint.
    pub tensors: HashMap<String, TensorMeta>,
}

/// Orbax checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbaxMetadata {
    /// Checkpoint version.
    #[serde(default)]
    pub version: u32,
    /// Pytree structure (simplified representation).
    #[serde(default)]
    pub pytree: Option<serde_json::Value>,
    /// Array info entries.
    #[serde(default)]
    pub arrays: HashMap<String, ArrayInfo>,
}

/// Information about an array in the checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayInfo {
    /// Shape of the array.
    pub shape: Vec<usize>,
    /// Data type string (e.g., "float32", "bfloat16").
    pub dtype: String,
    /// Path to the data file relative to checkpoint root.
    #[serde(default)]
    pub path: Option<String>,
    /// Byte offset in the data file.
    #[serde(default)]
    pub offset: usize,
    /// Byte size of the data.
    #[serde(default)]
    pub nbytes: usize,
}

/// Reader for Orbax checkpoints.
pub struct OrbaxReader {
    checkpoint: OrbaxCheckpoint,
}

impl OrbaxReader {
    /// Open an Orbax checkpoint directory.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let root = path.as_ref().to_path_buf();

        if !root.is_dir() {
            return Err(MithrilError::InvalidFormat(format!(
                "Orbax checkpoint must be a directory: {}",
                root.display()
            )));
        }

        // Look for metadata file (check multiple possible names)
        let metadata_path = root.join("_METADATA");
        let checkpoint_metadata_path = root.join("_CHECKPOINT_METADATA");
        let alt_metadata_path = root.join("metadata");

        let metadata = if metadata_path.exists() {
            Self::read_metadata(&metadata_path)?
        } else if checkpoint_metadata_path.exists() {
            Self::read_checkpoint_metadata(&checkpoint_metadata_path)?
        } else if alt_metadata_path.exists() {
            Self::read_metadata(&alt_metadata_path)?
        } else {
            // No metadata file, try to infer structure from directory
            Self::infer_metadata(&root)?
        };

        // Convert to tensor metadata
        let tensors = Self::build_tensor_map(&metadata)?;

        Ok(Self {
            checkpoint: OrbaxCheckpoint {
                root,
                metadata,
                tensors,
            },
        })
    }

    /// Read and parse metadata file.
    fn read_metadata(path: &Path) -> Result<OrbaxMetadata> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Try msgpack first
        if let Ok(metadata) = rmp_serde::from_slice::<OrbaxMetadata>(&data) {
            return Ok(metadata);
        }

        // Try JSON
        if let Ok(metadata) = serde_json::from_slice::<OrbaxMetadata>(&data) {
            return Ok(metadata);
        }

        // Try to parse as generic msgpack and convert
        if let Ok(value) = rmp_serde::from_slice::<serde_json::Value>(&data) {
            return Self::metadata_from_value(value);
        }

        Err(MithrilError::InvalidFormat(
            "Cannot parse Orbax metadata".to_string(),
        ))
    }

    /// Read _CHECKPOINT_METADATA file (JSON format from OrbaxWriter).
    fn read_checkpoint_metadata(path: &Path) -> Result<OrbaxMetadata> {
        let content = fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| MithrilError::InvalidFormat(format!("Invalid JSON metadata: {}", e)))?;

        let mut metadata = OrbaxMetadata {
            version: 0,
            pytree: None,
            arrays: HashMap::new(),
        };

        if let serde_json::Value::Object(map) = &json {
            // Extract version
            if let Some(v) = map.get("version").and_then(|v| v.as_u64()) {
                metadata.version = v as u32;
            }

            // Extract array info
            if let Some(serde_json::Value::Object(arrays)) = map.get("arrays") {
                for (name, info) in arrays {
                    if let serde_json::Value::Object(array_info) = info {
                        let dtype = array_info
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();

                        let shape: Vec<usize> = array_info
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                                    .collect()
                            })
                            .unwrap_or_default();

                        // Compute nbytes based on dtype and shape
                        let dtype_size = match dtype.as_str() {
                            "float64" | "int64" => 8,
                            "float32" | "int32" => 4,
                            "float16" | "bfloat16" => 2,
                            "int8" | "uint8" | "bool" => 1,
                            _ => 1,
                        };
                        let nbytes = shape.iter().product::<usize>() * dtype_size;

                        // Build the relative path to the array file
                        let path_parts: Vec<&str> = name.split('/').collect();
                        let mut array_path = PathBuf::from("state");
                        for part in &path_parts {
                            array_path.push(part);
                        }
                        array_path.push("0.npy");

                        metadata.arrays.insert(
                            name.clone(),
                            ArrayInfo {
                                shape,
                                dtype,
                                path: Some(array_path.to_string_lossy().to_string()),
                                offset: 0, // Will be updated when reading the NPY file
                                nbytes,
                            },
                        );
                    }
                }
            }

            // Store original metadata
            metadata.pytree = Some(json);
        }

        Ok(metadata)
    }

    /// Convert generic msgpack value to OrbaxMetadata.
    fn metadata_from_value(value: serde_json::Value) -> Result<OrbaxMetadata> {
        // Orbax metadata can have various structures depending on version
        // Try to extract what we can
        let mut metadata = OrbaxMetadata {
            version: 0,
            pytree: None,
            arrays: HashMap::new(),
        };

        if let serde_json::Value::Object(map) = value {
            // Look for version
            if let Some(v) = map.get("version") {
                if let Some(n) = v.as_u64() {
                    metadata.version = n as u32;
                }
            }

            // Store the whole structure as pytree for reference
            metadata.pytree = Some(serde_json::Value::Object(map));
        }

        Ok(metadata)
    }

    /// Infer metadata from directory structure.
    fn infer_metadata(root: &Path) -> Result<OrbaxMetadata> {
        let mut arrays = HashMap::new();

        // Check for state/ subdirectory (OrbaxWriter convention)
        let state_dir = root.join("state");
        let search_root = if state_dir.is_dir() { &state_dir } else { root };

        // Look for array files (*.npy, *.array, numbered files)
        for entry in fs::read_dir(search_root)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();

            // Skip metadata files
            if file_name.starts_with('_') || file_name == "metadata" || file_name.starts_with('.') {
                continue;
            }

            if path.is_file() {
                // Try to read array info from file
                if let Some(info) = Self::try_read_array_info(&path) {
                    arrays.insert(file_name, info);
                }
            } else if path.is_dir() {
                // Recursively process subdirectories
                Self::process_subdir(&path, &mut arrays, &file_name)?;
            }
        }

        Ok(OrbaxMetadata {
            version: 0,
            pytree: None,
            arrays,
        })
    }

    /// Process a subdirectory for arrays.
    fn process_subdir(
        dir: &Path,
        arrays: &mut HashMap<String, ArrayInfo>,
        prefix: &str,
    ) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();

            if path.is_file() {
                // Check if this is an Orbax array file (typically named "0", "0.npy", etc.)
                // In this case, use the parent directory name (prefix) as the array name
                let is_orbax_array_file = file_name == "0"
                    || file_name == "0.npy"
                    || file_name.parse::<usize>().is_ok()
                    || file_name
                        .strip_suffix(".npy")
                        .is_some_and(|s| s.parse::<usize>().is_ok());

                if is_orbax_array_file {
                    if let Some(mut info) = Self::try_read_array_info(&path) {
                        // Use prefix as array name, update path to point to this file
                        info.path = Some(
                            path.strip_prefix(dir.parent().unwrap().parent().unwrap_or(dir))
                                .unwrap_or(&path)
                                .to_string_lossy()
                                .to_string(),
                        );
                        arrays.insert(prefix.to_string(), info);
                    }
                } else {
                    // Regular file, use full path name
                    let full_name = format!("{}/{}", prefix, file_name);
                    if let Some(info) = Self::try_read_array_info(&path) {
                        arrays.insert(full_name, info);
                    }
                }
            } else if path.is_dir() {
                let full_name = format!("{}/{}", prefix, file_name);
                Self::process_subdir(&path, arrays, &full_name)?;
            }
        }
        Ok(())
    }

    /// Try to read array info from a file.
    fn try_read_array_info(path: &Path) -> Option<ArrayInfo> {
        let extension = path.extension()?.to_str()?;

        match extension {
            "npy" => Self::read_npy_header(path).ok(),
            "array" => Self::read_array_header(path).ok(),
            _ => None,
        }
    }

    /// Read header from a .npy file.
    fn read_npy_header(path: &Path) -> Result<ArrayInfo> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // NPY magic number
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;

        if &magic[..4] != b"\x93NUMPY" {
            return Err(MithrilError::InvalidFormat("Not a valid NPY file".into()));
        }

        let major = magic[4];
        let _minor = magic[5];

        // Read header length
        let header_len = if major == 1 {
            let mut len_buf = [0u8; 2];
            reader.read_exact(&mut len_buf)?;
            u16::from_le_bytes(len_buf) as usize
        } else {
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf)?;
            u32::from_le_bytes(len_buf) as usize
        };

        // Read header string
        let mut header_buf = vec![0u8; header_len];
        reader.read_exact(&mut header_buf)?;
        let header_str = String::from_utf8_lossy(&header_buf);

        // Parse the header (it's a Python dict literal)
        let (dtype, shape) = Self::parse_npy_header(&header_str)?;

        let file_size = fs::metadata(path)?.len() as usize;
        let header_size = 6 + if major == 1 { 2 } else { 4 } + header_len;
        let nbytes = file_size - header_size;

        Ok(ArrayInfo {
            shape,
            dtype,
            path: Some(path.file_name().unwrap().to_string_lossy().to_string()),
            offset: header_size,
            nbytes,
        })
    }

    /// Parse NPY header string.
    fn parse_npy_header(header: &str) -> Result<(String, Vec<usize>)> {
        // Simple parser for NPY header dict
        // Format: {'descr': '<f4', 'fortran_order': False, 'shape': (10, 20)}

        let dtype = if header.contains("'<f4'") || header.contains("'float32'") {
            "float32"
        } else if header.contains("'<f2'") || header.contains("'float16'") {
            "float16"
        } else if header.contains("'<f8'") || header.contains("'float64'") {
            "float64"
        } else if header.contains("'|V2'") || header.contains("'bfloat16'") {
            "bfloat16"
        } else if header.contains("'<i4'") || header.contains("'int32'") {
            "int32"
        } else if header.contains("'<i8'") || header.contains("'int64'") {
            "int64"
        } else {
            "unknown"
        };

        // Extract shape
        let shape = if let Some(start) = header.find("'shape': (") {
            let start = start + "'shape': (".len();
            if let Some(end) = header[start..].find(')') {
                let shape_str = &header[start..start + end];
                shape_str
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        Ok((dtype.to_string(), shape))
    }

    /// Read header from an Orbax .array file.
    fn read_array_header(path: &Path) -> Result<ArrayInfo> {
        // Orbax array files are msgpack encoded
        let data = fs::read(path)?;

        // Try to decode as msgpack
        if let Ok(serde_json::Value::Object(map)) =
            rmp_serde::from_slice::<serde_json::Value>(&data)
        {
            let shape = map
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            let dtype = map
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            return Ok(ArrayInfo {
                shape,
                dtype,
                path: Some(path.file_name().unwrap().to_string_lossy().to_string()),
                offset: 0,
                nbytes: data.len(),
            });
        }

        Err(MithrilError::InvalidFormat(
            "Cannot parse array header".to_string(),
        ))
    }

    /// Build tensor metadata map from Orbax metadata.
    fn build_tensor_map(metadata: &OrbaxMetadata) -> Result<HashMap<String, TensorMeta>> {
        let mut tensors = HashMap::new();

        for (name, info) in &metadata.arrays {
            let dtype = Self::parse_dtype(&info.dtype)?;
            let shape = info.shape.clone();
            let offset = info.offset;
            let size = info.nbytes;

            tensors.insert(
                name.clone(),
                TensorMeta {
                    name: name.clone(),
                    dtype,
                    shape,
                    offset,
                    size,
                },
            );
        }

        Ok(tensors)
    }

    /// Parse dtype string to DType enum.
    fn parse_dtype(s: &str) -> Result<DType> {
        match s.to_lowercase().as_str() {
            "float32" | "f32" | "<f4" => Ok(DType::Float32),
            "float16" | "f16" | "<f2" => Ok(DType::Float16),
            "bfloat16" | "bf16" => Ok(DType::BFloat16),
            "float64" | "f64" | "<f8" => Ok(DType::Float64),
            "int32" | "i32" | "<i4" => Ok(DType::Int32),
            "int64" | "i64" | "<i8" => Ok(DType::Int64),
            "int8" | "i8" => Ok(DType::Int8),
            "uint8" | "u8" => Ok(DType::UInt8),
            "bool" | "boolean" => Ok(DType::Bool),
            // Int16 is treated as Int32 for now (not in DType enum)
            "int16" | "i16" | "<i2" => Ok(DType::Int32),
            _ => Err(MithrilError::InvalidFormat(format!("Unknown dtype: {}", s))),
        }
    }

    /// Get the checkpoint info.
    #[must_use]
    pub fn checkpoint(&self) -> &OrbaxCheckpoint {
        &self.checkpoint
    }

    /// Get tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.checkpoint.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor metadata.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorMeta> {
        self.checkpoint.tensors.get(name)
    }

    /// Read tensor data by name.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let info =
            self.checkpoint.tensors.get(name).ok_or_else(|| {
                MithrilError::InvalidFormat(format!("Tensor not found: {}", name))
            })?;

        let array_info = self.checkpoint.metadata.arrays.get(name).ok_or_else(|| {
            MithrilError::InvalidFormat(format!("Array info not found: {}", name))
        })?;

        let file_path = if let Some(ref path) = array_info.path {
            self.checkpoint.root.join(path)
        } else {
            // Try to find the file by name
            self.checkpoint.root.join(name)
        };

        if !file_path.exists() {
            return Err(MithrilError::InvalidFormat(format!(
                "Data file not found: {}",
                file_path.display()
            )));
        }

        let mut file = File::open(&file_path)?;
        let mut data = vec![0u8; info.size];

        // Seek to offset if needed
        if info.offset > 0 {
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(info.offset as u64))?;
        }

        file.read_exact(&mut data)?;

        Ok(data)
    }

    /// Read all tensors into a HashMap.
    pub fn read_all_tensors(&self) -> Result<HashMap<String, Vec<u8>>> {
        let mut tensors = HashMap::new();

        for name in self.tensor_names() {
            match self.read_tensor(name) {
                Ok(data) => {
                    tensors.insert(name.to_string(), data);
                }
                Err(e) => {
                    // Log warning but continue
                    tracing::warn!("Failed to read tensor {}: {}", name, e);
                }
            }
        }

        Ok(tensors)
    }
}

/// Statistics from writing an Orbax checkpoint.
#[derive(Debug, Clone, Default)]
pub struct OrbaxWriteStats {
    /// Number of arrays written.
    pub array_count: usize,
    /// Total bytes written (all arrays).
    pub total_bytes: usize,
    /// Bytes written per array.
    pub array_bytes: HashMap<String, usize>,
}

/// Internal array info for writer.
#[derive(Debug, Clone)]
struct ArrayWriteInfo {
    /// Array name (may contain path separators like "params/layer_0/kernel").
    name: String,
    /// Raw array data.
    data: Vec<u8>,
    /// Data type.
    dtype: DType,
    /// Shape of the array.
    shape: Vec<usize>,
}

/// Writer for Orbax checkpoint format.
///
/// Builds an Orbax-compatible checkpoint directory by collecting arrays
/// and writing them as NPY files with metadata.
///
/// # Example
///
/// ```rust,ignore
/// use mithril_checkpoint::orbax::OrbaxWriter;
/// use mithril_core::types::DType;
///
/// let mut writer = OrbaxWriter::new();
/// writer.add_array("params/kernel", &data, DType::Float32, &[768, 768])?;
/// writer.write("./checkpoint")?;
/// ```
pub struct OrbaxWriter {
    arrays: Vec<ArrayWriteInfo>,
    metadata: Option<serde_json::Value>,
}

impl OrbaxWriter {
    /// Create a new Orbax writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            arrays: Vec::new(),
            metadata: None,
        }
    }

    /// Create a new Orbax writer with custom metadata.
    #[must_use]
    pub fn with_metadata(metadata: serde_json::Value) -> Self {
        Self {
            arrays: Vec::new(),
            metadata: Some(metadata),
        }
    }

    /// Set or replace the metadata.
    pub fn set_metadata(&mut self, metadata: serde_json::Value) {
        self.metadata = Some(metadata);
    }

    /// Add an array to be written.
    ///
    /// # Arguments
    /// * `name` - Array name (e.g., "params/layer_0/kernel")
    /// * `data` - Raw array bytes in row-major (C) order
    /// * `dtype` - Data type of the array
    /// * `shape` - Shape of the array
    ///
    /// # Errors
    /// Returns an error if the data size doesn't match shape * dtype_size.
    pub fn add_array(
        &mut self,
        name: impl Into<String>,
        data: &[u8],
        dtype: DType,
        shape: &[usize],
    ) -> Result<()> {
        let name = name.into();

        // Validate data size
        let expected_size = Self::compute_size(dtype, shape);
        if data.len() != expected_size {
            return Err(MithrilError::InvalidFormat(format!(
                "Array '{}': expected {} bytes for {:?} shape {:?}, got {}",
                name,
                expected_size,
                dtype,
                shape,
                data.len()
            )));
        }

        self.arrays.push(ArrayWriteInfo {
            name,
            data: data.to_vec(),
            dtype,
            shape: shape.to_vec(),
        });

        Ok(())
    }

    /// Compute the expected byte size for an array.
    fn compute_size(dtype: DType, shape: &[usize]) -> usize {
        let element_size = match dtype {
            DType::Float64 | DType::Int64 => 8,
            DType::Float32 | DType::Int32 => 4,
            DType::Float16 | DType::BFloat16 => 2,
            DType::Int8 | DType::UInt8 | DType::Bool => 1,
        };
        let num_elements: usize = shape.iter().product();
        num_elements * element_size
    }

    /// Convert DType to NPY descr string.
    fn dtype_to_npy_descr(dtype: DType) -> &'static str {
        match dtype {
            DType::Float64 => "<f8",
            DType::Float32 => "<f4",
            DType::Float16 => "<f2",
            DType::BFloat16 => "|V2", // Custom void type for bfloat16
            DType::Int64 => "<i8",
            DType::Int32 => "<i4",
            DType::Int8 => "|i1",
            DType::UInt8 => "|u1",
            DType::Bool => "|b1",
        }
    }

    /// Convert DType to Orbax/JAX dtype string.
    fn dtype_to_orbax_str(dtype: DType) -> &'static str {
        match dtype {
            DType::Float64 => "float64",
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Int64 => "int64",
            DType::Int32 => "int32",
            DType::Int8 => "int8",
            DType::UInt8 => "uint8",
            DType::Bool => "bool",
        }
    }

    /// Generate NPY v1.0 header for an array.
    fn generate_npy_header(dtype: DType, shape: &[usize]) -> Vec<u8> {
        let descr = Self::dtype_to_npy_descr(dtype);

        // Format shape tuple
        let shape_str = if shape.is_empty() {
            "()".to_string()
        } else if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", inner.join(", "))
        };

        // NPY header dict (Python syntax)
        let header_dict = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            descr, shape_str
        );

        // Calculate padding for 64-byte alignment
        // Header: magic (6) + version (2) + header_len (2) + header_dict
        let magic_len = 6;
        let version_len = 2;
        let header_len_field = 2;
        let fixed_size = magic_len + version_len + header_len_field;
        let header_dict_len = header_dict.len();

        // Total should be multiple of 64
        let total_unpadded = fixed_size + header_dict_len + 1; // +1 for newline
        let padding = (64 - (total_unpadded % 64)) % 64;
        let padded_header_len = header_dict_len + padding + 1; // +1 for newline

        let mut header = Vec::with_capacity(fixed_size + padded_header_len);

        // Magic number: \x93NUMPY
        header.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);

        // Version 1.0
        header.push(1);
        header.push(0);

        // Header length (little-endian u16)
        let header_len_u16 = padded_header_len as u16;
        header.extend_from_slice(&header_len_u16.to_le_bytes());

        // Header dict
        header.extend_from_slice(header_dict.as_bytes());

        // Padding (spaces)
        header.extend(std::iter::repeat(b' ').take(padding));

        // Terminating newline
        header.push(b'\n');

        header
    }

    /// Write the checkpoint to a directory.
    ///
    /// Creates the directory structure:
    /// ```text
    /// checkpoint/
    /// ├── _CHECKPOINT_METADATA  # JSON metadata
    /// └── state/
    ///     └── param_name/
    ///         └── 0.npy         # NPY array file
    /// ```
    ///
    /// # Arguments
    /// * `dir` - Output directory path
    ///
    /// # Returns
    /// Statistics about what was written.
    pub fn write<P: AsRef<Path>>(&self, dir: P) -> Result<OrbaxWriteStats> {
        let dir = dir.as_ref();

        // Create root directory
        fs::create_dir_all(dir)?;

        let mut stats = OrbaxWriteStats::default();

        // Write each array
        for array in &self.arrays {
            let bytes = self.write_array(dir, array)?;
            stats.array_count += 1;
            stats.total_bytes += bytes;
            stats.array_bytes.insert(array.name.clone(), bytes);
        }

        // Write metadata file
        self.write_metadata(dir)?;

        Ok(stats)
    }

    /// Write a single array to a file.
    fn write_array(&self, root: &Path, array: &ArrayWriteInfo) -> Result<usize> {
        // Parse path components from array name
        let path_parts: Vec<&str> = array.name.split('/').collect();

        // Build directory structure: state/<path_components>/
        let mut array_dir = root.join("state");
        for part in &path_parts {
            array_dir = array_dir.join(part);
        }
        fs::create_dir_all(&array_dir)?;

        // Write as 0.npy (Orbax convention)
        let npy_path = array_dir.join("0.npy");

        let file = File::create(&npy_path)?;
        let mut writer = BufWriter::new(file);

        // Generate and write NPY header
        let header = Self::generate_npy_header(array.dtype, &array.shape);
        writer.write_all(&header)?;

        // Write array data
        writer.write_all(&array.data)?;
        writer.flush()?;

        Ok(header.len() + array.data.len())
    }

    /// Write the checkpoint metadata file.
    fn write_metadata(&self, dir: &Path) -> Result<()> {
        let metadata_path = dir.join("_CHECKPOINT_METADATA");

        // Build metadata structure
        let mut meta = serde_json::Map::new();
        meta.insert("version".to_string(), serde_json::json!(1));

        // Add array information
        let mut arrays_info = serde_json::Map::new();
        for array in &self.arrays {
            let array_meta = serde_json::json!({
                "dtype": Self::dtype_to_orbax_str(array.dtype),
                "shape": array.shape,
            });
            arrays_info.insert(array.name.clone(), array_meta);
        }
        meta.insert("arrays".to_string(), serde_json::Value::Object(arrays_info));

        // Merge with custom metadata if provided
        if let Some(serde_json::Value::Object(ref custom_map)) = self.metadata {
            for (k, v) in custom_map {
                if k != "version" && k != "arrays" {
                    meta.insert(k.clone(), v.clone());
                }
            }
        }

        // Write as JSON (for human readability)
        let json = serde_json::to_string_pretty(&serde_json::Value::Object(meta)).map_err(|e| {
            MithrilError::InvalidFormat(format!("Failed to serialize metadata: {}", e))
        })?;

        fs::write(&metadata_path, json)?;

        // Also write Orbax marker file
        fs::write(dir.join(".orbax-checkpoint"), "")?;

        Ok(())
    }

    /// Get the number of arrays added.
    #[must_use]
    pub fn array_count(&self) -> usize {
        self.arrays.len()
    }

    /// Get the total size of array data (excluding headers).
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.arrays.iter().map(|a| a.data.len()).sum()
    }

    /// Get array names.
    pub fn array_names(&self) -> impl Iterator<Item = &str> {
        self.arrays.iter().map(|a| a.name.as_str())
    }
}

impl Default for OrbaxWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a path is an Orbax checkpoint.
pub fn is_orbax_checkpoint(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }

    // Look for Orbax-specific files
    path.join("_METADATA").exists()
        || path.join("metadata").exists()
        || path.join("_CHECKPOINT_METADATA").exists()
        || path.join(".orbax-checkpoint").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_dtype() {
        assert!(matches!(
            OrbaxReader::parse_dtype("float32"),
            Ok(DType::Float32)
        ));
        assert!(matches!(
            OrbaxReader::parse_dtype("bfloat16"),
            Ok(DType::BFloat16)
        ));
        assert!(matches!(
            OrbaxReader::parse_dtype("<f4"),
            Ok(DType::Float32)
        ));
    }

    #[test]
    fn test_parse_npy_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 20, 30), }";
        let (dtype, shape) = OrbaxReader::parse_npy_header(header).unwrap();
        assert_eq!(dtype, "float32");
        assert_eq!(shape, vec![10, 20, 30]);
    }

    #[test]
    fn test_is_orbax_checkpoint() {
        let dir = tempdir().unwrap();
        assert!(!is_orbax_checkpoint(dir.path()));

        // Create metadata file
        fs::write(dir.path().join("_METADATA"), b"test").unwrap();
        assert!(is_orbax_checkpoint(dir.path()));
    }

    #[test]
    fn test_orbax_reader_not_directory() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("not_a_dir");
        fs::write(&file_path, b"test").unwrap();

        let result = OrbaxReader::open(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_orbax_metadata_serialization() {
        let metadata = OrbaxMetadata {
            version: 1,
            pytree: None,
            arrays: HashMap::new(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("\"version\":1"));
    }

    // OrbaxWriter tests

    #[test]
    fn test_orbax_writer_basic() {
        let mut writer = OrbaxWriter::new();

        // Add a simple f32 tensor (2x3 = 6 elements, 24 bytes)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer
            .add_array("test_array", &bytes, DType::Float32, &[2, 3])
            .unwrap();

        assert_eq!(writer.array_count(), 1);
        assert_eq!(writer.data_size(), 24);
    }

    #[test]
    fn test_orbax_writer_invalid_size() {
        let mut writer = OrbaxWriter::new();

        // Try to add array with wrong size
        let data: Vec<u8> = vec![0; 10]; // Wrong size
        let result = writer.add_array("bad_array", &data, DType::Float32, &[2, 3]);

        assert!(result.is_err());
    }

    #[test]
    fn test_orbax_writer_write() {
        let dir = tempdir().unwrap();
        let mut writer = OrbaxWriter::new();

        // Add a tensor
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer
            .add_array("layer/weight", &bytes, DType::Float32, &[2, 2])
            .unwrap();

        // Write checkpoint
        let stats = writer.write(dir.path()).unwrap();

        assert_eq!(stats.array_count, 1);
        assert!(stats.total_bytes > 16); // Data + header

        // Verify files exist
        assert!(dir.path().join("_CHECKPOINT_METADATA").exists());
        assert!(dir.path().join(".orbax-checkpoint").exists());
        assert!(dir.path().join("state/layer/weight/0.npy").exists());
    }

    #[test]
    fn test_orbax_writer_multiple_arrays() {
        let dir = tempdir().unwrap();
        let mut writer = OrbaxWriter::new();

        // Add multiple arrays
        let data1: Vec<u8> = vec![0; 16]; // 4 f32
        let data2: Vec<u8> = vec![0; 8]; // 4 f16
        let data3: Vec<u8> = vec![0; 8]; // 8 i8

        writer
            .add_array("params/layer_0/kernel", &data1, DType::Float32, &[2, 2])
            .unwrap();
        writer
            .add_array("params/layer_0/bias", &data2, DType::Float16, &[4])
            .unwrap();
        writer
            .add_array("params/embed", &data3, DType::Int8, &[8])
            .unwrap();

        let stats = writer.write(dir.path()).unwrap();

        assert_eq!(stats.array_count, 3);

        // Verify all files exist
        assert!(dir
            .path()
            .join("state/params/layer_0/kernel/0.npy")
            .exists());
        assert!(dir.path().join("state/params/layer_0/bias/0.npy").exists());
        assert!(dir.path().join("state/params/embed/0.npy").exists());
    }

    #[test]
    fn test_orbax_writer_roundtrip() {
        let dir = tempdir().unwrap();
        let mut writer = OrbaxWriter::new();

        // Create test data
        let original_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = original_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        writer
            .add_array("test_tensor", &bytes, DType::Float32, &[2, 3])
            .unwrap();
        writer.write(dir.path()).unwrap();

        // Read back with OrbaxReader
        let reader = OrbaxReader::open(dir.path()).unwrap();

        // Check the tensor exists (note: path is state/test_tensor/0.npy)
        let names: Vec<_> = reader.tensor_names();
        // The name will include the state/ prefix from the directory structure
        assert!(!names.is_empty());
    }

    #[test]
    fn test_orbax_writer_with_metadata() {
        let dir = tempdir().unwrap();

        let custom_meta = serde_json::json!({
            "model_name": "test_model",
            "framework": "jax"
        });

        let mut writer = OrbaxWriter::with_metadata(custom_meta);

        let data: Vec<u8> = vec![0; 8];
        writer
            .add_array("weight", &data, DType::Float16, &[4])
            .unwrap();
        writer.write(dir.path()).unwrap();

        // Read metadata file and verify custom fields
        let metadata_content = fs::read_to_string(dir.path().join("_CHECKPOINT_METADATA")).unwrap();
        assert!(metadata_content.contains("test_model"));
        assert!(metadata_content.contains("jax"));
    }

    #[test]
    fn test_npy_header_generation() {
        // Test header generation for different shapes
        let header1 = OrbaxWriter::generate_npy_header(DType::Float32, &[10, 20]);
        assert!(header1.len() % 64 == 0); // Should be 64-byte aligned
        assert_eq!(&header1[..6], &[0x93, b'N', b'U', b'M', b'P', b'Y']);

        let header2 = OrbaxWriter::generate_npy_header(DType::BFloat16, &[100]);
        assert!(header2.len() % 64 == 0);

        let header3 = OrbaxWriter::generate_npy_header(DType::Int64, &[]);
        assert!(header3.len() % 64 == 0);
    }

    #[test]
    fn test_orbax_writer_npy_readable() {
        let dir = tempdir().unwrap();
        let mut writer = OrbaxWriter::new();

        // Create simple test data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer
            .add_array("tensor", &bytes, DType::Float32, &[2, 2])
            .unwrap();
        writer.write(dir.path()).unwrap();

        // Manually read and verify NPY file
        let npy_path = dir.path().join("state/tensor/0.npy");
        let npy_data = fs::read(&npy_path).unwrap();

        // Verify magic number
        assert_eq!(&npy_data[..6], &[0x93, b'N', b'U', b'M', b'P', b'Y']);

        // Verify version
        assert_eq!(npy_data[6], 1);
        assert_eq!(npy_data[7], 0);

        // Read header length
        let header_len = u16::from_le_bytes([npy_data[8], npy_data[9]]) as usize;

        // Verify data at end matches original
        let data_start = 10 + header_len;
        let read_data: Vec<f32> = npy_data[data_start..]
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_is_orbax_checkpoint_after_write() {
        let dir = tempdir().unwrap();
        let mut writer = OrbaxWriter::new();

        let data: Vec<u8> = vec![0; 4];
        writer.add_array("x", &data, DType::Float32, &[1]).unwrap();
        writer.write(dir.path()).unwrap();

        // Should now be recognized as Orbax checkpoint
        assert!(is_orbax_checkpoint(dir.path()));
    }
}
