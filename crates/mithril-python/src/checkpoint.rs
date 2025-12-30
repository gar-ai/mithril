//! Python bindings for mithril-checkpoint.

use mithril_checkpoint::{
    compressed_safetensors, CheckpointCompressor, CompressionConfig, CompressionStats,
    DeltaCompressor, DeltaStats, MstReader, MstWriter, OrbaxWriteStats, OrbaxWriter,
    QuantizeConfig, QuantizeMethod, Quantizer,
};
use mithril_core::types::DType;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

/// Python wrapper for CompressionConfig.
#[pyclass(name = "CompressionConfig")]
#[derive(Clone)]
pub struct PyCompressionConfig {
    inner: CompressionConfig,
}

#[pymethods]
impl PyCompressionConfig {
    /// Create a new compression config.
    ///
    /// Args:
    ///     zstd_level: Zstd compression level (1-22, default 3)
    ///     byte_grouping: Enable byte grouping for floating-point types (default True)
    #[new]
    #[pyo3(signature = (zstd_level=3, byte_grouping=true))]
    fn new(zstd_level: i32, byte_grouping: bool) -> Self {
        Self {
            inner: CompressionConfig {
                zstd_level,
                byte_grouping,
            },
        }
    }

    /// Create config optimized for compression speed.
    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: CompressionConfig::fast(),
        }
    }

    /// Create config optimized for compression ratio.
    #[staticmethod]
    fn best() -> Self {
        Self {
            inner: CompressionConfig::best(),
        }
    }

    /// Get the zstd compression level.
    #[getter]
    fn zstd_level(&self) -> i32 {
        self.inner.zstd_level
    }

    /// Get whether byte grouping is enabled.
    #[getter]
    fn byte_grouping(&self) -> bool {
        self.inner.byte_grouping
    }

    fn __repr__(&self) -> String {
        format!(
            "CompressionConfig(zstd_level={}, byte_grouping={})",
            self.inner.zstd_level, self.inner.byte_grouping
        )
    }
}

/// Python wrapper for CheckpointCompressor.
#[pyclass(name = "CheckpointCompressor")]
pub struct PyCheckpointCompressor {
    inner: CheckpointCompressor,
}

#[pymethods]
impl PyCheckpointCompressor {
    /// Create a new checkpoint compressor.
    ///
    /// Args:
    ///     config: Optional compression configuration. If None, uses default settings.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyCompressionConfig>) -> Self {
        let inner = match config {
            Some(cfg) => CheckpointCompressor::new(cfg.inner),
            None => CheckpointCompressor::default(),
        };
        Self { inner }
    }

    /// Compress tensor data.
    ///
    /// Args:
    ///     data: Raw tensor bytes
    ///     dtype: Data type string ("bf16", "fp16", "fp32", "fp64", "i8", "i32", "i64")
    ///     previous: Optional previous checkpoint bytes for delta encoding.
    ///               When provided, achieves 10-70x compression on consecutive checkpoints.
    ///
    /// Returns:
    ///     Compressed bytes
    ///
    /// Example:
    ///     # First checkpoint (no delta, ~1.4x compression)
    ///     compressed = compressor.compress(data, "bf16")
    ///
    ///     # Subsequent checkpoints (with delta, 10-70x compression)
    ///     compressed = compressor.compress(current, "bf16", previous=prev_data)
    #[pyo3(signature = (data, dtype, previous=None))]
    fn compress(&self, data: &[u8], dtype: &str, previous: Option<&[u8]>) -> PyResult<Vec<u8>> {
        let dtype = parse_dtype(dtype)?;
        self.inner
            .compress_with_delta(data, dtype, previous)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Decompress tensor data.
    ///
    /// Args:
    ///     data: Compressed bytes
    ///     dtype: Data type string ("bf16", "fp16", "fp32", etc.)
    ///     original_size: Expected size of decompressed data in bytes
    ///     previous: Optional previous checkpoint bytes for delta decoding.
    ///               Must match the previous used during compression.
    ///
    /// Returns:
    ///     Decompressed bytes matching original tensor data
    #[pyo3(signature = (data, dtype, original_size, previous=None))]
    fn decompress(
        &self,
        data: &[u8],
        dtype: &str,
        original_size: usize,
        previous: Option<&[u8]>,
    ) -> PyResult<Vec<u8>> {
        let dtype = parse_dtype(dtype)?;
        self.inner
            .decompress_with_delta(data, dtype, original_size, previous)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Compress raw bytes without byte grouping.
    ///
    /// This is useful for compressing pickle data or other non-tensor data
    /// where byte grouping doesn't provide benefits.
    ///
    /// Args:
    ///     data: Raw bytes to compress
    ///
    /// Returns:
    ///     Compressed bytes
    fn compress_raw(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        self.inner
            .compress_raw(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Decompress raw bytes without byte grouping.
    ///
    /// Args:
    ///     data: Compressed bytes
    ///     original_size: Expected size of decompressed data
    ///
    /// Returns:
    ///     Decompressed bytes
    fn decompress_raw(&self, data: &[u8], original_size: usize) -> PyResult<Vec<u8>> {
        self.inner
            .decompress_raw(data, original_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Calculate compression ratio.
    ///
    /// Args:
    ///     original_size: Original data size in bytes
    ///     compressed_size: Compressed data size in bytes
    ///
    /// Returns:
    ///     Compression ratio (original_size / compressed_size)
    #[staticmethod]
    fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        CheckpointCompressor::compression_ratio(original_size, compressed_size)
    }

    /// Compress tensor data with lossy quantization.
    ///
    /// This applies quantization before compression, achieving higher compression
    /// ratios at the cost of some precision loss.
    ///
    /// Args:
    ///     data: Raw tensor bytes
    ///     dtype: Data type string ("bf16", "fp16", "fp32")
    ///     shape: Shape of the tensor
    ///     quantize_config: Quantization configuration
    ///
    /// Returns:
    ///     Compressed quantized bytes (includes metadata for dequantization)
    ///
    /// Example:
    ///     config = mithril.QuantizeConfig(method="int8")
    ///     compressed = compressor.compress_quantized(data, "bf16", [1000], config)
    fn compress_quantized(
        &self,
        data: &[u8],
        dtype: &str,
        shape: Vec<usize>,
        quantize_config: PyQuantizeConfig,
    ) -> PyResult<Vec<u8>> {
        let dtype = parse_dtype(dtype)?;
        self.inner
            .compress_quantized(data, dtype, shape, &quantize_config.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Decompress and dequantize tensor data.
    ///
    /// Args:
    ///     data: Compressed quantized bytes (from compress_quantized)
    ///     dtype: Target data type ("bf16", "fp16", "fp32")
    ///
    /// Returns:
    ///     Dequantized tensor bytes in the target dtype
    fn decompress_dequantized(&self, data: &[u8], dtype: &str) -> PyResult<Vec<u8>> {
        let dtype = parse_dtype(dtype)?;
        self.inner
            .decompress_dequantized(data, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckpointCompressor(config=CompressionConfig(zstd_level={}, byte_grouping={}))",
            self.inner.config().zstd_level,
            self.inner.config().byte_grouping
        )
    }
}

/// Python wrapper for CompressionStats.
#[pyclass(name = "CompressionStats")]
pub struct PyCompressionStats {
    /// Original size in bytes
    #[pyo3(get)]
    pub original_size: usize,
    /// Compressed size in bytes
    #[pyo3(get)]
    pub compressed_size: usize,
    /// Compression ratio (original / compressed)
    #[pyo3(get)]
    pub ratio: f64,
}

#[pymethods]
impl PyCompressionStats {
    /// Create new stats from sizes.
    #[new]
    fn new(original_size: usize, compressed_size: usize) -> Self {
        let stats = CompressionStats::new(original_size, compressed_size);
        Self {
            original_size: stats.original_size,
            compressed_size: stats.compressed_size,
            ratio: stats.ratio,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CompressionStats(original_size={}, compressed_size={}, ratio={:.2})",
            self.original_size, self.compressed_size, self.ratio
        )
    }
}

/// Parse dtype string to DType enum.
fn parse_dtype(dtype: &str) -> PyResult<DType> {
    match dtype.to_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BFloat16),
        "fp16" | "float16" | "f16" => Ok(DType::Float16),
        "fp32" | "float32" | "f32" => Ok(DType::Float32),
        "fp64" | "float64" | "f64" => Ok(DType::Float64),
        "i8" | "int8" => Ok(DType::Int8),
        "i32" | "int32" => Ok(DType::Int32),
        "i64" | "int64" => Ok(DType::Int64),
        "u8" | "uint8" => Ok(DType::UInt8),
        "bool" => Ok(DType::Bool),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown dtype: '{}'. Supported: bf16, fp16, fp32, fp64, i8, i32, i64, u8, bool",
            dtype
        ))),
    }
}

// =============================================================================
// DeltaCompressor bindings
// =============================================================================

/// Python wrapper for DeltaStats.
///
/// Statistics from delta compression including sparsity information.
#[pyclass(name = "DeltaStats")]
pub struct PyDeltaStats {
    /// Original size in bytes
    #[pyo3(get)]
    pub original_size: usize,
    /// Compressed size in bytes
    #[pyo3(get)]
    pub compressed_size: usize,
    /// Compression ratio (original / compressed)
    #[pyo3(get)]
    pub ratio: f64,
    /// Sparsity of delta (fraction of zero bytes after XOR)
    #[pyo3(get)]
    pub sparsity: f64,
    /// Whether delta encoding was used
    #[pyo3(get)]
    pub used_delta: bool,
    /// Reference checkpoint key (if delta was used)
    #[pyo3(get)]
    pub reference_key: Option<String>,
}

impl From<DeltaStats> for PyDeltaStats {
    fn from(stats: DeltaStats) -> Self {
        Self {
            original_size: stats.original_size,
            compressed_size: stats.compressed_size,
            ratio: stats.ratio,
            sparsity: stats.sparsity,
            used_delta: stats.used_delta,
            reference_key: stats.reference_key,
        }
    }
}

#[pymethods]
impl PyDeltaStats {
    fn __repr__(&self) -> String {
        format!(
            "DeltaStats(original_size={}, compressed_size={}, ratio={:.2}, sparsity={:.4}, used_delta={}, reference_key={:?})",
            self.original_size, self.compressed_size, self.ratio, self.sparsity, self.used_delta, self.reference_key
        )
    }
}

/// Python wrapper for DeltaCompressor.
///
/// High-level compressor with automatic reference management for delta encoding.
/// Tracks previous checkpoints and automatically uses delta encoding when beneficial.
///
/// Example:
///     import mithril
///
///     # Create delta compressor
///     compressor = mithril.DeltaCompressor()
///
///     # First checkpoint (no delta available, ~1.4x compression)
///     compressed1, stats1 = compressor.compress_checkpoint("step_100", data1)
///     print(f"Ratio: {stats1.ratio:.1f}x, used_delta: {stats1.used_delta}")
///
///     # Second checkpoint (uses delta, 10-70x compression!)
///     compressed2, stats2 = compressor.compress_checkpoint("step_200", data2)
///     print(f"Ratio: {stats2.ratio:.1f}x, sparsity: {stats2.sparsity:.2%}")
#[pyclass(name = "DeltaCompressor")]
pub struct PyDeltaCompressor {
    inner: DeltaCompressor,
}

#[pymethods]
impl PyDeltaCompressor {
    /// Create a new delta compressor.
    ///
    /// Args:
    ///     config: Optional compression configuration. If None, uses default settings.
    ///     max_references: Maximum number of references to keep (default 10).
    #[new]
    #[pyo3(signature = (config=None, max_references=10))]
    fn new(config: Option<PyCompressionConfig>, max_references: usize) -> Self {
        let inner = match config {
            Some(cfg) => DeltaCompressor::new(cfg.inner).with_max_references(max_references),
            None => DeltaCompressor::default().with_max_references(max_references),
        };
        Self { inner }
    }

    /// Compress a checkpoint with automatic delta encoding.
    ///
    /// Uses the most recent stored checkpoint as the delta reference.
    /// The checkpoint is stored for future delta encoding.
    ///
    /// Args:
    ///     key: Unique identifier for this checkpoint (e.g., "step_100")
    ///     data: Raw checkpoint bytes
    ///     dtype: Data type string (default "bf16")
    ///
    /// Returns:
    ///     Tuple of (compressed_bytes, DeltaStats)
    #[pyo3(signature = (key, data, dtype="bf16"))]
    fn compress_checkpoint(
        &mut self,
        key: &str,
        data: &[u8],
        dtype: &str,
    ) -> PyResult<(Vec<u8>, PyDeltaStats)> {
        let dtype = parse_dtype(dtype)?;
        let (compressed, stats) = self
            .inner
            .compress_checkpoint_with_dtype(key, data, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((compressed, stats.into()))
    }

    /// Compress a checkpoint with a specific reference.
    ///
    /// Use this when you want explicit control over the delta reference.
    ///
    /// Args:
    ///     key: Unique identifier for this checkpoint
    ///     data: Raw checkpoint bytes
    ///     reference_key: Key of the reference checkpoint to use
    ///     dtype: Data type string (default "bf16")
    ///
    /// Returns:
    ///     Tuple of (compressed_bytes, DeltaStats)
    #[pyo3(signature = (key, data, reference_key, dtype="bf16"))]
    fn compress_with_reference(
        &mut self,
        key: &str,
        data: &[u8],
        reference_key: &str,
        dtype: &str,
    ) -> PyResult<(Vec<u8>, PyDeltaStats)> {
        let dtype = parse_dtype(dtype)?;
        let (compressed, stats) = self
            .inner
            .compress_with_reference(key, data, reference_key, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((compressed, stats.into()))
    }

    /// Decompress a checkpoint.
    ///
    /// Args:
    ///     data: Compressed checkpoint bytes
    ///     original_size: Expected decompressed size
    ///     reference_key: Key of reference checkpoint (if delta was used)
    ///     dtype: Data type string (default "bf16")
    ///
    /// Returns:
    ///     Decompressed bytes
    #[pyo3(signature = (data, original_size, reference_key=None, dtype="bf16"))]
    fn decompress_checkpoint(
        &self,
        data: &[u8],
        original_size: usize,
        reference_key: Option<&str>,
        dtype: &str,
    ) -> PyResult<Vec<u8>> {
        let dtype = parse_dtype(dtype)?;
        self.inner
            .decompress_checkpoint(data, original_size, reference_key, dtype)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Store a reference checkpoint for future delta encoding.
    ///
    /// Args:
    ///     key: Unique identifier for this checkpoint
    ///     data: Raw checkpoint bytes
    fn store_reference(&mut self, key: &str, data: &[u8]) {
        self.inner.store_reference(key, data);
    }

    /// Check if a reference exists.
    ///
    /// Args:
    ///     key: Reference key to check
    ///
    /// Returns:
    ///     True if the reference exists
    fn has_reference(&self, key: &str) -> bool {
        self.inner.has_reference(key)
    }

    /// Get the number of stored references.
    #[getter]
    fn reference_count(&self) -> usize {
        self.inner.reference_count()
    }

    /// Get the latest reference key.
    #[getter]
    fn latest_reference(&self) -> Option<&str> {
        self.inner.latest_reference()
    }

    /// Clear all stored references.
    fn clear_references(&mut self) {
        self.inner.clear_references();
    }

    fn __repr__(&self) -> String {
        format!(
            "DeltaCompressor(reference_count={}, latest={:?})",
            self.inner.reference_count(),
            self.inner.latest_reference()
        )
    }
}

// =============================================================================
// OrbaxWriter bindings
// =============================================================================

/// Statistics from writing an Orbax checkpoint.
#[pyclass(name = "OrbaxWriteStats")]
pub struct PyOrbaxWriteStats {
    /// Number of arrays written.
    #[pyo3(get)]
    pub array_count: usize,
    /// Total bytes written (all arrays).
    #[pyo3(get)]
    pub total_bytes: usize,
    /// Bytes written per array.
    #[pyo3(get)]
    pub array_bytes: HashMap<String, usize>,
}

impl From<OrbaxWriteStats> for PyOrbaxWriteStats {
    fn from(stats: OrbaxWriteStats) -> Self {
        Self {
            array_count: stats.array_count,
            total_bytes: stats.total_bytes,
            array_bytes: stats.array_bytes,
        }
    }
}

#[pymethods]
impl PyOrbaxWriteStats {
    fn __repr__(&self) -> String {
        format!(
            "OrbaxWriteStats(array_count={}, total_bytes={})",
            self.array_count, self.total_bytes
        )
    }
}

/// Python wrapper for OrbaxWriter.
///
/// Writer for Orbax checkpoint format (JAX/Flax).
/// Creates a directory containing NPY files and metadata.
///
/// Example:
///     import mithril
///     import numpy as np
///
///     writer = mithril.OrbaxWriter()
///
///     # Add arrays (name can include path separators)
///     weights = np.random.randn(768, 768).astype(np.float32)
///     writer.add_array("params/layer_0/kernel", weights)
///
///     bias = np.zeros(768, dtype=np.float32)
///     writer.add_array("params/layer_0/bias", bias)
///
///     # Write to directory
///     stats = writer.write("./checkpoint")
///     print(f"Wrote {stats.array_count} arrays, {stats.total_bytes} bytes")
#[pyclass(name = "OrbaxWriter")]
pub struct PyOrbaxWriter {
    inner: OrbaxWriter,
}

#[pymethods]
impl PyOrbaxWriter {
    /// Create a new Orbax writer.
    ///
    /// Args:
    ///     metadata: Optional JSON-serializable metadata to include.
    #[new]
    #[pyo3(signature = (metadata=None))]
    fn new(metadata: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let inner = if let Some(meta) = metadata {
            // Convert Python dict to serde_json::Value
            let json_str: String = Python::with_gil(|py| {
                let json_mod = py.import("json")?;
                let result = json_mod.call_method1("dumps", (meta,))?;
                result.extract::<String>()
            })?;
            let json_value: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            OrbaxWriter::with_metadata(json_value)
        } else {
            OrbaxWriter::new()
        };
        Ok(Self { inner })
    }

    /// Add an array to be written.
    ///
    /// Args:
    ///     name: Array name (e.g., "params/layer_0/kernel"). Use "/" for hierarchy.
    ///     data: NumPy array or bytes. If numpy array, will be converted to bytes.
    ///     dtype: Optional dtype string. Required if data is bytes, inferred if numpy array.
    ///     shape: Optional shape. Required if data is bytes, inferred if numpy array.
    ///
    /// Raises:
    ///     ValueError: If data size doesn't match shape * dtype_size.
    #[pyo3(signature = (name, data, dtype=None, shape=None))]
    fn add_array(
        &mut self,
        _py: Python<'_>,
        name: &str,
        data: &Bound<'_, PyAny>,
        dtype: Option<&str>,
        shape: Option<Vec<usize>>,
    ) -> PyResult<()> {
        // Try to handle numpy arrays
        let (bytes, inferred_dtype, inferred_shape) = if let Ok(array) = data.getattr("tobytes") {
            // It's a numpy-like object with tobytes()
            let bytes_obj = array.call0()?;
            let bytes: Vec<u8> = bytes_obj.extract()?;

            // Get dtype from array
            let dtype_obj = data.getattr("dtype")?;
            let dtype_name: String = dtype_obj.getattr("name")?.extract()?;

            // Get shape from array
            let shape_obj = data.getattr("shape")?;
            let shape: Vec<usize> = shape_obj.extract()?;

            (bytes, Some(dtype_name), Some(shape))
        } else if let Ok(bytes) = data.extract::<Vec<u8>>() {
            // It's raw bytes
            (bytes, None, None)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "data must be a numpy array or bytes",
            ));
        };

        // Determine dtype
        let dtype_str = dtype
            .map(|s| s.to_string())
            .or(inferred_dtype)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("dtype required when data is bytes")
            })?;
        let dtype = parse_dtype(&dtype_str)?;

        // Determine shape
        let shape = shape.or(inferred_shape).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("shape required when data is bytes")
        })?;

        self.inner
            .add_array(name, &bytes, dtype, &shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Write the checkpoint to a directory.
    ///
    /// Creates the directory structure:
    /// - _CHECKPOINT_METADATA (JSON)
    /// - .orbax-checkpoint (marker file)
    /// - state/<array_name>/0.npy (NPY files)
    ///
    /// Args:
    ///     path: Output directory path. Will be created if it doesn't exist.
    ///
    /// Returns:
    ///     OrbaxWriteStats with write statistics.
    fn write(&self, path: &str) -> PyResult<PyOrbaxWriteStats> {
        let path = PathBuf::from(path);
        self.inner
            .write(&path)
            .map(|stats| stats.into())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get the number of arrays added.
    #[getter]
    fn array_count(&self) -> usize {
        self.inner.array_count()
    }

    /// Get the total size of array data (excluding headers).
    #[getter]
    fn data_size(&self) -> usize {
        self.inner.data_size()
    }

    /// Get array names.
    fn array_names(&self) -> Vec<String> {
        self.inner.array_names().map(|s| s.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "OrbaxWriter(array_count={}, data_size={})",
            self.inner.array_count(),
            self.inner.data_size()
        )
    }
}

// =============================================================================
// Quantization bindings
// =============================================================================

/// Python wrapper for QuantizeConfig.
///
/// Configuration for lossy quantization of tensor data.
/// Quantization reduces precision to achieve higher compression ratios.
///
/// Example:
///     import mithril
///
///     # Int8 quantization (~2x compression)
///     config = mithril.QuantizeConfig(method="int8")
///
///     # NF4 quantization (~4x compression, QLoRA-style)
///     config = mithril.QuantizeConfig(method="nf4", group_size=64)
///
///     # Exclude certain layers from quantization
///     config = mithril.QuantizeConfig(
///         method="int8",
///         exclude_patterns=["embed", "lm_head", "layernorm"]
///     )
#[pyclass(name = "QuantizeConfig")]
#[derive(Clone)]
pub struct PyQuantizeConfig {
    inner: QuantizeConfig,
}

#[pymethods]
impl PyQuantizeConfig {
    /// Create a new quantization config.
    ///
    /// Args:
    ///     method: Quantization method ("int8", "nf4", "int4", "dynamic_int8")
    ///     group_size: Group size for grouped quantization (required for NF4)
    ///     exclude_patterns: List of patterns for tensor names to exclude from quantization
    #[new]
    #[pyo3(signature = (method="int8", group_size=None, exclude_patterns=None))]
    fn new(
        method: &str,
        group_size: Option<usize>,
        exclude_patterns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let quant_method = parse_quantize_method(method)?;
        let mut config = QuantizeConfig::new(quant_method);
        if let Some(gs) = group_size {
            config = config.with_group_size(gs);
        }
        if let Some(patterns) = exclude_patterns {
            config = config.with_exclude_patterns(patterns);
        }
        Ok(Self { inner: config })
    }

    /// Create Int8 linear quantization config.
    #[staticmethod]
    fn int8() -> Self {
        Self {
            inner: QuantizeConfig::int8(),
        }
    }

    /// Create NF4 quantization config with default group size (64).
    #[staticmethod]
    fn nf4() -> Self {
        Self {
            inner: QuantizeConfig::nf4(),
        }
    }

    /// Create NF4 quantization config with custom group size.
    #[staticmethod]
    fn nf4_with_group_size(group_size: usize) -> Self {
        Self {
            inner: QuantizeConfig::nf4_with_group_size(group_size),
        }
    }

    /// Get the quantization method.
    #[getter]
    fn method(&self) -> &str {
        match self.inner.method {
            QuantizeMethod::Int8Linear => "int8",
            QuantizeMethod::NF4 => "nf4",
            QuantizeMethod::Int4 => "int4",
            QuantizeMethod::DynamicInt8 => "dynamic_int8",
        }
    }

    /// Get the group size.
    #[getter]
    fn group_size(&self) -> Option<usize> {
        self.inner.group_size
    }

    /// Get the exclude patterns.
    #[getter]
    fn exclude_patterns(&self) -> Vec<String> {
        self.inner.exclude_patterns.clone()
    }

    /// Check if a tensor name should be excluded from quantization.
    fn should_exclude(&self, tensor_name: &str) -> bool {
        self.inner.should_exclude(tensor_name)
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantizeConfig(method='{}', group_size={:?}, exclude_patterns={:?})",
            self.method(),
            self.inner.group_size,
            self.inner.exclude_patterns
        )
    }
}

/// Parse quantize method string.
fn parse_quantize_method(method: &str) -> PyResult<QuantizeMethod> {
    match method.to_lowercase().as_str() {
        "int8" | "int8linear" => Ok(QuantizeMethod::Int8Linear),
        "nf4" => Ok(QuantizeMethod::NF4),
        "int4" => Ok(QuantizeMethod::Int4),
        "dynamic_int8" | "dynamicint8" => Ok(QuantizeMethod::DynamicInt8),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown quantization method: '{}'. Supported: int8, nf4, int4, dynamic_int8",
            method
        ))),
    }
}

/// Python wrapper for Quantizer.
///
/// Standalone quantizer for tensor data.
/// Use this when you need low-level control over quantization.
///
/// Example:
///     import mithril
///     import numpy as np
///
///     config = mithril.QuantizeConfig(method="int8")
///     quantizer = mithril.Quantizer(config)
///
///     # Quantize bf16 data
///     bf16_data = np.random.randn(1000).astype(np.float16).tobytes()
///     quantized = quantizer.quantize_bf16(bf16_data)
///
///     # Dequantize back
///     dequantized = quantizer.dequantize_to_bf16(quantized)
#[pyclass(name = "Quantizer")]
pub struct PyQuantizer {
    inner: Quantizer,
}

#[pymethods]
impl PyQuantizer {
    /// Create a new quantizer.
    ///
    /// Args:
    ///     config: Quantization configuration.
    #[new]
    fn new(config: PyQuantizeConfig) -> Self {
        Self {
            inner: Quantizer::new(config.inner),
        }
    }

    /// Quantize bf16 tensor data.
    ///
    /// Args:
    ///     data: Raw bf16 bytes (2 bytes per element, little-endian)
    ///     shape: Optional shape of the tensor
    ///
    /// Returns:
    ///     Serialized quantized tensor bytes (includes metadata for dequantization)
    #[pyo3(signature = (data, shape=None))]
    fn quantize_bf16(&self, data: &[u8], shape: Option<Vec<usize>>) -> PyResult<Vec<u8>> {
        let shape = shape.unwrap_or_else(|| vec![data.len() / 2]);
        let quantized = self
            .inner
            .quantize_bf16_with_shape(data, shape)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        quantized
            .to_bytes()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Quantize fp16 tensor data.
    ///
    /// Args:
    ///     data: Raw fp16 bytes (2 bytes per element, little-endian)
    ///     shape: Optional shape of the tensor
    ///
    /// Returns:
    ///     Serialized quantized tensor bytes
    #[pyo3(signature = (data, shape=None))]
    fn quantize_fp16(&self, data: &[u8], shape: Option<Vec<usize>>) -> PyResult<Vec<u8>> {
        let shape = shape.unwrap_or_else(|| vec![data.len() / 2]);
        let quantized = self
            .inner
            .quantize_fp16_with_shape(data, shape)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        quantized
            .to_bytes()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Quantize fp32 tensor data.
    ///
    /// Args:
    ///     data: Raw fp32 bytes (4 bytes per element, little-endian)
    ///     shape: Optional shape of the tensor
    ///
    /// Returns:
    ///     Serialized quantized tensor bytes
    #[pyo3(signature = (data, shape=None))]
    fn quantize_fp32(&self, data: &[u8], shape: Option<Vec<usize>>) -> PyResult<Vec<u8>> {
        let shape = shape.unwrap_or_else(|| vec![data.len() / 4]);
        let quantized = self
            .inner
            .quantize_fp32_with_shape(data, shape)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        quantized
            .to_bytes()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Dequantize to bf16 bytes.
    ///
    /// Args:
    ///     data: Serialized quantized tensor bytes (from quantize_*)
    ///
    /// Returns:
    ///     Dequantized bf16 bytes
    fn dequantize_to_bf16(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let quantized = mithril_checkpoint::QuantizedTensor::from_bytes(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.inner
            .dequantize_to_bf16(&quantized)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Dequantize to fp16 bytes.
    ///
    /// Args:
    ///     data: Serialized quantized tensor bytes (from quantize_*)
    ///
    /// Returns:
    ///     Dequantized fp16 bytes
    fn dequantize_to_fp16(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let quantized = mithril_checkpoint::QuantizedTensor::from_bytes(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.inner
            .dequantize_to_fp16(&quantized)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Dequantize to fp32 bytes.
    ///
    /// Args:
    ///     data: Serialized quantized tensor bytes (from quantize_*)
    ///
    /// Returns:
    ///     Dequantized fp32 bytes
    fn dequantize_to_fp32(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        let quantized = mithril_checkpoint::QuantizedTensor::from_bytes(data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.inner
            .dequantize_to_fp32(&quantized)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let method = match self.inner.config().method {
            QuantizeMethod::Int8Linear => "int8",
            QuantizeMethod::NF4 => "nf4",
            QuantizeMethod::Int4 => "int4",
            QuantizeMethod::DynamicInt8 => "dynamic_int8",
        };
        format!("Quantizer(method='{}')", method)
    }
}

// =============================================================================
// MST Format bindings
// =============================================================================

/// Tensor metadata for MST format.
#[pyclass(name = "MstTensorInfo")]
#[derive(Clone)]
pub struct PyMstTensorInfo {
    /// Data type of the tensor
    #[pyo3(get)]
    pub dtype: String,
    /// Shape of the tensor
    #[pyo3(get)]
    pub shape: Vec<usize>,
    /// Size in compressed data section
    #[pyo3(get)]
    pub compressed_size: usize,
    /// Original uncompressed size
    #[pyo3(get)]
    pub uncompressed_size: usize,
}

#[pymethods]
impl PyMstTensorInfo {
    fn __repr__(&self) -> String {
        format!(
            "MstTensorInfo(dtype='{}', shape={:?}, compressed_size={}, uncompressed_size={})",
            self.dtype, self.shape, self.compressed_size, self.uncompressed_size
        )
    }

    /// Compression ratio for this tensor.
    #[getter]
    fn ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            0.0
        } else {
            self.uncompressed_size as f64 / self.compressed_size as f64
        }
    }
}

/// Writer for MST (Mithril Safetensors) format.
///
/// The MST format is a compressed variant of safetensors that uses
/// byte grouping and zstd compression to achieve 10x+ compression.
///
/// Example:
///     import mithril
///     import numpy as np
///
///     # Create writer
///     writer = mithril.MstWriter()
///
///     # Add tensors
///     weights = np.random.randn(512, 768).astype(np.float16).tobytes()
///     writer.add_tensor("model.weight", weights, "fp16", [512, 768])
///
///     # Add metadata
///     writer.set_metadata({"format": "pytorch", "model": "bert"})
///
///     # Write file
///     writer.write("model.mst")
#[pyclass(name = "MstWriter")]
pub struct PyMstWriter {
    inner: MstWriter,
}

#[pymethods]
impl PyMstWriter {
    /// Create a new MST writer.
    ///
    /// Args:
    ///     config: Optional compression configuration.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyCompressionConfig>) -> Self {
        let inner = match config {
            Some(cfg) => MstWriter::with_config(cfg.inner),
            None => MstWriter::new(),
        };
        Self { inner }
    }

    /// Add a tensor to be written.
    ///
    /// Args:
    ///     name: Tensor name (e.g., "model.weight" or "layers.0.attention.query")
    ///     data: Raw tensor bytes or numpy array
    ///     dtype: Data type string ("bf16", "fp16", "fp32", etc.)
    ///     shape: Tensor shape
    #[pyo3(signature = (name, data, dtype, shape))]
    fn add_tensor(
        &mut self,
        _py: Python<'_>,
        name: &str,
        data: &Bound<'_, PyAny>,
        dtype: &str,
        shape: Vec<usize>,
    ) -> PyResult<()> {
        let bytes: Vec<u8> = if let Ok(array) = data.getattr("tobytes") {
            let bytes_obj = array.call0()?;
            bytes_obj.extract()?
        } else {
            data.extract()?
        };

        let dtype_enum = parse_dtype(dtype)?;
        self.inner.add_tensor(name, bytes, dtype_enum, shape);
        Ok(())
    }

    /// Set metadata for the MST file.
    ///
    /// Args:
    ///     metadata: Dictionary of string key-value pairs
    fn set_metadata(&mut self, metadata: HashMap<String, String>) {
        for (key, value) in metadata {
            self.inner.add_metadata(key, value);
        }
    }

    /// Write the MST file.
    ///
    /// Args:
    ///     path: Output file path
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get total compressed size of all tensors.
    #[getter]
    fn compressed_size(&self) -> usize {
        self.inner.compressed_size()
    }

    /// Get total uncompressed size of all tensors.
    #[getter]
    fn uncompressed_size(&self) -> usize {
        self.inner.uncompressed_size()
    }

    /// Get number of tensors.
    #[getter]
    fn tensor_count(&self) -> usize {
        self.inner.tensor_count()
    }

    /// Get compression ratio.
    #[getter]
    fn ratio(&self) -> f64 {
        if self.inner.compressed_size() == 0 {
            0.0
        } else {
            self.inner.uncompressed_size() as f64 / self.inner.compressed_size() as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MstWriter(tensor_count={}, uncompressed_size={}, compressed_size={}, ratio={:.2}x)",
            self.inner.tensor_count(),
            self.inner.uncompressed_size(),
            self.inner.compressed_size(),
            self.ratio()
        )
    }
}

/// Reader for MST (Mithril Safetensors) format.
///
/// Example:
///     import mithril
///
///     # Open MST file
///     reader = mithril.MstReader("model.mst")
///
///     # List tensors
///     print(reader.tensor_names())
///
///     # Get tensor info
///     info = reader.tensor_info("model.weight")
///     print(f"Shape: {info.shape}, dtype: {info.dtype}")
///
///     # Read tensor data
///     data = reader.read_tensor("model.weight")
///
///     # Read all tensors
///     all_tensors = reader.read_all()
#[pyclass(name = "MstReader")]
pub struct PyMstReader {
    inner: std::sync::Mutex<MstReader>,
}

#[pymethods]
impl PyMstReader {
    /// Open an MST file for reading.
    ///
    /// Args:
    ///     path: Path to the MST file
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = MstReader::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            inner: std::sync::Mutex::new(reader),
        })
    }

    /// List all tensor names in the file.
    fn tensor_names(&self) -> Vec<String> {
        let reader = self.inner.lock().unwrap();
        reader.tensor_names().map(|s| s.to_string()).collect()
    }

    /// Get metadata for a specific tensor.
    ///
    /// Args:
    ///     name: Tensor name
    ///
    /// Returns:
    ///     MstTensorInfo with dtype, shape, sizes
    fn tensor_info(&self, name: &str) -> PyResult<PyMstTensorInfo> {
        let reader = self.inner.lock().unwrap();
        let info = reader.tensor_info(name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Tensor not found: {}", name))
        })?;
        Ok(PyMstTensorInfo {
            dtype: info.dtype.clone(),
            shape: info.shape.clone(),
            compressed_size: info.compressed_size,
            uncompressed_size: info.uncompressed_size,
        })
    }

    /// Read and decompress a tensor.
    ///
    /// Args:
    ///     name: Tensor name
    ///
    /// Returns:
    ///     Decompressed tensor bytes
    fn read_tensor(&self, name: &str) -> PyResult<Vec<u8>> {
        let mut reader = self.inner.lock().unwrap();
        reader
            .read_tensor(name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Read and decompress all tensors.
    ///
    /// Returns:
    ///     Dictionary mapping tensor names to decompressed bytes
    fn read_all(&self) -> PyResult<HashMap<String, Vec<u8>>> {
        let mut reader = self.inner.lock().unwrap();
        reader
            .read_all()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get file metadata.
    fn metadata(&self) -> HashMap<String, String> {
        let reader = self.inner.lock().unwrap();
        reader.header().metadata.clone()
    }

    /// Get total uncompressed size.
    #[getter]
    fn total_uncompressed_size(&self) -> usize {
        let reader = self.inner.lock().unwrap();
        reader.header().total_uncompressed_size
    }

    /// Get total compressed size.
    #[getter]
    fn total_compressed_size(&self) -> usize {
        let reader = self.inner.lock().unwrap();
        reader.header().total_compressed_size
    }

    /// Get number of tensors.
    #[getter]
    fn tensor_count(&self) -> usize {
        let reader = self.inner.lock().unwrap();
        reader.header().tensors.len()
    }

    /// Get overall compression ratio.
    #[getter]
    fn ratio(&self) -> f64 {
        let reader = self.inner.lock().unwrap();
        let header = reader.header();
        if header.total_compressed_size == 0 {
            0.0
        } else {
            header.total_uncompressed_size as f64 / header.total_compressed_size as f64
        }
    }

    fn __repr__(&self) -> String {
        let reader = self.inner.lock().unwrap();
        let header = reader.header();
        format!(
            "MstReader(tensor_count={}, uncompressed_size={}, compressed_size={}, ratio={:.2}x)",
            header.tensors.len(),
            header.total_uncompressed_size,
            header.total_compressed_size,
            if header.total_compressed_size == 0 {
                0.0
            } else {
                header.total_uncompressed_size as f64 / header.total_compressed_size as f64
            }
        )
    }
}

/// Check if a file is in MST format.
///
/// Checks both the file extension (.mst) and magic bytes.
///
/// Args:
///     path: Path to check
///
/// Returns:
///     True if the file appears to be MST format
#[pyfunction]
pub fn is_mst(path: &str) -> bool {
    compressed_safetensors::is_mst(path)
}

/// Register checkpoint module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "checkpoint")?;
    m.add_class::<PyCompressionConfig>()?;
    m.add_class::<PyCheckpointCompressor>()?;
    m.add_class::<PyCompressionStats>()?;
    m.add_class::<PyMstWriter>()?;
    m.add_class::<PyMstReader>()?;
    m.add_class::<PyMstTensorInfo>()?;
    m.add_function(wrap_pyfunction!(is_mst, &m)?)?;
    m.add_class::<PyDeltaCompressor>()?;
    m.add_class::<PyDeltaStats>()?;
    m.add_class::<PyOrbaxWriter>()?;
    m.add_class::<PyOrbaxWriteStats>()?;
    m.add_class::<PyQuantizeConfig>()?;
    m.add_class::<PyQuantizer>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
