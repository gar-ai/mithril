//! Cache key generation for machine-independent caching.
//!
//! Generates portable cache keys from computation graphs and input specs.
//! Keys are based on content (graph structure, shapes, dtypes) rather than
//! file paths, enabling cross-machine cache sharing.
//!
//! ## Path Normalization
//!
//! Absolute paths in source code are normalized to prevent cache key
//! pollution. This enables cache sharing between:
//! - Different users on the same machine
//! - CI/CD environments with varying checkout paths
//! - Docker containers with different mount points
//!
//! ## Version Compatibility
//!
//! Cache keys include version information to ensure compatibility:
//! - PyTorch version (major.minor)
//! - CUDA version (major.minor)
//! - Triton version (major.minor)
//!
//! Caches are only shared when versions are compatible.

use mithril_core::hashing::{Blake3Hasher, HashFunction};
use mithril_core::types::DType;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A cache key that's portable across machines.
///
/// Unlike PyTorch's default cache keys (which include file paths),
/// these keys are based on:
/// - Graph structure (ops and connectivity)
/// - Input tensor metadata (shapes, dtypes)
/// - Device class (not specific GPU model)
///
/// This enables cache sharing across different machines with the same
/// compute capability.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Hash of the computation graph structure.
    pub graph_hash: [u8; 32],
    /// Input tensor specifications.
    pub inputs: Vec<InputSpec>,
    /// Target device class.
    pub device: DeviceClass,
}

impl CacheKey {
    /// Create a new cache key.
    #[must_use]
    pub fn new(graph_hash: [u8; 32], inputs: Vec<InputSpec>, device: DeviceClass) -> Self {
        Self {
            graph_hash,
            inputs,
            device,
        }
    }

    /// Create a cache key from raw bytes (graph IR or source code).
    ///
    /// This is useful when you have the raw representation of a computation
    /// graph and want to generate a key.
    #[must_use]
    pub fn from_bytes(bytes: &[u8], inputs: Vec<InputSpec>, device: DeviceClass) -> Self {
        let hasher = Blake3Hasher::new();
        let hash = hasher.hash(bytes);
        let mut graph_hash = [0u8; 32];
        graph_hash.copy_from_slice(&hash[..32]);

        Self {
            graph_hash,
            inputs,
            device,
        }
    }

    /// Convert to a storage key string.
    ///
    /// The storage key is a hex-encoded hash of the entire cache key,
    /// suitable for use as a filename or object key.
    #[must_use]
    pub fn to_storage_key(&self) -> String {
        let hasher = Blake3Hasher::new();

        // Serialize the key deterministically
        let serialized = self.to_bytes();
        let hash = hasher.hash(&serialized);

        // Use first 16 bytes (32 hex chars) for reasonable length
        hex::encode(&hash[..16])
    }

    /// Serialize the cache key to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        // Use a simple but deterministic serialization
        let mut bytes = Vec::new();

        // Graph hash (32 bytes)
        bytes.extend_from_slice(&self.graph_hash);

        // Number of inputs (4 bytes, little-endian)
        bytes.extend_from_slice(&(self.inputs.len() as u32).to_le_bytes());

        // Each input spec
        for input in &self.inputs {
            bytes.extend_from_slice(&input.to_bytes());
        }

        // Device class
        bytes.extend_from_slice(&self.device.to_bytes());

        bytes
    }

    /// Create a cache key from bytes.
    ///
    /// Returns `None` if the bytes are malformed.
    #[must_use]
    pub fn from_serialized_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 36 {
            return None;
        }

        let mut graph_hash = [0u8; 32];
        graph_hash.copy_from_slice(&bytes[0..32]);

        let num_inputs = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]) as usize;

        let mut offset = 36;
        let mut inputs = Vec::with_capacity(num_inputs);

        for _ in 0..num_inputs {
            let (input, consumed) = InputSpec::from_bytes(&bytes[offset..])?;
            inputs.push(input);
            offset += consumed;
        }

        let (device, _) = DeviceClass::from_bytes(&bytes[offset..])?;

        Some(Self {
            graph_hash,
            inputs,
            device,
        })
    }
}

/// Specification for an input tensor.
///
/// Describes the shape and data type of a tensor input to a compiled graph.
/// Shapes may include -1 for dynamic dimensions.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputSpec {
    /// Shape of the tensor. -1 indicates a dynamic dimension.
    pub shape: Vec<i64>,
    /// Data type of the tensor.
    pub dtype: DType,
}

impl InputSpec {
    /// Create a new input specification.
    #[must_use]
    pub fn new(shape: Vec<i64>, dtype: DType) -> Self {
        Self { shape, dtype }
    }

    /// Create an input spec with static shape.
    #[must_use]
    pub fn static_shape(shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            shape: shape.into_iter().map(|d| d as i64).collect(),
            dtype,
        }
    }

    /// Create an input spec with a dynamic batch dimension.
    #[must_use]
    pub fn dynamic_batch(shape: Vec<usize>, dtype: DType) -> Self {
        let mut dims: Vec<i64> = shape.into_iter().map(|d| d as i64).collect();
        if !dims.is_empty() {
            dims[0] = -1; // Dynamic batch
        }
        Self { shape: dims, dtype }
    }

    /// Check if this spec has any dynamic dimensions.
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        self.shape.iter().any(|&d| d < 0)
    }

    /// Get the number of elements (returns None if dynamic).
    #[must_use]
    pub fn numel(&self) -> Option<usize> {
        if self.is_dynamic() {
            None
        } else {
            Some(self.shape.iter().map(|&d| d as usize).product())
        }
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Number of dimensions (4 bytes)
        bytes.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());

        // Each dimension (8 bytes each, i64)
        for &dim in &self.shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }

        // Dtype (1 byte)
        bytes.push(dtype_to_u8(&self.dtype));

        bytes
    }

    /// Deserialize from bytes. Returns the spec and number of bytes consumed.
    fn from_bytes(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.len() < 4 {
            return None;
        }

        let num_dims = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        let required = 4 + num_dims * 8 + 1;
        if bytes.len() < required {
            return None;
        }

        let mut shape = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            let offset = 4 + i * 8;
            let dim = i64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            shape.push(dim);
        }

        let dtype = dtype_from_u8(bytes[4 + num_dims * 8])?;

        Some((Self { shape, dtype }, required))
    }
}

/// Device class for cache key generation.
///
/// This is more coarse-grained than specific GPU models, enabling
/// cache sharing across GPUs with the same compute capability.
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DeviceClass {
    /// Matches any CUDA device.
    #[default]
    CudaAny,
    /// Matches a specific CUDA compute capability.
    CudaCompute {
        /// Major version (e.g., 8 for Ampere, 9 for Hopper).
        major: u8,
        /// Minor version.
        minor: u8,
    },
    /// CPU target.
    Cpu,
}

impl DeviceClass {
    /// Create a device class from compute capability.
    #[must_use]
    pub fn cuda(major: u8, minor: u8) -> Self {
        Self::CudaCompute { major, minor }
    }

    /// Check if this device class is compatible with another.
    ///
    /// `CudaAny` is compatible with any CUDA device.
    /// `CudaCompute` is compatible with the same or higher compute capability.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        match (self, other) {
            // CudaAny is compatible with any CUDA
            (Self::CudaAny, Self::CudaAny) => true,
            (Self::CudaAny, Self::CudaCompute { .. }) => true,
            (Self::CudaCompute { .. }, Self::CudaAny) => true,

            // Same compute capability or higher is compatible
            (
                Self::CudaCompute {
                    major: m1,
                    minor: n1,
                },
                Self::CudaCompute {
                    major: m2,
                    minor: n2,
                },
            ) => {
                // Target (self) must be >= source (other) for forward compatibility
                (*m1, *n1) >= (*m2, *n2)
            }

            // CPU is only compatible with CPU
            (Self::Cpu, Self::Cpu) => true,

            // Different device types are incompatible
            _ => false,
        }
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::CudaAny => vec![0],
            Self::CudaCompute { major, minor } => vec![1, *major, *minor],
            Self::Cpu => vec![2],
        }
    }

    /// Deserialize from bytes. Returns the device class and bytes consumed.
    fn from_bytes(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.is_empty() {
            return None;
        }

        match bytes[0] {
            0 => Some((Self::CudaAny, 1)),
            1 => {
                if bytes.len() < 3 {
                    return None;
                }
                Some((
                    Self::CudaCompute {
                        major: bytes[1],
                        minor: bytes[2],
                    },
                    3,
                ))
            }
            2 => Some((Self::Cpu, 1)),
            _ => None,
        }
    }
}


/// Software version for cache compatibility checking.
///
/// Only major and minor versions are used for compatibility,
/// as patch versions typically don't affect compiled artifacts.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Version {
    /// Major version number.
    pub major: u16,
    /// Minor version number.
    pub minor: u16,
}

impl Version {
    /// Create a new version.
    #[must_use]
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Parse version from string like "2.1.0" or "2.1".
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() < 2 {
            return None;
        }

        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;

        Some(Self { major, minor })
    }

    /// Check if this version is compatible with another.
    ///
    /// Compatible means same major version and minor >= other's minor.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major && self.minor >= other.minor
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4);
        bytes.extend_from_slice(&self.major.to_le_bytes());
        bytes.extend_from_slice(&self.minor.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes.
    #[allow(dead_code)]
    fn from_bytes(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.len() < 4 {
            return None;
        }
        let major = u16::from_le_bytes([bytes[0], bytes[1]]);
        let minor = u16::from_le_bytes([bytes[2], bytes[3]]);
        Some((Self { major, minor }, 4))
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Software environment versions for cache compatibility.
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentVersions {
    /// PyTorch version.
    pub torch: Option<Version>,
    /// CUDA version (None for CPU-only).
    pub cuda: Option<Version>,
    /// Triton version (None if not used).
    pub triton: Option<Version>,
}

impl EnvironmentVersions {
    /// Create a new environment versions struct.
    #[must_use]
    pub fn new(torch: Option<Version>, cuda: Option<Version>, triton: Option<Version>) -> Self {
        Self {
            torch,
            cuda,
            triton,
        }
    }

    /// Create with only PyTorch version.
    #[must_use]
    pub fn torch_only(version: Version) -> Self {
        Self {
            torch: Some(version),
            cuda: None,
            triton: None,
        }
    }

    /// Check if this environment is compatible with another.
    ///
    /// All present versions must be compatible.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Check torch compatibility
        match (&self.torch, &other.torch) {
            (Some(a), Some(b)) if !a.is_compatible_with(b) => return false,
            _ => {}
        }

        // Check CUDA compatibility
        match (&self.cuda, &other.cuda) {
            (Some(a), Some(b)) if !a.is_compatible_with(b) => return false,
            (None, Some(_)) => return false, // Can't use CUDA cache without CUDA
            _ => {}
        }

        // Check Triton compatibility
        match (&self.triton, &other.triton) {
            (Some(a), Some(b)) if !a.is_compatible_with(b) => return false,
            _ => {}
        }

        true
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Torch version
        if let Some(v) = &self.torch {
            bytes.push(1);
            bytes.extend_from_slice(&v.to_bytes());
        } else {
            bytes.push(0);
        }

        // CUDA version
        if let Some(v) = &self.cuda {
            bytes.push(1);
            bytes.extend_from_slice(&v.to_bytes());
        } else {
            bytes.push(0);
        }

        // Triton version
        if let Some(v) = &self.triton {
            bytes.push(1);
            bytes.extend_from_slice(&v.to_bytes());
        } else {
            bytes.push(0);
        }

        bytes
    }
}


/// Extended cache key including version information.
///
/// Use this for production caching where version compatibility matters.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct VersionedCacheKey {
    /// Base cache key (graph, inputs, device).
    pub key: CacheKey,
    /// Environment versions.
    pub versions: EnvironmentVersions,
}

impl VersionedCacheKey {
    /// Create a new versioned cache key.
    #[must_use]
    pub fn new(key: CacheKey, versions: EnvironmentVersions) -> Self {
        Self { key, versions }
    }

    /// Check if this key is compatible with cached artifacts from another key.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Base keys must match exactly
        if self.key.to_storage_key() != other.key.to_storage_key() {
            return false;
        }

        // Device must be compatible
        if !self.key.device.is_compatible_with(&other.key.device) {
            return false;
        }

        // Versions must be compatible
        self.versions.is_compatible_with(&other.versions)
    }

    /// Convert to storage key string (includes version hash).
    #[must_use]
    pub fn to_storage_key(&self) -> String {
        let hasher = Blake3Hasher::new();

        let mut bytes = self.key.to_bytes();
        bytes.extend_from_slice(&self.versions.to_bytes());

        let hash = hasher.hash(&bytes);
        hex::encode(&hash[..16])
    }
}

/// Normalize a file path for cache key generation.
///
/// This function removes machine-specific path components to enable
/// cache sharing across different environments.
///
/// Transformations:
/// - `/home/user/project/src/model.py` → `src/model.py`
/// - `/Users/name/Documents/project/lib.py` → `lib.py`
/// - `C:\Users\name\project\model.py` → `model.py`
///
/// If the path cannot be normalized, returns a hash of the path.
#[must_use]
pub fn normalize_path(path: &str) -> String {
    let path = Path::new(path);

    // Try to extract meaningful suffix
    // Common project directories to look for
    let markers = ["src", "lib", "models", "scripts", "python", "torch"];

    let components: Vec<_> = path.components().collect();

    // Find the last occurrence of a marker directory
    for (i, comp) in components.iter().enumerate() {
        if let std::path::Component::Normal(name) = comp {
            if let Some(name_str) = name.to_str() {
                if markers.contains(&name_str) {
                    // Return path from marker onwards
                    let suffix: std::path::PathBuf = components[i..].iter().collect();
                    return suffix.to_string_lossy().to_string();
                }
            }
        }
    }

    // If no marker found, just use the filename
    if let Some(name) = path.file_name() {
        return name.to_string_lossy().to_string();
    }

    // Fall back to hashing the entire path
    let hasher = Blake3Hasher::new();
    let hash = hasher.hash(path.to_string_lossy().as_bytes());
    format!("path_{}", hex::encode(&hash[..8]))
}

/// Normalize source code by removing or hashing file paths.
///
/// This function processes source code to replace absolute paths
/// with normalized versions, enabling cache sharing.
#[must_use]
pub fn normalize_source(source: &str) -> String {
    // Common path patterns to look for (simplified regex-like matching)
    // Pattern: File paths starting with / or drive letter
    let lines: Vec<&str> = source.lines().collect();
    let normalized_lines: Vec<String> = lines.iter().map(|line| normalize_line(line)).collect();

    normalized_lines.join("\n")
}

/// Normalize a single line of source code.
fn normalize_line(line: &str) -> String {
    // Look for quoted path strings
    let mut result = String::new();
    let mut in_string = false;
    let mut string_char = '"';
    let mut current_string = String::new();
    let chars = line.chars();

    for c in chars {
        if !in_string {
            if c == '"' || c == '\'' {
                in_string = true;
                string_char = c;
                current_string.clear();
                result.push(c);
            } else {
                result.push(c);
            }
        } else if c == string_char && !current_string.ends_with('\\') {
            // End of string - check if it looks like a path
            if looks_like_path(&current_string) {
                let normalized = normalize_path(&current_string);
                result.push_str(&normalized);
            } else {
                result.push_str(&current_string);
            }
            result.push(c);
            in_string = false;
        } else {
            current_string.push(c);
        }
    }

    // Handle unterminated strings
    if in_string {
        result.push_str(&current_string);
    }

    result
}

/// Check if a string looks like a file path.
fn looks_like_path(s: &str) -> bool {
    // Unix absolute paths
    if s.starts_with('/') && s.contains('/') && !s.starts_with("//") {
        return true;
    }

    // Windows paths
    if s.len() >= 3 && s.chars().nth(1) == Some(':') {
        return true;
    }

    false
}

/// Convert DType to u8 for serialization.
fn dtype_to_u8(dtype: &DType) -> u8 {
    match dtype {
        DType::Float32 => 0,
        DType::Float16 => 1,
        DType::BFloat16 => 2,
        DType::Float64 => 3,
        DType::Int32 => 4,
        DType::Int64 => 5,
        DType::Int8 => 6,
        DType::UInt8 => 7,
        DType::Bool => 8,
    }
}

/// Convert u8 to DType.
fn dtype_from_u8(byte: u8) -> Option<DType> {
    match byte {
        0 => Some(DType::Float32),
        1 => Some(DType::Float16),
        2 => Some(DType::BFloat16),
        3 => Some(DType::Float64),
        4 => Some(DType::Int32),
        5 => Some(DType::Int64),
        6 => Some(DType::Int8),
        7 => Some(DType::UInt8),
        8 => Some(DType::Bool),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_deterministic() {
        let graph_hash = [1u8; 32];
        let inputs = vec![
            InputSpec::new(vec![1, 3, 224, 224], DType::Float32),
            InputSpec::new(vec![1, 1000], DType::Float32),
        ];
        let device = DeviceClass::CudaCompute { major: 8, minor: 0 };

        let key1 = CacheKey::new(graph_hash, inputs.clone(), device.clone());
        let key2 = CacheKey::new(graph_hash, inputs, device);

        assert_eq!(key1.to_storage_key(), key2.to_storage_key());
    }

    #[test]
    fn test_cache_key_different_inputs() {
        let graph_hash = [1u8; 32];
        let inputs1 = vec![InputSpec::new(vec![1, 3, 224, 224], DType::Float32)];
        let inputs2 = vec![InputSpec::new(vec![1, 3, 256, 256], DType::Float32)];

        let key1 = CacheKey::new(graph_hash, inputs1, DeviceClass::CudaAny);
        let key2 = CacheKey::new(graph_hash, inputs2, DeviceClass::CudaAny);

        assert_ne!(key1.to_storage_key(), key2.to_storage_key());
    }

    #[test]
    fn test_cache_key_from_bytes() {
        let inputs = vec![InputSpec::new(vec![1, 3, 224, 224], DType::Float32)];
        let key = CacheKey::from_bytes(b"graph source code", inputs, DeviceClass::Cpu);

        // Should produce valid storage key
        let storage_key = key.to_storage_key();
        assert_eq!(storage_key.len(), 32); // 16 bytes = 32 hex chars
    }

    #[test]
    fn test_cache_key_serialization_roundtrip() {
        let graph_hash = [42u8; 32];
        let inputs = vec![
            InputSpec::new(vec![1, 3, 224, 224], DType::Float32),
            InputSpec::new(vec![1, 1000], DType::Int64),
        ];
        let device = DeviceClass::CudaCompute { major: 9, minor: 0 };

        let key = CacheKey::new(graph_hash, inputs, device);
        let bytes = key.to_bytes();
        let recovered = CacheKey::from_serialized_bytes(&bytes).unwrap();

        assert_eq!(key, recovered);
    }

    #[test]
    fn test_input_spec_static() {
        let spec = InputSpec::static_shape(vec![1, 3, 224, 224], DType::Float32);

        assert!(!spec.is_dynamic());
        assert_eq!(spec.numel(), Some(1 * 3 * 224 * 224));
    }

    #[test]
    fn test_input_spec_dynamic() {
        let spec = InputSpec::dynamic_batch(vec![1, 3, 224, 224], DType::Float32);

        assert!(spec.is_dynamic());
        assert_eq!(spec.numel(), None);
        assert_eq!(spec.shape[0], -1);
    }

    #[test]
    fn test_device_class_compatibility() {
        let cuda_any = DeviceClass::CudaAny;
        let cuda_80 = DeviceClass::cuda(8, 0); // Ampere
        let cuda_90 = DeviceClass::cuda(9, 0); // Hopper
        let cpu = DeviceClass::Cpu;

        // CudaAny is compatible with any CUDA
        assert!(cuda_any.is_compatible_with(&cuda_any));
        assert!(cuda_any.is_compatible_with(&cuda_80));
        assert!(cuda_80.is_compatible_with(&cuda_any));

        // Higher compute capability can use lower's artifacts
        assert!(cuda_90.is_compatible_with(&cuda_80));
        assert!(!cuda_80.is_compatible_with(&cuda_90));

        // Same is compatible
        assert!(cuda_80.is_compatible_with(&cuda_80));

        // CPU only with CPU
        assert!(cpu.is_compatible_with(&cpu));
        assert!(!cpu.is_compatible_with(&cuda_any));
        assert!(!cuda_any.is_compatible_with(&cpu));
    }

    #[test]
    fn test_device_class_serialization_roundtrip() {
        let devices = vec![
            DeviceClass::CudaAny,
            DeviceClass::CudaCompute { major: 8, minor: 6 },
            DeviceClass::Cpu,
        ];

        for device in devices {
            let bytes = device.to_bytes();
            let (recovered, _) = DeviceClass::from_bytes(&bytes).unwrap();
            assert_eq!(device, recovered);
        }
    }

    #[test]
    fn test_input_spec_serialization_roundtrip() {
        let specs = vec![
            InputSpec::new(vec![1, 3, 224, 224], DType::Float32),
            InputSpec::new(vec![-1, 512], DType::BFloat16),
            InputSpec::new(vec![1024], DType::Int64),
        ];

        for spec in specs {
            let bytes = spec.to_bytes();
            let (recovered, _) = InputSpec::from_bytes(&bytes).unwrap();
            assert_eq!(spec, recovered);
        }
    }

    #[test]
    fn test_storage_key_length() {
        let key = CacheKey::new(
            [0u8; 32],
            vec![InputSpec::new(vec![1, 3, 224, 224], DType::Float32)],
            DeviceClass::CudaAny,
        );

        let storage_key = key.to_storage_key();

        // 32 hex characters (16 bytes)
        assert_eq!(storage_key.len(), 32);

        // Should be valid hex
        assert!(storage_key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // Version tests
    #[test]
    fn test_version_parse() {
        assert_eq!(Version::parse("2.1.0"), Some(Version::new(2, 1)));
        assert_eq!(Version::parse("2.1"), Some(Version::new(2, 1)));
        assert_eq!(Version::parse("12.3.5"), Some(Version::new(12, 3)));
        assert_eq!(Version::parse("2"), None);
        assert_eq!(Version::parse("invalid"), None);
    }

    #[test]
    fn test_version_compatibility() {
        let v21 = Version::new(2, 1);
        let v22 = Version::new(2, 2);
        let v30 = Version::new(3, 0);

        // Same version is compatible
        assert!(v21.is_compatible_with(&v21));

        // Higher minor can use lower minor's cache
        assert!(v22.is_compatible_with(&v21));

        // Lower minor cannot use higher minor's cache
        assert!(!v21.is_compatible_with(&v22));

        // Different major versions are incompatible
        assert!(!v30.is_compatible_with(&v21));
        assert!(!v21.is_compatible_with(&v30));
    }

    #[test]
    fn test_version_display() {
        assert_eq!(Version::new(2, 1).to_string(), "2.1");
        assert_eq!(Version::new(12, 0).to_string(), "12.0");
    }

    #[test]
    fn test_environment_versions_compatibility() {
        let env1 = EnvironmentVersions::new(
            Some(Version::new(2, 1)),
            Some(Version::new(12, 0)),
            Some(Version::new(2, 1)),
        );

        let env2 = EnvironmentVersions::new(
            Some(Version::new(2, 2)),
            Some(Version::new(12, 1)),
            Some(Version::new(2, 2)),
        );

        // Higher versions can use lower's cache
        assert!(env2.is_compatible_with(&env1));

        // Lower versions cannot use higher's cache
        assert!(!env1.is_compatible_with(&env2));
    }

    #[test]
    fn test_environment_versions_cuda_required() {
        let with_cuda =
            EnvironmentVersions::new(Some(Version::new(2, 1)), Some(Version::new(12, 0)), None);

        let without_cuda = EnvironmentVersions::new(Some(Version::new(2, 1)), None, None);

        // Can't use CUDA cache without CUDA
        assert!(!without_cuda.is_compatible_with(&with_cuda));

        // CUDA env can use CPU cache
        assert!(with_cuda.is_compatible_with(&without_cuda));
    }

    #[test]
    fn test_versioned_cache_key() {
        let key = CacheKey::new(
            [1u8; 32],
            vec![InputSpec::new(vec![1, 3, 224, 224], DType::Float32)],
            DeviceClass::CudaAny,
        );

        let versions = EnvironmentVersions::torch_only(Version::new(2, 1));
        let versioned = VersionedCacheKey::new(key, versions);

        let storage_key = versioned.to_storage_key();
        assert_eq!(storage_key.len(), 32);
    }

    // Path normalization tests
    #[test]
    fn test_normalize_path_with_src() {
        assert_eq!(
            normalize_path("/home/user/project/src/model.py"),
            "src/model.py"
        );
        assert_eq!(
            normalize_path("/Users/name/Documents/project/src/layers/attention.py"),
            "src/layers/attention.py"
        );
    }

    #[test]
    fn test_normalize_path_with_markers() {
        assert_eq!(normalize_path("/opt/project/lib/utils.py"), "lib/utils.py");
        assert_eq!(
            normalize_path("/app/models/transformer.py"),
            "models/transformer.py"
        );
    }

    #[test]
    fn test_normalize_path_filename_only() {
        assert_eq!(normalize_path("/var/app/random/dir/module.py"), "module.py");
    }

    #[test]
    fn test_normalize_source() {
        let source = r#"
import torch
# File: "/home/user/project/src/model.py"
path = "/Users/name/Documents/project/lib/utils.py"
"#;

        let normalized = normalize_source(source);

        // Paths should be normalized
        assert!(normalized.contains("src/model.py"));
        assert!(normalized.contains("lib/utils.py"));

        // Non-path strings should be preserved
        assert!(normalized.contains("import torch"));
    }

    #[test]
    fn test_normalize_source_preserves_non_paths() {
        let source = r#"
name = "my_model"
url = "https://example.com/model"
relative = "data/train.json"
"#;

        let normalized = normalize_source(source);

        // Non-absolute paths should be preserved
        assert!(normalized.contains(r#""my_model""#));
        assert!(normalized.contains(r#""https://example.com/model""#));
        assert!(normalized.contains(r#""data/train.json""#));
    }

    #[test]
    fn test_looks_like_path() {
        // Unix paths
        assert!(looks_like_path("/home/user/file.py"));
        assert!(looks_like_path("/usr/local/bin/python"));

        // Windows paths
        assert!(looks_like_path("C:\\Users\\name\\file.py"));

        // Not paths
        assert!(!looks_like_path("model"));
        assert!(!looks_like_path("data/train.json")); // relative
        assert!(!looks_like_path("https://example.com")); // URL
        assert!(!looks_like_path("//network/share")); // UNC starts with //
    }

    #[test]
    fn test_version_serialization_roundtrip() {
        let versions = vec![
            Version::new(0, 0),
            Version::new(2, 1),
            Version::new(12, 3),
            Version::new(u16::MAX, u16::MAX),
        ];

        for version in versions {
            let bytes = version.to_bytes();
            let (recovered, consumed) = Version::from_bytes(&bytes).unwrap();
            assert_eq!(version, recovered);
            assert_eq!(consumed, 4);
        }
    }
}
