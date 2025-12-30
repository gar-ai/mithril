//! Common types for Mithril.

use serde::{Deserialize, Serialize};

/// Data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// Brain floating point (16-bit)
    BFloat16,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// Boolean
    Bool,
}

impl DType {
    /// Size in bytes of a single element.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float64 | Self::Int64 => 8,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
        }
    }
}

/// Metadata about a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Name/key of the tensor
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Offset in bytes within storage
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
}

impl TensorMeta {
    /// Create new tensor metadata.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let size = numel * dtype.size_bytes();
        Self {
            name: name.into(),
            shape,
            dtype,
            offset: 0,
            size,
        }
    }

    /// Number of elements in the tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::BFloat16.size_bytes(), 2);
        assert_eq!(DType::Int8.size_bytes(), 1);
    }

    #[test]
    fn test_tensor_meta() {
        let meta = TensorMeta::new("weight", vec![1024, 512], DType::BFloat16);
        assert_eq!(meta.numel(), 1024 * 512);
        assert_eq!(meta.size, 1024 * 512 * 2);
    }
}
