//! Compression abstractions.

use crate::error::{MithrilError, Result};

/// Trait for compression algorithms.
pub trait Compressor: Send + Sync {
    /// Compress data.
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Decompress data.
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Compress with known output size hint.
    fn compress_with_hint(&self, data: &[u8], _size_hint: usize) -> Result<Vec<u8>> {
        self.compress(data)
    }

    /// Decompress with known output size.
    fn decompress_exact(&self, data: &[u8], output_size: usize) -> Result<Vec<u8>> {
        let decompressed = self.decompress(data)?;
        if decompressed.len() != output_size {
            return Err(MithrilError::Decompression(format!(
                "Expected {} bytes, got {}",
                output_size,
                decompressed.len()
            )));
        }
        Ok(decompressed)
    }
}

/// Zstd compressor with configurable level.
pub struct ZstdCompressor {
    level: i32,
}

impl ZstdCompressor {
    /// Create a new Zstd compressor with default level (3).
    #[must_use]
    pub fn new() -> Self {
        Self::with_level(3)
    }

    /// Create a new Zstd compressor with specified level.
    ///
    /// Level ranges from -7 (fastest) to 22 (best compression).
    /// Typical values: 1-4 for fast, 5-9 for balanced, 10+ for max compression.
    #[must_use]
    pub fn with_level(level: i32) -> Self {
        Self { level }
    }
}

impl Default for ZstdCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::encode_all(data, self.level).map_err(|e| MithrilError::Compression(e.to_string()))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| MithrilError::Decompression(e.to_string()))
    }
}

/// LZ4 compressor for maximum speed.
pub struct Lz4Compressor;

impl Lz4Compressor {
    /// Create a new LZ4 compressor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for Lz4Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for Lz4Compressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| MithrilError::Decompression(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_roundtrip() {
        let compressor = ZstdCompressor::new();
        let original = b"hello world, this is a test of compression!".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(original.as_slice(), decompressed.as_slice());
        assert!(compressed.len() < original.len());
    }

    #[test]
    fn test_lz4_roundtrip() {
        let compressor = Lz4Compressor::new();
        let original = b"hello world, this is a test of compression!".repeat(100);

        let compressed = compressor.compress(&original).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(original.as_slice(), decompressed.as_slice());
        assert!(compressed.len() < original.len());
    }
}
