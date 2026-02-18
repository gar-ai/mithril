//! Streaming/chunked compression for large checkpoint data.
//!
//! For very large models (70B+ parameters = 140GB+), buffering the entire
//! checkpoint in memory before compression is impractical. This module provides
//! a streaming API that compresses checkpoint data in chunks as it is written,
//! keeping memory usage bounded.
//!
//! ## Architecture
//!
//! The streaming pipeline mirrors the standard [`CheckpointCompressor`] pipeline
//! but operates on fixed-size blocks:
//!
//! 1. **Buffering** -- Incoming chunks are accumulated until a full block is ready.
//! 2. **Byte grouping** -- Each block is byte-grouped independently (bf16/fp16/fp32).
//! 3. **Zstd streaming compression** -- Grouped blocks are fed into a zstd streaming
//!    encoder that maintains dictionary context across blocks for good compression.
//! 4. **Finalization** -- On [`StreamingWriter::finish`], any remaining partial block
//!    is flushed and the zstd frame is finalized.
//!
//! ## Compatibility
//!
//! **Important:** Because byte grouping is applied per-block rather than over the
//! entire tensor, the compressed output is *not* byte-identical to
//! [`CheckpointCompressor::compress`]. However, decompression via
//! [`StreamingCompressor::decompress`] correctly reverses the process, and a
//! full round-trip is bit-exact.
//!
//! ## Example
//!
//! ```rust
//! use mithril_checkpoint::streaming::StreamingCompressor;
//! use mithril_checkpoint::pipeline::CompressionConfig;
//! use mithril_core::types::DType;
//!
//! let mut compressor = StreamingCompressor::new(CompressionConfig::default());
//!
//! // Compress bf16 data in 64 KB chunks
//! let data = vec![0u8; 200_000];
//! let mut writer = compressor.begin(DType::BFloat16);
//!
//! for chunk in data.chunks(65536) {
//!     writer.write_chunk(chunk).unwrap();
//! }
//!
//! let compressed = writer.finish().unwrap();
//!
//! // Decompress
//! let decompressed = StreamingCompressor::decompress(
//!     &compressed,
//!     DType::BFloat16,
//! ).unwrap();
//! assert_eq!(data, decompressed);
//! ```
//!
//! [`CheckpointCompressor`]: crate::pipeline::CheckpointCompressor

use std::io::Write;

use mithril_core::error::MithrilError;
use mithril_core::types::DType;
use mithril_core::Result;

use crate::bytegroup::{byte_group_bf16, byte_group_fp32, byte_ungroup_bf16, byte_ungroup_fp32};
use crate::pipeline::CompressionConfig;

/// Default block size for streaming compression (1 MB).
///
/// Each block is byte-grouped independently before being fed to zstd. Larger
/// blocks amortise the overhead of per-block grouping but increase peak memory.
/// 1 MB is a good balance for modern hardware.
const DEFAULT_BLOCK_SIZE: usize = 1024 * 1024;

/// Header magic bytes identifying a streaming-compressed checkpoint.
///
/// This distinguishes streaming output from single-shot [`CheckpointCompressor`]
/// output so that the correct decompression path is chosen.
const STREAMING_MAGIC: &[u8; 4] = b"MSTS"; // Mithril STreaming

/// Header version.
const HEADER_VERSION: u8 = 1;

/// Header size in bytes.
///
/// Layout (11 bytes total):
/// - [0..4]  magic       (4 bytes) -- `MSTS`
/// - [4]     version     (1 byte)
/// - [5]     dtype tag   (1 byte)
/// - [6]     flags       (1 byte)  -- bit 0: byte_grouping enabled
/// - [7..11] block_size  (4 bytes, little-endian u32)
const HEADER_SIZE: usize = 11;

/// Flag bit: byte grouping was enabled during compression.
const FLAG_BYTE_GROUPING: u8 = 0x01;

// ---------------------------------------------------------------------------
// StreamingCompressor
// ---------------------------------------------------------------------------

/// Streaming compressor for large checkpoint data.
///
/// Wraps a [`CompressionConfig`] and provides a chunked compression API that
/// keeps memory usage proportional to the block size rather than the full
/// tensor size.
///
/// See the [module-level documentation](self) for details.
pub struct StreamingCompressor {
    config: CompressionConfig,
    block_size: usize,
}

impl StreamingCompressor {
    /// Create a new streaming compressor with the given configuration.
    ///
    /// Uses the default block size of 1 MB.
    #[must_use]
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }

    /// Create a streaming compressor with a custom block size.
    ///
    /// The block size controls how much data is byte-grouped at once. It must
    /// be a positive value. For best results, choose a multiple of the dtype
    /// element size (2 for bf16/fp16, 4 for fp32).
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is zero.
    #[must_use]
    pub fn with_block_size(config: CompressionConfig, block_size: usize) -> Self {
        assert!(block_size > 0, "Block size must be positive");
        Self { config, block_size }
    }

    /// Begin a new streaming compression session.
    ///
    /// Returns a [`StreamingWriter`] that accepts chunks of raw tensor data
    /// and compresses them incrementally.
    ///
    /// # Arguments
    ///
    /// * `dtype` -- The data type of the tensor being compressed. This
    ///   determines the byte-grouping strategy (bf16/fp16 get 2-byte grouping,
    ///   fp32 gets 4-byte grouping, everything else passes through).
    pub fn begin(&mut self, dtype: DType) -> StreamingWriter<'_> {
        StreamingWriter::new(&self.config, self.block_size, dtype)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }

    /// Get the block size.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Decompress data produced by [`StreamingWriter::finish`].
    ///
    /// This is a static method because decompression does not require any
    /// mutable compressor state.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not valid streaming-compressed output
    /// or if zstd decompression fails.
    pub fn decompress(data: &[u8], dtype: DType) -> Result<Vec<u8>> {
        // Parse the header.
        if data.len() < HEADER_SIZE {
            return Err(MithrilError::Decompression(
                "Streaming data too short for header".into(),
            ));
        }
        if &data[0..4] != STREAMING_MAGIC {
            return Err(MithrilError::Decompression(
                "Invalid streaming magic bytes".into(),
            ));
        }
        if data[4] != HEADER_VERSION {
            return Err(MithrilError::Decompression(format!(
                "Unsupported streaming header version: {}",
                data[4]
            )));
        }

        // data[5] is the dtype tag -- we trust the caller's `dtype` for now
        // but could cross-check in a future version.

        let flags = data[6];
        let byte_grouping = (flags & FLAG_BYTE_GROUPING) != 0;

        let block_size = u32::from_le_bytes([data[7], data[8], data[9], data[10]]) as usize;
        let compressed_payload = &data[HEADER_SIZE..];

        // Decompress the zstd stream to get the concatenated (possibly grouped) blocks.
        let grouped_bytes = zstd::decode_all(compressed_payload)
            .map_err(|e| MithrilError::Decompression(e.to_string()))?;

        // If byte grouping was not used, the decompressed bytes are already the
        // raw tensor data.
        if !byte_grouping {
            return Ok(grouped_bytes);
        }

        // Ungroup each block.
        let element_size = dtype_element_size(dtype);
        let effective_block = align_block_size(block_size, element_size);

        let mut output = Vec::with_capacity(grouped_bytes.len());
        let mut offset = 0;
        while offset < grouped_bytes.len() {
            let remaining = grouped_bytes.len() - offset;
            let chunk_len = remaining.min(effective_block);
            let block = &grouped_bytes[offset..offset + chunk_len];

            let ungrouped = ungroup_block(block, dtype);
            output.extend_from_slice(&ungrouped);

            offset += chunk_len;
        }

        Ok(output)
    }
}

impl Default for StreamingCompressor {
    fn default() -> Self {
        Self::new(CompressionConfig::default())
    }
}

// ---------------------------------------------------------------------------
// StreamingWriter
// ---------------------------------------------------------------------------

/// A writer that accepts chunks of tensor data and compresses them
/// incrementally.
///
/// Created by [`StreamingCompressor::begin`]. Accumulates data in an internal
/// buffer; whenever the buffer reaches the configured block size, the block is
/// byte-grouped and fed into a zstd streaming encoder.
///
/// Call [`finish`](Self::finish) to flush any remaining data and retrieve the
/// final compressed bytes.
pub struct StreamingWriter<'a> {
    /// Reference to the compression config (borrows from StreamingCompressor).
    config: &'a CompressionConfig,
    /// Data type of the tensor being compressed.
    dtype: DType,
    /// Target block size before byte grouping.
    block_size: usize,
    /// Effective block size after alignment to dtype element boundaries.
    effective_block_size: usize,
    /// Accumulation buffer for incoming chunks.
    buffer: Vec<u8>,
    /// Total bytes written (before compression).
    total_written: usize,
    /// Zstd streaming encoder writing into `compressed_output`.
    encoder: zstd::Encoder<'static, Vec<u8>>,
    /// Whether the header has been written to the encoder yet.
    header_written: bool,
}

impl<'a> StreamingWriter<'a> {
    /// Create a new streaming writer.
    fn new(config: &'a CompressionConfig, block_size: usize, dtype: DType) -> Self {
        let element_size = dtype_element_size(dtype);
        let effective_block_size = align_block_size(block_size, element_size);

        // Pre-allocate: we write a small header + compressed data.
        let output_buf: Vec<u8> = Vec::with_capacity(1024 * 64);
        let encoder = zstd::Encoder::new(output_buf, config.zstd_level)
            .expect("Failed to create zstd encoder");

        Self {
            config,
            dtype,
            block_size,
            effective_block_size,
            buffer: Vec::with_capacity(effective_block_size),
            total_written: 0,
            encoder,
            header_written: false,
        }
    }

    /// Write a chunk of raw tensor data.
    ///
    /// The chunk can be of any size. Internally, data is accumulated and
    /// compressed in blocks of the configured block size.
    ///
    /// # Errors
    ///
    /// Returns an error if zstd compression fails (extremely unlikely for
    /// valid data).
    pub fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        self.total_written += data.len();

        let mut remaining = data;
        while !remaining.is_empty() {
            let space = self.effective_block_size - self.buffer.len();
            let to_copy = remaining.len().min(space);

            self.buffer.extend_from_slice(&remaining[..to_copy]);
            remaining = &remaining[to_copy..];

            if self.buffer.len() >= self.effective_block_size {
                self.flush_block()?;
            }
        }

        Ok(())
    }

    /// Finish compression and return the complete compressed output.
    ///
    /// This flushes any remaining buffered data, finalizes the zstd frame,
    /// and returns the compressed bytes including the streaming header.
    ///
    /// # Errors
    ///
    /// Returns an error if zstd finalization fails.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        // Flush any remaining buffered data.
        if !self.buffer.is_empty() {
            self.flush_block()?;
        }

        // Finalize the zstd stream.
        let compressed_payload = self
            .encoder
            .finish()
            .map_err(|e| MithrilError::Compression(e.to_string()))?;

        // Build the final output: header + compressed payload.
        let flags = if self.config.byte_grouping {
            FLAG_BYTE_GROUPING
        } else {
            0
        };

        let mut output = Vec::with_capacity(HEADER_SIZE + compressed_payload.len());
        output.extend_from_slice(STREAMING_MAGIC);
        output.push(HEADER_VERSION);
        output.push(dtype_to_tag(self.dtype));
        output.push(flags);
        output.extend_from_slice(&(self.block_size as u32).to_le_bytes());
        output.extend_from_slice(&compressed_payload);

        Ok(output)
    }

    /// Get the total number of uncompressed bytes written so far.
    #[must_use]
    pub fn bytes_written(&self) -> usize {
        self.total_written
    }

    /// Flush the internal buffer as one grouped block into the zstd encoder.
    fn flush_block(&mut self) -> Result<()> {
        let block = std::mem::take(&mut self.buffer);
        // Re-allocate the buffer for the next block.
        self.buffer = Vec::with_capacity(self.effective_block_size);

        let grouped = if self.config.byte_grouping {
            group_block(&block, self.dtype)
        } else {
            block
        };

        // Ensure the header flag is set (the header is actually written in
        // finish(); here we just track that data has been written).
        self.header_written = true;

        self.encoder
            .write_all(&grouped)
            .map_err(|e| MithrilError::Compression(e.to_string()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Return the element size in bytes for byte-grouping purposes.
///
/// Types that are not byte-grouped return 1 (no alignment requirement).
fn dtype_element_size(dtype: DType) -> usize {
    match dtype {
        DType::BFloat16 | DType::Float16 => 2,
        DType::Float32 => 4,
        _ => 1,
    }
}

/// Round `block_size` down to the nearest multiple of `element_size`.
///
/// This ensures that each block contains a whole number of elements so that
/// byte grouping works correctly.
fn align_block_size(block_size: usize, element_size: usize) -> usize {
    if element_size <= 1 {
        return block_size;
    }
    let aligned = (block_size / element_size) * element_size;
    // Ensure we don't produce a zero-sized block.
    if aligned == 0 {
        element_size
    } else {
        aligned
    }
}

/// Byte-group a block according to the given dtype.
fn group_block(data: &[u8], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::BFloat16 | DType::Float16 => {
            if data.len() % 2 == 0 {
                byte_group_bf16(data)
            } else {
                // Partial element at the end -- group what we can,
                // append the leftover byte.
                let aligned = (data.len() / 2) * 2;
                let mut grouped = byte_group_bf16(&data[..aligned]);
                grouped.extend_from_slice(&data[aligned..]);
                grouped
            }
        }
        DType::Float32 => {
            if data.len() % 4 == 0 {
                byte_group_fp32(data)
            } else {
                let aligned = (data.len() / 4) * 4;
                let mut grouped = byte_group_fp32(&data[..aligned]);
                grouped.extend_from_slice(&data[aligned..]);
                grouped
            }
        }
        _ => data.to_vec(),
    }
}

/// Byte-ungroup a block according to the given dtype.
fn ungroup_block(data: &[u8], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::BFloat16 | DType::Float16 => {
            if data.len() % 2 == 0 {
                byte_ungroup_bf16(data)
            } else {
                let aligned = (data.len() / 2) * 2;
                let mut ungrouped = byte_ungroup_bf16(&data[..aligned]);
                ungrouped.extend_from_slice(&data[aligned..]);
                ungrouped
            }
        }
        DType::Float32 => {
            if data.len() % 4 == 0 {
                byte_ungroup_fp32(data)
            } else {
                let aligned = (data.len() / 4) * 4;
                let mut ungrouped = byte_ungroup_fp32(&data[..aligned]);
                ungrouped.extend_from_slice(&data[aligned..]);
                ungrouped
            }
        }
        _ => data.to_vec(),
    }
}

/// Map a [`DType`] to a single-byte tag for the streaming header.
fn dtype_to_tag(dtype: DType) -> u8 {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compress data in one shot using the non-streaming pipeline for
    /// comparison.
    fn compress_oneshot(data: &[u8], dtype: DType, config: &CompressionConfig) -> Vec<u8> {
        use crate::pipeline::CheckpointCompressor;
        let compressor = CheckpointCompressor::new(config.clone());
        compressor.compress(data, dtype).unwrap()
    }

    /// Helper: decompress one-shot compressed data.
    fn decompress_oneshot(
        compressed: &[u8],
        dtype: DType,
        original_size: usize,
        config: &CompressionConfig,
    ) -> Vec<u8> {
        use crate::pipeline::CheckpointCompressor;
        let compressor = CheckpointCompressor::new(config.clone());
        compressor
            .decompress(compressed, dtype, original_size)
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // Round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_bf16() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = (0..10_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::BFloat16);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(data, decompressed, "bf16 round-trip must be bit-exact");
    }

    #[test]
    fn test_roundtrip_fp32() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = (0..5_000u32).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::Float32);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::Float32).unwrap();
        assert_eq!(data, decompressed, "fp32 round-trip must be bit-exact");
    }

    #[test]
    fn test_roundtrip_int8() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();
        let mut writer = compressor.begin(DType::Int8);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::Int8).unwrap();
        assert_eq!(data, decompressed, "int8 round-trip must be bit-exact");
    }

    #[test]
    fn test_roundtrip_empty() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = vec![];
        let mut writer = compressor.begin(DType::BFloat16);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(
            data, decompressed,
            "empty round-trip must produce empty output"
        );
    }

    // -----------------------------------------------------------------------
    // Various chunk sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_size_1kb() {
        roundtrip_with_chunk_size(1024, DType::BFloat16);
    }

    #[test]
    fn test_chunk_size_64kb() {
        roundtrip_with_chunk_size(64 * 1024, DType::BFloat16);
    }

    #[test]
    fn test_chunk_size_1mb() {
        roundtrip_with_chunk_size(1024 * 1024, DType::BFloat16);
    }

    #[test]
    fn test_chunk_size_1_byte() {
        // Extreme: write one byte at a time.
        roundtrip_with_chunk_size(1, DType::BFloat16);
    }

    #[test]
    fn test_chunk_size_odd() {
        // Chunk size that does not align with bf16 element size.
        roundtrip_with_chunk_size(333, DType::BFloat16);
    }

    #[test]
    fn test_chunk_size_fp32_various() {
        for &chunk_size in &[1, 7, 1024, 65536] {
            roundtrip_with_chunk_size(chunk_size, DType::Float32);
        }
    }

    fn roundtrip_with_chunk_size(chunk_size: usize, dtype: DType) {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        // Use 200 KB of data so we exercise multiple blocks with default 1 MB
        // block size, and many blocks with small chunk sizes.
        let data: Vec<u8> = (0..200_000u32)
            .flat_map(|x| {
                let val = (x % 256) as u8;
                std::iter::once(val)
            })
            .collect();

        let mut writer = compressor.begin(dtype);
        for chunk in data.chunks(chunk_size) {
            writer.write_chunk(chunk).unwrap();
        }
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, dtype).unwrap();
        assert_eq!(
            data, decompressed,
            "Round-trip failed for chunk_size={chunk_size}, dtype={dtype:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Streaming vs one-shot produce same decompressed output
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_decompresses_to_same_as_oneshot() {
        // The *compressed* bytes will differ (per-block vs whole-tensor grouping),
        // but decompression of the streaming output must produce the same raw data.
        let config = CompressionConfig::default();
        let data: Vec<u8> = (0..10_000u16).flat_map(|x| x.to_le_bytes()).collect();

        // One-shot
        let oneshot_compressed = compress_oneshot(&data, DType::BFloat16, &config);
        let oneshot_decompressed =
            decompress_oneshot(&oneshot_compressed, DType::BFloat16, data.len(), &config);

        // Streaming
        let mut compressor = StreamingCompressor::new(config);
        let mut writer = compressor.begin(DType::BFloat16);
        writer.write_chunk(&data).unwrap();
        let streaming_compressed = writer.finish().unwrap();
        let streaming_decompressed =
            StreamingCompressor::decompress(&streaming_compressed, DType::BFloat16).unwrap();

        assert_eq!(oneshot_decompressed, streaming_decompressed);
        assert_eq!(data, streaming_decompressed);
    }

    // -----------------------------------------------------------------------
    // Byte grouping disabled
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_no_byte_grouping() {
        let config = CompressionConfig {
            zstd_level: 3,
            byte_grouping: false,
        };
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = (0..10_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::BFloat16);
        for chunk in data.chunks(4096) {
            writer.write_chunk(chunk).unwrap();
        }
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(data, decompressed);
    }

    // -----------------------------------------------------------------------
    // Custom block size
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_block_size_small() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::with_block_size(config, 256);

        let data: Vec<u8> = (0..10_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::BFloat16);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_custom_block_size_large() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::with_block_size(config, 4 * 1024 * 1024);

        let data: Vec<u8> = (0..50_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::BFloat16);
        for chunk in data.chunks(8192) {
            writer.write_chunk(chunk).unwrap();
        }
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(data, decompressed);
    }

    // -----------------------------------------------------------------------
    // Compression config presets
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_config() {
        let config = CompressionConfig::fast();
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = (0..10_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::BFloat16);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_best_config() {
        let config = CompressionConfig::best();
        let mut compressor = StreamingCompressor::new(config);

        let data: Vec<u8> = (0..10_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer = compressor.begin(DType::BFloat16);
        writer.write_chunk(&data).unwrap();
        let compressed = writer.finish().unwrap();

        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(data, decompressed);
    }

    // -----------------------------------------------------------------------
    // bytes_written tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_written_tracking() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        let mut writer = compressor.begin(DType::BFloat16);
        assert_eq!(writer.bytes_written(), 0);

        writer.write_chunk(&[0u8; 1000]).unwrap();
        assert_eq!(writer.bytes_written(), 1000);

        writer.write_chunk(&[0u8; 500]).unwrap();
        assert_eq!(writer.bytes_written(), 1500);

        let _ = writer.finish().unwrap();
    }

    // -----------------------------------------------------------------------
    // Realistic weight data
    // -----------------------------------------------------------------------

    #[test]
    fn test_realistic_bf16_weights() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        // Simulate realistic neural network bf16 weights.
        let mut rng_state = 42u64;
        let data: Vec<u8> = (0..50_000)
            .flat_map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand = (rng_state >> 33) as u16;

                let exp = (123 + (rand % 9)) as u8;
                let mantissa = (rand & 0x7F) as u8;
                let sign = ((rand >> 15) & 1) as u8;

                let bf16 = ((sign as u16) << 15) | ((exp as u16) << 7) | (mantissa as u16);
                bf16.to_le_bytes()
            })
            .collect();

        let mut writer = compressor.begin(DType::BFloat16);
        // Write in 64 KB chunks, mimicking a real streaming scenario.
        for chunk in data.chunks(65536) {
            writer.write_chunk(chunk).unwrap();
        }
        let compressed = writer.finish().unwrap();

        // Verify round-trip.
        let decompressed = StreamingCompressor::decompress(&compressed, DType::BFloat16).unwrap();
        assert_eq!(
            data, decompressed,
            "Realistic bf16 round-trip must be bit-exact"
        );

        // The streaming output should achieve some compression (header overhead
        // makes it slightly larger than one-shot for small data, but the zstd
        // payload should still be smaller than the raw data).
        // Subtract the 10-byte header for a fair comparison.
        let payload_size = compressed.len() - 10;
        assert!(
            payload_size < data.len(),
            "Expected compression: payload={payload_size}, original={}",
            data.len()
        );
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_decompress_invalid_magic() {
        let bad_data = b"BADDxxxxxxxx";
        let result = StreamingCompressor::decompress(bad_data, DType::BFloat16);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_too_short() {
        let bad_data = b"MSTS";
        let result = StreamingCompressor::decompress(bad_data, DType::BFloat16);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_bad_version() {
        let mut bad_data = vec![0u8; HEADER_SIZE + 10];
        bad_data[..4].copy_from_slice(STREAMING_MAGIC);
        bad_data[4] = 99; // Bad version
        let result = StreamingCompressor::decompress(&bad_data, DType::BFloat16);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Block size alignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_size_alignment() {
        // Block size of 5 with bf16 (element_size=2) should align to 4.
        assert_eq!(align_block_size(5, 2), 4);
        // Block size of 7 with fp32 (element_size=4) should align to 4.
        assert_eq!(align_block_size(7, 4), 4);
        // Block size of 1 with bf16 should become element_size.
        assert_eq!(align_block_size(1, 2), 2);
        // Block size of 1024 with int8 (element_size=1) stays 1024.
        assert_eq!(align_block_size(1024, 1), 1024);
    }

    #[test]
    #[should_panic(expected = "Block size must be positive")]
    fn test_zero_block_size_panics() {
        let config = CompressionConfig::default();
        let _ = StreamingCompressor::with_block_size(config, 0);
    }

    // -----------------------------------------------------------------------
    // Multiple calls to begin (reuse compressor)
    // -----------------------------------------------------------------------

    #[test]
    fn test_reuse_compressor() {
        let config = CompressionConfig::default();
        let mut compressor = StreamingCompressor::new(config);

        // First compression session.
        let data1: Vec<u8> = (0..5_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer1 = compressor.begin(DType::BFloat16);
        writer1.write_chunk(&data1).unwrap();
        let compressed1 = writer1.finish().unwrap();

        // Second compression session with different data.
        let data2: Vec<u8> = (5_000..10_000u16).flat_map(|x| x.to_le_bytes()).collect();
        let mut writer2 = compressor.begin(DType::Float32);
        writer2.write_chunk(&data2).unwrap();
        let compressed2 = writer2.finish().unwrap();

        // Both should round-trip correctly.
        let dec1 = StreamingCompressor::decompress(&compressed1, DType::BFloat16).unwrap();
        let dec2 = StreamingCompressor::decompress(&compressed2, DType::Float32).unwrap();

        assert_eq!(data1, dec1);
        assert_eq!(data2, dec2);
    }
}
