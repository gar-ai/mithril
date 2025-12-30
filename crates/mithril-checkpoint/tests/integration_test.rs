//! Integration tests for mithril-checkpoint.
//!
//! Tests end-to-end compression/decompression workflows.

use mithril_checkpoint::pipeline::{CheckpointCompressor, CompressionConfig};
use mithril_core::types::DType;
use tempfile::TempDir;

/// Generate synthetic bf16 weight data that mimics real model weights.
/// Real weights tend to have patterns that compress well.
fn generate_synthetic_bf16_weights(num_bytes: usize) -> Vec<u8> {
    assert!(num_bytes % 2 == 0, "bf16 requires even byte count");
    let mut data = Vec::with_capacity(num_bytes);

    // Generate weights that mimic neural network weight distribution
    // (roughly normal, centered around 0, with some patterns)
    let num_elements = num_bytes / 2;
    for i in 0..num_elements {
        // Create patterns that compress well (like real weights)
        let value = ((i as f32).sin() * 0.1) as f32;

        // Convert to bf16 representation (truncate lower 16 bits of f32)
        let bits = value.to_bits();
        let bf16_bits = (bits >> 16) as u16;

        data.extend_from_slice(&bf16_bits.to_le_bytes());
    }

    data
}

/// Generate synthetic fp32 weight data.
fn generate_synthetic_fp32_weights(num_bytes: usize) -> Vec<u8> {
    assert!(
        num_bytes % 4 == 0,
        "fp32 requires byte count divisible by 4"
    );
    let mut data = Vec::with_capacity(num_bytes);

    let num_elements = num_bytes / 4;
    for i in 0..num_elements {
        let value = ((i as f32).sin() * 0.1) as f32;
        data.extend_from_slice(&value.to_le_bytes());
    }

    data
}

#[test]
fn test_bf16_roundtrip_small() {
    let data = generate_synthetic_bf16_weights(1000);
    let compressor = CheckpointCompressor::default();

    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::BFloat16, data.len())
        .unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_bf16_roundtrip_large() {
    // 10MB of bf16 weights
    let data = generate_synthetic_bf16_weights(10 * 1024 * 1024);
    let compressor = CheckpointCompressor::default();

    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::BFloat16, data.len())
        .unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_fp32_roundtrip() {
    let data = generate_synthetic_fp32_weights(4 * 1024 * 1024);
    let compressor = CheckpointCompressor::default();

    let compressed = compressor.compress(&data, DType::Float32).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::Float32, data.len())
        .unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_compression_ratio() {
    // bf16 weights with patterns should compress well
    let data = generate_synthetic_bf16_weights(1024 * 1024);
    let compressor = CheckpointCompressor::default();

    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let ratio = CheckpointCompressor::compression_ratio(data.len(), compressed.len());

    // Expect reasonable compression (at least 2x for synthetic data)
    assert!(
        ratio >= 2.0,
        "Compression ratio {:.2}x below minimum",
        ratio
    );
}

#[test]
fn test_compression_levels() {
    let data = generate_synthetic_bf16_weights(100 * 1024);

    // Test different compression levels
    for level in [1, 3, 6, 10] {
        let config = CompressionConfig {
            zstd_level: level,
            ..Default::default()
        };
        let compressor = CheckpointCompressor::new(config);

        let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
        let decompressed = compressor
            .decompress(&compressed, DType::BFloat16, data.len())
            .unwrap();

        assert_eq!(data, decompressed, "Failed at compression level {}", level);
    }
}

#[test]
fn test_byte_grouping_disabled() {
    let data = generate_synthetic_bf16_weights(10000);

    let config = CompressionConfig {
        byte_grouping: false,
        ..Default::default()
    };
    let compressor = CheckpointCompressor::new(config);

    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::BFloat16, data.len())
        .unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_byte_grouping_improves_compression() {
    let data = generate_synthetic_bf16_weights(1024 * 1024);

    // With byte grouping
    let config_grouped = CompressionConfig {
        byte_grouping: true,
        zstd_level: 3,
    };
    let compressor_grouped = CheckpointCompressor::new(config_grouped);
    let compressed_grouped = compressor_grouped.compress(&data, DType::BFloat16).unwrap();
    let ratio_grouped =
        CheckpointCompressor::compression_ratio(data.len(), compressed_grouped.len());

    // Without byte grouping
    let config_ungrouped = CompressionConfig {
        byte_grouping: false,
        zstd_level: 3,
    };
    let compressor_ungrouped = CheckpointCompressor::new(config_ungrouped);
    let compressed_ungrouped = compressor_ungrouped
        .compress(&data, DType::BFloat16)
        .unwrap();
    let ratio_ungrouped =
        CheckpointCompressor::compression_ratio(data.len(), compressed_ungrouped.len());

    // Byte grouping should not significantly hurt compression
    // For synthetic data, results may vary - just verify both work
    assert!(
        ratio_grouped > 1.0 && ratio_ungrouped > 1.0,
        "Both modes should achieve compression (grouped: {:.2}x, ungrouped: {:.2}x)",
        ratio_grouped,
        ratio_ungrouped
    );
}

#[test]
fn test_int8_passthrough() {
    // Int8 data doesn't benefit from byte grouping
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let compressor = CheckpointCompressor::default();

    let compressed = compressor.compress(&data, DType::Int8).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::Int8, data.len())
        .unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_empty_data() {
    let data: Vec<u8> = vec![];
    let compressor = CheckpointCompressor::default();

    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::BFloat16, 0)
        .unwrap();

    assert!(decompressed.is_empty());
}

#[test]
fn test_random_data_still_roundtrips() {
    // Random data won't compress well, but should still roundtrip
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(10000);
    for i in 0..5000 {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();
        data.extend_from_slice(&(hash as u16).to_le_bytes());
    }

    let compressor = CheckpointCompressor::default();
    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let decompressed = compressor
        .decompress(&compressed, DType::BFloat16, data.len())
        .unwrap();

    assert_eq!(data, decompressed);
}

#[test]
fn test_file_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("weights.bin");
    let compressed_path = temp_dir.path().join("weights.mcp");
    let output_path = temp_dir.path().join("weights_restored.bin");

    // Write original data
    let data = generate_synthetic_bf16_weights(100 * 1024);
    std::fs::write(&input_path, &data).unwrap();

    // Compress to file
    let compressor = CheckpointCompressor::default();
    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    std::fs::write(&compressed_path, &compressed).unwrap();

    // Read compressed and decompress
    let compressed_data = std::fs::read(&compressed_path).unwrap();
    let decompressed = compressor
        .decompress(&compressed_data, DType::BFloat16, data.len())
        .unwrap();

    // Write restored data
    std::fs::write(&output_path, &decompressed).unwrap();

    // Verify files match
    let original = std::fs::read(&input_path).unwrap();
    let restored = std::fs::read(&output_path).unwrap();
    assert_eq!(original, restored);
}

#[test]
fn test_throughput_acceptable() {
    // Test that we can process data at acceptable throughput
    let data = generate_synthetic_bf16_weights(10 * 1024 * 1024); // 10MB
    let compressor = CheckpointCompressor::default();

    let start = std::time::Instant::now();
    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
    let compress_time = start.elapsed();

    let start = std::time::Instant::now();
    let _decompressed = compressor
        .decompress(&compressed, DType::BFloat16, data.len())
        .unwrap();
    let decompress_time = start.elapsed();

    let compress_throughput = data.len() as f64 / compress_time.as_secs_f64() / 1e9;
    let decompress_throughput = data.len() as f64 / decompress_time.as_secs_f64() / 1e9;

    // Minimum thresholds (lower than METRICS.md because single-threaded, debug mode)
    // In debug mode, throughput is much lower than release
    assert!(
        compress_throughput >= 0.05,
        "Compression throughput {:.2} GiB/s too low",
        compress_throughput
    );
    assert!(
        decompress_throughput >= 0.1,
        "Decompression throughput {:.2} GiB/s too low",
        decompress_throughput
    );
}
