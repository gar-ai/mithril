//! Real-world validation tests using HuggingFace checkpoints.
//!
//! These tests require fixtures to be downloaded first:
//! ```bash
//! python scripts/download_hf_fixtures.py --checkpoints
//! ```
//!
//! Tests are marked `#[ignore]` to avoid running in CI without fixtures.

use mithril_checkpoint::{CheckpointCompressor, CompressionConfig, SafetensorsReader};
use std::path::PathBuf;

/// Get fixtures directory path.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures")
        .join("hf_checkpoints")
}

/// Test compression/decompression roundtrip on OPT-125M.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --checkpoints"]
fn test_opt125m_roundtrip() {
    let model_path = fixtures_dir().join("opt-125m");
    if !model_path.exists() {
        eprintln!(
            "Skipping test: OPT-125M not found at {}",
            model_path.display()
        );
        eprintln!("Run: python scripts/download_hf_fixtures.py --checkpoints");
        return;
    }

    // Find safetensors files
    let safetensors_files: Vec<_> = std::fs::read_dir(&model_path)
        .expect("Failed to read model directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !safetensors_files.is_empty(),
        "No safetensors files found in {:?}",
        model_path
    );

    let compressor = CheckpointCompressor::default();

    for entry in safetensors_files {
        let path = entry.path();
        println!("\nTesting: {}", path.display());

        // Read safetensors file
        let mut reader = SafetensorsReader::open(&path).expect("Failed to open safetensors");
        let tensor_names: Vec<String> = reader.tensor_names().map(String::from).collect();
        println!("  Tensors: {}", tensor_names.len());

        let mut total_original = 0usize;
        let mut total_compressed = 0usize;

        // Test each tensor
        for name in &tensor_names {
            let meta = reader.tensor_meta(name).expect("Tensor not found").clone();
            let data = reader.read_tensor(name).expect("Failed to read tensor");
            let original_size = data.len();

            // Compress
            let compressed = compressor
                .compress(&data, meta.dtype)
                .expect("Compression failed");

            // Decompress
            let decompressed = compressor
                .decompress(&compressed, meta.dtype, original_size)
                .expect("Decompression failed");

            // Verify roundtrip
            assert_eq!(data, decompressed, "Roundtrip failed for tensor {}", name);

            total_original += original_size;
            total_compressed += compressed.len();
        }

        let ratio = total_original as f64 / total_compressed as f64;
        println!(
            "  Total: {} -> {} bytes ({:.2}x compression)",
            total_original, total_compressed, ratio
        );

        // OPT-125M is bf16, so we expect good compression
        assert!(ratio > 5.0, "Expected >5x compression on bf16 weights");
    }
}

/// Test compression/decompression roundtrip on GPT2.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --checkpoints"]
fn test_gpt2_roundtrip() {
    let model_path = fixtures_dir().join("gpt2");
    if !model_path.exists() {
        eprintln!("Skipping test: GPT2 not found at {}", model_path.display());
        eprintln!("Run: python scripts/download_hf_fixtures.py --checkpoints");
        return;
    }

    // Find safetensors files
    let safetensors_files: Vec<_> = std::fs::read_dir(&model_path)
        .expect("Failed to read model directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !safetensors_files.is_empty(),
        "No safetensors files found in {:?}",
        model_path
    );

    let compressor = CheckpointCompressor::default();

    for entry in safetensors_files {
        let path = entry.path();
        println!("\nTesting: {}", path.display());

        let mut reader = SafetensorsReader::open(&path).expect("Failed to open safetensors");
        let tensor_names: Vec<String> = reader.tensor_names().map(String::from).collect();
        println!("  Tensors: {}", tensor_names.len());

        let mut total_original = 0usize;
        let mut total_compressed = 0usize;

        for name in &tensor_names {
            let meta = reader.tensor_meta(name).expect("Tensor not found").clone();
            let data = reader.read_tensor(name).expect("Failed to read tensor");
            let original_size = data.len();

            let compressed = compressor
                .compress(&data, meta.dtype)
                .expect("Compression failed");

            let decompressed = compressor
                .decompress(&compressed, meta.dtype, original_size)
                .expect("Decompression failed");

            assert_eq!(data, decompressed, "Roundtrip failed for tensor {}", name);

            total_original += original_size;
            total_compressed += compressed.len();
        }

        let ratio = total_original as f64 / total_compressed as f64;
        println!(
            "  Total: {} -> {} bytes ({:.2}x compression)",
            total_original, total_compressed, ratio
        );
    }
}

/// Benchmark compression throughput on real weights.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --checkpoints"]
fn test_compression_throughput() {
    let model_path = fixtures_dir().join("opt-125m");
    if !model_path.exists() {
        return;
    }

    // Find largest safetensors file
    let largest = std::fs::read_dir(&model_path)
        .expect("Failed to read model directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .max_by_key(|e| e.metadata().map(|m| m.len()).unwrap_or(0));

    if largest.is_none() {
        return;
    }

    let path = largest.unwrap().path();
    println!("Benchmarking throughput on: {}", path.display());

    // Read all tensor data
    let mut reader = SafetensorsReader::open(&path).expect("Failed to open");
    let all_data = reader.read_all_data().expect("Failed to read data");
    let size_mb = all_data.len() as f64 / (1024.0 * 1024.0);

    println!("  Data size: {:.2} MB", size_mb);

    // Benchmark compression
    let config = CompressionConfig {
        zstd_level: 3,
        byte_grouping: true,
    };
    let compressor = CheckpointCompressor::new(config);

    let start = std::time::Instant::now();
    let compressed = compressor
        .compress(&all_data, mithril_core::types::DType::BFloat16)
        .expect("Compression failed");
    let elapsed = start.elapsed();

    let throughput = size_mb / elapsed.as_secs_f64();
    let ratio = all_data.len() as f64 / compressed.len() as f64;

    println!("  Compression:");
    println!("    Time: {:.2}ms", elapsed.as_millis());
    println!("    Throughput: {:.2} MB/s", throughput);
    println!("    Ratio: {:.2}x", ratio);

    // METRICS.md target: ≥1 GiB/s compression (with parallel)
    // Single-threaded should hit ~500 MB/s minimum
    assert!(
        throughput > 100.0,
        "Compression throughput too low: {:.2} MB/s",
        throughput
    );

    // Benchmark decompression
    let start = std::time::Instant::now();
    let _decompressed = compressor
        .decompress(
            &compressed,
            mithril_core::types::DType::BFloat16,
            all_data.len(),
        )
        .expect("Decompression failed");
    let elapsed = start.elapsed();

    let throughput = size_mb / elapsed.as_secs_f64();
    println!("  Decompression:");
    println!("    Time: {:.2}ms", elapsed.as_millis());
    println!("    Throughput: {:.2} MB/s", throughput);

    // Decompression should be faster than compression
    assert!(
        throughput > 200.0,
        "Decompression throughput too low: {:.2} MB/s",
        throughput
    );
}
