//! Benchmarks for checkpoint compression and quantization.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mithril_checkpoint::bytegroup::{byte_group_bf16, byte_ungroup_bf16};
use mithril_checkpoint::pipeline::{CheckpointCompressor, CompressionConfig};
use mithril_checkpoint::quantize::{QuantizeConfig, QuantizeMethod, Quantizer};
use mithril_core::types::DType;
use std::fs;

/// Generate realistic bf16 tensor data (simulating neural network weights).
fn generate_bf16_data(size: usize) -> Vec<u8> {
    let mut rng_state = 42u64;
    (0..size / 2)
        .flat_map(|_| {
            // Simple LCG for reproducible pseudo-random data
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand = (rng_state >> 33) as u16;

            // bf16 format: 1 sign, 8 exp, 7 mantissa
            // Most weights are small, so exponents cluster around bias (127)
            let exp = 127 + ((rand % 8) as i16 - 4) as u8; // 123-131 mostly
            let mantissa = (rand & 0x7F) as u8;
            let sign = ((rand >> 15) & 1) as u8;

            let bf16 = ((sign as u16) << 15) | ((exp as u16) << 7) | (mantissa as u16);
            bf16.to_le_bytes()
        })
        .collect()
}

/// Load test fixture or generate data if not available.
fn load_test_data() -> Vec<u8> {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/checkpoints/small_model.bin"
    );

    fs::read(fixture_path).unwrap_or_else(|_| {
        // Generate 10MB of bf16-like data if fixture not available
        generate_bf16_data(10 * 1024 * 1024)
    })
}

fn bench_byte_grouping(c: &mut Criterion) {
    let data = load_test_data();
    let mut group = c.benchmark_group("byte_grouping");

    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("group_bf16", |b| b.iter(|| byte_group_bf16(&data)));

    let grouped = byte_group_bf16(&data);
    group.bench_function("ungroup_bf16", |b| b.iter(|| byte_ungroup_bf16(&grouped)));

    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let data = load_test_data();
    let mut group = c.benchmark_group("compression");

    group.throughput(Throughput::Bytes(data.len() as u64));

    // Test different compression levels
    for level in [1, 3, 6] {
        let compressor = CheckpointCompressor::new(CompressionConfig::with_level(level));

        group.bench_with_input(
            BenchmarkId::new("compress_bf16", format!("level_{}", level)),
            &data,
            |b, data| b.iter(|| compressor.compress(data, DType::BFloat16).unwrap()),
        );
    }

    // Benchmark decompression
    let compressor = CheckpointCompressor::default();
    let compressed = compressor.compress(&data, DType::BFloat16).unwrap();

    group.bench_function("decompress_bf16", |b| {
        b.iter(|| {
            compressor
                .decompress(&compressed, DType::BFloat16, data.len())
                .unwrap()
        })
    });

    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let data = load_test_data();

    let mut group = c.benchmark_group("compression_ratio");
    group.sample_size(10);

    // Measure compression ratio at different levels
    for level in [1, 3, 6, 10, 19] {
        let compressor = CheckpointCompressor::new(CompressionConfig::with_level(level));

        group.bench_with_input(
            BenchmarkId::new("ratio", format!("level_{}", level)),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = compressor.compress(data, DType::BFloat16).unwrap();
                    let ratio = data.len() as f64 / compressed.len() as f64;
                    ratio
                })
            },
        );
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let data = load_test_data();
    let compressor = CheckpointCompressor::default();

    let mut group = c.benchmark_group("roundtrip");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("compress_decompress", |b| {
        b.iter(|| {
            let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
            let decompressed = compressor
                .decompress(&compressed, DType::BFloat16, data.len())
                .unwrap();
            decompressed
        })
    });

    group.finish();
}

// =============================================================================
// Quantization Benchmarks
// =============================================================================

fn bench_quantization(c: &mut Criterion) {
    let data = load_test_data();

    let mut group = c.benchmark_group("quantization");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Int8 quantization
    let int8_config = QuantizeConfig::int8();
    let int8_quantizer = Quantizer::new(int8_config);

    group.bench_function("int8_quantize", |b| {
        b.iter(|| int8_quantizer.quantize_bf16(&data).unwrap())
    });

    let int8_quantized = int8_quantizer.quantize_bf16(&data).unwrap();
    group.bench_function("int8_dequantize", |b| {
        b.iter(|| int8_quantizer.dequantize_to_bf16(&int8_quantized).unwrap())
    });

    // NF4 quantization with different group sizes
    for group_size in [32, 64, 128] {
        // Ensure data size is divisible by group size
        let nf4_config = QuantizeConfig::nf4_with_group_size(group_size);
        let nf4_quantizer = Quantizer::new(nf4_config);

        group.bench_with_input(
            BenchmarkId::new("nf4_quantize", format!("g{}", group_size)),
            &data,
            |b, data| b.iter(|| nf4_quantizer.quantize_bf16(data).unwrap()),
        );

        let nf4_quantized = nf4_quantizer.quantize_bf16(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("nf4_dequantize", format!("g{}", group_size)),
            &nf4_quantized,
            |b, quantized| b.iter(|| nf4_quantizer.dequantize_to_bf16(quantized).unwrap()),
        );
    }

    // Int4 quantization
    let int4_config = QuantizeConfig::new(QuantizeMethod::Int4);
    let int4_quantizer = Quantizer::new(int4_config);

    group.bench_function("int4_quantize", |b| {
        b.iter(|| int4_quantizer.quantize_bf16(&data).unwrap())
    });

    let int4_quantized = int4_quantizer.quantize_bf16(&data).unwrap();
    group.bench_function("int4_dequantize", |b| {
        b.iter(|| int4_quantizer.dequantize_to_bf16(&int4_quantized).unwrap())
    });

    group.finish();

    // Print compression ratios (informational)
    let int8_quantized = int8_quantizer.quantize_bf16(&data).unwrap();
    let nf4_quantized = Quantizer::new(QuantizeConfig::nf4_with_group_size(64))
        .quantize_bf16(&data)
        .unwrap();
    let int4_quantized = int4_quantizer.quantize_bf16(&data).unwrap();

    eprintln!(
        "\nQuantization Compression Ratios ({}KB bf16 input):",
        data.len() / 1024
    );
    eprintln!("  Int8:  {:.2}x", int8_quantized.compression_ratio());
    eprintln!(
        "  NF4:   {:.2}x (group_size=64)",
        nf4_quantized.compression_ratio()
    );
    eprintln!("  Int4:  {:.2}x", int4_quantized.compression_ratio());
}

fn bench_quantization_ratio(c: &mut Criterion) {
    let data = load_test_data();

    let mut group = c.benchmark_group("quantization_ratio");
    group.sample_size(10);

    // Int8 ratio
    let int8_config = QuantizeConfig::int8();
    let int8_quantizer = Quantizer::new(int8_config);

    group.bench_function("int8_ratio", |b| {
        b.iter(|| {
            let quantized = int8_quantizer.quantize_bf16(&data).unwrap();
            quantized.compression_ratio()
        })
    });

    // NF4 ratios with different group sizes
    for group_size in [32, 64, 128] {
        let nf4_config = QuantizeConfig::nf4_with_group_size(group_size);
        let nf4_quantizer = Quantizer::new(nf4_config);

        group.bench_with_input(
            BenchmarkId::new("nf4_ratio", format!("g{}", group_size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let quantized = nf4_quantizer.quantize_bf16(data).unwrap();
                    quantized.compression_ratio()
                })
            },
        );
    }

    // Int4 ratio
    let int4_config = QuantizeConfig::new(QuantizeMethod::Int4);
    let int4_quantizer = Quantizer::new(int4_config);

    group.bench_function("int4_ratio", |b| {
        b.iter(|| {
            let quantized = int4_quantizer.quantize_bf16(&data).unwrap();
            quantized.compression_ratio()
        })
    });

    group.finish();
}

fn bench_quantized_roundtrip(c: &mut Criterion) {
    let data = load_test_data();
    let compressor = CheckpointCompressor::default();
    let shape = vec![data.len() / 2]; // bf16 elements

    let mut group = c.benchmark_group("quantized_roundtrip");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Int8 full pipeline: quantize → compress → decompress → dequantize
    let int8_config = QuantizeConfig::int8();
    group.bench_function("int8_full_pipeline", |b| {
        b.iter(|| {
            let compressed = compressor
                .compress_quantized(&data, DType::BFloat16, shape.clone(), &int8_config)
                .unwrap();
            let decompressed = compressor
                .decompress_dequantized(&compressed, DType::BFloat16)
                .unwrap();
            decompressed
        })
    });

    // NF4 full pipeline
    let nf4_config = QuantizeConfig::nf4_with_group_size(64);
    group.bench_function("nf4_g64_full_pipeline", |b| {
        b.iter(|| {
            let compressed = compressor
                .compress_quantized(&data, DType::BFloat16, shape.clone(), &nf4_config)
                .unwrap();
            let decompressed = compressor
                .decompress_dequantized(&compressed, DType::BFloat16)
                .unwrap();
            decompressed
        })
    });

    // Int4 full pipeline
    let int4_config = QuantizeConfig::new(QuantizeMethod::Int4);
    group.bench_function("int4_full_pipeline", |b| {
        b.iter(|| {
            let compressed = compressor
                .compress_quantized(&data, DType::BFloat16, shape.clone(), &int4_config)
                .unwrap();
            let decompressed = compressor
                .decompress_dequantized(&compressed, DType::BFloat16)
                .unwrap();
            decompressed
        })
    });

    // Compare with lossless compression
    group.bench_function("lossless_pipeline", |b| {
        b.iter(|| {
            let compressed = compressor.compress(&data, DType::BFloat16).unwrap();
            let decompressed = compressor
                .decompress(&compressed, DType::BFloat16, data.len())
                .unwrap();
            decompressed
        })
    });

    group.finish();

    // Print final compression ratios with zstd
    let int8_compressed = compressor
        .compress_quantized(&data, DType::BFloat16, shape.clone(), &int8_config)
        .unwrap();
    let nf4_compressed = compressor
        .compress_quantized(&data, DType::BFloat16, shape.clone(), &nf4_config)
        .unwrap();
    let int4_compressed = compressor
        .compress_quantized(&data, DType::BFloat16, shape.clone(), &int4_config)
        .unwrap();
    let lossless_compressed = compressor.compress(&data, DType::BFloat16).unwrap();

    eprintln!(
        "\nFull Pipeline Compression Ratios ({}KB bf16 input):",
        data.len() / 1024
    );
    eprintln!(
        "  Lossless:     {:.2}x ({} bytes)",
        data.len() as f64 / lossless_compressed.len() as f64,
        lossless_compressed.len()
    );
    eprintln!(
        "  Int8+zstd:    {:.2}x ({} bytes)",
        data.len() as f64 / int8_compressed.len() as f64,
        int8_compressed.len()
    );
    eprintln!(
        "  NF4+zstd:     {:.2}x ({} bytes)",
        data.len() as f64 / nf4_compressed.len() as f64,
        nf4_compressed.len()
    );
    eprintln!(
        "  Int4+zstd:    {:.2}x ({} bytes)",
        data.len() as f64 / int4_compressed.len() as f64,
        int4_compressed.len()
    );
}

criterion_group!(
    benches,
    bench_byte_grouping,
    bench_compression,
    bench_compression_ratio,
    bench_roundtrip,
    bench_quantization,
    bench_quantization_ratio,
    bench_quantized_roundtrip,
);
criterion_main!(benches);
