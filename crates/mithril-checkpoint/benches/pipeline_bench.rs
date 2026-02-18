//! End-to-end pipeline benchmarks for checkpoint compression.
//!
//! Tests full compress/decompress at realistic model sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mithril_checkpoint::delta::DeltaEncoder;
use mithril_checkpoint::pipeline::{CheckpointCompressor, CompressionConfig};
use mithril_core::types::DType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Generate synthetic bf16 weights with realistic distribution.
fn generate_bf16_weights(rng: &mut StdRng, num_params: usize) -> Vec<u8> {
    let mut weights = Vec::with_capacity(num_params * 2);
    for _ in 0..num_params {
        let exp = (124 + rng.gen_range(0..8)) as u8;
        let sign = rng.gen_range(0u8..2);
        let mantissa = rng.gen_range(0u8..128);
        let val = ((sign as u16) << 15) | ((exp as u16) << 7) | (mantissa as u16);
        weights.extend_from_slice(&val.to_le_bytes());
    }
    weights
}

/// Generate synthetic fp32 weights.
fn generate_fp32_weights(rng: &mut StdRng, num_params: usize) -> Vec<u8> {
    let mut weights = Vec::with_capacity(num_params * 4);
    for _ in 0..num_params {
        let val: f32 = rng.gen_range(-0.1..0.1);
        weights.extend_from_slice(&val.to_le_bytes());
    }
    weights
}

/// Perturb 0.5% of weights (simulate gradient update).
fn perturb_weights(weights: &mut [u8], rng: &mut StdRng, bytes_per_param: usize) {
    let num_params = weights.len() / bytes_per_param;
    let num_to_change = num_params / 200; // 0.5%
    for _ in 0..num_to_change {
        let idx = rng.gen_range(0..num_params) * bytes_per_param;
        weights[idx] ^= rng.gen_range(1u8..=7);
    }
}

fn bench_standalone_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("standalone_compress");
    let compressor = CheckpointCompressor::new(CompressionConfig::default());
    let mut rng = StdRng::seed_from_u64(42);

    // ~33M params Float16 (66MB, like GTE-Small)
    let bf16_data = generate_bf16_weights(&mut rng, 33_000_000);
    group.throughput(Throughput::Bytes(bf16_data.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("bf16", "33M_params"),
        &bf16_data,
        |b, data| {
            b.iter(|| compressor.compress(data, DType::BFloat16).unwrap());
        },
    );

    // ~22M params Float32 (88MB, like DINOv2)
    let fp32_data = generate_fp32_weights(&mut rng, 22_000_000);
    group.throughput(Throughput::Bytes(fp32_data.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fp32", "22M_params"),
        &fp32_data,
        |b, data| {
            b.iter(|| compressor.compress(data, DType::Float32).unwrap());
        },
    );

    group.finish();
}

fn bench_delta_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_compress");
    let compressor = CheckpointCompressor::new(CompressionConfig::default());
    let mut rng = StdRng::seed_from_u64(42);

    // 33M params bf16
    let baseline = generate_bf16_weights(&mut rng, 33_000_000);
    let mut step2 = baseline.clone();
    perturb_weights(&mut step2, &mut rng, 2);

    group.throughput(Throughput::Bytes(baseline.len() as u64));
    group.bench_function("bf16_33M_delta", |b| {
        b.iter(|| {
            compressor
                .compress_with_delta(&step2, DType::BFloat16, Some(&baseline))
                .unwrap()
        });
    });

    group.finish();
}

fn bench_xor_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_encode");
    let mut rng = StdRng::seed_from_u64(42);

    for &size_mb in &[1, 10, 66, 256] {
        let size = size_mb * 1_000_000;
        let prev: Vec<u8> = (0..size).map(|_| rng.gen()).collect();
        let mut curr = prev.clone();
        perturb_weights(&mut curr, &mut rng, 1);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("xor", format!("{}MB", size_mb)),
            &(&prev, &curr),
            |b, &(prev, curr)| {
                b.iter(|| DeltaEncoder::encode(curr, Some(prev)));
            },
        );
    }

    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress");
    let compressor = CheckpointCompressor::new(CompressionConfig::default());
    let mut rng = StdRng::seed_from_u64(42);

    let bf16_data = generate_bf16_weights(&mut rng, 33_000_000);
    let compressed = compressor.compress(&bf16_data, DType::BFloat16).unwrap();

    group.throughput(Throughput::Bytes(bf16_data.len() as u64));
    group.bench_function("bf16_33M", |b| {
        b.iter(|| {
            compressor
                .decompress(&compressed, DType::BFloat16, bf16_data.len())
                .unwrap()
        });
    });

    group.finish();
}

fn bench_xor_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_chunk_size");
    let mut rng = StdRng::seed_from_u64(42);

    // Use 66MB (realistic GTE-Small size)
    let size = 66_000_000;
    let prev: Vec<u8> = (0..size).map(|_| rng.gen()).collect();
    let mut curr = prev.clone();
    // Perturb 0.5% of bytes
    for i in (0..size).step_by(200) {
        curr[i] ^= rng.gen_range(1u8..=7);
    }

    for &chunk_kb in &[16, 32, 64, 128, 256, 512, 1024] {
        let chunk_size = chunk_kb * 1024;
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("xor_par", format!("{}KB", chunk_kb)),
            &chunk_size,
            |b, &cs| {
                b.iter(|| {
                    let mut delta = vec![0u8; size];
                    delta
                        .par_chunks_mut(cs)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let offset = chunk_idx * cs;
                            for (i, byte) in chunk.iter_mut().enumerate() {
                                *byte = curr[offset + i] ^ prev[offset + i];
                            }
                        });
                    delta
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_standalone_compress,
    bench_delta_compress,
    bench_xor_encode,
    bench_decompress,
    bench_xor_chunk_sizes,
);
criterion_main!(benches);
