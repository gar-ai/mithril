//! Mithril Real Model Checkpoint Compression Demo
//!
//! Compresses actual HuggingFace model weights (safetensors format) using
//! Mithril's byte-grouping + zstd pipeline and delta encoding. Demonstrates
//! real compression ratios on production model architectures.
//!
//! Run with: `cargo run --release --example real_checkpoint_demo`

use console::style;
use mithril_checkpoint::delta::DeltaCompressor;
use mithril_checkpoint::formats::SafetensorsReader;
use mithril_checkpoint::pipeline::{CheckpointCompressor, CompressionConfig};
use mithril_core::types::DType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::time::Instant;

/// A real model we can compress.
struct ModelInfo {
    name: &'static str,
    dir: &'static str,
    file: &'static str,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "GTE-Small (33M)",
        dir: "gte-small",
        file: "model.safetensors",
    },
    ModelInfo {
        name: "DINOv2-Small (22M)",
        dir: "dinov2-small",
        file: "model.safetensors",
    },
    ModelInfo {
        name: "MiniLM-L6 (22M)",
        dir: "minilm-l6-v2",
        file: "model.safetensors",
    },
    ModelInfo {
        name: "BGE-Small-EN (33M)",
        dir: "bge-small-en",
        file: "model.safetensors",
    },
];

fn find_fixtures_dir() -> PathBuf {
    let candidates = [
        PathBuf::from("fixtures/hf_checkpoints"),
        PathBuf::from("../fixtures/hf_checkpoints"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    eprintln!(
        "  {} Could not find fixtures/hf_checkpoints/ directory.",
        style("✗").red().bold()
    );
    eprintln!("  Run from the mithril workspace root.");
    std::process::exit(1);
}

fn main() {
    println!();
    println!(
        "  {}",
        style("╔══════════════════════════════════════════════════════╗")
            .cyan()
            .bold()
    );
    println!(
        "  {}",
        style("║     MITHRIL COMPRESSION ON REAL MODEL WEIGHTS       ║")
            .cyan()
            .bold()
    );
    println!(
        "  {}",
        style("╚══════════════════════════════════════════════════════╝")
            .cyan()
            .bold()
    );
    println!();

    let fixtures_dir = find_fixtures_dir();
    let compressor = CheckpointCompressor::new(CompressionConfig::default());

    // ─────────────────────────────────────────────────────────
    // Phase 1: Standalone compression of real safetensors
    // ─────────────────────────────────────────────────────────
    println!(
        "  {} Standalone Compression (byte-group + zstd)",
        style("▶").cyan().bold()
    );
    println!();

    println!(
        "  {0:─<22}┬{0:─<11}┬{0:─<12}┬{0:─<9}┬{0:─<8}┬{0:─<10}",
        ""
    );
    println!(
        "  {:>22}│{:>11}│{:>12}│{:>9}│{:>8}│{:>10}",
        style("Model").bold(),
        style("Raw Size").bold(),
        style("Compressed").bold(),
        style("Ratio").bold(),
        style("DType").bold(),
        style("Speed").bold(),
    );
    println!(
        "  {0:─<22}┼{0:─<11}┼{0:─<12}┼{0:─<9}┼{0:─<8}┼{0:─<10}",
        ""
    );

    let mut model_data: Vec<(String, Vec<u8>, DType)> = Vec::new();

    for m in MODELS {
        let path = fixtures_dir.join(m.dir).join(m.file);
        if !path.exists() {
            println!(
                "  {:>22}│ {} (file not found)",
                m.name,
                style("skipped").dim()
            );
            continue;
        }

        // Read safetensors
        let mut reader = match SafetensorsReader::open(&path) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "  {:>22}│ {} ({})",
                    m.name,
                    style("error").red(),
                    e
                );
                continue;
            }
        };

        // Determine dominant dtype from first tensor
        let header = reader.header();
        let dominant_dtype = header
            .tensors
            .values()
            .next()
            .map(|t| t.dtype)
            .unwrap_or(DType::Float32);

        // Read all tensor data
        let raw_data = match reader.read_all_data() {
            Ok(d) => d,
            Err(e) => {
                println!(
                    "  {:>22}│ {} ({})",
                    m.name,
                    style("error").red(),
                    e
                );
                continue;
            }
        };

        let raw_size = raw_data.len();

        // Compress
        let start = Instant::now();
        let compressed = compressor.compress(&raw_data, dominant_dtype).unwrap();
        let elapsed = start.elapsed();

        let compressed_size = compressed.len();
        let ratio = raw_size as f64 / compressed_size as f64;
        let speed_gibs =
            (raw_size as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed.as_secs_f64();

        let dtype_str = format!("{:?}", dominant_dtype);

        println!(
            "  {:>22}│{:>9} │{:>10} │{:>7.1}x │{:>7} │{:>7.1} GiB/s",
            m.name,
            format_bytes(raw_size),
            format_bytes(compressed_size),
            ratio,
            dtype_str,
            speed_gibs,
        );

        // Verify roundtrip
        let decompressed = compressor
            .decompress(&compressed, dominant_dtype, raw_size)
            .unwrap();
        assert_eq!(
            raw_data, decompressed,
            "Roundtrip failed for {}",
            m.name
        );

        model_data.push((m.name.to_string(), raw_data, dominant_dtype));
    }

    println!(
        "  {0:─<22}┴{0:─<11}┴{0:─<12}┴{0:─<9}┴{0:─<8}┴{0:─<10}",
        ""
    );
    println!();

    // ─────────────────────────────────────────────────────────
    // Phase 2: Delta compression (simulated fine-tuning)
    // ─────────────────────────────────────────────────────────
    if let Some((name, weights, dtype)) = model_data.first() {
        println!(
            "  {} Delta Compression — {} (simulated fine-tuning)",
            style("▶").cyan().bold(),
            name,
        );
        println!();

        let raw_size = weights.len();
        let bytes_per_param = match dtype {
            DType::Float32 => 4,
            DType::Float16 | DType::BFloat16 => 2,
            _ => 1,
        };

        println!(
            "  {} — {} raw, {} parameters",
            name,
            format_bytes(raw_size),
            format_count(raw_size / bytes_per_param),
        );
        println!();

        println!(
            "  {0:─<7}┬{0:─<12}┬{0:─<12}┬{0:─<9}┬{0:─<10}",
            ""
        );
        println!(
            "  {:>7}│{:>12}│{:>12}│{:>9}│{:>10}",
            style("Step").bold(),
            style("Raw Size").bold(),
            style("Compressed").bold(),
            style("Ratio").bold(),
            style("Changed").bold(),
        );
        println!(
            "  {0:─<7}┼{0:─<12}┼{0:─<12}┼{0:─<9}┼{0:─<10}",
            ""
        );

        let mut delta = DeltaCompressor::new(CompressionConfig::default());
        let mut current_weights = weights.clone();
        let mut rng = StdRng::seed_from_u64(42);

        let change_rates = [0.0, 0.005, 0.005, 0.005, 0.005];
        let mut total_raw = 0usize;
        let mut total_compressed = 0usize;

        for (step, &change_rate) in change_rates.iter().enumerate() {
            // Perturb weights (skip step 0 — baseline)
            if change_rate > 0.0 {
                perturb_weights(&mut current_weights, &mut rng, change_rate, bytes_per_param);
            }

            let key = format!("step_{}", step);
            let (compressed, stats) = delta
                .compress_checkpoint(&key, &current_weights)
                .expect("Compression failed");

            let compressed_size = compressed.len();
            total_raw += raw_size;
            total_compressed += compressed_size;

            let change_str = if step == 0 {
                "baseline".to_string()
            } else {
                format!("{:.1}%", change_rate * 100.0)
            };

            let ratio_str = if stats.ratio > 50.0 {
                style(format!("{:>6.0}x", stats.ratio))
                    .green()
                    .bold()
                    .to_string()
            } else if stats.ratio > 5.0 {
                style(format!("{:>6.1}x", stats.ratio))
                    .green()
                    .to_string()
            } else {
                format!("{:>6.1}x", stats.ratio)
            };

            println!(
                "  {:>7}│{:>10} │{:>10} │{}│{:>10}",
                step,
                format_bytes(raw_size),
                format_bytes(compressed_size),
                ratio_str,
                change_str,
            );

            // Verify roundtrip
            let reference_key = stats.reference_key.as_deref();
            let decompressed = delta
                .decompress_checkpoint(&compressed, raw_size, reference_key, *dtype)
                .expect("Decompression failed");
            assert_eq!(
                current_weights, decompressed,
                "Roundtrip failed at step {}",
                step
            );
        }

        println!(
            "  {0:─<7}┼{0:─<12}┼{0:─<12}┼{0:─<9}┼{0:─<10}",
            ""
        );

        let overall_ratio = total_raw as f64 / total_compressed as f64;
        println!(
            "  {:>7}│{:>10} │{:>10} │{:>6.1}x │",
            style("Total").bold(),
            format_bytes(total_raw),
            format_bytes(total_compressed),
            overall_ratio,
        );

        println!();
        println!(
            "  {} All roundtrip checks passed (byte-for-byte identical)",
            style("✓").green().bold()
        );
        println!(
            "  {} These are REAL model weights, not synthetic data",
            style("✓").green().bold()
        );
    }

    println!();
}

/// Perturb weights to simulate a gradient update during fine-tuning.
fn perturb_weights(weights: &mut [u8], rng: &mut StdRng, fraction: f64, bytes_per_param: usize) {
    let num_params = weights.len() / bytes_per_param;
    let num_to_change = (num_params as f64 * fraction) as usize;

    for _ in 0..num_to_change {
        let idx = rng.gen_range(0..num_params);
        let byte_idx = idx * bytes_per_param;

        // Small perturbation: flip low-order bits (mimics small gradient updates)
        let perturbation: u8 = rng.gen_range(1..=7);
        weights[byte_idx] ^= perturbation;
    }
}

/// Format bytes into human-readable form.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format large numbers with commas.
fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_safetensors_readable() {
        let fixtures = find_fixtures_dir();
        let path = fixtures.join("gte-small/model.safetensors");
        if !path.exists() {
            return;
        }
        let mut reader = SafetensorsReader::open(&path).unwrap();
        let header = reader.header();
        assert!(!header.tensors.is_empty(), "Should have tensors");

        let data = reader.read_all_data().unwrap();
        assert!(data.len() > 1_000_000, "Should have >1MB of tensor data");
    }

    #[test]
    fn test_real_weights_compress_above_1x() {
        let fixtures = find_fixtures_dir();
        let path = fixtures.join("gte-small/model.safetensors");
        if !path.exists() {
            return;
        }
        let mut reader = SafetensorsReader::open(&path).unwrap();
        let dtype = reader
            .header()
            .tensors
            .values()
            .next()
            .unwrap()
            .dtype;
        let data = reader.read_all_data().unwrap();

        let compressor = CheckpointCompressor::new(CompressionConfig::default());
        let compressed = compressor.compress(&data, dtype).unwrap();
        let ratio = data.len() as f64 / compressed.len() as f64;

        assert!(
            ratio > 1.0,
            "Real model weights should compress >1x, got {:.2}x",
            ratio
        );
    }

    #[test]
    fn test_real_weights_roundtrip() {
        let fixtures = find_fixtures_dir();
        let path = fixtures.join("gte-small/model.safetensors");
        if !path.exists() {
            return;
        }
        let mut reader = SafetensorsReader::open(&path).unwrap();
        let dtype = reader
            .header()
            .tensors
            .values()
            .next()
            .unwrap()
            .dtype;
        let data = reader.read_all_data().unwrap();

        let compressor = CheckpointCompressor::new(CompressionConfig::default());
        let compressed = compressor.compress(&data, dtype).unwrap();
        let decompressed = compressor.decompress(&compressed, dtype, data.len()).unwrap();

        assert_eq!(data, decompressed, "Real weights roundtrip must be exact");
    }

    #[test]
    fn test_real_weights_delta_high_ratio() {
        let fixtures = find_fixtures_dir();
        let path = fixtures.join("gte-small/model.safetensors");
        if !path.exists() {
            return;
        }
        let mut reader = SafetensorsReader::open(&path).unwrap();
        let data = reader.read_all_data().unwrap();

        let mut delta = DeltaCompressor::new(CompressionConfig::default());

        // Step 0: baseline
        let (_, stats0) = delta.compress_checkpoint("step_0", &data).unwrap();
        assert!(!stats0.used_delta);

        // Step 1: perturb 0.5% and compress
        let mut data2 = data.clone();
        let mut rng = StdRng::seed_from_u64(99);
        perturb_weights(&mut data2, &mut rng, 0.005, 4); // float32

        let (_, stats1) = delta.compress_checkpoint("step_1", &data2).unwrap();
        assert!(stats1.used_delta);
        assert!(
            stats1.ratio > 10.0,
            "Delta on real weights with 0.5% change should be >10x, got {:.1}x",
            stats1.ratio
        );
    }
}
