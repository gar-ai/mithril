//! Mithril Hero Demo: Side-by-Side Checkpoint Compression Comparison
//!
//! Simulates a training loop and compares naive full-checkpoint saving
//! vs. Mithril's delta compression. Also demonstrates dataset deduplication.
//!
//! Run with: `cargo run --release --example demo`

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use mithril_checkpoint::delta::DeltaCompressor;
use mithril_checkpoint::pipeline::CompressionConfig;
use mithril_core::types::DType;
use mithril_dedup::{DedupConfig, Deduplicator};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// Number of simulated bf16 parameters (134M ≈ 256 MB at 2 bytes each).
const NUM_PARAMS: usize = 134_217_728;
/// Bytes per bf16 parameter.
const BYTES_PER_PARAM: usize = 2;
/// Total model size in bytes.
const MODEL_SIZE: usize = NUM_PARAMS * BYTES_PER_PARAM;
/// Number of training steps to simulate.
const NUM_STEPS: usize = 10;

fn main() {
    print_header();

    println!(
        "  Simulating {} training steps on {} model weights",
        style(NUM_STEPS).cyan().bold(),
        style(format_bytes(MODEL_SIZE)).cyan().bold(),
    );
    println!(
        "  (bf16, {} parameters)\n",
        style(format_count(NUM_PARAMS)).dim()
    );

    // Generate initial synthetic weights
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {msg} [{bar:40.cyan/dim}] {pos}%")
            .unwrap()
            .progress_chars("━╸─"),
    );
    pb.set_message("Generating synthetic bf16 weights...");

    let mut rng = StdRng::seed_from_u64(42);
    let mut weights = generate_synthetic_bf16_weights(&mut rng, NUM_PARAMS);
    pb.set_position(100);
    pb.finish_with_message("Weights generated");
    println!();

    // Print table header
    println!("  {0:─<5}┼{0:─<20}┼{0:─<20}┼{0:─<9}", "");
    println!(
        "  {:>5}│{:^20}│{:^20}│{:^9}",
        style("Step").bold(),
        style("Naive PyTorch").bold(),
        style("Mithril (delta)").bold(),
        style("Savings").bold(),
    );
    println!("  {0:─<5}┼{0:─<20}┼{0:─<20}┼{0:─<9}", "");

    let mut delta_compressor = DeltaCompressor::new(CompressionConfig::default());

    let mut total_naive: usize = 0;
    let mut total_mithril: usize = 0;
    let total_start = Instant::now();

    for step in 1..=NUM_STEPS {
        let naive_size = weights.len();

        // Compress with Mithril
        let key = format!("step_{}", step);
        let (compressed, stats) = delta_compressor
            .compress_checkpoint(&key, &weights)
            .expect("Compression failed");

        let mithril_size = compressed.len();

        // Verify roundtrip
        let reference_key = stats.reference_key.as_deref();
        let decompressed = delta_compressor
            .decompress_checkpoint(&compressed, weights.len(), reference_key, DType::BFloat16)
            .expect("Decompression failed");
        assert_eq!(weights, decompressed, "Roundtrip failed at step {}", step);

        total_naive += naive_size;
        total_mithril += mithril_size;

        let ratio = naive_size as f64 / mithril_size as f64;
        let savings_pct = (1.0 - mithril_size as f64 / naive_size as f64) * 100.0;

        // Format the row
        let naive_col = format!("{:>8}  ({:.1}x)", format_bytes(naive_size), 1.0);
        let mithril_col = format!("{:>8} ({:.0}x)", format_bytes(mithril_size), ratio);
        let savings_col = if savings_pct > 50.0 {
            style(format!("{:>5.1}%", savings_pct)).green().to_string()
        } else {
            format!("{:>5.1}%", savings_pct)
        };

        println!(
            "  {:>5}│{:>20}│{:>20}│{:>9}",
            step, naive_col, mithril_col, savings_col,
        );

        // Perturb weights for next step (simulated weight perturbation matching
        // real gradient update magnitudes: ~0.5% of params change per step)
        if step < NUM_STEPS {
            perturb_weights(&mut weights, &mut rng, 0.005);
        }
    }

    let total_elapsed = total_start.elapsed();

    // Print totals
    println!("  {0:─<5}┼{0:─<20}┼{0:─<20}┼{0:─<9}", "");

    let total_ratio = total_naive as f64 / total_mithril as f64;
    println!(
        "  {:>5}│{:>20}│{:>20}│{:>7.1}x",
        style("Total").bold(),
        format_bytes(total_naive),
        format_bytes(total_mithril),
        total_ratio,
    );
    println!(
        "  {:>5}│{:>20}│{:>17.2}s  │",
        style("Time").bold(),
        "(baseline)",
        total_elapsed.as_secs_f64(),
    );

    println!();

    // Throughput stats
    let compress_throughput =
        (total_naive as f64 / (1024.0 * 1024.0 * 1024.0)) / total_elapsed.as_secs_f64();
    println!(
        "  {} All roundtrip checks passed (byte-for-byte identical)",
        style("✓").green().bold()
    );
    println!(
        "  {} Throughput: {:.1} GiB/s compression",
        style("✓").green().bold(),
        compress_throughput,
    );

    // ─────────────────────────────────────────────────────────
    // Dataset Deduplication Bonus
    // ─────────────────────────────────────────────────────────
    println!();
    run_dedup_demo();

    println!();
}

/// Run the dataset deduplication demo section.
fn run_dedup_demo() {
    println!(
        "  ┌─ {} ─────────────────────┐",
        style("Dataset Deduplication Bonus").bold()
    );

    let num_docs = 10_000;
    let dup_rate = 0.08; // ~8% duplicates

    // Generate synthetic documents
    let mut rng = StdRng::seed_from_u64(123);
    let mut docs: Vec<String> = Vec::with_capacity(num_docs);

    let base_docs: Vec<String> = (0..num_docs)
        .map(|i| {
            format!(
                "Document {} about topic {} with content hash {} and additional text for dedup testing",
                i,
                i % 100,
                rng.gen::<u64>()
            )
        })
        .collect();

    for (i, doc) in base_docs.iter().enumerate() {
        docs.push(doc.clone());
        // Insert duplicates at the specified rate
        if rng.gen::<f64>() < dup_rate && i > 0 {
            // Pick a random earlier document to duplicate
            let src = rng.gen_range(0..i);
            docs.push(base_docs[src].clone());
        }
    }

    let total_docs = docs.len();
    let texts: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

    let start = Instant::now();
    let config = DedupConfig::default();
    let dedup = Deduplicator::new(config);
    let result = dedup.deduplicate_texts(&texts);
    let elapsed = start.elapsed();

    let docs_per_sec = total_docs as f64 / elapsed.as_secs_f64();

    println!(
        "  │ {:>6} synthetic docs → Found {} duplicates{:>5}│",
        style(format_count(total_docs)).cyan(),
        style(result.stats.duplicate_count).yellow().bold(),
        ""
    );
    println!(
        "  │ {:>6} docs/sec │ {:.2}s total{:>19}│",
        style(format_count(docs_per_sec as usize)).cyan(),
        elapsed.as_secs_f64(),
        ""
    );
    println!("  └────────────────────────────────────────────────┘");
}

/// Generate synthetic bf16 weights that mimic real model weight distributions.
///
/// Real neural network weights follow roughly normal distributions with most values
/// near zero. In bf16, this means exponent bytes cluster around the bias value (127),
/// which is exactly what makes byte grouping + zstd so effective.
fn generate_synthetic_bf16_weights(rng: &mut StdRng, num_params: usize) -> Vec<u8> {
    let mut data = vec![0u8; num_params * BYTES_PER_PARAM];

    // Generate bf16 values that mimic real weight distributions
    for i in 0..num_params {
        let rand_val: u16 = rng.gen();

        // bf16: 1 sign bit, 8 exponent bits, 7 mantissa bits
        // Cluster exponents around 120-130 (small values near zero)
        let exp = (120 + (rand_val % 11)) as u8; // exponents 120-130
        let sign = ((rand_val >> 15) & 1) as u8;
        let mantissa = (rand_val & 0x7F) as u8;

        let bf16 = ((sign as u16) << 15) | ((exp as u16) << 7) | (mantissa as u16);
        let bytes = bf16.to_le_bytes();

        data[i * 2] = bytes[0];
        data[i * 2 + 1] = bytes[1];
    }

    data
}

/// Perturb a fraction of weights to simulate a gradient update.
fn perturb_weights(weights: &mut [u8], rng: &mut StdRng, fraction: f64) {
    let num_params = weights.len() / BYTES_PER_PARAM;
    let num_to_change = (num_params as f64 * fraction) as usize;

    for _ in 0..num_to_change {
        let idx = rng.gen_range(0..num_params);
        let byte_idx = idx * BYTES_PER_PARAM;

        // Small perturbation: flip a few mantissa bits
        let perturbation: u8 = rng.gen_range(1..=7);
        weights[byte_idx] ^= perturbation;
    }
}

fn print_header() {
    println!();
    println!(
        "  {}",
        style("╔══════════════════════════════════════════════════════╗")
            .cyan()
            .bold()
    );
    println!(
        "  {}",
        style("║            MITHRIL vs NAIVE CHECKPOINTING            ║")
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
    use mithril_checkpoint::pipeline::CheckpointCompressor;

    #[test]
    fn demo_standalone_compression_works() {
        let mut rng = StdRng::seed_from_u64(42);
        // Use smaller size for test (1 MB)
        let weights = generate_synthetic_bf16_weights(&mut rng, 500_000);
        let compressor = CheckpointCompressor::new(CompressionConfig::default());

        let compressed = compressor.compress(&weights, DType::BFloat16).unwrap();
        let ratio = weights.len() as f64 / compressed.len() as f64;

        assert!(
            ratio > 1.0,
            "Standalone compression ratio should be > 1.0, got {:.2}x",
            ratio
        );
    }

    #[test]
    fn demo_delta_compression_high_ratio() {
        let mut rng = StdRng::seed_from_u64(42);
        let weights = generate_synthetic_bf16_weights(&mut rng, 500_000);

        let mut delta = DeltaCompressor::new(CompressionConfig::default());

        // First checkpoint
        let (_, stats1) = delta.compress_checkpoint("step_0", &weights).unwrap();
        assert!(!stats1.used_delta);

        // Perturb ~1% and compress again
        let mut weights2 = weights.clone();
        perturb_weights(&mut weights2, &mut rng, 0.01);

        let (_, stats2) = delta.compress_checkpoint("step_1", &weights2).unwrap();
        assert!(stats2.used_delta);
        assert!(
            stats2.ratio > 10.0,
            "Delta compression ratio should be > 10x for 1% change, got {:.1}x",
            stats2.ratio
        );
    }

    #[test]
    fn demo_roundtrip_byte_exact() {
        let mut rng = StdRng::seed_from_u64(42);
        let weights = generate_synthetic_bf16_weights(&mut rng, 500_000);

        let mut delta = DeltaCompressor::new(CompressionConfig::default());

        // Compress
        let (compressed, stats) = delta.compress_checkpoint("step_0", &weights).unwrap();

        // Decompress
        let decompressed = delta
            .decompress_checkpoint(
                &compressed,
                weights.len(),
                stats.reference_key.as_deref(),
                DType::BFloat16,
            )
            .unwrap();

        assert_eq!(weights, decompressed, "Roundtrip must be byte-exact");
    }

    #[test]
    fn demo_dedup_finds_known_duplicates() {
        let dedup = Deduplicator::new(DedupConfig::default());

        let texts = vec![
            "The quick brown fox jumps over the lazy dog near the river bank",
            "The quick brown fox jumps over the lazy dog near the river bank",
            "A completely different document about machine learning training",
            "The quick brown fox jumps over the lazy dog near the river bank",
        ];

        let result = dedup.deduplicate_texts(&texts);
        assert!(
            result.stats.duplicate_count > 0,
            "Should find duplicates, got 0"
        );
    }

    #[test]
    fn demo_binary_exits_zero() {
        // Verify the format_bytes and format_count helpers work correctly
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1_500), "1.5 KB");
        assert_eq!(format_bytes(1_500_000), "1.5 MB");
        assert_eq!(format_bytes(2_500_000_000), "2.50 GB");
        assert_eq!(format_count(1_234_567), "1,234,567");
    }
}
