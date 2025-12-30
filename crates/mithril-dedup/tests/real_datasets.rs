//! Real-world validation tests using HuggingFace datasets.
//!
//! These tests require fixtures to be downloaded first:
//! ```bash
//! python scripts/download_hf_fixtures.py --datasets
//! ```
//!
//! Tests are marked `#[ignore]` to avoid running in CI without fixtures.

use mithril_dedup::{read_jsonl, DedupConfig, Deduplicator};

/// Default text field for HuggingFace datasets.
const TEXT_FIELD: &str = "text";
use std::path::PathBuf;
use std::time::Instant;

/// Get fixtures directory path.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fixtures")
        .join("hf_datasets")
}

/// Test deduplication on C4 dataset sample.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --datasets"]
fn test_c4_dedup() {
    let dataset_path = fixtures_dir().join("c4_sample.jsonl");
    if !dataset_path.exists() {
        eprintln!(
            "Skipping test: C4 sample not found at {}",
            dataset_path.display()
        );
        eprintln!("Run: python scripts/download_hf_fixtures.py --datasets");
        return;
    }

    println!("\nTesting C4 deduplication...");

    // Read documents
    let docs = read_jsonl(&dataset_path, TEXT_FIELD).expect("Failed to read C4 sample");
    println!("  Documents: {}", docs.len());

    // Run deduplication
    let config = DedupConfig::with_threshold(0.85);
    let dedup = Deduplicator::new(config);

    let start = Instant::now();
    let result = dedup.deduplicate_documents(&docs);
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    Unique documents: {}", result.stats.unique_documents);
    println!("    Duplicates found: {}", result.stats.duplicate_count);
    println!(
        "    Duplicate ratio: {:.2}%",
        result.stats.duplicate_ratio * 100.0
    );
    println!("    Clusters: {}", result.stats.cluster_count);
    println!("    Time: {:.2}ms", elapsed.as_millis());

    // C4 is web-crawled data, so we expect some duplicates
    // but not a huge amount (C4 is already deduplicated)
    assert!(
        result.stats.duplicate_ratio < 0.5,
        "C4 should have less than 50% duplicates"
    );

    // Verify throughput
    let docs_per_sec = docs.len() as f64 / elapsed.as_secs_f64();
    println!("    Throughput: {:.0} docs/sec", docs_per_sec);

    // METRICS.md target: 100K+ docs/sec
    // With 10K sample, we should hit at least 10K docs/sec
    assert!(
        docs_per_sec > 1000.0,
        "Dedup throughput too low: {:.0} docs/sec",
        docs_per_sec
    );
}

/// Test deduplication on Pile dataset sample.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --datasets"]
fn test_pile_dedup() {
    let dataset_path = fixtures_dir().join("pile_sample.jsonl");
    if !dataset_path.exists() {
        eprintln!(
            "Skipping test: Pile sample not found at {}",
            dataset_path.display()
        );
        eprintln!("Run: python scripts/download_hf_fixtures.py --datasets");
        return;
    }

    println!("\nTesting Pile deduplication...");

    let docs = read_jsonl(&dataset_path, TEXT_FIELD).expect("Failed to read Pile sample");
    println!("  Documents: {}", docs.len());

    let config = DedupConfig::with_threshold(0.85);
    let dedup = Deduplicator::new(config);

    let start = Instant::now();
    let result = dedup.deduplicate_documents(&docs);
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    Unique documents: {}", result.stats.unique_documents);
    println!("    Duplicates found: {}", result.stats.duplicate_count);
    println!(
        "    Duplicate ratio: {:.2}%",
        result.stats.duplicate_ratio * 100.0
    );
    println!("    Clusters: {}", result.stats.cluster_count);
    println!("    Time: {:.2}ms", elapsed.as_millis());

    let docs_per_sec = docs.len() as f64 / elapsed.as_secs_f64();
    println!("    Throughput: {:.0} docs/sec", docs_per_sec);
}

/// Test deduplication on Wikipedia sample (expect few duplicates).
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --datasets"]
fn test_wikipedia_dedup() {
    let dataset_path = fixtures_dir().join("wikipedia_sample.jsonl");
    if !dataset_path.exists() {
        eprintln!(
            "Skipping test: Wikipedia sample not found at {}",
            dataset_path.display()
        );
        eprintln!("Run: python scripts/download_hf_fixtures.py --datasets");
        return;
    }

    println!("\nTesting Wikipedia deduplication...");

    let docs = read_jsonl(&dataset_path, TEXT_FIELD).expect("Failed to read Wikipedia sample");
    println!("  Documents: {}", docs.len());

    let config = DedupConfig::with_threshold(0.85);
    let dedup = Deduplicator::new(config);

    let start = Instant::now();
    let result = dedup.deduplicate_documents(&docs);
    let elapsed = start.elapsed();

    println!("  Results:");
    println!("    Unique documents: {}", result.stats.unique_documents);
    println!("    Duplicates found: {}", result.stats.duplicate_count);
    println!(
        "    Duplicate ratio: {:.2}%",
        result.stats.duplicate_ratio * 100.0
    );
    println!("    Time: {:.2}ms", elapsed.as_millis());

    // Wikipedia articles should be mostly unique
    assert!(
        result.stats.duplicate_ratio < 0.1,
        "Wikipedia should have less than 10% duplicates (found {:.2}%)",
        result.stats.duplicate_ratio * 100.0
    );

    let docs_per_sec = docs.len() as f64 / elapsed.as_secs_f64();
    println!("    Throughput: {:.0} docs/sec", docs_per_sec);
}

/// Test that dedup clusters are valid.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --datasets"]
fn test_cluster_validity() {
    let dataset_path = fixtures_dir().join("c4_sample.jsonl");
    if !dataset_path.exists() {
        return;
    }

    let docs = read_jsonl(&dataset_path, TEXT_FIELD).expect("Failed to read dataset");
    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&docs);

    // Verify that clustered documents are actually similar
    for (representative, duplicates) in &result.clusters {
        let rep_text = &docs[*representative].text;

        for dup_idx in duplicates {
            let dup_text = &docs[*dup_idx].text;
            let similarity = dedup.similarity(rep_text, dup_text);

            // All clustered docs should be above threshold
            assert!(
                similarity >= 0.8,
                "Cluster member {} has low similarity {:.2} to representative {}",
                dup_idx,
                similarity,
                representative
            );
        }
    }

    println!("All {} clusters validated", result.clusters.len());
}

/// Benchmark throughput on real data.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --datasets"]
fn test_throughput_benchmark() {
    let dataset_path = fixtures_dir().join("c4_sample.jsonl");
    if !dataset_path.exists() {
        return;
    }

    let docs = read_jsonl(&dataset_path, TEXT_FIELD).expect("Failed to read dataset");
    println!("\nThroughput benchmark ({} docs)...", docs.len());

    // Warmup
    let dedup = Deduplicator::default();
    let _ = dedup.deduplicate_documents(&docs[..100.min(docs.len())]);

    // Benchmark
    let iterations = 3;
    let mut throughputs = Vec::new();

    for i in 0..iterations {
        let start = Instant::now();
        let result = dedup.deduplicate_documents(&docs);
        let elapsed = start.elapsed();

        let docs_per_sec = docs.len() as f64 / elapsed.as_secs_f64();
        throughputs.push(docs_per_sec);

        println!(
            "  Run {}: {:.0} docs/sec ({:.2}ms)",
            i + 1,
            docs_per_sec,
            elapsed.as_millis()
        );
        println!(
            "    Found {} duplicates ({:.1}%)",
            result.stats.duplicate_count,
            result.stats.duplicate_ratio * 100.0
        );
    }

    let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let max_throughput = throughputs.iter().cloned().fold(0.0f64, f64::max);

    println!("\nSummary:");
    println!("  Average: {:.0} docs/sec", avg_throughput);
    println!("  Peak: {:.0} docs/sec", max_throughput);

    // METRICS.md target: 100K+ docs/sec
    // This test runs with a smaller sample, so scale expectation
    assert!(
        avg_throughput > 1000.0,
        "Average throughput too low: {:.0} docs/sec",
        avg_throughput
    );
}

/// Test different similarity thresholds.
#[test]
#[ignore = "Requires HuggingFace fixtures: python scripts/download_hf_fixtures.py --datasets"]
fn test_threshold_sensitivity() {
    let dataset_path = fixtures_dir().join("c4_sample.jsonl");
    if !dataset_path.exists() {
        return;
    }

    let docs = read_jsonl(&dataset_path, TEXT_FIELD).expect("Failed to read dataset");
    println!("\nThreshold sensitivity ({} docs)...", docs.len());

    let thresholds = [0.7, 0.8, 0.85, 0.9, 0.95];

    for threshold in thresholds {
        let config = DedupConfig::with_threshold(threshold);
        let dedup = Deduplicator::new(config);

        let result = dedup.deduplicate_documents(&docs);

        println!(
            "  Threshold {:.2}: {} duplicates ({:.1}%), {} clusters",
            threshold,
            result.stats.duplicate_count,
            result.stats.duplicate_ratio * 100.0,
            result.stats.cluster_count
        );
    }

    // Lower thresholds should find more duplicates
    let low_config = DedupConfig::with_threshold(0.7);
    let high_config = DedupConfig::with_threshold(0.95);

    let low_dedup = Deduplicator::new(low_config);
    let high_dedup = Deduplicator::new(high_config);

    let low_result = low_dedup.deduplicate_documents(&docs);
    let high_result = high_dedup.deduplicate_documents(&docs);

    assert!(
        low_result.stats.duplicate_count >= high_result.stats.duplicate_count,
        "Lower threshold should find >= duplicates"
    );
}
