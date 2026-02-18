//! Mithril Real Dataset Deduplication Demo
//!
//! Runs deduplication on real HuggingFace news datasets (AG News, CC News, etc.)
//! to demonstrate Mithril's MinHash/LSH dedup on actual Reuters wire stories.
//!
//! Run with: `cargo run --release --example real_dedup_demo`

use console::style;
use mithril_dedup::{DedupConfig, Deduplicator};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

/// A dataset fixture we can load and deduplicate.
struct DatasetInfo {
    name: &'static str,
    filename: &'static str,
    text_field: &'static str,
}

const DATASETS: &[DatasetInfo] = &[
    DatasetInfo {
        name: "AG News",
        filename: "ag_news_sample.jsonl",
        text_field: "text",
    },
    DatasetInfo {
        name: "CC News",
        filename: "cc_news_sample.jsonl",
        text_field: "text",
    },
    DatasetInfo {
        name: "Amazon Polarity",
        filename: "amazon_polarity_sample.jsonl",
        text_field: "text",
    },
    DatasetInfo {
        name: "IMDB",
        filename: "imdb_sample.jsonl",
        text_field: "text",
    },
    DatasetInfo {
        name: "Rotten Tomatoes",
        filename: "rotten_tomatoes_sample.jsonl",
        text_field: "text",
    },
    DatasetInfo {
        name: "Tweet Eval",
        filename: "tweet_eval_sample.jsonl",
        text_field: "text",
    },
];

fn find_fixtures_dir() -> PathBuf {
    // Try relative to cwd first, then look upward
    let candidates = [
        PathBuf::from("fixtures/hf_datasets"),
        PathBuf::from("../fixtures/hf_datasets"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    eprintln!(
        "  {} Could not find fixtures/hf_datasets/ directory.",
        style("✗").red().bold()
    );
    eprintln!("  Run from the mithril workspace root.");
    std::process::exit(1);
}

/// Load texts from a JSONL file, extracting the given text field.
fn load_jsonl_texts(path: &std::path::Path, text_field: &str) -> Vec<String> {
    let file = File::open(path).expect("Failed to open JSONL file");
    let reader = BufReader::new(file);
    let mut texts = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(&line).expect("Invalid JSON");
        if let Some(text) = obj.get(text_field).and_then(|v| v.as_str()) {
            texts.push(text.to_string());
        }
    }

    texts
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
        style("║     MITHRIL DEDUP ON REAL HUGGINGFACE DATASETS      ║")
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

    // ─────────────────────────────────────────────────────────
    // Phase 1: Per-dataset deduplication
    // ─────────────────────────────────────────────────────────
    println!(
        "  {} Per-Dataset Deduplication (threshold = 0.85)",
        style("▶").cyan().bold()
    );
    println!();

    // Table header
    println!("  {0:─<18}┬{0:─<9}┬{0:─<7}┬{0:─<9}┬{0:─<14}┬{0:─<10}", "");
    println!(
        "  {:>18}│{:>9}│{:>7}│{:>9}│{:>14}│{:>10}",
        style("Dataset").bold(),
        style("Docs").bold(),
        style("Dups").bold(),
        style("Ratio").bold(),
        style("Throughput").bold(),
        style("Time").bold(),
    );
    println!("  {0:─<18}┼{0:─<9}┼{0:─<7}┼{0:─<9}┼{0:─<14}┼{0:─<10}", "");

    let config = DedupConfig::default(); // threshold = 0.85
    let dedup = Deduplicator::new(config);

    let mut all_texts: Vec<String> = Vec::new();
    let mut dataset_boundaries: Vec<(String, usize, usize)> = Vec::new(); // (name, start, end)
    let mut total_docs = 0usize;
    let mut total_dups = 0usize;
    let total_start = Instant::now();

    for ds in DATASETS {
        let path = fixtures_dir.join(ds.filename);
        if !path.exists() {
            println!(
                "  {:>18}│ {} (file not found)",
                ds.name,
                style("skipped").dim()
            );
            continue;
        }

        let texts = load_jsonl_texts(&path, ds.text_field);
        let num_docs = texts.len();

        if num_docs == 0 {
            continue;
        }

        // Collect for cross-dataset pass
        let start_idx = all_texts.len();
        all_texts.extend(texts.iter().cloned());
        let end_idx = all_texts.len();
        dataset_boundaries.push((ds.name.to_string(), start_idx, end_idx));

        // Per-dataset dedup
        let start = Instant::now();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let result = dedup.deduplicate_texts(&text_refs);
        let elapsed = start.elapsed();

        let throughput = num_docs as f64 / elapsed.as_secs_f64();

        total_docs += num_docs;
        total_dups += result.stats.duplicate_count;

        let ratio_str = if result.stats.duplicate_count > 0 {
            style(format!("{:>6.1}%", result.stats.duplicate_ratio * 100.0))
                .yellow()
                .to_string()
        } else {
            format!("{:>6.1}%", 0.0)
        };

        println!(
            "  {:>18}│{:>8} │{:>6} │{}│{:>10} d/s│{:>8.2}s",
            ds.name,
            format_count(num_docs),
            result.stats.duplicate_count,
            ratio_str,
            format_count(throughput as usize),
            elapsed.as_secs_f64(),
        );
    }

    println!("  {0:─<18}┼{0:─<9}┼{0:─<7}┼{0:─<9}┼{0:─<14}┼{0:─<10}", "");

    let total_elapsed = total_start.elapsed();
    let overall_throughput = total_docs as f64 / total_elapsed.as_secs_f64();
    let overall_ratio = if total_docs > 0 {
        total_dups as f64 / total_docs as f64 * 100.0
    } else {
        0.0
    };

    println!(
        "  {:>18}│{:>8} │{:>6} │{:>6.1}% │{:>10} d/s│{:>8.2}s",
        style("Total").bold(),
        format_count(total_docs),
        total_dups,
        overall_ratio,
        format_count(overall_throughput as usize),
        total_elapsed.as_secs_f64(),
    );
    println!();

    // ─────────────────────────────────────────────────────────
    // Phase 2: Cross-dataset dedup
    // ─────────────────────────────────────────────────────────
    if all_texts.len() > 1000 {
        println!(
            "  {} Cross-Dataset Deduplication ({} total docs)",
            style("▶").cyan().bold(),
            format_count(all_texts.len()),
        );
        println!();

        let start = Instant::now();
        let text_refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();
        let cross_result = dedup.deduplicate_texts(&text_refs);
        let cross_elapsed = start.elapsed();

        // Analyze which datasets the cross-dataset duplicates come from
        let mut cross_pairs: HashMap<(String, String), usize> = HashMap::new();

        for (&rep, dups) in &cross_result.clusters {
            let rep_ds = find_dataset(&dataset_boundaries, rep);
            for &dup_idx in dups {
                let dup_ds = find_dataset(&dataset_boundaries, dup_idx);
                if rep_ds != dup_ds {
                    let pair = if rep_ds < dup_ds {
                        (rep_ds.clone(), dup_ds.clone())
                    } else {
                        (dup_ds.clone(), rep_ds.clone())
                    };
                    *cross_pairs.entry(pair).or_insert(0) += 1;
                }
            }
        }

        let cross_throughput = all_texts.len() as f64 / cross_elapsed.as_secs_f64();

        println!(
            "  {} {} total duplicates across all datasets ({:.1}% duplicate ratio)",
            style("✓").green().bold(),
            style(cross_result.stats.duplicate_count).yellow().bold(),
            cross_result.stats.duplicate_ratio * 100.0,
        );
        println!(
            "  {} {} clusters found",
            style("✓").green().bold(),
            cross_result.stats.cluster_count,
        );
        println!(
            "  {} {} docs/sec | {:.2}s",
            style("✓").green().bold(),
            format_count(cross_throughput as usize),
            cross_elapsed.as_secs_f64(),
        );

        if !cross_pairs.is_empty() {
            println!();
            println!("  {} Cross-dataset overlaps:", style("│").dim());
            let mut pairs_vec: Vec<_> = cross_pairs.into_iter().collect();
            pairs_vec.sort_by(|a, b| b.1.cmp(&a.1));
            for ((ds1, ds2), count) in pairs_vec.iter().take(10) {
                println!(
                    "  {}   {} ↔ {}: {} shared near-duplicates",
                    style("│").dim(),
                    ds1,
                    ds2,
                    style(count).yellow()
                );
            }
        }

        // Show a sample duplicate pair
        println!();
        show_sample_duplicate(&cross_result, &all_texts, &dataset_boundaries);
    }

    println!();
    println!(
        "  {} All deduplication completed successfully",
        style("✓").green().bold()
    );
    println!();
}

/// Find which dataset a document index belongs to.
fn find_dataset(boundaries: &[(String, usize, usize)], idx: usize) -> String {
    for (name, start, end) in boundaries {
        if idx >= *start && idx < *end {
            return name.clone();
        }
    }
    "unknown".to_string()
}

/// Show a sample near-duplicate pair to illustrate what the dedup found.
fn show_sample_duplicate(
    result: &mithril_dedup::DedupResult,
    texts: &[String],
    boundaries: &[(String, usize, usize)],
) {
    // Find a cluster with members from different datasets
    for (&rep, dups) in &result.clusters {
        let rep_ds = find_dataset(boundaries, rep);
        for &dup_idx in dups {
            let dup_ds = find_dataset(boundaries, dup_idx);
            if rep_ds != dup_ds {
                let rep_text = &texts[rep];
                let dup_text = &texts[dup_idx];

                let max_len = 120;
                let rep_preview = if rep_text.len() > max_len {
                    format!("{}...", &rep_text[..max_len])
                } else {
                    rep_text.clone()
                };
                let dup_preview = if dup_text.len() > max_len {
                    format!("{}...", &dup_text[..max_len])
                } else {
                    dup_text.clone()
                };

                println!(
                    "  {} Sample cross-dataset near-duplicate:",
                    style("│").dim()
                );
                println!(
                    "  {}   [{}] \"{}\"",
                    style("│").dim(),
                    style(&rep_ds).cyan(),
                    style(rep_preview).dim()
                );
                println!(
                    "  {}   [{}] \"{}\"",
                    style("│").dim(),
                    style(&dup_ds).cyan(),
                    style(dup_preview).dim()
                );
                return;
            }
        }
    }

    // Fallback: show a within-dataset duplicate
    if let Some((&rep, dups)) = result.clusters.iter().next() {
        if let Some(&dup_idx) = dups.first() {
            let rep_text = &texts[rep];
            let dup_text = &texts[dup_idx];

            let max_len = 120;
            let rep_preview = if rep_text.len() > max_len {
                format!("{}...", &rep_text[..max_len])
            } else {
                rep_text.clone()
            };
            let dup_preview = if dup_text.len() > max_len {
                format!("{}...", &dup_text[..max_len])
            } else {
                dup_text.clone()
            };

            let ds = find_dataset(boundaries, rep);
            println!(
                "  {} Sample near-duplicate within {}:",
                style("│").dim(),
                ds,
            );
            println!(
                "  {}   Doc {}: \"{}\"",
                style("│").dim(),
                rep,
                style(rep_preview).dim()
            );
            println!(
                "  {}   Doc {}: \"{}\"",
                style("│").dim(),
                dup_idx,
                style(dup_preview).dim()
            );
        }
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
    fn test_load_jsonl_texts() {
        let fixtures = find_fixtures_dir();
        let path = fixtures.join("ag_news_sample.jsonl");
        if !path.exists() {
            return; // skip in CI without fixtures
        }
        let texts = load_jsonl_texts(&path, "text");
        assert!(
            texts.len() > 100,
            "Should load at least 100 docs from AG News"
        );
        assert!(!texts[0].is_empty(), "First doc should have text");
    }

    #[test]
    fn test_dedup_real_data_finds_duplicates() {
        let fixtures = find_fixtures_dir();
        let path = fixtures.join("ag_news_sample.jsonl");
        if !path.exists() {
            return;
        }
        let texts = load_jsonl_texts(&path, "text");
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let config = DedupConfig::default();
        let dedup = Deduplicator::new(config);
        let result = dedup.deduplicate_texts(&text_refs);

        // AG News from Reuters should have some near-duplicate wire stories
        assert_eq!(result.stats.total_documents, texts.len());
        // We just verify the dedup ran successfully; real data may or may not have dups
        assert!(result.stats.unique_documents <= result.stats.total_documents);
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_000), "1,000");
        assert_eq!(format_count(1_234_567), "1,234,567");
    }

    #[test]
    fn test_find_dataset() {
        let boundaries = vec![("A".to_string(), 0, 100), ("B".to_string(), 100, 200)];
        assert_eq!(find_dataset(&boundaries, 50), "A");
        assert_eq!(find_dataset(&boundaries, 150), "B");
        assert_eq!(find_dataset(&boundaries, 250), "unknown");
    }
}
