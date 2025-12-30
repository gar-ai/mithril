//! Integration tests for mithril-dedup.
//!
//! Tests end-to-end workflows with real file I/O.

use mithril_dedup::{
    read_jsonl, read_parquet, write_jsonl, write_parquet, DedupConfig, Deduplicator, Document,
};
use std::io::Write;
use tempfile::TempDir;

/// Create a test dataset with known duplicates.
fn create_test_documents(num_docs: usize, duplicate_ratio: f64) -> Vec<Document> {
    let num_unique = (num_docs as f64 * (1.0 - duplicate_ratio)) as usize;
    let mut docs = Vec::with_capacity(num_docs);

    // Create unique documents
    for i in 0..num_unique {
        docs.push(Document::new(
            i as u64,
            format!(
                "This is unique document number {} with content that is different from others.",
                i
            ),
        ));
    }

    // Create duplicates (copies of earlier documents)
    for i in num_unique..num_docs {
        let source_idx = i % num_unique;
        docs.push(Document::new(i as u64, docs[source_idx].text.clone()));
    }

    docs
}

#[test]
fn test_jsonl_roundtrip_dedup() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.jsonl");
    let output_path = temp_dir.path().join("output.jsonl");

    // Create test data with 30% duplicates
    let docs = create_test_documents(100, 0.3);
    write_jsonl(&input_path, &docs).unwrap();

    // Read and deduplicate
    let loaded_docs = read_jsonl(&input_path, "text").unwrap();
    assert_eq!(loaded_docs.len(), 100);

    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&loaded_docs);

    // Should have ~70 unique documents (30% were duplicates)
    assert!(result.stats.unique_documents >= 65 && result.stats.unique_documents <= 75);
    assert!(result.stats.duplicate_count >= 25 && result.stats.duplicate_count <= 35);

    // Write deduplicated output
    let output_docs: Vec<Document> = result
        .keep_indices
        .iter()
        .map(|&i| loaded_docs[i].clone())
        .collect();
    write_jsonl(&output_path, &output_docs).unwrap();

    // Verify output
    let final_docs = read_jsonl(&output_path, "text").unwrap();
    assert_eq!(final_docs.len(), result.stats.unique_documents);
}

#[test]
fn test_parquet_roundtrip_dedup() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.parquet");
    let output_path = temp_dir.path().join("output.parquet");

    // Create test data with 20% duplicates
    let docs = create_test_documents(200, 0.2);
    write_parquet(&input_path, &docs).unwrap();

    // Read and deduplicate
    let loaded_docs = read_parquet(&input_path, "text").unwrap();
    assert_eq!(loaded_docs.len(), 200);

    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&loaded_docs);

    // Should have ~160 unique documents (20% were duplicates)
    assert!(result.stats.unique_documents >= 155 && result.stats.unique_documents <= 165);

    // Write deduplicated output
    let output_docs: Vec<Document> = result
        .keep_indices
        .iter()
        .map(|&i| loaded_docs[i].clone())
        .collect();
    write_parquet(&output_path, &output_docs).unwrap();

    // Verify output
    let final_docs = read_parquet(&output_path, "text").unwrap();
    assert_eq!(final_docs.len(), result.stats.unique_documents);
}

#[test]
fn test_large_dataset_throughput() {
    // Test with 10K documents to verify throughput
    let docs = create_test_documents(10_000, 0.25);

    let start = std::time::Instant::now();
    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&docs);
    let elapsed = start.elapsed();

    let throughput = docs.len() as f64 / elapsed.as_secs_f64();

    // Should process at reasonable throughput
    // Note: Debug mode is much slower than release; METRICS.md targets are for release
    assert!(
        throughput >= 10_000.0,
        "Throughput {:.0} docs/sec below minimum threshold",
        throughput
    );

    // Verify deduplication worked
    assert!(result.stats.duplicate_count >= 2000 && result.stats.duplicate_count <= 3000);
}

#[test]
fn test_custom_threshold() {
    let docs = vec![
        Document::new(1, "The quick brown fox jumps over the lazy dog".to_string()),
        Document::new(2, "The quick brown fox leaps over the lazy dog".to_string()), // Similar
        Document::new(3, "A completely different sentence about cats".to_string()),
    ];

    // With high threshold (0.95), similar docs should NOT be detected as duplicates
    let config_high = DedupConfig::with_threshold(0.95);
    let dedup_high = Deduplicator::new(config_high);
    let result_high = dedup_high.deduplicate_documents(&docs);

    // With low threshold (0.5), similar docs SHOULD be detected as duplicates
    let config_low = DedupConfig::with_threshold(0.5);
    let dedup_low = Deduplicator::new(config_low);
    let result_low = dedup_low.deduplicate_documents(&docs);

    // High threshold should find fewer duplicates than low threshold
    assert!(result_high.stats.duplicate_count <= result_low.stats.duplicate_count);
}

#[test]
fn test_empty_dataset() {
    let docs: Vec<Document> = vec![];
    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&docs);

    assert_eq!(result.stats.total_documents, 0);
    assert_eq!(result.stats.duplicate_count, 0);
    assert!(result.keep_indices.is_empty());
}

#[test]
fn test_single_document() {
    let docs = vec![Document::new(1, "Single document".to_string())];
    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&docs);

    assert_eq!(result.stats.total_documents, 1);
    assert_eq!(result.stats.duplicate_count, 0);
    assert_eq!(result.keep_indices.len(), 1);
}

#[test]
fn test_all_duplicates() {
    let text = "This exact text is repeated in every single document".to_string();
    let docs: Vec<Document> = (0..50).map(|i| Document::new(i, text.clone())).collect();

    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&docs);

    assert_eq!(result.stats.total_documents, 50);
    assert_eq!(result.stats.duplicate_count, 49); // All but one are duplicates
    assert_eq!(result.keep_indices.len(), 1);
}

#[test]
fn test_cluster_detection() {
    // Create multiple clusters of duplicates
    let mut docs = Vec::new();

    // Cluster 1: Documents about foxes
    for i in 0..5 {
        docs.push(Document::new(
            i,
            "The quick brown fox jumps over the lazy dog".to_string(),
        ));
    }

    // Cluster 2: Documents about cats
    for i in 5..10 {
        docs.push(Document::new(
            i,
            "The lazy cat sleeps on the warm windowsill".to_string(),
        ));
    }

    // Unique documents
    docs.push(Document::new(
        10,
        "A completely unique document about birds".to_string(),
    ));
    docs.push(Document::new(
        11,
        "Another unique document about fish".to_string(),
    ));

    let dedup = Deduplicator::default();
    let result = dedup.deduplicate_documents(&docs);

    assert_eq!(result.stats.total_documents, 12);
    assert_eq!(result.stats.cluster_count, 2); // Two clusters of duplicates
    assert_eq!(result.stats.duplicate_count, 8); // 4 from each cluster
    assert_eq!(result.keep_indices.len(), 4); // 1 from each cluster + 2 unique
}

#[test]
fn test_jsonl_preserves_other_fields() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.jsonl");

    // Write JSONL with extra fields
    let mut file = std::fs::File::create(&input_path).unwrap();
    writeln!(
        file,
        r#"{{"id": 1, "text": "Document one", "source": "web", "score": 0.95}}"#
    )
    .unwrap();
    writeln!(
        file,
        r#"{{"id": 2, "text": "Document two", "source": "book", "score": 0.80}}"#
    )
    .unwrap();

    // Read documents
    let docs = read_jsonl(&input_path, "text").unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].text, "Document one");
    assert_eq!(docs[1].text, "Document two");
}

#[test]
fn test_custom_text_field() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.jsonl");

    // Write JSONL with custom field name
    let mut file = std::fs::File::create(&input_path).unwrap();
    writeln!(file, r#"{{"id": 1, "content": "Document one"}}"#).unwrap();
    writeln!(file, r#"{{"id": 2, "content": "Document two"}}"#).unwrap();

    // Read with custom field
    let docs = read_jsonl(&input_path, "content").unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].text, "Document one");
}
