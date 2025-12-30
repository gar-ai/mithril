//! Streaming deduplication for large-scale datasets.
//!
//! Processes documents in a streaming fashion without loading the entire dataset
//! into memory. Ideal for TB-scale datasets.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use mithril_dedup::streaming::{StreamingConfig, StreamingProcessor};
//!
//! let config = StreamingConfig::default();
//! let mut processor = StreamingProcessor::new(config);
//!
//! // Process a file
//! let stats = processor.process_jsonl("input.jsonl", "output.jsonl", "text").unwrap();
//! println!("Processed {} docs, found {} unique", stats.total, stats.unique);
//! ```
//!
//! ## Incremental Processing
//!
//! For append-only datasets, save and restore the index state:
//!
//! ```rust,no_run
//! use mithril_dedup::streaming::{StreamingConfig, StreamingProcessor, IncrementalIndex};
//!
//! // First run: process initial data
//! let mut processor = StreamingProcessor::new(StreamingConfig::default());
//! processor.process_jsonl("batch1.jsonl", "output1.jsonl", "text").unwrap();
//! processor.save_index("index.mdi").unwrap();
//!
//! // Later: load index and process new data
//! let mut processor = StreamingProcessor::load_index("index.mdi").unwrap();
//! processor.process_jsonl("batch2.jsonl", "output2.jsonl", "text").unwrap();
//! processor.save_index("index.mdi").unwrap();
//! ```

use crate::io::{IoError, JsonlReader, Result};
use crate::lsh_bloom::{LshBloomBuilder, LshBloomIndex};
use crate::minhash::MinHasher;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Configuration for streaming deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Number of MinHash permutations.
    pub num_permutations: usize,
    /// Similarity threshold (0.0 to 1.0).
    pub threshold: f64,
    /// Expected number of documents (for Bloom filter sizing).
    pub expected_docs: usize,
    /// False positive rate for Bloom filter.
    pub fp_rate: f64,
    /// Chunk size for progress reporting.
    pub report_interval: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            num_permutations: 128,
            threshold: 0.85,
            expected_docs: 10_000_000, // 10M default
            fp_rate: 0.001,
            report_interval: 100_000,
        }
    }
}

impl StreamingConfig {
    /// Create config for a specific dataset size.
    #[must_use]
    pub fn for_size(expected_docs: usize) -> Self {
        Self {
            expected_docs,
            ..Default::default()
        }
    }

    /// Set the similarity threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the number of permutations.
    #[must_use]
    pub fn with_permutations(mut self, n: usize) -> Self {
        self.num_permutations = n;
        self
    }
}

/// Statistics from streaming deduplication.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total documents processed.
    pub total: usize,
    /// Number of unique documents.
    pub unique: usize,
    /// Number of duplicates found.
    pub duplicates: usize,
    /// Duplicate ratio.
    pub duplicate_ratio: f64,
    /// Memory used by index (bytes).
    pub memory_bytes: usize,
    /// Processing time in seconds.
    pub elapsed_secs: f64,
}

impl StreamingStats {
    /// Throughput in documents per second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

/// Streaming deduplication processor.
///
/// Uses LSH with Bloom filters for memory-efficient deduplication.
/// Processes documents one at a time, immediately writing unique ones to output.
pub struct StreamingProcessor {
    config: StreamingConfig,
    index: LshBloomIndex,
    hasher: MinHasher,
    stats: StreamingStats,
}

impl StreamingProcessor {
    /// Create a new streaming processor.
    #[must_use]
    pub fn new(config: StreamingConfig) -> Self {
        let index = LshBloomBuilder::new()
            .num_permutations(config.num_permutations)
            .threshold(config.threshold)
            .expected_items(config.expected_docs)
            .fp_rate(config.fp_rate)
            .build();

        let hasher = MinHasher::new(config.num_permutations);

        Self {
            config,
            index,
            hasher,
            stats: StreamingStats::default(),
        }
    }

    /// Process a JSONL file in streaming mode.
    ///
    /// Reads documents one at a time, writes unique ones to output.
    /// Original JSON structure is preserved for kept documents.
    pub fn process_jsonl<P: AsRef<Path>>(
        &mut self,
        input: P,
        output: P,
        text_field: &str,
    ) -> Result<StreamingStats> {
        let start = std::time::Instant::now();

        let reader = JsonlReader::new(input, text_field)?;
        let out_file = File::create(output)?;
        let mut writer = BufWriter::new(out_file);

        // Note: We don't reset stats here to support incremental processing.
        // Call reset() explicitly if you want to start fresh.

        for result in reader {
            let (doc, original_line) = result?;

            self.stats.total += 1;

            // Compute signature and check for duplicate
            let signature = self.hasher.signature_from_text(&doc.text);
            let is_duplicate = self.index.check_and_insert(&signature);

            if is_duplicate {
                self.stats.duplicates += 1;
            } else {
                self.stats.unique += 1;
                // Write original line to preserve all fields
                writeln!(writer, "{}", original_line)?;
            }

            // Progress reporting
            if self.stats.total % self.config.report_interval == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "Processed {} docs ({} unique, {:.1}% dups) - {:.0} docs/sec",
                    self.stats.total,
                    self.stats.unique,
                    self.stats.duplicates as f64 / self.stats.total as f64 * 100.0,
                    self.stats.total as f64 / elapsed
                );
            }
        }

        writer.flush()?;

        self.stats.elapsed_secs = start.elapsed().as_secs_f64();
        self.stats.memory_bytes = self.index.memory_usage();
        self.stats.duplicate_ratio = if self.stats.total > 0 {
            self.stats.duplicates as f64 / self.stats.total as f64
        } else {
            0.0
        };

        Ok(self.stats.clone())
    }

    /// Process documents from a reader, writing unique ones to a writer.
    ///
    /// This is the low-level API for custom input/output.
    pub fn process_stream<R, W>(
        &mut self,
        input: R,
        mut output: W,
        text_field: &str,
    ) -> Result<StreamingStats>
    where
        R: BufRead,
        W: Write,
    {
        let start = std::time::Instant::now();
        // Note: We don't reset stats here to support incremental processing.
        // Call reset() explicitly if you want to start fresh.

        for (line_num, line_result) in input.lines().enumerate() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }

            let json: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| IoError::Parse {
                    line: line_num + 1,
                    message: e.to_string(),
                })?;

            let text = json
                .get(text_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| IoError::MissingField {
                    field: text_field.to_string(),
                    line: line_num + 1,
                })?;

            self.stats.total += 1;

            let signature = self.hasher.signature_from_text(text);
            let is_duplicate = self.index.check_and_insert(&signature);

            if is_duplicate {
                self.stats.duplicates += 1;
            } else {
                self.stats.unique += 1;
                writeln!(output, "{}", line)?;
            }
        }

        output.flush()?;

        self.stats.elapsed_secs = start.elapsed().as_secs_f64();
        self.stats.memory_bytes = self.index.memory_usage();
        self.stats.duplicate_ratio = if self.stats.total > 0 {
            self.stats.duplicates as f64 / self.stats.total as f64
        } else {
            0.0
        };

        Ok(self.stats.clone())
    }

    /// Check if a single document is a duplicate.
    ///
    /// Returns true if the document is a potential duplicate.
    /// The document is added to the index regardless.
    pub fn is_duplicate(&mut self, text: &str) -> bool {
        let signature = self.hasher.signature_from_text(text);
        self.index.check_and_insert(&signature)
    }

    /// Get the current statistics.
    #[must_use]
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Get estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }

    /// Reset the processor for a new deduplication run.
    pub fn reset(&mut self) {
        self.index.clear();
        self.stats = StreamingStats::default();
    }

    /// Save the index to a file for incremental processing.
    ///
    /// The saved index can be loaded later with `load_index()` to continue
    /// deduplication on new batches of data.
    ///
    /// # File Format
    /// The index is saved as a compressed binary file with `.mdi` extension
    /// (Mithril Dedup Index).
    pub fn save_index<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);

        let index_data = IncrementalIndex {
            version: 1,
            config: self.config.clone(),
            index: bincode_serialize(&self.index)?,
            cumulative_stats: CumulativeStats {
                total_processed: self.stats.total,
                total_unique: self.stats.unique,
                total_duplicates: self.stats.duplicates,
            },
        };

        // Use zstd compression for smaller files
        let mut encoder = zstd::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, &index_data)?;
        encoder.finish()?;

        Ok(())
    }

    /// Load a previously saved index to continue incremental processing.
    ///
    /// # Example
    /// ```rust,no_run
    /// use mithril_dedup::streaming::StreamingProcessor;
    ///
    /// // Load index from previous run
    /// let mut processor = StreamingProcessor::load_index("index.mdi").unwrap();
    ///
    /// // Continue processing new data
    /// processor.process_jsonl("new_batch.jsonl", "output.jsonl", "text").unwrap();
    ///
    /// // Save updated index
    /// processor.save_index("index.mdi").unwrap();
    /// ```
    pub fn load_index<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);

        let decoder = zstd::Decoder::new(reader)?;
        let index_data: IncrementalIndex = serde_json::from_reader(decoder)?;

        if index_data.version != 1 {
            return Err(IoError::Parse {
                line: 0,
                message: format!(
                    "Unsupported index version: {} (expected 1)",
                    index_data.version
                ),
            });
        }

        let index: LshBloomIndex = bincode_deserialize(&index_data.index)?;
        let hasher = MinHasher::new(index_data.config.num_permutations);

        Ok(Self {
            config: index_data.config,
            index,
            hasher,
            stats: StreamingStats {
                total: index_data.cumulative_stats.total_processed,
                unique: index_data.cumulative_stats.total_unique,
                duplicates: index_data.cumulative_stats.total_duplicates,
                duplicate_ratio: if index_data.cumulative_stats.total_processed > 0 {
                    index_data.cumulative_stats.total_duplicates as f64
                        / index_data.cumulative_stats.total_processed as f64
                } else {
                    0.0
                },
                memory_bytes: 0,
                elapsed_secs: 0.0,
            },
        })
    }

    /// Get the configuration used by this processor.
    #[must_use]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

// =============================================================================
// Incremental Index Persistence
// =============================================================================

/// Persisted index data for incremental processing.
///
/// This structure is saved to disk and can be loaded later to continue
/// deduplication on new batches of data.
#[derive(Serialize, Deserialize)]
pub struct IncrementalIndex {
    /// Format version for forward compatibility.
    pub version: u32,
    /// Configuration used to create the index.
    pub config: StreamingConfig,
    /// Serialized LSH Bloom index (bincode format).
    pub index: Vec<u8>,
    /// Cumulative statistics from all processed batches.
    pub cumulative_stats: CumulativeStats,
}

/// Cumulative statistics across all processed batches.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CumulativeStats {
    /// Total documents processed across all batches.
    pub total_processed: usize,
    /// Total unique documents found.
    pub total_unique: usize,
    /// Total duplicates detected.
    pub total_duplicates: usize,
}

/// Serialize with bincode for compact binary format.
fn bincode_serialize<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serialize(value).map_err(|e| IoError::Parse {
        line: 0,
        message: format!("bincode serialize error: {}", e),
    })
}

/// Deserialize from bincode.
fn bincode_deserialize<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T> {
    bincode::deserialize(data).map_err(|e| IoError::Parse {
        line: 0,
        message: format!("bincode deserialize error: {}", e),
    })
}

/// Process a JSONL file with default settings.
///
/// Convenience function for simple use cases.
pub fn deduplicate_jsonl<P: AsRef<Path>>(
    input: P,
    output: P,
    text_field: &str,
) -> Result<StreamingStats> {
    let config = StreamingConfig::default();
    let mut processor = StreamingProcessor::new(config);
    processor.process_jsonl(input, output, text_field)
}

/// Process a JSONL file with custom threshold.
pub fn deduplicate_jsonl_with_threshold<P: AsRef<Path>>(
    input: P,
    output: P,
    text_field: &str,
    threshold: f64,
) -> Result<StreamingStats> {
    let config = StreamingConfig::default().with_threshold(threshold);
    let mut processor = StreamingProcessor::new(config);
    processor.process_jsonl(input, output, text_field)
}

// =============================================================================
// Parquet streaming support
// =============================================================================

use crate::io::{read_parquet_full, write_parquet_full, RichDocument};

/// Process a Parquet file in streaming mode.
///
/// Note: Parquet files are read in batches, not truly streaming.
/// For very large files, consider splitting into chunks first.
pub fn deduplicate_parquet<P: AsRef<Path>>(
    input: P,
    output: P,
    text_column: &str,
    config: StreamingConfig,
) -> Result<StreamingStats> {
    let start = std::time::Instant::now();

    // Read all documents (Parquet requires full read for schema)
    let (docs, schema) = read_parquet_full(&input, text_column)?;

    let index = LshBloomBuilder::new()
        .num_permutations(config.num_permutations)
        .threshold(config.threshold)
        .expected_items(docs.len().max(config.expected_docs))
        .fp_rate(config.fp_rate)
        .build();

    let hasher = MinHasher::new(config.num_permutations);

    let mut unique_docs: Vec<RichDocument> = Vec::with_capacity(docs.len());
    let mut stats = StreamingStats::default();
    let mut index = index;

    for doc in docs {
        stats.total += 1;

        let signature = hasher.signature_from_text(&doc.text);
        let is_duplicate = index.check_and_insert(&signature);

        if is_duplicate {
            stats.duplicates += 1;
        } else {
            stats.unique += 1;
            unique_docs.push(doc);
        }
    }

    // Write unique documents
    write_parquet_full(&output, &unique_docs, &schema)?;

    stats.elapsed_secs = start.elapsed().as_secs_f64();
    stats.memory_bytes = index.memory_usage();
    stats.duplicate_ratio = if stats.total > 0 {
        stats.duplicates as f64 / stats.total as f64
    } else {
        0.0
    };

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_streaming_processor_basic() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        // First doc is unique
        assert!(!processor.is_duplicate("The quick brown fox jumps over the lazy dog"));

        // Same doc is duplicate
        assert!(processor.is_duplicate("The quick brown fox jumps over the lazy dog"));

        // Different doc is unique
        assert!(!processor.is_duplicate("A completely different document about something else"));
    }

    #[test]
    fn test_streaming_process_stream() {
        let input = r#"{"id": 1, "text": "Document one about topic A"}
{"id": 2, "text": "Document two about topic B"}
{"id": 3, "text": "Document one about topic A"}
{"id": 4, "text": "Document three about topic C"}"#;

        let mut output = Vec::new();
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        let stats = processor
            .process_stream(Cursor::new(input), &mut output, "text")
            .unwrap();

        assert_eq!(stats.total, 4);
        assert_eq!(stats.unique, 3);
        assert_eq!(stats.duplicates, 1);

        // Check output has 3 lines
        let output_str = String::from_utf8(output).unwrap();
        let lines: Vec<_> = output_str.lines().collect();
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_streaming_process_jsonl_file() {
        let input_content = r#"{"id": 1, "text": "First document"}
{"id": 2, "text": "Second document"}
{"id": 3, "text": "First document"}
{"id": 4, "text": "Third document"}"#;

        let input_file = NamedTempFile::new().unwrap();
        std::fs::write(input_file.path(), input_content).unwrap();

        let output_file = NamedTempFile::new().unwrap();

        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        let stats = processor
            .process_jsonl(input_file.path(), output_file.path(), "text")
            .unwrap();

        assert_eq!(stats.total, 4);
        assert_eq!(stats.unique, 3);
        assert_eq!(stats.duplicates, 1);

        // Verify output file
        let output_content = std::fs::read_to_string(output_file.path()).unwrap();
        let lines: Vec<_> = output_content.lines().collect();
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig::for_size(1_000_000)
            .with_threshold(0.9)
            .with_permutations(256);

        assert_eq!(config.expected_docs, 1_000_000);
        assert_eq!(config.threshold, 0.9);
        assert_eq!(config.num_permutations, 256);
    }

    #[test]
    fn test_streaming_stats() {
        let stats = StreamingStats {
            total: 1000,
            unique: 800,
            duplicates: 200,
            duplicate_ratio: 0.2,
            memory_bytes: 1024,
            elapsed_secs: 2.0,
        };

        assert_eq!(stats.throughput(), 500.0);
    }

    #[test]
    fn test_streaming_reset() {
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        assert!(!processor.is_duplicate("Test document"));
        assert!(processor.is_duplicate("Test document"));

        processor.reset();

        // After reset, same doc should be unique again
        assert!(!processor.is_duplicate("Test document"));
    }

    #[test]
    fn test_streaming_empty_lines() {
        let input = r#"{"id": 1, "text": "Document one"}

{"id": 2, "text": "Document two"}

"#;

        let mut output = Vec::new();
        let mut processor = StreamingProcessor::new(StreamingConfig::default());

        let stats = processor
            .process_stream(Cursor::new(input), &mut output, "text")
            .unwrap();

        assert_eq!(stats.total, 2);
        assert_eq!(stats.unique, 2);
    }

    #[test]
    fn test_convenience_function() {
        let input_content = r#"{"text": "Doc A"}
{"text": "Doc B"}
{"text": "Doc A"}"#;

        let input_file = NamedTempFile::new().unwrap();
        std::fs::write(input_file.path(), input_content).unwrap();

        let output_file = NamedTempFile::new().unwrap();

        let stats = deduplicate_jsonl(input_file.path(), output_file.path(), "text").unwrap();

        assert_eq!(stats.total, 3);
        assert_eq!(stats.unique, 2);
    }

    #[test]
    fn test_incremental_save_load() {
        let index_file = NamedTempFile::new().unwrap();

        // Process first batch
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        assert!(!processor.is_duplicate("Document one"));
        assert!(!processor.is_duplicate("Document two"));
        assert!(processor.is_duplicate("Document one")); // Duplicate

        // Save index
        processor.save_index(index_file.path()).unwrap();

        // Load index and continue
        let mut processor2 = StreamingProcessor::load_index(index_file.path()).unwrap();

        // Previous docs should still be detected as duplicates
        assert!(processor2.is_duplicate("Document one"));
        assert!(processor2.is_duplicate("Document two"));

        // New doc should be unique
        assert!(!processor2.is_duplicate("Document three"));
    }

    #[test]
    fn test_incremental_stats_preserved() {
        let index_file = NamedTempFile::new().unwrap();

        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor.is_duplicate("Doc A");
        processor.is_duplicate("Doc B");
        processor.is_duplicate("Doc A"); // Duplicate

        // Manually update stats (normally done by process_stream)
        processor.stats.total = 3;
        processor.stats.unique = 2;
        processor.stats.duplicates = 1;

        processor.save_index(index_file.path()).unwrap();

        let loaded = StreamingProcessor::load_index(index_file.path()).unwrap();

        // Stats should be preserved
        assert_eq!(loaded.stats().total, 3);
        assert_eq!(loaded.stats().unique, 2);
        assert_eq!(loaded.stats().duplicates, 1);
    }

    #[test]
    fn test_incremental_config_preserved() {
        let index_file = NamedTempFile::new().unwrap();

        let config = StreamingConfig::for_size(500_000)
            .with_threshold(0.9)
            .with_permutations(64);

        let mut processor = StreamingProcessor::new(config.clone());
        processor.is_duplicate("Test doc");
        processor.save_index(index_file.path()).unwrap();

        let loaded = StreamingProcessor::load_index(index_file.path()).unwrap();

        assert_eq!(loaded.config().expected_docs, 500_000);
        assert_eq!(loaded.config().threshold, 0.9);
        assert_eq!(loaded.config().num_permutations, 64);
    }

    #[test]
    fn test_incremental_multi_batch() {
        let index_file = NamedTempFile::new().unwrap();

        // Batch 1
        let batch1 = r#"{"text": "First batch doc 1"}
{"text": "First batch doc 2"}"#;

        let mut output1 = Vec::new();
        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        processor
            .process_stream(Cursor::new(batch1), &mut output1, "text")
            .unwrap();
        processor.save_index(index_file.path()).unwrap();

        // Batch 2 - includes a duplicate from batch 1
        let batch2 = r#"{"text": "Second batch doc 1"}
{"text": "First batch doc 1"}
{"text": "Second batch doc 2"}"#;

        let mut output2 = Vec::new();
        let mut processor2 = StreamingProcessor::load_index(index_file.path()).unwrap();
        let stats = processor2
            .process_stream(Cursor::new(batch2), &mut output2, "text")
            .unwrap();

        // Should have 5 total (2 from batch 1 + 3 from batch 2)
        assert_eq!(stats.total, 5);
        // 4 unique (2 from batch 1 + 2 new from batch 2)
        assert_eq!(stats.unique, 4);
        // 1 duplicate ("First batch doc 1" seen again)
        assert_eq!(stats.duplicates, 1);

        // Output should only have 2 lines (the new unique docs from batch 2)
        let output_str = String::from_utf8(output2).unwrap();
        let lines: Vec<_> = output_str.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_index_file_compressed() {
        let index_file = NamedTempFile::new().unwrap();

        let mut processor = StreamingProcessor::new(StreamingConfig::default());
        for i in 0..1000 {
            processor.is_duplicate(&format!("Document number {} with some text", i));
        }
        processor.stats.total = 1000;
        processor.stats.unique = 1000;

        processor.save_index(index_file.path()).unwrap();

        // File should be compressed and relatively small
        let file_size = std::fs::metadata(index_file.path()).unwrap().len();
        assert!(
            file_size < 600_000,
            "Index file should be compressed, was {} bytes",
            file_size
        );
    }
}
