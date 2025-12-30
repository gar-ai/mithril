//! mithril-dedup CLI - Data deduplication for ML training datasets.

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use indicatif::{ProgressBar, ProgressStyle};
use mithril_dedup::{
    deduplicate_parquet, read_jsonl_with_original, read_parquet, write_jsonl, write_jsonl_lines,
    write_parquet, DedupConfig, Deduplicator, Document, HybridConfig, HybridDeduplicator,
    InputFormat, MockBackend, SemanticConfig, SemanticDeduplicator, StreamingConfig,
    StreamingProcessor,
};
use serde::Serialize;
use std::collections::HashSet;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

/// JSON output for dedup results.
#[derive(Serialize)]
struct JsonOutput {
    input: String,
    output: Option<String>,
    total_documents: usize,
    unique_documents: usize,
    duplicates: usize,
    duplicate_ratio: f64,
    clusters: usize,
    elapsed_secs: f64,
    throughput_docs_s: f64,
}

/// File format for input/output.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Format {
    /// Auto-detect from file extension
    Auto,
    /// JSON Lines format
    Jsonl,
    /// Apache Parquet format
    Parquet,
}

/// Data deduplication for ML training datasets.
///
/// Efficiently detects and removes near-duplicate documents using MinHash and LSH.
/// Supports both JSONL and Parquet formats.
#[derive(Parser, Debug)]
#[command(name = "mithril-dedup")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input file path (JSONL or Parquet).
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// Output file path (JSONL or Parquet).
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Field/column name containing the text to deduplicate.
    #[arg(short = 'f', long, default_value = "text")]
    field: String,

    /// Input/output format (auto-detect from file extension by default).
    #[arg(long, value_enum, default_value = "auto")]
    format: Format,

    /// Similarity threshold (0.0-1.0). Documents with similarity >= threshold are duplicates.
    #[arg(short, long, default_value = "0.85")]
    threshold: f64,

    /// Number of MinHash permutations. More = more accurate, but slower.
    #[arg(short = 'p', long, default_value = "128")]
    permutations: usize,

    /// N-gram size for shingling.
    #[arg(short = 'n', long, default_value = "5")]
    ngram: usize,

    /// Skip candidate verification (faster but may have false positives).
    #[arg(long)]
    no_verify: bool,

    /// Print statistics only, don't write output.
    #[arg(long)]
    stats_only: bool,

    /// Verbose output.
    #[arg(short, long)]
    verbose: bool,

    /// Use streaming mode for memory-efficient processing.
    #[arg(long)]
    streaming: bool,

    /// Expected number of documents (for streaming mode Bloom filter sizing).
    #[arg(long, default_value = "10000000")]
    expected_docs: usize,

    /// Enable incremental processing (save/load index between runs).
    #[arg(long)]
    incremental: bool,

    /// Path to the index file for incremental processing (.mdi file).
    #[arg(long)]
    index_path: Option<PathBuf>,

    /// Output results as JSON.
    #[arg(long)]
    json: bool,

    /// Show progress bar.
    #[arg(long)]
    progress: bool,

    /// Use semantic (embedding-based) deduplication.
    /// Detects paraphrases and semantically similar documents.
    /// Slower than MinHash but more accurate for meaning-based duplicates.
    #[arg(long)]
    semantic: bool,

    /// Use hybrid deduplication (MinHash + semantic verification).
    /// Combines speed of MinHash with accuracy of semantic similarity.
    #[arg(long)]
    hybrid: bool,

    /// Semantic similarity threshold (0.0-1.0) for semantic/hybrid modes.
    #[arg(long, default_value = "0.9")]
    semantic_threshold: f32,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

/// Create a spinner for indeterminate progress.
fn create_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

/// Create a progress bar for determinate progress.
#[allow(dead_code)]
fn create_progress_bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Determine the effective format for a file path.
fn detect_format(path: &PathBuf, explicit_format: Format) -> Result<InputFormat, String> {
    match explicit_format {
        Format::Auto => InputFormat::from_path(path).ok_or_else(|| {
            format!(
                "Cannot detect format from file extension: {}",
                path.display()
            )
        }),
        Format::Jsonl => Ok(InputFormat::Jsonl),
        Format::Parquet => Ok(InputFormat::Parquet),
    }
}

/// Run deduplication in streaming mode.
fn run_streaming(
    args: &Cli,
    input: &PathBuf,
    input_format: InputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let output_path = match &args.output {
        Some(p) => p.clone(),
        None if args.stats_only => PathBuf::from("/dev/null"),
        None => {
            eprintln!("Error: output file required for streaming mode (use -o/--output)");
            std::process::exit(1);
        }
    };

    // Validate incremental args
    if args.incremental && args.index_path.is_none() {
        eprintln!("Error: --index-path is required when using --incremental");
        std::process::exit(1);
    }

    let config = StreamingConfig {
        num_permutations: args.permutations,
        threshold: args.threshold,
        expected_docs: args.expected_docs,
        fp_rate: 0.001,
        report_interval: if args.verbose { 100_000 } else { usize::MAX },
    };

    let pb = if args.progress && !args.json {
        Some(create_spinner("Running streaming deduplication..."))
    } else {
        None
    };

    if args.verbose && !args.json {
        eprintln!("Running streaming deduplication...");
        eprintln!("  Memory-efficient mode: processing documents one at a time");
        if args.incremental {
            eprintln!(
                "  Incremental mode: index will be saved/loaded from {}",
                args.index_path.as_ref().unwrap().display()
            );
        }
        eprintln!();
    }

    let stats = match input_format {
        InputFormat::Jsonl => {
            // Create or load processor
            let mut processor = if args.incremental {
                let index_path = args.index_path.as_ref().unwrap();
                if index_path.exists() {
                    if args.verbose && !args.json {
                        eprintln!("Loading existing index from {}...", index_path.display());
                    }
                    StreamingProcessor::load_index(index_path)?
                } else {
                    if args.verbose && !args.json {
                        eprintln!(
                            "Creating new index (will save to {})...",
                            index_path.display()
                        );
                    }
                    StreamingProcessor::new(config)
                }
            } else {
                StreamingProcessor::new(config)
            };

            let stats = processor.process_jsonl(input, &output_path, &args.field)?;

            // Save index if incremental
            if args.incremental {
                let index_path = args.index_path.as_ref().unwrap();
                if args.verbose && !args.json {
                    eprintln!("Saving index to {}...", index_path.display());
                }
                processor.save_index(index_path)?;
            }

            stats
        }
        InputFormat::Parquet => {
            if args.incremental {
                eprintln!("Error: incremental mode is not supported for Parquet files");
                std::process::exit(1);
            }
            deduplicate_parquet(input, &output_path, &args.field, config)?
        }
    };

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    if args.json {
        let output = JsonOutput {
            input: input.display().to_string(),
            output: if args.stats_only {
                None
            } else {
                Some(output_path.display().to_string())
            },
            total_documents: stats.total,
            unique_documents: stats.unique,
            duplicates: stats.duplicates,
            duplicate_ratio: stats.duplicate_ratio,
            clusters: 0,
            elapsed_secs: stats.elapsed_secs,
            throughput_docs_s: stats.throughput(),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!("Streaming Deduplication Results:");
        eprintln!(
            "  Total documents:   {} {}",
            stats.total,
            if args.incremental { "(cumulative)" } else { "" }
        );
        eprintln!(
            "  Unique documents:  {} {}",
            stats.unique,
            if args.incremental { "(cumulative)" } else { "" }
        );
        eprintln!(
            "  Duplicates found:  {} {}",
            stats.duplicates,
            if args.incremental { "(cumulative)" } else { "" }
        );
        eprintln!("  Duplicate ratio:   {:.2}%", stats.duplicate_ratio * 100.0);
        eprintln!();
        eprintln!("Performance:");
        eprintln!("  Processing time:   {:.3}s", stats.elapsed_secs);
        eprintln!("  Throughput:        {:.0} docs/sec", stats.throughput());
        eprintln!(
            "  Index memory:      {:.2} MB",
            stats.memory_bytes as f64 / 1_048_576.0
        );
        eprintln!();
        eprintln!("Total time: {:.3}s", start.elapsed().as_secs_f64());

        if args.stats_only {
            eprintln!();
            eprintln!("(Output not written: --stats-only mode)");
        }

        if args.incremental {
            eprintln!();
            eprintln!(
                "Index saved. Use --incremental --index-path {} on next run.",
                args.index_path.as_ref().unwrap().display()
            );
        }
    }

    Ok(())
}

/// Run semantic deduplication using embeddings.
fn run_semantic(
    args: &Cli,
    input: &PathBuf,
    input_format: InputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let pb = if args.progress && !args.json {
        Some(create_spinner("Running semantic deduplication..."))
    } else {
        None
    };

    if args.verbose && !args.json {
        eprintln!("Running semantic deduplication (using mock backend)...");
        eprintln!("  Threshold: {}", args.semantic_threshold);
    }

    // Read documents
    let docs: Vec<Document> = match input_format {
        InputFormat::Jsonl => {
            let docs_with_lines = read_jsonl_with_original(input, &args.field)?;
            docs_with_lines.into_iter().map(|(d, _)| d).collect()
        }
        InputFormat::Parquet => read_parquet(input, &args.field)?,
    };

    if docs.is_empty() {
        if !args.json {
            eprintln!("Warning: No documents found in input file");
        }
        return Ok(());
    }

    // Create semantic deduplicator with mock backend
    // TODO: Replace with real embedding backend (Candle, sentence-transformers)
    let backend = MockBackend::new(384);
    let config = SemanticConfig::with_threshold(args.semantic_threshold);
    let mut dedup = SemanticDeduplicator::new(backend, config);

    // Extract texts
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();

    // Deduplicate
    let result = dedup.deduplicate(&texts)?;

    let elapsed = start.elapsed();

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }

    // Write output
    if !args.stats_only {
        if let Some(ref output_path) = args.output {
            let unique_docs: Vec<_> = result
                .keep_indices
                .iter()
                .map(|&i| docs[i].clone())
                .collect();

            match input_format {
                InputFormat::Jsonl => write_jsonl(output_path, &unique_docs)?,
                InputFormat::Parquet => write_parquet(output_path, &unique_docs)?,
            }
        }
    }

    // Print results
    if args.json {
        let output = JsonOutput {
            input: input.display().to_string(),
            output: args.output.as_ref().map(|p| p.display().to_string()),
            total_documents: result.stats.total_documents,
            unique_documents: result.stats.unique_documents,
            duplicates: result.stats.duplicate_count,
            duplicate_ratio: result.stats.duplicate_ratio,
            clusters: 0,
            elapsed_secs: elapsed.as_secs_f64(),
            throughput_docs_s: docs.len() as f64 / elapsed.as_secs_f64(),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!("Semantic Deduplication Results:");
        eprintln!("  Total documents:   {}", result.stats.total_documents);
        eprintln!("  Unique documents:  {}", result.stats.unique_documents);
        eprintln!("  Duplicates found:  {}", result.stats.duplicate_count);
        eprintln!(
            "  Duplicate ratio:   {:.2}%",
            result.stats.duplicate_ratio * 100.0
        );
        eprintln!();
        eprintln!("Performance:");
        eprintln!("  Processing time:   {:.3}s", elapsed.as_secs_f64());
        eprintln!(
            "  Throughput:        {:.0} docs/sec",
            docs.len() as f64 / elapsed.as_secs_f64()
        );
    }

    Ok(())
}

/// Run hybrid deduplication (MinHash + semantic verification).
fn run_hybrid(
    args: &Cli,
    input: &PathBuf,
    input_format: InputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let pb = if args.progress && !args.json {
        Some(create_spinner("Running hybrid deduplication..."))
    } else {
        None
    };

    if args.verbose && !args.json {
        eprintln!("Running hybrid deduplication (MinHash + semantic)...");
        eprintln!("  MinHash threshold: {}", args.threshold);
        eprintln!("  Semantic threshold: {}", args.semantic_threshold);
    }

    // Read documents
    let docs: Vec<Document> = match input_format {
        InputFormat::Jsonl => {
            let docs_with_lines = read_jsonl_with_original(input, &args.field)?;
            docs_with_lines.into_iter().map(|(d, _)| d).collect()
        }
        InputFormat::Parquet => read_parquet(input, &args.field)?,
    };

    if docs.is_empty() {
        if !args.json {
            eprintln!("Warning: No documents found in input file");
        }
        return Ok(());
    }

    // Create hybrid deduplicator with mock backend
    // TODO: Replace with real embedding backend
    let backend = MockBackend::new(384);
    let config = HybridConfig {
        minhash_threshold: args.threshold,
        semantic_threshold: args.semantic_threshold,
        num_permutations: args.permutations,
        ngram_size: args.ngram,
        ..HybridConfig::default()
    };
    let mut dedup = HybridDeduplicator::new(backend, config);

    // Extract texts
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();

    // Deduplicate
    let result = dedup.deduplicate(&texts)?;

    let elapsed = start.elapsed();

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }

    // Write output
    if !args.stats_only {
        if let Some(ref output_path) = args.output {
            let unique_docs: Vec<_> = result
                .keep_indices
                .iter()
                .map(|&i| docs[i].clone())
                .collect();

            match input_format {
                InputFormat::Jsonl => write_jsonl(output_path, &unique_docs)?,
                InputFormat::Parquet => write_parquet(output_path, &unique_docs)?,
            }
        }
    }

    // Print results
    if args.json {
        let output = JsonOutput {
            input: input.display().to_string(),
            output: args.output.as_ref().map(|p| p.display().to_string()),
            total_documents: result.stats.total_documents,
            unique_documents: result.stats.unique_documents,
            duplicates: result.stats.duplicate_count,
            duplicate_ratio: result.stats.duplicate_ratio,
            clusters: 0,
            elapsed_secs: elapsed.as_secs_f64(),
            throughput_docs_s: docs.len() as f64 / elapsed.as_secs_f64(),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!("Hybrid Deduplication Results:");
        eprintln!("  Total documents:   {}", result.stats.total_documents);
        eprintln!("  Unique documents:  {}", result.stats.unique_documents);
        eprintln!("  Duplicates found:  {}", result.stats.duplicate_count);
        eprintln!(
            "  Duplicate ratio:   {:.2}%",
            result.stats.duplicate_ratio * 100.0
        );
        eprintln!("  MinHash candidates: {}", result.stats.candidate_pairs);
        eprintln!("  Semantic verified:  {}", result.stats.verified_pairs);
        eprintln!();
        eprintln!("Performance:");
        eprintln!("  Processing time:   {:.3}s", elapsed.as_secs_f64());
        eprintln!(
            "  Throughput:        {:.0} docs/sec",
            docs.len() as f64 / elapsed.as_secs_f64()
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    // Handle completions subcommand
    if let Some(Commands::Completions { shell }) = args.command {
        let mut cmd = Cli::command();
        generate(shell, &mut cmd, "mithril-dedup", &mut io::stdout());
        return Ok(());
    }

    // Require input file for dedup operations
    let input = args.input.clone().ok_or("Input file is required")?;

    // Validate arguments
    if args.threshold < 0.0 || args.threshold > 1.0 {
        eprintln!("Error: threshold must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    if args.permutations == 0 {
        eprintln!("Error: permutations must be > 0");
        std::process::exit(1);
    }

    if args.ngram == 0 {
        eprintln!("Error: ngram size must be > 0");
        std::process::exit(1);
    }

    if !args.stats_only && args.output.is_none() {
        eprintln!("Error: output file required (use -o/--output or --stats-only)");
        std::process::exit(1);
    }

    // Detect input format
    let input_format = detect_format(&input, args.format).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    // Print configuration
    if args.verbose && !args.json {
        eprintln!("Configuration:");
        eprintln!("  Input: {}", input.display());
        if let Some(ref output) = args.output {
            eprintln!("  Output: {}", output.display());
        }
        eprintln!("  Format: {:?}", input_format);
        eprintln!("  Text field: {}", args.field);
        eprintln!("  Threshold: {}", args.threshold);
        eprintln!("  Permutations: {}", args.permutations);
        eprintln!("  N-gram size: {}", args.ngram);
        eprintln!("  Verify candidates: {}", !args.no_verify);
        eprintln!("  Streaming mode: {}", args.streaming);
        if args.streaming {
            eprintln!("  Expected docs: {}", args.expected_docs);
            eprintln!("  Incremental mode: {}", args.incremental);
            if let Some(ref index_path) = args.index_path {
                eprintln!("  Index path: {}", index_path.display());
            }
        }
        eprintln!();
    }

    // Mode selection
    if args.streaming {
        return run_streaming(&args, &input, input_format);
    }
    if args.semantic {
        return run_semantic(&args, &input, input_format);
    }
    if args.hybrid {
        return run_hybrid(&args, &input, input_format);
    }

    // Read input file
    let start = Instant::now();

    let pb = if args.progress && !args.json {
        Some(create_spinner("Reading input file..."))
    } else {
        None
    };

    if args.verbose && !args.json {
        eprintln!("Reading input file...");
    }

    // Store documents and optionally original lines (for JSONL)
    let (docs, original_lines): (Vec<Document>, Option<Vec<String>>) = match input_format {
        InputFormat::Jsonl => {
            let docs_with_lines = read_jsonl_with_original(&input, &args.field)?;
            let (docs, lines): (Vec<_>, Vec<_>) = docs_with_lines.into_iter().unzip();
            (docs, Some(lines))
        }
        InputFormat::Parquet => {
            let docs = read_parquet(&input, &args.field)?;
            (docs, None)
        }
    };

    let read_time = start.elapsed();

    if let Some(ref pb) = pb {
        pb.set_message(format!("Read {} documents", docs.len()));
    }

    if args.verbose && !args.json {
        eprintln!(
            "Read {} documents in {:.2}s",
            docs.len(),
            read_time.as_secs_f64()
        );
    }

    if docs.is_empty() {
        if !args.json {
            eprintln!("Warning: No documents found in input file");
        }
        return Ok(());
    }

    // Create deduplicator
    let config = DedupConfig {
        threshold: args.threshold,
        num_permutations: args.permutations,
        ngram_size: args.ngram,
        verify_candidates: !args.no_verify,
    };

    let dedup = Deduplicator::new(config);

    // Run deduplication
    if let Some(ref pb) = pb {
        pb.set_message("Running deduplication...");
    }

    if args.verbose && !args.json {
        eprintln!("Running deduplication...");
    }

    let dedup_start = Instant::now();
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
    let result = dedup.deduplicate_texts(&texts);
    let dedup_time = dedup_start.elapsed();

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Print statistics
    let stats = &result.stats;
    let throughput = stats.total_documents as f64 / dedup_time.as_secs_f64();

    if args.json {
        let output = JsonOutput {
            input: input.display().to_string(),
            output: args.output.as_ref().map(|p| p.display().to_string()),
            total_documents: stats.total_documents,
            unique_documents: stats.unique_documents,
            duplicates: stats.duplicate_count,
            duplicate_ratio: stats.duplicate_ratio,
            clusters: stats.cluster_count,
            elapsed_secs: dedup_time.as_secs_f64(),
            throughput_docs_s: throughput,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!("Deduplication Results:");
        eprintln!("  Total documents:   {}", stats.total_documents);
        eprintln!("  Unique documents:  {}", stats.unique_documents);
        eprintln!("  Duplicates found:  {}", stats.duplicate_count);
        eprintln!("  Duplicate ratio:   {:.2}%", stats.duplicate_ratio * 100.0);
        eprintln!("  Duplicate clusters: {}", stats.cluster_count);
        eprintln!();
        eprintln!("Performance:");
        eprintln!("  Candidate pairs:   {}", stats.candidate_pairs);
        eprintln!("  Verified pairs:    {}", stats.verified_pairs);
        eprintln!("  Processing time:   {:.3}s", dedup_time.as_secs_f64());
        eprintln!("  Throughput:        {:.0} docs/sec", throughput);
    }

    // Write output if requested
    if !args.stats_only {
        if let Some(output_path) = &args.output {
            if args.verbose && !args.json {
                eprintln!();
                eprintln!("Writing output file...");
            }

            // Detect output format (use input format if auto)
            let output_format = detect_format(output_path, args.format).unwrap_or(input_format);

            // Collect indices to keep
            let keep_set: HashSet<usize> = result.keep_indices.iter().copied().collect();

            match output_format {
                InputFormat::Jsonl => {
                    let output_lines: Vec<String> = if let Some(ref lines) = original_lines {
                        lines
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| keep_set.contains(i))
                            .map(|(_, line)| line.clone())
                            .collect()
                    } else {
                        docs.iter()
                            .enumerate()
                            .filter(|(i, _)| keep_set.contains(i))
                            .map(|(_, doc)| {
                                serde_json::json!({
                                    "id": doc.id,
                                    "text": doc.text
                                })
                                .to_string()
                            })
                            .collect()
                    };
                    write_jsonl_lines(output_path, &output_lines)?;
                }
                InputFormat::Parquet => {
                    let output_docs: Vec<Document> = docs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| keep_set.contains(i))
                        .map(|(_, doc)| doc.clone())
                        .collect();
                    write_parquet(output_path, &output_docs)?;
                }
            }

            if args.verbose && !args.json {
                eprintln!(
                    "Wrote {} documents to {} ({:?})",
                    result.keep_indices.len(),
                    output_path.display(),
                    output_format
                );
            }
        }
    }

    if !args.json {
        eprintln!();
        eprintln!("Total time: {:.3}s", start.elapsed().as_secs_f64());
    }

    Ok(())
}
