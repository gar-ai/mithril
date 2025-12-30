//! mithril-checkpoint CLI - Checkpoint compression for PyTorch models.

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use indicatif::{ProgressBar, ProgressStyle};
use mithril_checkpoint::{
    CheckpointCompressor, CompressionConfig, CompressionStats, MstReader, MstWriter,
    QuantizeConfig, QuantizeMethod, SafetensorsReader, SafetensorsWriter,
};
use mithril_core::types::DType;
use rayon::prelude::*;
use serde::Serialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// JSON output for compression results.
#[derive(Serialize)]
struct JsonOutput {
    operation: String,
    input: String,
    output: Option<String>,
    original_size: usize,
    compressed_size: usize,
    ratio: f64,
    elapsed_secs: f64,
    throughput_mib_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensors: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantize_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group_size: Option<usize>,
}

/// File format for input/output.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Format {
    /// Auto-detect from file extension
    Auto,
    /// SafeTensors format (.safetensors)
    Safetensors,
    /// Mithril Safetensors compressed format (.mst)
    Mst,
    /// Raw binary tensor data
    Raw,
}

/// Data type for raw binary format.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DataTypeArg {
    /// BFloat16 (2 bytes per element)
    Bf16,
    /// Float16 (2 bytes per element)
    Fp16,
    /// Float32 (4 bytes per element)
    Fp32,
    /// Float64 (8 bytes per element)
    Fp64,
    /// Int8 (1 byte per element)
    I8,
    /// Int32 (4 bytes per element)
    I32,
    /// Int64 (8 bytes per element)
    I64,
}

impl From<DataTypeArg> for DType {
    fn from(arg: DataTypeArg) -> Self {
        match arg {
            DataTypeArg::Bf16 => DType::BFloat16,
            DataTypeArg::Fp16 => DType::Float16,
            DataTypeArg::Fp32 => DType::Float32,
            DataTypeArg::Fp64 => DType::Float64,
            DataTypeArg::I8 => DType::Int8,
            DataTypeArg::I32 => DType::Int32,
            DataTypeArg::I64 => DType::Int64,
        }
    }
}

/// Quantization method for lossy compression.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum QuantizeMethodArg {
    /// 8-bit linear quantization (2x compression, <0.1% accuracy loss)
    Int8,
    /// 4-bit NormalFloat quantization (4x compression, QLoRA-style)
    Nf4,
    /// 4-bit linear quantization
    Int4,
    /// Dynamic 8-bit per-channel quantization
    DynamicInt8,
}

impl From<QuantizeMethodArg> for QuantizeMethod {
    fn from(arg: QuantizeMethodArg) -> Self {
        match arg {
            QuantizeMethodArg::Int8 => QuantizeMethod::Int8Linear,
            QuantizeMethodArg::Nf4 => QuantizeMethod::NF4,
            QuantizeMethodArg::Int4 => QuantizeMethod::Int4,
            QuantizeMethodArg::DynamicInt8 => QuantizeMethod::DynamicInt8,
        }
    }
}

/// Checkpoint compression for PyTorch models.
///
/// Achieves 10x+ lossless compression on bf16/fp16 model weights through
/// byte grouping and zstd compression.
#[derive(Parser, Debug)]
#[command(name = "mithril-checkpoint")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compress a checkpoint file
    Compress(CompressArgs),
    /// Decompress a checkpoint file
    Decompress(DecompressArgs),
    /// Convert between formats (safetensors <-> mst)
    Convert(ConvertArgs),
    /// Show checkpoint information
    Info(InfoArgs),
    /// Generate shell completions
    Completions(CompletionsArgs),
}

#[derive(Parser, Debug)]
struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(value_enum)]
    shell: Shell,
}

#[derive(Parser, Debug)]
struct CompressArgs {
    /// Input file path
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file path
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Compression level (1-22, default 3)
    #[arg(short, long, default_value = "3", value_parser = clap::value_parser!(i32).range(1..=22))]
    level: i32,

    /// Disable byte grouping optimization
    #[arg(long)]
    no_byte_grouping: bool,

    /// Input format
    #[arg(long, value_enum, default_value = "auto")]
    format: Format,

    /// Data type for raw format (required if format=raw)
    #[arg(long, value_enum)]
    dtype: Option<DataTypeArg>,

    /// Print stats only, don't write output
    #[arg(long)]
    stats_only: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Path to previous checkpoint for delta encoding (achieves 100x+ compression)
    #[arg(long, value_name = "FILE")]
    previous: Option<PathBuf>,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Show progress bar
    #[arg(long)]
    progress: bool,

    /// Apply lossy quantization before compression (int8, nf4, int4, dynamic-int8)
    #[arg(long, value_enum)]
    quantize: Option<QuantizeMethodArg>,

    /// Group size for quantization (default: per-tensor, try 128 for better accuracy)
    #[arg(long)]
    group_size: Option<usize>,
}

#[derive(Parser, Debug)]
struct DecompressArgs {
    /// Input file path (compressed .mcp file)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file path
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Original size in bytes (required for decompression)
    #[arg(long)]
    original_size: usize,

    /// Data type (required for decompression)
    #[arg(long, value_enum)]
    dtype: DataTypeArg,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Path to previous checkpoint for delta decoding (must match compression)
    #[arg(long, value_name = "FILE")]
    previous: Option<PathBuf>,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Show progress bar
    #[arg(long)]
    progress: bool,
}

#[derive(Parser, Debug)]
struct InfoArgs {
    /// Input file path
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Show detailed tensor information
    #[arg(short, long)]
    detailed: bool,

    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

#[derive(Parser, Debug)]
struct ConvertArgs {
    /// Input file path (.safetensors or .mst)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file path (.mst or .safetensors)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Compression level for MST output (1-22, default 3)
    #[arg(short, long, default_value = "3", value_parser = clap::value_parser!(i32).range(1..=22))]
    level: i32,

    /// Disable byte grouping optimization (for MST output)
    #[arg(long)]
    no_byte_grouping: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Show progress bar
    #[arg(long)]
    progress: bool,
}

/// Create a progress bar with a standard style.
#[allow(dead_code)]
fn create_progress_bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(msg.to_string());
    pb
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

/// Detect the input format from file extension.
fn detect_format(path: &Path, explicit: Format) -> Result<Format, String> {
    match explicit {
        Format::Auto => {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());

            match ext.as_deref() {
                Some("safetensors") => Ok(Format::Safetensors),
                Some("mst") => Ok(Format::Mst),
                Some("bin") | Some("pt") | Some("raw") => Ok(Format::Raw),
                _ => Err(format!(
                    "Cannot detect format from extension: {}. Use --format to specify.",
                    path.display()
                )),
            }
        }
        other => Ok(other),
    }
}

fn compress_raw(args: &CompressArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dtype = args.dtype.ok_or("--dtype is required for raw format")?;

    let pb = if args.progress && !args.json {
        Some(create_spinner("Reading input file..."))
    } else {
        None
    };

    if args.verbose && !args.json {
        eprintln!("Reading raw file: {}", args.input.display());
    }

    let data = fs::read(&args.input)?;

    if let Some(ref pb) = pb {
        pb.set_message(format!("Read {} bytes", data.len()));
    }
    if args.verbose && !args.json {
        eprintln!("Read {} bytes", data.len());
    }

    // Read previous checkpoint if provided for delta encoding
    let previous_data = if let Some(prev_path) = &args.previous {
        if args.verbose && !args.json {
            eprintln!("Reading previous checkpoint: {}", prev_path.display());
        }
        Some(fs::read(prev_path)?)
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Compressing...");
    }

    let config = CompressionConfig {
        zstd_level: args.level,
        byte_grouping: !args.no_byte_grouping,
    };
    let compressor = CheckpointCompressor::new(config);

    let start = Instant::now();

    // Handle quantization if requested
    let (compressed, quantize_method_name) = if let Some(quantize_method) = args.quantize {
        // Build quantization config
        let quantize_config = QuantizeConfig {
            method: quantize_method.into(),
            group_size: args.group_size,
            exclude_patterns: vec![],
        };

        if args.verbose && !args.json {
            eprintln!(
                "Quantizing with {:?} (group_size: {:?})",
                quantize_method, args.group_size
            );
        }

        // Calculate shape from data size and dtype
        let element_size = match dtype {
            DataTypeArg::Bf16 | DataTypeArg::Fp16 => 2,
            DataTypeArg::Fp32 | DataTypeArg::I32 => 4,
            DataTypeArg::Fp64 | DataTypeArg::I64 => 8,
            DataTypeArg::I8 => 1,
        };
        let num_elements = data.len() / element_size;
        let shape = vec![num_elements];

        let compressed =
            compressor.compress_quantized(&data, dtype.into(), shape, &quantize_config)?;
        (compressed, Some(format!("{:?}", quantize_method)))
    } else {
        let compressed =
            compressor.compress_with_delta(&data, dtype.into(), previous_data.as_deref())?;
        (compressed, None)
    };

    let elapsed = start.elapsed();

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    let stats = CompressionStats::new(data.len(), compressed.len());
    let throughput = stats.original_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

    if args.json {
        let output = JsonOutput {
            operation: "compress".to_string(),
            input: args.input.display().to_string(),
            output: if args.stats_only {
                None
            } else {
                Some(args.output.display().to_string())
            },
            original_size: stats.original_size,
            compressed_size: stats.compressed_size,
            ratio: stats.ratio,
            elapsed_secs: elapsed.as_secs_f64(),
            throughput_mib_s: throughput,
            tensors: None,
            delta_mode: Some(args.previous.is_some()),
            quantize_method: quantize_method_name.clone(),
            group_size: args.group_size,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!("Compression Results:");
        if let Some(ref method) = quantize_method_name {
            eprintln!(
                "  Quantization:    {} (group_size: {:?})",
                method, args.group_size
            );
        }
        eprintln!("  Original size:   {} bytes", stats.original_size);
        eprintln!("  Compressed size: {} bytes", stats.compressed_size);
        eprintln!("  Ratio:           {:.2}x", stats.ratio);
        eprintln!("  Time:            {:.3}s", elapsed.as_secs_f64());
        eprintln!("  Throughput:      {:.2} MiB/s", throughput);
    }

    if !args.stats_only {
        fs::write(&args.output, &compressed)?;
        if args.verbose && !args.json {
            eprintln!();
            eprintln!("Wrote compressed data to: {}", args.output.display());
        }
    }

    Ok(())
}

fn compress_safetensors(args: &CompressArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.verbose {
        eprintln!("Reading safetensors file: {}", args.input.display());
    }

    let mut reader = SafetensorsReader::open(&args.input)?;
    let tensor_names: Vec<String> = reader.tensor_names().map(String::from).collect();

    if args.verbose {
        eprintln!("Found {} tensors", tensor_names.len());
    }

    let config = CompressionConfig {
        zstd_level: args.level,
        byte_grouping: !args.no_byte_grouping,
    };
    let compressor = CheckpointCompressor::new(config);

    // Build quantization config if requested
    let quantize_config = args.quantize.map(|method| QuantizeConfig {
        method: method.into(),
        exclude_patterns: vec![],
        group_size: args.group_size,
    });

    // Step 1: Read all tensor data sequentially (file I/O is not parallelizable)
    let tensors: Vec<(String, DType, Vec<usize>, Vec<u8>)> = tensor_names
        .iter()
        .map(|name| {
            let meta = reader
                .tensor_meta(name)
                .ok_or_else(|| format!("Tensor not found: {}", name))?
                .clone();
            let data = reader.read_tensor(name)?;
            Ok((name.clone(), meta.dtype, meta.shape.clone(), data))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    // Step 2: Compress in parallel using rayon
    let total_original = AtomicUsize::new(0);
    let total_compressed = AtomicUsize::new(0);
    let start = Instant::now();
    let verbose = args.verbose;
    let quantize_config_ref = &quantize_config;

    let results: Result<Vec<_>, _> = tensors
        .par_iter()
        .map(|(name, dtype, shape, data)| {
            let compressed = if let Some(qconfig) = quantize_config_ref {
                compressor
                    .compress_quantized(data, *dtype, shape.clone(), qconfig)
                    .map_err(|e| mithril_core::MithrilError::Compression(e.to_string()))?
            } else {
                compressor.compress(data, *dtype)?
            };

            total_original.fetch_add(data.len(), Ordering::Relaxed);
            total_compressed.fetch_add(compressed.len(), Ordering::Relaxed);

            if verbose {
                let ratio = CheckpointCompressor::compression_ratio(data.len(), compressed.len());
                eprintln!(
                    "  {}: {} -> {} bytes ({:.2}x)",
                    name,
                    data.len(),
                    compressed.len(),
                    ratio
                );
            }

            Ok::<_, mithril_core::MithrilError>(compressed)
        })
        .collect();

    // Ensure all compressions succeeded
    let _compressed_tensors = results?;

    let elapsed = start.elapsed();
    let stats = CompressionStats::new(
        total_original.load(Ordering::Relaxed),
        total_compressed.load(Ordering::Relaxed),
    );

    eprintln!();
    eprintln!("Compression Results:");
    eprintln!("  Tensors:         {}", tensor_names.len());
    eprintln!("  Original size:   {} bytes", stats.original_size);
    eprintln!("  Compressed size: {} bytes", stats.compressed_size);
    eprintln!("  Ratio:           {:.2}x", stats.ratio);
    eprintln!("  Time:            {:.3}s", elapsed.as_secs_f64());
    eprintln!(
        "  Throughput:      {:.2} MiB/s",
        stats.original_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0)
    );

    if !args.stats_only {
        // For safetensors, we output a compressed archive
        // For now, just compress the entire data section
        let all_data = reader.read_all_data()?;
        // Compute shape for entire data blob (treat as flat tensor)
        let dtype = DType::BFloat16;
        let element_size = dtype.size_bytes();
        let num_elements = all_data.len() / element_size;
        let shape = vec![num_elements];

        let compressed = if let Some(ref qconfig) = quantize_config {
            compressor.compress_quantized(&all_data, dtype, shape, qconfig)?
        } else {
            compressor.compress(&all_data, dtype)?
        };
        fs::write(&args.output, &compressed)?;

        if args.verbose {
            eprintln!();
            eprintln!("Wrote compressed data to: {}", args.output.display());
        }
    }

    Ok(())
}

fn run_compress(args: CompressArgs) -> Result<(), Box<dyn std::error::Error>> {
    let format = detect_format(&args.input, args.format)?;

    if args.verbose {
        eprintln!("Configuration:");
        eprintln!("  Input:         {}", args.input.display());
        eprintln!("  Output:        {}", args.output.display());
        eprintln!("  Format:        {:?}", format);
        eprintln!("  Level:         {}", args.level);
        eprintln!("  Byte grouping: {}", !args.no_byte_grouping);
        if let Some(dtype) = args.dtype {
            eprintln!("  Data type:     {:?}", dtype);
        }
        if let Some(prev) = &args.previous {
            eprintln!("  Delta mode:    enabled (previous: {})", prev.display());
        }
        if let Some(quantize) = &args.quantize {
            eprintln!(
                "  Quantization:  {:?} (group_size: {:?})",
                quantize, args.group_size
            );
        }
        eprintln!();
    }

    match format {
        Format::Raw => compress_raw(&args),
        Format::Safetensors => compress_safetensors(&args),
        Format::Mst => {
            eprintln!("Note: MST files are already compressed. Use 'convert' to change formats.");
            Ok(())
        }
        Format::Auto => unreachable!("Auto should be resolved by detect_format"),
    }
}

fn run_decompress(args: DecompressArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.verbose {
        eprintln!("Configuration:");
        eprintln!("  Input:         {}", args.input.display());
        eprintln!("  Output:        {}", args.output.display());
        eprintln!("  Original size: {}", args.original_size);
        eprintln!("  Data type:     {:?}", args.dtype);
        if let Some(prev) = &args.previous {
            eprintln!("  Delta mode:    enabled (previous: {})", prev.display());
        }
        eprintln!();
    }

    let compressed = fs::read(&args.input)?;

    if args.verbose {
        eprintln!("Read {} compressed bytes", compressed.len());
    }

    // Read previous checkpoint if provided for delta decoding
    let previous_data = if let Some(prev_path) = &args.previous {
        if args.verbose {
            eprintln!("Reading previous checkpoint: {}", prev_path.display());
        }
        Some(fs::read(prev_path)?)
    } else {
        None
    };

    let compressor = CheckpointCompressor::default();

    let start = Instant::now();
    let decompressed = compressor.decompress_with_delta(
        &compressed,
        args.dtype.into(),
        args.original_size,
        previous_data.as_deref(),
    )?;
    let elapsed = start.elapsed();

    eprintln!();
    eprintln!("Decompression Results:");
    eprintln!("  Compressed size:   {} bytes", compressed.len());
    eprintln!("  Decompressed size: {} bytes", decompressed.len());
    eprintln!("  Time:              {:.3}s", elapsed.as_secs_f64());
    eprintln!(
        "  Throughput:        {:.2} MiB/s",
        decompressed.len() as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0)
    );

    fs::write(&args.output, &decompressed)?;

    if args.verbose {
        eprintln!();
        eprintln!("Wrote decompressed data to: {}", args.output.display());
    }

    Ok(())
}

fn run_info(args: InfoArgs) -> Result<(), Box<dyn std::error::Error>> {
    let format = detect_format(&args.input, Format::Auto)?;

    match format {
        Format::Safetensors => {
            let reader = SafetensorsReader::open(&args.input)?;
            let header = reader.header();

            println!("Safetensors file: {}", args.input.display());
            println!("Tensors: {}", header.tensors.len());
            println!();

            if args.detailed {
                println!(
                    "{:<40} {:>12} {:>10} {:>20}",
                    "Name", "Size", "DType", "Shape"
                );
                println!("{}", "-".repeat(86));

                for (name, meta) in &header.tensors {
                    println!(
                        "{:<40} {:>12} {:>10?} {:>20}",
                        truncate_name(name, 40),
                        format_size(meta.size),
                        meta.dtype,
                        format_shape(&meta.shape)
                    );
                }
            } else {
                let total_size: usize = header.tensors.values().map(|m| m.size).sum();

                println!("Total tensor data: {}", format_size(total_size));

                // Show dtype distribution
                let mut dtype_counts: std::collections::HashMap<DType, usize> =
                    std::collections::HashMap::new();
                for meta in header.tensors.values() {
                    *dtype_counts.entry(meta.dtype).or_insert(0) += 1;
                }

                println!("\nData types:");
                for (dtype, count) in dtype_counts {
                    println!("  {:?}: {} tensors", dtype, count);
                }
            }
        }
        Format::Mst => {
            let reader = MstReader::open(&args.input)?;
            let header = reader.header();

            println!("MST (Mithril Safetensors) file: {}", args.input.display());
            println!("Tensors: {}", header.tensors.len());
            println!(
                "Compression: {} (level {})",
                header.compression.algorithm, header.compression.level
            );
            println!(
                "Byte grouping: {}",
                if header.compression.byte_grouping {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            println!();

            let ratio = if header.total_compressed_size > 0 {
                header.total_uncompressed_size as f64 / header.total_compressed_size as f64
            } else {
                0.0
            };
            println!(
                "Uncompressed size: {}",
                format_size(header.total_uncompressed_size)
            );
            println!(
                "Compressed size:   {}",
                format_size(header.total_compressed_size)
            );
            println!("Compression ratio: {:.2}x", ratio);

            if args.detailed {
                println!();
                println!(
                    "{:<40} {:>12} {:>12} {:>10} {:>20}",
                    "Name", "Uncomp.", "Comp.", "DType", "Shape"
                );
                println!("{}", "-".repeat(98));

                for (name, info) in &header.tensors {
                    println!(
                        "{:<40} {:>12} {:>12} {:>10} {:>20}",
                        truncate_name(name, 40),
                        format_size(info.uncompressed_size),
                        format_size(info.compressed_size),
                        &info.dtype,
                        format_shape(&info.shape)
                    );
                }
            }

            if !header.metadata.is_empty() {
                println!();
                println!("Metadata:");
                for (key, value) in &header.metadata {
                    println!("  {}: {}", key, value);
                }
            }
        }
        Format::Raw => {
            let metadata = fs::metadata(&args.input)?;
            println!("Raw binary file: {}", args.input.display());
            println!("Size: {}", format_size(metadata.len() as usize));
            println!("\nNote: Use --format safetensors for structured checkpoint files.");
        }
        Format::Auto => unreachable!(),
    }

    Ok(())
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - (max_len - 3)..])
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(", "))
}

fn run_convert(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input_format = detect_format(&args.input, Format::Auto)?;
    let output_format = detect_format(&args.output, Format::Auto)?;

    if args.verbose && !args.json {
        eprintln!("Configuration:");
        eprintln!(
            "  Input:         {} ({:?})",
            args.input.display(),
            input_format
        );
        eprintln!(
            "  Output:        {} ({:?})",
            args.output.display(),
            output_format
        );
        eprintln!("  Level:         {}", args.level);
        eprintln!("  Byte grouping: {}", !args.no_byte_grouping);
        eprintln!();
    }

    let pb = if args.progress && !args.json {
        Some(create_spinner("Converting..."))
    } else {
        None
    };

    let start = Instant::now();

    match (input_format, output_format) {
        (Format::Safetensors, Format::Mst) => {
            // Compress: safetensors -> mst
            if args.verbose && !args.json {
                eprintln!("Reading safetensors file...");
            }

            let mut reader = SafetensorsReader::open(&args.input)?;
            let header = reader.header().clone();

            let config = CompressionConfig {
                zstd_level: args.level,
                byte_grouping: !args.no_byte_grouping,
            };
            let mut writer = MstWriter::with_config(config);

            for name in header.tensors.keys() {
                let meta = header.tensors.get(name).unwrap();
                let data = reader.read_tensor(name)?;
                writer.add_tensor(name, data, meta.dtype, meta.shape.clone());
            }

            writer.write_file(&args.output)?;

            let elapsed = start.elapsed();
            let original_size = header.data_size;
            let compressed_size = writer.compressed_size();
            let ratio = original_size as f64 / compressed_size as f64;
            let throughput = original_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

            if let Some(pb) = pb {
                pb.finish_and_clear();
            }

            if args.json {
                let output = JsonOutput {
                    operation: "convert".to_string(),
                    input: args.input.display().to_string(),
                    output: Some(args.output.display().to_string()),
                    original_size,
                    compressed_size,
                    ratio,
                    elapsed_secs: elapsed.as_secs_f64(),
                    throughput_mib_s: throughput,
                    tensors: Some(header.tensors.len()),
                    delta_mode: None,
                    quantize_method: None,
                    group_size: None,
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                eprintln!();
                eprintln!("Conversion Results (safetensors -> mst):");
                eprintln!("  Tensors:         {}", header.tensors.len());
                eprintln!("  Original size:   {}", format_size(original_size));
                eprintln!("  Compressed size: {}", format_size(compressed_size));
                eprintln!("  Ratio:           {:.2}x", ratio);
                eprintln!("  Time:            {:.3}s", elapsed.as_secs_f64());
                eprintln!("  Throughput:      {:.2} MiB/s", throughput);
                eprintln!();
                eprintln!("Wrote: {}", args.output.display());
            }
        }
        (Format::Mst, Format::Safetensors) => {
            // Decompress: mst -> safetensors
            if args.verbose && !args.json {
                eprintln!("Reading MST file...");
            }

            let mut reader = MstReader::open(&args.input)?;
            let header = reader.header().clone();

            let mut writer = SafetensorsWriter::new();

            // Copy metadata
            for (key, value) in &header.metadata {
                writer.add_metadata(key, value);
            }

            // Decompress and add each tensor
            for (name, info) in &header.tensors {
                let data = reader.read_tensor(name)?;
                let dtype = match info.dtype.as_str() {
                    "F32" | "float32" => DType::Float32,
                    "F16" | "float16" => DType::Float16,
                    "BF16" | "bfloat16" => DType::BFloat16,
                    "F64" | "float64" => DType::Float64,
                    "I32" | "int32" => DType::Int32,
                    "I64" | "int64" => DType::Int64,
                    "I8" | "int8" => DType::Int8,
                    "U8" | "uint8" => DType::UInt8,
                    "BOOL" | "bool" => DType::Bool,
                    _ => DType::UInt8,
                };
                writer.add_tensor(name, data, dtype, info.shape.clone());
            }

            writer.write_file(&args.output)?;

            let elapsed = start.elapsed();
            let compressed_size = header.total_compressed_size;
            let decompressed_size = header.total_uncompressed_size;
            let throughput = decompressed_size as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

            if let Some(pb) = pb {
                pb.finish_and_clear();
            }

            if args.json {
                let output = JsonOutput {
                    operation: "convert".to_string(),
                    input: args.input.display().to_string(),
                    output: Some(args.output.display().to_string()),
                    original_size: decompressed_size,
                    compressed_size,
                    ratio: decompressed_size as f64 / compressed_size as f64,
                    elapsed_secs: elapsed.as_secs_f64(),
                    throughput_mib_s: throughput,
                    tensors: Some(header.tensors.len()),
                    delta_mode: None,
                    quantize_method: None,
                    group_size: None,
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                eprintln!();
                eprintln!("Conversion Results (mst -> safetensors):");
                eprintln!("  Tensors:           {}", header.tensors.len());
                eprintln!("  Compressed size:   {}", format_size(compressed_size));
                eprintln!("  Decompressed size: {}", format_size(decompressed_size));
                eprintln!("  Time:              {:.3}s", elapsed.as_secs_f64());
                eprintln!("  Throughput:        {:.2} MiB/s", throughput);
                eprintln!();
                eprintln!("Wrote: {}", args.output.display());
            }
        }
        (Format::Safetensors, Format::Safetensors) | (Format::Mst, Format::Mst) => {
            return Err("Input and output formats are the same. No conversion needed.".into());
        }
        _ => {
            return Err("Unsupported conversion. Use safetensors <-> mst.".into());
        }
    }

    Ok(())
}

fn run_completions(args: CompletionsArgs) {
    let mut cmd = Cli::command();
    generate(
        args.shell,
        &mut cmd,
        "mithril-checkpoint",
        &mut io::stdout(),
    );
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Compress(args) => run_compress(args),
        Commands::Decompress(args) => run_decompress(args),
        Commands::Convert(args) => run_convert(args),
        Commands::Info(args) => run_info(args),
        Commands::Completions(args) => {
            run_completions(args);
            return;
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
