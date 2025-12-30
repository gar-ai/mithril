//! CLI integration tests for mithril-checkpoint.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Get a Command for the mithril-checkpoint binary.
#[allow(deprecated)]
fn cmd() -> Command {
    Command::cargo_bin("mithril-checkpoint").unwrap()
}

// ============================================================================
// Help and Version Tests
// ============================================================================

#[test]
fn test_help() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Checkpoint compression for PyTorch models",
        ));
}

#[test]
fn test_version() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("mithril-checkpoint"));
}

#[test]
fn test_compress_help() {
    cmd()
        .args(["compress", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Compress a checkpoint file"));
}

#[test]
fn test_decompress_help() {
    cmd()
        .args(["decompress", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Decompress a checkpoint file"));
}

#[test]
fn test_info_help() {
    cmd()
        .args(["info", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Show checkpoint information"));
}

// ============================================================================
// Argument Validation Tests
// ============================================================================

#[test]
fn test_no_subcommand() {
    cmd()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage:"));
}

#[test]
fn test_compress_missing_input() {
    cmd()
        .args(["compress", "-o", "out.mcp"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("INPUT"));
}

#[test]
fn test_compress_missing_output() {
    cmd()
        .args(["compress", "input.bin"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--output"));
}

#[test]
fn test_compress_invalid_level() {
    cmd()
        .args(["compress", "input.bin", "-o", "out.mcp", "--level", "30"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("22").or(predicate::str::contains("invalid")));
}

#[test]
fn test_decompress_missing_original_size() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.mcp");
    fs::write(&input, b"dummy").unwrap();

    cmd()
        .args([
            "decompress",
            input.to_str().unwrap(),
            "-o",
            "out.bin",
            "--dtype",
            "bf16",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--original-size"));
}

#[test]
fn test_decompress_missing_dtype() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.mcp");
    fs::write(&input, b"dummy").unwrap();

    cmd()
        .args([
            "decompress",
            input.to_str().unwrap(),
            "-o",
            "out.bin",
            "--original-size",
            "1024",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--dtype"));
}

// ============================================================================
// Raw Format Compression Tests
// ============================================================================

#[test]
fn test_compress_raw_bf16() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("test.mcp");

    // Create test data: 512 zeros (1KB of bf16 zeros compresses extremely well)
    let data: Vec<u8> = vec![0u8; 1024];
    fs::write(&input, &data).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Compression Results:"));

    assert!(output.exists());

    // Compressed should be smaller
    let compressed_size = fs::metadata(&output).unwrap().len();
    assert!(compressed_size < data.len() as u64);
}

#[test]
fn test_compress_raw_stats_only() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("test.mcp");

    let data: Vec<u8> = vec![0u8; 1024];
    fs::write(&input, &data).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
            "--stats-only",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Compression Results:"));

    // Output should NOT be written
    assert!(!output.exists());
}

#[test]
fn test_compress_raw_verbose() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("test.mcp");

    let data: Vec<u8> = vec![0u8; 1024];
    fs::write(&input, &data).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
            "-v",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Configuration:"))
        .stderr(predicate::str::contains("Reading raw file:"));
}

#[test]
fn test_compress_raw_no_byte_grouping() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("test.mcp");

    let data: Vec<u8> = vec![0u8; 1024];
    fs::write(&input, &data).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
            "--no-byte-grouping",
        ])
        .assert()
        .success();

    assert!(output.exists());
}

// ============================================================================
// Compression Level Tests
// ============================================================================

#[test]
fn test_compress_level_1() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("test.mcp");

    let data: Vec<u8> = vec![42u8; 2048];
    fs::write(&input, &data).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
            "--level",
            "1",
        ])
        .assert()
        .success();
}

#[test]
fn test_compress_level_22() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("test.mcp");

    let data: Vec<u8> = vec![42u8; 2048];
    fs::write(&input, &data).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
            "--level",
            "22",
        ])
        .assert()
        .success();
}

// ============================================================================
// Different Data Types Tests
// ============================================================================

#[test]
fn test_compress_all_dtypes() {
    let temp = TempDir::new().unwrap();

    for dtype in ["bf16", "fp16", "fp32", "fp64", "i8", "i32", "i64"] {
        let input = temp.path().join(format!("test_{}.bin", dtype));
        let output = temp.path().join(format!("test_{}.mcp", dtype));

        let data: Vec<u8> = vec![0u8; 1024];
        fs::write(&input, &data).unwrap();

        cmd()
            .args([
                "compress",
                input.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
                "--format",
                "raw",
                "--dtype",
                dtype,
            ])
            .assert()
            .success();

        assert!(output.exists(), "Output for dtype {} should exist", dtype);
    }
}

// ============================================================================
// Compress/Decompress Roundtrip Tests
// ============================================================================

#[test]
fn test_roundtrip_bf16() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("original.bin");
    let compressed = temp.path().join("compressed.mcp");
    let decompressed = temp.path().join("decompressed.bin");

    // Create test data (bf16 values are 2 bytes each)
    let original_data: Vec<u8> = (0..512).flat_map(|i| [(i % 256) as u8, 0u8]).collect();
    fs::write(&input, &original_data).unwrap();

    // Compress
    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            compressed.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
        ])
        .assert()
        .success();

    // Decompress
    cmd()
        .args([
            "decompress",
            compressed.to_str().unwrap(),
            "-o",
            decompressed.to_str().unwrap(),
            "--original-size",
            &original_data.len().to_string(),
            "--dtype",
            "bf16",
        ])
        .assert()
        .success();

    // Verify roundtrip
    let recovered = fs::read(&decompressed).unwrap();
    assert_eq!(original_data, recovered, "Roundtrip should be lossless");
}

#[test]
fn test_roundtrip_fp32() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("original.bin");
    let compressed = temp.path().join("compressed.mcp");
    let decompressed = temp.path().join("decompressed.bin");

    // Create test data (fp32 values are 4 bytes each)
    let original_data: Vec<u8> = (0..256)
        .flat_map(|i| [(i % 256) as u8, 0u8, 0u8, 0u8])
        .collect();
    fs::write(&input, &original_data).unwrap();

    // Compress
    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            compressed.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "fp32",
        ])
        .assert()
        .success();

    // Decompress
    cmd()
        .args([
            "decompress",
            compressed.to_str().unwrap(),
            "-o",
            decompressed.to_str().unwrap(),
            "--original-size",
            &original_data.len().to_string(),
            "--dtype",
            "fp32",
        ])
        .assert()
        .success();

    // Verify roundtrip
    let recovered = fs::read(&decompressed).unwrap();
    assert_eq!(original_data, recovered, "Roundtrip should be lossless");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_compress_file_not_found() {
    let temp = TempDir::new().unwrap();
    let output = temp.path().join("out.mcp");

    cmd()
        .args([
            "compress",
            "/nonexistent/file.bin",
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
            "--dtype",
            "bf16",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error:"));
}

#[test]
fn test_decompress_file_not_found() {
    let temp = TempDir::new().unwrap();
    let output = temp.path().join("out.bin");

    cmd()
        .args([
            "decompress",
            "/nonexistent/file.mcp",
            "-o",
            output.to_str().unwrap(),
            "--original-size",
            "1024",
            "--dtype",
            "bf16",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error:"));
}

#[test]
fn test_raw_format_requires_dtype() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");
    let output = temp.path().join("out.mcp");

    fs::write(&input, b"test data").unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--format",
            "raw",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("dtype"));
}

// ============================================================================
// Info Command Tests
// ============================================================================

#[test]
fn test_info_raw_file() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.bin");

    fs::write(&input, vec![0u8; 4096]).unwrap();

    cmd()
        .args(["info", input.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Raw binary file:"))
        .stdout(predicate::str::contains("Size:"));
}

#[test]
fn test_info_file_not_found() {
    cmd()
        .args(["info", "/nonexistent/file.safetensors"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error:"));
}

// ============================================================================
// Auto Format Detection Tests
// ============================================================================

#[test]
fn test_auto_format_bin_extension() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("model.bin");
    let output = temp.path().join("model.mcp");

    fs::write(&input, vec![0u8; 1024]).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--dtype",
            "bf16",
        ])
        .assert()
        .success();
}

#[test]
fn test_auto_format_pt_extension() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("model.pt");
    let output = temp.path().join("model.mcp");

    fs::write(&input, vec![0u8; 1024]).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--dtype",
            "bf16",
        ])
        .assert()
        .success();
}

#[test]
fn test_unknown_extension() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("model.xyz");
    let output = temp.path().join("model.mcp");

    fs::write(&input, vec![0u8; 1024]).unwrap();

    cmd()
        .args([
            "compress",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Cannot detect format"));
}
