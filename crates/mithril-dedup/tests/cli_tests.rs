//! CLI integration tests for mithril-dedup.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Get a Command for the mithril-dedup binary.
#[allow(deprecated)]
fn cmd() -> Command {
    Command::cargo_bin("mithril-dedup").unwrap()
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
            "Data deduplication for ML training datasets",
        ));
}

#[test]
fn test_version() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("mithril-dedup"));
}

// ============================================================================
// Argument Validation Tests
// ============================================================================

#[test]
fn test_missing_input() {
    cmd()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Input file is required"));
}

#[test]
fn test_missing_output_no_stats_only() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    fs::write(&input, "").unwrap();

    cmd()
        .args([input.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("output file required"));
}

#[test]
fn test_invalid_threshold_too_high() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    fs::write(&input, "").unwrap();

    cmd()
        .args([
            input.to_str().unwrap(),
            "--threshold",
            "1.5",
            "--stats-only",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "threshold must be between 0.0 and 1.0",
        ));
}

#[test]
fn test_invalid_threshold_negative() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    fs::write(&input, "").unwrap();

    // Note: clap may parse -0.5 as an unknown option flag, which is also an error
    cmd()
        .args([
            input.to_str().unwrap(),
            "--threshold",
            "-0.5",
            "--stats-only",
        ])
        .assert()
        .failure();
}

#[test]
fn test_invalid_permutations() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    fs::write(&input, "").unwrap();

    cmd()
        .args([
            input.to_str().unwrap(),
            "--permutations",
            "0",
            "--stats-only",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("permutations must be > 0"));
}

#[test]
fn test_invalid_ngram() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    fs::write(&input, "").unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--ngram", "0", "--stats-only"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("ngram size must be > 0"));
}

// ============================================================================
// JSONL Format Tests
// ============================================================================

#[test]
fn test_jsonl_stats_only() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    // Create test JSONL with duplicates
    let content = r#"{"text": "hello world"}
{"text": "hello world"}
{"text": "different text here"}
{"text": "another unique document"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Total documents:"))
        .stderr(predicate::str::contains("Unique documents:"))
        .stderr(predicate::str::contains("Duplicates found:"));
}

#[test]
fn test_jsonl_dedup() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("input.jsonl");
    let output = temp.path().join("output.jsonl");

    // Create test JSONL with duplicates
    let content = r#"{"text": "hello world"}
{"text": "hello world"}
{"text": "different text here"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "-o", output.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("Deduplication Results:"));

    // Verify output exists and has fewer lines
    assert!(output.exists());
    let output_content = fs::read_to_string(&output).unwrap();
    let output_lines: Vec<_> = output_content.lines().collect();
    assert!(
        output_lines.len() < 3,
        "Should have fewer documents after dedup"
    );
}

#[test]
fn test_jsonl_custom_field() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"content": "hello world"}
{"content": "hello world again"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([
            input.to_str().unwrap(),
            "--field",
            "content",
            "--stats-only",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Total documents:"));
}

#[test]
fn test_jsonl_verbose() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only", "-v"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Configuration:"))
        .stderr(predicate::str::contains("Reading input file..."));
}

// ============================================================================
// Threshold Tests
// ============================================================================

#[test]
fn test_high_threshold() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "the quick brown fox"}
{"text": "the quick brown dog"}
{"text": "something completely different"}
"#;
    fs::write(&input, content).unwrap();

    // High threshold (0.95) should find fewer duplicates
    cmd()
        .args([
            input.to_str().unwrap(),
            "--threshold",
            "0.95",
            "--stats-only",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Deduplication Results:"));
}

#[test]
fn test_low_threshold() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "the quick brown fox"}
{"text": "the quick brown dog"}
{"text": "something completely different"}
"#;
    fs::write(&input, content).unwrap();

    // Low threshold (0.5) should find more duplicates
    cmd()
        .args([
            input.to_str().unwrap(),
            "--threshold",
            "0.5",
            "--stats-only",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Deduplication Results:"));
}

// ============================================================================
// Configuration Options Tests
// ============================================================================

#[test]
fn test_permutations_option() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([
            input.to_str().unwrap(),
            "--permutations",
            "64",
            "--stats-only",
        ])
        .assert()
        .success();
}

#[test]
fn test_ngram_option() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world foo bar baz"}
{"text": "different text abc def ghi"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--ngram", "3", "--stats-only"])
        .assert()
        .success();
}

#[test]
fn test_no_verify_option() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--no-verify", "--stats-only"])
        .assert()
        .success();
}

// ============================================================================
// Streaming Mode Tests
// ============================================================================

#[test]
fn test_streaming_mode_requires_output() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--streaming"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("output file required"));
}

#[test]
fn test_streaming_mode() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("input.jsonl");
    let output = temp.path().join("output.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--streaming",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Streaming Deduplication Results:"));

    assert!(output.exists());
}

#[test]
fn test_streaming_verbose() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("input.jsonl");
    let output = temp.path().join("output.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--streaming",
            "-v",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Memory-efficient mode"));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_file_not_found() {
    cmd()
        .args(["/nonexistent/file.jsonl", "--stats-only"])
        .assert()
        .failure();
}

#[test]
fn test_empty_file() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("empty.jsonl");

    fs::write(&input, "").unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .success()
        .stderr(predicate::str::contains("No documents found"));
}

#[test]
fn test_invalid_json() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("invalid.jsonl");

    fs::write(&input, "not valid json\n").unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .failure();
}

#[test]
fn test_missing_text_field() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("notext.jsonl");

    let content = r#"{"other": "hello world"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .failure();
}

// ============================================================================
// Format Detection Tests
// ============================================================================

#[test]
fn test_auto_detect_jsonl() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .success();
}

#[test]
fn test_unknown_extension() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.xyz");

    fs::write(&input, "some content").unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Cannot detect format"));
}

#[test]
fn test_explicit_format_jsonl() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.txt"); // Wrong extension

    let content = r#"{"text": "hello world"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--format", "jsonl", "--stats-only"])
        .assert()
        .success();
}

// ============================================================================
// Exact Duplicate Tests
// ============================================================================

#[test]
fn test_exact_duplicates() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    let output = temp.path().join("output.jsonl");

    // All identical documents
    let content = r#"{"text": "hello world"}
{"text": "hello world"}
{"text": "hello world"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "-o", output.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::is_match(r"Duplicates found:\s+2").unwrap());

    // Should output only 1 document
    let output_content = fs::read_to_string(&output).unwrap();
    let lines: Vec<_> = output_content.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 1);
}

#[test]
fn test_no_duplicates() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");
    let output = temp.path().join("output.jsonl");

    // All unique documents
    let content = r#"{"text": "first unique document with enough words"}
{"text": "second unique document with enough words"}
{"text": "third unique document with enough words"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "-o", output.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::is_match(r"Duplicates found:\s+0").unwrap());

    // Should output all 3 documents
    let output_content = fs::read_to_string(&output).unwrap();
    let lines: Vec<_> = output_content.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 3);
}

// ============================================================================
// Performance Stats Tests
// ============================================================================

#[test]
fn test_performance_stats_shown() {
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("test.jsonl");

    let content = r#"{"text": "hello world"}
{"text": "different text"}
"#;
    fs::write(&input, content).unwrap();

    cmd()
        .args([input.to_str().unwrap(), "--stats-only"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Performance:"))
        .stderr(predicate::str::contains("Processing time:"))
        .stderr(predicate::str::contains("Throughput:"))
        .stderr(predicate::str::contains("Total time:"));
}
