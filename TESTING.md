# Mithril Testing Strategy

This document defines the testing approach for all Mithril products, including research-aware TDD for exploratory components.

## Testing Philosophy

Mithril straddles production code and research:
- **mithril-core**: Production quality, traditional TDD
- **mithril-checkpoint**: Some research (compression algorithms)
- **mithril-dedup**: Some research (similarity thresholds)
- **mithril-cache**: Mostly production (well-understood problem)

We use a hybrid approach: **invariant tests** for correctness, **hypothesis tests** for assumptions, **exploration tests** for learning.

## Test Categories

### Invariant Tests (Write First, Must Pass)

Non-negotiable correctness properties. Write before implementation.

```rust
/// INVARIANT: Compression roundtrip is lossless
#[test]
fn invariant_compression_roundtrip_lossless() {
    let original = random_checkpoint(1024 * 1024);  // 1MB
    let compressed = compress(&original);
    let decompressed = decompress(&compressed);
    assert_eq!(original, decompressed);
}

/// INVARIANT: Dedup preserves unique documents
#[test]
fn invariant_dedup_preserves_unique() {
    let docs = vec![
        doc("unique one"),
        doc("unique two"),
        doc("unique three"),
    ];
    let result = deduplicate(&docs, 0.9);
    assert_eq!(result.keep.len(), 3);
}

/// INVARIANT: Cache hit returns same artifact
#[test]
fn invariant_cache_hit_exact_match() {
    let artifact = random_artifact();
    let key = cache.put(&artifact);
    let retrieved = cache.get(&key).unwrap();
    assert_eq!(artifact, retrieved);
}
```

### Hypothesis Tests (Validate Assumptions)

Encode performance expectations. May fail during development — that's learning.

```rust
/// HYPOTHESIS: Byte grouping improves bfloat16 compression by ~20%
#[test]
fn hypothesis_byte_grouping_improves_ratio() {
    let bf16_data = random_bfloat16(1_000_000);
    
    let without_grouping = zstd_compress(&bf16_data).len();
    let with_grouping = zstd_compress(&byte_group(&bf16_data)).len();
    
    let improvement = 1.0 - (with_grouping as f64 / without_grouping as f64);
    
    assert!(
        improvement >= 0.15,
        "Expected ≥15% improvement from byte grouping, got {:.1}%",
        improvement * 100.0
    );
}

/// HYPOTHESIS: MinHash similarity correlates with Jaccard (r² > 0.95)
#[test]
fn hypothesis_minhash_approximates_jaccard() {
    let pairs = generate_document_pairs(100);
    
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();
    
    for (doc1, doc2) in pairs {
        let minhash_sim = minhash_similarity(&doc1, &doc2, 128);
        let jaccard_sim = exact_jaccard(&doc1, &doc2);
        predictions.push(minhash_sim);
        actuals.push(jaccard_sim);
    }
    
    let correlation = pearson_r(&predictions, &actuals);
    assert!(correlation > 0.95, "MinHash correlation too low: {}", correlation);
}
```

### Exploration Tests (Run Manually)

Not assertions — measurements. Run to understand the problem space.

```rust
/// EXPLORATION: Compression ratios across model architectures
#[test]
#[ignore]  // cargo test exploration_ -- --ignored --nocapture
fn exploration_compression_by_architecture() {
    let checkpoints = [
        "pythia-70m", "pythia-410m", "pythia-1b",
        "llama-7b", "mistral-7b", "bert-base",
    ];
    
    println!("\n{:<20} {:>12} {:>12} {:>8}", "Model", "Original", "Compressed", "Ratio");
    println!("{:-<56}", "");
    
    for name in checkpoints {
        if let Ok(ckpt) = load_checkpoint(name) {
            let compressed = compress(&ckpt);
            let ratio = ckpt.len() as f64 / compressed.len() as f64;
            println!("{:<20} {:>12} {:>12} {:>7.1}x", 
                name, 
                format_bytes(ckpt.len()),
                format_bytes(compressed.len()),
                ratio
            );
        }
    }
}
```

### Development Workflow

**Step 1: Write invariants first (before ANY code)**
```rust
#[test]
fn invariant_roundtrip() { todo!() }
```

**Step 2: Make invariants compile with skeleton**
```rust
pub fn compress(data: &[u8]) -> Vec<u8> { unimplemented!() }
pub fn decompress(data: &[u8]) -> Vec<u8> { unimplemented!() }
```

**Step 3: Simplest implementation that passes invariants**
```rust
pub fn compress(data: &[u8]) -> Vec<u8> { 
    zstd::compress(data, 3).unwrap()  // Just works
}
```

**Step 4: Add hypothesis tests for performance expectations**
```rust
#[test]
fn hypothesis_compression_ratio() {
    // This may FAIL - that's learning
    assert!(ratio >= 10.0);
}
```

**Step 5: Improve implementation until hypotheses pass**

**Step 6: Run exploration tests to discover unknowns**
```bash
cargo test exploration_ -- --ignored --nocapture
```

**Summary:**

| Test Type | When to Write | Failure Means | CI Behavior |
|-----------|---------------|---------------|-------------|
| Invariant | Before code | Bug in code | Block merge |
| Hypothesis | After basic impl | Assumption wrong | Warn, don't block |
| Exploration | When learning | (Can't fail) | Weekly job |

## Testing Pyramid

```
                    ┌─────────────────┐
                    │    End-to-End   │  Few, slow, high confidence
                    │      Tests      │
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               │     Integration Tests      │  More, medium speed
               └─────────────┬─────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │              Unit Tests                  │  Many, fast
        └─────────────────────────────────────────┘
```

## Directory Structure

```
mithril/
├── crates/
│   ├── mithril-core/
│   │   └── src/
│   │       └── *.rs         # Unit tests in same file
│   ├── mithril-checkpoint/
│   │   ├── src/
│   │   │   └── *.rs         # Unit tests in same file
│   │   └── tests/           # Integration tests
│   │       └── *.rs
│   └── ...
├── tests/
│   ├── integration/         # Cross-crate integration tests
│   │   ├── checkpoint/
│   │   ├── dedup/
│   │   └── cache/
│   └── e2e/                 # End-to-end tests
│       ├── pytorch/
│       ├── jax/
│       └── datasets/
├── benches/
│   ├── checkpoint_bench.rs
│   ├── dedup_bench.rs
│   └── cache_bench.rs
└── fixtures/
    ├── checkpoints/         # Test checkpoint files
    ├── datasets/            # Test datasets
    └── models/              # Test model definitions
```

## Unit Tests

### Location and Naming

Unit tests live in the same file as the code they test:

```rust
// src/compression/zstd.rs

pub struct ZstdCompressor { /* ... */ }

impl ZstdCompressor {
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // ...
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compress_empty_returns_valid_output() {
        let compressor = ZstdCompressor::new(CompressionLevel::Default);
        let result = compressor.compress(&[]).unwrap();
        assert!(!result.is_empty()); // Zstd header even for empty
    }
    
    #[test]
    fn test_compress_roundtrip_preserves_data() {
        let compressor = ZstdCompressor::new(CompressionLevel::Default);
        let original = b"hello world";
        let compressed = compressor.compress(original).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }
    
    #[test]
    fn test_decompress_invalid_data_returns_error() {
        let compressor = ZstdCompressor::new(CompressionLevel::Default);
        let result = compressor.decompress(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }
}
```

### Test Naming Convention

```
test_<function>_<scenario>_<expected_result>

Examples:
- test_compress_empty_returns_valid_output
- test_compress_large_data_succeeds  
- test_decompress_invalid_data_returns_error
- test_minhash_identical_docs_returns_high_similarity
```

### Async Tests

Use `#[tokio::test]` for async tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_storage_put_get_roundtrip() {
        let storage = MemoryStorage::new();
        storage.put("key", Bytes::from("value")).await.unwrap();
        let result = storage.get("key").await.unwrap();
        assert_eq!(result.as_ref(), b"value");
    }
}
```

### Property-Based Tests

Use `proptest` for property-based testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_compression_roundtrip(data: Vec<u8>) {
        let compressor = ZstdCompressor::new(CompressionLevel::Default);
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        prop_assert_eq!(data, decompressed);
    }
    
    #[test]
    fn prop_minhash_similarity_bounds(
        doc1 in "[a-z ]{10,100}",
        doc2 in "[a-z ]{10,100}",
    ) {
        let hasher = MinHasher::new(128);
        let sig1 = hasher.hash_text(&doc1, 3);
        let sig2 = hasher.hash_text(&doc2, 3);
        let sim = MinHasher::similarity(&sig1, &sig2);
        
        prop_assert!(sim >= 0.0 && sim <= 1.0);
    }
}
```

## Integration Tests

### Location

Integration tests live in `crates/<product>/tests/`:

```rust
// crates/mithril-checkpoint/tests/pytorch_integration.rs

use mithril_checkpoint::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_compress_pytorch_state_dict() {
    let checkpoint = load_test_checkpoint("fixtures/pythia-410m.pt");
    let compressor = CheckpointCompressor::new(default_config());
    
    let compressed = compressor.compress(&checkpoint, None).await.unwrap();
    let decompressed = compressor.decompress(&compressed).await.unwrap();
    
    // Verify all tensors match
    for (name, tensor) in checkpoint.tensors {
        assert_eq!(tensor.data, decompressed.tensors[&name].data);
    }
}

#[tokio::test]
async fn test_delta_encoding_reduces_size() {
    let ckpt1 = load_test_checkpoint("fixtures/step_1000.pt");
    let ckpt2 = load_test_checkpoint("fixtures/step_1100.pt");
    
    let compressor = CheckpointCompressor::new(default_config());
    
    // Compress without delta
    let no_delta = compressor.compress(&ckpt2, None).await.unwrap();
    
    // Compress with delta
    let with_delta = compressor.compress(&ckpt2, Some(&ckpt1)).await.unwrap();
    
    // Delta should be smaller
    assert!(with_delta.compressed_size < no_delta.compressed_size);
}
```

### Test Fixtures

Store test fixtures in `fixtures/` directory:

```
fixtures/
├── checkpoints/
│   ├── pythia-410m.pt        # Small model checkpoint
│   ├── step_1000.pt          # Training checkpoint
│   └── step_1100.pt          # Next checkpoint (for delta)
├── datasets/
│   ├── duplicates.jsonl      # Known duplicates
│   ├── unique.jsonl          # Known unique docs
│   └── mixed.jsonl           # Mixed dataset
└── models/
    └── simple_model.py       # Test model definition
```

### Fixture Generation

```rust
// tests/fixtures.rs

pub fn generate_random_checkpoint(size_mb: usize) -> Checkpoint {
    let mut tensors = HashMap::new();
    let num_values = size_mb * 1024 * 1024 / 4; // f32 = 4 bytes
    
    tensors.insert(
        "weight".to_string(),
        TensorData {
            data: (0..num_values)
                .map(|_| rand::random::<f32>())
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            shape: vec![num_values],
            dtype: DType::Float32,
        },
    );
    
    Checkpoint { tensors, metadata: HashMap::new() }
}

pub fn generate_similar_checkpoints(size_mb: usize, delta: f32) -> (Checkpoint, Checkpoint) {
    let ckpt1 = generate_random_checkpoint(size_mb);
    let mut ckpt2 = ckpt1.clone();
    
    // Modify delta% of values
    // ...
    
    (ckpt1, ckpt2)
}
```

## End-to-End Tests

### PyTorch Integration

```python
# tests/e2e/pytorch/test_checkpoint_e2e.py

import pytest
import torch
import mithril.checkpoint as ckpt

@pytest.fixture
def model():
    return torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )

def test_save_load_preserves_model(model, tmp_path):
    """Full roundtrip: train → save → load → verify"""
    compressor = ckpt.Compressor(storage=f"local://{tmp_path}")
    
    # Initial state
    x = torch.randn(1, 100)
    original_output = model(x)
    
    # Save
    compressor.save(model.state_dict(), key="test")
    
    # Load into new model
    loaded = compressor.load("test")
    new_model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    new_model.load_state_dict(loaded)
    
    # Verify identical output
    new_output = new_model(x)
    assert torch.allclose(original_output, new_output)

def test_compression_ratio(model, tmp_path):
    """Verify compression achieves target ratio"""
    compressor = ckpt.Compressor(storage=f"local://{tmp_path}")
    
    meta = compressor.save(model.state_dict(), key="test")
    
    # Should achieve at least 2x compression on random weights
    assert meta.ratio >= 2.0
```

### Dataset Deduplication

```python
# tests/e2e/datasets/test_dedup_e2e.py

import pytest
import mithril.dedup as dedup

def test_dedup_removes_exact_duplicates(tmp_path):
    """Exact duplicates should be removed"""
    docs = [
        {"id": 0, "text": "This is document one."},
        {"id": 1, "text": "This is document two."},
        {"id": 2, "text": "This is document one."},  # Exact duplicate
    ]
    
    duplicator = dedup.Deduplicator(threshold=1.0)
    result = duplicator.deduplicate(docs)
    
    assert len(result.keep) == 2
    assert 2 in result.remove

def test_dedup_detects_near_duplicates(tmp_path):
    """Near-duplicates should be detected"""
    docs = [
        {"id": 0, "text": "The quick brown fox jumps over the lazy dog."},
        {"id": 1, "text": "The quick brown fox leaps over the lazy dog."},  # Near-dup
        {"id": 2, "text": "Something completely different here."},
    ]
    
    duplicator = dedup.Deduplicator(threshold=0.8)
    result = duplicator.deduplicate(docs)
    
    assert len(result.keep) == 2
```

## Benchmarks

### Criterion Benchmarks

```rust
// benches/checkpoint_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use mithril_checkpoint::*;

fn bench_compression(c: &mut Criterion) {
    let sizes = [1, 10, 100, 1000]; // MB
    
    let mut group = c.benchmark_group("compression");
    
    for size in sizes {
        let data = generate_random_data(size * 1024 * 1024);
        
        group.throughput(Throughput::Bytes(data.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("zstd", size),
            &data,
            |b, data| {
                let compressor = ZstdCompressor::new(CompressionLevel::Default);
                b.iter(|| compressor.compress(data))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("lz4", size),
            &data,
            |b, data| {
                let compressor = Lz4Compressor::new(CompressionLevel::Default);
                b.iter(|| compressor.compress(data))
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_compression);
criterion_main!(benches);
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench checkpoint_bench

# Generate HTML report
cargo bench -- --save-baseline main

# Compare to baseline
cargo bench -- --baseline main
```

## Test Data

### Downloading Test Fixtures

```bash
# Script to download test fixtures
#!/bin/bash
# scripts/download_fixtures.sh

FIXTURES_DIR="fixtures"
mkdir -p $FIXTURES_DIR/checkpoints

# Download small model checkpoint
wget -O $FIXTURES_DIR/checkpoints/pythia-70m.pt \
    "https://huggingface.co/EleutherAI/pythia-70m/resolve/main/pytorch_model.bin"
```

### Generating Test Data

```python
# scripts/generate_fixtures.py

import torch
import json

def generate_test_checkpoint(path, size_mb=10):
    """Generate a test checkpoint of given size"""
    num_params = size_mb * 1024 * 1024 // 4  # f32 = 4 bytes
    
    state_dict = {
        "weight": torch.randn(num_params),
    }
    
    torch.save(state_dict, path)

def generate_duplicate_dataset(path, num_docs=1000, dup_ratio=0.3):
    """Generate dataset with known duplicates"""
    unique_docs = [
        {"id": i, "text": f"Unique document number {i} with some content."}
        for i in range(int(num_docs * (1 - dup_ratio)))
    ]
    
    # Add duplicates
    docs = unique_docs.copy()
    for i in range(int(num_docs * dup_ratio)):
        original = unique_docs[i % len(unique_docs)]
        docs.append({"id": len(docs), "text": original["text"]})
    
    with open(path, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      
      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Run unit tests
        run: cargo test --workspace
      
      - name: Run integration tests
        run: cargo test --workspace -- --ignored
      
      - name: Run Python tests
        run: |
          pip install maturin pytest
          cd crates/mithril-python
          maturin develop
          pytest tests/

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmarks
        run: cargo bench -- --save-baseline pr
      
      - name: Compare benchmarks
        run: cargo bench -- --baseline main --load-baseline pr
```

## Coverage

### Setup

```bash
# Install coverage tools
cargo install cargo-tarpaulin

# Run with coverage
cargo tarpaulin --out Html
```

### Coverage Targets

| Crate | Target Coverage |
|-------|----------------|
| mithril-core | 80% |
| mithril-checkpoint | 70% |
| mithril-dedup | 70% |
| mithril-cache | 70% |

## Test Utilities

### Common Test Helpers

```rust
// tests/common/mod.rs

use tempfile::TempDir;

/// Create temporary directory that auto-cleans
pub fn temp_dir() -> TempDir {
    tempfile::tempdir().unwrap()
}

/// Create in-memory storage for testing
pub fn test_storage() -> MemoryStorage {
    MemoryStorage::new()
}

/// Generate random bytes
pub fn random_bytes(size: usize) -> Vec<u8> {
    (0..size).map(|_| rand::random()).collect()
}

/// Assert two byte slices are equal with better error message
pub fn assert_bytes_eq(expected: &[u8], actual: &[u8]) {
    if expected != actual {
        let diff_pos = expected.iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b);
        panic!(
            "Byte mismatch at position {:?}. Expected len: {}, actual len: {}",
            diff_pos, expected.len(), actual.len()
        );
    }
}
```

### Mock Implementations

```rust
// tests/mocks/mod.rs

use std::sync::{Arc, Mutex};

/// Mock storage that tracks operations
pub struct MockStorage {
    data: Arc<Mutex<HashMap<String, Bytes>>>,
    operations: Arc<Mutex<Vec<Operation>>>,
}

#[derive(Clone, Debug)]
pub enum Operation {
    Get(String),
    Put(String, usize),
    Delete(String),
}

impl MockStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            operations: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn operations(&self) -> Vec<Operation> {
        self.operations.lock().unwrap().clone()
    }
}

#[async_trait]
impl StorageBackend for MockStorage {
    async fn get(&self, key: &str) -> Result<Bytes, StorageError> {
        self.operations.lock().unwrap().push(Operation::Get(key.to_string()));
        self.data.lock().unwrap()
            .get(key)
            .cloned()
            .ok_or(StorageError::NotFound(key.to_string()))
    }
    
    // ... other methods
}
```
