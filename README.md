# Mithril

High-performance ML infrastructure toolkit for checkpoint compression, dataset deduplication, and torch.compile caching.

## Features

- **mithril-checkpoint**: Checkpoint compression at >2 GiB/s throughput
  - Lossless: 1.3-1.5x standalone, **10-3500x with delta encoding** between training steps
  - Lossy quantization: INT8/FP16/FP8 with configurable error bounds
  - PyTorch DCP integration for FSDP distributed checkpoints
  - S3-compatible storage (AWS, MinIO, etc.)

- **mithril-dedup**: Dataset deduplication at 400-600K docs/sec
  - MinHash/LSH for near-duplicate detection
  - Semantic deduplication with HNSW index
  - Optional Candle backend for native Rust embeddings
  - Parquet and JSONL support

- **mithril-cache**: Compilation caching for torch.compile
  - TorchInductor/Triton cache hook integration
  - vLLM and SGLang kernel cache managers
  - Content-addressable storage with LRU eviction
  - S3-compatible remote cache

## Installation

### Rust

```bash
cargo add mithril-checkpoint
cargo add mithril-dedup
cargo add mithril-cache
```

### Python

```bash
# From source (requires Rust toolchain)
cd crates/mithril-python
pip install maturin
maturin develop
```

## Quick Start

### Python

```python
import mithril

# Checkpoint compression
config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
compressor = mithril.CheckpointCompressor(config)
compressed = compressor.compress(data, "bf16")
decompressed = compressor.decompress(compressed, "bf16", len(data))

# Dataset deduplication
config = mithril.DedupConfig(threshold=0.85)
deduplicator = mithril.Deduplicator(config)
result = deduplicator.deduplicate(["doc1", "doc1 copy", "doc2"])
print(f"Keep: {result.keep_indices}")  # [0, 2]
print(f"Duplicates found: {result.stats.duplicate_count}")

# Similarity computation
sim = deduplicator.similarity("text1", "text2")

# Semantic deduplication (embedding-based)
from mithril import SemanticDeduplicator, SemanticConfig
config = SemanticConfig(similarity_threshold=0.9, embedding_dim=384)
dedup = SemanticDeduplicator(config)
result = dedup.deduplicate(embeddings)  # numpy array of embeddings

# Cache management with torch.compile hooks
# This automatically sets TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR
config = mithril.CacheConfig("/tmp/mithril-cache")
config = config.with_inductor(True).with_triton(True)
manager = mithril.CacheManager(config)

# Now torch.compile will use mithril-managed cache directories
import torch
@torch.compile
def my_model(x):
    return x * 2 + 1

# Check cache stats
print(manager.stats())
print(f"Inductor cache: {manager.inductor_dir()}")
print(f"Triton cache: {manager.triton_dir()}")

# Delta compression for training checkpoints (10-3500x compression!)
delta = mithril.DeltaCompressor()
_, stats1 = delta.compress_checkpoint("step_100", checkpoint_bytes)
_, stats2 = delta.compress_checkpoint("step_200", checkpoint_bytes)  # Uses delta
print(f"Compression: {stats2.ratio:.0f}x, Sparsity: {stats2.sparsity:.1%}")

# PyTorch DCP integration for FSDP
mithril.save_compressed(state_dict, "./checkpoint")
mithril.load_compressed(state_dict, "./checkpoint")

# Content-addressable storage
store = mithril.ContentStore("/tmp/cas")
address = store.put(content)
content = store.get(address)

# vLLM kernel caching
from mithril import VllmCacheManager, VllmCacheConfig
config = VllmCacheConfig("/tmp/vllm-cache")
manager = VllmCacheManager(config)
manager.set_environment()  # Sets VLLM_CACHE_ROOT, HF_HOME, TRITON_CACHE_DIR

# SGLang kernel caching
from mithril import SglangCacheManager, SglangCacheConfig
config = SglangCacheConfig("/tmp/sglang-cache")
manager = SglangCacheManager(config)
manager.set_environment()  # Sets cache environment variables
```

### Rust

```rust
use mithril_checkpoint::{CheckpointCompressor, CompressionConfig};
use mithril_dedup::{Deduplicator, DedupConfig};
use mithril_cache::ContentStore;

// Checkpoint compression
let config = CompressionConfig::default();
let compressor = CheckpointCompressor::new(config);
let compressed = compressor.compress(&data, DType::BFloat16)?;
let decompressed = compressor.decompress(&compressed, DType::BFloat16, data.len())?;

// Dataset deduplication
let config = DedupConfig::default().with_threshold(0.85);
let deduplicator = Deduplicator::new(config);
let result = deduplicator.deduplicate_texts(&["doc1", "doc2", "doc1 copy"]);
println!("Unique: {:?}", result.keep_indices);

// Content store
let store = ContentStore::new("/tmp/cas")?;
let address = store.put(&content).await?;
let retrieved = store.get(&address).await?;
```

## Python API Guide

### Two Compression Modes

Mithril offers two compression approaches depending on your data format:

| Method | Use Case | Best For |
|--------|----------|----------|
| `compress_raw(data)` | Pickle/safetensors file data | `torch.save()` output, file bytes |
| `compress(data, dtype)` | Raw aligned tensor bytes | `tensor.numpy().tobytes()` |

**When to use which:**

```python
import mithril
import torch
import io

config = mithril.CompressionConfig(zstd_level=3)
compressor = mithril.CheckpointCompressor(config)

# Option 1: compress_raw for pickle/file data
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
pickle_data = buffer.getvalue()
compressed = compressor.compress_raw(pickle_data)
decompressed = compressor.decompress_raw(compressed, len(pickle_data))

# Option 2: compress with dtype for raw tensor bytes (better compression!)
tensor = model.weight.to(torch.bfloat16)
raw_bytes = tensor.view(torch.uint16).numpy().tobytes()
compressed = compressor.compress(raw_bytes, "bf16")
decompressed = compressor.decompress(compressed, "bf16", len(raw_bytes))
```

### Byte Grouping (10-40% Better Compression)

Byte grouping separates high/low bytes of multi-byte values, improving compression for bf16/fp16/fp32 tensors:

```python
# Enable byte grouping for raw tensor bytes
config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
compressor = mithril.CheckpointCompressor(config)

# Without byte grouping: ~1.29x compression
# With byte grouping:    ~1.41x compression (on real model weights)
compressed = compressor.compress(raw_bytes, "bf16")
```

**Note:** Byte grouping only works with `compress(data, dtype)`, not `compress_raw()`.

### Delta Compression (3-70x Improvement)

For consecutive training checkpoints, delta compression achieves dramatic improvements:

```python
import numpy as np

# Simulate training: base weights + small update
base_tensor = torch.randn(512, 512, dtype=torch.bfloat16)
base_bytes = base_tensor.view(torch.uint16).numpy().tobytes()

updated_tensor = base_tensor + torch.randn_like(base_tensor) * 0.001
updated_bytes = updated_tensor.view(torch.uint16).numpy().tobytes()

config = mithril.CompressionConfig(zstd_level=3)
compressor = mithril.CheckpointCompressor(config)

# Regular compression
regular = compressor.compress(updated_bytes, "bf16")

# Delta compression - pass previous checkpoint
delta = compressor.compress(updated_bytes, "bf16", previous=base_bytes)

# Delta is typically 3-70x smaller!
print(f"Regular: {len(regular)} bytes")
print(f"Delta: {len(delta)} bytes")

# Decompress with the same previous reference
decompressed = compressor.decompress(delta, "bf16", len(updated_bytes), previous=base_bytes)
```

### Supported Data Types

| dtype | Bytes | Use Case |
|-------|-------|----------|
| `bf16` | 2 | LLM weights (recommended) |
| `fp16` | 2 | Mixed precision training |
| `fp32` | 4 | Full precision |
| `fp64` | 8 | Scientific computing |
| `i8` | 1 | Quantized models |
| `i32` | 4 | Integer tensors |
| `i64` | 8 | Large indices |
| `u8` | 1 | Byte data |
| `bool` | 1 | Masks |

## CLI

### Checkpoint Compression

```bash
# Compress a safetensors file
mithril-checkpoint compress model.safetensors -o model.mcp

# Decompress
mithril-checkpoint decompress model.mcp -o model.safetensors

# Show file info
mithril-checkpoint info model.mcp
```

### Dataset Deduplication

```bash
# Deduplicate JSONL dataset
mithril-dedup --input data.jsonl --output deduped.jsonl --threshold 0.85

# With custom text field
mithril-dedup --input data.jsonl --output deduped.jsonl --text-field content
```

## Benchmarks

Performance measured on Apple M2 (10MB bf16 data, 10K documents):

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Checkpoint compress (bf16) | 2.0-2.4 GiB/s | zstd level 3 |
| Checkpoint decompress | 2.0 GiB/s | |
| Delta compress | 10-3500x ratio | 99% sparsity between steps |
| Deduplication | 400-600K docs/sec | MinHash + LSH |
| Semantic dedup | Depends on backend | HNSW index |
| Content store put | 70+ MB/s | Blake3 hashing |
| Cache key generation | 4M keys/sec | |

Run benchmarks:

```bash
cargo bench --workspace --exclude mithril-python
```

### Delta Encoding Example

```python
# First checkpoint: 365x compression (repetitive data)
# Second checkpoint: 3531x compression with delta (99% sparsity)
# Improvement: 9.7x better with delta encoding
```

## Architecture

```text
mithril/
├── crates/
│   ├── mithril-core/       # Shared types and utilities
│   ├── mithril-checkpoint/ # Checkpoint compression
│   ├── mithril-dedup/      # Dataset deduplication
│   ├── mithril-cache/      # Compilation caching
│   └── mithril-python/     # Python bindings (PyO3)
├── scripts/
│   ├── generate_fixtures.py    # Generate test fixtures
│   └── download_hf_fixtures.py # Download HuggingFace models
└── fixtures/               # Test data
```

## API Reference

### mithril.checkpoint

| Class | Description |
|-------|-------------|
| `CompressionConfig` | Configuration for compression (zstd_level, byte_grouping) |
| `CheckpointCompressor` | Compress/decompress checkpoint data |
| `DeltaCompressor` | Delta encoding between checkpoints (10-3500x compression) |
| `DeltaStats` | Statistics from delta compression |
| `save_compressed` | Save FSDP checkpoint with compression (DCP integration) |
| `load_compressed` | Load FSDP checkpoint with decompression |

### mithril.dedup

| Class | Description |
|-------|-------------|
| `DedupConfig` | Configuration (threshold, num_permutations, ngram_size) |
| `Deduplicator` | Deduplicate texts, compute similarity |
| `DedupResult` | Results with keep_indices, remove_indices, clusters, stats |
| `DedupStats` | Statistics (total_documents, unique_documents, duplicate_count) |
| `SemanticConfig` | Semantic dedup config (similarity_threshold, embedding_dim) |
| `SemanticDeduplicator` | Embedding-based deduplication with HNSW index |

### mithril.cache

| Class | Description |
|-------|-------------|
| `CacheConfig` | Cache configuration (root_dir, max_size) |
| `CacheManager` | Manage cache entries with LRU eviction |
| `ContentStore` | Content-addressable storage with Blake3 hashing |
| `VllmCacheConfig` | vLLM kernel cache configuration |
| `VllmCacheManager` | Cache manager for vLLM compiled kernels |
| `SglangCacheConfig` | SGLang kernel cache configuration |
| `SglangCacheManager` | Cache manager for SGLang attention/sampling kernels |

## Testing

```bash
# Run all tests
cargo test --workspace

# Run with HuggingFace fixtures (requires download)
python scripts/download_hf_fixtures.py --all
cargo test --workspace -- --ignored
```

## License

MIT OR Apache-2.0
