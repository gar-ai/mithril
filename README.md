# Mithril

**High-performance ML infrastructure toolkit for checkpoint compression, dataset deduplication, and torch.compile caching.**

Built in Rust. Exposed to Python via PyO3. Designed for training pipelines where storage and I/O are the bottleneck.

---

## Why Mithril

Training large models produces enormous checkpoints every N steps. Mithril exploits the structure of floating-point weights -- separating exponent and mantissa bytes before compression, then XOR-encoding consecutive checkpoints to produce sparse delta buffers that compress to near-nothing. The result: **141x compression between training steps** with byte-for-byte lossless roundtrips.

## Key Features

- **Checkpoint compression** -- Byte grouping + zstd pipeline achieving 2.3 GiB/s throughput on bf16 weights
- **Delta encoding** -- XOR between consecutive checkpoints yields 99.3% sparsity and 141x compression ratios
- **Dataset deduplication** -- MinHash/LSH near-duplicate detection at 80K-100K+ docs/sec
- **torch.compile caching** -- Manages TorchInductor and Triton compilation caches with content-addressable storage and LRU eviction
- **Lossless guarantees** -- Every roundtrip is verified byte-for-byte identical
- **Python bindings** -- Full API exposed via PyO3 with `maturin develop`

## Benchmarks

All measurements on Apple M2, zstd level 3.

### Throughput

| Operation | Throughput |
|---|---|
| Delta compress (33M bf16 params) | **2.32 GiB/s** |
| Standalone compress (22M fp32 params) | **546 MiB/s** |
| Decompress (33M bf16 params) | **609 MiB/s** |
| XOR encode (256 MB) | **12.8 GiB/s** |

### Hero Demo -- 256 MB bf16 Synthetic Model (10 Training Steps)

| Step | Naive (torch.save) | Mithril | Compression |
|---|---|---|---|
| Step 1 (standalone) | 256 MB | 197 MB | 1.3x |
| Steps 2-10 (delta) | 256 MB each | ~1.8 MB each | **141-142x** |
| **Total (10 steps)** | **2.68 GB** | **219 MB** | **12.2x overall** |

Delta steps achieve **99.3% storage savings**. All roundtrips byte-for-byte identical.

### Real Model Weights

| Model | Type | Standalone | Delta (0.5% perturbation) |
|---|---|---|---|
| GTE-Small (33M) | Float16 | 1.2-2.2x | 141-142x |
| DINOv2-Small (22M) | Float32 | 1.2-2.2x | 141-142x |
| MiniLM-L6 (22M) | Float32 | 1.2-2.2x | 141-142x |
| BGE-Small-EN (33M) | Float16 | 1.2-2.2x | 141-142x |

### Dataset Deduplication

| Dataset | Throughput | Duplicates Found |
|---|---|---|
| CC News | 80K-100K+ docs/sec | ~15% |
| AG News | 80K-100K+ docs/sec | ~0.8% |

## Quick Start

### Build from Source (Rust)

```bash
git clone https://github.com/your-org/mithril.git
cd mithril
cargo build --release
```

### Python (via maturin)

```bash
cd crates/mithril-python
pip install maturin
maturin develop --release
```

After installation, `import mithril` is available in Python.

## Usage

### Python

```python
import mithril

# --- Checkpoint Compression ---
config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
compressor = mithril.CheckpointCompressor(config)

# Compress raw tensor bytes with dtype-aware byte grouping
compressed = compressor.compress(raw_bytes, "bf16")
decompressed = compressor.decompress(compressed, "bf16", len(raw_bytes))
assert decompressed == raw_bytes  # byte-for-byte identical

# Delta compression between training steps
delta = mithril.DeltaCompressor()
_, stats1 = delta.compress_checkpoint("step_100", checkpoint_bytes)
_, stats2 = delta.compress_checkpoint("step_200", checkpoint_bytes_v2)
print(f"Delta ratio: {stats2.ratio:.0f}x, Sparsity: {stats2.sparsity:.1%}")

# --- Dataset Deduplication ---
config = mithril.DedupConfig(threshold=0.85)
deduplicator = mithril.Deduplicator(config)
result = deduplicator.deduplicate(["doc1", "doc1 copy", "doc2"])
print(f"Keep: {result.keep_indices}")       # [0, 2]
print(f"Duplicates: {result.stats.duplicate_count}")  # 1

# --- Cache Management ---
config = mithril.CacheConfig("/tmp/mithril-cache")
config = config.with_inductor(True).with_triton(True)
manager = mithril.CacheManager(config)
print(manager.stats())
```

### Rust

```rust
use mithril_checkpoint::{CheckpointCompressor, CompressionConfig};
use mithril_checkpoint::delta::DeltaCompressor;
use mithril_dedup::{Deduplicator, DedupConfig};
use mithril_core::types::DType;

// Standalone compression
let config = CompressionConfig::default();
let compressor = CheckpointCompressor::new(config);
let compressed = compressor.compress(&data, DType::BFloat16)?;
let decompressed = compressor.decompress(&compressed, DType::BFloat16, data.len())?;
assert_eq!(data, decompressed);

// Delta compression across training steps
let mut delta = DeltaCompressor::new(CompressionConfig::default());
let (blob, stats) = delta.compress_checkpoint("step_0", &weights)?;
// ... after training step ...
let (blob2, stats2) = delta.compress_checkpoint("step_1", &weights_v2)?;
// stats2.ratio will be ~141x for small weight updates

// Dataset deduplication
let config = DedupConfig::default().with_threshold(0.85);
let dedup = Deduplicator::new(config);
let result = dedup.deduplicate_texts(&["doc1", "doc2", "doc1 copy"]);
println!("Unique: {:?}", result.keep_indices);
```

## Architecture

```
mithril/
  crates/
    mithril-core/         Shared types, compression, hashing, storage
    mithril-checkpoint/   Checkpoint compression pipeline
    mithril-dedup/        Dataset deduplication (MinHash/LSH)
    mithril-cache/        torch.compile cache management
    mithril-python/       PyO3 Python bindings
  examples/
    demo.rs               Hero demo: synthetic 256MB model, 10 training steps
    real_checkpoint_demo.rs   Compression on real HuggingFace safetensors
    real_dedup_demo.rs        Dedup on real HF datasets (AG News, CC News, etc.)
```

### The Core Insight: Byte Grouping

Neural network weights in bf16/fp16/fp32 formats store bytes interleaved: `[high0, low0, high1, low1, ...]` where the high byte contains the exponent and the low byte contains the mantissa. Exponents in trained models cluster tightly (most weights are small values near zero), but interleaving destroys that locality.

Mithril's byte grouping separates the streams: `[high0, high1, ..., low0, low1, ...]`. Now the exponent bytes form a highly repetitive sequence that zstd compresses aggressively. For 4-byte fp32 values, the same principle applies across all four byte lanes.

Combined with delta encoding (XOR between consecutive checkpoints), unchanged bytes become zeros. A checkpoint where 0.5% of parameters changed becomes >99% zeros after XOR -- and those zeros compress to almost nothing.

### Performance Details

- SIMD auto-vectorization via `chunks_exact` patterns for byte grouping and XOR operations
- Parallel processing with rayon for data above 1 MB
- Zero-copy `Cow<[u8]>` paths for dtypes that skip byte grouping (u8, i8, bool)
- Blake3 content-addressable hashing for cache deduplication
- 484+ tests across the workspace including property-based tests with proptest

## Running the Demos

```bash
# Hero demo: 256MB synthetic model, 10 training steps
# Shows side-by-side comparison of naive vs Mithril compression
cargo run --release --example demo

# Real model weights: compress actual HuggingFace safetensors
# Requires: python scripts/download_hf_fixtures.py --all
cargo run --release --example real_checkpoint_demo

# Real dataset dedup: MinHash/LSH on AG News, CC News, etc.
# Requires: python scripts/download_hf_fixtures.py --all
cargo run --release --example real_dedup_demo
```

## Project Structure

| Crate | Description |
|---|---|
| `mithril-core` | Shared infrastructure: types, compression backends (zstd, lz4), hashing (xxhash3, blake3), cloud storage (S3/GCS) |
| `mithril-checkpoint` | Checkpoint compression pipeline: byte grouping, delta encoding, safetensors/ONNX/GGUF format support, CLI |
| `mithril-dedup` | Dataset deduplication: MinHash, LSH, Bloom filters, semantic dedup (HNSW + optional Candle embeddings), Parquet/JSONL I/O |
| `mithril-cache` | Compilation cache management: TorchInductor/Triton hooks, content-addressable storage, LRU eviction, vLLM/SGLang kernel caches |
| `mithril-python` | PyO3 bindings exposing the full API to Python |

## Testing

```bash
# Run all tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace --exclude mithril-python

# Run with real HuggingFace fixtures (downloads models and datasets)
python scripts/download_hf_fixtures.py --all
cargo test --workspace -- --ignored
```

## License

MIT OR Apache-2.0
