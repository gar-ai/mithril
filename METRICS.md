# Mithril Success Metrics

This document defines how we measure success for each product.

## Core Principles

1. **Measurable**: Every metric has a specific number or threshold
2. **Comparable**: Metrics compare against baselines (existing tools, no tool)
3. **User-focused**: Metrics reflect what users care about
4. **Automated**: Metrics are collected automatically in CI

## mithril-checkpoint Metrics

### Primary Metrics

| Metric | Target | Achieved | How to Measure |
|--------|--------|----------|----------------|
| **Compression Ratio (standalone)** | 1.3-1.5x (lossless) | ✅ 1.4x | `original_size / compressed_size` |
| **Compression Ratio (with delta)** | 10-20x (lossless) | ✅ 100x+ | With previous checkpoint reference |
| **Delta Compression Ratio** | 39-70x | ✅ 103-104x | XOR + byte grouping + zstd |
| **Compression Throughput** | ≥2.5 GiB/s | 1.6-2.1 GiB/s | Bytes/second (Apple M2) |
| **Decompression Throughput** | ≥3.5 GiB/s | 2.1 GiB/s | Bytes/second (Apple M2) |
| **Memory Overhead** | ≤2x checkpoint size | ✅ | Peak RSS during compress |

> **Note**: Delta encoding (v0.2) achieves 100x+ compression on consecutive training checkpoints
> where ~1% of weights change per step. Standalone compression of random/pre-trained models
> achieves ~1.4x. Throughput targets are for 16-core CPU; Apple M2 results shown above.

### Real Model Validation (GPT-2, 548MB)

| Test | Result | Target | Status |
|------|--------|--------|--------|
| **Standalone (pre-trained)** | 1.28x | 1.3-1.5x | ~✅ |
| **Delta (simulated fine-tuning)** | 389x | 39-70x | ✅ (5.5x better) |
| **Roundtrip** | Bit-exact | Bit-exact | ✅ |

> Run with: `python scripts/test_hf_models.py`

### Comprehensive Model Validation (Stress Tests)

Tested across diverse model architectures to validate compression robustness:

| Category | Model | Size | Standalone | Delta | Dtype |
|----------|-------|------|------------|-------|-------|
| **Language** | GPT-2 | 548 MB | 1.28x | 390x | fp32 |
| **Language** | Qwen3-0.6B | 1.5 GB | 1.50x | 649x | bf16 |
| **Language** | TinyLlama-1.1B | 2.2 GB | 1.50x | 556x | bf16 |
| **Vision** | ViT-Base | 346 MB | 1.17x | 386x | fp32 |
| **Vision** | DINOv2-Small | 88 MB | 1.16x | 336x | fp32 |
| **Embedding** | MiniLM-L6-v2 | 91 MB | 1.18x | 376x | fp32 |
| **Embedding** | BGE-Small-EN | 133 MB | 2.14x | 375x | fp32 |
| **Embedding** | GTE-Small | 67 MB | 1.17x | 491x | fp32 |
| **Average** | - | - | **1.39x** | **445x** | - |

**Key Findings:**
- bf16 models (Qwen3, TinyLlama) achieve better standalone compression (1.50x) due to byte grouping
- Delta compression exceeds targets by 6-10x across all architectures (336-649x vs 39-70x target)
- All roundtrips are bit-exact
- Vision models compress slightly less (1.16-1.17x) due to more random weight distributions

**Edge Cases Tested:**
- Very small tensors (<1KB): Works correctly
- Large tensors (~100MB): 1.17x compression
- All-zeros tensor: 28,777x compression (best case)
- Random data: 1.17x compression (worst case)
- Delta chain (10 consecutive): Bit-exact roundtrips

> Run with: `python scripts/stress_test.py`

### Secondary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Accuracy Preservation** | Bit-exact (lossless) | Checksum match after roundtrip |
| **Integration Overhead** | <5% training time | With DCP/Orbax hooks |
| **Cold Start Time** | <100ms | Time to initialize compressor |

### Benchmarks

```bash
# Run checkpoint benchmarks
cargo bench -p mithril-checkpoint

# Output format
checkpoint/compress/100MB    time: [45.2 ms 45.8 ms 46.4 ms]
                             thrpt: [2.16 GiB/s 2.18 GiB/s 2.21 GiB/s]

checkpoint/decompress/100MB  time: [28.1 ms 28.5 ms 28.9 ms]
                             thrpt: [3.46 GiB/s 3.51 GiB/s 3.56 GiB/s]
```

### Comparison Baselines

| Baseline | What It Tells Us |
|----------|------------------|
| **Raw (no compression)** | Minimum improvement threshold |
| **gzip -9** | Common baseline, slow |
| **zstd -3** | Good baseline, fast |
| **LMC paper results** | Academic state-of-art |

## mithril-dedup Metrics

### Primary Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Throughput** | ≥100K docs/sec | Documents processed per second |
| **Memory (LSH)** | <16GB for 1B docs | Peak RSS |
| **Memory (LSHBloom)** | <1GB for 1B docs | Peak RSS |
| **Precision** | ≥0.95 | True duplicates / reported duplicates |
| **Recall** | ≥0.90 | Found duplicates / actual duplicates |

### Validation Results (Real HuggingFace Datasets)

Tested on 7 real datasets from HuggingFace with manually verified duplicates:

| Dataset | Docs | Duplicates | Ratio | Throughput | Status |
|---------|------|------------|-------|------------|--------|
| **CC News** | 10,000 | 1,490 | 14.9% | 68K docs/sec | ✅ Verified |
| **AG News** | 5,000 | 39 | 0.78% | 352K docs/sec | ✅ Verified |
| **IMDB** | 5,000 | 9 | 0.18% | 105K docs/sec | ✅ Verified |
| **Amazon Polarity** | 10,000 | 4 | 0.04% | 225K docs/sec | ✅ Verified |
| **Wikitext** | 2,183 | 0 | 0.00% | 181K docs/sec | ✅ Clean |
| **Rotten Tomatoes** | 4,451 | 0 | 0.00% | 506K docs/sec | ✅ Clean |
| **Tweet Eval** | 4,859 | 0 | 0.00% | 551K docs/sec | ✅ Clean |

**Key Findings:**

- CC News (news syndication) has 14.9% duplicates - manually verified as real near-duplicates
- Throughput varies by duplicate density: more candidates = more verification = slower
- Clean datasets (Wikipedia, reviews) show 0% duplicates as expected
- Average throughput: **284K docs/sec** (2.8x above 100K target)

**Duplicate Verification:**

- Similarity scores 0.97-1.00 on verified pairs
- Exact duplicates (1.00) found in CC News
- Near-duplicates (0.97-0.98) are same stories with minor edits

> Run with: `./target/release/mithril-dedup <input.jsonl> -o <output.jsonl>`

### Python Bindings Performance

| Dataset | Throughput | vs CLI |
|---------|------------|--------|
| Tweet Eval | 20K docs/sec | 28x slower |
| AG News | 13K docs/sec | 27x slower |
| IMDB | 2.5K docs/sec | 42x slower |
| **Average** | **12K docs/sec** | ~25x overhead |

> Note: Python bindings have overhead due to PyO3 data copying. For production workloads, use the Rust CLI.

### Secondary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Jaccard Correlation** | ≥0.95 | MinHash estimate vs exact |
| **Index Build Time** | <1 hour for 1B docs | One-time cost |
| **Query Latency** | <10ms | Single document lookup |

### Benchmarks

```bash
# Run dedup benchmarks
cargo bench -p mithril-dedup

# Output format
dedup/minhash/10K_docs       time: [12.3 ms 12.5 ms 12.7 ms]
                             thrpt: [787K docs/s 800K docs/s 813K docs/s]

dedup/full_pipeline/100K     time: [1.23 s 1.25 s 1.27 s]
                             thrpt: [78.7K docs/s 80.0K docs/s 81.3K docs/s]
```

### Comparison Baselines

| Baseline | What It Tells Us |
|----------|------------------|
| **text-dedup** | Current Python standard |
| **NeMo Curator** | GPU-accelerated alternative |
| **Exact comparison** | Ground truth (O(n²), unusable at scale) |

### Test Datasets

| Dataset | Size | Known Duplicates | Purpose |
|---------|------|------------------|---------|
| **Synthetic-1K** | 1K docs | 30% exact | Unit tests |
| **Synthetic-100K** | 100K docs | 30% near | Integration tests |
| **C4-sample** | 1M docs | Unknown | Real-world benchmark |

## mithril-cache Metrics

### Primary Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Cold Start (cache hit)** | <1s | Time from import to compiled model |
| **Cache Hit Rate** | ≥80% | Hits / (Hits + Misses) over time |
| **Local Lookup Latency** | <10ms | Time to check local cache |
| **Remote Lookup Latency** | <500ms | Time to fetch from S3/GCS |

### Secondary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Cache Size Efficiency** | <2x artifact size | Storage overhead |
| **Cross-machine Hit Rate** | ≥60% | Hits when artifact from different machine |
| **Compilation Fallback Time** | <5% overhead | When cache miss triggers recompile |

### Benchmarks

```bash
# Run cache benchmarks
cargo bench -p mithril-cache

# Output format
cache/local_lookup           time: [1.23 ms 1.25 ms 1.27 ms]
cache/remote_lookup          time: [245 ms 250 ms 255 ms]
cache/key_generation         time: [12.3 µs 12.5 µs 12.7 µs]
```

### Comparison Baselines

| Baseline | What It Tells Us |
|----------|------------------|
| **torch.compile cold** | 67s (H100) - what we're fixing |
| **torch.compile warm (local)** | Current best case |
| **No caching** | Worst case |

### Real-World Tests

```python
# Test script for real metrics
import time
import torch
import mithril.cache

def measure_cold_start():
    # Fresh Python process
    start = time.time()
    model = torch.compile(MyModel())
    model(dummy_input)  # Trigger compilation
    return time.time() - start

# Run with: python -c "from test_metrics import *; print(measure_cold_start())"
```

## Automated Metric Collection

### CI Integration

```yaml
# .github/workflows/metrics.yml
name: Collect Metrics

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmarks
        run: cargo bench --workspace -- --save-baseline ${{ github.sha }}
      
      - name: Upload metrics
        uses: actions/upload-artifact@v3
        with:
          name: metrics-${{ github.sha }}
          path: target/criterion/
      
      - name: Post to metrics dashboard
        run: ./scripts/post_metrics.sh
```

### Metrics Dashboard

Track metrics over time:

```
┌─────────────────────────────────────────────────────────────┐
│                  Checkpoint Compression                      │
├─────────────────────────────────────────────────────────────┤
│  Throughput (GiB/s)                                         │
│  3.5 ┤                                    ╭──────           │
│  3.0 ┤                          ╭─────────╯                 │
│  2.5 ┤           ╭──────────────╯                           │
│  2.0 ┤───────────╯                                          │
│      └──────────────────────────────────────────────────    │
│        v0.1    v0.2    v0.3    v0.4    v0.5                │
└─────────────────────────────────────────────────────────────┘
```

## Definition of Done

A product is "done" when:

### MVP (v0.1)

- [ ] All primary metrics meet targets
- [ ] Unit test coverage ≥70%
- [ ] Integration tests pass
- [ ] Python bindings work
- [ ] Basic documentation exists

### Production Ready (v1.0)

- [ ] All metrics meet targets consistently (3 consecutive runs)
- [ ] Benchmarks automated in CI
- [ ] Performance regression tests
- [ ] Real-world validation (one external user)
- [ ] Complete documentation

## Metric Definitions

### Throughput

```
Throughput = Bytes Processed / Wall Clock Time

Measured as:
- Median of 10 runs
- After 3 warmup runs
- Single-threaded unless specified
```

### Compression Ratio

```
Compression Ratio = Original Size / Compressed Size

Higher is better.
Measured on standardized test data.
```

### Latency

```
Latency = Time from request to response

Measured as:
- P50 (median)
- P99 (tail latency)
- Max observed
```

### Memory

```
Memory = Peak RSS during operation

Measured using:
- /proc/self/status on Linux
- getrusage() on macOS
```

## Regression Detection

### Thresholds

| Metric Type | Regression Threshold |
|-------------|---------------------|
| Throughput | >10% decrease |
| Latency | >20% increase |
| Memory | >25% increase |

### Alerts

```yaml
# In CI, fail if regression detected
- name: Check for regressions
  run: |
    cargo bench -- --baseline main --load-baseline pr
    if grep -q "regressed" benchmark_results.txt; then
      echo "Performance regression detected!"
      exit 1
    fi
```

## User-Facing Metrics

What users see in logs/output:

```
mithril-checkpoint: Compressed 329 GiB → 8.4 GiB (39.2x) in 142s (2.32 GiB/s)
mithril-dedup: Processed 1.4M docs in 14.2s (98.6K docs/s), found 412K duplicates (29.4%)
mithril-cache: Cache hit! Loaded in 0.8s (saved ~67s compilation time)
```
