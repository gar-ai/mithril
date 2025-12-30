# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-27

### Added

#### mithril-checkpoint
- Lossless bf16/fp16 checkpoint compression at 2+ GiB/s throughput
- Delta encoding between training steps achieving 10-3500x compression ratios
- Lossy quantization support (INT8, FP16, FP8) with configurable error bounds
- PyTorch Distributed Checkpoint (DCP) integration for FSDP workloads
- S3-compatible storage backend (AWS S3, MinIO, etc.)
- Safetensors read support for HuggingFace model loading
- Byte grouping optimization for better compression of floating-point data

#### mithril-dedup
- MinHash/LSH near-duplicate detection at 400-600K documents/second
- Semantic deduplication with HNSW index for embedding-based similarity
- Optional Candle backend for native Rust embeddings (no Python required)
- Parquet input/output with HuggingFace Datasets compatibility
- JSON Lines (JSONL) input/output support
- CLI tool: `mithril-dedup --input data.jsonl --output deduped.jsonl`
- Configurable Jaccard similarity threshold (default 0.85)

#### mithril-cache
- TorchInductor cache directory management for torch.compile
- Triton kernel cache interception via environment variables
- vLLM kernel cache manager (`VllmCacheManager`)
- SGLang kernel cache manager (`SglangCacheManager`)
- Content-addressable storage with Blake3 hashing
- LRU eviction to prevent unbounded cache growth
- S3-compatible remote cache backend

#### mithril-python
- Python bindings for all modules via PyO3
- `CheckpointCompressor` for compress/decompress operations
- `DeltaCompressor` for delta encoding between checkpoints
- `Deduplicator` for text deduplication with MinHash/LSH
- `SemanticDeduplicator` for embedding-based deduplication
- `CacheManager` for torch.compile cache management
- `ContentStore` for content-addressable storage
- `save_compressed()` / `load_compressed()` for DCP integration
- Lazy loading for optional torch dependency

#### mithril-core
- Shared `LocalStorage` backend for all crates
- Zstd compression with configurable levels
- XxHash for fast hashing
- S3Storage backend with async operations
- Common error types and result handling

### Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Checkpoint compress | Throughput | 2.0-2.4 GiB/s |
| Checkpoint decompress | Throughput | 2.0 GiB/s |
| Delta encoding | Compression ratio | 10-3500x |
| Deduplication | Throughput | 400-600K docs/sec |
| Content store | Write speed | 70+ MB/s |
| Cache key generation | Rate | 4M keys/sec |

### Technical Notes

- Minimum Rust version: 1.75
- Python support: 3.9+
- License: MIT OR Apache-2.0
