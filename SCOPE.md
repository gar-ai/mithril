# Mithril Scope Control

This document defines MVP boundaries and helps avoid scope creep.

## Core Philosophy

> **Ship something small that works, then iterate.**

Every feature has a cost: implementation time, maintenance burden, testing surface, documentation. The MVP should be the smallest thing that delivers value.

## Decision Framework

When considering a feature, ask:

```
1. Does it block the core use case?
   YES → Must have for MVP
   NO  → Continue...

2. Can users work around its absence?
   YES → Post-MVP
   NO  → Continue...

3. Is it technically required by something in MVP?
   YES → Must have for MVP
   NO  → Post-MVP
```

## mithril-checkpoint Scope

### MVP (v0.1) - MUST HAVE

| Feature | Rationale | Integration |
|---------|-----------|-------------|
| bfloat16 byte grouping | Core compression technique | — |
| Zstd compression | Standard, fast, good ratios | — |
| Local storage backend | Proves value without cloud complexity | — |
| Python bindings | Primary user interface | PyO3 |
| PyTorch state_dict support | Most common format | P0 |
| **PyTorch DCP SavePlanner** | Modern checkpoint API, FSDP users | P0 |
| Safetensors read support | HuggingFace models use this | P0 |

### Post-MVP - DEFER

| Feature | Why Defer | When to Add | Priority |
|---------|-----------|-------------|----------|
| ~~S3/GCS storage~~ | ✅ COMPLETED: S3Storage with MinIO support | v0.1 | — |
| Safetensors write support | Read-only enough for MVP | v0.2 | P1 |
| DeepSpeed integration | Their checkpoints already work OK | v0.2 | P1 |
| ~~Delta encoding~~ | ✅ COMPLETED: 10-3500x with XOR delta | v0.1 | — |
| Streaming compression | Memory optimization | v0.2 | P2 |
| JAX/Orbax support | PyTorch first | v0.3 | P2 |
| ~~Lossy quantization~~ | ✅ COMPLETED: INT8/FP16/FP8 with error bounds | v0.1 | — |
| NVMe pipelining | FastPersist-style async | v0.3 | P2 |
| GPU compression | Diminishing returns on CPU | v0.4+ | P3 |
| W&B/MLflow integration | Nice-to-have, not core | v0.4+ | P3 |

### Key Integration Insight

**FSDP is where compression helps most.** Sharded checkpoints across many ranks = massive storage. Target PyTorch DCP users with FSDP first, not DeepSpeed users (they already have decent tooling).

### OUT OF SCOPE - NEVER

| Feature | Why |
|---------|-----|
| Model architecture optimization | Different product |
| Training optimization | Different product |
| Checkpoint diffing UI | Not core value |
| Cloud checkpoint management | Different product |

### MVP Definition of Done

```
[x] Can compress PyTorch checkpoint
[x] Can decompress back to identical bytes
[x] Achieves ≥10x compression on test checkpoints (with delta encoding)
[x] Throughput ≥1 GiB/s (2.0-2.4 GiB/s achieved)
[x] Python API works: compressor.save(state_dict, path)
[x] Python API works: state_dict = compressor.load(path)
[x] Tests pass (400+ tests)
[x] Basic README exists
```

## mithril-dedup Scope

### MVP (v0.1) - MUST HAVE

| Feature | Rationale | Integration |
|---------|-----------|-------------|
| MinHash signatures | Core algorithm | — |
| LSH bucketing | Required for efficiency | — |
| JSON Lines input | Common format | P0 |
| **Parquet input** | Universal ML data format | P0 (arrow-rs) |
| Duplicate detection | Core use case | — |
| CLI interface | Immediate usability | clap |
| Python bindings | Programmatic access | PyO3 |
| **HF Datasets compatibility** | Standard for ML datasets | P0 (format compat) |

### Post-MVP - DEFER

| Feature | Why Defer | When to Add | Priority |
|---------|-----------|-------------|----------|
| S3/GCS input | Local files prove value first | v0.2 | P1 |
| LSHBloom (memory opt) | Standard LSH works for reasonable sizes | v0.2 | P1 |
| Streaming mode | Batch is enough for MVP | v0.2 | P2 |
| Suffix array exact dedup | Different use case | v0.2 | P2 |
| Incremental index | Batch rebuild is fine | v0.3 | P2 |
| ~~Semantic dedup (embeddings)~~ | ✅ COMPLETED: HNSW index, MockBackend, Candle backend | v0.1 | — |
| Ray/Spark connectors | **SKIP** - single-node Rust scales fine | Never | P3 |
| NVIDIA RAPIDS | **SKIP** - GPU overkill for this | Never | P3 |

### Key Integration Insight

**Single-node Rust with rayon can process TB-scale data.** Don't add Spark/Ray complexity. A startup's customers probably don't have Spark clusters anyway. If they do, they can run mithril per-partition.

**HF Datasets "compatibility" means format compatibility**, not library integration. Read/write the same Parquet files with the same schema. Users can:
```python
# Load with HF
dataset = load_dataset("parquet", data_files="data.parquet")
# Dedup with mithril  
mithril_dedup data.parquet -o deduped.parquet
# Load deduped with HF
dataset = load_dataset("parquet", data_files="deduped.parquet")
```

### OUT OF SCOPE - NEVER

| Feature | Why |
|---------|-----|
| Data cleaning/filtering | Different product |
| Text preprocessing | User's responsibility |
| Dataset hosting | Different product |
| Training data attribution | Different product |

### MVP Definition of Done

```
[x] Can process JSON Lines file
[x] Detects exact duplicates (Jaccard = 1.0)
[x] Detects near-duplicates (Jaccard ≥ 0.85)
[x] Throughput ≥50K docs/sec on single core (400-600K achieved)
[x] CLI: mithril-dedup input.jsonl -o output.jsonl
[x] Python: dedup.deduplicate(docs)
[x] Tests pass (400+ tests)
[x] Basic README exists
```

## mithril-cache Scope

### MVP (v0.1) - MUST HAVE

| Feature | Rationale | Integration |
|---------|-----------|-------------|
| Local disk cache | Basic functionality | — |
| Content-addressable storage | Core architecture | — |
| **TorchInductor cache intercept** | Where artifacts live | P0 (SHALLOW) |
| **Triton cache intercept** | Compiled kernels | P0 (env vars) |
| Cache key generation | Required for cache | — |
| LRU eviction | Prevent disk fill | — |
| Python bindings | User interface | PyO3 |

### Post-MVP - DEFER

| Feature | Why Defer | When to Add | Priority |
|---------|-----------|-------------|----------|
| S3/GCS remote cache | Local proves value first | v0.2 | P1 |
| Cross-machine sharing | Needs remote cache | v0.2 | P1 |
| Custom cache key normalization | Start with shallow integration | v0.2 | P1 |
| Version compatibility matrix | Start with strict matching | v0.2 | P2 |
| Deep TorchInductor hooks | APIs are unstable | v0.3+ | P2 |
| Cache warming/prefetch | Optimization | v0.3 | P2 |
| JAX/XLA caching | PyTorch first | v0.3 | P2 |
| Cache analytics/metrics | Nice to have | v0.2 | P2 |
| ~~vLLM/SGLang~~ | ✅ COMPLETED: VllmCacheManager, SglangCacheManager | v0.1 | — |
| Redis distributed cache | Overkill for most users | v0.4+ | P3 |

### Key Integration Insight: Shallow First

**TorchInductor's internal APIs are unstable.** They change between minor PyTorch versions. Strategy:

```
MVP (v0.1):   Intercept cache DIRECTORY, provide better storage layer
              └── Just redirect TORCHINDUCTOR_CACHE_DIR
              └── Just redirect TRITON_CACHE_DIR
              └── Minimal Python patching

v0.2:         Custom cache key normalization (path-independent)
              └── Patch torch._inductor.codecache lightly

v0.3+:        Deep TorchInductor hooks (IF APIs stabilize)
              └── Only if PyTorch team stabilizes these
```

**vLLM/SGLang are inference frameworks** — they compile once, run millions of times. The *acute* pain is during *training iteration* where developers hit cold starts repeatedly. Don't prioritize inference frameworks.

### OUT OF SCOPE - NEVER

| Feature | Why |
|---------|-----|
| Model optimization | Different product |
| Compilation optimization | Different product |
| Cloud cache hosting service | Different business model |
| General-purpose build cache | Focus on ML |

### MVP Definition of Done

```
[x] Local cache stores torch.compile artifacts
[x] Cache hit reduces cold start from 67s to <5s
[x] Cache survives Python process restart
[x] LRU eviction keeps cache under size limit
[x] Python: cache.init(local_dir="...")
[x] Works with torch.compile out of the box
[x] Tests pass (400+ tests)
[x] Basic README exists
```

## Shared Core Scope

### MVP - MUST HAVE

| Component | Rationale |
|-----------|-----------|
| LocalStorage | All products need it |
| ZstdCompressor | Checkpoint needs it |
| XxHash | All products need hashing |
| Basic error types | Clean error handling |
| PyO3 utilities | Python bindings |

### Post-MVP - DEFER

| Component | Why Defer |
|-----------|-----------|
| ~~S3Storage~~ | ✅ COMPLETED: S3Storage with MinIO support |
| GcsStorage | Local is enough for MVP |
| LZ4Compressor | Zstd covers MVP |
| Adaptive compression | Optimization |
| Progress reporting | Nice to have |

## Red Flags: Scope Creep Indicators

Watch for these phrases in discussions:

| Phrase | What It Means | Response |
|--------|---------------|----------|
| "While we're at it..." | Scope creep | "Let's track that for v0.2" |
| "It would be nice if..." | Feature creep | "Does it block MVP?" |
| "Users might want..." | Speculation | "Let's validate with actual users" |
| "It's easy to add..." | Underestimation | "Easy to add != should add" |
| "Competitors have..." | Fear-driven | "What's our core value?" |

## Handling Feature Requests

### Process

1. **Acknowledge**: "Good idea, let me think about it"
2. **Evaluate**: Use the decision framework above
3. **Document**: Add to post-MVP list with rationale
4. **Communicate**: "We'll consider this for v0.2 after we validate MVP"

### Post-MVP Backlog Template

```markdown
## Feature: [Name]

**Requested by**: [Who]
**Request date**: [When]

**Description**: 
[What is it]

**Value**:
[Why would users want it]

**Cost**:
[Rough estimate: Small/Medium/Large]

**Dependencies**:
[What needs to exist first]

**Target version**: v0.X
```

## Version Roadmap

```
v0.1 (MVP) — ✅ COMPLETE
├── Core functionality ✅
├── Local storage + S3 storage ✅
├── PyTorch DCP for checkpoint ✅
├── torch.compile/TorchInductor/Triton cache ✅
├── vLLM/SGLang cache managers ✅
├── Delta encoding (10-3500x compression) ✅
├── Lossy quantization (INT8/FP16/FP8) ✅
├── Semantic dedup with HNSW ✅
├── Parquet + JSONL for dedup ✅
├── 400+ tests passing ✅
└── pip install mithril works ✅

v0.2 — +4 weeks
├── ~~S3/GCS storage (all products)~~ ✅ DONE in v0.1
├── ~~Delta encoding (checkpoint)~~ ✅ DONE in v0.1
├── LSHBloom memory optimization (dedup)
├── Cross-machine cache sharing
├── Safetensors write support
├── DeepSpeed integration (checkpoint)
└── Cache key normalization (cache)

v0.3 — +4 weeks  
├── JAX/Orbax support (checkpoint)
├── Streaming/incremental mode (dedup)
├── Deep TorchInductor hooks (cache, IF stable)
├── Cache analytics/metrics
└── Performance optimization pass

v0.4+
├── ~~Lossy quantization (checkpoint)~~ ✅ DONE in v0.1
├── ~~Semantic dedup with embeddings~~ ✅ DONE in v0.1
├── GPU acceleration (if needed)
├── Based on user feedback
└── ~~vLLM/SGLang (cache)~~ ✅ DONE in v0.1
```

### Integration Timeline (Realistic)

**Checkpoint:**
```
Week 1-3:   Byte grouping + zstd ──► Local storage
Week 4-6:   PyTorch DCP SavePlanner ──► safetensors read
Week 7-10:  S3/GCS backend ──► Delta encoding
Week 11-14: DeepSpeed hooks ──► JAX/Orbax
```

**Dedup:**
```
Week 1-2:   MinHash + LSH ──► JSONL I/O
Week 3-4:   Parquet I/O (arrow-rs) ──► CLI
Week 5-6:   Python bindings ──► Polish
Week 7-10:  S3/GCS input ──► LSHBloom
Week 11+:   Streaming mode ──► Incremental index
```

**Cache:**
```
Week 1-3:   CAS + local storage ──► LRU eviction
Week 4-6:   Env var interception (SHALLOW) ──► Python API
Week 7-10:  S3/GCS remote cache ──► Cross-machine sharing
Week 11-14: Cache key normalization ──► Light patching
Week 15+:   Deep hooks (IF PyTorch stabilizes APIs)
```

**Note:** These timelines assume dedicated focus. With parallel agents, products develop simultaneously but each follows this progression.

## Time Estimates

### MVP Timeline (Per Product)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Core algorithms | 2 weeks | Working Rust code |
| Integration | 1 week | Storage + pipeline |
| Python bindings | 1 week | Working Python API |
| Testing | 1 week | Tests pass, basic benchmarks |
| Documentation | 3 days | README, examples |
| **Total** | **5-6 weeks** | **Shippable MVP** |

### Parallel Development

With three agents:

```
Week 1-2: All agents build core algorithms
Week 3:   All agents integrate with mithril-core
Week 4:   All agents add Python bindings
Week 5:   All agents test and document
Week 6:   Buffer / integration testing
```

## Cutting Scope

When behind schedule, cut in this order:

1. **First to cut**: Nice-to-have features
2. **Second to cut**: Performance optimizations
3. **Third to cut**: Secondary storage backends
4. **Last resort**: Reduce test coverage (document gaps)

**Never cut**:
- Core functionality
- Basic error handling
- Basic documentation

## Questions to Ask Daily

1. Am I working on MVP-critical code?
2. Could this be simpler?
3. Am I gold-plating?
4. What's the minimum that works?
5. Is this blocking something else?

## Anti-Patterns

### Over-Engineering
```rust
// BAD: Premature abstraction
trait CompressionStrategy<T: AsRef<[u8]>, E: Error> {
    type Output: Into<Vec<u8>>;
    fn compress(&self, data: T) -> Result<Self::Output, E>;
}

// GOOD: Simple and direct
pub fn compress(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    zstd::compress(data, 3)
}
```

### Premature Optimization
```rust
// BAD: SIMD before it's needed
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// GOOD: Simple first, optimize later
fn hash(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}
```

### Feature Creep in Comments
```rust
// BAD: TODO that expands scope
// TODO: Add support for S3, GCS, Azure, and custom backends

// GOOD: Focused TODO
// TODO: Add error context for debugging
```
