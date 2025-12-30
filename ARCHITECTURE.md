# Mithril Architecture

Mithril is a Rust-based ML infrastructure suite comprising three products that share a common core. This document defines the system architecture that enables three parallel development agents to build without collision.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                            MITHRIL                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │    CHECKPOINT  │  │     DEDUP      │  │     CACHE      │        │
│  │                │  │                │  │                │        │
│  │  Checkpoint    │  │     Data       │  │  Compilation   │        │
│  │  Compression   │  │ Deduplication  │  │    Caching     │        │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │
│          │                   │                   │                  │
│          └───────────────────┼───────────────────┘                  │
│                              │                                      │
│                   ┌──────────▼──────────┐                           │
│                   │    MITHRIL-CORE     │                           │
│                   │                     │                           │
│                   │  • Storage Layer    │                           │
│                   │  • Compression      │                           │
│                   │  • Hashing/CAS      │                           │
│                   │  • PyO3 Bindings    │                           │
│                   │  • Async Runtime    │                           │
│                   └─────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mithril/
├── Cargo.toml                    # Workspace root
├── README.md
├── docs/
│   ├── ARCHITECTURE.md           # This document
│   ├── INTERFACES.md             # Core API contracts
│   ├── STYLE_GUIDE.md            # Code patterns for all agents
│   ├── TESTING.md                # Testing strategy
│   ├── METRICS.md                # Success criteria
│   ├── SCOPE.md                  # MVP boundaries
│   ├── RESEARCH.md               # Papers and prior art
│   ├── checkpoint/
│   │   └── SPEC.md               # Checkpoint compression spec
│   ├── dedup/
│   │   └── SPEC.md               # Deduplication spec
│   └── cache/
│       └── SPEC.md               # Compilation caching spec
├── crates/
│   ├── mithril-core/             # Shared infrastructure (BUILD FIRST)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── storage/          # Storage abstraction layer
│   │       │   ├── mod.rs
│   │       │   ├── local.rs
│   │       │   ├── s3.rs
│   │       │   └── gcs.rs
│   │       ├── compression/      # Compression primitives
│   │       │   ├── mod.rs
│   │       │   ├── zstd.rs
│   │       │   ├── lz4.rs
│   │       │   └── huffman.rs
│   │       ├── hashing/          # Hashing and CAS
│   │       │   ├── mod.rs
│   │       │   ├── xxhash.rs
│   │       │   ├── content_address.rs
│   │       │   └── minhash.rs
│   │       ├── async_runtime/    # Tokio utilities
│   │       │   └── mod.rs
│   │       └── types/            # Shared types
│   │           └── mod.rs
│   │
│   ├── mithril-checkpoint/       # Checkpoint compression (Agent 1)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── delta/            # Delta encoding
│   │       ├── quantize/         # Adaptive quantization
│   │       ├── bytegroup/        # bfloat16 byte grouping
│   │       └── pipeline/         # Compression pipeline
│   │
│   ├── mithril-dedup/            # Deduplication (Agent 2)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── minhash/          # MinHash implementation
│   │       ├── lsh/              # Locality-sensitive hashing
│   │       ├── suffix/           # Suffix array dedup
│   │       └── cluster/          # Connected components
│   │
│   ├── mithril-cache/            # Compilation caching (Agent 3)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cas/              # Content-addressable storage
│   │       ├── keys/             # Cache key generation
│   │       ├── artifacts/        # Artifact management
│   │       └── hooks/            # Framework hooks
│   │
│   └── mithril-python/           # Unified Python bindings
│       ├── Cargo.toml
│       ├── pyproject.toml
│       └── src/
│           ├── lib.rs
│           ├── checkpoint.rs
│           ├── dedup.rs
│           └── cache.rs
│
├── python/
│   └── mithril/                  # Python package
│       ├── __init__.py
│       ├── checkpoint/
│       ├── dedup/
│       └── cache/
│
├── tests/
│   ├── integration/
│   └── benchmarks/
│
└── examples/
    ├── checkpoint/
    ├── dedup/
    └── cache/
```

## Development Phases

### Phase 0: Core Foundation (Week 1-2)
**Single agent or coordinated effort**

The shared core MUST be built first. Products depend on these interfaces being stable.

```
mithril-core provides:
├── StorageBackend trait          # All products need storage
├── Compressor trait              # Checkpoint primary, others secondary
├── ContentAddressable trait      # Cache primary, others use
├── HashFunction trait            # Dedup primary, others use
├── AsyncHandle utilities         # All products need async
└── Error types                   # Unified error handling
```

### Phase 1: Parallel Product Development (Week 3+)
**Three agents working independently**

Each agent owns their crate completely. Communication happens through:
1. The defined interfaces in `INTERFACES.md`
2. The shared core library
3. No direct cross-product dependencies

```
Agent 1 (Checkpoint)     Agent 2 (Dedup)          Agent 3 (Cache)
     │                        │                        │
     ▼                        ▼                        ▼
mithril-checkpoint/      mithril-dedup/           mithril-cache/
     │                        │                        │
     └────────────────────────┼────────────────────────┘
                              │
                              ▼
                        mithril-core/
                        (STABLE after Phase 0)
```

### Phase 2: Python Integration (Week 6+)
**Can begin once any product has stable Rust API**

The `mithril-python` crate provides unified bindings.

## Agent Boundaries

### What Agents CAN Do
- Modify anything within their assigned crate
- Add new modules/files within their crate
- Add dependencies to their crate's Cargo.toml
- Create integration tests in `tests/integration/{product}/`
- Create examples in `examples/{product}/`

### What Agents CANNOT Do
- Modify `mithril-core/` after Phase 0 (propose changes via docs)
- Modify another agent's crate
- Add cross-product dependencies (checkpoint cannot depend on dedup)
- Modify shared configuration without coordination

### Requesting Core Changes
If an agent needs core functionality:
1. Document the need in `docs/CORE_REQUESTS.md`
2. Specify the trait/function signature needed
3. Wait for core update before depending on it

## Dependency Graph

```
mithril-python
     │
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
mithril-checkpoint  mithril-dedup  mithril-cache
     │              │              │
     └──────────────┼──────────────┘
                    ▼
              mithril-core
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  tokio          object_store   pyo3
     │              │              
     ▼              ▼              
  rayon          zstd/lz4        
```

## Key Design Decisions

### 1. Workspace Structure
Using Cargo workspaces allows independent compilation while sharing dependencies. Each product is a separate crate to enforce boundaries.

### 2. Trait-Based Abstraction
Core functionality exposed via traits enables:
- Mocking for tests
- Alternative implementations
- Clear contracts between components

### 3. Async-First
All I/O operations are async (tokio). CPU-bound work uses rayon for parallelism. This matches ML workload patterns (large I/O, parallel computation).

### 4. Python as Primary Interface
While the core is Rust, users interact via Python. The Rust API is internal; Python API is the product. Design Rust APIs to be PyO3-friendly.

### 5. Storage Abstraction
All products use the same storage abstraction (`StorageBackend` trait). This enables:
- Local development with filesystem
- Production deployment with S3/GCS
- Testing with in-memory storage

## Build Commands

```bash
# Build everything
cargo build --workspace

# Build specific product
cargo build -p mithril-checkpoint
cargo build -p mithril-dedup
cargo build -p mithril-cache

# Run tests for specific product
cargo test -p mithril-checkpoint
cargo test -p mithril-dedup
cargo test -p mithril-cache

# Build Python package
cd crates/mithril-python
maturin develop

# Run benchmarks
cargo bench -p mithril-checkpoint
```

## Configuration

Each product has its own configuration, but format is consistent:

```toml
# mithril.toml (example)
[core]
storage_backend = "local"  # or "s3", "gcs"
temp_dir = "/tmp/mithril"

[checkpoint]
compression_level = 3
delta_enabled = true

[dedup]
algorithm = "minhash"
threshold = 0.85

[cache]
max_size_gb = 100
eviction_policy = "lru"
```

## Error Handling

All products use a unified error type from core:

```rust
// mithril-core/src/error.rs
#[derive(Error, Debug)]
pub enum MithrilError {
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Compression error: {0}")]
    Compression(#[from] CompressionError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    // Product-specific errors wrap their own types
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
    
    #[error("Dedup error: {0}")]
    Dedup(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
}
```

## Logging and Telemetry

Use `tracing` crate for structured logging:

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(data))]
pub async fn compress_checkpoint(data: &[u8]) -> Result<Vec<u8>> {
    info!(size = data.len(), "Starting compression");
    // ...
}
```

## Ecosystem Integrations

### Shared Infrastructure (All Products)

All three products share common Rust infrastructure:

| Component | Crate/Approach | Notes |
|-----------|---------------|-------|
| Python bindings | PyO3 | Mature, well-documented |
| Cloud storage | object_store | Apache Arrow project, production-ready |
| Async I/O | tokio + rayon | Async for I/O, rayon for CPU parallelism |
| Data interchange | Arrow | Zero-copy with Python |
| Tensor format | safetensors | Increasingly standard, fast |

### Integration Priority Framework

**Tier 1: Framework Lock-In (Must Have)**
These determine whether customers can adopt without rewriting their stack:

| Product | Critical Integration | Risk if Missing |
|---------|---------------------|-----------------|
| Checkpoint | PyTorch DCP | 🔴 Cannot serve PyTorch users (90%+ market) |
| Cache | TorchInductor + Triton | 🔴 torch.compile is the only target |
| Dedup | HF Datasets + Parquet | 🟡 Friction kills adoption |

**Tier 2: Storage (High Priority, Post-MVP)**
Required for production but not for validation:

| Integration | Priority | Notes |
|-------------|----------|-------|
| Local filesystem | P0 (MVP) | Proves value fast |
| S3/GCS | P1 (v0.2) | Production deployment |
| Safetensors | P1 (v0.2) | HuggingFace ecosystem |

**Tier 3: Ecosystem Expansion (Post-MVP)**

| Product | Integration | Target Segment | Priority |
|---------|-------------|----------------|----------|
| Checkpoint | DeepSpeed | Large-scale training | P1 |
| Checkpoint | JAX/Orbax | Google/TPU users | P2 |
| Cache | vLLM/SGLang | Inference (less pain) | P3 |
| Dedup | Ray/Spark | Enterprise (overkill) | P3 |

### Integration Complexity Reality Check

```
PyTorch DCP ──────► PyO3 bindings to SavePlanner API
                    └── MEDIUM: Well-documented, stable API

DeepSpeed ────────► Checkpoint callback hooks
                    └── LOW: Callback-based, straightforward

TorchInductor ────► Deep Python internals, UNDOCUMENTED
                    └── HIGH: APIs change between minor versions
                    └── Strategy: Shallow integration first

Triton Cache ─────► File-based, environment variables
                    └── MEDIUM: Binary portability issues

JAX/Orbax ────────► Custom CheckpointHandler
                    └── HIGH: Different paradigm (functional)
```

### What We're NOT Integrating (And Why)

| Integration | Why Skip |
|-------------|----------|
| Spark/Ray | Single-node Rust + rayon handles TB-scale. Distributed adds massive complexity. |
| NVIDIA RAPIDS | GPU acceleration for dataframes is overkill for dedup |
| LangChain/LlamaIndex | Different use case (RAG), not training data |
| W&B/MLflow | Nice-to-have, not core value |

## Next Steps

1. Read `INTERFACES.md` for core API contracts
2. Read `STYLE_GUIDE.md` for coding patterns
3. Read your product's `SPEC.md` for implementation details
4. Read `SCOPE.md` to understand MVP boundaries
5. Start with `mithril-core` if assigned, or wait for it to stabilize
