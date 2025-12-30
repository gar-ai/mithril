# Mithril Research References

Papers, repos, and prior art informing each product.

## mithril-checkpoint

### Papers

| Paper | Key Insight | Link |
|-------|-------------|------|
| **ExCP (ICML 2024)** | Weight-momentum joint compression, 70x ratio | [arXiv](https://arxiv.org/abs/2406.01234) |
| **LMC (May 2025)** | bfloat16 byte grouping, 2.78 GiB/s | [arXiv](https://arxiv.org/abs/2505.xxxxx) |
| **ZipNN (Nov 2024)** | Exploits FP exponent skewness, 33-50% lossless | [arXiv](https://arxiv.org/abs/2411.05239) |
| **ImPart (April 2025)** | SVD importance-aware delta sparsification, 2× compression | [arXiv](https://arxiv.org/abs/2504.13237) |
| **Delta-CoMe (June 2024)** | Mixed-precision delta quantization by singular value | [arXiv](https://arxiv.org/abs/2406.08903) |
| **DynaQuant** | Gradient-based sensitivity for quantization | [Paper](https://example.com) |
| **Check-N-Run (NSDI 2022)** | Meta's differential compression | [USENIX](https://www.usenix.org/conference/nsdi22) |

### Key Techniques

**bfloat16 Byte Grouping (from LMC)**
```
Before: [h0, l0, h1, l1, h2, l2, ...]  (interleaved)
After:  [h0, h1, h2, ..., l0, l1, l2, ...]  (grouped)

High bytes (exponent) compress better together.
Low bytes (mantissa) compress better together.
Result: ~20% better compression ratio.
```

**ZipNN Exponent Skewness (v0.2)**
```
Neural network weights have NON-UNIFORM exponent distributions.
General compressors (zstd) don't exploit this.
ZipNN codes exponents separately → 33-50% better compression.
80GB/s decompression speed.
```

**ImPart Importance-Aware Delta (v0.2)**
```
Not all weight deltas matter equally.
SVD decomposition: delta = U @ S @ V^T
Top singular vectors → high precision (8-bit)
Tail singular vectors → low precision (2-bit)
Result: 2× compression vs uniform sparsity.
```

**Delta Encoding (from Check-N-Run)**
```
checkpoint_n = checkpoint_{n-1} XOR delta_n

Consecutive checkpoints are similar.
XOR produces mostly zeros → compresses extremely well.
After 17+ steps: 39-70x compression possible.
```

### Integration References

| Reference | What to Learn | Link |
|-----------|---------------|------|
| **PyTorch DCP Source** | SavePlanner/LoadPlanner API | [GitHub](https://github.com/pytorch/pytorch/tree/main/torch/distributed/checkpoint) |
| **FSDP Checkpoint Tutorial** | How FSDP shards checkpoints | [PyTorch Docs](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) |
| **safetensors** | Fast, safe tensor format | [GitHub](https://github.com/huggingface/safetensors) |
| **DeepSpeed Checkpointing** | Their optimization approach | [DeepSpeed Docs](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) |

### Repos

| Repo | What It Does | Link |
|------|--------------|------|
| **torch.distributed.checkpoint** | PyTorch DCP, our integration target | [GitHub](https://github.com/pytorch/pytorch) |
| **safetensors** | HuggingFace tensor format | [GitHub](https://github.com/huggingface/safetensors) |
| **orbax** | JAX checkpointing | [GitHub](https://github.com/google/orbax) |
| **DeepSpeed** | Includes checkpoint utilities | [GitHub](https://github.com/microsoft/DeepSpeed) |

## mithril-dedup

### Papers

| Paper | Key Insight | Link |
|-------|-------------|------|
| **Deduplicating Training Data** (Google 2021) | 3.5x memorization reduction | [arXiv](https://arxiv.org/abs/2107.06499) |
| **SemDeDup** (Meta 2023) | Semantic dedup via embeddings | [arXiv](https://arxiv.org/abs/2303.09540) |
| **LSHBloom (2024)** | 54x memory reduction for LSH | [Paper](https://example.com) |
| **FED (2024)** | 107x speedup for dedup | [Paper](https://example.com) |
| **D4** | Data quality via dedup | [arXiv](https://arxiv.org/abs/2308.12284) |

### Key Techniques

**MinHash Signature**
```
For each hash function h_i:
  sig[i] = min(h_i(token) for token in document)

Jaccard(A, B) ≈ count(sig_A[i] == sig_B[i]) / num_hashes

128-256 permutations typical.
3-5 gram shingling typical.
```

**LSH Bucketing**
```
Split signature into b bands of r rows each.
Hash each band → bucket.
Documents in same bucket = candidate duplicates.

Threshold ≈ (1/b)^(1/r)
For 0.85 threshold: b=20, r=5 works well.
```

**LSHBloom Optimization**
```
Replace HashMap<BandHash, Vec<DocId>> with BloomFilter.

Memory: 54x reduction (paper claims).
Tradeoff: False positives require verification pass.
```

### Repos

| Repo | What It Does | Link |
|------|--------------|------|
| **text-dedup** | Python reference implementation | [GitHub](https://github.com/ChenghaoMou/text-dedup) |
| **deduplicate-text-datasets** | Google's suffix array impl | [GitHub](https://github.com/google-research/deduplicate-text-datasets) |
| **NeMo Curator** | NVIDIA's GPU dedup | [GitHub](https://github.com/NVIDIA/NeMo-Curator) |
| **datatrove** | HuggingFace data processing | [GitHub](https://github.com/huggingface/datatrove) |

### Datasets for Testing

| Dataset | Size | Notes |
|---------|------|-------|
| **C4** | 300GB+ | Known duplicates, good benchmark |
| **The Pile** | 800GB | Diverse sources |
| **RedPajama** | 1.2T tokens | Recent, well-documented |

## mithril-cache

### Background

**The Problem**
```
torch.compile cold start times:
- Diffusers on H100: 67 seconds
- Large models: 17-30 minutes (triggers NCCL timeout)
- Warm cache often doesn't transfer between machines
```

**Why Caches Fail**
```python
# PyTorch cache key includes:
cache_key = hash(
    source_code_hash,  # Includes file paths! 
    torch_version,
    cuda_version,
    gpu_name,
    ...
)

# Different machine = different paths = cache miss
```

### Papers/Docs

| Resource | Key Insight | Link |
|----------|-------------|------|
| **PyTorch RFC: Mega-Cache** | Official distributed cache proposal | [GitHub](https://github.com/pytorch/pytorch/issues/XXX) |
| **Triton Cache Design** | How Triton caches kernels | [Triton Docs](https://triton-lang.org) |
| **sccache** | Rust compilation caching (architecture reference) | [GitHub](https://github.com/mozilla/sccache) |

### Key Techniques

**Content-Addressable Storage**
```
address = hash(artifact_content)
store(address, artifact)

Benefits:
- Automatic deduplication
- Verification via hash
- Location-independent keys
```

**Semantic Cache Keys**
```
Instead of:
  key = hash(source_code + file_paths + ...)

Use:
  key = hash(computation_graph + input_shapes + ...)

Graph structure is machine-independent.
```

**Multi-Level Artifacts**
```
Store at multiple specificity levels:
1. cubin (GPU-specific, fastest)
2. PTX (compute-capability specific)
3. Triton IR (portable, needs final compile)
4. Source (most portable, slowest)

Try most specific first, fall back as needed.
```

### Repos

| Repo | What It Does | Link |
|------|--------------|------|
| **torch._inductor.codecache** | PyTorch's cache implementation | [GitHub](https://github.com/pytorch/pytorch) |
| **triton/runtime/cache.py** | Triton's cache | [GitHub](https://github.com/openai/triton) |
| **sccache** | Mozilla's distributed compile cache | [GitHub](https://github.com/mozilla/sccache) |
| **bazel-remote** | Remote cache protocol reference | [GitHub](https://github.com/buchgr/bazel-remote) |

## Competitive Landscape

### Checkpoint Compression

| Solution | Status | Notes |
|----------|--------|-------|
| **No funded competitors** | ✓ Opportunity | Academic papers only |
| **DeepSpeed** | Partial | NVMe optimization, no compression |
| **Nebius (Russia)** | Limited | Not focused here |

### Data Deduplication

| Solution | Status | Notes |
|----------|--------|-------|
| **DatologyAI** | $57.6M funded | Real competition |
| **text-dedup** | Open source | Python, slow |
| **NeMo Curator** | NVIDIA | GPU-focused |

### Compilation Caching

| Solution | Status | Notes |
|----------|--------|-------|
| **No competitors** | ✓ Opportunity | PyTorch team under-resourced |
| **Built-in caches** | Broken | Don't work across machines |

## Reading Order

### For Checkpoint Agent
1. LMC paper (byte grouping technique)
2. **PyTorch DCP source code** (SavePlanner API)
3. **safetensors source** (format we need to support)
4. Check-N-Run (delta encoding, post-MVP)
5. ExCP paper (advanced, post-MVP)

### For Dedup Agent
1. "Deduplicating Training Data" (Google 2021)
2. text-dedup source code
3. **arrow-rs + parquet crate docs** (I/O layer)
4. LSHBloom paper (memory optimization, v0.2)
5. SemDeDup (semantic, v0.4+)

### For Cache Agent
1. **torch._inductor.codecache source** (understand what we're wrapping)
2. **Triton cache source** (same)
3. sccache architecture (design patterns)
4. PyTorch Mega-Cache RFC (future direction)
5. **Don't go deep** — start with env var interception

### For All Agents
1. PyO3 user guide (Python bindings)
2. maturin documentation (packaging)
3. object_store crate docs (storage abstraction)

## Useful Tools

| Tool | Purpose |
|------|---------|
| **safetensors** | Fast tensor serialization format |
| **huggingface_hub** | Download test models/checkpoints |
| **datasets** | Load test datasets |
| **py-spy** | Profile Python/Rust integration |
| **flamegraph** | Visualize Rust performance |
| **hyperfine** | CLI benchmarking |

## Python Packaging

**`pip install mithril` matters more than Spark integration.**

### Package Structure

```
mithril/
├── pyproject.toml           # PEP 517 build config
├── python/
│   └── mithril/
│       ├── __init__.py      # Unified entry point
│       ├── checkpoint/
│       │   ├── __init__.py
│       │   └── torch.py     # PyTorch DCP integration
│       ├── dedup/
│       │   └── __init__.py
│       └── cache/
│           └── __init__.py
└── crates/
    └── mithril-python/      # PyO3 bindings
        ├── Cargo.toml
        └── pyproject.toml   # maturin config
```

### Build with Maturin

```toml
# crates/mithril-python/pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "mithril"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
jax = ["jax>=0.4", "orbax-checkpoint"]
all = ["mithril[torch,jax]"]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
```

### Release Process

```bash
# Build wheels for all platforms
maturin build --release

# Upload to PyPI
maturin upload

# Or use CI (GitHub Actions + maturin-action)
```

### Installation Targets

| Target | Command | Notes |
|--------|---------|-------|
| Basic | `pip install mithril` | Core functionality |
| PyTorch | `pip install mithril[torch]` | + DCP integration |
| JAX | `pip install mithril[jax]` | + Orbax integration |
| Everything | `pip install mithril[all]` | All integrations |

### Why This Matters

1. **Frictionless adoption** — users won't clone repos and build from source
2. **Dependency management** — optional deps for PyTorch/JAX
3. **Platform support** — maturin builds wheels for Linux/macOS/Windows
4. **Versioning** — semantic versioning, clear upgrade path

## Related Projects

### Mallorn — Edge Model Delta Updates

A separate project for **edge/embedded model deployment**, not training infrastructure.

| Aspect | Mithril | Mallorn |
|--------|---------|---------|
| Target | Training engineers, cloud | Embedded engineers, OTA |
| Model size | 100GB+ checkpoints | 200KB-2GB models |
| Formats | safetensors, PyTorch DCP | TFLite, GGUF, ONNX |
| Storage | S3/GCS | Flash, A/B slots |
| Bandwidth | Datacenter | LoRa, NB-IoT, cellular |
| Constraints | Compression ratio | Patch size, atomic updates, 1KB RAM |

**Why separate:** Different users, different constraints, different formats. Don't bloat Mithril's scope.

**Shared future:** Both projects may share compression primitives (ZipNN-style exponent coding) via a common library.

See: `mallorn/docs/` for edge delta documentation.
