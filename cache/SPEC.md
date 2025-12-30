# mithril-cache Specification

Compilation caching for ML frameworks. Target: Reduce torch.compile cold starts from 67s to <1s.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CACHE ARCHITECTURE                           │
│                                                                  │
│  ┌──────────────┐                         ┌──────────────────┐  │
│  │   PyTorch    │                         │   Cache Store    │  │
│  │  Inductor    │───────┐                 │                  │  │
│  └──────────────┘       │                 │  ┌────────────┐  │  │
│                         │   ┌─────────┐   │  │   Local    │  │  │
│  ┌──────────────┐       ├──▶│ Mithril │──▶│  │   Disk     │  │  │
│  │   Triton     │───────┤   │  Cache  │   │  └────────────┘  │  │
│  │   Kernels    │       │   │  Layer  │   │        │         │  │
│  └──────────────┘       │   └─────────┘   │        ▼         │  │
│                         │        │        │  ┌────────────┐  │  │
│  ┌──────────────┐       │        │        │  │   Remote   │  │  │
│  │    JAX/XLA   │───────┘        │        │  │  (S3/GCS)  │  │  │
│  └──────────────┘                │        │  └────────────┘  │  │
│                                  │        │                  │  │
│                                  ▼        └──────────────────┘  │
│                         ┌─────────────┐                         │
│                         │   Content   │                         │
│                         │ Addressable │                         │
│                         │   Storage   │                         │
│                         └─────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## The Problem

### Current Pain Points

1. **torch.compile cold starts**: 67 seconds on H100 for diffusers models
2. **Cache portability failures**: Users report warm caches don't work across machines
3. **NCCL timeouts**: 17-30 minute compilation triggers distributed training failures
4. **No sharing**: Each developer recompiles the same kernels

### What Gets Compiled

```
User Code (Python)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ TorchDynamo: Captures Python bytecode → FX Graph             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ AOT Autograd: Traces forward + backward → Joint Graph        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ TorchInductor: Generates Triton/C++ code                     │
└──────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Triton Compiler     │    │ C++ Compiler        │
│ Python → Triton-IR  │    │ C++ → .so           │
│ → LLVM-IR → PTX     │    │                     │
│ → cubin             │    │                     │
└─────────────────────┘    └─────────────────────┘
```

### Current Cache Keys (PyTorch)

```python
# Actual cache key components from TorchInductor
cache_key = hash(
    torch_version,
    triton_version,
    cuda_version,
    gpu_name,           # e.g., "NVIDIA H100"
    graph_hash,         # FX graph structure
    input_shapes,       # Tensor shapes
    input_dtypes,       # Tensor dtypes
    compiler_flags,     # Optimization settings
    source_code_hash,   # Problem: includes file paths!
)
```

## Core Components

### 1. Semantic Cache Keys

Replace brittle file-path-based keys with content-based keys.

```rust
/// Cache key that's portable across machines
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Hash of the computation graph (not source code)
    pub graph_hash: [u8; 32],
    /// Input tensor metadata
    pub inputs: Vec<InputSpec>,
    /// Target device class
    pub device: DeviceClass,
    /// Framework version requirements
    pub requirements: VersionRequirements,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputSpec {
    pub shape: Vec<i64>,      // -1 for dynamic dims
    pub dtype: DType,
    pub layout: TensorLayout,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DeviceClass {
    /// Matches any CUDA device
    CudaAny,
    /// Matches specific compute capability
    CudaCompute { major: u8, minor: u8 },
    /// Matches specific GPU model
    CudaModel(String),
    /// CPU
    Cpu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VersionRequirements {
    /// Minimum PyTorch version
    pub torch_min: Option<semver::Version>,
    /// Minimum Triton version  
    pub triton_min: Option<semver::Version>,
    /// CUDA compute capability
    pub cuda_compute: Option<(u8, u8)>,
}

impl CacheKey {
    /// Generate cache key from FX graph
    pub fn from_fx_graph(graph: &FxGraph, inputs: &[TensorMeta]) -> Self {
        // Hash graph structure, not source code
        let graph_hash = Self::hash_graph_structure(graph);
        
        let inputs = inputs.iter()
            .map(|t| InputSpec {
                shape: t.shape.clone(),
                dtype: t.dtype,
                layout: t.layout,
            })
            .collect();
        
        Self {
            graph_hash,
            inputs,
            device: DeviceClass::CudaAny,
            requirements: VersionRequirements::default(),
        }
    }
    
    fn hash_graph_structure(graph: &FxGraph) -> [u8; 32] {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        
        // Hash ops and their connectivity, not metadata
        for node in graph.nodes() {
            hasher.update(node.op.as_bytes());
            hasher.update(&node.target.to_le_bytes());
            for arg in &node.args {
                hasher.update(&arg.to_le_bytes());
            }
        }
        
        hasher.finalize().into()
    }
    
    /// Convert to storage key
    pub fn to_storage_key(&self) -> String {
        let hash = blake3::hash(&bincode::serialize(self).unwrap());
        format!("cache/{}", hex::encode(&hash.as_bytes()[..16]))
    }
}
```

### 2. Multi-Level Artifact Storage

Store artifacts at different levels of specificity for maximum reuse.

```rust
/// Artifact types that can be cached
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CacheArtifact {
    /// Triton kernel (most specific to least specific)
    TritonKernel {
        /// CUDA binary (GPU-specific)
        cubin: Option<Vec<u8>>,
        /// PTX assembly (compute-capability specific)
        ptx: Option<String>,
        /// Triton IR (portable)
        ttir: Option<String>,
    },
    
    /// C++ compiled kernel
    CppKernel {
        /// Shared object (platform-specific)
        so: Option<Vec<u8>>,
        /// Source code (portable)
        source: String,
    },
    
    /// FX Graph (portable)
    FxGraph {
        serialized: Vec<u8>,
    },
    
    /// Autotune results
    AutotuneResult {
        config: HashMap<String, serde_json::Value>,
        benchmark_ns: u64,
    },
}

/// Artifact with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredArtifact {
    pub artifact: CacheArtifact,
    pub key: CacheKey,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub created_by: String,  // machine identifier
    pub hit_count: u64,
    /// Environment where artifact was created
    pub source_env: Environment,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Environment {
    pub torch_version: String,
    pub triton_version: Option<String>,
    pub cuda_version: Option<String>,
    pub gpu_model: Option<String>,
    pub driver_version: Option<String>,
}
```

### 3. Cache Manager

Coordinates local and remote caches.

```rust
/// Main cache manager
pub struct CacheManager {
    /// Local disk cache
    local: LocalCache,
    /// Optional remote cache
    remote: Option<Box<dyn RemoteCache>>,
    /// Compatibility checker
    compat: CompatibilityChecker,
    /// Metrics
    metrics: CacheMetrics,
}

impl CacheManager {
    pub async fn new(config: CacheConfig) -> Result<Self>;
    
    /// Get artifact from cache
    pub async fn get(&self, key: &CacheKey) -> Result<Option<StoredArtifact>> {
        // Check local first
        if let Some(artifact) = self.local.get(key).await? {
            self.metrics.record_hit("local");
            return Ok(Some(artifact));
        }
        
        // Check remote
        if let Some(remote) = &self.remote {
            if let Some(artifact) = remote.get(key).await? {
                // Verify compatibility
                if self.compat.is_compatible(&artifact)? {
                    // Store locally for next time
                    self.local.put(key, &artifact).await?;
                    self.metrics.record_hit("remote");
                    return Ok(Some(artifact));
                } else {
                    self.metrics.record_incompatible();
                }
            }
        }
        
        self.metrics.record_miss();
        Ok(None)
    }
    
    /// Store artifact in cache
    pub async fn put(&self, key: &CacheKey, artifact: &CacheArtifact) -> Result<()> {
        let stored = StoredArtifact {
            artifact: artifact.clone(),
            key: key.clone(),
            created_at: chrono::Utc::now(),
            created_by: hostname::get()?.to_string_lossy().to_string(),
            hit_count: 0,
            source_env: Environment::current()?,
        };
        
        // Store locally
        self.local.put(key, &stored).await?;
        
        // Store remotely (async, don't wait)
        if let Some(remote) = &self.remote {
            let remote = remote.clone();
            let key = key.clone();
            let stored = stored.clone();
            tokio::spawn(async move {
                let _ = remote.put(&key, &stored).await;
            });
        }
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.metrics.stats()
    }
}
```

### 4. Compatibility Checker

Determines if cached artifacts can be used in current environment.

```rust
/// Checks if artifacts are compatible with current environment
pub struct CompatibilityChecker {
    current_env: Environment,
    /// Compatibility rules
    rules: Vec<Box<dyn CompatibilityRule>>,
}

impl CompatibilityChecker {
    pub fn new() -> Result<Self> {
        let current_env = Environment::current()?;
        
        let rules: Vec<Box<dyn CompatibilityRule>> = vec![
            Box::new(TorchVersionRule),
            Box::new(CudaComputeRule),
            Box::new(TritonVersionRule),
        ];
        
        Ok(Self { current_env, rules })
    }
    
    pub fn is_compatible(&self, artifact: &StoredArtifact) -> Result<bool> {
        for rule in &self.rules {
            if !rule.check(&self.current_env, &artifact.source_env, &artifact.artifact)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

trait CompatibilityRule: Send + Sync {
    fn check(
        &self,
        current: &Environment,
        source: &Environment,
        artifact: &CacheArtifact,
    ) -> Result<bool>;
}

struct CudaComputeRule;

impl CompatibilityRule for CudaComputeRule {
    fn check(
        &self,
        current: &Environment,
        source: &Environment,
        artifact: &CacheArtifact,
    ) -> Result<bool> {
        // cubin requires exact match
        // PTX requires compatible compute capability
        match artifact {
            CacheArtifact::TritonKernel { cubin: Some(_), .. } => {
                // Exact GPU match required for cubin
                Ok(current.gpu_model == source.gpu_model)
            }
            CacheArtifact::TritonKernel { ptx: Some(_), .. } => {
                // Compute capability must be >= source
                // (forward compatibility)
                Ok(true) // Simplified; real impl checks versions
            }
            _ => Ok(true),
        }
    }
}
```

## Integration Strategy: Shallow First

### The Problem with Deep Integration

TorchInductor's internal APIs are **undocumented and unstable**. They change between PyTorch minor versions:

```python
# These are INTERNAL APIs that we'd need to hook:
torch._inductor.codecache  # Changes frequently
torch._dynamo.cache  # Also unstable
triton.runtime.cache  # More stable, but still internal
```

**Risk**: Deep integration breaks on every PyTorch update, creating maintenance burden.

### MVP Strategy: Directory Interception (v0.1)

Start shallow — just intercept the cache directories:

```python
import os
import mithril.cache as cache

# Initialize mithril cache (does the interception)
cache.init(local_dir="~/.cache/mithril")

# This sets:
# - TORCHINDUCTOR_CACHE_DIR → mithril-managed directory
# - TRITON_CACHE_DIR → mithril-managed directory

# Now torch.compile uses our directories
import torch
model = torch.compile(model)  # Artifacts go to mithril cache
```

```rust
// Rust implementation
pub fn init(config: CacheConfig) -> Result<CacheManager> {
    let manager = CacheManager::new(config)?;
    
    // Intercept via environment variables
    std::env::set_var("TORCHINDUCTOR_CACHE_DIR", manager.inductor_dir());
    std::env::set_var("TRITON_CACHE_DIR", manager.triton_dir());
    
    // Optionally set TORCH_COMPILE_CACHE_DIR (newer PyTorch)
    if let Some(dir) = manager.compile_cache_dir() {
        std::env::set_var("TORCH_COMPILE_CACHE_DIR", dir);
    }
    
    Ok(manager)
}
```

**What this gives us:**
- Better storage management (LRU eviction, size limits)
- Foundation for remote cache (sync these directories to S3)
- No dependency on unstable PyTorch internals

**What this doesn't give us:**
- Cross-machine portability (cache keys still include paths)
- Custom cache key generation

### v0.2 Strategy: Light Patching

Add cache key normalization without deep hooks:

```python
# Monkey-patch the cache key function
import torch._inductor.codecache as codecache

_original_code_hash = codecache.code_hash

def _normalized_code_hash(code):
    # Normalize file paths out of the hash
    normalized = normalize_source_paths(code)
    return _original_code_hash(normalized)

codecache.code_hash = _normalized_code_hash
```

```rust
// Rust support for path normalization
pub fn normalize_source_paths(code: &str) -> String {
    // Replace absolute paths with relative or hashed versions
    let re = regex::Regex::new(r#"['"](/[^'"]+\.py)['"]"#).unwrap();
    re.replace_all(code, |caps: &regex::Captures| {
        let path = &caps[1];
        let hash = xxhash_rust::xxh3::xxh3_64(path.as_bytes());
        format!("\"__normalized_{}\"", hash)
    }).to_string()
}
```

### v0.3+ Strategy: Deep Hooks (If Stable)

Only pursue deep integration if PyTorch stabilizes these APIs:

```python
# Hypothetical future API (doesn't exist yet)
from torch.compiler import CacheBackend

class MithrilCacheBackend(CacheBackend):
    def get(self, key: CacheKey) -> Optional[Artifact]:
        return self.manager.get(key)
    
    def put(self, key: CacheKey, artifact: Artifact):
        self.manager.put(key, artifact)

# Register with PyTorch
torch.compiler.set_cache_backend(MithrilCacheBackend())
```

**Wait for PyTorch team to provide stable extension points.** Don't fight unstable internals.

## Framework Hooks

### TorchInductor Cache (P0 - MVP)

```rust
/// Hook into PyTorch's cache system via environment variables
pub struct InductorHook {
    cache_dir: PathBuf,
    manager: Arc<CacheManager>,
}

impl InductorHook {
    pub fn install(manager: Arc<CacheManager>) -> Result<Self> {
        let cache_dir = manager.local_path().join("inductor");
        std::fs::create_dir_all(&cache_dir)?;
        
        // Set environment variable - PyTorch will use this directory
        std::env::set_var("TORCHINDUCTOR_CACHE_DIR", &cache_dir);
        
        // Optional: Watch directory for new artifacts
        // (for eager sync to remote cache)
        
        Ok(Self { cache_dir, manager })
    }
}
```

### Triton Cache (P0 - MVP)

```rust
/// Hook into Triton's cache system
pub struct TritonHook {
    cache_dir: PathBuf,
    manager: Arc<CacheManager>,
}

impl TritonHook {
    pub fn install(manager: Arc<CacheManager>) -> Result<Self> {
        let cache_dir = manager.local_path().join("triton");
        std::fs::create_dir_all(&cache_dir)?;
        
        std::env::set_var("TRITON_CACHE_DIR", &cache_dir);
        
        Ok(Self { cache_dir, manager })
    }
}
```

### vLLM/SGLang (P3 - Maybe Never)

**These are inference frameworks.** They compile once, run millions of times. The compile cache pain is during *training iteration* where developers hit cold starts repeatedly.

Don't prioritize inference frameworks until training use case is solid.

## Public API

### Rust API

```rust
/// Configuration for cache
pub struct CacheConfig {
    /// Local cache directory
    pub local_dir: PathBuf,
    /// Maximum local cache size
    pub max_local_size_gb: u64,
    /// Remote cache URL (s3://, gs://, or http://)
    pub remote_url: Option<String>,
    /// Eviction policy
    pub eviction: EvictionPolicy,
    /// Enable telemetry
    pub telemetry: bool,
}

pub enum EvictionPolicy {
    Lru,
    Lfu,
    SizeBased,
    AgeBased { max_age_days: u32 },
}

// Main entry point
pub async fn init(config: CacheConfig) -> Result<CacheManager>;

// Or use builder
pub fn builder() -> CacheBuilder;

pub struct CacheBuilder {
    config: CacheConfig,
}

impl CacheBuilder {
    pub fn local_dir(mut self, path: impl AsRef<Path>) -> Self;
    pub fn max_size_gb(mut self, size: u64) -> Self;
    pub fn remote(mut self, url: &str) -> Self;
    pub fn eviction(mut self, policy: EvictionPolicy) -> Self;
    pub async fn build(self) -> Result<CacheManager>;
}
```

### Python API

```python
import mithril.cache as cache

# Initialize cache (call once at startup)
cache.init(
    local_dir="~/.cache/mithril",
    max_size_gb=100,
    remote="s3://my-bucket/ml-cache",
)

# Automatic integration with torch.compile
import torch
model = torch.compile(model)  # Now uses mithril cache

# Manual cache operations
cache.warmup(["model_v1", "model_v2"])  # Pre-fetch from remote
cache.clear()  # Clear local cache
stats = cache.stats()  # Get hit/miss statistics

# Export/import for CI/CD
cache.export("cache_bundle.tar.zst")
cache.import_("cache_bundle.tar.zst")
```

### Environment Variables

```bash
# Local cache directory
export MITHRIL_CACHE_DIR="~/.cache/mithril"

# Remote cache
export MITHRIL_REMOTE_URL="s3://bucket/cache"

# Maximum local cache size (GB)
export MITHRIL_MAX_SIZE_GB=100

# Disable remote cache
export MITHRIL_OFFLINE=1

# Debug logging
export MITHRIL_LOG=debug
```

## Implementation Plan

### Phase 1: Core CAS (Week 1-2)
- [ ] Implement content-addressable storage
- [ ] Local disk cache with LRU eviction
- [ ] Cache key generation
- [ ] Basic get/put operations

### Phase 2: Compatibility Layer (Week 3-4)
- [ ] Environment detection (torch, triton, cuda versions)
- [ ] Compatibility rules engine
- [ ] Multi-level artifact storage (cubin, PTX, source)
- [ ] Fallback compilation when incompatible

### Phase 3: Framework Integration (Week 5-6)
- [ ] PyTorch Inductor hooks
- [ ] Triton cache hooks
- [ ] Environment variable interception
- [ ] Integration tests with torch.compile

### Phase 4: Remote Cache (Week 7-8)
- [ ] S3 backend
- [ ] GCS backend
- [ ] Cache synchronization
- [ ] Warmup/prefetch

### Phase 5: Python Bindings (Week 9-10)
- [ ] PyO3 bindings
- [ ] High-level Python API
- [ ] torch.compile integration
- [ ] CLI tools

### Phase 6: Telemetry & Polish (Week 11-12)
- [ ] Hit/miss metrics
- [ ] Cache size tracking
- [ ] Performance benchmarks
- [ ] Documentation

## Benchmarks

### Target Metrics

| Metric | Target | Baseline |
|--------|--------|----------|
| Cold start (local hit) | <1s | 67s |
| Cold start (remote hit) | <5s | 67s |
| Cache hit rate | >80% | 0% (no sharing) |
| Local lookup latency | <10ms | N/A |
| Remote lookup latency | <500ms | N/A |

### Benchmark Suite

```rust
// benches/cache_bench.rs
fn bench_cache_lookup(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let manager = rt.block_on(CacheManager::new(test_config())).unwrap();
    
    // Pre-populate cache
    let key = CacheKey::test();
    let artifact = CacheArtifact::test();
    rt.block_on(manager.put(&key, &artifact)).unwrap();
    
    c.bench_function("local_cache_hit", |b| {
        b.to_async(&rt).iter(|| async {
            manager.get(&key).await.unwrap()
        })
    });
}

fn bench_key_generation(c: &mut Criterion) {
    let graph = load_test_fx_graph();
    let inputs = test_input_specs();
    
    c.bench_function("cache_key_generation", |b| {
        b.iter(|| CacheKey::from_fx_graph(&graph, &inputs))
    });
}
```

## Testing Strategy

### Unit Tests
- Cache key generation determinism
- LRU eviction correctness
- Compatibility rule logic
- Artifact serialization roundtrip

### Integration Tests
- Full torch.compile with caching
- Cache hit after restart
- Remote cache sync
- Multi-process cache access

### End-to-End Tests
```python
def test_torch_compile_cache():
    cache.init(local_dir=tmpdir)
    
    model = SimpleModel()
    compiled = torch.compile(model)
    
    # First run: cache miss
    t1 = time.time()
    compiled(input1)
    cold_time = time.time() - t1
    
    # Restart Python process...
    
    # Second run: cache hit
    t2 = time.time()
    compiled(input1)
    warm_time = time.time() - t2
    
    assert warm_time < cold_time / 10  # 10x faster
```

## Error Handling

```rust
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Cache key generation failed: {0}")]
    KeyGeneration(String),
    
    #[error("Artifact not found: {0}")]
    NotFound(String),
    
    #[error("Incompatible artifact: {reason}")]
    Incompatible { reason: String },
    
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Remote cache unavailable: {0}")]
    RemoteUnavailable(String),
    
    #[error("Compilation failed after cache miss: {0}")]
    CompilationFailed(String),
}
```

## Security Considerations

### Artifact Integrity
```rust
/// Verify artifact hasn't been tampered with
pub fn verify_artifact(artifact: &StoredArtifact, expected_hash: &[u8]) -> bool {
    let actual_hash = blake3::hash(&bincode::serialize(&artifact.artifact).unwrap());
    actual_hash.as_bytes() == expected_hash
}
```

### Remote Cache Authentication
```rust
pub struct RemoteCacheAuth {
    /// AWS credentials for S3
    aws_credentials: Option<aws_credential_types::Credentials>,
    /// GCP credentials for GCS
    gcp_credentials: Option<String>,
    /// API key for custom backends
    api_key: Option<String>,
}
```

## References

- PyTorch TorchInductor cache: `torch/_inductor/codecache.py`
- Triton cache: `triton/runtime/cache.py`
- PyTorch Mega-Cache API: `torch.compiler.save_cache_artifacts()`
- Bazel Remote Execution API (inspiration for remote cache protocol)
- sccache (Rust compilation cache, architectural reference)
