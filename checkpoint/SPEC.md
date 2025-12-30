# mithril-checkpoint Specification

Checkpoint compression for ML training. Target: 39-70x compression with 3+ GiB/s throughput.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPRESSION PIPELINE                         │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Input   │──▶│  Delta   │──▶│  Byte    │──▶│   Entropy    │ │
│  │ Checkpoint│   │ Encoding │   │ Grouping │   │  (Zstd/LZ4)  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│       │              │              │                │          │
│       ▼              ▼              ▼                ▼          │
│   Raw tensors   XOR with prev   Separate exp/    Final         │
│   (bfloat16)    checkpoint      mantissa bytes   compressed    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Algorithms

### 1. Delta Encoding

Exploits the fact that consecutive checkpoints are similar (weights change slowly during training).

**MVP (v0.1): Simple XOR encoding**

```rust
/// Delta encoding between checkpoints
pub struct DeltaEncoder {
    /// Previous checkpoint signature (lightweight, not full data)
    previous_signature: Option<CheckpointSignature>,
}

impl DeltaEncoder {
    /// Encode current checkpoint as delta from previous
    pub fn encode(&mut self, current: &[u8], previous: Option<&[u8]>) -> Vec<u8> {
        match previous {
            Some(prev) => {
                // XOR encoding - identical bytes become zeros (compress well)
                current.iter()
                    .zip(prev.iter())
                    .map(|(c, p)| c ^ p)
                    .collect()
            }
            None => current.to_vec(),
        }
    }
    
    /// Decode delta back to full checkpoint
    pub fn decode(&self, delta: &[u8], previous: &[u8]) -> Vec<u8> {
        delta.iter()
            .zip(previous.iter())
            .map(|(d, p)| d ^ p)
            .collect()
    }
}
```

**v0.2+: Importance-Aware Delta Compression**

Research insights to incorporate:

| Paper | Technique | Benefit |
|-------|-----------|---------|
| **ZipNN** (IBM/MIT 2024) | Exploits floating-point exponent skewness | 33-50% lossless compression, 80GB/s decompression |
| **ImPart** (April 2025) | SVD-based importance-aware sparsification | 2× better compression than uniform sparsity |
| **Delta-CoMe** (June 2024) | Mixed-precision delta quantization | Allocate bits by singular value magnitude |

```rust
/// v0.2: Importance-aware delta encoding (post-MVP)
pub struct ImportanceAwareDelta {
    /// Number of singular vectors to retain
    top_k_singular: usize,
    /// Quantization bits for large vs small singular values
    high_bits: u8,  // e.g., 8 bits for top singular vectors
    low_bits: u8,   // e.g., 2 bits for tail
}

impl ImportanceAwareDelta {
    /// Compress weight delta using SVD decomposition
    /// Key insight from ImPart: not all weight deltas contribute equally
    pub fn compress(&self, delta: &Tensor) -> CompressedDelta {
        // 1. Reshape to 2D matrix
        // 2. SVD decomposition: delta = U @ S @ V^T
        // 3. Keep top-k singular vectors at high precision
        // 4. Aggressively quantize remaining (they matter less)
        // 5. Compress with ZipNN-style exponent-aware encoding
        todo!("v0.2 implementation")
    }
}

/// v0.2: ZipNN-style compression exploiting FP exponent distribution
/// Neural network weights have skewed exponent distributions
/// (clustered around small values) - exploit this for better compression
pub struct NeuralCompressor {
    /// Use neural-network-specific entropy coding
    use_exponent_coding: bool,
}
```

The key insight from ZipNN: neural network weights have **non-uniform exponent distributions** that general-purpose compressors (zstd, lz4) don't exploit. By coding exponents separately with knowledge of this skewness, we get 33-50% better compression.

/// Lightweight signature for identifying checkpoints
#[derive(Clone)]
pub struct CheckpointSignature {
    pub hash: [u8; 32],
    pub size: u64,
    pub step: u64,
}
```

### 2. bfloat16 Byte Grouping

Based on LMC paper. bfloat16 values have structure: exponent bytes compress differently than mantissa bytes.

```
bfloat16 layout: [SEEEEEEE EMMMMMMM]
                  ^^^^^^^^ high byte (sign + exponent)
                           ^^^^^^^^ low byte (mantissa)
```

```rust
/// Byte grouping for bfloat16 tensors
pub struct ByteGrouper;

impl ByteGrouper {
    /// Group bytes by position within bfloat16 values
    /// Input: [h0, l0, h1, l1, h2, l2, ...]
    /// Output: [h0, h1, h2, ..., l0, l1, l2, ...]
    pub fn group(data: &[u8]) -> Vec<u8> {
        let len = data.len();
        assert!(len % 2 == 0, "bfloat16 data must have even length");
        
        let mut result = Vec::with_capacity(len);
        let half = len / 2;
        
        // High bytes first
        for i in (0..len).step_by(2) {
            result.push(data[i]);
        }
        // Low bytes second
        for i in (1..len).step_by(2) {
            result.push(data[i]);
        }
        
        result
    }
    
    /// Ungroup bytes back to interleaved format
    pub fn ungroup(data: &[u8]) -> Vec<u8> {
        let len = data.len();
        let half = len / 2;
        
        let mut result = Vec::with_capacity(len);
        for i in 0..half {
            result.push(data[i]);        // High byte
            result.push(data[half + i]); // Low byte
        }
        
        result
    }
}
```

### 3. Adaptive Quantization

Based on DynaQuant. Use gradient magnitude to determine which weights can tolerate lower precision.

```rust
/// Sensitivity-aware quantization
pub struct AdaptiveQuantizer {
    /// Target bits per value
    target_bits: u8,
    /// Maximum restarts before accuracy degradation
    max_restarts: u32,
}

impl AdaptiveQuantizer {
    pub fn new(target_bits: u8) -> Self {
        Self {
            target_bits,
            max_restarts: 10,  // DynaQuant limit
        }
    }
    
    /// Quantize tensor with sensitivity scores
    /// Returns quantized data + metadata for reconstruction
    pub fn quantize(&self, data: &[f32], sensitivity: Option<&[f32]>) -> QuantizedTensor {
        match sensitivity {
            Some(sens) => self.quantize_adaptive(data, sens),
            None => self.quantize_uniform(data),
        }
    }
    
    fn quantize_uniform(&self, data: &[f32]) -> QuantizedTensor {
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / ((1 << self.target_bits) - 1) as f32;
        
        let quantized: Vec<u8> = data.iter()
            .map(|&v| ((v - min) / scale).round() as u8)
            .collect();
        
        QuantizedTensor {
            data: quantized,
            min,
            scale,
            bits: self.target_bits,
        }
    }
    
    /// Dequantize back to f32
    pub fn dequantize(&self, tensor: &QuantizedTensor) -> Vec<f32> {
        tensor.data.iter()
            .map(|&q| tensor.min + (q as f32) * tensor.scale)
            .collect()
    }
}

pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub min: f32,
    pub scale: f32,
    pub bits: u8,
}
```

## File Format

```
┌─────────────────────────────────────────┐
│              MITHRIL CHECKPOINT          │
├─────────────────────────────────────────┤
│ Magic: "MCKP" (4 bytes)                 │
│ Version: u32                            │
│ Flags: u32                              │
│   bit 0: has_delta                      │
│   bit 1: has_byte_grouping              │
│   bit 2: has_quantization               │
│ Header Length: u32                      │
├─────────────────────────────────────────┤
│ Header (JSON, compressed):              │
│   - original_size                       │
│   - tensor_metadata[]                   │
│   - compression_params                  │
│   - previous_checkpoint_hash            │
│   - training_step                       │
├─────────────────────────────────────────┤
│ Tensor Blocks:                          │
│   ┌─────────────────────────────────┐  │
│   │ Block Header:                    │  │
│   │   - tensor_name                  │  │
│   │   - compressed_size              │  │
│   │   - original_size                │  │
│   │   - compression_method           │  │
│   ├─────────────────────────────────┤  │
│   │ Compressed Data                  │  │
│   └─────────────────────────────────┘  │
│   ... more blocks ...                   │
├─────────────────────────────────────────┤
│ Footer:                                 │
│   - checksum (xxhash64)                 │
│   - total_compressed_size               │
└─────────────────────────────────────────┘
```

## Public API

### Rust API

```rust
// High-level API
pub struct CheckpointCompressor {
    config: CompressorConfig,
    storage: Box<dyn StorageBackend>,
    signature_cache: LruCache<String, CheckpointSignature>,
}

impl CheckpointCompressor {
    pub fn new(config: CompressorConfig, storage: Box<dyn StorageBackend>) -> Self;
    
    /// Compress a checkpoint
    pub async fn compress(
        &mut self,
        checkpoint: &Checkpoint,
        previous: Option<&str>,  // Key of previous checkpoint
    ) -> Result<CompressedCheckpoint>;
    
    /// Decompress a checkpoint
    pub async fn decompress(
        &self,
        compressed: &CompressedCheckpoint,
    ) -> Result<Checkpoint>;
    
    /// Save compressed checkpoint to storage
    pub async fn save(
        &mut self,
        checkpoint: &Checkpoint,
        key: &str,
        previous_key: Option<&str>,
    ) -> Result<CheckpointMeta>;
    
    /// Load and decompress checkpoint from storage
    pub async fn load(&self, key: &str) -> Result<Checkpoint>;
}

pub struct CompressorConfig {
    /// Enable delta encoding (default: true)
    pub delta_enabled: bool,
    /// Enable byte grouping for bfloat16 (default: true)
    pub byte_grouping: bool,
    /// Compression level (1-22 for zstd)
    pub compression_level: i32,
    /// Enable lossy quantization (default: false)
    pub quantization: Option<QuantizationConfig>,
}

pub struct Checkpoint {
    pub tensors: HashMap<String, TensorData>,
    pub metadata: HashMap<String, serde_json::Value>,
}

pub struct TensorData {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}
```

### Python API

```python
import mithril.checkpoint as ckpt

# Create compressor with local storage
compressor = ckpt.Compressor(
    storage="local:///tmp/checkpoints",
    delta_enabled=True,
    byte_grouping=True,
    compression_level=3,
)

# Compress and save PyTorch checkpoint
state_dict = model.state_dict()
meta = compressor.save(state_dict, key="step_1000", previous_key="step_900")
print(f"Compressed {meta.original_size} -> {meta.compressed_size} ({meta.ratio:.1f}x)")

# Load checkpoint
loaded = compressor.load("step_1000")
model.load_state_dict(loaded)

# Integration with PyTorch DCP
from mithril.checkpoint.torch import MithrilSavePlanner, MithrilLoadPlanner

# Use as custom planner with torch.distributed.checkpoint
torch.distributed.checkpoint.save(
    state_dict,
    storage_writer=...,
    planner=MithrilSavePlanner(compressor),
)
```

## Framework Integration

### PyTorch DCP Integration (P0 - MVP)

PyTorch Distributed Checkpoint (DCP) is the modern API for saving FSDP/distributed checkpoints. We integrate via custom SavePlanner/LoadPlanner.

```python
# User-facing API
from mithril.checkpoint.torch import MithrilSavePlanner, MithrilLoadPlanner
import torch.distributed.checkpoint as dcp

# Save with compression
dcp.save(
    state_dict,
    storage_writer=dcp.FileSystemWriter(path),
    planner=MithrilSavePlanner(compression_level=3),
)

# Load with decompression  
dcp.load(
    state_dict,
    storage_reader=dcp.FileSystemReader(path),
    planner=MithrilLoadPlanner(),
)
```

```rust
// Rust side: implement transform hooks
pub struct MithrilSavePlanner {
    compressor: CheckpointCompressor,
}

impl MithrilSavePlanner {
    /// Hook called by DCP to transform tensors before saving
    pub fn transform_tensor(&self, tensor: &TensorData) -> Result<TransformedTensor> {
        // Apply byte grouping for bfloat16
        let grouped = if tensor.dtype == DType::BFloat16 {
            ByteGrouper::group(&tensor.data)
        } else {
            tensor.data.clone()
        };
        
        // Apply compression
        let compressed = self.compressor.compress_tensor(&grouped)?;
        
        Ok(TransformedTensor {
            data: compressed,
            original_size: tensor.data.len(),
            metadata: CompressionMeta {
                byte_grouped: tensor.dtype == DType::BFloat16,
                compressor: "zstd".to_string(),
            },
        })
    }
}
```

**Why DCP?** FSDP users have the biggest checkpoints (sharded across ranks). This is where compression helps most. DeepSpeed users already have decent tooling.

### Safetensors Support (P0 - MVP Read, P1 - Write)

Safetensors is increasingly the standard format, especially for HuggingFace models.

```python
# Read safetensors, compress, save as mithril format
from mithril.checkpoint import Compressor
from safetensors import safe_open

compressor = Compressor(storage="local:///checkpoints")

# Load from safetensors
with safe_open("model.safetensors", framework="pt") as f:
    state_dict = {k: f.get_tensor(k) for k in f.keys()}

# Save compressed
compressor.save(state_dict, key="model_v1")
```

```rust
// Rust: safetensors reading via safetensors crate
use safetensors::SafeTensors;

pub fn load_safetensors(path: &Path) -> Result<HashMap<String, TensorData>> {
    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    
    let mut result = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        result.insert(name.to_string(), TensorData {
            data: tensor.data().to_vec(),
            shape: tensor.shape().to_vec(),
            dtype: convert_dtype(tensor.dtype()),
        });
    }
    Ok(result)
}
```

### DeepSpeed Integration (P1 - v0.2)

DeepSpeed has its own checkpoint system. Integration is callback-based:

```python
# deepspeed_config.json
{
    "checkpoint": {
        "use_mithril": true,
        "mithril_config": {
            "compression_level": 3,
            "byte_grouping": true
        }
    }
}
```

**Why P1?** DeepSpeed's existing checkpoint system already works reasonably well. The acute pain is vanilla PyTorch DCP users who don't have these optimizations.

### JAX/Orbax Integration (P2 - v0.3)

```python
import orbax.checkpoint as ocp
from mithril.checkpoint.orbax import MithrilCheckpointHandler

handler = MithrilCheckpointHandler(compression_level=3)
checkpointer = ocp.Checkpointer(handler)
```

**Why P2?** Different paradigm (functional), smaller market share than PyTorch. Get PyTorch right first.

## Implementation Plan

### Phase 1: Core Compression (Week 1-2)
- [ ] Implement `ByteGrouper` for bfloat16
- [ ] Implement `DeltaEncoder` with XOR encoding
- [ ] Implement file format writer/reader
- [ ] Basic zstd compression integration
- [ ] Unit tests for each component

### Phase 2: Pipeline Integration (Week 3-4)
- [ ] Implement `CheckpointCompressor` struct
- [ ] Async storage integration
- [ ] Signature caching for delta references
- [ ] Streaming compression for large checkpoints
- [ ] Integration tests with real checkpoints

### Phase 3: Python Bindings (Week 5-6)
- [ ] PyO3 bindings for `Compressor`
- [ ] NumPy array handling
- [ ] Async Python support
- [ ] PyTorch state_dict conversion

### Phase 4: Framework Integration (Week 7-8)
- [ ] PyTorch DCP SavePlanner/LoadPlanner
- [ ] DeepSpeed hooks
- [ ] Orbax CheckpointHandler
- [ ] End-to-end integration tests

## Benchmarks

### Target Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Compression ratio (lossless) | 10-20x | vs raw checkpoint size |
| Compression ratio (with delta) | 39-70x | vs raw, after 17+ steps |
| Compression throughput | 2.5+ GiB/s | on 16 cores |
| Decompression throughput | 3.5+ GiB/s | on 16 cores |
| Memory overhead | <2x checkpoint size | peak memory during compress |

### Benchmark Suite

```rust
// benches/checkpoint_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

fn bench_byte_grouping(c: &mut Criterion) {
    let data = generate_bfloat16_data(100_000_000); // 100M values = 200MB
    
    let mut group = c.benchmark_group("byte_grouping");
    group.throughput(Throughput::Bytes(data.len() as u64));
    
    group.bench_function("group", |b| {
        b.iter(|| ByteGrouper::group(&data))
    });
    
    group.bench_function("ungroup", |b| {
        let grouped = ByteGrouper::group(&data);
        b.iter(|| ByteGrouper::ungroup(&grouped))
    });
}

fn bench_compression_pipeline(c: &mut Criterion) {
    let checkpoint = load_test_checkpoint("pythia-410m");
    let compressor = CheckpointCompressor::new(default_config());
    
    let mut group = c.benchmark_group("compression");
    group.throughput(Throughput::Bytes(checkpoint.size() as u64));
    
    group.bench_function("compress_no_delta", |b| {
        b.iter(|| compressor.compress(&checkpoint, None))
    });
    
    group.bench_function("compress_with_delta", |b| {
        let previous = load_test_checkpoint("pythia-410m-prev");
        b.iter(|| compressor.compress(&checkpoint, Some(&previous)))
    });
}
```

## Testing Strategy

### Unit Tests
- `ByteGrouper`: roundtrip, edge cases (empty, odd length error)
- `DeltaEncoder`: roundtrip, identical input produces zeros
- `AdaptiveQuantizer`: accuracy bounds, edge values
- File format: parse/write roundtrip, corruption detection

### Integration Tests
- Compress/decompress real PyTorch checkpoints
- Verify model accuracy after compression roundtrip
- Test with different model architectures (transformer, CNN, etc.)
- Test with different dtypes (float32, float16, bfloat16)

### Property Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn byte_grouping_roundtrip(data: Vec<u8>) {
        prop_assume!(data.len() % 2 == 0);
        let grouped = ByteGrouper::group(&data);
        let ungrouped = ByteGrouper::ungroup(&grouped);
        prop_assert_eq!(data, ungrouped);
    }
}
```

## Error Handling

```rust
#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("Invalid checkpoint format: {0}")]
    InvalidFormat(String),
    
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u64, actual: u64 },
    
    #[error("Missing previous checkpoint for delta decoding: {0}")]
    MissingPrevious(String),
    
    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(DType),
    
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error(transparent)]
    Compression(#[from] CompressionError),
    
    #[error(transparent)]
    Storage(#[from] StorageError),
    
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

## References

- ExCP (ICML 2024): Weight-momentum joint compression
- LMC (May 2025): bfloat16 byte grouping, 2.78 GiB/s
- DynaQuant: Gradient-based sensitivity, 39x compression
- Check-N-Run (NSDI 2022): Meta's differential compression
- DeepSpeed FastPersist: NVMe pipelining (no compression)
- Orbax: JAX checkpointing, async multi-tier
