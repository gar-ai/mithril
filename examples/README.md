# Mithril Examples

## Rust Demo: Side-by-Side Compression Comparison

Simulates a 10-step training loop on 256 MB of synthetic bf16 model weights,
comparing naive full-checkpoint saving vs. Mithril's delta compression.
Also demonstrates dataset deduplication throughput.

### Run

```bash
cargo run --release --example demo
```

### Expected Output

- Step 1: ~1.3x compression (standalone, no delta reference yet)
- Steps 2-10: ~140x compression (delta encoding exploits <1% weight change per step)
- All roundtrip checks pass (byte-for-byte identical reconstruction)
- Dataset dedup: ~600K+ docs/sec on 10K synthetic documents

### Tests

```bash
cargo test --example demo
```

Runs 5 tests: standalone compression, delta high-ratio, roundtrip byte-exact,
dedup finds duplicates, and helper function correctness.

---

## Python Demo: Training Loop with Mithril Checkpoints

Fine-tunes `distilgpt2` for a few steps and compresses each checkpoint
using Mithril's DeltaCompressor Python bindings.

### Prerequisites

```bash
# Build Python bindings
cd /path/to/mithril
maturin develop --release

# Install Python dependencies
pip install torch transformers
```

### Run

```bash
python examples/training_demo.py
python examples/training_demo.py --steps 10
python examples/training_demo.py --model distilgpt2 --steps 5 --lr 1e-4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `distilgpt2` | HuggingFace model name |
| `--steps` | `5` | Number of training steps |
| `--lr` | `5e-5` | Learning rate |

---

## Real Model Checkpoint Compression

Compresses actual HuggingFace model weights (safetensors format) — GTE-Small,
DINOv2-Small, MiniLM, BGE-Small-EN — demonstrating Mithril compression on
production model architectures. No synthetic data.

### Run

```bash
cargo run --release --example real_checkpoint_demo
```

### Expected Output

- Standalone compression: 1.2-2.2x on real Float16/Float32 weights
- Delta compression (simulated 0.5% fine-tuning): **141-142x** between steps
- All roundtrips byte-exact on real model weights

### Tests

```bash
cargo test --example real_checkpoint_demo
```

Runs 4 tests: safetensors readable, compression >1x, roundtrip exact, delta >10x.

---

## Real Dataset Dedup Demo

Runs MinHash/LSH deduplication on real HuggingFace datasets (AG News, CC News,
Amazon reviews, IMDB, etc.). Demonstrates per-dataset and cross-dataset
duplicate detection on ~39K real documents.

### Run

```bash
cargo run --release --example real_dedup_demo
```

### Expected Output

- Per-dataset dedup table with doc counts, duplicates, ratios, and throughput
- CC News: ~15% duplicates (syndicated wire stories)
- AG News: ~0.8% duplicates (Reuters republished)
- Cross-dataset dedup across all 39K docs
- 80K-100K+ docs/sec throughput

### Tests

```bash
cargo test --example real_dedup_demo
```

Runs 4 tests: JSONL loading, real data dedup, format_count, find_dataset.

---

## Existing Examples

| File | Description |
|------|-------------|
| `pytorch_checkpoint.py` | Basic checkpoint compression with Mithril |
| `torch_compile_cache.py` | torch.compile cache management |
| `dedup_dataset.py` | Dataset deduplication |
