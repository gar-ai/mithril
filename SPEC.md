# mithril-dedup Specification

Data deduplication for ML training datasets. Target: Process 1TB+ with <$100 compute cost.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEDUPLICATION PIPELINE                       │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Input   │──▶│ Tokenize │──▶│ MinHash  │──▶│     LSH      │ │
│  │Documents │   │ / Shingle│   │Signature │   │  Bucketing   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│                                                      │          │
│                                                      ▼          │
│                                            ┌──────────────┐     │
│  ┌──────────┐   ┌──────────┐              │   Candidate   │     │
│  │  Output  │◀──│  Filter  │◀─────────────│    Pairs     │     │
│  │ Unique   │   │Duplicates│              └──────────────┘     │
│  └──────────┘   └──────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Algorithms

### 1. MinHash Signatures

Approximates Jaccard similarity between documents.

```rust
use std::collections::HashSet;

/// MinHash signature generator
pub struct MinHasher {
    /// Number of hash permutations (more = more accurate, more memory)
    num_permutations: usize,
    /// Hash seeds for each permutation
    seeds: Vec<u64>,
}

impl MinHasher {
    pub fn new(num_permutations: usize) -> Self {
        let mut rng = rand::thread_rng();
        let seeds: Vec<u64> = (0..num_permutations)
            .map(|_| rng.gen())
            .collect();
        
        Self { num_permutations, seeds }
    }
    
    /// Generate MinHash signature from token set
    pub fn signature(&self, tokens: &HashSet<u64>) -> MinHashSignature {
        let mut min_hashes = vec![u64::MAX; self.num_permutations];
        
        for &token in tokens {
            for (i, &seed) in self.seeds.iter().enumerate() {
                let hash = self.hash_with_seed(token, seed);
                min_hashes[i] = min_hashes[i].min(hash);
            }
        }
        
        MinHashSignature { values: min_hashes }
    }
    
    /// Hash token with seed using xxhash
    fn hash_with_seed(&self, token: u64, seed: u64) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(
            &token.to_le_bytes(),
            seed
        )
    }
    
    /// Estimate Jaccard similarity from signatures
    pub fn similarity(sig1: &MinHashSignature, sig2: &MinHashSignature) -> f64 {
        let matches = sig1.values.iter()
            .zip(sig2.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        
        matches as f64 / sig1.values.len() as f64
    }
}

#[derive(Clone, Debug)]
pub struct MinHashSignature {
    pub values: Vec<u64>,
}
```

### 2. Locality-Sensitive Hashing (LSH)

Groups similar documents into buckets for efficient candidate pair generation.

```rust
/// LSH index for finding candidate duplicates
pub struct LshIndex {
    /// Number of bands (groups of rows)
    num_bands: usize,
    /// Rows per band
    rows_per_band: usize,
    /// Buckets for each band: band_id -> hash -> doc_ids
    buckets: Vec<HashMap<u64, Vec<DocId>>>,
}

impl LshIndex {
    /// Create LSH index with target similarity threshold
    /// 
    /// For threshold t with b bands and r rows:
    /// P(candidates) ≈ 1 - (1 - t^r)^b
    pub fn with_threshold(num_permutations: usize, threshold: f64) -> Self {
        // Optimize b and r for given threshold
        let (num_bands, rows_per_band) = Self::optimize_params(num_permutations, threshold);
        
        Self {
            num_bands,
            rows_per_band,
            buckets: (0..num_bands).map(|_| HashMap::new()).collect(),
        }
    }
    
    fn optimize_params(n: usize, t: f64) -> (usize, usize) {
        // Find b, r where b * r = n and threshold behavior is optimal
        let mut best = (1, n);
        let mut best_diff = f64::MAX;
        
        for b in 1..=n {
            if n % b == 0 {
                let r = n / b;
                // Threshold where P(candidate) = 0.5
                let estimated_t = (1.0 / b as f64).powf(1.0 / r as f64);
                let diff = (estimated_t - t).abs();
                if diff < best_diff {
                    best = (b, r);
                    best_diff = diff;
                }
            }
        }
        
        best
    }
    
    /// Add document signature to index
    pub fn insert(&mut self, doc_id: DocId, signature: &MinHashSignature) {
        for band in 0..self.num_bands {
            let start = band * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_hash = self.hash_band(&signature.values[start..end]);
            
            self.buckets[band]
                .entry(band_hash)
                .or_insert_with(Vec::new)
                .push(doc_id);
        }
    }
    
    /// Find candidate duplicate pairs
    pub fn candidates(&self) -> impl Iterator<Item = (DocId, DocId)> + '_ {
        self.buckets.iter()
            .flat_map(|band_buckets| {
                band_buckets.values()
                    .filter(|bucket| bucket.len() > 1)
                    .flat_map(|bucket| {
                        // Generate all pairs in bucket
                        bucket.iter().enumerate()
                            .flat_map(move |(i, &id1)| {
                                bucket[i+1..].iter().map(move |&id2| (id1, id2))
                            })
                    })
            })
    }
    
    fn hash_band(&self, values: &[u64]) -> u64 {
        let bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        xxhash_rust::xxh3::xxh3_64(&bytes)
    }
}
```

### 3. LSHBloom (Memory-Efficient LSH)

Replace hash tables with Bloom filters for 54x memory reduction.

```rust
use bitvec::prelude::*;

/// Memory-efficient LSH using Bloom filters
pub struct LshBloom {
    num_bands: usize,
    rows_per_band: usize,
    /// Bloom filters for each band
    filters: Vec<BloomFilter>,
    /// Document IDs with their signatures (for verification)
    documents: Vec<(DocId, MinHashSignature)>,
}

pub struct BloomFilter {
    bits: BitVec,
    num_hashes: usize,
}

impl BloomFilter {
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let num_bits = Self::optimal_bits(expected_items, fp_rate);
        let num_hashes = Self::optimal_hashes(num_bits, expected_items);
        
        Self {
            bits: bitvec![0; num_bits],
            num_hashes,
        }
    }
    
    pub fn insert(&mut self, item: u64) {
        for i in 0..self.num_hashes {
            let hash = self.hash_with_index(item, i);
            let idx = (hash as usize) % self.bits.len();
            self.bits.set(idx, true);
        }
    }
    
    pub fn contains(&self, item: u64) -> bool {
        (0..self.num_hashes).all(|i| {
            let hash = self.hash_with_index(item, i);
            let idx = (hash as usize) % self.bits.len();
            self.bits[idx]
        })
    }
    
    fn optimal_bits(n: usize, p: f64) -> usize {
        (-(n as f64) * p.ln() / (2.0_f64.ln().powi(2))).ceil() as usize
    }
    
    fn optimal_hashes(m: usize, n: usize) -> usize {
        ((m as f64 / n as f64) * 2.0_f64.ln()).ceil() as usize
    }
    
    fn hash_with_index(&self, item: u64, index: usize) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(&item.to_le_bytes(), index as u64)
    }
}
```

### 4. Suffix Array Deduplication

For exact substring deduplication (repeated passages).

```rust
/// Suffix array for finding repeated substrings
pub struct SuffixArrayDedup {
    /// Minimum duplicate length to report
    min_length: usize,
}

impl SuffixArrayDedup {
    pub fn new(min_length: usize) -> Self {
        Self { min_length }
    }
    
    /// Find all duplicate substrings in text
    pub fn find_duplicates(&self, text: &str) -> Vec<DuplicateSpan> {
        // Build suffix array using divsufsort or similar
        let sa = self.build_suffix_array(text.as_bytes());
        let lcp = self.build_lcp_array(text.as_bytes(), &sa);
        
        let mut duplicates = Vec::new();
        
        for i in 1..sa.len() {
            if lcp[i] >= self.min_length {
                duplicates.push(DuplicateSpan {
                    position: sa[i] as usize,
                    length: lcp[i],
                    duplicate_of: sa[i - 1] as usize,
                });
            }
        }
        
        duplicates
    }
    
    fn build_suffix_array(&self, data: &[u8]) -> Vec<i32> {
        // Use cdivsufsort crate
        let mut sa = vec![0i32; data.len()];
        cdivsufsort::sort(data, &mut sa);
        sa
    }
    
    fn build_lcp_array(&self, data: &[u8], sa: &[i32]) -> Vec<usize> {
        // Kasai's algorithm for LCP array
        let n = data.len();
        let mut rank = vec![0usize; n];
        let mut lcp = vec![0usize; n];
        
        for i in 0..n {
            rank[sa[i] as usize] = i;
        }
        
        let mut k = 0usize;
        for i in 0..n {
            if rank[i] == 0 {
                k = 0;
                continue;
            }
            
            let j = sa[rank[i] - 1] as usize;
            while i + k < n && j + k < n && data[i + k] == data[j + k] {
                k += 1;
            }
            
            lcp[rank[i]] = k;
            k = k.saturating_sub(1);
        }
        
        lcp
    }
}

pub struct DuplicateSpan {
    pub position: usize,
    pub length: usize,
    pub duplicate_of: usize,
}
```

### 5. Union-Find for Duplicate Clusters

Group duplicate documents into clusters.

```rust
/// Union-Find data structure for clustering duplicates
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    
    /// Find root of element with path compression
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    
    /// Union two elements by rank
    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        
        if rx == ry {
            return;
        }
        
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
    
    /// Get all clusters
    pub fn clusters(&mut self) -> HashMap<usize, Vec<usize>> {
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            clusters.entry(root).or_default().push(i);
        }
        clusters
    }
}
```

## Ecosystem Integration

### HuggingFace Datasets Compatibility (P0 - MVP)

We don't integrate with the `datasets` library directly. We achieve compatibility by reading/writing the same file formats with the same schemas.

```python
# User workflow - no library integration needed
from datasets import load_dataset

# 1. Export HF dataset to parquet
dataset = load_dataset("c4", "en", split="train")
dataset.to_parquet("c4_train.parquet")

# 2. Deduplicate with mithril
# $ mithril-dedup c4_train.parquet -o c4_deduped.parquet --text-column text

# 3. Load deduped back into HF
deduped = load_dataset("parquet", data_files="c4_deduped.parquet")
```

**Why format compatibility over library integration?** 
- Simpler implementation
- No Python dependency conflicts
- Works with any tool that reads Parquet
- Users already know this workflow

### Parquet/Arrow Support (P0 - MVP)

Native Parquet support via `arrow-rs` and `parquet` crates:

```rust
use arrow::array::StringArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub fn read_parquet_texts(path: &Path, text_column: &str) -> Result<Vec<Document>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    
    let mut documents = Vec::new();
    let mut doc_id = 0;
    
    for batch in reader {
        let batch = batch?;
        let texts = batch
            .column_by_name(text_column)
            .ok_or_else(|| DedupError::InvalidConfig(format!("Column '{}' not found", text_column)))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DedupError::InvalidConfig("Text column must be string type".into()))?;
        
        for text in texts.iter().flatten() {
            documents.push(Document {
                id: doc_id,
                text: text.to_string(),
            });
            doc_id += 1;
        }
    }
    
    Ok(documents)
}
```

### JSON Lines Support (P0 - MVP)

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct JsonDoc {
    #[serde(default)]
    id: Option<u64>,
    text: String,
}

pub fn read_jsonl(path: &Path, text_field: &str) -> Result<Vec<Document>> {
    let file = BufReader::new(File::open(path)?);
    let mut documents = Vec::new();
    
    for (line_num, line) in file.lines().enumerate() {
        let line = line?;
        let json: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| DedupError::Parse { line: line_num, message: e.to_string() })?;
        
        let text = json.get(text_field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| DedupError::Parse { 
                line: line_num, 
                message: format!("Field '{}' not found or not string", text_field) 
            })?;
        
        documents.push(Document {
            id: line_num as u64,
            text: text.to_string(),
        });
    }
    
    Ok(documents)
}
```

### What We're NOT Integrating

| Integration | Why Skip |
|-------------|----------|
| **Spark/Ray** | Single-node Rust + rayon handles TB-scale. Distributed adds massive complexity for marginal benefit. |
| **NVIDIA RAPIDS** | GPU acceleration is overkill for MinHash/LSH. CPU is fast enough. |
| **NeMo Curator** | Competitor, not integration target. |
| **Embedding models** | Semantic dedup is v0.4+ at earliest. Keep MVP simple. |

### Scaling Without Distribution

**Claim: Single-node Rust can process TB-scale data.**

Math:
- 100K docs/sec throughput target
- 1TB dataset ≈ 1B documents (assuming 1KB avg)
- Processing time: 1B / 100K = 10,000 seconds ≈ 3 hours

3 hours on a single machine is acceptable for a batch job. If customers need faster:
1. Run on bigger machine (more cores → more rayon parallelism)
2. Shard input files, run mithril per-shard, merge results
3. Wait for v0.3+ if we decide distributed is needed

**Don't add Spark/Ray complexity until proven necessary.**

## File Formats

### Input Formats

```rust
pub enum InputFormat {
    /// JSON Lines (one JSON object per line)
    JsonLines { text_field: String },
    /// Parquet files (via arrow-rs)
    Parquet { text_column: String },
    /// Plain text (one document per line or file)
    PlainText,
    /// CSV/TSV
    Csv { text_column: String, delimiter: char },
}

impl InputFormat {
    /// Auto-detect format from file extension
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "jsonl" | "json" => Some(Self::JsonLines { text_field: "text".into() }),
            "parquet" => Some(Self::Parquet { text_column: "text".into() }),
            "txt" => Some(Self::PlainText),
            "csv" => Some(Self::Csv { text_column: "text".into(), delimiter: ',' }),
            "tsv" => Some(Self::Csv { text_column: "text".into(), delimiter: '\t' }),
            _ => None,
        }
    }
}
```

### Output Format

```rust
/// Deduplication result
pub struct DedupResult {
    /// Document IDs to keep
    pub keep: Vec<DocId>,
    /// Document IDs to remove
    pub remove: Vec<DocId>,
    /// Duplicate clusters (for inspection)
    pub clusters: Vec<DuplicateCluster>,
    /// Statistics
    pub stats: DedupStats,
}

pub struct DuplicateCluster {
    /// Representative document (kept)
    pub representative: DocId,
    /// Duplicate documents (removed)
    pub duplicates: Vec<DocId>,
    /// Average pairwise similarity
    pub avg_similarity: f64,
}

pub struct DedupStats {
    pub total_documents: u64,
    pub unique_documents: u64,
    pub duplicate_documents: u64,
    pub duplicate_ratio: f64,
    pub processing_time_secs: f64,
    pub peak_memory_bytes: u64,
}
```

## Public API

### Rust API

```rust
/// Main deduplication engine
pub struct Deduplicator {
    config: DedupConfig,
    minhasher: MinHasher,
    lsh: LshIndex,
}

impl Deduplicator {
    pub fn new(config: DedupConfig) -> Self;
    
    /// Deduplicate documents from an iterator
    pub fn deduplicate<I>(&mut self, documents: I) -> Result<DedupResult>
    where
        I: Iterator<Item = Document>;
    
    /// Add documents to index (for streaming/incremental)
    pub fn add_document(&mut self, doc: Document) -> Option<Vec<DocId>>;
    
    /// Check if document is duplicate of any indexed document
    pub fn is_duplicate(&self, doc: &Document) -> Option<DuplicateInfo>;
    
    /// Get all duplicate clusters
    pub fn clusters(&self) -> Vec<DuplicateCluster>;
}

pub struct DedupConfig {
    /// Similarity threshold (0.0-1.0, default 0.85)
    pub threshold: f64,
    /// Number of MinHash permutations (default 128)
    pub num_permutations: usize,
    /// N-gram size for shingling (default 5)
    pub ngram_size: usize,
    /// Algorithm variant
    pub algorithm: DedupAlgorithm,
    /// Memory limit (enables LSHBloom when exceeded)
    pub memory_limit_bytes: Option<u64>,
}

pub enum DedupAlgorithm {
    /// MinHash + LSH (default)
    MinHashLsh,
    /// MinHash + LSH with Bloom filters
    MinHashLshBloom,
    /// Exact substring with suffix arrays
    SuffixArray { min_length: usize },
    /// SimHash (faster, less accurate)
    SimHash,
}

pub struct Document {
    pub id: DocId,
    pub text: String,
}
```

### Python API

```python
import mithril.dedup as dedup

# Simple API
duplicator = dedup.Deduplicator(
    threshold=0.85,
    num_permutations=128,
    ngram_size=5,
)

# From JSON Lines file
result = duplicator.deduplicate_file(
    "data.jsonl",
    text_field="text",
    output="unique.jsonl",
)
print(f"Removed {result.duplicate_ratio:.1%} duplicates")

# From iterator
docs = [{"id": i, "text": text} for i, text in enumerate(texts)]
result = duplicator.deduplicate(docs)

# Streaming / incremental
for doc in stream:
    duplicates = duplicator.add(doc)
    if duplicates:
        print(f"Doc {doc['id']} is duplicate of {duplicates}")

# Check single document
info = duplicator.is_duplicate(new_doc)
if info:
    print(f"Duplicate of {info.duplicate_of} (similarity: {info.similarity})")
```

### CLI

```bash
# Basic usage
mithril-dedup input.jsonl -o unique.jsonl --field text

# With options
mithril-dedup input.jsonl \
    --threshold 0.85 \
    --permutations 256 \
    --ngram 5 \
    --output unique.jsonl \
    --clusters clusters.json \
    --stats stats.json

# Suffix array mode (exact substrings)
mithril-dedup input.txt \
    --algorithm suffix-array \
    --min-length 50 \
    --output deduped.txt

# Streaming mode
cat input.jsonl | mithril-dedup --stream --field text > unique.jsonl
```

## Implementation Plan

### Phase 1: Core Algorithms (Week 1-2)
- [ ] Implement `MinHasher` with SIMD optimization
- [ ] Implement `LshIndex` with configurable bands
- [ ] Implement `UnionFind` for clustering
- [ ] Implement text tokenization / shingling
- [ ] Unit tests with known duplicate pairs

### Phase 2: Memory Efficiency (Week 3)
- [ ] Implement `LshBloom` variant
- [ ] Memory-mapped file support for large datasets
- [ ] Streaming processing mode
- [ ] Memory usage benchmarks

### Phase 3: I/O and Formats (Week 4)
- [ ] JSON Lines reader/writer
- [ ] Parquet reader (via arrow)
- [ ] CSV reader
- [ ] Parallel file processing

### Phase 4: Python Bindings (Week 5)
- [ ] PyO3 bindings for `Deduplicator`
- [ ] Iterator support
- [ ] DataFrame integration (pandas, polars)
- [ ] Progress callbacks

### Phase 5: CLI and Polish (Week 6)
- [ ] CLI with clap
- [ ] Progress bars
- [ ] Statistics output
- [ ] Documentation and examples

## Benchmarks

### Target Metrics

| Metric | Target | Comparison |
|--------|--------|------------|
| Throughput | 100K+ docs/sec | text-dedup: ~10K docs/sec |
| Memory (LSH) | <16GB for 1B docs | text-dedup: OOM at 100M |
| Memory (LSHBloom) | <1GB for 1B docs | 54x reduction |
| Accuracy (Jaccard) | >0.95 correlation | vs exact Jaccard |

### Benchmark Suite

```rust
// benches/dedup_bench.rs
fn bench_minhash(c: &mut Criterion) {
    let documents: Vec<String> = load_test_docs(10_000);
    let hasher = MinHasher::new(128);
    
    c.bench_function("minhash_signature", |b| {
        b.iter(|| {
            for doc in &documents {
                let tokens = tokenize(doc, 5);
                hasher.signature(&tokens);
            }
        })
    });
}

fn bench_dedup_pipeline(c: &mut Criterion) {
    let documents = load_test_docs(100_000);
    
    let mut group = c.benchmark_group("dedup_pipeline");
    group.sample_size(10);
    group.throughput(Throughput::Elements(documents.len() as u64));
    
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let mut dedup = Deduplicator::new(default_config());
            dedup.deduplicate(documents.iter().cloned())
        })
    });
}
```

## Testing Strategy

### Unit Tests
- MinHash: verify similarity estimates vs exact Jaccard
- LSH: verify candidate pair recall
- Union-Find: verify cluster correctness
- Tokenizer: edge cases (empty, unicode, very long)

### Integration Tests
- Deduplicate known dataset (C4 sample)
- Verify duplicate detection accuracy
- Roundtrip: dedup -> verify unique -> add known duplicate -> detect

### Property Tests
```rust
proptest! {
    #[test]
    fn minhash_similarity_bounds(
        doc1 in "[a-z ]{10,100}",
        doc2 in "[a-z ]{10,100}",
    ) {
        let hasher = MinHasher::new(128);
        let sig1 = hasher.hash_text(&doc1, 3);
        let sig2 = hasher.hash_text(&doc2, 3);
        let sim = MinHasher::similarity(&sig1, &sig2);
        
        prop_assert!(sim >= 0.0 && sim <= 1.0);
    }
    
    #[test]
    fn identical_docs_high_similarity(doc in "[a-z ]{10,100}") {
        let hasher = MinHasher::new(128);
        let sig1 = hasher.hash_text(&doc, 3);
        let sig2 = hasher.hash_text(&doc, 3);
        let sim = MinHasher::similarity(&sig1, &sig2);
        
        prop_assert!(sim > 0.99);
    }
}
```

## Error Handling

```rust
#[derive(Error, Debug)]
pub enum DedupError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
    
    #[error("Memory limit exceeded: {used} > {limit}")]
    MemoryLimit { used: u64, limit: u64 },
    
    #[error("Invalid document: {0}")]
    InvalidDocument(String),
}
```

## Performance Optimizations

### SIMD Hashing
```rust
// Use xxhash-rust with SIMD features
[dependencies]
xxhash-rust = { version = "0.8", features = ["xxh3", "xxh64", "nightly"] }
```

### Parallel Processing
```rust
use rayon::prelude::*;

// Parallel signature computation
let signatures: Vec<_> = documents
    .par_iter()
    .map(|doc| {
        let tokens = tokenize(&doc.text, config.ngram_size);
        (doc.id, hasher.signature(&tokens))
    })
    .collect();
```

### Memory-Mapped Files
```rust
use memmap2::MmapOptions;

// For very large datasets
let file = File::open(path)?;
let mmap = unsafe { MmapOptions::new().map(&file)? };
// Process mmap as byte slice
```

## References

- "Deduplicating Training Data Makes Language Models Better" (Google, 2021)
- LSHBloom (2024): 54x space reduction
- BigCode deduplication: 256 permutations, 0.85 threshold
- text-dedup: Reference Python implementation
- Google deduplicate-text-datasets: Rust suffix array implementation
