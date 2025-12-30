use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mithril_dedup::{lsh::LshIndex, minhash::MinHasher, Deduplicator};

fn generate_documents(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            format!(
                "Document number {} contains information about topic {} with various keywords and text content for testing deduplication algorithms effectively",
                i,
                i % 10
            )
        })
        .collect()
}

fn generate_documents_with_duplicates(count: usize, dup_ratio: f64) -> Vec<String> {
    let unique_count = ((1.0 - dup_ratio) * count as f64) as usize;
    let mut docs: Vec<String> = (0..unique_count)
        .map(|i| {
            format!(
                "Document number {} contains information about topic {} with various keywords and text content for testing",
                i,
                i % 10
            )
        })
        .collect();

    // Add duplicates
    let dup_count = count - unique_count;
    for i in 0..dup_count {
        docs.push(docs[i % unique_count].clone());
    }

    docs
}

fn bench_minhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash");

    let hasher = MinHasher::new(128);
    let text = "The quick brown fox jumps over the lazy dog and this is some additional text to make it longer for realistic benchmarking purposes";

    group.bench_function("signature_128", |b| {
        b.iter(|| hasher.signature_from_text(black_box(text)))
    });

    let hasher_256 = MinHasher::new(256);
    group.bench_function("signature_256", |b| {
        b.iter(|| hasher_256.signature_from_text(black_box(text)))
    });

    // Benchmark tokenization
    group.bench_function("tokenize", |b| b.iter(|| hasher.tokenize(black_box(text))));

    // Benchmark similarity computation
    let sig1 = hasher.signature_from_text(text);
    let sig2 =
        hasher.signature_from_text("Different text with some overlap jumps over the lazy dog");

    group.bench_function("similarity", |b| {
        b.iter(|| MinHasher::similarity(black_box(&sig1), black_box(&sig2)))
    });

    group.finish();
}

fn bench_lsh(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh");

    let hasher = MinHasher::new(128);
    let docs = generate_documents(1000);
    let signatures: Vec<_> = docs.iter().map(|d| hasher.signature_from_text(d)).collect();

    group.bench_function("insert_1000", |b| {
        b.iter(|| {
            let mut lsh = LshIndex::with_threshold(128, 0.85);
            for (i, sig) in signatures.iter().enumerate() {
                lsh.insert(i as u64, sig);
            }
            lsh
        })
    });

    // Benchmark candidate generation
    let mut lsh = LshIndex::with_threshold(128, 0.85);
    for (i, sig) in signatures.iter().enumerate() {
        lsh.insert(i as u64, sig);
    }

    group.bench_function("candidates_1000", |b| b.iter(|| lsh.candidates_vec()));

    group.finish();
}

fn bench_dedup_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("dedup_pipeline");
    group.sample_size(10);

    for size in [100, 1000, 10000] {
        let docs = generate_documents_with_duplicates(size, 0.3);
        let texts: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &texts, |b, texts| {
            let dedup = Deduplicator::default();
            b.iter(|| dedup.deduplicate_texts(black_box(texts)))
        });
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.sample_size(10);

    // Benchmark to verify 100K docs/sec target
    let docs = generate_documents(10000);
    let texts: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

    group.throughput(Throughput::Elements(10000));
    group.bench_function("10k_docs", |b| {
        let dedup = Deduplicator::default();
        b.iter(|| dedup.deduplicate_texts(black_box(&texts)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_minhash,
    bench_lsh,
    bench_dedup_pipeline,
    bench_throughput
);
criterion_main!(benches);
