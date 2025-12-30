use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mithril_cache::cas::ContentStore;
use mithril_cache::eviction::{CacheEntry, LruCache};
use mithril_cache::keys::{CacheKey, DeviceClass, InputSpec};
use mithril_core::types::DType;
use tempfile::TempDir;

fn bench_cache_key_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_key");

    // Typical ML model input specs
    let inputs = vec![
        InputSpec::new(vec![1, 3, 224, 224], DType::Float32),
        InputSpec::new(vec![1, 1000], DType::Float32),
    ];

    group.bench_function("from_bytes", |b| {
        let graph_bytes = b"def forward(x): return x * 2 + 1";
        b.iter(|| {
            CacheKey::from_bytes(
                black_box(graph_bytes),
                black_box(inputs.clone()),
                black_box(DeviceClass::CudaAny),
            )
        })
    });

    group.bench_function("to_storage_key", |b| {
        let key = CacheKey::from_bytes(b"graph", inputs.clone(), DeviceClass::CudaAny);
        b.iter(|| black_box(&key).to_storage_key())
    });

    group.bench_function("serialization_roundtrip", |b| {
        let key = CacheKey::from_bytes(b"graph", inputs.clone(), DeviceClass::CudaAny);
        b.iter(|| {
            let bytes = black_box(&key).to_bytes();
            CacheKey::from_serialized_bytes(&bytes)
        })
    });

    group.finish();
}

fn bench_lru_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_cache");

    group.bench_function("put_1000_entries", |b| {
        b.iter(|| {
            let mut cache = LruCache::new(1024 * 1024 * 1024); // 1GB
            for i in 0..1000 {
                cache.put(CacheEntry::new(format!("key_{i}"), 1024));
            }
            black_box(cache.len())
        })
    });

    group.bench_function("get_hot", |b| {
        let mut cache = LruCache::new(1024 * 1024 * 1024);
        for i in 0..1000 {
            cache.put(CacheEntry::new(format!("key_{i}"), 1024));
        }

        b.iter(|| {
            // Access a "hot" key and check if it exists
            black_box(cache.get("key_500").is_some())
        })
    });

    group.bench_function("eviction_pressure", |b| {
        b.iter(|| {
            let mut cache = LruCache::new(10 * 1024); // 10KB limit
            for i in 0..1000 {
                cache.put(CacheEntry::new(format!("key_{i}"), 1024));
            }
            black_box(cache.len())
        })
    });

    group.finish();
}

fn bench_content_store(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("content_store");

    // Small content (typical cache key metadata)
    let small_content = vec![0u8; 256];
    group.throughput(Throughput::Bytes(small_content.len() as u64));

    group.bench_function("put_256b", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        b.to_async(&rt).iter(|| async {
            // Use different content each time to avoid caching
            let content: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
            black_box(store.put(&content).await.unwrap())
        })
    });

    group.bench_function("get_256b", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let address = rt.block_on(async { store.put(&small_content).await.unwrap() });

        b.to_async(&rt)
            .iter(|| async { black_box(store.get(&address).await.unwrap()) })
    });

    group.bench_function("exists", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let address = rt.block_on(async { store.put(&small_content).await.unwrap() });

        b.to_async(&rt)
            .iter(|| async { black_box(store.exists(&address).await.unwrap()) })
    });

    group.finish();

    // Benchmark larger content (typical compiled kernel)
    let mut group = c.benchmark_group("content_store_1mb");
    let large_content = vec![0u8; 1024 * 1024]; // 1MB
    group.throughput(Throughput::Bytes(large_content.len() as u64));

    group.bench_function("put_1mb", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        b.to_async(&rt)
            .iter(|| async { black_box(store.put(&large_content).await.unwrap()) })
    });

    group.bench_function("get_1mb", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let address = rt.block_on(async { store.put(&large_content).await.unwrap() });

        b.to_async(&rt)
            .iter(|| async { black_box(store.get(&address).await.unwrap()) })
    });

    group.finish();
}

fn bench_lookup_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("lookup_latency");

    // This benchmark measures the end-to-end lookup time
    // Target: <10ms
    group.bench_function("full_lookup_path", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        // Pre-populate with typical content
        let content = vec![0u8; 1024]; // 1KB typical metadata
        let address = rt.block_on(async { store.put(&content).await.unwrap() });

        b.to_async(&rt).iter(|| async {
            // 1. Check if exists
            let exists = store.exists(&address).await.unwrap();
            assert!(exists);

            // 2. Get content
            let data = store.get(&address).await.unwrap();
            black_box(data)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_key_generation,
    bench_lru_cache,
    bench_content_store,
    bench_lookup_latency
);
criterion_main!(benches);
