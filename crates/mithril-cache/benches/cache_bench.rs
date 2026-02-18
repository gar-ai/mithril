use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use mithril_cache::cas::ContentStore;
use mithril_cache::eviction::{CacheEntry, LruCache};
use mithril_cache::hooks::{CacheConfig, CacheManager};
use mithril_cache::keys::{CacheKey, DeviceClass, InputSpec};
use mithril_core::types::DType;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Cache Key Generation
// ---------------------------------------------------------------------------

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

    // Benchmark key generation with a large graph IR (realistic torch.compile payload)
    group.bench_function("from_bytes_large_graph", |b| {
        let large_graph: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();
        b.iter(|| {
            CacheKey::from_bytes(
                black_box(&large_graph),
                black_box(inputs.clone()),
                black_box(DeviceClass::CudaCompute {
                    major: 8,
                    minor: 0,
                }),
            )
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// LRU Cache
// ---------------------------------------------------------------------------

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

    group.bench_function("get_cold_miss", |b| {
        let mut cache = LruCache::new(1024 * 1024 * 1024);
        for i in 0..1000 {
            cache.put(CacheEntry::new(format!("key_{i}"), 1024));
        }

        b.iter(|| black_box(cache.get("nonexistent_key").is_none()))
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

    group.bench_function("peek_no_reorder", |b| {
        let mut cache = LruCache::new(1024 * 1024 * 1024);
        for i in 0..1000 {
            cache.put(CacheEntry::new(format!("key_{i}"), 1024));
        }

        b.iter(|| black_box(cache.peek("key_500").is_some()))
    });

    group.bench_function("contains_check", |b| {
        let mut cache = LruCache::new(1024 * 1024 * 1024);
        for i in 0..1000 {
            cache.put(CacheEntry::new(format!("key_{i}"), 1024));
        }

        b.iter(|| black_box(cache.contains("key_500")))
    });

    group.bench_function("remove_entry", |b| {
        b.iter(|| {
            let mut cache = LruCache::new(1024 * 1024 * 1024);
            for i in 0..100 {
                cache.put(CacheEntry::new(format!("key_{i}"), 1024));
            }
            // Remove all entries
            for i in 0..100 {
                cache.remove(&format!("key_{i}"));
            }
            black_box(cache.len())
        })
    });

    group.finish();
}

/// Benchmark LRU cache with varying numbers of entries to show scaling behavior.
fn bench_lru_cache_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_cache_scaling");

    for &num_entries in &[100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("put", num_entries),
            &num_entries,
            |b, &n| {
                b.iter(|| {
                    let mut cache = LruCache::new(u64::MAX);
                    for i in 0..n {
                        cache.put(CacheEntry::new(format!("key_{i}"), 1024));
                    }
                    black_box(cache.len())
                })
            },
        );
    }

    // Benchmark get on caches of different sizes
    for &num_entries in &[100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("get_in_populated", num_entries),
            &num_entries,
            |b, &n| {
                let mut cache = LruCache::new(u64::MAX);
                for i in 0..n {
                    cache.put(CacheEntry::new(format!("key_{i}"), 1024));
                }
                let target = format!("key_{}", n / 2);
                b.iter(|| black_box(cache.get(&target).is_some()))
            },
        );
    }

    group.finish();
}

/// Simulate a realistic workload: 80% cache hits, 20% new entries (Zipf-like).
fn bench_lru_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_mixed_workload");

    group.bench_function("80_hit_20_miss_1000ops", |b| {
        b.iter(|| {
            let mut cache = LruCache::new(512 * 1024); // 512KB
            // Pre-populate with 100 "warm" entries
            for i in 0..100 {
                cache.put(CacheEntry::new(format!("warm_{i}"), 1024));
            }

            let mut evicted_total = 0usize;
            for i in 0..1000 {
                if i % 5 == 0 {
                    // 20% miss -> insert new entry
                    let evicted =
                        cache.put(CacheEntry::new(format!("new_{i}"), 2048));
                    evicted_total += evicted.len();
                } else {
                    // 80% hit -> access existing entry
                    let key = format!("warm_{}", i % 100);
                    cache.get(&key);
                }
            }
            black_box((cache.len(), evicted_total))
        })
    });

    // Workload skewed toward a small number of hot keys (typical kernel reuse)
    group.bench_function("hot_set_reuse_1000ops", |b| {
        b.iter(|| {
            let mut cache = LruCache::new(1024 * 1024); // 1MB
            // Insert 200 entries
            for i in 0..200 {
                cache.put(CacheEntry::new(format!("kern_{i}"), 4096));
            }
            // Repeatedly access only the 10 hottest keys (simulates model reuse)
            for i in 0..1000 {
                let key = format!("kern_{}", i % 10);
                cache.get(&key);
            }
            black_box(cache.stats().total_hits)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Content-Addressable Store
// ---------------------------------------------------------------------------

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
            // Use different content each time to avoid CAS dedup short-circuit
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

    group.bench_function("compute_address_256b", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        b.iter(|| black_box(store.compute_address(&small_content)))
    });

    group.bench_function("put_idempotent_256b", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        // Pre-write so that subsequent puts are idempotent (CAS dedup path)
        rt.block_on(async { store.put(&small_content).await.unwrap() });

        b.to_async(&rt)
            .iter(|| async { black_box(store.put(&small_content).await.unwrap()) })
    });

    group.bench_function("delete_256b", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        b.to_async(&rt).iter(|| async {
            let addr = store.put(&small_content).await.unwrap();
            black_box(store.delete(&addr).await.unwrap())
        })
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

    group.bench_function("compute_address_1mb", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        b.iter(|| black_box(store.compute_address(&large_content)))
    });

    group.finish();
}

/// Benchmark content store across a range of payload sizes to characterise
/// throughput scaling (important for understanding Triton kernel artifact sizes).
fn bench_content_store_scaling(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("content_store_scaling");

    for &size in &[256usize, 4096, 65536, 262144, 1048576] {
        let label = match size {
            256 => "256B",
            4096 => "4KB",
            65536 => "64KB",
            262144 => "256KB",
            1048576 => "1MB",
            _ => unreachable!(),
        };

        let content: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("put", label), &content, |b, data| {
            let tmp = TempDir::new().unwrap();
            let store = ContentStore::new(tmp.path()).unwrap();

            b.to_async(&rt)
                .iter(|| async { black_box(store.put(data).await.unwrap()) })
        });

        group.bench_with_input(BenchmarkId::new("get", label), &content, |b, data| {
            let tmp = TempDir::new().unwrap();
            let store = ContentStore::new(tmp.path()).unwrap();

            let address = rt.block_on(async { store.put(data).await.unwrap() });

            b.to_async(&rt)
                .iter(|| async { black_box(store.get(&address).await.unwrap()) })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// CacheManager init and lookup
// ---------------------------------------------------------------------------

fn bench_cache_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_manager");

    group.bench_function("init", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let config = CacheConfig::new(tmp.path()).with_max_size_bytes(10 * 1024 * 1024 * 1024);
            let manager = CacheManager::init(config).unwrap();
            black_box(manager.root_dir().to_path_buf())
        })
    });

    group.bench_function("record_entry", |b| {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path()).with_max_size_bytes(10 * 1024 * 1024 * 1024);
        let mut manager = CacheManager::init(config).unwrap();
        let mut i = 0u64;

        b.iter(|| {
            i += 1;
            let evicted = manager.record_entry(format!("kernel_{i}"), 4096);
            black_box(evicted)
        })
    });

    group.bench_function("record_access_hit", |b| {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path()).with_max_size_bytes(10 * 1024 * 1024 * 1024);
        let mut manager = CacheManager::init(config).unwrap();

        // Pre-populate
        for i in 0..100 {
            manager.record_entry(format!("kernel_{i}"), 4096);
        }

        b.iter(|| black_box(manager.record_access("kernel_50")))
    });

    group.bench_function("record_access_miss", |b| {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path()).with_max_size_bytes(10 * 1024 * 1024 * 1024);
        let mut manager = CacheManager::init(config).unwrap();

        for i in 0..100 {
            manager.record_entry(format!("kernel_{i}"), 4096);
        }

        b.iter(|| black_box(manager.record_access("nonexistent_kernel")))
    });

    group.bench_function("stats", |b| {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::new(tmp.path()).with_max_size_bytes(10 * 1024 * 1024 * 1024);
        let mut manager = CacheManager::init(config).unwrap();

        for i in 0..500 {
            manager.record_entry(format!("kernel_{i}"), 4096);
        }

        b.iter(|| black_box(manager.stats()))
    });

    group.bench_function("eviction_on_record_entry", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let config = CacheConfig::new(tmp.path())
                .with_max_size_bytes(50 * 1024); // 50KB limit
            let mut manager = CacheManager::init(config).unwrap();

            let mut total_evicted = 0usize;
            for i in 0..200 {
                let evicted = manager.record_entry(format!("kernel_{i}"), 4096);
                total_evicted += evicted.len();
            }
            black_box(total_evicted)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Lookup Latency (end-to-end)
// ---------------------------------------------------------------------------

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

    // Full path with key generation + CAS lookup (realistic torch.compile path)
    group.bench_function("keygen_then_cas_lookup", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let inputs = vec![
            InputSpec::new(vec![1, 3, 224, 224], DType::Float32),
            InputSpec::new(vec![1, 1000], DType::Float32),
        ];

        // Store a compiled kernel artifact
        let kernel_bytes = vec![0xABu8; 8192]; // 8KB compiled kernel
        let address = rt.block_on(async { store.put(&kernel_bytes).await.unwrap() });

        b.to_async(&rt).iter(|| async {
            // Step 1: generate cache key from graph IR
            let key = CacheKey::from_bytes(
                b"def forward(self, x): return self.linear(x)",
                inputs.clone(),
                DeviceClass::CudaAny,
            );
            let _storage_key = key.to_storage_key();

            // Step 2: check if artifact exists in CAS and retrieve it
            let data = store.get(&address).await.unwrap();
            black_box(data)
        })
    });

    // Lookup miss path
    group.bench_function("miss_path", |b| {
        let tmp = TempDir::new().unwrap();
        let store = ContentStore::new(tmp.path()).unwrap();

        let fake_address = "0".repeat(64);

        b.to_async(&rt).iter(|| async {
            let exists = store.exists(&fake_address).await.unwrap();
            assert!(!exists);
            black_box(exists)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_cache_key_generation,
    bench_lru_cache,
    bench_lru_cache_scaling,
    bench_lru_mixed_workload,
    bench_content_store,
    bench_content_store_scaling,
    bench_cache_manager,
    bench_lookup_latency,
);
criterion_main!(benches);
