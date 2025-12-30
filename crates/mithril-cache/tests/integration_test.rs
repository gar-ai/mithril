//! Integration tests for mithril-cache.
//!
//! Tests end-to-end cache workflows: store, retrieve, evict.

use mithril_cache::{
    CacheConfig, CacheEntry, CacheKey, ContentStore, DeviceClass, InputSpec, LruCache,
};
use mithril_core::types::DType;
use std::time::Duration;
use tempfile::TempDir;

/// Generate synthetic compilation artifact data.
fn generate_artifact(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

// ============================================================================
// ContentStore tests (async)
// ============================================================================

#[tokio::test]
async fn test_content_store_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let data = generate_artifact(1024);
    let address = store.put(&data).await.unwrap();

    let retrieved = store.get(&address).await.unwrap().unwrap();
    assert_eq!(data, retrieved);
}

#[tokio::test]
async fn test_content_store_deduplication() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let data = generate_artifact(1024);

    // Store same data twice
    let addr1 = store.put(&data).await.unwrap();
    let addr2 = store.put(&data).await.unwrap();

    // Should get same address (content-addressable)
    assert_eq!(addr1, addr2);
}

#[tokio::test]
async fn test_content_store_multiple_artifacts() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let artifacts: Vec<Vec<u8>> = (0..10).map(|i| generate_artifact(1024 * (i + 1))).collect();

    let mut addresses = Vec::new();
    for data in &artifacts {
        let addr = store.put(data).await.unwrap();
        addresses.push(addr);
    }

    // Retrieve all and verify
    for (addr, original) in addresses.iter().zip(artifacts.iter()) {
        let retrieved = store.get(addr).await.unwrap().unwrap();
        assert_eq!(*original, retrieved);
    }
}

#[tokio::test]
async fn test_content_store_exists() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let data = generate_artifact(1024);
    let address = store.put(&data).await.unwrap();

    assert!(store.exists(&address).await.unwrap());
    assert!(!store.exists("nonexistent").await.unwrap());
}

#[tokio::test]
async fn test_content_store_delete() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let data = generate_artifact(1024);
    let address = store.put(&data).await.unwrap();

    assert!(store.exists(&address).await.unwrap());
    store.delete(&address).await.unwrap();
    assert!(!store.exists(&address).await.unwrap());
}

#[tokio::test]
async fn test_content_store_size() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let data = generate_artifact(1024);
    let address = store.put(&data).await.unwrap();

    let size = store.size(&address).await.unwrap();
    assert_eq!(size, Some(1024));

    assert_eq!(store.size("nonexistent").await.unwrap(), None);
}

#[tokio::test]
async fn test_large_artifact_handling() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    // 10MB artifact (typical compiled kernel size)
    let data = generate_artifact(10 * 1024 * 1024);
    let address = store.put(&data).await.unwrap();

    let retrieved = store.get(&address).await.unwrap().unwrap();
    assert_eq!(data, retrieved);
}

// ============================================================================
// LruCache tests (sync)
// ============================================================================

#[test]
fn test_lru_cache_basic() {
    let mut cache = LruCache::new(10 * 1024 * 1024); // 10MB

    let entry = CacheEntry::new("key1".to_string(), 1024);
    cache.put(entry);

    assert!(cache.get("key1").is_some());
    assert!(cache.get("key2").is_none());
}

#[test]
fn test_lru_cache_eviction() {
    // Small cache that can only hold ~2KB
    let mut cache = LruCache::new(2048);

    // Add entries that total more than cache size
    for i in 0..5 {
        let entry = CacheEntry::new(format!("key{}", i), 1024); // Each is 1KB
        cache.put(entry);
    }

    // Oldest entries should be evicted
    assert!(cache.current_size() <= 2048);
}

#[test]
fn test_lru_order() {
    let mut cache = LruCache::new(3072); // Holds ~3 entries

    // Add 3 entries
    cache.put(CacheEntry::new("a".to_string(), 1000));
    cache.put(CacheEntry::new("b".to_string(), 1000));
    cache.put(CacheEntry::new("c".to_string(), 1000));

    // Access 'a' to make it recently used
    cache.get("a");

    // Add new entry - should evict 'b' (oldest untouched)
    cache.put(CacheEntry::new("d".to_string(), 1000));

    // 'a' and 'd' should still be there
    assert!(cache.get("a").is_some());
    assert!(cache.get("d").is_some());
}

#[test]
fn test_lru_clear() {
    let mut cache = LruCache::new(10240);

    cache.put(CacheEntry::new("key1".to_string(), 1024));
    cache.put(CacheEntry::new("key2".to_string(), 1024));

    assert_eq!(cache.len(), 2);

    cache.clear();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// ============================================================================
// CacheKey tests (sync)
// ============================================================================

#[test]
fn test_cache_key_generation() {
    let key = CacheKey::new(
        [1u8; 32],
        vec![InputSpec::static_shape(
            vec![1, 3, 224, 224],
            DType::Float32,
        )],
        DeviceClass::cuda(8, 0),
    );

    let storage_key = key.to_storage_key();
    assert!(!storage_key.is_empty());
    assert_eq!(storage_key.len(), 32); // Blake3 first 16 bytes = 32 hex chars
}

#[test]
fn test_cache_key_deterministic() {
    let key1 = CacheKey::new(
        [42u8; 32],
        vec![InputSpec::static_shape(vec![1, 512], DType::Float32)],
        DeviceClass::Cpu,
    );

    let key2 = CacheKey::new(
        [42u8; 32],
        vec![InputSpec::static_shape(vec![1, 512], DType::Float32)],
        DeviceClass::Cpu,
    );

    assert_eq!(key1.to_storage_key(), key2.to_storage_key());
}

#[test]
fn test_cache_key_different_inputs() {
    let key1 = CacheKey::new(
        [1u8; 32],
        vec![InputSpec::static_shape(vec![1, 512], DType::Float32)],
        DeviceClass::Cpu,
    );

    let key2 = CacheKey::new(
        [1u8; 32],
        vec![InputSpec::static_shape(vec![2, 512], DType::Float32)], // Different batch
        DeviceClass::Cpu,
    );

    assert_ne!(key1.to_storage_key(), key2.to_storage_key());
}

#[test]
fn test_cache_key_from_bytes() {
    let graph_ir = b"def forward(x): return x * 2";
    let key = CacheKey::from_bytes(
        graph_ir,
        vec![InputSpec::static_shape(vec![1, 10], DType::Float32)],
        DeviceClass::Cpu,
    );

    let storage_key = key.to_storage_key();
    assert!(!storage_key.is_empty());
}

#[test]
fn test_cache_key_serialization() {
    let key = CacheKey::new(
        [99u8; 32],
        vec![
            InputSpec::static_shape(vec![1, 3, 224, 224], DType::Float32),
            InputSpec::static_shape(vec![1, 1000], DType::Float32),
        ],
        DeviceClass::cuda(8, 6),
    );

    let bytes = key.to_bytes();
    let restored = CacheKey::from_serialized_bytes(&bytes).unwrap();

    assert_eq!(key.to_storage_key(), restored.to_storage_key());
}

// ============================================================================
// CacheManager tests (sync)
// ============================================================================

#[test]
fn test_cache_manager_init() {
    let temp_dir = TempDir::new().unwrap();
    let config = CacheConfig::new(temp_dir.path());
    let manager = mithril_cache::init(config).unwrap();

    // Should create expected directories
    assert!(manager.inductor_dir().exists());
    assert!(manager.triton_dir().exists());
}

#[test]
fn test_cache_manager_clear() {
    let temp_dir = TempDir::new().unwrap();
    let config = CacheConfig::new(temp_dir.path());
    let mut manager = mithril_cache::init(config).unwrap();

    // Create a file in the cache
    std::fs::write(manager.inductor_dir().join("test.txt"), "test").unwrap();

    manager.clear().unwrap();

    // Directory should be empty
    let entries: Vec<_> = std::fs::read_dir(manager.inductor_dir()).unwrap().collect();
    assert!(entries.is_empty());
}

#[test]
fn test_cache_manager_record_entry() {
    let temp_dir = TempDir::new().unwrap();
    let config = CacheConfig::new(temp_dir.path());
    let mut manager = mithril_cache::init(config).unwrap();

    // Record some entries
    manager.record_entry("artifact1".to_string(), 1024);
    manager.record_entry("artifact2".to_string(), 2048);

    let stats = manager.stats();
    assert_eq!(stats.lru_entry_count, 2);
    assert_eq!(stats.lru_size_bytes, 3072);
}

#[test]
fn test_cache_manager_record_access() {
    let temp_dir = TempDir::new().unwrap();
    let config = CacheConfig::new(temp_dir.path());
    let mut manager = mithril_cache::init(config).unwrap();

    manager.record_entry("artifact1".to_string(), 1024);

    // Access the entry
    let found = manager.record_access("artifact1");
    assert!(found);

    // Access non-existent entry
    let not_found = manager.record_access("nonexistent");
    assert!(!not_found);
}

#[test]
fn test_cache_config_builder() {
    let config = CacheConfig::new("/tmp/test-cache")
        .with_max_size_gb(20)
        .with_inductor(true)
        .with_triton(false);

    assert_eq!(config.max_size_bytes, 20 * 1024 * 1024 * 1024);
    assert!(config.enable_inductor);
    assert!(!config.enable_triton);
}

// ============================================================================
// Performance tests
// ============================================================================

#[tokio::test]
async fn test_lookup_latency() {
    let temp_dir = TempDir::new().unwrap();
    let store = ContentStore::new(temp_dir.path()).unwrap();

    let data = generate_artifact(100 * 1024); // 100KB artifact
    let address = store.put(&data).await.unwrap();

    // Measure lookup latency
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = store.get(&address).await;
    }
    let elapsed = start.elapsed();
    let avg_latency = elapsed / 100;

    // Target is <10ms per METRICS.md
    assert!(
        avg_latency < Duration::from_millis(10),
        "Average lookup latency {:?} exceeds 10ms target",
        avg_latency
    );
}

// ============================================================================
// Device class tests
// ============================================================================

#[test]
fn test_device_class_compatibility() {
    // CudaAny is compatible with any CUDA
    assert!(DeviceClass::CudaAny.is_compatible_with(&DeviceClass::CudaAny));
    assert!(DeviceClass::CudaAny.is_compatible_with(&DeviceClass::cuda(8, 0)));
    assert!(DeviceClass::cuda(8, 0).is_compatible_with(&DeviceClass::CudaAny));

    // Same compute capability
    assert!(DeviceClass::cuda(8, 0).is_compatible_with(&DeviceClass::cuda(8, 0)));

    // Higher compute capability is compatible with lower
    assert!(DeviceClass::cuda(8, 6).is_compatible_with(&DeviceClass::cuda(8, 0)));
    assert!(DeviceClass::cuda(9, 0).is_compatible_with(&DeviceClass::cuda(8, 6)));

    // CPU only compatible with CPU
    assert!(DeviceClass::Cpu.is_compatible_with(&DeviceClass::Cpu));
    assert!(!DeviceClass::Cpu.is_compatible_with(&DeviceClass::CudaAny));
}

#[test]
fn test_input_spec_dynamic() {
    let spec = InputSpec::dynamic_batch(vec![1, 3, 224, 224], DType::Float32);
    assert!(spec.is_dynamic());
    assert_eq!(spec.shape[0], -1); // First dimension is dynamic
    assert!(spec.numel().is_none()); // Can't compute numel for dynamic
}

#[test]
fn test_input_spec_static() {
    let spec = InputSpec::static_shape(vec![1, 3, 224, 224], DType::Float32);
    assert!(!spec.is_dynamic());
    assert_eq!(spec.numel(), Some(1 * 3 * 224 * 224));
}
