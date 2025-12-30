//! LRU cache eviction for local cache management.
//!
//! Implements Least Recently Used (LRU) eviction to keep the cache
//! within size limits while maximizing cache hit rates.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// A cache entry with metadata for eviction decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The storage key/address for this entry.
    pub key: String,
    /// Size in bytes.
    pub size: u64,
    /// When the entry was created.
    #[serde(skip)]
    #[allow(dead_code)]
    created_at: Option<Instant>,
    /// When the entry was last accessed.
    #[serde(skip)]
    last_accessed: Option<Instant>,
    /// Number of times this entry has been accessed.
    pub hit_count: u64,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub fn new(key: String, size: u64) -> Self {
        let now = Instant::now();
        Self {
            key,
            size,
            created_at: Some(now),
            last_accessed: Some(now),
            hit_count: 0,
        }
    }

    /// Record an access to this entry.
    pub fn record_access(&mut self) {
        self.last_accessed = Some(Instant::now());
        self.hit_count += 1;
    }

    /// Get time since last access.
    #[must_use]
    pub fn time_since_access(&self) -> Option<Duration> {
        self.last_accessed.map(|t| t.elapsed())
    }
}

/// Node in the LRU linked list.
#[allow(dead_code)]
struct LruNode {
    key: String,
    prev: Option<String>,
    next: Option<String>,
}

/// LRU cache with size-based eviction.
///
/// Tracks cache entries and their access patterns to evict
/// least-recently-used entries when the cache exceeds its size limit.
///
/// # Example
///
/// ```
/// use mithril_cache::eviction::{LruCache, CacheEntry};
///
/// let mut cache = LruCache::new(1024); // 1KB limit
///
/// cache.put(CacheEntry::new("key1".to_string(), 256));
/// cache.put(CacheEntry::new("key2".to_string(), 256));
///
/// // Access key1 to make it recently used
/// cache.get("key1");
///
/// // When we exceed capacity, least recently used (key2) will be evicted
/// cache.put(CacheEntry::new("key3".to_string(), 768));
///
/// assert!(cache.get("key1").is_some());
/// assert!(cache.get("key2").is_none()); // evicted
/// ```
pub struct LruCache {
    /// Maximum cache size in bytes.
    max_size_bytes: u64,
    /// Current cache size in bytes.
    current_size: u64,
    /// Cache entries by key.
    entries: HashMap<String, CacheEntry>,
    /// LRU ordering: nodes keyed by entry key.
    nodes: HashMap<String, LruNode>,
    /// Head of LRU list (most recently used).
    head: Option<String>,
    /// Tail of LRU list (least recently used).
    tail: Option<String>,
}

impl LruCache {
    /// Create a new LRU cache with the given size limit.
    #[must_use]
    pub fn new(max_size_bytes: u64) -> Self {
        Self {
            max_size_bytes,
            current_size: 0,
            entries: HashMap::new(),
            nodes: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    /// Get the maximum cache size in bytes.
    #[must_use]
    pub fn max_size(&self) -> u64 {
        self.max_size_bytes
    }

    /// Get the current cache size in bytes.
    #[must_use]
    pub fn current_size(&self) -> u64 {
        self.current_size
    }

    /// Get the number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by key, updating its access time.
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        if self.entries.contains_key(key) {
            // Move to front (most recently used)
            self.move_to_front(key);

            // Update access time and hit count
            if let Some(entry) = self.entries.get_mut(key) {
                entry.record_access();
            }

            self.entries.get(key)
        } else {
            None
        }
    }

    /// Check if an entry exists without updating access time.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Peek at an entry without updating its access time.
    #[must_use]
    pub fn peek(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    /// Put an entry into the cache.
    ///
    /// If the entry already exists, it's updated and moved to front.
    /// If adding the entry would exceed capacity, least-recently-used
    /// entries are evicted until there's room.
    ///
    /// Returns a list of evicted keys.
    pub fn put(&mut self, entry: CacheEntry) -> Vec<String> {
        let key = entry.key.clone();
        let size = entry.size;

        // If entry already exists, remove it first
        if let Some(old) = self.entries.remove(&key) {
            self.current_size -= old.size;
            self.remove_node(&key);
        }

        // Evict entries if needed
        let evicted = self.evict_if_needed(size);

        // Add the new entry
        self.entries.insert(key.clone(), entry);
        self.current_size += size;

        // Add to front of LRU list
        self.add_to_front(&key);

        evicted
    }

    /// Remove an entry from the cache.
    ///
    /// Returns the removed entry if it existed.
    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size -= entry.size;
            self.remove_node(key);
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.nodes.clear();
        self.head = None;
        self.tail = None;
        self.current_size = 0;
    }

    /// Get all keys in LRU order (most recent first).
    #[must_use]
    pub fn keys_lru_order(&self) -> Vec<String> {
        let mut keys = Vec::with_capacity(self.entries.len());
        let mut current = self.head.clone();

        while let Some(key) = current {
            keys.push(key.clone());
            current = self.nodes.get(&key).and_then(|n| n.next.clone());
        }

        keys
    }

    /// Get cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let total_hits: u64 = self.entries.values().map(|e| e.hit_count).sum();

        CacheStats {
            entry_count: self.entries.len(),
            current_size_bytes: self.current_size,
            max_size_bytes: self.max_size_bytes,
            utilization: if self.max_size_bytes > 0 {
                self.current_size as f64 / self.max_size_bytes as f64
            } else {
                0.0
            },
            total_hits,
        }
    }

    /// Evict entries until we have room for `needed_bytes`.
    ///
    /// Returns a list of evicted keys.
    fn evict_if_needed(&mut self, needed_bytes: u64) -> Vec<String> {
        let mut evicted = Vec::new();

        // If the single entry is bigger than max, we still add it
        // but evict everything else first
        let target = self.max_size_bytes.saturating_sub(needed_bytes);

        while self.current_size > target {
            if let Some(key) = self.tail.clone() {
                if let Some(entry) = self.entries.remove(&key) {
                    self.current_size -= entry.size;
                    self.remove_node(&key);
                    evicted.push(key);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        evicted
    }

    /// Add a key to the front of the LRU list.
    fn add_to_front(&mut self, key: &str) {
        let node = LruNode {
            key: key.to_string(),
            prev: None,
            next: self.head.clone(),
        };

        if let Some(old_head) = &self.head {
            if let Some(old_node) = self.nodes.get_mut(old_head) {
                old_node.prev = Some(key.to_string());
            }
        }

        self.nodes.insert(key.to_string(), node);
        self.head = Some(key.to_string());

        if self.tail.is_none() {
            self.tail = Some(key.to_string());
        }
    }

    /// Remove a key from the LRU list.
    fn remove_node(&mut self, key: &str) {
        if let Some(node) = self.nodes.remove(key) {
            // Update prev node's next pointer
            if let Some(prev_key) = &node.prev {
                if let Some(prev_node) = self.nodes.get_mut(prev_key) {
                    prev_node.next = node.next.clone();
                }
            } else {
                // This was the head
                self.head = node.next.clone();
            }

            // Update next node's prev pointer
            if let Some(next_key) = &node.next {
                if let Some(next_node) = self.nodes.get_mut(next_key) {
                    next_node.prev = node.prev.clone();
                }
            } else {
                // This was the tail
                self.tail = node.prev.clone();
            }
        }
    }

    /// Move a key to the front of the LRU list.
    fn move_to_front(&mut self, key: &str) {
        if self.head.as_deref() == Some(key) {
            return; // Already at front
        }

        self.remove_node(key);
        self.add_to_front(key);
    }
}

/// Cache statistics.
#[derive(Clone, Debug)]
pub struct CacheStats {
    /// Number of entries in the cache.
    pub entry_count: usize,
    /// Current cache size in bytes.
    pub current_size_bytes: u64,
    /// Maximum cache size in bytes.
    pub max_size_bytes: u64,
    /// Cache utilization (0.0 to 1.0).
    pub utilization: f64,
    /// Total number of cache hits.
    pub total_hits: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic_operations() {
        let mut cache = LruCache::new(1000);

        // Add entries
        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 200));
        cache.put(CacheEntry::new("key3".to_string(), 300));

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.current_size(), 600);

        // Get entries
        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LruCache::new(500);

        // Fill cache
        cache.put(CacheEntry::new("key1".to_string(), 200));
        cache.put(CacheEntry::new("key2".to_string(), 200));

        assert_eq!(cache.current_size(), 400);
        assert_eq!(cache.len(), 2);

        // Adding this should evict key1 (least recently used)
        let evicted = cache.put(CacheEntry::new("key3".to_string(), 200));

        assert_eq!(evicted, vec!["key1"]);
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_lru_access_updates_order() {
        let mut cache = LruCache::new(500);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 100));
        cache.put(CacheEntry::new("key3".to_string(), 100));

        // Order should be: key3, key2, key1 (most to least recent)
        assert_eq!(cache.keys_lru_order(), vec!["key3", "key2", "key1"]);

        // Access key1 to make it most recent
        cache.get("key1");

        // Order should now be: key1, key3, key2
        assert_eq!(cache.keys_lru_order(), vec!["key1", "key3", "key2"]);

        // Adding new entry should evict key2 (now least recent)
        let evicted = cache.put(CacheEntry::new("key4".to_string(), 300));

        assert_eq!(evicted, vec!["key2"]);
        assert!(cache.contains("key1"));
        assert!(!cache.contains("key2"));
        assert!(cache.contains("key3"));
        assert!(cache.contains("key4"));
    }

    #[test]
    fn test_lru_update_existing() {
        let mut cache = LruCache::new(500);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 100));

        assert_eq!(cache.current_size(), 200);

        // Update key1 with larger size
        cache.put(CacheEntry::new("key1".to_string(), 300));

        assert_eq!(cache.current_size(), 400);
        assert_eq!(cache.len(), 2);

        // key1 should now be most recent
        assert_eq!(cache.keys_lru_order(), vec!["key1", "key2"]);
    }

    #[test]
    fn test_lru_remove() {
        let mut cache = LruCache::new(500);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 200));

        assert_eq!(cache.current_size(), 300);

        let removed = cache.remove("key1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().size, 100);

        assert_eq!(cache.current_size(), 200);
        assert_eq!(cache.len(), 1);
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_lru_clear() {
        let mut cache = LruCache::new(500);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 200));

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_size(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_peek_does_not_update_order() {
        let mut cache = LruCache::new(500);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 100));
        cache.put(CacheEntry::new("key3".to_string(), 100));

        // Order: key3, key2, key1
        assert_eq!(cache.keys_lru_order(), vec!["key3", "key2", "key1"]);

        // Peek at key1 - should not update order
        assert!(cache.peek("key1").is_some());
        assert_eq!(cache.keys_lru_order(), vec!["key3", "key2", "key1"]);

        // Get at key1 - should update order
        cache.get("key1");
        assert_eq!(cache.keys_lru_order(), vec!["key1", "key3", "key2"]);
    }

    #[test]
    fn test_lru_stats() {
        let mut cache = LruCache::new(1000);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 200));

        // Access key1 a few times
        cache.get("key1");
        cache.get("key1");
        cache.get("key1");

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.current_size_bytes, 300);
        assert_eq!(stats.max_size_bytes, 1000);
        assert!((stats.utilization - 0.3).abs() < 0.01);
        assert_eq!(stats.total_hits, 3);
    }

    #[test]
    fn test_lru_evicts_multiple() {
        let mut cache = LruCache::new(400);

        cache.put(CacheEntry::new("key1".to_string(), 100));
        cache.put(CacheEntry::new("key2".to_string(), 100));
        cache.put(CacheEntry::new("key3".to_string(), 100));
        cache.put(CacheEntry::new("key4".to_string(), 100));

        assert_eq!(cache.current_size(), 400);

        // Adding a 300 byte entry should evict key1, key2, key3
        let evicted = cache.put(CacheEntry::new("key5".to_string(), 300));

        assert_eq!(evicted.len(), 3);
        assert!(evicted.contains(&"key1".to_string()));
        assert!(evicted.contains(&"key2".to_string()));
        assert!(evicted.contains(&"key3".to_string()));

        assert!(cache.contains("key4"));
        assert!(cache.contains("key5"));
    }

    #[test]
    fn test_cache_entry_access_tracking() {
        let mut entry = CacheEntry::new("test".to_string(), 100);

        assert_eq!(entry.hit_count, 0);

        entry.record_access();
        assert_eq!(entry.hit_count, 1);

        entry.record_access();
        entry.record_access();
        assert_eq!(entry.hit_count, 3);

        // Time since access should be very small
        let elapsed = entry.time_since_access().unwrap();
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn test_lru_single_item_larger_than_max() {
        let mut cache = LruCache::new(100);

        // Add a small item first
        cache.put(CacheEntry::new("small".to_string(), 50));

        // Add an item larger than max - should evict everything and still add it
        let evicted = cache.put(CacheEntry::new("large".to_string(), 200));

        assert_eq!(evicted, vec!["small"]);
        assert!(cache.contains("large"));
        assert_eq!(cache.current_size(), 200);
    }
}
