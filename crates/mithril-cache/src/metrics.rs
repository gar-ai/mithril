//! Cache metrics collection and reporting.
//!
//! Provides observability into cache performance with support for
//! Prometheus and JSON export formats.
//!
//! ## Example
//!
//! ```rust
//! use mithril_cache::metrics::CacheMetrics;
//!
//! let metrics = CacheMetrics::new();
//!
//! // Record cache operations
//! metrics.record_hit(1024);
//! metrics.record_miss();
//! metrics.record_eviction(512);
//!
//! // Export for monitoring
//! println!("{}", metrics.export_prometheus());
//! println!("{}", metrics.export_json());
//! ```

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Thread-safe cache metrics collector.
///
/// All operations are lock-free and suitable for high-performance scenarios.
#[derive(Debug)]
pub struct CacheMetrics {
    /// Number of cache hits
    hits: AtomicU64,
    /// Number of cache misses
    misses: AtomicU64,
    /// Number of cache evictions
    evictions: AtomicU64,
    /// Total bytes served from cache
    bytes_hit: AtomicU64,
    /// Total bytes written to cache
    bytes_written: AtomicU64,
    /// Total bytes evicted from cache
    bytes_evicted: AtomicU64,
    /// Estimated compilation time saved (milliseconds)
    time_saved_ms: AtomicU64,
    /// Creation time for uptime calculation
    created_at: Instant,
}

impl CacheMetrics {
    /// Create a new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            bytes_hit: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_evicted: AtomicU64::new(0),
            time_saved_ms: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Record a cache hit.
    ///
    /// # Arguments
    /// * `bytes` - Size of the cached item in bytes
    pub fn record_hit(&self, bytes: u64) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.bytes_hit.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a cache hit with estimated time saved.
    ///
    /// # Arguments
    /// * `bytes` - Size of the cached item in bytes
    /// * `time_saved_ms` - Estimated compilation time saved in milliseconds
    pub fn record_hit_with_time(&self, bytes: u64, time_saved_ms: u64) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.bytes_hit.fetch_add(bytes, Ordering::Relaxed);
        self.time_saved_ms
            .fetch_add(time_saved_ms, Ordering::Relaxed);
    }

    /// Record a cache miss.
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache write.
    ///
    /// # Arguments
    /// * `bytes` - Size of the item written in bytes
    pub fn record_write(&self, bytes: u64) {
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a cache eviction.
    ///
    /// # Arguments
    /// * `bytes` - Size of the evicted item in bytes
    pub fn record_eviction(&self, bytes: u64) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.bytes_evicted.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get the current snapshot of metrics.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        MetricsSnapshot {
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            bytes_hit: self.bytes_hit.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_evicted: self.bytes_evicted.load(Ordering::Relaxed),
            time_saved_ms: self.time_saved_ms.load(Ordering::Relaxed),
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
            uptime_secs: self.created_at.elapsed().as_secs_f64(),
        }
    }

    /// Get the cache hit rate (0.0 to 1.0).
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get the total number of cache hits.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get the total number of cache misses.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get the total estimated time saved in milliseconds.
    #[must_use]
    pub fn time_saved_ms(&self) -> u64 {
        self.time_saved_ms.load(Ordering::Relaxed)
    }

    /// Get the total estimated time saved in seconds.
    #[must_use]
    pub fn time_saved_secs(&self) -> f64 {
        self.time_saved_ms.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Export metrics in Prometheus format.
    ///
    /// Returns a string suitable for Prometheus scraping.
    #[must_use]
    pub fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot();

        format!(
            r#"# HELP mithril_cache_hits_total Total number of cache hits
# TYPE mithril_cache_hits_total counter
mithril_cache_hits_total {}

# HELP mithril_cache_misses_total Total number of cache misses
# TYPE mithril_cache_misses_total counter
mithril_cache_misses_total {}

# HELP mithril_cache_evictions_total Total number of cache evictions
# TYPE mithril_cache_evictions_total counter
mithril_cache_evictions_total {}

# HELP mithril_cache_bytes_hit_total Total bytes served from cache
# TYPE mithril_cache_bytes_hit_total counter
mithril_cache_bytes_hit_total {}

# HELP mithril_cache_bytes_written_total Total bytes written to cache
# TYPE mithril_cache_bytes_written_total counter
mithril_cache_bytes_written_total {}

# HELP mithril_cache_bytes_evicted_total Total bytes evicted from cache
# TYPE mithril_cache_bytes_evicted_total counter
mithril_cache_bytes_evicted_total {}

# HELP mithril_cache_time_saved_seconds Estimated compilation time saved
# TYPE mithril_cache_time_saved_seconds counter
mithril_cache_time_saved_seconds {:.3}

# HELP mithril_cache_hit_rate Cache hit rate
# TYPE mithril_cache_hit_rate gauge
mithril_cache_hit_rate {:.4}

# HELP mithril_cache_uptime_seconds Cache uptime in seconds
# TYPE mithril_cache_uptime_seconds gauge
mithril_cache_uptime_seconds {:.1}
"#,
            snapshot.hits,
            snapshot.misses,
            snapshot.evictions,
            snapshot.bytes_hit,
            snapshot.bytes_written,
            snapshot.bytes_evicted,
            snapshot.time_saved_ms as f64 / 1000.0,
            snapshot.hit_rate,
            snapshot.uptime_secs,
        )
    }

    /// Export metrics in JSON format.
    ///
    /// Returns a JSON string with all metrics.
    #[must_use]
    pub fn export_json(&self) -> String {
        let snapshot = self.snapshot();
        serde_json::to_string_pretty(&snapshot).unwrap_or_else(|_| "{}".to_string())
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.bytes_hit.store(0, Ordering::Relaxed);
        self.bytes_written.store(0, Ordering::Relaxed);
        self.bytes_evicted.store(0, Ordering::Relaxed);
        self.time_saved_ms.store(0, Ordering::Relaxed);
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of cache metrics.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total bytes served from cache
    pub bytes_hit: u64,
    /// Total bytes written to cache
    pub bytes_written: u64,
    /// Total bytes evicted
    pub bytes_evicted: u64,
    /// Estimated time saved in milliseconds
    pub time_saved_ms: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Uptime in seconds
    pub uptime_secs: f64,
}

impl MetricsSnapshot {
    /// Get the hit rate as a percentage string.
    #[must_use]
    pub fn hit_rate_percent(&self) -> String {
        format!("{:.1}%", self.hit_rate * 100.0)
    }

    /// Get a human-readable time saved string.
    #[must_use]
    pub fn time_saved_human(&self) -> String {
        let secs = self.time_saved_ms as f64 / 1000.0;
        if secs < 60.0 {
            format!("{:.1}s", secs)
        } else if secs < 3600.0 {
            format!("{:.1}m", secs / 60.0)
        } else {
            format!("{:.1}h", secs / 3600.0)
        }
    }

    /// Get a human-readable bytes hit string.
    #[must_use]
    pub fn bytes_hit_human(&self) -> String {
        format_bytes(self.bytes_hit)
    }

    /// Get a human-readable bytes written string.
    #[must_use]
    pub fn bytes_written_human(&self) -> String {
        format_bytes(self.bytes_written)
    }
}

/// Format bytes in human-readable form.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Summary report for cache metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Cache hit rate
    pub hit_rate: f64,
    /// Total operations (hits + misses)
    pub total_operations: u64,
    /// Total time saved
    pub time_saved_secs: f64,
    /// Total bytes served
    pub bytes_served: u64,
    /// Recommendation based on metrics
    pub recommendation: String,
}

impl MetricsSnapshot {
    /// Generate a summary report with recommendations.
    #[must_use]
    pub fn summary(&self) -> MetricsSummary {
        let total = self.hits + self.misses;
        let recommendation = if total < 100 {
            "Not enough data for recommendations".to_string()
        } else if self.hit_rate > 0.9 {
            "Excellent! Cache is working optimally".to_string()
        } else if self.hit_rate > 0.7 {
            "Good cache performance. Consider increasing cache size for better hit rate".to_string()
        } else if self.hit_rate > 0.5 {
            "Moderate performance. Review cache key generation or increase cache size".to_string()
        } else if self.hit_rate > 0.3 {
            "Low hit rate. Check if cache is properly configured".to_string()
        } else {
            "Very low hit rate. Cache may not be effective for this workload".to_string()
        };

        MetricsSummary {
            hit_rate: self.hit_rate,
            total_operations: total,
            time_saved_secs: self.time_saved_ms as f64 / 1000.0,
            bytes_served: self.bytes_hit,
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_basic() {
        let metrics = CacheMetrics::new();

        metrics.record_hit(1024);
        metrics.record_hit(2048);
        metrics.record_miss();

        assert_eq!(metrics.hits(), 2);
        assert_eq!(metrics.misses(), 1);
    }

    #[test]
    fn test_hit_rate() {
        let metrics = CacheMetrics::new();

        metrics.record_hit(100);
        metrics.record_hit(100);
        metrics.record_hit(100);
        metrics.record_miss();

        let rate = metrics.hit_rate();
        assert!((rate - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_hit_rate_empty() {
        let metrics = CacheMetrics::new();
        assert_eq!(metrics.hit_rate(), 0.0);
    }

    #[test]
    fn test_bytes_tracking() {
        let metrics = CacheMetrics::new();

        metrics.record_hit(1000);
        metrics.record_write(2000);
        metrics.record_eviction(500);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.bytes_hit, 1000);
        assert_eq!(snapshot.bytes_written, 2000);
        assert_eq!(snapshot.bytes_evicted, 500);
    }

    #[test]
    fn test_time_saved() {
        let metrics = CacheMetrics::new();

        metrics.record_hit_with_time(1000, 5000); // 5 seconds saved
        metrics.record_hit_with_time(1000, 3000); // 3 seconds saved

        assert_eq!(metrics.time_saved_ms(), 8000);
        assert!((metrics.time_saved_secs() - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_snapshot() {
        let metrics = CacheMetrics::new();

        metrics.record_hit(100);
        metrics.record_miss();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 1);
        assert_eq!(snapshot.misses, 1);
        assert!((snapshot.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = CacheMetrics::new();
        metrics.record_hit(1024);

        let prometheus = metrics.export_prometheus();
        assert!(prometheus.contains("mithril_cache_hits_total 1"));
        assert!(prometheus.contains("mithril_cache_bytes_hit_total 1024"));
    }

    #[test]
    fn test_json_export() {
        let metrics = CacheMetrics::new();
        metrics.record_hit(1024);

        let json = metrics.export_json();
        assert!(json.contains("\"hits\": 1"));
        assert!(json.contains("\"bytes_hit\": 1024"));
    }

    #[test]
    fn test_reset() {
        let metrics = CacheMetrics::new();

        metrics.record_hit(100);
        metrics.record_miss();
        assert_eq!(metrics.hits(), 1);

        metrics.reset();
        assert_eq!(metrics.hits(), 0);
        assert_eq!(metrics.misses(), 0);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let metrics = Arc::new(CacheMetrics::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let m = Arc::clone(&metrics);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    m.record_hit(100);
                    m.record_miss();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(metrics.hits(), 1000);
        assert_eq!(metrics.misses(), 1000);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_time_saved_human() {
        let snapshot = MetricsSnapshot {
            hits: 0,
            misses: 0,
            evictions: 0,
            bytes_hit: 0,
            bytes_written: 0,
            bytes_evicted: 0,
            time_saved_ms: 45000,
            hit_rate: 0.0,
            uptime_secs: 0.0,
        };

        assert_eq!(snapshot.time_saved_human(), "45.0s");
    }

    #[test]
    fn test_summary_recommendations() {
        let high_rate = MetricsSnapshot {
            hits: 95,
            misses: 5,
            evictions: 0,
            bytes_hit: 0,
            bytes_written: 0,
            bytes_evicted: 0,
            time_saved_ms: 0,
            hit_rate: 0.95,
            uptime_secs: 0.0,
        };
        assert!(high_rate.summary().recommendation.contains("Excellent"));

        let low_rate = MetricsSnapshot {
            hits: 20,
            misses: 80,
            evictions: 0,
            bytes_hit: 0,
            bytes_written: 0,
            bytes_evicted: 0,
            time_saved_ms: 0,
            hit_rate: 0.2,
            uptime_secs: 0.0,
        };
        assert!(low_rate.summary().recommendation.contains("Very low"));
    }
}
