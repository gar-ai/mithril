//! vLLM cache integration.
//!
//! This module provides caching support for vLLM inference framework,
//! enabling faster cold starts by caching compiled kernels.

use mithril_core::hashing::{Blake3Hasher, HashFunction};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during vLLM cache operations.
#[derive(Debug, Error)]
pub enum VllmError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Cache error: {0}")]
    Cache(String),
}

pub type Result<T> = std::result::Result<T, VllmError>;

/// Configuration for vLLM cache.
#[derive(Debug, Clone)]
pub struct VllmCacheConfig {
    pub cache_dir: PathBuf,
    pub max_size_bytes: u64,
    pub cache_kernels: bool,
    pub cache_tokenizers: bool,
    pub remote_url: Option<String>,
}

impl VllmCacheConfig {
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
            max_size_bytes: 10 * 1024 * 1024 * 1024,
            cache_kernels: true,
            cache_tokenizers: true,
            remote_url: None,
        }
    }

    pub fn with_max_size_gb(mut self, gb: f64) -> Self {
        self.max_size_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
        self
    }

    pub fn with_remote(mut self, url: &str) -> Self {
        self.remote_url = Some(url.to_string());
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct WarmupStats {
    pub kernels_loaded: usize,
    pub bytes_loaded: u64,
    pub elapsed_secs: f64,
}

#[derive(Debug, Clone, Default)]
pub struct VllmCacheStats {
    pub total_size_bytes: u64,
    pub kernel_count: usize,
    pub tokenizer_count: usize,
}

pub struct VllmCacheManager {
    config: VllmCacheConfig,
    hasher: Blake3Hasher,
}

impl VllmCacheManager {
    pub fn new(config: VllmCacheConfig) -> Result<Self> {
        std::fs::create_dir_all(config.cache_dir.join("kernels"))?;
        std::fs::create_dir_all(config.cache_dir.join("tokenizers"))?;
        std::fs::create_dir_all(config.cache_dir.join("cas"))?;
        Ok(Self {
            config,
            hasher: Blake3Hasher::new(),
        })
    }

    pub fn cache_dir(&self) -> &Path {
        &self.config.cache_dir
    }
    pub fn kernels_dir(&self) -> PathBuf {
        self.config.cache_dir.join("kernels")
    }
    pub fn tokenizers_dir(&self) -> PathBuf {
        self.config.cache_dir.join("tokenizers")
    }
    fn cas_dir(&self) -> PathBuf {
        self.config.cache_dir.join("cas")
    }

    pub fn put_kernel(&self, key: &str, data: &[u8]) -> Result<()> {
        if !self.config.cache_kernels {
            return Ok(());
        }
        let hash = hex::encode(self.hasher.hash(data));
        let cas_path = self.cas_dir().join(&hash);
        if !cas_path.exists() {
            std::fs::write(&cas_path, data)?;
        }
        std::fs::write(self.kernels_dir().join(format!("{}.hash", key)), &hash)?;
        Ok(())
    }

    pub fn get_kernel(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if !self.config.cache_kernels {
            return Ok(None);
        }
        let mapping = self.kernels_dir().join(format!("{}.hash", key));
        if !mapping.exists() {
            return Ok(None);
        }
        let hash = std::fs::read_to_string(&mapping)?;
        let cas_path = self.cas_dir().join(&hash);
        if cas_path.exists() {
            Ok(Some(std::fs::read(&cas_path)?))
        } else {
            Ok(None)
        }
    }

    pub fn warmup(&self) -> Result<WarmupStats> {
        let start = std::time::Instant::now();
        let mut stats = WarmupStats::default();
        if self.kernels_dir().exists() {
            for entry in std::fs::read_dir(self.kernels_dir())? {
                if entry?.path().extension().map_or(false, |e| e == "hash") {
                    stats.kernels_loaded += 1;
                }
            }
        }
        stats.elapsed_secs = start.elapsed().as_secs_f64();
        Ok(stats)
    }

    pub fn stats(&self) -> Result<VllmCacheStats> {
        let mut stats = VllmCacheStats::default();
        if self.kernels_dir().exists() {
            for entry in std::fs::read_dir(self.kernels_dir())? {
                if entry?.path().extension().map_or(false, |e| e == "hash") {
                    stats.kernel_count += 1;
                }
            }
        }
        if self.cas_dir().exists() {
            for entry in std::fs::read_dir(self.cas_dir())? {
                if let Ok(meta) = entry?.metadata() {
                    stats.total_size_bytes += meta.len();
                }
            }
        }
        Ok(stats)
    }

    pub fn clear(&self) -> Result<()> {
        for dir in [self.kernels_dir(), self.cas_dir()] {
            if dir.exists() {
                for entry in std::fs::read_dir(&dir)? {
                    std::fs::remove_file(entry?.path())?;
                }
            }
        }
        Ok(())
    }

    pub fn set_environment(&self) {
        std::env::set_var("VLLM_CACHE_ROOT", &self.config.cache_dir);
        std::env::set_var("HF_HOME", self.tokenizers_dir());
        std::env::set_var("TRITON_CACHE_DIR", self.kernels_dir());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_vllm_cache_creation() {
        let dir = tempdir().unwrap();
        let manager = VllmCacheManager::new(VllmCacheConfig::new(dir.path())).unwrap();
        assert!(manager.kernels_dir().exists());
    }

    #[test]
    fn test_kernel_cache() {
        let dir = tempdir().unwrap();
        let manager = VllmCacheManager::new(VllmCacheConfig::new(dir.path())).unwrap();
        manager.put_kernel("test", b"data").unwrap();
        assert_eq!(manager.get_kernel("test").unwrap(), Some(b"data".to_vec()));
        assert!(manager.get_kernel("missing").unwrap().is_none());
    }

    #[test]
    fn test_cache_stats() {
        let dir = tempdir().unwrap();
        let manager = VllmCacheManager::new(VllmCacheConfig::new(dir.path())).unwrap();
        manager.put_kernel("k1", b"d1").unwrap();
        manager.put_kernel("k2", b"d2").unwrap();
        assert_eq!(manager.stats().unwrap().kernel_count, 2);
    }
}
