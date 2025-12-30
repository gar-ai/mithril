//! SGLang cache integration.
//!
//! This module provides caching support for SGLang inference framework,
//! enabling faster cold starts by caching compiled kernels.

use mithril_core::hashing::{Blake3Hasher, HashFunction};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during SGLang cache operations.
#[derive(Debug, Error)]
pub enum SglangError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Cache error: {0}")]
    Cache(String),
}

pub type Result<T> = std::result::Result<T, SglangError>;

/// Configuration for SGLang cache.
#[derive(Debug, Clone)]
pub struct SglangCacheConfig {
    pub cache_dir: PathBuf,
    pub max_size_bytes: u64,
    pub cache_attention: bool,
    pub cache_sampling: bool,
    pub remote_url: Option<String>,
}

impl SglangCacheConfig {
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
            max_size_bytes: 10 * 1024 * 1024 * 1024,
            cache_attention: true,
            cache_sampling: true,
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
pub struct SglangCacheStats {
    pub total_size_bytes: u64,
    pub attention_kernel_count: usize,
    pub sampling_kernel_count: usize,
}

pub struct SglangCacheManager {
    config: SglangCacheConfig,
    hasher: Blake3Hasher,
}

impl SglangCacheManager {
    pub fn new(config: SglangCacheConfig) -> Result<Self> {
        std::fs::create_dir_all(config.cache_dir.join("attention"))?;
        std::fs::create_dir_all(config.cache_dir.join("sampling"))?;
        std::fs::create_dir_all(config.cache_dir.join("triton"))?;
        std::fs::create_dir_all(config.cache_dir.join("cas"))?;
        Ok(Self {
            config,
            hasher: Blake3Hasher::new(),
        })
    }

    pub fn cache_dir(&self) -> &Path {
        &self.config.cache_dir
    }
    pub fn attention_dir(&self) -> PathBuf {
        self.config.cache_dir.join("attention")
    }
    pub fn sampling_dir(&self) -> PathBuf {
        self.config.cache_dir.join("sampling")
    }
    pub fn triton_dir(&self) -> PathBuf {
        self.config.cache_dir.join("triton")
    }
    fn cas_dir(&self) -> PathBuf {
        self.config.cache_dir.join("cas")
    }

    pub fn put_attention_kernel(&self, key: &str, data: &[u8]) -> Result<()> {
        if !self.config.cache_attention {
            return Ok(());
        }
        let hash = hex::encode(self.hasher.hash(data));
        let cas_path = self.cas_dir().join(&hash);
        if !cas_path.exists() {
            std::fs::write(&cas_path, data)?;
        }
        std::fs::write(self.attention_dir().join(format!("{}.hash", key)), &hash)?;
        Ok(())
    }

    pub fn get_attention_kernel(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if !self.config.cache_attention {
            return Ok(None);
        }
        let mapping = self.attention_dir().join(format!("{}.hash", key));
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

    pub fn put_sampling_kernel(&self, key: &str, data: &[u8]) -> Result<()> {
        if !self.config.cache_sampling {
            return Ok(());
        }
        let hash = hex::encode(self.hasher.hash(data));
        let cas_path = self.cas_dir().join(&hash);
        if !cas_path.exists() {
            std::fs::write(&cas_path, data)?;
        }
        std::fs::write(self.sampling_dir().join(format!("{}.hash", key)), &hash)?;
        Ok(())
    }

    pub fn get_sampling_kernel(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if !self.config.cache_sampling {
            return Ok(None);
        }
        let mapping = self.sampling_dir().join(format!("{}.hash", key));
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
        for dir in [self.attention_dir(), self.sampling_dir()] {
            if dir.exists() {
                for entry in std::fs::read_dir(&dir)? {
                    if entry?.path().extension().is_some_and(|e| e == "hash") {
                        stats.kernels_loaded += 1;
                    }
                }
            }
        }
        stats.elapsed_secs = start.elapsed().as_secs_f64();
        Ok(stats)
    }

    pub fn stats(&self) -> Result<SglangCacheStats> {
        let mut stats = SglangCacheStats::default();
        if self.attention_dir().exists() {
            for entry in std::fs::read_dir(self.attention_dir())? {
                if entry?.path().extension().is_some_and(|e| e == "hash") {
                    stats.attention_kernel_count += 1;
                }
            }
        }
        if self.sampling_dir().exists() {
            for entry in std::fs::read_dir(self.sampling_dir())? {
                if entry?.path().extension().is_some_and(|e| e == "hash") {
                    stats.sampling_kernel_count += 1;
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
        for dir in [self.attention_dir(), self.sampling_dir(), self.cas_dir()] {
            if dir.exists() {
                for entry in std::fs::read_dir(&dir)? {
                    std::fs::remove_file(entry?.path())?;
                }
            }
        }
        Ok(())
    }

    pub fn set_environment(&self) {
        std::env::set_var("TRITON_CACHE_DIR", self.triton_dir());
        std::env::set_var("HF_HOME", self.config.cache_dir.join("hf_home"));
        std::env::set_var("SGLANG_CACHE_DIR", &self.config.cache_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_sglang_cache_creation() {
        let dir = tempdir().unwrap();
        let manager = SglangCacheManager::new(SglangCacheConfig::new(dir.path())).unwrap();
        assert!(manager.attention_dir().exists());
        assert!(manager.sampling_dir().exists());
    }

    #[test]
    fn test_attention_kernel_cache() {
        let dir = tempdir().unwrap();
        let manager = SglangCacheManager::new(SglangCacheConfig::new(dir.path())).unwrap();
        manager.put_attention_kernel("test", b"data").unwrap();
        assert_eq!(
            manager.get_attention_kernel("test").unwrap(),
            Some(b"data".to_vec())
        );
    }

    #[test]
    fn test_sampling_kernel_cache() {
        let dir = tempdir().unwrap();
        let manager = SglangCacheManager::new(SglangCacheConfig::new(dir.path())).unwrap();
        manager.put_sampling_kernel("test", b"data").unwrap();
        assert_eq!(
            manager.get_sampling_kernel("test").unwrap(),
            Some(b"data".to_vec())
        );
    }

    #[test]
    fn test_cache_stats() {
        let dir = tempdir().unwrap();
        let manager = SglangCacheManager::new(SglangCacheConfig::new(dir.path())).unwrap();
        manager.put_attention_kernel("a1", b"d1").unwrap();
        manager.put_sampling_kernel("s1", b"d2").unwrap();
        let stats = manager.stats().unwrap();
        assert_eq!(stats.attention_kernel_count, 1);
        assert_eq!(stats.sampling_kernel_count, 1);
    }
}
