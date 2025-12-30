//! Python bindings for Mithril.
//!
//! Provides Python access to:
//! - `mithril.checkpoint`: Checkpoint compression for PyTorch models
//! - `mithril.cache`: Compilation caching for torch.compile
//! - `mithril.dedup`: Dataset deduplication with MinHash/LSH
//!
//! ## Example
//!
//! ```python
//! import mithril
//!
//! # Checkpoint compression (sync)
//! config = mithril.checkpoint.CompressionConfig(zstd_level=3)
//! compressor = mithril.checkpoint.CheckpointCompressor(config)
//! compressed = compressor.compress(data, "bf16")
//!
//! # Cache management (sync)
//! config = mithril.cache.CacheConfig("/tmp/cache").with_max_size_gb(10)
//! manager = mithril.cache.CacheManager(config)
//!
//! # Content store (sync)
//! store = mithril.cache.ContentStore("/tmp/cas")
//! address = store.put(content)
//!
//! # Deduplication
//! config = mithril.dedup.DedupConfig(threshold=0.85)
//! deduplicator = mithril.dedup.Deduplicator(config)
//! result = deduplicator.deduplicate(["doc1", "doc1 copy", "doc2"])
//! print(result.keep_indices)  # [0, 2]
//! ```

mod cache;
mod checkpoint;
mod dedup;

use pyo3::prelude::*;

/// Mithril: ML Infrastructure Toolkit
///
/// Submodules:
///   - checkpoint: Checkpoint compression for PyTorch models
///   - cache: Compilation caching for torch.compile
///   - dedup: Dataset deduplication with MinHash/LSH
#[pymodule]
fn _mithril(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    checkpoint::register(m)?;
    cache::register(m)?;
    dedup::register(m)?;

    // Also expose top-level convenience classes
    m.add_class::<checkpoint::PyCompressionConfig>()?;
    m.add_class::<checkpoint::PyCheckpointCompressor>()?;
    m.add_class::<checkpoint::PyDeltaCompressor>()?;
    m.add_class::<checkpoint::PyDeltaStats>()?;
    m.add_class::<checkpoint::PyOrbaxWriter>()?;
    m.add_class::<checkpoint::PyOrbaxWriteStats>()?;
    m.add_class::<checkpoint::PyQuantizeConfig>()?;
    m.add_class::<checkpoint::PyQuantizer>()?;
    m.add_class::<checkpoint::PyMstWriter>()?;
    m.add_class::<checkpoint::PyMstReader>()?;
    m.add_class::<checkpoint::PyMstTensorInfo>()?;
    m.add_function(pyo3::wrap_pyfunction!(checkpoint::is_mst, m)?)?;
    m.add_class::<cache::PyCacheConfig>()?;
    m.add_class::<cache::PyCacheManager>()?;
    m.add_class::<cache::PyContentStore>()?;
    m.add_class::<cache::PyRemoteCacheConfig>()?;
    #[cfg(feature = "s3")]
    m.add_class::<cache::PyS3RemoteCache>()?;
    m.add_class::<dedup::PyDedupConfig>()?;
    m.add_class::<dedup::PyDeduplicator>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Mithril: ML Infrastructure Toolkit")?;

    Ok(())
}
