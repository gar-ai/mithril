//! Python bindings for mithril-dedup.

use mithril_dedup::{
    DedupConfig, DedupResult, DedupStats, Deduplicator, HybridConfig, HybridDeduplicator,
    MockBackend, SemanticConfig, SemanticDeduplicator,
};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for DedupConfig.
#[pyclass(name = "DedupConfig")]
#[derive(Clone)]
pub struct PyDedupConfig {
    inner: DedupConfig,
}

#[pymethods]
impl PyDedupConfig {
    /// Create a new dedup config.
    ///
    /// Args:
    ///     threshold: Similarity threshold (0.0-1.0). Default: 0.85
    ///     num_permutations: Number of MinHash permutations. Default: 128
    ///     ngram_size: N-gram size for shingling. Default: 5
    #[new]
    #[pyo3(signature = (threshold=0.85, num_permutations=128, ngram_size=5))]
    fn new(threshold: f64, num_permutations: usize, ngram_size: usize) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threshold must be between 0.0 and 1.0",
            ));
        }
        Ok(Self {
            inner: DedupConfig {
                threshold,
                num_permutations,
                ngram_size,
                verify_candidates: true,
            },
        })
    }

    /// Create config with just a threshold.
    #[staticmethod]
    fn with_threshold(threshold: f64) -> PyResult<Self> {
        Self::new(threshold, 128, 5)
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    #[getter]
    fn num_permutations(&self) -> usize {
        self.inner.num_permutations
    }

    #[getter]
    fn ngram_size(&self) -> usize {
        self.inner.ngram_size
    }

    fn __repr__(&self) -> String {
        format!(
            "DedupConfig(threshold={}, num_permutations={}, ngram_size={})",
            self.inner.threshold, self.inner.num_permutations, self.inner.ngram_size
        )
    }
}

/// Python wrapper for DedupStats.
#[pyclass(name = "DedupStats")]
pub struct PyDedupStats {
    #[pyo3(get)]
    pub total_documents: usize,
    #[pyo3(get)]
    pub unique_documents: usize,
    #[pyo3(get)]
    pub duplicate_count: usize,
    #[pyo3(get)]
    pub duplicate_ratio: f64,
    #[pyo3(get)]
    pub cluster_count: usize,
    #[pyo3(get)]
    pub candidate_pairs: usize,
    #[pyo3(get)]
    pub verified_pairs: usize,
    #[pyo3(get)]
    pub processing_time_secs: f64,
}

impl From<DedupStats> for PyDedupStats {
    fn from(stats: DedupStats) -> Self {
        Self {
            total_documents: stats.total_documents,
            unique_documents: stats.unique_documents,
            duplicate_count: stats.duplicate_count,
            duplicate_ratio: stats.duplicate_ratio,
            cluster_count: stats.cluster_count,
            candidate_pairs: stats.candidate_pairs,
            verified_pairs: stats.verified_pairs,
            processing_time_secs: stats.processing_time_secs,
        }
    }
}

#[pymethods]
impl PyDedupStats {
    fn __repr__(&self) -> String {
        format!(
            "DedupStats(total={}, unique={}, duplicates={}, ratio={:.2}%)",
            self.total_documents,
            self.unique_documents,
            self.duplicate_count,
            self.duplicate_ratio * 100.0
        )
    }
}

/// Python wrapper for DedupResult.
#[pyclass(name = "DedupResult")]
pub struct PyDedupResult {
    #[pyo3(get)]
    pub keep_indices: Vec<usize>,
    #[pyo3(get)]
    pub remove_indices: Vec<usize>,
    #[pyo3(get)]
    pub clusters: HashMap<usize, Vec<usize>>,
    #[pyo3(get)]
    pub stats: Py<PyDedupStats>,
}

impl PyDedupResult {
    fn from_result(py: Python<'_>, result: DedupResult) -> PyResult<Self> {
        let stats = Py::new(py, PyDedupStats::from(result.stats))?;
        Ok(Self {
            keep_indices: result.keep_indices,
            remove_indices: result.remove_indices,
            clusters: result.clusters,
            stats,
        })
    }
}

#[pymethods]
impl PyDedupResult {
    fn __repr__(&self, py: Python<'_>) -> String {
        let stats = self.stats.bind(py).borrow();
        format!(
            "DedupResult(keep={}, remove={}, clusters={})",
            self.keep_indices.len(),
            self.remove_indices.len(),
            stats.cluster_count
        )
    }
}

/// Python wrapper for Deduplicator.
#[pyclass(name = "Deduplicator")]
pub struct PyDeduplicator {
    inner: Deduplicator,
}

#[pymethods]
impl PyDeduplicator {
    /// Create a new deduplicator.
    ///
    /// Args:
    ///     config: Optional dedup configuration. Uses defaults if not provided.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyDedupConfig>) -> Self {
        let config = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: Deduplicator::new(config),
        }
    }

    /// Deduplicate a list of text strings.
    ///
    /// Args:
    ///     texts: List of text documents to deduplicate
    ///
    /// Returns:
    ///     DedupResult with indices to keep/remove and statistics
    fn deduplicate(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<PyDedupResult> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let result = self.inner.deduplicate_texts(&text_refs);
        PyDedupResult::from_result(py, result)
    }

    /// Compute similarity between two texts.
    ///
    /// Args:
    ///     text1: First text
    ///     text2: Second text
    ///
    /// Returns:
    ///     Jaccard similarity estimate (0.0 to 1.0)
    fn similarity(&self, text1: &str, text2: &str) -> f64 {
        self.inner.similarity(text1, text2)
    }

    /// Get the configuration used by this deduplicator.
    fn config(&self) -> PyDedupConfig {
        PyDedupConfig {
            inner: self.inner.config().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Deduplicator(threshold={})", self.inner.config().threshold)
    }
}

// ============================================================================
// Semantic Deduplication
// ============================================================================

/// Python wrapper for SemanticConfig.
#[pyclass(name = "SemanticConfig")]
#[derive(Clone)]
pub struct PySemanticConfig {
    inner: SemanticConfig,
}

#[pymethods]
impl PySemanticConfig {
    /// Create a new semantic dedup config.
    ///
    /// Args:
    ///     threshold: Cosine similarity threshold (0.0-1.0). Default: 0.9
    ///     batch_size: Batch size for embedding computation. Default: 64
    ///     num_neighbors: Number of neighbors to search in ANN. Default: 10
    ///     ef_search: EF parameter for HNSW search. Default: 100
    #[new]
    #[pyo3(signature = (threshold=0.9, batch_size=64, num_neighbors=10, ef_search=100))]
    fn new(
        threshold: f32,
        batch_size: usize,
        num_neighbors: usize,
        ef_search: usize,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threshold must be between 0.0 and 1.0",
            ));
        }
        Ok(Self {
            inner: SemanticConfig {
                threshold,
                batch_size,
                num_neighbors,
                ef_search,
            },
        })
    }

    /// Create config with just a threshold.
    #[staticmethod]
    fn with_threshold(threshold: f32) -> PyResult<Self> {
        Self::new(threshold, 64, 10, 100)
    }

    /// Create a fast config (less accurate but faster).
    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: SemanticConfig::fast(),
        }
    }

    /// Create a precise config (more accurate but slower).
    #[staticmethod]
    fn precise() -> Self {
        Self {
            inner: SemanticConfig::precise(),
        }
    }

    #[getter]
    fn threshold(&self) -> f32 {
        self.inner.threshold
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    #[getter]
    fn num_neighbors(&self) -> usize {
        self.inner.num_neighbors
    }

    #[getter]
    fn ef_search(&self) -> usize {
        self.inner.ef_search
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticConfig(threshold={}, batch_size={}, num_neighbors={})",
            self.inner.threshold, self.inner.batch_size, self.inner.num_neighbors
        )
    }
}

/// Python wrapper for SemanticDeduplicator.
///
/// Uses embedding-based similarity to detect semantically similar documents,
/// including paraphrases that MinHash would miss.
///
/// Note: Currently uses a mock embedding backend. For production use,
/// integrate with sentence-transformers or similar embedding models.
#[pyclass(name = "SemanticDeduplicator")]
pub struct PySemanticDeduplicator {
    inner: SemanticDeduplicator<MockBackend>,
}

#[pymethods]
impl PySemanticDeduplicator {
    /// Create a new semantic deduplicator.
    ///
    /// Args:
    ///     config: Optional semantic configuration. Uses defaults if not provided.
    ///     embedding_dim: Embedding dimension for the mock backend. Default: 384
    #[new]
    #[pyo3(signature = (config=None, embedding_dim=384))]
    fn new(config: Option<PySemanticConfig>, embedding_dim: usize) -> Self {
        let config = config.map(|c| c.inner).unwrap_or_default();
        let backend = MockBackend::new(embedding_dim);
        Self {
            inner: SemanticDeduplicator::new(backend, config),
        }
    }

    /// Deduplicate a list of text strings using semantic similarity.
    ///
    /// Args:
    ///     texts: List of text documents to deduplicate
    ///
    /// Returns:
    ///     DedupResult with indices to keep/remove and statistics
    fn deduplicate(&mut self, py: Python<'_>, texts: Vec<String>) -> PyResult<PyDedupResult> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let result = self
            .inner
            .deduplicate(&text_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        PyDedupResult::from_result(py, result)
    }

    /// Get the number of documents in the index.
    fn index_size(&self) -> usize {
        self.inner.index_size()
    }

    /// Clear the index.
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticDeduplicator(index_size={})",
            self.inner.index_size()
        )
    }
}

// ============================================================================
// Hybrid Deduplication
// ============================================================================

/// Python wrapper for HybridConfig.
#[pyclass(name = "HybridConfig")]
#[derive(Clone)]
pub struct PyHybridConfig {
    inner: HybridConfig,
}

#[pymethods]
impl PyHybridConfig {
    /// Create a new hybrid dedup config.
    ///
    /// Args:
    ///     minhash_threshold: MinHash threshold for candidate filtering (0.0-1.0). Default: 0.7
    ///     semantic_threshold: Semantic threshold for verification (0.0-1.0). Default: 0.9
    ///     num_permutations: Number of MinHash permutations. Default: 128
    ///     ngram_size: N-gram size for shingling. Default: 5
    #[new]
    #[pyo3(signature = (minhash_threshold=0.7, semantic_threshold=0.9, num_permutations=128, ngram_size=5))]
    fn new(
        minhash_threshold: f64,
        semantic_threshold: f32,
        num_permutations: usize,
        ngram_size: usize,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&minhash_threshold) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "minhash_threshold must be between 0.0 and 1.0",
            ));
        }
        if !(0.0..=1.0).contains(&semantic_threshold) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "semantic_threshold must be between 0.0 and 1.0",
            ));
        }
        Ok(Self {
            inner: HybridConfig {
                minhash_threshold,
                semantic_threshold,
                num_permutations,
                ngram_size,
                embed_batch_size: 64,
                verify_with_semantic: true,
            },
        })
    }

    /// Create a fast configuration (more MinHash filtering, less semantic).
    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: HybridConfig::fast(),
        }
    }

    /// Create a precise configuration (more semantic verification).
    #[staticmethod]
    fn precise() -> Self {
        Self {
            inner: HybridConfig::precise(),
        }
    }

    #[getter]
    fn minhash_threshold(&self) -> f64 {
        self.inner.minhash_threshold
    }

    #[getter]
    fn semantic_threshold(&self) -> f32 {
        self.inner.semantic_threshold
    }

    #[getter]
    fn num_permutations(&self) -> usize {
        self.inner.num_permutations
    }

    #[getter]
    fn ngram_size(&self) -> usize {
        self.inner.ngram_size
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridConfig(minhash_threshold={}, semantic_threshold={}, num_permutations={})",
            self.inner.minhash_threshold,
            self.inner.semantic_threshold,
            self.inner.num_permutations
        )
    }
}

/// Python wrapper for HybridDeduplicator.
///
/// Combines MinHash (fast syntactic matching) with semantic similarity
/// (accurate meaning-based matching) for high-quality deduplication.
///
/// Stage 1: MinHash finds candidate duplicates quickly
/// Stage 2: Semantic similarity verifies candidates
///
/// Note: Currently uses a mock embedding backend. For production use,
/// integrate with sentence-transformers or similar embedding models.
#[pyclass(name = "HybridDeduplicator")]
pub struct PyHybridDeduplicator {
    inner: HybridDeduplicator<MockBackend>,
}

#[pymethods]
impl PyHybridDeduplicator {
    /// Create a new hybrid deduplicator.
    ///
    /// Args:
    ///     config: Optional hybrid configuration. Uses defaults if not provided.
    ///     embedding_dim: Embedding dimension for the mock backend. Default: 384
    #[new]
    #[pyo3(signature = (config=None, embedding_dim=384))]
    fn new(config: Option<PyHybridConfig>, embedding_dim: usize) -> Self {
        let config = config.map(|c| c.inner).unwrap_or_default();
        let backend = MockBackend::new(embedding_dim);
        Self {
            inner: HybridDeduplicator::new(backend, config),
        }
    }

    /// Deduplicate a list of text strings using hybrid approach.
    ///
    /// Args:
    ///     texts: List of text documents to deduplicate
    ///
    /// Returns:
    ///     DedupResult with indices to keep/remove and statistics
    fn deduplicate(&mut self, py: Python<'_>, texts: Vec<String>) -> PyResult<PyDedupResult> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let result = self
            .inner
            .deduplicate(&text_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        PyDedupResult::from_result(py, result)
    }

    /// Get the number of documents in the index.
    fn index_size(&self) -> usize {
        self.inner.index_size()
    }

    /// Clear both MinHash and semantic indices.
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!("HybridDeduplicator(index_size={})", self.inner.index_size())
    }
}

/// Register dedup module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "dedup")?;
    m.add_class::<PyDedupConfig>()?;
    m.add_class::<PyDedupStats>()?;
    m.add_class::<PyDedupResult>()?;
    m.add_class::<PyDeduplicator>()?;
    m.add_class::<PySemanticConfig>()?;
    m.add_class::<PySemanticDeduplicator>()?;
    m.add_class::<PyHybridConfig>()?;
    m.add_class::<PyHybridDeduplicator>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
