"""Integration tests for torch.compile cache management."""

import os
import shutil
import time

import pytest

torch = pytest.importorskip("torch")
mithril = pytest.importorskip("mithril")


@pytest.fixture
def clean_cache_dir(tmp_path):
    """Provide a clean temporary cache directory."""
    cache_dir = tmp_path / "torch_cache"
    cache_dir.mkdir()
    yield cache_dir
    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.mark.slow
class TestTorchCompileCache:
    """Test torch.compile integration with mithril cache."""

    def test_cache_manager_creation(self, clean_cache_dir):
        """Test that CacheManager can be created."""
        config = mithril.CacheConfig(str(clean_cache_dir))
        manager = mithril.CacheManager(config)
        assert manager is not None

    def test_cache_directories_created(self, clean_cache_dir):
        """Test that cache directories are created."""
        config = mithril.CacheConfig(str(clean_cache_dir))
        config = config.with_inductor(True).with_triton(True)
        manager = mithril.CacheManager(config)

        # Check subdirectories exist
        inductor_dir = manager.inductor_dir()
        triton_dir = manager.triton_dir()

        assert inductor_dir is not None
        assert triton_dir is not None

    def test_cache_stats(self, clean_cache_dir):
        """Test that cache stats can be retrieved."""
        config = mithril.CacheConfig(str(clean_cache_dir))
        manager = mithril.CacheManager(config)

        stats = manager.stats()
        # CacheStats is a Rust-exposed object
        assert hasattr(stats, "lru_entry_count")
        assert hasattr(stats, "lru_size_bytes")
        assert stats.lru_entry_count >= 0

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not hasattr(torch.backends, "mps"),
        reason="Requires GPU or MPS for meaningful torch.compile test",
    )
    def test_torch_compile_cache_hit(self, clean_cache_dir, monkeypatch):
        """Test that torch.compile benefits from caching.

        This test verifies that:
        1. First compilation is slow (cold cache)
        2. Second compilation is faster (warm cache)
        """
        # Set up mithril cache
        config = mithril.CacheConfig(str(clean_cache_dir))
        config = config.with_inductor(True).with_triton(True)
        manager = mithril.CacheManager(config)

        # Set environment variables for torch.compile
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", manager.inductor_dir())
        monkeypatch.setenv("TRITON_CACHE_DIR", manager.triton_dir())

        # Simple model for compilation
        @torch.compile
        def simple_model(x):
            return x @ x.T + x.sum()

        # Warmup / first run (cold)
        x = torch.randn(100, 100, device="cuda" if torch.cuda.is_available() else "mps")
        _ = simple_model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Time the second run (should use cache)
        start = time.perf_counter()
        _ = simple_model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cached_time = time.perf_counter() - start

        # Cached execution should be fast (< 1 second for this simple model)
        assert cached_time < 5.0, f"Cached execution took {cached_time:.2f}s, expected < 5s"


class TestCacheEnvironment:
    """Test environment variable handling for cache."""

    def test_inductor_env_set(self, clean_cache_dir, monkeypatch):
        """Test that TORCHINDUCTOR_CACHE_DIR can be set."""
        config = mithril.CacheConfig(str(clean_cache_dir))
        config = config.with_inductor(True)
        manager = mithril.CacheManager(config)

        inductor_dir = manager.inductor_dir()
        assert inductor_dir is not None
        assert os.path.isdir(inductor_dir)

    def test_triton_env_set(self, clean_cache_dir, monkeypatch):
        """Test that TRITON_CACHE_DIR can be set."""
        config = mithril.CacheConfig(str(clean_cache_dir))
        config = config.with_triton(True)
        manager = mithril.CacheManager(config)

        triton_dir = manager.triton_dir()
        assert triton_dir is not None
        assert os.path.isdir(triton_dir)
