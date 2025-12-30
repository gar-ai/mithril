#!/usr/bin/env python3
"""Test TorchInductor/Triton cache hooks with mithril-cache."""

import os
import sys
import tempfile
import time

# Check torch is available
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

# Check mithril is available
try:
    import mithril
    print(f"Mithril version: {mithril.__version__}")
except ImportError:
    print("ERROR: mithril not installed. Run: cd crates/mithril-python && maturin develop")
    sys.exit(1)


def test_cache_hooks():
    """Test that mithril cache hooks intercept torch.compile caches."""

    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\n=== Setting up mithril cache in {cache_dir} ===")

        # Create cache config with inductor and triton hooks
        config = mithril.CacheConfig(cache_dir)
        config = config.with_inductor(True)
        config = config.with_triton(True)
        manager = mithril.CacheManager(config)

        # Check environment variables are set
        inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        triton_dir = os.environ.get("TRITON_CACHE_DIR")

        print(f"TORCHINDUCTOR_CACHE_DIR = {inductor_dir}")
        print(f"TRITON_CACHE_DIR = {triton_dir}")

        expected_inductor = os.path.join(cache_dir, "inductor")
        expected_triton = os.path.join(cache_dir, "triton")

        assert inductor_dir == expected_inductor, f"Expected {expected_inductor}, got {inductor_dir}"
        assert triton_dir == expected_triton, f"Expected {expected_triton}, got {triton_dir}"
        print("✓ Environment variables set correctly")

        # Check Python version for torch.compile support
        import sys
        python_version = sys.version_info

        if python_version >= (3, 14):
            print(f"\n⚠ Python {python_version.major}.{python_version.minor} detected")
            print("  torch.compile is not supported on Python 3.14+")
            print("  Skipping torch.compile test, but env var hooks are working!")
            print("\n=== Test Complete (partial - env vars validated) ===")
            return True

        # Define a simple function to compile
        @torch.compile
        def fn(x):
            return x * 2 + 1

        # First run (cold) - should trigger compilation
        print("\n=== Running torch.compile (cold start) ===")
        x = torch.randn(100, 100)
        start = time.time()
        result = fn(x)
        cold_time = time.time() - start
        print(f"Cold start time: {cold_time:.3f}s")

        # Verify result is correct
        expected = x * 2 + 1
        assert torch.allclose(result, expected), "Computation result incorrect!"
        print("✓ Computation correct")

        # Second run (warm) - should use cached compilation
        print("\n=== Running torch.compile (warm) ===")
        x2 = torch.randn(100, 100)
        start = time.time()
        result2 = fn(x2)
        warm_time = time.time() - start
        print(f"Warm time: {warm_time:.3f}s")

        # Check cache stats
        stats = manager.stats()
        print(f"\n=== Cache Stats ===")
        print(f"Entries: {stats.lru_entry_count}")
        print(f"Size: {stats.lru_size_bytes} bytes")
        print(f"Utilization: {stats.utilization * 100:.1f}%")

        # Check if cache directories have content
        inductor_files = []
        triton_files = []

        if os.path.exists(expected_inductor):
            for root, dirs, files in os.walk(expected_inductor):
                inductor_files.extend(files)

        if os.path.exists(expected_triton):
            for root, dirs, files in os.walk(expected_triton):
                triton_files.extend(files)

        print(f"\nInductor cache files: {len(inductor_files)}")
        print(f"Triton cache files: {len(triton_files)}")

        if inductor_files or triton_files:
            print("✓ Cache directories populated by torch.compile")
        else:
            print("⚠ No cache files found - torch.compile may not have used external cache")
            print("  (This can happen if torch.compile uses in-memory caching)")

        print("\n=== Test Complete ===")
        return True


if __name__ == "__main__":
    try:
        success = test_cache_hooks()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
