#!/usr/bin/env python3
"""Example: torch.compile caching with Mithril.

This example demonstrates how to use Mithril's caching system to speed up
torch.compile cold starts.

Requirements:
    pip install torch mithril
"""

import os
import tempfile
import time

import mithril

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Note: torch not installed. Running cache-only demo.")


def setup_cache(cache_dir: str, max_size_gb: int = 10) -> mithril.cache.CacheManager:
    """Set up a Mithril cache for torch.compile artifacts.

    Args:
        cache_dir: Directory for cache storage
        max_size_gb: Maximum cache size in GB

    Returns:
        Configured CacheManager instance
    """
    config = mithril.CacheConfig(cache_dir).with_max_size_gb(max_size_gb)
    # Enable Inductor and Triton cache integration
    config = config.with_inductor(True).with_triton(True)
    return mithril.CacheManager(config)


def demo_content_store():
    """Demonstrate content-addressable storage."""
    print("\n--- Content-Addressable Storage Demo ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = mithril.ContentStore(tmpdir)
        print(f"Created ContentStore at: {store.root()}")

        # Store some content
        content = b"This is sample compiled kernel bytecode" * 100
        print(f"\nStoring {len(content)} bytes...")

        address = store.put(content)
        print(f"Content address: {address[:16]}...")

        # Compute address without storing
        computed_addr = store.compute_address(content)
        assert address == computed_addr, "Address mismatch!"
        print("Address computation verified")

        # Retrieve content
        retrieved = store.get(address)
        assert retrieved == content, "Content mismatch!"
        print("Content retrieval verified")

        # Check existence
        exists = store.exists(address)
        print(f"Content exists: {exists}")

        # Non-existent content
        fake_addr = "0" * 64
        missing = store.get(fake_addr)
        print(f"Non-existent content returns: {missing}")

        # Delete content
        deleted = store.delete(address)
        print(f"Content deleted: {deleted}")

        # Verify deletion
        exists_after = store.exists(address)
        print(f"Content exists after delete: {exists_after}")


def demo_cache_manager():
    """Demonstrate LRU cache management."""
    print("\n--- Cache Manager Demo ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = mithril.CacheConfig(tmpdir).with_max_size_gb(1)
        manager = mithril.CacheManager(config)

        print(f"Cache manager: {manager}")
        print(f"Inductor dir: {manager.inductor_dir()}")
        print(f"Triton dir: {manager.triton_dir()}")

        # Record some cache entries
        print("\nRecording cache entries...")
        entries = [
            ("kernel_abc123", 1024 * 1024),      # 1 MB
            ("kernel_def456", 2 * 1024 * 1024),  # 2 MB
            ("kernel_ghi789", 512 * 1024),       # 512 KB
        ]

        for key, size in entries:
            evicted = manager.record_entry(key, size)
            print(f"  Added {key} ({size // 1024} KB), evicted: {evicted}")

        # Show stats
        stats = manager.stats()
        print(f"\nCache stats: {stats}")
        print(f"  Entries: {stats.lru_entry_count}")
        print(f"  Size: {stats.lru_size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Utilization: {stats.utilization:.1%}")
        print(f"  Total hits: {stats.total_hits}")

        # Record access (cache hit)
        print("\nRecording cache access...")
        found = manager.record_access("kernel_abc123")
        print(f"  kernel_abc123 found: {found}")

        found = manager.record_access("nonexistent_kernel")
        print(f"  nonexistent_kernel found: {found}")

        # Updated stats
        stats = manager.stats()
        print(f"  Total hits after access: {stats.total_hits}")

        # Clear cache
        print("\nClearing cache...")
        manager.clear()
        stats = manager.stats()
        print(f"  Entries after clear: {stats.lru_entry_count}")


def demo_torch_compile_integration():
    """Demonstrate torch.compile cache integration."""
    if not HAS_TORCH:
        print("\n--- torch.compile Integration (skipped - torch not installed) ---")
        return

    print("\n--- torch.compile Integration Demo ---")

    with tempfile.TemporaryDirectory() as cache_dir:
        # Set up Mithril cache
        manager = setup_cache(cache_dir, max_size_gb=5)

        # Set environment variables for PyTorch's Inductor/Triton caches
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = manager.inductor_dir()
        os.environ["TRITON_CACHE_DIR"] = manager.triton_dir()

        print(f"Inductor cache: {manager.inductor_dir()}")
        print(f"Triton cache: {manager.triton_dir()}")

        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(512, 256)
                self.linear2 = torch.nn.Linear(256, 128)
                self.linear3 = torch.nn.Linear(128, 10)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = torch.relu(self.linear2(x))
                return self.linear3(x)

        # First compilation (cold start)
        print("\nFirst compilation (cold start)...")
        model = SimpleModel()
        compiled_model = torch.compile(model, mode="reduce-overhead")

        dummy_input = torch.randn(32, 512)

        start = time.perf_counter()
        _ = compiled_model(dummy_input)
        cold_time = time.perf_counter() - start
        print(f"  Cold start time: {cold_time:.2f}s")

        # Second call (warm)
        start = time.perf_counter()
        _ = compiled_model(dummy_input)
        warm_time = time.perf_counter() - start
        print(f"  Warm call time: {warm_time * 1000:.2f}ms")

        # Check cache stats
        stats = manager.stats()
        print(f"\nCache after compilation:")
        print(f"  Entries: {stats.lru_entry_count}")
        print(f"  Size: {stats.lru_size_bytes / 1024:.1f} KB")

        # Note: In a real scenario, the cache would persist across Python processes
        print("\nNote: For cross-process caching, use a persistent cache directory")
        print("and ensure TORCHINDUCTOR_CACHE_DIR is set before importing torch.")


def main():
    print("=" * 60)
    print("Mithril torch.compile Caching Example")
    print("=" * 60)

    # Demo content-addressable storage
    demo_content_store()

    # Demo cache manager
    demo_cache_manager()

    # Demo torch.compile integration
    demo_torch_compile_integration()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
