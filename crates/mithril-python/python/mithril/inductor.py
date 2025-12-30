"""TorchInductor compilation cache integration.

This module provides hooks to integrate Mithril's caching layer with
TorchInductor's FxGraph compilation cache.

## Quick Start

```python
import mithril
from mithril.inductor import patch_inductor, unpatch_inductor

# Patch TorchInductor to use Mithril cache
patch_inductor(cache_dir="/path/to/cache")

# Now torch.compile will use Mithril's cache
import torch
model = torch.nn.Linear(10, 10)
compiled = torch.compile(model)

# When done, optionally unpatch
unpatch_inductor()
```

## Remote Cache (S3/GCS)

```python
from mithril.inductor import patch_inductor

# S3 remote backing
patch_inductor(
    cache_dir="/tmp/cache",
    remote_bucket="my-inductor-cache",
    remote_prefix="inductor/v1/",
    remote_region="us-west-2",
    auto_push=True,   # Push to remote on cache store
    lazy_pull=True,   # Pull from remote on cache miss
)
```

## Environment Variables

- `MITHRIL_CACHE_DIR`: Override cache directory
- `MITHRIL_INDUCTOR_ENABLED`: Set to "0" to disable patching
- `MITHRIL_INDUCTOR_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `MITHRIL_INDUCTOR_REMOTE_BACKEND`: Remote backend type ("s3", "gcs", or empty to disable)
- `MITHRIL_INDUCTOR_REMOTE_BUCKET`: Remote bucket name
- `MITHRIL_INDUCTOR_REMOTE_PREFIX`: Remote key prefix
- `MITHRIL_INDUCTOR_REMOTE_REGION`: AWS region (default: "us-east-1")
- `MITHRIL_INDUCTOR_REMOTE_ENDPOINT`: Custom endpoint (for MinIO, LocalStack)
- `MITHRIL_INDUCTOR_AUTO_PUSH`: Set to "1" to auto-push on store
- `MITHRIL_INDUCTOR_LAZY_PULL`: Set to "1" to lazy-pull on miss (default: "1")

## Compatibility

Tested with:
- PyTorch 2.0, 2.1, 2.2, 2.3, 2.4, 2.5
- CUDA 11.8, 12.1

Note: The internal TorchInductor APIs may change between PyTorch versions.
This module attempts to handle version differences gracefully.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("mithril.inductor")

# Track patching state
_original_load: Optional[Callable] = None
_original_save: Optional[Callable] = None
_cache_manager: Optional[Any] = None
_patched: bool = False


def _get_pytorch_version() -> Tuple[int, int, int]:
    """Get PyTorch version as tuple."""
    try:
        import torch
        version = torch.__version__.split('+')[0]  # Strip CUDA suffix
        parts = version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2].split('a')[0].split('b')[0].split('rc')[0]))
    except Exception:
        return (0, 0, 0)


def _is_inductor_available() -> bool:
    """Check if TorchInductor is available."""
    try:
        import torch._inductor
        return True
    except ImportError:
        return False


def _get_fx_graph_cache():
    """Get FxGraphCache class, handling version differences."""
    version = _get_pytorch_version()

    try:
        if version >= (2, 1, 0):
            # PyTorch 2.1+
            from torch._inductor.codecache import FxGraphCache
            return FxGraphCache
        elif version >= (2, 0, 0):
            # PyTorch 2.0
            from torch._inductor.codecache import PyCodeCache
            return PyCodeCache
        else:
            logger.warning(f"PyTorch {version} is not supported, need >= 2.0")
            return None
    except ImportError as e:
        logger.warning(f"Failed to import FxGraphCache: {e}")
        return None


def _compute_cache_key(gm_hash: str, example_inputs: Any, config: Any) -> str:
    """Compute a stable cache key from graph hash and inputs."""
    key_parts = [gm_hash]

    # Add input shapes and dtypes
    try:
        import torch
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor):
                key_parts.append(str(inp.shape))
                key_parts.append(str(inp.dtype))
                key_parts.append(str(inp.device.type))
    except Exception:
        pass

    # Add relevant config
    try:
        key_parts.append(str(hash(pickle.dumps(config))))
    except Exception:
        pass

    combined = ":".join(key_parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


class MithrilInductorCache:
    """Mithril-backed cache for TorchInductor compilations.

    Supports optional remote backing (S3/GCS) for cross-machine cache sharing.
    When remote is configured:
    - Local cache is checked first
    - On miss, lazy pull from remote (if lazy_pull=True)
    - On store, auto push to remote (if auto_push=True)
    """

    def __init__(
        self,
        cache_dir: str | Path,
        remote_bucket: Optional[str] = None,
        remote_prefix: str = "inductor/",
        remote_region: str = "us-east-1",
        remote_endpoint: Optional[str] = None,
        auto_push: bool = False,
        lazy_pull: bool = True,
    ):
        """Initialize the Inductor cache.

        Args:
            cache_dir: Directory to store cached artifacts.
            remote_bucket: S3/GCS bucket name for remote backing (optional).
            remote_prefix: Key prefix in remote bucket.
            remote_region: AWS region for S3.
            remote_endpoint: Custom endpoint (for MinIO, LocalStack).
            auto_push: Automatically push to remote after local writes.
            lazy_pull: Pull from remote on cache miss.
        """
        from mithril import CacheConfig, CacheManager

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        config = CacheConfig(str(self.cache_dir))
        self._manager = CacheManager(config)

        # Remote cache config
        self._remote = None
        self._auto_push = auto_push
        self._lazy_pull = lazy_pull

        if remote_bucket:
            self._init_remote(
                remote_bucket,
                remote_prefix,
                remote_region,
                remote_endpoint,
                auto_push,
                lazy_pull,
            )

        # Stats
        self.hits = 0
        self.misses = 0
        self.remote_hits = 0
        self.remote_misses = 0
        self.pushes = 0
        self.push_errors = 0

    def _init_remote(
        self,
        bucket: str,
        prefix: str,
        region: str,
        endpoint: Optional[str],
        auto_push: bool,
        lazy_pull: bool,
    ):
        """Initialize remote cache backend."""
        try:
            from mithril import S3RemoteCache, RemoteCacheConfig

            remote_config = RemoteCacheConfig(
                auto_push=auto_push,
                lazy_pull=lazy_pull,
            )

            self._remote = S3RemoteCache(
                local_path=str(self.cache_dir),
                bucket=bucket,
                prefix=prefix,
                region=region,
                endpoint=endpoint,
                config=remote_config,
            )
            logger.info(f"Remote cache enabled: s3://{bucket}/{prefix}")
        except ImportError:
            logger.warning("S3RemoteCache not available (built without s3 feature)")
        except Exception as e:
            logger.warning(f"Failed to initialize remote cache: {e}")

    def get(self, key: str) -> Optional[bytes]:
        """Get cached artifact.

        Checks local cache first, then remote (if lazy_pull is enabled).

        Args:
            key: Cache key.

        Returns:
            Cached data or None if not found.
        """
        try:
            cache_path = self.cache_dir / f"{key}.mithril"

            # Check local first
            if cache_path.exists():
                self.hits += 1
                logger.debug(f"Local cache hit: {key[:8]}...")
                return cache_path.read_bytes()

            # Try remote if lazy_pull is enabled
            if self._remote and self._lazy_pull:
                remote_data = self._get_from_remote(key)
                if remote_data is not None:
                    # Store locally for next time
                    try:
                        cache_path.write_bytes(remote_data)
                    except Exception as e:
                        logger.debug(f"Failed to write local cache: {e}")
                    self.hits += 1
                    self.remote_hits += 1
                    logger.debug(f"Remote cache hit: {key[:8]}...")
                    return remote_data
                self.remote_misses += 1

            self.misses += 1
            return None
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            self.misses += 1
            return None

    def _get_from_remote(self, key: str) -> Optional[bytes]:
        """Fetch artifact from remote cache."""
        if not self._remote:
            return None
        try:
            # Use content address as the key
            return self._remote.get(key)
        except Exception as e:
            logger.debug(f"Remote get error: {e}")
            return None

    def put(self, key: str, data: bytes) -> bool:
        """Store artifact in cache.

        Stores locally, and optionally pushes to remote (if auto_push is enabled).

        Args:
            key: Cache key.
            data: Data to cache.

        Returns:
            True if stored successfully.
        """
        try:
            cache_path = self.cache_dir / f"{key}.mithril"
            cache_path.write_bytes(data)

            # Auto-push to remote if enabled
            if self._remote and self._auto_push:
                self._push_to_remote(key, data)

            return True
        except Exception as e:
            logger.debug(f"Cache put error: {e}")
            return False

    def _push_to_remote(self, key: str, data: bytes):
        """Push artifact to remote cache."""
        if not self._remote:
            return
        try:
            self._remote.put(data)
            self.pushes += 1
            logger.debug(f"Pushed to remote: {key[:8]}...")
        except Exception as e:
            self.push_errors += 1
            logger.debug(f"Remote push error: {e}")

    def sync(self) -> Dict[str, Any]:
        """Sync local and remote caches bidirectionally.

        Returns:
            Dictionary with sync statistics.
        """
        if not self._remote:
            return {"enabled": False}

        try:
            stats = self._remote.sync()
            return {
                "enabled": True,
                "pushed": stats.push.uploaded,
                "pulled": stats.pull.downloaded,
                "push_skipped": stats.push.skipped,
                "pull_skipped": stats.pull.skipped,
            }
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {"enabled": True, "error": str(e)}

    def push(self) -> Dict[str, Any]:
        """Push all local artifacts to remote.

        Returns:
            Dictionary with push statistics.
        """
        if not self._remote:
            return {"enabled": False}

        try:
            stats = self._remote.push()
            return {
                "enabled": True,
                "uploaded": stats.uploaded,
                "skipped": stats.skipped,
                "bytes": stats.bytes,
                "failed": stats.failed,
            }
        except Exception as e:
            logger.error(f"Push failed: {e}")
            return {"enabled": True, "error": str(e)}

    def pull(self) -> Dict[str, Any]:
        """Pull all remote artifacts to local.

        Useful for warming the cache before training starts.

        Returns:
            Dictionary with pull statistics.
        """
        if not self._remote:
            return {"enabled": False}

        try:
            stats = self._remote.pull()
            return {
                "enabled": True,
                "downloaded": stats.downloaded,
                "skipped": stats.skipped,
                "bytes": stats.bytes,
                "failed": stats.failed,
            }
        except Exception as e:
            logger.error(f"Pull failed: {e}")
            return {"enabled": True, "error": str(e)}

    def warmup(self) -> Dict[str, Any]:
        """Warm local cache from remote.

        Alias for pull() - useful for CI/CD cache warming.
        """
        return self.pull()

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def remote_hit_rate(self) -> float:
        """Get remote cache hit rate (among local misses)."""
        total = self.remote_hits + self.remote_misses
        return self.remote_hits / total if total > 0 else 0.0

    @property
    def has_remote(self) -> bool:
        """Check if remote cache is configured."""
        return self._remote is not None


def _make_cached_load(original_load: Callable, cache: MithrilInductorCache) -> Callable:
    """Create a cached version of FxGraphCache.load."""

    @functools.wraps(original_load)
    def cached_load(*args, **kwargs):
        # Try to compute cache key
        try:
            if len(args) >= 2:
                gm_hash = str(args[0])
                example_inputs = args[1] if len(args) > 1 else []
                config = args[2] if len(args) > 2 else {}
                key = _compute_cache_key(gm_hash, example_inputs, config)

                # Check Mithril cache first
                cached_data = cache.get(key)
                if cached_data is not None:
                    logger.debug(f"Mithril cache hit for {key[:8]}...")
                    try:
                        return pickle.loads(cached_data)
                    except Exception as e:
                        logger.debug(f"Failed to unpickle cached data: {e}")
        except Exception as e:
            logger.debug(f"Cache key computation failed: {e}")

        # Fall back to original
        result = original_load(*args, **kwargs)

        # Store in Mithril cache if we got a result
        if result is not None:
            try:
                key = _compute_cache_key(str(args[0]), args[1] if len(args) > 1 else [],
                                         args[2] if len(args) > 2 else {})
                cache.put(key, pickle.dumps(result))
                logger.debug(f"Stored in Mithril cache: {key[:8]}...")
            except Exception as e:
                logger.debug(f"Failed to store in Mithril cache: {e}")

        return result

    return cached_load


def _make_cached_save(original_save: Callable, cache: MithrilInductorCache) -> Callable:
    """Create a cached version of FxGraphCache.save."""

    @functools.wraps(original_save)
    def cached_save(*args, **kwargs):
        # Let the original save happen
        result = original_save(*args, **kwargs)

        # Also store in Mithril cache
        try:
            if len(args) >= 3:
                gm_hash = str(args[0])
                example_inputs = args[1] if len(args) > 1 else []
                config = args[2] if len(args) > 2 else {}
                key = _compute_cache_key(gm_hash, example_inputs, config)

                # Store the compiled code
                if len(args) > 3:
                    cache.put(key, pickle.dumps(args[3]))
                    logger.debug(f"Saved to Mithril cache: {key[:8]}...")
        except Exception as e:
            logger.debug(f"Failed to save to Mithril cache: {e}")

        return result

    return cached_save


def _get_remote_config_from_env() -> Dict[str, Any]:
    """Get remote cache configuration from environment variables."""
    config = {}

    backend = os.environ.get("MITHRIL_INDUCTOR_REMOTE_BACKEND", "")
    if backend.lower() in ("s3", "gcs"):
        bucket = os.environ.get("MITHRIL_INDUCTOR_REMOTE_BUCKET")
        if bucket:
            config["remote_bucket"] = bucket
            config["remote_prefix"] = os.environ.get(
                "MITHRIL_INDUCTOR_REMOTE_PREFIX", "inductor/"
            )
            config["remote_region"] = os.environ.get(
                "MITHRIL_INDUCTOR_REMOTE_REGION", "us-east-1"
            )
            config["remote_endpoint"] = os.environ.get(
                "MITHRIL_INDUCTOR_REMOTE_ENDPOINT"
            )
            config["auto_push"] = os.environ.get(
                "MITHRIL_INDUCTOR_AUTO_PUSH", "0"
            ) == "1"
            config["lazy_pull"] = os.environ.get(
                "MITHRIL_INDUCTOR_LAZY_PULL", "1"
            ) == "1"

    return config


def patch_inductor(
    cache_dir: Optional[str | Path] = None,
    log_level: str = "WARNING",
    remote_bucket: Optional[str] = None,
    remote_prefix: str = "inductor/",
    remote_region: str = "us-east-1",
    remote_endpoint: Optional[str] = None,
    auto_push: bool = False,
    lazy_pull: bool = True,
) -> bool:
    """Patch TorchInductor to use Mithril's caching layer.

    Args:
        cache_dir: Directory to store cached artifacts.
                   Defaults to MITHRIL_CACHE_DIR env or ~/.mithril/inductor.
        log_level: Logging level for mithril.inductor logger.
        remote_bucket: S3/GCS bucket name for remote backing (optional).
                       Can also be set via MITHRIL_INDUCTOR_REMOTE_BUCKET.
        remote_prefix: Key prefix in remote bucket.
        remote_region: AWS region for S3.
        remote_endpoint: Custom endpoint (for MinIO, LocalStack).
        auto_push: Automatically push to remote after local writes.
        lazy_pull: Pull from remote on cache miss.

    Returns:
        True if patched successfully, False otherwise.

    Example:
        # Local only
        patch_inductor(cache_dir="/tmp/cache")

        # With S3 remote backing
        patch_inductor(
            cache_dir="/tmp/cache",
            remote_bucket="my-bucket",
            remote_prefix="inductor/",
            auto_push=True,
        )

        # Using environment variables
        # MITHRIL_INDUCTOR_REMOTE_BACKEND=s3
        # MITHRIL_INDUCTOR_REMOTE_BUCKET=my-bucket
        patch_inductor()
    """
    global _original_load, _original_save, _cache_manager, _patched

    if _patched:
        logger.debug("Already patched, skipping")
        return True

    # Check if disabled
    if os.environ.get("MITHRIL_INDUCTOR_ENABLED", "1") == "0":
        logger.info("Mithril Inductor hooks disabled via MITHRIL_INDUCTOR_ENABLED=0")
        return False

    # Set up logging
    log_level = os.environ.get("MITHRIL_INDUCTOR_LOG_LEVEL", log_level)
    logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))

    # Check availability
    if not _is_inductor_available():
        logger.warning("TorchInductor not available")
        return False

    FxGraphCache = _get_fx_graph_cache()
    if FxGraphCache is None:
        logger.warning("Could not get FxGraphCache class")
        return False

    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.environ.get(
            "MITHRIL_CACHE_DIR",
            str(Path.home() / ".mithril" / "inductor")
        )

    # Get remote config from env if not specified in args
    env_config = _get_remote_config_from_env()
    if remote_bucket is None:
        remote_bucket = env_config.get("remote_bucket")
        remote_prefix = env_config.get("remote_prefix", remote_prefix)
        remote_region = env_config.get("remote_region", remote_region)
        remote_endpoint = env_config.get("remote_endpoint", remote_endpoint)
        auto_push = env_config.get("auto_push", auto_push)
        lazy_pull = env_config.get("lazy_pull", lazy_pull)

    # Create cache
    try:
        _cache_manager = MithrilInductorCache(
            cache_dir,
            remote_bucket=remote_bucket,
            remote_prefix=remote_prefix,
            remote_region=remote_region,
            remote_endpoint=remote_endpoint,
            auto_push=auto_push,
            lazy_pull=lazy_pull,
        )
        logger.info(f"Mithril Inductor cache initialized at {cache_dir}")
        if remote_bucket:
            logger.info(f"Remote backing: s3://{remote_bucket}/{remote_prefix}")
    except Exception as e:
        logger.error(f"Failed to create cache: {e}")
        return False

    # Patch load/save methods
    try:
        version = _get_pytorch_version()

        if version >= (2, 1, 0):
            # PyTorch 2.1+: FxGraphCache uses class methods
            if hasattr(FxGraphCache, 'load'):
                _original_load = FxGraphCache.load
                FxGraphCache.load = staticmethod(_make_cached_load(_original_load, _cache_manager))

            if hasattr(FxGraphCache, 'save'):
                _original_save = FxGraphCache.save
                FxGraphCache.save = staticmethod(_make_cached_save(_original_save, _cache_manager))
        else:
            # PyTorch 2.0: Different API
            logger.warning("PyTorch 2.0 inductor hooks are experimental")

        _patched = True
        logger.info("TorchInductor successfully patched")
        return True

    except Exception as e:
        logger.error(f"Failed to patch TorchInductor: {e}")
        return False


def unpatch_inductor() -> bool:
    """Remove Mithril's TorchInductor patches.

    Returns:
        True if unpatched successfully.
    """
    global _original_load, _original_save, _cache_manager, _patched

    if not _patched:
        logger.debug("Not patched, nothing to unpatch")
        return True

    FxGraphCache = _get_fx_graph_cache()
    if FxGraphCache is None:
        return False

    try:
        if _original_load is not None:
            FxGraphCache.load = _original_load
            _original_load = None

        if _original_save is not None:
            FxGraphCache.save = _original_save
            _original_save = None

        _patched = False
        logger.info("TorchInductor unpatched")
        return True

    except Exception as e:
        logger.error(f"Failed to unpatch: {e}")
        return False


def get_stats() -> Dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dictionary with cache stats including remote stats if enabled.
    """
    if _cache_manager is None:
        return {"enabled": False}

    stats = {
        "enabled": True,
        "hits": _cache_manager.hits,
        "misses": _cache_manager.misses,
        "hit_rate": _cache_manager.hit_rate,
        "cache_dir": str(_cache_manager.cache_dir),
        "has_remote": _cache_manager.has_remote,
    }

    if _cache_manager.has_remote:
        stats.update({
            "remote_hits": _cache_manager.remote_hits,
            "remote_misses": _cache_manager.remote_misses,
            "remote_hit_rate": _cache_manager.remote_hit_rate,
            "pushes": _cache_manager.pushes,
            "push_errors": _cache_manager.push_errors,
        })

    return stats


def is_patched() -> bool:
    """Check if TorchInductor is patched.

    Returns:
        True if patched.
    """
    return _patched


def has_remote() -> bool:
    """Check if remote cache is configured.

    Returns:
        True if remote cache is enabled.
    """
    return _cache_manager is not None and _cache_manager.has_remote


def sync_cache() -> Dict[str, Any]:
    """Sync local and remote caches bidirectionally.

    This pushes local artifacts to remote and pulls remote artifacts to local.
    Useful for keeping caches in sync across machines.

    Returns:
        Dictionary with sync statistics, or {"enabled": False} if no remote.
    """
    if _cache_manager is None:
        return {"enabled": False, "error": "Inductor cache not initialized"}
    return _cache_manager.sync()


def push_cache() -> Dict[str, Any]:
    """Push all local cached artifacts to remote storage.

    Uploads all artifacts in the local cache to the configured remote bucket.

    Returns:
        Dictionary with push statistics, or {"enabled": False} if no remote.
    """
    if _cache_manager is None:
        return {"enabled": False, "error": "Inductor cache not initialized"}
    return _cache_manager.push()


def pull_cache() -> Dict[str, Any]:
    """Pull all remote cached artifacts to local storage.

    Downloads all artifacts from the configured remote bucket to local cache.
    Useful for warming the cache before training starts.

    Returns:
        Dictionary with pull statistics, or {"enabled": False} if no remote.
    """
    if _cache_manager is None:
        return {"enabled": False, "error": "Inductor cache not initialized"}
    return _cache_manager.pull()


def warmup_cache() -> Dict[str, Any]:
    """Warm local cache from remote.

    Alias for pull_cache() - downloads remote artifacts to local cache.
    Useful for CI/CD pipelines to warm the cache before training.

    Example:
        # In CI/CD setup script
        from mithril.inductor import patch_inductor, warmup_cache

        patch_inductor(
            remote_bucket="my-bucket",
            remote_prefix="inductor/",
        )
        stats = warmup_cache()
        print(f"Downloaded {stats['downloaded']} artifacts")

    Returns:
        Dictionary with pull statistics.
    """
    return pull_cache()


def get_cache() -> Optional[MithrilInductorCache]:
    """Get the underlying cache manager.

    Returns:
        MithrilInductorCache instance, or None if not initialized.
    """
    return _cache_manager


# Auto-patch on import if MITHRIL_INDUCTOR_AUTO_PATCH=1
if os.environ.get("MITHRIL_INDUCTOR_AUTO_PATCH", "0") == "1":
    patch_inductor()
