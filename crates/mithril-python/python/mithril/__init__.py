"""Mithril: High-performance ML infrastructure toolkit.

This package provides:
- Checkpoint compression for PyTorch models
- Dataset deduplication with MinHash/LSH
- Compilation caching for torch.compile
"""

from mithril._mithril import (
    # Checkpoint
    CompressionConfig,
    CheckpointCompressor,
    DeltaCompressor,
    DeltaStats,
    OrbaxWriter,
    OrbaxWriteStats,
    QuantizeConfig,
    Quantizer,
    MstWriter,
    MstReader,
    MstTensorInfo,
    is_mst,
    # Cache
    CacheConfig,
    CacheManager,
    ContentStore,
    RemoteCacheConfig,
    # Dedup
    DedupConfig,
    Deduplicator,
    # Submodules
    checkpoint,
    cache,
    dedup,
    # Metadata
    __version__,
)

# Optional S3 remote cache (requires s3 feature)
try:
    from mithril._mithril import S3RemoteCache
    _has_s3 = True
except ImportError:
    _has_s3 = False

# Lazy imports for optional dependencies
_dcp_names = ("MithrilStorageWriter", "MithrilStorageReader", "save_compressed", "load_compressed")
_deepspeed_names = ("MithrilDeepSpeedCheckpoint", "compress_deepspeed_checkpoint", "decompress_deepspeed_checkpoint")
_inductor_names = (
    "patch_inductor", "unpatch_inductor", "MithrilInductorCache",
    "get_stats", "is_patched", "has_remote", "sync_cache", "push_cache", "pull_cache", "warmup_cache", "get_cache",
)
_async_names = (
    "compress_async", "decompress_async", "compress_raw_async", "decompress_raw_async",
    "compress_file_async", "decompress_file_async", "store_async", "retrieve_async",
)
_streaming_names = ("StreamingCompressor", "StreamingDecompressor")

def __getattr__(name):
    """Lazy import for DCP, DeepSpeed, Inductor, and async API."""
    if name in _dcp_names:
        from mithril import dcp
        return getattr(dcp, name)
    if name in _deepspeed_names:
        from mithril import deepspeed
        return getattr(deepspeed, name)
    if name in _inductor_names:
        from mithril import inductor
        return getattr(inductor, name)
    if name in _async_names:
        from mithril import async_api
        return getattr(async_api, name)
    if name in _streaming_names:
        from mithril import streaming
        return getattr(streaming, name)
    raise AttributeError(f"module 'mithril' has no attribute '{name}'")

__all__ = [
    # Checkpoint
    "CompressionConfig",
    "CheckpointCompressor",
    "DeltaCompressor",
    "DeltaStats",
    "OrbaxWriter",
    "OrbaxWriteStats",
    "QuantizeConfig",
    "Quantizer",
    "MstWriter",
    "MstReader",
    "MstTensorInfo",
    "is_mst",
    # Cache
    "CacheConfig",
    "CacheManager",
    "ContentStore",
    "RemoteCacheConfig",
    # Dedup
    "DedupConfig",
    "Deduplicator",
    # DCP integration (lazy loaded)
    "MithrilStorageWriter",
    "MithrilStorageReader",
    "save_compressed",
    "load_compressed",
    # DeepSpeed integration (lazy loaded)
    "MithrilDeepSpeedCheckpoint",
    "compress_deepspeed_checkpoint",
    "decompress_deepspeed_checkpoint",
    # Inductor integration (lazy loaded)
    "patch_inductor",
    "unpatch_inductor",
    "MithrilInductorCache",
    "get_stats",
    "is_patched",
    "has_remote",
    "sync_cache",
    "push_cache",
    "pull_cache",
    "warmup_cache",
    "get_cache",
    # Submodules
    "checkpoint",
    "cache",
    "dedup",
    # Async API (lazy loaded)
    "compress_async",
    "decompress_async",
    "compress_raw_async",
    "decompress_raw_async",
    "compress_file_async",
    "decompress_file_async",
    "store_async",
    "retrieve_async",
    # Streaming API (lazy loaded)
    "StreamingCompressor",
    "StreamingDecompressor",
]

# Add S3RemoteCache to exports if available
if _has_s3:
    __all__.append("S3RemoteCache")
