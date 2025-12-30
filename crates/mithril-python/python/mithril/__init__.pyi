"""Type stubs for mithril package."""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

__version__: str

# =============================================================================
# Checkpoint Module
# =============================================================================

class CompressionConfig:
    """Configuration for checkpoint compression."""

    def __init__(
        self,
        zstd_level: int = 3,
        byte_grouping: bool = True,
        num_threads: Optional[int] = None,
    ) -> None: ...
    @property
    def zstd_level(self) -> int: ...
    @property
    def byte_grouping(self) -> bool: ...

class CheckpointCompressor:
    """Compresses and decompresses checkpoint data."""

    def __init__(self, config: Optional[CompressionConfig] = None) -> None: ...
    def compress(self, data: bytes, dtype: str) -> bytes:
        """Compress raw tensor data."""
        ...
    def decompress(self, data: bytes, dtype: str, original_size: int) -> bytes:
        """Decompress to original tensor data."""
        ...

class DeltaCompressor:
    """Delta compression between checkpoints."""

    def __init__(self) -> None: ...
    def compress_checkpoint(
        self, checkpoint_id: str, data: bytes
    ) -> Tuple[bytes, DeltaStats]:
        """Compress a checkpoint, using delta encoding if possible."""
        ...
    def clear(self) -> None:
        """Clear the reference checkpoint."""
        ...

class DeltaStats:
    """Statistics from delta compression."""

    @property
    def ratio(self) -> float:
        """Compression ratio achieved."""
        ...
    @property
    def sparsity(self) -> float:
        """Fraction of zeros after XOR delta."""
        ...

class OrbaxWriter:
    """Writer for Orbax checkpoint format."""

    def __init__(self) -> None: ...
    def write(self, path: Union[str, Path], state_dict: Dict[str, Any]) -> OrbaxWriteStats: ...

class OrbaxWriteStats:
    """Statistics from Orbax write operation."""

    @property
    def tensors_written(self) -> int: ...
    @property
    def bytes_written(self) -> int: ...

# =============================================================================
# Cache Module
# =============================================================================

class CacheConfig:
    """Configuration for cache manager."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_bytes: Optional[int] = None,
    ) -> None: ...
    def with_inductor(self, enabled: bool) -> "CacheConfig": ...
    def with_triton(self, enabled: bool) -> "CacheConfig": ...
    @property
    def cache_dir(self) -> str: ...

class CacheManager:
    """Manages torch.compile cache directories."""

    def __init__(self, config: CacheConfig) -> None: ...
    def inductor_dir(self) -> Optional[str]:
        """Get the TorchInductor cache directory."""
        ...
    def triton_dir(self) -> Optional[str]:
        """Get the Triton cache directory."""
        ...
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...
    def clear(self) -> None:
        """Clear all cached data."""
        ...

class ContentStore:
    """Content-addressable storage."""

    def __init__(self, path: Union[str, Path]) -> None: ...
    async def put(self, data: bytes) -> str:
        """Store data and return content address."""
        ...
    async def get(self, address: str) -> Optional[bytes]:
        """Retrieve data by content address."""
        ...
    async def get_bytes(self, address: str) -> Optional[bytes]:
        """Retrieve data as bytes."""
        ...
    async def exists(self, address: str) -> bool:
        """Check if content exists."""
        ...
    async def delete(self, address: str) -> bool:
        """Delete content by address."""
        ...
    async def list_all(self) -> List[str]:
        """List all content addresses."""
        ...
    def compute_address(self, data: bytes) -> str:
        """Compute content address for data."""
        ...

class RemoteCacheConfig:
    """Configuration for remote cache."""

    def __init__(
        self,
        auto_push: bool = False,
        lazy_pull: bool = True,
        max_concurrent: int = 8,
    ) -> None: ...

# =============================================================================
# Dedup Module
# =============================================================================

class DedupConfig:
    """Configuration for deduplication."""

    def __init__(
        self,
        num_permutations: int = 128,
        threshold: float = 0.85,
        num_bands: Optional[int] = None,
        rows_per_band: Optional[int] = None,
    ) -> None: ...
    @property
    def num_permutations(self) -> int: ...
    @property
    def threshold(self) -> float: ...

class Deduplicator:
    """Text deduplication using MinHash/LSH."""

    def __init__(self, config: Optional[DedupConfig] = None) -> None: ...
    def deduplicate(self, documents: List[str]) -> List[int]:
        """Return indices of unique documents."""
        ...
    def find_duplicates(self, documents: List[str]) -> List[Tuple[int, int, float]]:
        """Find duplicate pairs with similarity scores."""
        ...

# =============================================================================
# S3 Remote Cache (optional)
# =============================================================================

class S3RemoteCache:
    """S3-compatible remote cache backend."""

    def __init__(
        self,
        bucket: str,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None: ...
    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def list(self, prefix: str = "") -> List[str]: ...

# =============================================================================
# DCP Integration (lazy loaded, requires torch)
# =============================================================================

def save_compressed(
    state_dict: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    config: Optional[CompressionConfig] = None,
) -> Dict[str, Any]:
    """Save a state dict with compression using PyTorch DCP."""
    ...

def load_compressed(
    state_dict: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    config: Optional[CompressionConfig] = None,
) -> None:
    """Load a compressed checkpoint into a state dict."""
    ...

class MithrilStorageWriter:
    """PyTorch DCP StorageWriter with Mithril compression."""
    ...

class MithrilStorageReader:
    """PyTorch DCP StorageReader for Mithril compressed checkpoints."""
    ...

# =============================================================================
# DeepSpeed Integration (lazy loaded, requires deepspeed)
# =============================================================================

class MithrilDeepSpeedCheckpoint:
    """DeepSpeed checkpoint with Mithril compression."""
    ...

def compress_deepspeed_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Compress a DeepSpeed checkpoint."""
    ...

def decompress_deepspeed_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """Decompress a Mithril-compressed DeepSpeed checkpoint."""
    ...

# =============================================================================
# Inductor Integration (lazy loaded, requires torch)
# =============================================================================

def patch_inductor(cache_dir: Optional[Union[str, Path]] = None) -> None:
    """Patch TorchInductor to use Mithril cache."""
    ...

def unpatch_inductor() -> None:
    """Remove Mithril patch from TorchInductor."""
    ...

class MithrilInductorCache:
    """Mithril-managed TorchInductor cache."""
    ...

def get_stats() -> Dict[str, Any]:
    """Get inductor cache statistics."""
    ...

def is_patched() -> bool:
    """Check if TorchInductor is currently patched."""
    ...

def has_remote() -> bool:
    """Check if remote cache is configured."""
    ...

def sync_cache() -> Dict[str, Any]:
    """Sync local and remote caches bidirectionally."""
    ...

def push_cache() -> Dict[str, Any]:
    """Push local cache to remote."""
    ...

def pull_cache() -> Dict[str, Any]:
    """Pull remote cache to local."""
    ...

def warmup_cache() -> Dict[str, Any]:
    """Warm local cache from remote (alias for pull)."""
    ...

def get_cache() -> Optional[ContentStore]:
    """Get the current cache instance."""
    ...

# =============================================================================
# Submodules
# =============================================================================

class checkpoint:
    """Checkpoint compression submodule."""

    CompressionConfig = CompressionConfig
    CheckpointCompressor = CheckpointCompressor
    DeltaCompressor = DeltaCompressor
    DeltaStats = DeltaStats

class cache:
    """Cache management submodule."""

    CacheConfig = CacheConfig
    CacheManager = CacheManager
    ContentStore = ContentStore
    RemoteCacheConfig = RemoteCacheConfig

class dedup:
    """Deduplication submodule."""

    DedupConfig = DedupConfig
    Deduplicator = Deduplicator
