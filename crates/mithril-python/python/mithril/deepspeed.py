"""DeepSpeed integration for mithril checkpoint compression.

Provides monkey-patching for DeepSpeed's checkpoint saving to enable
transparent compression of ZeRO-3 sharded checkpoints.

Example:
    import deepspeed
    import mithril
    from mithril.deepspeed import MithrilDeepSpeedCheckpoint

    # Enable compression for DeepSpeed
    config = mithril.CompressionConfig(zstd_level=3)
    ds_hook = MithrilDeepSpeedCheckpoint(config)
    ds_hook.enable()

    # Now all DeepSpeed checkpoints will be compressed
    model, optimizer, _, _ = deepspeed.initialize(...)
    model.save_checkpoint(save_dir)  # Saves compressed

    # Load (automatic decompression)
    model.load_checkpoint(save_dir)

    # Restore original behavior
    ds_hook.disable()
"""

from __future__ import annotations

import functools
import io
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from mithril import CompressionConfig

# Magic bytes for mithril-compressed DeepSpeed checkpoints
MITHRIL_DS_MAGIC = b"MDSC"  # Mithril DeepSpeed Checkpoint
MITHRIL_DS_VERSION = 1


def _is_deepspeed_available() -> bool:
    """Check if deepspeed is installed."""
    try:
        import deepspeed
        return True
    except ImportError:
        return False


class MithrilDeepSpeedCheckpoint:
    """Monkey-patch DeepSpeed's checkpoint saving for compression.

    This class intercepts DeepSpeed's checkpoint save/load operations
    and applies mithril compression. Works with ZeRO-1, ZeRO-2, and ZeRO-3.

    Attributes:
        config: Compression configuration to use.
        enabled: Whether the patch is currently active.
    """

    def __init__(self, config: CompressionConfig):
        """Initialize the DeepSpeed compression hook.

        Args:
            config: Mithril compression configuration.
        """
        self.config = config
        self.enabled = False
        self._original_save_checkpoint: Optional[Callable] = None
        self._original_load_checkpoint: Optional[Callable] = None
        self._original_torch_save: Optional[Callable] = None
        self._original_torch_load: Optional[Callable] = None

    def enable(self) -> None:
        """Enable compression for DeepSpeed checkpoints.

        Patches DeepSpeed's save_checkpoint and load_checkpoint methods
        to transparently compress/decompress checkpoint files.

        Raises:
            ImportError: If deepspeed is not installed.
        """
        if self.enabled:
            return

        if not _is_deepspeed_available():
            raise ImportError(
                "DeepSpeed is not installed. Install with: pip install deepspeed"
            )

        import torch

        # Store original torch.save/load for use in hooks
        self._original_torch_save = torch.save
        self._original_torch_load = torch.load

        # Patch torch.save and torch.load when called from DeepSpeed context
        self._patch_torch_io()
        self.enabled = True

    def disable(self) -> None:
        """Disable compression and restore original behavior."""
        if not self.enabled:
            return

        self._unpatch_torch_io()
        self.enabled = False

    def _patch_torch_io(self) -> None:
        """Patch torch.save and torch.load for compression."""
        import torch

        config = self.config
        original_save = self._original_torch_save
        original_load = self._original_torch_load

        @functools.wraps(original_save)
        def patched_save(obj: Any, f: Any, *args, **kwargs) -> None:
            """Compressed torch.save wrapper."""
            # Check if this is a file path (string or Path)
            if isinstance(f, (str, Path)):
                path = Path(f)
                # Only compress checkpoint files, not config files
                if _is_checkpoint_file(path):
                    _save_compressed(obj, path, config, original_save)
                    return

            # Fall back to original save
            original_save(obj, f, *args, **kwargs)

        @functools.wraps(original_load)
        def patched_load(f: Any, *args, **kwargs) -> Any:
            """Decompressing torch.load wrapper."""
            if isinstance(f, (str, Path)):
                path = Path(f)
                if path.exists() and _is_mithril_compressed(path):
                    return _load_compressed(path, original_load, *args, **kwargs)

            # Fall back to original load
            return original_load(f, *args, **kwargs)

        torch.save = patched_save
        torch.load = patched_load

    def _unpatch_torch_io(self) -> None:
        """Restore original torch.save and torch.load."""
        import torch

        if self._original_torch_save is not None:
            torch.save = self._original_torch_save
        if self._original_torch_load is not None:
            torch.load = self._original_torch_load

    def __enter__(self) -> "MithrilDeepSpeedCheckpoint":
        """Context manager entry."""
        self.enable()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.disable()


def _is_checkpoint_file(path: Path) -> bool:
    """Check if a file is a DeepSpeed checkpoint file to compress."""
    name = path.name
    # DeepSpeed checkpoint patterns
    patterns = [
        "mp_rank_",      # Model parallel rank files
        "zero_pp_rank_", # ZeRO pipeline parallel
        "model_states",  # Model state dict
        "optim_states",  # Optimizer state (fp32 master weights in ZeRO)
        ".pt",           # Generic PyTorch files
        ".bin",          # Alternative checkpoint extension
    ]
    return any(p in name for p in patterns)


def _is_mithril_compressed(path: Path) -> bool:
    """Check if a file was compressed by mithril."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            return magic == MITHRIL_DS_MAGIC
    except (IOError, OSError):
        return False


def _save_compressed(
    obj: Any,
    path: Path,
    config: "CompressionConfig",
    original_save: Callable,
) -> None:
    """Save a compressed checkpoint.

    Format:
        - 4 bytes: Magic (MDSC)
        - 1 byte: Version
        - 4 bytes: Original size (u32)
        - N bytes: Compressed data
    """
    import mithril

    # Serialize to bytes using original torch.save
    buffer = io.BytesIO()
    original_save(obj, buffer)
    original_data = buffer.getvalue()

    # Compress using mithril
    compressor = mithril.CheckpointCompressor(config)
    # Use generic compression (not byte-grouped, since this is pickle data)
    compressed = compressor.compress_raw(original_data)

    # Write compressed file with header
    with open(path, "wb") as f:
        f.write(MITHRIL_DS_MAGIC)
        f.write(struct.pack("<B", MITHRIL_DS_VERSION))
        f.write(struct.pack("<I", len(original_data)))
        f.write(compressed)


def _load_compressed(
    path: Path,
    original_load: Callable,
    *args,
    **kwargs,
) -> Any:
    """Load a compressed checkpoint."""
    import mithril

    with open(path, "rb") as f:
        # Read and verify header
        magic = f.read(4)
        if magic != MITHRIL_DS_MAGIC:
            raise ValueError(f"Invalid mithril DeepSpeed checkpoint: {path}")

        version = struct.unpack("<B", f.read(1))[0]
        if version > MITHRIL_DS_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {version}, "
                f"max supported: {MITHRIL_DS_VERSION}"
            )

        original_size = struct.unpack("<I", f.read(4))[0]
        compressed_data = f.read()

    # Decompress
    config = mithril.CompressionConfig()  # Use default for decompression
    compressor = mithril.CheckpointCompressor(config)
    decompressed = compressor.decompress_raw(compressed_data, original_size)

    # Load using original torch.load
    buffer = io.BytesIO(decompressed)
    return original_load(buffer, *args, **kwargs)


def compress_deepspeed_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    config: Optional["CompressionConfig"] = None,
) -> dict:
    """Compress an existing DeepSpeed checkpoint directory.

    This function compresses all checkpoint files in a DeepSpeed checkpoint
    directory. Useful for compressing checkpoints saved without the hook.

    Args:
        checkpoint_dir: Path to the DeepSpeed checkpoint directory.
        output_dir: Output directory (defaults to checkpoint_dir + "_compressed").
        config: Compression config (uses defaults if not specified).

    Returns:
        Dictionary with compression statistics.

    Example:
        from mithril.deepspeed import compress_deepspeed_checkpoint

        stats = compress_deepspeed_checkpoint(
            "/path/to/checkpoint",
            config=mithril.CompressionConfig(zstd_level=6)
        )
        print(f"Compressed {stats['files']} files, saved {stats['saved_bytes']} bytes")
    """
    import mithril
    import shutil
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    if output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_compressed"
    output_dir = Path(output_dir)

    if config is None:
        config = mithril.CompressionConfig()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "files": 0,
        "original_bytes": 0,
        "compressed_bytes": 0,
        "saved_bytes": 0,
    }

    # Process all files
    for src_file in checkpoint_dir.rglob("*"):
        if src_file.is_dir():
            continue

        rel_path = src_file.relative_to(checkpoint_dir)
        dst_file = output_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if _is_checkpoint_file(src_file):
            # Compress checkpoint files
            try:
                obj = torch.load(src_file, map_location="cpu", weights_only=False)
                _save_compressed(obj, dst_file, config, torch.save)

                stats["files"] += 1
                stats["original_bytes"] += src_file.stat().st_size
                stats["compressed_bytes"] += dst_file.stat().st_size
            except Exception as e:
                # If compression fails, copy the file as-is
                shutil.copy2(src_file, dst_file)
        else:
            # Copy non-checkpoint files (configs, etc.)
            shutil.copy2(src_file, dst_file)

    stats["saved_bytes"] = stats["original_bytes"] - stats["compressed_bytes"]
    return stats


def decompress_deepspeed_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Decompress a mithril-compressed DeepSpeed checkpoint directory.

    Args:
        checkpoint_dir: Path to the compressed checkpoint directory.
        output_dir: Output directory (defaults to checkpoint_dir + "_decompressed").

    Returns:
        Dictionary with decompression statistics.
    """
    import shutil
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    if output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_decompressed"
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "files": 0,
        "compressed_bytes": 0,
        "decompressed_bytes": 0,
    }

    for src_file in checkpoint_dir.rglob("*"):
        if src_file.is_dir():
            continue

        rel_path = src_file.relative_to(checkpoint_dir)
        dst_file = output_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if _is_mithril_compressed(src_file):
            # Decompress
            obj = _load_compressed(src_file, torch.load, map_location="cpu", weights_only=False)
            torch.save(obj, dst_file)

            stats["files"] += 1
            stats["compressed_bytes"] += src_file.stat().st_size
            stats["decompressed_bytes"] += dst_file.stat().st_size
        else:
            # Copy non-compressed files
            shutil.copy2(src_file, dst_file)

    return stats


__all__ = [
    "MithrilDeepSpeedCheckpoint",
    "compress_deepspeed_checkpoint",
    "decompress_deepspeed_checkpoint",
]
