"""PyTorch Distributed Checkpoint (DCP) integration for Mithril.

This module provides compressed checkpoint saving/loading for FSDP models
using PyTorch's torch.distributed.checkpoint API.

Strategy: We wrap DCP's standard save/load with file-level compression.
This is simpler and more robust than trying to intercept the DCP pipeline.

Example:
    import torch
    import mithril
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # Wrap model with FSDP
    model = FSDP(MyModel())

    # Save with compression (drop-in replacement for dcp.save)
    mithril.save_compressed(
        state_dict={"model": model.state_dict()},
        checkpoint_path="./checkpoints/step_1000",
    )

    # Load with decompression (drop-in replacement for dcp.load)
    state_dict = {"model": model.state_dict()}
    mithril.load_compressed(
        state_dict=state_dict,
        checkpoint_path="./checkpoints/step_1000",
    )
    model.load_state_dict(state_dict["model"])
"""

from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import (
        FileSystemWriter,
        FileSystemReader,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mithril._mithril import CheckpointCompressor, CompressionConfig


# File extension for mithril compressed files
MITHRIL_EXT = ".mcp"
# Magic bytes for verification
MITHRIL_MAGIC = b"MCPT"  # Mithril CheckPoint
MITHRIL_VERSION = 1


def _dtype_to_str(dtype: "torch.dtype") -> str:
    """Convert torch dtype to mithril dtype string."""
    if not HAS_TORCH:
        return "fp32"
    dtype_map = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.float64: "fp64",
        torch.int8: "i8",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.uint8: "u8",
        torch.bool: "bool",
    }
    return dtype_map.get(dtype, "fp32")


class MithrilStorageWriter:
    """Storage writer that compresses checkpoint data using Mithril.

    This writer uses standard DCP FileSystemWriter, then compresses the
    resulting files for 10x+ compression on bf16/fp16 model weights.

    Example:
        writer = MithrilStorageWriter("./checkpoint")

        # Using with DCP directly
        import torch.distributed.checkpoint as dcp
        dcp.save(state_dict, storage_writer=writer.fs_writer)
        writer.compress_files()

        # Or use the convenience function
        mithril.save_compressed(state_dict, "./checkpoint")
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        config: Optional[CompressionConfig] = None,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 4,
    ):
        """Initialize the Mithril storage writer.

        Args:
            path: Directory to save checkpoint to.
            config: Compression configuration. Defaults to balanced settings.
            single_file_per_rank: If True, each rank writes to a single file.
            sync_files: If True, sync files after writing.
            thread_count: Number of threads for parallel compression.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DCP integration. Install with: pip install torch")

        self.path = Path(path)
        self.config = config or CompressionConfig()
        self.compressor = CheckpointCompressor(self.config)
        self.thread_count = thread_count

        # Create underlying FileSystemWriter
        self.fs_writer = FileSystemWriter(
            path=str(path),
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=1,  # We'll parallelize compression ourselves
        )

        # Track stats
        self._original_bytes = 0
        self._compressed_bytes = 0

    def compress_files(self, delete_originals: bool = True) -> Dict[str, Any]:
        """Compress all checkpoint files in the directory.

        Args:
            delete_originals: If True, delete original files after compression.

        Returns:
            Compression statistics.
        """
        # Find all .pt and .distcp files
        files_to_compress = []
        for pattern in ["*.pt", "*.distcp", "__*_0"]:
            files_to_compress.extend(self.path.glob(pattern))

        # Also find rank-specific files
        for f in self.path.iterdir():
            if f.is_file() and not f.suffix and not f.name.startswith("."):
                # Likely a DCP rank file (no extension)
                files_to_compress.append(f)

        # Filter to only include actual data files
        files_to_compress = [
            f for f in files_to_compress
            if f.is_file() and not f.name.endswith(MITHRIL_EXT)
        ]

        if not files_to_compress:
            return {"original_bytes": 0, "compressed_bytes": 0, "ratio": 1.0}

        def compress_file(file_path: Path) -> tuple:
            """Compress a single file."""
            data = file_path.read_bytes()
            original_size = len(data)

            # Use compress_raw for file data (safetensors format)
            # compress() with dtype requires aligned raw tensor bytes
            compressed = self.compressor.compress_raw(data)

            # Write compressed file with header
            output_path = file_path.with_suffix(file_path.suffix + MITHRIL_EXT)
            with open(output_path, "wb") as f:
                # Write header
                f.write(MITHRIL_MAGIC)
                f.write(struct.pack("<B", MITHRIL_VERSION))
                f.write(struct.pack("<Q", original_size))
                f.write(compressed)

            compressed_size = output_path.stat().st_size

            if delete_originals:
                file_path.unlink()

            return (original_size, compressed_size)

        # Compress files in parallel
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            results = list(executor.map(compress_file, files_to_compress))

        self._original_bytes = sum(r[0] for r in results)
        self._compressed_bytes = sum(r[1] for r in results)

        return self.compression_stats

    @property
    def compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        ratio = (
            self._original_bytes / max(self._compressed_bytes, 1)
            if self._compressed_bytes > 0
            else 1.0
        )
        return {
            "original_bytes": self._original_bytes,
            "compressed_bytes": self._compressed_bytes,
            "ratio": ratio,
        }


class MithrilStorageReader:
    """Storage reader that decompresses checkpoint data using Mithril.

    This reader decompresses Mithril-compressed files, then uses standard
    DCP FileSystemReader to load the checkpoint.

    Example:
        reader = MithrilStorageReader("./checkpoint")
        reader.decompress_files()

        # Using with DCP directly
        import torch.distributed.checkpoint as dcp
        dcp.load(state_dict, storage_reader=reader.fs_reader)

        # Or use the convenience function
        mithril.load_compressed(state_dict, "./checkpoint")
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        config: Optional[CompressionConfig] = None,
        thread_count: int = 4,
    ):
        """Initialize the Mithril storage reader.

        Args:
            path: Directory to load checkpoint from.
            config: Compression configuration (must match writer config).
            thread_count: Number of threads for parallel decompression.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DCP integration. Install with: pip install torch")

        self.path = Path(path)
        self.config = config or CompressionConfig()
        self.compressor = CheckpointCompressor(self.config)
        self.thread_count = thread_count

        # Will be created after decompression
        self.fs_reader = None
        self._temp_files: List[Path] = []

    def decompress_files(self, in_place: bool = False) -> None:
        """Decompress all compressed checkpoint files.

        Args:
            in_place: If True, replace compressed files with decompressed ones.
                     If False, decompress to temp files (cleaned up on close).
        """
        # Find all .mcp files
        compressed_files = list(self.path.glob(f"*{MITHRIL_EXT}"))

        if not compressed_files:
            # No compression, use files directly
            self.fs_reader = FileSystemReader(str(self.path))
            return

        def decompress_file(file_path: Path) -> Path:
            """Decompress a single file."""
            with open(file_path, "rb") as f:
                # Read and verify header
                magic = f.read(4)
                if magic != MITHRIL_MAGIC:
                    raise ValueError(f"Invalid mithril file: {file_path}")

                version = struct.unpack("<B", f.read(1))[0]
                if version != MITHRIL_VERSION:
                    raise ValueError(f"Unsupported mithril version: {version}")

                original_size = struct.unpack("<Q", f.read(8))[0]
                compressed = f.read()

            # Decompress (use decompress_raw to match compress_raw)
            decompressed = self.compressor.decompress_raw(compressed, original_size)

            # Write decompressed file
            # Remove the .mcp extension to get original filename
            output_path = Path(str(file_path)[:-len(MITHRIL_EXT)])

            output_path.write_bytes(decompressed)

            if in_place:
                file_path.unlink()

            return output_path

        # Decompress files in parallel
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            decompressed_files = list(executor.map(decompress_file, compressed_files))

        if not in_place:
            self._temp_files = decompressed_files

        self.fs_reader = FileSystemReader(str(self.path))

    def cleanup(self) -> None:
        """Clean up temporary decompressed files."""
        for f in self._temp_files:
            if f.exists():
                f.unlink()
        self._temp_files = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


def save_compressed(
    state_dict: Dict[str, Any],
    checkpoint_path: Union[str, os.PathLike],
    config: Optional[CompressionConfig] = None,
    process_group: Optional[Any] = None,
    thread_count: int = 4,
) -> Dict[str, Any]:
    """Save a state dict with Mithril compression.

    This is a high-level convenience function that wraps
    torch.distributed.checkpoint.save with Mithril compression.

    Args:
        state_dict: The state dict to save.
        checkpoint_path: Directory to save checkpoint to.
        config: Compression configuration.
        process_group: Optional distributed process group.
        thread_count: Number of threads for parallel compression.

    Returns:
        Compression statistics dict with original_bytes, compressed_bytes, ratio.

    Example:
        import mithril

        # Save model checkpoint with compression
        stats = mithril.save_compressed(
            {"model": model.state_dict()},
            "./checkpoints/step_1000",
        )
        print(f"Compression ratio: {stats['ratio']:.1f}x")
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for DCP integration. Install with: pip install torch")

    # Create directory if needed
    path = Path(checkpoint_path)
    path.mkdir(parents=True, exist_ok=True)

    # Create writer
    writer = MithrilStorageWriter(path, config=config, thread_count=thread_count)

    # Save using standard DCP
    dcp.save(state_dict, storage_writer=writer.fs_writer, process_group=process_group)

    # Compress the files
    stats = writer.compress_files(delete_originals=True)

    return stats


def load_compressed(
    state_dict: Dict[str, Any],
    checkpoint_path: Union[str, os.PathLike],
    config: Optional[CompressionConfig] = None,
    process_group: Optional[Any] = None,
    thread_count: int = 4,
) -> None:
    """Load a state dict with Mithril decompression.

    This is a high-level convenience function that wraps
    torch.distributed.checkpoint.load with Mithril decompression.

    Args:
        state_dict: The state dict to load into (modified in place).
        checkpoint_path: Directory to load checkpoint from.
        config: Compression configuration (must match writer config).
        process_group: Optional distributed process group.
        thread_count: Number of threads for parallel decompression.

    Example:
        import mithril

        # Load model checkpoint with decompression
        state_dict = {"model": model.state_dict()}
        mithril.load_compressed(
            state_dict,
            "./checkpoints/step_1000",
        )
        model.load_state_dict(state_dict["model"])
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for DCP integration. Install with: pip install torch")

    # Create reader and decompress files
    with MithrilStorageReader(checkpoint_path, config=config, thread_count=thread_count) as reader:
        reader.decompress_files(in_place=False)

        # Load using standard DCP
        dcp.load(state_dict, storage_reader=reader.fs_reader, process_group=process_group)
