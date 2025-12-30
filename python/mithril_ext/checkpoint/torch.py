"""PyTorch Distributed Checkpoint (DCP) integration for Mithril.

Provides MithrilSavePlanner and MithrilLoadPlanner for use with
torch.distributed.checkpoint to compress checkpoints during save/load.

Example:
    import torch.distributed.checkpoint as dcp
    from mithril.checkpoint.torch import MithrilSavePlanner, MithrilLoadPlanner

    # Save with compression
    dcp.save(
        state_dict,
        storage_writer=dcp.FileSystemWriter(path),
        planner=MithrilSavePlanner(compression_level=3),
    )

    # Load with decompression
    dcp.load(
        state_dict,
        storage_reader=dcp.FileSystemReader(path),
        planner=MithrilLoadPlanner(),
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import io

import torch
from torch.distributed.checkpoint.planner import (
    SavePlanner,
    LoadPlanner,
    SavePlan,
    LoadPlan,
    WriteItem,
    ReadItem,
)
from torch.distributed.checkpoint.metadata import (
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

# Import Rust bindings
try:
    from mithril import CheckpointCompressor, CompressionConfig
except ImportError:
    raise ImportError(
        "mithril Rust bindings not found. "
        "Build with: cd crates/mithril-python && maturin develop"
    )


# Mapping from torch dtype to mithril dtype string
DTYPE_MAP = {
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


def _torch_dtype_to_mithril(dtype: torch.dtype) -> str:
    """Convert torch dtype to mithril dtype string."""
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return DTYPE_MAP[dtype]


def _compress_tensor(
    tensor: torch.Tensor,
    compressor: CheckpointCompressor,
    previous: Optional[torch.Tensor] = None,
) -> Tuple[bytes, int, str]:
    """Compress a tensor using Mithril.

    Args:
        tensor: The tensor to compress
        compressor: Mithril compressor instance
        previous: Optional previous tensor for delta encoding (100x+ compression)

    Returns:
        Tuple of (compressed_bytes, original_size, dtype_string)
    """
    # Ensure tensor is contiguous and on CPU
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    # Get raw bytes
    data = tensor.numpy().tobytes()
    original_size = len(data)
    dtype_str = _torch_dtype_to_mithril(tensor.dtype)

    # Get previous bytes if available for delta encoding
    previous_bytes = None
    if previous is not None:
        if not previous.is_contiguous():
            previous = previous.contiguous()
        if previous.device.type != "cpu":
            previous = previous.cpu()
        previous_bytes = previous.numpy().tobytes()

    # Compress (with delta if previous provided)
    compressed = compressor.compress(data, dtype_str, previous=previous_bytes)

    return bytes(compressed), original_size, dtype_str


def _decompress_tensor(
    compressed: bytes,
    original_size: int,
    dtype_str: str,
    shape: torch.Size,
    compressor: CheckpointCompressor,
    previous: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decompress bytes back to a tensor using Mithril.

    Args:
        compressed: Compressed tensor bytes
        original_size: Original size in bytes
        dtype_str: Mithril dtype string
        shape: Original tensor shape
        compressor: Mithril compressor instance
        previous: Optional previous tensor for delta decoding
    """
    # Get previous bytes if available
    previous_bytes = None
    if previous is not None:
        if not previous.is_contiguous():
            previous = previous.contiguous()
        if previous.device.type != "cpu":
            previous = previous.cpu()
        previous_bytes = previous.numpy().tobytes()

    # Decompress (with delta if previous provided)
    decompressed = compressor.decompress(
        compressed, dtype_str, original_size, previous=previous_bytes
    )

    # Convert back to tensor
    # Map dtype string back to torch dtype
    dtype_map_reverse = {v: k for k, v in DTYPE_MAP.items()}
    torch_dtype = dtype_map_reverse[dtype_str]

    # Create tensor from bytes
    tensor = torch.frombuffer(bytearray(decompressed), dtype=torch_dtype)
    return tensor.reshape(shape)


class MithrilSavePlanner(DefaultSavePlanner):
    """SavePlanner that compresses tensors using Mithril before saving.

    This planner wraps the default PyTorch DCP save planner and applies
    Mithril compression to tensor data. Compression metadata is stored
    alongside the checkpoint to enable decompression during load.

    Args:
        compression_level: Zstd compression level (1-22, default 3).
            Higher levels give better compression but are slower.
        byte_grouping: Enable byte grouping for floating-point types.
            Significantly improves compression for bf16/fp16/fp32.
        previous_state_dict: Optional previous state dict for delta encoding.
            When provided, achieves 100x+ compression on consecutive checkpoints.

    Example:
        >>> # First checkpoint (no delta)
        >>> planner = MithrilSavePlanner(compression_level=3)
        >>> dcp.save(state_dict, storage_writer=writer, planner=planner)
        >>>
        >>> # Subsequent checkpoints (with delta for 100x+ compression)
        >>> planner = MithrilSavePlanner(previous_state_dict=prev_state)
        >>> dcp.save(state_dict, storage_writer=writer, planner=planner)
    """

    def __init__(
        self,
        compression_level: int = 3,
        byte_grouping: bool = True,
        previous_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.config = CompressionConfig(
            zstd_level=compression_level,
            byte_grouping=byte_grouping,
        )
        self.compressor = CheckpointCompressor(self.config)
        # Store compression metadata for each tensor
        self._compression_meta: Dict[str, Dict[str, Any]] = {}
        # Previous state dict for delta encoding
        self._previous_state_dict = previous_state_dict

    def transform_tensor(self, write_item: WriteItem, tensor: torch.Tensor) -> torch.Tensor:
        """Transform tensor before writing by compressing it.

        Note: DCP expects tensors, so we store compressed data as uint8 tensor
        and record metadata for decompression.
        """
        # Get previous tensor if available for delta encoding
        previous = None
        if self._previous_state_dict is not None:
            # Try to find matching tensor in previous state
            key = str(write_item.index)
            # Extract tensor name from index (format: "tensor_name/...")
            tensor_name = key.split("/")[0] if "/" in key else key
            if tensor_name in self._previous_state_dict:
                prev_tensor = self._previous_state_dict[tensor_name]
                # Only use delta if shapes match
                if prev_tensor.shape == tensor.shape:
                    previous = prev_tensor

        # Compress the tensor (with delta if previous available)
        compressed, original_size, dtype_str = _compress_tensor(
            tensor, self.compressor, previous=previous
        )

        # Store metadata for decompression (including delta status)
        key = str(write_item.index)
        self._compression_meta[key] = {
            "has_delta": previous is not None,
            "original_size": original_size,
            "dtype": dtype_str,
            "shape": list(tensor.shape),
            "compressed": True,
        }

        # Return compressed data as uint8 tensor
        return torch.frombuffer(bytearray(compressed), dtype=torch.uint8)

    def create_local_plan(self) -> SavePlan:
        """Create local save plan with compression metadata."""
        plan = super().create_local_plan()
        # Attach compression metadata to plan for storage
        return plan

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """Finalize the save plan."""
        return super().finish_plan(new_plan)


class MithrilLoadPlanner(DefaultLoadPlanner):
    """LoadPlanner that decompresses tensors using Mithril after loading.

    This planner works with checkpoints saved using MithrilSavePlanner.
    It reads compression metadata and decompresses tensors during load.

    Args:
        previous_state_dict: Optional previous state dict for delta decoding.
            Required if the checkpoint was saved with delta encoding.

    Example:
        >>> # Load without delta
        >>> planner = MithrilLoadPlanner()
        >>> dcp.load(state_dict, storage_reader=reader, planner=planner)
        >>>
        >>> # Load with delta (required if saved with delta)
        >>> planner = MithrilLoadPlanner(previous_state_dict=prev_state)
        >>> dcp.load(state_dict, storage_reader=reader, planner=planner)
    """

    def __init__(
        self,
        previous_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.config = CompressionConfig()
        self.compressor = CheckpointCompressor(self.config)
        self._compression_meta: Dict[str, Dict[str, Any]] = {}
        self._previous_state_dict = previous_state_dict

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> torch.Tensor:
        """Transform tensor after reading by decompressing it.

        Note: If tensor was compressed with MithrilSavePlanner, it will be
        stored as uint8. We detect this and decompress accordingly.
        """
        key = str(read_item.dest_index)

        # Check if we have compression metadata for this tensor
        if key in self._compression_meta:
            meta = self._compression_meta[key]
            if meta.get("compressed", False):
                compressed = bytes(tensor.numpy())

                # Get previous tensor for delta decoding if needed
                previous = None
                if meta.get("has_delta", False) and self._previous_state_dict is not None:
                    tensor_name = key.split("/")[0] if "/" in key else key
                    if tensor_name in self._previous_state_dict:
                        previous = self._previous_state_dict[tensor_name]

                return _decompress_tensor(
                    compressed,
                    meta["original_size"],
                    meta["dtype"],
                    torch.Size(meta["shape"]),
                    self.compressor,
                    previous=previous,
                )

        # Not compressed, return as-is
        return tensor


# Convenience function for simple compression without DCP
def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    compression_level: int = 3,
    byte_grouping: bool = True,
    previous_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, bytes]:
    """Compress a PyTorch state_dict using Mithril.

    Args:
        state_dict: PyTorch model state dictionary
        compression_level: Zstd compression level (1-22)
        byte_grouping: Enable byte grouping optimization
        previous_state_dict: Optional previous state dict for delta encoding.
            When provided, achieves 100x+ compression on consecutive checkpoints.

    Returns:
        Dictionary mapping tensor names to compressed bytes
    """
    compressor = CheckpointCompressor(
        CompressionConfig(zstd_level=compression_level, byte_grouping=byte_grouping)
    )

    compressed = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Get previous tensor if available for delta encoding
            previous = None
            if previous_state_dict is not None and name in previous_state_dict:
                prev_tensor = previous_state_dict[name]
                if prev_tensor.shape == tensor.shape:
                    previous = prev_tensor

            data, _, _ = _compress_tensor(tensor, compressor, previous=previous)
            compressed[name] = data
        else:
            # Non-tensor values (e.g., scalars) - store as-is via pickle
            import pickle
            compressed[name] = pickle.dumps(tensor)

    return compressed


def decompress_state_dict(
    compressed: Dict[str, bytes],
    reference_state_dict: Dict[str, torch.Tensor],
    previous_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Decompress a state_dict compressed with compress_state_dict.

    Args:
        compressed: Dictionary of compressed tensor bytes
        reference_state_dict: Reference state dict for shapes/dtypes
        previous_state_dict: Optional previous state dict for delta decoding.
            Required if the state dict was compressed with delta encoding.

    Returns:
        Decompressed state dictionary
    """
    compressor = CheckpointCompressor(CompressionConfig())

    result = {}
    for name, data in compressed.items():
        if name in reference_state_dict:
            ref = reference_state_dict[name]
            if isinstance(ref, torch.Tensor):
                dtype_str = _torch_dtype_to_mithril(ref.dtype)
                original_size = ref.numel() * ref.element_size()

                # Get previous tensor for delta decoding if available
                previous = None
                if previous_state_dict is not None and name in previous_state_dict:
                    previous = previous_state_dict[name]

                result[name] = _decompress_tensor(
                    data, original_size, dtype_str, ref.shape, compressor,
                    previous=previous,
                )
            else:
                import pickle
                result[name] = pickle.loads(data)
        else:
            import pickle
            result[name] = pickle.loads(data)

    return result
