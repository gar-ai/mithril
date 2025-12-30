"""Streaming compression API for large checkpoints.

This module provides a streaming interface for compressing and decompressing
checkpoint files without loading the entire checkpoint into memory.

Example:
    # Compress tensors to MST format
    with StreamingCompressor("model.mst") as compressor:
        for name, tensor in model.state_dict().items():
            data = tensor.numpy().tobytes()
            shape = list(tensor.shape)
            dtype = str(tensor.dtype).replace("torch.", "")
            compressor.write_tensor(name, data, dtype, shape)

    # Read tensors back
    with StreamingDecompressor("model.mst") as decompressor:
        for name, data, info in decompressor:
            print(f"{name}: {info.shape} {info.dtype}")
"""

from typing import Iterator, Tuple, Dict, Any, Optional
from mithril._mithril import (
    MstWriter,
    MstReader,
    MstTensorInfo,
    CompressionConfig,
)


class StreamingCompressor:
    """Context manager for streaming tensor compression to MST format.

    Compresses tensors one at a time without loading the full checkpoint
    into memory. Each tensor is compressed as it's added.

    Example:
        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        with StreamingCompressor("model.mst", config) as compressor:
            compressor.write_tensor("layer1.weight", weight_bytes, "bf16", [768, 768])
            compressor.write_tensor("layer1.bias", bias_bytes, "bf16", [768])

        print(f"Compression ratio: {compressor.stats['ratio']:.2f}x")

    Attributes:
        path: Output file path
        config: Compression configuration
        stats: Dictionary with compression statistics after finalize()
    """

    def __init__(
        self,
        path: str,
        config: Optional[CompressionConfig] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Initialize streaming compressor.

        Args:
            path: Output file path (typically .mst extension)
            config: Optional compression configuration
            metadata: Optional metadata to include in the file
        """
        self.path = path
        self.config = config
        self._metadata = metadata or {}
        self._writer: Optional[MstWriter] = None
        self._finalized = False
        self.stats: Dict[str, Any] = {}

    def __enter__(self) -> "StreamingCompressor":
        """Start the streaming compression context."""
        self._writer = MstWriter(self.config)
        if self._metadata:
            self._writer.set_metadata(self._metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize and close the compression context."""
        if not self._finalized and self._writer is not None:
            self.finalize()
        return None

    def write_tensor(
        self,
        name: str,
        data: bytes,
        dtype: str,
        shape: list,
    ) -> None:
        """Write a tensor to the compressed file.

        Args:
            name: Tensor name (e.g., "model.layer1.weight")
            data: Raw tensor bytes
            dtype: Data type string ("bf16", "fp16", "fp32", etc.)
            shape: Tensor shape as list of integers

        Raises:
            RuntimeError: If called after finalize() or outside context
        """
        if self._writer is None:
            raise RuntimeError("StreamingCompressor must be used as context manager")
        if self._finalized:
            raise RuntimeError("Cannot write after finalize()")

        self._writer.add_tensor(name, data, dtype, shape)

    def set_metadata(self, key: str, value: str) -> None:
        """Add metadata to the file.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if self._writer is None:
            self._metadata[key] = value
        else:
            self._writer.set_metadata({key: value})

    def finalize(self) -> Dict[str, Any]:
        """Finalize and write the compressed file.

        This is called automatically when exiting the context manager,
        but can be called explicitly if needed.

        Returns:
            Dictionary with compression statistics
        """
        if self._finalized:
            return self.stats

        if self._writer is None:
            raise RuntimeError("StreamingCompressor must be used as context manager")

        # Write the file
        self._writer.write(self.path)

        # Collect stats
        self.stats = {
            "tensor_count": self._writer.tensor_count,
            "uncompressed_size": self._writer.uncompressed_size,
            "compressed_size": self._writer.compressed_size,
            "ratio": self._writer.ratio,
        }

        self._finalized = True
        return self.stats

    @property
    def tensor_count(self) -> int:
        """Number of tensors written so far."""
        if self._writer is None:
            return 0
        return self._writer.tensor_count

    @property
    def uncompressed_size(self) -> int:
        """Total uncompressed size of tensors written."""
        if self._writer is None:
            return 0
        return self._writer.uncompressed_size

    @property
    def compressed_size(self) -> int:
        """Total compressed size of tensors written."""
        if self._writer is None:
            return 0
        return self._writer.compressed_size


class StreamingDecompressor:
    """Context manager for streaming tensor decompression from MST format.

    Reads and decompresses tensors one at a time, making it memory-efficient
    for large checkpoints.

    Example:
        with StreamingDecompressor("model.mst") as decompressor:
            # Iterate over all tensors
            for name, data, info in decompressor:
                tensor = torch.frombuffer(data, dtype=torch.bfloat16)
                print(f"{name}: {tensor.shape}")

            # Or read specific tensors
            weight = decompressor.read_tensor("layer1.weight")

    Attributes:
        path: Input file path
        tensor_names: List of tensor names in the file
        metadata: File metadata dictionary
    """

    def __init__(self, path: str):
        """Initialize streaming decompressor.

        Args:
            path: Input MST file path
        """
        self.path = path
        self._reader: Optional[MstReader] = None
        self._iterated_names: set = set()

    def __enter__(self) -> "StreamingDecompressor":
        """Open the file for streaming decompression."""
        self._reader = MstReader(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the decompression context."""
        self._reader = None
        return None

    def __iter__(self) -> Iterator[Tuple[str, bytes, MstTensorInfo]]:
        """Iterate over all tensors in the file.

        Yields:
            Tuples of (name, data, info) for each tensor
        """
        if self._reader is None:
            raise RuntimeError("StreamingDecompressor must be used as context manager")

        for name in self._reader.tensor_names():
            if name not in self._iterated_names:
                self._iterated_names.add(name)
                info = self._reader.tensor_info(name)
                data = self._reader.read_tensor(name)
                yield name, data, info

    def read_tensor(self, name: str) -> bytes:
        """Read and decompress a specific tensor.

        Args:
            name: Tensor name

        Returns:
            Decompressed tensor bytes

        Raises:
            KeyError: If tensor name not found
        """
        if self._reader is None:
            raise RuntimeError("StreamingDecompressor must be used as context manager")
        return self._reader.read_tensor(name)

    def tensor_info(self, name: str) -> MstTensorInfo:
        """Get metadata for a specific tensor.

        Args:
            name: Tensor name

        Returns:
            MstTensorInfo with dtype, shape, sizes
        """
        if self._reader is None:
            raise RuntimeError("StreamingDecompressor must be used as context manager")
        return self._reader.tensor_info(name)

    @property
    def tensor_names(self) -> list:
        """List of tensor names in the file."""
        if self._reader is None:
            return []
        return self._reader.tensor_names()

    @property
    def metadata(self) -> Dict[str, str]:
        """File metadata dictionary."""
        if self._reader is None:
            return {}
        return self._reader.metadata()

    @property
    def tensor_count(self) -> int:
        """Number of tensors in the file."""
        if self._reader is None:
            return 0
        return self._reader.tensor_count

    @property
    def ratio(self) -> float:
        """Overall compression ratio."""
        if self._reader is None:
            return 0.0
        return self._reader.ratio
