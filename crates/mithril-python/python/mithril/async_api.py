"""Async API for Mithril compression operations.

This module provides async wrappers around the synchronous Mithril API
using asyncio.to_thread() for non-blocking compression/decompression.

Example:
    import asyncio
    import mithril
    from mithril.async_api import compress_async, decompress_async
    
    async def main():
        data = b"..." * 1000000
        compressed = await compress_async(data, "bf16")
        decompressed = await decompress_async(compressed, "bf16", len(data))
    
    asyncio.run(main())
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

from mithril import (
    CompressionConfig,
    CheckpointCompressor,
    DeltaCompressor,
    ContentStore,
)


async def compress_async(
    data: bytes,
    dtype: str,
    config: Optional[CompressionConfig] = None,
    previous: Optional[bytes] = None,
) -> bytes:
    """Async compression with optional delta encoding.
    
    Args:
        data: Raw tensor bytes to compress
        dtype: Data type (bf16, fp16, fp32, etc.)
        config: Compression configuration
        previous: Previous checkpoint bytes for delta compression
    
    Returns:
        Compressed bytes
    """
    compressor = CheckpointCompressor(config or CompressionConfig())
    if previous is not None:
        return await asyncio.to_thread(compressor.compress, data, dtype, previous=previous)
    return await asyncio.to_thread(compressor.compress, data, dtype)


async def decompress_async(
    data: bytes,
    dtype: str,
    original_size: int,
    previous: Optional[bytes] = None,
) -> bytes:
    """Async decompression with optional delta decoding.
    
    Args:
        data: Compressed bytes
        dtype: Data type
        original_size: Original uncompressed size
        previous: Previous checkpoint bytes for delta decompression
    
    Returns:
        Decompressed bytes
    """
    compressor = CheckpointCompressor(CompressionConfig())
    if previous is not None:
        return await asyncio.to_thread(
            compressor.decompress, data, dtype, original_size, previous=previous
        )
    return await asyncio.to_thread(compressor.decompress, data, dtype, original_size)


async def compress_raw_async(
    data: bytes,
    config: Optional[CompressionConfig] = None,
) -> bytes:
    """Async raw compression (for pickle/safetensors files).
    
    Args:
        data: Raw bytes to compress
        config: Compression configuration
    
    Returns:
        Compressed bytes
    """
    compressor = CheckpointCompressor(config or CompressionConfig())
    return await asyncio.to_thread(compressor.compress_raw, data)


async def decompress_raw_async(
    data: bytes,
    original_size: int,
) -> bytes:
    """Async raw decompression.
    
    Args:
        data: Compressed bytes
        original_size: Original uncompressed size
    
    Returns:
        Decompressed bytes
    """
    compressor = CheckpointCompressor(CompressionConfig())
    return await asyncio.to_thread(compressor.decompress_raw, data, original_size)


async def compress_file_async(
    input_path: str,
    output_path: str,
    config: Optional[CompressionConfig] = None,
) -> Dict[str, Any]:
    """Async file compression.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        config: Compression configuration
    
    Returns:
        Dict with original_size, compressed_size, ratio
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Read file (in thread to avoid blocking)
    data = await asyncio.to_thread(input_file.read_bytes)
    original_size = len(data)
    
    # Compress
    compressor = CheckpointCompressor(config or CompressionConfig())
    compressed = await asyncio.to_thread(compressor.compress_raw, data)
    compressed_size = len(compressed)
    
    # Write file
    await asyncio.to_thread(output_file.write_bytes, compressed)
    
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": original_size / compressed_size if compressed_size > 0 else 1.0,
    }


async def decompress_file_async(
    input_path: str,
    output_path: str,
    original_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Async file decompression.
    
    Args:
        input_path: Path to compressed file
        output_path: Path to output file
        original_size: Original size (auto-detected for zstd if not provided)
    
    Returns:
        Dict with decompressed_size
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Read compressed file
    compressed = await asyncio.to_thread(input_file.read_bytes)
    
    # Decompress
    compressor = CheckpointCompressor(CompressionConfig())
    # If original_size not provided, try to auto-detect (works for zstd)
    size = original_size if original_size else len(compressed) * 20
    decompressed = await asyncio.to_thread(compressor.decompress_raw, compressed, size)
    
    # Write file
    await asyncio.to_thread(output_file.write_bytes, decompressed)
    
    return {
        "decompressed_size": len(decompressed),
    }


async def store_async(store: ContentStore, content: bytes) -> str:
    """Async content store put.
    
    Args:
        store: ContentStore instance
        content: Content to store
    
    Returns:
        Content address (hex string)
    """
    return await asyncio.to_thread(store.put, content)


async def retrieve_async(store: ContentStore, address: str) -> bytes:
    """Async content store get.
    
    Args:
        store: ContentStore instance
        address: Content address
    
    Returns:
        Retrieved content
    """
    return await asyncio.to_thread(store.get, address)


__all__ = [
    "compress_async",
    "decompress_async",
    "compress_raw_async",
    "decompress_raw_async",
    "compress_file_async",
    "decompress_file_async",
    "store_async",
    "retrieve_async",
]
