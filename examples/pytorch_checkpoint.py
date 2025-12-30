#!/usr/bin/env python3
"""Example: PyTorch checkpoint compression with Mithril.

This example demonstrates how to compress and decompress PyTorch model weights
using Mithril's checkpoint compression.

Requirements:
    pip install torch safetensors mithril
"""

import os
import tempfile
import time

# Check if torch is available
try:
    import torch
    from safetensors.torch import save_file, load_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Note: torch/safetensors not installed. Running with synthetic data.")

import mithril


def create_sample_model_weights():
    """Create sample model weights (or synthetic data if torch unavailable)."""
    if HAS_TORCH:
        # Create a small transformer-like model's weights
        weights = {
            "embedding.weight": torch.randn(32000, 768, dtype=torch.bfloat16),
            "encoder.layer.0.attention.query.weight": torch.randn(768, 768, dtype=torch.bfloat16),
            "encoder.layer.0.attention.key.weight": torch.randn(768, 768, dtype=torch.bfloat16),
            "encoder.layer.0.attention.value.weight": torch.randn(768, 768, dtype=torch.bfloat16),
            "encoder.layer.0.mlp.fc1.weight": torch.randn(3072, 768, dtype=torch.bfloat16),
            "encoder.layer.0.mlp.fc2.weight": torch.randn(768, 3072, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32000, 768, dtype=torch.bfloat16),
        }
        return weights
    else:
        # Create synthetic bf16 data (random bytes)
        import struct
        import random
        size = 10 * 1024 * 1024  # 10 MB
        data = bytes(random.getrandbits(8) for _ in range(size))
        return data


def compress_safetensors_file(input_path: str, output_path: str) -> dict:
    """Compress a safetensors file using Mithril.

    Args:
        input_path: Path to input .safetensors file
        output_path: Path for compressed output

    Returns:
        Dict with compression statistics
    """
    # Read the safetensors file
    with open(input_path, "rb") as f:
        data = f.read()

    original_size = len(data)

    # Create compressor with default settings
    config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
    compressor = mithril.CheckpointCompressor(config)

    # Compress (bf16 is most common for modern models)
    start = time.perf_counter()
    compressed = compressor.compress(data, "bf16")
    compress_time = time.perf_counter() - start

    compressed_size = len(compressed)

    # Write compressed data
    with open(output_path, "wb") as f:
        f.write(compressed)

    return {
        "original_size_mb": original_size / (1024 * 1024),
        "compressed_size_mb": compressed_size / (1024 * 1024),
        "ratio": original_size / compressed_size,
        "compress_time_ms": compress_time * 1000,
        "throughput_gbps": (original_size / (1024**3)) / compress_time,
    }


def decompress_checkpoint(input_path: str, original_size: int) -> tuple:
    """Decompress a Mithril-compressed checkpoint.

    Args:
        input_path: Path to compressed file
        original_size: Original size in bytes

    Returns:
        Tuple of (decompressed_data, stats_dict)
    """
    with open(input_path, "rb") as f:
        compressed = f.read()

    config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
    compressor = mithril.CheckpointCompressor(config)

    start = time.perf_counter()
    decompressed = compressor.decompress(compressed, "bf16", original_size)
    decompress_time = time.perf_counter() - start

    return decompressed, {
        "decompress_time_ms": decompress_time * 1000,
        "throughput_gbps": (original_size / (1024**3)) / decompress_time,
    }


def main():
    print("=" * 60)
    print("Mithril Checkpoint Compression Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        if HAS_TORCH:
            # Create sample model weights
            print("\n1. Creating sample model weights...")
            weights = create_sample_model_weights()

            # Save to safetensors
            safetensors_path = os.path.join(tmpdir, "model.safetensors")
            save_file(weights, safetensors_path)

            original_size = os.path.getsize(safetensors_path)
            print(f"   Saved {len(weights)} tensors ({original_size / 1024 / 1024:.1f} MB)")

            # Compress
            print("\n2. Compressing with Mithril...")
            compressed_path = os.path.join(tmpdir, "model.mcp")
            stats = compress_safetensors_file(safetensors_path, compressed_path)

            print(f"   Original:   {stats['original_size_mb']:.1f} MB")
            print(f"   Compressed: {stats['compressed_size_mb']:.1f} MB")
            print(f"   Ratio:      {stats['ratio']:.2f}x")
            print(f"   Time:       {stats['compress_time_ms']:.1f} ms")
            print(f"   Throughput: {stats['throughput_gbps']:.2f} GiB/s")

            # Decompress
            print("\n3. Decompressing...")
            decompressed, decomp_stats = decompress_checkpoint(compressed_path, original_size)

            print(f"   Time:       {decomp_stats['decompress_time_ms']:.1f} ms")
            print(f"   Throughput: {decomp_stats['throughput_gbps']:.2f} GiB/s")

            # Verify roundtrip
            print("\n4. Verifying roundtrip...")
            with open(safetensors_path, "rb") as f:
                original_data = f.read()

            if original_data == decompressed:
                print("   Roundtrip verified: data matches exactly")
            else:
                print("   ERROR: Data mismatch!")

        else:
            # Synthetic data example
            print("\n1. Creating synthetic bf16 data...")
            data = create_sample_model_weights()
            original_size = len(data)
            print(f"   Created {original_size / 1024 / 1024:.1f} MB of synthetic data")

            # Compress
            print("\n2. Compressing with Mithril...")
            config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
            compressor = mithril.CheckpointCompressor(config)

            start = time.perf_counter()
            compressed = compressor.compress(data, "bf16")
            compress_time = time.perf_counter() - start

            ratio = len(data) / len(compressed)
            print(f"   Original:   {len(data) / 1024 / 1024:.1f} MB")
            print(f"   Compressed: {len(compressed) / 1024 / 1024:.1f} MB")
            print(f"   Ratio:      {ratio:.2f}x")
            print(f"   Time:       {compress_time * 1000:.1f} ms")

            # Decompress
            print("\n3. Decompressing...")
            start = time.perf_counter()
            decompressed = compressor.decompress(compressed, "bf16", len(data))
            decompress_time = time.perf_counter() - start

            print(f"   Time:       {decompress_time * 1000:.1f} ms")

            # Verify
            print("\n4. Verifying roundtrip...")
            if data == decompressed:
                print("   Roundtrip verified: data matches exactly")
            else:
                print("   ERROR: Data mismatch!")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
