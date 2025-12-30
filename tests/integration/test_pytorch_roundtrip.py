"""PyTorch roundtrip integration tests for mithril-checkpoint.

These tests verify that PyTorch model weights can be:
1. Saved with torch.save()
2. Compressed with mithril
3. Decompressed with mithril
4. Loaded with torch.load()
5. Verified to be identical to the original

NOTE: Mithril has two compression modes:
- compress_raw/decompress_raw: For arbitrary data like torch.save pickle format
- compress/decompress with dtype: For raw aligned tensor bytes (byte grouping)
"""

import io
import tempfile
from pathlib import Path

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")


class TestTorchSaveRoundtrip:
    """Test roundtrip with torch.save() format using compress_raw."""

    def test_simple_tensor_roundtrip(self):
        """Test compressing and decompressing a simple tensor."""
        import mithril

        original = torch.randn(100, 100, dtype=torch.float32)

        buffer = io.BytesIO()
        torch.save(original, buffer)
        data = buffer.getvalue()

        # Use compress_raw for pickle data
        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)

        assert len(compressed) <= len(data), "Compression should not expand data"

        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(original, recovered), "Roundtrip should be lossless"

    def test_bf16_tensor_roundtrip(self):
        """Test roundtrip with bfloat16 tensors (common for LLMs)."""
        import mithril

        original = torch.randn(256, 256, dtype=torch.bfloat16)

        buffer = io.BytesIO()
        torch.save(original, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)

        ratio = len(data) / len(compressed)
        assert ratio > 1.0, f"Expected some compression, got {ratio:.2f}x"

        decompressed = compressor.decompress_raw(compressed, len(data))
        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(original, recovered)

    def test_fp16_tensor_roundtrip(self):
        """Test roundtrip with float16 tensors."""
        import mithril

        original = torch.randn(128, 512, dtype=torch.float16)

        buffer = io.BytesIO()
        torch.save(original, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(original, recovered)


class TestStateDict:
    """Test roundtrip with model state_dict format."""

    def test_linear_model_state_dict(self):
        """Test compressing a simple linear model's state dict."""
        import mithril

        model = torch.nn.Linear(256, 128, dtype=torch.float32)
        state_dict = model.state_dict()

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered_state = torch.load(io.BytesIO(decompressed), weights_only=True)

        for key in state_dict:
            assert torch.equal(
                state_dict[key], recovered_state[key]
            ), f"Mismatch in {key}"

    def test_mlp_model_state_dict(self):
        """Test with a multi-layer MLP."""
        import mithril

        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
        state_dict = model.state_dict()

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered_state = torch.load(io.BytesIO(decompressed), weights_only=True)

        for key in state_dict:
            assert torch.equal(state_dict[key], recovered_state[key])


class TestRawTensorBytes:
    """Test byte-grouped compression on raw tensor bytes (not pickle)."""

    def test_fp32_raw_bytes(self):
        """Test byte-grouped compression on raw fp32 bytes."""
        import mithril

        tensor = torch.randn(500, 500, dtype=torch.float32)
        raw_bytes = tensor.numpy().tobytes()

        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress(raw_bytes, "fp32")
        decompressed = compressor.decompress(compressed, "fp32", len(raw_bytes))

        import numpy as np
        recovered = torch.from_numpy(
            np.frombuffer(decompressed, dtype=np.float32).reshape(500, 500).copy()
        )
        assert torch.equal(tensor, recovered)

    def test_bf16_raw_bytes(self):
        """Test byte-grouped compression on raw bf16 bytes."""
        import mithril
        import numpy as np

        tensor = torch.randn(500, 500, dtype=torch.bfloat16)
        raw_bytes = tensor.view(torch.uint16).numpy().tobytes()

        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress(raw_bytes, "bf16")

        # bf16 achieves good compression with byte grouping
        ratio = len(raw_bytes) / len(compressed)
        assert ratio > 1.3, f"Expected >1.3x compression for bf16, got {ratio:.2f}x"

        decompressed = compressor.decompress(compressed, "bf16", len(raw_bytes))

        recovered_uint16 = np.frombuffer(decompressed, dtype=np.uint16).reshape(500, 500)
        recovered = torch.from_numpy(recovered_uint16.copy()).view(torch.bfloat16)
        assert torch.equal(tensor, recovered)

    def test_delta_compression_raw_bytes(self):
        """Test delta compression on raw tensor bytes."""
        import mithril
        import numpy as np

        # Base weights
        base_tensor = torch.randn(512, 512, dtype=torch.bfloat16)
        base_bytes = base_tensor.view(torch.uint16).numpy().tobytes()

        # Updated weights (small change from training)
        updated_tensor = base_tensor + torch.randn_like(base_tensor) * 0.001
        updated_bytes = updated_tensor.view(torch.uint16).numpy().tobytes()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)

        # Regular compression
        regular_compressed = compressor.compress(updated_bytes, "bf16")

        # Delta compression (pass previous checkpoint)
        delta_compressed = compressor.compress(updated_bytes, "bf16", previous=base_bytes)

        # Delta should be much smaller since weights changed only slightly
        delta_ratio = len(regular_compressed) / len(delta_compressed)
        assert delta_ratio > 3.0, \
            f"Delta should be >3x smaller than regular, got {delta_ratio:.2f}x"

        # Verify roundtrip
        decompressed = compressor.decompress(
            delta_compressed, "bf16", len(updated_bytes), previous=base_bytes
        )

        recovered_uint16 = np.frombuffer(decompressed, dtype=np.uint16).reshape(512, 512)
        recovered_tensor = torch.from_numpy(recovered_uint16.copy()).view(torch.bfloat16)

        assert torch.equal(updated_tensor, recovered_tensor)


class TestCompressionLevels:
    """Test different compression levels."""

    def test_level_1_fastest(self):
        """Test fastest compression level."""
        import mithril

        data = torch.randn(500, 500, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(data, buffer)
        raw = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=1)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(raw)
        decompressed = compressor.decompress_raw(compressed, len(raw))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(data, recovered)

    def test_level_19_best(self):
        """Test best compression level."""
        import mithril

        data = torch.randn(500, 500, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(data, buffer)
        raw = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=19)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(raw)
        decompressed = compressor.decompress_raw(compressed, len(raw))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(data, recovered)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_tensor(self):
        """Test handling of empty tensors."""
        import mithril

        empty = torch.tensor([], dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(empty, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(empty, recovered)

    def test_scalar_tensor(self):
        """Test handling of scalar tensors."""
        import mithril

        scalar = torch.tensor(3.14159, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(scalar, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(scalar, recovered)

    def test_multidimensional_tensor(self):
        """Test handling of high-dimensional tensors."""
        import mithril

        tensor = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert torch.equal(tensor, recovered)

    def test_mixed_dtype_state_dict(self):
        """Test state dict with mixed dtypes."""
        import mithril

        state_dict = {
            "fp32_weights": torch.randn(100, 100, dtype=torch.float32),
            "bf16_weights": torch.randn(100, 100, dtype=torch.bfloat16),
            "int64_indices": torch.randint(0, 100, (50,), dtype=torch.int64),
        }

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        decompressed = compressor.decompress_raw(compressed, len(data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        for key in state_dict:
            assert torch.equal(state_dict[key], recovered[key])


class TestFileOperations:
    """Test file-based operations."""

    def test_file_roundtrip(self):
        """Test compressing and decompressing files."""
        import mithril

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "original.pt"
            compressed_path = Path(tmpdir) / "compressed.mcp"
            recovered_path = Path(tmpdir) / "recovered.pt"

            # Create and save original
            model = torch.nn.Linear(512, 256, dtype=torch.bfloat16)
            torch.save(model.state_dict(), original_path)

            # Read, compress, write
            with open(original_path, "rb") as f:
                data = f.read()

            config = mithril.CompressionConfig(zstd_level=3)
            compressor = mithril.CheckpointCompressor(config)
            compressed = compressor.compress_raw(data)

            with open(compressed_path, "wb") as f:
                f.write(compressed)

            # Verify compressed file is smaller or equal
            assert compressed_path.stat().st_size <= original_path.stat().st_size

            # Read, decompress, write
            with open(compressed_path, "rb") as f:
                compressed_data = f.read()

            decompressed = compressor.decompress_raw(compressed_data, len(data))

            with open(recovered_path, "wb") as f:
                f.write(decompressed)

            # Load and verify
            original_state = torch.load(original_path, weights_only=True)
            recovered_state = torch.load(recovered_path, weights_only=True)

            for key in original_state:
                assert torch.equal(original_state[key], recovered_state[key])


class TestContentStore:
    """Test content-addressable storage integration."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving compressed checkpoints."""
        import mithril

        with tempfile.TemporaryDirectory() as tmpdir:
            store = mithril.ContentStore(tmpdir)

            # Create checkpoint data
            model = torch.nn.Linear(128, 64, dtype=torch.bfloat16)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            data = buffer.getvalue()

            # Compress
            config = mithril.CompressionConfig(zstd_level=3)
            compressor = mithril.CheckpointCompressor(config)
            compressed = compressor.compress_raw(data)

            # Store
            address = store.put(compressed)
            assert address is not None
            assert len(address) > 0

            # Retrieve
            retrieved = store.get(address)
            assert retrieved == compressed

            # Decompress and verify
            decompressed = compressor.decompress_raw(retrieved, len(data))
            recovered = torch.load(io.BytesIO(decompressed), weights_only=True)

            original_state = model.state_dict()
            for key in original_state:
                assert torch.equal(original_state[key], recovered[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
