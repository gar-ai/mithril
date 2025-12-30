"""Integration tests for PyTorch Distributed Checkpoint (DCP) roundtrip."""

import shutil

import pytest

torch = pytest.importorskip("torch")
mithril = pytest.importorskip("mithril")


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Provide a temporary directory for checkpoints."""
    ckpt_dir = tmp_path / "checkpoint"
    yield ckpt_dir
    # Cleanup
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


class TestDCPRoundtrip:
    """Test save_compressed and load_compressed functions."""

    def test_simple_state_dict_roundtrip(self, checkpoint_dir):
        """Test basic save/load with a simple state dict."""
        state_dict = {
            "weights": torch.randn(100, 100, dtype=torch.bfloat16),
            "bias": torch.randn(100, dtype=torch.bfloat16),
        }

        # Save with compression
        stats = mithril.save_compressed(state_dict, str(checkpoint_dir))

        # Verify compression happened
        assert stats["ratio"] >= 1.0, f"Expected compression ratio >= 1.0, got {stats['ratio']}"

        # Load back
        loaded = {
            "weights": torch.empty_like(state_dict["weights"]),
            "bias": torch.empty_like(state_dict["bias"]),
        }
        mithril.load_compressed(loaded, str(checkpoint_dir))

        # Verify data integrity
        assert torch.equal(state_dict["weights"], loaded["weights"]), "Weights mismatch after roundtrip"
        assert torch.equal(state_dict["bias"], loaded["bias"]), "Bias mismatch after roundtrip"

    def test_large_state_dict_roundtrip(self, checkpoint_dir):
        """Test with a larger state dict (simulating a real model)."""
        # Simulate a small transformer layer
        hidden_size = 512
        state_dict = {
            "attention.q_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16),
            "attention.k_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16),
            "attention.v_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16),
            "attention.o_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16),
            "mlp.gate_proj.weight": torch.randn(hidden_size * 4, hidden_size, dtype=torch.bfloat16),
            "mlp.up_proj.weight": torch.randn(hidden_size * 4, hidden_size, dtype=torch.bfloat16),
            "mlp.down_proj.weight": torch.randn(hidden_size, hidden_size * 4, dtype=torch.bfloat16),
            "norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        }

        # Save
        stats = mithril.save_compressed(state_dict, str(checkpoint_dir))
        print(f"Compression ratio: {stats['ratio']:.2f}x")

        # Load
        loaded = {k: torch.empty_like(v) for k, v in state_dict.items()}
        mithril.load_compressed(loaded, str(checkpoint_dir))

        # Verify all tensors
        for key in state_dict:
            assert torch.equal(state_dict[key], loaded[key]), f"Mismatch in {key}"

    def test_mixed_dtype_roundtrip(self, checkpoint_dir):
        """Test with mixed data types."""
        state_dict = {
            "bf16_tensor": torch.randn(100, 100, dtype=torch.bfloat16),
            "fp32_tensor": torch.randn(50, 50, dtype=torch.float32),
            "int64_tensor": torch.randint(0, 100, (20, 20), dtype=torch.int64),
        }

        # Save
        stats = mithril.save_compressed(state_dict, str(checkpoint_dir))

        # Load
        loaded = {k: torch.empty_like(v) for k, v in state_dict.items()}
        mithril.load_compressed(loaded, str(checkpoint_dir))

        # Verify
        for key in state_dict:
            assert torch.equal(state_dict[key], loaded[key]), f"Mismatch in {key}"

    def test_compression_ratio(self, checkpoint_dir):
        """Test that we achieve meaningful compression."""
        # Use highly compressible data (zeros)
        state_dict = {
            "zeros": torch.zeros(1000, 1000, dtype=torch.bfloat16),
        }

        stats = mithril.save_compressed(state_dict, str(checkpoint_dir))

        # Zeros should compress very well
        assert stats["ratio"] > 10.0, f"Expected high compression for zeros, got {stats['ratio']:.2f}x"

    def test_random_data_compression(self, checkpoint_dir):
        """Test compression of random data (less compressible)."""
        state_dict = {
            "random": torch.randn(500, 500, dtype=torch.bfloat16),
        }

        stats = mithril.save_compressed(state_dict, str(checkpoint_dir))

        # Random data should still achieve some compression due to byte grouping
        assert stats["ratio"] >= 1.0, f"Expected ratio >= 1.0, got {stats['ratio']:.2f}x"

        # Verify roundtrip
        loaded = {"random": torch.empty_like(state_dict["random"])}
        mithril.load_compressed(loaded, str(checkpoint_dir))
        assert torch.equal(state_dict["random"], loaded["random"])


class TestDCPConfiguration:
    """Test DCP with different compression configurations."""

    def test_custom_compression_config(self, checkpoint_dir):
        """Test with custom CompressionConfig."""
        config = mithril.CompressionConfig(zstd_level=6, byte_grouping=True)

        state_dict = {"weights": torch.randn(100, 100, dtype=torch.bfloat16)}

        stats = mithril.save_compressed(state_dict, str(checkpoint_dir), config=config)

        loaded = {"weights": torch.empty_like(state_dict["weights"])}
        mithril.load_compressed(loaded, str(checkpoint_dir), config=config)

        assert torch.equal(state_dict["weights"], loaded["weights"])

    def test_high_compression_level(self, checkpoint_dir):
        """Test with maximum compression level."""
        config = mithril.CompressionConfig(zstd_level=19, byte_grouping=True)

        state_dict = {"weights": torch.randn(100, 100, dtype=torch.bfloat16)}

        stats = mithril.save_compressed(state_dict, str(checkpoint_dir), config=config)

        # High compression should yield better ratio (but slower)
        loaded = {"weights": torch.empty_like(state_dict["weights"])}
        mithril.load_compressed(loaded, str(checkpoint_dir), config=config)

        assert torch.equal(state_dict["weights"], loaded["weights"])
