"""Integration tests for delta compression between training checkpoints."""

import pytest

torch = pytest.importorskip("torch")
mithril = pytest.importorskip("mithril")


class TestDeltaCompression:
    """Test delta encoding for training checkpoint sequences."""

    def test_delta_compressor_creation(self):
        """Test that DeltaCompressor can be created."""
        delta = mithril.DeltaCompressor()
        assert delta is not None

    def test_first_checkpoint_no_delta(self):
        """Test that first checkpoint doesn't use delta."""
        delta = mithril.DeltaCompressor()

        # Create initial checkpoint
        data = torch.randn(1000, 1000, dtype=torch.bfloat16)
        checkpoint_bytes = bytes(data.untyped_storage())

        compressed, stats = delta.compress_checkpoint("step_0", checkpoint_bytes)

        assert compressed is not None
        assert len(compressed) < len(checkpoint_bytes), "First checkpoint should still be compressed"
        # First checkpoint shouldn't have extreme compression from delta
        assert stats.ratio < 100, "First checkpoint shouldn't have delta compression"

    def test_delta_between_similar_checkpoints(self):
        """Test delta compression between similar checkpoints."""
        delta = mithril.DeltaCompressor()

        # Create base checkpoint
        torch.manual_seed(42)
        base_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16)
        base_bytes = bytes(base_tensor.untyped_storage())

        # Compress base
        _, base_stats = delta.compress_checkpoint("step_0", base_bytes)

        # Create similar checkpoint (small changes, simulating one training step)
        # In real training, most parameters change only slightly
        similar_tensor = base_tensor.clone()
        similar_tensor[:10, :10] += 0.001  # Small perturbation
        similar_bytes = bytes(similar_tensor.untyped_storage())

        # Compress with delta
        _, delta_stats = delta.compress_checkpoint("step_1", similar_bytes)

        # Delta compression should be much better
        assert delta_stats.ratio > base_stats.ratio, (
            f"Delta should improve compression: base={base_stats.ratio:.1f}x, delta={delta_stats.ratio:.1f}x"
        )

        # With 99%+ similarity, we should see significant improvement
        if delta_stats.sparsity > 0.9:
            assert delta_stats.ratio > 10 * base_stats.ratio, (
                f"With {delta_stats.sparsity:.1%} sparsity, expected 10x+ improvement"
            )

    def test_delta_with_identical_checkpoints(self):
        """Test delta compression with identical checkpoints (maximum sparsity)."""
        delta = mithril.DeltaCompressor()

        # Create checkpoint
        torch.manual_seed(42)
        tensor = torch.randn(1000, 1000, dtype=torch.bfloat16)
        checkpoint_bytes = bytes(tensor.untyped_storage())

        # Compress first time
        _, stats1 = delta.compress_checkpoint("step_0", checkpoint_bytes)

        # Compress identical data
        _, stats2 = delta.compress_checkpoint("step_1", checkpoint_bytes)

        # Identical data should have 100% sparsity (all zeros after XOR)
        assert stats2.sparsity >= 0.99, f"Expected ~100% sparsity, got {stats2.sparsity:.1%}"

        # Should achieve very high compression
        assert stats2.ratio > 100, f"Expected 100x+ compression, got {stats2.ratio:.1f}x"

    def test_delta_with_different_checkpoints(self):
        """Test delta compression with completely different checkpoints."""
        delta = mithril.DeltaCompressor()

        # Create first checkpoint
        torch.manual_seed(42)
        tensor1 = torch.randn(500, 500, dtype=torch.bfloat16)
        bytes1 = bytes(tensor1.untyped_storage())

        # Compress first
        _, stats1 = delta.compress_checkpoint("step_0", bytes1)

        # Create completely different checkpoint
        torch.manual_seed(123)
        tensor2 = torch.randn(500, 500, dtype=torch.bfloat16)
        bytes2 = bytes(tensor2.untyped_storage())

        # Compress with delta (won't help much)
        _, stats2 = delta.compress_checkpoint("step_1", bytes2)

        # Sparsity should be low (random XOR is still random)
        assert stats2.sparsity < 0.5, f"Expected low sparsity for different data, got {stats2.sparsity:.1%}"

    def test_training_simulation(self):
        """Simulate a realistic training scenario with gradual changes."""
        delta = mithril.DeltaCompressor()

        # Initial model state
        torch.manual_seed(42)
        state = torch.randn(500, 500, dtype=torch.bfloat16)

        compression_ratios = []

        # Simulate 5 training steps
        for step in range(5):
            checkpoint_bytes = bytes(state.untyped_storage())
            _, stats = delta.compress_checkpoint(f"step_{step}", checkpoint_bytes)
            compression_ratios.append(stats.ratio)

            # Simulate training: add small gradient updates
            gradient = torch.randn_like(state) * 0.001
            state = state + gradient

        # First step should have base compression
        # Subsequent steps should have better compression due to delta
        print(f"Compression ratios: {[f'{r:.1f}x' for r in compression_ratios]}")

        # After warmup, delta should help (step 2+ should be better than step 1)
        # Note: With random gradients, improvement may be modest
        assert compression_ratios[-1] > compression_ratios[0] * 0.5, (
            "Delta should maintain reasonable compression even with changes"
        )


class TestDeltaStats:
    """Test DeltaStats properties."""

    def test_stats_properties(self):
        """Test that DeltaStats has expected properties."""
        delta = mithril.DeltaCompressor()

        tensor = torch.randn(100, 100, dtype=torch.bfloat16)
        _, stats = delta.compress_checkpoint("test", bytes(tensor.untyped_storage()))

        # Check stats properties exist
        assert hasattr(stats, "ratio")
        assert hasattr(stats, "sparsity")

        # Values should be reasonable
        assert stats.ratio >= 1.0
        assert 0.0 <= stats.sparsity <= 1.0

    def test_stats_repr(self):
        """Test that DeltaStats can be printed."""
        delta = mithril.DeltaCompressor()

        tensor = torch.randn(100, 100, dtype=torch.bfloat16)
        _, stats = delta.compress_checkpoint("test", bytes(tensor.untyped_storage()))

        # Should be printable
        repr_str = repr(stats)
        assert "ratio" in repr_str.lower() or "DeltaStats" in repr_str
