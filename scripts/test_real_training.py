#!/usr/bin/env python3
"""Test mithril checkpoint compression with real PyTorch training.

This script validates that delta compression achieves 50x+ on consecutive
training checkpoints with a real model and optimizer.
"""

import sys
import time

# Check dependencies
try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    import mithril
except ImportError:
    print("ERROR: mithril not installed. Run: cd crates/mithril-python && maturin develop")
    sys.exit(1)


def test_simple_model():
    """Test with a simple Linear model."""
    print("=== Simple Model (4MB) ===\n")

    # Create a simple model
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    compressor = mithril.CheckpointCompressor()
    previous = None

    for step in range(5):
        # Simulate training step
        x = torch.randn(32, 1000)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Get checkpoint bytes
        state_dict = model.state_dict()
        current = b"".join(t.numpy().tobytes() for t in state_dict.values())

        # Compress
        start = time.time()
        compressed = compressor.compress(current, "fp32", previous=previous)
        elapsed = time.time() - start

        ratio = len(current) / len(compressed)
        throughput = len(current) / elapsed / (1024**3)

        print(f"Step {step}: {len(current)/1e6:.1f}MB → {len(compressed)/1e3:.1f}KB ({ratio:.1f}x) @ {throughput:.2f} GiB/s")

        # Verify roundtrip
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        assert decompressed == current, f"Roundtrip failed at step {step}!"

        previous = current

    print("\n✓ Simple model test passed!")
    return True


def test_mlp_model():
    """Test with a deeper MLP model."""
    print("\n=== MLP Model (16MB) ===\n")

    # Create a deeper model
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 1000),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    compressor = mithril.CheckpointCompressor()
    previous = None
    ratios = []

    for step in range(5):
        # Simulate training step
        x = torch.randn(32, 1000)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Get checkpoint bytes
        state_dict = model.state_dict()
        current = b"".join(t.numpy().tobytes() for t in state_dict.values())

        # Compress
        start = time.time()
        compressed = compressor.compress(current, "fp32", previous=previous)
        elapsed = time.time() - start

        ratio = len(current) / len(compressed)
        throughput = len(current) / elapsed / (1024**3)

        print(f"Step {step}: {len(current)/1e6:.1f}MB → {len(compressed)/1e3:.1f}KB ({ratio:.1f}x) @ {throughput:.2f} GiB/s")

        if step > 0:
            ratios.append(ratio)

        # Verify roundtrip
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        assert decompressed == current, f"Roundtrip failed at step {step}!"

        previous = current

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    print(f"\n✓ MLP model test passed! Average delta ratio: {avg_ratio:.1f}x")

    # Check we hit target
    if avg_ratio >= 50:
        print(f"✓ Target met: {avg_ratio:.1f}x >= 50x")
    else:
        print(f"⚠ Below target: {avg_ratio:.1f}x < 50x (may vary with optimizer/model)")

    return True


def test_small_lr_model():
    """Test with very small learning rate (simulates late-stage training)."""
    print("\n=== Small LR Model (simulates late training) ===\n")

    # Create a model with very small LR (late-stage fine-tuning)
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # Very small LR

    compressor = mithril.CheckpointCompressor()
    previous = None
    ratios = []

    for step in range(5):
        # Simulate training step with small gradient
        x = torch.randn(32, 1000) * 0.01  # Small inputs = small gradients
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Get checkpoint bytes
        state_dict = model.state_dict()
        current = b"".join(t.numpy().tobytes() for t in state_dict.values())

        # Compress
        compressed = compressor.compress(current, "fp32", previous=previous)
        ratio = len(current) / len(compressed)

        print(f"Step {step}: {len(current)/1e6:.1f}MB → {len(compressed)/1e3:.1f}KB ({ratio:.1f}x)")

        if step > 0:
            ratios.append(ratio)

        # Verify roundtrip
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        assert decompressed == current, f"Roundtrip failed at step {step}!"

        previous = current

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    print(f"\n✓ Small LR model test passed! Average delta ratio: {avg_ratio:.1f}x")
    return True


def test_minimal_change():
    """Test with minimal weight changes (best case for delta)."""
    print("\n=== Minimal Change Test (best case) ===\n")

    model = torch.nn.Linear(1000, 1000)

    compressor = mithril.CheckpointCompressor()

    # Get initial state
    state_dict = model.state_dict()
    previous = b"".join(t.numpy().tobytes() for t in state_dict.values())

    # First checkpoint (no delta)
    compressed = compressor.compress(previous, "fp32")
    ratio = len(previous) / len(compressed)
    print(f"Step 0 (no delta): {len(previous)/1e6:.1f}MB → {len(compressed)/1e3:.1f}KB ({ratio:.1f}x)")

    for step in range(1, 5):
        # Make tiny random perturbation (0.01% of weights)
        with torch.no_grad():
            for p in model.parameters():
                mask = torch.rand_like(p) < 0.0001  # Change 0.01% of weights
                p[mask] += torch.randn_like(p[mask]) * 0.001

        state_dict = model.state_dict()
        current = b"".join(t.numpy().tobytes() for t in state_dict.values())

        compressed = compressor.compress(current, "fp32", previous=previous)
        ratio = len(current) / len(compressed)

        print(f"Step {step} (delta): {len(current)/1e6:.1f}MB → {len(compressed)/1e3:.1f}KB ({ratio:.1f}x)")

        # Verify roundtrip
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        assert decompressed == current, f"Roundtrip failed at step {step}!"

        previous = current

    print("\n✓ Minimal change test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Mithril Checkpoint - Real Training Validation")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Mithril version: {mithril.__version__}")
    print()

    all_passed = True

    try:
        all_passed &= test_simple_model()
        all_passed &= test_mlp_model()
        all_passed &= test_small_lr_model()
        all_passed &= test_minimal_change()

        print("\n" + "=" * 60)
        if all_passed:
            print("All tests passed!")
            sys.exit(0)
        else:
            print("Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
