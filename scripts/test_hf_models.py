#!/usr/bin/env python3
"""Test mithril compression on real HuggingFace models.

This script validates mithril-checkpoint compression against real model weights:
1. Standalone compression (target: 1.3-1.5x)
2. Delta compression with simulated fine-tuning (target: 39-70x)

Usage:
    python scripts/test_hf_models.py
"""

import sys
import time
from pathlib import Path

# Check dependencies
try:
    import torch
    from safetensors.torch import load_file
except ImportError:
    print("ERROR: Install deps: pip install torch safetensors")
    sys.exit(1)

try:
    import mithril
except ImportError:
    print("ERROR: mithril not installed. Run: cd crates/mithril-python && maturin develop")
    sys.exit(1)


def test_standalone_compression(model_path: Path):
    """Test compression of pre-trained model weights (no delta)."""
    print(f"\n=== Standalone Compression: {model_path.name} ===\n")

    # Load model
    state_dict = load_file(model_path)

    compressor = mithril.CheckpointCompressor()
    total_original = 0
    total_compressed = 0
    total_time = 0

    for name, tensor in state_dict.items():
        # Handle dtype
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
            dtype = "fp16"
        elif tensor.dtype == torch.float16:
            dtype = "fp16"
        else:
            dtype = "fp32"

        data = tensor.numpy().tobytes()

        start = time.time()
        compressed = compressor.compress(data, dtype)
        total_time += time.time() - start

        ratio = len(data) / len(compressed)
        total_original += len(data)
        total_compressed += len(compressed)

        if len(data) > 1_000_000:  # Only print large tensors
            print(f"  {name}: {len(data)/1e6:.1f}MB -> {len(compressed)/1e6:.2f}MB ({ratio:.2f}x)")

    overall_ratio = total_original / total_compressed
    throughput = total_original / total_time / (1024**3)
    print(f"\nTotal: {total_original/1e6:.1f}MB -> {total_compressed/1e6:.1f}MB ({overall_ratio:.2f}x) @ {throughput:.2f} GiB/s")

    return overall_ratio, throughput


def test_delta_compression(model_path: Path, num_steps: int = 5):
    """Simulate fine-tuning and test delta compression."""
    print(f"\n=== Delta Compression (Simulated Fine-tuning): {model_path.name} ===\n")

    state_dict = load_file(model_path)
    compressor = mithril.CheckpointCompressor()

    # Convert to a single byte buffer
    def state_to_bytes(sd):
        parts = []
        for name, tensor in sorted(sd.items()):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            parts.append(tensor.numpy().tobytes())
        return b"".join(parts)

    previous = state_to_bytes(state_dict)

    # First checkpoint (no delta)
    compressed = compressor.compress(previous, "fp32")
    print(f"Step 0 (baseline): {len(previous)/1e6:.1f}MB -> {len(compressed)/1e6:.1f}MB ({len(previous)/len(compressed):.1f}x)")

    ratios = []
    throughputs = []

    for step in range(1, num_steps + 1):
        # Simulate fine-tuning: small perturbation to weights
        with torch.no_grad():
            for name, tensor in state_dict.items():
                if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    # Change ~0.1% of weights slightly (simulates gradient update)
                    mask = torch.rand_like(tensor.float()) < 0.001
                    noise = torch.randn_like(tensor.float()) * 0.0001
                    if tensor.dtype == torch.bfloat16:
                        tensor.copy_(tensor.float().add_(noise * mask.float()).to(torch.bfloat16))
                    else:
                        tensor.add_((noise * mask.float()).to(tensor.dtype))

        current = state_to_bytes(state_dict)

        start = time.time()
        compressed = compressor.compress(current, "fp32", previous=previous)
        elapsed = time.time() - start

        ratio = len(current) / len(compressed)
        throughput = len(current) / elapsed / (1024**3)
        ratios.append(ratio)
        throughputs.append(throughput)

        print(f"Step {step} (delta): {len(current)/1e6:.1f}MB -> {len(compressed)/1e3:.0f}KB ({ratio:.0f}x) @ {throughput:.2f} GiB/s")

        # Verify roundtrip
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        assert decompressed == current, f"Roundtrip failed at step {step}!"

        previous = current

    avg_ratio = sum(ratios) / len(ratios)
    avg_throughput = sum(throughputs) / len(throughputs)
    print(f"\nAverage delta ratio: {avg_ratio:.0f}x @ {avg_throughput:.2f} GiB/s")
    print("Roundtrip verification: PASSED")

    return avg_ratio, avg_throughput


def test_cli_compression(model_path: Path):
    """Test CLI compression on the model file."""
    import subprocess
    import tempfile

    print(f"\n=== CLI Compression Test ===\n")

    with tempfile.NamedTemporaryFile(suffix=".mcp", delete=False) as f:
        output_path = f.name

    try:
        # Run CLI compress
        cmd = [
            "cargo", "run", "-p", "mithril-checkpoint", "--release", "--",
            "compress", str(model_path), "-o", output_path
        ]
        print(f"Running: {' '.join(cmd[-4:])}")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=model_path.parent.parent.parent.parent)
        if result.returncode != 0:
            print(f"CLI failed: {result.stderr}")
            return None

        # Check sizes
        original_size = model_path.stat().st_size
        compressed_size = Path(output_path).stat().st_size
        ratio = original_size / compressed_size

        print(f"Original:   {original_size/1e6:.1f} MB")
        print(f"Compressed: {compressed_size/1e6:.1f} MB")
        print(f"Ratio:      {ratio:.2f}x")

        return ratio

    finally:
        Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    gpt2_path = project_root / "fixtures/hf_checkpoints/gpt2/model.safetensors"

    if not gpt2_path.exists():
        print(f"ERROR: {gpt2_path} not found")
        print("Run: python scripts/download_hf_fixtures.py --checkpoints")
        sys.exit(1)

    print("=" * 60)
    print("Mithril - Real HuggingFace Model Validation")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Mithril version: {mithril.__version__}")
    print(f"Model: {gpt2_path}")
    print(f"Model size: {gpt2_path.stat().st_size / 1e6:.1f} MB")

    # Test 1: Standalone compression
    standalone_ratio, standalone_throughput = test_standalone_compression(gpt2_path)

    # Test 2: Delta compression
    delta_ratio, delta_throughput = test_delta_compression(gpt2_path)

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Standalone compression: {standalone_ratio:.2f}x @ {standalone_throughput:.2f} GiB/s (target: 1.3-1.5x)")
    print(f"Delta compression:      {delta_ratio:.0f}x @ {delta_throughput:.2f} GiB/s (target: 39-70x)")

    passed = True
    if standalone_ratio < 1.3:
        print(f"\n! Standalone below target: {standalone_ratio:.2f}x < 1.3x")
        passed = False
    if delta_ratio < 39:
        print(f"\n! Delta below target: {delta_ratio:.0f}x < 39x")
        passed = False

    if passed:
        print("\n ALL TARGETS MET!")
        sys.exit(0)
    else:
        print("\n Some targets not met - see details above")
        sys.exit(1)
