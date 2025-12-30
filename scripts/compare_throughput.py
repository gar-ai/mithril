#!/usr/bin/env python3
"""Compare CLI vs Python throughput for mithril-checkpoint.

This script benchmarks both methods to identify where the bottleneck is.
"""

import subprocess
import sys
import time
from pathlib import Path

try:
    import mithril
except ImportError:
    print("ERROR: mithril not installed")
    sys.exit(1)


def benchmark_python(data: bytes, dtype: str, iterations: int = 3) -> float:
    """Benchmark Python bindings compression."""
    compressor = mithril.CheckpointCompressor()

    # Warmup
    compressor.compress(data, dtype)

    times = []
    for _ in range(iterations):
        start = time.time()
        compressed = compressor.compress(data, dtype)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    throughput = len(data) / avg_time / (1024**2)  # MiB/s
    ratio = len(data) / len(compressed)

    return throughput, ratio


def benchmark_cli(input_path: Path, iterations: int = 3) -> float:
    """Benchmark CLI compression."""
    output_path = Path("/tmp/benchmark_output.mcp")

    # Warmup
    subprocess.run(
        ["./target/release/mithril-checkpoint", "compress",
         str(input_path), "-o", str(output_path)],
        capture_output=True, check=True
    )

    times = []
    for _ in range(iterations):
        start = time.time()
        result = subprocess.run(
            ["./target/release/mithril-checkpoint", "compress",
             str(input_path), "-o", str(output_path)],
            capture_output=True, text=True
        )
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    # Parse throughput from output
    original_size = input_path.stat().st_size
    throughput = original_size / avg_time / (1024**2)  # MiB/s

    output_path.unlink(missing_ok=True)
    return throughput


def main():
    project_root = Path(__file__).parent.parent

    print("=" * 60)
    print("Mithril Checkpoint - Throughput Comparison")
    print("=" * 60)
    print(f"Mithril version: {mithril.__version__}")

    # Test files
    test_files = [
        ("GPT-2", project_root / "fixtures/hf_checkpoints/gpt2/model.safetensors"),
    ]

    print("\n" + "=" * 60)
    print("Raw Data Benchmark (Python bindings)")
    print("=" * 60)

    # Create test data of various sizes
    sizes = [
        ("1 MB", 1 * 1024 * 1024),
        ("10 MB", 10 * 1024 * 1024),
        ("100 MB", 100 * 1024 * 1024),
    ]

    print(f"\n{'Size':<12} {'Throughput':<15} {'Ratio':<10}")
    print("-" * 40)

    import random
    random.seed(42)

    for name, size in sizes:
        # Generate pseudo-random bf16-like data
        data = bytes(random.getrandbits(8) for _ in range(size))
        throughput, ratio = benchmark_python(data, "fp32")
        print(f"{name:<12} {throughput:>10.1f} MiB/s {ratio:>8.2f}x")

    print("\n" + "=" * 60)
    print("Model File Benchmark")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Method':<12} {'Throughput':<15}")
    print("-" * 50)

    for name, path in test_files:
        if not path.exists():
            print(f"{name:<20} {'N/A':<12} File not found")
            continue

        # For safetensors, we need to use CLI (Python bindings expect raw bytes)
        # Skip Python benchmark for model files - it processes tensor-by-tensor

        # CLI benchmark (processes entire safetensors file)
        cli_throughput = benchmark_cli(path)
        print(f"{name:<20} {'CLI':<12} {cli_throughput:>10.1f} MiB/s")

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    print("""
Possible bottlenecks in Python bindings:
1. PyO3 data copying (Python bytes → Rust Vec<u8>)
2. GIL acquisition overhead
3. Memory allocation patterns
4. Tensor iteration overhead in stress_test.py

The CLI processes the entire file at once, while Python
may be processing tensor-by-tensor with overhead.
""")


if __name__ == "__main__":
    main()
