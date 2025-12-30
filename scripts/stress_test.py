#!/usr/bin/env python3
"""Comprehensive stress tests for mithril-checkpoint.

Tests compression across diverse model architectures:
- Language models (GPT-2, Qwen3, TinyLlama)
- Vision models (ViT, CLIP, DINOv2)
- Embedding models (MiniLM, BGE, GTE)

Also tests edge cases and stress scenarios.

Usage:
    python scripts/stress_test.py [--all] [--language] [--vision] [--embedding] [--edge-cases]
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass
class TestResult:
    """Result from a compression test."""
    model_name: str
    category: str
    original_size_mb: float
    compressed_size_mb: float
    ratio: float
    throughput_gibs: float
    dtype: str
    num_tensors: int
    delta_ratio: Optional[float] = None


def get_dtype_str(tensor: torch.Tensor) -> str:
    """Get mithril dtype string from tensor dtype."""
    dtype_map = {
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
    return dtype_map.get(tensor.dtype, "fp32")


def test_model(model_path: Path, model_name: str, category: str) -> Optional[TestResult]:
    """Test standalone compression on a model."""
    if not model_path.exists():
        print(f"  Skipping {model_name}: not downloaded")
        return None

    print(f"\n  Testing {model_name}...")

    try:
        state_dict = load_file(model_path)
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        return None

    compressor = mithril.CheckpointCompressor()
    total_original = 0
    total_compressed = 0
    total_time = 0
    detected_dtype = "unknown"

    for name, tensor in state_dict.items():
        # Detect and record dtype
        if detected_dtype == "unknown":
            detected_dtype = get_dtype_str(tensor)

        # Handle bfloat16 (not supported by numpy)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
            dtype_str = "fp16"
        else:
            dtype_str = get_dtype_str(tensor)

        # Skip non-float tensors for now
        if dtype_str not in ("fp16", "fp32", "bf16"):
            continue

        try:
            data = tensor.numpy().tobytes()
        except Exception:
            continue

        start = time.time()
        compressed = compressor.compress(data, dtype_str)
        total_time += time.time() - start

        total_original += len(data)
        total_compressed += len(compressed)

    if total_original == 0:
        print(f"  No compressible tensors in {model_name}")
        return None

    ratio = total_original / total_compressed
    throughput = total_original / total_time / (1024**3) if total_time > 0 else 0

    print(f"    Size: {total_original/1e6:.1f}MB -> {total_compressed/1e6:.1f}MB")
    print(f"    Ratio: {ratio:.2f}x @ {throughput:.2f} GiB/s")
    print(f"    Dtype: {detected_dtype} ({len(state_dict)} tensors)")

    return TestResult(
        model_name=model_name,
        category=category,
        original_size_mb=total_original / 1e6,
        compressed_size_mb=total_compressed / 1e6,
        ratio=ratio,
        throughput_gibs=throughput,
        dtype=detected_dtype,
        num_tensors=len(state_dict),
    )


def test_delta_compression(model_path: Path, model_name: str, num_steps: int = 3) -> Optional[float]:
    """Test delta compression with simulated fine-tuning."""
    if not model_path.exists():
        return None

    print(f"\n  Delta test on {model_name}...")

    try:
        state_dict = load_file(model_path)
    except Exception as e:
        print(f"  Error: {e}")
        return None

    compressor = mithril.CheckpointCompressor()

    # Convert to bytes (handle bf16)
    def state_to_bytes(sd):
        parts = []
        for name, tensor in sorted(sd.items()):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            try:
                parts.append(tensor.numpy().tobytes())
            except Exception:
                pass
        return b"".join(parts)

    previous = state_to_bytes(state_dict)
    if len(previous) == 0:
        return None

    ratios = []
    for step in range(1, num_steps + 1):
        # Simulate fine-tuning: small perturbation
        with torch.no_grad():
            for name, tensor in state_dict.items():
                if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    mask = torch.rand_like(tensor.float()) < 0.001
                    noise = torch.randn_like(tensor.float()) * 0.0001
                    if tensor.dtype == torch.bfloat16:
                        tensor.copy_(tensor.float().add_(noise * mask.float()).to(torch.bfloat16))
                    else:
                        tensor.add_((noise * mask.float()).to(tensor.dtype))

        current = state_to_bytes(state_dict)
        compressed = compressor.compress(current, "fp32", previous=previous)
        ratio = len(current) / len(compressed)
        ratios.append(ratio)

        # Verify roundtrip
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        assert decompressed == current, f"Roundtrip failed at step {step}!"

        previous = current

    avg_ratio = sum(ratios) / len(ratios)
    print(f"    Delta ratio: {avg_ratio:.0f}x (avg of {num_steps} steps)")
    return avg_ratio


def test_edge_cases():
    """Test edge cases and stress scenarios."""
    print("\n=== Edge Cases ===\n")
    compressor = mithril.CheckpointCompressor()
    all_passed = True

    # Test 1: Very small tensor
    print("  Testing very small tensor (<1KB)...")
    small = torch.randn(10, 10, dtype=torch.float32)
    data = small.numpy().tobytes()
    compressed = compressor.compress(data, "fp32")
    ratio = len(data) / len(compressed)
    print(f"    {len(data)} bytes -> {len(compressed)} bytes ({ratio:.2f}x)")

    # Verify roundtrip
    decompressed = compressor.decompress(compressed, "fp32", len(data))
    assert decompressed == data, "Small tensor roundtrip failed!"
    print("    Roundtrip: PASSED")

    # Test 2: Large tensor (100MB)
    print("\n  Testing large tensor (~100MB)...")
    large = torch.randn(5000, 5000, dtype=torch.float32)
    data = large.numpy().tobytes()
    start = time.time()
    compressed = compressor.compress(data, "fp32")
    elapsed = time.time() - start
    ratio = len(data) / len(compressed)
    throughput = len(data) / elapsed / (1024**3)
    print(f"    {len(data)/1e6:.1f}MB -> {len(compressed)/1e6:.1f}MB ({ratio:.2f}x) @ {throughput:.2f} GiB/s")

    # Test 3: Delta chain (10 consecutive checkpoints)
    print("\n  Testing delta chain (10 checkpoints)...")
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    previous = None
    chain_ratios = []

    for step in range(10):
        x = torch.randn(32, 1000)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        state_dict = model.state_dict()
        current = b"".join(t.numpy().tobytes() for t in state_dict.values())

        compressed = compressor.compress(current, "fp32", previous=previous)
        ratio = len(current) / len(compressed)

        if step > 0:
            chain_ratios.append(ratio)

        # Verify
        decompressed = compressor.decompress(compressed, "fp32", len(current), previous=previous)
        if decompressed != current:
            print(f"    Step {step}: Roundtrip FAILED!")
            all_passed = False
            break

        previous = current

    if chain_ratios:
        avg = sum(chain_ratios) / len(chain_ratios)
        print(f"    Average delta ratio: {avg:.1f}x")
    print("    Roundtrip: PASSED" if all_passed else "    Roundtrip: FAILED")

    # Test 4: All zeros (best case)
    print("\n  Testing all-zeros tensor...")
    zeros = torch.zeros(1000, 1000, dtype=torch.float32)
    data = zeros.numpy().tobytes()
    compressed = compressor.compress(data, "fp32")
    ratio = len(data) / len(compressed)
    print(f"    {len(data)/1e6:.1f}MB -> {len(compressed)/1e3:.1f}KB ({ratio:.0f}x)")

    # Test 5: Random data (worst case)
    print("\n  Testing random data (worst case)...")
    random_data = torch.randn(1000, 1000, dtype=torch.float32)
    data = random_data.numpy().tobytes()
    compressed = compressor.compress(data, "fp32")
    ratio = len(data) / len(compressed)
    print(f"    {len(data)/1e6:.1f}MB -> {len(compressed)/1e6:.2f}MB ({ratio:.2f}x)")

    return all_passed


def run_category_tests(fixtures_dir: Path, category: str, models: List[Tuple[str, str]]) -> List[TestResult]:
    """Run tests for a category of models."""
    print(f"\n{'=' * 60}")
    print(f"Category: {category.upper()}")
    print("=" * 60)

    results = []
    checkpoints_dir = fixtures_dir / "hf_checkpoints"

    for local_name, display_name in models:
        # Find model.safetensors
        model_dir = checkpoints_dir / local_name
        model_path = model_dir / "model.safetensors"

        # Some models have different file names
        if not model_path.exists():
            # Try to find any .safetensors file
            safetensor_files = list(model_dir.glob("*.safetensors"))
            if safetensor_files:
                model_path = safetensor_files[0]

        result = test_model(model_path, display_name, category)
        if result:
            # Also test delta compression
            delta_ratio = test_delta_compression(model_path, display_name)
            if delta_ratio:
                result.delta_ratio = delta_ratio
            results.append(result)

    return results


def print_summary(results: List[TestResult]):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Group by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    print(f"\n{'Model':<25} {'Size':>10} {'Ratio':>8} {'Throughput':>12} {'Delta':>8}")
    print("-" * 70)

    for category, cat_results in categories.items():
        print(f"\n{category.upper()}")
        for r in cat_results:
            delta_str = f"{r.delta_ratio:.0f}x" if r.delta_ratio else "N/A"
            print(f"  {r.model_name:<23} {r.original_size_mb:>8.1f}MB {r.ratio:>7.2f}x {r.throughput_gibs:>10.2f} GiB/s {delta_str:>8}")

    # Overall stats
    if results:
        avg_ratio = sum(r.ratio for r in results) / len(results)
        avg_throughput = sum(r.throughput_gibs for r in results) / len(results)
        delta_results = [r for r in results if r.delta_ratio]
        avg_delta = sum(r.delta_ratio for r in delta_results) / len(delta_results) if delta_results else 0

        print(f"\n{'-' * 70}")
        print(f"  {'AVERAGE':<23} {'':>10} {avg_ratio:>7.2f}x {avg_throughput:>10.2f} GiB/s {avg_delta:>7.0f}x")


def main():
    parser = argparse.ArgumentParser(description="Stress test mithril-checkpoint")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--language", action="store_true", help="Test language models")
    parser.add_argument("--vision", action="store_true", help="Test vision models")
    parser.add_argument("--embedding", action="store_true", help="Test embedding models")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests")
    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.language, args.vision, args.embedding, args.edge_cases]):
        args.all = True

    project_root = Path(__file__).parent.parent
    fixtures_dir = project_root / "fixtures"

    print("=" * 60)
    print("Mithril Checkpoint - Comprehensive Stress Tests")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Mithril version: {mithril.__version__}")
    print(f"Fixtures: {fixtures_dir}")

    all_results = []

    # Language models
    if args.all or args.language:
        language_models = [
            ("gpt2", "GPT-2"),
            ("qwen3-0.6b", "Qwen3-0.6B"),
            ("tinyllama-1.1b", "TinyLlama-1.1B"),
        ]
        results = run_category_tests(fixtures_dir, "language", language_models)
        all_results.extend(results)

    # Vision models
    if args.all or args.vision:
        vision_models = [
            ("vit-base", "ViT-Base"),
            ("clip-vit-base", "CLIP-ViT-Base"),
            ("dinov2-small", "DINOv2-Small"),
        ]
        results = run_category_tests(fixtures_dir, "vision", vision_models)
        all_results.extend(results)

    # Embedding models
    if args.all or args.embedding:
        embedding_models = [
            ("minilm-l6-v2", "MiniLM-L6-v2"),
            ("bge-small-en", "BGE-Small-EN"),
            ("gte-small", "GTE-Small"),
        ]
        results = run_category_tests(fixtures_dir, "embedding", embedding_models)
        all_results.extend(results)

    # Edge cases
    if args.all or args.edge_cases:
        edge_passed = test_edge_cases()

    # Summary
    if all_results:
        print_summary(all_results)

    print("\n" + "=" * 60)
    print("Stress tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
