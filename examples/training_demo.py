#!/usr/bin/env python3
"""Mithril Training Demo: Checkpoint compression during fine-tuning.

Fine-tunes distilgpt2 for a few steps and compresses checkpoints
using Mithril's DeltaCompressor. Extracts raw tensor bytes from
state_dicts (avoiding pickle overhead) so that byte grouping and
delta encoding achieve high compression ratios on consecutive
training checkpoints.

Supports two modes:
  - Full fine-tuning: All parameters updated (realistic baseline)
  - Frozen layers: Only last N layers trained (simulates LoRA/PEFT)
    Frozen parameters are identical between steps, so XOR delta
    encoding achieves extreme compression ratios.

Prerequisites:
    pip install torch transformers
    cd /path/to/mithril && maturin develop --release

Usage:
    python examples/training_demo.py
    python examples/training_demo.py --steps 5
    python examples/training_demo.py --freeze-layers 1
    python examples/training_demo.py --compare
"""

import argparse
import sys
import time


def check_dependencies():
    """Check that required packages are available."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import mithril  # noqa: F401
    except ImportError:
        missing.append("mithril (run: cd /path/to/mithril && maturin develop --release)")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch transformers")
        print("For mithril: cd /path/to/mithril && maturin develop --release")
        sys.exit(1)


def serialize_state_dict(state_dict) -> tuple[bytes, str]:
    """Serialize a PyTorch state_dict to raw tensor bytes.

    Returns (bytes, dtype_str) where dtype_str is the dominant dtype.
    Extracts raw tensor data without pickle overhead for optimal compression.
    """
    buffers = []
    dtype_counts = {}

    for name, tensor in state_dict.items():
        # Convert to contiguous CPU tensor and get raw bytes
        t = tensor.detach().cpu().contiguous()
        raw = t.numpy().tobytes()
        buffers.append(raw)

        # Track dtype
        dt = str(tensor.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + tensor.numel()

    # Determine dominant dtype
    dominant = max(dtype_counts, key=dtype_counts.get)
    dtype_map = {
        "torch.float32": "fp32",
        "torch.float16": "fp16",
        "torch.bfloat16": "bf16",
        "torch.int8": "i8",
        "torch.int32": "i32",
        "torch.int64": "i64",
    }
    dtype_str = dtype_map.get(dominant, "uint8")

    return b"".join(buffers), dtype_str


def freeze_model_layers(model, keep_last_n=1):
    """Freeze all transformer layers except the last N.

    This simulates LoRA/PEFT fine-tuning where most parameters are frozen.
    The full state_dict still contains all weights, but frozen weights
    remain identical between steps -- perfect for delta compression.

    Returns the number of trainable and frozen parameters.
    """
    import torch

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last N transformer layers + LM head
    if hasattr(model, "transformer"):
        # GPT-2 style
        layers = model.transformer.h
        total_layers = len(layers)
        for i in range(max(0, total_layers - keep_last_n), total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        # Always unfreeze LM head
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA style
        layers = model.model.layers
        total_layers = len(layers)
        for i in range(max(0, total_layers - keep_last_n), total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def run_training(model, tokenizer, compressor, args, freeze_layers=None, label=None):
    """Run training loop and return compression stats."""
    import torch

    if label:
        print(f"\n  --- {label} ---")

    if freeze_layers is not None:
        trainable, frozen = freeze_model_layers(model, keep_last_n=freeze_layers)
        pct_frozen = frozen / (trainable + frozen) * 100
        print(f"  Frozen: {frozen:,} params ({pct_frozen:.1f}%) | Trainable: {trainable:,}")
    else:
        trainable = sum(p.numel() for p in model.parameters())
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        print(f"  All {trainable:,} parameters trainable")

    # Create training data
    texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Machine learning models require large datasets for effective training.",
        "Gradient descent optimizes neural network weights iteratively.",
        "Attention mechanisms allow models to focus on relevant input tokens.",
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=64,
    )
    inputs["labels"] = inputs["input_ids"].clone()

    # Setup optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # Clear compressor references for a clean run
    compressor.clear_references()

    print(f"\n  {'Step':>6} | {'Loss':>8} | {'Raw Size':>10} | {'Compressed':>10} | {'Ratio':>7} | {'Sparsity':>8} | {'Delta':>5}")
    print(f"  {'─' * 6}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 7}─┼─{'─' * 8}─┼─{'─' * 5}")

    total_raw = 0
    total_compressed = 0
    start_time = time.time()
    ratios = []

    model.train()
    for step in range(1, args.steps + 1):
        # Training step
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Serialize state dict (includes ALL weights, frozen and trainable)
        state_dict = model.state_dict()
        raw_bytes, dtype_str = serialize_state_dict(state_dict)
        raw_size = len(raw_bytes)

        # Compress with Mithril
        compressed, stats = compressor.compress_checkpoint(
            f"step_{step}", raw_bytes, dtype_str
        )
        compressed_size = len(compressed)

        total_raw += raw_size
        total_compressed += compressed_size
        ratios.append(stats.ratio)

        delta_marker = "yes" if stats.used_delta else "no"
        sparsity_str = f"{stats.sparsity:.1%}" if stats.used_delta else "n/a"
        print(
            f"  {step:>6} | {loss.item():>8.4f} | "
            f"{raw_size / 1e6:>8.1f} MB | "
            f"{compressed_size / 1e6:>8.1f} MB | "
            f"{stats.ratio:>6.1f}x | "
            f"{sparsity_str:>8} | "
            f"{delta_marker:>5}"
        )

    elapsed = time.time() - start_time

    overall_ratio = total_raw / total_compressed if total_compressed > 0 else 0
    print(f"  {'─' * 6}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 7}─┼─{'─' * 8}─┼─{'─' * 5}")
    print(
        f"  {'Total':>6} | {'':>8} | "
        f"{total_raw / 1e6:>8.1f} MB | "
        f"{total_compressed / 1e6:>8.1f} MB | "
        f"{overall_ratio:>6.1f}x |{'':>10}|"
    )
    print(f"\n  Time: {elapsed:.1f}s | Storage saved: {(1 - total_compressed / total_raw) * 100:.1f}%")

    # Return stats for comparison
    avg_delta_ratio = sum(ratios[1:]) / len(ratios[1:]) if len(ratios) > 1 else 0
    return {
        "total_raw": total_raw,
        "total_compressed": total_compressed,
        "overall_ratio": overall_ratio,
        "avg_delta_ratio": avg_delta_ratio,
        "elapsed": elapsed,
        "savings_pct": (1 - total_compressed / total_raw) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Mithril training checkpoint demo")
    parser.add_argument("--model", default="distilgpt2", help="HuggingFace model name")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--freeze-layers", type=int, default=None,
        help="Freeze all but last N layers (simulates LoRA/PEFT)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both full and frozen modes side-by-side for comparison"
    )
    args = parser.parse_args()

    check_dependencies()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mithril

    print("=" * 72)
    print("  Mithril Training Checkpoint Compression Demo")
    print("=" * 72)

    # Load model and tokenizer
    print(f"\n  Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    compressor = mithril.DeltaCompressor()

    if args.compare:
        # Run both modes and compare
        print(f"\n  Comparing full fine-tuning vs frozen layers ({args.steps} steps)")

        # Full fine-tuning
        model_full = AutoModelForCausalLM.from_pretrained(args.model)
        stats_full = run_training(
            model_full, tokenizer, compressor, args,
            freeze_layers=None, label="Full Fine-Tuning (all parameters updated)"
        )

        # Frozen layers (keep last 1)
        model_frozen = AutoModelForCausalLM.from_pretrained(args.model)
        stats_frozen = run_training(
            model_frozen, tokenizer, compressor, args,
            freeze_layers=1, label="Frozen Layers (last 1 layer trainable, simulates LoRA)"
        )

        # Comparison table
        print(f"\n  {'=' * 68}")
        print(f"  COMPARISON SUMMARY")
        print(f"  {'=' * 68}")
        print(f"  {'':>24} | {'Full Fine-Tune':>16} | {'Frozen Layers':>16}")
        print(f"  {'─' * 24}─┼─{'─' * 16}─┼─{'─' * 16}")
        print(f"  {'Total raw':>24} | {stats_full['total_raw']/1e6:>13.1f} MB | {stats_frozen['total_raw']/1e6:>13.1f} MB")
        print(f"  {'Total compressed':>24} | {stats_full['total_compressed']/1e6:>13.1f} MB | {stats_frozen['total_compressed']/1e6:>13.1f} MB")
        print(f"  {'Overall ratio':>24} | {stats_full['overall_ratio']:>15.1f}x | {stats_frozen['overall_ratio']:>15.1f}x")
        print(f"  {'Avg delta ratio':>24} | {stats_full['avg_delta_ratio']:>15.1f}x | {stats_frozen['avg_delta_ratio']:>15.1f}x")
        print(f"  {'Storage saved':>24} | {stats_full['savings_pct']:>14.1f}% | {stats_frozen['savings_pct']:>14.1f}%")
        print(f"  {'─' * 24}─┼─{'─' * 16}─┼─{'─' * 16}")

        improvement = stats_frozen['overall_ratio'] / stats_full['overall_ratio']
        print(f"\n  Delta compression is {improvement:.0f}x more effective with frozen layers.")
        print(f"  This is why Mithril pairs perfectly with LoRA/PEFT fine-tuning.")

    else:
        # Single mode
        run_training(
            model, tokenizer, compressor, args,
            freeze_layers=args.freeze_layers,
            label="Frozen Layers" if args.freeze_layers else "Full Fine-Tuning"
        )

    print(f"\n{'=' * 72}")
    print("  Done!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
