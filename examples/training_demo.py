#!/usr/bin/env python3
"""Mithril Training Demo: Checkpoint compression during fine-tuning.

Fine-tunes distilgpt2 for a few steps and compresses checkpoints
using Mithril's DeltaCompressor. Shows per-step compression stats
demonstrating how delta encoding achieves 10-70x compression on
consecutive training checkpoints.

Prerequisites:
    pip install torch transformers
    cd /path/to/mithril && maturin develop --release

Usage:
    python examples/training_demo.py
    python examples/training_demo.py --steps 5
    python examples/training_demo.py --model distilgpt2 --steps 10
"""

import argparse
import io
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


def serialize_state_dict(state_dict) -> bytes:
    """Serialize a PyTorch state_dict to bytes using torch.save."""
    import torch

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Mithril training checkpoint demo")
    parser.add_argument("--model", default="distilgpt2", help="HuggingFace model name")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    check_dependencies()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mithril

    print("=" * 60)
    print("  Mithril Training Checkpoint Compression Demo")
    print("=" * 60)

    # Load model and tokenizer
    print(f"\n  Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Create synthetic training data
    texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Machine learning models require large datasets for effective training.",
        "Gradient descent optimizes neural network weights iteratively.",
        "Attention mechanisms allow models to focus on relevant input tokens.",
    ]

    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    inputs["labels"] = inputs["input_ids"].clone()

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create Mithril delta compressor
    compressor = mithril.DeltaCompressor()

    print(f"\n  Training for {args.steps} steps with Mithril checkpoint compression\n")
    print(f"  {'Step':>6} | {'Loss':>8} | {'Raw Size':>10} | {'Compressed':>10} | {'Ratio':>7} | {'Delta':>5}")
    print(f"  {'─' * 6}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 7}─┼─{'─' * 5}")

    total_raw = 0
    total_compressed = 0
    start_time = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        # Training step
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Serialize state dict
        state_dict = model.state_dict()
        raw_bytes = serialize_state_dict(state_dict)
        raw_size = len(raw_bytes)

        # Compress with Mithril (uses delta from previous step automatically)
        compressed, stats = compressor.compress_checkpoint(
            f"step_{step}", raw_bytes, "bf16"
        )
        compressed_size = len(compressed)

        total_raw += raw_size
        total_compressed += compressed_size

        # Print step stats
        delta_marker = "yes" if stats.used_delta else "no"
        print(
            f"  {step:>6} | {loss.item():>8.4f} | "
            f"{raw_size / 1e6:>8.1f} MB | "
            f"{compressed_size / 1e6:>8.1f} MB | "
            f"{stats.ratio:>6.1f}x | "
            f"{delta_marker:>5}"
        )

    elapsed = time.time() - start_time

    # Print summary
    overall_ratio = total_raw / total_compressed if total_compressed > 0 else 0
    print(f"  {'─' * 6}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 7}─┼─{'─' * 5}")
    print(
        f"  {'Total':>6} | {'':>8} | "
        f"{total_raw / 1e6:>8.1f} MB | "
        f"{total_compressed / 1e6:>8.1f} MB | "
        f"{overall_ratio:>6.1f}x |"
    )

    print(f"\n  Time: {elapsed:.1f}s ({args.steps} steps)")
    print(f"  Storage saved: {(1 - total_compressed / total_raw) * 100:.1f}%")
    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
