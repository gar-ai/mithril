#!/usr/bin/env python3
"""Download HuggingFace fixtures for real-world validation testing.

This script downloads real model checkpoints and datasets for testing
mithril-checkpoint and mithril-dedup against real-world data.

Requirements:
    pip install huggingface_hub datasets

Usage:
    python scripts/download_hf_fixtures.py [--checkpoints] [--datasets] [--all]
"""
import argparse
import json
import os
from pathlib import Path


def download_model(checkpoints_dir: Path, repo_id: str, local_name: str) -> bool:
    """Download a single model from HuggingFace.

    Returns True if successful, False otherwise.
    """
    from huggingface_hub import snapshot_download

    print(f"\nDownloading {repo_id}...")
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=checkpoints_dir / local_name,
            allow_patterns=["*.safetensors", "config.json"],
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded to: {model_path}")

        # Show sizes
        total_size = 0
        for f in Path(model_path).glob("*.safetensors"):
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {f.name}: {size_mb:.1f} MB")

        if total_size == 0:
            print("  Warning: No .safetensors files found")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


# Model registry organized by category
MODELS = {
    "language": [
        ("openai-community/gpt2", "gpt2"),
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b"),
    ],
    "vision": [
        ("google/vit-base-patch16-224", "vit-base"),
        ("openai/clip-vit-base-patch32", "clip-vit-base"),
        ("facebook/dinov2-small", "dinov2-small"),
    ],
    "embedding": [
        ("sentence-transformers/all-MiniLM-L6-v2", "minilm-l6-v2"),
        ("BAAI/bge-small-en-v1.5", "bge-small-en"),
        ("thenlper/gte-small", "gte-small"),
    ],
}


def download_checkpoints(fixtures_dir: Path, categories: list = None) -> None:
    """Download real model checkpoints from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    checkpoints_dir = fixtures_dir / "hf_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Default to all categories
    if categories is None:
        categories = list(MODELS.keys())

    results = {"success": [], "failed": []}

    for category in categories:
        if category not in MODELS:
            print(f"Unknown category: {category}")
            continue

        print(f"\n{'=' * 40}")
        print(f"Category: {category.upper()}")
        print("=" * 40)

        for repo_id, local_name in MODELS[category]:
            if download_model(checkpoints_dir, repo_id, local_name):
                results["success"].append(local_name)
            else:
                results["failed"].append(local_name)

    # Summary
    print(f"\n{'=' * 40}")
    print("Download Summary")
    print("=" * 40)
    print(f"Success: {len(results['success'])} models")
    if results["failed"]:
        print(f"Failed: {', '.join(results['failed'])}")


def download_datasets(fixtures_dir: Path, sample_size: int = 10000) -> None:
    """Download sample datasets from HuggingFace for dedup testing."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        return

    datasets_dir = fixtures_dir / "hf_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Download C4 sample (English web text)
    print(f"\nDownloading allenai/c4 sample ({sample_size} docs)...")
    try:
        ds = load_dataset(
            "allenai/c4",
            "en",
            split=f"train[:{sample_size}]",
            streaming=False,
            trust_remote_code=True,
        )

        output_path = datasets_dir / "c4_sample.jsonl"
        with open(output_path, 'w') as f:
            for i, item in enumerate(ds):
                doc = {
                    "id": f"c4_{i}",
                    "text": item["text"],
                    "url": item.get("url", ""),
                }
                f.write(json.dumps(doc) + '\n')

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Created: {output_path} ({size_mb:.1f} MB, {sample_size} docs)")

    except Exception as e:
        print(f"  Error downloading c4: {e}")

    # Download Pile sample (diverse text corpus)
    print(f"\nDownloading EleutherAI/pile sample ({sample_size} docs)...")
    try:
        ds = load_dataset(
            "EleutherAI/pile",
            split=f"train[:{sample_size}]",
            streaming=False,
            trust_remote_code=True,
        )

        output_path = datasets_dir / "pile_sample.jsonl"
        with open(output_path, 'w') as f:
            for i, item in enumerate(ds):
                doc = {
                    "id": f"pile_{i}",
                    "text": item["text"],
                    "meta": item.get("meta", {}),
                }
                f.write(json.dumps(doc) + '\n')

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Created: {output_path} ({size_mb:.1f} MB, {sample_size} docs)")

    except Exception as e:
        print(f"  Error downloading pile: {e}")

    # Download Wikipedia sample (clean text, fewer duplicates expected)
    print(f"\nDownloading wikipedia sample ({sample_size} docs)...")
    try:
        ds = load_dataset(
            "wikipedia",
            "20220301.en",
            split=f"train[:{sample_size}]",
            streaming=False,
            trust_remote_code=True,
        )

        output_path = datasets_dir / "wikipedia_sample.jsonl"
        with open(output_path, 'w') as f:
            for i, item in enumerate(ds):
                doc = {
                    "id": f"wiki_{i}",
                    "text": item["text"],
                    "title": item.get("title", ""),
                }
                f.write(json.dumps(doc) + '\n')

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Created: {output_path} ({size_mb:.1f} MB, {sample_size} docs)")

    except Exception as e:
        print(f"  Error downloading wikipedia: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace fixtures for real-world testing"
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Download model checkpoints (all categories)"
    )
    parser.add_argument(
        "--language",
        action="store_true",
        help="Download language models (GPT-2, Qwen3, TinyLlama)"
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Download vision models (ViT, CLIP, DINOv2)"
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Download embedding models (MiniLM, BGE, GTE)"
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Download text datasets (c4, pile, wikipedia)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download everything"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of documents to sample from each dataset (default: 10000)"
    )
    args = parser.parse_args()

    # Determine which categories to download
    categories = []
    if args.language:
        categories.append("language")
    if args.vision:
        categories.append("vision")
    if args.embedding:
        categories.append("embedding")

    # --checkpoints or --all downloads all categories
    if args.checkpoints or args.all:
        categories = None  # None means all

    if not (categories or args.datasets or args.all):
        parser.print_help()
        print("\nNo action specified. Examples:")
        print("  --checkpoints          Download all model checkpoints")
        print("  --language             Download only language models")
        print("  --vision --embedding   Download vision and embedding models")
        print("  --datasets             Download text datasets")
        print("  --all                  Download everything")
        return

    # Project root
    project_root = Path(__file__).parent.parent
    fixtures_dir = project_root / "fixtures"

    print("=" * 60)
    print("HuggingFace Fixtures Downloader")
    print("=" * 60)
    print(f"Output directory: {fixtures_dir}")

    if categories is not None or args.all or args.checkpoints:
        download_checkpoints(fixtures_dir, categories)

    if args.all or args.datasets:
        download_datasets(fixtures_dir, args.sample_size)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
