#!/usr/bin/env python3
"""Test mithril-dedup with real HuggingFace datasets.

Tests MinHash + LSH deduplication on C4, Pile, and Wikipedia samples.

Usage:
    python scripts/test_dedup.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import mithril
except ImportError:
    print("ERROR: mithril not installed. Run: cd crates/mithril-python && maturin develop")
    sys.exit(1)


@dataclass
class DatasetResult:
    """Result from testing a dataset."""
    name: str
    total_docs: int
    unique_docs: int
    duplicate_count: int
    duplicate_ratio: float
    throughput: float  # docs/sec
    processing_time: float


def load_jsonl(path: Path, text_field: str = "text") -> List[str]:
    """Load texts from a JSONL file."""
    texts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                if text_field in doc:
                    texts.append(doc[text_field])
    return texts


def test_dataset(
    path: Path,
    name: str,
    threshold: float = 0.85,
    text_field: str = "text",
) -> Optional[DatasetResult]:
    """Test deduplication on a single dataset."""
    if not path.exists():
        print(f"  Skipping {name}: file not found")
        return None

    print(f"\n  Testing {name}...")

    # Load texts
    texts = load_jsonl(path, text_field)
    if not texts:
        print(f"  No texts found in {name}")
        return None

    print(f"    Loaded {len(texts)} documents")

    # Configure deduplicator
    config = mithril.dedup.DedupConfig(
        threshold=threshold,
        num_permutations=128,
        ngram_size=5,
    )
    dedup = mithril.dedup.Deduplicator(config)

    # Run deduplication
    start = time.time()
    result = dedup.deduplicate(texts)
    elapsed = time.time() - start

    stats = result.stats
    throughput = stats.total_documents / elapsed if elapsed > 0 else 0

    print(f"    Unique: {stats.unique_documents} ({100 - stats.duplicate_ratio * 100:.1f}%)")
    print(f"    Duplicates: {stats.duplicate_count} ({stats.duplicate_ratio * 100:.1f}%)")
    print(f"    Clusters: {stats.cluster_count}")
    print(f"    Throughput: {throughput:.0f} docs/sec")
    print(f"    Time: {elapsed:.2f}s")

    # Show some example duplicates
    if result.clusters and len(result.clusters) > 0:
        print(f"\n    Sample duplicate cluster:")
        # Get first cluster
        for rep_idx, dup_indices in list(result.clusters.items())[:1]:
            print(f"      Representative (idx {rep_idx}):")
            print(f"        {texts[rep_idx][:100]}...")
            for dup_idx in dup_indices[:2]:
                print(f"      Duplicate (idx {dup_idx}):")
                print(f"        {texts[dup_idx][:100]}...")

    return DatasetResult(
        name=name,
        total_docs=stats.total_documents,
        unique_docs=stats.unique_documents,
        duplicate_count=stats.duplicate_count,
        duplicate_ratio=stats.duplicate_ratio,
        throughput=throughput,
        processing_time=elapsed,
    )


def test_threshold_sensitivity(texts: List[str], name: str):
    """Test how threshold affects duplicate detection."""
    print(f"\n  Threshold sensitivity for {name}:")
    print(f"    {'Threshold':<12} {'Duplicates':<12} {'Ratio':<10}")
    print(f"    {'-' * 34}")

    for threshold in [0.70, 0.80, 0.85, 0.90, 0.95]:
        config = mithril.dedup.DedupConfig(threshold=threshold)
        dedup = mithril.dedup.Deduplicator(config)
        result = dedup.deduplicate(texts)
        stats = result.stats
        print(f"    {threshold:<12.2f} {stats.duplicate_count:<12} {stats.duplicate_ratio * 100:.1f}%")


def test_similarity_examples(dedup: mithril.dedup.Deduplicator):
    """Test similarity computation on example pairs."""
    print("\n  Similarity examples:")

    pairs = [
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox jumps over the lazy dog."),
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox leaps over the lazy dog."),
        ("Machine learning is a subset of artificial intelligence.",
         "Deep learning is a subset of machine learning."),
        ("Hello world", "Goodbye world"),
        ("Completely different text here", "Nothing similar at all"),
    ]

    for text1, text2 in pairs:
        sim = dedup.similarity(text1, text2)
        print(f"    {sim:.2f}: '{text1[:40]}...' vs '{text2[:40]}...'")


def main():
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "fixtures" / "hf_datasets"

    print("=" * 60)
    print("Mithril Dedup - HuggingFace Dataset Validation")
    print("=" * 60)
    print(f"Mithril version: {mithril.__version__}")
    print(f"Datasets dir: {datasets_dir}")

    # Test each dataset
    datasets = [
        ("c4_sample.jsonl", "C4 (web text)", "text"),
        ("pile_sample.jsonl", "Pile (diverse)", "text"),
        ("wikipedia_sample.jsonl", "Wikipedia", "text"),
    ]

    results = []
    all_texts = []

    print("\n" + "=" * 60)
    print("Dataset Deduplication Tests")
    print("=" * 60)

    for filename, name, text_field in datasets:
        path = datasets_dir / filename
        result = test_dataset(path, name, text_field=text_field)
        if result:
            results.append(result)
            # Collect texts for combined test
            all_texts.extend(load_jsonl(path, text_field))

    # Test combined dataset
    if all_texts:
        print("\n" + "=" * 60)
        print("Combined Dataset Test")
        print("=" * 60)

        config = mithril.dedup.DedupConfig(threshold=0.85)
        dedup = mithril.dedup.Deduplicator(config)

        start = time.time()
        result = dedup.deduplicate(all_texts)
        elapsed = time.time() - start

        stats = result.stats
        throughput = stats.total_documents / elapsed

        print(f"\n  Combined ({len(all_texts)} docs):")
        print(f"    Unique: {stats.unique_documents}")
        print(f"    Duplicates: {stats.duplicate_count} ({stats.duplicate_ratio * 100:.1f}%)")
        print(f"    Cross-dataset clusters: {stats.cluster_count}")
        print(f"    Throughput: {throughput:.0f} docs/sec")

        # Threshold sensitivity
        test_threshold_sensitivity(all_texts[:1000], "sample")

        # Similarity examples
        test_similarity_examples(dedup)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if results:
        print(f"\n{'Dataset':<20} {'Docs':<10} {'Dups':<10} {'Ratio':<10} {'Throughput':<15}")
        print("-" * 65)

        total_docs = 0
        total_time = 0
        for r in results:
            print(f"{r.name:<20} {r.total_docs:<10} {r.duplicate_count:<10} {r.duplicate_ratio * 100:.1f}%{'':<6} {r.throughput:.0f} docs/sec")
            total_docs += r.total_docs
            total_time += r.processing_time

        avg_throughput = total_docs / total_time if total_time > 0 else 0
        print("-" * 65)
        print(f"{'TOTAL':<20} {total_docs:<10} {'':<10} {'':<10} {avg_throughput:.0f} docs/sec")

        # Check targets
        print("\n" + "=" * 60)
        print("Target Verification")
        print("=" * 60)

        target_throughput = 100000
        if avg_throughput >= target_throughput:
            print(f"  Throughput: {avg_throughput:.0f} docs/sec >= {target_throughput} target")
        else:
            print(f"  Throughput: {avg_throughput:.0f} docs/sec < {target_throughput} target")
            print(f"    Note: Python bindings have overhead; Rust CLI should be faster")
    else:
        print("\nNo datasets found. Run:")
        print("  .venv/bin/python scripts/download_hf_fixtures.py --datasets")

    print("\n" + "=" * 60)
    print("Dedup tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
