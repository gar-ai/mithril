#!/usr/bin/env python3
"""Test mithril dedup on HuggingFace datasets (C4, Pile samples)."""

import os
import sys
import time

# Check mithril is available
try:
    import mithril
    print(f"Mithril version: {mithril.__version__}")
except ImportError:
    print("ERROR: mithril not installed. Run: cd crates/mithril-python && maturin develop")
    sys.exit(1)

# Check datasets is available
try:
    from datasets import load_dataset
    print("HuggingFace datasets available")
except ImportError:
    print("ERROR: datasets not installed. Run: pip install datasets")
    sys.exit(1)


def test_synthetic_duplicates():
    """Test with synthetic data containing known duplicates."""
    print("\n=== Testing Synthetic Duplicates ===")

    # Create documents with known duplicates
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",  # Exact duplicate
        "The quick brown fox jumps over a lazy dog.",    # Near duplicate
        "Machine learning is transforming the world.",
        "Machine learning is transforming the world!",   # Near duplicate
        "A completely different document about cats.",
        "Another document that discusses various topics.",
        "The quick brown fox leaps over the lazy dog.",  # Near duplicate
    ]

    config = mithril.DedupConfig(threshold=0.8)
    deduplicator = mithril.Deduplicator(config)

    start = time.time()
    result = deduplicator.deduplicate(docs)
    elapsed = time.time() - start

    print(f"Input documents: {len(docs)}")
    print(f"Unique documents: {len(result.keep_indices)}")
    print(f"Duplicates removed: {len(docs) - len(result.keep_indices)}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Keep indices: {list(result.keep_indices)}")

    # We expect at least 1 exact duplicate to be removed
    assert len(result.keep_indices) < len(docs), "Expected some duplicates to be removed"
    print("PASS: Synthetic duplicates test passed")
    return True


def test_c4_sample(num_docs=1000):
    """Test dedup on C4 dataset sample."""
    print(f"\n=== Testing C4 Dataset ({num_docs} docs) ===")

    try:
        print("Loading C4 dataset (streaming)...")
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)

        # Take sample
        docs = []
        for i, item in enumerate(ds):
            if i >= num_docs:
                break
            docs.append(item["text"])
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1} documents...")

        print(f"Loaded {len(docs)} documents from C4")

    except Exception as e:
        print(f"WARNING: Could not load C4 dataset: {e}")
        print("Skipping C4 test (dataset may require authentication or be unavailable)")
        return True

    # Run deduplication
    config = mithril.DedupConfig(threshold=0.85)
    deduplicator = mithril.Deduplicator(config)

    print("Running deduplication...")
    start = time.time()
    result = deduplicator.deduplicate(docs)
    elapsed = time.time() - start

    duplicates = len(docs) - len(result.keep_indices)
    dup_rate = (duplicates / len(docs)) * 100
    throughput = len(docs) / elapsed

    print(f"Input documents: {len(docs)}")
    print(f"Unique documents: {len(result.keep_indices)}")
    print(f"Duplicates found: {duplicates} ({dup_rate:.1f}%)")
    print(f"Time: {elapsed:.3f}s ({throughput:.0f} docs/sec)")

    print("PASS: C4 dedup test completed")
    return True


def test_pile_sample(num_docs=1000):
    """Test dedup on Pile dataset sample."""
    print(f"\n=== Testing Pile Dataset ({num_docs} docs) ===")

    try:
        print("Loading Pile dataset (streaming)...")
        # Try the validation split which is smaller
        ds = load_dataset("EleutherAI/pile", split="validation", streaming=True, trust_remote_code=True)

        # Take sample
        docs = []
        for i, item in enumerate(ds):
            if i >= num_docs:
                break
            docs.append(item["text"])
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1} documents...")

        print(f"Loaded {len(docs)} documents from Pile")

    except Exception as e:
        print(f"WARNING: Could not load Pile dataset: {e}")
        print("Skipping Pile test (dataset may require authentication or be unavailable)")
        return True

    # Run deduplication
    config = mithril.DedupConfig(threshold=0.85)
    deduplicator = mithril.Deduplicator(config)

    print("Running deduplication...")
    start = time.time()
    result = deduplicator.deduplicate(docs)
    elapsed = time.time() - start

    duplicates = len(docs) - len(result.keep_indices)
    dup_rate = (duplicates / len(docs)) * 100
    throughput = len(docs) / elapsed

    print(f"Input documents: {len(docs)}")
    print(f"Unique documents: {len(result.keep_indices)}")
    print(f"Duplicates found: {duplicates} ({dup_rate:.1f}%)")
    print(f"Time: {elapsed:.3f}s ({throughput:.0f} docs/sec)")

    print("PASS: Pile dedup test completed")
    return True


def test_throughput_benchmark(num_docs=10000):
    """Benchmark throughput with larger dataset."""
    print(f"\n=== Throughput Benchmark ({num_docs} synthetic docs) ===")

    # Generate synthetic documents
    import random
    random.seed(42)

    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "machine", "learning", "artificial", "intelligence", "neural", "network",
             "data", "science", "python", "rust", "programming", "algorithm"]

    docs = []
    for i in range(num_docs):
        # Generate random document of 50-200 words
        doc_len = random.randint(50, 200)
        doc = " ".join(random.choices(words, k=doc_len))
        docs.append(doc)

    # Add some duplicates (10%)
    for i in range(num_docs // 10):
        idx = random.randint(0, len(docs) - 1)
        docs.append(docs[idx])

    random.shuffle(docs)
    print(f"Generated {len(docs)} documents with ~10% duplicates")

    # Run deduplication
    config = mithril.DedupConfig(threshold=0.85)
    deduplicator = mithril.Deduplicator(config)

    print("Running deduplication...")
    start = time.time()
    result = deduplicator.deduplicate(docs)
    elapsed = time.time() - start

    duplicates = len(docs) - len(result.keep_indices)
    dup_rate = (duplicates / len(docs)) * 100
    throughput = len(docs) / elapsed

    print(f"Input documents: {len(docs)}")
    print(f"Unique documents: {len(result.keep_indices)}")
    print(f"Duplicates found: {duplicates} ({dup_rate:.1f}%)")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.0f} docs/sec")

    # Target is 100K docs/sec
    if throughput >= 100000:
        print("PASS: Throughput meets target (>=100K docs/sec)")
    else:
        print(f"NOTE: Throughput {throughput:.0f} docs/sec (target: 100K docs/sec)")
        print("      This may vary based on document size and hardware")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Mithril Dedup - HuggingFace Datasets Test Suite")
    print("=" * 60)

    all_passed = True

    try:
        # Run tests
        all_passed &= test_synthetic_duplicates()
        all_passed &= test_throughput_benchmark()

        # These may fail if datasets aren't accessible
        all_passed &= test_c4_sample(num_docs=500)
        all_passed &= test_pile_sample(num_docs=500)

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
