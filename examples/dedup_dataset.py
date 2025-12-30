#!/usr/bin/env python3
"""Example: Dataset deduplication with Mithril.

This example demonstrates how to deduplicate text datasets using Mithril's
MinHash/LSH-based deduplication.

Requirements:
    pip install mithril
"""

import json
import os
import tempfile
import time
from typing import Iterator

import mithril


def generate_sample_dataset(num_docs: int = 1000, dup_ratio: float = 0.3) -> list[dict]:
    """Generate a sample dataset with known duplicates.

    Args:
        num_docs: Total number of documents
        dup_ratio: Fraction of documents that are duplicates

    Returns:
        List of document dicts with 'id' and 'text' fields
    """
    import random
    random.seed(42)

    unique_count = int(num_docs * (1 - dup_ratio))
    docs = []

    templates = [
        "This is document {} containing information about {}.",
        "Article {} discusses the subject of {} in detail.",
        "Report number {} covers the analysis of {}.",
        "Study {} examines the relationship between {} and related factors.",
        "Paper {} presents findings on {} with comprehensive data.",
    ]

    topics = [
        "machine learning", "data processing", "neural networks",
        "optimization", "training", "inference", "models", "datasets",
        "transformers", "attention mechanisms", "gradient descent",
        "backpropagation", "embeddings", "tokenization",
    ]

    # Create unique documents
    for i in range(unique_count):
        template = random.choice(templates)
        topic = random.choice(topics)
        text = template.format(i, topic)
        # Add random suffix to make each truly unique
        text += f" Additional context: {random.random():.10f}"
        docs.append({"id": f"doc_{i}", "text": text, "is_duplicate": False})

    # Create exact duplicates
    for i in range(num_docs - unique_count):
        original = docs[i % unique_count]
        docs.append({
            "id": f"dup_{i}",
            "text": original["text"],
            "is_duplicate": True,
            "duplicate_of": original["id"],
        })

    random.shuffle(docs)
    return docs


def deduplicate_texts(texts: list[str], threshold: float = 0.85) -> mithril.dedup.DedupResult:
    """Deduplicate a list of texts.

    Args:
        texts: List of text strings
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        DedupResult with keep_indices, remove_indices, and stats
    """
    config = mithril.DedupConfig(threshold=threshold)
    deduplicator = mithril.Deduplicator(config)
    return deduplicator.deduplicate(texts)


def deduplicate_jsonl(
    input_path: str,
    output_path: str,
    text_field: str = "text",
    threshold: float = 0.85,
) -> dict:
    """Deduplicate a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path for deduplicated output
        text_field: Name of the text field in each JSON object
        threshold: Similarity threshold

    Returns:
        Dict with deduplication statistics
    """
    # Read all documents
    docs = []
    texts = []
    with open(input_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            docs.append(doc)
            texts.append(doc.get(text_field, ""))

    # Deduplicate
    start = time.perf_counter()
    result = deduplicate_texts(texts, threshold)
    elapsed = time.perf_counter() - start

    # Write unique documents
    keep_set = set(result.keep_indices)
    with open(output_path, "w") as f:
        for i, doc in enumerate(docs):
            if i in keep_set:
                f.write(json.dumps(doc) + "\n")

    return {
        "total_documents": result.stats.total_documents,
        "unique_documents": result.stats.unique_documents,
        "duplicate_count": result.stats.duplicate_count,
        "duplicate_ratio": result.stats.duplicate_ratio,
        "cluster_count": result.stats.cluster_count,
        "processing_time_secs": elapsed,
        "throughput_docs_per_sec": len(texts) / elapsed,
    }


def compute_pairwise_similarity(texts: list[str]) -> list[tuple[int, int, float]]:
    """Compute similarity between all pairs of texts.

    This is useful for understanding the duplicate clusters.

    Args:
        texts: List of text strings

    Returns:
        List of (i, j, similarity) tuples for pairs with similarity > 0
    """
    deduplicator = mithril.Deduplicator()
    pairs = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = deduplicator.similarity(texts[i], texts[j])
            if sim > 0:
                pairs.append((i, j, sim))

    return sorted(pairs, key=lambda x: -x[2])


def main():
    print("=" * 60)
    print("Mithril Dataset Deduplication Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate sample dataset
        print("\n1. Generating sample dataset...")
        docs = generate_sample_dataset(num_docs=1000, dup_ratio=0.3)

        # Count actual duplicates for verification
        actual_dups = sum(1 for d in docs if d.get("is_duplicate", False))
        print(f"   Generated {len(docs)} documents ({actual_dups} are duplicates)")

        # Save to JSONL
        input_path = os.path.join(tmpdir, "input.jsonl")
        with open(input_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        # Deduplicate
        print("\n2. Deduplicating...")
        output_path = os.path.join(tmpdir, "output.jsonl")
        stats = deduplicate_jsonl(input_path, output_path, threshold=0.85)

        print(f"   Total documents:    {stats['total_documents']}")
        print(f"   Unique documents:   {stats['unique_documents']}")
        print(f"   Duplicates found:   {stats['duplicate_count']}")
        print(f"   Duplicate ratio:    {stats['duplicate_ratio']:.1%}")
        print(f"   Clusters:           {stats['cluster_count']}")
        print(f"   Processing time:    {stats['processing_time_secs']:.3f}s")
        print(f"   Throughput:         {stats['throughput_docs_per_sec']:.0f} docs/sec")

        # Verify output
        print("\n3. Verifying output...")
        with open(output_path, "r") as f:
            output_docs = [json.loads(line) for line in f]
        print(f"   Output contains {len(output_docs)} documents")

        # Example: similarity between specific texts
        print("\n4. Similarity examples...")
        deduplicator = mithril.Deduplicator()

        pairs = [
            ("Hello world", "Hello world"),
            ("Hello world", "Hello there"),
            ("Machine learning is great", "Machine learning is awesome"),
            ("Completely different text", "Nothing in common"),
        ]

        for text1, text2 in pairs:
            sim = deduplicator.similarity(text1, text2)
            print(f"   '{text1[:30]}...' vs '{text2[:30]}...': {sim:.2f}")

        # Example: finding near-duplicates
        print("\n5. Near-duplicate detection...")
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumped over the lazy dog.",  # Near dup
            "A completely unrelated sentence about cats.",
            "The quick brown fox leaps over the lazy dog.",  # Near dup
        ]

        result = deduplicate_texts(sample_texts, threshold=0.7)
        print(f"   Input: {len(sample_texts)} texts")
        print(f"   Unique: {len(result.keep_indices)} texts (indices: {result.keep_indices})")
        print(f"   Clusters: {result.clusters}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
