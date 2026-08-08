"""
Benchmark duplicate candidate-search strategies.

Compares:

1. Brute-force pairwise search
2. BK-tree candidate search

Important:
Image hashes are computed ONCE and reused by both methods.
This isolates the duplicate-search performance rather than
measuring image loading/hashing repeatedly.
"""

import time
import sys
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import load_embeddings
from src.duplicates import (
    compute_multi_hash,
    hashes_match,
    build_phash_bktree,
    query_bktree,
    HASH_THRESHOLD,
)


# ============================================================
# Configuration
# ============================================================

SPLIT = "train"

SAMPLE_SIZES = [
    500,
    2000,
    5000,
    None,       # full dataset
]


# ============================================================
# Hashing
# ============================================================

print("\n==============================================")
print("Preparing benchmark data")
print("==============================================")

_, _, paths = load_embeddings(SPLIT)

print(f"Total images available: {len(paths):,}")

print("\nComputing hashes once...")

t0 = time.perf_counter()

hashes_dict = {}

for idx, path in enumerate(paths):

    h = compute_multi_hash(path)

    if h is not None:
        hashes_dict[str(path)] = h

    if (idx + 1) % 1000 == 0:
        print(
            f"  Hashed "
            f"{idx + 1:,}/{len(paths):,}"
        )

hash_time = time.perf_counter() - t0

print(
    f"\nHash preparation time: "
    f"{hash_time:.2f}s"
)


# ============================================================
# Brute-force benchmark
# ============================================================

def brute_force_duplicates(hashes_dict):
    """
    Original O(n²) candidate search.

    Uses the SAME validated multi-hash rule as the BK-tree
    implementation.

    Returns:
        flagged dictionary
        number of pair comparisons
    """

    paths_list = list(hashes_dict.keys())

    flagged = {}

    comparisons = 0

    n = len(paths_list)

    for i in range(n):

        path_i = paths_list[i]

        if path_i in flagged:
            continue

        for j in range(i + 1, n):

            path_j = paths_list[j]

            if path_j in flagged:
                continue

            comparisons += 1

            if hashes_match(
                hashes_dict[path_i],
                hashes_dict[path_j]
            ):
                flagged[path_j] = path_i

    return flagged, comparisons


# ============================================================
# Benchmark
# ============================================================

print("\n")
print("=" * 75)
print("DUPLICATE DETECTION BENCHMARK")
print("=" * 75)

print(
    f"\n{'N':>8}"
    f"{'Brute(s)':>14}"
    f"{'BK-tree(s)':>14}"
    f"{'Speedup':>12}"
    f"{'Brute pairs':>16}"
    f"{'BK candidates':>16}"
)

print("-" * 90)


for requested_size in SAMPLE_SIZES:

    if requested_size is None:
        n = len(hashes_dict)
    else:
        n = min(
            requested_size,
            len(hashes_dict)
        )

    sample_paths = list(
        hashes_dict.keys()
    )[:n]

    sample_hashes = {
        path: hashes_dict[path]
        for path in sample_paths
    }

    print(f"\nRunning N = {n:,}")


    # ========================================================
    # Brute force
    # ========================================================

    t0 = time.perf_counter()

    brute_flagged, brute_comparisons = (
        brute_force_duplicates(
            sample_hashes
        )
    )

    brute_time = (
        time.perf_counter() - t0
    )


    # ========================================================
    # BK-tree
    # ========================================================

    t1 = time.perf_counter()

    tree = build_phash_bktree(
        sample_hashes
    )

    bk_flagged = query_bktree(
        tree,
        sample_hashes,
        HASH_THRESHOLD
    )

    bk_time = (
        time.perf_counter() - t1
    )


    # ========================================================
    # Compare results
    # ========================================================

    brute_set = set(
        brute_flagged.keys()
    )

    bk_set = set(
        bk_flagged.keys()
    )

    same_results = (
        brute_set == bk_set
    )

    missing_from_bk = (
        brute_set - bk_set
    )

    extra_in_bk = (
        bk_set - brute_set
    )


    # ========================================================
    # Speed
    # ========================================================

    speedup = (
        brute_time / bk_time
        if bk_time > 0
        else 0
    )


    print(
        f"{n:>8,}"
        f"{brute_time:>14.3f}"
        f"{bk_time:>14.3f}"
        f"{speedup:>11.2f}x"
        f"{brute_comparisons:>16,}"
        f"{'N/A':>16}"
    )

    print(
        f"  Brute duplicates : "
        f"{len(brute_flagged):,}"
    )

    print(
        f"  BK-tree duplicates: "
        f"{len(bk_flagged):,}"
    )

    print(
        f"  Results identical : "
        f"{'YES' if same_results else 'NO'}"
    )

    if not same_results:

        print(
            f"  Missing from BK-tree: "
            f"{len(missing_from_bk):,}"
        )

        print(
            f"  Extra in BK-tree: "
            f"{len(extra_in_bk):,}"
        )


# ============================================================
# Final note
# ============================================================

print("\n")
print("=" * 75)
print("BENCHMARK COMPLETE")
print("=" * 75)

print(
    "\nHash computation was performed once and excluded "
    "from search-runtime comparison."
)

print(
    "This benchmark measures the cost of duplicate "
    "candidate search itself."
)