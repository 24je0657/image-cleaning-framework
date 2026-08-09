import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
try:
    from src.embeddings import load_embeddings
except ModuleNotFoundError:
    from embeddings import load_embeddings
import os

from pybktree import BKTree
import time



HASH_THRESHOLD = 5  # Hamming distance threshold for image hash comparison
# Confidence Band
COSINE_BANDS = {
    "DEFINITE" : 0.99,   # auto-remove
    "LIKELY"   : 0.97,   # flag for REMOVE
    "POSSIBLE" : 0.93,   # flag for REVIEW only
}
REPORTS_DIR = "reports"  # Directory to save the duplicate reports
os.makedirs(REPORTS_DIR, exist_ok=True)  # Create the reports directory if it doesn't exist


# ----- Exact Duplicate Detection -----

def compute_multi_hash(image_path):
    """
    Compute 4 perceptual hash types.
    Each captures different transformation invariances:
      phash  → DCT-based, robust to compression/brightness
      dhash  → gradient-based, robust to contrast changes
      whash  → wavelet-based, robust to minor crops
      ahash  → average-based, fastest, least robust
    Returns dict of hashes or None if image unreadable.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        return {
            "phash" : imagehash.phash(img),
            "dhash" : imagehash.dhash(img),
            "whash" : imagehash.whash(img),
            "ahash" : imagehash.average_hash(img),
        }
    except Exception as e:
        print(f"  Hash failed for {image_path}: {e}")
        return None


def hashes_match(h1: dict, h2: dict, threshold: int = HASH_THRESHOLD) -> bool:
    """
    Two images are duplicates when atleast 2 out of pHash + (dHash OR wHash OR aHash) matches
    within hamming threshold.
    Catches more transformation variants than single hash.
    """


    phash_match = (
        abs(h1["phash"] - h2["phash"]) <= threshold
    )

    if not phash_match:
        return False

    supporting_matches = sum([
        abs(h1["dhash"] - h2["dhash"]) <= threshold,
        abs(h1["whash"] - h2["whash"]) <= threshold,
        abs(h1["ahash"] - h2["ahash"]) <= threshold
    ])

    return supporting_matches >= 1
# ─── BK-TREE HELPERS ──────────────────────────────────────────

def phash_hamming(item1: tuple, item2: tuple) -> int:
    """
    Hamming distance between two (phash_int, file_path) tuples.
    """
    return bin(item1[0] ^ item2[0]).count("1")


def build_phash_bktree(hashes_dict: dict) -> BKTree:
    """
    Build a BK-tree using pHash as the search key.

    The BK-tree is only used for candidate retrieval.
    Final duplicate verification is still performed using
    the existing multi-hash rule.
    """

    items = []

    for path, hashes in hashes_dict.items():

        # Convert ImageHash -> hexadecimal string -> integer
        phash_int = int(str(hashes["phash"]), 16)

        items.append(
            (phash_int, path)
        )

    return BKTree(
        phash_hamming,
        items
    )


def query_bktree(
    tree: BKTree,
    hashes_dict: dict,
    threshold: int = HASH_THRESHOLD
) -> dict:
    """
    Query every image against the pHash BK-tree.

    The BK-tree returns candidates whose pHash is within
    the specified Hamming distance.

    Each candidate is then checked using the existing
    multi-hash duplicate rule.
    """

    flagged = {}

    paths_list = list(hashes_dict.keys())

    for path in paths_list:

        if path in flagged:
            continue

        # Convert ImageHash -> hexadecimal string -> integer
        phash_int = int(
            str(hashes_dict[path]["phash"]),
            16
        )

        query = (
            phash_int,
            path
        )

        candidates = tree.find(
            query,
            threshold
        )

        for dist, (cand_phash_int, cand_path) in candidates:

            # Don't compare image with itself
            if cand_path == path:
                continue

            # Already marked as duplicate
            if cand_path in flagged:
                continue

            # Existing validated rule:
            # pHash + at least one supporting hash
            if hashes_match(
                hashes_dict[path],
                hashes_dict[cand_path]
            ):
                flagged[cand_path] = path

    return flagged
    
    
def Find_Exact_Duplicates(paths_array, split="train"):
    """
    Production duplicate detection using a BK-tree.

    Precision validation:
      v1 — any 2-of-4, threshold=5
            68% observed precision (34/50)

      v2 — pHash + 1 corroborating hash
            100% observed precision (50/50)

    Current approach:
      v3 — BK-tree indexed on pHash

    Pipeline:
      Step 1 — Hash all images
      Step 2 — Build BK-tree on pHash
      Step 3 — Query BK-tree for pHash candidates
      Step 4 — Verify candidates using the
               validated multi-hash rule

    Duplicate rule:
      pHash required + at least one corroborating hash.

    The BK-tree is only used to improve candidate retrieval.
    """

    print(f"\n{'=' * 30}")
    print(f"Exact Duplicate Detection — {split}")
    print(f"{'=' * 30}")

    print("Method    : BK-tree on pHash + multi-hash verification")
    print(f"Threshold : Hamming <= {HASH_THRESHOLD}")
    print("Rule      : pHash required + 1 corroborating hash")

    # ============================================================
    # Step 1 — Compute hashes
    # ============================================================

    hashes_dict = {}

    for idx, path in enumerate(paths_array):

        h = compute_multi_hash(path)

        if h is not None:
            hashes_dict[str(path)] = h

        if (idx + 1) % 1000 == 0:
            print(
                f"  Hashed {idx + 1:,}/{len(paths_array):,}"
            )

    print(
        f"\n  Hashing complete     : "
        f"{len(hashes_dict):,} images"
    )

    # ============================================================
    # Step 2 — Build BK-tree
    # ============================================================

    tree = build_phash_bktree(hashes_dict)

    print("  BK-tree built successfully")

    # ============================================================
    # Step 3 + 4 — Query + multi-hash verification
    # ============================================================

    flagged = query_bktree(
        tree,
        hashes_dict,
        HASH_THRESHOLD
    )

    print(
        f"  Hash-level flagged   : "
        f"{len(flagged):,}"
    )

    # ============================================================
    # Summary
    # ============================================================

    n_total = len(paths_array)

    print(
        f"\n── Summary ────────────────────────────────────────"
    )

    print(
        f"  Total images        : "
        f"{n_total:,}"
    )

    print(
        f"  Duplicates found    : "
        f"{len(flagged):,}"
    )

    print(
        f"\n── Validation History ────────────────────────────"
    )

    print(
        "  v1 (2-of-4)         : "
        "68% observed precision (34/50)"
    )

    print(
        "  v2 (pHash anchor)   : "
        "100% observed precision (50/50)"
    )

    print(
        "  Current rule        : "
        "pHash + 1 corroborating hash"
    )

    # ============================================================
    # Build report
    # ============================================================

    df = pd.DataFrame({
        "file_path": list(hashes_dict.keys())
    })

    df["Exact_duplicates"] = False
    df["duplicate_of"] = None

    # Replace the entire loop + df construction with this
    df = pd.DataFrame({"file_path": list(hashes_dict.keys())})

    # Single-pass map — O(n) not O(n × flagged)
    df["Exact_duplicates"] = df["file_path"].isin(flagged)
    df["duplicate_of"]     = df["file_path"].map(flagged)

    return df, flagged


# ---- Near Duplicate Detection(COSINE_SIMILARITY) -----

def Find_Near_Duplicates(embeddings, paths_array, split = "train"):
    """
    Three-tier near-duplicate detection using confidence bands.

    DEFINITE  (≥0.99) → REMOVE automatically
    LIKELY    (≥0.97) → flag for REMOVE
    POSSIBLE  (≥0.93) → flag for REVIEW only

    This replaces the binary flag/no-flag decision with
    a graded confidence score that the Decision Engine
    can weight appropriately.
    """

    print(f"\n{'='*30}")
    print(f"Near Duplicate Detection for {split} set")
    print(f"\n{'='*30}")

    n = len(embeddings)
    chunk_size = 500  # Adjust based on available memory
    flagged = set()
    records = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]

        #----- Cosine Similarity b/w this chunk and the entire embeddings -----
        similarity_matrix = cosine_similarity(chunk, embeddings)

        for i, row in enumerate(similarity_matrix):
            global_i = start + i
            for j, score in enumerate(row):
                if j <= global_i :
                    continue  # Avoid self-comparison and duplicate pairs
                if score >= COSINE_BANDS["POSSIBLE"]:
                    if score >= COSINE_BANDS["DEFINITE"]:
                        confidence = "DEFINITE"
                    elif score >= COSINE_BANDS["LIKELY"]:
                        confidence = "LIKELY"
                    else :
                        confidence = "POSSIBLE"
                    flagged.add(j)
                    records.append({
                         "file_path": paths_array[j],
                        "near_duplicate_of": paths_array[global_i],
                        "cosine_score": round(float(score), 4),
                        "near_duplicate": True,
                        "dup_confidence":confidence,
                        })

        if(start + chunk_size) % 2000 == 0:
            print(f" Processed {start +chunk_size}/{n})")

    print(f"\n Near Duplicate pairs found: {len(records)}")

    df_near = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["file_path", "near_duplicate_of", "cosine_score", 
                 "near_duplicate","dup_confidence"]
    )
    if len(df_near) > 0:
       for tier in ["DEFINITE","LIKELY","POSSIBLE"]:
          count = (df_near["dup_confidence"] == tier).sum()
          print(f"    {tier:10s}: {count}")

    return df_near , flagged

# ----- Main Execution -----

if __name__ == "__main__":
    for split in ["train", "val"]:
        embeddings, labels, paths_array = load_embeddings(split)

        # ----- Exact Duplicate -----
        df_exact, dup_groups = Find_Exact_Duplicates(paths_array, split)


        # ----- Near Duplicate -----
        df_near, near_flagged = Find_Near_Duplicates(embeddings, paths_array, split)

        # ----- Merge both reports  -----

        df_combined = df_exact.copy()

        if not df_near.empty:

        # Keep the strongest near-duplicate evidence for each image
            confidence_rank = {
                "POSSIBLE": 1,
                "LIKELY": 2,
                "DEFINITE": 3
            }

            df_near["confidence_rank"] = (
                df_near["dup_confidence"].map(confidence_rank)
            )

            df_near_best = (
                df_near
                .sort_values(
                    ["file_path", "confidence_rank", "cosine_score"],
                    ascending=[True, False, False]
                )
                .drop_duplicates(
                    subset=["file_path"],
                    keep="first"
                )
            )

            # Merge the strongest near-duplicate evidence
            # into the image-level master report
            df_combined = df_combined.merge(
                df_near_best[
                    [
                        "file_path",
                        "cosine_score",
                        "near_duplicate_of",
                        "dup_confidence"
                    ]
                ],
                on="file_path",
                how="left"
            )

            df_combined["near_duplicate"] = (
                df_combined["cosine_score"].notna()
            )

        else:

            df_combined["near_duplicate"] = False
            df_combined["cosine_score"] = np.nan
            df_combined["near_duplicate_of"] = np.nan
            df_combined["dup_confidence"] = np.nan

        # ----- Save the report -----
        out_path = f"{REPORTS_DIR}/{split}_duplicates_report.csv"
        df_combined.to_csv(out_path, index=False)
        print(f"\n Duplicate report saved to: {out_path}")

        # ---- Print Summary -----
        print(f"\n --- {split.upper()} SET SUMMARY ---")
        print(f" Exact duplicates :{df_exact['Exact_duplicates'].sum()}")
        print(f" Near Duplicates :{len(near_flagged)}")
        total_duplicates = df_combined['Exact_duplicates'].sum() + len(near_flagged)
        print(f" Total Duplicates :{total_duplicates}")







    