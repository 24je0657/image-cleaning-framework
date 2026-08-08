import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import load_embeddings
import os

HASH_THRESHOLD = 5  # Hamming distance threshold for image hash comparison
COSINE_THRESHOLD = 0.97  # Cosine similarity threshold for embedding comparison
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
    
    
def Find_Exact_Duplicates(paths_array, split = "train"):
    """Multi-hash duplicate detection.
    Flags image pair if any hash type matches within threshold.
    """
    print(f"\n{'='*30}")
    print(f"Exact Duplicate Detection for {split} set")
    print(f"\n{'='*30}")

    hashes = {}
    records = []

    for idx, path in enumerate(paths_array):
        h = compute_multi_hash(path)
        if h is None:
            continue
        hashes[str(path)] = h
        records.append({"file_path":str(path)})


        if(idx + 1) % 1000 == 0:
            print(f"Hashed {idx + 1}/{len(paths_array)} images")

        
    # ----- Compare all pairs using multi-hash voting -----
    paths_list = list(hashes.keys())
    n =  len(paths_list)
    flagged = {}
    for i in range(n):
        if paths_list[i] in flagged:
            continue
        for j in range(i+1,n):
            if paths_list[j] in flagged:
                continue 
            if hashes_match(hashes[paths_list[i]],hashes[paths_list[j]]):
                flagged[paths_list[j]] = paths_list[i]

    print(f"\n Total Images Processed: {len(paths_array)}")
    print(f" Duplicate Images Found: {len(flagged)}")


    # ----- Build a DataFrame for reporting -----
    df = pd.DataFrame(records)
    df["Exact_duplicates"] = False
    df["duplicate_of"] = None

    for duplicate_path, original_path in flagged.items():
        df.loc[
            df["file_path"] == duplicate_path,
            "Exact_duplicates"
        ] = True

        df.loc[
            df["file_path"] == duplicate_path,
            "duplicate_of"
        ] = original_path
    

    return df, flagged


# ---- Near Duplicate Detection(COSINE_SIMILARITY) -----

def Find_Near_Duplicates(embeddings, paths_array, split = "train"):
    """Compute Pairwise Cosine Similarity on ResNet50 Embeddings.
    Flag Pairs above the COSINE_THRESHOLD as Near Duplicates.
    Use chunking to avoid memory issues for large datasets."""

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
                if j <= global_i:
                    continue  # Avoid self-comparison and duplicate pairs
                if score >= COSINE_THRESHOLD:
                    if global_i not in flagged:
                        flagged.add(j)
                        records.append({
                            "file_path": paths_array[j],
                            "near_duplicate_of": paths_array[global_i],
                            "cosine_score": round(float(score), 4),
                            "near_duplicate": True
                        })

        if(start + chunk_size) % 2000 == 0:
            print(f" Processed {start +chunk_size}/{n})")

    print(f"\n Near Duplicate pairs found: {len(records)}")

    df_near = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["file_path", "near_duplicate_of", "cosine_score", 
                 "near_duplicate"]
    )

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
            df_combined["near_duplicate"] = df_combined["file_path"].isin(
                df_near["file_path"]
            )
            df_combined["cosine_score"] = df_combined["file_path"].map(
                df_near.set_index("file_path")["cosine_score"]
            )
            df_combined["near_duplicate_of"] = df_combined["file_path"].map(
                df_near.set_index("file_path")["near_duplicate_of"]
            )
        else:
            df_combined["near_duplicate"] = False
            df_combined["cosine_score"] = np.nan
            df_combined["near_duplicate_of"] = np.nan

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







    