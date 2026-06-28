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

def compute_phash(image_path):
    """Compute the Perceptual Hash Of one image."""
    try:
        img = Image.open(image_path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception as e:
        print(f"Hash Failed for {image_path}: {e}")
        return None
    
def Find_Exact_Duplicates(paths_array, split = "train"):
    """Group Images with Identical Hashes -> Exact Duplicates."""
    print(f"\n{'='*30}")
    print(f"Exact Duplicate Detection for {split} set")
    print(f"\n{'='*30}")

    hash_map = {}
    records = []

    for idx, path in enumerate(paths_array):
        phash = compute_phash(path)
        if phash is None:
            continue

        if phash not in hash_map:
            hash_map[phash] = []

        hash_map[phash].append(path)
        records.append({"file_path": path, "phash": phash})

        if(idx + 1) % 1000 == 0:
            print(f"Hashed {idx + 1}/{len(paths_array)} images")

        
    # ----- Find groups with more than one image (duplicates) -----
    duplicate_groups = {}
    for h , ps in hash_map.items():
        if len(ps) > 1:
            duplicate_groups[h] = ps

    print(f"\n Total Images Processed: {len(paths_array)}")
    print(f" Duplicate Groups Found: {len(duplicate_groups)}")

    total_dup = sum(len(v)-1 for v in duplicate_groups.values())
    print(f" Total Duplicate Images found: {total_dup}")

    # ----- Build a DataFrame for reporting -----
    df = pd.DataFrame(records)
    df["Exact_duplicates"] = False
    df["duplicate_of"] = None

    for phash, group_paths in duplicate_groups.items():
        keep = group_paths[0]  # Keep the first image in the group
        remove = group_paths[1:]  # Mark the rest as duplicates

        for p in remove:
            df.loc[df["file_path"] == p, "Exact_duplicates"] = True
            df.loc[df["file_path"] == p, "duplicate_of"] = keep

    return df, duplicate_groups


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







    