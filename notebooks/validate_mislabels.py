# notebooks/validate_mislabels.py
"""
Visual validation of HIGH-confidence mislabeled images.
Opens side-by-side grid: flagged image vs 3 nearest neighbors
from the predicted class — lets you judge if the flag is correct.

Run: python notebooks/validate_mislabels.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL        import Image
from pathlib    import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing    import normalize

sys.path.insert(0, ".")


# ─── CONFIG ───────────────────────────────────────────────────────────────────
SPLIT          = "train"        # change to "val" for val set
N_IMAGES       = 20             # how many HIGH-confidence to review
N_NEIGHBORS    = 3              # nearest neighbors from predicted class to show
OUTPUT_DIR     = Path("notebooks/validation_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data(split):
    mis_df     = pd.read_csv(f"reports/{split}_mislabel_report.csv")
    embeddings = np.load(f"embeddings/{split}_embeddings.npy")
    labels     = np.load(f"embeddings/{split}_labels.npy",
                         allow_pickle=True)
    paths      = np.load(f"embeddings/{split}_file_paths.npy",
                         allow_pickle=True)
    return mis_df, embeddings, labels, paths


# ─── NEAREST NEIGHBOR LOOKUP ──────────────────────────────────────────────────

def get_neighbors_from_class(query_emb, all_embs, all_paths,
                              all_labels, target_class, n=3):
    """
    Find N nearest neighbors of query_emb within target_class.
    Returns list of (path, similarity_score).
    """
    mask       = all_labels == target_class
    class_embs = all_embs[mask]
    class_paths= all_paths[mask]

    normed_query = normalize(query_emb.reshape(1, -1))
    normed_class = normalize(class_embs)
    sims         = cosine_similarity(normed_query, normed_class)[0]
    top_n        = np.argsort(sims)[::-1][:n]

    return [(class_paths[i], round(float(sims[i]), 4)) for i in top_n]


# ─── SINGLE IMAGE VALIDATION CARD ─────────────────────────────────────────────

def make_validation_card(row, embeddings, labels, paths, idx):
    """
    Create one validation card:
      Left:  flagged image with its declared class
      Right: N nearest neighbors from the predicted class

    This lets you answer visually:
      "Does this image actually look like the predicted class?"
    """
    flagged_path    = row["file_path"]
    declared_class  = row["declared_class"]
    predicted_class = row["nn_predicted"]
    votes           = row["votes"]
    confidence      = row["confidence"]
    mislabel_votes  = row.get("mislabel_votes", votes)

    # Find embedding index for flagged image
    path_idx = np.where(paths == flagged_path)[0]
    if len(path_idx) == 0:
        print(f"  Embedding not found for {flagged_path}")
        return None

    query_emb = embeddings[path_idx[0]]
    neighbors = get_neighbors_from_class(
        query_emb, embeddings, paths,
        labels, predicted_class, N_NEIGHBORS
    )

    # ── Build figure ──────────────────────────────────────────────
    fig_width  = 3 + N_NEIGHBORS * 2.5
    fig, axes  = plt.subplots(1, 1 + N_NEIGHBORS,
                               figsize=(fig_width, 3.2))
    fig.patch.set_facecolor("#F8F8F8")

    # ── Left: flagged image ───────────────────────────────────────
    try:
        img = Image.open(flagged_path).convert("RGB")
        axes[0].imshow(img)
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Load error\n{e}",
                     ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=8)
        axes[0].set_facecolor("#FFE0E0")

    axes[0].set_title(
        f"⚠ FLAGGED\n"
        f"Declared: {declared_class}\n"
        f"Predicted: {predicted_class}\n"
        f"Confidence: {confidence} ({votes}/3)",
        fontsize=8, color="#C62828", fontweight="bold",
        pad=4
    )
    axes[0].axis("off")

    # Red border on flagged image
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#C62828")
        spine.set_linewidth(3)

    # ── Right: nearest neighbors from predicted class ─────────────
    for ni, (nbr_path, sim_score) in enumerate(neighbors):
        ax = axes[ni + 1]
        try:
            nbr_img = Image.open(nbr_path).convert("RGB")
            ax.imshow(nbr_img)
        except Exception:
            ax.text(0.5, 0.5, "Load error",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        ax.set_title(
            f"✓ {predicted_class.upper()}\n"
            f"Similarity: {sim_score:.3f}\n"
            f"{Path(nbr_path).name}",
            fontsize=7, color="#1B5E20", pad=4
        )
        ax.axis("off")

        # Green border on neighbors
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E7D32")
            spine.set_linewidth(2)

    # ── Overall title ─────────────────────────────────────────────
    fig.suptitle(
        f"[{idx+1}]  {Path(flagged_path).name}  |  "
        f"Declared: {declared_class}  →  Predicted: {predicted_class}",
        fontsize=9, fontweight="bold", y=1.02
    )

    plt.tight_layout(pad=0.5)
    return fig


# ─── VALIDATION LOOP ──────────────────────────────────────────────────────────

def validate(split=SPLIT, n=N_IMAGES):
    print(f"\nLoading {split} data...")
    mis_df, embeddings, labels, paths = load_data(split)

    # Only HIGH confidence — most reliable flags
    high_conf = mis_df[
        mis_df["confidence"] == "HIGH"
    ].head(n)

    print(f"HIGH-confidence mislabels found : {len(mis_df[mis_df['confidence']=='HIGH'])}")
    print(f"Reviewing first {len(high_conf)} images...")
    print(f"Output directory : {OUTPUT_DIR}")

    results = []

    for idx, (_, row) in enumerate(high_conf.iterrows()):
        print(f"\n  [{idx+1}/{len(high_conf)}] "
              f"{Path(row['file_path']).name}  "
              f"declared={row['declared_class']}  "
              f"predicted={row['nn_predicted']}")

        fig = make_validation_card(row, embeddings, labels, paths, idx)

        if fig is not None:
            out_path = OUTPUT_DIR / f"{split}_{idx+1:02d}_{row['declared_class']}_to_{row['nn_predicted']}.png"
            fig.savefig(out_path, dpi=120,
                        bbox_inches="tight",
                        facecolor="#F8F8F8")
            plt.close(fig)
            print(f"    Saved → {out_path.name}")
            results.append({
                "idx"            : idx + 1,
                "file_path"      : row["file_path"],
                "declared"       : row["declared_class"],
                "predicted"      : row["nn_predicted"],
                "votes"          : row["votes"],
                "confidence"     : row["confidence"],
                "output_file"    : str(out_path),
                "human_verdict"  : ""    # fill in manually
            })

    # ── Save validation log ───────────────────────────────────────
    results_df = pd.DataFrame(results)
    log_path   = OUTPUT_DIR / f"{split}_validation_log.csv"
    results_df.to_csv(log_path, index=False)

    print(f"\n{'='*60}")
    print(f"  {len(results)} validation cards saved to {OUTPUT_DIR}/")
    print(f"  Log saved → {log_path}")
    print(f"\n  NEXT STEP:")
    print(f"  Open each image in {OUTPUT_DIR}/")
    print(f"  For each: does the flagged image look like")
    print(f"  the PREDICTED class rather than its DECLARED class?")
    print(f"  Fill 'human_verdict' column in {log_path}")
    print(f"  TRUE  = correctly flagged (is mislabeled)")
    print(f"  FALSE = false positive (label is correct)")
    print(f"{'='*60}")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        choices=["train", "val"])
    parser.add_argument("--n",     default=20, type=int,
                        help="Number of HIGH-confidence images to review")
    args = parser.parse_args()

    df = validate(args.split, args.n)