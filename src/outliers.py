import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----- Configuration -----
EMBED_DIR = "embeddings"
REPORTS_DIR = "reports"
PCA_COMPONENTS = 128  # Reduce embeddings to 128 dimensions
CONTAMINATION = 0.05  # Expect 5% of the data to be outliers
RANDOM_STATE = 42  # For reproducibility

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_embeddings(split = "train"):
    embeddings = np.load(f"{EMBED_DIR}/{split}_embeddings.npy")
    labels = np.load(f"{EMBED_DIR}/{split}_labels.npy", allow_pickle=True)
    paths = np.load(f"{EMBED_DIR}/{split}_file_paths.npy", allow_pickle=True)

    print(f"Loaded {split} embeddings: {embeddings.shape}")
    return embeddings, labels, paths


# ----- PCA Dimensionality Reduction -----
def Reduce_Dimensions(embeddings, n_components = PCA_COMPONENTS):
    """Standardize => PCA
    Returns Reduced Embeddings + fitted PCA object
    """
    print(f"Reducing dimensions {embeddings.shape[1]} => {n_components} using PCA...")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_.sum() * 100

    print(f" Variance explained by {n_components} components: {explained:.2f}%")
    print(f" Reduced embeddings shape: {reduced.shape}")

    return reduced, pca, scaler


# ----- Isolation Forest -----

def run_isolation_forest(reduced_embeddings, contamination = CONTAMINATION):
    """Fit Isolation Forest on PCA-reduced embeddings
      and return outlier scores and predictions."""
    
    print(f" fitting Isolation Forest with contamination={contamination}...")
    clf = IsolationForest(n_estimators = 200,
                          contamination = contamination,
                          random_state = RANDOM_STATE,
                          n_jobs = -1
                          )
    predictions = clf.fit_predict(reduced_embeddings)
    scores = clf.decision_function(reduced_embeddings)

    n_outliers = np.sum(predictions == -1)

    print(f" Outliers detected: {n_outliers} / {len(reduced_embeddings)} ({n_outliers/len(reduced_embeddings)*100:.2f}%)")

    return predictions, scores, clf


# ----- Per class Isolation Forest -----
def run_per_class_isolation_forest(embeddings, labels, paths, contamination = CONTAMINATION):
    """Run Isolation forest Seperately per class
    This catches Outliers within each class
    """

    print(f" \n Per class Isolation Forest ------------- ")
    all_records = []

    for class_name in np.unique(labels):
        mask = labels == class_name
        class_emb = embeddings[mask]
        class_paths = paths[mask]

        n_comp = min(PCA_COMPONENTS, len(class_emb) - 1) # Ensure we don't exceed number of samples
        scaler = StandardScaler()
        scaled = scaler.fit_transform(class_emb)
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        reduced = pca.fit_transform(scaled)

        clf = IsolationForest(n_estimators=200,
                                contamination=contamination,
                                random_state=RANDOM_STATE,
                                n_jobs=-1)
        predictions = clf.fit_predict(reduced)
        scores = clf.decision_function(reduced)

        n_outliers = np.sum(predictions == -1)
        print(f" [{class_name}] {len(class_emb)} images => {n_outliers} outliers")

        for path, pred, score in zip(class_paths, predictions, scores):
            all_records.append({

                "file_path":path,
                "class":class_name,
                "per_class_outlier": pred == -1,
                "per_class_score": round(float(score), 6)

            })

    return pd.DataFrame(all_records)

# ----- Visualization -----
def plot_outliers(reduced, predictions, labels, split = "train"):
    """Project to 2D with PCA for Visualization 
    color by class and mark outliers with red 'x'"""

    print("\n Generating Outlier Scatter Plot...")

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca_2d.fit_transform(reduced)
    colors = {"cat":"blue", "dog":"orange", "elephant":"green", 
              "horse":"purple", "lion":"brown"}
    
    fig, ax = plt.subplots(figsize=(12, 8))

    for class_name in np.unique(labels):
        mask = labels == class_name
        ax.scatter(
            coords_2d[mask, 0], 
            coords_2d[mask, 1], 
            c=colors.get(class_name, "gray"),
            label=class_name,
            alpha=0.6,
            s = 10
        )

    # Mark outliers
    outlier_mask = predictions == -1
    ax.scatter(
        coords_2d[outlier_mask, 0], 
        coords_2d[outlier_mask, 1], 
        c="red", 
        marker="x", 
        label="Outlier",
        s = 40,
        zorder = 5
    )

    ax.set_title(f"Outlier Detection Scatter Plot ({split} set)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend(markerscale=2)
    plt.tight_layout()

    out_path = f"{REPORTS_DIR}/{split}_outlier_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f" Scatter plot saved to {out_path}")


# ---- Main Function -----
def detect_outliers(split = "train"):

    print(f"\n{'='*30}")
    print(f"Outlier Detection for {split} set")
    print(f"{'='*30}")

    # ----- Load Embeddings -----
    embeddings, labels, paths = load_embeddings(split)

    # ----- Global Isolation Forest -----
    reduced, pca, scaler = Reduce_Dimensions(embeddings)
    predictions, scores, clf = run_isolation_forest(reduced)

    # ----- Per Class Isolation Forest -----
    df_per_class = run_per_class_isolation_forest(embeddings, labels, paths)

    # ----- Build Global Report -----
    df_global = pd.DataFrame({
        "file_path": paths,
        "class": labels,
        "global_outlier": predictions == -1,
        "global_score": [round(float(score), 6) for score in scores]
    })

    # ----- Merge Global and Per Class Reports -----
    df_final = df_global.merge(
        df_per_class[["file_path", "per_class_outlier", "per_class_score"]],
        on="file_path",
        how="left"
    )

    # ----- Combined Flag Outlier if either method flags it -----

    df_final["is_outlier"] = (
        df_final["global_outlier"] | df_final["per_class_outlier"]
    )

    # ----- Summary Statistics -----
    print(f" \n Overall Summary ------------- ")

    summary = df_final.groupby("class").agg(
        total_images = ("file_path", "count"),
        outliers = ("is_outlier", "sum")
    ).reset_index()

    summary["outlier_%"] = ((summary["outliers"] / summary["total_images"]) * 100) .round(2)

    print(summary.to_string(index=False))

    # ----- Top 10 Most Anamalous -----
    print(f" \n Top 10 Most Anomalous Images ------------- ")
    print(df_final.nsmallest(10, "global_score")[
        ["file_path", "class", "global_score", "per_class_score"]].to_string(index=False))
    
    # ----- Visualization -----
    plot_outliers(reduced, predictions, labels, split)

    # ----- Save Final Report -----
    report_path = f"{REPORTS_DIR}/{split}_outlier_report.csv"
    df_final.to_csv(report_path, index=False)
    print(f" Final report saved to {report_path}")

    return df_final

if __name__ == "__main__":
    for split in ["train", "val"]:
        detect_outliers(split)





        







