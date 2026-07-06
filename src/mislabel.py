import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# -------- Configuration -----
EMBED_DIR = "embeddings"
REPORTS_DIR = "reports"
N_CLASSES = 5  # Number of classes in the dataset
K_NEIGHBORS = 11  # Number of neighbors for KNN
PCA_COMPONENTS = 128  # Reduce embeddings to 128 dimensions
RANDOM_STATE = 42  # For reproducibility
K_MEANS = 25

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_embeddings(split = "train"):
    embeddings = np.load(f"{EMBED_DIR}/{split}_embeddings.npy")
    labels = np.load(f"{EMBED_DIR}/{split}_labels.npy", allow_pickle=True)
    paths = np.load(f"{EMBED_DIR}/{split}_file_paths.npy", allow_pickle=True)

    print(f"Loaded {split} embeddings: {embeddings.shape}")
    return embeddings, labels, paths

def reduce_dimensions(embeddings):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_.sum() * 100

    print(f" PCA: {embeddings.shape[1]} -> {PCA_COMPONENTS} dims " 
          f"explained variance: {explained:.2f}%")

    return reduced

# ----- Kmeans Clustering -----
def method1_KMeans(reduced, labels):

    """Cluster All Embeddings to K clusters 
    Map each Cluster into its dominant class
    Image flagged if declared class != cluster dominant class
    """

    print(f"\n Method 1 : Kmeans Clustering ------------")

    kmeans = KMeans(n_clusters = K_MEANS, 
                    random_state=RANDOM_STATE, n_init=10)
    
    cluster_ids = kmeans.fit_predict(reduced)

    # ----- Map each cluster to its dominant class -----
    cluster_to_class = {}

    for cid in range(K_MEANS):
        mask = cluster_ids == cid
        class_counts = Counter(labels[mask])
        dominant = class_counts.most_common(1)[0][0]
        cluster_to_class[cid] = dominant

        class_counts = {str(k): v for k, v in Counter(labels[mask]).items()}
        print(f"Cluster {cid} -> '{dominant}': {class_counts}")

        

    predicted = np.array([cluster_to_class[c] for c in cluster_ids])
    km_flag = predicted != labels

    
    print(f" \n KMean suspects : {km_flag.sum()} Images")

    return km_flag, predicted

# ----- Nearest Neighbour Voting -----
def method2_nearest_neighbors(reduced, labels):

    """Find K nearest Neighbours using PCA reduced embeddings
    L2 Normalize 1st -> Cosine similarity via euclidian distance"""

    print(f" \n Method 2 : Nearest Neighbor voting ---------")
    normed = normalize(reduced, norm = "l2")

    nn  = NearestNeighbors(
        n_neighbors = K_NEIGHBORS,
        metric = "euclidean",
        n_jobs = -1
    )

    nn.fit(normed)
    distance, indices = nn.kneighbors(normed)

    nn_flag = np.zeros(len(labels), dtype = bool)
    nn_voted_cls = []

    for i in range(len(labels)):
        neighbor_labels = labels[indices[i][1:]]
        vote_counts = Counter(neighbor_labels)
        top_vote = vote_counts.most_common(1)[0][0]
        own_class_votes = vote_counts.get(labels[i],0)

        nn_voted_cls.append(top_vote)

        # Flag if own class < 40 % of neighbors 
        if own_class_votes < len(neighbor_labels) * 0.4 :
            nn_flag[i] = True
    
    print(f" NN Suspects {nn_flag.sum()} Images ")
    return nn_flag, np.array(nn_voted_cls)


# ----- Method 3 Centroid Distance -----
def method3_centroid_distance(reduced,labels):

    """ Compute Mean Embeddding (Centroid) per class
     If image is closer to a different class centroid then it own => Suspicious"""
    
    print(f" \n Method 3 Centroid Distance ----------")
    normed = normalize(reduced, norm = "l2")
    classes = np.unique(labels)
    centroids = {
        cls : normed[labels == cls].mean(axis=0)
        for cls in classes
    }

    cd_flag = np.zeros(len(labels),dtype = bool)
    cd_voted_cls = []

    for i in range(len(labels)):
        vec = normed[i]
        dists = {
            cls : np.linalg.norm(vec - centroids[cls])
            for cls in classes
        }

        closest = min(dists, key = dists.get)
        cd_voted_cls.append(closest)

        if closest != labels[i]:
            cd_flag[i] = True
        
    print(f" Centroid Suspects : {cd_flag.sum()} Images")
    return cd_flag, np.array(cd_voted_cls)


# ----- Confusion Matrix Visualisation -----
def plot_confusion_matrix(labels, nn_pred, split = "train"):
    """
    Plot declared class vs NN-predicted class as a heatmap.
    Off-diagonal cells = potential mislabels.
    """

    classes = sorted(np.unique(labels))
    matrix  = np.zeros((len(classes), len(classes)), dtype=int)

    cls_idx = {c: i for i, c in enumerate(classes)}

    for true, pred in zip(labels, nn_pred):
        matrix[cls_idx[true]][cls_idx[pred]] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot      = True,
        fmt        = "d",
        xticklabels= classes,
        yticklabels= classes,
        cmap       = "Blues",
        ax         = ax
    )
    ax.set_xlabel("Predicted class (NN voting)")
    ax.set_ylabel("Declared class (folder label)")
    ax.set_title(f"Label mismatch heatmap — {split} set\n"
                 f"(Off-diagonal = mislabel candidates)")
    plt.tight_layout()

    out_path = f"{REPORTS_DIR}/{split}_mislabel_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved → {out_path}")


# ----- Main detect function -----
def detect_mislabels(split = "train"):
    print(f"\n{'='*30}")
    print(f"MISLABEL DETECTION — {split.upper()} SET")
    print(f"{'='*30}")

    embeddings, labels, paths = load_embeddings(split)
    reduced = reduce_dimensions(embeddings)

    # ----- Run all Three methods -----
    km_flag, km_pred = method1_KMeans(reduced, labels)
    nn_flag, nn_pred = method2_nearest_neighbors(reduced, labels)
    cd_flag, cd_pred = method3_centroid_distance(reduced, labels)


    # voting system
    votes = km_flag.astype(int) + nn_flag.astype(int) + cd_flag.astype(int)

    def confidence(v):
        if v == 3: return "HIGH"
        if v == 2: return "MEDIUM"
        if v == 1: return "LOW"
        return "CLEAN"
    
    confidence_arr = np.array([confidence(v) for v in votes])
    is_mislabeled  = votes >= 2   # flag MEDIUM + HIGH only

    # Build Report
    df = pd.DataFrame({
        "file_path"          : paths,
        "declared_class"     : labels,
        "kmeans_predicted"   : km_pred,
        "nn_predicted"       : nn_pred,
        "centroid_predicted" : cd_pred,
        "kmeans_flag"        : km_flag,
        "nn_flag"            : nn_flag,
        "centroid_flag"      : cd_flag,
        "votes"              : votes,
        "confidence"         : confidence_arr,
        "is_mislabeled"      : is_mislabeled
    })

    # ----- Overall Summary -----
    print(f"\n OVERALL SUMMARY ------------------")
    print(f"  Total images        : {len(df)}")
    print(f"  HIGH confidence     : {(confidence_arr == 'HIGH').sum()}")
    print(f"  MEDIUM confidence   : {(confidence_arr == 'MEDIUM').sum()}")
    print(f"  LOW  confidence     : {(confidence_arr == 'LOW').sum()}")
    print(f"  Total flagged (≥2)  : {is_mislabeled.sum()} "
          f"({100*is_mislabeled.mean():.2f}%)")
    
    # ----- Per class breakdown -----
    print(f" \n Per class Breakdown ---------------")
    for cls in sorted(np.unique(labels)):
        mask  = labels == cls
        flagd = is_mislabeled[mask].sum()
        total = mask.sum()
        print(f"  [{cls:10s}] {flagd:4d} / {total}  ({100*flagd/total:.1f}%)")

    # ----- Top 10 suspicious -----
    print(f"\n TOP 10 MOST SUSPICIOUS -----------------")
    suspicious = df[df["is_mislabeled"]].nlargest(10, "votes")
    print(suspicious[[
        "file_path", "declared_class",
        "nn_predicted", "votes", "confidence"
    ]].to_string(index=False))

    # ── Visualize confusion matrix
    plot_confusion_matrix(labels, nn_pred, split)

    # ── Save report
    out_path = f"{REPORTS_DIR}/{split}_mislabel_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Report saved → {out_path}")

    return df


if  __name__ == "__main__":
    for split in ["train","val"]:
        detect_mislabels(split)




















