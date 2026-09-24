"""
Mislabel detection service — 3-method voting.
"""
import time
import numpy as np
from sklearn.cluster       import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors     import NearestNeighbors
from collections           import Counter


class MislabelService:

    def __init__(self,
                 k_kmeans     : int = 25,
                 k_neighbors  : int = 11,
                 pca_dims     : int = 128,
                 vote_threshold: int = 2):
        self.k_kmeans      = k_kmeans
        self.k_neighbors   = k_neighbors
        self.pca_dims      = pca_dims
        self.vote_threshold = vote_threshold

    def _reduce(self, embeddings: np.ndarray) -> np.ndarray:
        scaled = StandardScaler().fit_transform(embeddings)
        n_comp = min(self.pca_dims, embeddings.shape[0] - 1)
        return PCA(n_components=n_comp,
                   random_state=42).fit_transform(scaled)

    def _kmeans(self, reduced, labels):
        km      = KMeans(n_clusters=self.k_kmeans,
                         random_state=42, n_init=10)
        cids    = km.fit_predict(reduced)
        cmap    = {
            c: Counter(labels[cids==c]).most_common(1)[0][0]
            for c in range(self.k_kmeans)
            if (cids==c).sum() > 0
        }
        pred    = np.array([cmap.get(c, "") for c in cids])
        return pred != labels

    def _knn(self, reduced, labels):
        normed = normalize(reduced, norm="l2")
        nn     = NearestNeighbors(n_neighbors=self.k_neighbors,
                                   metric="euclidean", n_jobs=-1)
        nn.fit(normed)
        _, idxs = nn.kneighbors(normed)
        flags   = np.zeros(len(labels), dtype=bool)
        preds   = []
        for i, neighbors in enumerate(idxs):
            nbr_labels = labels[neighbors[1:]]
            vote       = Counter(nbr_labels).most_common(1)[0][0]
            preds.append(vote)
            own_votes  = (nbr_labels == labels[i]).sum()
            if own_votes < len(nbr_labels) * 0.4:
                flags[i] = True
        return flags, np.array(preds)

    def _centroid(self, reduced, labels):
        normed    = normalize(reduced, norm="l2")
        centroids = {
            cls: normed[labels==cls].mean(axis=0)
            for cls in np.unique(labels)
        }
        flags = np.zeros(len(labels), dtype=bool)
        preds = []
        for i, vec in enumerate(normed):
            closest = min(centroids,
                          key=lambda c: np.linalg.norm(vec - centroids[c]))
            preds.append(closest)
            if closest != labels[i]:
                flags[i] = True
        return flags, np.array(preds)

    def detect(self,
               embeddings : np.ndarray,
               labels     : np.ndarray,
               paths      : np.ndarray) -> dict:
        t0      = time.time()
        reduced = self._reduce(embeddings)

        km_flag             = self._kmeans(reduced, labels)
        nn_flag, nn_preds   = self._knn(reduced, labels)
        cd_flag, _          = self._centroid(reduced, labels)

        votes     = km_flag.astype(int) + nn_flag.astype(int) + cd_flag.astype(int)
        flagged   = votes >= self.vote_threshold

        def conf(v):
            return "HIGH" if v==3 else "MEDIUM" if v==2 else "LOW"

        # Per-class breakdown
        per_class = {}
        for cls in np.unique(labels):
            mask = labels == cls
            n_fl = int(flagged[mask].sum())
            per_class[cls] = {
                "total"      : int(mask.sum()),
                "mislabeled" : n_fl,
                "pct"        : round(100 * n_fl / int(mask.sum()), 2)
            }

        suspicious = [
            {
                "path"       : str(paths[i]),
                "declared"   : str(labels[i]),
                "predicted"  : str(nn_preds[i]),
                "votes"      : int(votes[i]),
                "confidence" : conf(votes[i])
            }
            for i in np.where(flagged)[0]
        ]
        suspicious.sort(key=lambda x: -x["votes"])

        return {
            "module"       : "mislabel_detection",
            "flagged"      : int(flagged.sum()),
            "total"        : len(labels),
            "flag_pct"     : round(100*flagged.sum()/len(labels), 2),
            "high"         : int((votes==3).sum()),
            "medium"       : int((votes==2).sum()),
            "low"          : int((votes==1).sum()),
            "per_class"    : per_class,
            "duration_s"   : round(time.time() - t0, 2),
            "worst_images" : suspicious[:20],
        }