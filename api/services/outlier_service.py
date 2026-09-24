"""
Isolation Forest outlier detection service.
"""
import time
import numpy as np
from sklearn.ensemble      import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class OutlierService:

    def __init__(self,
                 contamination : float = 0.05,
                 pca_dims      : int   = 128,
                 n_estimators  : int   = 200):
        self.contamination = contamination
        self.pca_dims      = pca_dims
        self.n_estimators  = n_estimators

    def _reduce(self, embeddings: np.ndarray) -> np.ndarray:
        scaled  = StandardScaler().fit_transform(embeddings)
        n_comp  = min(self.pca_dims, embeddings.shape[0] - 1)
        reduced = PCA(n_components=n_comp,
                      random_state=42).fit_transform(scaled)
        return reduced

    def detect(self,
               embeddings : np.ndarray,
               labels     : np.ndarray,
               paths      : np.ndarray) -> dict:
        t0      = time.time()
        reduced = self._reduce(embeddings)

        clf     = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            random_state  = 42,
            n_jobs        = -1
        )
        preds  = clf.fit_predict(reduced)
        scores = clf.decision_function(reduced)

        outlier_mask  = preds == -1
        outlier_paths = paths[outlier_mask].tolist()

        # Per-class breakdown
        per_class = {}
        for cls in np.unique(labels):
            mask    = labels == cls
            n_out   = int((preds[mask] == -1).sum())
            per_class[cls] = {
                "total"    : int(mask.sum()),
                "outliers" : n_out,
                "pct"      : round(100 * n_out / int(mask.sum()), 2)
            }

        return {
            "module"        : "outlier_detection",
            "flagged"       : int(outlier_mask.sum()),
            "total"         : len(embeddings),
            "flag_pct"      : round(100*outlier_mask.sum()/len(embeddings), 2),
            "contamination" : self.contamination,
            "pca_dims"      : self.pca_dims,
            "per_class"     : per_class,
            "duration_s"    : round(time.time() - t0, 2),
            "worst_images"  : [
                {"path": str(p), "score": round(float(s), 6)}
                for p, s in sorted(
                    zip(paths[outlier_mask], scores[outlier_mask]),
                    key=lambda x: x[1]
                )[:20]
            ]
        }