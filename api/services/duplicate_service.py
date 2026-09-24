"""
Duplicate detection service.
Wraps src/duplicates.py for API use.
Returns structured dicts — no HTTP, no FastAPI imports here.
"""
import time
import numpy as np
import pandas as pd
import imagehash
from PIL     import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateService:

    def __init__(self, cosine_threshold: float = 0.97, hash_threshold: int = 5):
        self.cosine_threshold = cosine_threshold
        self.hash_threshold   = hash_threshold

    def find_exact_duplicates(self, paths: np.ndarray) -> dict:
        t0       = time.time()
        hash_map = {}

        for path in paths:
            try:
                h = str(imagehash.phash(Image.open(path).convert("RGB")))
                hash_map.setdefault(h, []).append(str(path))
            except Exception:
                continue

        dup_groups = {h: ps for h, ps in hash_map.items() if len(ps) > 1}
        flagged    = sum(len(v) - 1 for v in dup_groups.values())

        return {
            "module"       : "exact_duplicates",
            "flagged"      : flagged,
            "total"        : len(paths),
            "flag_pct"     : round(100 * flagged / max(len(paths), 1), 2),
            "duration_s"   : round(time.time() - t0, 2),
            "groups"       : len(dup_groups),
        }

    def find_near_duplicates(self,
                             embeddings : np.ndarray,
                             paths      : np.ndarray) -> dict:
        t0      = time.time()
        n       = len(embeddings)
        flagged = set()
        pairs   = []
        CHUNK   = 500

        for start in range(0, n, CHUNK):
            end    = min(start + CHUNK, n)
            chunk  = embeddings[start:end]
            sim    = cosine_similarity(chunk, embeddings)

            for i, row in enumerate(sim):
                gi = start + i
                for j, score in enumerate(row):
                    if j <= gi:
                        continue
                    if score >= self.cosine_threshold and gi not in flagged:
                        flagged.add(j)
                        pairs.append({
                            "image_1"      : str(paths[gi]),
                            "image_2"      : str(paths[j]),
                            "cosine_score" : round(float(score), 4)
                        })

        return {
            "module"     : "near_duplicates",
            "flagged"    : len(flagged),
            "total"      : n,
            "flag_pct"   : round(100 * len(flagged) / max(n, 1), 2),
            "duration_s" : round(time.time() - t0, 2),
            "pairs"      : pairs[:50],   # cap at 50 pairs in response
        }