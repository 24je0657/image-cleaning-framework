"""
Blur and noise detection service.
No HTTP imports — pure business logic.
"""
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL     import Image
from pathlib import Path
from typing  import Optional

from api.dependencies import get_autoencoder, ae_transform, DEVICE


class QualityService:

    def __init__(self, blur_threshold: float = 100.0):
        self.blur_threshold = blur_threshold

    # ── Blur ──────────────────────────────────────────────────────────────────

    def compute_blur_score(self, image_path: str) -> Optional[float]:
        try:
            img  = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4)
        except Exception:
            return None

    def detect_blur_batch(self, paths: np.ndarray) -> dict:
        t0      = time.time()
        scores  = []
        flagged = []

        for path in paths:
            score = self.compute_blur_score(str(path))
            if score is None:
                continue
            scores.append(score)
            if score < self.blur_threshold:
                flagged.append({"path": str(path), "blur_score": score})

        return {
            "module"        : "blur_detection",
            "flagged"       : len(flagged),
            "total"         : len(scores),
            "flag_pct"      : round(100*len(flagged)/max(len(scores),1), 2),
            "mean_score"    : round(float(np.mean(scores)), 2) if scores else 0,
            "threshold"     : self.blur_threshold,
            "duration_s"    : round(time.time() - t0, 2),
            "worst_images"  : sorted(flagged,
                                     key=lambda x: x["blur_score"])[:20],
        }

    # ── Noise ─────────────────────────────────────────────────────────────────

    def compute_reconstruction_error(self, image_path: str) -> Optional[float]:
        ae = get_autoencoder()
        if ae is None:
            return None
        try:
            img    = Image.open(image_path).convert("RGB")
            tensor = ae_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                recon = ae(tensor)
                error = nn.MSELoss()(recon, tensor).item()
            return round(error, 6)
        except Exception:
            return None

    def detect_noise_batch(self, paths: np.ndarray) -> dict:
        t0     = time.time()
        errors = []

        for path in paths:
            err = self.compute_reconstruction_error(str(path))
            if err is not None:
                errors.append({"path": str(path), "error": err})

        if not errors:
            return {"module": "noise_detection", "flagged": 0,
                    "total": len(paths), "flag_pct": 0.0,
                    "duration_s": round(time.time()-t0, 2),
                    "note": "Autoencoder not available"}

        mean_err  = float(np.mean([e["error"] for e in errors]))
        std_err   = float(np.std([e["error"] for e in errors]))
        threshold = mean_err + 2 * std_err
        flagged   = [e for e in errors if e["error"] > threshold]

        return {
            "module"       : "noise_detection",
            "flagged"      : len(flagged),
            "total"        : len(errors),
            "flag_pct"     : round(100*len(flagged)/max(len(errors),1), 2),
            "mean_error"   : round(mean_err, 6),
            "threshold"    : round(threshold, 6),
            "duration_s"   : round(time.time() - t0, 2),
            "worst_images" : sorted(flagged,
                                    key=lambda x: -x["error"])[:20],
        }