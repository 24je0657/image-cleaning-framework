# api/worker.py
"""
All heavy CPU work goes here.
These functions run in a ThreadPoolExecutor
so they never block the FastAPI event loop.
"""
import time
import numpy  as np
import pandas as pd
import torch
import cv2
import imagehash
from PIL        import Image
from pathlib    import Path
from sklearn.ensemble   import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from collections import Counter

from api.dependencies import (
    get_resnet, get_autoencoder,
    resnet_transform, ae_transform, DEVICE
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─── SINGLE IMAGE ANALYSIS ────────────────────────────────────────────────────

def analyze_single_image(image_path: str) -> dict:
    """
    Full pipeline on one image.
    Returns structured result dict.
    Called via run_in_executor — blocking is fine here.
    """
    path   = Path(image_path)
    result = {
        "file_path"            : image_path,
        "class_name"           : path.parent.name,
        "is_blurry"            : False,
        "blur_score"           : None,
        "is_noisy"             : False,
        "reconstruction_error" : None,
        "is_duplicate"         : False,
        "is_outlier"           : False,
        "is_mislabeled"        : False,
        "mislabel_confidence"  : None,
        "predicted_class"      : None,
        "priority_score"       : 0,
        "verdict"              : "CLEAN",
        "flags"                : []
    }

    try:
        img_pil = Image.open(path).convert("RGB")
        img_cv  = cv2.imread(str(path))
    except Exception as e:
        result["verdict"] = f"ERROR: {e}"
        return result

    # ── Blur score
    gray        = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur_score  = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4)
    result["blur_score"] = blur_score
    if blur_score < 100:
        result["is_blurry"] = True
        result["flags"].append("blurry")

    # ── Noise / reconstruction error
    ae = get_autoencoder()
    if ae is not None:
        tensor = ae_transform(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = ae(tensor)
            error = torch.nn.MSELoss()(recon, tensor).item()
        result["reconstruction_error"] = round(error, 6)
        # Threshold: load from precomputed stats or use fixed
        if error > 0.02:
            result["is_noisy"] = True
            result["flags"].append("noisy")

    # ── Embedding (for outlier + mislabel context)
    resnet = get_resnet()
    tensor = resnet_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(tensor).squeeze().cpu().numpy()

    # ── Priority score
    weights = {
        "blurry": 6, "noisy": 5, "duplicate": 10,
        "outlier": 4, "mislabeled": 7
    }
    result["priority_score"] = sum(
        weights.get(f, 0) for f in result["flags"]
    )

    # ── Verdict
    score = result["priority_score"]
    if score == 0:
        result["verdict"] = "CLEAN"
    elif score >= 10:
        result["verdict"] = "REMOVE — high priority"
    elif score >= 5:
        result["verdict"] = "REVIEW — medium priority"
    else:
        result["verdict"] = "REVIEW — low priority"

    return result


# ─── FULL PIPELINE (runs as background job) ───────────────────────────────────

def run_full_pipeline(
    job_id       : str,
    jobs         : dict,
    split        : str,
    config       : dict
):
    """
    Full pipeline run on an entire dataset split.
    Updates jobs[job_id] with progress as it goes.
    Designed to run in ThreadPoolExecutor.
    """
    import sys
    sys.path.insert(0, ".")   # allow src/ imports

    start_total = time.time()
    jobs[job_id]["status"]  = "running"
    jobs[job_id]["message"] = "Starting pipeline..."

    module_results = []

    def update(progress: int, message: str):
        jobs[job_id]["progress"] = progress
        jobs[job_id]["message"]  = message
        jobs[job_id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        # ── Module 1: Preprocessing
        update(5, "Running preprocessing validation...")
        t0 = time.time()
        from src.preprocessing import validate_dataset
        df_valid = validate_dataset("data/raw/Animal", split)
        invalid  = (~df_valid["valid"]).sum()
        module_results.append(ModuleResultData(
            "Preprocessing", "complete",
            int(invalid), len(df_valid), round(time.time()-t0, 2)
        ))

        # ── Module 2: Embeddings
        update(15, "Extracting ResNet50 embeddings...")
        t0 = time.time()
        from src.embeddings import extract_embeddings, load_embeddings
        extract_embeddings(split)
        embeddings, labels, paths = load_embeddings(split)
        module_results.append(ModuleResultData(
            "Embedding extractor", "complete",
            len(embeddings), len(embeddings), round(time.time()-t0, 2)
        ))

        # ── Module 3: Duplicates
        if config.get("run_duplicates", True):
            update(30, "Detecting duplicates...")
            t0 = time.time()
            from src.duplicates import find_exact_duplicates, find_near_duplicates
            df_exact, _ = find_exact_duplicates(paths, split)
            df_near,  _ = find_near_duplicates(embeddings, paths, split)
            n_dup = int(df_exact["exact_duplicate"].sum())
            n_near= len(df_near)
            module_results.append(ModuleResultData(
                "Duplicate detector", "complete",
                n_dup + n_near, len(paths), round(time.time()-t0, 2)
            ))

        # ── Module 4: Blur + Noise
        if config.get("run_blur", True):
            update(50, "Detecting blur and noise...")
            t0 = time.time()
            from src.quality import detect_blur, detect_noise
            df_blur  = detect_blur(split)
            df_noise = detect_noise(split)
            n_blur   = int(df_blur["is_blurry"].sum())
            n_noise  = int(df_noise["is_noisy"].sum())
            module_results.append(ModuleResultData(
                "Quality detector", "complete",
                n_blur + n_noise, len(df_blur), round(time.time()-t0, 2)
            ))

        # ── Module 5: Outliers
        if config.get("run_outliers", True):
            update(68, "Running Isolation Forest...")
            t0 = time.time()
            from src.outliers import detect_outliers
            df_out = detect_outliers(split)
            n_out  = int(df_out["is_outlier"].sum())
            module_results.append(ModuleResultData(
                "Outlier detector", "complete",
                n_out, len(df_out), round(time.time()-t0, 2)
            ))

        # ── Module 6: Mislabels
        if config.get("run_mislabels", True):
            update(82, "Detecting mislabeled images...")
            t0 = time.time()
            from src.mislabel import detect_mislabels
            df_mis = detect_mislabels(split)
            n_mis  = int(df_mis["is_mislabeled"].sum())
            module_results.append(ModuleResultData(
                "Mislabel detector", "complete",
                n_mis, len(df_mis), round(time.time()-t0, 2)
            ))

        # ── Module 7: Decision Engine
        update(93, "Running decision engine...")
        t0 = time.time()
        from src.decision_engine import build_master_report
        df_master = build_master_report(split)
        module_results.append(ModuleResultData(
            "Decision engine", "complete",
            int(df_master["verdict"].str.startswith("REMOVE").sum()),
            len(df_master), round(time.time()-t0, 2)
        ))

        # ── Done
        duration = round(time.time() - start_total, 2)
        total    = len(df_master)
        clean    = int((df_master["verdict"] == "CLEAN").sum())
        remove   = int(df_master["verdict"].str.startswith("REMOVE").sum())
        review   = int(df_master["verdict"].str.startswith("REVIEW").sum())

        jobs[job_id]["status"]   = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Pipeline complete"
        jobs[job_id]["result"]   = {
            "split"          : split,
            "total_images"   : total,
            "clean"          : clean,
            "to_remove"      : remove,
            "to_review"      : review,
            "duration_s"     : duration,
            "report_path"    : f"reports/{split}_master_report.csv",
            "modules"        : [m.__dict__ for m in module_results]
        }

    except Exception as e:
        jobs[job_id]["status"]  = "failed"
        jobs[job_id]["error"]   = str(e)
        jobs[job_id]["message"] = f"Pipeline failed: {e}"


class ModuleResultData:
    def __init__(self, module, status, flagged, total, duration):
        self.module     = module
        self.status     = status
        self.flagged    = flagged
        self.total      = total
        self.flag_pct   = round(100 * flagged / max(total, 1), 2)
        self.duration_s = duration