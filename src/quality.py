import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ==============================
# AutoEncoder (PyTorh) 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as image
# ==============================



# ----- Configuration -----
ROOT_DIR = "data/raw/Animal"
REPORTS_DIR = "reports"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BLUR_THRESHOLD = 100.0 # Below this it is Blurry (tune if needed)
os.makedirs(REPORTS_DIR, exist_ok = True)
CONTAMINATION_LIMIT = 0.30


# =============================
# AutoEncoder Configuration
MODEL_PATH = "models/autoencoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_IMG_SIZE = 128  # Resize images to 128x128 for the autoencoder
NOISE_THRESHOLD = None # Set dynamically from mean + 2*std after first pass
ae_transform = transforms.Compose([
    transforms.Resize((AE_IMG_SIZE, AE_IMG_SIZE)),
    transforms.ToTensor(),
])
# =============================




# ----- CORE BLUR SCORE FUNCTION -----
def compute_blur_score(image_path):

    """Compute Laplacian Variance of an image
    Higher Score => Sharp
    Lower Score => Blurry.
    Returns Float Score or None if Image can't be read ."""

    try:
        img = cv2.imread(str(image_path))
        if img is None :
            return None
        
        # ----- Convert to gray scale as bluir detection no need color -----
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ----- Apply Laplacian and compute variance -----
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()
        return round(score,4)
    
    except Exception as e :
        print(f" Error Processing Image {image_path}: {e}")
        return None
    

# ----- Dataset-Level Blur Detection -----
# ----- Dataset-Level Blur Detection -----
def detect_blur(split="train", use_adaptive=True, percentile=5):
    """
    Two-pass blur detection with contamination guard.

    Pass 1:
        Compute blur scores for all images.

    Pass 2:
        If adaptive mode is enabled:
            - Estimate contamination per class using
              the global BLUR_THRESHOLD.
            - If contamination > 30%, use global threshold.
            - Otherwise, use class-specific percentile threshold.

        If adaptive mode is disabled:
            - Use global BLUR_THRESHOLD for all images.

    This prevents an overly blurry class from causing its
    adaptive threshold to become too lenient.
    """
    split_path = Path(ROOT_DIR) / split
    records    = []

    classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
    print(f"\n{'='*30}")
    print(f"Blur detection - {split.upper()} Set")
    print(f"{'='*30}")
    print(f"Classes   - {classes}")
    print(f"Mode      - {'Adaptive(two pass) per-class' if use_adaptive else 'Global'}")
    if not use_adaptive:
        print(f"Threshold - {BLUR_THRESHOLD}")

    # ── First pass: collect all Blur scores
    for class_name in classes:
        class_dir = split_path / class_name
        img_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS  
        ])

        print(f"Processing {class_name} - {len(img_files)} images")

        for idx, img_path in enumerate(img_files):
            score = compute_blur_score(img_path)
            if score is None:
                continue

            records.append({
                "file_path"  : str(img_path),
                "class"      : class_name,
                "blur_score" : score,
            })

            if (idx + 1) % 500 == 0:
                print(f"  {idx+1} / {len(img_files)} done...")

    df = pd.DataFrame(records)

    # ── Second pass: Choose threshold
    if use_adaptive:
        
        class_thresholds = {}
        threshold_methods = {}
        print(
            f"\n── Contamination Check "
            f"(global threshold={BLUR_THRESHOLD}) ──"
        )

        for cls in df["class"].unique():

            cls_scores = df[
                df["class"] == cls
            ]["blur_score"]

            # Estimate how much of this class falls below
            # the global blur threshold.
            contamination_rate = (
                cls_scores < BLUR_THRESHOLD
            ).mean()

            if contamination_rate > CONTAMINATION_LIMIT:

                # Class is potentially contaminated.
                # Do not allow the adaptive threshold to
                # become too lenient.
                threshold = BLUR_THRESHOLD

                threshold_methods[cls] = "global_fallback"

                print(
                    f"  [{cls:10s}] "
                    f"contamination="
                    f"{contamination_rate:.0%} "
                    f"> {CONTAMINATION_LIMIT:.0%} "
                    f"→ GLOBAL fallback "
                    f"({threshold})"
                )
            
            else:

                # Class is sufficiently clean.
                # Adaptive percentile threshold is safe.
                threshold = cls_scores.quantile(
                    percentile / 100
                )

                threshold_methods[cls] = (
                    "adaptive_percentile"
                )

                print(
                    f"  [{cls:10s}] "
                    f"contamination="
                    f"{contamination_rate:.0%} "
                    f"≤ {CONTAMINATION_LIMIT:.0%} "
                    f"→ adaptive p{percentile} "
                    f"({threshold:.1f})"
                )
            class_thresholds[cls] = round(
                threshold,
                2
            )

        df["threshold_used"] = df[
            "class"
        ].map(class_thresholds)

        df["threshold_method"] = df[
            "class"
        ].map(threshold_methods)

    else:

        df["threshold_used"] = BLUR_THRESHOLD
        df["threshold_method"] = "global"
        # ============================================================
    # Final blur decision
    # ============================================================

    df["is_blurry"] = (
        df["blur_score"]
        < df["threshold_used"]
    )

    # ============================================================
    # Per-class statistics
    # ============================================================

    print(
        f"\n----- Per Class Blur Statistics -----"
    )

    stats = df.groupby("class").agg(
        total=("blur_score", "count"),
        blurry_count=("is_blurry", "sum"),
        mean_score=("blur_score", "mean"),
        min_score=("blur_score", "min"),
        max_score=("blur_score", "max"),
        threshold=("threshold_used", "first"),
        method=("threshold_method", "first")
    ).reset_index()

    stats["blurry_%"] = (
        stats["blurry_count"]
        / stats["total"]
        * 100
    ).round(2)

    print(
        stats.to_string(index=False)
    )

    # ============================================================
    # Overall summary
    # ============================================================

    total_blurry = df["is_blurry"].sum()

    print(
        f"\n----- Overall Summary -----"
    )

    print(
        f" Total images  : {len(df)}"
    )

    print(
        f" Blurry images : "
        f"{total_blurry} "
        f"({100 * total_blurry / len(df):.2f}%)"
    )

    print(
        f" Mean Score    : "
        f"{df['blur_score'].mean():.2f}"
    )

    print(
        f" Min Score     : "
        f"{df['blur_score'].min():.2f}"
    )

    print(
        f" Max Score     : "
        f"{df['blur_score'].max():.2f}"
    )


    # ── Save report
    out_path = f"{REPORTS_DIR}/{split}_blur_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Report saved → {out_path}")

    return df




# =============================
# AutoEncoder Model Definition

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
# =============================

# =============================
# Load Trained AutoEncoder Model
def load_autoencoder():
    """Load the trained autoencoder model from inference."""
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f" Autoencoder model loaded Sucessfully from {MODEL_PATH} !")
    return model

# ============================


# =============================
# Reconstruction Error Function

def compute_reconstruction_error(model, image_path):
    """ Compute MSE b/w Original and Reconstructed image using AutoEncoder.
    High Error => Image is Noisy/Anamolous"""

    try:
        img = image.open(image_path).convert("RGB")
        img_tensor = ae_transform(img).unsqueeze(0).to(DEVICE) # (1,3,128,128)

        with torch.no_grad():
            recon = model(img_tensor)
            error = nn.MSELoss()(recon, img_tensor).item()
        return error
    
    except Exception as e:
        print(f" Error Processing Image {image_path}: {e}")
        return None
    
# ===========================


# ===========================
# Dataset Level Noise Detection using AutoEncoder

def detect_noise(split = "train"):

    """ Detect Noisy Images using AutoEncoder Reconstruction Error."""

    split_path = Path(ROOT_DIR)/split
    model = load_autoencoder()
    records = []

    classes = sorted(
        [d.name for d in split_path.iterdir() if d.is_dir()]
    )

    print(f"\n{'='*30}")
    print(f"Noise detection - {split.upper()} Set")
    print(f"{'='*30}")

    for class_name in classes:
        class_dir = split_path / class_name
        img_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file and f.suffix.lower() in VALID_EXTENSIONS
        ])

        print(f"Processing {class_name}- {len(img_files)} images")

        for idx, img_path in enumerate(img_files):
            error = compute_reconstruction_error(model, img_path)
            if error is None:
                continue

            records.append({
                "file_path" : str(img_path),
                "class" : class_name,
                "reconstruction_error" : error
            })

            if(idx+1) % 500 == 0:
                print(f" {idx+1} / {len(img_files)} done...")

    df = pd.DataFrame(records)

    #Dynamically set threshold as mean + 2*std
    mean_error = df["reconstruction_error"].mean()
    std_error = df["reconstruction_error"].std()
    threshold = mean_error + 2*std_error

    df["is_noisy"] = df["reconstruction_error"] > threshold

    # per-class statistics
    print(f"\n ----- Per class Noise Statistics -----")
    stats = df.groupby("class").agg(
        total = ("reconstruction_error","count"),
        noisy_count = ("is_noisy", "sum"),
        mean_error = ("reconstruction_error", "mean"),
    ).reset_index()

    stats["noisy_%"] = (stats["noisy_count"] / stats["total"] *100).round(2)
    print(stats.to_string(index = False))

    # Overall Summary
    print(f"\n ----- Overall Summary -----")
    print(f" Total images :{len(df)}")
    print(f" Mean Reconstruction Error : {mean_error:.4f}")
    print(f" Std Reconstruction Error : {std_error:.4f}")
    print(f" Noise Threshold : {threshold:.4f}")
    print(f" Noisy images :{df['is_noisy'].sum()} ({100 * df['is_noisy'].sum() / len(df):.2f}%)")

    # Save Report
    out_path = f"{REPORTS_DIR}/{split}_noise_report.csv"
    df.to_csv(out_path, index = False)
    print(f" Noise report saved to {out_path}")

    return df

# ===========================
# ===========================
# Noise Refinement
# ===========================

def refine_noise_flags(split="train"):
    """
    Refine AutoEncoder noise flags using independent blur and
    outlier evidence.

    Decision logic:

    1. High reconstruction error + blurry
       -> CONFIRMED_NOISY

    2. High reconstruction error + borderline blur
       -> REVIEW_NOISE

    3. High reconstruction error + sharp + outlier
       -> REVIEW_OOD_OR_NOISE

    4. High reconstruction error + sharp + not outlier
       -> REVIEW_POSSIBLE_OOD

    The system does NOT automatically classify sharp images as OOD.
    They are treated as ambiguous cases requiring review.
    """

    noise_path   = f"{REPORTS_DIR}/{split}_noise_report.csv"
    blur_path    = f"{REPORTS_DIR}/{split}_blur_report.csv"
    outlier_path = f"{REPORTS_DIR}/{split}_outliers_report.csv"

    if not os.path.exists(noise_path):
        print("Noise report not found — run detect_noise() first")
        return None

    noise_df = pd.read_csv(noise_path)

    # ── Merge blur scores and adaptive thresholds ──────────────
    if os.path.exists(blur_path):

        blur_df = pd.read_csv(blur_path)[
            ["file_path", "blur_score", "threshold_used"]
        ]

        noise_df = noise_df.merge(
            blur_df,
            on="file_path",
            how="left"
        )

    else:

        noise_df["blur_score"] = np.nan
        noise_df["threshold_used"] = np.nan

    # ── Merge outlier flags ────────────────────────────────────
    if os.path.exists(outlier_path):

        out_df = pd.read_csv(outlier_path)[
            ["file_path", "is_outlier"]
        ]

        noise_df = noise_df.merge(
            out_df,
            on="file_path",
            how="left"
        )

    else:

        noise_df["is_outlier"] = False

    noise_df["is_outlier"] = (
        noise_df["is_outlier"]
        .fillna(False)
        .astype(bool)
    )

    # ── Calculate blur ratio ───────────────────────────────────
    # Compares an image's blur score with its own class baseline.
    #
    # < 1.0  -> below class blur threshold
    # 1 - 2   -> borderline
    # > 2.0   -> relatively sharp

    noise_df["blur_ratio"] = np.where(
        noise_df["threshold_used"] > 0,
        noise_df["blur_score"] / noise_df["threshold_used"],
        np.nan
    )

    # ── Refine individual noise verdict ────────────────────────
    def refined_verdict(row):

        # Image was not originally flagged by AutoEncoder
        if not row["is_noisy"]:
            return False, "not_flagged"

        blur_ratio = row.get("blur_ratio")
        is_outlier = bool(row.get("is_outlier", False))

        # If blur information is unavailable,
        # do not make an aggressive decision.
        if pd.isna(blur_ratio):

            if is_outlier:
                return True, "review_ood_or_noise"

            return False, "review_insufficient_evidence"

        # ──────────────────────────────────────────────────────
        # CASE 1:
        # Reconstruction error is high and image is below
        # its class-specific blur threshold.
        #
        # Strongest evidence of genuine image degradation.
        # ──────────────────────────────────────────────────────
        if blur_ratio < 1.0:

            return True, "confirmed_noisy"

        # ──────────────────────────────────────────────────────
        # CASE 2:
        # Image is close to its blur threshold.
        #
        # Evidence is ambiguous, so send for review.
        # ──────────────────────────────────────────────────────
        if blur_ratio < 2.0:

            return True, "review_noise"

        # ──────────────────────────────────────────────────────
        # CASE 3:
        # Image is relatively sharp AND unusual according
        # to the outlier detector.
        #
        # Could be OOD or genuine noise.
        # Do not automatically remove.
        # ──────────────────────────────────────────────────────
        if is_outlier:

            return True, "review_ood_or_noise"

        # ──────────────────────────────────────────────────────
        # CASE 4:
        # High reconstruction error but image is relatively
        # sharp and not an outlier.
        #
        # AutoEncoder may be reacting to unusual content,
        # pose, background, or another distributional property.
        #
        # Do NOT call it definitely OOD.
        # Send it for review.
        # ──────────────────────────────────────────────────────
        return False, "review_possible_ood"

    # ── Apply refinement ───────────────────────────────────────
    results = noise_df.apply(
        lambda r: refined_verdict(r),
        axis=1
    )

    noise_df["is_noisy_refined"] = results.apply(
        lambda x: x[0]
    )

    noise_df["noise_verdict"] = results.apply(
        lambda x: x[1]
    )

    # ── Summary ────────────────────────────────────────────────
    original = int(
        noise_df["is_noisy"].sum()
    )

    refined = int(
        noise_df["is_noisy_refined"].sum()
    )

    confirmed_noisy = int(
        (
            noise_df["noise_verdict"]
            == "confirmed_noisy"
        ).sum()
    )

    review_noise = int(
        (
            noise_df["noise_verdict"]
            == "review_noise"
        ).sum()
    )

    review_ood_or_noise = int(
        (
            noise_df["noise_verdict"]
            == "review_ood_or_noise"
        ).sum()
    )

    review_possible_ood = int(
        (
            noise_df["noise_verdict"]
            == "review_possible_ood"
        ).sum()
    )

    print(
        f"\n── Noise Refinement ───────────────────────────────"
    )

    print(
        f"  Original AE flags       : {original}"
    )

    print(
        f"  Confirmed noisy         : {confirmed_noisy}"
    )

    print(
        f"  Review — noise          : {review_noise}"
    )

    print(
        f"  Review — OOD/noise      : {review_ood_or_noise}"
    )

    print(
        f"  Review — possible OOD   : {review_possible_ood}"
    )

    print(
        f"  Final noisy/review      : {refined}"
    )

    print(
        f"\n── Verdict Distribution ───────────────────────────"
    )

    verdicts = noise_df[
        "noise_verdict"
    ].value_counts()

    for v, c in verdicts.items():

        print(
            f"  {v:30s}: {c}"
        )

    # ── Save refined report ───────────────────────────────────
    out_path = (
        f"{REPORTS_DIR}/{split}_noise_report.csv"
    )

    noise_df.to_csv(
        out_path,
        index=False
    )

    print(
        f"\n✅ Refined noise report saved → {out_path}"
    )

    return noise_df



# ----- Main Execution -----

if __name__ == "__main__":

    # ── Blur Detection ─────────────────────────────────────────
    for split in ["train", "val"]:

        df_blur = detect_blur(split)

        print(
            f"\n Top 10 Blurriest Images "
            f"[{split.upper()}]"
        )

        top_blur = df_blur.nsmallest(
            10,
            "blur_score"
        )[
            ["file_path", "class", "blur_score"]
        ]

        print(
            top_blur.to_string(index=False)
        )


    # ── Noise Detection ────────────────────────────────────────
    for split in ["train", "val"]:

        df_noise = detect_noise(split)

        print(
            f"\n Top 10 Noisiest Images "
            f"[{split.upper()}]"
        )

        print(
            df_noise.nlargest(
                10,
                "reconstruction_error"
            )[
                [
                    "file_path",
                    "class",
                    "reconstruction_error"
                ]
            ].to_string(index=False)
        )

        # ── Medium 4: Noise Refinement ─────────────────────────
        df_noise_refined = refine_noise_flags(split)

        print(
            f"\n Top 10 Refined Noise Cases "
            f"[{split.upper()}]"
        )

        print(
            df_noise_refined[
                df_noise_refined["is_noisy_refined"]
            ].sort_values(
                "reconstruction_error",
                ascending=False
            ).head(10)[
                [
                    "file_path",
                    "class",
                    "reconstruction_error",
                    "blur_score",
                    "is_outlier",
                    "noise_verdict"
                ]
            ].to_string(index=False)
        )
        
    












