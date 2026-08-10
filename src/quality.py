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



# ----- Main Execution -----

if __name__ == "__main__":
    # Blur Detection
    for split in ["train","val"]:
        df_blur = detect_blur(split)

        print(f" \n Top 10 Blurriest Images [{split.upper()}]")

        top_blur = df_blur.nsmallest(10, "blur_score")[
            ["file_path" , "class", "blur_score"]
        ]
        print(top_blur.to_string(index = False))

    # Noise Detection
    for split in ["train","val"]:
        df_noise = detect_noise(split)

        print(f" \n Top 10 Noisiest Images [{split.upper()}]")
        print(df_noise.nlargest(10, "reconstruction_error")[
            ["file_path" , "class", "reconstruction_error"]
            ].to_string(index = False))
        
    












