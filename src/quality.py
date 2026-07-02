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
def detect_blur(split = "train"):

    """ Walk through Animal/<split>/<class> and compute blur score for every image.
    Save report to reports/<split>_blur_report.csv
    """

    split_path = Path(ROOT_DIR)/split
    records = []

    classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
    print(f"\n{'='*30}")
    print(f"Blur detection - {split.upper()} Set")
    print(f"{'='*30}")
    print(f"Classes - {classes}")
    print(f"Blur Threshold - {BLUR_THRESHOLD}")

    for class_name in classes:
        class_dir = split_path / class_name
        img_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file and f.suffix.lower() in VALID_EXTENSIONS
        ])

        print(f"Processing {class_name}- {len(img_files)} images")

        for idx, img_path in enumerate(img_files):
            score = compute_blur_score(img_path)
            if score is None :
                continue

            records.append({
                "file_path" : str(img_path),
                "class" : class_name,
                "blur_score" : score,
                "is_blurry" : score < BLUR_THRESHOLD
            })
            if(idx+1) % 500 == 0:
                print(f" {idx+1} / {len(img_files)} done...")

    df = pd.DataFrame(records)


    print(f"\n ----- Per class Blur Statistics -----")
    stats = df.groupby("class").agg(

        total = ("blur_score","count"),
        blurry_count = ("is_blurry", "sum"),
        mean_score = ("blur_score", "mean"),
        min_score = ("blur_score", "min"),
        max_score = ("blur_score", "max")
    ).reset_index()

    stats["blurry_%"] = (stats["blurry_count"] / stats["total"] *100).round(2)

    print(stats.to_string(index = False))


    # ----- Overall Summary -----
    total_blurry = df["is_blurry"].sum()
    print(f"\n ----- Overall Summary -----")
    print(f" Total images :{len(df)}")
    print(f" Blurry images :{total_blurry} ({100 * total_blurry / len(df):.2f}%)")
    print(f" Mean Score : {df["blur_score"].mean():.2f}")
    print(f" Min Score : {df["blur_score"].min():.2f}")
    print(f" Max Score :{df["blur_score"].max():.2f}")

    # ----- Save Report -----
    out_path = f"{REPORTS_DIR}/{split}_blur_report.csv"
    df.to_csv(out_path, index = False)

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
        
    












