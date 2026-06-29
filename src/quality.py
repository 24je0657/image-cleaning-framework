import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ----- Configuration -----
ROOT_DIR = "data/raw/Animal"
REPORTS_DIR = "reports"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BLUR_THRESHOLD = 100.0 # Below this it is Blurry (tune if needed)
os.makedirs(REPORTS_DIR, exist_ok = True)

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



# ----- Main Execution -----

if __name__ == "__main__":
    for split in ["train","val"]:
        df_blur = detect_blur(split)

        print(f" \n Top 10 Blurriest Images [{split.upper()}]")

        top_blur = df_blur.nsmallest(10, "blur_score")[
            ["file_path" , "class", "blur_score"]
        ]
        print(top_blur.to_string(index = False))












