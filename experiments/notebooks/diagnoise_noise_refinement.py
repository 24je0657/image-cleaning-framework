# notebooks/diagnose_noise_refinement.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


df = pd.read_csv(
    "reports/train_noise_report.csv"
)


# ============================================================
# Only originally flagged images
# ============================================================

flagged = df[
    df["is_noisy"] == True
].copy()


print(
    f"Total originally flagged : {len(flagged)}"
)

print("\nVerdict breakdown:")
print(
    flagged["noise_verdict"].value_counts()
)


# ============================================================
# Blur score comparison
# ============================================================

print("\nBlur score stats by verdict:")

print(
    flagged
    .groupby("noise_verdict")["blur_score"]
    .agg([
        "mean",
        "min",
        "max",
        "count"
    ])
    .round(2)
)


# ============================================================
# Reconstruction error comparison
# ============================================================

print(
    "\nReconstruction error stats by verdict:"
)

print(
    flagged
    .groupby("noise_verdict")["reconstruction_error"]
    .agg([
        "mean",
        "min",
        "max",
        "count"
    ])
    .round(6)
)


# ============================================================
# Visual diagnosis
# 8 confirmed noisy vs 8 downgraded OOD
# ============================================================

confirmed = flagged[
    flagged["noise_verdict"] == "confirmed_noisy"
].head(8)

downgraded = flagged[
    flagged["noise_verdict"] == "downgraded_ood"
].head(8)


fig, axes = plt.subplots(
    4,
    4,
    figsize=(16, 12)
)

fig.suptitle(
    "Noise Refinement Diagnosis",
    fontsize=14
)


for i in range(4):

    # --------------------------------------------------------
    # Confirmed noisy — image 1
    # --------------------------------------------------------

    if i < len(confirmed):

        row = confirmed.iloc[i]

        img = Image.open(
            row["file_path"]
        ).convert("RGB")

        axes[i, 0].imshow(img)

        axes[i, 0].set_title(
            f"CONFIRMED NOISY\n"
            f"err={row['reconstruction_error']:.4f}\n"
            f"blur={row['blur_score']:.0f}",
            fontsize=8
        )

        axes[i, 0].axis("off")


    # --------------------------------------------------------
    # Downgraded OOD — image 1
    # --------------------------------------------------------

    if i < len(downgraded):

        row = downgraded.iloc[i]

        img = Image.open(
            row["file_path"]
        ).convert("RGB")

        axes[i, 1].imshow(img)

        axes[i, 1].set_title(
            f"DOWNGRADED OOD\n"
            f"err={row['reconstruction_error']:.4f}\n"
            f"blur={row['blur_score']:.0f}",
            fontsize=8
        )

        axes[i, 1].axis("off")


# ============================================================
# Second set of 4 images
# ============================================================

confirmed_2 = confirmed.iloc[4:8]

downgraded_2 = downgraded.iloc[4:8]


for i in range(4):

    if i < len(confirmed_2):

        row = confirmed_2.iloc[i]

        img = Image.open(
            row["file_path"]
        ).convert("RGB")

        axes[i, 2].imshow(img)

        axes[i, 2].set_title(
            f"CONFIRMED NOISY\n"
            f"err={row['reconstruction_error']:.4f}\n"
            f"blur={row['blur_score']:.0f}",
            fontsize=8
        )

        axes[i, 2].axis("off")


    if i < len(downgraded_2):

        row = downgraded_2.iloc[i]

        img = Image.open(
            row["file_path"]
        ).convert("RGB")

        axes[i, 3].imshow(img)

        axes[i, 3].set_title(
            f"DOWNGRADED OOD\n"
            f"err={row['reconstruction_error']:.4f}\n"
            f"blur={row['blur_score']:.0f}",
            fontsize=8
        )

        axes[i, 3].axis("off")


plt.tight_layout()

plt.savefig(
    "notebooks/noise_refinement_diagnosis.png",
    dpi=120
)

plt.close()

print(
    "\nSaved → "
    "notebooks/noise_refinement_diagnosis.png"
)