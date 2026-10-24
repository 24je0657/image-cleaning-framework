# Standard library
import os
from pathlib import Path

# Third-party libraries
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


# ----- Configuration -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224

ROOT_DIR = "data/raw/Animal"
EMB_DIR = "embeddings"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

os.makedirs(EMB_DIR, exist_ok=True)


# ----- Image Transformation -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ----- Load Pre-trained Model -----
def load_model():
    """Load pretrained ResNet50 and remove the final FC layer."""

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model.eval()
    model.to(DEVICE)

    print(f"Model loaded on {DEVICE}.")

    return model


# ----- Single Image Embedding -----
def load_image(path):
    """Load one image safely and return None if it fails."""

    try:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        return img

    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


# ----- Extraction Function -----
def extract_embeddings(
    split="train",
    root_dir=ROOT_DIR,
    embed_dir=EMB_DIR,
):
    """
    Extract ResNet50 embeddings from images in:
        root_dir/split/class_name/

    Save embeddings, labels, and file paths as .npy files
    inside embed_dir.
    """

    split_path = Path(root_dir) / split
    embed_path = Path(embed_dir)

    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Dataset split directory not found: {split_path}"
        )

    # Create the output directory if necessary.
    embed_path.mkdir(parents=True, exist_ok=True)

    model = load_model()

    all_embeddings = []
    all_labels = []
    all_file_paths = []

    classes = sorted([
        d.name for d in split_path.iterdir()
        if d.is_dir()
    ])

    if not classes:
        raise ValueError(
            f"No class directories found in {split_path}"
        )

    print(f"\nClasses found: {classes}")

    for class_name in classes:
        class_dir = split_path / class_name

        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in VALID_EXTENSIONS
        ])

        print(
            f"\nProcessing [{class_name}] "
            f"with {len(image_files)} images."
        )

        batch_paths = []
        batch_tensors = []

        for idx, img_path in enumerate(image_files):

            tensor = load_image(img_path)

            if tensor is None:
                print(
                    f"Skipping image {img_path} "
                    "due to loading error."
                )
                continue

            batch_tensors.append(tensor)
            batch_paths.append(str(img_path))

            # Process a full batch or the final image in this class.
            if (
                len(batch_tensors) == BATCH_SIZE
                or idx == len(image_files) - 1
            ):
                if not batch_tensors:
                    continue

                batch = torch.stack(batch_tensors).to(DEVICE)

                with torch.no_grad():
                    features = model(batch)
                    features = features.squeeze(-1).squeeze(-1)
                    features = features.cpu().numpy()

                all_embeddings.append(features)

                all_labels.extend(
                    [class_name] * len(features)
                )

                all_file_paths.extend(batch_paths)

                # Reset batch lists.
                batch_tensors = []
                batch_paths = []

            if (idx + 1) % 500 == 0:
                print(
                    f"{idx + 1} / {len(image_files)} images "
                    f"processed for class [{class_name}]."
                )

    if not all_embeddings:
        raise ValueError(
            f"No valid images were processed in {split_path}"
        )

    # ----- Stack and Save -----
    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)
    file_paths_array = np.array(all_file_paths)

    np.save(
        embed_path / f"{split}_embeddings.npy",
        embeddings_array,
    )

    np.save(
        embed_path / f"{split}_labels.npy",
        labels_array,
    )

    np.save(
        embed_path / f"{split}_file_paths.npy",
        file_paths_array,
    )

    print(f"\n{split} embeddings saved!")
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(
        f"Labels distribution: "
        f"{np.unique(labels_array, return_counts=True)}"
    )
    print(f"File paths shape: {file_paths_array.shape}")


# ----- Embeddings Loader Used by Downstream Modules -----
def load_embeddings(
    split="train",
    embed_dir=EMB_DIR,
):
    """
    Load cached embeddings, labels, and file paths
    from .npy files.
    """

    embed_path = Path(embed_dir)

    embeddings_file = embed_path / f"{split}_embeddings.npy"
    labels_file = embed_path / f"{split}_labels.npy"
    paths_file = embed_path / f"{split}_file_paths.npy"

    # Give a clear error if extraction has not been run.
    for file_path in (embeddings_file, labels_file, paths_file):
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Embedding file not found: {file_path}"
            )

    embeddings = np.load(embeddings_file)

    labels = np.load(
        labels_file,
        allow_pickle=True,
    )

    file_paths = np.load(
        paths_file,
        allow_pickle=True,
    )

    print(
        f"Loaded {split} embeddings: {embeddings.shape}, "
        f"labels: {labels.shape}, "
        f"file paths: {file_paths.shape}"
    )

    return embeddings, labels, file_paths


# ----- Main Execution -----
if __name__ == "__main__":
    for split in ["train", "val"]:
        extract_embeddings(split)