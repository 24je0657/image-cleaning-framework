import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50 , ResNet50_Weights
from PIL import Image
from pathlib import Path

# ----- Configuration -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
ROOT_DIR = "data/raw/Animal"
EMB_DIR = "embeddings"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
os.makedirs(EMB_DIR, exist_ok = True)

# ----- Image Transformation -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                          std = [0.229, 0.224, 0.225])
])

# ----- Load Pre-trained Model -----
def load_model():
    """Load the ResNet50 pretrained model and remove the fc layer -> feature extractor."""
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last fc layer
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded on {DEVICE}.")
    return model

# ----- Single Image Embedding -----
def load_image(path):
    """ Load one image safely and return None if it fails.  """
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None
    
# ----- Extraction function -----
def extract_embeddings(split = "train"):
    """ Walk through Animal/{split}/<class>/*.jpg, extract ResNet50 embeddings in batches and save embeddings, labels, and file paths as .npy files."""
    split_path = Path(ROOT_DIR) / split
    model = load_model()
    all_embeddings = []
    all_labels = []
    all_file_paths = []
    classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
    print(f"\n classses found: {classes}")

    for class_name in classes:
        class_dir = split_path / class_name
        image_files = sorted([f for f in class_dir.iterdir() 
                              if f.suffix.lower() in VALID_EXTENSIONS])
        print(f"\n processing [{class_name}] with {len(image_files)} images.")
        
        batch_paths = []
        batch_tensors = []
        for idx , img_path in enumerate(image_files):
            tensor = load_image(img_path)
            if tensor is None:
                print(f"Skipping image {img_path} due to loading error.")
                continue  
                
            batch_tensors.append(tensor)
            batch_paths.append(str(img_path))

            if len(batch_tensors) == BATCH_SIZE or idx == len(image_files) - 1:
                if not batch_tensors:
                    continue  # Skip if no valid images in the batch
                    
                batch = torch.stack(batch_tensors).to(DEVICE)

                with torch.no_grad():
                    features = model(batch) # batch shape (BATCH_SIZE, 2048, 1, 1)
                    features = features.squeeze(-1).squeeze(-1)  # shape (BATCH_SIZE, 2048)
                    features = features.cpu().numpy()
                
                all_embeddings.append(features)
                all_labels.extend([class_name] * len(features))
                all_file_paths.extend(batch_paths)

                # --- Reset batch lists for next iteration ---
                batch_tensors = []
                batch_paths = []

                if (idx+1) % 500 == 0:
                    print(f" {idx+1} / {len(image_files)} images processed for class [{class_name}].")

    # ----- Stack and Save -----
    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)
    file_paths_array = np.array(all_file_paths)

    np.save(f"{EMB_DIR}/{split}_embeddings.npy", embeddings_array)
    np.save(f"{EMB_DIR}/{split}_labels.npy", labels_array)
    np.save(f"{EMB_DIR}/{split}_file_paths.npy", file_paths_array)

    print(f" \n {split} Embeddings saved!")
    print(f" embeddings shape: {embeddings_array.shape}")
    print(f" Labels shape: {np.unique(labels_array , return_counts=True)}")


# ----- Embeddings Loader used for all downstream modules -----
def load_embeddings(split = "train"):
    """Load cached embeddings, labels, and file paths from .npy files."""
    embeddings = np.load(f"{EMB_DIR}/{split}_embeddings.npy")
    labels = np.load(f"{EMB_DIR}/{split}_labels.npy")
    file_paths = np.load(f"{EMB_DIR}/{split}_file_paths.npy")

    print(f" Loaded {split} embeddings: {embeddings.shape}, labels: {labels.shape}, file paths: {file_paths.shape}")
    return embeddings, labels, file_paths

# ----- Main Execution -----
if __name__ == "__main__":
    for split in ["train", "val"]:
        extract_embeddings(split)

                

