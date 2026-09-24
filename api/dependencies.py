# api/dependencies.py
"""
Models loaded ONCE at startup and shared across all requests.
This is critical — you never want to reload ResNet50 per request.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from functools import lru_cache
from pathlib   import Path
import numpy    as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvAutoencoder(nn.Module):
    """Must match training architecture exactly."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@lru_cache(maxsize=1)
def get_resnet():
    """Load ResNet50 once, cache forever."""
    print("Loading ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)
    print(f"ResNet50 loaded on {DEVICE}")
    return model


@lru_cache(maxsize=1)
def get_autoencoder():
    """Load autoencoder once, cache forever."""
    model_path = "models/autoencoder.pth"
    if not Path(model_path).exists():
        print("⚠ autoencoder.pth not found — noise detection disabled")
        return None
    print("Loading autoencoder...")
    model = ConvAutoencoder()
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.eval().to(DEVICE)
    print("Autoencoder loaded")
    return model


# ── Transforms ────────────────────────────────────────────────────────────────
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

ae_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])