from fastapi          import APIRouter
from pathlib          import Path
import time

from api.schemas      import HealthResponse
from api.dependencies import get_resnet, get_autoencoder

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Verify API is up and models are loaded."""
    return HealthResponse(
        status        = "ok",
        version       = "1.0.0",
        models_loaded = Path("models/autoencoder.pth").exists(),
        reports_ready = {
            "train_master"     : Path("reports/train_master_report.csv").exists(),
            "val_master"       : Path("reports/val_master_report.csv").exists(),
            "train_embeddings" : Path("embeddings/train_embeddings.npy").exists(),
        },
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    )