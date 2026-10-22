from fastapi          import APIRouter
from pathlib          import Path
from datetime import datetime, timezone


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
            "animal_train_master"     : Path("reports/Animal/train_master_report.csv").exists(),
            "animal_val_master"       : Path("reports/Animal/val_master_report.csv").exists(),
            "intel_train_master"      : Path("reports/intel/train_master_report.csv").exists(),
            "intel_test_master"       : Path("reports/intel/test_master_report.csv").exists(),
            "animal_embeddings"       : Path("embeddings/Animal/train_embeddings.npy").exists(),
            "intel_embeddings"        : Path("embeddings/intel/train_embeddings.npy").exists(),
        },
        timestamp = datetime.now(timezone.utc)
    )