import uuid
import asyncio
from pathlib    import Path
from fastapi    import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from api.schemas      import SingleImageRequest, ImageAnalysisResult
from api.worker       import executor
from api.services.quality_service    import QualityService
from api.services.duplicate_service  import DuplicateService

router   = APIRouter(tags=["Analysis"])
qual_svc = QualityService()


async def _analyze(path: str) -> dict:
    loop = asyncio.get_event_loop()

    blur_score = await loop.run_in_executor(
        executor, qual_svc.compute_blur_score, path
    )
    noise_err  = await loop.run_in_executor(
        executor, qual_svc.compute_reconstruction_error, path
    )

    is_blurry = blur_score is not None and blur_score < 100
    is_noisy  = noise_err  is not None and noise_err  > 0.02

    flags = []
    if is_blurry: flags.append("blurry")
    if is_noisy:  flags.append("noisy")

    weights       = {"blurry": 6, "noisy": 5}
    priority      = sum(weights.get(f, 0) for f in flags)
    class_name    = Path(path).parent.name

    if priority == 0:     verdict = "CLEAN"
    elif priority >= 10:  verdict = "REMOVE — high priority"
    elif priority >= 5:   verdict = "REVIEW — medium priority"
    else:                 verdict = "REVIEW — low priority"

    return {
        "file_path"            : path,
        "class_name"           : class_name,
        "blur_score"           : blur_score,
        "is_blurry"            : is_blurry,
        "reconstruction_error" : noise_err,
        "is_noisy"             : is_noisy,
        "is_duplicate"         : False,
        "is_outlier"           : False,
        "is_mislabeled"        : False,
        "mislabel_confidence"  : None,
        "predicted_class"      : None,
        "priority_score"       : priority,
        "verdict"              : verdict,
        "flags"                : flags
    }


@router.post("/analyze/image")
async def analyze_image(request: SingleImageRequest):
    """Analyze a single image by local path."""
    if not Path(request.image_path).exists():
        raise HTTPException(404, f"Image not found: {request.image_path}")
    result = await _analyze(request.image_path)
    return JSONResponse(content=result)


@router.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Upload an image and analyze it immediately."""
    tmp = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    try:
        tmp.write_bytes(await file.read())
        result = await _analyze(str(tmp))
        result["original_filename"] = file.filename
        return JSONResponse(content=result)
    finally:
        if tmp.exists():
            tmp.unlink()