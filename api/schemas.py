from pydantic import BaseModel, Field
from typing   import Optional, List, Dict, Any
from enum     import Enum


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL   = "val"


# ── Requests ──────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    split              : SplitType = SplitType.TRAIN
    run_duplicates     : bool  = True
    run_blur           : bool  = True
    run_noise          : bool  = True
    run_outliers       : bool  = True
    run_mislabels      : bool  = True
    blur_threshold     : float = Field(100.0, ge=0)
    cosine_threshold   : float = Field(0.97,  ge=0, le=1.0)
    contamination      : float = Field(0.05,  ge=0.01, le=0.5)


class SingleImageRequest(BaseModel):
    image_path : str
    run_all    : bool = True


class ExportRequest(BaseModel):
    split              : SplitType = SplitType.TRAIN
    remove_duplicates  : bool = True
    remove_blurry      : bool = True
    remove_noisy       : bool = False
    remove_outliers    : bool = False
    remove_mislabeled  : bool = True
    min_priority_score : int  = Field(10, ge=0)


# ── Responses ─────────────────────────────────────────────────────────────────

class ModuleResult(BaseModel):
    module     : str
    flagged    : int
    total      : int
    flag_pct   : float
    duration_s : float
    status     : str = "complete"


class JobResponse(BaseModel):
    job_id     : str
    status     : JobStatus
    progress   : int
    message    : str
    created_at : str
    updated_at : str
    result     : Optional[Dict[str, Any]] = None
    error      : Optional[str]            = None


class ImageAnalysisResult(BaseModel):
    file_path             : str
    class_name            : str
    blur_score            : Optional[float]
    is_blurry             : bool
    reconstruction_error  : Optional[float]
    is_noisy              : bool
    is_duplicate          : bool
    is_outlier            : bool
    is_mislabeled         : bool
    mislabel_confidence   : Optional[str]
    predicted_class       : Optional[str]
    priority_score        : int
    verdict               : str
    flags                 : List[str]


class HealthResponse(BaseModel):
    status        : str
    version       : str
    models_loaded : bool
    reports_ready : Dict[str, bool]
    timestamp     : str


class StatsResponse(BaseModel):
    split                : str
    total_images         : int
    clean                : int
    to_remove            : int
    to_review            : int
    issue_breakdown      : Dict[str, int]
    per_class            : Dict[str, Dict[str, int]]
    verdict_distribution : Dict[str, int]