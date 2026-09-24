import asyncio
import shutil
import pandas as pd
from pathlib          import Path
from fastapi          import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import ExportRequest
from api.worker  import jobs, executor, make_job, submit
import uuid

router = APIRouter(tags=["Reports"])


@router.get("/stats/{split}")
async def get_stats(split: str):
    """Summary stats from precomputed master report."""
    path = Path(f"reports/{split}_master_report.csv")
    if not path.exists():
        raise HTTPException(404, f"No master report for '{split}'. "
                                  "Run /pipeline/run first.")
    loop = asyncio.get_event_loop()
    df   = await loop.run_in_executor(executor, pd.read_csv, str(path))

    total = len(df)
    def safe(col):
        return int(df[col].astype(bool).sum()) if col in df.columns else 0

    return JSONResponse(content={
        "split"         : split,
        "total_images"  : total,
        "clean"         : int((df["verdict"] == "CLEAN").sum()),
        "to_remove"     : int(df["verdict"].str.startswith("REMOVE").sum()),
        "to_review"     : int(df["verdict"].str.startswith("REVIEW").sum()),
        "efficiency_pct": round(
            100*(df["verdict"]=="CLEAN").sum()/total, 2),
        "issue_breakdown": {
            "exact_duplicates": safe("exact_duplicate"),
            "near_duplicates" : safe("near_duplicate"),
            "blurry"          : safe("is_blurry"),
            "noisy"           : safe("is_noisy"),
            "outliers"        : safe("is_outlier"),
            "mislabeled"      : safe("is_mislabeled"),
        },
        "verdict_distribution": df["verdict"].value_counts().to_dict(),
        "per_class": {
            cls: {
                "total" : int(len(df[df["class"]==cls])),
                "clean" : int((df[df["class"]==cls]["verdict"]=="CLEAN").sum()),
                "remove": int(df[df["class"]==cls]["verdict"]
                               .str.startswith("REMOVE").sum()),
                "review": int(df[df["class"]==cls]["verdict"]
                               .str.startswith("REVIEW").sum()),
            }
            for cls in ["cat","dog","elephant","horse","lion"]
            if cls in df["class"].values
        }
    })


@router.get("/reports/{split}/{report_type}")
async def download_report(split: str, report_type: str):
    """Download any CSV report directly."""
    path = Path(f"reports/{split}_{report_type}_report.csv")
    if not path.exists():
        raise HTTPException(404, f"Report not found: {path.name}")
    return FileResponse(str(path),
                        filename=path.name,
                        media_type="text/csv")


@router.post("/export")
async def export_dataset(request: ExportRequest):
    """Export cleaned dataset as async background job."""
    master_path = f"reports/{request.split}_master_report.csv"
    if not Path(master_path).exists():
        raise HTTPException(404, "Run pipeline first")

    job_id           = str(uuid.uuid4())
    jobs[job_id]     = make_job(job_id)

    def do_export():
        try:
            df    = pd.read_csv(master_path)
            mask  = pd.Series(False, index=df.index)

            if request.remove_duplicates:
                mask |= df.get("exact_duplicate", False).astype(bool)
                mask |= df.get("near_duplicate",  False).astype(bool)
            if request.remove_blurry:
                mask |= df.get("is_blurry", False).astype(bool)
            if request.remove_noisy:
                mask |= df.get("is_noisy",  False).astype(bool)
            if request.remove_outliers:
                mask |= df.get("is_outlier",False).astype(bool)
            if request.remove_mislabeled:
                mask |= (df.get("is_mislabeled", False).astype(bool) &
                         (df.get("mislabel_confidence","") == "HIGH"))
            mask |= df["priority_score"] >= request.min_priority_score

            keep       = df[~mask]
            clean_root = Path("data_clean") / request.split
            copied     = 0

            for i,(_, row) in enumerate(keep.iterrows()):
                src = Path(row["file_path"])
                dst = clean_root / row["class"] / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    pass
                jobs[job_id]["progress"] = int(100*i/len(keep))

            jobs[job_id].update({
                "status"  : "completed",
                "progress": 100,
                "message" : f"Exported {copied} images",
                "result"  : {
                    "copied"     : copied,
                    "removed"    : int(mask.sum()),
                    "output_path": f"data_clean/{request.split}/"
                }
            })
        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": str(e)})

    await submit(do_export)
    return {"job_id": job_id, "message": "Export started"}