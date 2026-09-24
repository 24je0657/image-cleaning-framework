"""
Pipeline orchestrator — calls services in order,
updates job state, handles errors per module.
"""
import time
import sys
sys.path.insert(0, ".")

import numpy as np
from api.services.duplicate_service  import DuplicateService
from api.services.quality_service    import QualityService
from api.services.outlier_service    import OutlierService
from api.services.mislabel_service   import MislabelService
from api.services.decision_engine    import DecisionEngineService


class PipelineService:

    def __init__(self, config: dict):
        self.config = config
        self.dup_svc = DuplicateService(
            cosine_threshold=config.get("cosine_threshold", 0.97))
        self.qual_svc = QualityService(
            blur_threshold=config.get("blur_threshold", 100.0))
        self.out_svc  = OutlierService(
            contamination=config.get("contamination", 0.05))
        self.mis_svc  = MislabelService()
        self.de_svc   = DecisionEngineService()

    def run(self, job_id: str, jobs: dict, split: str):
        """
        Full pipeline. Called inside ThreadPoolExecutor.
        Updates jobs[job_id] at every stage.
        """
        start = time.time()
        jobs[job_id]["status"]  = "running"
        module_results          = []

        def update(pct: int, msg: str):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["message"]  = msg

        def record(result: dict):
            module_results.append(result)

        try:
            # ── Embeddings
            update(10, "Extracting ResNet50 embeddings...")
            from src.embeddings import extract_embeddings, load_embeddings
            extract_embeddings(split)
            embeddings, labels, paths = load_embeddings(split)

            # ── Duplicates
            if self.config.get("run_duplicates", True):
                update(25, "Detecting duplicates...")
                record(self.dup_svc.find_exact_duplicates(paths))
                record(self.dup_svc.find_near_duplicates(embeddings, paths))

            # ── Blur
            if self.config.get("run_blur", True):
                update(42, "Detecting blur...")
                record(self.qual_svc.detect_blur_batch(paths))

            # ── Noise
            if self.config.get("run_noise", True):
                update(55, "Detecting noise...")
                record(self.qual_svc.detect_noise_batch(paths))

            # ── Outliers
            if self.config.get("run_outliers", True):
                update(68, "Running Isolation Forest...")
                record(self.out_svc.detect(embeddings, labels, paths))

            # ── Mislabels
            if self.config.get("run_mislabels", True):
                update(82, "Detecting mislabeled images...")
                record(self.mis_svc.detect(embeddings, labels, paths))

            # ── Decision Engine
            update(94, "Building master report...")
            summary = self.de_svc.merge_reports(split)

            jobs[job_id].update({
                "status"   : "completed",
                "progress" : 100,
                "message"  : "Pipeline complete ✅",
                "result"   : {
                    **summary,
                    "modules"     : module_results,
                    "duration_s"  : round(time.time() - start, 2),
                }
            })

        except Exception as e:
            jobs[job_id].update({
                "status"  : "failed",
                "message" : f"Pipeline failed: {e}",
                "error"   : str(e)
            })