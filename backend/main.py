"""
Klassify – FastAPI backend.

All REST API endpoints.  Designed to be independently runnable:

    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from configs.settings import settings
from experiments.experiment_tracker import list_experiments, get_tracker
from experiments.model_registry import registry
from ml.dataset_loader import (
    detect_feature_types,
    get_class_distribution,
    load_dataset,
    profile_dataset,
)
from ml.model_factory import get_available_models, get_model_display_names
from utils.exceptions import KlassifyError
from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade ML experimentation platform API.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_id: str
    target_col: str
    model_key: str
    experiment_name: str = "default"
    custom_params: Dict[str, Any] = Field(default_factory=dict)
    tuning_strategy: Optional[str] = None
    tuning_n_iter: int = 20
    run_cv: bool = False
    cv_folds: int = 5
    test_size: float = 0.2
    scaler: str = "standard"
    encoding: str = "onehot"


class ExperimentRequest(BaseModel):
    dataset_id: str
    target_col: str
    model_keys: List[str]
    experiment_name: str = "default"
    tuning_strategy: Optional[str] = None
    run_cv: bool = False
    test_size: float = 0.2
    scaler: str = "standard"


class PredictRequest(BaseModel):
    model_key: str
    version: Optional[str] = None
    data: List[List[float]]


# ──────────────────────────────────────────────────────────────────────────────
# In-memory dataset store (maps dataset_id → absolute path)
# In production, replace with a database-backed store.
# ──────────────────────────────────────────────────────────────────────────────

_DATASET_STORE: Dict[str, Path] = {}


def _resolve_dataset(dataset_id: str) -> Path:
    if dataset_id not in _DATASET_STORE:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found. Upload it first.")
    return _DATASET_STORE[dataset_id]


# ──────────────────────────────────────────────────────────────────────────────
# Error handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(KlassifyError)
async def klassify_error_handler(request, exc: KlassifyError):
    return JSONResponse(status_code=400, content={"error": exc.code, "message": exc.message})


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": settings.APP_VERSION}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset routes
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/upload_dataset", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV/Parquet/JSON dataset.  Returns a ``dataset_id``."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    dataset_id = uuid.uuid4().hex[:8]
    dest = settings.UPLOAD_DIR / f"{dataset_id}{suffix}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = dest.stat().st_size / 1024**2
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        dest.unlink()
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_UPLOAD_SIZE_MB} MB")

    _DATASET_STORE[dataset_id] = dest
    df = load_dataset(dest)
    logger.info("Uploaded dataset %s (%d rows)", dataset_id, len(df))
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
    }


@app.get("/api/v1/dataset/{dataset_id}/profile", tags=["Dataset"])
def dataset_profile(dataset_id: str):
    """Return a full profiling report for an uploaded dataset."""
    path = _resolve_dataset(dataset_id)
    df = load_dataset(path)
    return profile_dataset(df)


@app.get("/api/v1/dataset/{dataset_id}/preview", tags=["Dataset"])
def dataset_preview(dataset_id: str, n: int = 20):
    """Return first *n* rows as JSON."""
    path = _resolve_dataset(dataset_id)
    df = load_dataset(path)
    return df.head(n).to_dict(orient="records")


@app.get("/api/v1/dataset/{dataset_id}/class_distribution", tags=["Dataset"])
def class_distribution(dataset_id: str, target_col: str):
    path = _resolve_dataset(dataset_id)
    df = load_dataset(path)
    return get_class_distribution(df, target_col)


# ──────────────────────────────────────────────────────────────────────────────
# Model routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/models", tags=["Models"])
def get_models():
    """Return all available model keys and display names."""
    return get_model_display_names()


@app.get("/api/v1/registry/models", tags=["Registry"])
def list_registered_models():
    return {"models": registry.list_models()}


@app.get("/api/v1/registry/models/{model_key}/versions", tags=["Registry"])
def list_model_versions(model_key: str):
    return {"versions": registry.list_versions(model_key)}


@app.get("/api/v1/registry/leaderboard", tags=["Registry"])
def leaderboard(metric: str = "accuracy"):
    return registry.leaderboard(metric=metric)


# ──────────────────────────────────────────────────────────────────────────────
# Training routes
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/train_model", tags=["Training"])
def train_model_sync(req: TrainRequest):
    """Synchronous single-model training (blocks until complete)."""
    from backend.training_service import training_service

    path = _resolve_dataset(req.dataset_id)
    result = training_service.run_single(
        dataset_path=path,
        target_col=req.target_col,
        model_key=req.model_key,
        experiment_name=req.experiment_name,
        custom_params=req.custom_params,
        tuning_strategy=req.tuning_strategy,
        tuning_n_iter=req.tuning_n_iter,
        run_cv=req.run_cv,
        cv_folds=req.cv_folds,
        test_size=req.test_size,
        scaler=req.scaler,
        encoding=req.encoding,
    )
    return result


@app.post("/api/v1/train_model/async", tags=["Training"])
def train_model_async(req: TrainRequest):
    """Submit training as an async Celery task. Returns task_id."""
    from backend.celery_worker import train_model_task

    path = _resolve_dataset(req.dataset_id)
    task = train_model_task.delay(
        dataset_path=str(path),
        target_col=req.target_col,
        model_key=req.model_key,
        experiment_name=req.experiment_name,
        options={
            "custom_params": req.custom_params,
            "tuning_strategy": req.tuning_strategy,
            "run_cv": req.run_cv,
            "test_size": req.test_size,
        },
    )
    return {"task_id": task.id, "status": "PENDING"}


@app.get("/api/v1/tasks/{task_id}", tags=["Training"])
def get_task_status(task_id: str):
    """Poll async task status and result."""
    from backend.celery_worker import celery_app as _celery

    result = _celery.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


@app.post("/api/v1/run_experiment", tags=["Training"])
def run_experiment(req: ExperimentRequest):
    """Train all *model_keys* and return a comparison."""
    from backend.training_service import training_service

    path = _resolve_dataset(req.dataset_id)
    return training_service.run_experiment(
        dataset_path=path,
        target_col=req.target_col,
        model_keys=req.model_keys,
        experiment_name=req.experiment_name,
        tuning_strategy=req.tuning_strategy,
        run_cv=req.run_cv,
        test_size=req.test_size,
        scaler=req.scaler,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics / Experiment routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/experiments", tags=["Experiments"])
def get_experiments():
    return {"experiments": list_experiments()}


@app.get("/api/v1/experiments/{experiment_name}/runs", tags=["Experiments"])
def get_runs(experiment_name: str):
    tracker = get_tracker(experiment_name)
    return tracker.list_runs()


@app.get("/api/v1/experiments/{experiment_name}/runs/{run_id}", tags=["Experiments"])
def get_run(experiment_name: str, run_id: str):
    tracker = get_tracker(experiment_name)
    return tracker.get_run(run_id)


@app.get("/api/v1/experiments/{experiment_name}/best", tags=["Experiments"])
def get_best_run(experiment_name: str, metric: str = "accuracy"):
    tracker = get_tracker(experiment_name)
    best = tracker.get_best_run(metric=metric)
    if best is None:
        raise HTTPException(404, "No finished runs found.")
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Prediction route
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/predict", tags=["Prediction"])
def predict(req: PredictRequest):
    """Run inference using a registered model version."""
    from backend.training_service import training_service

    X = np.array(req.data)
    return training_service.predict(req.model_key, X, req.version)


# ──────────────────────────────────────────────────────────────────────────────
# Model export
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/registry/models/{model_key}/export", tags=["Registry"])
def export_model(model_key: str, version: Optional[str] = None, fmt: str = "joblib"):
    """Download a model artifact (joblib / onnx)."""
    if fmt == "onnx":
        path = registry.export_onnx(model_key, version)
    else:
        entry = registry._get_entry(model_key, version)
        path = Path(entry["artifact_path"])

    if not path.exists():
        raise HTTPException(404, "Artifact not found.")
    return FileResponse(path=str(path), filename=path.name, media_type="application/octet-stream")
