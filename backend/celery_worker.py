"""
Klassify – Celery async worker.

Background tasks for long-running training jobs.
Uses Redis as both broker and result backend.

Usage::

    celery -A backend.celery_worker worker --loglevel=info
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Celery
from celery.utils.log import get_task_logger

from configs.settings import settings

# ── App ────────────────────────────────────────────────────────────────────────
celery_app = Celery(
    "klassify",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=86400,  # 24 h
    worker_prefetch_multiplier=1,
)

logger = get_task_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="klassify.train_model_task", max_retries=2)
def train_model_task(
    self,
    dataset_path: str,
    target_col: str,
    model_key: str,
    experiment_name: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Async training task.

    Args:
        dataset_path:    Absolute path to the dataset file.
        target_col:      Name of the target column.
        model_key:       Key from model_configs.yaml.
        experiment_name: Experiment to log under.
        options:         Additional kwargs forwarded to ``training_service.run_single``.

    Returns:
        Training result summary dict (JSON-serialisable).
    """
    from backend.training_service import training_service

    self.update_state(state="PROGRESS", meta={"status": "Starting training…"})
    logger.info("Task %s – training %s on %s", self.request.id, model_key, dataset_path)

    try:
        result = training_service.run_single(
            dataset_path=Path(dataset_path),
            target_col=target_col,
            model_key=model_key,
            experiment_name=experiment_name,
            **(options or {}),
        )
        return {"status": "SUCCESS", **result}
    except Exception as exc:
        logger.exception("Training task failed: %s", exc)
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(bind=True, name="klassify.run_experiment_task")
def run_experiment_task(
    self,
    dataset_path: str,
    target_col: str,
    model_keys: List[str],
    experiment_name: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Async multi-model experiment task.
    """
    from backend.training_service import training_service

    self.update_state(state="PROGRESS", meta={"status": "Running experiment…", "n_models": len(model_keys)})
    logger.info("Experiment task %s – %d models", self.request.id, len(model_keys))

    results = training_service.run_experiment(
        dataset_path=Path(dataset_path),
        target_col=target_col,
        model_keys=model_keys,
        experiment_name=experiment_name,
        **(options or {}),
    )
    return {"status": "SUCCESS", "results": results}


@celery_app.task(name="klassify.health_check")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "worker": "klassify-worker"}
