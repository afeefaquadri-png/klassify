"""
Klassify – Experiment tracker.

Provides an MLflow-inspired experiment tracking layer that:
* Persists run metadata + metrics as JSON on disk.
* Assigns unique run IDs with timestamps.
* Supports listing, filtering, and comparing experiments.
* Is framework-agnostic (no MLflow dependency required).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.settings import settings
from utils.exceptions import ExperimentError
from utils.logger import get_logger

logger = get_logger(__name__)

_EXPERIMENTS_DIR = settings.EXPERIMENT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Data model (plain dicts for JSON-serializability)
# ──────────────────────────────────────────────────────────────────────────────

def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# ExperimentTracker
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    File-backed experiment tracker.

    Each experiment is stored as a directory under ``EXPERIMENT_DIR/<experiment_name>/``
    with individual runs stored as ``<run_id>.json``.

    Usage::

        tracker = ExperimentTracker("iris_classification")
        run_id = tracker.start_run(model_key="random_forest", dataset="iris.csv")
        tracker.log_params(run_id, {"n_estimators": 100})
        tracker.log_metrics(run_id, {"accuracy": 0.96, "f1": 0.95})
        tracker.end_run(run_id)
    """

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.experiment_dir = _EXPERIMENTS_DIR / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def start_run(
        self,
        model_key: str,
        dataset: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a new run; returns its *run_id*."""
        run_id = _new_run_id()
        run_data: Dict[str, Any] = {
            "run_id": run_id,
            "experiment": self.experiment_name,
            "model_key": model_key,
            "dataset": dataset,
            "status": "RUNNING",
            "start_time": _now_iso(),
            "end_time": None,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": tags or {},
        }
        self._save_run(run_id, run_data)
        logger.info("Started run %s for model=%s", run_id, model_key)
        return run_id

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Merge *params* into an existing run."""
        run = self._load_run(run_id)
        run["params"].update({k: _to_serializable(v) for k, v in params.items()})
        self._save_run(run_id, run)

    def log_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        """Merge *metrics* into an existing run."""
        run = self._load_run(run_id)
        run["metrics"].update({k: _to_serializable(v) for k, v in metrics.items()})
        self._save_run(run_id, run)

    def log_artifact(self, run_id: str, artifact_path: str) -> None:
        """Record an artifact file path."""
        run = self._load_run(run_id)
        run["artifacts"].append(artifact_path)
        self._save_run(run_id, run)

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Mark a run as finished."""
        run = self._load_run(run_id)
        run["status"] = status
        run["end_time"] = _now_iso()
        self._save_run(run_id, run)
        logger.info("Ended run %s – status=%s", run_id, status)

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return self._load_run(run_id)

    def list_runs(
        self,
        model_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all runs, optionally filtered."""
        runs = []
        for path in sorted(self.experiment_dir.glob("*.json")):
            try:
                run = json.loads(path.read_text())
                if model_key and run.get("model_key") != model_key:
                    continue
                if status and run.get("status") != status:
                    continue
                runs.append(run)
            except Exception:
                pass
        return sorted(runs, key=lambda r: r.get("start_time", ""), reverse=True)

    def compare_runs(self, run_ids: List[str], metric: str = "accuracy") -> List[Dict[str, Any]]:
        """Return sorted comparison of specified runs by *metric*."""
        rows = []
        for rid in run_ids:
            try:
                run = self._load_run(rid)
                row = {
                    "run_id": rid,
                    "model_key": run.get("model_key"),
                    "start_time": run.get("start_time"),
                    metric: run.get("metrics", {}).get(metric),
                    **{f"param_{k}": v for k, v in run.get("params", {}).items()},
                }
                rows.append(row)
            except Exception:
                pass
        return sorted(rows, key=lambda r: (r.get(metric) or 0), reverse=True)

    def get_best_run(self, metric: str = "accuracy", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """Return the run with the best value for *metric*."""
        runs = self.list_runs(status="FINISHED")
        valid = [r for r in runs if r.get("metrics", {}).get(metric) is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r["metrics"][metric]) if higher_is_better \
            else min(valid, key=lambda r: r["metrics"][metric])

    # ── I/O ───────────────────────────────────────────────────────────────

    def _run_path(self, run_id: str) -> Path:
        return self.experiment_dir / f"{run_id}.json"

    def _save_run(self, run_id: str, data: Dict[str, Any]) -> None:
        self._run_path(run_id).write_text(json.dumps(data, indent=2, default=str))

    def _load_run(self, run_id: str) -> Dict[str, Any]:
        path = self._run_path(run_id)
        if not path.exists():
            raise ExperimentError(f"Run '{run_id}' not found in experiment '{self.experiment_name}'")
        return json.loads(path.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# Global registry of trackers
# ──────────────────────────────────────────────────────────────────────────────

_trackers: Dict[str, ExperimentTracker] = {}


def get_tracker(experiment_name: str) -> ExperimentTracker:
    """Get or create an :class:`ExperimentTracker` for *experiment_name*."""
    if experiment_name not in _trackers:
        _trackers[experiment_name] = ExperimentTracker(experiment_name)
    return _trackers[experiment_name]


def list_experiments() -> List[str]:
    """Return all experiment names on disk."""
    if not _EXPERIMENTS_DIR.exists():
        return []
    return [d.name for d in _EXPERIMENTS_DIR.iterdir() if d.is_dir()]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_serializable(value: Any) -> Any:
    """Convert numpy / non-JSON types to native Python."""
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
