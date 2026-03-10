"""
Klassify – Model Registry.

Manages versioned model artifacts:
* Save / load via joblib or pickle.
* ONNX export (optional).
* Version tagging and metadata lookup.
* Leaderboard across all registered models.
"""

from __future__ import annotations

import json
import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.base import ClassifierMixin

from configs.settings import settings
from utils.exceptions import RegistryError
from utils.logger import get_logger

logger = get_logger(__name__)

_REGISTRY_ROOT = settings.MODEL_DIR
_REGISTRY_INDEX = _REGISTRY_ROOT / "registry_index.json"


# ──────────────────────────────────────────────────────────────────────────────
# Index helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_index() -> Dict[str, Any]:
    _REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    if not _REGISTRY_INDEX.exists():
        return {"models": {}}
    return json.loads(_REGISTRY_INDEX.read_text())


def _save_index(index: Dict[str, Any]) -> None:
    _REGISTRY_INDEX.write_text(json.dumps(index, indent=2, default=str))


# ──────────────────────────────────────────────────────────────────────────────
# ModelRegistry
# ──────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Filesystem-backed versioned model registry.

    Models are stored under::

        MODEL_DIR/<model_key>/v<version>/<model_key>.joblib

    Each version entry in the registry index captures metadata, metrics,
    and artifact paths.
    """

    def register(
        self,
        model: ClassifierMixin,
        model_key: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Persist *model* and return its assigned version string.

        Args:
            model:           Fitted scikit-learn compatible classifier.
            model_key:       Logical model name (e.g. ``"random_forest"``).
            metrics:         Evaluation metrics dict.
            params:          Hyperparameters used.
            experiment_name: Associated experiment (optional).
            run_id:          Associated run ID (optional).
            tags:            Free-form key-value metadata.

        Returns:
            Version string (e.g. ``"v3"``).
        """
        index = _load_index()
        versions = index["models"].setdefault(model_key, {})
        version_num = len(versions) + 1
        version = f"v{version_num}"

        model_dir = _REGISTRY_ROOT / model_key / version
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = model_dir / f"{model_key}.joblib"
        joblib.dump(model, artifact_path)

        entry: Dict[str, Any] = {
            "version": version,
            "model_key": model_key,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "artifact_path": str(artifact_path),
            "metrics": {k: _to_serializable(v) for k, v in metrics.items()
                        if not isinstance(v, (list, dict))},  # skip curves
            "params": {k: _to_serializable(v) for k, v in params.items()},
            "experiment_name": experiment_name,
            "run_id": run_id,
            "tags": tags or {},
        }
        versions[version] = entry
        _save_index(index)
        logger.info("Registered %s %s at %s", model_key, version, artifact_path)
        return version

    def load(self, model_key: str, version: Optional[str] = None) -> ClassifierMixin:
        """
        Load a model from the registry.

        Args:
            model_key: Logical model name.
            version:   Specific version string (e.g. ``"v2"``); defaults to latest.

        Returns:
            Loaded classifier.

        Raises:
            RegistryError: If the model / version is not found.
        """
        entry = self._get_entry(model_key, version)
        path = Path(entry["artifact_path"])
        if not path.exists():
            raise RegistryError(f"Artifact file missing: {path}")
        model = joblib.load(path)
        logger.info("Loaded %s %s from %s", model_key, entry["version"], path)
        return model

    def export_onnx(
        self,
        model_key: str,
        version: Optional[str] = None,
        n_features: int = 1,
    ) -> Path:
        """
        Export model to ONNX format.  Requires ``skl2onnx``.

        Returns path to the exported ``.onnx`` file.
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise RegistryError("skl2onnx is required for ONNX export: pip install skl2onnx")

        model = self.load(model_key, version)
        entry = self._get_entry(model_key, version)
        onnx_path = Path(entry["artifact_path"]).with_suffix(".onnx")

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_path.write_bytes(onnx_model.SerializeToString())
        logger.info("Exported ONNX model to %s", onnx_path)
        return onnx_path

    def list_models(self) -> List[str]:
        """Return all registered model keys."""
        return list(_load_index()["models"].keys())

    def list_versions(self, model_key: str) -> List[Dict[str, Any]]:
        """Return all version entries for *model_key*, newest first."""
        index = _load_index()
        versions = index["models"].get(model_key, {})
        entries = list(versions.values())
        return sorted(entries, key=lambda e: e.get("registered_at", ""), reverse=True)

    def get_latest_version(self, model_key: str) -> Optional[str]:
        """Return the latest version string, or None if not registered."""
        versions = self.list_versions(model_key)
        return versions[0]["version"] if versions else None

    def leaderboard(self, metric: str = "accuracy") -> List[Dict[str, Any]]:
        """
        Return a sorted leaderboard of best versions per model ranked by *metric*.
        """
        index = _load_index()
        rows = []
        for model_key, versions in index["models"].items():
            for entry in versions.values():
                val = entry.get("metrics", {}).get(metric)
                if val is not None:
                    rows.append({
                        "model_key": model_key,
                        "version": entry["version"],
                        "registered_at": entry["registered_at"],
                        metric: val,
                        **{f"metric_{k}": v for k, v in entry.get("metrics", {}).items()},
                    })
        return sorted(rows, key=lambda r: r.get(metric, 0), reverse=True)

    def delete(self, model_key: str, version: str) -> None:
        """Remove a specific model version from registry and disk."""
        index = _load_index()
        versions = index["models"].get(model_key, {})
        if version not in versions:
            raise RegistryError(f"Version {version} of {model_key} not found.")
        entry = versions.pop(version)
        model_dir = Path(entry["artifact_path"]).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        if not versions:
            del index["models"][model_key]
        _save_index(index)
        logger.info("Deleted %s %s", model_key, version)

    # ── Private ───────────────────────────────────────────────────────────

    def _get_entry(self, model_key: str, version: Optional[str]) -> Dict[str, Any]:
        index = _load_index()
        versions = index["models"].get(model_key)
        if not versions:
            raise RegistryError(f"Model '{model_key}' not registered.")
        if version is None:
            # latest
            entries = sorted(versions.values(), key=lambda e: e.get("registered_at", ""), reverse=True)
            return entries[0]
        if version not in versions:
            raise RegistryError(f"Version '{version}' of '{model_key}' not found.")
        return versions[version]


# ── Singleton ──────────────────────────────────────────────────────────────────

registry = ModelRegistry()


# ──────────────────────────────────────────────────────────────────────────────

def _to_serializable(value: Any) -> Any:
    import numpy as np
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
