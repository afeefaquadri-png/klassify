"""
Klassify – Training Service.

High-level orchestration that combines:
  Dataset loading → Preprocessing → Training → Evaluation → Tracking → Registry
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from configs.settings import settings
from experiments.experiment_tracker import ExperimentTracker, get_tracker
from experiments.model_registry import registry
from ml.dataset_loader import detect_feature_types, get_class_distribution, load_dataset
from ml.preprocessing import get_feature_names_out, prepare_data
from ml.trainer import TrainingConfig, TrainingResult, train_model, train_multiple
from utils.exceptions import DatasetError, TrainingError
from utils.logger import get_logger

logger = get_logger(__name__)


class TrainingService:
    """
    Application-layer service that orchestrates end-to-end ML runs.

    This is the primary entry point called by the API layer and the
    Streamlit frontend.  All heavy lifting is delegated to domain modules.
    """

    # ── Single-model run ──────────────────────────────────────────────────

    def run_single(
        self,
        dataset_path: Path,
        target_col: str,
        model_key: str,
        experiment_name: str,
        *,
        custom_params: Optional[Dict[str, Any]] = None,
        tuning_strategy: Optional[str] = None,
        tuning_cv: int = 3,
        tuning_n_iter: int = 20,
        run_cv: bool = False,
        cv_folds: int = 5,
        test_size: float = 0.2,
        scaler: str = "standard",
        encoding: str = "onehot",
        register_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline for a single model.

        Returns a JSON-serialisable summary dict.
        """
        tracker = get_tracker(experiment_name)

        # ── Load & preprocess ─────────────────────────────────────────────
        df = load_dataset(dataset_path)
        feature_types = detect_feature_types(df)

        X, y, le, ct = prepare_data(
            df, target_col, feature_types,
            scaler=scaler, encoding=encoding,
        )
        class_names = list(le.classes_)

        # Derive feature names for explainability
        numeric_cols = [c for c, t in feature_types.items()
                        if t == "numeric" and c != target_col and c in df.columns]
        categorical_cols = [c for c, t in feature_types.items()
                            if t == "categorical" and c != target_col and c in df.columns]
        feature_names = get_feature_names_out(ct, numeric_cols, categorical_cols)

        # ── Training config ───────────────────────────────────────────────
        config = TrainingConfig(
            model_key=model_key,
            custom_params=custom_params or {},
            test_size=test_size,
            tuning_strategy=tuning_strategy,
            tuning_cv=tuning_cv,
            tuning_n_iter=tuning_n_iter,
            run_cv=run_cv,
            cv_folds=cv_folds,
            scoring="f1_weighted",
        )

        # ── Experiment tracking ───────────────────────────────────────────
        run_id = tracker.start_run(
            model_key=model_key,
            dataset=dataset_path.name,
        )
        tracker.log_params(run_id, config.custom_params or config.__dict__)

        try:
            result: TrainingResult = train_model(
                X, y, config,
                class_names=class_names,
                feature_names=feature_names,
            )
        except Exception as exc:
            tracker.end_run(run_id, status="FAILED")
            raise TrainingError(str(exc)) from exc

        # Log scalar metrics only
        scalar_metrics = {
            k: v for k, v in result.metrics.items()
            if isinstance(v, (int, float)) and v is not None
        }
        tracker.log_metrics(run_id, scalar_metrics)
        tracker.end_run(run_id)

        # ── Registry ──────────────────────────────────────────────────────
        version: Optional[str] = None
        if register_model:
            version = registry.register(
                model=result.model,
                model_key=model_key,
                metrics=scalar_metrics,
                params=result.best_params,
                experiment_name=experiment_name,
                run_id=run_id,
            )
            tracker.log_artifact(run_id, str(settings.MODEL_DIR / model_key / version))

        return {
            "run_id": run_id,
            "model_key": model_key,
            "version": version,
            "metrics": scalar_metrics,
            "best_params": result.best_params,
            "cv_results": result.cv_results,
            "training_time_s": result.training_time_s,
            "tuning_time_s": result.tuning_time_s,
            "class_names": class_names,
            "n_features": X.shape[1],
            "n_samples": len(y),
        }

    # ── Multi-model experiment ─────────────────────────────────────────────

    def run_experiment(
        self,
        dataset_path: Path,
        target_col: str,
        model_keys: List[str],
        experiment_name: str,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all *model_keys* and return a comparison dict.
        """
        logger.info(
            "Running experiment '%s' with %d models on %s",
            experiment_name, len(model_keys), dataset_path.name,
        )
        results: Dict[str, Dict[str, Any]] = {}
        for key in model_keys:
            try:
                results[key] = self.run_single(
                    dataset_path=dataset_path,
                    target_col=target_col,
                    model_key=key,
                    experiment_name=experiment_name,
                    **kwargs,
                )
            except Exception as exc:
                logger.error("Failed %s: %s", key, exc)
                results[key] = {"error": str(exc), "model_key": key}
        return results

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        model_key: str,
        X: np.ndarray,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference using a registered model.
        """
        model = registry.load(model_key, version)
        predictions = model.predict(X)
        proba: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        return {
            "predictions": predictions.tolist(),
            "probabilities": proba.tolist() if proba is not None else None,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
training_service = TrainingService()
