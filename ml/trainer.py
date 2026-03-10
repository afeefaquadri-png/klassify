"""
Klassify – Core model trainer.

Orchestrates:
  1. Train/test splitting
  2. Model fitting
  3. Optional hyperparameter tuning (grid / random / bayesian)
  4. Optional cross-validation
  5. Metrics computation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)

from ml.metrics import compute_metrics, cross_validate_model
from ml.model_factory import build_model, get_param_distributions, get_param_grid
from utils.exceptions import HyperparameterError, TrainingError
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_key: str
    custom_params: Dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2
    random_state: int = 42
    # Tuning
    tuning_strategy: Optional[str] = None  # "grid", "random", "bayesian"
    tuning_cv: int = 3
    tuning_n_iter: int = 20              # for random/bayesian
    scoring: str = "f1_weighted"
    # Cross-validation
    run_cv: bool = False
    cv_folds: int = 5
    cv_stratified: bool = True


@dataclass
class TrainingResult:
    model_key: str
    model: ClassifierMixin
    best_params: Dict[str, Any]
    metrics: Dict[str, Any]
    cv_results: Optional[Dict[str, Any]]
    training_time_s: float
    tuning_time_s: float
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> TrainingResult:
    """
    Full training pipeline for a single model.

    Args:
        X:           Feature matrix (preprocessed).
        y:           Encoded label vector.
        config:      :class:`TrainingConfig` instance.
        class_names: Human-readable class labels.
        feature_names: Feature column names post-transform.

    Returns:
        Populated :class:`TrainingResult`.

    Raises:
        TrainingError: On any training failure.
    """
    logger.info("Starting training: model=%s, tuning=%s", config.model_key, config.tuning_strategy)

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # ── Build base model ──────────────────────────────────────────────────
    try:
        model = build_model(config.model_key, config.custom_params or None)
    except Exception as exc:
        raise TrainingError(f"Model build failed: {exc}") from exc

    # ── Hyperparameter tuning ─────────────────────────────────────────────
    best_params = dict(model.get_params())
    tuning_time_s = 0.0

    if config.tuning_strategy:
        model, best_params, tuning_time_s = _tune(
            model=model,
            X_train=X_train,
            y_train=y_train,
            model_key=config.model_key,
            strategy=config.tuning_strategy,
            cv=config.tuning_cv,
            n_iter=config.tuning_n_iter,
            scoring=config.scoring,
        )

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(f"Model fit failed for {config.model_key}: {exc}") from exc
    training_time_s = time.perf_counter() - t0
    logger.info("Trained %s in %.2fs", config.model_key, training_time_s)

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = compute_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        training_time=training_time_s,
    )

    # ── Cross-validate (optional) ─────────────────────────────────────────
    cv_results: Optional[Dict] = None
    if config.run_cv:
        cv_results = cross_validate_model(
            model=clone(model),
            X=X,
            y=y,
            n_splits=config.cv_folds,
            stratified=config.cv_stratified,
            scoring=config.scoring,
        )

    return TrainingResult(
        model_key=config.model_key,
        model=model,
        best_params=best_params,
        metrics=metrics,
        cv_results=cv_results,
        training_time_s=training_time_s,
        tuning_time_s=tuning_time_s,
        feature_names=feature_names,
        class_names=class_names,
    )


def train_multiple(
    X: np.ndarray,
    y: np.ndarray,
    model_keys: List[str],
    base_config: TrainingConfig,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, TrainingResult]:
    """
    Train several models under a shared configuration.

    Returns a mapping of {model_key: TrainingResult}.
    Failures for individual models are logged and skipped.
    """
    results: Dict[str, TrainingResult] = {}
    for key in model_keys:
        cfg = TrainingConfig(
            model_key=key,
            custom_params=base_config.custom_params,
            test_size=base_config.test_size,
            random_state=base_config.random_state,
            tuning_strategy=base_config.tuning_strategy,
            tuning_cv=base_config.tuning_cv,
            tuning_n_iter=base_config.tuning_n_iter,
            scoring=base_config.scoring,
            run_cv=base_config.run_cv,
            cv_folds=base_config.cv_folds,
        )
        try:
            results[key] = train_model(X, y, cfg, class_names, feature_names)
            logger.info("Completed %s – accuracy=%.4f", key, results[key].metrics["accuracy"])
        except Exception as exc:
            logger.error("Training failed for %s: %s", key, exc)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tune(
    model: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_key: str,
    strategy: str,
    cv: int,
    n_iter: int,
    scoring: str,
) -> tuple:
    """Run hyperparameter search; returns (best_model, best_params, elapsed)."""
    t0 = time.perf_counter()

    if strategy == "grid":
        param_grid = get_param_grid(model_key)
        if not param_grid:
            logger.warning("No param_grid for %s – skipping grid search", model_key)
            return model, dict(model.get_params()), 0.0
        searcher = GridSearchCV(
            estimator=clone(model),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )

    elif strategy == "random":
        param_dist = get_param_distributions(model_key)
        if not param_dist:
            logger.warning("No param_distributions for %s – skipping random search", model_key)
            return model, dict(model.get_params()), 0.0
        searcher = RandomizedSearchCV(
            estimator=clone(model),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            random_state=42,
        )

    elif strategy == "bayesian":
        try:
            from skopt import BayesSearchCV
        except ImportError:
            raise HyperparameterError(
                "scikit-optimize is required for Bayesian search: pip install scikit-optimize"
            )
        param_dist = get_param_distributions(model_key)
        searcher = BayesSearchCV(
            estimator=clone(model),
            search_spaces=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            random_state=42,
        )
    else:
        raise HyperparameterError(f"Unknown tuning strategy: '{strategy}'")

    try:
        searcher.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(f"Hyperparameter search failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        "Tuning complete (%s): best_score=%.4f params=%s in %.1fs",
        strategy, searcher.best_score_, searcher.best_params_, elapsed,
    )
    return searcher.best_estimator_, searcher.best_params_, elapsed
