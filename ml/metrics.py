"""
Klassify – Model evaluation metrics.

Computes the full suite of classification metrics and returns
a structured, JSON-serialisable result dict.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    log_loss,
    matthews_corrcoef,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate *model* on the test split and return a comprehensive metrics dict.

    Args:
        model:         Fitted classifier.
        X_test:        Test feature matrix.
        y_test:        True labels (integer-encoded).
        class_names:   Human-readable class labels.
        training_time: Seconds taken to train (injected externally).

    Returns:
        Dict with all scalar metrics, curves, and confusion matrix.
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    n_classes = len(np.unique(y_test))
    multiclass = n_classes > 2
    average = "weighted" if multiclass else "binary"

    # ── Probability estimates ─────────────────────────────────────────────
    y_prob: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            pass

    # ── Core scalars ──────────────────────────────────────────────────────
    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, average=average, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, average=average, zero_division=0)), 6),
        "f1": round(float(f1_score(y_test, y_pred, average=average, zero_division=0)), 6),
        "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 6),
        "training_time_s": round(training_time, 4),
        "inference_time_s": round(inference_time, 6),
        "n_test_samples": int(len(y_test)),
    }

    # ── ROC-AUC ───────────────────────────────────────────────────────────
    if y_prob is not None:
        try:
            if multiclass:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            else:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            metrics["roc_auc"] = round(float(auc), 6)
        except Exception as e:
            logger.warning("Could not compute ROC-AUC: %s", e)
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = round(float(log_loss(y_test, y_prob)), 6)
        except Exception:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ── Per-class report ──────────────────────────────────────────────────
    target_names = class_names if class_names else [str(i) for i in sorted(np.unique(y_test))]
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    # ── ROC curve data (binary only) ──────────────────────────────────────
    if y_prob is not None and not multiclass:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
        prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob[:, 1])
        metrics["pr_curve"] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": pr_thresh.tolist(),
            "average_precision": round(float(average_precision_score(y_test, y_prob[:, 1])), 6),
        }
    else:
        metrics["roc_curve"] = None
        metrics["pr_curve"] = None

    return metrics


def cross_validate_model(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    scoring: str = "f1_weighted",
) -> Dict[str, Any]:
    """
    Run k-fold (or stratified k-fold) cross-validation.

    Returns mean, std, and per-fold scores.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) \
        if stratified else n_splits

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "scoring": scoring,
        "n_splits": n_splits,
        "stratified": stratified,
        "scores": scores.tolist(),
        "mean": round(float(scores.mean()), 6),
        "std": round(float(scores.std()), 6),
        "min": round(float(scores.min()), 6),
        "max": round(float(scores.max()), 6),
    }
