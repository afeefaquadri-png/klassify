"""
Klassify – SHAP Explainability.

Wraps the ``shap`` library to provide model-agnostic explanations.
Falls back gracefully if shap is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 200,
    background_samples: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP values for *model* on *X*.

    Automatically selects the appropriate explainer:
    * ``TreeExplainer``  – tree-based models (RF, GBM, XGBoost, LightGBM)
    * ``LinearExplainer`` – linear models (Logistic Regression)
    * ``KernelExplainer`` – all others (slower)

    Args:
        model:             Fitted classifier.
        X:                 Feature matrix (numpy array).
        feature_names:     Column names for display.
        max_samples:       Maximum samples to explain (for speed).
        background_samples: Background dataset size for KernelExplainer.

    Returns:
        Dict with keys ``shap_values``, ``feature_names``, ``base_value``,
        or ``None`` if shap is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap package not installed – explainability unavailable.")
        return None

    # Sub-sample for speed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = _get_explainer(shap, model, X, background_samples)
    if explainer is None:
        return None

    try:
        shap_vals = explainer.shap_values(X_sample)
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return None

    # For multiclass, shap_vals is a list of arrays – take mean abs
    if isinstance(shap_vals, list):
        shap_arr = np.array(shap_vals)           # (n_classes, n_samples, n_features)
        mean_abs = np.abs(shap_arr).mean(axis=(0, 1))  # (n_features,)
        sample_vals = shap_arr[0]               # use class 0 for waterfall
    else:
        sample_vals = shap_vals
        mean_abs = np.abs(sample_vals).mean(axis=0)

    base_value = float(explainer.expected_value) \
        if not isinstance(explainer.expected_value, (list, np.ndarray)) \
        else float(explainer.expected_value[0])

    names = feature_names or [f"f{i}" for i in range(X_sample.shape[1])]

    return {
        "shap_values": sample_vals.tolist(),
        "mean_abs_shap": mean_abs.tolist(),
        "feature_names": names,
        "base_value": base_value,
        "n_samples": len(X_sample),
    }


def shap_summary_data(shap_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SHAP result to a tidy DataFrame for plotting.

    Returns columns: ``feature``, ``mean_abs_shap`` sorted descending.
    """
    df = pd.DataFrame({
        "feature": shap_result["feature_names"],
        "mean_abs_shap": shap_result["mean_abs_shap"],
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_explainer(shap: Any, model: Any, X: np.ndarray, background_samples: int):
    model_type = type(model).__name__.lower()
    tree_models = {
        "randomforestclassifier", "gradientboostingclassifier",
        "xgbclassifier", "lgbmclassifier", "decisiontreeclassifier",
        "extratreesclassifier",
    }
    linear_models = {
        "logisticregression", "sgdclassifier", "linearsvc",
        "ridgeclassifier",
    }

    try:
        if model_type in tree_models:
            return shap.TreeExplainer(model)
        elif model_type in linear_models:
            background = shap.sample(X, background_samples)
            return shap.LinearExplainer(model, background)
        else:
            background = shap.sample(X, background_samples)
            return shap.KernelExplainer(model.predict_proba, background)
    except Exception as exc:
        logger.warning("Could not create SHAP explainer: %s", exc)
        return None
