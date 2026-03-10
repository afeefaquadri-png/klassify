"""
Klassify – Model factory.

Reads model_configs.yaml and instantiates any registered classifier
by key.  Third-party packages (xgboost, lightgbm) are imported lazily
so the platform still works if they are not installed.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sklearn.base import ClassifierMixin

from utils.exceptions import ModelNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_configs.yaml"


def _load_configs() -> Dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_MODEL_CONFIGS: Dict[str, Any] = _load_configs()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_available_models() -> List[str]:
    """Return all registered model keys."""
    return list(_MODEL_CONFIGS.keys())


def get_model_display_names() -> Dict[str, str]:
    """Return {key: display_name} for all registered models."""
    return {k: v["display_name"] for k, v in _MODEL_CONFIGS.items()}


def get_model_config(model_key: str) -> Dict[str, Any]:
    """Return full config dict for *model_key*."""
    if model_key not in _MODEL_CONFIGS:
        raise ModelNotFoundError(f"Unknown model key: '{model_key}'")
    return _MODEL_CONFIGS[model_key]


def build_model(
    model_key: str,
    custom_params: Optional[Dict[str, Any]] = None,
) -> ClassifierMixin:
    """
    Instantiate a classifier by key.

    Args:
        model_key:     Key from ``model_configs.yaml`` (e.g. ``"random_forest"``).
        custom_params: Override any default hyperparameters.

    Returns:
        Instantiated (unfitted) scikit-learn compatible classifier.

    Raises:
        ModelNotFoundError: If *model_key* is not registered.
    """
    config = get_model_config(model_key)
    class_path: str = config["class"]
    params: Dict[str, Any] = dict(config.get("default_params", {}))

    # Custom params override defaults
    if custom_params:
        params.update(custom_params)

    # Handle None values for max_depth etc.
    params = {k: (None if v == "null" or v is None else v) for k, v in params.items()}

    model = _import_class(class_path)(**params)
    logger.info("Built model: %s with params=%s", config["display_name"], params)
    return model


def get_param_grid(model_key: str) -> Dict[str, Any]:
    """Return the grid-search parameter grid for *model_key*."""
    config = get_model_config(model_key)
    return config.get("param_grid", {})


def get_param_distributions(model_key: str) -> Dict[str, Any]:
    """
    Return scipy-compatible distributions for random / Bayesian search.

    Converts the YAML spec into actual scipy.stats distributions.
    """
    from scipy.stats import loguniform, randint

    config = get_model_config(model_key)
    raw = config.get("param_distributions", {})
    distributions: Dict[str, Any] = {}

    for param, spec in raw.items():
        dist = spec.get("dist")
        if dist == "loguniform":
            distributions[param] = loguniform(spec["low"], spec["high"])
        elif dist == "randint":
            distributions[param] = randint(spec["low"], spec["high"])
        else:
            distributions[param] = spec  # pass through (e.g. list)

    return distributions


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _import_class(class_path: str) -> type:
    """Dynamically import a class given its dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Cannot import {class_path}: {exc}") from exc
