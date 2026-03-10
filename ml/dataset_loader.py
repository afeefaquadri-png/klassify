"""
Klassify – Dataset loader.

Responsibilities
----------------
* Load CSV / Parquet / JSON files into pandas DataFrames.
* Detect feature types (numeric, categorical, datetime, text).
* Generate a lightweight dataset profile report.
* Cache loaded datasets in-process to avoid repeated I/O.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.exceptions import DatasetError, UnsupportedFileTypeError
from utils.logger import get_logger

logger = get_logger(__name__)

# In-process LRU-style cache: {file_hash: DataFrame}
_DATASET_CACHE: Dict[str, pd.DataFrame] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load a dataset from *path* into a DataFrame.

    Supported formats: ``.csv``, ``.parquet``, ``.json``.
    Results are cached in-memory by file content hash.

    Args:
        path: Absolute path to the dataset file.

    Returns:
        Loaded :class:`pandas.DataFrame`.

    Raises:
        UnsupportedFileTypeError: If the file extension is not supported.
        DatasetError: If loading fails for any other reason.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".json"}:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix}. Allowed: .csv, .parquet, .json"
        )

    file_hash = _hash_file(path)
    if file_hash in _DATASET_CACHE:
        logger.debug("Dataset cache hit: %s", path.name)
        return _DATASET_CACHE[file_hash].copy()

    logger.info("Loading dataset: %s", path.name)
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_json(path)
    except Exception as exc:
        raise DatasetError(f"Failed to load {path.name}: {exc}") from exc

    _DATASET_CACHE[file_hash] = df
    logger.info("Loaded %d rows × %d cols from %s", len(df), df.shape[1], path.name)
    return df.copy()


def detect_feature_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer feature types for each column.

    Returns a mapping of ``{column_name: type_label}`` where *type_label* is
    one of ``"numeric"``, ``"categorical"``, ``"datetime"``, ``"text"``,
    or ``"unknown"``.
    """
    types: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
        elif _looks_like_datetime(series):
            types[col] = "datetime"
        elif _looks_like_text(series):
            types[col] = "text"
        else:
            types[col] = "categorical"
    return types


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a lightweight profiling report.

    Returns a dict with per-column statistics plus dataset-level summary.
    """
    feature_types = detect_feature_types(df)
    columns_info: Dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "feature_type": feature_types[col],
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique()),
        }
        if feature_types[col] == "numeric":
            desc = series.describe()
            info.update(
                {
                    "mean": _safe_float(desc.get("mean")),
                    "std": _safe_float(desc.get("std")),
                    "min": _safe_float(desc.get("min")),
                    "max": _safe_float(desc.get("max")),
                    "q25": _safe_float(desc.get("25%")),
                    "median": _safe_float(desc.get("50%")),
                    "q75": _safe_float(desc.get("75%")),
                    "skewness": _safe_float(series.skew()),
                    "kurtosis": _safe_float(series.kurtosis()),
                }
            )
        else:
            vc = series.value_counts()
            info["top_values"] = vc.head(10).to_dict()

        columns_info[col] = info

    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        "feature_types_summary": {
            ft: sum(1 for v in feature_types.values() if v == ft)
            for ft in ("numeric", "categorical", "datetime", "text", "unknown")
        },
        "columns": columns_info,
    }


def get_class_distribution(df: pd.DataFrame, target_col: str) -> Dict[str, int]:
    """Return class value counts for *target_col*."""
    return df[target_col].value_counts().to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_datetime(series: pd.Series, sample_size: int = 200) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    parsed = 0
    for val in sample:
        try:
            pd.to_datetime(str(val))
            parsed += 1
        except Exception:
            pass
    return parsed / len(sample) > 0.8


def _looks_like_text(series: pd.Series, sample_size: int = 200) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().head(sample_size).astype(str)
    avg_words = sample.str.split().apply(len).mean()
    return avg_words > 4


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return None if np.isnan(v) or np.isinf(v) else round(v, 6)
    except Exception:
        return None
