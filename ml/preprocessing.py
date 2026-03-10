"""
Klassify – Preprocessing pipeline.

Builds a scikit-learn Pipeline that handles:
* Missing values  (numeric → median imputation, categorical → most-frequent)
* Encoding        (ordinal / one-hot for categoricals)
* Scaling         (standard / minmax / robust – user-selectable)
* Feature engineering hooks (polynomial, interaction terms)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from utils.exceptions import DatasetError
from utils.logger import get_logger

logger = get_logger(__name__)

ScalerType = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_types: Dict[str, str],
    *,
    scaler: str = "standard",
    encoding: str = "onehot",
    drop_high_cardinality: bool = True,
    cardinality_threshold: int = 50,
    polynomial_degree: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, ColumnTransformer]:
    """
    Prepare features and target for model training.

    Args:
        df:                    Full dataframe.
        target_col:            Name of the target column.
        feature_types:         Output of ``detect_feature_types()``.
        scaler:                One of ``standard``, ``minmax``, ``robust``.
        encoding:              One of ``onehot``, ``ordinal``.
        drop_high_cardinality: Drop categoricals with too many unique values.
        cardinality_threshold: Max unique values before a column is dropped.
        polynomial_degree:     If set, adds polynomial features to numerics.

    Returns:
        Tuple of (X_transformed, y_encoded, label_encoder, column_transformer).
    """
    if target_col not in df.columns:
        raise DatasetError(f"Target column '{target_col}' not found in dataset.")

    df = df.copy()
    y_raw = df.pop(target_col)

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    # Separate feature groups
    numeric_cols = [
        c for c, t in feature_types.items()
        if t == "numeric" and c != target_col and c in df.columns
    ]
    categorical_cols = [
        c for c, t in feature_types.items()
        if t == "categorical" and c != target_col and c in df.columns
    ]
    # Drop datetime / text cols (not yet supported)
    drop_cols = [
        c for c, t in feature_types.items()
        if t in ("datetime", "text", "unknown") and c != target_col and c in df.columns
    ]
    if drop_cols:
        logger.warning("Dropping unsupported columns: %s", drop_cols)

    if drop_high_cardinality:
        hi_card = [
            c for c in categorical_cols
            if df[c].nunique() > cardinality_threshold
        ]
        if hi_card:
            logger.warning("Dropping high-cardinality columns: %s", hi_card)
            categorical_cols = [c for c in categorical_cols if c not in hi_card]

    if not numeric_cols and not categorical_cols:
        raise DatasetError("No usable feature columns remain after preprocessing.")

    # Build sub-pipelines
    scaler_cls = ScalerType.get(scaler, StandardScaler)

    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler_cls()),
    ]
    if polynomial_degree and polynomial_degree > 1:
        numeric_steps.append(
            ("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False))
        )

    numeric_pipe = Pipeline(numeric_steps)

    encoder_cls = OneHotEncoder(handle_unknown="ignore", sparse_output=False) \
        if encoding == "onehot" else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder_cls),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    X = df[numeric_cols + categorical_cols]
    X_transformed = ct.fit_transform(X)

    logger.info(
        "Preprocessed: %d samples, %d features → %d transformed features",
        len(y), len(numeric_cols) + len(categorical_cols), X_transformed.shape[1],
    )
    return X_transformed, y, le, ct


def get_feature_names_out(ct: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Extract human-readable feature names after transformation.
    """
    names: List[str] = []
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            # May have poly features
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names.extend(last_step.get_feature_names_out(cols).tolist())
            else:
                names.extend(cols)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            if hasattr(encoder, "get_feature_names_out"):
                names.extend(encoder.get_feature_names_out(cols).tolist())
            else:
                names.extend(cols)
    return names
