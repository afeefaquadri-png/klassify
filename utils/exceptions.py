"""
Klassify – Application exception hierarchy.
"""

from __future__ import annotations


class KlassifyError(Exception):
    """Base exception for all Klassify errors."""

    def __init__(self, message: str, code: str = "KLASSIFY_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


# ── Data layer ────────────────────────────────────────────────────────────────

class DatasetError(KlassifyError):
    """Raised when dataset operations fail."""


class ValidationError(DatasetError):
    """Raised when uploaded data fails validation."""


class UnsupportedFileTypeError(ValidationError):
    """Raised for disallowed file extensions."""


class FileSizeLimitError(ValidationError):
    """Raised when file exceeds the configured size limit."""


# ── ML / Training layer ───────────────────────────────────────────────────────

class TrainingError(KlassifyError):
    """Raised when model training fails."""


class ModelNotFoundError(KlassifyError):
    """Raised when a requested model key doesn't exist."""


class HyperparameterError(TrainingError):
    """Raised for invalid hyperparameter configurations."""


class ExperimentError(KlassifyError):
    """Raised when experiment tracking operations fail."""


# ── Inference layer ───────────────────────────────────────────────────────────

class PredictionError(KlassifyError):
    """Raised when prediction / inference fails."""


# ── Registry ──────────────────────────────────────────────────────────────────

class RegistryError(KlassifyError):
    """Raised when model registry operations fail."""
