"""
Klassify – Test suite.

Run: pytest tests/test_core.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_breast_cancer


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def iris_csv(tmp_path_factory):
    """Write Iris dataset to a temp CSV and return its path."""
    data = load_iris(as_frame=True)
    df = data.frame
    df["target"] = data.target
    path = tmp_path_factory.mktemp("data") / "iris.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def binary_csv(tmp_path_factory):
    """Write breast cancer dataset to a temp CSV."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df["target"] = data.target
    path = tmp_path_factory.mktemp("data") / "cancer.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def iris_preprocessed(iris_csv):
    """Return preprocessed X, y, le, ct for Iris."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.dataset_loader import detect_feature_types, load_dataset
    from ml.preprocessing import prepare_data

    df = load_dataset(iris_csv)
    ft = detect_feature_types(df)
    X, y, le, ct = prepare_data(df, "target", ft)
    return X, y, le, ct


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loader
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetLoader:
    def test_load_csv(self, iris_csv):
        from ml.dataset_loader import load_dataset
        df = load_dataset(iris_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150

    def test_cache_hit(self, iris_csv):
        from ml.dataset_loader import load_dataset, _DATASET_CACHE
        df1 = load_dataset(iris_csv)
        df2 = load_dataset(iris_csv)
        assert df1.shape == df2.shape

    def test_unsupported_extension(self, tmp_path):
        from ml.dataset_loader import load_dataset
        from utils.exceptions import UnsupportedFileTypeError
        bad = tmp_path / "test.xlsx"
        bad.write_text("data")
        with pytest.raises(UnsupportedFileTypeError):
            load_dataset(bad)

    def test_detect_feature_types(self, iris_csv):
        from ml.dataset_loader import detect_feature_types, load_dataset
        df = load_dataset(iris_csv)
        ft = detect_feature_types(df)
        numeric_count = sum(1 for t in ft.values() if t == "numeric")
        assert numeric_count >= 4

    def test_profile_dataset(self, iris_csv):
        from ml.dataset_loader import load_dataset, profile_dataset
        df = load_dataset(iris_csv)
        profile = profile_dataset(df)
        assert profile["n_rows"] == 150
        assert "columns" in profile
        assert profile["n_columns"] == df.shape[1]

    def test_class_distribution(self, iris_csv):
        from ml.dataset_loader import get_class_distribution, load_dataset
        df = load_dataset(iris_csv)
        dist = get_class_distribution(df, "target")
        assert len(dist) == 3
        assert sum(dist.values()) == 150


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_prepare_data_shape(self, iris_preprocessed):
        X, y, le, ct = iris_preprocessed
        assert X.shape[0] == 150
        assert len(y) == 150
        assert X.shape[1] >= 4

    def test_label_encoder(self, iris_preprocessed):
        _, y, le, _ = iris_preprocessed
        assert len(le.classes_) == 3

    def test_missing_target_raises(self, iris_csv):
        from ml.dataset_loader import detect_feature_types, load_dataset
        from ml.preprocessing import prepare_data
        from utils.exceptions import DatasetError
        df = load_dataset(iris_csv)
        ft = detect_feature_types(df)
        with pytest.raises(DatasetError):
            prepare_data(df, "nonexistent_col", ft)


# ──────────────────────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────────────────────

class TestModelFactory:
    def test_get_available_models(self):
        from ml.model_factory import get_available_models
        models = get_available_models()
        assert "logistic_regression" in models
        assert "random_forest" in models
        assert len(models) >= 8

    def test_build_logistic_regression(self):
        from ml.model_factory import build_model
        model = build_model("logistic_regression")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_random_forest(self):
        from ml.model_factory import build_model
        model = build_model("random_forest", {"n_estimators": 10})
        assert model.n_estimators == 10

    def test_unknown_model_raises(self):
        from ml.model_factory import build_model
        from utils.exceptions import ModelNotFoundError
        with pytest.raises(ModelNotFoundError):
            build_model("unicorn_model")

    def test_get_param_grid(self):
        from ml.model_factory import get_param_grid
        grid = get_param_grid("random_forest")
        assert "n_estimators" in grid

    @pytest.mark.parametrize("key", [
        "logistic_regression", "decision_tree", "random_forest",
        "naive_bayes", "knn",
    ])
    def test_all_models_build(self, key):
        from ml.model_factory import build_model
        model = build_model(key)
        assert model is not None


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_train_logistic_regression(self, iris_preprocessed):
        from ml.trainer import TrainingConfig, train_model
        X, y, le, _ = iris_preprocessed
        config = TrainingConfig(model_key="logistic_regression", test_size=0.2)
        result = train_model(X, y, config, class_names=list(le.classes_))
        assert result.metrics["accuracy"] > 0.5
        assert result.model is not None

    def test_train_random_forest(self, iris_preprocessed):
        from ml.trainer import TrainingConfig, train_model
        X, y, le, _ = iris_preprocessed
        config = TrainingConfig(
            model_key="random_forest",
            custom_params={"n_estimators": 20},
        )
        result = train_model(X, y, config)
        assert result.metrics["f1"] > 0

    def test_train_with_cv(self, iris_preprocessed):
        from ml.trainer import TrainingConfig, train_model
        X, y, le, _ = iris_preprocessed
        config = TrainingConfig(
            model_key="decision_tree",
            run_cv=True,
            cv_folds=3,
        )
        result = train_model(X, y, config)
        assert result.cv_results is not None
        assert "mean" in result.cv_results

    def test_train_multiple(self, iris_preprocessed):
        from ml.trainer import TrainingConfig, train_multiple
        X, y, _, _ = iris_preprocessed
        base = TrainingConfig(model_key="logistic_regression")
        results = train_multiple(X, y, ["logistic_regression", "decision_tree"], base)
        assert len(results) == 2
        assert all("accuracy" in r.metrics for r in results.values())


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_compute_metrics_keys(self, iris_preprocessed):
        from ml.metrics import compute_metrics
        from ml.model_factory import build_model
        from sklearn.model_selection import train_test_split

        X, y, le, _ = iris_preprocessed
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = build_model("logistic_regression")
        model.fit(X_tr, y_tr)
        m = compute_metrics(model, X_te, y_te, class_names=list(le.classes_))

        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in m

    def test_cross_validate(self, iris_preprocessed):
        from ml.metrics import cross_validate_model
        from ml.model_factory import build_model

        X, y, _, _ = iris_preprocessed
        model = build_model("decision_tree")
        cv = cross_validate_model(model, X, y, n_splits=3)
        assert cv["mean"] > 0
        assert len(cv["scores"]) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Tracker
# ──────────────────────────────────────────────────────────────────────────────

class TestExperimentTracker:
    def test_full_run_lifecycle(self, tmp_path, monkeypatch):
        monkeypatch.setattr("configs.settings.settings.EXPERIMENT_DIR", tmp_path)
        from experiments.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.experiment_name = "test_exp"
        tracker.experiment_dir = tmp_path / "test_exp"
        tracker.experiment_dir.mkdir()

        run_id = tracker.start_run("random_forest", dataset="iris.csv")
        assert (tracker.experiment_dir / f"{run_id}.json").exists()

        tracker.log_params(run_id, {"n_estimators": 100})
        tracker.log_metrics(run_id, {"accuracy": 0.95, "f1": 0.94})
        tracker.end_run(run_id)

        run = tracker.get_run(run_id)
        assert run["status"] == "FINISHED"
        assert run["metrics"]["accuracy"] == 0.95
        assert run["params"]["n_estimators"] == 100

    def test_list_runs(self, tmp_path):
        from experiments.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.experiment_name = "test_exp2"
        tracker.experiment_dir = tmp_path / "test_exp2"
        tracker.experiment_dir.mkdir()

        for model in ["lr", "rf"]:
            rid = tracker.start_run(model)
            tracker.log_metrics(rid, {"accuracy": 0.9})
            tracker.end_run(rid)

        runs = tracker.list_runs()
        assert len(runs) == 2

    def test_best_run(self, tmp_path):
        from experiments.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.experiment_name = "test_exp3"
        tracker.experiment_dir = tmp_path / "test_exp3"
        tracker.experiment_dir.mkdir()

        rid1 = tracker.start_run("lr")
        tracker.log_metrics(rid1, {"accuracy": 0.80})
        tracker.end_run(rid1)

        rid2 = tracker.start_run("rf")
        tracker.log_metrics(rid2, {"accuracy": 0.95})
        tracker.end_run(rid2)

        best = tracker.get_best_run("accuracy")
        assert best["run_id"] == rid2


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_register_and_load(self, tmp_path, monkeypatch, iris_preprocessed):
        monkeypatch.setattr("configs.settings.settings.MODEL_DIR", tmp_path)
        # Re-import to pick up monkeypatched path
        import importlib
        import experiments.model_registry as mr_mod
        monkeypatch.setattr(mr_mod, "_REGISTRY_ROOT", tmp_path)
        monkeypatch.setattr(mr_mod, "_REGISTRY_INDEX", tmp_path / "registry_index.json")

        reg = mr_mod.ModelRegistry()
        from ml.model_factory import build_model
        from sklearn.model_selection import train_test_split

        X, y, le, _ = iris_preprocessed
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2)
        model = build_model("logistic_regression")
        model.fit(X_tr, y_tr)

        version = reg.register(model, "logistic_regression", {"accuracy": 0.95}, {})
        assert version == "v1"

        loaded = reg.load("logistic_regression")
        assert hasattr(loaded, "predict")

    def test_leaderboard_ordering(self, tmp_path, monkeypatch, iris_preprocessed):
        monkeypatch.setattr("configs.settings.settings.MODEL_DIR", tmp_path)
        import experiments.model_registry as mr_mod
        monkeypatch.setattr(mr_mod, "_REGISTRY_ROOT", tmp_path)
        monkeypatch.setattr(mr_mod, "_REGISTRY_INDEX", tmp_path / "ri2.json")

        reg = mr_mod.ModelRegistry()
        from ml.model_factory import build_model
        from sklearn.model_selection import train_test_split

        X, y, _, _ = iris_preprocessed
        X_tr, _, y_tr, _ = train_test_split(X, y)

        for key, acc in [("logistic_regression", 0.80), ("decision_tree", 0.95)]:
            m = build_model(key)
            m.fit(X_tr, y_tr)
            reg.register(m, key, {"accuracy": acc}, {})

        board = reg.leaderboard("accuracy")
        assert board[0]["model_key"] == "decision_tree"


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (smoke tests – no assertion on visuals, just no crash)
# ──────────────────────────────────────────────────────────────────────────────

class TestVisualization:
    def test_correlation_heatmap(self):
        from visualization.plot_engine import plot_correlation_heatmap
        df = pd.DataFrame(np.random.randn(50, 4), columns=list("ABCD"))
        fig = plot_correlation_heatmap(df)
        assert fig is not None

    def test_class_balance(self):
        from visualization.plot_engine import plot_class_balance
        fig = plot_class_balance({"A": 30, "B": 20, "C": 10})
        assert fig is not None

    def test_confusion_matrix(self):
        from visualization.plot_engine import plot_confusion_matrix
        cm = [[40, 3], [2, 35]]
        fig = plot_confusion_matrix(cm, ["neg", "pos"])
        assert fig is not None

    def test_feature_importance(self):
        from visualization.plot_engine import plot_feature_importance
        imp = np.array([0.3, 0.5, 0.2, 0.1])
        fig = plot_feature_importance(imp, ["a", "b", "c", "d"])
        assert fig is not None

    def test_model_comparison(self):
        from visualization.plot_engine import plot_model_comparison
        results = {
            "lr": {"accuracy": 0.9, "f1": 0.88, "precision": 0.87, "recall": 0.89, "roc_auc": 0.95},
            "rf": {"accuracy": 0.95, "f1": 0.94, "precision": 0.93, "recall": 0.95, "roc_auc": 0.98},
        }
        fig = plot_model_comparison(results)
        assert fig is not None


# ──────────────────────────────────────────────────────────────────────────────
# Integration
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_iris(self, iris_csv, tmp_path, monkeypatch):
        """End-to-end: load → preprocess → train → metrics → registry."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        monkeypatch.setattr("configs.settings.settings.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("configs.settings.settings.EXPERIMENT_DIR", tmp_path / "experiments")
        (tmp_path / "models").mkdir()
        (tmp_path / "experiments").mkdir()

        import experiments.model_registry as mr_mod
        monkeypatch.setattr(mr_mod, "_REGISTRY_ROOT", tmp_path / "models")
        monkeypatch.setattr(mr_mod, "_REGISTRY_INDEX", tmp_path / "models" / "idx.json")

        from ml.dataset_loader import detect_feature_types, load_dataset
        from ml.preprocessing import get_feature_names_out, prepare_data
        from ml.trainer import TrainingConfig, train_model
        from ml.metrics import compute_metrics

        df = load_dataset(iris_csv)
        ft = detect_feature_types(df)
        X, y, le, ct = prepare_data(df, "target", ft)

        config = TrainingConfig(model_key="random_forest", custom_params={"n_estimators": 10})
        result = train_model(X, y, config, class_names=list(le.classes_))

        assert result.metrics["accuracy"] > 0.5
        assert result.model is not None
        assert result.metrics["confusion_matrix"] is not None
