# Klassify — ML Experimentation Platform

> An end-to-end classification platform. Upload a CSV, explore your data, train and compare 10 classifiers, explain every prediction — no boilerplate required.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34%2B-FF4B4B?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Using the UI](#using-the-ui)
- [REST API Reference](#rest-api-reference)
- [Supported Classifiers](#supported-classifiers)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Running Tests](#running-tests)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

**Klassify** is a production-grade machine learning experimentation platform built for data scientists and ML practitioners who need to iterate fast.

It combines an interactive **Streamlit** frontend with a **FastAPI** REST backend, a file-backed **experiment tracker**, a versioned **model registry**, and **SHAP** explainability — all wired together into a single coherent workflow.

You can use Klassify entirely through the browser UI, or drive it programmatically through the REST API for integration into existing pipelines.

---

## Features

### Dataset Management
- Upload CSV files up to 100 MB via drag-and-drop
- Automatic column-type detection: `numeric`, `categorical`, `datetime`, `text`
- Full dataset profile: row/column counts, missing values, duplicates, memory usage
- Per-column statistics: mean, std, quartiles, skewness, kurtosis, top values
- Configurable preprocessing: scaler choice, categorical encoding, target selection

### Exploratory Data Analysis
- **Distributions** — histogram grid for all numeric features (up to 16 columns)
- **Correlation** — Pearson heatmap to detect multicollinearity
- **Class Balance** — bar chart of target class frequencies
- **PCA** — 2-D PCA scatter coloured by class (post-preprocessing)
- **Outliers** — interactive box plots per feature

All charts are fully interactive via Plotly (zoom, pan, hover, toggle series).

### Model Training
- Train up to **10 classifiers** in a single run
- Configurable test split (10%–40%) with stratified train/test partitioning
- Stratified **k-fold cross-validation** (2–10 folds) with per-fold score charts
- Three **hyperparameter tuning** strategies:
  - `grid` — exhaustive GridSearchCV
  - `random` — RandomizedSearchCV
  - `bayesian` — BayesSearchCV via scikit-optimize
- Sequential training with live progress bar and per-model status

### Model Evaluation
- Performance summary table: Accuracy, F1, Precision, Recall, ROC-AUC, training time
- Side-by-side grouped bar chart for visual comparison
- Per-model: confusion matrix, ROC curves, PR curves, feature importance
- Decision boundary visualisation (PCA-projected for high-dimensional data)
- CV fold score charts with mean, std, min, max

### Explainability (SHAP)
- Auto-selects the best SHAP explainer per model type:
  - Tree models → `TreeExplainer` (fast)
  - Linear models → `LinearExplainer`
  - Others (SVM, KNN, MLP) → `KernelExplainer`
- Mean absolute SHAP bar chart ranked by feature impact
- Full SHAP value table for further analysis

### Experiment Tracking
- Every training run logged automatically as a JSON file
- Tracks: model key, params, metrics, artifact paths, status, timestamps
- Experiment selector, runs table, best-run badge, raw JSON inspector
- `list_experiments()`, `compare_runs()`, `get_best_run()` API

### Model Registry
- Versioned artifact storage (v1, v2, …) via `joblib`
- Registry index (`registry_index.json`) with full metadata per version
- Leaderboard sortable by any metric across all versions
- ONNX export for portable deployment

### REST API
- Full **FastAPI** backend — all UI capabilities available over HTTP
- Async training via **Celery + Redis** with task polling
- Swagger docs auto-generated at `/docs`

---

## Project Structure

```
klassify/
├── backend/
│   ├── main.py                # FastAPI app — all REST endpoints
│   ├── training_service.py    # Orchestration: dataset → preprocess → train → register
│   └── celery_worker.py       # Async Celery tasks (train, experiment, health)
│
├── ml/
│   ├── dataset_loader.py      # Load (CSV/Parquet/JSON), cache, profile, detect types
│   ├── preprocessing.py       # ColumnTransformer pipeline (impute → scale → encode)
│   ├── model_factory.py       # Dynamic model instantiation from model_configs.yaml
│   ├── trainer.py             # Training, tuning (grid/random/bayesian), CV
│   └── metrics.py             # Accuracy, F1, ROC-AUC, confusion matrix, curves
│
├── experiments/
│   ├── experiment_tracker.py  # File-backed MLflow-style run tracker
│   └── model_registry.py      # Versioned model store + leaderboard
│
├── visualization/
│   ├── plot_engine.py         # All Plotly charts (heatmap, ROC, PCA, boundary, …)
│   └── shap_explainer.py      # SHAP value computation + summary
│
├── frontend/
│   └── streamlit_app.py       # Multi-page Streamlit UI (Home, Dataset, EDA, …)
│
├── configs/
│   ├── settings.py            # Pydantic settings — all env-driven
│   └── model_configs.yaml     # 10 classifiers with default params + tuning grids
│
├── utils/
│   ├── logger.py              # Structured stdout logger
│   └── exceptions.py          # Typed exception hierarchy (KlassifyError subtypes)
│
├── data/
│   ├── uploads/               # Uploaded dataset files (temp)
│   ├── experiments/           # Run JSON files per experiment
│   └── models/                # Versioned model artifacts + registry_index.json
│
├── tests/
│   └── test_core.py           # Pytest suite: unit + integration
│
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml     # api + frontend + worker + redis
│
├── .streamlit/
│   └── config.toml            # Streamlit dark theme config
│
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (only needed for async Celery training — UI works without it)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/afeefaquadri-png/klassify.git
cd klassify/klassify

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Streamlit UI
streamlit run frontend/streamlit_app.py

# 4. (Optional) Start the FastAPI backend
uvicorn backend.main:app --reload --port 8000

# 5. (Optional) Start the Celery async worker
celery -A backend.celery_worker worker --loglevel=info
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI (Swagger docs) | http://localhost:8000/docs |
| FastAPI (ReDoc) | http://localhost:8000/redoc |

### Docker

```bash
cd deployment
docker compose up --build
```

Services started: `api` (port 8000), `frontend` (port 8501), `worker` (Celery), `redis`.

---

## Using the UI

The Streamlit app is a multi-page interface. Use the sidebar to navigate between pages. The **Session Status** panel tracks your progress through the workflow.

### 1 · Dataset

1. Click **Browse files** and upload a `.csv` file (max 100 MB).
2. Review the auto-generated profile — row/column counts, missing cells, duplicates.
3. Inspect the **Column Details** table (type, missing %, unique count).
4. Select a **target column**, **scaler**, and **encoding** strategy.
5. Click **Preprocess Data** — the success banner confirms the feature matrix shape and class list.

> High-cardinality categorical columns (> 50 unique values) are automatically dropped.

### 2 · EDA

Navigate to **EDA** after uploading a dataset. Five tabs are available:

| Tab | Contents |
|---|---|
| Distributions | Histogram grid for numeric features |
| Correlation | Pearson heatmap — spot multicollinearity |
| Class Balance | Target class frequency bar chart |
| PCA | 2-D scatter coloured by class *(requires preprocessing)* |
| Outliers | Box plot for any selected numeric feature |

### 3 · Train

> Preprocessing must be complete before training.

1. Select one or more classifiers from the multiselect dropdown.
2. Choose a **tuning strategy**: `none` (use defaults), `grid`, or `random`.
3. Toggle **Cross-validation** on/off and set the number of folds (2–10).
4. Set the **test split** fraction (0.10–0.40).
5. Enter an **experiment name** to group this run in the log.
6. Click **Train Selected Models**.

Each model trains sequentially with a live progress bar. Results are logged and the model is versioned automatically.

### 4 · Results

- **Performance Summary** table — all models side-by-side.
- **Comparison chart** — grouped bar chart across metrics.
- **Model inspector** — select a model to view:
  - Confusion matrix
  - Feature importance (or coefficient magnitudes)
  - ROC and PR curves
  - Decision boundary (PCA-projected)
- **CV panel** (if cross-validation was run) — fold scores, mean, std, min, max.

### 5 · Explainability

1. Select a trained model.
2. Click **Compute SHAP Values**.
3. View the mean |SHAP| feature importance bar chart and summary table.

> KernelExplainer (used for SVM, KNN, MLP) can take 30–60 s on large datasets.

### 6 · Leaderboard

Ranks all registered model versions by the selected metric (accuracy, F1, ROC-AUC, precision, recall). Re-training after tuning creates a new version — compare them here.

### 7 · Experiment Log

Browse all recorded runs. Select an experiment name, view the runs table, see the best-run badge, and inspect any run's raw JSON detail.

---

## REST API Reference

All endpoints are prefixed with `/api/v1`. Interactive docs: **http://localhost:8000/docs**

### Dataset

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload_dataset` | Upload a CSV/Parquet/JSON file. Returns `dataset_id`. |
| `GET` | `/dataset/{id}/profile` | Full profiling report (types, stats, missing). |
| `GET` | `/dataset/{id}/preview` | First N rows as JSON (default N=20). |
| `GET` | `/dataset/{id}/class_distribution` | Target column value counts. |

### Training

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/train_model` | Synchronous single-model training. |
| `POST` | `/train_model/async` | Submit async training job. Returns `task_id`. |
| `GET` | `/tasks/{task_id}` | Poll async task status and result. |
| `POST` | `/run_experiment` | Train multiple models sequentially. |

**Train request body:**

```json
{
  "dataset_id": "abc123",
  "target_col": "species",
  "model_key": "random_forest",
  "experiment_name": "iris_baseline",
  "tuning_strategy": "random",
  "tuning_n_iter": 20,
  "run_cv": true,
  "cv_folds": 5,
  "test_size": 0.2,
  "scaler": "standard",
  "encoding": "onehot",
  "custom_params": {}
}
```

### Models & Registry

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/models` | List available classifier keys + display names. |
| `GET` | `/registry/models` | List all registered model keys. |
| `GET` | `/registry/models/{key}/versions` | Version history for a model. |
| `GET` | `/registry/leaderboard` | All versions ranked by metric. |
| `GET` | `/registry/models/{key}/export` | Download model (joblib or ONNX). |

### Experiments

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/experiments` | List all experiment names. |
| `GET` | `/experiments/{name}/runs` | All runs in an experiment. |
| `GET` | `/experiments/{name}/runs/{run_id}` | Single run detail. |
| `GET` | `/experiments/{name}/best` | Best run by metric. |

### Inference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Run inference with a registered model. |

**Predict request body:**

```json
{
  "model_key": "random_forest",
  "version": "v2",
  "data": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
}
```

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check. |

---

## Supported Classifiers

| Key | Classifier | Library |
|---|---|---|
| `logistic_regression` | Logistic Regression | scikit-learn |
| `knn` | K-Nearest Neighbours | scikit-learn |
| `svm` | Support Vector Machine | scikit-learn |
| `decision_tree` | Decision Tree | scikit-learn |
| `random_forest` | Random Forest | scikit-learn |
| `gradient_boosting` | Gradient Boosting | scikit-learn |
| `xgboost` | XGBoost | xgboost |
| `lightgbm` | LightGBM | lightgbm |
| `naive_bayes` | Naive Bayes (Gaussian) | scikit-learn |
| `mlp` | Neural Network (MLP) | scikit-learn |

All models are defined in `configs/model_configs.yaml` with default params, Grid Search grids, and Random/Bayesian search distributions. Adding a new classifier requires only a YAML entry — no code changes.

---

## Configuration

All settings are driven by environment variables or a `.env` file. See `configs/settings.py` for the full list.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |
| `UPLOAD_DIR` | `data/uploads` | Uploaded file storage |
| `MODEL_DIR` | `data/models` | Model artifact storage |
| `EXPERIMENT_DIR` | `data/experiments` | Experiment JSON storage |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum upload size |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Celery result backend |
| `DEFAULT_TEST_SIZE` | `0.2` | Default train/test split |
| `DEFAULT_CV_FOLDS` | `5` | Default CV folds |
| `DEFAULT_RANDOM_STATE` | `42` | Global random seed |
| `MAX_TRAINING_TIME_SECONDS` | `3600` | Training timeout |

---

## Architecture

```
┌────────────────────────────────────────────────┐
│               Streamlit Frontend               │
│   Home · Dataset · EDA · Train · Results       │
│   Leaderboard · Explainability · Experiment Log│
└───────────────────────┬────────────────────────┘
                        │ HTTP / in-process
┌───────────────────────▼────────────────────────┐
│              FastAPI REST API                  │
│  /upload · /train · /predict · /export · /health│
└──────┬──────────────┬──────────────────────────┘
       │              │
       │        ┌─────▼──────┐
       │        │   Celery   │ ← Redis broker
       │        │   Worker   │
       │        └─────┬──────┘
       │              │
┌──────▼──────────────▼──────────────────────────┐
│             Training Service                   │
│  load → detect types → preprocess → tune →    │
│  train → evaluate → track → register          │
└──────┬──────────────┬──────────────────────────┘
       │              │
┌──────▼──────┐ ┌─────▼────────┐ ┌──────────────┐
│  ML Domain  │ │  Experiment  │ │    Model     │
│  loader     │ │   Tracker    │ │   Registry   │
│  preprocess │ │  (JSON/disk) │ │ (joblib/ONNX)│
│  trainer    │ └──────────────┘ └──────────────┘
│  metrics    │
└─────────────┘
```

**Key design decisions:**

- **File-backed persistence** — no database dependency; experiments as JSON, models as joblib files.
- **Sklearn-centric** — all classifiers implement `ClassifierMixin`, enabling a uniform pipeline.
- **Lazy imports** — XGBoost, LightGBM, SHAP imported only when needed for graceful degradation.
- **Config-driven models** — adding a classifier requires only a YAML entry, no code changes.
- **Sync default, async optional** — UI calls are synchronous; async training available via Celery for API consumers.

---

## Running Tests

```bash
pytest tests/test_core.py -v
```

The test suite covers:

- Dataset loader (CSV loading, type detection, profiling, caching)
- Preprocessing pipeline (imputation, scaling, encoding, feature name extraction)
- Model factory (YAML loading, dynamic instantiation, param grid retrieval)
- Trainer (train/test split, cross-validation, hyperparameter tuning)
- Metrics (accuracy, F1, ROC-AUC, confusion matrix, classification report)
- Experiment tracker (start/log/end run, list, best run, comparison)
- Model registry (register, load, versioning, leaderboard)
- Visualisations (plot functions return valid Plotly figures)
- End-to-end integration (upload → preprocess → train → evaluate → register)

---

## Roadmap

- [ ] AutoML mode — train all models and auto-select the best
- [ ] Regression task support
- [ ] Multi-label classification
- [ ] Online model serving endpoint
- [ ] PostgreSQL-backed experiment store
- [ ] S3 / GCS artifact storage backend
- [ ] Prometheus metrics + Grafana dashboard
- [ ] LLM-powered dataset insights and model recommendations
- [ ] Differential privacy training options

---

## License

MIT © Klassify Contributors
