# 🧪 Klassify – Multi-Model ML Experimentation Platform

> A production-grade, interactive machine learning experimentation environment.
> Train, compare, explain, and deploy classifiers – all in one platform.

---

## ✨ Features

| Capability | Details |
|---|---|
| **Dataset Management** | CSV upload, schema detection, profiling, missing value analysis |
| **EDA** | Correlation heatmaps, distributions, class balance, PCA, outlier plots |
| **10 Classifiers** | LR, KNN, SVM, Decision Tree, Random Forest, GBM, XGBoost, LightGBM, Naive Bayes, MLP |
| **Hyperparameter Tuning** | Grid Search, Random Search, Bayesian Optimization (scikit-optimize) |
| **Cross-Validation** | Stratified k-fold with per-fold score charts |
| **Experiment Tracking** | MLflow-inspired file-backed tracker with run history & comparisons |
| **Model Registry** | Versioned artifact storage, leaderboard, joblib/ONNX export |
| **Explainability** | SHAP values, feature importance, partial dependence |
| **REST API** | Full FastAPI backend with async Celery training jobs |
| **Interactive UI** | Multi-page Streamlit frontend with dark theme |
| **Docker** | Multi-stage Dockerfile + Docker Compose (api + frontend + worker + redis) |

---

## 🚀 Quick Start

### Local development

```bash
# 1. Clone & install
git clone https://github.com/your-org/klassify.git
cd klassify
pip install -r requirements.txt

# 2. Start the API
uvicorn backend.main:app --reload --port 8000

# 3. Start the UI
streamlit run frontend/streamlit_app.py

# 4. (Optional) Start async worker
celery -A backend.celery_worker worker --loglevel=info
```

### Docker (recommended for production)

```bash
cd deployment
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| Celery Flower | http://localhost:5555 (add `--profile monitoring`) |

---

## 🗂 Project Structure

```
klassify/
├── backend/
│   ├── main.py              # FastAPI app + all REST endpoints
│   ├── training_service.py  # Orchestration layer
│   └── celery_worker.py     # Async Celery tasks
├── ml/
│   ├── dataset_loader.py    # Load, cache, profile datasets
│   ├── preprocessing.py     # sklearn ColumnTransformer pipeline
│   ├── model_factory.py     # Dynamic model instantiation from YAML
│   ├── trainer.py           # Training + hyperparameter tuning
│   └── metrics.py           # Full metrics suite
├── experiments/
│   ├── experiment_tracker.py  # File-backed MLflow-style tracker
│   └── model_registry.py      # Versioned model artifact store
├── visualization/
│   ├── plot_engine.py       # All Plotly charts
│   └── shap_explainer.py    # SHAP integration
├── frontend/
│   └── streamlit_app.py     # Multi-page Streamlit UI
├── configs/
│   ├── settings.py          # Pydantic settings (env-driven)
│   └── model_configs.yaml   # Model hyperparameter registry
├── utils/
│   ├── logger.py            # Structured logging
│   └── exceptions.py        # Typed exception hierarchy
├── tests/
│   └── test_core.py         # Pytest suite (unit + integration)
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/upload_dataset` | Upload a CSV dataset |
| `GET` | `/api/v1/dataset/{id}/profile` | Full dataset profiling report |
| `GET` | `/api/v1/dataset/{id}/preview` | First N rows |
| `POST` | `/api/v1/train_model` | Synchronous single-model training |
| `POST` | `/api/v1/train_model/async` | Submit async training (Celery) |
| `GET` | `/api/v1/tasks/{task_id}` | Poll async task status |
| `POST` | `/api/v1/run_experiment` | Train all selected models |
| `GET` | `/api/v1/models` | List available classifiers |
| `GET` | `/api/v1/experiments` | List all experiments |
| `GET` | `/api/v1/experiments/{name}/runs` | List runs in an experiment |
| `GET` | `/api/v1/registry/leaderboard` | Model leaderboard |
| `GET` | `/api/v1/registry/models/{key}/export` | Download model artifact |
| `POST` | `/api/v1/predict` | Run inference |

Full interactive docs at **http://localhost:8000/docs**

---

## ⚙️ Configuration

All settings are driven by environment variables (or a `.env` file):

```env
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=100
REDIS_URL=redis://localhost:6379/0
DEFAULT_CV_FOLDS=5
DEFAULT_TEST_SIZE=0.2
```

See `configs/settings.py` for the full list.

---

## 🧪 Running Tests

```bash
pytest tests/test_core.py -v
```

Covers: dataset loader, preprocessing, model factory, trainer, metrics, experiment tracker, model registry, visualizations, and end-to-end pipeline integration.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────┐
│          Streamlit Frontend              │
│  (EDA · Train · Results · Leaderboard)  │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│           FastAPI REST API               │
│  /upload · /train · /predict · /export  │
└──────────────────┬───────────────────────┘
                   │
       ┌───────────┼────────────┐
       │           │            │
┌──────▼──┐  ┌────▼────┐  ┌───▼────────┐
│ Training │  │Experiment│  │  Registry  │
│ Service  │  │ Tracker  │  │  (joblib)  │
└──────┬──┘  └─────────┘  └────────────┘
       │
┌──────▼──────────────────────────────────┐
│          ML Domain Layer                 │
│  loader · preprocessing · trainer ·     │
│  model_factory · metrics                │
└──────────────────────────────────────────┘
       │
┌──────▼───────────────┐
│  Celery + Redis       │
│  (async training)     │
└───────────────────────┘
```

---

## 🗺 Roadmap

- [ ] AutoML mode (train all models, pick best automatically)
- [ ] Regression task support
- [ ] Online model serving endpoint
- [ ] PostgreSQL-backed experiment store
- [ ] S3 / GCS artifact storage backend
- [ ] Prometheus metrics endpoint
- [ ] LLM-powered dataset insights

---

## 📄 License

MIT © Klassify Contributors
