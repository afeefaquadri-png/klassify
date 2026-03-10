"""
Klassify – Streamlit Frontend
═══════════════════════════════════════════════════════════════════
Interactive ML experimentation platform.

Run:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import os

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://klassify-1.onrender.com"
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klassify",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ───────── Main App Container ───────── */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ───────── Sidebar Styling ───────── */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #181825 0%, #11111b 100%);
    border-right: 1px solid #313244;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #cdd6f4 !important;
}

/* ───────── Headings ───────── */

h1, h2, h3 {
    color: #cdd6f4;
    font-weight: 600;
}

h1 {
    border-bottom: 1px solid #313244;
    padding-bottom: 0.3rem;
}

/* ───────── Metric Cards ───────── */

.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #313244 100%);
    border: 1px solid #45475a;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    transition: all 0.2s ease-in-out;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

/* Streamlit metric component */

[data-testid="stMetric"] {
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 0.7rem 1rem;
}

/* Metric label */

[data-testid="stMetricLabel"] {
    color: #a6adc8 !important;
}

/* Metric value */

[data-testid="stMetricValue"] {
    color: #cdd6f4 !important;
}

/* ───────── Buttons ───────── */

.stButton>button {
    background: linear-gradient(135deg, #89b4fa, #74c7ec);
    color: #11111b;
    border-radius: 6px;
    border: none;
    font-weight: 600;
    padding: 0.4rem 1rem;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #74c7ec, #89b4fa);
    transform: scale(1.03);
}

/* ───────── Dataframes / Tables ───────── */

[data-testid="stDataFrame"] {
    border: 1px solid #313244;
    border-radius: 8px;
    overflow: hidden;
}

/* ───────── Chart Containers ───────── */

.chart-container {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid #45475a;
}

/* ───────── File Upload Area ───────── */

[data-testid="stFileUploader"] {
    border: 1px dashed #45475a;
    border-radius: 8px;
    padding: 1rem;
}

/* ───────── Input Fields ───────── */

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: #1e1e2e !important;
    color: #cdd6f4 !important;
    border-radius: 6px !important;
}

/* ───────── Footer Fix ───────── */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "df": None,
        "dataset_path": None,
        "target_col": None,
        "feature_types": None,
        "X": None,
        "y": None,
        "le": None,
        "ct": None,
        "feature_names": None,
        "training_results": {},     # {model_key: TrainingResult}
        "experiment_summaries": {}, # {model_key: summary_dict}
        "experiment_name": "default_experiment",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧪 Klassify")
    st.markdown("*ML Experimentation Platform*")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📂  Dataset",
            "🔍  EDA",
            "🏋️  Train",
            "📊  Results",
            "🏆  Leaderboard",
            "🔬  Explainability",
            "📜  Experiment Log",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("v1.0.0  ·  Klassify")

page_key = page.split("  ")[1]


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Dataset
# ──────────────────────────────────────────────────────────────────────────────

if page_key == "Dataset":
    st.title("📂 Dataset Management")

    col_up, col_info = st.columns([1, 2])

    with col_up:
        st.subheader("Upload")
        uploaded = st.file_uploader(
            "Drop a CSV file here",
            type=["csv"],
            help="CSV files up to 100 MB",
        )
        if uploaded:
            import tempfile, os
            from ml.dataset_loader import detect_feature_types, load_dataset, profile_dataset

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)

            df = load_dataset(tmp_path)
            ft = detect_feature_types(df)
            st.session_state.df = df
            st.session_state.dataset_path = tmp_path
            st.session_state.feature_types = ft
            # Reset downstream
            st.session_state.X = None
            st.session_state.training_results = {}
            st.success(f"✅ Loaded **{uploaded.name}** – {len(df):,} rows × {df.shape[1]} cols")

    if st.session_state.df is not None:
        df = st.session_state.df
        ft = st.session_state.feature_types

        with col_info:
            st.subheader("Preview")
            st.dataframe(df.head(10), use_container_width=True)

        st.divider()
        st.subheader("Dataset Profile")

        from ml.dataset_loader import profile_dataset
        profile = profile_dataset(df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{profile['n_rows']:,}")
        m2.metric("Columns", profile["n_columns"])
        m3.metric("Missing cells", f"{profile['total_missing']:,}")
        m4.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

        st.divider()
        st.subheader("Column Details")
        col_df = pd.DataFrame([
            {
                "Column": col,
                "Type": info["feature_type"],
                "Dtype": info["dtype"],
                "Missing %": f"{info['missing_pct']}%",
                "Unique": info["unique_count"],
            }
            for col, info in profile["columns"].items()
        ])
        st.dataframe(col_df, use_container_width=True)

        st.divider()
        st.subheader("⚙️ Preprocessing Setup")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            target_col = st.selectbox("Target column", options=list(df.columns))
            st.session_state.target_col = target_col
        with pcol2:
            scaler = st.selectbox("Scaler", ["standard", "minmax", "robust"])
        with pcol3:
            encoding = st.selectbox("Categorical encoding", ["onehot", "ordinal"])

        if st.button("🔄 Preprocess Data", type="primary"):
            from ml.preprocessing import get_feature_names_out, prepare_data
            with st.spinner("Preprocessing…"):
                try:
                    X, y, le, ct = prepare_data(
                        df, target_col, ft,
                        scaler=scaler, encoding=encoding,
                    )
                    num_cols = [c for c, t in ft.items() if t == "numeric" and c != target_col and c in df.columns]
                    cat_cols = [c for c, t in ft.items() if t == "categorical" and c != target_col and c in df.columns]
                    feature_names = get_feature_names_out(ct, num_cols, cat_cols)
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.le = le
                    st.session_state.ct = ct
                    st.session_state.feature_names = feature_names
                    st.success(f"✅ Ready: {X.shape[0]:,} samples × {X.shape[1]} features | Classes: {list(le.classes_)}")
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first (Dataset page).")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distributions", "Correlation", "Class Balance", "PCA", "Outliers"
    ])

    from visualization.plot_engine import (
        plot_class_balance, plot_correlation_heatmap,
        plot_feature_distributions, plot_pca_scatter,
    )

    with tab1:
        fig = plot_feature_distributions(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        target = st.session_state.target_col or df.columns[-1]
        target = st.selectbox("Target column", df.columns,
                              index=list(df.columns).index(target) if target in df.columns else 0)
        from ml.dataset_loader import get_class_distribution
        dist = get_class_distribution(df, target)
        fig = plot_class_balance(dist)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if st.session_state.X is not None:
            fig = plot_pca_scatter(
                st.session_state.X, st.session_state.y,
                class_names=list(st.session_state.le.classes_) if st.session_state.le else None,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run preprocessing first to enable PCA view.")

    with tab5:
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            import plotly.express as px
            col = st.selectbox("Feature", numeric.columns)
            fig = px.box(df, y=col, title=f"Box plot – {col}")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Train
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "Train":
    st.title("🏋️ Model Training")

    if st.session_state.X is None:
        st.warning("Complete preprocessing on the Dataset page first.")
        st.stop()

    from ml.model_factory import get_model_display_names
    display_names = get_model_display_names()

    st.subheader("Select Models")
    selected_models = st.multiselect(
        "Models to train",
        options=list(display_names.keys()),
        default=["logistic_regression", "random_forest", "gradient_boosting"],
        format_func=lambda k: display_names[k],
    )

    st.divider()
    st.subheader("Training Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        tuning = st.selectbox("Hyperparameter tuning", ["none", "grid", "random"])
        tuning_strategy = None if tuning == "none" else tuning
    with col2:
        run_cv = st.checkbox("Cross-validation", value=True)
        cv_folds = st.slider("CV folds", 2, 10, 5) if run_cv else 5
    with col3:
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
        exp_name = st.text_input("Experiment name", value=st.session_state.experiment_name)
        st.session_state.experiment_name = exp_name

    if st.button("🚀 Train Selected Models", type="primary", disabled=not selected_models):
        from ml.trainer import TrainingConfig, train_model
        from experiments.experiment_tracker import get_tracker
        from experiments.model_registry import registry

        tracker = get_tracker(exp_name)
        X = st.session_state.X
        y = st.session_state.y
        class_names = list(st.session_state.le.classes_)
        feature_names = st.session_state.feature_names

        progress = st.progress(0)
        status = st.empty()
        results = {}

        for i, key in enumerate(selected_models):
            status.info(f"Training **{display_names[key]}** ({i+1}/{len(selected_models)})…")
            config = TrainingConfig(
                model_key=key,
                test_size=test_size,
                tuning_strategy=tuning_strategy,
                run_cv=run_cv,
                cv_folds=cv_folds,
            )
            run_id = tracker.start_run(key, dataset=str(st.session_state.dataset_path))
            try:
                result = train_model(X, y, config, class_names, feature_names)
                scalar_m = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))}
                tracker.log_params(run_id, result.best_params)
                tracker.log_metrics(run_id, scalar_m)
                tracker.end_run(run_id)
                version = registry.register(
                    result.model, key, scalar_m, result.best_params,
                    experiment_name=exp_name, run_id=run_id,
                )
                results[key] = result
                st.session_state.training_results[key] = result
                st.success(f"✅ {display_names[key]}  accuracy={result.metrics['accuracy']:.4f}")
            except Exception as e:
                tracker.end_run(run_id, "FAILED")
                st.error(f"❌ {display_names[key]}: {e}")
            progress.progress((i + 1) / len(selected_models))

        status.success("🎉 Training complete!")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Results
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "Results":
    st.title("📊 Model Results")
    results = st.session_state.training_results
    if not results:
        st.warning("Train some models first.")
        st.stop()

    from visualization.plot_engine import (
        plot_confusion_matrix, plot_model_comparison,
        plot_roc_curves, plot_pr_curves, plot_feature_importance,
        plot_decision_boundary,
    )
    from ml.model_factory import get_model_display_names
    names = get_model_display_names()

    # ── Summary table ─────────────────────────────────────────────────────
    st.subheader("Performance Summary")
    rows = []
    for key, res in results.items():
        m = res.metrics
        rows.append({
            "Model": names.get(key, key),
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc') or 0:.4f}",
            "Train (s)": f"{m.get('training_time_s', 0):.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Comparison chart ──────────────────────────────────────────────────
    metrics_map = {key: res.metrics for key, res in results.items()}
    fig = plot_model_comparison(metrics_map)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # ── Per-model details ─────────────────────────────────────────────────
    selected = st.selectbox("Inspect model", list(results.keys()), format_func=lambda k: names.get(k, k))
    res = results[selected]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(res.metrics["confusion_matrix"], res.class_names)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        st.subheader("Feature Importance")
        model = res.model
        fn = res.feature_names or [f"f{i}" for i in range(st.session_state.X.shape[1])]
        if hasattr(model, "feature_importances_"):
            fig_fi = plot_feature_importance(model.feature_importances_, fn)
            st.plotly_chart(fig_fi, use_container_width=True)
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
            fig_fi = plot_feature_importance(imp, fn)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

    st.divider()
    tab_roc, tab_pr, tab_db = st.tabs(["ROC Curves", "PR Curves", "Decision Boundary"])
    with tab_roc:
        fig = plot_roc_curves(metrics_map)
        st.plotly_chart(fig, use_container_width=True)
    with tab_pr:
        fig = plot_pr_curves(metrics_map)
        st.plotly_chart(fig, use_container_width=True)
    with tab_db:
        if st.session_state.X is not None:
            with st.spinner("Rendering decision boundary…"):
                fig = plot_decision_boundary(
                    res.model, st.session_state.X, st.session_state.y,
                    feature_names=res.feature_names,
                    class_names=res.class_names,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── CV results ────────────────────────────────────────────────────────
    if res.cv_results:
        st.divider()
        st.subheader("Cross-Validation Results")
        cv = res.cv_results
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV Mean", f"{cv['mean']:.4f}")
        m2.metric("CV Std", f"{cv['std']:.4f}")
        m3.metric("CV Min", f"{cv['min']:.4f}")
        m4.metric("CV Max", f"{cv['max']:.4f}")
        import plotly.graph_objects as go
        fig_cv = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv['scores']))],
            y=cv["scores"],
            marker_color="#89B4FA",
        ))
        fig_cv.update_layout(title=f"CV Scores ({cv['scoring']})", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_cv, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Leaderboard
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "Leaderboard":
    st.title("🏆 Model Leaderboard")
    from experiments.model_registry import registry

    metric = st.selectbox("Rank by", ["accuracy", "f1", "roc_auc", "precision", "recall"])
    board = registry.leaderboard(metric=metric)

    if not board:
        st.info("No registered models yet. Train some models first.")
    else:
        df_board = pd.DataFrame(board)
        st.dataframe(df_board, use_container_width=True)

        import plotly.express as px
        if metric in df_board.columns:
            fig = px.bar(df_board.head(10), x="model_key", y=metric,
                         color="model_key", title=f"Top 10 Models by {metric}")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Explainability
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "Explainability":
    st.title("🔬 Model Explainability")
    results = st.session_state.training_results
    if not results or st.session_state.X is None:
        st.warning("Train models and preprocess data first.")
        st.stop()

    from ml.model_factory import get_model_display_names
    names = get_model_display_names()
    selected = st.selectbox("Model", list(results.keys()), format_func=lambda k: names.get(k, k))
    model = results[selected].model
    feature_names = results[selected].feature_names or [f"f{i}" for i in range(st.session_state.X.shape[1])]

    if st.button("Compute SHAP Values", type="primary"):
        from visualization.shap_explainer import compute_shap_values, shap_summary_data
        with st.spinner("Computing SHAP values (this may take a moment)…"):
            shap_result = compute_shap_values(model, st.session_state.X, feature_names)

        if shap_result is None:
            st.warning("SHAP computation failed or shap package not installed.")
        else:
            df_shap = shap_summary_data(shap_result)
            st.subheader("Mean |SHAP| Feature Importance")
            import plotly.express as px
            fig = px.bar(
                df_shap.head(20), x="mean_abs_shap", y="feature",
                orientation="h", title="SHAP Feature Importance",
                color="mean_abs_shap", color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_shap, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Experiment Log
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "Experiment Log":
    st.title("📜 Experiment Log")
    from experiments.experiment_tracker import list_experiments, get_tracker

    experiments = list_experiments()
    if not experiments:
        st.info("No experiments recorded yet.")
        st.stop()

    exp_name = st.selectbox("Experiment", experiments)
    tracker = get_tracker(exp_name)
    runs = tracker.list_runs()

    if not runs:
        st.info("No runs for this experiment.")
    else:
        rows = []
        for r in runs:
            rows.append({
                "run_id": r["run_id"],
                "model": r["model_key"],
                "status": r["status"],
                "accuracy": r.get("metrics", {}).get("accuracy"),
                "f1": r.get("metrics", {}).get("f1"),
                "roc_auc": r.get("metrics", {}).get("roc_auc"),
                "started": r["start_time"],
            })
        df_runs = pd.DataFrame(rows)
        st.dataframe(df_runs, use_container_width=True)

        best = tracker.get_best_run("accuracy")
        if best:
            st.success(f"🥇 Best run: **{best['run_id']}** ({best['model_key']}) — accuracy {best['metrics'].get('accuracy', 'N/A'):.4f}")

        run_id = st.selectbox("Inspect run", [r["run_id"] for r in runs])
        run_detail = tracker.get_run(run_id)
        st.json(run_detail)
