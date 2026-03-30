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

/* ═══════════════════════════════════════
   GLOBAL DARK THEME ENFORCEMENT
   ═══════════════════════════════════════ */

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
.main, section.main {
    background-color: #11111b !important;
    color: #cdd6f4 !important;
}

/* Top toolbar */
[data-testid="stHeader"] {
    background: #11111b !important;
    border-bottom: 1px solid #1e1e2e;
}

/* Main content padding */
.block-container {
    padding: 2rem 2rem 3rem 2rem;
    max-width: 1200px;
}

/* ═══════════════════════════════════════
   TYPOGRAPHY & LINKS
   ═══════════════════════════════════════ */

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #cdd6f4 !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}

h1 { border-bottom: 1px solid #313244; padding-bottom: 0.4rem; }

p, li, span, label,
.stMarkdown p, .stMarkdown li {
    color: #cdd6f4 !important;
}

/* Kill Streamlit's default orange links */
a, .stMarkdown a {
    color: #89b4fa !important;
    text-decoration: none !important;
}
a:hover, .stMarkdown a:hover {
    color: #74c7ec !important;
    text-decoration: underline !important;
}

/* ═══════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════ */

section[data-testid="stSidebar"] {
    background: #181825 !important;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #cdd6f4 !important;
}

/* Sidebar radio active item */
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] {
    color: #89b4fa !important;
    font-weight: 600;
}

/* ═══════════════════════════════════════
   NATIVE STREAMLIT COMPONENTS
   ═══════════════════════════════════════ */

/* Metric cards */
[data-testid="stMetric"] {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #a6adc8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #cdd6f4 !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #a6e3a1 !important; }

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: #89b4fa;
    color: #11111b;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.5rem 1.4rem;
    letter-spacing: 0.01em;
    transition: all 0.18s ease;
}
.stButton > button[kind="primary"]:hover {
    background: #74c7ec;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(137,180,250,0.35);
}

/* Secondary buttons */
.stButton > button:not([kind="primary"]) {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.18s ease;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #89b4fa;
    color: #89b4fa;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1e1e2e;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #313244;
    gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #a6adc8 !important;
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.2rem;
    font-size: 0.88rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #89b4fa !important;
    border-bottom: 2px solid #89b4fa !important;
    background: #11111b !important;
    font-weight: 700;
}

/* Selectbox / input */
[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background: #1e1e2e !important;
    border-color: #313244 !important;
    color: #cdd6f4 !important;
    border-radius: 8px !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #89b4fa !important;
    border-color: #89b4fa !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1e1e2e;
    border: 1.5px dashed #45475a;
    border-radius: 10px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #89b4fa;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #313244;
    border-radius: 10px;
    overflow: hidden;
}

/* Alert / info boxes */
[data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 3px;
}

/* Divider */
hr { border-color: #313244 !important; margin: 1.5rem 0; }

/* Code blocks */
code, pre {
    background: #181825 !important;
    color: #cdd6f4 !important;
    border: 1px solid #313244 !important;
    border-radius: 6px;
}

/* ═══════════════════════════════════════
   LANDING PAGE COMPONENTS
   ═══════════════════════════════════════ */

/* Hero */
.hero-wrap {
    background: linear-gradient(145deg, #181825 0%, #1e1e2e 50%, #181825 100%);
    border: 1px solid #313244;
    border-radius: 20px;
    padding: 4rem 3rem 3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.hero-wrap::before {
    content: "";
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 30%, rgba(137,180,250,0.07) 0%, transparent 100%),
        radial-gradient(ellipse 50% 40% at 80% 70%, rgba(116,199,236,0.06) 0%, transparent 100%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(166,227,161,0.04) 0%, transparent 100%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #89b4fa;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(135deg, #cdd6f4 0%, #89b4fa 40%, #74c7ec 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #a6adc8;
    max-width: 620px;
    margin: 0 auto 2rem;
    line-height: 1.65;
    font-weight: 400;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid;
}

/* Stat strip */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin: 1.5rem 0;
}
.stat-item {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1.4rem 0.75rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-item:hover {
    border-color: #89b4fa;
    transform: translateY(-3px);
}
.stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: #89b4fa;
    line-height: 1;
    margin-bottom: 0.3rem;
    letter-spacing: -0.03em;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #6c7086;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* About block */
.about-block {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    line-height: 1.7;
    color: #a6adc8;
    font-size: 0.95rem;
}
.about-block strong { color: #cdd6f4; }

/* Section heading */
.sec-head {
    font-size: 1.05rem;
    font-weight: 700;
    color: #cdd6f4;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-head::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #313244;
}

/* Feature card */
.feat-card {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.feat-card:hover {
    border-color: #89b4fa;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(137,180,250,0.1);
}
.feat-icon { font-size: 1.5rem; margin-bottom: 0.5rem; display: block; }
.feat-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: #cdd6f4;
    margin: 0 0 0.35rem 0;
}
.feat-desc { font-size: 0.8rem; color: #6c7086; line-height: 1.55; margin: 0; }

/* Workflow step */
.wf-step {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.85rem 1.1rem;
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-left: 2px solid #89b4fa;
    border-radius: 0 10px 10px 0;
    margin-bottom: 0.6rem;
}
.wf-num {
    width: 24px; height: 24px;
    background: #89b4fa;
    color: #11111b;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 800;
    flex-shrink: 0;
}
.wf-title { font-size: 0.88rem; font-weight: 700; color: #cdd6f4; margin: 0 0 0.15rem; }
.wf-desc  { font-size: 0.78rem; color: #6c7086; margin: 0; line-height: 1.45; }

/* Classifier pills */
.clf-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.25rem;
}
.clf-pill {
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid;
    background: rgba(255,255,255,0.04);
}

/* Quick-start box */
.qs-box {
    background: #181825;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
}

/* Tip callout */
.tip-box {
    background: rgba(137,180,250,0.06);
    border: 1px solid rgba(137,180,250,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    font-size: 0.85rem;
    color: #a6adc8;
    margin-top: 0.75rem;
}
.tip-box strong { color: #89b4fa; }

/* Guide table in user guide */
.guide-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 0.75rem 0;
}
.guide-table th {
    background: #313244;
    color: #cdd6f4;
    font-weight: 700;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.guide-table td {
    padding: 0.5rem 0.75rem;
    color: #a6adc8;
    border-bottom: 1px solid #1e1e2e;
}
.guide-table tr:last-child td { border-bottom: none; }
.guide-table tr:hover td { background: rgba(137,180,250,0.04); }

/* ═══════════════════════════════════════
   MISC
   ═══════════════════════════════════════ */

footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

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
        "training_results": {},
        "experiment_summaries": {},
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
    st.markdown("""
    <div style="padding: 0.5rem 0 0.25rem 0;">
        <div style="font-size:1.6rem; font-weight:800; background:linear-gradient(135deg,#89b4fa,#74c7ec);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
            🧪 Klassify
        </div>
        <div style="font-size:0.78rem; color:#6c7086; margin-top:0.1rem;">ML Experimentation Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
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

    # Sidebar status summary
    has_data = st.session_state.df is not None
    has_preprocessed = st.session_state.X is not None
    has_results = bool(st.session_state.training_results)

    def _status_row(done, label):
        dot = "background:#a6e3a1" if done else "background:#313244;border:1px solid #45475a"
        txt = "#cdd6f4" if done else "#6c7086"
        return (f'<div style="display:flex;align-items:center;gap:0.55rem;'
                f'padding:0.35rem 0;font-size:0.82rem;color:{txt}">'
                f'<span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;{dot}"></span>'
                f'{label}</div>')

    st.markdown(
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'color:#6c7086;font-weight:700;margin-bottom:0.4rem">Session Status</div>'
        + _status_row(has_data, "Dataset loaded")
        + _status_row(has_preprocessed, "Data preprocessed")
        + _status_row(has_results, "Models trained"),
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("v1.0.0  ·  Klassify")

page_key = page.split("  ")[1]


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Home (Landing)
# ──────────────────────────────────────────────────────────────────────────────

if page_key == "Home":

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">ML Experimentation Platform</div>
        <div class="hero-title">Klassify</div>
        <div class="hero-sub">
            An end-to-end classification platform — upload a CSV, explore your data,
            train &amp; compare 10 classifiers, and explain every prediction.
            No boilerplate. No setup. Just results.
        </div>
        <div class="hero-tags">
            <span class="hero-tag" style="color:#89b4fa;border-color:rgba(137,180,250,0.3);background:rgba(137,180,250,0.07)">10 Classifiers</span>
            <span class="hero-tag" style="color:#74c7ec;border-color:rgba(116,199,236,0.3);background:rgba(116,199,236,0.07)">Hyperparameter Tuning</span>
            <span class="hero-tag" style="color:#a6e3a1;border-color:rgba(166,227,161,0.3);background:rgba(166,227,161,0.07)">SHAP Explainability</span>
            <span class="hero-tag" style="color:#f9e2af;border-color:rgba(249,226,175,0.3);background:rgba(249,226,175,0.07)">Experiment Tracking</span>
            <span class="hero-tag" style="color:#cba6f7;border-color:rgba(203,166,247,0.3);background:rgba(203,166,247,0.07)">Interactive EDA</span>
            <span class="hero-tag" style="color:#f38ba8;border-color:rgba(243,139,168,0.3);background:rgba(243,139,168,0.07)">REST API</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat strip ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="stat-strip">
        <div class="stat-item"><div class="stat-num">10</div><div class="stat-lbl">Classifiers</div></div>
        <div class="stat-item"><div class="stat-num">3</div><div class="stat-lbl">Tuning Modes</div></div>
        <div class="stat-item"><div class="stat-num">10×</div><div class="stat-lbl">Cross-Validation</div></div>
        <div class="stat-item"><div class="stat-num">5+</div><div class="stat-lbl">Chart Types</div></div>
        <div class="stat-item"><div class="stat-num">ONNX</div><div class="stat-lbl">Model Export</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="about-block">
        <strong>Klassify</strong> is a production-grade ML experimentation platform for data scientists
        who want to iterate fast — without writing boilerplate.<br><br>
        Upload any tabular CSV dataset, explore it with interactive charts, then train and compare
        up to 10 classifiers in a single click. Every experiment is automatically tracked, every
        model is versioned, and SHAP explainability is built-in so you always know <em>why</em>
        the model made a prediction.<br><br>
        The same capabilities are available through a <strong>FastAPI REST API</strong> with async
        Celery support — making Klassify easy to embed in existing ML pipelines.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Features & Workflow ────────────────────────────────────────────────────
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="sec-head">Platform Features</div>', unsafe_allow_html=True)

        features = [
            ("📂", "Dataset Management",
             "Upload CSV files up to 100 MB with automatic column-type detection, "
             "missing-value profiling, and duplicate tracking."),
            ("🔍", "Exploratory Data Analysis",
             "Histograms, correlation heatmaps, class-balance charts, PCA scatter, "
             "and box plots — all powered by Plotly."),
            ("🏋️", "Multi-Model Training",
             "Train up to 10 classifiers in one run with configurable test splits "
             "and stratified cross-validation."),
            ("⚙️", "Hyperparameter Tuning",
             "Grid Search, Random Search, or Bayesian Optimisation — just pick a "
             "strategy before hitting Train."),
            ("📊", "Rich Evaluation",
             "Confusion matrices, ROC/PR curves, feature importance, decision "
             "boundaries, and per-fold CV scores."),
            ("🔬", "SHAP Explainability",
             "Auto-selects TreeExplainer, LinearExplainer, or KernelExplainer "
             "based on the model type."),
            ("🏆", "Experiment Registry",
             "Every run logged as JSON. Models versioned and stored locally; "
             "export to ONNX for portable deployment."),
            ("🚀", "REST API",
             "Full FastAPI backend with async Celery. Upload, train, poll, "
             "and infer — all over HTTP."),
        ]

        for i in range(0, len(features), 2):
            c1, c2 = st.columns(2, gap="small")
            for col, (icon, title, desc) in zip([c1, c2], features[i:i+2]):
                with col:
                    st.markdown(f"""
                    <div class="feat-card">
                        <span class="feat-icon">{icon}</span>
                        <p class="feat-name">{title}</p>
                        <p class="feat-desc">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec-head">How It Works</div>', unsafe_allow_html=True)

        steps = [
            ("Upload your data",
             "Drop a CSV on the Dataset page. Column types are detected and the dataset is profiled instantly."),
            ("Configure preprocessing",
             "Pick a target column, scaler, and encoding strategy, then click Preprocess."),
            ("Explore with EDA",
             "Check distributions, correlations, class balance, PCA projections, and outliers."),
            ("Select & train models",
             "Choose classifiers, set tuning strategy and CV folds, then hit Train."),
            ("Analyse results",
             "Performance table, comparison charts, confusion matrices, ROC/PR curves, and CV scores."),
            ("Explain predictions",
             "Compute SHAP values and rank features by mean absolute impact."),
            ("Track & compare",
             "Leaderboard ranks all model versions; Experiment Log stores every run's raw detail."),
        ]

        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="wf-step">
                <div class="wf-num">{i}</div>
                <div>
                    <p class="wf-title">{title}</p>
                    <p class="wf-desc">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Supported classifiers ─────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Supported Classifiers</div>', unsafe_allow_html=True)

    classifiers = [
        ("Logistic Regression", "#89b4fa"),
        ("K-Nearest Neighbours", "#74c7ec"),
        ("Support Vector Machine", "#a6e3a1"),
        ("Decision Tree", "#f9e2af"),
        ("Random Forest", "#cba6f7"),
        ("Gradient Boosting", "#f38ba8"),
        ("XGBoost", "#fab387"),
        ("LightGBM", "#94e2d5"),
        ("Naive Bayes", "#89dceb"),
        ("Neural Network (MLP)", "#b4befe"),
    ]

    pills = "".join(
        f'<span class="clf-pill" style="color:{c};border-color:rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.35)">{n}</span>'
        for n, c in classifiers
    )
    st.markdown(f'<div class="clf-grid">{pills}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Quick Start ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Quick Start</div>', unsafe_allow_html=True)
    st.markdown("""
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Launch the app
    streamlit run klassify/frontend/streamlit_app.py

    # 3. (Optional) Start the FastAPI backend
    uvicorn klassify.backend.main:app --reload --port 8000
    ```
    """)

    st.markdown("""
    <div class="tip-box">
        <strong>Tip:</strong>
        Watch the <em>Session Status</em> panel in the sidebar — it tracks your progress
        through Dataset → Preprocessing → Training.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Quick User Guide ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📖 Quick User Guide</p>', unsafe_allow_html=True)

    guide_tabs = st.tabs([
        "1 · Upload Data",
        "2 · Explore (EDA)",
        "3 · Train Models",
        "4 · View Results",
        "5 · Explainability",
        "6 · Leaderboard",
        "7 · Experiment Log",
    ])

    with guide_tabs[0]:
        st.markdown("""
        #### 📂 Dataset Page — Upload & Preprocess

        **Step 1 — Upload your CSV**
        - Click **Browse files** or drag-and-drop a `.csv` file (max 100 MB).
        - Klassify automatically detects each column's type: *numeric*, *categorical*, *datetime*, or *text*.
        - A preview of the first 10 rows and a full column profile appear instantly.

        **Step 2 — Review the profile**
        - The **Dataset Profile** cards show row/column counts, missing cells, and duplicate rows.
        - The **Column Details** table shows dtype, feature type, missing %, and unique-value count for every column.

        **Step 3 — Configure preprocessing**
        | Setting | Options | What it does |
        |---|---|---|
        | Target column | Any column | The label the models will predict |
        | Scaler | Standard · MinMax · Robust | Normalises numeric features |
        | Categorical encoding | One-Hot · Ordinal | Encodes string/category columns |

        **Step 4 — Click "Preprocess Data"**
        - Imputes missing values, scales numeric features, and encodes categoricals.
        - The success banner shows the resulting shape and class list.
        - Once done, the ✅ checkmarks in the sidebar update automatically.

        > **Tip:** High-cardinality columns (>50 unique values) are automatically dropped.
        """)

    with guide_tabs[1]:
        st.markdown("""
        #### 🔍 EDA Page — Explore your data

        Navigate to **EDA** from the sidebar. Five tabs are available:

        | Tab | What you'll see |
        |---|---|
        | **Distributions** | Histogram grid for all numeric columns (up to 16) |
        | **Correlation** | Pearson correlation heatmap — spot multicollinearity |
        | **Class Balance** | Bar chart of target class frequencies — detect imbalance |
        | **PCA** | 2-D PCA scatter coloured by class *(requires preprocessing)* |
        | **Outliers** | Interactive box plot — select any numeric column |

        **Reading the charts**
        - In the **Correlation** tab, values near ±1 indicate strong linear relationships between features.
        - In the **Class Balance** tab, heavily skewed bars may require resampling before training.
        - In the **PCA** tab, well-separated clusters suggest the features are discriminative.

        > **Tip:** All charts are fully interactive — zoom, pan, hover for exact values, and click legend items to toggle series.
        """)

    with guide_tabs[2]:
        st.markdown("""
        #### 🏋️ Train Page — Train & tune classifiers

        > Preprocessing must be complete before training.

        **Step 1 — Select models**
        Choose one or more from the 10 available classifiers using the multiselect dropdown.
        Defaults: Logistic Regression, Random Forest, Gradient Boosting.

        **Step 2 — Configure training options**

        | Option | Description |
        |---|---|
        | **Hyperparameter tuning** | `none` — use defaults; `grid` — exhaustive grid search; `random` — random search |
        | **Cross-validation** | Toggle on/off; set number of folds (2–10) |
        | **Test split** | Fraction of data held out for evaluation (0.10–0.40) |
        | **Experiment name** | Label for grouping runs in the Experiment Log |

        **Step 3 — Click "Train Selected Models"**
        - Each model trains sequentially with a live progress bar.
        - Success/failure messages appear per model.
        - All results are automatically saved to the Experiment Log and Model Registry.

        > **Tip:** Start with `none` tuning for a fast baseline, then re-run with `random` or `grid` on your best model.
        """)

    with guide_tabs[3]:
        st.markdown("""
        #### 📊 Results Page — Evaluate trained models

        > At least one model must be trained first.

        **Performance Summary table** — top of the page shows Accuracy, F1, Precision, Recall, ROC-AUC, and training time for every model side-by-side.

        **Comparison chart** — grouped bar chart for visual metric comparison across models.

        **Per-model inspector** — select any model from the dropdown to see:

        | Panel | Description |
        |---|---|
        | **Confusion Matrix** | Row-normalised heatmap of true vs. predicted classes |
        | **Feature Importance** | Top 20 features ranked by importance or coefficient magnitude |
        | **ROC Curves** | Overlay of all models' ROC curves with AUC scores |
        | **PR Curves** | Precision-Recall curves — more informative for imbalanced datasets |
        | **Decision Boundary** | 2-D PCA projection of the model's decision regions |

        **Cross-Validation panel** *(if CV was enabled)*
        - Mean, Std, Min, Max CV scores shown as metric cards.
        - Bar chart of per-fold scores for the selected model.

        > **Tip:** For imbalanced classes, prioritise F1 and PR-AUC over raw accuracy.
        """)

    with guide_tabs[4]:
        st.markdown("""
        #### 🔬 Explainability Page — Understand predictions with SHAP

        > Models must be trained and data preprocessed before using this page.

        **Step 1 — Select a model** from the dropdown.

        **Step 2 — Click "Compute SHAP Values"**
        - Klassify automatically selects the best SHAP explainer for the model type:
          - **Tree models** (Random Forest, XGBoost, LightGBM, Gradient Boosting) → `TreeExplainer` (fast)
          - **Linear models** (Logistic Regression) → `LinearExplainer`
          - **All others** (SVM, KNN, MLP) → `KernelExplainer` (slower — may take 30–60 s)

        **Interpreting results**
        - **Mean |SHAP| bar chart** — features ranked by average absolute impact on predictions.
          Longer bar = greater influence on the model's output.
        - **SHAP summary table** — exact mean |SHAP| value per feature for further analysis.

        > **Tip:** A feature with high SHAP importance but low model feature importance may indicate non-linear interactions that tree-based SHAP captures better.
        """)

    with guide_tabs[5]:
        st.markdown("""
        #### 🏆 Leaderboard Page — Compare all registered model versions

        Every model trained on any experiment is registered here. The leaderboard lets you compare across different runs and experiment configurations.

        **Rank by** — choose the metric to sort by:
        `accuracy` · `f1` · `roc_auc` · `precision` · `recall`

        The table shows model key, version, registered date, and all tracked metrics.
        The bar chart displays the **Top 10** models by the selected metric.

        > **Tip:** Re-training the same model after tuning creates a new version (v2, v3, …). Compare versions in the leaderboard to confirm tuning improved performance.
        """)

    with guide_tabs[6]:
        st.markdown("""
        #### 📜 Experiment Log — Browse & inspect run history

        Every training run is logged as a JSON file on disk. This page lets you inspect them without leaving the app.

        **Experiment selector** — choose an experiment name (corresponds to the name set on the Train page).

        **Runs table** — columns: run ID, model, status (FINISHED / FAILED), accuracy, F1, ROC-AUC, start time.

        **Best run badge** — automatically highlights the run with the highest accuracy in the selected experiment.

        **Run inspector** — select any run ID to view the raw JSON: params, metrics, artifact paths, and tags logged during that run.

        > **Tip:** Use different experiment names for different datasets or problem setups — it keeps runs grouped logically and makes the log easier to navigate.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Dataset
# ──────────────────────────────────────────────────────────────────────────────

elif page_key == "Dataset":
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
            import tempfile
            from ml.dataset_loader import detect_feature_types, load_dataset, profile_dataset

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)

            df = load_dataset(tmp_path)
            ft = detect_feature_types(df)
            st.session_state.df = df
            st.session_state.dataset_path = tmp_path
            st.session_state.feature_types = ft
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
