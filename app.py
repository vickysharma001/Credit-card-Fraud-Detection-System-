from __future__ import annotations

import os
import io
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ============================================================
# FraudGuard AI | Credit Card Fraud Detection
# Production-style Streamlit interface for a saved sklearn/XGBoost model
# ============================================================

st.set_page_config(
    page_title="FraudGuard AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "FraudGuard AI"
MODEL_CANDIDATES = [
    "models/credit_fraud_model.pkl",
    "notebook/models/credit_fraud_model.pkl",
    "credit_fraud_model.pkl",
    "models/credit_fraud_model.joblib",
]
DEFAULT_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
          radial-gradient(circle at 12% 0%, rgba(37,99,235,.16), transparent 28%),
          radial-gradient(circle at 92% 8%, rgba(124,58,237,.13), transparent 25%),
          #080d18;
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(148,163,184,.14);
    }
    [data-testid="stHeader"] { background: rgba(8,13,24,.85); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc; }
    p, label, span, div { }
    .hero {
        padding: 26px 28px; border-radius: 20px; margin-bottom: 22px;
        background: linear-gradient(115deg, rgba(30,64,175,.38), rgba(76,29,149,.30)),
                    rgba(15,23,42,.88);
        border: 1px solid rgba(129,140,248,.24);
        box-shadow: 0 18px 55px rgba(0,0,0,.22);
    }
    .hero-kicker { color:#93c5fd; font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
    .hero-title { color:#fff; font-family:'Space Grotesk',sans-serif; font-size:34px; font-weight:700; margin:7px 0 4px; }
    .hero-sub { color:#cbd5e1; font-size:14px; }
    .metric-card {
        background: linear-gradient(145deg, rgba(20,31,51,.95), rgba(12,20,35,.95));
        border: 1px solid rgba(148,163,184,.15); border-radius:16px;
        padding:18px 19px; min-height:112px;
    }
    .metric-label { color:#94a3b8; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { color:#f8fafc; font-family:'Space Grotesk',sans-serif; font-size:27px; font-weight:700; margin-top:8px; }
    .metric-help { color:#64748b; font-size:11px; margin-top:3px; }
    .section-note { color:#94a3b8; font-size:13px; }
    .result-box { padding:22px; border-radius:18px; border:1px solid rgba(148,163,184,.2); background:#101a2b; }
    .small-muted { color:#94a3b8; font-size:12px; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg,#2563eb,#7c3aed); border:0; color:white;
        border-radius:10px; font-weight:700; min-height:44px;
    }
    div.stButton > button[kind="primary"]:hover { border:0; filter:brightness(1.12); }
    div[data-testid="stForm"] { background:rgba(15,23,42,.55); border:1px solid rgba(148,163,184,.15); padding:18px; border-radius:16px; }
    [data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.15); border-radius:12px; }
    hr { border-color:rgba(148,163,184,.15); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Helpers -----------------------------
@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.isfile(path):
            bundle = joblib.load(path)
            if isinstance(bundle, dict) and "model" in bundle:
                model = bundle["model"]
                features = bundle.get("feature_names") or getattr(model, "feature_names_in_", None)
                if features is None:
                    features = DEFAULT_FEATURES
                return {
                    "model": model,
                    "model_name": bundle.get("model_name", type(model).__name__),
                    "threshold": float(bundle.get("threshold", 0.5)),
                    "feature_names": list(features),
                    "metrics": bundle.get("metrics", {}),
                    "path": path,
                }
            # Also support a plain estimator saved directly with joblib
            model = bundle
            features = getattr(model, "feature_names_in_", DEFAULT_FEATURES)
            return {
                "model": model,
                "model_name": type(model).__name__,
                "threshold": 0.5,
                "feature_names": list(features),
                "metrics": {},
                "path": path,
            }
    return None


def get_fraud_probability(model, frame: pd.DataFrame) -> np.ndarray:
    """Return probability for class label 1 (fraud), handling class order safely."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", [0, 1]))
        fraud_idx = classes.index(1) if 1 in classes else probs.shape[1] - 1
        return np.asarray(probs[:, fraud_idx], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(frame), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -35, 35)))

    raise RuntimeError("This model does not support predict_proba() or decision_function().")


def prepare_frame(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_names if col not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing[:8]))
    frame = data.loc[:, feature_names].copy()
    for col in feature_names:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Input contains blank or non-numeric feature values.")
    return frame


def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-help">{help_text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def show_prediction(probability: float, threshold: float):
    is_fraud = probability >= threshold
    color = "#fb7185" if is_fraud else "#34d399"
    title = "Potential Fraud Detected" if is_fraud else "Likely Genuine Transaction"
    explanation = (
        "The model risk score is at or above the configured decision threshold."
        if is_fraud else
        "The model risk score is below the configured decision threshold."
    )
    st.markdown(
        f"""<div class="result-box">
        <div style="color:{color};font-weight:700;font-size:12px;letter-spacing:1.5px;text-transform:uppercase;">MODEL ASSESSMENT</div>
        <div style="font-family:Space Grotesk;font-size:25px;font-weight:700;color:#f8fafc;margin:8px 0;">{title}</div>
        <div style="color:#cbd5e1;font-size:13px;">{explanation}</div>
        <div style="font-size:36px;font-weight:700;color:{color};margin-top:15px;">{probability:.2%}</div>
        <div class="small-muted">Estimated fraud probability</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(float(probability), 0.0), 1.0))
    st.caption(f"Decision threshold: {threshold:.2%}")


# ----------------------------- Load model -----------------------------
try:
    bundle = load_model()
except Exception as exc:
    st.error("Model file was found, but it could not be loaded.")
    st.code(str(exc))
    st.info(
        "If this mentions libomp.dylib on macOS, run `brew install libomp`, "
        "then restart Streamlit. Make sure the model was saved with the same "
        "scikit-learn/XGBoost versions used for training."
    )
    st.stop()

if bundle is None:
    st.error("Trained model not found.")
    st.write("Expected one of these paths:")
    for p in MODEL_CANDIDATES:
        st.code(p)
    st.info("Save your trained model bundle as `models/credit_fraud_model.pkl` and restart this app.")
    st.stop()

model = bundle["model"]
feature_names = bundle["feature_names"]
threshold = min(max(bundle["threshold"], 0.0), 1.0)
metrics = bundle["metrics"] or {}


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard AI")
    st.caption("Credit Card Fraud Intelligence")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Single Transaction", "Batch Analysis", "Model Insights", "About"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Model loaded")
    st.caption(f"Algorithm: {bundle['model_name']}")
    st.caption(f"Threshold: {threshold:.3f}")
    st.caption(f"Features: {len(feature_names)}")
    st.divider()
    st.caption("Internal decision-support demo • Not a payment authorization system")


# ----------------------------- Header -----------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">AI-powered risk intelligence</div>
      <div class="hero-title">🛡️ FraudGuard AI</div>
      <div class="hero-sub">Credit Card Fraud Detection & Transaction Risk Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Overview -----------------------------
if page == "Overview":
    st.subheader("System Overview")
    st.markdown(
        '<p class="section-note">Monitor model configuration and access transaction-level fraud scoring.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Model", str(bundle["model_name"])[:18], "Loaded from local artifact")
    with c2:
        metric_card("Input Features", str(len(feature_names)), "Features expected by model")
    with c3:
        metric_card("Decision Threshold", f"{threshold:.3f}", "Configured fraud cut-off")
    with c4:
        metric_card("Model Status", "ONLINE", "Artifact loaded successfully")

    st.markdown("### Model performance snapshot")
    if metrics:
        cols = st.columns(min(4, len(metrics)))
        preferred = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
        shown = [(k, metrics[k]) for k in preferred if k in metrics]
        if not shown:
            shown = list(metrics.items())
        for i, (key, value) in enumerate(shown[:4]):
            try:
                display = f"{float(value):.3f}"
            except (TypeError, ValueError):
                display = str(value)
            with cols[i % len(cols)]:
                metric_card(key.replace("_", " ").title(), display, "Saved test-set metric")
    else:
        st.info("No metrics were saved in the model bundle. Open Model Insights for details.")

    st.markdown("### Workflow")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**01 · Input**")
        st.write("Enter transaction features or upload a CSV file.")
    with w2:
        st.markdown("**02 · Score**")
        st.write("The trained classifier estimates the fraud probability.")
    with w3:
        st.markdown("**03 · Review**")
        st.write("Review the risk score and export batch results.")

    st.warning(
        "Important: V1–V28 are anonymized PCA-transformed features in the commonly used "
        "European card dataset. Arbitrary zeros are not real transaction data and should "
        "only be used to test that the interface works."
    )


# ----------------------------- Single transaction -----------------------------
elif page == "Single Transaction":
    st.subheader("Single Transaction Assessment")
    st.markdown(
        '<p class="section-note">Provide all features expected by the trained model. '
        'For the standard dataset, V1–V28 are anonymized PCA features.</p>',
        unsafe_allow_html=True,
    )

    known_basic = {"Time", "Amount"}
    basic_cols = [c for c in feature_names if c in known_basic]
    advanced_cols = [c for c in feature_names if c not in known_basic]

    with st.form("single_transaction_form"):
        st.markdown("#### Transaction details")
        basic_values = {}
        bcols = st.columns(2)
        for i, col in enumerate(basic_cols):
            with bcols[i % 2]:
                if col.lower() == "amount":
                    basic_values[col] = st.number_input(
                        "Transaction amount", min_value=0.0, value=100.0, step=10.0,
                        help="Amount in the same unit/currency as the training dataset."
                    )
                else:
                    basic_values[col] = st.number_input(
                        "Transaction time", min_value=0.0, value=0.0, step=60.0,
                        help="Time value in the same format used during model training."
                    )

        advanced_values = {}
        if advanced_cols:
            with st.expander(f"Advanced model features ({len(advanced_cols)})", expanded=False):
                st.caption("Enter the actual transformed feature values from your dataset.")
                grid = st.columns(3)
                for i, col in enumerate(advanced_cols):
                    with grid[i % 3]:
                        advanced_values[col] = st.number_input(
                            col, value=0.0, step=0.01, format="%.6f", key=f"single_{col}"
                        )

        submitted = st.form_submit_button("Analyze Transaction", type="primary", use_container_width=True)

    if submitted:
        row = {**basic_values, **advanced_values}
        input_df = pd.DataFrame([{c: row.get(c, 0.0) for c in feature_names}])
        try:
            input_df = prepare_frame(input_df, feature_names)
            probability = float(get_fraud_probability(model, input_df)[0])
            st.markdown("### Assessment result")
            left, right = st.columns([1, 1])
            with left:
                show_prediction(probability, threshold)
            with right:
                st.markdown("#### Transaction summary")
                st.dataframe(
                    pd.DataFrame({"Feature": feature_names, "Value": [input_df.iloc[0][c] for c in feature_names]}),
                    use_container_width=True, hide_index=True, height=380
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ----------------------------- Batch analysis -----------------------------
elif page == "Batch Analysis":
    st.subheader("Batch Transaction Analysis")
    st.markdown(
        '<p class="section-note">Upload a CSV containing the model input columns. '
        'The target column `Class` is not required.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)
            st.write(f"Uploaded file: **{len(source_df):,} rows** · {len(source_df.columns)} columns")
            with st.expander("Preview uploaded data"):
                st.dataframe(source_df.head(10), use_container_width=True)
            missing = [c for c in feature_names if c not in source_df.columns]
            if missing:
                st.error("Required model columns are missing: " + ", ".join(missing))
            elif st.button("Run Batch Detection", type="primary"):
                with st.spinner("Scoring transactions..."):
                    model_input = prepare_frame(source_df, feature_names)
                    probs = get_fraud_probability(model, model_input)
                    results = source_df.copy()
                    results["Fraud_Probability"] = probs
                    results["Prediction"] = np.where(probs >= threshold, "Potential Fraud", "Likely Genuine")
                    results["Prediction_Class"] = (probs >= threshold).astype(int)

                fraud_count = int((results["Prediction_Class"] == 1).sum())
                genuine_count = len(results) - fraud_count
                a, b, c = st.columns(3)
                with a:
                    metric_card("Transactions Scored", f"{len(results):,}")
                with b:
                    metric_card("Potential Fraud", f"{fraud_count:,}")
                with c:
                    metric_card("Likely Genuine", f"{genuine_count:,}")

                st.markdown("### Scored transactions")
                st.dataframe(
                    results.sort_values("Fraud_Probability", ascending=False),
                    use_container_width=True, height=420
                )
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored results (CSV)",
                    data=csv_bytes,
                    file_name=f"fraudguard_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Could not process this file: {exc}")


# ----------------------------- Model insights -----------------------------
elif page == "Model Insights":
    st.subheader("Model Insights & Evaluation")
    st.markdown(
        '<p class="section-note">Evaluation metrics below are read from the saved model bundle. '
        'They are not recalculated on the uploaded transaction data.</p>',
        unsafe_allow_html=True,
    )
    if metrics:
        metric_items = list(metrics.items())
        cols = st.columns(3)
        for i, (key, value) in enumerate(metric_items):
            try:
                value_text = f"{float(value):.4f}"
            except (TypeError, ValueError):
                value_text = str(value)
            with cols[i % 3]:
                metric_card(key.replace("_", " ").title(), value_text, "Saved evaluation metric")

        numeric = {}
        for key, value in metrics.items():
            try:
                numeric[key.replace("_", " ").title()] = float(value)
            except (TypeError, ValueError):
                pass
        if numeric:
            fig = go.Figure(go.Bar(
                x=list(numeric.keys()), y=list(numeric.values()),
                marker=dict(color=list(range(len(numeric))), colorscale="Blues"),
                text=[f"{v:.3f}" for v in numeric.values()], textposition="outside"
            ))
            fig.update_layout(
                title="Saved model metrics", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1.12], title="Score"),
                margin=dict(l=10, r=10, t=55, b=10), height=360
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The model bundle does not contain saved metrics.")
        st.code("bundle['metrics'] = {'precision': ..., 'recall': ..., 'f1_score': ..., 'roc_auc': ..., 'pr_auc': ...}")

    st.markdown("### Feature schema")
    st.caption("The model expects these columns in exactly this order:")
    st.code(", ".join(feature_names))


# ----------------------------- About -----------------------------
elif page == "About":
    st.subheader("About FraudGuard AI")
    st.write(
        "FraudGuard AI is a machine-learning demonstration application that scores "
        "credit-card transactions for potential fraud. It uses a previously trained "
        "classification model saved as a Joblib artifact."
    )
    st.markdown("#### Technology stack")
    st.write("- Python and Streamlit — application interface")
    st.write("- Pandas and NumPy — input processing")
    st.write("- Scikit-learn / XGBoost — machine-learning inference")
    st.write("- Plotly — interactive visualizations")
    st.markdown("#### Responsible use")
    st.warning(
        "This is an academic/demo decision-support tool, not a bank-grade fraud prevention "
        "system. A model score is not proof of fraud. Real deployments require validated "
        "data pipelines, monitoring, security controls, human review, and periodic retraining."
    )
    st.markdown("#### Loaded artifact")
    st.code(bundle["path"])


# ----------------------------- Footer -----------------------------
st.divider()
st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:11px;padding:8px 0;">'
    'FRAUDGUARD AI · CREDIT CARD FRAUD DETECTION · MACHINE LEARNING PROJECT'
    '</div>',
    unsafe_allow_html=True,
)
