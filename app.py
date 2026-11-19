
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FraudGuard AI | Credit Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. FUTURISTIC DARK THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(37,99,235,.18), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(124,58,237,.16), transparent 35%),
        #080d19;
    color: #f1f5f9;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a30, #0b1222);
    border-right: 1px solid rgba(148,163,184,.13);
}

[data-testid="stSidebar"] * {
    color: #dbeafe;
}

h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

p, label {
    color: #a9b9d0;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,
        rgba(25,40,68,.95),
        rgba(14,24,43,.95));
    border: 1px solid rgba(99,145,220,.22);
    padding: 20px;
    border-radius: 17px;
    box-shadow: 0 10px 30px rgba(0,0,0,.18);
}

[data-testid="stMetricLabel"] {
    color: #9fb2d0 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
}

div.stButton > button,
div.stDownloadButton > button {
    background: linear-gradient(100deg, #6366f1, #2563eb);
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 45px;
    font-weight: 700;
    transition: all .2s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: linear-gradient(100deg, #818cf8, #3b82f6);
    box-shadow: 0 0 22px rgba(99,102,241,.35);
    transform: translateY(-2px);
    color: white;
}

div[data-testid="stForm"] {
    background: rgba(20,33,57,.72);
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 18px;
    padding: 22px;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #111e35;
    border: 1px solid #293d60;
    border-radius: 9px;
    color: white;
}

input {
    color: white !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}

[data-testid="stFileUploader"] {
    background: rgba(20,33,57,.6);
    border: 1px dashed #526b99;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293d60;
    border-radius: 12px;
}

hr {
    border-color: rgba(148,163,184,.15);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

#MainMenu, footer {
    visibility: hidden;
}

.small-muted {
    color: #91a4c2;
    font-size: 13px;
}

.glass-card {
    background: linear-gradient(145deg,
        rgba(29,45,76,.8),
        rgba(15,25,45,.9));
    border: 1px solid rgba(129,160,220,.2);
    border-radius: 18px;
    padding: 22px;
    min-height: 120px;
}

.gradient-text {
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL LOADING
# =========================================================

MODEL_PATHS = [
    Path("models/credit_fraud_model.pkl"),
    Path("credit_fraud_model.pkl"),
    Path("models/credit_fraud_model.joblib")
]


@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            bundle = joblib.load(path)

            if not isinstance(bundle, dict):
                raise ValueError(
                    "Expected a dictionary model bundle."
                )

            required = [
                "model",
                "threshold",
                "feature_names",
                "model_name"
            ]

            missing = [
                key for key in required
                if key not in bundle
            ]

            if missing:
                raise ValueError(
                    f"Model bundle is missing: {missing}"
                )

            return bundle, str(path)

    raise FileNotFoundError(
        "Model file not found. Check your models folder."
    )


try:
    bundle, model_path = load_model()

    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_names = list(bundle["feature_names"])
    model_name = bundle["model_name"]
    metrics = bundle.get("metrics", {})

    model_loaded = True
    model_error = None

except Exception as e:
    model_loaded = False
    model_error = str(e)
    metrics = {}
    feature_names = []
    model_name = "Not loaded"
    threshold = 0.5


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def fraud_probability(data):
    """
    Return probability for class 1 (fraud).
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "This model does not support predict_proba()."
        )

    probabilities = model.predict_proba(data)

    classes = getattr(model, "classes_", None)

    if classes is not None and 1 in classes:
        fraud_index = list(classes).index(1)
    else:
        fraud_index = 1

    return probabilities[:, fraud_index]


def format_metric(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def find_metric(metric_dict, possible_names):
    if not isinstance(metric_dict, dict):
        return None

    for name in possible_names:
        if name in metric_dict:
            return metric_dict[name]

    return None


def make_gauge(value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={
                "suffix": "%",
                "font": {"size": 36, "color": "white"}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#94a3b8"
                },
                "bar": {"color": "#8b5cf6"},
                "bgcolor": "#15233c",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#123c38"
                    },
                    {
                        "range": [30, 70],
                        "color": "#55431c"
                    },
                    {
                        "range": [70, 100],
                        "color": "#52242e"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=15, r=15, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    return fig


def predict_single(input_data):
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    probability = float(
        fraud_probability(input_df)[0]
    )

    prediction = int(probability >= threshold)

    return probability, prediction


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 10px 0 22px 0;">
        <div style="font-size:38px;">🛡️</div>
        <div style="font-size:23px;font-weight:900;color:white;">
            FRAUD<span style="color:#818cf8;">GUARD</span>
        </div>
        <div style="font-size:11px;letter-spacing:2px;color:#8298bd;">
            ARTIFICIAL INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Fraud Prediction",
            "Batch Detection",
            "Model Performance",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    if model_loaded:
        st.success("● ML Model Online")
        st.caption(f"Model: {model_name}")
    else:
        st.error("● Model Offline")
        st.caption("Check the model file.")

    st.markdown("---")
    st.caption("FRAUDGUARD AI | 2026")


# =========================================================
# 6. TOP HEADER
# =========================================================

st.markdown("""
<div style="
    padding:25px 28px;
    border-radius:20px;
    background:linear-gradient(110deg,
        rgba(30,58,138,.55),
        rgba(76,29,149,.48),
        rgba(15,23,42,.85));
    border:1px solid rgba(129,140,248,.25);
    margin-bottom:25px;
">
    <div style="
        font-size:12px;
        color:#a5b4fc;
        font-weight:800;
        letter-spacing:3px;
    ">
        INTELLIGENT TRANSACTION SECURITY
    </div>

    <h1 style="
        margin:9px 0;
        font-size:34px;
        color:white;
    ">
        🛡️ FraudGuard AI
    </h1>

    <div style="color:#cbd5e1;font-size:14px;">
        Next-generation Credit Card Fraud Detection & Risk Analysis
    </div>
</div>
""", unsafe_allow_html=True)


if not model_loaded:
    st.error(f"Model loading error: {model_error}")
    st.info(
        "Make sure your model bundle is saved at "
        "'models/credit_fraud_model.pkl' or in the project root."
    )
    st.stop()


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("## Overview")
    st.caption(
        "Your machine learning fraud detection control center."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "SYSTEM STATUS",
            "ACTIVE",
            "Model loaded"
        )

    with col2:
        st.metric(
            "ML ALGORITHM",
            str(model_name)
        )

    with col3:
        st.metric(
            "DECISION THRESHOLD",
            f"{threshold:.3f}"
        )

    with col4:
        st.metric(
            "INPUT FEATURES",
            len(feature_names)
        )

    st.markdown("---")

    left, right = st.columns([1.5, 1])

    with left:

        st.markdown("### ⚡ Detection Workflow")

        st.markdown("""
        <div class="glass-card">
            <div style="font-size:13px;color:#60a5fa;font-weight:700;">
                STEP 01
            </div>
            <h3>Transaction Input</h3>
            <p>Provide transaction features or upload a CSV dataset.</p>
            <div style="color:#818cf8;font-size:24px;">↓</div>

            <div style="font-size:13px;color:#a78bfa;font-weight:700;">
                STEP 02
            </div>
            <h3>Machine Learning Analysis</h3>
            <p>The trained model analyzes patterns and calculates fraud probability.</p>
            <div style="color:#818cf8;font-size:24px;">↓</div>

            <div style="font-size:13px;color:#22d3ee;font-weight:700;">
                STEP 03
            </div>
            <h3>Risk Classification</h3>
            <p>The probability is compared against the saved decision threshold.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("### 🧠 Model Information")

        st.markdown(
            f"""
            <div class="glass-card">
                <p class="small-muted">SELECTED MODEL</p>
                <h3>{model_name}</h3>
                <hr>
                <p class="small-muted">MODEL THRESHOLD</p>
                <h2 class="gradient-text">{threshold:.4f}</h2>
                <hr>
                <p class="small-muted">MODEL INPUT FEATURES</p>
                <h2>{len(feature_names)}</h2>
                <hr>
                <p class="small-muted">MODEL FILE</p>
                <p style="font-size:12px;word-break:break-all;">
                    {model_path}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📈 Model Performance Snapshot")

    metric_items = [
        ("Precision", ["precision", "Precision"]),
        ("Recall", ["recall", "Recall"]),
        ("F1 Score", ["f1", "f1_score", "F1"]),
        ("ROC-AUC", ["roc_auc", "roc_auc_score", "ROC-AUC"]),
        ("PR-AUC", ["pr_auc", "average_precision", "PR-AUC"])
    ]

    if isinstance(metrics, dict) and metrics:

        metric_cols = st.columns(len(metric_items))

        for col, (label, keys) in zip(metric_cols, metric_items):

            value = find_metric(metrics, keys)

            with col:
                if value is not None:
                    st.metric(label, format_metric(value))
                else:
                    st.metric(label, "N/A")

    else:
        st.info(
            "Performance metrics were not found in the saved model bundle. "
            "Open Model Performance to see available information."
        )

    st.caption(
        "Dashboard values show model configuration, not live banking transactions."
    )


# =========================================================
# 8. SINGLE FRAUD PREDICTION
# =========================================================

elif page == "Fraud Prediction":

    st.markdown("## 🔍 Transaction Intelligence")
    st.caption(
        "Analyze an individual transaction using your trained ML model."
    )

    st.info(
        "Enter all model features. V1–V28 are anonymized PCA-transformed "
        "features from the dataset, not ordinary card details."
    )

    if "Time" not in feature_names or "Amount" not in feature_names:
        st.warning(
            "This form expects the standard dataset features: "
            "Time, V1–V28 and Amount. Your saved feature names differ."
        )

    with st.form("fraud_prediction_form"):

        st.markdown("### 💳 Transaction Information")

        col1, col2 = st.columns(2)

        input_values = {}

        with col1:
            time_value = st.number_input(
                "Transaction Time",
                min_value=0.0,
                value=10000.0,
                step=100.0
            )

        with col2:
            amount_value = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

        if "Time" in feature_names:
            input_values["Time"] = time_value

        if "Amount" in feature_names:
            input_values["Amount"] = amount_value

        st.markdown("---")
        st.markdown("### 🧬 PCA Feature Values")

        pca_features = [
            name for name in feature_names
            if name not in ["Time", "Amount"]
        ]

        if pca_features:

            st.caption(
                "For a demonstration, values default to zero. "
                "Use actual transformed values for meaningful testing."
            )

            feature_cols = st.columns(3)

            for i, feature in enumerate(pca_features):

                with feature_cols[i % 3]:

                    input_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.6f",
                        key=f"single_{feature}"
                    )

        submitted = st.form_submit_button(
            "⚡ ANALYZE TRANSACTION",
            use_container_width=True
        )

    if submitted:

        try:
            ordered_values = {
                name: input_values[name]
                for name in feature_names
            }

            probability, prediction = predict_single(
                ordered_values
            )

            st.markdown("---")
            st.markdown("## Analysis Result")

            result_col, gauge_col = st.columns([1, 1])

            with result_col:

                st.metric(
                    "FRAUD PROBABILITY",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(np.clip(probability, 0, 1))
                )

                if prediction == 1:

                    st.error("🚨 HIGH RISK — FRAUD FLAGGED")

                    st.write(
                        "The model has classified this transaction "
                        "as potentially fraudulent."
                    )

                else:

                    st.success("✅ GENUINE — NOT FLAGGED")

                    st.write(
                        "The model has not classified this transaction "
                        "as fraudulent at the selected threshold."
                    )

                st.caption(
                    f"Decision threshold: {threshold:.4f}"
                )

            with gauge_col:

                st.plotly_chart(
                    make_gauge(probability),
                    use_container_width=True
                )

            st.caption(
                "This is a machine learning prediction, not proof of fraud. "
                "A real financial institution should use additional verification."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# =========================================================
# 9. BATCH DETECTION
# =========================================================

elif page == "Batch Detection":

    st.markdown("## 📂 Batch Transaction Intelligence")

    st.caption(
        "Upload multiple transactions and analyze them together."
    )

    st.markdown("### Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV containing your model input features",
        type=["csv"]
    )

    if uploaded_file:

        try:

            data = pd.read_csv(uploaded_file)

            st.markdown("### Dataset Preview")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

            missing_columns = [
                col for col in feature_names
                if col not in data.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing model input columns: {missing_columns}"
                )

            elif data.empty:

                st.warning("The uploaded CSV contains no transactions.")

            else:

                st.success(
                    f"Dataset loaded successfully: {len(data):,} rows"
                )

                if st.button(
                    "🚀 RUN FRAUD DETECTION",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing transactions..."
                    ):

                        input_data = data[feature_names].copy()

                        probabilities = fraud_probability(
                            input_data
                        )

                        predictions = (
                            probabilities >= threshold
                        ).astype(int)

                        results = data.copy()

                        results["Fraud_Probability"] = probabilities
                        results["Prediction"] = predictions

                        results["Status"] = np.where(
                            predictions == 1,
                            "Fraud",
                            "Genuine"
                        )

                    total = len(results)
                    fraud_count = int(predictions.sum())
                    genuine_count = total - fraud_count

                    st.markdown("---")
                    st.markdown("### 📊 Detection Summary")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "TOTAL TRANSACTIONS",
                        f"{total:,}"
                    )

                    c2.metric(
                        "FRAUD FLAGGED",
                        f"{fraud_count:,}"
                    )

                    c3.metric(
                        "GENUINE",
                        f"{genuine_count:,}"
                    )

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:

                        fig = px.pie(
                            names=["Fraud", "Genuine"],
                            values=[
                                fraud_count,
                                genuine_count
                            ],
                            hole=0.65,
                            title="Prediction Distribution",
                            color_discrete_sequence=[
                                "#ef4444",
                                "#6366f1"
                            ]
                        )

                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="white",
                            legend_title_text=""
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    with chart_col2:

                        fig2 = px.histogram(
                            results,
                            x="Fraud_Probability",
                            nbins=30,
                            title="Fraud Probability Distribution",
                            color_discrete_sequence=["#818cf8"]
                        )

                        fig2.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="white"
                        )

                        st.plotly_chart(
                            fig2,
                            use_container_width=True
                        )

                    st.markdown("### Transaction Results")

                    filter_status = st.selectbox(
                        "Filter Transactions",
                        ["All", "Fraud", "Genuine"]
                    )

                    if filter_status == "Fraud":
                        display_data = results[
                            results["Status"] == "Fraud"
                        ]

                    elif filter_status == "Genuine":
                        display_data = results[
                            results["Status"] == "Genuine"
                        ]

                    else:
                        display_data = results

                    st.dataframe(
                        display_data,
                        use_container_width=True
                    )

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ DOWNLOAD DETECTION REPORT",
                        data=csv_data,
                        file_name="FraudGuard_Detection_Report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to process CSV: {e}"
            )

    else:

        st.markdown("""
        <div class="glass-card">
            <h3>📁 Batch Analysis</h3>
            <p>
                Upload your transaction dataset to get fraud predictions,
                probability scores, visual analytics and a downloadable report.
            </p>
            <p class="small-muted">
                Required columns: all feature names used during model training.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Download Input Template")

    template = pd.DataFrame(
        [{name: 0.0 for name in feature_names}]
    )

    st.download_button(
        "⬇️ DOWNLOAD CSV TEMPLATE",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="fraud_input_template.csv",
        mime="text/csv"
    )


# =========================================================
# 10. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown("## 📈 Model Performance Center")

    st.caption(
        "Evaluation metrics saved with your trained model."
    )

    st.markdown("### Selected Model")

    st.markdown(
        f"""
        <div class="glass-card">
            <p class="small-muted">BEST SELECTED MODEL</p>
            <h2 class="gradient-text">{model_name}</h2>
            <p>Decision threshold: {threshold:.4f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if isinstance(metrics, dict) and metrics:

        st.markdown("### Evaluation Metrics")

        metric_items = [
            ("Accuracy", ["accuracy", "accuracy_score"]),
            ("Precision", ["precision", "Precision"]),
            ("Recall", ["recall", "Recall"]),
            ("F1 Score", ["f1", "f1_score", "F1"]),
            ("ROC-AUC", ["roc_auc", "roc_auc_score"]),
            ("PR-AUC", ["pr_auc", "average_precision"])
        ]

        cols = st.columns(3)

        for i, (label, keys) in enumerate(metric_items):

            value = find_metric(metrics, keys)

            with cols[i % 3]:

                if value is not None:
                    st.metric(
                        label,
                        format_metric(value)
                    )
                else:
                    st.metric(label, "N/A")

        st.markdown("---")
        st.markdown("### Raw Saved Metrics")

        st.json(metrics)

    else:

        st.warning(
            "No evaluation metrics were saved in the model bundle."
        )

        st.info(
            "Save your test metrics in the model bundle during training "
            "to display them here."
        )

    st.markdown("### Understanding the Metrics")

    st.markdown("""
    - **Precision:** Out of transactions flagged as fraud, how many were actually fraud?
    - **Recall:** Out of all actual fraud transactions, how many did the model detect?
    - **F1 Score:** Harmonic balance between precision and recall.
    - **ROC-AUC:** Measures how well the model separates the two classes across thresholds.
    - **PR-AUC:** Particularly useful for evaluating highly imbalanced datasets.
    - **Accuracy:** Overall percentage of correct predictions; it can be misleading for imbalanced data.
    """)


# =========================================================
# 11. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown("## 🧠 About FraudGuard AI")

    st.markdown("""
    <div class="glass-card">
        <h2 class="gradient-text">
            Credit Card Fraud Detection Using Machine Learning
        </h2>
        <p>
            FraudGuard AI is a supervised machine learning application
            designed to identify potentially fraudulent credit card transactions.
            It analyzes transaction features and predicts the probability
            of a transaction belonging to the fraud class.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Dataset Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.markdown("""
        **Input Features**

        - Time
        - V1 to V28
        - Amount
        """)

    with feature_col2:

        st.markdown("""
        **Target Variable**

        - Class 0: Genuine transaction
        - Class 1: Fraudulent transaction
        """)

    st.markdown("---")

    st.markdown("### Machine Learning Workflow")

    workflow = [
        ("01", "Data Collection", "Credit card transaction dataset"),
        ("02", "Data Preprocessing", "Cleaning and preparing input features"),
        ("03", "Class Imbalance Handling", "Addressing rare fraud examples"),
        ("04", "Model Training", "Training and comparing ML algorithms"),
        ("05", "Model Evaluation", "Precision, recall, F1 and PR-AUC"),
        ("06", "Deployment", "Streamlit-based prediction interface")
    ]

    for number, title, description in workflow:

        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom:12px;min-height:unset;">
                <span style="color:#818cf8;font-weight:800;">
                    {number}
                </span>
                <span style="font-size:18px;font-weight:700;color:white;">
                    &nbsp; {title}
                </span>
                <p style="margin:8px 0 0 32px;">
                    {description}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### Important Disclaimer")

    st.warning(
        "This application is an educational machine learning project. "
        "Its predictions are risk indicators, not proof of fraud or "
        "a substitute for financial institution verification."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;padding:10px;color:#64748b;">
    <div style="font-weight:700;color:#a5b4fc;">
        🛡️ FRAUDGUARD AI
    </div>
    <div style="font-size:12px;margin-top:5px;">
        Intelligent Credit Card Fraud Detection System
    </div>
    <div style="font-size:11px;margin-top:5px;">
        Built with Python • Scikit-learn • Streamlit • Plotly
    </div>
</div>
""", unsafe_allow_html=True)
