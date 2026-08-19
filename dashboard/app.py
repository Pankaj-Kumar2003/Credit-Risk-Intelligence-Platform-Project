"""
Streamlit dashboard for the Credit Risk Intelligence Platform.
Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.helper import load_config, load_json, resolve_path
from src.utils.model_loader import load_artifacts

st.set_page_config(page_title="Credit Risk Intelligence Platform", layout="wide", page_icon="💳")

config = load_config()


@st.cache_resource
def get_artifacts():
    try:
        return load_artifacts(config)
    except FileNotFoundError:
        return None


@st.cache_data
def get_raw_data():
    path = resolve_path(config["data"]["raw_path"])
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def get_test_data():
    path = resolve_path(config["data"]["test_path"])
    return pd.read_csv(path) if path.exists() else None


artifacts = get_artifacts()
raw_df = get_raw_data()
test_df = get_test_data()

st.sidebar.title("Credit Risk Intelligence Platform")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Dataset Explorer", "Feature Analysis", "Model Performance",
     "Risk Prediction", "SHAP Explainability", "Drift Monitoring", "Model Comparison"],
)

if artifacts is None:
    st.sidebar.warning("⚠️ No trained model found. Run `python main.py --stage all` first.")

# ------------------------------------------------------------------
# PROJECT OVERVIEW
# ------------------------------------------------------------------
if page == "Project Overview":
    st.title("Credit Risk Intelligence Platform")
    st.markdown(
        "Predicts loan-applicant default probability, explains every prediction, "
        "and monitors model health over time."
    )

    col1, col2, col3, col4 = st.columns(4)
    if raw_df is not None:
        col1.metric("Applications", f"{len(raw_df):,}")
        col2.metric("Historical Default Rate", f"{raw_df[config['data']['target_column']].mean():.1%}")
        col3.metric("Avg Credit Score", f"{raw_df['credit_score'].mean():.0f}")
        col4.metric("Avg Loan Amount", f"${raw_df['loan_amount'].mean():,.0f}")
    else:
        st.info("No dataset found yet — run `python scripts/generate_mock_data.py`.")

    st.subheader("Pipeline")
    st.code(
        "Data Sources -> Ingestion -> Validation -> Cleaning -> Feature Engineering\n"
        "  -> Model Training (LogReg / RF / XGBoost / LightGBM) -> MLflow Tracking\n"
        "  -> SHAP Explainability -> FastAPI + Streamlit -> Drift Monitoring",
        language="text",
    )

# ------------------------------------------------------------------
# DATASET EXPLORER
# ------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    if raw_df is None:
        st.warning("No raw dataset found.")
    else:
        st.dataframe(raw_df.head(200), use_container_width=True)
        st.subheader("Missing Values")
        missing = raw_df.isna().mean().sort_values(ascending=False)
        st.bar_chart(missing[missing > 0])

        st.subheader("Target Distribution")
        target_col = config["data"]["target_column"]
        fig = px.pie(raw_df, names=target_col, title="Default vs. Repaid", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# FEATURE ANALYSIS
# ------------------------------------------------------------------
elif page == "Feature Analysis":
    st.title("Feature Analysis")
    if raw_df is None:
        st.warning("No raw dataset found.")
    else:
        numeric_cols = config["data"]["numerical_features"]
        selected = st.selectbox("Select a numeric feature", numeric_cols)
        fig = px.histogram(raw_df, x=selected, color=config["data"]["target_column"],
                            barmode="overlay", opacity=0.6, title=f"Distribution of {selected} by default status")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = raw_df[numeric_cols + [config["data"]["target_column"]]].corr()
        fig2 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    metrics_path = resolve_path("reports/champion_evaluation_metrics.json")
    if not metrics_path.exists():
        st.warning("No evaluation report found. Run `python main.py --stage evaluate`.")
    else:
        metrics = load_json(metrics_path)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ROC-AUC", metrics["roc_auc"])
        col2.metric("PR-AUC", metrics["pr_auc"])
        col3.metric("Precision", metrics["precision"])
        col4.metric("Recall", metrics["recall"])
        col5.metric("F1 Score", metrics["f1"])

        image_cols = st.columns(2)
        for i, name in enumerate(["roc_curve.png", "pr_curve.png", "confusion_matrix.png",
                                   "calibration_curve.png", "lift_gain_chart.png"]):
            image_path = resolve_path(f"reports/{name}")
            if image_path.exists():
                image_cols[i % 2].image(str(image_path), caption=name.replace("_", " ").replace(".png", "").title())

# ------------------------------------------------------------------
# RISK PREDICTION
# ------------------------------------------------------------------
elif page == "Risk Prediction":
    st.title("Score a New Applicant")
    if artifacts is None:
        st.warning("Train the model first.")
    else:
        from src.prediction.predict import CreditRiskPredictor

        with st.form("applicant_form"):
            c1, c2, c3 = st.columns(3)
            annual_income = c1.number_input("Annual income", 10000, 1000000, 65000)
            loan_amount = c1.number_input("Loan amount", 500, 200000, 15000)
            credit_score = c1.slider("Credit score", 300, 850, 690)
            employment_years = c2.number_input("Employment years", 0.0, 45.0, 4.0)
            age = c2.number_input("Age", 18, 90, 35)
            existing_debt = c2.number_input("Existing debt", 0, 500000, 8000)
            credit_limit = c3.number_input("Credit limit", 500, 300000, 20000)
            num_open_accounts = c3.number_input("Open accounts", 0, 40, 5)
            num_credit_inquiries_6m = c3.number_input("Inquiries (6m)", 0, 20, 1)
            num_previous_defaults = st.slider("Previous defaults", 0, 10, 0)
            delinquencies_2y = st.slider("Delinquencies (2y)", 0, 10, 0)
            loan_purpose = st.selectbox("Loan purpose", ["debt_consolidation", "home_improvement", "medical",
                                                           "auto", "small_business", "major_purchase", "education", "other"])
            home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            submitted = st.form_submit_button("Score Applicant")

        if submitted:
            predictor = CreditRiskPredictor(artifacts)
            record = {
                "applicant_id": "DASHBOARD_APPLICANT", "annual_income": annual_income, "loan_amount": loan_amount,
                "credit_score": credit_score, "employment_years": employment_years, "age": age,
                "existing_debt": existing_debt, "credit_limit": credit_limit, "num_open_accounts": num_open_accounts,
                "num_credit_inquiries_6m": num_credit_inquiries_6m, "num_previous_defaults": num_previous_defaults,
                "delinquencies_2y": delinquencies_2y, "loan_purpose": loan_purpose, "home_ownership": home_ownership,
                "application_date": "2026-01-01", "account_open_date": "2021-01-01",
            }
            result = predictor.predict_single(record)
            st.metric("Default Probability", f"{result['default_probability']:.1%}")
            st.metric("Risk Category", result["risk_category"])

# ------------------------------------------------------------------
# SHAP EXPLAINABILITY
# ------------------------------------------------------------------
elif page == "SHAP Explainability":
    st.title("Model Explainability (SHAP)")
    importance_path = resolve_path("reports/shap_global_importance.csv")
    if not importance_path.exists():
        st.warning("Run `python -m src.explainability.shap_explainer` first.")
    else:
        importance_df = pd.read_csv(importance_path)
        fig = px.bar(importance_df.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                     orientation="h", title="Global Feature Importance (mean |SHAP value|)")
        st.plotly_chart(fig, use_container_width=True)

    summary_path = resolve_path("reports/shap_summary_plot.png")
    if summary_path.exists():
        st.image(str(summary_path), caption="SHAP Summary Plot")

# ------------------------------------------------------------------
# DRIFT MONITORING
# ------------------------------------------------------------------
elif page == "Drift Monitoring":
    st.title("Data Drift Monitoring")
    drift_json_path = resolve_path("reports/drift_report.json")
    drift_html_path = resolve_path(config["monitoring"]["report_path"])

    if not drift_json_path.exists():
        st.warning("Run `python -m src.monitoring.drift_monitor` first.")
    else:
        with open(drift_json_path) as f:
            drift = json.load(f)
        st.json({"summary_available": True})
        if drift_html_path.exists():
            with open(drift_html_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=800, scrolling=True)

# ------------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown(
        "Model comparison metrics are logged to MLflow during training. "
        "Launch the MLflow UI to compare Logistic Regression, Random Forest, "
        "XGBoost, and LightGBM side by side:"
    )
    st.code("mlflow ui --backend-store-uri mlruns", language="bash")
    st.markdown("Then open [http://localhost:5000](http://localhost:5000)")
