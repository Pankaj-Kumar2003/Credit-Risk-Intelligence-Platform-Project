# Credit Risk Intelligence Platform

This project is my end-to-end credit risk modeling portfolio project. I built
it to go beyond a notebook and create a realistic ML system that follows the
full lifecycle of a production credit model: ingesting raw applicant data,
validating it, cleaning it, engineering risk features, training and comparing
models, explaining prediction results, and monitoring whether the model still
performs as the data changes over time.

I wanted to understand how a credit decision system is designed in practice,
where the model is only one part of the solution. The real value comes from
clean data pipelines, business-aware features, explainability for stakeholders,
and monitoring so a model does not silently drift after deployment.

## My Project Goals

- Build a complete machine learning workflow for credit default prediction.
- Learn how data quality checks and schema validation impact model reliability.
- Create business-relevant features such as debt-to-income ratios and utilization
  metrics instead of only training on raw columns.
- Compare multiple model families and track experiments with MLflow.
- Add explainability so a prediction is not just a probability but something a
  reviewer can interpret.
- Explore production monitoring by checking for data drift over time.
- Deliver an interactive interface for both API-based and dashboard-based access.

## Learning Outcomes

Through this project, I strengthened my understanding of:

- end-to-end ML pipeline design from raw files to model inference
- feature engineering for financial risk use cases
- model evaluation with metrics that matter for imbalance-heavy lending data
- SHAP-based explainability for stakeholder trust and model interpretation
- data drift monitoring and why retraining decisions need to be data-driven
- deployment patterns using FastAPI and Streamlit for model serving and review

## Business Problem

Lending institutions need a fast and defensible way to estimate whether a
borrower is likely to default. In real scenarios, manual review is slow,
black-box predictions are hard to explain, and stale models can create risk
without any visible signal until losses begin to appear.

This project addresses that problem by combining prediction, interpretation, and
monitoring in one system. A user can load applicant data, get a risk score,
understand which factors are pushing the decision, and monitor whether the
incoming data distribution has shifted enough to require review or retraining.

## Project Approach and Design Decisions

I designed this project around a practical workflow rather than a toy example:

- Data ingestion is configuration-driven so the pipeline can map source columns
  to a canonical schema without hardcoding dataset-specific logic.
- Schema validation and data checks run before training so bad inputs are caught
  early.
- Preprocessing and feature engineering are implemented as a reusable pipeline
  instead of ad hoc transformations in one notebook.
- Model training compares multiple algorithms and logs experiments to MLflow to
  support reproducibility and comparison.
- Hyperparameter search is included so model tuning is not left out of the
  workflow.
- Evaluation focuses on credit-relevant metrics such as ROC-AUC, PR-AUC, and
  calibration-oriented monitoring.
- SHAP explainability is added to show the top drivers behind individual
  predictions.
- A FastAPI service and Streamlit dashboard separate the serving layer from the
  experimentation layer, which is closer to how ML products are built.
- Drift monitoring is added as a production-minded safeguard against model
  degradation in the field.

## Dataset

The pipeline is intentionally dataset-agnostic: `config/config.yaml` maps
canonical column names such as `annual_income`, `credit_score`, and
`default_flag` to the actual names used by the source file. This lets the
project work with different public credit datasets or a generated synthetic
version when no external dataset is available.

Possible data sources include:

- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

If a dataset is not available yet, this project can generate a realistic mock
credit dataset:

```bash
python scripts/generate_mock_data.py
```

This creates `data/raw/loan_applications.csv` with the same schema expected by
later stages, including realistic missing values and a non-trivial default
signal.

## Installation

```bash
git clone <repo-url>
cd credit-risk-intelligence-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Get data (or drop a real dataset into data/raw/ and update config.yaml)
python scripts/generate_mock_data.py

# 2. Run the full pipeline: preprocess -> train -> evaluate -> explain -> monitor
python main.py --stage all

# Or run stages individually
python main.py --stage preprocess
python main.py --stage train
python main.py --stage tune       # Optuna hyperparameter search
python main.py --stage evaluate
python main.py --stage explain
python main.py --stage monitor

# 3. Serve
uvicorn api.main:app --reload
streamlit run dashboard/app.py

# 4. Inspect experiments
mlflow ui --backend-store-uri mlruns
```

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) has eight pages: Project
Overview, Dataset Explorer, Feature Analysis, Model Performance, Risk
Prediction (score a new applicant interactively), SHAP Explainability,
Drift Monitoring, and Model Comparison.

<!-- Add screenshots here after running `streamlit run dashboard/app.py` -->

## API

FastAPI service (`api/main.py`), endpoints:

| Method | Path                  | Purpose                              |
|--------|-----------------------|---------------------------------------|
| GET    | `/health`              | Liveness + model-loaded check         |
| GET    | `/model-info`          | Model type and feature list           |
| POST   | `/predict`              | Score a single applicant              |
| POST   | `/predict-batch`        | Score a list of applicants            |
| GET    | `/feature-importance`  | Global SHAP feature importance        |

Interactive docs: `http://localhost:8000/docs`

## Docker

```bash
docker-compose up --build
```

Starts the API (`:8000`), dashboard (`:8501`), and MLflow UI (`:5000`)
together.

## Testing

```bash
pytest -v
```

Covers ingestion column-mapping, schema/quality validation, feature
engineering invariants, model construction, and API health/predict
endpoints.

## Folder Structure

```
credit-risk-intelligence-platform/
├── data/{raw,interim,processed,external}
├── notebooks/                 # exploratory notebooks (01-05)
├── config/config.yaml         # single source of truth for paths & params
├── src/
│   ├── ingestion/              # raw file -> canonical schema
│   ├── validation/             # schema + data quality checks
│   ├── preprocessing/          # cleaning + full pipeline orchestration
│   ├── features/                # business feature engineering
│   ├── training/                 # 4-model training + MLflow logging
│   ├── tuning/                    # Optuna hyperparameter search
│   ├── evaluation/                 # metrics, curves, lift/gain
│   ├── explainability/              # SHAP global + local
│   ├── monitoring/                   # Evidently drift reports
│   ├── prediction/                    # inference service
│   └── utils/                          # config, logging, model loading
├── api/main.py                # FastAPI service
├── dashboard/app.py           # Streamlit dashboard
├── models/                    # persisted champion model + preprocessor
├── reports/                   # evaluation plots, SHAP plots, drift reports
├── tests/                     # pytest suite
├── Dockerfile / docker-compose.yml
├── requirements.txt
└── main.py                    # pipeline entry point
```

## Results

Run `python main.py --stage all` against your own data and the champion
model's metrics will be written to `reports/champion_evaluation_metrics.json`
and displayed on the dashboard's Model Performance page. On the synthetic
dataset shipped with `scripts/generate_mock_data.py`, LightGBM/XGBoost
typically land around ROC-AUC 0.82–0.86 versus ~0.75 for the logistic
regression baseline — replace with your real numbers once trained on an
actual dataset.

## Future Improvements

- Add a feature store (Feast) for consistent online/offline feature parity
- Champion/challenger shadow deployment before promoting a new model
- Automated retraining trigger off the Evidently drift report
- Fairness/bias audits across protected attributes
- CI pipeline (GitHub Actions) running `pytest` + `main.py --stage all` on every PR

## Project Profile

This project is a personal portfolio implementation demonstrating end-to-end
credit risk modeling, model monitoring, and MLOps practices. It is intended
as a learning and showcase project for machine learning and data science work.

If you adapt this repository for your own portfolio, make sure the final
version reflects your own implementation details, naming, configuration, and
project documentation.