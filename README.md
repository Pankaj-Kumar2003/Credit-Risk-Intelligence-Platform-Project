# Credit Risk Intelligence Platform

An end-to-end machine learning platform that predicts loan-applicant default
risk, explains every prediction, and monitors model health in production.
Built to reflect how credit risk models are actually developed and deployed
in banking and fintech, not a Kaggle notebook.

## Business Problem

Lenders need to decide, quickly and defensibly, whether an applicant is
likely to default. Manual underwriting doesn't scale, black-box models don't
survive a compliance review, and models that go stale silently cost money.
This platform addresses all three: it predicts risk, explains *why*, and
tells you when the data has drifted enough that the model needs a refresh.

**What it produces for each applicant:**
- Default probability
- Risk category (Low / Moderate / High / Very High)
- SHAP-based explanation of the top drivers behind that score

## Architecture

```
Data Sources (Home Credit / Lending Club / Give Me Some Credit — or the
              included synthetic generator)
   |
   v
Ingestion (src/ingestion) --------- column-mapping layer, dataset-agnostic
   |
   v
Validation (src/validation) ------- schema checks + data quality checks
   |
   v
Cleaning (src/preprocessing) ------ dedup, imputation, outlier capping
   |
   v
Feature Engineering (src/features)  DTI, utilization, tenure, risk-history
   |
   v
Train/Test Split + Encoding (src/preprocessing/preprocessing_pipeline.py)
   |
   v
Model Training (src/training) ----- LogReg / RandomForest / XGBoost / LightGBM
   |                                  tracked in MLflow
   v
Hyperparameter Tuning (src/tuning)  Optuna, logged as nested MLflow runs
   |
   v
Evaluation (src/evaluation) ------- AUC, PR-AUC, calibration, lift/gain
   |
   v
Explainability (src/explainability) SHAP global + local explanations
   |
   v
   +-----------------+------------------+
   |                                    |
   v                                    v
FastAPI (api/)                  Streamlit Dashboard (dashboard/)
   |                                    |
   +------------------+-----------------+
                       v
        Drift Monitoring (src/monitoring, Evidently AI)
```

## Dataset

The pipeline is dataset-agnostic by design: `config/config.yaml` maps
canonical column names (`annual_income`, `credit_score`, `default_flag`, …)
onto whatever the source file actually calls them. Point it at:

- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

No dataset yet? Generate a realistic synthetic one:

```bash
python scripts/generate_mock_data.py
```

This writes `data/raw/loan_applications.csv` with the same schema the rest
of the pipeline expects, including realistic missingness and a
non-linearly-separable default signal.

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