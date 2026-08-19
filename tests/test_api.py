from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_without_model_returns_503_or_200():
    # Model may or may not be trained in the test environment — either
    # a clean 503 (untrained) or a 200 (trained) is acceptable; a 500
    # crash is not.
    payload = {
        "applicant_id": "TEST001", "annual_income": 65000, "loan_amount": 15000,
        "credit_score": 700, "employment_years": 5, "age": 34, "existing_debt": 8000,
        "credit_limit": 20000, "num_open_accounts": 4, "num_credit_inquiries_6m": 1,
        "num_previous_defaults": 0, "delinquencies_2y": 0, "loan_purpose": "auto",
        "home_ownership": "RENT",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)
