import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.training.train_model import get_candidate_models


def test_get_candidate_models_returns_four_models(config):
    models = get_candidate_models(config)
    assert set(models.keys()) == {"logistic_regression", "random_forest", "xgboost", "lightgbm"}


def test_logistic_regression_trains_and_predicts(config):
    models = get_candidate_models(config)
    lr: LogisticRegression = models["logistic_regression"]

    import numpy as np
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
    y = (x["f0"] + rng.normal(size=100) > 0).astype(int)

    lr.fit(x, y)
    proba = lr.predict_proba(x)[:, 1]
    assert proba.shape[0] == 100
    assert ((proba >= 0) & (proba <= 1)).all()
