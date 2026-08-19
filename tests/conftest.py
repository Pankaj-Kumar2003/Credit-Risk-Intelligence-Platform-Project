"""Shared pytest fixtures: a small synthetic dataset used by every test
module so tests don't depend on data/raw/ existing or on training time."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.utils.helper import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def sample_raw_df():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "applicant_id": [f"APP{i}" for i in range(n)],
        "annual_income": rng.uniform(20000, 150000, n),
        "loan_amount": rng.uniform(1000, 50000, n),
        "credit_score": rng.uniform(400, 820, n),
        "employment_years": rng.uniform(0, 30, n),
        "age": rng.integers(21, 70, n),
        "existing_debt": rng.uniform(0, 50000, n),
        "credit_limit": rng.uniform(1000, 80000, n),
        "num_open_accounts": rng.integers(0, 15, n),
        "num_credit_inquiries_6m": rng.integers(0, 6, n),
        "num_previous_defaults": rng.integers(0, 3, n),
        "delinquencies_2y": rng.integers(0, 4, n),
        "loan_purpose": rng.choice(["debt_consolidation", "auto", "medical"], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "application_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "account_open_date": pd.date_range("2018-01-01", periods=n, freq="D"),
        "default_flag": rng.integers(0, 2, n),
    })
    return df
