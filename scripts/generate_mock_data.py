"""
Generates a synthetic loan-applicant dataset that mirrors the schema
of Home Credit / Lending Club style datasets, using the column mapping
defined in config/config.yaml. Run this once before the pipeline if
you don't have a real dataset in data/raw/ yet:

    python scripts/generate_mock_data.py

Swap this out for a real download (Home Credit / Lending Club / Give
Me Some Credit) and the rest of the pipeline works unchanged, because
every downstream module reads columns through the config mapping.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_mock_dataset(n_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_rows)
    employment_years = np.clip(rng.normal(6, 5, n_rows), 0, 40).round(1)
    annual_income = np.clip(rng.lognormal(mean=10.8, sigma=0.5, size=n_rows), 12000, 500000).round(2)
    credit_score = np.clip(rng.normal(670, 80, n_rows), 300, 850).round(0)
    loan_amount = np.clip(rng.lognormal(mean=9.2, sigma=0.6, size=n_rows), 1000, 100000).round(2)
    existing_debt = np.clip(rng.lognormal(mean=8.8, sigma=0.9, size=n_rows), 0, 200000).round(2)
    credit_limit = np.clip(rng.lognormal(mean=9.5, sigma=0.6, size=n_rows), 500, 150000).round(2)
    num_open_accounts = rng.integers(0, 20, n_rows)
    num_credit_inquiries_6m = rng.poisson(1.2, n_rows)
    num_previous_defaults = rng.poisson(0.25, n_rows)
    delinquencies_2y = rng.poisson(0.4, n_rows)

    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "auto",
         "small_business", "major_purchase", "education", "other"],
        size=n_rows,
        p=[0.30, 0.15, 0.08, 0.12, 0.10, 0.10, 0.05, 0.10],
    )
    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"], size=n_rows, p=[0.42, 0.40, 0.15, 0.03]
    )

    application_date = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n_rows), unit="D"
    )
    account_open_date = application_date - pd.to_timedelta(
        (employment_years * 365).astype(int) + rng.integers(0, 400, n_rows), unit="D"
    )

    # Latent default risk signal — realistic, not linearly separable.
    debt_to_income = existing_debt / np.maximum(annual_income, 1)
    utilization = existing_debt / np.maximum(credit_limit, 1)

    risk_logit = (
        -2.4  # intercept, calibrated so the base default rate lands near real-world (~15-20%)
        -0.012 * (credit_score - 650)
        + 1.8 * debt_to_income
        + 1.1 * utilization
        + 0.55 * num_previous_defaults
        + 0.35 * delinquencies_2y
        + 0.15 * num_credit_inquiries_6m
        - 0.04 * employment_years
        + 0.000006 * loan_amount
        - 0.000004 * annual_income
        + rng.normal(0, 0.6, n_rows)
    )
    default_prob = 1 / (1 + np.exp(-risk_logit))
    default_flag = (rng.uniform(0, 1, n_rows) < default_prob).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"APP{100000 + i}" for i in range(n_rows)],
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "age": age,
        "existing_debt": existing_debt,
        "credit_limit": credit_limit,
        "num_open_accounts": num_open_accounts,
        "num_credit_inquiries_6m": num_credit_inquiries_6m,
        "num_previous_defaults": num_previous_defaults,
        "delinquencies_2y": delinquencies_2y,
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
        "application_date": application_date,
        "account_open_date": account_open_date,
        "default_flag": default_flag,
    })

    # Inject realistic messiness: missing values, a few duplicates.
    for col in ["credit_score", "employment_years", "existing_debt"]:
        mask = rng.uniform(0, 1, n_rows) < 0.02
        df.loc[mask, col] = np.nan

    dup_rows = df.sample(frac=0.005, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main() -> None:
    config = load_config()
    raw_path = resolve_path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_mock_dataset()
    df.to_csv(raw_path, index=False)
    logger.info(f"Mock dataset written to {raw_path} — shape={df.shape}, default_rate={df['default_flag'].mean():.3f}")


if __name__ == "__main__":
    main()
