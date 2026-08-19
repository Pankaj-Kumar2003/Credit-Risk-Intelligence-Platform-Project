from src.features.feature_engineering import FeatureEngineer
from src.preprocessing.data_cleaning import DataCleaner


def test_feature_engineering_adds_expected_columns(config, sample_raw_df):
    cleaned = DataCleaner(config).fit_transform(sample_raw_df)
    engineered = FeatureEngineer(config).engineer(cleaned)

    for col in ["debt_to_income_ratio", "credit_utilization_ratio", "heuristic_risk_score",
                "adverse_event_count", "income_bracket"]:
        assert col in engineered.columns


def test_heuristic_risk_score_is_bounded(config, sample_raw_df):
    cleaned = DataCleaner(config).fit_transform(sample_raw_df)
    engineered = FeatureEngineer(config).engineer(cleaned)
    assert engineered["heuristic_risk_score"].between(0, 100).all()


def test_debt_to_income_ratio_nonnegative(config, sample_raw_df):
    cleaned = DataCleaner(config).fit_transform(sample_raw_df)
    engineered = FeatureEngineer(config).engineer(cleaned)
    assert (engineered["debt_to_income_ratio"] >= 0).all()
