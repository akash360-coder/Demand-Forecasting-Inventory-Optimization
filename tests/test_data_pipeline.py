from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.feature_engineering import build_feature_matrix


def test_dataset_gen_and_features():
    df = ensure_dataset()
    assert len(df) > 0
    features = build_feature_matrix(df.head(50).copy())
    assert "lag_1" in features.columns
    assert "rolling_mean_7" in features.columns
