import hashlib

from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.forecasting import train_and_select_model


def test_isolated_training_preserves_production_artifact(tmp_path, monkeypatch):
    from src.demand_intelligence import forecasting

    production = tmp_path / "production.joblib"
    production.write_bytes(b"canonical-production-artifact")
    before = hashlib.sha256(production.read_bytes()).hexdigest()
    monkeypatch.setattr(forecasting, "PRODUCTION_MODEL_PATH", production)
    train_and_select_model(ensure_dataset().head(300), output_path=tmp_path / "candidate.joblib")
    assert hashlib.sha256(production.read_bytes()).hexdigest() == before


def test_training_output_path_is_created(tmp_path):
    output = tmp_path / "nested" / "candidate.joblib"
    train_and_select_model(ensure_dataset().head(300), output_path=output)
    assert output.exists()


def test_training_output_path_is_distinct_from_default(tmp_path):
    from src.demand_intelligence.forecasting import PRODUCTION_MODEL_PATH

    output = tmp_path / "candidate.joblib"
    train_and_select_model(ensure_dataset().head(300), output_path=output)
    assert output.resolve() != PRODUCTION_MODEL_PATH.resolve()


def test_isolated_training_artifact_loads(tmp_path):
    import joblib

    output = tmp_path / "candidate.joblib"
    train_and_select_model(ensure_dataset().head(300), output_path=output)
    artifact = joblib.load(output)
    assert artifact["model"] is not None
    assert artifact["feature_columns"]


def test_default_production_path_remains_configured():
    from src.demand_intelligence.forecasting import PRODUCTION_MODEL_PATH

    assert PRODUCTION_MODEL_PATH.name == "production_forecast_model.joblib"
