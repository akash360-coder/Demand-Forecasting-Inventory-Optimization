import json
from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestRegressor

import src.demand_intelligence.model_registry as registry


def record(tmp_path, version="v1", status="candidate", production=False):
    artifact = tmp_path / f"{version}.joblib"
    joblib.dump({"model": RandomForestRegressor(), "feature_columns": registry.FEATURE_COLUMNS}, artifact)
    return {"model_version": version, "model_name": "Test Model", "model_type": "RandomForestRegressor", "artifact_path": str(artifact), "metrics": {"wmape": 10.0, "mae": 1.0, "rmse": 2.0, "mape": 3.0}, "status": status, "is_production": production, "is_champion": production, "feature_names": registry.FEATURE_COLUMNS, "feature_count": 18, "parent_model_version": "parent", "training_run_id": f"train_{version}", "metadata": {"artifact_sha256": "hash"}}


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(registry, "REGISTRY_FILE", tmp_path / "registry" / "registry.json")
    monkeypatch.setattr(registry, "PRODUCTION_MODEL_PATH", tmp_path / "production.joblib")
    monkeypatch.setattr(registry, "ensure_current_registered", lambda: None)
    return tmp_path


def test_register_model_persists_metadata(isolated_registry):
    item = registry.register_model(record(isolated_registry))
    assert registry.REGISTRY_FILE.exists()
    assert registry.get_model("v1")["model_name"] == "Test Model"


def test_register_model_preserves_version(isolated_registry):
    assert registry.register_model(record(isolated_registry))["model_version"] == "v1"


def test_register_model_preserves_type(isolated_registry):
    assert registry.register_model(record(isolated_registry))["model_type"] == "RandomForestRegressor"


def test_register_model_preserves_training_run(isolated_registry):
    assert registry.register_model(record(isolated_registry))["training_run_id"] == "train_v1"


def test_register_model_preserves_artifact_path(isolated_registry):
    item = record(isolated_registry)
    assert registry.register_model(item)["artifact_path"] == item["artifact_path"]


def test_register_model_preserves_hash(isolated_registry):
    assert registry.register_model(record(isolated_registry))["metadata"]["artifact_sha256"] == "hash"


def test_register_model_preserves_feature_contract(isolated_registry):
    assert registry.register_model(record(isolated_registry))["feature_count"] == 18


def test_register_model_preserves_metrics(isolated_registry):
    assert registry.register_model(record(isolated_registry))["metrics"]["wmape"] == 10


def test_register_model_preserves_parent_version(isolated_registry):
    assert registry.register_model(record(isolated_registry))["parent_model_version"] == "parent"


@pytest.mark.parametrize("status", ["candidate", "validated", "approved", "production", "rejected", "archived"])
def test_registry_statuses_are_supported(isolated_registry, status):
    item = record(isolated_registry, version=status, status=status)
    assert registry.register_model(item)["status"] == status


def test_duplicate_version_is_rejected(isolated_registry):
    registry.register_model(record(isolated_registry))
    with pytest.raises(ValueError):
        registry.register_model(record(isolated_registry))


def test_unknown_version_is_rejected(isolated_registry):
    with pytest.raises(ValueError):
        registry.get_model("missing")


def test_listing_returns_only_registered_models(isolated_registry):
    registry.register_model(record(isolated_registry))
    assert [item["model_version"] for item in registry.list_models()] == ["v1"]


def test_production_champion_is_unique(isolated_registry):
    registry.register_model(record(isolated_registry, production=True, status="production"))
    registry.register_model(record(isolated_registry, version="v2", production=True, status="production"))
    models = registry.list_models()
    assert sum(item["is_production"] for item in models) == 1
    assert sum(item["is_champion"] for item in models) == 1


def test_promotion_updates_champion_state(isolated_registry):
    registry.register_model(record(isolated_registry, production=True, status="production"))
    registry.register_model(record(isolated_registry, version="v2"))
    promoted = registry.promote_model("v2")
    assert promoted["is_production"] is True
    assert registry.get_model("v1")["status"] == "archived"


def test_incompatible_promotion_is_rejected(isolated_registry):
    item = record(isolated_registry)
    bad = Path(item["artifact_path"])
    joblib.dump({"model": RandomForestRegressor(), "feature_columns": ["bad"]}, bad)
    registry.register_model(item)
    with pytest.raises(ValueError):
        registry.promote_model("v1")


def test_missing_artifact_is_rejected(isolated_registry):
    item = record(isolated_registry)
    Path(item["artifact_path"]).unlink()
    registry.register_model(item)
    with pytest.raises(ValueError):
        registry.promote_model("v1")


def test_registry_json_is_valid(isolated_registry):
    registry.register_model(record(isolated_registry))
    assert json.loads(registry.REGISTRY_FILE.read_text())
