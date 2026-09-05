from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

from app.services.forecast_service import _load_dataframe
from src.demand_intelligence.forecasting import PRODUCTION_MODEL_PATH, train_and_select_model
from src.demand_intelligence.model_registry import REGISTRY_DIR, _write, compare_models, ensure_current_registered, list_models


def retrain_model() -> dict[str, Any]:
    champion = ensure_current_registered()
    champion_backup = REGISTRY_DIR / champion["model_version"] / "model.joblib"
    champion_backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRODUCTION_MODEL_PATH, champion_backup)
    champion["artifact_path"] = str(champion_backup)
    artifact = train_and_select_model(_load_dataframe())
    candidate = {"model_version": artifact["model_version"], "model_name": artifact["model_name"], "metrics": artifact["metrics"], "training_run_id": f"train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"}
    comparison = compare_models(champion["metrics"], candidate["metrics"])
    candidate_path = REGISTRY_DIR / candidate["model_version"] / "model.joblib"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRODUCTION_MODEL_PATH, candidate_path)
    records = list_models()
    for item in records:
        if item["model_version"] == champion["model_version"]:
            item["artifact_path"] = str(champion_backup)
    for item in records:
        item["is_champion"] = False
        item["is_production"] = False
        if item["status"] == "production":
            item["status"] = "archived"
    records.append({
        "model_id": f"{candidate['model_name'].lower().replace(' ', '-')}-{candidate['model_version']}",
        "model_name": candidate["model_name"], "model_type": type(artifact["model"]).__name__,
        "model_version": candidate["model_version"], "artifact_path": str(candidate_path),
        "created_at": artifact.get("trained_at", datetime.now(timezone.utc).isoformat()),
        "feature_count": len(artifact.get("feature_columns", [])), "feature_names": artifact.get("feature_columns", []),
        "target_name": "units_sold", "metrics": candidate["metrics"],
        "status": "production" if comparison["decision"] == "PROMOTED" else "rejected",
        "is_champion": comparison["decision"] == "PROMOTED", "is_production": comparison["decision"] == "PROMOTED",
        "parent_model_version": champion["model_version"], "training_run_id": candidate["training_run_id"], "metadata": {},
    })
    if comparison["decision"] == "PROMOTED":
        shutil.copy2(candidate_path, PRODUCTION_MODEL_PATH)
        for item in records:
            if item["model_version"] == candidate["model_version"]:
                item["status"] = "production"
                item["is_champion"] = True
                item["is_production"] = True
    else:
        shutil.copy2(champion_backup, PRODUCTION_MODEL_PATH)
        for item in records:
            if item["model_version"] == champion["model_version"]:
                item["status"] = "production"
                item["is_champion"] = True
                item["is_production"] = True
    _write(records)
    return {**candidate, "champion_model_version": champion["model_version"], **comparison}
