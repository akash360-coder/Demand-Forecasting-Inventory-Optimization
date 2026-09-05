from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS
from src.demand_intelligence.forecasting import PRODUCTION_MODEL_PATH, load_production_model

REGISTRY_DIR = PRODUCTION_MODEL_PATH.parent / "registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read() -> list[dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return []
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _write(records: list[dict[str, Any]]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def ensure_current_registered() -> dict[str, Any]:
    artifact = load_production_model()
    version = str(artifact.get("model_version", "unknown"))
    records = _read()
    existing = next((item for item in records if item["model_version"] == version), None)
    if existing:
        return existing
    record = {
        "model_id": f"{artifact.get('model_name', 'model').lower().replace(' ', '-')}-{version}",
        "model_name": artifact.get("model_name", type(artifact["model"]).__name__),
        "model_type": type(artifact["model"]).__name__,
        "model_version": version,
        "artifact_path": str(PRODUCTION_MODEL_PATH),
        "created_at": artifact.get("trained_at", _now()),
        "feature_count": len(artifact.get("feature_columns", FEATURE_COLUMNS)),
        "feature_names": artifact.get("feature_columns", FEATURE_COLUMNS),
        "target_name": "units_sold",
        "metrics": artifact.get("metrics", {}),
        "status": "production",
        "is_champion": True,
        "is_production": True,
        "parent_model_version": None,
        "training_run_id": f"train_{version}",
        "metadata": {"artifact_sha256": _hash(PRODUCTION_MODEL_PATH)},
    }
    for item in records:
        item["is_champion"] = False
        item["is_production"] = False
        if item["status"] == "production":
            item["status"] = "archived"
    records.append(record)
    _write(records)
    return record


def list_models() -> list[dict[str, Any]]:
    ensure_current_registered()
    return _read()


def get_model(version: str) -> dict[str, Any]:
    record = next((item for item in list_models() if item["model_version"] == version), None)
    if record is None:
        raise ValueError(f"Model version {version!r} is not registered.")
    return record


def register_model(record: dict[str, Any]) -> dict[str, Any]:
    required = {"model_version", "model_name", "model_type", "artifact_path", "metrics", "status"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Registry metadata is missing: {', '.join(sorted(missing))}.")
    records = _read()
    if any(item["model_version"] == record["model_version"] for item in records):
        raise ValueError(f"Model version {record['model_version']!r} is already registered.")
    normalized = {
        "model_id": record.get("model_id", f"{record['model_name']}-{record['model_version']}"),
        "model_name": record["model_name"],
        "model_type": record["model_type"],
        "model_version": record["model_version"],
        "artifact_path": record["artifact_path"],
        "created_at": record.get("created_at", _now()),
        "feature_count": record.get("feature_count", len(record.get("feature_names", []))),
        "feature_names": record.get("feature_names", FEATURE_COLUMNS),
        "target_name": record.get("target_name", "units_sold"),
        "metrics": record["metrics"],
        "status": record["status"],
        "is_champion": bool(record.get("is_champion", False)),
        "is_production": bool(record.get("is_production", False)),
        "parent_model_version": record.get("parent_model_version"),
        "training_run_id": record.get("training_run_id", f"train_{record['model_version']}"),
        "metadata": record.get("metadata", {}),
    }
    if normalized["is_production"] or normalized["is_champion"]:
        for item in records:
            item["is_production"] = False
            item["is_champion"] = False
            if item["status"] == "production":
                item["status"] = "archived"
    records.append(normalized)
    _write(records)
    return normalized


def promote_model(version: str) -> dict[str, Any]:
    record = get_model(version)
    if not Path(record["artifact_path"]).exists():
        raise ValueError("Registered model artifact is missing.")
    artifact = joblib.load(record["artifact_path"])
    if not hasattr(artifact.get("model"), "predict") or artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Registered model artifact is incompatible.")
    shutil.copy2(record["artifact_path"], PRODUCTION_MODEL_PATH)
    records = list_models()
    for item in records:
        item["is_champion"] = item["model_version"] == version
        item["is_production"] = item["model_version"] == version
        item["status"] = "production" if item["model_version"] == version else "archived"
    _write(records)
    return get_model(version)


def compare_models(champion_metrics: dict[str, float], challenger_metrics: dict[str, float], minimum_improvement: float = 0.01) -> dict[str, Any]:
    champion = float(champion_metrics["wmape"])
    challenger = float(challenger_metrics["wmape"])
    improvement = (champion - challenger) / champion if champion else 0.0
    promoted = challenger < champion and improvement >= minimum_improvement
    return {"champion_metrics": champion_metrics, "challenger_metrics": challenger_metrics, "wmape_improvement": improvement, "decision": "PROMOTED" if promoted else "REJECTED", "reason": "Challenger meets the configured WMAPE improvement threshold." if promoted else "Challenger does not meet the configured WMAPE improvement threshold."}


def rollback_to_model(version: str) -> dict[str, Any]:
    record = get_model(version)
    artifact_path = Path(record["artifact_path"])
    if not artifact_path.exists():
        raise ValueError("Registered model artifact is missing.")
    artifact = joblib.load(artifact_path)
    if not hasattr(artifact.get("model"), "predict") or artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Registered model artifact is incompatible.")
    shutil.copy2(artifact_path, PRODUCTION_MODEL_PATH)
    records = list_models()
    for item in records:
        item["is_champion"] = item["model_version"] == version
        item["is_production"] = item["model_version"] == version
        item["status"] = "production" if item["model_version"] == version else "archived"
    _write(records)
    return get_model(version)
