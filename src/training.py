"""
Reproducible training entry point for the demand forecasting pipeline.

Usage:
    python -m src.training
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.data_validation import get_clean_dataset, validate_data_quality
from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS
from src.demand_intelligence.forecasting import run_experiment, train_and_select_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline() -> dict:
    """
    Execute the complete training pipeline.

    Returns:
        Dictionary containing training results and model metadata.
    """
    logger.info("=" * 70)
    logger.info("DEMAND FORECASTING PIPELINE - TRAINING")
    logger.info("=" * 70)

    # STEP 1: Load dataset
    logger.info("\n[STEP 1] Loading dataset...")
    df = ensure_dataset()
    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Products: {df['product_id'].nunique()}")
    logger.info(f"  Stores: {df['store_id'].nunique()}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # STEP 2: Validate data quality
    logger.info("\n[STEP 2] Validating data quality...")
    report = validate_data_quality(df)
    logger.info(f"  {report.summary()}")

    if not report.is_valid:
        critical = [i for i in report.issues if i.severity == "CRITICAL"]
        if critical:
            raise ValueError(f"Data quality issues prevent training: {critical}")

    # STEP 3: Clean dataset
    logger.info("\n[STEP 3] Cleaning dataset...")
    df_clean = get_clean_dataset(df)
    logger.info(f"  Cleaned dataset: {len(df_clean):,} rows")
    logger.info(f"  Removed {len(df) - len(df_clean)} invalid records")

    # STEP 4: Run experiments
    logger.info("\n[STEP 4] Running model experiments...")
    logger.info("  Evaluating: Naive, Seasonal Naive, Moving Average, Random Forest, XGBoost, LightGBM")
    experiment = run_experiment(df_clean)

    logger.info(f"\n  Experiment Results:")
    logger.info(f"  Dataset rows used: {experiment['dataset_rows']:,}")
    logger.info(f"  Validation strategy: {experiment['validation_strategy']}")
    logger.info(f"\n  Model Comparison (Validation Set):")
    logger.info(f"  {'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'WMAPE':<10}")
    logger.info("  " + "-" * 60)

    for result in experiment["results"]:
        name = result["model_name"]
        metrics = result["validation_metrics"]
        marker = " ← SELECTED" if name == experiment["selected_model"] else ""
        logger.info(
            f"  {name:<20} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} "
            f"{metrics['mape']:<10.2f} {metrics['wmape']:<10.2f}{marker}"
        )

    logger.info(f"\n  Selected Model: {experiment['selected_model']}")
    logger.info(f"  Validation Metrics: {experiment['selected_validation_metrics']}")

    # STEP 5: Train and persist final model
    logger.info("\n[STEP 5] Training and persisting final model...")
    artifact = train_and_select_model(df_clean)
    logger.info(f"  Model saved to: {Path('models/production_forecast_model.joblib').resolve()}")
    logger.info(f"  Model version: {artifact['model_version']}")
    logger.info(f"  Features: {len(FEATURE_COLUMNS)} features")

    # Create summary report
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": {
            "total_rows": len(df),
            "valid_rows": len(df_clean),
            "removed_rows": len(df) - len(df_clean),
            "products": df_clean["product_id"].nunique(),
            "stores": df_clean["store_id"].nunique(),
            "date_range": {
                "start": str(df_clean["date"].min()),
                "end": str(df_clean["date"].max()),
            },
        },
        "validation": {
            "total_issues": len(report.issues),
            "critical_issues": len([i for i in report.issues if i.severity == "CRITICAL"]),
            "warnings": len([i for i in report.issues if i.severity == "WARNING"]),
        },
        "feature_engineering": {
            "total_features": len(FEATURE_COLUMNS),
            "feature_list": FEATURE_COLUMNS,
        },
        "experiment": {
            "strategy": experiment["validation_strategy"],
            "models_evaluated": len(experiment["results"]),
            "selected_model": experiment["selected_model"],
        },
        "metrics": {
            "validation": {
                "mae": float(experiment["selected_validation_metrics"]["mae"]),
                "rmse": float(experiment["selected_validation_metrics"]["rmse"]),
                "mape": float(experiment["selected_validation_metrics"]["mape"]),
                "wmape": float(experiment["selected_validation_metrics"]["wmape"]),
            },
            "test": {
                "mae": float(experiment["selected_test_metrics"]["mae"]),
                "rmse": float(experiment["selected_test_metrics"]["rmse"]),
                "mape": float(experiment["selected_test_metrics"]["mape"]),
                "wmape": float(experiment["selected_test_metrics"]["wmape"]),
            },
        },
        "model": {
            "name": artifact["model_name"],
            "version": artifact["model_version"],
            "trained_at": artifact["trained_at"],
        },
    }

    # Save summary report
    report_path = Path("models/training_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  Training summary saved to: {report_path.resolve()}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE ✓")
    logger.info("=" * 70)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Start API: python -m uvicorn app.main:app --reload")
    logger.info(f"  2. Test forecast: curl http://localhost:8000/api/v1/forecast")
    logger.info(f"  3. View performance: curl http://localhost:8000/api/v1/models/performance")
    logger.info("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    run_training_pipeline()
