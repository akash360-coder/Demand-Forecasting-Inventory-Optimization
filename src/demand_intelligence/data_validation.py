"""Comprehensive data validation for demand forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    """Single validation issue found in data."""

    category: str
    severity: str
    message: str
    affected_rows: int | None = None
    details: dict[str, Any] | None = None


@dataclass
class ValidationReport:
    """Complete validation report for dataset."""

    dataset_rows: int
    valid_rows: int
    issues: list[ValidationIssue]
    timestamp: str
    is_valid: bool

    def summary(self) -> str:
        """Return human-readable summary."""
        msg = f"Dataset: {self.dataset_rows} rows, Valid: {self.valid_rows} rows\n"
        if not self.issues:
            return msg + "✓ No issues found"
        msg += f"Issues: {len(self.issues)}\n"
        for issue in self.issues:
            severity_icon = "🔴" if issue.severity == "CRITICAL" else "🟡" if issue.severity == "WARNING" else "🔵"
            msg += f"{severity_icon} [{issue.category}] {issue.message}"
            if issue.affected_rows is not None:
                msg += f" ({issue.affected_rows} rows)"
            msg += "\n"
        return msg


def validate_required_columns(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate that all required columns exist."""
    issues: list[ValidationIssue] = []
    required = {"date", "product_id", "store_id", "units_sold", "price"}
    missing = required - set(df.columns)
    if missing:
        issues.append(
            ValidationIssue(
                category="SCHEMA",
                severity="CRITICAL",
                message=f"Missing required columns: {missing}",
            )
        )
    return issues


def validate_missing_values(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate missing values in critical columns."""
    issues: list[ValidationIssue] = []
    critical_cols = ["date", "product_id", "store_id", "units_sold"]
    for col in critical_cols:
        if col not in df.columns:
            continue
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            severity = "CRITICAL" if missing_count > len(df) * 0.05 else "WARNING"
            issues.append(
                ValidationIssue(
                    category="MISSING_VALUES",
                    severity=severity,
                    message=f"Column '{col}' has {missing_count} missing values",
                    affected_rows=missing_count,
                )
            )
    return issues


def validate_duplicates(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate duplicate records."""
    issues: list[ValidationIssue] = []
    key_cols = ["date", "product_id", "store_id"]
    existing_cols = [c for c in key_cols if c in df.columns]
    if not existing_cols:
        return issues

    duplicates = df.duplicated(subset=existing_cols, keep=False).sum()
    if duplicates > 0:
        issues.append(
            ValidationIssue(
                category="DUPLICATES",
                severity="WARNING",
                message=f"Found {duplicates} duplicate records (date × product × store)",
                affected_rows=duplicates,
            )
        )
    return issues


def validate_dates(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate date column."""
    issues: list[ValidationIssue] = []
    if "date" not in df.columns:
        return issues

    # Check if date can be converted
    try:
        dates = pd.to_datetime(df["date"])
    except Exception as e:
        issues.append(
            ValidationIssue(
                category="DATE_FORMAT",
                severity="CRITICAL",
                message=f"Cannot parse date column: {str(e)}",
            )
        )
        return issues

    # Check for future dates (beyond today)
    today = pd.Timestamp.now().normalize()
    future_dates = (dates > today).sum()
    if future_dates > 0:
        issues.append(
            ValidationIssue(
                category="DATE_RANGE",
                severity="WARNING",
                message=f"Found {future_dates} dates in the future",
                affected_rows=future_dates,
            )
        )

    # Check chronological ordering
    if not dates.is_monotonic_increasing:
        issues.append(
            ValidationIssue(
                category="DATE_ORDER",
                severity="WARNING",
                message="Dates are not in chronological order (consider sorting)",
            )
        )

    return issues


def validate_demand(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate demand/units_sold column."""
    issues: list[ValidationIssue] = []
    if "units_sold" not in df.columns:
        return issues

    units = df["units_sold"].astype(float)

    # Check for negative values
    negative = (units < 0).sum()
    if negative > 0:
        issues.append(
            ValidationIssue(
                category="NEGATIVE_DEMAND",
                severity="CRITICAL",
                message=f"Found {negative} negative demand values",
                affected_rows=negative,
            )
        )

    # Check for outliers (unusually high)
    if len(units) > 10:
        q75 = units.quantile(0.75)
        q25 = units.quantile(0.25)
        iqr = q75 - q25
        upper_bound = q75 + 3 * iqr
        outliers = (units > upper_bound).sum()
        if outliers > 0 and outliers < len(units) * 0.1:
            issues.append(
                ValidationIssue(
                    category="DEMAND_OUTLIERS",
                    severity="INFO",
                    message=f"Found {outliers} potential outliers in demand",
                    affected_rows=outliers,
                )
            )

    return issues


def validate_prices(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate price column."""
    issues: list[ValidationIssue] = []
    if "price" not in df.columns:
        return issues

    price = df["price"].astype(float)

    # Check for negative prices
    negative = (price <= 0).sum()
    if negative > 0:
        issues.append(
            ValidationIssue(
                category="INVALID_PRICE",
                severity="CRITICAL",
                message=f"Found {negative} non-positive price values",
                affected_rows=negative,
            )
        )

    return issues


def validate_identifiers(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate product and store identifiers."""
    issues: list[ValidationIssue] = []

    if "product_id" in df.columns:
        unique_products = df["product_id"].nunique()
        if unique_products == 0:
            issues.append(
                ValidationIssue(
                    category="INVALID_IDENTIFIER",
                    severity="CRITICAL",
                    message="No unique products found",
                )
            )

    if "store_id" in df.columns:
        unique_stores = df["store_id"].nunique()
        if unique_stores == 0:
            issues.append(
                ValidationIssue(
                    category="INVALID_IDENTIFIER",
                    severity="CRITICAL",
                    message="No unique stores found",
                )
            )

    return issues


def validate_data_quality(df: pd.DataFrame) -> ValidationReport:
    """
    Comprehensive data validation.

    Returns:
        ValidationReport with all found issues and summary.
    """
    issues: list[ValidationIssue] = []

    # Run all validations
    issues.extend(validate_required_columns(df))
    issues.extend(validate_missing_values(df))
    issues.extend(validate_duplicates(df))
    issues.extend(validate_dates(df))
    issues.extend(validate_demand(df))
    issues.extend(validate_prices(df))
    issues.extend(validate_identifiers(df))

    # Calculate valid rows (no critical issues affecting rows)
    critical_row_issues = sum(i.affected_rows or 0 for i in issues if i.severity == "CRITICAL")
    valid_rows = max(0, len(df) - critical_row_issues)

    # Dataset is valid if no critical issues exist
    is_valid = not any(i.severity == "CRITICAL" for i in issues)

    return ValidationReport(
        dataset_rows=len(df),
        valid_rows=valid_rows,
        issues=issues,
        timestamp=datetime.now().isoformat(),
        is_valid=is_valid,
    )


def get_clean_dataset(df: pd.DataFrame, handle_duplicates: str = "first") -> pd.DataFrame:
    """
    Get cleaned dataset by removing/fixing common issues.

    Args:
        df: Input dataframe
        handle_duplicates: How to handle duplicates ('first', 'last', or 'drop')

    Returns:
        Cleaned dataframe
    """
    clean = df.copy()

    # Remove duplicates
    key_cols = ["date", "product_id", "store_id"]
    existing_cols = [c for c in key_cols if c in clean.columns]
    if existing_cols:
        clean = clean.drop_duplicates(subset=existing_cols, keep=handle_duplicates)

    # Remove negative demand
    if "units_sold" in clean.columns:
        clean = clean[clean["units_sold"] >= 0]

    # Remove zero/negative prices
    if "price" in clean.columns:
        clean = clean[clean["price"] > 0]

    # Remove rows with critical missing values
    critical_cols = ["date", "product_id", "store_id", "units_sold"]
    existing_critical = [c for c in critical_cols if c in clean.columns]
    clean = clean.dropna(subset=existing_critical)

    # Sort by date
    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"])
        clean = clean.sort_values("date").reset_index(drop=True)

    return clean
