"""Select and validate ZIP/borough features for PCA and K-Means.

Input:  data/processed/area_modeling_features_with_persistence.csv
Output: data/processed/selected_modeling_features.csv
Reports: outputs/tables/feature_selection/

Run from project root:
    python -m src.features.select_modeling_features
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PROCESSED_DATA_DIR, OUTPUT_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "area_modeling_features_with_persistence.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "selected_modeling_features.csv"
REPORT_DIR = OUTPUT_DIR / "tables" / "feature_selection"

# Areas with fewer reports still appear in the output; this is only a warning.
MIN_COMPLAINT_SUPPORT = 30

BASE_FEATURES = [
    "total_complaints",
    "active_months",
    "monthly_cv",
    "repeated_complaint_rate",
    "persistent_complaint_rate",
    "avg_recurrence_span_days",
]

EXCLUDED_FEATURES = {
    "area_id": "Geographic identifier",
    "borough": "Categorical geographic identifier",
    "incident_zip": "Geographic identifier",
    "latitude": "Reserved for spatial analysis",
    "longitude": "Reserved for spatial analysis",
    "mean_latitude": "Reserved for spatial analysis",
    "mean_longitude": "Reserved for spatial analysis",
    "category_share_sum": "Validation-only feature",
    "monthly_mean": "Redundant with total complaint volume",
    "total_monthly_complaints": "Redundant with total complaint volume",
    "monthly_std": "Initially excluded due to correlation",
    "peak_month_complaints": "Highly correlated with volume",
    "peak_month_share": "Highly correlated with monthly CV",
    "repeated_location_groups": "Highly correlated with repeated complaint count",
    "repeated_complaint_count": "Use normalized rate initially",
    "persistent_location_count": "Use normalized persistence rate initially",
    "persistent_complaint_count": "Use normalized persistence rate initially",
    "resolution_available_rate": "Resolution-status inconsistency",
    "unresolved_rate": "Resolution-status inconsistency",
    "median_resolution_hours": "Temporarily excluded pending validation",
    "p90_resolution_hours": "Temporarily excluded pending validation",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input missing: {INPUT_FILE}\n"
            "Run create_persistence_features.py before feature selection."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if df.empty:
        raise ValueError("Engineered input is empty.")
    if "area_id" not in df or df["area_id"].isna().any():
        raise ValueError("area_id missing or contains null values.")
    if df["area_id"].astype(str).str.strip().eq("").any():
        raise ValueError("area_id contains blank values.")
    if df["area_id"].duplicated().any():
        raise ValueError("Duplicate area IDs detected.")

    logger.info("Loaded %s areas and %s columns", *df.shape)
    return df


def select_features(df):
    shares = sorted(c for c in df.columns if c.startswith("share_"))
    counts = sorted(c for c in df.columns if c.startswith("count_"))

    if not shares or not counts:
        raise ValueError("Expected count_* and share_* category columns.")

    # Share and count columns must represent exactly the same categories.
    expected_shares = {"share_" + c[6:] for c in counts}
    if set(shares) != expected_shares:
        raise ValueError(
            "Mismatch between category count and share columns. "
            f"Missing share columns: {sorted(expected_shares - set(shares))}; "
            f"unexpected share columns: {sorted(set(shares) - expected_shares)}"
        )

    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Required base features missing: {missing}")

    logger.info("ZIP-level category count/share pairs: %s", len(shares))
    if len(shares) == 14:
        logger.info(
            "14 mapped categories are expected when Bridge Condition "
            "has no usable ZIP identifiers."
        )
    elif len(shares) != 15:
        logger.warning(
            "Found %s categories, rather than the anticipated 14 or 15. "
            "Confirm upstream aggregation coverage.",
            len(shares),
        )

    return BASE_FEATURES + shares, shares, counts


def validate_features(df, candidates, shares, counts):
    """Validate full compositions BEFORE removing any constant model features."""
    numeric_cols = candidates + counts
    numeric_df = df[numeric_cols].copy()

    nonnumeric = numeric_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnumeric:
        raise TypeError(f"Nonnumeric columns found: {nonnumeric}")

    missing = numeric_df.isna().sum()
    if (missing > 0).any():
        raise ValueError("Missing feature values:\n" + missing[missing > 0].to_string())
    if not np.isfinite(numeric_df.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite feature values detected.")

    if (df["total_complaints"] <= 0).any():
        raise ValueError("Area total_complaints must be positive.")
    if (df[counts] < 0).any().any():
        raise ValueError("Negative category counts detected.")
    if not np.allclose(
        df[counts].sum(axis=1).to_numpy(dtype=float),
        df["total_complaints"].to_numpy(dtype=float),
        rtol=0, atol=1e-8,
    ):
        raise ValueError("Category counts do not equal total_complaints.")

    if ((df[shares] < -1e-10) | (df[shares] > 1 + 1e-10)).any().any():
        raise ValueError("Complaint category shares must be between 0 and 1.")
    if not np.allclose(df[shares].sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Complaint category shares do not sum to 1.")

    for rate in ("repeated_complaint_rate", "persistent_complaint_rate"):
        if ((df[rate] < 0) | (df[rate] > 1)).any():
            raise ValueError(f"{rate} outside the expected 0-to-1 range.")
    for col in ("monthly_cv", "avg_recurrence_span_days", "active_months"):
        if (df[col] < 0).any():
            raise ValueError(f"Negative values detected in {col}.")

    # A constant column cannot distinguish one area from another.
    constant = [c for c in candidates if df[c].nunique(dropna=False) <= 1]
    selected = [c for c in candidates if c not in constant]
    if not selected:
        raise ValueError("No variable features remain for modeling.")

    if constant:
        logger.warning("Excluded zero-variance modeling columns: %s", constant)
    logger.info("All original category shares sum to 1; totals match category counts.")
    logger.info("Selected %s nonconstant numerical features", len(selected))
    return df[selected].copy(), selected, constant


def create_selection_report(df, selected_features, constant_features):
    selected = set(selected_features)
    constant = set(constant_features)
    rows = []
    for column in df.columns:
        if column == "area_id":
            status, reason = "METADATA", "Geographic identifier"
        elif column in constant:
            status, reason = "EXCLUDED", "Zero variance across geographical areas"
        elif column in selected:
            status, reason = "SELECTED", "Used for initial clustering"
        elif column in EXCLUDED_FEATURES:
            status, reason = "EXCLUDED", EXCLUDED_FEATURES[column]
        else:
            status, reason = "REVIEW", "Not selected; evaluate separately"
        rows.append({"feature": column, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def main():
    df = load_data()
    candidates, shares, counts = select_features(df)
    feature_df, selected, constant = validate_features(df, candidates, shares, counts)

    model_df = pd.concat([df[["area_id"]].reset_index(drop=True),
                          feature_df.reset_index(drop=True)], axis=1)
    metadata_cols = [c for c in (
        "area_id", "borough", "incident_zip", "latitude", "longitude"
    ) if c in df.columns]
    metadata = df[metadata_cols].copy()
    report = create_selection_report(df, selected, constant)

    low_support = int((df["total_complaints"] < MIN_COMPLAINT_SUPPORT).sum())
    if low_support:
        logger.warning(
            "%s of %s areas have fewer than %s complaints. "
            "Review before peer comparison; no areas were removed.",
            low_support, len(df), MIN_COMPLAINT_SUPPORT,
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(OUTPUT_FILE, index=False)
    metadata.to_csv(REPORT_DIR / "area_metadata.csv", index=False)
    report.to_csv(REPORT_DIR / "feature_selection_report.csv", index=False)

    print("\nFEATURE SELECTION COMPLETE")
    print("Areas:", len(model_df))
    print("ZIP-level complaint categories:", len(shares))
    print("Selected modeling features:", len(selected))
    print("Excluded constant features:", constant or "None")
    print(f"Low-support areas (<{MIN_COMPLAINT_SUPPORT} complaints):", low_support)
    print("Missing values:", int(feature_df.isna().sum().sum()))
    print("Duplicate area IDs:", int(model_df["area_id"].duplicated().sum()))
    print("Category count/share validation: PASSED")
    print("Modeling CSV:", OUTPUT_FILE)
    print("Reports:", REPORT_DIR)
    logger.info("Feature selection saved successfully.")


if __name__ == "__main__":
    main()
