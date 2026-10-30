import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MASTER_DATA_FILE, PROCESSED_DATA_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "area_composition_features.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "area_modeling_features.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def create_area_id(df):
    df = df.copy()

    df["borough"] = df["borough"].astype(str).str.strip().str.upper()

    df["incident_zip"] = (
        pd.to_numeric(df["incident_zip"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(5)
    )

    df["area_id"] = df["borough"] + "_" + df["incident_zip"]

    return df


def main():

    logger.info("Loading datasets...")

    master = pd.read_csv(MASTER_DATA_FILE, low_memory=False)
    area_features = pd.read_csv(INPUT_FILE)

    required_columns = [
        "created_date",
        "borough",
        "incident_zip"
    ]

    missing = [
        col for col in required_columns
        if col not in master.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prepare complaint records
    master["created_date"] = pd.to_datetime(
        master["created_date"],
        errors="coerce"
    )

    master = master.dropna(
        subset=["created_date", "borough", "incident_zip"]
    )

    master = create_area_id(master)

    # Keep only areas present in the composition dataset
    valid_areas = set(area_features["area_id"])

    master = master[master["area_id"].isin(valid_areas)].copy()

    if master.empty:
        raise ValueError("No matching geographic areas found.")

    # Create monthly periods
    master["month"] = master["created_date"].dt.to_period("M")

    # Count complaints by area and month
    monthly_counts = (
        master.groupby(["area_id", "month"])
        .size()
        .rename("monthly_complaints")
        .reset_index()
    )

    # Complete missing months with zero complaints
    all_months = pd.period_range(
        master["month"].min(),
        master["month"].max(),
        freq="M"
    )

    area_month_index = pd.MultiIndex.from_product(
        [area_features["area_id"].unique(), all_months],
        names=["area_id", "month"]
    )

    monthly_counts = (
        monthly_counts
        .set_index(["area_id", "month"])
        .reindex(area_month_index, fill_value=0)
        .reset_index()
    )

    # Calculate temporal features
    temporal_features = (
        monthly_counts.groupby("area_id")["monthly_complaints"]
        .agg(
            active_months=lambda x: (x > 0).sum(),
            monthly_mean="mean",
            monthly_std="std",
            peak_month_complaints="max",
            total_monthly_complaints="sum"
        )
        .reset_index()
    )

    temporal_features["monthly_cv"] = (
        temporal_features["monthly_std"]
        / temporal_features["monthly_mean"].replace(0, np.nan)
    )

    temporal_features["peak_month_share"] = (
        temporal_features["peak_month_complaints"]
        / temporal_features["total_monthly_complaints"].replace(0, np.nan)
    )

    temporal_features = temporal_features.fillna(0)

    # Merge with existing area-level features
    final_df = area_features.merge(
        temporal_features,
        on="area_id",
        how="left",
        validate="one_to_one"
    )

    if final_df["monthly_mean"].isna().any():
        raise ValueError("Missing temporal features detected.")

    final_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Temporal feature engineering completed.")
    logger.info("Final dataset shape: %s", final_df.shape)
    logger.info("Temporal features: 6")
    logger.info("Output saved to: %s", OUTPUT_FILE)

    print("\nTemporal feature preview:")
    print(
        final_df[
            [
                "area_id",
                "active_months",
                "monthly_mean",
                "monthly_std",
                "monthly_cv",
                "peak_month_share"
            ]
        ].head()
    )


if __name__ == "__main__":
    main()