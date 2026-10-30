import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MASTER_DATA_FILE, PROCESSED_DATA_DIR


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = PROCESSED_DATA_DIR / "area_modeling_features.csv"

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "area_modeling_features_with_persistence.csv"
)

COORDINATE_PRECISION = 4

MIN_REPEATED_COMPLAINTS = 2

MIN_PERSISTENT_COMPLAINTS = 5

MIN_PERSISTENCE_DAYS = 90


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data(df):

    df = df.copy()

    required_columns = [
        "unique_key",
        "created_date",
        "complaint_type",
        "borough",
        "incident_zip",
        "latitude",
        "longitude"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.drop_duplicates(subset=["unique_key"])

    df["created_date"] = pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["incident_zip"] = (
        pd.to_numeric(df["incident_zip"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(5)
    )

    df["borough"] = (
        df["borough"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["complaint_type"] = (
        df["complaint_type"]
        .astype("string")
        .str.strip()
    )

    # Remove records without required information
    df = df.dropna(
        subset=[
            "created_date",
            "latitude",
            "longitude",
            "complaint_type"
        ]
    )

    df = df[
        (df["latitude"].between(-90, 90))
        & (df["longitude"].between(-180, 180))
        & (df["borough"] != "UNSPECIFIED")
        & (df["incident_zip"] != "<NA>")
        & (df["complaint_type"] != "")
    ].copy()

    return df


# --------------------------------------------------
# IDENTIFY REPEATED LOCATIONS
# --------------------------------------------------

def identify_repeated_locations(df):

    logger.info("Identifying repeated complaint locations...")

    df["latitude_rounded"] = df["latitude"].round(
        COORDINATE_PRECISION
    )

    df["longitude_rounded"] = df["longitude"].round(
        COORDINATE_PRECISION
    )

    location_columns = [
        "borough",
        "incident_zip",
        "complaint_type",
        "latitude_rounded",
        "longitude_rounded"
    ]

    location_summary = (
        df.groupby(location_columns)
        .agg(
            complaint_count=("unique_key", "nunique"),
            first_report=("created_date", "min"),
            last_report=("created_date", "max")
        )
        .reset_index()
    )

    location_summary["recurrence_span_days"] = (
        location_summary["last_report"]
        - location_summary["first_report"]
    ).dt.total_seconds() / 86400

    # Repeated locations: at least 2 reports
    location_summary["is_repeated"] = (
        location_summary["complaint_count"]
        >= MIN_REPEATED_COMPLAINTS
    )

    # Persistent locations: at least 5 reports
    # spanning at least 90 days
    location_summary["is_persistent"] = (
        (location_summary["complaint_count"]
         >= MIN_PERSISTENT_COMPLAINTS)
        &
        (location_summary["recurrence_span_days"]
         >= MIN_PERSISTENCE_DAYS)
    )

    return location_summary


# --------------------------------------------------
# AREA-LEVEL PERSISTENCE FEATURES
# --------------------------------------------------

def create_area_features(location_summary):

    logger.info("Creating area-level persistence features...")

    repeated = location_summary[
        location_summary["is_repeated"]
    ].copy()

    persistent = location_summary[
        location_summary["is_persistent"]
    ].copy()

    repeated_features = (
        repeated.groupby(["borough", "incident_zip"])
        .agg(
            repeated_location_groups=("complaint_count", "size"),
            repeated_complaint_count=("complaint_count", "sum")
        )
        .reset_index()
    )

    persistent_features = (
        persistent.groupby(["borough", "incident_zip"])
        .agg(
            persistent_location_count=("complaint_count", "size"),
            persistent_complaint_count=("complaint_count", "sum"),
            avg_recurrence_span_days=(
                "recurrence_span_days", "mean"
            )
        )
        .reset_index()
    )

    area_features = repeated_features.merge(
        persistent_features,
        on=["borough", "incident_zip"],
        how="outer"
    )

    return area_features


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    logger.info("Loading master dataset...")

    master = pd.read_csv(
        MASTER_DATA_FILE,
        low_memory=False
    )

    area_features = pd.read_csv(INPUT_FILE)

    master = prepare_data(master)

    logger.info(
        "Valid complaint records with coordinates: %s",
        len(master)
    )

    # Identify recurring locations
    location_summary = identify_repeated_locations(master)

    # Create area-level features
    persistence_features = create_area_features(
        location_summary
    )

    # Generate area ID
    persistence_features["area_id"] = (
        persistence_features["borough"]
        + "_"
        + persistence_features["incident_zip"]
    )

    # Merge with existing modeling features
    final_df = area_features.merge(
        persistence_features.drop(
            columns=["borough", "incident_zip"]
        ),
        on="area_id",
        how="left",
        validate="one_to_one"
    )

    # Areas without repeated complaints receive zero
    count_columns = [
        "repeated_location_groups",
        "repeated_complaint_count",
        "persistent_location_count",
        "persistent_complaint_count"
    ]

    final_df[count_columns] = (
        final_df[count_columns].fillna(0)
    )

    final_df["avg_recurrence_span_days"] = (
        final_df["avg_recurrence_span_days"].fillna(0)
    )

    # Calculate rates
    final_df["repeated_complaint_rate"] = (
        final_df["repeated_complaint_count"]
        / final_df["total_complaints"].replace(0, np.nan)
    )

    final_df["persistent_complaint_rate"] = (
        final_df["persistent_complaint_count"]
        / final_df["total_complaints"].replace(0, np.nan)
    )

    final_df[
        ["repeated_complaint_rate", "persistent_complaint_rate"]
    ] = final_df[
        ["repeated_complaint_rate", "persistent_complaint_rate"]
    ].fillna(0)

    # Validation
    if final_df["area_id"].duplicated().any():
        raise ValueError("Duplicate area IDs detected.")

    if final_df[count_columns].isna().any().any():
        raise ValueError("Missing persistence counts detected.")

    if (
        final_df["repeated_complaint_rate"] > 1
    ).any():
        raise ValueError("Repeated complaint rate exceeds 1.")

    if (
        final_df["persistent_complaint_rate"] > 1
    ).any():
        raise ValueError("Persistent complaint rate exceeds 1.")

    # Save results
    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Summary
    print("\nPERSISTENCE ANALYSIS SUMMARY")
    print("-" * 40)

    print("Unique location-category groups:", len(location_summary))

    print(
        "Repeated location groups:",
        int(location_summary["is_repeated"].sum())
    )

    print(
        "Persistent location groups:",
        int(location_summary["is_persistent"].sum())
    )

    print(
        "Areas with persistent locations:",
        int((final_df["persistent_location_count"] > 0).sum())
    )

    print("\nFinal dataset shape:", final_df.shape)

    print("\nFeature preview:")

    print(
        final_df[
            [
                "area_id",
                "repeated_location_groups",
                "persistent_location_count",
                "repeated_complaint_rate",
                "persistent_complaint_rate",
                "avg_recurrence_span_days"
            ]
        ].head()
    )

    logger.info("Persistence features saved successfully.")
    logger.info("Output: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()