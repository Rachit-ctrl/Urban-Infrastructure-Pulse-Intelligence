"""Build ZIP-level infrastructure reporting features from the cleaned NYC 311 master.

Run from the PROJECT ROOT (the folder containing config/ and src/):
    python -m src.features.aggregate_area_features

Reads:  config.config.MASTER_DATA_FILE
Writes: data/processed/area_features_by_zip.csv

Designed to preserve the existing feature column names used by downstream modules.
Only the columns needed for aggregation are loaded (rather than all 53 columns).
"""

import numpy as np
import pandas as pd

from config.config import MASTER_DATA_FILE, PROCESSED_DATA_DIR

INPUT_FILE = MASTER_DATA_FILE
OUTPUT_DIR = PROCESSED_DATA_DIR
OUTPUT_FILE = OUTPUT_DIR / "area_features_by_zip.csv"

REQUIRED_COLUMNS = [
    "unique_key", "borough", "incident_zip", "complaint_type",
    "latitude", "longitude", "resolution_hours", "is_closed_status",
]
OPTIONAL_COLUMNS = ["has_valid_coordinates"]
READ_COLUMNS = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


def load_data():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Check MASTER_DATA_FILE in config/config.py and run cleaning first."
        )

    # Avoid loading unnecessary text/location columns from the 53-column master.
    df = pd.read_csv(
        INPUT_FILE,
        usecols=lambda col: col in READ_COLUMNS,
        dtype={
            "unique_key": "string",
            "borough": "string",
            "incident_zip": "string",
            "complaint_type": "string",
        },
        low_memory=False,
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns missing from cleaned CSV: {missing}")

    print(f"Loaded {len(df):,} rows; {len(df.columns)} required/optional columns.")
    return df


def prepare_data(df):
    original_count = len(df)
    duplicate_ids = int(df["unique_key"].duplicated().sum())
    missing_ids = int(df["unique_key"].isna().sum())
    if missing_ids:
        raise ValueError(f"Found {missing_ids:,} rows without complaint IDs.")
    if duplicate_ids:
        print(f"WARNING: {duplicate_ids:,} duplicate IDs; keeping first occurrence.")
        df = df.drop_duplicates(subset="unique_key").copy()
    else:
        df = df.copy()

    # Same geographical key as the 100K pipeline: (borough, numeric ZIP).
    df["borough"] = df["borough"].str.strip().str.upper()
    df["complaint_type"] = df["complaint_type"].str.strip()
    zip_numeric = pd.to_numeric(df["incident_zip"].str.strip(), errors="coerce")
    # Exclude missing or non-integral ZIP values rather than quietly truncating them.
    valid_zip = zip_numeric.notna() & np.isfinite(zip_numeric) & (zip_numeric % 1 == 0)
    valid_borough = df["borough"].notna() & df["borough"].ne("") & df["borough"].ne("UNSPECIFIED")
    usable = (valid_zip & valid_borough).fillna(False)

    excluded_area = int((~usable).sum())
    df = df.loc[usable].copy()
    df["incident_zip"] = zip_numeric.loc[usable].astype("int64").astype(str)

    for col in ("resolution_hours", "latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["resolution_hours"] < 0, "resolution_hours"] = np.nan

    # A location is only a suitable area-coordinate summary when BOTH
    # coordinates exist and pass the cleaning stage's validity check.
    if "has_valid_coordinates" in df.columns:
        geo_valid = (
            df["has_valid_coordinates"].astype("string")
            .str.strip().str.lower().isin(["true", "1"])
        )
    else:
        geo_valid = df["latitude"].notna() & df["longitude"].notna()
        print("NOTE: has_valid_coordinates absent; using nonmissing coordinate pairs.")
    df.loc[~geo_valid, ["latitude", "longitude"]] = np.nan

    status = df["is_closed_status"].astype("string").str.strip().str.lower()
    unknown_status = int((~status.isin(["true", "false", "1", "0"]) & status.notna()).sum())
    if unknown_status:
        print(f"WARNING: {unknown_status:,} unrecognized is_closed_status values; treating as not closed.")
    df["_unresolved"] = ~status.isin(["true", "1"])
    df["_resolution_available"] = df["resolution_hours"].notna()

    print(f"Unique complaint IDs kept: {len(df) + excluded_area:,}")
    print(f"Excluded from ZIP-level aggregation: {excluded_area:,} rows (missing/invalid borough or ZIP)")
    print(f"Records represented by ZIP groups: {len(df):,} / {original_count:,}")
    if df.empty:
        raise ValueError("No usable records with borough and ZIP remain.")
    return df


def create_area_features(df):
    keys = ["borough", "incident_zip"]
    grouped = df.groupby(keys, observed=True, sort=True)

    area_features = grouped.agg(
        total_complaints=("unique_key", "count"),
        category_diversity=("complaint_type", "nunique"),
        median_resolution_hours=("resolution_hours", "median"),
        p90_resolution_hours=("resolution_hours", lambda x: x.quantile(0.90)),
        resolution_available_rate=("_resolution_available", "mean"),
        unresolved_rate=("_unresolved", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()

    category_counts = pd.crosstab(
        index=[df["borough"], df["incident_zip"]],
        columns=df["complaint_type"],
    )
    # Preserve the exact column naming convention from the 100K code.
    category_counts.columns = [
        f"count_{str(col).lower().replace(' ', '_').replace('/', '_')}"
        for col in category_counts.columns
    ]
    category_counts = category_counts.reset_index()

    area_features = area_features.merge(
        category_counts, on=keys, how="left", validate="one_to_one"
    )
    category_cols = [c for c in category_counts.columns if c not in keys]
    area_features[category_cols] = area_features[category_cols].fillna(0).astype("int64")
    area_features.insert(
        0, "area_id", area_features["borough"].astype(str)
        + "_" + area_features["incident_zip"].astype(str),
    )

    # Checks for accidental duplication or mismatched group merges.
    assert area_features["area_id"].is_unique, "Duplicate area_id values detected."
    assert int(area_features["total_complaints"].sum()) == len(df), "Area counts do not reconcile."
    missing_category = int(df["complaint_type"].isna().sum())
    assert int(area_features[category_cols].to_numpy().sum()) == len(df) - missing_category, (
        "Complaint category counts do not reconcile."
    )
    if missing_category:
        print(f"WARNING: {missing_category:,} records have missing complaint_type values.")
    return area_features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = load_data()
    prepared = prepare_data(source)
    areas = create_area_features(prepared)
    areas.to_csv(OUTPUT_FILE, index=False)
    print("\nZIP-LEVEL FEATURE AGGREGATION COMPLETE")
    print("Output:", OUTPUT_FILE)
    print("Number of ZIP/borough areas:", len(areas))
    print("Rows represented:", f"{int(areas['total_complaints'].sum()):,}")
    print("Feature columns:", len(areas.columns))
    print("Number of category-count features:", sum(c.startswith("count_") for c in areas.columns))
    print("Areas with missing median resolution:", int(areas["median_resolution_hours"].isna().sum()))
    print("Areas with missing mean coordinates:", int(areas["latitude"].isna().sum()))
    print("\nFirst five ZIP-level rows:")
    print(areas.head().to_string(index=False))


if __name__ == "__main__":
    main()
