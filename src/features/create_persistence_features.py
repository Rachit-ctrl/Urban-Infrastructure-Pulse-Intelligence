"""Build repeated-location and persistence features for NYC 311 ZIP/borough areas.

Run from the project's root directory:
    python -m src.features.create_persistence_features

Inputs:
    config.config.MASTER_DATA_FILE  (cleaned 2023-2025 complaints)
    data/processed/area_modeling_features.csv
Output (overwrites the previous 100K feature file in the working project):
    data/processed/area_modeling_features_with_persistence.csv

A location-category group is defined by (borough + ZIP, complaint type,
latitude rounded to 4 decimal places, longitude rounded to 4 decimal places).
Repeated: at least 2 complaints in such a group.
Persistent: at least 5 complaints spanning at least 90 days.

The rates use ALL ZIP-area complaints as denominators. Complaints lacking
valid locations cannot be observed as recurring at a specific point, so the
rates should be treated as observed lower bounds, not estimates of true
physical issue recurrence. Repeated reports need not be the same issue.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MASTER_DATA_FILE, PROCESSED_DATA_DIR  # noqa: E402

INPUT_FILE = PROCESSED_DATA_DIR / "area_modeling_features.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "area_modeling_features_with_persistence.csv"

COORDINATE_PRECISION = 4
MIN_REPEATED_COMPLAINTS = 2
MIN_PERSISTENT_COMPLAINTS = 5
MIN_PERSISTENCE_DAYS = 90

REQUIRED_COLUMNS = [
    "unique_key", "created_date", "complaint_type", "borough",
    "incident_zip", "latitude", "longitude",
]
OPTIONAL_COLUMNS = ["has_valid_coordinates"]

COUNT_COLUMNS = [
    "repeated_location_groups",
    "repeated_complaint_count",
    "persistent_location_count",
    "persistent_complaint_count",
]
RESULT_COLUMNS = COUNT_COLUMNS + [
    "avg_recurrence_span_days",
    "repeated_complaint_rate",
    "persistent_complaint_rate",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_master():
    """Only load columns used by recurrence calculations (not all 53)."""
    if not MASTER_DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Cleaned master not found: {MASTER_DATA_FILE}. Run cleaning first."
        )
    df = pd.read_csv(
        MASTER_DATA_FILE,
        usecols=lambda name: name in set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS),
        dtype={
            "unique_key": "string",
            "complaint_type": "string",
            "borough": "string",
            "incident_zip": "string",
            "has_valid_coordinates": "string",
        },
        low_memory=False,
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in master CSV: {missing}")
    logger.info("Loaded %s cleaned complaint rows (only %d columns)", f"{len(df):,}", len(df.columns))
    return df


def load_area_features():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Missing temporal area features: {INPUT_FILE}. "
            "Run the composition and temporal feature scripts first."
        )
    areas = pd.read_csv(INPUT_FILE, dtype={"area_id": "string"}, low_memory=False)
    if areas.empty or not {"area_id", "total_complaints"}.issubset(areas.columns):
        raise ValueError("Area modeling CSV is empty or lacks area_id/total_complaints.")
    if areas["area_id"].isna().any() or areas["area_id"].duplicated().any():
        raise ValueError("Area modeling input contains missing or duplicated area IDs.")
    totals = pd.to_numeric(areas["total_complaints"], errors="coerce")
    if totals.isna().any() or totals.le(0).any() or totals.ne(totals.round()).any():
        raise ValueError("All area total_complaints must be positive integers.")
    if any(col in areas.columns for col in RESULT_COLUMNS):
        raise ValueError("Input already has persistence columns; expected area_modeling_features.csv.")
    return areas


def prepare_data(df, valid_area_ids=None):
    """Normalize geography identically to ZIP aggregation; filter spatially usable rows."""
    required = set(REQUIRED_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["unique_key"].isna().any() or df["unique_key"].duplicated().any():
        raise ValueError("Missing or duplicate complaint IDs in cleaned master. Review cleaning.")

    borough = df["borough"].astype("string").str.strip().str.upper()
    zip_number = pd.to_numeric(df["incident_zip"], errors="coerce")
    valid_area = (
        borough.notna() & borough.ne("") & borough.ne("UNSPECIFIED")
        & zip_number.notna() & np.isfinite(zip_number)
        & zip_number.eq(zip_number.round())
    ).fillna(False)

    created = pd.to_datetime(df["created_date"], errors="coerce")
    complaint_type = df["complaint_type"].astype("string").str.strip()
    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    valid_coords = (
        latitude.notna() & longitude.notna()
        & latitude.between(-90, 90) & longitude.between(-180, 180)
    ).fillna(False)
    if "has_valid_coordinates" in df.columns:
        # Matches the cleaning-stage flag; only accept explicitly valid rows.
        valid_flag = (
            df["has_valid_coordinates"].astype("string")
            .str.strip().str.lower().isin(["true", "1"])
        )
        valid_coords &= valid_flag
    else:
        logger.warning("No has_valid_coordinates flag; using numeric coordinate ranges only")

    usable = (
        valid_area & created.notna() & complaint_type.notna()
        & complaint_type.ne("") & valid_coords
    ).fillna(False)
    result = pd.DataFrame({
        "area_id": (
            borough.loc[usable].astype(str) + "_"
            + zip_number.loc[usable].astype("int64").astype(str)
        ),
        "created_date": created.loc[usable],
        "complaint_type": complaint_type.loc[usable],
        "latitude": latitude.loc[usable],
        "longitude": longitude.loc[usable],
        "unique_key": df.loc[usable, "unique_key"],
    })
    if valid_area_ids is not None:
        initial = len(result)
        result = result[result["area_id"].isin(valid_area_ids)].copy()
        if len(result) < initial:
            logger.warning(
                "%s spatially eligible complaints did not match area_modeling_features IDs",
                f"{initial - len(result):,}",
            )
    logger.info(
        "Spatially eligible records: %s / %s (%.1f%%)",
        f"{len(result):,}", f"{len(df):,}",
        100 * len(result) / len(df) if len(df) else 0,
    )
    return result


def identify_repeated_locations(df):
    """One row per category and rounded coordinate pair within an area."""
    columns = [
        "area_id", "complaint_type", "latitude_rounded", "longitude_rounded",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns + [
            "complaint_count", "first_report", "last_report", "recurrence_span_days",
            "is_repeated", "is_persistent",
        ])
    prepared = df.copy()
    prepared["latitude_rounded"] = prepared["latitude"].round(COORDINATE_PRECISION)
    prepared["longitude_rounded"] = prepared["longitude"].round(COORDINATE_PRECISION)
    groups = (
        prepared.groupby(columns, observed=True, sort=False)
        .agg(
            complaint_count=("unique_key", "size"),
            first_report=("created_date", "min"),
            last_report=("created_date", "max"),
        )
        .reset_index()
    )
    if int(groups["complaint_count"].sum()) != len(prepared):
        raise ValueError("Location-group counts do not reconcile with eligible records.")
    groups["recurrence_span_days"] = (
        (groups["last_report"] - groups["first_report"])
        .dt.total_seconds() / 86400.0
    )
    groups["is_repeated"] = groups["complaint_count"].ge(MIN_REPEATED_COMPLAINTS)
    groups["is_persistent"] = (
        groups["complaint_count"].ge(MIN_PERSISTENT_COMPLAINTS)
        & groups["recurrence_span_days"].ge(MIN_PERSISTENCE_DAYS)
    )
    return groups


def create_area_features(location_summary):
    """Aggregate repeat/persistence diagnostics; returns an indexed feature table."""
    if location_summary.empty:
        return pd.DataFrame(columns=["area_id"] + COUNT_COLUMNS + ["avg_recurrence_span_days"])
    groups = location_summary.copy()
    groups["_repeated_complaints"] = groups["complaint_count"].where(groups["is_repeated"], 0)
    groups["_persistent_complaints"] = groups["complaint_count"].where(groups["is_persistent"], 0)
    groups["_persistent_span"] = groups["recurrence_span_days"].where(groups["is_persistent"])
    return (
        groups.groupby("area_id", observed=True, sort=False)
        .agg(
            repeated_location_groups=("is_repeated", "sum"),
            repeated_complaint_count=("_repeated_complaints", "sum"),
            persistent_location_count=("is_persistent", "sum"),
            persistent_complaint_count=("_persistent_complaints", "sum"),
            avg_recurrence_span_days=("_persistent_span", "mean"),
        )
        .reset_index()
    )


def compute_persistence_features(area_features, spatial_records):
    """Join all geographic area records and validate rates before saving."""
    location_summary = identify_repeated_locations(spatial_records)
    persistence = create_area_features(location_summary)
    final = area_features.merge(persistence, on="area_id", how="left", validate="one_to_one")
    if len(final) != len(area_features):
        raise ValueError("Area count changed during persistence merge.")
    final[COUNT_COLUMNS] = final[COUNT_COLUMNS].fillna(0).astype("int64")
    final["avg_recurrence_span_days"] = final["avg_recurrence_span_days"].fillna(0.0)

    total = pd.to_numeric(final["total_complaints"], errors="raise")
    final["repeated_complaint_rate"] = final["repeated_complaint_count"] / total
    final["persistent_complaint_rate"] = final["persistent_complaint_count"] / total
    if (
        (final[COUNT_COLUMNS] < 0).any().any()
        or final["repeated_complaint_count"].gt(total).any()
        or final["persistent_complaint_count"].gt(final["repeated_complaint_count"]).any()
        or final["persistent_location_count"].gt(final["repeated_location_groups"]).any()
        or final["repeated_complaint_rate"].gt(1).any()
        or final["persistent_complaint_rate"].gt(final["repeated_complaint_rate"]).any()
    ):
        raise ValueError("Persistence counts/rates inconsistent with area totals.")
    return final, location_summary


def main():
    logger.info("Loading full-data area features and relevant master columns...")
    areas = load_area_features()
    master = load_master()
    spatial_records = prepare_data(master, valid_area_ids=set(areas["area_id"].astype(str)))
    final, groups = compute_persistence_features(areas, spatial_records)

    # Helpful audit: an area can have low apparent recurrence if most records
    # lack coordinates; do not confuse 'zero observed' with 'no physical issue'.
    matched_by_area = spatial_records.groupby("area_id").size()
    indexed_totals = pd.Series(
        final["total_complaints"].astype(int).to_numpy(),
        index=final["area_id"].astype(str),
    )
    coverage = matched_by_area.reindex(indexed_totals.index, fill_value=0) / indexed_totals
    logger.info(
        "Spatial observation coverage across areas: median %.1f%%; min %.1f%%",
        100 * coverage.median(), 100 * coverage.min(),
    )
    logger.info("Areas with <50%% usable spatial records: %d", int(coverage.lt(0.5).sum()))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)

    print("\nPERSISTENCE FEATURE ENGINEERING COMPLETE")
    print("Master records loaded:", f"{len(master):,}")
    print("Records eligible for spatial recurrence:", f"{len(spatial_records):,}")
    print("ZIP/borough areas:", f"{len(final):,}")
    print("Location-category groups:", f"{len(groups):,}")
    print("Repeated groups:", f"{int(groups['is_repeated'].sum()):,}")
    print("Persistent groups:", f"{int(groups['is_persistent'].sum()):,}")
    print("Areas with persistent locations:", f"{int((final['persistent_location_count'] > 0).sum()):,}")
    print("Final shape:", final.shape)
    print("Saved:", OUTPUT_FILE)
    print("\nFirst five area summaries:")
    print(final[[
        "area_id", "repeated_location_groups", "persistent_location_count",
        "repeated_complaint_rate", "persistent_complaint_rate",
        "avg_recurrence_span_days",
    ]].head().to_string(index=False))


if __name__ == "__main__":
    main()
