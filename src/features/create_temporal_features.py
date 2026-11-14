"""Create monthly activity features for ZIP/borough areas in NYC 311.

Input:
    data/processed/nyc311_infrastructure_cleaned.csv
    data/processed/area_composition_features.csv
Output:
    data/processed/area_modeling_features.csv

Run from project root:
    python -m src.features.create_temporal_features
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    MASTER_DATA_FILE,
    PROCESSED_DATA_DIR,
    STUDY_START_DATE,
    STUDY_END_DATE,
)

INPUT_FILE = PROCESSED_DATA_DIR / "area_composition_features.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "area_modeling_features.csv"
CHUNK_SIZE = 100_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_area_ids(chunk):
    """Match area_id formatting in aggregate_area_features.py."""
    borough = chunk["borough"].astype("string").str.strip().str.upper()
    zip_number = pd.to_numeric(chunk["incident_zip"], errors="coerce")

    # Only exact whole-number ZIPs can match the original ZIP aggregation.
    usable = (
        borough.notna()
        & borough.ne("")
        & borough.ne("UNSPECIFIED")
        & zip_number.notna()
        & np.isfinite(zip_number)
        & zip_number.ge(0)
        & zip_number.le(99999)
        & zip_number.eq(zip_number.round())
    ).fillna(False)

    cleaned = chunk.loc[usable, ["created_date"]].copy()
    cleaned["area_id"] = (
        borough.loc[usable].astype(str)
        + "_"
        + zip_number.loc[usable].astype("int64").astype(str)
    )
    return cleaned


def load_monthly_counts(area_ids, start_date, end_exclusive):
    """Count monthly complaints in chunks without reading all 53 columns."""
    if not MASTER_DATA_FILE.exists():
        raise FileNotFoundError(f"Cleaned master file missing: {MASTER_DATA_FILE}")

    required = ["created_date", "borough", "incident_zip"]
    master_columns = set(pd.read_csv(MASTER_DATA_FILE, nrows=0).columns)
    missing = set(required) - master_columns
    if missing:
        raise ValueError(f"Cleaned master missing columns: {sorted(missing)}")

    counts = None
    total_read = 0
    kept = 0
    invalid_dates = 0
    out_of_range = 0
    excluded_areas = 0

    reader = pd.read_csv(
        MASTER_DATA_FILE,
        usecols=required,
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in reader:
        total_read += len(chunk)
        valid = prepare_area_ids(chunk)
        excluded_areas += len(chunk) - len(valid)
        dates = pd.to_datetime(valid["created_date"], errors="coerce")
        invalid_dates += int(dates.isna().sum())
        in_range = dates.ge(start_date) & dates.lt(end_exclusive)
        out_of_range += int((dates.notna() & ~in_range).sum())
        valid = valid.loc[in_range].copy()
        valid["month"] = dates.loc[in_range].dt.to_period("M")
        valid = valid[valid["area_id"].isin(area_ids)]
        kept += len(valid)

        monthly_chunk = valid.groupby(["area_id", "month"]).size()
        counts = monthly_chunk if counts is None else counts.add(monthly_chunk, fill_value=0)
        logger.info("Scanned %s rows; %s matched valid areas/dates", f"{total_read:,}", f"{kept:,}")

    logger.info(
        "Total read: %s | matched: %s | unusable geography: %s | "
        "invalid dates in geographically usable rows: %s | out of study period: %s",
        f"{total_read:,}", f"{kept:,}", f"{excluded_areas:,}",
        f"{invalid_dates:,}", f"{out_of_range:,}",
    )
    if counts is None:
        counts = pd.Series(
            dtype="int64",
            index=pd.MultiIndex.from_arrays([[], []], names=["area_id", "month"]),
        )
    return counts.astype("int64")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Area composition file not found: {INPUT_FILE}. "
            "Run create_composition_features first."
        )

    area_features = pd.read_csv(INPUT_FILE, low_memory=False)
    needed = {"area_id", "total_complaints"}
    if not needed.issubset(area_features.columns):
        raise ValueError(f"Area composition missing: {sorted(needed - set(area_features.columns))}")
    if area_features.empty or area_features["area_id"].isna().any():
        raise ValueError("Area composition is empty or has missing area IDs.")
    if area_features["area_id"].duplicated().any():
        raise ValueError("Duplicate area_id in composition features.")

    expected = pd.to_numeric(area_features["total_complaints"], errors="coerce")
    if expected.isna().any() or expected.lt(0).any() or expected.ne(expected.round()).any():
        raise ValueError("total_complaints must be nonnegative whole numbers.")

    start = pd.Timestamp(STUDY_START_DATE)
    end = pd.Timestamp(STUDY_END_DATE)
    if end < start:
        raise ValueError("STUDY_END_DATE is before STUDY_START_DATE.")
    end_exclusive = end + pd.Timedelta(days=1)
    all_months = pd.period_range(start=start, end=end, freq="M")
    logger.info("Using %d months (%s through %s)", len(all_months), all_months[0], all_months[-1])

    area_ids = area_features["area_id"].astype(str).tolist()
    monthly_counts = load_monthly_counts(set(area_ids), start, end_exclusive)

    complete_index = pd.MultiIndex.from_product(
        [area_ids, all_months], names=["area_id", "month"]
    )
    complete = monthly_counts.reindex(complete_index, fill_value=0).astype("int64")
    monthly_matrix = complete.unstack("month")  # One row per geographic area

    temporal = pd.DataFrame(index=monthly_matrix.index)
    temporal["active_months"] = monthly_matrix.gt(0).sum(axis=1)
    temporal["monthly_mean"] = monthly_matrix.mean(axis=1)
    temporal["monthly_std"] = monthly_matrix.std(axis=1).fillna(0)
    temporal["peak_month_complaints"] = monthly_matrix.max(axis=1)
    temporal["total_monthly_complaints"] = monthly_matrix.sum(axis=1)
    temporal["monthly_cv"] = (
        temporal["monthly_std"] / temporal["monthly_mean"].replace(0, np.nan)
    ).fillna(0)
    temporal["peak_month_share"] = (
        temporal["peak_month_complaints"]
        / temporal["total_monthly_complaints"].replace(0, np.nan)
    ).fillna(0)

    # Important: temporal totals must equal original ZIP-level totals.
    expected_by_area = pd.Series(
        expected.astype("int64").to_numpy(), index=area_ids
    )
    differences = temporal["total_monthly_complaints"] - expected_by_area
    inconsistent = differences[differences.ne(0)]
    if not inconsistent.empty:
        examples = pd.DataFrame({
            "area_total": expected_by_area.loc[inconsistent.index],
            "monthly_total": temporal.loc[inconsistent.index, "total_monthly_complaints"],
            "difference": inconsistent,
        }).head(10)
        raise ValueError(
            "Monthly complaint totals do not match aggregation totals for "
            f"{len(inconsistent)} areas. Check ZIP normalization/date ranges.\n{examples}"
        )

    final_df = area_features.merge(
        temporal.reset_index(), on="area_id", how="left", validate="one_to_one"
    )
    if len(final_df) != len(area_features) or final_df[temporal.columns].isna().any().any():
        raise ValueError("Missing temporal values or number of areas changed during merge.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Temporal feature engineering complete.")
    logger.info("Areas: %s | features: %s", len(final_df), len(final_df.columns))
    logger.info("Monthly totals verified for all areas. Output: %s", OUTPUT_FILE)
    print("\nTemporal feature preview:")
    print(final_df[[
        "area_id", "active_months", "monthly_mean", "monthly_std",
        "monthly_cv", "peak_month_share", "total_monthly_complaints"
    ]].head().to_string(index=False))


if __name__ == "__main__":
    main()
