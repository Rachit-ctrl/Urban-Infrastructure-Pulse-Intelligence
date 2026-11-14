"""Clean full NYC 311 infrastructure data (2023-2025) in bounded memory.

Put this file at <project_root>/src/preprocessing/clean_full_nyc311.py.
Run from the project root: python -m src.preprocessing.clean_full_nyc311

Input : data/raw/nyc311_all_15_complaints_2023_2025.csv
Output: data/processed/nyc311_infrastructure_cleaned.csv
        data/processed/nyc311_infrastructure_geographic.csv
        data/reference/nyc311_data_quality_report.csv

Keeps *all* source records. Invalid dates/coordinates are flagged, not deleted
from the master. Spatial subset includes only valid broad-NYC coordinate pairs.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COMPLAINT_TYPES = {
    "Street Condition", "Traffic Signal Condition", "Street Light Condition",
    "Sewer", "Sidewalk Condition", "Root/Sewer/Sidewalk Condition",
    "Curb Condition", "Street Sign - Damaged", "Street Sign - Dangling",
    "Street Sign - Missing", "Water Leak", "Sewer Maintenance",
    "Bridge Condition", "Water Maintenance", "Water Drainage",
}

# Broad, conservative geographic sanity bounds (NOT an NYC boundary polygon).
# They exclude clearly implausible coordinates, not every point outside NYC.
MIN_LAT, MAX_LAT = 40.40, 41.00
MIN_LON, MAX_LON = -74.30, -73.60
START_DATE = pd.Timestamp("2023-01-01")
END_EXCLUSIVE = pd.Timestamp("2026-01-01")
CHUNK_SIZE = 50_000

DERIVED_COLS = [
    "created_year", "created_month", "created_day_of_week",
    "resolution_hours", "has_valid_coordinates", "coordinate_quality_flag",
    "date_quality_flag", "has_closed_date", "is_closed_status",
]
REQUIRED_RAW_COLS = {
    "unique_key", "created_date", "closed_date", "complaint_type", "borough",
    "status", "latitude", "longitude",
}


def parse_date_column(series: pd.Series) -> pd.Series:
    # NYC 311 exports may mix timestamps with and without fractional seconds.
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        # Fallback for older pandas: parse distinct formats independently.
        result = pd.to_datetime(series, errors="coerce")
        if result.isna().any():
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                missing = result.isna() & series.notna()
                if not missing.any():
                    break
                result.loc[missing] = pd.to_datetime(series.loc[missing], format=fmt, errors="coerce")
        return result


def nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def clean_chunk(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return cleaned master chunk, valid-geo mask and independent metrics."""
    df = raw.copy()

    created = parse_date_column(df["created_date"])
    closed = parse_date_column(df["closed_date"])
    raw_closed_nonblank = nonblank(df["closed_date"])

    invalid_created = created.isna()
    outside_dates = created.notna() & ((created < START_DATE) | (created >= END_EXCLUSIVE))
    closed_parse_fail = raw_closed_nonblank & closed.isna()
    missing_closed = closed.isna()  # includes blank or unparseable dates
    backwards = created.notna() & closed.notna() & (closed < created)
    valid_duration = created.notna() & closed.notna() & ~backwards

    df["created_year"] = created.dt.year.astype("Int64")
    df["created_month"] = created.dt.month.astype("Int64")
    df["created_day_of_week"] = created.dt.day_name()

    hours = pd.Series(np.nan, index=df.index, dtype="float64")
    hours.loc[valid_duration] = (
        (closed.loc[valid_duration] - created.loc[valid_duration])
        .dt.total_seconds() / 3600.0
    ).round(3)
    df["resolution_hours"] = hours

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    missing_coords = lat.isna() | lon.isna() | ~np.isfinite(lat) | ~np.isfinite(lon)
    in_bounds = lat.between(MIN_LAT, MAX_LAT) & lon.between(MIN_LON, MAX_LON)
    geo_valid = (~missing_coords & in_bounds).fillna(False)
    outside_coords = ~missing_coords & ~in_bounds
    df["latitude"] = lat
    df["longitude"] = lon
    df["has_valid_coordinates"] = geo_valid.astype(bool)
    df["coordinate_quality_flag"] = np.select(
        [missing_coords, outside_coords],
        ["missing_or_non_numeric_coordinate", "outside_broad_nyc_bounds"],
        default="valid_coordinate_range",
    )

    df["date_quality_flag"] = np.select(
        [invalid_created, backwards, missing_closed],
        ["missing_or_invalid_created_date", "closed_before_created", "open_or_missing_closed_date"],
        default="valid_or_not_evaluable",
    )
    df["has_closed_date"] = closed.notna().astype(bool)
    df["is_closed_status"] = (
        df["status"].astype("string").str.strip().str.casefold().eq("closed")
        .fillna(False).astype(bool)
    )

    # Preserve source column order, append the nine 100K-compatible derived fields.
    df = df[list(raw.columns) + DERIVED_COLS]
    metrics = {
        "created_date_parse_failures": int(invalid_created.sum()),
        "created_date_outside_study_period": int(outside_dates.sum()),
        "nonblank_closed_date_parse_failures": int(closed_parse_fail.sum()),
        "closed_before_created": int(backwards.sum()),
        "missing_or_unparseable_closed_date": int(missing_closed.sum()),
        "missing_latitude": int(lat.isna().sum()),
        "missing_longitude": int(lon.isna().sum()),
        "missing_either_coordinate": int(missing_coords.sum()),
        "outside_broad_nyc_bounds": int(outside_coords.sum()),
        "valid_coordinate_range_records": int(geo_valid.sum()),
        "valid_resolution_hours": int(valid_duration.sum()),
        "closed_status_records": int(df["is_closed_status"].sum()),
    }
    return df, geo_valid, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Resolve project root from package location, independent of VS Code CWD.
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input", type=Path,
                        default=project_root / "data/raw/nyc311_all_15_complaints_2023_2025.csv")
    parser.add_argument("--output", type=Path,
                        default=project_root / "data/processed/nyc311_infrastructure_cleaned.csv")
    parser.add_argument("--geo-output", type=Path,
                        default=project_root / "data/processed/nyc311_infrastructure_geographic.csv")
    parser.add_argument("--report", type=Path,
                        default=project_root / "data/reference/nyc311_data_quality_report.csv")
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    geo = args.geo_output.resolve()
    report = args.report.resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {inp}")
    if len({inp, out, geo, report}) < 4:
        raise ValueError("Input, cleaned output, geographic output and report must be different files.")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be a positive integer")

    headers = pd.read_csv(inp, nrows=0).columns.tolist()
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate raw column names; inspect the source header.")
    missing = REQUIRED_RAW_COLS - set(headers)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")
    already_cleaned = set(DERIVED_COLS) & set(headers)
    if already_cleaned:
        raise ValueError(f"Source appears to be already cleaned: {sorted(already_cleaned)}")

    for path in (out, geo, report):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = out.with_name(out.name + ".part")
    geo_tmp = geo.with_name(geo.name + ".part")
    report_tmp = report.with_name(report.name + ".part")

    totals = Counter()
    categories = Counter()
    date_flags = Counter()
    coordinate_flags = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    missing_ids = 0
    total_rows = 0
    geo_rows = 0
    chunks = 0

    try:
        with out_tmp.open("w", encoding="utf-8", newline="") as master_fp, \
             geo_tmp.open("w", encoding="utf-8", newline="") as geo_fp:
            for chunk in pd.read_csv(inp, dtype="string", keep_default_na=True,
                                     chunksize=args.chunksize, low_memory=False):
                chunks += 1
                total_rows += len(chunk)
                ids = chunk["unique_key"].str.strip()
                missing_ids += int((~nonblank(ids)).sum())
                for key in ids.dropna():
                    if key:
                        if key in seen_ids:
                            duplicate_ids += 1
                        else:
                            seen_ids.add(key)

                categories.update(chunk["complaint_type"].fillna("<MISSING>").tolist())
                cleaned, geo_mask, metrics = clean_chunk(chunk)
                totals.update(metrics)
                date_flags.update(cleaned["date_quality_flag"].value_counts().to_dict())
                coordinate_flags.update(cleaned["coordinate_quality_flag"].value_counts().to_dict())

                cleaned.to_csv(master_fp, index=False, header=(chunks == 1),
                               quoting=csv.QUOTE_MINIMAL)
                cleaned.loc[geo_mask].to_csv(geo_fp, index=False,
                                             header=(chunks == 1),
                                             quoting=csv.QUOTE_MINIMAL)
                geo_rows += int(geo_mask.sum())
                print(f"Processed {total_rows:,} rows | geo-valid {geo_rows:,}", flush=True)

        unexpected = sorted(set(categories) - COMPLAINT_TYPES)
        outside_study = totals["created_date_outside_study_period"]
        if unexpected or outside_study:
            raise ValueError(
                f"Input failed 2023-2025/15-type selection validation. "
                f"Unexpected categories: {unexpected}; valid dates outside period: {outside_study:,}. "
                "Existing output files have not been replaced."
            )
        if totals["valid_coordinate_range_records"] != geo_rows:
            raise AssertionError("Geographic subset row count mismatch")

        rows = []
        def add(section: str, metric: str, value: int, notes: str = ""):
            rows.append({"section": section, "metric": metric, "value": value, "notes": notes})
        add("overall", "source_records", total_rows)
        add("overall", "columns_original", len(headers))
        add("overall", "unique_complaint_ids", len(seen_ids))
        add("overall", "duplicate_complaint_id_occurrences", duplicate_ids)
        add("overall", "missing_complaint_ids", missing_ids)
        for metric in ("created_date_parse_failures", "created_date_outside_study_period",
                       "nonblank_closed_date_parse_failures", "closed_before_created",
                       "missing_or_unparseable_closed_date"):
            add("dates", metric, totals[metric])
        for metric in ("missing_latitude", "missing_longitude", "missing_either_coordinate",
                       "outside_broad_nyc_bounds", "valid_coordinate_range_records"):
            add("coordinates", metric, totals[metric])
        add("outputs", "cleaned_master_records", total_rows, "All input records retained")
        add("outputs", "geographic_subset_records", geo_rows,
            "Valid numerical coordinates within conservative bounding box")
        add("outputs", "valid_resolution_hours", totals["valid_resolution_hours"])
        for category in sorted(COMPLAINT_TYPES):
            add("category", category, categories[category], "record_count")
        for flag, count in sorted(date_flags.items()):
            add("date_quality_flag", flag, count)
        for flag, count in sorted(coordinate_flags.items()):
            add("coordinate_quality_flag", flag, count)
        pd.DataFrame(rows, columns=["section", "metric", "value", "notes"]).to_csv(
            report_tmp, index=False
        )

        # Publish finished outputs only after successful validation.
        os.replace(out_tmp, out)
        os.replace(geo_tmp, geo)
        os.replace(report_tmp, report)
    finally:
        for path in (out_tmp, geo_tmp, report_tmp):
            if path.exists():
                path.unlink()

    print("\nCLEANING COMPLETE")
    print(f"Raw records:          {total_rows:,}")
    print(f"Cleaned master rows:  {total_rows:,} (none dropped)")
    print(f"Geographic rows:      {geo_rows:,}")
    print(f"Duplicate IDs:        {duplicate_ids:,}")
    print(f"Invalid created dates:{totals['created_date_parse_failures']:>10,}")
    print(f"Backwards dates:      {totals['closed_before_created']:,}")
    print(f"Missing coordinates:  {totals['missing_either_coordinate']:,}")
    print(f"Cleaned master: {out}")
    print(f"Geographic CSV: {geo}")
    print(f"Quality report: {report}")


if __name__ == "__main__":
    main()
