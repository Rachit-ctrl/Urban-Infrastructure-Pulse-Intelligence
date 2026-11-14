"""Descriptive comparison of an uploaded reporting day with historical matched weekdays.

Read-only computation. This module does not alter PCA, K-Means, DBSCAN,
anomaly detection, stored models, or dashboard rankings.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


WINDOW_SUPPORT = {
    "same_quarter_weekdays": {"min_active_days": 10, "min_reports": 30},
    "all_matching_weekdays": {"min_active_days": 20, "min_reports": 50},
}


def compare_historical_comparable_weekdays(
    backend_records: pd.DataFrame,
    historical_daily_counts: pd.DataFrame,
    manifest: Mapping,
    reporting_day: str,
    *,
    expected_analyzed: int | None = None,
    min_current_reports: int = 5,
) -> dict:
    """Return two pooled historical weekday category-mix comparisons per current area.

    `same_quarter_weekdays`: matching weekday and calendar quarter from 2023–2025.
    `all_matching_weekdays`: matching weekday across the full historical period.

    The historical mix is pooled across matching days, not the share of a single
    historical day. A supported result is descriptive, not a statistical anomaly.
    """
    required_backend = {
        "created_date", "processing_status", "zip_validation_status",
        "borough", "incident_zip", "complaint_category",
    }
    missing = required_backend - set(backend_records.columns)
    if missing:
        raise ValueError(f"Missing required backend columns: {sorted(missing)}")
    required_manifest = {
        "schema", "historical_start", "historical_end", "supported_categories",
        "observed_eligible_categories", "eligible_rows", "sparse_daily_entries",
    }
    missing_manifest = required_manifest - set(manifest)
    if missing_manifest:
        raise ValueError(f"Missing manifest fields: {sorted(missing_manifest)}")

    target = pd.Timestamp(reporting_day)
    if pd.isna(target) or target.time() != pd.Timestamp(target.date()).time():
        raise ValueError("reporting_day must be a calendar date")
    start = pd.Timestamp(manifest["historical_start"])
    end = pd.Timestamp(manifest["historical_end"])
    if target <= end:
        raise ValueError("Reporting day must be after historical baseline to avoid leakage")

    categories = sorted(set(str(c).strip().upper() for c in manifest["supported_categories"]))
    observed = set(str(c).strip().upper() for c in manifest["observed_eligible_categories"])
    if not observed <= set(categories):
        raise ValueError("Manifest observed categories not included in supported taxonomy")

    history = historical_daily_counts.copy(deep=True)
    if list(history.columns) != list(manifest["schema"]):
        raise ValueError("Historical snapshot columns do not match manifest schema")
    if len(history) != int(manifest["sparse_daily_entries"]):
        raise ValueError("Historical snapshot row count does not match manifest")
    if history.duplicated(["area_id", "complaint_category", "historical_day"]).any():
        raise ValueError("Duplicate area/category/day keys in historical snapshot")
    history["historical_day"] = pd.to_datetime(
        history["historical_day"], format="%Y-%m-%d", errors="raise"
    )
    history["complaint_count"] = pd.to_numeric(
        history["complaint_count"], errors="raise"
    )
    if not history["historical_day"].between(start, end).all():
        raise ValueError("Historical snapshot contains dates outside manifest range")
    if not history["complaint_count"].gt(0).all():
        raise ValueError("Historical sparse snapshot has nonpositive counts")
    if int(history["complaint_count"].sum()) != int(manifest["eligible_rows"]):
        raise ValueError("Historical snapshot complaint counts do not reconcile")
    if not set(history["complaint_category"]).issubset(observed):
        raise ValueError("Historical snapshot contains categories outside observed taxonomy")

    records = backend_records.copy(deep=True)
    eligible = records.loc[
        records["created_date"].astype("string").str.slice(0, 10).eq(target.strftime("%Y-%m-%d"))
        & records["processing_status"].astype("string").str.strip().eq("processed")
        & records["zip_validation_status"].astype("string").str.strip().eq("zcta_consistent")
    ].copy()
    if expected_analyzed is not None and len(eligible) != int(expected_analyzed):
        raise ValueError(
            f"Verified-day comparison selected {len(eligible)} reports "
            f"but pipeline selected {expected_analyzed}. Check filters before proceeding."
        )
    if not len(eligible):
        return {
            "status": "no_verified_records", "eligible_reports": 0,
            "summary": pd.DataFrame(), "details": pd.DataFrame(),
            "unmatched_areas": [],
        }

    zips = eligible["incident_zip"].astype("string").str.strip()
    if not zips.str.fullmatch(r"\d{5}").fillna(False).all():
        raise ValueError("Verified-day input contains invalid ZIP code")
    eligible["area_id"] = (
        eligible["borough"].astype("string").str.strip().str.upper() + "_" + zips
    )
    eligible["complaint_category"] = (
        eligible["complaint_category"].astype("string").str.strip().str.upper()
    )
    if eligible["area_id"].isna().any() or eligible["complaint_category"].isna().any():
        raise ValueError("Verified-day input contains a missing area or category")
    current_categories = set(eligible["complaint_category"].unique())
    if not current_categories <= set(categories):
        raise ValueError("Uploaded verified category not in historical supported taxonomy")

    calendar = pd.date_range(start, end, freq="D")
    all_weekdays = calendar[calendar.dayofweek == target.dayofweek]
    same_quarter = all_weekdays[all_weekdays.quarter == target.quarter]
    historical_weekdays = history.loc[
        history["historical_day"].dt.dayofweek.eq(target.dayofweek)
    ]
    selected_windows = {
        "same_quarter_weekdays": (
            historical_weekdays.loc[
                historical_weekdays["historical_day"].dt.quarter.eq(target.quarter)
            ], len(same_quarter)
        ),
        "all_matching_weekdays": (historical_weekdays, len(all_weekdays)),
    }

    known_areas = set(history["area_id"])
    summary_rows = []
    detail_rows = []
    for area, group in eligible.groupby("area_id", sort=True):
        current_counts = group["complaint_category"].value_counts()
        current_total = int(len(group))
        current_top = str(current_counts.idxmax())
        uncovered = sorted(set(current_counts.index) - observed)

        for window_name, (window_data, calendar_days) in selected_windows.items():
            support = WINDOW_SUPPORT[window_name]
            area_history = window_data.loc[window_data["area_id"].eq(area)]
            historic_counts = area_history.groupby("complaint_category")["complaint_count"].sum()
            historic_total = int(historic_counts.sum())
            active_days = int(area_history["historical_day"].nunique())

            if area not in known_areas:
                status = "no_historical_area"
            elif (active_days < support["min_active_days"]
                  or historic_total < support["min_reports"]):
                status = "insufficient_historical_support"
            elif uncovered:
                status = "category_without_historical_coverage"
            elif current_total < min_current_reports:
                status = "small_current_sample"
            else:
                status = "supported"

            historic_top = str(historic_counts.idxmax()) if status == "supported" else None
            distance_differences = []
            for category in categories:
                current_n = int(current_counts.get(category, 0))
                hist_n = int(historic_counts.get(category, 0))
                current_pct = 100 * current_n / current_total
                historical_pct = (
                    100 * hist_n / historic_total
                    if status == "supported" and category in observed
                    else np.nan
                )
                if status == "supported" and category in observed:
                    distance_differences.append(abs(current_pct - historical_pct))
                detail_rows.append({
                    "area_id": area,
                    "reference_window": window_name,
                    "complaint_category": category,
                    "comparison_status": status,
                    "historical_reports": hist_n,
                    "current_reports": current_n,
                    "historical_share_pct": round(historical_pct, 3) if pd.notna(historical_pct) else np.nan,
                    "current_share_pct": round(current_pct, 3),
                    "share_change_pp": round(current_pct - historical_pct, 3)
                        if pd.notna(historical_pct) else np.nan,
                })
            summary_rows.append({
                "area_id": area,
                "reference_window": window_name,
                "comparison_status": status,
                "historical_calendar_days": calendar_days,
                "historical_active_days": active_days,
                "historical_reports": historic_total,
                "current_reports": current_total,
                "historical_top_category": historic_top,
                "current_top_category": current_top,
                "top_category_changed": historic_top != current_top
                    if status == "supported" else None,
                "category_mix_distance": round(sum(distance_differences) / 200, 4)
                    if status == "supported" else np.nan,
                "uncovered_current_categories": ", ".join(uncovered),
            })

    return {
        "status": "compared",
        "eligible_reports": int(len(eligible)),
        "matched_area_count": int(eligible["area_id"].isin(known_areas).groupby(eligible["area_id"]).all().sum()),
        "unmatched_areas": sorted(set(eligible["area_id"]) - known_areas),
        "summary": pd.DataFrame(summary_rows),
        "details": pd.DataFrame(detail_rows),
    }
