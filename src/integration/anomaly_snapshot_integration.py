"""Load historical daily counts and combine three models.

Integrates:
- New-batch area K-Means
- New-batch spatial DBSCAN
- Historical-versus-incoming anomaly screening

Historical models remain unchanged.
Anomaly flags are screening evidence only.
"""

import json

import numpy as np
import pandas as pd

from src.integration.combined_batch_intelligence import (
    analyze_combined_daily_batch
)

from src.models.new_batch_anomaly_screen import (
    screen_daily_anomalies
)


def load_anomaly_snapshot(
    snapshot_path,
    manifest_path
):
    """Load and validate saved historical evidence."""

    with open(
        manifest_path,
        encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    required_manifest = {
        "historical_start",
        "historical_end",
        "calendar_days",
        "eligible_rows",
        "area_count",
        "category_count",
        "sparse_daily_entries",
        "supported_categories",
        "observed_eligible_categories",
        "categories_without_eligible_area_records"
    }

    missing = required_manifest - set(manifest)

    if missing:
        raise ValueError(
            f"Missing manifest fields: {sorted(missing)}"
        )

    frame = pd.read_csv(
        snapshot_path,
        compression="gzip",
        dtype="string",
        keep_default_na=False
    )

    required_columns = [
        "area_id",
        "complaint_category",
        "historical_day",
        "complaint_count"
    ]

    if list(frame.columns) != required_columns:
        raise ValueError(
            "Unexpected historical snapshot schema."
        )

    if frame.duplicated(
        required_columns[:3]
    ).any():
        raise ValueError(
            "Duplicate historical daily-count keys."
        )

    valid_counts = frame[
        "complaint_count"
    ].str.fullmatch(
        r"[1-9][0-9]*"
    ).all()

    if not valid_counts:
        raise ValueError(
            "Historical counts must be positive integers."
        )

    frame["complaint_count"] = (
        frame["complaint_count"].astype("int64")
    )

    calendar = pd.date_range(
        manifest["historical_start"],
        manifest["historical_end"],
        freq="D"
    )

    if len(calendar) != int(
        manifest["calendar_days"]
    ):
        raise ValueError(
            "Historical calendar length mismatch."
        )

    historical_days = frame[
        "historical_day"
    ].astype(str)

    parsed_days = pd.to_datetime(
        historical_days,
        format="%Y-%m-%d",
        errors="coerce"
    )

    if parsed_days.isna().any():
        raise ValueError(
            "Invalid historical day in snapshot."
        )

    expected_days = {
        day.strftime("%Y-%m-%d")
        for day in calendar
    }

    if set(historical_days) != expected_days:
        raise ValueError(
            "Snapshot calendar coverage mismatch."
        )

    area_ids = sorted(
        frame["area_id"].astype(str).unique()
    )

    observed_categories = sorted(
        frame["complaint_category"]
        .astype(str).unique()
    )

    supported_categories = sorted({
        str(category).strip().upper()
        for category in manifest[
            "supported_categories"
        ]
    })

    missing_categories = sorted(
        set(supported_categories)
        - set(observed_categories)
    )

    if len(area_ids) != int(
        manifest["area_count"]
    ):
        raise ValueError(
            "Historical area count mismatch."
        )

    if len(supported_categories) != int(
        manifest["category_count"]
    ):
        raise ValueError(
            "Supported taxonomy count mismatch."
        )

    if observed_categories != sorted(
        manifest["observed_eligible_categories"]
    ):
        raise ValueError(
            "Observed category metadata mismatch."
        )

    if missing_categories != sorted(
        manifest[
            "categories_without_eligible_area_records"
        ]
    ):
        raise ValueError(
            "Missing-category metadata mismatch."
        )

    if len(frame) != int(
        manifest["sparse_daily_entries"]
    ):
        raise ValueError(
            "Daily-count row total mismatch."
        )

    if int(
        frame["complaint_count"].sum()
    ) != int(manifest["eligible_rows"]):
        raise ValueError(
            "Historical complaint total mismatch."
        )

    counts = {
        (str(area), str(category), str(day)): int(count)
        for area, category, day, count in frame[
            required_columns
        ].itertuples(
            index=False,
            name=None
        )
    }

    return {
        "daily_counts": counts,
        "calendar": calendar,
        "area_ids": area_ids,
        "supported_categories":
            supported_categories,
        "observed_categories":
            observed_categories,
        "missing_categories":
            missing_categories,
        "manifest": manifest,
        "snapshot_rows": int(len(frame))
    }


def _joined_unique(values):
    return " | ".join(
        sorted({
            str(value)
            for value in values.dropna()
            if str(value).strip()
        })
    )


def analyze_batch_with_anomaly_context(
    records,
    baseline,
    reporting_day,
    min_incoming_reports=8
):
    """Combine three independent incoming analyses.

    Keeps all original complaint records and
    all reporting-day exclusions traceable.
    """

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a DataFrame.")

    categories = baseline[
        "supported_categories"
    ]

    combined = analyze_combined_daily_batch(
        records,
        categories,
        reporting_day
    )

    anomaly = screen_daily_anomalies(
        records=records,
        supported_categories=categories,
        historical_daily_counts=baseline[
            "daily_counts"
        ],
        historical_calendar=baseline[
            "calendar"
        ],
        historical_area_ids=baseline[
            "area_ids"
        ],
        reporting_day=reporting_day,
        min_incoming_reports=min_incoming_reports
    )

    # ----------------------------------------------
    # 1. Attach per-record anomaly audit
    # ----------------------------------------------

    fused = combined[
        "combined_records"
    ].copy()

    audit = anomaly["record_audit"][[
        "source_row",
        "anomaly_input_status",
        "anomaly_review_status"
    ]]

    fused = fused.merge(
        audit,
        on="source_row",
        how="left",
        validate="one_to_one",
        sort=False
    )

    fused["anomaly_area_id"] = (
        fused["borough"]
        .astype(str)
        .str.strip()
        .str.upper()
        + "_"
        + fused["incident_zip"]
        .astype(str)
        .str.strip()
    )

    fused["anomaly_category"] = (
        fused["complaint_category"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ----------------------------------------------
    # 2. Attach area-category reference evidence
    # ----------------------------------------------

    comparison = anomaly[
        "comparisons"
    ][[
        "area_id",
        "complaint_category",
        "incoming_reports",
        "seasonal_support",
        "seasonal_daily_mean",
        "seasonal_daily_max",
        "all_weekday_support",
        "all_weekday_daily_max"
    ]].rename(columns={
        "area_id":
            "anomaly_area_id",
        "complaint_category":
            "anomaly_category",
        "incoming_reports":
            "anomaly_area_category_reports",
        "seasonal_support":
            "anomaly_seasonal_support",
        "seasonal_daily_mean":
            "anomaly_historical_daily_mean",
        "seasonal_daily_max":
            "anomaly_seasonal_max",
        "all_weekday_support":
            "anomaly_broad_support",
        "all_weekday_daily_max":
            "anomaly_broad_max"
    })

    fused = fused.merge(
        comparison,
        on=[
            "anomaly_area_id",
            "anomaly_category"
        ],
        how="left",
        validate="many_to_one",
        sort=False
    )

    # An excluded complaint must not inherit an
    # anomaly comparison simply because another
    # eligible complaint has the same area/category.
    eligible_mask = fused[
        "anomaly_input_status"
    ].eq("eligible")

    evidence_columns = [
        "anomaly_area_category_reports",
        "anomaly_seasonal_support",
        "anomaly_historical_daily_mean",
        "anomaly_seasonal_max",
        "anomaly_broad_support",
        "anomaly_broad_max"
    ]

    # Pandas nullable dtypes are required because
    # excluded complaints must have missing evidence.
    # Ordinary numpy bool/int columns cannot reliably
    # accept pd.NA through DataFrame assignment.

    boolean_columns = [
        "anomaly_seasonal_support",
        "anomaly_broad_support"
    ]

    integer_columns = [
        "anomaly_area_category_reports",
        "anomaly_seasonal_max",
        "anomaly_broad_max"
    ]

    float_columns = [
        "anomaly_historical_daily_mean"
    ]

    for column in boolean_columns:
        fused[column] = fused[column].astype(
            "boolean"
        )

    for column in integer_columns:
        fused[column] = pd.to_numeric(
            fused[column],
            errors="coerce"
        ).astype("Int64")

    for column in float_columns:
        fused[column] = pd.to_numeric(
            fused[column],
            errors="coerce"
        ).astype("Float64")

    # Assign one column at a time to avoid
    # mixed-dtype block assignment problems.
    excluded_mask = ~eligible_mask

    if excluded_mask.any():
        for column in evidence_columns:
            fused.loc[
                excluded_mask, column
            ] = pd.NA

    fused["anomaly_candidate"] = fused[
        "anomaly_review_status"
    ].eq(
        "cross_window_consistent_candidate"
    )

    # ----------------------------------------------
    # 3. Explain how the models complement each other
    # ----------------------------------------------

    flagged_rows = fused.loc[
        fused["anomaly_candidate"]
    ].copy()

    context_columns = [
        "area_id",
        "complaint_category",
        "linked_kmeans_patterns",
        "dbscan_candidate_complaints",
        "distinct_dbscan_candidates",
        "historical_established_complaints",
        "historical_outside_complaints"
    ]

    if flagged_rows.empty:

        context = pd.DataFrame(
            columns=context_columns
        )

    else:

        context = (
            flagged_rows.groupby([
                "anomaly_area_id",
                "anomaly_category"
            ])
            .agg(
                linked_kmeans_patterns=(
                    "new_kmeans_description",
                    _joined_unique
                ),
                dbscan_candidate_complaints=(
                    "new_dbscan_status",
                    lambda values: int(
                        values.eq(
                            "candidate_concentration"
                        ).sum()
                    )
                ),
                distinct_dbscan_candidates=(
                    "new_dbscan_candidate_id",
                    "nunique"
                ),
                historical_established_complaints=(
                    "dbscan_status",
                    lambda values: int(
                        values.eq(
                            "established_cluster"
                        ).sum()
                    )
                ),
                historical_outside_complaints=(
                    "dbscan_status",
                    lambda values: int(
                        values.eq(
                            "outside_existing_clusters"
                        ).sum()
                    )
                )
            )
            .reset_index()
            .rename(columns={
                "anomaly_area_id":
                    "area_id",
                "anomaly_category":
                    "complaint_category"
            })
        )

    candidates = (
        anomaly["candidates"]
        .merge(
            context,
            on=[
                "area_id",
                "complaint_category"
            ],
            how="left",
            validate="one_to_one"
        )
    )

    if not candidates.empty:

        candidates["explanation"] = [
            (
                f"{int(row.incoming_reports)} "
                f"{row.complaint_category} complaints "
                f"were reported in {row.area_id} "
                f"on {reporting_day}. "
                f"This count exceeds the observed "
                f"seasonal historical maximum "
                f"({int(row.seasonal_daily_max)}) "
                f"and all-weekday historical maximum "
                f"({int(row.all_weekday_daily_max)}). "
                f"This is an exploratory reporting "
                f"increase, not a confirmed defect."
            )
            for row in candidates.itertuples(
                index=False
            )
        ]

        candidates["suggested_review"] = (
            "Check complaint descriptions, duplicates, "
            "location accuracy, historical reporting "
            "coverage and recent operational records "
            "before considering investigation."
        )

    else:

        candidates["explanation"] = pd.Series(
            dtype="string"
        )

        candidates["suggested_review"] = pd.Series(
            dtype="string"
        )

    # ----------------------------------------------
    # 4. Final reconciliation
    # ----------------------------------------------

    if len(fused) != len(records):
        raise AssertionError(
            "Integrated records did not reconcile."
        )

    if fused["source_row"].tolist() != (
        records["source_row"].tolist()
    ):
        raise AssertionError(
            "Original record order changed."
        )

    if (
        int(fused["anomaly_candidate"].sum())
        != anomaly["stats"][
            "candidate_complaints"
        ]
    ):
        raise AssertionError(
            "Anomaly candidate counts disagree."
        )

    if (
        int(fused["anomaly_candidate"].sum())
        != int(
            candidates[
                "incoming_reports"
            ].sum()
        )
    ):
        raise AssertionError(
            "Candidate evidence did not reconcile."
        )

    if fused.loc[
        ~eligible_mask,
        "anomaly_candidate"
    ].any():
        raise AssertionError(
            "Excluded records cannot be candidates."
        )

    if (
        combined["stats"]["selected_records"]
        != anomaly["stats"]["selected_records"]
    ):
        raise AssertionError(
            "Daily selections disagree."
        )

    stats = {
        "input_records": len(fused),
        "reporting_day": reporting_day,
        "selected_records":
            anomaly["stats"]["selected_records"],
        "kmeans_status":
            combined["stats"]["kmeans_status"],
        "kmeans_assigned_complaints":
            combined["stats"][
                "kmeans_assigned_complaints"
            ],
        "dbscan_reportable_candidates":
            combined["stats"][
                "dbscan_reportable_candidates"
            ],
        "dbscan_candidate_complaints":
            combined["stats"][
                "dbscan_candidate_complaints"
            ],
        "anomaly_eligible_records":
            anomaly["stats"]["eligible_records"],
        "anomaly_candidate_combinations":
            anomaly["stats"]["candidate_combinations"],
        "anomaly_candidate_complaints":
            anomaly["stats"]["candidate_complaints"],
        "historical_baseline_areas":
            len(baseline["area_ids"]),
        "historically_observed_categories":
            len(baseline["observed_categories"])
    }

    return {
        "combined_records": fused,
        "anomaly_candidates": candidates,
        "anomaly_comparisons":
            anomaly["comparisons"],
        "date_audit": anomaly["record_audit"],
        "kmeans_dbscan_output": combined,
        "anomaly_output": anomaly,
        "stats": stats
    }
