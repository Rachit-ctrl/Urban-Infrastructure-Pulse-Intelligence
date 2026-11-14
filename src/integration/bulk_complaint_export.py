"""Safe CSV exports for bulk complaint outcomes.

Exported review queues are not operational
risk rankings or repair assignments.
"""

from io import BytesIO
import json
import pandas as pd


REVIEW_ZIP_STATUSES = {
    "zcta_mismatch",
    "boundary_review_required"
}


def _safe_cell(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)

    # Prevent spreadsheet formula execution
    # when exported text is opened in Excel.
    if text.lstrip().startswith(
        ("=", "+", "-", "@")
    ) or text.startswith(("\t", "\r", "\n")):
        return "'" + text

    return text


def csv_bytes(frame):
    work = frame.copy()

    for column in work.columns:

        if pd.api.types.is_numeric_dtype(
            work[column]
        ):
            continue

        work[column] = work[column].map(
            _safe_cell
        )

    return work.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n"
    ).encode("utf-8-sig")


def build_bulk_export_package(batch_output):

    records = batch_output["records"].copy()

    required = {
        "source_row",
        "complaint_id",
        "processing_status",
        "validation_errors",
        "zip_validation_status",
        "dbscan_cluster_id",
        "kmeans_display_allowed"
    }

    if not required.issubset(records.columns):
        raise ValueError(
            "Bulk processor output schema changed."
        )

    if records.empty:
        raise ValueError("No complaint outcomes.")

    if records["source_row"].duplicated().any():
        raise ValueError(
            "Duplicate source-row identifiers."
        )

    statuses = records["processing_status"]

    allowed = {
        "processed",
        "geographic_review_required",
        "input_rejected",
        "backend_error"
    }

    if (
        statuses.isna().any()
        or not set(statuses).issubset(allowed)
    ):
        raise ValueError(
            "Unexpected processing status."
        )

    geo_mask = statuses.eq(
        "geographic_review_required"
    )

    invalid_mask = statuses.eq(
        "input_rejected"
    )

    error_mask = statuses.eq(
        "backend_error"
    )

    processed_mask = statuses.eq(
        "processed"
    )

    zip_mask = records[
        "zip_validation_status"
    ].isin(REVIEW_ZIP_STATUSES)

    # Review queues are subsets. An otherwise
    # processed complaint may need ZIP review.
    review_mask = (
        geo_mask | zip_mask | error_mask
    )

    records["review_reason"] = ""

    records.loc[
        geo_mask, "review_reason"
    ] = "geographic_review"

    records.loc[
        zip_mask & ~geo_mask,
        "review_reason"
    ] = "zip_validation_review"

    records.loc[
        error_mask, "review_reason"
    ] = "backend_processing_error"

    review = records.loc[
        review_mask
    ].copy()

    rejected = records.loc[
        invalid_mask
    ].copy()

    processed = records.loc[
        processed_mask
    ].copy()

    zip_review = records.loc[
        zip_mask
    ].copy()

    errors = records.loc[
        error_mask
    ].copy()

    stats = batch_output.get("stats", {})

    if int(stats.get("total", -1)) != len(records):
        raise ValueError(
            "Batch input count mismatch."
        )

    status_counts = {
        name: int(statuses.eq(name).sum())
        for name in sorted(allowed)
    }

    for status, stat_name in [
        ("processed", "processed"),
        ("geographic_review_required",
         "geographic_review"),
        ("input_rejected", "input_rejected"),
        ("backend_error", "backend_errors")
    ]:
        if status_counts[status] != int(
            stats.get(stat_name, -1)
        ):
            raise ValueError(
                "Batch status totals do not agree."
            )

    summary = {
        "schema_version": "bulk_export_v1",
        "input_type": "synthetic_test_batch",
        "historical_reference_period": "2023-2025",
        "total_complaints": len(records),
        "status_counts": status_counts,
        "review_queue_rows": len(review),
        "zip_review_rows": len(zip_review),
        "rejected_input_rows": len(rejected),
        "backend_error_rows": len(errors),
        "historical_hotspot_associations": int(
            records["dbscan_cluster_id"]
            .notna().sum()
        ),
        "historical_kmeans_profiles_displayed": int(
            records["kmeans_display_allowed"]
            .eq(True).sum()
        ),
        "backend_evaluations": int(
            stats.get("backend_evaluations", 0)
        ),
        "notes": [
            "This batch contains synthetic test records.",
            "Review queues are not safety rankings.",
            "ZIP review can coexist with processed status.",
            "Historical clusters are not retrained.",
            "No new hotspots are discovered in this phase."
        ]
    }

    tables = {
        "all_complaint_outcomes.csv": records,
        "processed_complaints.csv": processed,
        "review_queue.csv": review,
        "zip_review.csv": zip_review,
        "rejected_inputs.csv": rejected,
        "backend_errors.csv": errors
    }

    files = {
        filename: csv_bytes(table)
        for filename, table in tables.items()
    }

    files["batch_summary.json"] = json.dumps(
        summary,
        indent=2,
        allow_nan=False
    ).encode("utf-8")

    return {
        "files": files,
        "tables": tables,
        "summary": summary
    }
