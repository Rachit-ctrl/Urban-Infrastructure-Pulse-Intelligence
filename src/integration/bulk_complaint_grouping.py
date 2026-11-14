"""Summaries of incoming complaints and historical context.

These are descriptive reporting groups, not
operational safety priorities or new ML clusters.
"""

import pandas as pd


def summarize_bulk_complaints(
    batch_output,
    hotspot_reference
):

    records = batch_output["records"].copy()
    reference = hotspot_reference.copy()

    required = {
        "source_row",
        "complaint_id",
        "created_date",
        "processing_status",
        "borough",
        "complaint_category",
        "latitude",
        "longitude",
        "dbscan_status",
        "dbscan_cluster_id",
        "zip_validation_status",
        "kmeans_display_allowed",
        "kmeans_area_id",
        "kmeans_cluster_id",
        "kmeans_reporting_profile"
    }

    if not required.issubset(records.columns):
        raise ValueError(
            "Bulk processor output schema changed."
        )

    if records.empty:
        raise ValueError("Empty batch results.")

    if records["source_row"].duplicated().any():
        raise ValueError("Duplicate source rows.")

    if not {
        "dbscan_cluster_id",
        "total_complaints",
        "primary_borough",
        "dominant_category"
    }.issubset(reference.columns):
        raise ValueError(
            "Historical hotspot reference changed."
        )

    if reference[
        "dbscan_cluster_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate historical hotspot IDs."
        )

    records["dbscan_cluster_id"] = pd.to_numeric(
        records["dbscan_cluster_id"],
        errors="coerce"
    ).astype("Int64")

    records["kmeans_cluster_id"] = pd.to_numeric(
        records["kmeans_cluster_id"],
        errors="coerce"
    ).astype("Int64")

    reference["dbscan_cluster_id"] = pd.to_numeric(
        reference["dbscan_cluster_id"],
        errors="raise"
    ).astype("Int64")

    # ----------------------------------------------
    # 2. Separate input processing outcomes
    # ----------------------------------------------

    processed = records.loc[
        records["processing_status"].eq(
            "processed"
        )
    ].copy()

    geographic_review = records.loc[
        records["processing_status"].eq(
            "geographic_review_required"
        )
    ].copy()

    rejected = records.loc[
        records["processing_status"].eq(
            "input_rejected"
        )
    ].copy()

    backend_errors = records.loc[
        records["processing_status"].eq(
            "backend_error"
        )
    ].copy()

    if (
        len(processed)
        + len(geographic_review)
        + len(rejected)
        + len(backend_errors)
        != len(records)
    ):
        raise ValueError(
            "Batch processing totals do not reconcile."
        )

    # ----------------------------------------------
    # 3. Group processed complaints by borough
    # ----------------------------------------------

    by_borough = (
        processed.groupby("borough")
        .size()
        .reset_index(name="new_complaints")
        .sort_values(
            ["new_complaints", "borough"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------
    # 4. Group processed complaints by category
    # ----------------------------------------------

    by_category = (
        processed.groupby("complaint_category")
        .size()
        .reset_index(name="new_complaints")
        .sort_values(
            ["new_complaints", "complaint_category"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )

    if len(processed):
        by_category["share_of_processed_pct"] = (
            100
            * by_category["new_complaints"]
            / len(processed)
        ).round(2)
    else:
        by_category["share_of_processed_pct"] = (
            pd.Series(dtype=float)
        )

    # ----------------------------------------------
    # 5. Group new reports by historical DBSCAN ID
    # ----------------------------------------------

    associated = processed.loc[
        processed["dbscan_status"].eq(
            "established_cluster"
        )
    ].copy()

    if associated[
        "dbscan_cluster_id"
    ].isna().any():
        raise ValueError(
            "Established hotspot has no cluster ID."
        )

    # Exact coordinates are descriptive only.
    # Multiple reports at the same coordinate
    # do not prove multiple physical defects.
    associated["location_key"] = [
        repr(float(lat)) + "|" + repr(float(lon))
        for lat, lon in zip(
            associated["latitude"],
            associated["longitude"]
        )
    ]

    hotspot_groups = (
        associated.groupby("dbscan_cluster_id")
        .agg(
            incoming_complaints=(
                "source_row", "size"
            ),
            distinct_submitted_locations=(
                "location_key", "nunique"
            ),
            incoming_categories=(
                "complaint_category", "nunique"
            )
        )
        .reset_index()
    )

    historical_columns = reference[[
        "dbscan_cluster_id",
        "total_complaints",
        "primary_borough",
        "dominant_category"
    ]].rename(columns={
        "total_complaints":
            "historical_complaints_2023_2025",
        "primary_borough":
            "historical_primary_borough",
        "dominant_category":
            "historical_dominant_category"
    })

    hotspot_groups = hotspot_groups.merge(
        historical_columns,
        on="dbscan_cluster_id",
        how="left",
        validate="one_to_one"
    )

    if hotspot_groups[
        "historical_complaints_2023_2025"
    ].isna().any():
        raise ValueError(
            "Associated hotspot missing from "
            "historical reference."
        )

    hotspot_groups = hotspot_groups.sort_values(
        ["incoming_complaints", "dbscan_cluster_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # ----------------------------------------------
    # 6. Category distribution within each hotspot
    # ----------------------------------------------

    by_hotspot_category = (
        associated.groupby([
            "dbscan_cluster_id",
            "complaint_category"
        ])
        .size()
        .reset_index(name="new_complaints")
    )

    if not by_hotspot_category.empty:

        totals = (
            hotspot_groups.set_index(
                "dbscan_cluster_id"
            )["incoming_complaints"]
        )

        by_hotspot_category[
            "share_within_incoming_hotspot_pct"
        ] = (
            100
            * by_hotspot_category["new_complaints"]
            / by_hotspot_category[
                "dbscan_cluster_id"
            ].map(totals)
        ).round(2)

    # ----------------------------------------------
    # 7. Group displayed historical K-Means profiles
    # ----------------------------------------------

    shown = processed.loc[
        processed[
            "kmeans_display_allowed"
        ].eq(True)
    ].copy()

    if shown[
        [
            "kmeans_area_id",
            "kmeans_cluster_id",
            "kmeans_reporting_profile"
        ]
    ].isna().any().any():
        raise ValueError(
            "Displayed area profile is incomplete."
        )

    area_groups = (
        shown.groupby([
            "kmeans_area_id",
            "kmeans_cluster_id",
            "kmeans_reporting_profile",
            "zip_validation_status"
        ], dropna=False)
        .size()
        .reset_index(name="new_complaints")
        .sort_values(
            "new_complaints",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------
    # 8. Review cases are not a safety ranking
    # ----------------------------------------------

    zip_review = records.loc[
        records["zip_validation_status"].isin([
            "zcta_mismatch",
            "boundary_review_required"
        ])
    ].copy()

    review_mask = (
        records["processing_status"].isin([
            "geographic_review_required",
            "backend_error"
        ])
        | records["zip_validation_status"].isin([
            "zcta_mismatch",
            "boundary_review_required"
        ])
    )

    review_queue = records.loc[
        review_mask
    ].copy()

    # ----------------------------------------------
    # 9. Daily incoming complaint counts
    # ----------------------------------------------

    by_date = (
        records.groupby([
            "created_date",
            "processing_status"
        ], dropna=False)
        .size()
        .reset_index(name="complaints")
    )

    # ----------------------------------------------
    # 10. Summary with explicit denominators
    # ----------------------------------------------

    summary = {
        "total_input": len(records),
        "processed": len(processed),
        "geographic_review": len(
            geographic_review
        ),
        "input_rejected": len(rejected),
        "backend_errors": len(backend_errors),
        "associated_with_historical_hotspots":
            len(associated),
        "distinct_associated_historical_hotspots":
            len(hotspot_groups),
        "historical_area_profiles_displayed":
            len(shown),
        "zip_review": len(zip_review),
        "review_queue_rows": len(review_queue)
    }

    return {
        "summary": summary,
        "by_borough": by_borough,
        "by_category": by_category,
        "historical_hotspots": hotspot_groups,
        "by_hotspot_category":
            by_hotspot_category,
        "historical_area_profiles":
            area_groups,
        "by_date": by_date,
        "review_queue": review_queue
    }
