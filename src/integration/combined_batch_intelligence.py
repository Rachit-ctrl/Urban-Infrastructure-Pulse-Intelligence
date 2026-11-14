"""Combine daily new-batch and historical complaint context.

Historical model results are used as existing
record-level context, never as new model inputs.

No infrastructure defects or priorities are inferred.
"""

import pandas as pd

from src.integration.daily_batch_window import (
    select_daily_complaints
)

from src.models.new_batch_area_kmeans import (
    analyze_new_batch_areas
)

from src.models.new_batch_dbscan import (
    analyze_new_batch_spatial
)


def _joined_unique(values):
    return " | ".join(
        sorted({
            str(value)
            for value in values.dropna()
            if str(value).strip()
        })
    )


def analyze_combined_daily_batch(
    records,
    supported_categories,
    reporting_day,
    k=3,
    min_area_size=20,
    eps_m=125,
    min_samples=5,
    min_report_count=5,
    min_distinct_locations=2,
    random_state=18
):
    """Analyze one NYC reporting day and retain full audit."""

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a DataFrame.")

    if records.columns.duplicated().any():
        raise ValueError("Duplicate column names.")

    source = records.copy().reset_index(drop=True)

    # 1. Select one NYC-local day.
    daily = select_daily_complaints(
        source,
        reporting_day
    )

    selected = daily["selected_records"]

    # 2. Fit the two independent new-batch models.
    km = analyze_new_batch_areas(
        selected,
        supported_categories,
        k=k,
        min_area_size=min_area_size,
        random_state=random_state
    )

    spatial = analyze_new_batch_spatial(
        selected,
        supported_categories,
        eps_m=eps_m,
        min_samples=min_samples,
        min_report_count=min_report_count,
        min_distinct_locations=min_distinct_locations
    )

    # 3. Prepare per-complaint K-Means eligibility.
    km_rows = (
        km["record_eligibility"][[
            "source_row",
            "area_id",
            "eligibility_reason"
        ]]
        .rename(columns={
            "area_id": "new_kmeans_area_id",
            "eligibility_reason":
                "new_kmeans_eligibility"
        })
    )

    # 4. Prepare area assignments. These columns
    # may not exist if K-Means could not fit.
    km_areas = km["area_summary"].copy()

    for column in [
        "new_batch_cluster",
        "new_batch_description"
    ]:
        if column not in km_areas.columns:
            km_areas[column] = pd.NA

    km_areas = km_areas[[
        "area_id",
        "incoming_complaints",
        "meets_minimum",
        "new_batch_cluster",
        "new_batch_description"
    ]].rename(columns={
        "area_id": "new_kmeans_area_id",
        "incoming_complaints":
            "new_kmeans_area_complaints",
        "meets_minimum":
            "new_kmeans_area_meets_minimum",
        "new_batch_cluster":
            "new_kmeans_cluster",
        "new_batch_description":
            "new_kmeans_description"
    })

    # 5. Prepare per-complaint DBSCAN results.
    spatial_rows = (
        spatial["records"][[
            "source_row",
            "record_eligibility",
            "new_batch_status",
            "raw_candidate_id",
            "new_batch_candidate_id"
        ]]
        .rename(columns={
            "record_eligibility":
                "new_dbscan_eligibility",
            "new_batch_status":
                "new_dbscan_status",
            "raw_candidate_id":
                "new_dbscan_raw_candidate_id",
            "new_batch_candidate_id":
                "new_dbscan_candidate_id"
        })
    )

    date_rows = daily["audit"][[
        "source_row",
        "nyc_reporting_day",
        "date_parse_status",
        "daily_window_status"
    ]]

    # 6. Use source_row, never model cluster IDs,
    # as the join key.
    combined = (
        source
        .merge(
            date_rows,
            on="source_row",
            how="left",
            validate="one_to_one",
            sort=False
        )
        .merge(
            km_rows,
            on="source_row",
            how="left",
            validate="one_to_one",
            sort=False
        )
        .merge(
            km_areas,
            on="new_kmeans_area_id",
            how="left",
            validate="many_to_one",
            sort=False
        )
        .merge(
            spatial_rows,
            on="source_row",
            how="left",
            validate="one_to_one",
            sort=False
        )
    )

    selected_mask = combined[
        "daily_window_status"
    ].eq("selected_day")

    # 7. Preserve excluded date records explicitly.
    for column in [
        "new_kmeans_eligibility",
        "new_dbscan_eligibility",
        "new_dbscan_status"
    ]:
        combined.loc[
            ~selected_mask, column
        ] = "not_in_reporting_day"

    combined["new_kmeans_fit_status"] = (
        "not_in_reporting_day"
    )

    combined.loc[
        selected_mask,
        "new_kmeans_fit_status"
    ] = km["status"]

    # 8. Build fresh candidate context. Do not
    # reuse briefs from a different reporting day.
    candidate_rows = combined.loc[
        combined["new_dbscan_status"].eq(
            "candidate_concentration"
        )
    ].copy()

    context_columns = [
        "new_dbscan_candidate_id",
        "associated_kmeans_areas",
        "linked_kmeans_area_ids",
        "linked_kmeans_patterns"
    ]

    if candidate_rows.empty:

        context = pd.DataFrame(
            columns=context_columns
        )

    else:

        context = (
            candidate_rows.groupby(
                "new_dbscan_candidate_id"
            )
            .agg(
                associated_kmeans_areas=(
                    "new_kmeans_area_id",
                    "nunique"
                ),
                linked_kmeans_area_ids=(
                    "new_kmeans_area_id",
                    _joined_unique
                ),
                linked_kmeans_patterns=(
                    "new_kmeans_description",
                    _joined_unique
                )
            )
            .reset_index()
        )

    candidate_intelligence = (
        spatial["candidate_profiles"]
        .merge(
            context,
            left_on="new_batch_candidate_id",
            right_on="new_dbscan_candidate_id",
            how="left",
            validate="one_to_one"
        )
        .drop(
            columns=["new_dbscan_candidate_id"]
        )
    )

    guidance = {
        "SEWER":
            "Review sewer complaint details and "
            "relevant maintenance history.",
        "STREET CONDITION":
            "Review roadway complaint descriptions "
            "and reported locations.",
        "STREET LIGHT CONDITION":
            "Verify whether reports describe "
            "the same lighting assets.",
        "TRAFFIC SIGNAL CONDITION":
            "Verify reported intersections "
            "and traffic-signal equipment."
    }

    candidate_intelligence[
        "suggested_review"
    ] = (
        candidate_intelligence[
            "complaint_category"
        ]
        .map(guidance)
        .fillna(
            "Review the complaint descriptions "
            "and submitted locations."
        )
        + " Check for duplicate reports and "
        + "confirm evidence before field action."
    )

    # 9. Reconcile full original input coverage.
    if len(combined) != len(source):
        raise AssertionError(
            "Combined output changed record count."
        )

    if (
        combined["source_row"].tolist()
        != source["source_row"].tolist()
    ):
        raise AssertionError(
            "Complaint order or identity changed."
        )

    if (
        spatial["stats"]["total_records"]
        != daily["stats"]["selected_records"]
    ):
        raise AssertionError(
            "DBSCAN did not receive the selected day."
        )

    if (
        km["stats"]["total_records"]
        != daily["stats"]["selected_records"]
    ):
        raise AssertionError(
            "K-Means did not receive the selected day."
        )

    if (
        candidate_intelligence[
            "incoming_complaints"
        ].sum() != spatial["stats"][
            "candidate_complaints"
        ]
    ):
        raise AssertionError(
            "Candidate complaint counts disagree."
        )

    stats = {
        "input_records": len(source),
        "reporting_day": reporting_day,
        "selected_records":
            daily["stats"]["selected_records"],
        "date_exclusions": (
            daily["stats"]["outside_selected_day"]
            + daily["stats"][
                "invalid_or_missing_date"
            ]
        ),
        "kmeans_status": km["status"],
        "kmeans_eligible_areas":
            km["stats"]["eligible_areas"],
        "kmeans_assigned_complaints": int(
            combined[
                "new_kmeans_cluster"
            ].notna().sum()
        ),
        "dbscan_reportable_candidates":
            spatial["stats"][
                "reportable_candidates"
            ],
        "dbscan_candidate_complaints":
            spatial["stats"][
                "candidate_complaints"
            ],
        "dbscan_below_reporting_threshold":
            spatial["stats"][
                "below_reporting_threshold"
            ],
        "dbscan_unclustered":
            spatial["stats"]["unclustered"],
        "dbscan_excluded":
            spatial["stats"]["excluded_records"]
    }

    return {
        "combined_records": combined,
        "selected_records": selected.copy(),
        "date_audit": daily["audit"],
        "area_summary": km["area_summary"],
        "candidate_intelligence":
            candidate_intelligence,
        "kmeans_output": km,
        "dbscan_output": spatial,
        "stats": stats
    }
