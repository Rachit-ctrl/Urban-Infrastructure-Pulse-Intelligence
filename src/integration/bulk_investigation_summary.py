"""Evidence-based briefs for incoming complaint batches.

Historical hotspot association is not proof of
a defect, repair urgency, or safety risk.
"""

import pandas as pd


def build_investigation_briefs(
    batch_output,
    historical_hotspots
):
    records = batch_output["records"].copy()
    responses = batch_output[
        "responses_by_source_row"
    ]

    reference = historical_hotspots.copy()

    required = {
        "source_row",
        "processing_status",
        "borough",
        "complaint_category",
        "latitude",
        "longitude",
        "dbscan_status",
        "dbscan_cluster_id",
        "zip_validation_status",
        "kmeans_display_allowed",
        "featured_count",
        "supporting_count",
        "limited_evidence_count"
    }

    if not required.issubset(records.columns):
        raise ValueError(
            "Unexpected bulk processor schema."
        )

    if records["source_row"].duplicated().any():
        raise ValueError("Duplicate source rows.")

    reference = reference.set_index(
        "dbscan_cluster_id",
        verify_integrity=True
    )

    # Only established historical associations
    # become historical hotspot briefs.
    established = records.loc[
        records["processing_status"].eq(
            "processed"
        )
        & records["dbscan_status"].eq(
            "established_cluster"
        )
    ].copy()

    established["dbscan_cluster_id"] = (
        pd.to_numeric(
            established["dbscan_cluster_id"],
            errors="raise"
        ).astype(int)
    )

    briefs = []

    for (cluster_id, category), group in (
        established.groupby([
            "dbscan_cluster_id",
            "complaint_category"
        ])
    ):
        cluster_id = int(cluster_id)

        if cluster_id not in reference.index:
            raise ValueError(
                "Historical hotspot reference missing."
            )

        historical = reference.loc[cluster_id]

        incoming_count = len(group)

        location_count = len(set(
            zip(
                group["latitude"].astype(float),
                group["longitude"].astype(float)
            )
        ))

        historical_count = int(
            historical["total_complaints"]
        )

        representative_row = int(
            group["source_row"].iloc[0]
        )

        response = responses.get(
            representative_row
        )

        if response is None:
            raise ValueError(
                "Missing representative backend evidence."
            )

        if (
            response["dbscan"]["cluster_id"]
            != cluster_id
        ):
            raise ValueError(
                "Representative hotspot mismatch."
            )

        category_context = (
            response["dbscan"].get(
                "category_context"
            ) or {}
        )

        historical_category_count = (
            category_context.get(
                "category_complaints"
            )
        )

        if historical_category_count is not None:
            historical_category_count = int(
                historical_category_count
            )

        actions = response.get(
            "recommendations"
        ) or {}

        featured = len(
            actions.get("featured", [])
        )

        supporting = len(
            actions.get("supporting", [])
        )

        limited = len(
            actions.get("limited_evidence", [])
        )

        for column, expected in [
            ("featured_count", featured),
            ("supporting_count", supporting),
            ("limited_evidence_count", limited)
        ]:
            if not group[column].eq(expected).all():
                raise ValueError(
                    "Recommendation count inconsistency."
                )

        zip_review_count = int(
            group["zip_validation_status"].isin([
                "zcta_mismatch",
                "boundary_review_required"
            ]).sum()
        )

        unverified_zip_count = int(
            group["zip_validation_status"].eq(
                "zcta_unavailable"
            ).sum()
        )

        displayed_area_count = int(
            group["kmeans_display_allowed"]
            .eq(True).sum()
        )

        # Avoid asserting that repeated input
        # records represent distinct defects.
        repeated_location_review = (
            incoming_count > location_count
        )

        evidence = (
            f"{incoming_count} incoming complaint "
            f"records in category {category} were "
            f"associated with historical hotspot "
            f"{cluster_id}, across "
            f"{location_count} distinct submitted "
            f"coordinate(s). The hotspot contains "
            f"{historical_count} historical reports "
            f"from 2023-2025."
        )

        if historical_category_count is not None:
            evidence += (
                f" Of these historical reports, "
                f"{historical_category_count} were "
                f"in category {category}."
            )

        safeguards = []

        if repeated_location_review:
            safeguards.append(
                "Check whether repeated coordinates "
                "represent duplicate reports or "
                "different incidents."
            )

        if zip_review_count:
            safeguards.append(
                f"{zip_review_count} incoming records "
                f"require ZIP geographic review."
            )

        if unverified_zip_count:
            safeguards.append(
                f"{unverified_zip_count} incoming "
                f"records use an unavailable ZCTA "
                f"reference."
            )

        safeguards.append(
            "Historical hotspot association does "
            "not establish physical defect severity."
        )

        suggested_review = (
            "Review related complaint descriptions, "
            "maintenance history and potential "
            "duplicate submissions. Consider "
            "coordinated field investigation when "
            "supported by verified operational "
            "information."
        )

        briefs.append({
            "historical_hotspot_id": cluster_id,
            "complaint_category": category,
            "incoming_complaints": incoming_count,
            "distinct_submitted_locations":
                location_count,
            "historical_complaints_2023_2025":
                historical_count,
            "historical_category_complaints":
                historical_category_count,
            "historical_primary_borough":
                historical["primary_borough"],
            "historical_dominant_category":
                historical["dominant_category"],
            "zip_review_records":
                zip_review_count,
            "zcta_unavailable_records":
                unverified_zip_count,
            "historical_area_profiles_displayed":
                displayed_area_count,
            "featured_investigations":
                featured,
            "supporting_investigations":
                supporting,
            "limited_evidence_investigations":
                limited,
            "representative_source_row":
                representative_row,
            "repeat_location_review":
                repeated_location_review,
            "evidence_explanation": evidence,
            "review_safeguards": " ".join(
                safeguards
            ),
            "suggested_investigation":
                suggested_review,
            "operational_priority":
                "not_assessed"
        })

    brief_table = pd.DataFrame.from_records(
        briefs
    )

    if not brief_table.empty:
        brief_table = brief_table.sort_values(
            [
                "incoming_complaints",
                "historical_hotspot_id"
            ],
            ascending=[False, True]
        ).reset_index(drop=True)

    # Complaints outside established historical
    # clusters must not disappear from the queue.
    unmatched = records.loc[
        records["processing_status"].eq(
            "processed"
        )
        & ~records["dbscan_status"].eq(
            "established_cluster"
        )
    ].copy()

    unmatched_groups = (
        unmatched.groupby(
            [
                "borough",
                "complaint_category",
                "dbscan_status"
            ],
            dropna=False
        )
        .size()
        .reset_index(name="incoming_complaints")
    )

    review_counts = {
        "geographic_review": int(
            records["processing_status"].eq(
                "geographic_review_required"
            ).sum()
        ),
        "zip_review": int(
            records["zip_validation_status"]
            .isin([
                "zcta_mismatch",
                "boundary_review_required"
            ]).sum()
        ),
        "invalid_input": int(
            records["processing_status"].eq(
                "input_rejected"
            ).sum()
        ),
        "backend_error": int(
            records["processing_status"].eq(
                "backend_error"
            ).sum()
        )
    }

    summary = {
        "total_complaints": len(records),
        "established_hotspot_reports":
            len(established),
        "historical_hotspot_category_briefs":
            len(brief_table),
        "unmatched_processed_reports":
            len(unmatched),
        "review_counts": review_counts,
        "interpretation": (
            "Briefs are descriptive investigation "
            "aids, not safety rankings or confirmed "
            "repair instructions."
        )
    }

    return {
        "briefs": brief_table,
        "unmatched_groups": unmatched_groups,
        "summary": summary
    }
