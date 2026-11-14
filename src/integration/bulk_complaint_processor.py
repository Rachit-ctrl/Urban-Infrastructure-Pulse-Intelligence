"""Batch processing using saved historical services.

Each input record receives a traceable outcome.
Existing models are never retrained or updated.
"""

import pandas as pd

from src.integration.bulk_complaint_contract import (
    validate_batch
)


def process_complaint_batch(
    frame,
    supported_categories,
    adapter,
    use_cache=True
):
    validated = validate_batch(
        frame,
        supported_categories
    )

    results = []
    responses_by_source_row = {}
    cache = {}
    backend_evaluations = 0

    for record in validated.to_dict("records"):

        output = dict(record)

        output.update({
            "processing_status": None,
            "system_status": None,
            "borough": None,
            "geography_status": None,
            "zip_validation_status": None,
            "dbscan_status": None,
            "dbscan_cluster_id": None,
            "kmeans_status": None,
            "kmeans_area_id": None,
            "kmeans_cluster_id": None,
            "kmeans_reporting_profile": None,
            "kmeans_display_allowed": False,
            "featured_count": 0,
            "supporting_count": 0,
            "limited_evidence_count": 0,
            "warning_count": 0,
            "processing_error": None
        })

        if record["validation_status"] != (
            "ready_for_geographic_validation"
        ):
            output["processing_status"] = (
                "input_rejected"
            )

            results.append(output)
            continue

        zip_value = record["incident_zip"]

        if pd.isna(zip_value):
            zip_value = None

        # The historical backend is deterministic
        # for identical normalized input values.
        # Caching avoids repeating costly lookups.
        key = (
            float(record["latitude"]),
            float(record["longitude"]),
            str(record["complaint_category"]),
            zip_value
        )

        try:

            if use_cache and key in cache:
                response = cache[key]

            else:
                backend_evaluations += 1

                response = adapter.evaluate(
                    key[0],
                    key[1],
                    key[2],
                    key[3]
                )

                if use_cache:
                    cache[key] = response

            geography = response.get(
                "geography"
            ) or {}

            dbscan = response.get(
                "dbscan"
            ) or {}

            kmeans = response.get(
                "kmeans"
            ) or {}

            zip_info = response.get(
                "zip_validation"
            ) or {}

            recommendations = response.get(
                "recommendations"
            ) or {}

            warnings = response.get(
                "warnings"
            ) or []

            needs_review = (
                response.get("status")
                == "geographic_review_required"
            )

            output.update({
                "processing_status": (
                    "geographic_review_required"
                    if needs_review
                    else "processed"
                ),
                "system_status": response.get(
                    "status"
                ),
                "borough": geography.get(
                    "borough"
                ),
                "geography_status": geography.get(
                    "status"
                ),
                "zip_validation_status":
                    zip_info.get("status"),
                "dbscan_status": dbscan.get(
                    "status"
                ),
                "dbscan_cluster_id":
                    dbscan.get("cluster_id"),
                "kmeans_status": kmeans.get(
                    "status"
                ),
                "kmeans_area_id": kmeans.get(
                    "area_id"
                ),
                "kmeans_cluster_id":
                    kmeans.get("cluster_id"),
                "kmeans_reporting_profile":
                    kmeans.get("reporting_profile"),
                "kmeans_display_allowed": (
                    kmeans.get("display_allowed")
                    is True
                ),
                "featured_count": len(
                    recommendations.get(
                        "featured", []
                    )
                ),
                "supporting_count": len(
                    recommendations.get(
                        "supporting", []
                    )
                ),
                "limited_evidence_count": len(
                    recommendations.get(
                        "limited_evidence", []
                    )
                ),
                "warning_count": len(warnings)
            })

            # Preserve full historical evidence
            # separately, indexed by source row.
            responses_by_source_row[
                int(record["source_row"])
            ] = response

        except Exception as exc:

            output["processing_status"] = (
                "backend_error"
            )

            output["processing_error"] = (
                type(exc).__name__
                + ": "
                + str(exc)[:250]
            )

        results.append(output)

    table = pd.DataFrame.from_records(results)

    return {
        "records": table,
        "responses_by_source_row":
            responses_by_source_row,
        "stats": {
            "total": len(table),
            "processed": int(
                table["processing_status"]
                .eq("processed").sum()
            ),
            "geographic_review": int(
                table["processing_status"]
                .eq("geographic_review_required")
                .sum()
            ),
            "input_rejected": int(
                table["processing_status"]
                .eq("input_rejected").sum()
            ),
            "backend_errors": int(
                table["processing_status"]
                .eq("backend_error").sum()
            ),
            "backend_evaluations":
                backend_evaluations,
            "cache_entries": len(cache)
        }
    }
