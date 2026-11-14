"""Convert three-model output into investigation briefs.

Explanations are descriptive screening evidence,
not confirmed infrastructure faults, risk scores,
statistical significance or repair priorities.
"""

import math
import numpy as np
import pandas as pd


def _unique(values):
    return sorted({
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    })


def _historical_relationship(group):

    statuses = group[
        "dbscan_status"
    ].astype(str)

    established = int(
        statuses.eq("established_cluster").sum()
    )

    outside = int(
        statuses.eq(
            "outside_existing_clusters"
        ).sum()
    )

    ambiguous = int(
        statuses.eq(
            "ambiguous_association"
        ).sum()
    )

    other = (
        len(group)
        - established
        - outside
        - ambiguous
    )

    if outside == len(group):
        relationship = (
            "entirely_outside_historical_associations"
        )

    elif established == len(group):
        relationship = (
            "entirely_associated_with_historical_clusters"
        )

    else:
        relationship = (
            "mixed_or_uncertain_historical_associations"
        )

    return {
        "relationship": relationship,
        "established": established,
        "outside": outside,
        "ambiguous": ambiguous,
        "other": other
    }


def _representative_location(group):
    """Return an actual submitted coordinate."""

    coordinates = group[[
        "latitude",
        "longitude"
    ]].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if coordinates.isna().any().any():
        raise ValueError(
            "Invalid spatial candidate coordinates"
        )

    values = coordinates.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite candidate coordinates"
        )

    center_lat, center_lon = (
        values.mean(axis=0)
    )

    lat = np.radians(values[:, 0])
    lon = np.radians(values[:, 1])

    lat0 = math.radians(
        float(center_lat)
    )

    lon0 = math.radians(
        float(center_lon)
    )

    a = (
        np.sin((lat - lat0) / 2) ** 2
        + np.cos(lat0)
        * np.cos(lat)
        * np.sin((lon - lon0) / 2) ** 2
    )

    distances = (
        2
        * 6371008.8
        * np.arcsin(
            np.sqrt(np.clip(a, 0, 1))
        )
    )

    chosen = values[
        int(np.argmin(distances))
    ]

    distinct = len(
        np.unique(values, axis=0)
    )

    return (
        float(chosen[0]),
        float(chosen[1]),
        int(distinct)
    )


def build_investigation_briefs(
    integrated_result,
    synthetic_data=False
):
    """Build two types of candidate explanation.

    1. Area-category reporting-increase briefs.
    2. Local spatial concentration briefs.

    Also returns record audit and map-ready GeoJSON.
    """

    rows = integrated_result[
        "combined_records"
    ]

    anomaly_candidates = integrated_result[
        "anomaly_candidates"
    ]

    profiles = integrated_result[
        "kmeans_dbscan_output"
    ]["dbscan_output"]["candidate_profiles"]

    reporting_day = str(
        integrated_result[
            "stats"
        ]["reporting_day"]
    )

    required_columns = {
        "source_row",
        "complaint_id",
        "daily_window_status",
        "complaint_category",
        "latitude",
        "longitude",
        "anomaly_area_id",
        "anomaly_category",
        "anomaly_input_status",
        "anomaly_review_status",
        "anomaly_candidate",
        "new_kmeans_description",
        "new_dbscan_status",
        "new_dbscan_candidate_id",
        "dbscan_status"
    }

    missing = (
        required_columns
        - set(rows.columns)
    )

    if missing:
        raise ValueError(
            f"Missing record columns: {sorted(missing)}"
        )

    if rows["source_row"].duplicated().any():
        raise ValueError(
            "Duplicate source_row values"
        )

    if rows["anomaly_candidate"].isna().any():
        raise ValueError(
            "Missing anomaly candidate flags"
        )

    # ----------------------------------------------
    # 2. Area/category anomaly explanations
    # ----------------------------------------------

    anomaly_briefs = []

    for item in anomaly_candidates.itertuples(
        index=False
    ):

        area = str(item.area_id)
        category = str(
            item.complaint_category
        )

        group = rows.loc[
            rows["anomaly_area_id"].eq(area)
            & rows["anomaly_category"].eq(
                category
            )
            & rows["anomaly_input_status"].eq(
                "eligible"
            )
        ]

        count = len(group)

        if (
            count != int(item.incoming_reports)
            or not group[
                "anomaly_candidate"
            ].all()
        ):
            raise ValueError(
                f"Anomaly count mismatch: "
                f"{area}, {category}"
            )

        spatial_count = int(
            group["new_dbscan_status"]
            .eq("candidate_concentration")
            .sum()
        )

        patterns = _unique(
            group["new_kmeans_description"]
        )

        spatial_ids = _unique(
            group["new_dbscan_candidate_id"]
        )

        historical = (
            _historical_relationship(group)
        )

        explanation = (
            f"{count} {category} complaints "
            f"were submitted in {area} on "
            f"{reporting_day}. The count exceeds "
            f"the historical seasonal maximum "
            f"of {int(item.seasonal_daily_max)} "
            f"and all-weekday maximum of "
            f"{int(item.all_weekday_daily_max)}. "
            f"{spatial_count} complaints also "
            f"belong to new-batch spatial "
            f"concentrations. Reporting pattern: "
            f"{', '.join(patterns) if patterns else 'unavailable'}. "
            f"This is an exploratory reporting "
            f"increase, not a confirmed defect."
        )

        anomaly_briefs.append({
            "evidence_type":
                "area_category_reporting_increase",
            "area_id": area,
            "complaint_category": category,
            "reporting_day": reporting_day,
            "incoming_complaints": count,
            "seasonal_historical_max":
                int(item.seasonal_daily_max),
            "all_weekday_historical_max":
                int(item.all_weekday_daily_max),
            "kmeans_reporting_patterns":
                patterns,
            "complaints_in_spatial_candidates":
                spatial_count,
            "linked_spatial_candidate_ids":
                spatial_ids,
            "historical_dbscan_associations":
                historical,
            "screening_status":
                "cross_window_consistent_candidate",
            "explanation": explanation,
            "suggested_review": (
                "Check report descriptions, "
                "duplicate submissions, location "
                "accuracy, category definitions "
                "and recent operational records "
                "before considering an investigation."
            ),
            "synthetic_data": bool(
                synthetic_data
            )
        })

    # ----------------------------------------------
    # 3. Spatial concentration explanations
    # ----------------------------------------------

    spatial_briefs = []
    map_features = []

    for profile in profiles.itertuples(
        index=False
    ):

        candidate_id = str(
            profile.new_batch_candidate_id
        )

        group = rows.loc[
            rows["new_dbscan_status"].eq(
                "candidate_concentration"
            )
            & rows[
                "new_dbscan_candidate_id"
            ].eq(candidate_id)
        ]

        count = len(group)

        if count != int(
            profile.incoming_complaints
        ):
            raise ValueError(
                f"Spatial count mismatch: "
                f"{candidate_id}"
            )

        categories = _unique(
            group["complaint_category"]
        )

        if (
            len(categories) != 1
            or categories[0] != str(
                profile.complaint_category
            )
        ):
            raise ValueError(
                f"Mixed spatial categories: "
                f"{candidate_id}"
            )

        lat, lon, distinct = (
            _representative_location(group)
        )

        if distinct != int(
            profile.distinct_locations
        ):
            raise ValueError(
                "Spatial location count mismatch"
            )

        historical = (
            _historical_relationship(group)
        )

        anomaly_overlap = int(
            group["anomaly_candidate"].sum()
        )

        area_ids = _unique(
            group["anomaly_area_id"]
        )

        patterns = _unique(
            group["new_kmeans_description"]
        )

        linked_anomaly_groups = sorted({
            (
                f"{row.anomaly_area_id} / "
                f"{row.anomaly_category}"
            )
            for row in group.loc[
                group["anomaly_candidate"]
            ].itertuples(index=False)
        })

        explanation = (
            f"{count} {categories[0]} complaints "
            f"at {distinct} distinct submitted "
            f"coordinates form spatial candidate "
            f"{candidate_id}. Historical "
            f"association: {historical['outside']} "
            f"outside, {historical['established']} "
            f"established, "
            f"{historical['ambiguous']} ambiguous "
            f"and {historical['other']} other. "
            f"{anomaly_overlap} members also "
            f"belong to area-category reporting "
            f"increase candidates. The map pin "
            f"is a submitted complaint coordinate, "
            f"not a verified infrastructure fault."
        )

        spatial_briefs.append({
            "evidence_type":
                "new_batch_spatial_concentration",
            "candidate_id": candidate_id,
            "complaint_category":
                categories[0],
            "reporting_day": reporting_day,
            "incoming_complaints": count,
            "distinct_submitted_locations":
                distinct,
            "representative_latitude":
                round(lat, 7),
            "representative_longitude":
                round(lon, 7),
            "area_ids": area_ids,
            "kmeans_reporting_patterns":
                patterns,
            "complaints_in_anomaly_candidate_groups":
                anomaly_overlap,
            "linked_anomaly_area_categories":
                linked_anomaly_groups,
            "historical_dbscan_associations":
                historical,
            "explanation": explanation,
            "suggested_review": (
                "Review nearby report descriptions, "
                "duplicate coordinates and location "
                "accuracy. Verify conditions before "
                "any field action."
            ),
            "synthetic_data": bool(
                synthetic_data
            )
        })

        map_features.append({
            "type": "Feature",
            "id": candidate_id,
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "candidate_id": candidate_id,
                "complaint_category":
                    categories[0],
                "incoming_complaints": count,
                "distinct_locations": distinct,
                "historical_relationship":
                    historical["relationship"],
                "complaints_in_anomaly_candidate_groups":
                    anomaly_overlap,
                "synthetic_data": bool(
                    synthetic_data
                )
            }
        })

    # ----------------------------------------------
    # 4. Traceable status for every original record
    # ----------------------------------------------

    audit_rows = []

    for item in rows.itertuples(
        index=False
    ):

        audit_rows.append({
            "source_row": str(
                item.source_row
            ),
            "complaint_id": str(
                item.complaint_id
            ),
            "daily_window_status": str(
                item.daily_window_status
            ),
            "anomaly_input_status": str(
                item.anomaly_input_status
            ),
            "anomaly_review_status": str(
                item.anomaly_review_status
            ),
            "anomaly_candidate": bool(
                item.anomaly_candidate
            ),
            "new_dbscan_status": str(
                item.new_dbscan_status
            ),
            "new_dbscan_candidate_id": (
                None
                if pd.isna(
                    item.new_dbscan_candidate_id
                )
                else str(
                    item.new_dbscan_candidate_id
                )
            )
        })

    # ----------------------------------------------
    # 5. Reconciliation
    # ----------------------------------------------

    stats = {
        "input_records": len(rows),
        "reporting_day": reporting_day,
        "anomaly_candidate_groups":
            len(anomaly_briefs),
        "anomaly_candidate_complaints":
            int(
                rows["anomaly_candidate"].sum()
            ),
        "spatial_candidate_groups":
            len(spatial_briefs),
        "spatial_candidate_complaints":
            int(
                rows["new_dbscan_status"]
                .eq("candidate_concentration")
                .sum()
            ),
        "spatial_groups_entirely_outside_history":
            sum(
                brief[
                    "historical_dbscan_associations"
                ]["relationship"]
                == "entirely_outside_historical_associations"
                for brief in spatial_briefs
            ),
        "synthetic_data": bool(
            synthetic_data
        )
    }

    if (
        stats["anomaly_candidate_complaints"]
        != sum(
            item["incoming_complaints"]
            for item in anomaly_briefs
        )
    ):
        raise ValueError(
            "Anomaly brief totals do not reconcile"
        )

    if (
        stats["spatial_candidate_complaints"]
        != sum(
            item["incoming_complaints"]
            for item in spatial_briefs
        )
    ):
        raise ValueError(
            "Spatial brief totals do not reconcile"
        )

    return {
        "metadata": {
            "reporting_day": reporting_day,
            "synthetic_data": bool(
                synthetic_data
            ),
            "interpretation": (
                "Exploratory reporting evidence, "
                "not verified infrastructure faults, "
                "risk scores or automatic repair "
                "priorities."
            )
        },
        "stats": stats,
        "anomaly_briefs": anomaly_briefs,
        "spatial_briefs": spatial_briefs,
        "record_audit": audit_rows,
        "map_geojson": {
            "type": "FeatureCollection",
            "name":
                "new_batch_spatial_candidate_locations",
            "features": map_features
        }
    }
