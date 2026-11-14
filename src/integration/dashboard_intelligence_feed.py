"""Dashboard data contract for three-model intelligence.

All counts describe complaint reports, not verified
infrastructure faults or automatic repair priorities.
"""

import pandas as pd


def _text(value):
    if value is None or pd.isna(value):
        return None
    return str(value)


def _unique(values):
    return sorted({
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    })


def build_dashboard_feed(integrated, bundle):
    """Build reusable JSON-compatible dashboard data."""

    records = integrated[
        "combined_records"
    ].copy()

    briefs = bundle["anomaly_briefs"]
    spatial_briefs = bundle["spatial_briefs"]
    geojson = bundle["map_geojson"]

    required = {
        "source_row",
        "complaint_id",
        "daily_window_status",
        "anomaly_area_id",
        "anomaly_category",
        "anomaly_input_status",
        "anomaly_review_status",
        "anomaly_candidate",
        "new_kmeans_description",
        "kmeans_reporting_profile",
        "new_dbscan_status",
        "new_dbscan_candidate_id",
        "dbscan_status"
    }

    missing = required - set(records.columns)

    if missing:
        raise ValueError(
            f"Missing dashboard columns: {sorted(missing)}"
        )

    if records["source_row"].duplicated().any():
        raise ValueError("Duplicate source_row values")

    record_audit = bundle["record_audit"]

    if len(record_audit) != len(records):
        raise ValueError(
            "Brief record audit count mismatch"
        )

    if [
        str(item["source_row"])
        for item in record_audit
    ] != records["source_row"].astype(str).tolist():
        raise ValueError(
            "Brief and integrated source rows differ"
        )

    anomaly_mask = (
        records["anomaly_candidate"]
        .fillna(False)
        .astype(bool)
    )

    spatial_mask = (
        records["new_dbscan_status"]
        .eq("candidate_concentration")
        .fillna(False)
    )

    eligible_mask = (
        records["anomaly_input_status"]
        .eq("eligible")
        .fillna(False)
    )

    both = int(
        (anomaly_mask & spatial_mask).sum()
    )

    anomaly_only = int(
        (anomaly_mask & ~spatial_mask).sum()
    )

    spatial_only = int(
        (spatial_mask & ~anomaly_mask).sum()
    )

    neither = int(
        (~anomaly_mask & ~spatial_mask).sum()
    )

    n_anomaly = int(anomaly_mask.sum())
    n_spatial = int(spatial_mask.sum())

    if n_anomaly != int(
        bundle["stats"]["anomaly_candidate_complaints"]
    ):
        raise ValueError(
            "Anomaly brief complaint total mismatch"
        )

    if n_spatial != int(
        bundle["stats"]["spatial_candidate_complaints"]
    ):
        raise ValueError(
            "Spatial brief complaint total mismatch"
        )

    if len(briefs) != int(
        bundle["stats"]["anomaly_candidate_groups"]
    ):
        raise ValueError(
            "Anomaly brief group count mismatch"
        )

    if len(spatial_briefs) != int(
        bundle["stats"]["spatial_candidate_groups"]
    ):
        raise ValueError(
            "Spatial brief group count mismatch"
        )

    if both != sum(
        int(item[
            "complaints_in_anomaly_candidate_groups"
        ])
        for item in spatial_briefs
    ):
        raise ValueError(
            "Anomaly-spatial overlap mismatch"
        )

    anomaly_keys = {
        (
            str(row.anomaly_area_id),
            str(row.anomaly_category)
        )
        for row in records.loc[
            anomaly_mask
        ].itertuples(index=False)
    }

    brief_keys = {
        (
            str(item["area_id"]),
            str(item["complaint_category"])
        )
        for item in briefs
    }

    if anomaly_keys != brief_keys:
        raise ValueError(
            "Anomaly brief identities do not reconcile"
        )

    candidate_ids = {
        str(value)
        for value in records.loc[
            spatial_mask,
            "new_dbscan_candidate_id"
        ].dropna()
    }

    spatial_ids = {
        str(item["candidate_id"])
        for item in spatial_briefs
    }

    map_features = geojson["features"]

    map_ids = {
        str(item["id"])
        for item in map_features
    }

    if (
        candidate_ids != spatial_ids
        or spatial_ids != map_ids
        or len(map_features) != len(spatial_ids)
    ):
        raise ValueError(
            "Spatial brief and map IDs disagree"
        )

    # ----------------------------------------------
    # Area-level K-Means reporting summaries
    # ----------------------------------------------

    area_rows = records.loc[
        eligible_mask
    ].copy()

    area_cards = []

    for area_id, group in area_rows.groupby(
        "anomaly_area_id",
        sort=True
    ):

        category_counts = (
            group["anomaly_category"]
            .astype(str)
            .value_counts()
        )

        breakdown = [
            {
                "complaint_category": category,
                "incoming_complaints": int(count)
            }
            for category, count in sorted(
                category_counts.items(),
                key=lambda item: (
                    -item[1], item[0]
                )
            )
        ]

        in_spatial = group[
            "new_dbscan_status"
        ].eq("candidate_concentration")

        area_cards.append({
            "area_id": str(area_id),
            "incoming_complaints": int(len(group)),
            "reporting_patterns": _unique(
                group["new_kmeans_description"]
            ),
            "historical_kmeans_profiles": _unique(
                group["kmeans_reporting_profile"]
            ),
            "category_breakdown": breakdown,
            "dominant_category": (
                breakdown[0]["complaint_category"]
                if breakdown else None
            ),
            "anomaly_candidate_complaints": int(
                group["anomaly_candidate"].sum()
            ),
            "spatial_candidate_complaints": int(
                in_spatial.sum()
            ),
            "spatial_candidate_ids": _unique(
                group.loc[
                    in_spatial,
                    "new_dbscan_candidate_id"
                ]
            ),
            "historical_established_complaints": int(
                group["dbscan_status"].eq(
                    "established_cluster"
                ).sum()
            ),
            "historical_outside_complaints": int(
                group["dbscan_status"].eq(
                    "outside_existing_clusters"
                ).sum()
            ),
            "historical_ambiguous_complaints": int(
                group["dbscan_status"].eq(
                    "ambiguous_association"
                ).sum()
            )
        })

    # ----------------------------------------------
    # Record-level dashboard table
    # ----------------------------------------------

    record_rows = []

    for row in records.itertuples(index=False):

        record_rows.append({
            "source_row": _text(row.source_row),
            "complaint_id": _text(row.complaint_id),
            "area_id": _text(row.anomaly_area_id),
            "complaint_category": _text(
                row.anomaly_category
            ),
            "reporting_day_status": _text(
                row.daily_window_status
            ),
            "anomaly_input_status": _text(
                row.anomaly_input_status
            ),
            "new_kmeans_pattern": _text(
                row.new_kmeans_description
            ),
            "historical_kmeans_profile": _text(
                row.kmeans_reporting_profile
            ),
            "new_dbscan_status": _text(
                row.new_dbscan_status
            ),
            "new_dbscan_candidate_id": _text(
                row.new_dbscan_candidate_id
            ),
            "historical_dbscan_status": _text(
                row.dbscan_status
            ),
            "anomaly_review_status": _text(
                row.anomaly_review_status
            ),
            "anomaly_candidate": bool(
                row.anomaly_candidate
            ),
            "spatial_candidate": (
                row.new_dbscan_status
                == "candidate_concentration"
            )
        })

    patterns = sorted({
        pattern
        for area in area_cards
        for pattern in area["reporting_patterns"]
    })

    summary = {
        "input_complaints": int(len(records)),
        "selected_day_complaints": int(
            records["daily_window_status"]
            .eq("selected_day").sum()
        ),
        "anomaly_eligible_complaints": int(
            eligible_mask.sum()
        ),
        "areas_with_eligible_reports":
            len(area_cards),
        "new_kmeans_patterns": len(patterns),
        "spatial_candidate_groups":
            len(spatial_briefs),
        "spatial_candidate_complaints": n_spatial,
        "anomaly_candidate_groups":
            len(briefs),
        "anomaly_candidate_complaints": n_anomaly,
        "complaints_in_both_signals": both,
        "complaints_anomaly_only": anomaly_only,
        "complaints_spatial_only": spatial_only,
        "complaints_in_neither_signal": neither,
        "distinct_complaints_in_either_signal":
            n_anomaly + n_spatial - both,
        "historical_baseline_areas": int(
            integrated["stats"][
                "historical_baseline_areas"
            ]
        )
    }

    if (
        both + anomaly_only
        + spatial_only + neither
        != len(records)
    ):
        raise ValueError(
            "Dashboard report partition mismatch"
        )

    if sum(
        area["incoming_complaints"]
        for area in area_cards
    ) != int(eligible_mask.sum()):
        raise ValueError(
            "Dashboard area totals mismatch"
        )

    return {
        "metadata": {
            "schema_version": "1.0",
            "reporting_day": str(
                integrated["stats"]["reporting_day"]
            ),
            "synthetic_data": bool(
                bundle["metadata"]["synthetic_data"]
            ),
            "evidence_policy": (
                "Screening candidates and reporting "
                "patterns only. No validated defects, "
                "risk scores, statistical significance "
                "claims or automatic repair priorities."
            ),
            "counting_policy": (
                "Anomaly and spatial complaint counts "
                "may overlap. Use distinct_complaints_"
                "in_either_signal for their union."
            )
        },
        "summary": summary,
        "filters": {
            "boroughs": sorted({
                str(area["area_id"]).split(
                    "_", 1
                )[0]
                for area in area_cards
            }),
            "categories": sorted({
                str(value)
                for value in area_rows[
                    "anomaly_category"
                ].dropna().unique()
            }),
            "kmeans_patterns": patterns,
            "anomaly_review_statuses": _unique(
                records["anomaly_review_status"]
            )
        },
        "area_cards": area_cards,
        "anomaly_cards": [
            {
                "id": (
                    f"ANOMALY::{item['area_id']}::"
                    f"{item['complaint_category']}"
                ),
                **item
            }
            for item in briefs
        ],
        "spatial_cards": [
            dict(item)
            for item in spatial_briefs
        ],
        "records": record_rows,
        "map_geojson": geojson
    }
