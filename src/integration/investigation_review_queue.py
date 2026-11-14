"""Evidence-led human investigation review queue.

The ranking is a transparent review-order heuristic,
NOT an infrastructure risk or repair-priority score.
"""


def display_location(area_id):
    text = str(area_id)

    if "_" not in text:
        return text

    borough, zip_code = text.rsplit("_", 1)

    return (
        borough.title()
        + " — ZIP "
        + zip_code
    )


def suggested_action(category, spatial=False):

    name = str(category).upper()

    if "TRAFFIC SIGNAL" in name:

        action = (
            "Identify the relevant intersection "
            "from the original complaint details. "
            "Compare reports to determine whether "
            "they describe the same signal issue, "
            "review previous signal maintenance "
            "and service records, and verify "
            "operation at the reported location."
        )

    elif "STREET LIGHT" in name:

        action = (
            "Compare reported locations and any "
            "available pole or fixture identifiers. "
            "Check whether reports describe the "
            "same outage, review recent lighting "
            "maintenance records, and verify "
            "the reported condition."
        )

    elif "SEWER" in name:

        action = (
            "Review the sewer complaint descriptions "
            "for a possible shared blockage, drainage "
            "or infrastructure issue. Compare nearby "
            "locations, examine recent sewer "
            "maintenance and cleaning history, "
            "and verify conditions before field action."
        )

    elif (
        "STREET CONDITION" in name
        or "SIDEWALK" in name
    ):

        action = (
            "Compare report descriptions and "
            "locations to identify repeated damage "
            "or reports about the same road or "
            "sidewalk segment. Review repair and "
            "maintenance history, then consider "
            "a site inspection to verify the issue."
        )

    else:

        action = (
            "Review the original complaints, "
            "compare locations and descriptions "
            "for a possible common cause, "
            "examine recent maintenance records, "
            "and verify the reported conditions."
        )

    if spatial:

        action += (
            " Pay particular attention to the "
            "newly identified spatial grouping; "
            "a concentration does not itself "
            "confirm one shared defect."
        )

    return action


def build_review_queue(feed):
    """Generate ranked, evidence-grounded review cards."""

    required = {
        "metadata",
        "summary",
        "anomaly_cards",
        "spatial_cards",
        "area_cards",
        "map_geojson"
    }

    if not required.issubset(feed):
        raise ValueError(
            "Dashboard feed missing required sections."
        )

    summary = feed["summary"]
    anomalies = feed["anomaly_cards"]
    spatial = feed["spatial_cards"]

    if len(anomalies) != int(
        summary["anomaly_candidate_groups"]
    ):
        raise ValueError(
            "Anomaly group count mismatch."
        )

    if len(spatial) != int(
        summary["spatial_candidate_groups"]
    ):
        raise ValueError(
            "Spatial group count mismatch."
        )

    spatial_ids = [
        str(item["candidate_id"])
        for item in spatial
    ]

    if len(set(spatial_ids)) != len(spatial_ids):
        raise ValueError(
            "Duplicate spatial candidate IDs."
        )

    features = feed["map_geojson"]["features"]

    if set(spatial_ids) != {
        str(item["id"])
        for item in features
    }:
        raise ValueError(
            "Map candidate identities disagree."
        )

    queue = []

    # Area/category historical reporting increases.
    for item in anomalies:

        area_id = str(item["area_id"])

        category = str(
            item["complaint_category"]
        )

        count = int(
            item["incoming_complaints"]
        )

        seasonal = int(
            item["seasonal_historical_max"]
        )

        broad = int(
            item["all_weekday_historical_max"]
        )

        overlap = int(
            item["complaints_in_spatial_candidates"]
        )

        if (
            overlap < 0
            or overlap > count
            or count <= seasonal
            or count <= broad
        ):
            raise ValueError(
                "Invalid anomaly comparison evidence."
            )

        has_spatial = overlap > 0

        if has_spatial:

            why = (
                f"{count} {category.lower()} reports "
                f"were received in {display_location(area_id)}. "
                f"The historical seasonal daily "
                f"maximum was {seasonal}, and the "
                f"all-weekday daily maximum was "
                f"{broad}. {overlap} of the new "
                f"reports also belong to one or "
                f"more spatial concentrations. "
                f"Both reporting-volume and "
                f"spatial screening signals "
                f"support reviewing this area."
            )

        else:

            why = (
                f"{count} {category.lower()} reports "
                f"were received in {display_location(area_id)}, "
                f"compared with historical daily "
                f"maxima of {seasonal} (seasonal) "
                f"and {broad} (all-weekday). "
                f"No new spatial concentration "
                f"is linked to these reports."
            )

        queue.append({
            "kind": "area_category",
            "tier": (
                1 if has_spatial else 2
            ),
            "review_label": (
                "Review first — two screening signals"
                if has_spatial
                else "Review reporting increase"
            ),
            "area_ids": [area_id],
            "location": display_location(
                area_id
            ),
            "category": category,
            "incoming_reports": count,
            "seasonal_max": seasonal,
            "all_weekday_max": broad,
            "historical_excess": (
                count - max(seasonal, broad)
            ),
            "spatial_overlap_reports": overlap,
            "candidate_id": None,
            "linked_spatial_ids": [
                str(value)
                for value in item.get(
                    "linked_spatial_candidate_ids", []
                )
            ],
            "kmeans_patterns": [
                str(value)
                for value in item.get(
                    "kmeans_reporting_patterns", []
                )
            ],
            "why": why,
            "suggested_action":
                suggested_action(
                    category,
                    spatial=has_spatial
                ),
            "evidence_note": (
                "Cross-window consistent historical "
                "screening candidate. Not a "
                "statistically confirmed anomaly."
            )
        })

    # Spatial findings without linked anomaly records.
    for item in spatial:

        count = int(
            item["incoming_complaints"]
        )

        overlap = int(
            item[
                "complaints_in_anomaly_candidate_groups"
            ]
        )

        if overlap < 0 or overlap > count:
            raise ValueError(
                "Invalid spatial overlap count."
            )

        if overlap != 0:
            continue

        candidate_id = str(
            item["candidate_id"]
        )

        category = str(
            item["complaint_category"]
        )

        areas = [
            str(value)
            for value in item["area_ids"]
        ]

        locations = ", ".join(
            display_location(area)
            for area in areas
        )

        distinct = int(
            item["distinct_submitted_locations"]
        )

        historical = item[
            "historical_dbscan_associations"
        ]

        outside = int(
            historical["outside"]
        )

        why = (
            f"{count} {category.lower()} reports "
            f"at {distinct} distinct submitted "
            f"coordinates form spatial candidate "
            f"{candidate_id} in {locations}. "
            f"{outside} reports are outside "
            f"established historical DBSCAN "
            f"associations. No qualifying "
            f"area/category reporting-increase "
            f"candidate is linked to this spatial "
            f"group. The concentration merits "
            f"verification, but is not a "
            f"confirmed infrastructure incident."
        )

        queue.append({
            "kind": "spatial_only",
            "tier": 3,
            "review_label":
                "Review spatial concentration",
            "area_ids": areas,
            "location": locations,
            "category": category,
            "incoming_reports": count,
            "seasonal_max": None,
            "all_weekday_max": None,
            "historical_excess": 0,
            "spatial_overlap_reports": 0,
            "candidate_id": candidate_id,
            "linked_spatial_ids": [
                candidate_id
            ],
            "kmeans_patterns": [
                str(value)
                for value in item.get(
                    "kmeans_reporting_patterns", []
                )
            ],
            "why": why,
            "suggested_action":
                suggested_action(
                    category,
                    spatial=True
                ),
            "evidence_note": (
                "Spatial screening only. "
                "This is not a historical-count "
                "anomaly conclusion."
            )
        })

    # Ranking is for human review, not field urgency.
    queue.sort(
        key=lambda item: (
            item["tier"],
            -item["historical_excess"],
            -item["spatial_overlap_reports"],
            -item["incoming_reports"],
            item["location"],
            item["category"]
        )
    )

    for index, item in enumerate(
        queue,
        start=1
    ):
        item["review_order"] = index

    first = sum(
        item["tier"] == 1
        for item in queue
    )

    historical_only = sum(
        item["tier"] == 2
        for item in queue
    )

    spatial_only = sum(
        item["tier"] == 3
        for item in queue
    )

    # Readable area summaries; never show raw
    # category_breakdown dictionaries as objects.
    area_table = []

    for area in feed["area_cards"]:

        area_table.append({
            "area_id": str(
                area["area_id"]
            ),
            "location": display_location(
                area["area_id"]
            ),
            "reports": int(
                area["incoming_complaints"]
            ),
            "reporting_pattern": ", ".join(
                map(
                    str,
                    area.get(
                        "reporting_patterns", []
                    )
                )
            ),
            "dominant_category": str(
                area["dominant_category"]
            ),
            "anomaly_associated": int(
                area["anomaly_candidate_complaints"]
            ),
            "spatial_associated": int(
                area["spatial_candidate_complaints"]
            )
        })

    return {
        "metadata": {
            "reporting_day": feed[
                "metadata"
            ]["reporting_day"],
            "synthetic_data": bool(
                feed["metadata"].get(
                    "synthetic_data", False
                )
            ),
            "review_policy": (
                "Human screening order only. "
                "Not a risk score, verified defect, "
                "statistical significance claim "
                "or repair dispatch instruction."
            )
        },
        "summary": {
            "reports": int(
                summary["input_complaints"]
            ),
            "distinct_with_signals": int(
                summary[
                    "distinct_complaints_in_either_signal"
                ]
            ),
            "review_first_groups": first,
            "historical_only_groups":
                historical_only,
            "spatial_only_groups": spatial_only,
            "review_items": len(queue),
            "all_spatial_candidates":
                len(spatial)
        },
        "queue": queue,
        "area_table": area_table,
        "spatial_cards": spatial,
        "map_geojson": feed["map_geojson"]
    }
